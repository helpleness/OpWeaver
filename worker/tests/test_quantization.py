"""Real Olive static quantization, shared NPZ datasets and lossy quality semantics."""

import json
import os
import uuid
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from opweaver.benchmark import digest_file
from opweaver.datasets import DatasetFeeds, create_dataset, dataset_pair, load_dataset
from opweaver.lab_runner import execute
from opweaver.lab_tasks import OptimizationTasks
from opweaver.lab_types import ExperimentConfig
from opweaver.olive_adapter import preview
from opweaver.onnx_examples import cnn
from opweaver.onnx_import import import_onnx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def quant_project(tmp_path):
    project = tmp_path / "quant project"
    project.mkdir()
    source = project / "cnn.onnx"
    graph = cnn()
    rng = np.random.default_rng(190)
    # Unequal weights/output classes ensure rounding errors are actually measurable.
    for tensor in graph.graph.initializer:
        array = rng.normal(0, 0.23, tensor.dims).astype(np.float32)
        tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
    onnx.save(graph, source)
    folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
    report = import_onnx(source, project, folder)
    (folder / "result.json").write_text(json.dumps(report), encoding="utf-8")
    model_id = report["artifact"]["id"]
    records = {}
    for role, seed, count in [("calibration", 23, 8), ("evaluation", 1023, 4)]:
        files = []
        for i in range(count):
            path = project / f"{role}-{i}.npz"
            np.savez(
                path,
                pixels=np.random.default_rng(seed + i)
                .normal(0, 1, (2, 3, 8, 8))
                .astype(np.float32),
            )
            files.append(str(path))
        records[role] = create_dataset(
            project,
            {
                "source_model_id": model_id,
                "name": role,
                "role": role,
                "files": files,
                "count": count,
                "mapping": {"image": "pixels"},
            },
        )
    config = {
        "source_model_id": model_id,
        "inputs": {"shapes": {"image": [2, 3, 8, 8]}},
        "benchmark": {"warmup": 3, "runs": 30, "batch_size": 2, "batch_input": "image"},
        "optimization": {
            "backend_id": "microsoft.olive",
            "environment": {
                "python_path": os.environ.get(
                    "OPWEAVER_OLIVE_PYTHON", str(ROOT / ".tools" / "olive-env")
                )
            },
            "calibration_dataset_id": records["calibration"]["id"],
            "workflow": {
                "id": "static_int8",
                "passes": [{"id": "quant", "type": "OnnxStaticQuantization"}],
            },
        },
        "evaluation": {
            "dataset_id": records["evaluation"]["id"],
            "gates": {"max_mean_abs_error": 0.0},
        },
    }
    return project, config, records


def test_dataset_mapping_streaming_integrity_and_reuse(quant_project):
    project, config, records = quant_project
    calibration = records["calibration"]
    assert calibration["batch_count"] == 8 and calibration["example_count"] == 16
    assert set(DatasetFeeds(calibration)[0]) == {"image"}
    assert dataset_pair(project, ExperimentConfig.model_validate(config))["independent"]
    config["evaluation"]["dataset_id"] = calibration["id"]
    with pytest.raises(ValueError, match="overlap"):
        dataset_pair(project, ExperimentConfig.model_validate(config))
    config["evaluation"]["allow_calibration_reuse"] = True
    assert not dataset_pair(project, ExperimentConfig.model_validate(config))["independent"]
    target = Path(calibration["manifest_path"]).parent / calibration["samples"][0]["path"]
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_dataset(project, calibration["id"])


@pytest.mark.parametrize(
    "bad", ["missing", "corrupt", "empty", "dtype", "shape", "nan", "mapping", "count", "object"]
)
def test_dataset_rejects_invalid_data(quant_project, bad):
    project, config, _ = quant_project
    path = project / "bad.npz"
    value = np.ones((2, 3, 8, 8), np.float32)
    if bad == "empty":
        value = np.ones((0, 3, 8, 8), np.float32)
    if bad == "dtype":
        value = value.astype(np.float64)
    if bad == "shape":
        value = value[:, :, :, :4]
    if bad == "nan":
        value.flat[0] = np.nan
    if bad == "object":
        value = np.array([object()])
    if bad != "missing":
        np.savez(path, image=value)
    if bad == "corrupt":
        path.write_bytes(b"broken NPZ")
    raw = {"source_model_id": config["source_model_id"], "files": [str(path)], "count": 1}
    if bad == "mapping":
        raw["mapping"] = {"unknown": "image"}
    if bad == "count":
        raw["count"] = 2
    with pytest.raises((ValueError, OSError, __import__("zipfile").BadZipFile)):
        create_dataset(project, raw)


def test_shared_multi_input_dynamic_and_labeled_top1(tmp_path):
    from onnx import TensorProto
    from onnx import helper as h
    from opweaver.benchmark import create_session
    from opweaver.lab_types import BenchmarkConfig, EvaluationConfig
    from opweaver.onnx_examples import model, value
    from opweaver.task_evaluation import evaluate_task, quality_gate

    graph = model(
        [h.make_node("Add", ["x", "y"], ["scores"])],
        [value("x", ["B", 2]), value("y", ["B", 2])],
        [value("scores", ["B", 2])],
    )
    source = tmp_path / "multi.onnx"
    onnx.save(graph, source)
    folder = tmp_path / ".opweaver" / "imports" / uuid.uuid4().hex
    report = import_onnx(source, tmp_path, folder)
    (folder / "result.json").write_text(json.dumps(report))
    files = []
    for i in range(3):
        # Explicit supervised toy task: class 0 has the larger supplied first score.
        path = tmp_path / f"labeled-{i}.npz"
        np.savez(
            path,
            left=np.array([[4 + i, 1], [1, 4 + i]], np.float32),
            right=np.zeros((2, 2), np.float32),
            labels=np.array([0, 1], np.int64),
        )
        files.append(str(path))
    raw = {
        "source_model_id": report["artifact"]["id"],
        "role": "evaluation",
        "files": files,
        "count": 3,
        "mapping": {"x": "left", "y": "right"},
        "label_key": "labels",
    }
    dataset = create_dataset(tmp_path, raw)
    session, _ = create_session(source, BenchmarkConfig())
    config = EvaluationConfig(
        task="classification_top1", output_name="scores", gates={"max_accuracy_drop": 0}
    )
    result = evaluate_task(session, session, dataset, config)
    assert (
        result["samples"] == 6 and result["baseline_accuracy"] == result["optimized_accuracy"] == 1
    )
    assert (
        quality_gate(
            {"mean_abs_error": 0, "max_abs_error": 0, "changed_fraction": 0}, result, config.gates
        )["status"]
        == "passed"
    )
    raw.update(mode="random", files=[], label_key=None, shapes={"x": [2, 2], "y": [2, 2]})
    unlabeled = create_dataset(tmp_path, raw)
    with pytest.raises(ValueError, match="labels"):
        evaluate_task(session, session, unlabeled, config)
    assert TensorProto.FLOAT == graph.graph.input[0].type.tensor_type.elem_type


@pytest.mark.parametrize(
    "parameters",
    [
        {"calibrate_method": "Imaginary"},
        {"data_config": {}},
        {"quant_preprocess": True},
        {"quant_format": "QOperator", "activation_type": "int8"},
        {"op_types_to_quantize": ["Imaginary"]},
    ],
)
def test_quantization_preflight_rejects_unsupported_contract(quant_project, parameters):
    project, config, _ = quant_project
    config["optimization"]["workflow"]["passes"][0]["parameters"] = parameters
    with pytest.raises(ValueError):
        preview(project, config)


@pytest.mark.parametrize(
    "method,format,activation,channel",
    [
        ("MinMax", "QDQ", "int8", True),
        ("Percentile", "QDQ", "uint8", False),
        ("Entropy", "QOperator", "uint8", False),
    ],
)
def test_real_static_olive_data_config_quality_benchmark_export(
    quant_project, tmp_path, method, format, activation, channel
):
    project, config, records = quant_project
    if not Path(config["optimization"]["environment"]["python_path"]).exists():
        pytest.skip("Independent Olive environment not installed")
    config["optimization"]["workflow"]["passes"][0]["parameters"] = {
        "calibrate_method": method,
        "quant_format": format,
        "activation_type": activation,
        "per_channel": channel,
    }
    if method == "Percentile":
        config["optimization"]["workflow"]["passes"][0]["parameters"].update(
            precision="uint8", op_types_to_exclude=["Gemm"]
        )
    before = digest_file(project / "cnn.onnx")
    if channel:
        config["optimization"]["workflow"]["passes"].insert(
            0, {"id": "preprocess", "type": "OnnxQuantizationPreprocess"}
        )
    checked = preview(project, config)
    native = checked["native_workflow"]["passes"]["quant"]["config"]
    assert native["data_config"]["type"] == "OpWeaverNPZDataContainerV1"
    assert native["calibrate_method"] == method and native["quant_preprocess"] is False
    folder = project / ".opweaver" / "optimizations" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    result = execute({"project": str(project), "config": checked["config"]}, folder)
    assert result["success"], (
        result["error"],
        (folder / "olive" / "stderr.log").read_text()[-4000:],
    )
    assert result["quantization"]["present"] and result["quantization"]["integer_weights"]
    assert result["quantization"]["representation"] == format
    assert result["datasets"]["independent"]
    assert result["workflow_execution"]["calibration_usage"][0]["unique_batches"] == 8
    assert result["validation"]["execution_valid"]
    assert not result["validation"]["passed"]  # lossy quantization is not strict FP32 equivalence
    assert result["validation"]["numerical_summary"]["mean_abs_error"] > 0
    assert result["validation"]["quality_gate"]["status"] == "failed"
    assert result["validation"]["task_quality"]["status"] == "not_evaluated"
    assert result["baseline"]["conditions_hash"] == result["optimized"]["conditions_hash"]
    assert len(result["optimized"]["latencies_ms"]) == 30
    assert digest_file(project / "cnn.onnx") == before
    tasks = OptimizationTasks()
    reopened = tasks.load(project, result["id"])
    assert reopened["data_availability"]["calibration"]["available"]
    exported = tasks.export(
        {"project": str(project), "id": result["id"], "directory": str(tmp_path)}
    )
    assert Path(exported["json"]).is_file()
    assert (Path(exported["directory"]) / "datasets" / "calibration" / "manifest.json").is_file()
    if channel:
        assert list((Path(exported["directory"]) / "preprocessed").rglob("*.onnx"))
        assert result["preprocessed_artifacts"][0]["parent_model_id"] == result["source_model_id"]
        assert (
            result["optimization"]["artifact"]["parent_model_id"]
            == result["preprocessed_artifacts"][0]["id"]
        )
    Path(records["evaluation"]["manifest_path"]).unlink()
    assert not tasks.load(project, result["id"])["data_availability"]["evaluation"]["available"]
