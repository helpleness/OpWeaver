"""Real ORT QDQ execution, conservative matching, bounded capture and durable jobs."""

import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from opweaver.benchmark import digest_file
from opweaver.classification_examples import generate_classification_examples
from opweaver.datasets import DatasetFeeds, create_dataset
from opweaver.diagnostic_capture import augment_model, collect
from opweaver.diagnostic_engine import QuantizationDiagnosticEngine
from opweaver.diagnostic_metrics import aggregate_metrics, tensor_metrics
from opweaver.diagnostic_tasks import DiagnosticTasks
from opweaver.diagnostic_types import DiagnosticConfig
from opweaver.lab_reports import write_json
from opweaver.onnx_import import import_onnx


class Reader(CalibrationDataReader):
    def __init__(self, feeds):
        self.feeds = iter(feeds)

    def get_next(self):
        return next(self.feeds, None)


@pytest.fixture
def diagnostic_project(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    source, project = tmp_path / "source", tmp_path / "project"
    fixture = generate_classification_examples(source)
    project.mkdir()
    (project / "project.opweaver.json").write_text("{}")

    def imported(path, parent=None):
        folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
        result = import_onnx(path, project, folder)
        result["artifact"]["parent_model_id"] = parent
        write_json(folder / "result.json", result)
        return result

    original = imported(source / "color-classifier.onnx")
    data = create_dataset(
        project,
        {
            "source_model_id": original["artifact"]["id"],
            "role": "diagnostic",
            "mode": "images",
            "image": {
                "manifest_path": str(source / "diagnostic.csv"),
                "input_name": "image",
                "class_mapping": fixture["classes"],
                "batch_size": 2,
                "preprocessing": fixture["preprocessing"],
                "provenance": "synthetic_test",
            },
        },
    )

    def quantized(format=QuantFormat.QDQ, transform=None):
        target = source / (uuid.uuid4().hex + ".onnx")
        quantize_static(
            source / "color-classifier.onnx",
            target,
            Reader(DatasetFeeds(data)),
            quant_format=format,
            activation_type=QuantType.QUInt8
            if format == QuantFormat.QOperator
            else QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "Gemm"],
        )
        if transform:
            model = onnx.load(target)
            transform(model)
            onnx.save(model, target)
        return imported(target, original["artifact"]["id"])

    candidate = quantized()
    config = {
        "fp32_model_id": original["artifact"]["id"],
        "quantized_model_id": candidate["artifact"]["id"],
        "dataset_id": data["id"],
        "max_samples": 3,
    }

    def run(**overrides):
        folder = project / ".opweaver" / "diagnostics" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        result = QuantizationDiagnosticEngine().execute(
            {"project": str(project), "config": {**config, **overrides}}, folder
        )
        return result, folder

    return project, original, candidate, data, config, run, quantized, imported


def test_known_metrics_and_element_weighted_aggregation():
    a, b = np.array([1, 2, 3], np.float32), np.array([2, 3, 4], np.float32)
    value = tensor_metrics(a, b)
    assert value["valid"]
    assert value["mae"] == value["mse"] == value["rmse"] == value["max_abs_error"] == 1
    assert value["mean_relative_error"] == pytest.approx((1 + 0.5 + 1 / 3) / 3)
    assert value["cosine_similarity"] == pytest.approx(20 / math.sqrt(14 * 29))
    second = tensor_metrics(np.array([1], np.float32), np.array([4], np.float32))
    aggregate = aggregate_metrics([{"metrics": value}, {"metrics": second}])
    assert aggregate["mae"] == 1.5 and aggregate["mean_sample_mae"] == 2
    assert aggregate["mse"] == 3 and aggregate["rmse"] == math.sqrt(3)
    assert aggregate["max_abs_error"] == 3 and aggregate["elements"] == 4
    zero = tensor_metrics(np.zeros(3, np.float32), np.zeros(3, np.float32))
    assert zero["valid"] and zero["mae"] == 0
    assert zero["cosine_similarity"] is zero["mean_relative_error"] is None


@pytest.mark.parametrize("bad", ["nan", "inf", "shape", "dtype", "empty", "integer", "overflow"])
def test_invalid_tensor_metrics_are_not_zero(bad):
    a, b = np.ones(2, np.float32), np.ones(2, np.float32)
    if bad in ("nan", "inf"):
        b[0] = float(bad)
    elif bad == "shape":
        b = b.reshape(1, 2)
    elif bad == "dtype":
        b = b.astype(np.float16)
    elif bad == "empty":
        a = b = np.zeros(0, np.float32)
    elif bad == "integer":
        a = b = np.zeros(2, np.int8)
    else:
        a, b = np.array([1e308]), np.array([-1e308])
    result = tensor_metrics(a, b)
    assert not result["valid"] and result["reason"] and result["mae"] is None


def test_real_capture_and_complete_qdq_diagnostic(diagnostic_project, tmp_path):
    project, original, candidate, data, _, run, _, _ = diagnostic_project
    source = project / original["artifact"]["file_path"]
    before = digest_file(source)
    augmented = tmp_path / "capture.onnx"
    assert augment_model(source, augmented, {"features"}) == {"features"}
    feed = DatasetFeeds(data)[0]
    activation = collect(augmented, feed, "CPUExecutionProvider")
    np.testing.assert_array_equal(activation["features"][0], feed["image"].reshape(-1))
    assert activation["features"][0].shape == (384,)
    result, folder = run()
    assert result["success"], result["error"]
    assert result["matching_summary"]["comparable_tensors"] == 5
    assert result["matching_summary"]["coverage"] == 1
    assert len(result["captures"]) == 3
    assert len({c["feed_fingerprint"] for c in result["captures"]}) == 3
    rows = [v for v in result["records"] if v["kind"] == "activation"]
    assert all(v["aggregate"]["valid_samples"] == 3 for v in rows)
    assert any(v["aggregate"]["mae"] > 0 for v in rows)
    conv = next(v for v in rows if v["fp32_tensor"] == "features")
    assert conv["match_method"].startswith("ort_qdq")
    assert conv["samples"][0]["fp32_concrete_inferred_shape"] == [2, 3, 8, 8]
    assert conv["samples"][0]["fp32_capture_shape"] == [384]
    assert conv["samples"][0]["local_qdq_metrics"]["valid"]
    weights = [v for v in result["records"] if v["kind"] == "weight"]
    assert len(weights) == 3 and all(v["aggregate"]["valid_samples"] == 1 for v in weights)
    stem = next(v for v in weights if v["fp32_tensor"] == "stem_weight")
    assert stem["quantization"]["axis"] == 0
    assert stem["quantization"]["scale"]["elements"] == 3
    assert result["lineage"]["confirmed"]
    assert digest_file(source) == before
    assert not (folder / "temporary").exists()
    assert not list(source.parent.glob("*inferred*"))
    assert (
        result["resources"]["exposed_fp32_tensors"]
        + result["resources"]["exposed_quantized_tensors"]
        <= result["config"]["max_tensors"]
    )
    reopened = DiagnosticTasks().load(project, result["id"])
    assert all(v["available"] for v in reopened["availability"].values())
    export = DiagnosticTasks().export(
        {"project": str(project), "id": result["id"], "directory": str(tmp_path)}
    )
    report = json.loads((Path(export["directory"]) / "report.json").read_text())
    assert report["records"] == result["records"]
    assert "Activation Error" in (Path(export["directory"]) / "report.md").read_text()
    assert "Weight Error" in (Path(export["directory"]) / "report.md").read_text()
    assert candidate["artifact"]["id"] == result["models"]["quantized"]["id"]


def test_limits_sample_scope_and_memory(diagnostic_project):
    _, _, _, _, _, run, _, _ = diagnostic_project
    result, _ = run(max_samples=1, selected_nodes=["onnx_0"], include_weights=False)
    assert result["success"] and len(result["captures"]) == 1
    assert len(result["records"]) == 1 and result["resources"]["selected_batches"] == 1
    limited, _ = run(max_tensors=2, selected_nodes=["onnx_0"], include_weights=False)
    assert not limited["success"] and limited["records"][0]["match_status"] == "Unsupported"
    assert "Max Tensors" in limited["records"][0]["message"]
    disk, folder = run(disk_budget_mb=1)
    assert not disk["success"] and "disk budget" in disk["error"]
    assert not (folder / "temporary").exists()
    invalid, _ = run(selected_nodes=["not-a-node"])
    assert not invalid["success"] and "Selected original node" in invalid["error"]


def test_node_renames_match_but_tensor_renames_are_not_guessed(diagnostic_project):
    _, _, _, _, _, run, quantized, _ = diagnostic_project

    def rename_nodes(model):
        for index, node in enumerate(model.graph.node):
            node.name = f"renamed_{index}"

    changed = quantized(transform=rename_nodes)
    result, _ = run(quantized_model_id=changed["artifact"]["id"], include_weights=False)
    assert result["success"] and result["matching_summary"]["coverage"] == 1

    def rename_tensor(model):
        for node in model.graph.node:
            for names in (node.input, node.output):
                for i, name in enumerate(names):
                    if name == "positive":
                        names[i] = "renamed_positive"
        for value in model.graph.value_info:
            if value.name == "positive":
                value.name = "renamed_positive"

    changed = quantized(transform=rename_tensor)
    partial, _ = run(quantized_model_id=changed["artifact"]["id"], include_weights=False)
    assert partial["success"] and 0 < partial["matching_summary"]["coverage"] < 1
    missing = next(v for v in partial["records"] if v["fp32_tensor"] == "positive")
    assert missing["match_status"] == "Unmatched" and missing["aggregate"] is None


def test_qoperator_failure_is_durable_and_explicit(diagnostic_project):
    project, _, _, _, _, run, quantized, _ = diagnostic_project
    other = quantized(format=QuantFormat.QOperator)
    result, folder = run(quantized_model_id=other["artifact"]["id"])
    assert result["status"] == "Failed" and "QOperator" in result["error"]
    assert result["quantization"]["representation"] == "QOperator"
    assert not result["captures"] and not (folder / "temporary").exists()
    assert DiagnosticTasks().load(project, result["id"])["error"] == result["error"]


def test_changed_graph_contract_is_not_a_name_based_match(diagnostic_project):
    _, _, _, _, _, run, quantized, _ = diagnostic_project

    def replace_relu(model):
        next(n for n in model.graph.node if n.op_type == "Relu").op_type = "Sigmoid"

    changed = quantized(transform=replace_relu)
    result, _ = run(quantized_model_id=changed["artifact"]["id"], include_weights=False)
    assert result["success"] and 0 < result["matching_summary"]["coverage"] < 1
    row = next(r for r in result["records"] if r["fp32_tensor"] == "positive")
    assert row["match_status"] == "Partially Matched" and row["aggregate"] is None
    assert "producer" in row["message"].lower() or "contract" in row["message"].lower()


def test_incompatible_input_names_fail_before_capture(diagnostic_project):
    _, _, _, _, _, run, quantized, _ = diagnostic_project

    def rename_input(model):
        model.graph.input[0].name = "other_image"
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == "image":
                    node.input[i] = "other_image"

    changed = quantized(transform=rename_input)
    result, folder = run(quantized_model_id=changed["artifact"]["id"])
    assert not result["success"] and "input" in result["error"].lower()
    assert not result["captures"] and not (folder / "temporary").exists()


def test_final_role_is_not_a_diagnostic_shortcut(diagnostic_project):
    project, original, _, data, _, run, _, _ = diagnostic_project
    raw = dict(data["config"])
    raw["role"] = "final_evaluation"
    final = create_dataset(project, raw)
    result, _ = run(dataset_id=final["id"])
    assert not result["success"] and "diagnostic role" in result["error"]
    result, _ = run()
    assert not result["success"] and "held-out final evaluation" in result["error"]
    assert original["artifact"]["id"] == result["config"]["fp32_model_id"]


def test_missing_resources_keep_historical_metrics(diagnostic_project):
    project, original, _, _, _, run, _, _ = diagnostic_project
    result, _ = run(max_samples=1)
    assert result["success"]
    (project / original["artifact"]["file_path"]).unlink()
    restored = DiagnosticTasks().load(project, result["id"])
    assert not restored["availability"]["fp32"]["available"]
    assert restored["records"] == result["records"]


def wait_job(tasks, id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = tasks.snapshot(id)
        if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            tasks.threads[id].join(timeout=2)
            return job
        time.sleep(0.02)
    raise AssertionError("Diagnostic did not terminate")


def test_async_start_progress_cancel_restore_and_worker_exclusion(diagnostic_project):
    project, _, _, _, config, _, _, _ = diagnostic_project
    events, tasks = [], DiagnosticTasks()
    tasks.emit = events.append
    try:
        job = tasks.start({"project": str(project), "config": {**config, "max_samples": 6}})
        ended = wait_job(tasks, job["id"])
        assert ended["status"] == "Succeeded", ended["error"]
        # Tiny models can finish a capture phase between supervisor polls.
        # The durable phase log must still contain actual capture progress.
        assert any("[capture]" in line for line in ended["result"]["logs"])
        assert any(v["data"].get("phase") for v in events)
        assert DiagnosticTasks().history(project)[0]["id"] == job["id"]
        cancelled = tasks.start({"project": str(project), "config": config})
        tasks.stop(cancelled["id"])
        ended = wait_job(tasks, cancelled["id"])
        assert ended["status"] == "Cancelled" and not ended["result"]["success"]
        assert not (Path(ended["artifact_dir"]) / "temporary").exists()
        assert DiagnosticTasks().load(project, cancelled["id"])["status"] == "Cancelled"
    finally:
        tasks.shutdown()


def test_schema_and_config_bounds():
    root = Path(__file__).resolve().parents[2]
    schema = root / "schemas/quantization-diagnostic.schema.json"
    if schema.exists():
        from opweaver.diagnostic_types import DiagnosticResult

        assert json.loads(schema.read_text()) == DiagnosticResult.model_json_schema()
    with pytest.raises(ValueError):
        DiagnosticConfig(
            fp32_model_id="a" * 32, quantized_model_id="b" * 32, dataset_id="c" * 32, max_tensors=0
        )
