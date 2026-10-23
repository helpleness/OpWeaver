"""Real local images, preprocessing, labels, split isolation and Olive evaluation."""

import json
import uuid
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from opweaver.benchmark import create_session
from opweaver.classification_examples import generate_classification_examples
from opweaver.dataset_sources import ImagePreprocessor
from opweaver.datasets import (
    create_dataset,
    dataset_overlap,
    dataset_pair,
    load_dataset,
    sample_arrays,
)
from opweaver.lab_runner import execute
from opweaver.lab_tasks import OptimizationTasks
from opweaver.lab_types import (
    BenchmarkConfig,
    EvaluationConfig,
    ExperimentConfig,
    ImagePreprocessing,
)
from opweaver.onnx_import import import_onnx
from opweaver.task_evaluation import evaluate_single, evaluate_task
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def image_project(tmp_path):
    fixture = generate_classification_examples(tmp_path / "images")
    project = tmp_path / "project"
    project.mkdir()
    folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
    model = import_onnx(tmp_path / "images" / "color-classifier.onnx", project, folder)
    (folder / "result.json").write_text(json.dumps(model))

    def config(role="validation", **kwargs):
        image = {
            "manifest_path": str(tmp_path / "images" / fixture["datasets"][role]),
            "input_name": "image",
            "class_mapping": fixture["classes"],
            "batch_size": 2,
            "preprocessing": fixture["preprocessing"],
            "provenance": "synthetic_test",
            **kwargs,
        }
        return {
            "source_model_id": model["artifact"]["id"],
            "role": role,
            "mode": "images",
            "image": image,
        }

    return project, model, fixture, config


def test_image_preprocessor_exact_color_layout_normalization(tmp_path):
    pixels = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], np.uint8)
    path = tmp_path / "pixels.png"
    Image.fromarray(pixels).save(path)
    config = ImagePreprocessing(
        resize=[2, 2],
        center_crop=[2, 1],
        color="BGR",
        layout="NCHW",
        scale=0.1,
        mean=[1, 2, 3],
        std=[2, 2, 2],
    )
    result = ImagePreprocessor(config).transform(path)
    expected = (pixels[:, :1, ::-1].astype(np.float64) * 0.1 - np.array([1, 2, 3])) / 2
    np.testing.assert_allclose(result, expected.transpose(2, 0, 1))
    assert result.dtype == np.float32 and result.shape == (3, 2, 1)
    assert ImagePreprocessor(ImagePreprocessing(layout="NHWC", dtype="uint8")).transform(
        path
    ).shape == (2, 2, 3)
    with pytest.raises(ValueError, match="uint8"):
        ImagePreprocessor(ImagePreprocessing(dtype="uint8", scale=0.5))


def test_image_manifest_snapshot_metrics_and_held_out_roles(image_project):
    project, model, fixture, config = image_project
    records = {role: create_dataset(project, config(role)) for role in fixture["datasets"]}
    data = records["validation"]
    assert (
        data["example_count"] == 12
        and data["batch_count"] == 6
        and data["observed_class_count"] == 6
    )
    assert set(data["label_distribution"].values()) == {2}
    assert data["preprocessing"]["center_crop"] == [8, 8]
    feed, labels = sample_arrays(data["manifest_path"], data["samples"][0])
    assert feed["image"].shape == (2, 3, 8, 8) and labels.tolist() == [0, 1]
    session, _ = create_session(project / model["artifact"]["file_path"], BenchmarkConfig())
    evaluation = EvaluationConfig(task="classification_top1", output_name="scores")
    metrics = evaluate_single(session, data, evaluation)
    assert metrics["samples"] == metrics["correct"] == 12 and metrics["incorrect"] == 0
    assert metrics["top1_accuracy"] == metrics["top5_accuracy"] == 1
    pair = evaluate_task(session, session, data, evaluation)
    assert pair["accuracy_drop_percentage_points"] == pair["accuracy_change_percent"] == 0
    cfg = ExperimentConfig(
        source_model_id=model["artifact"]["id"],
        optimization={"calibration_dataset_id": records["calibration"]["id"]},
        evaluation={
            "dataset_id": data["id"],
            "final_dataset_id": records["final_evaluation"]["id"],
        },
    )
    assert dataset_pair(project, cfg)["independent"]
    cfg.evaluation.dataset_id = records["final_evaluation"]["id"]
    with pytest.raises(ValueError, match="Final evaluation"):
        dataset_pair(project, cfg)
    # Raw data and a portable replay CSV are preserved independently of the source tree.
    assert (Path(data["manifest_path"]).parent / "source/replay.csv").is_file()
    source = Path(data["manifest_path"]).parent / data["source_files"][1]["path"]
    source.unlink()
    with pytest.raises(OSError):
        load_dataset(project, data["id"])


@pytest.mark.parametrize(
    "bad",
    [
        "class_map",
        "name",
        "dtype",
        "shape",
        "missing",
        "duplicate",
        "crop",
        "empty",
        "std",
        "corrupt",
        "normalized_duplicate",
        "escaped",
        "fractional_label",
    ],
)
def test_image_validation_is_more_than_decoding(image_project, bad):
    project, _, _, config = image_project
    raw = config()
    image = raw["image"]
    if bad == "class_map":
        image["class_mapping"] = {"wrong": 0}
    if bad == "name":
        image["input_name"] = "wrong"
    if bad == "dtype":
        image["preprocessing"] = {**image["preprocessing"], "dtype": "float64"}
    if bad == "shape":
        image["preprocessing"] = {**image["preprocessing"], "center_crop": [6, 6]}
    if bad == "crop":
        image["preprocessing"] = {**image["preprocessing"], "center_crop": [20, 20]}
    if bad == "std":
        image["preprocessing"] = {**image["preprocessing"], "std": [0, 1, 1]}
    if bad == "normalized_duplicate":
        image["preprocessing"] = {**image["preprocessing"], "scale": 0}
    if bad == "fractional_label":
        image["class_mapping"] = {"red": 0.1}
    csv_path = Path(image["manifest_path"])
    if bad == "empty":
        csv_path.write_text("path,label\n")
    if bad == "missing":
        csv_path.write_text("path,label\nmissing.png,red\n")
    if bad == "duplicate":
        first = csv_path.read_text().splitlines()[1]
        csv_path.write_text("path,label\n" + first + "\n" + first + "\n")
    if bad == "corrupt":
        (csv_path.parent / "validation/sample-00.png").write_bytes(b"corrupt PNG data")
    if bad == "escaped":
        (csv_path.parent.parent / "escaped.png").write_bytes(
            (csv_path.parent / "validation/sample-00.png").read_bytes()
        )
        csv_path.write_text("path,label\n../escaped.png,red\n")
    before = set((project / ".opweaver/datasets").glob("*"))
    with pytest.raises((ValueError, OSError)):
        create_dataset(project, raw)
    assert set((project / ".opweaver/datasets").glob("*")) == before


def test_image_split_overlap_rebatching_and_reprocessing(image_project):
    project, model, _, config = image_project
    a = create_dataset(project, config("calibration"))
    raw = config("calibration", batch_size=3)
    raw["role"] = "validation"
    b = create_dataset(project, raw)
    assert b["overlaps"][0]["dataset_id"] == a["id"]
    assert "Detectable sample overlap" in " ".join(b["warnings"])
    # No equal batch hashes, but all twelve original images/tensors are shared.
    overlap = dataset_overlap(a, b)
    assert len([v for v in overlap if v.startswith("image:")]) == 12
    assert len([v for v in overlap if v.startswith("tensor:")]) == 12
    assert not any(v.startswith("batch:") for v in overlap)
    cfg = ExperimentConfig(
        source_model_id=model["artifact"]["id"],
        optimization={"calibration_dataset_id": a["id"]},
        evaluation={"dataset_id": b["id"]},
    )
    with pytest.raises(ValueError, match="overlap"):
        dataset_pair(project, cfg)
    cfg.evaluation.allow_calibration_reuse = True
    assert not dataset_pair(project, cfg)["independent"]
    raw["role"] = "final_evaluation"
    raw["image"]["preprocessing"] = {**raw["image"]["preprocessing"], "color": "BGR"}
    final = create_dataset(project, raw)
    assert any(v.startswith("image:") for v in dataset_overlap(a, final))
    cfg.evaluation.final_dataset_id = final["id"]
    with pytest.raises(ValueError, match="Final evaluation overlaps"):
        dataset_pair(project, cfg)


def test_image_last_batch_label_mapping_and_replay(image_project):
    project, _, _, config = image_project
    raw = config(batch_size=5)
    a = create_dataset(project, raw)
    assert [s["labels"] for s in a["samples"]] == [5, 5, 2]
    raw["image"]["drop_last"] = True
    b = create_dataset(project, raw)
    assert b["example_count"] == 10 and sum(b["label_distribution"].values()) == 10
    assert "discarded 2" in " ".join(b["warnings"])
    assert b["source_adapter"]["pillow_version"] and b["source_adapter"]["numpy_version"]
    raw["image"]["manifest_path"] = str(Path(b["manifest_path"]).parent / "source/replay.csv")
    replay = create_dataset(project, raw)
    assert replay["fingerprint"] == b["fingerprint"]
    for original, copied in zip(b["samples"], replay["samples"], strict=True):
        assert original["content_sha256"] == copied["content_sha256"]
        assert original["label_sha256"] == copied["label_sha256"]
    raw["image"]["class_mapping"] = {"red": True}
    with pytest.raises(ValueError):
        create_dataset(project, raw)


def test_real_classifier_incorrect_predictions_top5_and_percent_units(image_project, tmp_path):
    project, model, _, config = image_project
    data = create_dataset(project, config())
    original = project / model["artifact"]["file_path"]
    graph = onnx.load(original)
    for weight in graph.graph.initializer:
        if weight.name == "head_weight":
            array = numpy_helper.to_array(weight).copy()
            array[:, [0, 1]] = array[:, [1, 0]]
            weight.CopyFrom(numpy_helper.from_array(array, weight.name))
    changed = tmp_path / "wrong-class-order.onnx"
    onnx.save(graph, changed)
    a, _ = create_session(original, BenchmarkConfig())
    b, _ = create_session(changed, BenchmarkConfig())
    evaluation = EvaluationConfig(task="classification_top1", output_name="scores")
    result = evaluate_task(a, b, data, evaluation)
    assert result["optimized_metrics"]["correct"] == 8
    assert result["optimized_metrics"]["incorrect"] == 4
    assert result["accuracy_drop_percentage_points"] == pytest.approx(100 / 3)
    assert result["accuracy_change_percent"] == pytest.approx(-100 / 3)
    assert result["optimized_top5_accuracy"] == 1
    reversed_pair = evaluate_task(b, a, data, evaluation)
    assert reversed_pair["accuracy_drop_percentage_points"] == pytest.approx(-100 / 3)
    assert reversed_pair["accuracy_change_percent"] == pytest.approx(50)
    # Negating all six logits makes the previously correct label rank last.
    graph = onnx.load(original)
    for weight in graph.graph.initializer:
        if weight.name in ("head_weight", "head_bias"):
            weight.CopyFrom(numpy_helper.from_array(-numpy_helper.to_array(weight), weight.name))
    negative = tmp_path / "true-class-last.onnx"
    onnx.save(graph, negative)
    last, _ = create_session(negative, BenchmarkConfig())
    recovered = evaluate_task(last, a, data, evaluation)
    assert recovered["baseline_accuracy"] == recovered["baseline_top5_accuracy"] == 0
    assert recovered["accuracy_change_percent"] is None  # Zero denominator, not infinity.
    assert recovered["accuracy_drop_percentage_points"] == -100
    data["class_mapping"]["invalid_unused_label"] = 6
    with pytest.raises(ValueError, match="class mapping"):
        evaluate_single(a, data, evaluation)


def test_top5_is_not_fabricated_for_two_class_model(image_project, tmp_path):
    project, model, _, config = image_project
    raw = config()
    csv_path = Path(raw["image"]["manifest_path"])
    csv_path.write_text(
        "\n".join(
            row
            for row in csv_path.read_text().splitlines()
            if row == "path,label" or row.endswith((",red", ",green"))
        )
        + "\n"
    )
    raw["image"]["class_mapping"] = {"red": 0, "green": 1}
    data = create_dataset(project, raw)
    graph = onnx.load(project / model["artifact"]["file_path"])
    for weight in graph.graph.initializer:
        if weight.name in ("head_weight", "head_bias"):
            array = numpy_helper.to_array(weight)[..., :2].copy()
            weight.CopyFrom(numpy_helper.from_array(array, weight.name))
    graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2
    target = tmp_path / "two-class.onnx"
    onnx.save(graph, target)
    session, _ = create_session(target, BenchmarkConfig())
    result = evaluate_single(
        session, data, EvaluationConfig(task="classification_top1", output_name="scores")
    )
    assert result["correct"] == result["samples"] == 4
    assert result["top5_accuracy"] is None and result["top5_correct"] is None
    assert result["top5_status"] == "not_applicable_fewer_than_5_classes"


def test_real_images_fp32_baseline_then_olive_int8_and_export(image_project, tmp_path):
    project, model, _, config = image_project
    records = {
        role: create_dataset(project, config(role))
        for role in ("calibration", "validation", "final_evaluation")
    }
    experiment = {
        "source_model_id": model["artifact"]["id"],
        "mode": "baseline",
        "benchmark": {"runs": 15, "warmup": 2},
        "evaluation": {
            "dataset_id": records["validation"]["id"],
            "task": "classification_top1",
            "output_name": "scores",
            "final_dataset_id": records["final_evaluation"]["id"],
        },
    }

    def run(cfg):
        folder = project / ".opweaver" / "optimizations" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        return execute({"project": str(project), "config": cfg}, folder)

    baseline = run(experiment)
    assert baseline["success"], baseline["error"]
    assert baseline["baseline_evaluation"]["correct"] == 12
    assert baseline["phases"]["baseline"] == "Succeeded"
    environment = ROOT / ".tools/olive-env"
    if not environment.exists():
        pytest.skip("Install independent Olive environment")
    experiment.update(
        mode="optimize",
        optimization={
            "backend_id": "microsoft.olive",
            "environment": {"python_path": str(environment)},
            "calibration_dataset_id": records["calibration"]["id"],
            "workflow": {"passes": [{"id": "quant", "type": "OnnxStaticQuantization"}]},
        },
    )
    result = run(experiment)
    assert result["success"], result["error"]
    assert result["validation"]["task_quality"]["baseline_accuracy"] == 1
    assert result["validation"]["task_quality"]["optimized_metrics"]["samples"] == 12
    assert result["validation"]["task_quality"]["optimized_top5_accuracy"] is not None
    assert result["final_evaluation"]["samples"] == 12
    assert result["optimized"]["conditions_hash"] == result["baseline"]["conditions_hash"]
    exported = OptimizationTasks().export(
        {"project": str(project), "id": result["id"], "directory": str(tmp_path)}
    )
    root = Path(exported["directory"])
    assert (root / "datasets/evaluation/source/replay.csv").is_file()
    assert len(list((root / "datasets/evaluation/source").glob("image-*"))) == 12
