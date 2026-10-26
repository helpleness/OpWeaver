"""Real multi-input inference, preflight budgets, and termination during native ORT work."""

import json
import time
import uuid
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper as h
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
from opweaver.datasets import create_dataset
from opweaver.diagnostic_engine import QuantizationDiagnosticEngine
from opweaver.diagnostic_tasks import DiagnosticTasks
from opweaver.lab_reports import write_json
from opweaver.onnx_import import import_onnx
from opweaver.process_owner import terminate_process
from opweaver.storage import RunStore


class Reader(CalibrationDataReader):
    def __init__(self, feeds):
        self.feeds = iter(feeds)

    def get_next(self):
        return next(self.feeds, None)


def pair_project(root, model, feeds):
    project = root / "project"
    project.mkdir(parents=True)
    (project / "project.opweaver.json").write_text("{}")
    source, quantized = root / "float.onnx", root / "quantized.onnx"
    onnx.save(model, source)
    quantize_static(
        source,
        quantized,
        Reader(feeds[:1]),
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )

    def imported(path, parent=None):
        folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
        result = import_onnx(path, project, folder)
        result["artifact"]["parent_model_id"] = parent
        write_json(folder / "result.json", result)
        return result["artifact"]["id"]

    original = imported(source)
    candidate = imported(quantized, original)
    paths = []
    for index, feed in enumerate(feeds):
        path = root / f"feed-{index}.npz"
        np.savez(path, **feed)
        paths.append(str(path))
    dataset = create_dataset(
        project,
        {
            "source_model_id": original,
            "role": "diagnostic",
            "mode": "npz",
            "files": paths,
            "count": len(paths),
        },
    )
    return project, {
        "fp32_model_id": original,
        "quantized_model_id": candidate,
        "dataset_id": dataset["id"],
    }


def execute(project, config):
    folder = project / ".opweaver/diagnostics" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    return QuantizationDiagnosticEngine().execute(
        {"project": str(project), "config": config}, folder
    ), folder


def test_dynamic_multi_input_npz_and_variable_batch_aggregation(tmp_path):
    weights = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 12, "w")
    graph = h.make_graph(
        [
            h.make_node("MatMul", ["x", "w"], ["hidden"], name="multiply"),
            h.make_node("Add", ["hidden", "y"], ["output"], name="add"),
        ],
        "multi-input",
        [
            h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["B", 4]),
            h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["B", 3]),
        ],
        [h.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ["B", 3])],
        [weights],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    rng = np.random.default_rng(42)
    feeds = [
        {
            "x": rng.normal(size=(n, 4)).astype(np.float32),
            "y": rng.normal(size=(n, 3)).astype(np.float32),
        }
        for n in (2, 3, 1)
    ]
    project, config = pair_project(tmp_path, model, feeds)
    result, _ = execute(project, config)
    assert result["success"], result["error"]
    assert result["matching_summary"]["coverage"] == 1
    assert [v["example_count"] for v in result["captures"]] == [2, 3, 1]
    row = next(v for v in result["records"] if v["fp32_tensor"] == "hidden")
    assert row["aggregate"]["elements"] == 18 and row["aggregate"]["valid_samples"] == 3
    expected = sum(v["metrics"]["mae"] * v["metrics"]["elements"] for v in row["samples"]) / 18
    assert row["aggregate"]["mae"] == pytest.approx(expected)
    assert [s["fp32_concrete_inferred_shape"] for s in row["samples"]] == [[2, 3], [3, 3], [1, 3]]


@pytest.fixture(scope="module")
def heavy_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnostic-real-native-work")
    rng = np.random.default_rng(98)
    weight = numpy_helper.from_array((rng.normal(size=(1024, 1024)) / 32).astype(np.float32), "w")
    nodes = [
        h.make_node("MatMul", ["x" if i == 0 else f"h{i - 1}", "w"], [f"h{i}"], name=f"matmul_{i}")
        for i in range(32)
    ]
    graph = h.make_graph(
        nodes,
        "bounded diagnostic termination test",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["B", 1024])],
        [h.make_tensor_value_info("h31", onnx.TensorProto.FLOAT, ["B", 1024])],
        [weight],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    feeds = [{"x": rng.uniform(-1, 1, size=(512, 1024)).astype(np.float32)} for _ in range(16)]
    project, config = pair_project(root, model, feeds)
    return project, {
        **config,
        "selected_nodes": ["onnx_31"],
        "max_samples": 16,
        "max_tensors": 4,
        "include_weights": False,
    }


def test_preflight_rejects_capture_budget_before_running(heavy_pair):
    project, config = heavy_pair
    result, folder = execute(project, {**config, "memory_budget_mb": 1})
    assert not result["success"] and "before inference" in result["error"]
    assert result["resources"]["working_bytes_estimate"] > 1024**2
    assert not result["captures"] and not (folder / "temporary").exists()


@pytest.mark.parametrize("mode", ["cancel", "timeout", "crash"])
def test_actual_native_inference_termination_and_cleanup(heavy_pair, monkeypatch, mode):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    project, config = heavy_pair
    tasks = DiagnosticTasks()
    try:
        job = tasks.start(
            {
                "project": str(project),
                "config": {**config, "timeout": 3 if mode == "timeout" else 30},
            }
        )
        if mode != "timeout":
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                current = tasks.snapshot(job["id"])
                if current.get("phase") == "capture":
                    break
                assert current["state"] not in ("FAILED", "COMPLETED"), current.get("error")
                time.sleep(0.02)
            assert current.get("phase") == "capture"
            if mode == "cancel":
                tasks.stop(job["id"])
            else:
                terminate_process(tasks.processes[job["id"]])
        tasks.threads[job["id"]].join(timeout=20)
        ended = tasks.snapshot(job["id"])
        assert ended["status"] == ("Cancelled" if mode == "cancel" else "Failed"), ended
        if mode == "timeout":
            assert "timed out" in ended["error"]
        if mode == "crash":
            assert "exited" in ended["error"]
        assert not ended["result"]["success"]
        assert not (Path(ended["artifact_dir"]) / "temporary").exists()
        saved = DiagnosticTasks().load(project, job["id"])
        assert saved["status"] == ended["status"]
        # Simulate a persisted interrupted record left by a lost supervising Worker.
        record = {k: v for k, v in ended.items() if k != "result"}
        record.update(state="RUNNING", status="Running")
        RunStore(project).set_metadata(f"diagnostic:{job['id']}", record)
        leftover = Path(ended["artifact_dir"]) / "temporary"
        leftover.mkdir()
        (leftover / "partial.onnx").write_bytes(b"unfinished diagnostic-only artifact")
        recovered = DiagnosticTasks().load(project, job["id"])
        assert recovered["status"] == "Failed" and "interrupted" in recovered["error"]
        assert not leftover.exists()
    finally:
        tasks.shutdown()


def test_changed_model_failure_is_a_report(tmp_path):
    w = numpy_helper.from_array(np.eye(4, dtype=np.float32), "w")
    graph = h.make_graph(
        [h.make_node("MatMul", ["x", "w"], ["out"], name="mm")],
        "model",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["B", 4])],
        [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, ["B", 4])],
        [w],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    project, config = pair_project(tmp_path, model, [{"x": np.ones((2, 4), np.float32)}])
    folder = project / ".opweaver/imports" / config["quantized_model_id"]
    artifact = json.loads((folder / "result.json").read_text())["artifact"]
    target = project / artifact["file_path"]
    target.write_bytes(b"changed model payload")
    failed, _ = execute(project, config)
    assert not failed["success"] and "hash" in failed["error"]


def test_actual_nonfinite_feed_output_is_excluded_without_double_counting(tmp_path):
    graph = h.make_graph(
        [h.make_node("MatMul", ["x", "w"], ["hidden"]), h.make_node("Sqrt", ["hidden"], ["out"])],
        "partially comparable actual feeds",
        [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["B", 4])],
        [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, ["B", 4])],
        [numpy_helper.from_array(np.eye(4, dtype=np.float32), "w")],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    feeds = [{"x": np.full((2, 4), v, np.float32)} for v in (1, -1)]
    project, config = pair_project(tmp_path, model, feeds)
    result, _ = execute(project, {**config, "max_samples": 2})
    assert result["success"]
    row = next(r for r in result["records"] if r["fp32_tensor"] == "out")
    assert row["match_status"] == "Partially Matched"
    assert [s["sample_index"] for s in row["samples"]] == [0, 1]
    assert row["aggregate"]["valid_samples"] == 1 and row["aggregate"]["invalid_samples"] == 1
    assert row["samples"][1]["metrics"]["mae"] is None
