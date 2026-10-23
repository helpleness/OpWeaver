import io
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from opweaver.benchmark import compare_benchmarks, digest_file
from opweaver.lab_inputs import check_feed, load_npz, prepare_inputs
from opweaver.lab_reports import export_experiment, verify_artifact
from opweaver.lab_runner import execute
from opweaver.lab_tasks import OptimizationTasks
from opweaver.lab_types import BenchmarkConfig, Experiment, InputConfig
from opweaver.lab_validation import compare_tensor
from opweaver.model_artifact import ModelArtifact
from opweaver.model_files import atomic_replace
from opweaver.model_tasks import ModelTasks
from opweaver.onnx_examples import generate_examples
from opweaver.onnx_import import import_onnx
from opweaver.optimization import OrtCPUBackend, capabilities
from opweaver.process_owner import terminate_process
from opweaver.storage import RunStore


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    source = tmp_path / "source"
    generate_examples(source)
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.opweaver.json").write_text("{}")

    def imported(name="tiny-cnn"):
        folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
        result = import_onnx(source / f"{name}.onnx", project, folder)
        (folder / "result.json").write_text(json.dumps(result))
        return result

    def run(name="tiny-cnn", **overrides):
        original = imported(name)
        folder = project / ".opweaver" / "optimizations" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        config = {
            "source_model_id": original["artifact"]["id"],
            "inputs": {"shapes": {"image": [2, 3, 8, 8]}},
            "benchmark": {"warmup": 2, "runs": 12, "batch_size": 2, "batch_input": "image"},
            **overrides,
        }
        return execute({"project": str(project), "config": config}, folder), folder, original

    return SimpleNamespace(project=project, source=source, imported=imported, run=run)


def test_real_baseline_sampling_and_statistics(setup):
    result, folder, _ = setup.run(mode="baseline")
    assert result["status"] == "Succeeded", result["error"]
    baseline = result["baseline"]
    timings = baseline["latencies_ms"]
    assert len(timings) == 12 and all(t > 0 for t in timings)
    assert baseline["mean_ms"] == pytest.approx(np.mean(timings))
    assert baseline["p50_ms"] == pytest.approx(np.percentile(timings, 50))
    assert baseline["p95_ms"] == pytest.approx(np.percentile(timings, 95))
    assert baseline["requests_per_second"] == pytest.approx(1000 / np.mean(timings))
    assert baseline["samples_per_second"] == pytest.approx(2000 / np.mean(timings))
    assert baseline["session_init_ms"] > 0 and baseline["warmup_ms"] > 0
    assert baseline["input_config"]["seed"] == 42
    assert baseline["session_options"]["graph_optimization_level"] == "disable"
    assert result["optimization"] is None and result["phases"]["optimize"] == "Skipped"
    assert (folder / "inputs" / "sample-0.npz").is_file()


@pytest.mark.parametrize("level", ["disable", "basic", "extended", "all"])
def test_real_optimization_validation_persistence_and_export(setup, tmp_path, level):
    before = digest_file(setup.source / "tiny-cnn.onnx")
    result, folder, original = setup.run(optimization={"level": level})
    assert result["status"] == "Succeeded", result["error"]
    assert digest_file(setup.source / "tiny-cnn.onnx") == before
    verify_artifact(setup.project, ModelArtifact.model_validate(original["artifact"]))
    artifact = result["optimization"]["artifact"]
    assert artifact["id"] != result["source_model_id"]
    assert artifact["parent_model_id"] == result["source_model_id"]
    assert artifact["experiment_id"] == result["id"]
    assert artifact["target_environment"]["provider"] == "CPUExecutionProvider"
    assert result["validation"]["passed"] and result["validation"]["samples"] == 3
    assert result["baseline"]["conditions_hash"] == result["optimized"]["conditions_hash"]
    assert result["baseline"]["input_digest"] == result["optimized"]["input_digest"]
    assert all(state == "Succeeded" for state in result["phases"].values())
    assert result["optimization"]["applied_passes"] is None
    assert ModelTasks().load(setup.project, artifact["id"])["artifact"] == artifact
    loaded = OptimizationTasks().load(setup.project, result["id"])
    assert loaded == Experiment.model_validate(result).model_dump()
    exported = export_experiment(setup.project, loaded, tmp_path)
    report = json.loads(Path(exported["json"]).read_text())
    assert report["baseline"]["latencies_ms"] == result["baseline"]["latencies_ms"]
    assert (
        digest_file(Path(exported["directory"]) / exported["models"]["optimized"])
        == artifact["sha256"]
    )
    assert "Numerical consistency" in Path(exported["markdown"]).read_text()
    with pytest.raises(FileExistsError):
        export_experiment(setup.project, loaded, tmp_path)


def test_multi_input_output_npz_and_boolean_exact(setup):
    files = []
    for i in range(3):
        path = setup.source / f"sample-{i}.npz"
        np.savez(path, x=np.full((2, 4), i, np.float32), y=np.full((2, 4), i + 1, np.float32))
        files.append(str(path))
    result, _, _ = setup.run(
        "multi-input-output",
        inputs={"mode": "npz", "files": files},
        benchmark={"runs": 5, "warmup": 0},
    )
    assert result["status"] == "Succeeded", result["error"]
    assert len(result["validation"]["outputs"]) == 9
    assert any(v["comparison"] == "exact integer/boolean" for v in result["validation"]["outputs"])
    assert result["baseline"]["samples_per_second"] is None
    assert result["baseline"]["warmup_ms"] == 0


@pytest.mark.parametrize("level", ["disable", "basic", "extended", "all"])
def test_external_weights_are_reserialized_into_independent_artifact(setup, tmp_path, level):
    before = digest_file(setup.source / "weights.bin")
    result, _, _ = setup.run("external-cnn", optimization={"level": level})
    assert result["status"] == "Succeeded", result["error"]
    assert result["validation"]["passed"]
    artifact = result["optimization"]["artifact"]
    assert digest_file(setup.source / "weights.bin") == before
    assert (
        artifact["external_files"]
        and artifact["external_files"][0]["path"] == "optimized.onnx.data"
    )
    exported = export_experiment(setup.project, result, tmp_path)
    assert (Path(exported["directory"]) / "optimized" / "optimized.onnx.data").is_file()


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0.0, 0.00001, True),
        (0.0, 0.01, False),
        (10.0, 10.001, True),
        (np.nan, np.nan, False),
        (np.inf, np.inf, True),
        (np.inf, -np.inf, False),
    ],
)
def test_float_tolerance_nonfinite_and_zero_reference(a, b, expected):
    result = compare_tensor(np.array([a]), np.array([b]), atol=1e-5, rtol=1e-3)
    assert result["passed"] is expected
    json.dumps(result, allow_nan=False)


def test_integer_bool_empty_dtype_and_unsupported_output():
    assert not compare_tensor(
        np.array([2**63], dtype=np.uint64), np.array([2**63 + 1], dtype=np.uint64), 1, 1
    )["passed"]
    assert not compare_tensor(np.array([True]), np.array([False]), 1, 1)["passed"]
    assert compare_tensor(np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0, 0)[
        "passed"
    ]
    assert not compare_tensor(np.ones(1, np.float32), np.ones(1, np.float64), 1, 1)["passed"]
    assert not compare_tensor(np.ones(1, np.complex64), np.ones(1, np.complex64), 1, 1)["passed"]
    assert not compare_tensor({}, {}, 1, 1)["passed"]


@pytest.mark.parametrize(
    "mean,outcome",
    [
        (5, "Speedup"),
        (10, "Unchanged"),
        (20, "Regression"),
        (0, "Unavailable"),
        (None, "Unavailable"),
    ],
)
def test_honest_comparison(mean, outcome):
    a = {"conditions_hash": "same", "mean_ms": 10, "requests_per_second": 100, "model_bytes": 100}
    b = {"conditions_hash": "same", "mean_ms": mean, "requests_per_second": 50, "model_bytes": 120}
    result = compare_benchmarks(a, b)
    assert result["outcome"] == outcome
    assert result["size_change_percent"] == pytest.approx(20)
    b["conditions_hash"] = "different"
    assert compare_benchmarks(a, b)["speedup"] is None
    assert compare_benchmarks(None, b)["outcome"] == "Unavailable"


def test_inputs_no_guessed_shapes_repeatable_and_exact_validation(tmp_path):
    meta = [SimpleNamespace(name="x", type="tensor(float)", shape=["B", 4])]
    with pytest.raises(ValueError, match="dynamic shape"):
        prepare_inputs(meta, InputConfig(), 2, tmp_path / "a", BenchmarkConfig())
    config = InputConfig(shapes={"x": [2, 4]}, seed=3)
    a, _ = prepare_inputs(meta, config, 2, tmp_path / "a", BenchmarkConfig())
    b, _ = prepare_inputs(meta, config, 2, tmp_path / "b", BenchmarkConfig())
    assert np.array_equal(a[0]["x"], b[0]["x"]) and not np.array_equal(a[0]["x"], a[1]["x"])
    for feed, error in [
        ({"z": np.ones((2, 4), np.float32)}, "names/count"),
        ({"x": np.ones((2, 3), np.float32)}, "shape mismatch"),
        ({"x": np.ones((2, 4), np.float64)}, "dtype must"),
    ]:
        with pytest.raises(ValueError, match=error):
            check_feed(meta, feed)
    path = tmp_path / "pickle.npz"
    np.savez(path, x=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_npz(path)
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        payload, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)}
    )
    path = tmp_path / "invalid-header.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.npy", payload.getvalue())
    with pytest.raises(ValueError, match="payload/input budget"):
        load_npz(path)


@pytest.mark.parametrize(
    "name,inputs,error",
    [
        ("custom", {"shapes": {"x": [2, 4]}}, "not a registered"),
        ("tiny-cnn", {"shapes": {"image": [2, 9, 8, 8]}}, "shape mismatch"),
        ("tiny-cnn", {}, "dynamic shape"),
    ],
)
def test_pipeline_compatibility_and_input_errors_are_durable(setup, name, inputs, error):
    result, folder, _ = setup.run(name, inputs=inputs)
    assert result["status"] == "Failed"
    assert error.lower() in result["error"].lower()
    assert result["optimized"] is None and result["validation"] is None
    assert json.loads((folder / "experiment.json").read_text())["error"] == result["error"]


def test_bad_optimizer_file_fails_without_false_validation(setup, monkeypatch):
    def corrupt(self, path, output, config, benchmark_config, progress):
        output.write_bytes(b"corrupt model")
        return {"duration_ms": 1, "environment": {}}

    monkeypatch.setattr(OrtCPUBackend, "optimize", corrupt)
    result, _, _ = setup.run()
    assert result["status"] == "Failed" and result["validation"] is None
    assert result["phases"]["optimize"] == "Failed"


def wait_job(tasks, id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = tasks.snapshot(id)
        if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            tasks.threads[id].join(timeout=2)
            return job
        time.sleep(0.02)
    raise AssertionError("Lab job did not terminate")


def test_async_lifecycle_cancel_crash_and_history(setup):
    events = []
    tasks = OptimizationTasks(events.append)
    source = setup.imported()
    params = {
        "project": str(setup.project),
        "config": {
            "source_model_id": source["artifact"]["id"],
            "inputs": {"shapes": {"image": [2, 3, 8, 8]}},
            "benchmark": {"runs": 5},
        },
    }
    try:
        job = tasks.start(params)
        completed = wait_job(tasks, job["id"])
        assert completed["status"] == "Succeeded", completed["error"]
        assert {e["data"]["status"] for e in events} >= {"Pending", "Validating", "Succeeded"}
        assert OptimizationTasks().history(setup.project)[0]["id"] == job["id"]
        job = tasks.start(params)
        tasks.stop(job["id"])
        cancelled = wait_job(tasks, job["id"])
        assert cancelled["status"] == "Cancelled"
        assert OptimizationTasks().load(setup.project, job["id"])["status"] == "Cancelled"
        job = tasks.start(params)
        deadline = time.monotonic() + 3
        while job["id"] not in tasks.processes and time.monotonic() < deadline:
            time.sleep(0.01)
        tasks.processes[job["id"]].kill()
        crashed = wait_job(tasks, job["id"])
        assert crashed["status"] == "Failed" and "exited" in crashed["error"]
        loaded = OptimizationTasks().load(setup.project, job["id"])
        assert loaded["status"] == "Failed"
    finally:
        tasks.shutdown()


def test_interrupted_history_and_source_hash_guard(setup):
    result, folder, _ = setup.run(mode="baseline")
    result["status"], result["success"] = "Running", False
    (folder / "experiment.json").write_text(json.dumps(result))
    RunStore(setup.project).set_metadata(
        f"optimization:{folder.name}", {"id": folder.name, "state": "RUNNING"}
    )
    loaded = OptimizationTasks().load(setup.project, folder.name)
    assert loaded["status"] == "Failed" and "interrupted" in loaded["error"]
    path = setup.project / result["original"]["file_path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        verify_artifact(setup.project, ModelArtifact.model_validate(result["original"]))
    with pytest.raises(ValueError, match="Invalid experiment"):
        OptimizationTasks().load(setup.project, "../escape")


def test_unimplemented_backends_are_not_available():
    available = capabilities()
    assert available[0]["available"] and available[0]["levels"]
    assert available[1]["implemented"] and not available[1]["available"]
    assert available[1]["environment"]["status"] == "Not Installed"
    assert not available[2]["available"] and not available[2]["implemented"]
    schema = Path(__file__).resolve().parents[2] / "schemas" / "optimization-experiment.schema.json"
    assert json.loads(schema.read_text()) == Experiment.model_json_schema()


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_during_inference_and_actual_timeout(setup, cancel):
    tasks = OptimizationTasks()
    source = setup.imported()
    try:
        job = tasks.start(
            {
                "project": str(setup.project),
                "config": {
                    "source_model_id": source["artifact"]["id"],
                    "inputs": {"shapes": {"image": [512, 3, 8, 8]}},
                    "benchmark": {"runs": 100000, "warmup": 10000},
                    "timeout": 3 if not cancel else 30,
                },
            }
        )
        deadline = time.monotonic() + 10
        if cancel:
            while time.monotonic() < deadline:
                current = tasks.snapshot(job["id"])
                if current.get("phase") == "baseline":
                    break
                time.sleep(0.02)
            assert current.get("phase") == "baseline"
            tasks.stop(job["id"])
        ended = wait_job(tasks, job["id"])
        assert ended["status"] == ("Cancelled" if cancel else "Failed")
        assert not ended["result"]["success"]
        if not cancel:
            assert "timed out" in ended["error"]
        # The subprocess has closed its pipes; a venv interpreter child cannot
        # keep running and publish a later success after cancellation/timeout.
        saved = OptimizationTasks().load(setup.project, job["id"])
        assert saved["status"] == ended["status"]
    finally:
        tasks.shutdown()


def test_disposable_process_exits_when_supervisor_dies(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    startup = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    parent = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import os,time; print(os.getpid(),flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        **startup,
    )
    child = None
    try:
        owner = int(parent.stdout.readline())
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "from opweaver.process_owner import watch_owner; import sys,time; watch_owner(int(sys.argv[1])); print('ready',flush=True); time.sleep(30)",
                str(owner),
            ],
            stdout=subprocess.PIPE,
            **startup,
        )
        assert child.stdout.readline().strip() == b"ready"
        terminate_process(parent)
        parent.wait(timeout=5)
        child.communicate(timeout=5)
        assert child.returncode == 72
    finally:
        terminate_process(parent)
        parent.communicate(timeout=5)
        if child:
            terminate_process(child)
            child.communicate(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing transient error")
def test_atomic_report_replace_retries_temporary_file_lock(tmp_path, monkeypatch):
    source, target = tmp_path / "report.tmp", tmp_path / "report.json"
    source.write_text('{"status":"Succeeded"}')
    target.write_text('{"status":"Running"}')
    replace = os.replace
    attempts = []

    def locked_then_available(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("WinError 5: another reader holds the report")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", locked_then_available)
    atomic_replace(source, target)
    assert len(attempts) == 3 and json.loads(target.read_text())["status"] == "Succeeded"
