"""Real independent Olive workflows plus bounded contract/failure tests."""

import importlib.metadata
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import onnx
import pytest
from onnx import helper as h
from opweaver import olive_runner
from opweaver.benchmark import digest_file
from opweaver.lab_runner import execute
from opweaver.lab_tasks import OptimizationTasks
from opweaver.lab_types import OptimizationConfig, OptimizationEnvironment
from opweaver.model_artifact import ModelArtifact
from opweaver.model_tasks import ModelTasks
from opweaver.olive_adapter import OliveBackend, preview, save_workflow
from opweaver.onnx_examples import generate_examples, model, value
from opweaver.onnx_import import import_onnx
from opweaver.optimization_environment import configure, interpreter, invoke, probe, settings
from opweaver.storage import RunStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def olive():
    path = os.environ.get("OPWEAVER_OLIVE_PYTHON", str(ROOT / ".tools" / "olive-env"))
    if not Path(path).exists():
        pytest.skip("Independent Olive environment not installed; run scripts/setup_olive.ps1")
    result = probe({"python_path": path}, force=True)
    assert result["status"] == "Available", result
    return {"python_path": str(interpreter(path))}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    generate_examples(root / "source")
    onnx.save(
        model(
            [
                h.make_node("Identity", ["x"], ["a"]),
                h.make_node("Identity", ["a"], ["b"]),
                h.make_node("Relu", ["b"], ["y"]),
            ],
            [value("x", [2, 4])],
            [value("y", [2, 4])],
        ),
        root / "source" / "redundant.onnx",
    )
    return root


def configuration(project, environment, name="redundant"):
    folder = project / ".opweaver" / "imports" / uuid.uuid4().hex
    report = import_onnx(project / "source" / f"{name}.onnx", project, folder)
    (folder / "result.json").write_text(json.dumps(report), encoding="utf-8")
    return {
        "source_model_id": report["artifact"]["id"],
        "inputs": {"shapes": {"x": [2, 4]}},
        "benchmark": {"runs": 12, "warmup": 2},
        "optimization": {
            "backend_id": "microsoft.olive",
            "environment": environment,
            "workflow": {
                "id": "two_pass",
                "passes": [
                    {"id": "first", "type": "OnnxPeepholeOptimizer"},
                    {"id": "disabled", "type": "OnnxPeepholeOptimizer", "enabled": False},
                    {
                        "id": "second",
                        "type": "OnnxOpVersionConversion",
                        "parameters": {"target_opset": 18},
                    },
                ],
                "output": {"model_name": "result.onnx", "external_data_threshold": 0},
            },
        },
    }


def run(project, config):
    folder = project / ".opweaver" / "optimizations" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    return execute({"project": str(project), "config": config}, folder), folder


def test_not_configured_and_bad_interpreter(tmp_path):
    assert probe()["status"] == "Not Installed"
    assert probe({"python_path": "C:/missing/python.exe"})["status"] == "Environment Error"
    configure(tmp_path, {"python_path": "C:/missing/python.exe"})
    assert probe(settings(tmp_path))["status"] == "Environment Error"
    with pytest.raises(ValueError, match="independent"):
        interpreter(sys.executable)


def test_actual_empty_venv_with_spaces(tmp_path):
    path = tmp_path / "empty environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(path)], check=True, capture_output=True
    )
    result = probe({"python_path": str(path)}, force=True)
    assert result["status"] == "Not Installed" and not result["available"]
    assert result["isolated_venv"] and result["versions"]["olive-ai"] is None


def test_version_and_dependency_classification(monkeypatch):
    original = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.12.1" if name == "olive-ai" else original(name),
    )
    assert olive_runner.environment_probe()["status"] == "Version Incompatible"
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.13.0" if name == "olive-ai" else original(name),
    )
    original_import = olive_runner.importlib.import_module

    def missing(name):
        if name == "onnxscript":
            raise ModuleNotFoundError("dependency removed for classification test")
        return original_import(name)

    monkeypatch.setattr(olive_runner.importlib, "import_module", missing)
    result = olive_runner.environment_probe()
    assert result["status"] == "Missing Dependency"
    assert result["errors"][0]["dependency"] == "onnxscript"


def test_environment_snapshot_and_real_native_validation(project, olive):
    configure(project, olive)
    assert settings(project)["python_path"] == olive["python_path"]
    config = configuration(project, None)
    checked = preview(project, config)
    native = checked["native_workflow"]
    assert list(native["passes"]) == ["first", "second"]
    assert native["passes"]["second"]["config"]["target_opset"] == 18
    assert native["passes"]["first"]["config"]["size_threshold"] == 0
    assert native["input_model"]["config"]["model_path"].endswith("redundant.onnx")
    assert native["engine"]["search_strategy"] is False
    assert checked["config"]["optimization"]["environment"]["python_path"] == olive["python_path"]
    saved = save_workflow(project, config)
    assert RunStore(project).metadata_list("optimization-workflow:") == [saved]
    assert json.loads(json.dumps(native)) == native
    assert probe(olive)["telemetry_disabled"]
    assert importlib.util.find_spec("olive") is None  # no dependency injected into Worker


@pytest.mark.parametrize(
    "bad",
    ["unknown_pass", "duplicate_id", "unknown_parameter", "wrong_type", "empty", "shell_hook"],
)
def test_preflight_rejects_unavailable_and_bad_config(project, olive, bad):
    config = configuration(project, olive)
    workflow = config["optimization"]["workflow"]
    if bad == "unknown_pass":
        workflow["passes"][0]["type"] = "ImaginaryPass"
    elif bad == "duplicate_id":
        workflow["passes"][1]["id"] = "first"
    elif bad == "unknown_parameter":
        workflow["passes"][0]["parameters"] = {"user_script": "anything.py"}
    elif bad == "wrong_type":
        workflow["passes"][0]["parameters"] = {"onnxscript_optimize": "true"}
    elif bad == "empty":
        workflow["passes"] = []
    else:
        workflow["shell"] = "arbitrary command"
    with pytest.raises(ValueError):
        OptimizationTasks().start({"project": str(project), "config": config})
    assert not (project / ".opweaver" / "optimizations").exists()


@pytest.mark.parametrize("name", ["redundant", "external-cnn", "multi-input-output"])
def test_real_olive_workflow_validation_benchmark_export_reopen(project, olive, name, tmp_path):
    config = configuration(project, olive, name)
    config["inputs"]["shapes"] = (
        {"image": [2, 3, 8, 8]}
        if name == "external-cnn"
        else {"x": [2, 4], **({"y": [2, 4]} if name == "multi-input-output" else {})}
    )
    before = digest_file(project / "source" / f"{name}.onnx")
    result, folder = run(project, config)
    assert result["success"], result["error"]
    state = result["workflow_execution"]
    assert (
        state["status"] == "Succeeded" and state["environment"]["versions"]["olive-ai"] == "0.13.0"
    )
    assert [p["pass_type"] for p in state["executed_passes"]] == [
        "onnxpeepholeoptimizer",
        "onnxopversionconversion",
    ]
    assert len(state["requested_passes"]) == 3
    assert state["candidates"] and state["telemetry_disabled"]
    assert (folder / "olive" / "executed-workflow.json").is_file()
    assert "Running pass first" in (folder / "olive" / "stdout.log").read_text()
    assert result["optimization"]["backend_id"] == "microsoft.olive"
    assert result["optimization"]["artifact"]["file_path"].endswith("result.onnx")
    assert result["validation"]["passed"] and result["validation"]["samples"] == 3
    assert result["baseline"]["conditions_hash"] == result["optimized"]["conditions_hash"]
    assert result["comparison"]["outcome"] in ("Speedup", "Regression", "Unchanged")
    assert len(result["optimized"]["latencies_ms"]) == 12
    assert digest_file(project / "source" / f"{name}.onnx") == before
    if name == "redundant":
        assert result["graph_comparison"]["original_nodes"] == 3
        assert result["graph_comparison"]["optimized_nodes"] == 1
    if name == "multi-input-output":
        assert len(result["validation"]["outputs"]) == 9
    loaded = OptimizationTasks().load(project, result["id"])
    assert loaded["workflow_execution"] == state
    assert (
        ModelTasks().load(project, result["optimization"]["artifact"]["id"])["artifact"][
            "parent_model_id"
        ]
        == result["source_model_id"]
    )
    exported = OptimizationTasks().export(
        {"project": str(project), "id": result["id"], "directory": str(tmp_path)}
    )
    bundle = Path(exported["directory"])
    assert (bundle / "workflow" / "executed-workflow.json").is_file()
    assert (bundle / "workflow" / "stdout.log").read_bytes() == (
        folder / "olive" / "stdout.log"
    ).read_bytes()
    if name == "multi-input-output":
        config["inputs"] = {
            "mode": "npz",
            "files": [str(folder / "inputs" / f"sample-{i}.npz") for i in range(3)],
        }
        replay, _ = run(project, config)
        assert replay["success"], replay["error"]
        assert replay["validation"]["passed"]
        assert replay["baseline"]["input_digest"] == result["baseline"]["input_digest"]


def test_no_observed_transformation_is_not_faked(project, olive):
    config = configuration(project, olive)
    config["optimization"]["workflow"]["passes"] = [
        {
            "id": "noop",
            "type": "OnnxPeepholeOptimizer",
            "parameters": {
                "onnxscript_optimize": False,
                "onnxoptimizer_optimize": False,
                "fuse_reshape_operations": False,
            },
        }
    ]
    result, _ = run(project, config)
    assert result["success"], result["error"]
    assert result["workflow_execution"]["candidates"][0]["observed_graph_change"] is False
    assert result["graph_comparison"]["optimized_nodes"] == 3
    assert result["optimization"]["applied_passes"] is None


@pytest.mark.parametrize("fault", ["failure", "absent_result", "missing_artifact"])
def test_injected_external_process_failures_retain_evidence(project, olive, monkeypatch, fault):
    # Fault injection complements (and does not replace) the real workflow tests above.
    bridge = project / "fault_bridge.py"
    result = (
        {"success": False, "error": "injected Olive failure"}
        if fault == "failure"
        else {
            "success": True,
            "candidates": [{"id": "missing", "model_path": str(project / "not-created.onnx")}],
            "selected_candidate_id": "missing",
        }
    )
    bridge.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        + (
            "pass\n"
            if fault == "absent_result"
            else f"Path(sys.argv[3]).write_text(json.dumps({result!r}))\n"
        )
    )
    monkeypatch.setattr("opweaver.olive_adapter.bridge_path", lambda: bridge)
    config = configuration(project, olive)
    failed, folder = run(project, config)
    assert failed["status"] == "Failed" and not failed["success"]
    assert failed["optimization"] is None and failed["optimized"] is None
    assert failed["workflow_execution"]["native_workflow"]
    assert (folder / "olive" / "stdout.log").is_file()


def test_real_task_timeout_preserves_workflow(project, olive):
    config = configuration(project, olive)
    config["timeout"] = 8
    # Fresh Lab child probes the environment then starts a fresh Olive child;
    # eight seconds bounds that actual process tree, not a mocked optimizer.
    manager = OptimizationTasks()
    job = manager.start({"project": str(project), "config": config})
    manager.threads[job["id"]].join(30)
    result = manager.snapshot(job["id"])
    assert result["state"] == "FAILED" and "timed out" in result["error"]
    assert not result["result"]["success"]


def test_custom_domain_capability_constraint(project, olive):
    config = configuration(project, olive)
    adapter = OliveBackend(olive, {"model": {"opsets": {"vendor.custom": 1}}})
    assert not any(p["available"] for p in adapter.probe().passes)
    with pytest.raises(ValueError, match="unavailable"):
        adapter.validate(
            ModelArtifact.model_validate(
                ModelTasks().load(project, config["source_model_id"])["artifact"]
            ),
            OptimizationConfig.model_validate(config["optimization"]),
        )


def test_actual_olive_invalid_native_error(olive):
    with pytest.raises(ValueError):
        invoke(
            OptimizationEnvironment.model_validate(olive),
            "validate",
            {
                "native": {
                    "input_model": {"type": "ONNXModel", "config": {"model_path": "missing.onnx"}},
                    "engine": {"unknown_setting": True},
                }
            },
        )


def test_cancel_actual_olive_process(project, olive):
    config = configuration(project, olive)
    manager = OptimizationTasks()
    job = manager.start({"project": str(project), "config": config})
    folder = Path(job["artifact_dir"])
    deadline = time.monotonic() + 45
    while not (folder / "olive" / "process.json").is_file():
        assert time.monotonic() < deadline, manager.snapshot(job["id"])
        assert manager.snapshot(job["id"])["state"] not in ("FAILED", "COMPLETED")
        time.sleep(0.1)
    manager.stop(job["id"])
    manager.threads[job["id"]].join(15)
    result = manager.snapshot(job["id"])
    assert result["state"] == "CANCELLED"
    assert result["result"]["workflow_execution"]["status"] == "Cancelled"
    assert not result["result"]["success"]
    assert result["result"]["optimization"] is None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        pid = json.loads((folder / "olive" / "process.json").read_text())["pid"]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if handle:
            try:
                assert kernel.WaitForSingleObject(handle, 5000) == 0, (
                    "Nested Olive interpreter survived cancellation"
                )
            finally:
                kernel.CloseHandle(handle)
