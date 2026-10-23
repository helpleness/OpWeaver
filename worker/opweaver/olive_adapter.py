"""Versioned local Olive adapter. Benchmarking remains in the Worker engine."""

import copy
import json
import os
import subprocess
import time
from pathlib import Path

from .lab_reports import verify_artifact, write_json
from .lab_types import BackendCapability, ExperimentConfig, OptimizationEnvironment
from .model_artifact import ModelArtifact
from .model_tasks import ModelTasks
from .optimization import OptimizationBackend
from .optimization_environment import (
    bridge_path,
    interpreter,
    invoke,
    local_environment,
    probe,
    settings,
)
from .process_owner import terminate_process
from .storage import RunStore


class OliveBackend(OptimizationBackend):
    id = "microsoft.olive"

    def __init__(self, environment=None, model=None):
        self.environment = environment
        self.model = model
        self.calibration = None

    def probe(self):
        found = probe(self.environment)
        passes = copy.deepcopy(found["passes"])
        # Only the standard local graph contract has been verified for these passes.
        unsupported = []
        if self.model:
            unsupported = [op for op in self.model["model"]["opsets"] if op not in ("", "ai.onnx")]
        for item in passes:
            if unsupported:
                item.update(
                    available=False,
                    reason="Custom operator domains require an unverified pass contract",
                )
        return BackendCapability(
            id=self.id,
            name="Microsoft Olive",
            available=found["available"],
            version=found["versions"].get("olive-ai"),
            formats=["onnx"],
            devices=["cpu"],
            precisions=["preserve"],
            dependencies=[
                "Independent Python; olive-ai==0.13.x, CPU ORT, onnxscript; see environment snapshot"
            ],
            reason=found["reason"],
            environment=found,
            passes=passes,
            note="Local ordered workflows; no evaluator, search, remote models or custom scripts. Exact rewrite rules are not inferred.",
        )

    def validate(self, artifact, config):
        if config.environment and config.environment.backend_id != self.id:
            raise ValueError("Optimization environment belongs to a different backend")
        found = OliveBackend(config.environment or self.environment, self.model).probe()
        if not found.available:
            raise ValueError(found.reason)
        if artifact.format != "onnx":
            raise ValueError("Olive local workflows require an ONNX artifact")
        workflow = config.workflow
        if not workflow or not any(p.enabled for p in workflow.passes):
            raise ValueError("Enable at least one Olive pass")
        ids = [p.id for p in workflow.passes]
        if len(ids) != len(set(ids)):
            raise ValueError("Workflow pass IDs must be unique")
        manifest = {p["type"]: p for p in found.passes}
        for item in workflow.passes:
            if item.type not in manifest:
                raise ValueError(f"Pass {item.type} is outside the verified local contract")
            capability = manifest[item.type]
            if item.enabled and not capability["available"]:
                raise ValueError(f"Pass {item.type} is unavailable: {capability.get('reason', '')}")
            parameters = {p["name"]: p for p in capability["parameters"]}
            for name, value in item.parameters.items():
                if name not in parameters:
                    raise ValueError(f"Unsupported {item.type} parameter: {name}")
                spec = parameters[name]
                kind = spec["type"]
                expected = {"boolean": bool, "integer": int, "enum": str, "strings": list}[kind]
                if type(value) is not expected or (
                    kind == "strings"
                    and any(type(v) is not str or not v or len(v) > 256 for v in value)
                ):
                    raise ValueError(f"{name} requires {kind}")
                if kind == "strings" and len(value) > 1024:
                    raise ValueError("Too many operator/node filters")
                if kind == "enum" and value not in spec["choices"]:
                    raise ValueError(f"Unsupported {name}: {value}")
                if kind == "strings" and spec.get("choices") and set(value) - set(spec["choices"]):
                    raise ValueError(f"Unknown operator types in {name}")
                if expected is int and not spec["minimum"] <= value <= spec["maximum"]:
                    raise ValueError(f"{name} is outside the installed pass range")
                if item.enabled and value and not spec.get("available", True):
                    raise ValueError(spec["reason"])
            if item.enabled and item.type == "OnnxStaticQuantization":
                if not config.calibration_dataset_id:
                    raise ValueError(
                        "Static quantization requires a registered calibration dataset"
                    )
                if (
                    item.parameters.get("per_channel")
                    and item.parameters.get("precision", "int8") != "int8"
                ):
                    raise ValueError("Verified CPU per-channel quantization requires int8 weights")
                activation = item.parameters.get("activation_type", "int8")
                weight = item.parameters.get("precision", "int8")
                if activation == "int8" and weight == "uint8":
                    raise ValueError(
                        "Signed activations with unsigned weights are unsupported by static quantization"
                    )
                if (
                    item.parameters.get("quant_format", "QDQ") == "QOperator"
                    and activation != "uint8"
                ):
                    raise ValueError(
                        "Verified CPU QOperator execution requires uint8 activations; use QDQ for int8 activations"
                    )
        return found

    def build_workflow(self, path, directory, config):
        found = self.probe()
        manifest = {p["type"]: p for p in found.passes}
        workflow = config.workflow
        passes = {}
        for item in workflow.passes:
            if not item.enabled:
                continue
            parameters = {p["name"]: p["default"] for p in manifest[item.type]["parameters"]}
            parameters.update(item.parameters)
            if item.type == "OnnxStaticQuantization":
                if not self.calibration:
                    raise ValueError("A verified calibration dataset is required")
                parameters.update(
                    quant_preprocess=False,
                    prepare_qdq_config=False,
                    calibration_providers=["CPUExecutionProvider"],
                    data_config={
                        "name": "opweaver_" + item.id,
                        "type": "OpWeaverNPZDataContainerV1",
                        "user_script": str(Path(__file__).with_name("olive_data.py")),
                        "load_dataset_config": {
                            "type": "opweaver_npz_manifest_v1",
                            "params": {
                                "manifest_path": self.calibration["manifest_path"],
                                "manifest_sha256": self.calibration["manifest_sha256"],
                                "usage_path": str(
                                    directory / (item.id + "-calibration-usage.json")
                                ),
                            },
                        },
                    },
                )
            parameters.update(
                save_as_external_data=workflow.output.save_external_data,
                all_tensors_to_one_file=True,
                external_data_name=workflow.output.model_name + ".data",
                size_threshold=workflow.output.external_data_threshold,
            )
            passes[item.id] = {"type": item.type, "config": parameters}
        local = {
            "type": "LocalSystem",
            "config": {
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]
            },
        }
        return {
            "workflow_id": workflow.id,
            "input_model": {"type": "ONNXModel", "config": {"model_path": str(path)}},
            "engine": {
                "host": local,
                "target": local,
                "search_strategy": False,
                "evaluate_input_model": False,
                "output_dir": str(directory / "output"),
                "cache_dir": str(directory / "cache"),
                "enable_shared_cache": False,
                "log_severity_level": workflow.log_level,
                "log_to_file": False,
            },
            "passes": passes,
        }

    def optimize(self, path, output, config, benchmark_config, progress):
        directory = output.parent / "olive"
        directory.mkdir(exist_ok=False)
        found = self.probe()
        native = self.build_workflow(path, directory, config)
        request = {
            "output_model_name": config.workflow.output.model_name,
            "native": native,
            "source": str(path),
            "requested_passes": [p.model_dump() for p in config.workflow.passes],
            "owner_pid": os.getpid(),
            "cancellation": str(output.parent / "cancel.request"),
        }
        write_json(directory / "request.json", request)
        write_json(directory / "native-workflow.json", native)
        state = {
            "environment": found.environment,
            "generic_config": config.model_dump(),
            "native_workflow": native,
            "requested_passes": request["requested_passes"],
            "status": "Running",
            "started_at": time.time(),
            "raw_logs": ["olive/stdout.log", "olive/stderr.log"],
        }
        write_json(directory / "execution.json", state)
        started = time.perf_counter_ns()
        startup = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        progress(
            None, "Olive workflow starting; raw logs follow (no percentage is supplied by Olive)"
        )
        with (
            (directory / "stdout.log").open("wb") as stdout,
            (directory / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                [
                    str(interpreter(config.environment.python_path)),
                    "-I",
                    "-u",
                    str(bridge_path()),
                    "run",
                    str(directory / "request.json"),
                    str(directory / "result.json"),
                ],
                cwd=directory,
                env=local_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                **startup,
            )
            offsets = {"stdout": 0, "stderr": 0}
            try:
                while True:
                    for stream in offsets:
                        with (directory / f"{stream}.log").open("rb") as source:
                            source.seek(offsets[stream])
                            chunk = source.read(32768)
                            offsets[stream] = source.tell()
                        if chunk:
                            progress(
                                None,
                                f"[Olive {stream}] "
                                + chunk.decode("utf-8", errors="replace").rstrip(),
                            )
                    if process.poll() is not None:
                        break
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    terminate_process(process)
        result_path = directory / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {
                "success": False,
                "error": f"Olive process exited {process.returncode} without a result",
            }
        )
        state.update(
            result,
            status="Succeeded" if result.get("success") else "Failed",
            finished_at=time.time(),
        )
        state["calibration_usage"] = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in directory.glob("*-calibration-usage.json")
        ]
        executed_path = directory / "executed-workflow.json"
        if executed_path.is_file():
            state["executed_workflow"] = json.loads(executed_path.read_text(encoding="utf-8"))
        passes_path = directory / "pass-results.json"
        if passes_path.is_file():
            state.update(json.loads(passes_path.read_text(encoding="utf-8")))
        write_json(directory / "execution.json", state)
        if process.returncode or not result.get("success"):
            raise ValueError(result.get("error", f"Olive process exited {process.returncode}"))
        if any(p.enabled and p.type == "OnnxStaticQuantization" for p in config.workflow.passes):
            if not state["calibration_usage"] or any(
                u["unique_batches"] != self.calibration["batch_count"]
                for u in state["calibration_usage"]
            ):
                raise ValueError("Olive did not consume every recorded calibration batch")
        candidate = next(
            (c for c in result["candidates"] if c["id"] == result["selected_candidate_id"]), None
        )
        if not candidate:
            raise ValueError("Olive returned no selected candidate")
        model_path = Path(candidate["model_path"]).resolve(strict=True)
        if not model_path.is_relative_to(directory.resolve()) or not model_path.is_file():
            raise ValueError("Olive candidate is missing or outside the experiment")
        if candidate["inference_config"]:
            raise ValueError(
                "Olive candidate requires unverified inference settings; identical CPU benchmarking cannot be guaranteed"
            )
        return {
            "model_path": model_path,
            "duration_ms": (time.perf_counter_ns() - started) / 1e6,
            "environment": {
                "provider": "CPUExecutionProvider",
                "optimization_environment": found.environment,
            },
            "backend_version": found.version,
            "workflow": state,
            "note": result["transformation_note"],
            "serialization_options": config.workflow.output.model_dump(),
        }


def workflow_config(project, raw):
    config = ExperimentConfig.model_validate(raw)
    if config.optimization.backend_id != OliveBackend.id:
        raise ValueError("Workflow editing requires Microsoft Olive")
    if config.optimization.environment is None:
        config.optimization.environment = OptimizationEnvironment.model_validate(settings(project))
    model = ModelTasks().load(project, config.source_model_id)
    adapter = OliveBackend(config.optimization.environment.model_dump(), model)
    adapter.validate(ModelArtifact.model_validate(model["artifact"]), config.optimization)
    from .datasets import dataset_pair

    pair = dataset_pair(project, config)
    adapter.calibration = pair["calibration"]
    if pair["calibration"] or pair["evaluation"]:
        from .benchmark import create_session
        from .datasets import DatasetFeeds
        from .lab_inputs import check_feed

        session, _ = create_session(
            verify_artifact(project, ModelArtifact.model_validate(model["artifact"])),
            config.benchmark,
        )
        for role in ("calibration", "evaluation"):
            if pair[role]:
                for feed in DatasetFeeds(pair[role]):
                    check_feed(session.get_inputs(), feed)
    if (
        any(
            p.enabled and p.type == "OnnxStaticQuantization"
            for p in config.optimization.workflow.passes
        )
        and not pair["evaluation"]
    ):
        raise ValueError(
            "Static quantization requires a separate evaluation dataset (or explicit reuse)"
        )
    path = verify_artifact(project, ModelArtifact.model_validate(model["artifact"]))
    return config, adapter, path


def preview(project, raw):
    config, adapter, path = workflow_config(project, raw)
    # Preview paths are illustrative, never passed through from editable JSON.
    directory = Path(project).resolve() / ".opweaver" / "optimizations" / "PREVIEW" / "olive"
    native = adapter.build_workflow(path, directory, config.optimization)
    result = invoke(config.optimization.environment, "validate", {"native": native})
    return {
        "config": config.model_dump(),
        "native_workflow": native,
        "validated_workflow": result["executed_config"],
        "note": "Validated preview only. Execution uses a new experiment directory and saves the actual resolved configuration.",
    }


def save_workflow(project, raw):
    validated = preview(project, raw)
    config = validated["config"]
    id = config["optimization"]["workflow"]["id"]
    record = {"id": id, "saved_at": time.time(), "config": config}
    RunStore(Path(project)).set_metadata(f"optimization-workflow:{id}", record)
    return record
