"""JSON Lines RPC v1 over stdio. stdout is protocol-only; stderr is diagnostics."""

import json
import sys
import threading
import traceback
from pathlib import Path

import torch

from .benchmark import compare_benchmarks
from .builder import GraphModule
from .code_tasks import CaptureTasks
from .codegen import generate
from .diagnostic_tasks import DiagnosticTasks
from .importers import builtin_importers
from .ir import Graph
from .lab_tasks import OptimizationTasks
from .model_tasks import ModelTasks
from .optimization import capabilities
from .registry import builtin_registry
from .source import inspect_source, patch_parameter
from .storage import RunStore
from .training import TrainingConfig, TrainingEngine, environment
from .validation import validate


class Worker:
    def __init__(self, send):
        self.send = send
        self.registry = builtin_registry()
        self.engine = TrainingEngine(lambda event: send({"protocol": 1, **event}))
        self.code_tasks = CaptureTasks(lambda event: send({"protocol": 1, **event}))
        self.model_tasks = ModelTasks(lambda event: send({"protocol": 1, **event}))
        self.lab_tasks = OptimizationTasks(lambda event: send({"protocol": 1, **event}))
        self.diagnostic_tasks = DiagnosticTasks(lambda event: send({"protocol": 1, **event}))

    def dispatch(self, method: str, params: dict):
        if method.startswith("diagnostic."):
            if method == "diagnostic.start":
                if self.engine.active() or any(
                    t.is_alive()
                    for manager in (self.code_tasks, self.model_tasks, self.lab_tasks)
                    for t in manager.threads.values()
                ):
                    raise ValueError("Stop other compute/import tasks before diagnostics")
                return self.diagnostic_tasks.start(params)
            if method == "diagnostic.status":
                return self.diagnostic_tasks.snapshot(params["id"])
            if method == "diagnostic.stop":
                return self.diagnostic_tasks.stop(params["id"])
            if method == "diagnostic.history":
                return self.diagnostic_tasks.history(params["project"])
            if method == "diagnostic.load":
                return self.diagnostic_tasks.load(params["project"], params["id"])
            if method == "diagnostic.export":
                return self.diagnostic_tasks.export(params)
            raise ValueError("Unknown diagnostic operation")
        if method in (
            "lab.start",
            "lab.dataset.create",
            "train.start",
            "forward",
            "model.import.start",
            "code.capture.start",
        ) and any(t.is_alive() for t in self.diagnostic_tasks.threads.values()):
            raise ValueError(
                "Stop the quantization diagnostic before starting other compute/import work"
            )
        if method.startswith("lab.dataset."):
            from .datasets import availability, create_dataset, load_dataset

            if method == "lab.dataset.create":
                if self.engine.active() or any(
                    t.is_alive()
                    for manager in (self.code_tasks, self.model_tasks, self.lab_tasks)
                    for t in manager.threads.values()
                ):
                    raise ValueError(
                        "Wait until compute/import tasks finish before preparing datasets"
                    )
                return create_dataset(params["project"], params["config"])
            if method == "lab.dataset.load":
                return load_dataset(params["project"], params["id"])
            if method == "lab.dataset.list":
                return [
                    {
                        **r,
                        "availability": availability(params["project"], {"evaluation": r})[
                            "evaluation"
                        ],
                    }
                    for r in RunStore(Path(params["project"])).metadata_list("dataset:")
                ]
            raise ValueError("Unknown dataset operation")
        if method == "lab.backends":
            from .optimization_environment import settings

            model = (
                ModelTasks().load(params["project"], params["source_model_id"])
                if params.get("source_model_id")
                else None
            )
            return capabilities(settings(params.get("project")), model)
        if method.startswith("lab.environment."):
            from .optimization_environment import configure, probe, settings

            if method == "lab.environment.configure":
                configure(params["project"], params["environment"])
            return probe(settings(params["project"]), force=True)
        if method == "lab.workflow.preview":
            from .olive_adapter import preview

            return preview(params["project"], params["config"])
        if method == "lab.workflow.save":
            from .olive_adapter import save_workflow

            return save_workflow(params["project"], params["config"])
        if method == "lab.workflow.list":
            return RunStore(Path(params["project"])).metadata_list("optimization-workflow:")
        if method == "lab.start":
            if self.engine.active() or any(
                t.is_alive()
                for manager in [self.code_tasks, self.model_tasks]
                for t in manager.threads.values()
            ):
                raise ValueError("Stop other compute/import tasks before benchmarking")
            return self.lab_tasks.start(params)
        if method == "lab.status":
            return self.lab_tasks.snapshot(params["id"])
        if method == "lab.stop":
            return self.lab_tasks.stop(params["id"])
        if method == "lab.history":
            return self.lab_tasks.history(params["project"])
        if method == "lab.load":
            return self.lab_tasks.load(params["project"], params["id"])
        if method == "lab.export":
            return self.lab_tasks.export(params)
        if method == "lab.compare":
            reference = self.lab_tasks.load(params["project"], params["reference_id"])
            candidate = self.lab_tasks.load(params["project"], params["candidate_id"])
            left = reference["optimized"] or reference["baseline"]
            right = candidate["optimized"] or candidate["baseline"]
            return {
                "reference_id": reference["id"],
                "candidate_id": candidate["id"],
                "reference": {k: v for k, v in (left or {}).items() if k != "latencies_ms"},
                "candidate": {k: v for k, v in (right or {}).items() if k != "latencies_ms"},
                "comparison": compare_benchmarks(left, right),
            }
        if method in ("train.start", "forward", "model.import.start", "code.capture.start") and any(
            t.is_alive() for t in self.lab_tasks.threads.values()
        ):
            raise ValueError(
                "Stop the Optimization Lab experiment before starting another compute task"
            )
        if method == "environment":
            return environment()
        if method == "registry":
            return self.registry.manifests()
        if method == "validate":
            return validate(Graph.model_validate(params["graph"]), self.registry).as_dict()
        if method == "code.inspect":
            return inspect_source(params["source"], params.get("file_path", "model.py"))
        if method == "code.generate":
            return generate(
                Graph.model_validate(params["graph"]),
                self.registry,
                params.get("file_path", "src/model.py"),
                params.get("class_name", "OpWeaverModel"),
            )
        if method == "code.patch":
            op = self.registry.get(params["operator_id"])
            op.parameters.model_validate({**params["parameters"], params["key"]: params["value"]})
            return patch_parameter(
                params["source"],
                params["class_name"],
                params["module_path"],
                params["key"],
                params["value"],
                params["expected_hash"],
            )
        if method == "code.capture.start":
            return self.code_tasks.start(params)
        if method == "code.capture.status":
            return self.code_tasks.snapshot(params["id"])
        if method == "code.capture.stop":
            return self.code_tasks.stop(params["id"])
        if method == "model.import.start":
            return self.model_tasks.start(params)
        if method == "model.importers":
            return builtin_importers().manifests()
        if method == "model.probe":
            return builtin_importers().probe(params["source"])
        if method == "model.import.status":
            return self.model_tasks.snapshot(params["id"])
        if method == "model.import.stop":
            return self.model_tasks.stop(params["id"])
        if method == "model.list":
            return self.model_tasks.history(params["project"])
        if method == "model.load":
            return self.model_tasks.load(params["project"], params["id"])
        if method == "forward":
            if self.engine.active():
                raise ValueError("Stop training before running a separate forward inspection")
            graph = Graph.model_validate(params["graph"])
            model = GraphModule(graph)
            tensors = []
            for id in model.inputs:
                shape = model.nodes[id].params.get("shape", ["B", 8])
                if any(isinstance(d, str) and d != "B" for d in shape):
                    raise ValueError(
                        "Forward preview resolves only symbol B; use the Python API for other dynamic dimensions"
                    )
                tensors.append(torch.randn([4 if d == "B" else d for d in shape]))
            model.eval()
            with torch.no_grad():
                output = model(*tensors)
            outputs = (output,) if isinstance(output, torch.Tensor) else output
            return {
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": "cpu",
                "outputs": [
                    {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "mean": t.float().mean().item(),
                        "min": t.min().item(),
                        "max": t.max().item(),
                    }
                    for t in outputs
                ],
            }
        if method == "train.start":
            return self.engine.start(
                Graph.model_validate(params["graph"]),
                TrainingConfig.model_validate(params["config"]),
                Path(params["project"]),
                params.get("resume"),
            )
        if method == "train.stop":
            return self.engine.stop(params["id"])
        if method == "train.status":
            return self.engine.snapshot(params["id"])
        if method == "train.history":
            return RunStore(Path(params["project"])).recover(set(self.engine.jobs))
        if method == "shutdown":
            self.diagnostic_tasks.shutdown()
            self.lab_tasks.shutdown()
            self.model_tasks.shutdown()
            self.code_tasks.shutdown()
            self.engine.shutdown()
            return {"stopped": True}
        raise ValueError(f"Unknown method: {method}")


def main():
    torch.set_num_threads(2)
    lock = threading.Lock()

    def send(value):
        with lock:
            sys.stdout.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            sys.stdout.flush()

    worker = Worker(send)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("protocol") != 1:
                raise ValueError("Unsupported RPC protocol; expected 1")
            result = worker.dispatch(request["method"], request.get("params", {}))
            send({"protocol": 1, "id": request["id"], "result": result})
            if request["method"] == "shutdown":
                return
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send(
                {
                    "protocol": 1,
                    "id": request.get("id"),
                    "error": {"code": type(exc).__name__, "message": str(exc)},
                }
            )
    worker.engine.shutdown()
    worker.code_tasks.shutdown()
    worker.model_tasks.shutdown()
    worker.lab_tasks.shutdown()
    worker.diagnostic_tasks.shutdown()


if __name__ == "__main__":
    main()
