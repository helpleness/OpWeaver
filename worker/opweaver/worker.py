"""JSON Lines RPC v1 over stdio. stdout is protocol-only; stderr is diagnostics."""

import json
import sys
import threading
import traceback
from pathlib import Path

import torch

from .builder import GraphModule
from .code_tasks import CaptureTasks
from .codegen import generate
from .ir import Graph
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

    def dispatch(self, method: str, params: dict):
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


if __name__ == "__main__":
    main()
