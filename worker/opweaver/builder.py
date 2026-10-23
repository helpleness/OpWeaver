"""Compile validated Graph IR to registered, autograd-compatible eager modules."""

import torch
from torch import nn

from .ir import Graph, Port
from .registry import OperatorRegistry, builtin_registry
from .validation import validate


class GraphModule(nn.Module):
    def __init__(self, graph: Graph, registry: OperatorRegistry | None = None):
        super().__init__()
        registry = registry or builtin_registry()
        report = validate(graph, registry)
        if not report.valid:
            raise ValueError("\n".join(issue["message"] for issue in report.issues))
        self.graph = graph.model_copy(deep=True)
        self.order = report.order
        self.nodes = {node.id: node for node in self.graph.nodes}
        self.inputs = [n.id for n in graph.nodes if n.operator.id == "core.input"]
        self.outputs = [n.id for n in graph.nodes if n.operator.id == "core.output"]
        self.incoming = {(e.target, e.target_port): (e.source, e.source_port) for e in graph.edges}
        self.layers = nn.ModuleDict()
        for node in graph.nodes:
            if node.operator.id not in ("core.input", "core.output") and not node.share_with:
                op = registry.get(node.operator.id, node.operator.version)
                self.layers[node.id] = op.factory(op.parameters.model_validate(node.params))
        for node in graph.nodes:
            if node.share_with:
                self.layers[node.id] = self.layers[node.share_with]

    def forward(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(args) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} input tensors, got {len(args)}")
        values = {(id, "out"): value for id, value in zip(self.inputs, args)}
        symbols = {}

        def check_port(id: str, port: Port, value: torch.Tensor) -> None:
            spec = port.tensor
            if spec is None:
                return
            if len(value.shape) != len(spec.shape):
                raise ValueError(f"Port {id}:{port.name} rank contract failed")
            for expected, actual in zip(spec.shape, value.shape):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(f"Port {id}:{port.name} shape contract failed")
                if isinstance(expected, str):
                    if expected in symbols and symbols[expected] != actual:
                        raise ValueError(f"Port {id}:{port.name} symbolic shape contract failed")
                    symbols[expected] = actual
            if str(value.dtype).removeprefix("torch.") != spec.dtype:
                raise ValueError(f"Port {id}:{port.name} dtype contract failed")
            if spec.device != "any" and value.device.type != spec.device:
                raise ValueError(f"Port {id}:{port.name} device contract failed")

        for id, value in zip(self.inputs, args):
            params = self.nodes[id].params
            shape = params.get("shape", ["B", 8])
            if len(value.shape) != len(shape):
                raise ValueError(f"Input {id} expects rank {len(shape)}, got {value.ndim}")
            for expected, actual in zip(shape, value.shape):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(f"Input {id} expects shape {shape}, got {list(value.shape)}")
                if isinstance(expected, str):
                    if expected in symbols and symbols[expected] != actual:
                        raise ValueError(
                            f"Symbolic dimension {expected} is inconsistent across inputs"
                        )
                    symbols[expected] = actual
            if str(value.dtype).removeprefix("torch.") != params.get("dtype", "float32"):
                raise ValueError(f"Input {id} dtype mismatch")
            device = params.get("device", "any")
            if device != "any" and value.device.type != device:
                raise ValueError(f"Input {id} requires device {device}")
            check_port(id, self.nodes[id].outputs[0], value)
        results = {}
        for id in self.order:
            node = self.nodes[id]
            if node.operator.id == "core.input":
                continue
            inputs = [values[self.incoming[id, port.name]] for port in node.inputs]
            for port, value in zip(node.inputs, inputs):
                check_port(id, port, value)
            if node.operator.id == "core.output":
                results[id] = inputs[0]
                continue
            output = self.layers[id](*inputs)
            outputs = (output,) if isinstance(output, torch.Tensor) else tuple(output)
            if len(outputs) != len(node.outputs):
                raise ValueError(f"Operator {id} produced an incorrect number of tensors")
            for port, value in zip(node.outputs, outputs):
                check_port(id, port, value)
            values.update({(id, port.name): tensor for port, tensor in zip(node.outputs, outputs)})
        ordered = tuple(results[id] for id in self.outputs)
        return ordered[0] if len(ordered) == 1 else ordered
