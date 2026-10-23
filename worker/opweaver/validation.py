"""Deterministic graph validation; diagnostics retain node/edge locations."""

from collections import deque
from dataclasses import dataclass, field

from .ir import Graph, TensorSpec
from .registry import OperatorRegistry


@dataclass
class ValidationResult:
    issues: list[dict] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    tensors: dict[str, dict] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not any(issue["severity"] == "error" for issue in self.issues)

    def error(self, message: str, node: str | None = None, edge: str | None = None) -> None:
        self.issues.append({"severity": "error", "message": message, "node": node, "edge": edge})

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "issues": self.issues,
            "order": self.order,
            "tensors": self.tensors,
        }


def validate(graph: Graph, registry: OperatorRegistry) -> ValidationResult:
    result = ValidationResult()
    nodes = {node.id: node for node in graph.nodes}
    if len(nodes) != len(graph.nodes):
        result.error("Duplicate node IDs")
    if len({edge.id for edge in graph.edges}) != len(graph.edges):
        result.error("Duplicate edge IDs")
    if graph.modules or any(node.module_ref for node in graph.nodes):
        result.error(
            "Nested module execution is reserved for IR v1 extensions and is not implemented yet"
        )
    if graph.plugins:
        result.error(
            "External plugin loading requires explicit trust and is not enabled in this release"
        )
    operators, parameters = {}, {}
    for node in graph.nodes:
        try:
            op = registry.get(node.operator.id, node.operator.version)
            operators[node.id] = op
            parameters[node.id] = op.parameters.model_validate(node.params)
            if (
                tuple(port.name for port in node.inputs) != op.inputs
                or tuple(port.name for port in node.outputs) != op.outputs
            ):
                result.error(
                    f"{op.label} port declaration must be inputs={op.inputs}, outputs={op.outputs}",
                    node.id,
                )
            if node.share_with:
                owner = nodes.get(node.share_with)
                if (
                    node.operator.id in ("core.input", "core.output")
                    or not owner
                    or owner.share_with
                    or owner.id == node.id
                    or owner.operator != node.operator
                    or owner.params != node.params
                ):
                    result.error(
                        "Shared parameters require an unshared owner with identical operator and parameters",
                        node.id,
                    )
        except (ValueError, TypeError) as exc:
            result.error(str(exc), node.id)
    adjacency = {id: [] for id in nodes}
    indegree = {id: 0 for id in nodes}
    incoming = {}
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes:
            result.error(
                f"Edge references a missing node: {edge.source} → {edge.target}",
                edge=edge.id,
            )
            continue
        source, target = nodes[edge.source], nodes[edge.target]
        if edge.source_port not in [p.name for p in source.outputs] or edge.target_port not in [
            p.name for p in target.inputs
        ]:
            result.error("Edge references an invalid input/output port", edge.target, edge.id)
            continue
        key = (edge.target, edge.target_port)
        if key in incoming:
            result.error(
                f"Input port {edge.target_port} has multiple producers",
                edge.target,
                edge.id,
            )
        incoming[key] = edge
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(id for id in nodes if indegree[id] == 0)
    while queue:
        id = queue.popleft()
        result.order.append(id)
        for target in adjacency[id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(result.order) != len(nodes):
        result.error(
            "Cycle detected. IR v1 executes DAGs; use a future control-flow module for loops."
        )
    for node in graph.nodes:
        for port in node.inputs:
            if (node.id, port.name) not in incoming:
                result.error(f"Required input '{port.name}' is not connected", node.id)
    if not any(n.operator.id == "core.input" for n in graph.nodes):
        result.error("Graph requires at least one Input")
    if not any(n.operator.id == "core.output" for n in graph.nodes):
        result.error("Graph requires at least one Output")
    if not result.valid:
        return result

    def check_declared(actual: TensorSpec, declared: TensorSpec | None) -> None:
        if declared is None:
            return
        if actual.dtype != declared.dtype or len(actual.shape) != len(declared.shape):
            raise ValueError(
                f"Port tensor contract mismatch: inferred {actual}, declared {declared}"
            )
        if any(
            isinstance(a, int) and isinstance(b, int) and a != b
            for a, b in zip(actual.shape, declared.shape)
        ):
            raise ValueError(
                f"Port shape mismatch: inferred {actual.shape}, declared {declared.shape}"
            )
        if actual.device != declared.device and "any" not in (
            actual.device,
            declared.device,
        ):
            raise ValueError("Port device mismatch")

    inferred = {}
    for id in result.order:
        node, op = nodes[id], operators[id]
        try:
            args = []
            for port in node.inputs:
                edge = incoming[id, port.name]
                spec = inferred[edge.source, edge.source_port]
                check_declared(spec, port.tensor)
                args.append(spec)
                result.tensors[f"{id}:{port.name}"] = spec.model_dump()
            outputs = op.infer(args, parameters[id])
            if len(outputs) != len(node.outputs):
                raise ValueError("Operator shape function returned an incorrect number of outputs")
            for spec in args + outputs:
                if spec.dtype not in op.dtypes:
                    raise ValueError(
                        f"{op.label} does not support {spec.dtype}; supported: {op.dtypes}"
                    )
                if spec.device != "any" and spec.device not in op.devices:
                    raise ValueError(f"{op.label} does not support device {spec.device}")
            for port, spec in zip(node.outputs, outputs):
                check_declared(spec, port.tensor)
                inferred[id, port.name] = spec
                result.tensors[f"{id}:{port.name}"] = spec.model_dump()
        except KeyError:
            result.error("Cannot infer tensor because an upstream node is invalid", id)
        except (ValueError, TypeError) as exc:
            result.error(str(exc), id)
    return result
