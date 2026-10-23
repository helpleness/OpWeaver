"""Small pass contract. No quantization, compilation or benchmark claims."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .ir import Graph


@dataclass(frozen=True)
class PassSpec:
    id: str
    version: str
    input_type: str
    stage: Literal["pre_training", "training", "post_training", "export_ir", "backend"]
    backends: tuple[str, ...]
    operators: tuple[str, ...]
    precisions: tuple[str, ...]
    calibration: bool
    changes_parameters: bool
    changes_structure: bool
    requires_retraining: bool
    artifact_type: str
    validation: str


@dataclass
class PassResult:
    graph: Graph
    accepted: bool
    reason: str


class PassManager:
    def __init__(self):
        self.passes: dict[str, tuple[PassSpec, Callable[[Graph], Graph]]] = {}

    def register(self, spec: PassSpec, transform: Callable[[Graph], Graph]):
        if spec.id in self.passes:
            raise ValueError(f"Duplicate pass {spec.id}")
        self.passes[spec.id] = (spec, transform)

    def run(self, id: str, graph: Graph, verify: Callable[[Graph, Graph], bool]) -> PassResult:
        try:
            candidate = self.passes[id][1](graph.model_copy(deep=True))
            if not verify(graph.model_copy(deep=True), candidate.model_copy(deep=True)):
                return PassResult(
                    graph,
                    False,
                    "Validation rejected candidate; original graph retained",
                )
            return PassResult(candidate, True, "Validated")
        except Exception as exc:
            return PassResult(graph, False, f"Pass failed; original graph retained: {exc}")


def dead_code_elimination(graph: Graph) -> Graph:
    live = {node.id for node in graph.nodes if node.operator.id == "core.output"}
    changed = True
    while changed:
        before = len(live)
        live.update(edge.source for edge in graph.edges if edge.target in live)
        live.update(node.share_with for node in graph.nodes if node.id in live and node.share_with)
        changed = len(live) != before
    graph.nodes = [node for node in graph.nodes if node.id in live]
    graph.edges = [edge for edge in graph.edges if edge.source in live and edge.target in live]
    return graph


def verify_eager(original: Graph, candidate: Graph) -> bool:
    """CPU smoke validation with preserved weights/RNG. Not a proof for all inputs."""
    import torch

    from .builder import GraphModule

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        before, after = GraphModule(original).eval(), GraphModule(candidate).eval()
        if before.inputs != after.inputs or before.outputs != after.outputs:
            return False
        weights = before.state_dict()
        after.load_state_dict({key: weights[key] for key in after.state_dict()}, strict=True)
        for batch in (1, 4, 7):
            inputs = [
                torch.randn(
                    [
                        batch if d == "B" else d
                        for d in before.nodes[id].params.get("shape", ["B", 8])
                    ]
                )
                for id in before.inputs
            ]
            with torch.no_grad():
                torch.testing.assert_close(before(*inputs), after(*inputs), atol=1e-6, rtol=1e-5)
    return True
