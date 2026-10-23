"""Offline example factories, also used to generate committed fixtures."""

from .ir import Edge, Graph, GraphMetadata, Node, OperatorSpec, Port
from .registry import builtin_registry


def node(id: str, operator: str, **params) -> Node:
    op = builtin_registry().get(f"core.{operator}")
    return Node(
        id=id,
        label=op.label,
        operator=OperatorSpec(id=op.id),
        params=op.parameters.model_validate(params).model_dump(),
        inputs=[Port(name=name) for name in op.inputs],
        outputs=[Port(name=name) for name in op.outputs],
    )


def mlp() -> Graph:
    nodes = [
        node("input", "input"),
        node("hidden", "linear", in_features=8, out_features=16),
        node("relu", "relu"),
        node("head", "linear", in_features=16, out_features=3),
        node("output", "output"),
    ]
    return Graph(
        id="synthetic-mlp",
        metadata=GraphMetadata(name="Synthetic classifier"),
        nodes=nodes,
        edges=[
            Edge(id=f"e{i}", source=a.id, target=b.id)
            for i, (a, b) in enumerate(zip(nodes, nodes[1:]))
        ],
    )


def residual() -> Graph:
    nodes = [
        node("input", "input"),
        node("branch", "linear", in_features=8, out_features=8),
        node("relu", "relu"),
        node("sum", "add"),
        node("head", "linear", in_features=8, out_features=3),
        node("output", "output"),
    ]
    edges = [
        Edge(id="e1", source="input", target="branch"),
        Edge(id="e2", source="branch", target="relu"),
        Edge(id="e3", source="relu", target="sum", target_port="a"),
        Edge(id="e4", source="input", target="sum", target_port="b"),
        Edge(id="e5", source="sum", target="head"),
        Edge(id="e6", source="head", target="output"),
    ]
    return Graph(
        id="residual-mlp",
        metadata=GraphMetadata(name="Residual classifier"),
        nodes=nodes,
        edges=edges,
    )
