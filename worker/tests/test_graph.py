import json

import pytest
import torch
from opweaver.builder import GraphModule
from opweaver.examples import mlp, node, residual
from opweaver.ir import Edge, Graph, TensorSpec
from opweaver.passes import PassManager, PassSpec, dead_code_elimination, verify_eager
from opweaver.registry import EmptyParams, OperatorPlugin, builtin_registry
from opweaver.validation import validate


def report(graph):
    return validate(graph, builtin_registry())


def test_round_trip_and_schema():
    graph = mlp()
    assert Graph.model_validate_json(graph.model_dump_json()) == graph
    assert "GraphMetadata" in Graph.model_json_schema()["$defs"]
    assert "positions" not in json.loads(graph.model_dump_json())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_node",
        "port",
        "cycle",
        "shape",
        "dtype",
        "duplicate",
        "params",
        "operator",
        "unconnected",
    ],
)
def test_invalid_graph(fault):
    graph = mlp()
    if fault == "missing_node":
        graph.edges[0].source = "ghost"
    elif fault == "port":
        graph.edges[0].source_port = "wrong"
    elif fault == "cycle":
        graph.edges[1].source = "relu"
    elif fault == "shape":
        graph.nodes[1].params["in_features"] = 999
    elif fault == "dtype":
        graph.nodes[0].params["dtype"] = "int64"
    elif fault == "duplicate":
        graph.nodes.append(graph.nodes[0])
    elif fault == "params":
        graph.nodes[1].params["out_features"] = -1
    elif fault == "operator":
        graph.nodes[1].operator.version = "999"
    elif fault == "unconnected":
        graph.edges.pop()
    assert not report(graph).valid
    assert report(graph).issues[0]["message"]


def test_topological_sort_independent_of_node_order():
    graph = residual()
    graph.nodes.reverse()
    result = report(graph)
    assert result.valid
    for edge in graph.edges:
        assert result.order.index(edge.source) < result.order.index(edge.target)


def test_port_tensor_contract():
    graph = mlp()
    graph.nodes[1].inputs[0].tensor = TensorSpec(shape=["B", 99])
    assert not report(graph).valid


def test_dynamic_port_contract_is_checked_at_runtime():
    graph = mlp()
    graph.nodes[1].inputs[0].tensor = TensorSpec(shape=[1, 8], device="cpu")
    model = GraphModule(graph)
    assert model(torch.randn(1, 8)).shape == (1, 3)
    with pytest.raises(ValueError, match="shape contract failed"):
        model(torch.randn(4, 8))


def test_forward_backward_and_optimizer_updates():
    model = GraphModule(residual())
    assert isinstance(model, torch.nn.Module)
    x = torch.randn(12, 8)
    before = model.layers["branch"].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    y = model(x)
    assert y.shape == (12, 3)
    y.square().mean().backward()
    assert model.layers["branch"].weight.grad is not None
    optimizer.step()
    assert not torch.equal(before, model.layers["branch"].weight)


def test_multiple_inputs_outputs():
    graph = Graph(
        id="multi",
        nodes=[
            node("x", "input"),
            node("y", "input"),
            node("sum", "add"),
            node("a", "output"),
            node("b", "output"),
        ],
        edges=[
            Edge(id="1", source="x", target="sum", target_port="a"),
            Edge(id="2", source="y", target="sum", target_port="b"),
            Edge(id="3", source="sum", target="a"),
            Edge(id="4", source="x", target="b"),
        ],
    )
    model = GraphModule(graph)
    x, y = torch.randn(2, 8), torch.randn(2, 8)
    a, b = model(x, y)
    assert torch.equal(a, x + y) and torch.equal(b, x)
    with pytest.raises(ValueError, match="inconsistent"):
        model(x, torch.randn(3, 8))


def test_shared_parameters():
    graph = residual()
    shared = node("shared", "linear", in_features=8, out_features=8)
    shared.share_with = "branch"
    graph.nodes.append(shared)
    graph.edges.append(Edge(id="extra", source="input", target="shared"))
    model = GraphModule(graph)
    assert model.layers["shared"].weight is model.layers["branch"].weight


def test_pass_rollback_and_dce():
    graph = mlp()
    graph.nodes.append(node("unused", "relu"))
    graph.edges.append(Edge(id="unused_edge", source="input", target="unused"))
    manager = PassManager()
    spec = PassSpec(
        "dce",
        "1",
        "GraphIR",
        "pre_training",
        ("pytorch.eager",),
        (),
        ("float32",),
        False,
        False,
        True,
        False,
        "GraphIR",
        "caller-provided numerical validation",
    )
    manager.register(spec, dead_code_elimination)
    result = manager.run("dce", graph, verify_eager)
    assert result.accepted and len(result.graph.nodes) == 5
    assert len(graph.nodes) == 6
    rejected = manager.run("dce", graph, lambda old, new: False)
    assert not rejected.accepted and rejected.graph == graph


def test_committed_schema_matches_authoritative_models():
    from pathlib import Path

    from opweaver.training import TrainingConfig

    path = Path(__file__).resolve().parents[2] / "schemas" / "graph-ir.schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == Graph.model_json_schema()
    config_path = path.with_name("training-config.schema.json")
    assert json.loads(config_path.read_text(encoding="utf-8")) == TrainingConfig.model_json_schema()
    from opweaver.source import AnalysisGraph, SourceMapping

    for name, model in [("analysis-graph", AnalysisGraph), ("source-mapping", SourceMapping)]:
        assert (
            json.loads(path.with_name(name + ".schema.json").read_text(encoding="utf-8"))
            == model.model_json_schema()
        )


def test_registered_custom_operator_with_two_outputs_and_autograd():
    from opweaver.ir import Node, OperatorSpec, Port

    class Fork(torch.nn.Module):
        def forward(self, value):
            return value * 2, value * 3

    registry = builtin_registry()
    registry.register(
        OperatorPlugin(
            "test.fork",
            "Fork",
            "Custom",
            ("in",),
            ("a", "b"),
            EmptyParams,
            lambda p: Fork(),
            lambda args, p: [args[0], args[0]],
            "Test two outputs",
        )
    )
    graph = Graph(
        id="fork",
        nodes=[
            node("input", "input"),
            Node(
                id="fork",
                operator=OperatorSpec(id="test.fork"),
                inputs=[Port(name="in")],
                outputs=[Port(name="a"), Port(name="b")],
            ),
            node("a", "output"),
            node("b", "output"),
        ],
        edges=[
            Edge(id="1", source="input", target="fork"),
            Edge(id="2", source="fork", source_port="a", target="a"),
            Edge(id="3", source="fork", source_port="b", target="b"),
        ],
    )
    x = torch.randn(2, 8, requires_grad=True)
    a, b = GraphModule(graph, registry)(x)
    (a.sum() + b.sum()).backward()
    assert torch.equal(x.grad, torch.full_like(x, 5))
