"""Explicit, CPU FX capture. Executed only inside a dedicated short-lived process."""

import importlib.util
import operator
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from .ir import Edge, Graph, GraphMetadata, Node, OperatorSpec, Port
from .registry import builtin_registry
from .source import (
    AnalysisGraph,
    AnalysisNode,
    digest,
    graph_hash,
    parse_ast,
    read_source,
    source_mapping,
)
from .validation import validate


def serializable(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "parameters": value.numel()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, fx.Node):
        return {"node": value.name}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def tensor_metadata(node):
    meta = node.meta.get("tensor_meta")
    if hasattr(meta, "shape"):
        return {
            "shape": list(meta.shape),
            "dtype": str(meta.dtype).removeprefix("torch."),
            "device": "cpu",
        }
    if isinstance(meta, (tuple, list)):
        return {
            "items": [
                {"shape": list(m.shape), "dtype": str(m.dtype)} for m in meta if hasattr(m, "shape")
            ]
        }
    return {}


def make_inputs(specs):
    if not specs:
        raise ValueError("Configure at least one example input")
    tensors = []
    for spec in specs:
        shape = spec["shape"]
        if not shape or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in shape):
            raise ValueError("Example input dimensions must be positive concrete integers")
        count = 1
        for dim in shape:
            count *= dim
        if count > 16_000_000:
            raise ValueError("Example input exceeds the 16M-element capture limit")
        dtype = spec.get("dtype", "float32")
        if dtype not in ("float32", "float64", "int64", "bool"):
            raise ValueError(f"Unsupported capture dtype {dtype}")
        tensors.append(
            torch.randn(shape, dtype=getattr(torch, dtype))
            if dtype.startswith("float")
            else torch.zeros(shape, dtype=getattr(torch, dtype))
        )
    return tensors


class SourceTracer(fx.Tracer):
    def __init__(self, file_path, registry):
        super().__init__()
        self.file_path = str(Path(file_path).resolve())
        self.registry = registry
        self.record_stack_traces = True

    def is_leaf_module(self, module, module_qualified_name):
        return any(
            type(module) is op.module_type for op in self.registry.operators()
        ) or super().is_leaf_module(module, module_qualified_name)

    def create_node(self, *args, **kwargs):
        node = super().create_node(*args, **kwargs)
        frame = sys._getframe(1)
        while frame:
            if frame.f_code.co_filename == self.file_path:
                node.meta["source_line"] = frame.f_lineno
                break
            frame = frame.f_back
        return node


def load_model(file_path, entry, kwargs, source=None):
    path = Path(file_path).resolve()
    if path.suffix.lower() != ".py" or not entry.isidentifier():
        raise ValueError("Select a Python file and a top-level model class/factory")
    spec = importlib.util.spec_from_file_location("opweaver_user_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    # Compile the exact saved source, avoiding timestamp-based stale .pyc reuse.
    exec(
        compile(
            (source if source is not None else read_source(path)).removeprefix("\ufeff"),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    constructor = getattr(module, entry, None)
    if constructor is None:
        raise ValueError(f"Entry '{entry}' was not found")
    model = constructor(**kwargs)
    if not isinstance(model, nn.Module):
        raise ValueError("Selected entry did not construct torch.nn.Module")
    return model.cpu().eval()


def capture(file_path: str, entry: str, kwargs: dict, specs: list[dict], registry=None) -> dict:
    registry = registry or builtin_registry()
    source = read_source(file_path)
    torch.manual_seed(42)
    tensors = make_inputs(specs)
    model = load_model(file_path, entry, kwargs, source)
    with torch.no_grad():
        reference = model(*(tensor.clone() for tensor in tensors))
    tracer = SourceTracer(file_path, registry)
    traced = fx.GraphModule(model, tracer.trace(model))
    with torch.no_grad():
        ShapeProp(traced).propagate(*tensors)
        outputs = traced(*tensors)
    analysis_nodes, analysis_edges, mapping, design_nodes, design_edges, diagnostics = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    reference_matches = True
    try:
        torch.testing.assert_close(reference, outputs, atol=1e-6, rtol=1e-5)
    except (AssertionError, TypeError, ValueError) as exc:
        reference_matches = False
        diagnostics.append(
            {
                "severity": "warning",
                "message": "FX output differs from the original module for the configured inputs; editable conversion is disabled. "
                + str(exc)[:500],
            }
        )
    translated, values, input_index, owners = {}, {}, 0, {}
    ids = getattr(model, "_opweaver_node_ids", {})
    input_ids = getattr(model, "_opweaver_input_ids", {})
    output_ids = getattr(model, "_opweaver_output_ids", [])
    expression_ids = getattr(model, "_opweaver_expression_ids", {})
    from .source import inspect_source

    inspect_source(source, file_path)
    import ast

    expressions_by_line = {}
    for statement in ast.walk(parse_ast(source)):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            if name in expression_ids:
                expressions_by_line[statement.lineno] = expression_ids[name]
    conversion_ok = reference_matches
    node_ids = set()
    for fx_node in traced.graph.nodes:
        target, params, op = str(fx_node.target), {}, None
        module_path = target if fx_node.op == "call_module" else ""
        id = f"fx_{fx_node.name}"
        if fx_node.op == "placeholder":
            id = input_ids.get(target, f"input_{fx_node.name}")
            op = registry.get("core.input")
            params = {
                "shape": specs[input_index].get("design_shape", specs[input_index]["shape"]),
                "dtype": specs[input_index].get("dtype", "float32"),
                "device": "any",
            }
            input_index += 1
        elif fx_node.op == "call_module":
            module = traced.get_submodule(target)
            op = next((op for op in registry.operators() if type(module) is op.module_type), None)
            id = ids.get(target, "module_" + re.sub(r"\W", "_", target))
            if op:
                try:
                    if getattr(module, "inplace", False) or (
                        isinstance(module, nn.GELU) and module.approximate != "none"
                    ):
                        raise ValueError(
                            "In-place/approximate settings have no lossless Design Graph adapter"
                        )
                    params = op.parameters.model_validate(op.module_parameters(module)).model_dump()
                except (ValueError, TypeError) as exc:
                    diagnostics.append({"severity": "warning", "message": str(exc), "node": id})
                    op = None
            if not op:
                params = {"module_type": type(module).__qualname__, "representation": repr(module)}
        elif fx_node.op == "call_function":
            functions = {
                operator.add: "core.add",
                torch.add: "core.add",
                operator.mul: "core.multiply",
                torch.mul: "core.multiply",
                torch.cat: "core.concat",
                torch.concat: "core.concat",
                torch.relu: "core.relu",
                torch.flatten: "core.flatten",
            }
            if fx_node.target in functions:
                op = registry.get(functions[fx_node.target])
                params = {}
                if op.id == "core.concat":
                    params = {
                        "dim": fx_node.kwargs.get(
                            "dim", fx_node.args[1] if len(fx_node.args) > 1 else 0
                        )
                    }
                if op.id == "core.flatten":
                    params = {
                        "start_dim": fx_node.kwargs.get(
                            "start_dim", fx_node.args[1] if len(fx_node.args) > 1 else 0
                        ),
                        "end_dim": fx_node.kwargs.get(
                            "end_dim", fx_node.args[2] if len(fx_node.args) > 2 else -1
                        ),
                    }
                if fx_node.kwargs.get("alpha", 1) != 1 or fx_node.kwargs.get("out") is not None:
                    op = None
            id = expressions_by_line.get(fx_node.meta.get("source_line"), id)
        elif fx_node.op == "call_method" and target == "flatten":
            op = registry.get("core.flatten")
            params = {
                "start_dim": fx_node.kwargs.get(
                    "start_dim", fx_node.args[1] if len(fx_node.args) > 1 else 0
                ),
                "end_dim": fx_node.kwargs.get(
                    "end_dim", fx_node.args[2] if len(fx_node.args) > 2 else -1
                ),
            }
        if id in node_ids:
            id = f"fx_{fx_node.name}"
        node_ids.add(id)
        values[fx_node] = id
        dependencies = list(fx_node.all_input_nodes)
        if fx_node.op == "output":
            flat_outputs = fx_node.args[0]
            flat_outputs = (flat_outputs,) if isinstance(flat_outputs, fx.Node) else flat_outputs
            if not isinstance(flat_outputs, (tuple, list)) or not all(
                isinstance(n, fx.Node) for n in flat_outputs
            ):
                conversion_ok = False
                diagnostics.append(
                    {
                        "severity": "warning",
                        "message": "Dictionary/nested outputs remain analysis-only",
                    }
                )
                flat_outputs = dependencies
            for index, dependency in enumerate(flat_outputs):
                output_id = output_ids[index] if index < len(output_ids) else f"output_{index}"
                analysis_nodes.append(
                    AnalysisNode(
                        id=output_id,
                        label=f"Output {index}",
                        operator="core.output",
                        kind="output",
                        inputs=["in"],
                        outputs=[],
                        tensors=tensor_metadata(dependency),
                    )
                )
                analysis_edges.append(
                    {
                        "id": f"edge_{len(analysis_edges)}",
                        "source": values[dependency],
                        "source_port": "out",
                        "target": output_id,
                        "target_port": "in",
                    }
                )
                design_nodes.append(
                    Node(
                        id=output_id,
                        label="Output",
                        operator=OperatorSpec(id="core.output"),
                        inputs=[Port(name="in")],
                    )
                )
                design_edges.append(
                    Edge(
                        id=f"edge_{len(design_edges)}", source=values[dependency], target=output_id
                    )
                )
                mapping.append(
                    source_mapping(
                        source,
                        file_path,
                        entry,
                        output_id,
                        call_line=fx_node.meta.get("source_line"),
                    )
                )
            continue
        ports = list(op.inputs) if op else [f"arg_{i}" for i in range(len(dependencies))]
        all_arguments_are_tensors = len(dependencies) == len(ports)
        # Deduplicated all_input_nodes loses repeated operands (x + x); recover ordered leaves.
        leaves = []
        fx.node.map_arg((fx_node.args, fx_node.kwargs), lambda n: leaves.append(n))
        if fx_node.op in ("call_function", "call_method"):
            all_arguments_are_tensors = len(leaves) == len(ports)
            if (
                op
                and op.id in ("core.add", "core.multiply")
                and not (
                    len(fx_node.args) == 2 and all(isinstance(a, fx.Node) for a in fx_node.args)
                )
            ):
                all_arguments_are_tensors = False
        else:
            leaves = dependencies
        if op and not all_arguments_are_tensors:
            op = None
            ports = [f"arg_{i}" for i in range(len(leaves))]
        if not op:
            conversion_ok = False
            diagnostics.append(
                {
                    "severity": "warning",
                    "message": f"{fx_node.op} {target} is captured but has no complete editable Design Graph adapter",
                    "node": id,
                }
            )
        label = (
            op.label
            if op
            else (
                target
                if fx_node.op != "call_module"
                else type(traced.get_submodule(target)).__name__
            )
        )
        analysis_nodes.append(
            AnalysisNode(
                id=id,
                label=label,
                operator=op.id if op else target,
                kind=fx_node.op,
                module_path=module_path,
                parameters=params
                if op
                else {
                    **params,
                    "args": serializable(fx_node.args),
                    "kwargs": serializable(fx_node.kwargs),
                },
                inputs=ports,
                tensors=tensor_metadata(fx_node),
            )
        )
        for index, dep in enumerate(leaves):
            analysis_edges.append(
                {
                    "id": f"edge_{len(analysis_edges)}",
                    "source": values[dep],
                    "source_port": "out",
                    "target": id,
                    "target_port": ports[index] if index < len(ports) else f"arg_{index}",
                }
            )
        mapping.append(
            source_mapping(
                source, file_path, entry, id, module_path, fx_node.meta.get("source_line")
            )
        )
        if op:
            share_with = owners.get(module_path) if module_path else None
            if module_path and not share_with:
                owners[module_path] = id
            design_nodes.append(
                Node(
                    id=id,
                    label=label,
                    operator=OperatorSpec(id=op.id),
                    params=params,
                    inputs=[Port(name=n) for n in op.inputs],
                    outputs=[Port(name=n) for n in op.outputs],
                    share_with=share_with,
                )
            )
            for port, dep in zip(op.inputs, leaves):
                design_edges.append(
                    Edge(
                        id=f"edge_{len(design_edges)}",
                        source=values[dep],
                        target=id,
                        target_port=port,
                    )
                )
            translated[fx_node] = op.id
    projected = Graph(
        id=f"code-{entry}",
        metadata=GraphMetadata(name=entry),
        nodes=design_nodes,
        edges=design_edges,
    )
    report = validate(projected, registry) if conversion_ok else None
    if report and not report.valid:
        diagnostics.extend(report.issues)
        conversion_ok = False
    analysis = AnalysisGraph(
        id=f"fx-{entry}",
        name=entry,
        source_hash=digest(source),
        nodes=analysis_nodes,
        edges=analysis_edges,
        diagnostics=diagnostics,
    )
    output_list = (reference,) if isinstance(reference, torch.Tensor) else reference
    stats = []
    if isinstance(output_list, (tuple, list)):
        stats = [
            {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "mean": t.float().mean().item(),
                "min": t.min().item(),
                "max": t.max().item(),
            }
            for t in output_list
            if isinstance(t, torch.Tensor)
        ]
    return {
        "success": True,
        "analysis": analysis.model_dump(),
        "design_graph": projected.model_dump() if conversion_ok else None,
        "source_hash": digest(source),
        "graph_hash": graph_hash(projected)
        if conversion_ok
        else digest(analysis.model_dump_json()),
        "source_map": [m.model_dump() for m in mapping],
        "validation": {
            "passed": reference_matches,
            "atol": 1e-6,
            "rtol": 1e-5,
            "scope": "Original module versus FX on configured inputs only",
        },
        "diagnostics": diagnostics,
        "forward": {
            "outputs": stats,
            "parameters": sum(p.numel() for p in model.parameters()),
            "mode": "eval",
            "device": "cpu",
        },
        "statistics": {
            "nodes": len(analysis_nodes),
            "operators": dict(Counter(n.operator for n in analysis_nodes)),
            "parameters": sum(p.numel() for p in model.parameters()),
            "file_bytes": Path(file_path).stat().st_size,
        },
        "capture": {
            "backend": "torch.fx",
            "pytorch": str(torch.__version__),
            "input_specs": specs,
            "specialized": "CPU eval; example-bound shapes; Python construction/static branches are specialized",
        },
    }
