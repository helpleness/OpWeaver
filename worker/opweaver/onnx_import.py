"""Read real ONNX structure without inference, Python execution or weight expansion."""

import math
import re
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import onnx
from onnx import AttributeProto, TensorProto

from .ir import Edge
from .model_artifact import ModelArtifact, OnnxReport
from .model_files import copy_file
from .source import AnalysisGraph, AnalysisNode

PROTO_LIMIT = 256 * 1024 * 1024
VIEW_NODE_LIMIT = 20_000
DTYPES = {
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "COMPLEX64": "complex64",
    "COMPLEX128": "complex128",
}
BITS = {
    "FLOAT": 32,
    "DOUBLE": 64,
    "FLOAT16": 16,
    "BFLOAT16": 16,
    "BOOL": 8,
    "COMPLEX64": 64,
    "COMPLEX128": 128,
}


def dtype_name(value):
    try:
        name = TensorProto.DataType.Name(value)
        return DTYPES.get(name, name.lower())
    except ValueError:
        return f"unknown({value})"


def value_info(value):
    result = {"name": value.name, "dtype": "unknown", "shape": None}
    kind = value.type.WhichOneof("value")
    result["type"] = kind or "unknown"
    if kind in ("tensor_type", "sparse_tensor_type"):
        tensor = getattr(value.type, kind)
        result["dtype"] = dtype_name(tensor.elem_type)
        if tensor.HasField("shape"):
            result["shape"] = [
                d.dim_value if d.HasField("dim_value") else d.dim_param or None
                for d in tensor.shape.dim
            ]
    else:
        result["type_description"] = str(value.type)[:2000]
    return result


def tensor_info(tensor, shape=None):
    shape = list(tensor.dims) if shape is None else list(shape)
    count = math.prod(shape)
    name = TensorProto.DataType.Name(tensor.data_type)
    bits = BITS.get(name)
    if bits is None:
        match = re.match(r"(?:U?INT|FLOAT)(\d+)", name)
        bits = int(match[1]) if match else None
    return {
        "name": tensor.name,
        "shape": shape,
        "dtype": dtype_name(tensor.data_type),
        "elements": count,
        "logical_bytes": (count * bits + 7) // 8 if bits else None,
        "external": tensor.data_location == TensorProto.EXTERNAL,
        "storage": dict((v.key, v.value) for v in tensor.external_data),
    }


def all_tensors(message):
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type:
            for child in value if field.is_repeated else [value]:
                yield from all_tensors(child)


def graphs(graph, prefix=""):
    yield prefix, graph
    for i, node in enumerate(graph.node):
        for attribute in node.attribute:
            children = [attribute.g] if attribute.type == AttributeProto.GRAPH else attribute.graphs
            for j, child in enumerate(children):
                yield from graphs(child, f"{prefix}{i}/{attribute.name}:{j}/")


def free_inputs(graph):
    local = {v.name for v in graph.input} | {v.name for v in graph.initializer}
    local |= {v.values.name for v in graph.sparse_initializer}
    local |= {name for n in graph.node for name in n.output if name}
    used = {name for n in graph.node for name in n.input if name}
    used |= {v.name for v in graph.output}
    for node in graph.node:
        for attr in node.attribute:
            children = [attr.g] if attr.type == AttributeProto.GRAPH else attr.graphs
            for child in children:
                used |= free_inputs(child)
    return used - local


def graph_summary(graph):
    return {
        "name": graph.name,
        "nodes": len(graph.node),
        "operators": dict(Counter(n.op_type for n in graph.node)),
        "inputs": [value_info(v) for v in graph.input],
        "outputs": [value_info(v) for v in graph.output],
        "captured_inputs": sorted(free_inputs(graph)),
        "note": "Subgraph metadata; control flow is preserved, not flattened",
    }


def attribute_value(attr):
    if attr.ref_attr_name:
        return {"ref_attr_name": attr.ref_attr_name}
    kind = attr.type
    if kind == AttributeProto.TENSOR:
        return tensor_info(attr.t)
    if kind == AttributeProto.TENSORS:
        return {
            "tensors": [tensor_info(t) for t in attr.tensors[:64]],
            "count": len(attr.tensors),
            "truncated": len(attr.tensors) > 64,
        }
    if kind == AttributeProto.SPARSE_TENSOR:
        return {**tensor_info(attr.sparse_tensor.values, attr.sparse_tensor.dims), "sparse": True}
    if kind == AttributeProto.SPARSE_TENSORS:
        return {
            "tensors": [
                {**tensor_info(t.values, t.dims), "sparse": True} for t in attr.sparse_tensors[:64]
            ],
            "count": len(attr.sparse_tensors),
            "truncated": len(attr.sparse_tensors) > 64,
        }
    if kind == AttributeProto.GRAPH:
        return graph_summary(attr.g)
    if kind == AttributeProto.GRAPHS:
        return [graph_summary(g) for g in attr.graphs]
    try:
        value = onnx.helper.get_attribute_value(attr)
    except ValueError:
        return {"unrecognized_attribute_type": attr.type, "name": attr.name}
    if isinstance(value, bytes):
        return {
            "preview": value[:4096].decode("utf-8", errors="replace"),
            "bytes": len(value),
            "truncated": len(value) > 4096,
        }
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, list):
        return {
            "values": [
                str(v)[:4096]
                if isinstance(v, bytes)
                or hasattr(v, "DESCRIPTOR")
                or isinstance(v, float)
                and not math.isfinite(v)
                else v
                for v in value[:256]
            ],
            "count": len(value),
            "truncated": len(value) > 256,
        }
    return str(value)[:4096]


def build_analysis(
    graph, tensors, top_weights, opsets, known, graph_id, graph_name, source_hash, prefix=""
):
    if not prefix and free_inputs(graph):
        raise ValueError(f"Unresolved top-level ONNX tensors: {sorted(free_inputs(graph))}")
    initializers = {w["name"]: w for w in top_weights}
    tensors = {**tensors, **initializers}
    nodes, edges, producers = [], [], {}

    def producer(name, node_id, port):
        if not name or name in producers:
            raise ValueError(f"Duplicate or empty ONNX tensor producer: {name}")
        producers[name] = (node_id, port)

    for i, name in enumerate(sorted(free_inputs(graph))):
        id = f"capture_{i}"
        producer(name, id, "out_0")
        meta = tensors.get(name, {"name": name, "shape": None, "dtype": "unknown"})
        nodes.append(
            AnalysisNode(
                id=id,
                label=name,
                operator="onnx.capture",
                kind="captured_input",
                outputs=["out_0"],
                tensors=meta,
                details={"output_tensors": {"out_0": meta}},
            )
        )
    for i, value in enumerate(graph.input):
        if value.name in initializers:
            continue
        id = f"input_{i}"
        producer(value.name, id, "out_0")
        nodes.append(
            AnalysisNode(
                id=id,
                label=value.name,
                operator="onnx.input",
                kind="graph_input",
                outputs=["out_0"],
                tensors=tensors.get(value.name, value_info(value)),
                details={"output_tensors": {"out_0": tensors.get(value.name, value_info(value))}},
            )
        )
    for i, weight in enumerate(top_weights):
        id = f"weight_{i}"
        producer(weight["name"], id, "out_0")
        nodes.append(
            AnalysisNode(
                id=id,
                label=weight["name"],
                operator="onnx.initializer",
                kind="initializer",
                outputs=["out_0"],
                tensors=weight,
                details={
                    "output_tensors": {"out_0": weight},
                    "overridable": any(v.name == weight["name"] for v in graph.input),
                },
            )
        )
    for i, node in enumerate(graph.node):
        id = f"onnx_{i}"
        explicit = [(f"in_{j}", name, False) for j, name in enumerate(node.input) if name]
        implicit = set()
        for attr in node.attribute:
            children = [attr.g] if attr.type == AttributeProto.GRAPH else attr.graphs
            for child in children:
                implicit |= free_inputs(child)
        explicit_names = set(node.input)
        inputs = [
            *explicit,
            *[
                (f"capture_{j}", name, True)
                for j, name in enumerate(sorted(implicit - explicit_names))
            ],
        ]
        input_meta, output_meta = {}, {}
        for port, name, capture in inputs:
            if name not in producers:
                raise ValueError(
                    f"Unresolved tensor '{name}' at node {node.name or i}; graph is not a valid topological ONNX graph"
                )
            source_id, source_port = producers[name]
            edges.append(
                Edge(
                    id=f"edge_{len(edges)}",
                    source=source_id,
                    source_port=source_port,
                    target=id,
                    target_port=port,
                )
            )
            input_meta[port] = {
                **tensors.get(name, {"name": name, "shape": None, "dtype": "unknown"}),
                "implicit_capture": capture,
            }
        for j, name in enumerate(node.output):
            if name:
                producer(name, id, f"out_{j}")
                output_meta[f"out_{j}"] = tensors.get(
                    name, {"name": name, "shape": None, "dtype": "unknown"}
                )
        status = known[(node.domain, node.op_type, node.overload)]
        nodes.append(
            AnalysisNode(
                id=id,
                label=node.name or node.op_type,
                operator=f"{node.domain or 'ai.onnx'}::{node.op_type}",
                kind="onnx_node",
                module_path=node.op_type,
                inputs=[p for p, _, _ in inputs],
                outputs=list(output_meta),
                parameters={a.name: attribute_value(a) for a in node.attribute},
                tensors=next(iter(output_meta.values()), {}),
                details={
                    "subgraphs": [
                        {"id": f"{prefix}{i}/{a.name}:{j}/", "name": child.name or a.name}
                        for a in node.attribute
                        for j, child in enumerate(
                            [a.g] if a.type == AttributeProto.GRAPH else a.graphs
                        )
                    ],
                    "name": node.name,
                    "op_type": node.op_type,
                    "domain": node.domain or "ai.onnx",
                    "opset": opsets.get(node.domain),
                    "schema_status": status,
                    "known": status != "unknown",
                    "doc_string": node.doc_string[:4096],
                    "input_tensors": input_meta,
                    "output_tensors": output_meta,
                },
            )
        )
    for i, value in enumerate(graph.output):
        if value.name not in producers:
            raise ValueError(f"ONNX output '{value.name}' has no producer")
        id = f"output_{i}"
        source_id, source_port = producers[value.name]
        edges.append(
            Edge(
                id=f"edge_{len(edges)}",
                source=source_id,
                source_port=source_port,
                target=id,
                target_port="in_0",
            )
        )
        nodes.append(
            AnalysisNode(
                id=id,
                label=value.name,
                operator="onnx.output",
                kind="graph_output",
                inputs=["in_0"],
                outputs=[],
                tensors=tensors[value.name],
                details={"input_tensors": {"in_0": tensors[value.name]}},
            )
        )
    if len(nodes) > VIEW_NODE_LIMIT:
        raise ValueError(
            f"Model exceeds the {VIEW_NODE_LIMIT:,}-node viewer limit; no partial graph was published"
        )
    return AnalysisGraph(
        id=graph_id,
        name=graph_name,
        origin="onnx",
        source_hash=source_hash,
        source_model_id=graph_id.removeprefix("onnx-").split(":")[0],
        nodes=nodes,
        edges=edges,
    )


def import_onnx(source, project, folder):
    source, project, folder = (
        Path(source).resolve(),
        Path(project).resolve(),
        Path(folder).resolve(),
    )
    if source.stat().st_size > PROTO_LIMIT:
        raise ValueError(
            "ONNX protobuf exceeds the 256 MiB analysis limit; external weights are supported"
        )
    if not folder.is_relative_to(project):
        raise ValueError("Artifact directory escapes the project")
    model_dir = folder / "model"
    copied = model_dir / source.name
    identity = copy_file(source, copied)
    model = onnx.load_model(copied, load_external_data=False)
    if not model.HasField("graph") or not model.ir_version or not model.graph.output:
        raise ValueError("Invalid ONNX model: graph, IR version and outputs are required")

    external = {}
    for tensor in all_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        fields = {v.key: v.value for v in tensor.external_data}
        if len(fields) != len(tensor.external_data):
            raise ValueError("Ambiguous external tensor metadata")
        location = fields.get("location", "")
        relative = PurePosixPath(location.replace("\\", "/"))
        if not location or relative.is_absolute() or ".." in relative.parts or ":" in location:
            raise ValueError("External weight location must stay inside the model directory")
        weight = (source.parent / str(relative)).resolve(strict=True)
        if not weight.is_relative_to(source.parent) or not weight.is_file() or weight == source:
            raise ValueError(
                "External weight symlink/path escapes the model directory or aliases the model"
            )
        offset, length = int(fields.get("offset", 0)), fields.get("length")
        if (
            offset < 0
            or offset > weight.stat().st_size
            or length is not None
            and (int(length) < 0 or offset + int(length) > weight.stat().st_size)
        ):
            raise ValueError("External tensor byte range is outside its weight file")
        normalized = relative.as_posix()
        if normalized not in external:
            external[normalized] = {
                "path": normalized,
                **copy_file(weight, model_dir / str(relative)),
            }

    diagnostics, unknown = [], Counter()
    opsets = {o.domain: o.version for o in model.opset_import}
    functions = {(f.domain, f.name, f.overload) for f in model.functions}
    known = {}
    for node in (n for _, g in graphs(model.graph) for n in g.node):
        signature = (node.domain, node.op_type, node.overload)
        try:
            onnx.defs.get_schema(node.op_type, opsets.get(node.domain, 0), node.domain)
            known[signature] = "schema"
        except onnx.defs.SchemaError:
            known[signature] = "local_function" if signature in functions else "unknown"
            if known[signature] == "unknown":
                unknown[f"{node.domain or 'ai.onnx'}::{node.op_type}"] += 1
    checker_status = "passed"
    try:
        onnx.checker.check_model(str(copied), full_check=False)
    except onnx.checker.ValidationError as exc:
        if "No Op registered" not in str(exc) or not unknown:
            raise ValueError(f"ONNX checker rejected the model: {exc}") from exc
        checker_status = "incomplete"
        diagnostics.append({"severity": "warning", "message": f"ONNX checker incomplete: {exc}"})
    if unknown:
        diagnostics.append(
            {
                "severity": "warning",
                "message": "Unregistered operator schemas retained: " + ", ".join(unknown),
            }
        )
    shape_status = "inferred"
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=False, data_prop=False
        )
    except Exception as exc:
        inferred, shape_status = model, "failed"
        diagnostics.append(
            {
                "severity": "warning",
                "message": f"Shape inference unavailable; declared metadata retained: {exc}",
            }
        )
    graph = model.graph
    tensors = {
        v.name: value_info(v)
        for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }
    weights = []
    for prefix, nested in graphs(graph):
        for tensor in nested.initializer:
            weights.append({**tensor_info(tensor), "graph_path": prefix, "sparse": False})
        for sparse in nested.sparse_initializer:
            weights.append(
                {
                    **tensor_info(sparse.values, sparse.dims),
                    "graph_path": prefix,
                    "sparse": True,
                    "stored_elements": math.prod(sparse.values.dims),
                }
            )
    top_weights = [w for w in weights if not w["graph_path"]]
    initializers = {w["name"]: w for w in top_weights}
    if len(initializers) != len(top_weights):
        raise ValueError("Duplicate ONNX initializer names")
    tensors.update(initializers)
    id = folder.name
    analysis = build_analysis(
        graph,
        tensors,
        top_weights,
        opsets,
        known,
        f"onnx-{id}",
        graph.name or source.stem,
        identity["sha256"],
    )
    subgraphs = {}
    for prefix, child in graphs(inferred.graph):
        if not prefix:
            continue
        child_tensors = {
            **tensors,
            **{v.name: value_info(v) for v in [*child.input, *child.value_info, *child.output]},
        }
        child_weights = [w for w in weights if w["graph_path"] == prefix]
        subgraphs[prefix] = build_analysis(
            child,
            child_tensors,
            child_weights,
            opsets,
            known,
            f"onnx-{id}:{prefix}",
            child.name or prefix,
            identity["sha256"],
            prefix,
        )
    all_nodes = [n for g in [analysis, *subgraphs.values()] for n in g.nodes]
    if len(all_nodes) > VIEW_NODE_LIMIT:
        raise ValueError(
            f"Model exceeds the {VIEW_NODE_LIMIT:,}-node viewer limit; no partial graph published"
        )
    missing = sum(
        1
        for n in all_nodes
        if n.kind == "onnx_node"
        for v in n.details["output_tensors"].values()
        if v.get("shape") is None or v.get("dtype") == "unknown"
    )
    if missing:
        shape_status = "partial" if shape_status != "failed" else shape_status
        diagnostics.append(
            {
                "severity": "warning",
                "message": f"{missing} operator outputs have unknown shape/dtype; unknown values are not inferred guesses",
            }
        )
    if any(
        a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
        for n in graph.node
        for a in n.attribute
    ):
        diagnostics.append(
            {
                "severity": "info",
                "message": "Control-flow subgraphs can be opened independently; they are not flattened. Implicit outer tensor dependencies are connected.",
            }
        )
    distribution = Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in graph.node)
    id = folder.name
    artifact = ModelArtifact(
        id=id,
        name=graph.name or source.stem,
        file_path=copied.relative_to(project).as_posix(),
        sha256=identity["sha256"],
        file_bytes=identity["file_bytes"],
        total_file_bytes=identity["file_bytes"] + sum(v["file_bytes"] for v in external.values()),
        external_files=list(external.values()),
        imported_at=time.time(),
        source_name=source.name,
        source_path=str(source),
        importer_id="onnx",
    )
    normalized = []
    for d in diagnostics:
        message = d["message"]
        code = (
            "UNKNOWN_OPERATOR"
            if "Unregistered" in message
            else "SHAPE_INFERENCE"
            if "Shape" in message or "shape/dtype" in message
            else "SUBGRAPHS"
            if "subgraphs" in message
            else "CHECKER_INCOMPLETE"
        )
        normalized.append(
            {
                **d,
                "code": code,
                "source": "onnx.analysis",
                "suggested_action": "Inspect the original schema and declared tensor metadata."
                if d["severity"] == "warning"
                else "",
            }
        )
    for n in all_nodes:
        if any(
            isinstance(v, dict) and "unrecognized_attribute_type" in v
            for v in n.parameters.values()
        ):
            normalized.append(
                {
                    "severity": "warning",
                    "code": "UNKNOWN_ATTRIBUTE",
                    "message": "An attribute type could not be decoded; its type identifier and original model are preserved.",
                    "node": n.id,
                    "source": "onnx.attributes",
                }
            )
        if n.kind == "onnx_node" and n.details.get("known") is False:
            normalized.append(
                {
                    "severity": "warning",
                    "code": "UNKNOWN_OPERATOR",
                    "message": f"Unregistered schema: {n.operator}",
                    "node": n.id,
                    "source": "onnx.schema",
                    "suggested_action": "Install a compatible runtime extension before attempting inference.",
                }
            )
        if n.kind == "onnx_node" and any(
            v.get("shape") is None or v.get("dtype") == "unknown"
            for v in n.details["output_tensors"].values()
        ):
            normalized.append(
                {
                    "severity": "warning",
                    "code": "UNKNOWN_SHAPE",
                    "message": "Output shape/dtype is unknown; no dimensions were guessed.",
                    "node": n.id,
                    "source": "onnx.shape_inference",
                }
            )
    if any(any(not isinstance(d, int) for d in (v.get("shape") or [])) for v in tensors.values()):
        normalized.append(
            {
                "severity": "info",
                "code": "DYNAMIC_SHAPE",
                "message": "Symbolic/unknown dimensions are preserved.",
                "source": "onnx.metadata",
            }
        )
    if external:
        normalized.append(
            {
                "severity": "info",
                "code": "EXTERNAL_DATA",
                "message": f"{len(external)} external weight files copied without expanding their tensor data.",
                "source": "onnx.external_data",
            }
        )
    analysis.diagnostics = normalized
    quantization = None
    if any(
        n.op_type in ("QuantizeLinear", "DequantizeLinear", "QGemm", "MatMulInteger", "ConvInteger")
        or n.op_type.startswith("QLinear")
        for n in graph.node
    ) or any(t.data_type in (TensorProto.INT8, TensorProto.UINT8) for t in graph.initializer):
        from .quantization_analysis import analyze_quantization

        quantization = analyze_quantization(copied)
    return OnnxReport(
        quantization=quantization,
        artifact=artifact,
        analysis=analysis,
        model={
            "ir_version": model.ir_version,
            "opsets": opsets,
            "producer": model.producer_name,
            "producer_version": model.producer_version,
            "model_version": model.model_version,
            "domain": model.domain,
            "doc_string": model.doc_string[:4096],
            "metadata": [{"key": p.key, "value": p.value[:4096]} for p in model.metadata_props],
            "local_functions": [f"{f.domain}::{f.name}" for f in model.functions],
            "parser_version": onnx.__version__,
            "overridable_initializers": [v.name for v in graph.input if v.name in initializers],
            "scope": "Read-only ONNX structure; no inference backend executed; no PyTorch source reconstruction",
        },
        inputs=[value_info(v) for v in graph.input if v.name not in initializers],
        outputs=[tensors[v.name] for v in graph.output],
        initializers=weights,
        statistics={
            "nodes": len(graph.node),
            "graph_nodes": len(analysis.nodes),
            "edges": len(analysis.edges),
            "operators": dict(distribution),
            "initializer_count": len(weights),
            "initializer_elements": sum(w["elements"] for w in weights),
            "parameters_note": "Initializer elements, including constants; ONNX does not identify trainable parameters",
            "logical_tensor_bytes": sum(w["logical_bytes"] for w in weights)
            if all(w["logical_bytes"] is not None for w in weights)
            else None,
            "unknown_operators": dict(unknown),
            "flops": None,
            "weight_dtypes": dict(
                Counter(
                    {
                        dtype: sum(w["elements"] for w in weights if w["dtype"] == dtype)
                        for dtype in {w["dtype"] for w in weights}
                    }
                )
            ),
            "subgraphs": len(subgraphs),
        },
        validation={
            "checker": checker_status,
            "shape_inference": shape_status,
            "runtime": "not_executed",
        },
        diagnostics=normalized,
        subgraphs=subgraphs,
    ).model_dump()
