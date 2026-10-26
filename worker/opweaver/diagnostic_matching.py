"""Conservative tensor identity: ORT QDQ matching plus recursive producer contracts."""

from collections import Counter

from onnx import TensorProto
from onnxruntime.quantization import qdq_loss_debug as debug
from onnxruntime.quantization.quant_utils import QUANT_INPUT_SUFFIX

from .diagnostic_types import LayerErrorRecord

FLOAT_TYPES = (TensorProto.FLOAT, TensorProto.FLOAT16)


def metadata(value):
    tensor = value.type.tensor_type
    shape = None
    if tensor.HasField("shape"):
        shape = [
            d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in tensor.shape.dim
        ]
    return {"shape": shape, "dtype": tensor.elem_type}


class GraphIndex:
    def __init__(self, model):
        self.model = model
        self.nodes = {f"onnx_{i}": n for i, n in enumerate(model.graph.node)}
        self.producers = {
            name: (id, n) for id, n in self.nodes.items() for name in n.output if name
        }
        self.initializers = {v.name: v for v in model.graph.initializer}
        self.inputs = {v.name for v in model.graph.input if v.name not in self.initializers}
        self.values = {
            v.name: metadata(v)
            for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]
        }
        self.opsets = {(v.domain or "ai.onnx"): v.version for v in model.opset_import}
        self.name_counts = Counter(n.name for n in model.graph.node)

    def unwrap(self, name):
        visited = set()
        while name in self.producers:
            if name in visited:
                raise ValueError("Cycle in QDQ tensor path")
            visited.add(name)
            _, node = self.producers[name]
            if node.domain not in ("", "ai.onnx") or node.op_type not in (
                "QuantizeLinear",
                "DequantizeLinear",
            ):
                break
            if len(node.input) < 2 or any(v not in self.initializers for v in node.input[1:]):
                raise ValueError("Dynamic/non-constant QDQ parameters are unsupported")
            if node.op_type == "DequantizeLinear" and node.input[0] in self.producers:
                _, quant = self.producers[node.input[0]]
                if quant.op_type != "QuantizeLinear" or list(node.input[1:]) != list(
                    quant.input[1:]
                ):
                    raise ValueError(
                        "Q/DQ parameters or structure differ; cannot prove numerical meaning"
                    )
            name = node.input[0]
        return name

    def compute_producer(self, name):
        return self.producers.get(self.unwrap(name), (None, None))


class TensorMatcher:
    def __init__(self, reference, candidate):
        self.reference, self.candidate = reference, candidate
        self.memo = {}
        # Name markers are metadata only. Numerical matching is called again on actual captures.
        self.templates = debug.create_activation_matching(
            {name: [name] for name, v in candidate.values.items() if v["dtype"] in FLOAT_TYPES},
            {name: [name] for name, v in reference.values.items() if v["dtype"] in FLOAT_TYPES},
        )

    def canonical(self, name):
        if name.endswith(QUANT_INPUT_SUFFIX):
            base = name[: -len(QUANT_INPUT_SUFFIX)]
            if base in self.reference.values and any(
                n.op_type == "QuantizeLinear" and n.input[0] == name
                for n in self.candidate.nodes.values()
            ):
                return base
        return name

    def contract(self, original, quantized, depth=0):
        key = original, quantized
        if key in self.memo:
            return self.memo[key]
        if depth > 128:
            return False, "Producer proof exceeds depth 128; choose a smaller/shallower scope"
        a, b = self.reference, self.candidate
        try:
            raw = b.unwrap(quantized)
        except ValueError as exc:
            return False, str(exc)
        if original in a.inputs:
            result = (raw == original and raw in b.inputs, "Model input identity differs")
        elif original in a.initializers:
            weight = a.initializers[original]
            target = b.initializers.get(raw)
            result = (
                target is not None
                and raw in (original, original + "_quantized")
                and list(target.dims) == list(weight.dims),
                "Initializer identity/shape differs",
            )
        elif original not in a.producers or raw not in b.producers:
            result = False, "No corresponding producer tensor"
        else:
            _, left = a.producers[original]
            _, right = b.producers[raw]
            domain = left.domain or "ai.onnx"
            if domain not in ("", "ai.onnx") or left.op_type in ("If", "Loop", "Scan"):
                result = (
                    False,
                    "Custom/local-function or control-flow producer semantics are unsupported",
                )
            elif self.canonical(raw) != original:
                result = False, "Tensor was renamed beyond a verified ORT QDQ suffix"
            elif domain != (right.domain or "ai.onnx") or left.op_type != right.op_type:
                result = False, "Producer operator was fused/replaced or domain differs"
            elif a.opsets.get(domain) != b.opsets.get(domain):
                result = False, "Producer opset differs; equivalence is not assumed"
            elif list(left.output).index(original) != list(right.output).index(raw):
                result = False, "Producer output port differs"
            elif [
                v.SerializeToString(deterministic=True)
                for v in sorted(left.attribute, key=lambda x: x.name)
            ] != [
                v.SerializeToString(deterministic=True)
                for v in sorted(right.attribute, key=lambda x: x.name)
            ]:
                result = False, "Producer attributes differ"
            elif len(left.input) != len(right.input):
                result = False, "Producer input arity differs"
            else:
                result = (
                    True,
                    "Recursive tensor/producer/QDQ contract verified; node names are not used",
                )
                for x, y in zip(left.input, right.input, strict=True):
                    if not x and not y:
                        continue
                    matched, reason = self.contract(x, y, depth + 1)
                    if not matched:
                        result = False, f"Upstream {x}: {reason}"
                        break
        self.memo[key] = result
        return result

    def plan(self, config):
        reference, candidate = self.reference, self.candidate
        missing = set(config.selected_nodes) - set(reference.nodes)
        if missing:
            raise ValueError(f"Selected original node IDs do not exist: {sorted(missing)}")
        if any(not v.isidentifier() for v in config.operator_types):
            raise ValueError("Operator filters must be standard ONNX operator type names")
        nodes = [
            (id, n)
            for id, n in reference.nodes.items()
            if (not config.selected_nodes or id in config.selected_nodes)
            and (not config.operator_types or n.op_type in config.operator_types)
        ]
        if not nodes:
            raise ValueError("The selected node/operator scope is empty")
        scope_count = sum(len([v for v in n.output if v]) for _, n in nodes)
        nodes = nodes[:64]
        rows, plans, left_names, right_names = [], {}, set(), set()
        for id, node in nodes:
            for output in node.output:
                if not output or len(rows) >= 256:
                    continue
                row = LayerErrorRecord(
                    id=f"{id}:{list(node.output).index(output)}",
                    fp32_tensor=output,
                    fp32_node_id=id,
                    node_name=node.name,
                    node_name_unique=bool(node.name and reference.name_counts[node.name] == 1),
                    operator_type=node.op_type,
                    fp32_shape=reference.values.get(output, {}).get("shape"),
                ).model_dump()
                rows.append(row)
                if node.domain not in ("", "ai.onnx") or node.op_type in ("If", "Loop", "Scan"):
                    row.update(
                        match_status="Unsupported",
                        message="Custom domain/control-flow capture is unsupported",
                    )
                    continue
                if reference.values.get(output, {}).get("dtype") not in FLOAT_TYPES:
                    row.update(
                        match_status="Unsupported",
                        message="Only inferred float32/float16 tensors can be captured by this ORT helper",
                    )
                    continue
                template = self.templates.get(output)
                if template and "float" in template:
                    pre, post = template["pre_qdq"][0], template["post_qdq"][0]
                    method = "ort_qdq_loss_debug+recursive_producer_contract"
                elif (
                    output in candidate.values and candidate.values[output]["dtype"] in FLOAT_TYPES
                ):
                    pre = post = output
                    method = "unchanged_tensor+recursive_producer_contract"
                else:
                    row["message"] = (
                        "ORT QDQ matching found no counterpart; fused/renamed/missing tensors are not guessed"
                    )
                    continue
                row.update(
                    quantized_tensor=post,
                    quantized_shape=candidate.values.get(post, {}).get("shape"),
                    match_method=method,
                )
                matched, reason = self.contract(output, post)
                row["message"] = reason
                try:
                    compute_id, _ = candidate.compute_producer(post)
                    row["quantized_node_id"] = compute_id
                    row["quantized_tensor_producer_id"] = candidate.producers.get(post, (None,))[0]
                except ValueError:
                    pass
                if not matched:
                    row["match_status"] = "Partially Matched"
                    continue
                new_left, new_right = left_names | {output}, right_names | {pre, post}
                if len(new_left) + len(new_right) > config.max_tensors:
                    row.update(
                        match_status="Unsupported",
                        message="Max Tensors capture limit reached; select a smaller scope or raise the limit",
                    )
                    continue
                row["match_status"] = "Matched"
                left_names, right_names = new_left, new_right
                plans[row["id"]] = {
                    "reference": output,
                    "pre": pre,
                    "post": post,
                    "qdq": method.startswith("ort_qdq"),
                }
        return rows, plans, scope_count
