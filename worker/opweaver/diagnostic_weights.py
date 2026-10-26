"""Bounded ORT weight matching with explicit affine/broadcast verification."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import qdq_loss_debug as debug

from .diagnostic_metrics import aggregate_metrics, tensor_metrics
from .diagnostic_types import LayerErrorRecord


def parameter_info(tensor):
    array = numpy_helper.to_array(tensor)
    return {
        "name": tensor.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "elements": int(array.size),
        "preview": array.reshape(-1)[:16].tolist(),
        "preview_truncated": bool(array.size > 16),
    }


def weight_records(reference, candidate, node_ids, config, folder, progress):
    rows, selected = [], {}
    dequants = {}
    for id, node in candidate.nodes.items():
        if (
            node.op_type == "DequantizeLinear"
            and node.domain in ("", "ai.onnx")
            and node.input[0] in candidate.initializers
            and node.input[0].endswith("_quantized")
        ):
            dequants.setdefault(node.input[0][:-10], []).append((id, node))
    for id, original in reference.nodes.items():
        if id not in node_ids:
            continue
        for port, name in enumerate(original.input):
            value = reference.initializers.get(name)
            if (
                value is None
                or value.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16)
                or len(rows) >= 64
            ):
                continue
            row = LayerErrorRecord(
                id=f"weight:{id}:{port}",
                fp32_tensor=name,
                fp32_node_id=id,
                node_name=original.name,
                node_name_unique=bool(original.name and reference.name_counts[original.name] == 1),
                operator_type=original.op_type,
                fp32_shape=list(value.dims),
                kind="weight",
                match_method="ort_create_weight_matching+affine_broadcast_verification",
            ).model_dump()
            rows.append(row)
            matches = dequants.get(name, [])
            if len(matches) != 1:
                row["message"] = (
                    "No unique ORT-named initializer/DQ counterpart; renamed weights are not guessed"
                )
                continue
            dq_id, dq = matches[0]
            weight = candidate.initializers[dq.input[0]]
            row.update(
                quantized_tensor=weight.name,
                quantized_shape=list(weight.dims),
                quantized_tensor_producer_id=dq_id,
            )
            if weight.data_type not in (TensorProto.INT8, TensorProto.UINT8, TensorProto.INT32):
                row.update(
                    match_status="Unsupported",
                    message="Only int8/uint8 weights and int32 quantized bias are supported",
                )
                continue
            consumers = [
                node_id
                for node_id, n in candidate.nodes.items()
                if n.op_type == original.op_type
                and n.domain == original.domain
                and len(n.input) > port
                and n.input[port] in dq.output
            ]
            row["quantized_node_id"] = consumers[0] if len(consumers) == 1 else None
            if any(v not in candidate.initializers for v in dq.input[1:]):
                row.update(
                    match_status="Unsupported",
                    message="Weight scale/zero point is not a constant initializer",
                )
                continue
            selected[name] = dq
            row.update(
                match_status="Matched",
                message="Initializer identity and actual DQ parameters; weight error is independent of activation matching",
            )
    if not selected:
        return rows
    float_values = [reference.initializers[n] for n in selected]
    quant_names = {name for dq in selected.values() for name in dq.input}
    quant_values = [candidate.initializers[n] for n in quant_names]
    estimate = sum(v.ByteSize() for v in [*float_values, *quant_values]) * 24
    if estimate > config.memory_budget_mb * 1024**2:
        for row in rows:
            if row["match_status"] == "Matched":
                row.update(
                    match_status="Unsupported",
                    message="Selected weight matching/statistics exceeds the working tensor budget; narrow nodes or disable weights",
                )
        return rows
    float_nodes = [
        helper.make_node("Identity", [v.name], [v.name + "_diagnostic"]) for v in float_values
    ]
    float_outputs = [
        helper.make_tensor_value_info(v.name + "_diagnostic", v.data_type, list(v.dims))
        for v in float_values
    ]
    quant_outputs = [
        helper.make_tensor_value_info(
            dq.output[0],
            reference.initializers[name].data_type,
            list(reference.initializers[name].dims),
        )
        for name, dq in selected.items()
    ]
    a = helper.make_model(
        helper.make_graph(
            float_nodes, "diagnostic selected reference weights", [], float_outputs, float_values
        ),
        opset_imports=reference.model.opset_import,
        ir_version=reference.model.ir_version,
    )
    b = helper.make_model(
        helper.make_graph(
            list(selected.values()),
            "diagnostic selected QDQ weights",
            [],
            quant_outputs,
            quant_values,
        ),
        opset_imports=candidate.model.opset_import,
        ir_version=candidate.model.ir_version,
    )
    left, right = folder / "reference-weights.onnx", folder / "quantized-weights.onnx"
    existing = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    if existing + (a.ByteSize() + b.ByteSize()) * 3 > config.disk_budget_mb * 1024**2:
        for row in rows:
            if row["match_status"] == "Matched":
                row.update(
                    match_status="Unsupported",
                    message="Selected weight instrumentation would exceed the temporary disk budget",
                )
        return rows
    onnx.save(a, left)
    onnx.save(b, right)
    progress("weights", 0, "ORT matching selected constant weights with per-channel parameters")
    try:
        matched = debug.create_weight_matching(str(left), str(right))
    except Exception as exc:
        for row in rows:
            if row["match_status"] == "Matched":
                row.update(
                    match_status="Unsupported",
                    message=f"ORT weight matching failed: {type(exc).__name__}: {exc}",
                )
        return rows
    for row in rows:
        if row["match_status"] != "Matched":
            continue
        name = row["fp32_tensor"]
        dq, pair = selected[name], matched.get(name)
        if pair is None:
            row.update(match_status="Unmatched", message="ORT did not return this weight pair")
            continue
        try:
            values = candidate.initializers
            quant = numpy_helper.to_array(values[dq.input[0]])
            scale = numpy_helper.to_array(values[dq.input[1]])
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("Weight scale must contain finite positive values")
            zero = (
                numpy_helper.to_array(values[dq.input[2]])
                if len(dq.input) > 2
                else np.zeros(scale.shape, dtype=np.int32)
            )
            axis = next((int(attr.i) for attr in dq.attribute if attr.name == "axis"), 1)
            if scale.size != 1:
                if (
                    scale.ndim != 1
                    or zero.shape != scale.shape
                    or not -quant.ndim <= axis < quant.ndim
                    or scale.size != quant.shape[axis]
                ):
                    raise ValueError("Invalid per-channel scale/zero-point shape/axis")
                shape = [1] * quant.ndim
                shape[axis] = scale.size
                scale, zero = scale.reshape(shape), zero.reshape(shape)
            affine = (quant.astype(np.float64) - zero.astype(np.float64)) * scale.astype(np.float64)
            if pair["dequantized"].shape != affine.shape or not np.allclose(
                pair["dequantized"], affine, rtol=1e-5, atol=1e-8
            ):
                raise ValueError(
                    "Installed ORT weight helper differs from affine DQ semantics (including integer subtraction/broadcast); metric withheld"
                )
            original = pair["float"]
            dequantized = pair["dequantized"].astype(original.dtype)
            metrics = tensor_metrics(original, dequantized)
            row["samples"] = [
                {
                    "sample_index": 0,
                    "scope": "One constant weight tensor, independent of dataset batches",
                    "metrics": metrics,
                }
            ]
            row["aggregate"] = aggregate_metrics(row["samples"])
            row["quantization"] = {
                "dtype": str(quant.dtype),
                "axis": axis if scale.size != 1 else None,
                "scale": parameter_info(values[dq.input[1]]),
                "zero_point": parameter_info(values[dq.input[2]])
                if len(dq.input) > 2
                else {"implicit": 0},
                "dequantized_dtype": str(dequantized.dtype),
                "working_bytes_estimate": estimate,
            }
            if not metrics["valid"]:
                row.update(match_status="Unsupported", message=metrics["reason"])
        except (ValueError, TypeError, IndexError) as exc:
            row.update(match_status="Unsupported", message=str(exc))
    return rows
