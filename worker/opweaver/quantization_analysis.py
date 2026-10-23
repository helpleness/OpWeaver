"""Observed top-level ONNX quantization representation, never inferred from filenames."""

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def analyze_quantization(path):
    graph = onnx.load(str(path), load_external_data=False).graph
    counts = Counter(n.op_type for n in graph.node)
    dequantized = {n.output[0] for n in graph.node if n.op_type == "DequantizeLinear"}
    qoperators = [
        n
        for n in graph.node
        if n.op_type.startswith("QLinear") or n.op_type in ("MatMulInteger", "ConvInteger", "QGemm")
    ]
    parameters = {
        name
        for n in graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
        for name in n.input[1:3]
    }
    # QOperator parameter positions follow the ONNX QLinear*/QGemm signatures.
    for n in qoperators:
        indices = (1, 2, 4, 5, 7, 8) if n.op_type == "QGemm" else (1, 2, 4, 5, 6, 7)
        parameters.update(n.input[i] for i in indices if i < len(n.input))
    int_weights = [
        {"name": t.name, "dtype": TensorProto.DataType.Name(t.data_type), "shape": list(t.dims)}
        for t in graph.initializer
        if t.data_type in (TensorProto.INT8, TensorProto.UINT8)
        and t.name not in parameters
        and np.prod(t.dims) > 1
    ]
    observed = [
        {
            "name": n.name,
            "op_type": n.op_type,
            "representation": "QOperator" if n in qoperators else "QDQ inputs",
        }
        for n in graph.node
        if n in qoperators
        or (
            n.op_type not in ("QuantizeLinear", "DequantizeLinear")
            and any(v in dequantized for v in n.input)
        )
    ]
    summaries = []
    for t in graph.initializer:
        if t.name in parameters and len(summaries) < 256 and np.prod(t.dims) <= 4096:
            array = numpy_helper.to_array(t, base_dir=str(Path(path).parent))
            summaries.append(
                {
                    "name": t.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "elements": int(array.size),
                    "preview": array.reshape(-1)[:8].tolist(),
                }
            )
    present = bool(
        counts["QuantizeLinear"] or counts["DequantizeLinear"] or qoperators or int_weights
    )
    return {
        "present": present,
        "representation": "QOperator"
        if qoperators
        else "QDQ"
        if counts["DequantizeLinear"]
        else "Integer weights"
        if int_weights
        else "None",
        "quantize_linear": counts["QuantizeLinear"],
        "dequantize_linear": counts["DequantizeLinear"],
        "qoperator_count": len(qoperators),
        "integer_weights": int_weights,
        "observed_quantized_nodes": observed,
        "scale_zero_point_tensors": summaries,
        "remaining_float_initializers": [
            t.name
            for t in graph.initializer
            if t.data_type == TensorProto.FLOAT and t.name not in parameters
        ],
        "scope": "Top-level graph. QDQ input boundaries/QOperator nodes and integer initializers are observed; fusion, complete integer coverage and kernel execution are not inferred.",
    }
