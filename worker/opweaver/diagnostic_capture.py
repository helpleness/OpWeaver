"""ORT-owned instrumentation/collection, pruned to bounded selected floating outputs."""

import math

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from onnxruntime.quantization import qdq_loss_debug as debug
from onnxruntime.quantization.calibrate import CalibrationDataReader

POSTFIX = "_ReshapedSavedOutput"  # Verified ORT 1.31 helper output contract.


def session_options():
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = options.enable_cpu_mem_arena = False
    options.log_severity_level = 3
    return options


class SingleFeed(CalibrationDataReader):
    def __init__(self, feed):
        self.feed = feed

    def get_next(self):
        result, self.feed = self.feed, None
        return result


def augment_model(source, target, selected):
    debug.modify_model_output_intermediate_tensors(source, target)
    graph = onnx.load(target)
    outputs = {name + POSTFIX for name in selected}
    available = {v.name for v in graph.graph.output if v.name.endswith(POSTFIX)}
    retained = [
        n
        for n in graph.graph.node
        if not (n.op_type == "Reshape" and n.name.endswith(POSTFIX)) or n.name in outputs
    ]
    del graph.graph.node[:]
    graph.graph.node.extend(retained)
    kept_outputs = [v for v in graph.graph.output if v.name in outputs]
    del graph.graph.output[:]
    graph.graph.output.extend(kept_outputs)
    onnx.save(graph, target)
    return {v[: -len(POSTFIX)] for v in available & outputs}


def symbols_for(index, feed):
    symbols = {}
    for name in index.inputs:
        dims = index.values.get(name, {}).get("shape")
        if dims is None:
            continue
        for expected, actual in zip(dims, feed[name].shape, strict=True):
            if isinstance(expected, str):
                if expected in symbols and symbols[expected] != actual:
                    raise ValueError(f"Input symbol {expected} differs between inputs")
                symbols[expected] = actual
    return symbols


def concrete_shape(index, name, feed):
    info = index.values.get(name)
    if not info or info["shape"] is None:
        raise ValueError("Unknown rank")
    symbols = symbols_for(index, feed)
    shape = [symbols.get(v, v) if isinstance(v, str) else v for v in info["shape"]]
    if any(type(v) is not int or v < 0 for v in shape):
        raise ValueError("Unbound/data-dependent inferred shape")
    return shape


def estimated_bytes(index, names, feed):
    return sum(
        math.prod(concrete_shape(index, n, feed))
        * np.dtype(helper.tensor_dtype_to_np_dtype(index.values[n]["dtype"])).itemsize
        for n in names
    )


def collect(path, feed, provider):
    return debug.collect_activations(
        str(path),
        SingleFeed(feed),
        session_options=session_options(),
        execution_providers=[provider],
    )
