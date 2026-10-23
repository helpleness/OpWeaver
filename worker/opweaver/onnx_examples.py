"""Small offline fixtures built with the ONNX public helper API."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper


def model(nodes, inputs, outputs, weights=(), name="Offline model", domains=()):
    return h.make_model(
        h.make_graph(nodes, name, inputs, outputs, initializer=weights),
        producer_name="OpWeaver offline fixtures",
        ir_version=10,
        opset_imports=[h.make_opsetid("", 18), *[h.make_opsetid(d, 1) for d in domains]],
    )


def value(name, shape, dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


def cnn():
    weights = [
        numpy_helper.from_array(np.full(shape, 0.1, dtype=np.float32), name)
        for name, shape in [
            ("conv_weight", (2, 3, 3, 3)),
            ("conv_bias", (2,)),
            ("head_weight", (18, 3)),
            ("head_bias", (3,)),
        ]
    ]
    return model(
        [
            h.make_node(
                "Conv",
                ["image", "conv_weight", "conv_bias"],
                ["features"],
                name="encoder/conv",
                kernel_shape=[3, 3],
            ),
            h.make_node("Relu", ["features"], ["activation"], name="encoder/relu"),
            h.make_node(
                "MaxPool",
                ["activation"],
                ["pooled"],
                name="encoder/pool",
                kernel_shape=[2, 2],
                strides=[2, 2],
            ),
            h.make_node("Flatten", ["pooled"], ["flat"], name="classifier/flatten", axis=1),
            h.make_node(
                "Gemm", ["flat", "head_weight", "head_bias"], ["scores"], name="classifier/head"
            ),
        ],
        [value("image", ["B", 3, 8, 8])],
        [value("scores", ["B", 3])],
        weights,
        "Tiny CNN",
    )


def residual():
    return model(
        [
            h.make_node("Identity", ["x"], ["branch"], name="branch"),
            h.make_node("Add", ["x", "branch"], ["sum"], name="residual/add"),
            h.make_node("Relu", ["sum"], ["y"], name="activation"),
        ],
        [value("x", ["B", 4])],
        [value("y", ["B", 4])],
        name="Residual",
    )


def dynamic():
    return model(
        [h.make_node("Identity", ["x"], ["y"], name="dynamic/identity")],
        [value("x", ["B", None, 4])],
        [value("y", ["B", None, 4])],
        name="Dynamic shapes",
    )


def custom():
    return model(
        [
            h.make_node(
                "Fancy",
                ["x"],
                ["y"],
                name="vendor/custom",
                domain="com.example",
                alpha=0.25,
                axes=[0, 1],
                label="custom",
            )
        ],
        [value("x", ["B", 4])],
        [value("y", ["B", 4])],
        name="Custom operator",
        domains=["com.example"],
    )


def multi_output():
    return model(
        [
            h.make_node("Split", ["x", ""], ["left", "right"], name="split", axis=1, num_outputs=2),
            h.make_node("Add", ["left", "right"], ["sum"], name="merge"),
        ],
        [value("x", [2, 4])],
        [value("sum", [2, 2]), value("right", [2, 2])],
        name="Multi output",
    )


def conditional():
    then = h.make_graph(
        [h.make_node("Add", ["x", "x"], ["a"], name="double")],
        "Then branch",
        [],
        [value("a", ["B", 4])],
    )
    other = h.make_graph(
        [h.make_node("Identity", ["x"], ["b"], name="identity")],
        "Else branch",
        [],
        [value("b", ["B", 4])],
    )
    return model(
        [
            h.make_node(
                "If", ["condition"], ["y"], name="conditional", then_branch=then, else_branch=other
            )
        ],
        [value("condition", [], T.BOOL), value("x", ["B", 4])],
        [value("y", ["B", 4])],
        name="Conditional branches",
    )


def multi_input_output():
    return model(
        [
            h.make_node("Add", ["x", "y"], ["sum"], name="add"),
            h.make_node("Mul", ["x", "y"], ["product"], name="multiply"),
            h.make_node("Greater", ["x", "y"], ["greater"], name="compare"),
        ],
        [value("x", ["B", 4]), value("y", ["B", 4])],
        [value("sum", ["B", 4]), value("product", ["B", 4]), value("greater", ["B", 4], T.BOOL)],
        name="Multi input / output",
    )


def large_chain(count=600):
    return model(
        [
            h.make_node(
                "Identity",
                ["x" if i == 0 else f"v_{i - 1}"],
                ["y" if i == count - 1 else f"v_{i}"],
                name=f"stack/layer_{i}",
            )
            for i in range(count)
        ],
        [value("x", ["B", 4])],
        [value("y", ["B", 4])],
        name="Large chain",
    )


def redundant_identity():
    return model(
        [
            h.make_node("Identity", ["x"], ["a"], name="redundant/first"),
            h.make_node("Identity", ["a"], ["b"], name="redundant/second"),
            h.make_node("Relu", ["b"], ["y"], name="activation"),
        ],
        [value("x", [2, 4])],
        [value("y", [2, 4])],
        name="Olive Identity elimination",
    )


def generate_examples(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, factory in [
        ("tiny-cnn", cnn),
        ("residual", residual),
        ("dynamic", dynamic),
        ("custom", custom),
        ("multi-output", multi_output),
        ("conditional", conditional),
        ("multi-input-output", multi_input_output),
        ("large-chain", large_chain),
        ("redundant-identity", redundant_identity),
    ]:
        onnx.save_model(factory(), directory / f"{name}.onnx")
    external = directory / "weights.bin"
    if external.exists():
        external.unlink()
    onnx.save_model(
        cnn(),
        directory / "external-cnn.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    (directory / "corrupt.onnx").write_bytes(b"not an ONNX protobuf")
