"""Offline, explicitly synthetic CNN + disjoint NPZ calibration/evaluation batches."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper
from opweaver.onnx_examples import model, value


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(190)
    weights = [
        numpy_helper.from_array(rng.normal(0, 0.12, shape).astype(np.float32), name)
        for name, shape in [
            ("conv_weight", (16, 3, 3, 3)),
            ("conv_bias", (16,)),
            ("head_weight", (144, 32)),
            ("head_bias", (32,)),
        ]
    ]
    graph = model(
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
        [value("scores", ["B", 32])],
        weights,
        "Synthetic quantization CNN",
    )
    onnx.save(graph, folder / "quantization-cnn.onnx")
    manifest = {
        "scope": "Synthetic offline test fixture, not a trained task model or representative production dataset",
        "mapping": {"image": "pixels"},
        "shape": [2, 3, 8, 8],
        "datasets": {},
    }
    for role, seed, count in [("calibration", 2026, 32), ("evaluation", 12026, 8)]:
        directory = folder / role
        directory.mkdir(exist_ok=True)
        files = []
        for index in range(count):
            path = directory / f"batch-{index:03d}.npz"
            np.savez(
                path,
                pixels=np.random.default_rng(seed + index)
                .normal(0, 1, (2, 3, 8, 8))
                .astype(np.float32),
            )
            files.append(path.relative_to(folder).as_posix())
        manifest["datasets"][role] = {"seed": seed, "count": count, "files": files}
    (folder / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "examples" / "quantization")
