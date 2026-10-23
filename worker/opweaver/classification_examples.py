"""Offline supervised color classification fixture. Synthetic, not business coverage."""

import csv
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper

from .onnx_examples import model, value


def generate_classification_examples(folder):
    from PIL import Image

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    classes = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    prototypes = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.8, 0.1],
            [0.1, 0.8, 0.8],
            [0.8, 0.1, 0.8],
        ],
        np.float32,
    )
    weights = [
        numpy_helper.from_array(np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1), "stem_weight"),
        numpy_helper.from_array(prototypes.T.copy(), "head_weight"),
        numpy_helper.from_array(-0.5 * np.sum(prototypes**2, axis=1), "head_bias"),
    ]
    graph = model(
        [
            h.make_node(
                "Conv",
                ["image", "stem_weight"],
                ["features"],
                name="stem/conv",
                kernel_shape=[1, 1],
            ),
            h.make_node("Relu", ["features"], ["positive"], name="stem/relu"),
            h.make_node("GlobalAveragePool", ["positive"], ["pooled"], name="pool/average"),
            h.make_node("Flatten", ["pooled"], ["flat"], name="head/flatten", axis=1),
            h.make_node(
                "Gemm", ["flat", "head_weight", "head_bias"], ["scores"], name="head/classifier"
            ),
        ],
        [value("image", ["B", 3, 8, 8])],
        [value("scores", ["B", 6])],
        weights,
        "Synthetic supervised color classifier",
    )
    onnx.save(graph, folder / "color-classifier.onnx")
    fixture = {
        "scope": "Synthetic supervised color/prototype fixture; not a trained business classifier or representative production data",
        "classes": dict(zip(classes, range(6), strict=True)),
        "datasets": {},
    }
    for role, seed in [
        ("calibration", 100),
        ("validation", 1100),
        ("diagnostic", 2100),
        ("final_evaluation", 3100),
    ]:
        directory = folder / role
        directory.mkdir(exist_ok=True)
        csv_path = folder / f"{role}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label"])
            writer.writeheader()
            for index in range(12):
                label = index % 6
                shape = (12 + index % 3, 14 + index % 2, 3)
                rng = np.random.default_rng(seed + index)
                pixels = np.clip(prototypes[label] * 255 + rng.normal(0, 6, shape), 0, 255).astype(
                    np.uint8
                )
                path = directory / f"sample-{index:02d}{'.png' if index % 2 == 0 else '.jpg'}"
                Image.fromarray(pixels).save(path)
                writer.writerow(
                    {"path": path.relative_to(folder).as_posix(), "label": classes[label]}
                )
        fixture["datasets"][role] = csv_path.relative_to(folder).as_posix()
    fixture["preprocessing"] = {
        "resize": [10, 10],
        "center_crop": [8, 8],
        "layout": "NCHW",
        "color": "RGB",
        "dtype": "float32",
        "scale": 1 / 255,
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
    }
    (folder / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    return fixture
