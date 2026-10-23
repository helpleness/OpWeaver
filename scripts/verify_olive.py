"""Explicit offline smoke run using the independent Olive environment."""

import json
import sys
import uuid
from pathlib import Path

import onnx
from onnx import helper as h
from opweaver.lab_runner import execute
from opweaver.onnx_examples import model, value
from opweaver.onnx_import import import_onnx


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "redundant-identity.onnx"
    onnx.save(
        model(
            [
                h.make_node("Identity", ["x"], ["a"]),
                h.make_node("Identity", ["a"], ["b"]),
                h.make_node("Relu", ["b"], ["y"]),
            ],
            [value("x", [2, 4])],
            [value("y", [2, 4])],
        ),
        source,
    )
    imported_folder = root / ".opweaver" / "imports" / uuid.uuid4().hex
    imported = import_onnx(source, root, imported_folder)
    (imported_folder / "result.json").write_text(json.dumps(imported))
    folder = root / ".opweaver" / "optimizations" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    result = execute(
        {
            "project": str(root),
            "config": {
                "source_model_id": imported["artifact"]["id"],
                "inputs": {"shapes": {"x": [2, 4]}},
                "benchmark": {"runs": 30, "warmup": 3},
                "optimization": {
                    "backend_id": "microsoft.olive",
                    "environment": {"python_path": str(Path(sys.argv[2]).resolve())},
                    "workflow": {
                        "passes": [
                            {"id": "peephole", "type": "OnnxPeepholeOptimizer"},
                            {
                                "id": "opset",
                                "type": "OnnxOpVersionConversion",
                                "parameters": {"target_opset": 18},
                            },
                        ]
                    },
                },
            },
        },
        folder,
    )
    print(
        json.dumps(
            {
                "folder": str(folder),
                "status": result["status"],
                "error": result["error"],
                "graph_comparison": result["graph_comparison"],
                "comparison": result["comparison"],
            },
            indent=2,
        )
    )
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
