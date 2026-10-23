"""Generate small local ONNX fixtures; no model/weight downloads."""

from pathlib import Path

from opweaver.onnx_examples import generate_examples

if __name__ == "__main__":
    generate_examples(Path(__file__).resolve().parents[1] / "examples" / "onnx")
    print(
        "Generated examples/onnx (CNN, residual, dynamic, custom, multi-output, If, external data, corrupt)"
    )
