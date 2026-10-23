"""Disposable ONNX parser process; never executes an ONNX inference graph."""

import json
import sys
import traceback
from pathlib import Path


def main():
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        from .importers import builtin_importers

        result = (
            builtin_importers()
            .resolve(request["file_path"])
            .import_model(request, result_path.parent)
        )
        encoded = json.dumps(result, ensure_ascii=True, allow_nan=False)
        if len(encoded.encode("utf-8")) > 24 * 1024 * 1024:
            raise ValueError(
                "ONNX analysis exceeds the 24 MiB viewer report limit; no partial graph published"
            )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        encoded = json.dumps(
            {
                "success": False,
                "diagnostics": [
                    {
                        "severity": "error",
                        "code": "IMPORT_FAILED",
                        "source": request.get("importer_id", "model.import"),
                        "message": f"{type(exc).__name__}: {exc}",
                        "suggested_action": "Check the model format, structure and referenced external files. Previous valid artifacts were retained.",
                    }
                ],
            }
        )
    result_path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
