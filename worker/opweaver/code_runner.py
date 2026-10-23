"""A disposable user-code host. Never used to open or statically inspect files."""

import contextlib
import json
import sys
import traceback
from pathlib import Path

from .capture import capture
from .source import digest, read_source


def main():
    request_file, result_file = map(Path, sys.argv[1:3])
    request = json.loads(request_file.read_text(encoding="utf-8"))
    try:
        import torch

        torch.set_num_threads(2)
        source = read_source(request["file_path"])
        if digest(source) != request["expected_hash"]:
            raise ValueError(
                "Source changed before capture started. Reload and parse the saved version."
            )
        with contextlib.redirect_stdout(sys.stderr):
            result = capture(
                request["file_path"],
                request["entry"],
                request.get("constructor", {}),
                request["inputs"],
            )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result = {
            "success": False,
            "source_hash": request.get("expected_hash"),
            "diagnostics": [
                {
                    "severity": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "line": getattr(exc, "lineno", None),
                    "hint": "FX cannot capture data-dependent Python control flow. The previous valid graph is retained; source was not changed.",
                }
            ],
        }
    result_file.write_text(json.dumps(result, ensure_ascii=True, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
