"""Disposable diagnostics process; parent ownership and cancellation reuse existing lifecycle."""

import json
import sys
from pathlib import Path

from .diagnostic_engine import QuantizationDiagnosticEngine
from .lab_reports import write_json
from .process_owner import watch_owner


def main():
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("owner_pid"):
        watch_owner(request["owner_pid"], result_path.parent / "cancel.request")
    result = QuantizationDiagnosticEngine().execute(request, result_path.parent)
    write_json(result_path, result)


if __name__ == "__main__":
    main()
