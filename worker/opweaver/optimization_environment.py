"""Project-scoped optimization environments; never installs into the Worker."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .lab_types import OptimizationEnvironment
from .process_owner import terminate_process
from .storage import RunStore

ENV_KEY = "optimization-environment:microsoft.olive"
SUPPORTED_OLIVE = "0.13.x"
_CACHE = {}


def local_environment():
    return {
        **os.environ,
        "OLIVE_DISABLE_TELEMETRY": "1",
        "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }


def interpreter(path):
    if not path:
        raise ValueError(
            "Configure an independent Olive Python interpreter or virtual environment directory"
        )
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if candidate.is_dir():
        options = [candidate / "Scripts" / "python.exe", candidate / "bin" / "python"]
        candidate = next((p for p in options if p.is_file()), options[0])
    if not candidate.is_file() or not candidate.name.lower().startswith("python"):
        raise ValueError(
            "Python interpreter does not exist; choose python.exe, bin/python, or its virtual environment directory"
        )
    # Do not resolve the final symlink: POSIX venv Python relies on its own path.
    if os.path.normcase(str(candidate)) == os.path.normcase(os.path.abspath(sys.executable)):
        raise ValueError(
            "Choose an independent optimization environment, not the OpWeaver Worker interpreter"
        )
    return candidate


def settings(project):
    records = RunStore(Path(project)).metadata_list(ENV_KEY) if project else []
    return OptimizationEnvironment.model_validate(records[0] if records else {}).model_dump()


def configure(project, value):
    config = OptimizationEnvironment.model_validate(value)
    if config.backend_id != "microsoft.olive":
        raise ValueError("This environment configuration is for Microsoft Olive")
    if config.python_path:
        try:
            config.python_path = str(interpreter(config.python_path))
        except ValueError:
            # Preserve the explicitly selected invalid environment so its probe
            # status can be shown and corrected, rather than displaying old success.
            config.python_path = os.path.abspath(Path(config.python_path).expanduser())
    RunStore(Path(project)).set_metadata(ENV_KEY, config.model_dump())
    return config.model_dump()


def bridge_path():
    return Path(__file__).with_name("olive_runner.py")


def invoke(environment, mode, request, folder=None, timeout=30):
    python = interpreter(environment.python_path)
    with tempfile.TemporaryDirectory(prefix="opweaver-olive-probe-") as temporary:
        directory = Path(folder or temporary)
        request_path, result_path = (
            directory / f"{mode}-request.json",
            directory / f"{mode}-result.json",
        )
        request_path.write_text(
            json.dumps({**request, "owner_pid": os.getpid()}, ensure_ascii=True), encoding="utf-8"
        )
        startup = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = subprocess.Popen(
            [
                str(python),
                "-I",
                "-u",
                str(bridge_path()),
                mode,
                str(request_path),
                str(result_path),
            ],
            cwd=directory,
            env=local_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **startup,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process(process)
            process.communicate(timeout=5)
            raise TimeoutError("Olive environment/configuration check timed out") from None
        if process.returncode or not result_path.is_file():
            raise RuntimeError(
                f"Olive {mode} process exited {process.returncode}: {stderr.decode('utf-8', errors='replace')[-1600:]}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("success", True):
            raise ValueError(result.get("error", "Olive configuration failed"))
        return result


def probe(environment=None, force=False):
    environment = OptimizationEnvironment.model_validate(environment or {})
    base = {
        "schema_version": "1.0",
        "backend_id": "microsoft.olive",
        "python_path": environment.python_path,
        "status": "Not Installed",
        "available": False,
        "versions": {},
        "passes": [],
        "errors": [],
        "supported_olive": SUPPORTED_OLIVE,
    }
    if not environment.python_path:
        return {
            **base,
            "reason": "No independent Olive environment configured; ORT CPU remains available",
        }
    try:
        path = interpreter(environment.python_path)
        key = str(path)
        if not force and key in _CACHE and time.monotonic() - _CACHE[key][0] < 30:
            return _CACHE[key][1]
        result = invoke(environment, "probe", {})
        result = {**base, **result, "python_path": str(path)}
        _CACHE[key] = (time.monotonic(), result)
        return result
    except Exception as exc:
        return {
            **base,
            "status": "Environment Error",
            "reason": str(exc),
            "errors": [{"code": type(exc).__name__, "message": str(exc)}],
        }
