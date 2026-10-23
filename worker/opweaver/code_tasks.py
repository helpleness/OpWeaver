"""Asynchronous, cancellable capture subprocesses with durable local reports."""

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .model_files import atomic_replace
from .process_owner import terminate_process
from .storage import RunStore


class CaptureTasks:
    event_name = "code.task"
    runner_module = "opweaver.code_runner"
    task_directory = "captures"
    metadata_prefix = "capture"
    task_label = "Model capture"

    def __init__(self, emit=None):
        self.emit = emit or (lambda event: None)
        self.lock = threading.RLock()
        self.jobs = {}
        self.processes = {}
        self.threads = {}
        self.cancelled = set()

    def snapshot(self, id):
        with self.lock:
            if id not in self.jobs:
                raise ValueError(
                    "Capture task is not live; its durable report is in the project analysis directory"
                )
            return json.loads(json.dumps(self.jobs[id]))

    def update(self, id, **values):
        with self.lock:
            self.jobs[id].update(values)
            snapshot = self.snapshot(id)
            folder = Path(snapshot["artifact_dir"])
            temporary = folder / "job.tmp"
            temporary.write_text(json.dumps(snapshot), encoding="utf-8")
            atomic_replace(temporary, folder / "job.json")
            RunStore(Path(snapshot["project"])).set_metadata(
                f"{self.metadata_prefix}:{id}", self.metadata_record(snapshot)
            )
            self.emit({"event": self.event_name, "data": snapshot})

    def metadata_record(self, snapshot):
        return snapshot

    def poll_progress(self, id, folder):
        """Optional child progress hook; capture/import behavior stays unchanged."""

    def start(self, params):
        if params.get("execute_confirmed") is not True:
            raise ValueError(
                "Capture executes Python code. Start it explicitly with execute_confirmed=true."
            )
        project = Path(params["project"]).resolve()
        file_path = Path(params["file_path"]).resolve()
        if not file_path.is_relative_to(project / "src") or file_path.suffix != ".py":
            raise ValueError("Capture source must be a Python file in the project src directory")
        if not 3 <= params.get("timeout", 30) <= 120:
            raise ValueError("Capture timeout must be between 3 and 120 seconds")
        return self.create(params, project, file_path)

    def create(self, params, project, file_path):
        with self.lock:
            if any(t.is_alive() for t in self.threads.values()):
                raise ValueError("A model capture is already running")
            id = uuid.uuid4().hex
            folder = project / ".opweaver" / self.task_directory / id
            if not folder.resolve().is_relative_to(project):
                raise ValueError("Task artifact directory escapes the project")
            folder.mkdir(parents=True, exist_ok=True)
            request = {**params, "file_path": str(file_path), "owner_pid": os.getpid()}
            (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
            self.jobs[id] = {
                "id": id,
                "state": "CREATED",
                "project": str(project),
                "file_path": str(file_path),
                "source_hash": params.get("expected_hash", ""),
                "artifact_dir": str(folder),
                "created_at": time.time(),
                "result": None,
                "error": None,
                "logs": "",
            }
            self.update(id)
            thread = threading.Thread(target=self.run, args=(id, request, folder), daemon=True)
            self.threads[id] = thread
            thread.start()
            return self.snapshot(id)

    def stop(self, id):
        with self.lock:
            if self.snapshot(id)["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                return self.snapshot(id)
            self.cancelled.add(id)
            (Path(self.jobs[id]["artifact_dir"]) / "cancel.request").touch()
            process = self.processes.get(id)
            if process and process.poll() is None:
                terminate_process(process)
            self.update(id, state="STOPPING")
            return self.snapshot(id)

    def run(self, id, request, folder):
        try:
            startup = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            with self.lock:
                if id in self.cancelled:
                    self.update(id, state="CANCELLED")
                    return
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        self.runner_module,
                        str(folder / "request.json"),
                        str(folder / "result.json"),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    **startup,
                )
                self.processes[id] = process
                self.update(id, state="RUNNING")
            try:
                deadline = time.monotonic() + request.get("timeout", 30)
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        self.poll_progress(id, folder)
                        if time.monotonic() >= deadline:
                            raise
                self.poll_progress(id, folder)
            except subprocess.TimeoutExpired:
                (folder / "cancel.request").touch()
                terminate_process(process)
                stdout, stderr = process.communicate()
                (folder / "stdout.log").write_bytes(stdout)
                (folder / "stderr.log").write_bytes(stderr)
                raise TimeoutError(
                    f"{self.task_label} timed out. Child process was terminated; the source and previous graph were retained."
                ) from None
            (folder / "stdout.log").write_bytes(stdout)
            (folder / "stderr.log").write_bytes(stderr)
            logs = stderr.decode("utf-8", errors="replace")[-16000:]
            if id in self.cancelled:
                self.update(id, state="CANCELLED", logs=logs)
                return
            if process.returncode or not (folder / "result.json").exists():
                raise RuntimeError(
                    f"Capture subprocess exited with code {process.returncode}: {logs[-1500:]}"
                )
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.update(
                id,
                state="COMPLETED" if result["success"] else "FAILED",
                result=result,
                logs=logs,
                error=None if result["success"] else result["diagnostics"][0]["message"],
            )
        except Exception as exc:
            with self.lock:
                self.jobs[id].update(state="FAILED", error=str(exc))
            try:
                self.update(id)
            except Exception:
                self.emit({"event": self.event_name, "data": self.snapshot(id)})
        finally:
            with self.lock:
                process = self.processes.pop(id, None)
                if process and process.poll() is None:
                    terminate_process(process)
                    process.wait(timeout=3)

    def shutdown(self):
        for id in list(self.jobs):
            self.stop(id)
        for thread in list(self.threads.values()):
            thread.join(timeout=3)
