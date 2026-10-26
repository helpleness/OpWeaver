"""Durable diagnostic jobs using the existing child-process lifecycle, independently of Olive."""

import json
import time
from pathlib import Path

from .code_tasks import CaptureTasks
from .diagnostic_reports import (
    cleanup_temporary,
    diagnostic_folder,
    export_diagnostic,
    save_diagnostic,
)
from .diagnostic_types import DiagnosticConfig, DiagnosticResult
from .lab_reports import verify_artifact
from .lab_tasks import STATUSES, TERMINAL
from .model_artifact import ModelArtifact
from .model_tasks import ModelTasks
from .storage import RunStore


class DiagnosticTasks(CaptureTasks):
    event_name = "diagnostic.task"
    runner_module = "opweaver.diagnostic_runner"
    task_directory = "diagnostics"
    metadata_prefix = "diagnostic"
    task_label = "Quantization diagnostic"

    def start(self, params):
        project = Path(params["project"]).resolve(strict=True)
        config = DiagnosticConfig.model_validate(params["config"])
        original = ModelTasks().load(project, config.fp32_model_id)
        return self.create(
            {
                "project": str(project),
                "config": config.model_dump(),
                "timeout": config.timeout,
                "created_at": time.time(),
            },
            project,
            project / original["artifact"]["file_path"],
        )

    def metadata_record(self, snapshot):
        return {k: v for k, v in snapshot.items() if k not in ("result", "logs")}

    def update(self, id, **values):
        with self.lock:
            state = values.get("state", self.jobs[id]["state"])
            # CaptureTasks may set FAILED before calling update without a state argument.
            # The lifecycle state is authoritative, including abrupt exit/timeout paths.
            values["status"] = STATUSES[state]
            folder = Path(self.jobs[id]["artifact_dir"])
            if state == "CREATED":
                request = json.loads((folder / "request.json").read_text())
                report = DiagnosticResult(
                    id=id, config=request["config"], created_at=request["created_at"]
                ).model_dump()
                save_diagnostic(folder, report)
            if state in TERMINAL:
                report = json.loads((folder / "diagnostic.json").read_text(encoding="utf-8"))
                report.update(
                    status=STATUSES[state], success=state == "COMPLETED", finished_at=time.time()
                )
                report["resources"]["temporary_cleaned"] = cleanup_temporary(folder)
                if state != "COMPLETED":
                    report["error"] = (
                        values.get("error")
                        or self.jobs[id].get("error")
                        or (
                            "Diagnostic cancelled; subprocess terminated"
                            if state == "CANCELLED"
                            else "Diagnostic subprocess failed"
                        )
                    )
                    report["diagnostics"] = [{"severity": "error", "message": report["error"]}]
                if values.get("logs"):
                    report["logs"].append("[runtime stderr] " + values["logs"][-16000:])
                save_diagnostic(folder, report)
                values.update(
                    result=report,
                    error=report["error"],
                    logs="\n".join(report["logs"]),
                    matching_summary=report["matching_summary"],
                )
            super().update(id, **values)

    def poll_progress(self, id, folder):
        try:
            path = folder / "progress.json"
            if not path.is_file():
                return
            revision = path.stat().st_mtime_ns
            if revision == self.jobs[id].get("progress_revision"):
                return
            self.update(
                id, **json.loads(path.read_text(encoding="utf-8")), progress_revision=revision
            )
        except (OSError, json.JSONDecodeError):
            pass

    def history(self, project):
        store = RunStore(Path(project))
        records = store.metadata_list("diagnostic:")
        for record in records:
            if record["id"] not in self.jobs and record["state"] not in TERMINAL:
                folder = diagnostic_folder(project, record["id"])
                report = json.loads((folder / "diagnostic.json").read_text(encoding="utf-8"))
                report.update(
                    status="Failed",
                    success=False,
                    error="Diagnostic worker was interrupted before completion",
                    finished_at=time.time(),
                )
                report["resources"]["temporary_cleaned"] = cleanup_temporary(folder)
                report["diagnostics"] = [{"severity": "error", "message": report["error"]}]
                save_diagnostic(folder, report)
                record.update(state="FAILED", status="Failed", error=report["error"])
                store.set_metadata(f"diagnostic:{record['id']}", record)
        return records

    def load(self, project, id):
        self.history(project)
        path = diagnostic_folder(project, id) / "diagnostic.json"
        if path.stat().st_size > 24 * 1024**2:
            raise ValueError("Diagnostic report exceeds the viewer limit")
        report = DiagnosticResult.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
        available = {}
        for key in ("fp32", "quantized"):
            item = report["models"].get(key)
            if item:
                try:
                    verify_artifact(project, ModelArtifact.model_validate(item))
                    available[key] = {"available": True}
                except (ValueError, OSError) as exc:
                    available[key] = {"available": False, "reason": str(exc)}
        if report["dataset"]:
            from .datasets import load_dataset

            try:
                current = load_dataset(project, report["dataset"]["id"])
                if current["manifest_sha256"] != report["dataset"]["manifest_sha256"]:
                    raise ValueError("Diagnostic dataset manifest changed")
                available["dataset"] = {"available": True}
            except (ValueError, OSError) as exc:
                available["dataset"] = {"available": False, "reason": str(exc)}
        # Availability describes current references; historical statistics are retained.
        return {**report, "availability": available}

    def export(self, params):
        report = self.load(params["project"], params["id"])
        return export_diagnostic(params["project"], report, params["directory"])
