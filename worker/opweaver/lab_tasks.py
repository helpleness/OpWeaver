"""Optimization experiments reuse CaptureTasks process ownership and cancellation."""

import json
import time
from pathlib import Path

from .code_tasks import CaptureTasks
from .lab_reports import experiment_folder, export_experiment, save_report
from .lab_types import Experiment, ExperimentConfig
from .model_tasks import ModelTasks
from .storage import RunStore

TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")
STATUSES = {
    "CREATED": "Pending",
    "RUNNING": "Running",
    "STOPPING": "Running",
    "COMPLETED": "Succeeded",
    "FAILED": "Failed",
    "CANCELLED": "Cancelled",
}


class OptimizationTasks(CaptureTasks):
    event_name = "lab.task"
    runner_module = "opweaver.lab_runner"
    task_directory = "optimizations"
    metadata_prefix = "optimization"
    task_label = "Optimization experiment"

    def start(self, params):
        project = Path(params["project"]).resolve(strict=True)
        config = ExperimentConfig.model_validate(params["config"])
        report = ModelTasks().load(project, config.source_model_id)
        if report["artifact"]["format"] != "onnx":
            raise ValueError("Optimization Lab currently executes imported ONNX artifacts only")
        if config.mode == "optimize" and config.optimization.backend_id == "microsoft.olive":
            from .olive_adapter import preview

            config = ExperimentConfig.model_validate(
                preview(project, config.model_dump())["config"]
            )
        return self.create(
            {
                "project": str(project),
                "config": config.model_dump(),
                "timeout": config.timeout,
                "created_at": time.time(),
            },
            project,
            project / report["artifact"]["file_path"],
        )

    def metadata_record(self, snapshot):
        return {k: v for k, v in snapshot.items() if k not in ("result", "logs")}

    def update(self, id, **values):
        with self.lock:
            state = values.get("state", self.jobs[id]["state"])
            if "state" in values or "status" not in self.jobs[id]:
                values["status"] = STATUSES[state]
                if state == "RUNNING" and not self.jobs[id].get("progress_revision"):
                    values.update(status="Validating", phase="environment", progress=0)
            if state == "CREATED":
                folder = Path(self.jobs[id]["artifact_dir"])
                request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
                save_report(
                    folder,
                    Experiment(
                        id=id,
                        source_model_id=request["config"]["source_model_id"],
                        config=request["config"],
                        created_at=request["created_at"],
                    ).model_dump(),
                )
            if state in TERMINAL:
                values["status"] = STATUSES[state]
                folder = Path(self.jobs[id]["artifact_dir"])
                path = folder / "experiment.json"
                if path.is_file():
                    report = json.loads(path.read_text(encoding="utf-8"))
                else:
                    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
                    report = Experiment(
                        id=id,
                        source_model_id=request["config"]["source_model_id"],
                        config=request["config"],
                        created_at=request["created_at"],
                    ).model_dump()
                report.update(
                    status=STATUSES[state], success=state == "COMPLETED", finished_at=time.time()
                )
                if values.get("logs"):
                    report["logs"].append("[runtime stderr]\n" + values["logs"][-16000:])
                if state != "COMPLETED":
                    report["error"] = (
                        values.get("error")
                        or self.jobs[id].get("error")
                        or (
                            "Experiment cancelled; its subprocess was terminated"
                            if state == "CANCELLED"
                            else "Experiment process failed"
                        )
                    )
                    for phase, status in report["phases"].items():
                        if status == "Running":
                            report["phases"][phase] = STATUSES[state]
                olive_state = folder / "olive" / "execution.json"
                if olive_state.is_file():
                    workflow = json.loads(olive_state.read_text(encoding="utf-8"))
                    workflow.update(status=STATUSES[state], error=report.get("error"))
                    report["workflow_execution"] = workflow
                save_report(folder, report)
                values.update(
                    result=report,
                    phases=report["phases"],
                    logs="\n".join(report["logs"]),
                    error=report["error"],
                )
            super().update(id, **values)

    def poll_progress(self, id, folder):
        try:
            path = folder / "progress.json"
            if not path.exists():
                return
            revision = path.stat().st_mtime_ns
            if self.jobs[id].get("progress_revision") == revision:
                return
            value = json.loads(path.read_text(encoding="utf-8"))
            self.update(id, **value, progress_revision=revision)
        except (OSError, json.JSONDecodeError):
            pass  # Atomic replacement can be briefly locked by Windows file indexers.

    def history(self, project):
        store = RunStore(Path(project))
        records = store.metadata_list("optimization:")
        for record in records:
            if record["id"] not in self.jobs and record["state"] not in TERMINAL:
                record.update(
                    state="FAILED",
                    status="Failed",
                    error="Experiment worker was interrupted before completion",
                )
                folder = experiment_folder(project, record["id"])
                path = folder / "experiment.json"
                if path.is_file():
                    report = json.loads(path.read_text(encoding="utf-8"))
                    report.update(
                        status="Failed",
                        success=False,
                        error=record["error"],
                        finished_at=time.time(),
                    )
                    report["phases"] = {
                        key: "Failed" if value == "Running" else value
                        for key, value in report["phases"].items()
                    }
                    save_report(folder, report)
                store.set_metadata(f"optimization:{record['id']}", record)
        return records

    def load(self, project, id):
        self.history(project)
        path = experiment_folder(project, id) / "experiment.json"
        if path.stat().st_size > 24 * 1024 * 1024:
            raise ValueError("Experiment report exceeds the viewer limit")
        report = Experiment.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
        if report.get("datasets"):
            from .datasets import availability

            report["data_availability"] = availability(project, report["datasets"])
        return report

    def export(self, params):
        report = self.load(params["project"], params["id"])
        if report["status"] not in ("Succeeded", "Failed", "Cancelled"):
            raise ValueError("Wait until the experiment finishes before exporting")
        return export_experiment(params["project"], report, params["directory"])
