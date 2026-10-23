"""ONNX imports share the existing cancellable subprocess task lifecycle."""

import re
from pathlib import Path

from .code_tasks import CaptureTasks
from .importers import builtin_importers
from .model_artifact import OnnxReport
from .storage import RunStore


class ModelTasks(CaptureTasks):
    event_name = "model.task"
    runner_module = "opweaver.import_runner"
    task_directory = "imports"
    metadata_prefix = "model-import"
    task_label = "Model import"

    def start(self, params):
        project = Path(params["project"]).resolve()
        source = Path(params["source"]).resolve(strict=True)
        if not (project / "project.opweaver.json").is_file():
            raise ValueError("Open or create a project before importing a model")
        importer = builtin_importers().resolve(source)
        if not source.is_file():
            raise ValueError("Select a model file")
        if importer.requires_execution and params.get("execute_confirmed") is not True:
            raise ValueError(
                "Python import must be explicitly triggered with execute_confirmed=true"
            )
        if not 3 <= params.get("timeout", 120) <= 120:
            raise ValueError("Import timeout must be between 3 and 120 seconds")
        return self.create(
            {**params, "importer_id": importer.id, "timeout": params.get("timeout", 120)},
            project,
            source,
        )

    def metadata_record(self, snapshot):
        result = snapshot.get("result") or {}
        return {
            **{k: v for k, v in snapshot.items() if k not in ("result", "logs")},
            "artifact": result.get("artifact"),
        }

    def history(self, project):
        store = RunStore(Path(project))
        records = store.metadata_list("model-import:")
        for record in records:
            if record["id"] not in self.jobs and record["state"] not in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            ):
                record.update(
                    state="FAILED",
                    error="Import worker exited before completion. Previous valid models were retained.",
                )
                store.set_metadata(f"model-import:{record['id']}", record)
        return records

    def load(self, project, id):
        if not re.fullmatch(r"[0-9a-f]{32}", id):
            raise ValueError("Invalid model artifact ID")
        root = Path(project).resolve()
        folder = (root / ".opweaver" / "imports" / id).resolve(strict=True)
        if not folder.is_relative_to(root):
            raise ValueError("Model report escapes the project")
        report_path = folder / "result.json"
        if report_path.stat().st_size > 24 * 1024 * 1024:
            raise ValueError("Model report exceeds the viewer limit")
        report = OnnxReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        model_path = (root / report.artifact.file_path).resolve(strict=True)
        allowed = folder if report.artifact.format == "onnx" else root / "src"
        if not model_path.is_relative_to(allowed) or report.artifact.id != id:
            raise ValueError("Model artifact reference does not belong to this import")
        return report.model_dump()
