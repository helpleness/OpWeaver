"""Durable reports and explicit export of verified model/input snapshots."""

import json
import re
from pathlib import Path, PurePosixPath

from .benchmark import digest_file
from .model_files import atomic_replace, copy_file


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2), encoding="utf-8"
    )
    atomic_replace(temporary, path)


def experiment_folder(project, id, exists=True):
    if not re.fullmatch(r"[0-9a-f]{32}", id):
        raise ValueError("Invalid experiment ID")
    root = Path(project).resolve(strict=True)
    folder = (root / ".opweaver" / "optimizations" / id).resolve(strict=exists)
    if not folder.is_relative_to(root):
        raise ValueError("Experiment path escapes the project")
    return folder


def verify_artifact(project, artifact):
    root = Path(project).resolve(strict=True)
    path = (root / artifact.file_path).resolve(strict=True)
    allowed = (root / ".opweaver" / "imports" / artifact.id).resolve(strict=True)
    if (
        not allowed.is_relative_to(root)
        or not path.is_relative_to(allowed)
        or digest_file(path) != artifact.sha256
    ):
        raise ValueError(
            "Model file no longer matches the imported artifact/hash; import a new snapshot"
        )
    for external in artifact.external_files:
        relative = PurePosixPath(external["path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in external["path"]:
            raise ValueError("Invalid external artifact path")
        external_path = (path.parent / external["path"]).resolve(strict=True)
        if (
            not external_path.is_relative_to(allowed)
            or digest_file(external_path) != external["sha256"]
        ):
            raise ValueError("External model data no longer matches its imported snapshot")
    return path


def markdown_report(report):
    config = report["config"]
    original = report.get("original") or {}
    optimized = (report.get("optimization") or {}).get("artifact") or {}
    lines = [
        "# OpWeaver Optimization Experiment",
        "",
        f"- Experiment: `{report['id']}`",
        f"- Status: **{report['status']}**",
        f"- Source model: `{report['source_model_id']}`",
        f"- Source: `{original.get('source_path', '')}`",
        f"- Source SHA256: `{original.get('sha256', '')}`",
        f"- Optimized artifact: `{optimized.get('id', 'not generated')}`",
        "",
        "## Measured results",
        "",
        "| Metric | Baseline | Optimized |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Model bytes", "model_bytes"),
        ("Session initialization ms", "session_init_ms"),
        ("Warmup total ms", "warmup_ms"),
        ("Mean ms", "mean_ms"),
        ("P50 ms", "p50_ms"),
        ("P95 ms", "p95_ms"),
        ("Min ms", "min_ms"),
        ("Max ms", "max_ms"),
        ("requests/s (concurrency=1)", "requests_per_second"),
        ("samples/s (declared batch)", "samples_per_second"),
    ]:
        lines.append(
            f"| {label} | {(report.get('baseline') or {}).get(key, 'unavailable')} | {(report.get('optimized') or {}).get(key, 'unavailable')} |"
        )
    for label, value in [
        ("Comparison", report.get("comparison")),
        ("Graph changes", report.get("graph_comparison")),
        ("Optimization config", config["optimization"]),
        ("Benchmark config", config["benchmark"]),
        ("Input config", config["inputs"]),
        ("Saved input samples", report.get("input_samples")),
        ("Environment", report.get("environment")),
        ("Optimization result / target environment", report.get("optimization")),
        ("Workflow requested / executed / observed", report.get("workflow_execution")),
        ("Numerical validation", report.get("validation")),
        ("FP32 standalone task evaluation", report.get("baseline_evaluation")),
        ("Held-out final evaluation", report.get("final_evaluation")),
        ("Calibration / evaluation datasets and independence", report.get("datasets")),
        ("Observed quantization representation", report.get("quantization")),
        ("Preprocessed artifact lineage", report.get("preprocessed_artifacts")),
        ("Referenced data availability on reopen", report.get("data_availability")),
        ("Phase states", report.get("phases")),
        ("Warnings / failure", {"warnings": report.get("warnings"), "error": report.get("error")}),
    ]:
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "```json",
                json.dumps(value, indent=2, ensure_ascii=True),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Raw latency samples (ms)",
            "",
            "```json",
            json.dumps(
                {
                    side: (report.get(side) or {}).get("latencies_ms")
                    for side in ["baseline", "optimized"]
                }
            ),
            "```",
            "",
            "## Execution log",
            "",
            "```text",
            *report.get("logs", []),
            "```",
            "",
            "Numerical consistency covers only the recorded samples and is not task accuracy. Sequential CPU measurements are subject to noise. Node reduction alone does not establish a speedup. No GPU or memory measurement was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def save_report(folder, report):
    write_json(folder / "experiment.json", report)
    (folder / "report.md").write_text(markdown_report(report), encoding="utf-8")


def export_experiment(project, report, destination):
    destination = Path(destination).resolve(strict=True)
    if not destination.is_dir():
        raise ValueError("Select an existing export directory")
    target = destination / f"opweaver-{report['id']}"
    target.mkdir(exist_ok=False)
    exported = {}
    for key, artifact in [
        ("original", report.get("original")),
        ("optimized", (report.get("optimization") or {}).get("artifact")),
        *[
            (f"preprocessed/{i}", artifact)
            for i, artifact in enumerate(report.get("preprocessed_artifacts", []))
        ],
    ]:
        if not artifact:
            continue
        from .model_artifact import ModelArtifact

        parsed = ModelArtifact.model_validate(artifact)
        source = verify_artifact(project, parsed)
        model_target = target / key / source.name
        copy_file(source, model_target)
        for ext in parsed.external_files:
            copy_file(source.parent / ext["path"], model_target.parent / ext["path"])
        exported[key] = model_target.relative_to(target).as_posix()
    folder = experiment_folder(project, report["id"])
    for role in ("calibration", "evaluation", "final_evaluation"):
        dataset = (report.get("datasets") or {}).get(role)
        if dataset:
            from .datasets import load_dataset

            current = load_dataset(project, dataset["id"])
            if current["manifest_sha256"] != dataset["manifest_sha256"]:
                raise ValueError("Recorded dataset fingerprint no longer matches")
            source = Path(current["manifest_path"])
            copy_file(source, target / "datasets" / role / "manifest.json")
            for sample in current["samples"]:
                copy_file(
                    source.parent / sample["path"], target / "datasets" / role / sample["path"]
                )
            exported[role + "_dataset"] = f"datasets/{role}/manifest.json"
            for original in current.get("source_files", []):
                copy_file(
                    source.parent / original["path"], target / "datasets" / role / original["path"]
                )
    if report.get("workflow_execution"):
        for name in [
            "request.json",
            "native-workflow.json",
            "executed-workflow.json",
            "execution.json",
            "result.json",
            "pass-results.json",
            "process.json",
            "stdout.log",
            "stderr.log",
        ]:
            source = (folder / "olive" / name).resolve()
            if source.is_file() and source.is_relative_to(folder):
                copy_file(source, target / "workflow" / name)
        for source in (folder / "olive").glob("*-calibration-usage.json"):
            copy_file(source, target / "workflow" / source.name)
        # Preserve explicit preprocessing artifacts from Olive's recorded footprint.
        for index, item in enumerate(
            []
            if report.get("preprocessed_artifacts")
            else report["workflow_execution"].get("executed_passes", [])
        ):
            if item["pass_type"].lower() != "onnxquantizationpreprocess":
                continue
            resource = item.get("artifact_path")
            if resource:
                path = Path(resource).resolve()
                if path.is_relative_to(folder) and path.is_file():
                    import onnx

                    from .onnx_import import all_tensors

                    copy_file(path, target / "preprocessed" / str(index) / path.name)
                    model = onnx.load(str(path), load_external_data=False)
                    for tensor in all_tensors(model):
                        external = {v.key: v.value for v in tensor.external_data}.get("location")
                        if external:
                            source = (path.parent / external).resolve(strict=True)
                            if not source.is_relative_to(folder) or not source.is_relative_to(
                                path.parent
                            ):
                                raise ValueError(
                                    "Preprocessed external data escapes the artifact directory"
                                )
                            copy_file(source, target / "preprocessed" / str(index) / external)
    for sample in report.get("input_samples", []):
        source = (folder / "inputs" / sample["path"]).resolve(strict=True)
        if not source.is_relative_to(folder) or digest_file(source) != sample["sha256"]:
            raise ValueError("Saved experiment input snapshot is missing or changed")
        copy_file(source, target / "inputs" / source.name)
    write_json(target / "report.json", {**report, "exported_files": exported})
    (target / "report.md").write_text(
        markdown_report(report)
        + "\nExported model paths:\n\n"
        + json.dumps(exported, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "directory": str(target),
        "json": str(target / "report.json"),
        "markdown": str(target / "report.md"),
        "models": exported,
    }
