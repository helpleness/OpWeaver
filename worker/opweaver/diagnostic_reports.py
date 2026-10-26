"""Scoped diagnostics storage, cleanup and human-readable local exports."""

import json
import re
import shutil
import time
from pathlib import Path

from .lab_reports import write_json
from .model_files import copy_file


def diagnostic_folder(project, id, exists=True):
    if not re.fullmatch(r"[0-9a-f]{32}", id):
        raise ValueError("Invalid diagnostic ID")
    root = Path(project).resolve(strict=True)
    scope = (root / ".opweaver" / "diagnostics").resolve()
    folder = (scope / id).resolve(strict=exists)
    if not scope.is_relative_to(root) or not folder.is_relative_to(scope):
        raise ValueError("Diagnostic directory escapes the project")
    return folder


def cleanup_temporary(folder):
    folder = Path(folder).resolve(strict=True)
    target = folder / "temporary"
    if not target.exists():
        return True
    if target.is_symlink() or target.resolve().parent != folder:
        raise ValueError("Temporary cleanup target escapes its diagnostic directory")
    for attempt in range(4):
        try:
            shutil.rmtree(target)
            return True
        except PermissionError:
            if attempt == 3:
                return False
            time.sleep(0.1)
    return False


def markdown_report(report):
    lines = [
        "# OpWeaver Quantization Diagnostic",
        "",
        f"Diagnostic: `{report['id']}` · {report['status']}",
        "",
        "High error requires investigation; it does not establish a root cause.",
        "",
    ]
    for title, key in [
        ("Models and source experiment", "models"),
        ("Source lineage", "lineage"),
        ("Observed quantization/configuration", "quantization"),
        ("Diagnostic dataset and preprocessing", "dataset"),
        ("Capture/resource configuration", "config"),
        ("Resources and sample semantics", "resources"),
        ("Environment and tool versions", "environment"),
        ("Matching coverage", "matching_summary"),
        ("Current reference availability", "availability"),
        ("Actual captured batches (metadata only)", "captures"),
    ]:
        lines += [
            f"## {title}",
            "",
            "```json",
            json.dumps(report.get(key), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    lines += [
        "## Statistical semantics",
        "",
        report["sample_unit"],
        "",
        "MAE/MSE/relative mean use valid element counts; RMSE is sqrt(global MSE). Max error spans valid elements. Cosine is the mean of defined per-feed cosines. Weight tensors are evaluated once, separately from dataset batches.",
        "",
    ]
    for kind, title in [("activation", "Activation Error"), ("weight", "Weight Error")]:
        lines += [
            f"## {title}",
            "",
            "| Original node / Tensor | Status | MAE | RMSE | Max absolute | Cosine |",
            "|---|---|---:|---:|---:|---:|",
        ]
        rows = [v for v in report["records"] if v["kind"] == kind]
        rows.sort(
            key=lambda v: (
                (v.get("aggregate") or {}).get("mae")
                if (v.get("aggregate") or {}).get("mae") is not None
                else -1
            ),
            reverse=True,
        )
        for row in rows:
            m = row.get("aggregate") or {}
            label = (row["node_name"] or row["fp32_node_id"]) + " / " + row["fp32_tensor"]
            lines.append(
                "| "
                + " | ".join(
                    str(v).replace("|", "\\|").replace("\n", " ")
                    for v in [
                        label,
                        row["match_status"],
                        m.get("mae"),
                        m.get("rmse"),
                        m.get("max_abs_error"),
                        m.get("cosine_similarity"),
                    ]
                )
                + " |"
            )
        lines += [
            "",
            "Full matching methods, shapes, per-feed statistics and unmatched reasons are in the JSON report.",
            "",
        ]
        unavailable = [r for r in rows if r["match_status"] != "Matched"]
        if unavailable:
            lines += ["| Tensor | Status | Reason |", "|---|---|---|"]
            for row in unavailable:
                lines.append(
                    "| "
                    + " | ".join(
                        str(row[k]).replace("|", "\\|").replace("\n", " ")
                        for k in ("fp32_tensor", "match_status", "message")
                    )
                    + " |"
                )
            lines.append("")
    for title, key in [
        ("Limits", "limitations"),
        ("Warnings", "warnings"),
        ("Errors", "diagnostics"),
        ("Logs", "logs"),
    ]:
        lines += [
            f"## {title}",
            "",
            "```json",
            json.dumps(report.get(key), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    return "\n".join(lines)


def save_diagnostic(folder, report):
    write_json(folder / "diagnostic.json", report)
    temporary = folder / "report.tmp"
    temporary.write_text(markdown_report(report), encoding="utf-8")
    from .model_files import atomic_replace

    atomic_replace(temporary, folder / "report.md")


def export_diagnostic(project, report, directory):
    if report["status"] not in ("Succeeded", "Failed", "Cancelled"):
        raise ValueError("Wait until the diagnostic finishes before exporting")
    parent = Path(directory).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Select an existing export parent directory")
    target = parent / f"opweaver-diagnostic-{report['id']}"
    target.mkdir(exist_ok=False)
    write_json(target / "report.json", report)
    (target / "report.md").write_text(markdown_report(report), encoding="utf-8")
    source = diagnostic_folder(project, report["id"])
    for name in ("stdout.log", "stderr.log"):
        if (source / name).is_file():
            copy_file(source / name, target / name)
    return {"directory": str(target), "files": ["report.json", "report.md"]}
