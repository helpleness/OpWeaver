"""Verify exported evidence bindings and package final artifacts with hashes."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .luna_review import validate


def package():
    root = Path(__file__).resolve().parents[2]
    out = root / "exports/episodic_memory_report_20261026"
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    for suite in ("long", "fresh", "public", "regression"):
        records = [r for r in data["records"] if r["suite"] == suite]
        validate([r["packet"] for r in records], [r["review"] for r in records])
        for r in records:
            answer = r["original_output"]["result"].get("answer")
            if answer is not None:
                assert answer == r["packet"]["answer"]
    assert len(data["records"]) == 376
    html = (out / "逐题对照与证据.html").read_text(encoding="utf-8")
    assert html.count('<article class="answer ') == 376
    assert html.count('<section class="case"') == 136
    shutil.copy2(Path(__file__), out / "evidence/code_snapshot" / Path(__file__).name)
    shutil.copy2(Path(__file__).parent / "README.md", out / "evidence/code_snapshot/README.md")
    audit = dict(
        record_count=376,
        paired_case_count=136,
        review_binding_and_quote_validation="passed for all four suites",
        raw_output_identity="passed",
        semantic_labels="AI judgments with evidence adjudication, not human gold",
        original_runtime_review_status="Some original output manifests say pending because they predate review; final review status is in each review directory audit.json.",
        document_pages=14,
        render_engine="Microsoft Word 16 PDF export and bundled Poppler",
        renderer_fallback_reason="Packaged render_docx.py failed: LibreOffice soffice.exe was not found on PATH",
        visual_review="All 14 page images inspected; final changed pages 1 and 7 re-inspected; other 12 pages pixel-identical to inspected revision",
        attention_tests=3,
        long_cache_sensitivity="3/3 answers identical to one-shot full-history prefill; 2 additional short examples identical",
        excluded="Full Qwen backbone, multi-GB source corpus, and internal page images",
    )
    (out / "EXPORT_VALIDATION.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    entries = [
        dict(
            path=p.relative_to(out).as_posix(),
            bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    ]
    (out / "MANIFEST.json").write_text(
        json.dumps(dict(files=entries), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, arcname=out.name + "/" + p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for e in entries:
            assert hashlib.sha256(z.read(out.name + "/" + e["path"])).hexdigest() == e["sha256"]
    print(
        json.dumps(
            dict(
                files=len(entries) + 1,
                archive=str(archive),
                bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    package()
