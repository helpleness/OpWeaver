"""Exposed regression set plus independent-process persisted-memory verification."""

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from .evaluate_semantic_memory import outcome
from .generate_data import FIELDS, TASKS
from .qwen_memory import save_json
from .semantic_memory import SemanticMemory


def main(args):
    torch.set_num_threads(4)
    out = args.run / "verification"
    out.mkdir(exist_ok=False)
    model = SemanticMemory.load(args.run)
    threshold = json.loads((args.run / "calibration/calibration.json").read_text(encoding="utf-8"))[
        "threshold"
    ]
    suite = Path("experiments/episodic_memory/runs/qwen_scale_comparison_01")
    events = json.loads((suite / "events.json").read_text(encoding="utf-8"))
    items = json.loads((suite / "selection.json").read_text(encoding="utf-8"))
    lookup = {e["id"]: e for e in events}
    store = model.store(out / "regression.sqlite", create=True)
    model.write_events(store, [e for e in events if e["stored"]])
    store.close()
    store = model.store(out / "regression.sqlite")
    rows = []
    for item in items:
        e = lookup[item["id"]]
        field = TASKS.index(item["task"])
        result = model.ask_checked(store, e["owner"], item["novel"], threshold)
        rows.append(
            dict(
                id=e["id"],
                field=field,
                question=item["novel"],
                result=result,
                outcome=outcome(
                    result["answer"], FIELDS[item["task"]][e["values"][item["task"]]], field
                ),
            )
        )
    store.close()
    save_json(
        out / "regression.json",
        dict(
            note="Previously exposed 60-query regression, not a new test",
            counts=dict(Counter(r["outcome"] for r in rows)),
            rows=rows,
        ),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    cases = json.loads((args.suite / "test.json").read_text(encoding="utf-8"))
    evaluated = [
        json.loads(s)
        for s in (args.run / "test/results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chosen = [
        next(r for r in evaluated if r["outcome"] == "correct"),
        next(r for r in evaluated if r["kind"] == "conflict"),
    ]
    byid = {c["id"]: c for c in cases}
    results = []
    for row in chosen:
        c = byid[row["id"]]
        cmd = [
            sys.executable,
            "-m",
            "experiments.episodic_memory.semantic_demo",
            "--run",
            str(args.run),
            "--db",
            str(args.run / "test/memories.sqlite"),
            "--owner",
            c["owner"],
            "ask",
            "--question",
            c["question"],
        ]
        process = subprocess.run(
            cmd,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        result = json.loads(process.stdout.splitlines()[-1])
        assert result["answer"] == row["result"]["answer"]
        results.append(
            dict(id=c["id"], answer=result["answer"], reason=result["reason"], exact=True)
        )
    save_json(out / "restart.json", results)
    print(
        json.dumps(
            dict(regression=dict(Counter(r["outcome"] for r in rows)), restart_exact=len(results))
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--suite", type=Path, default=Path("experiments/episodic_memory/runs/semantic_suite_01")
    )
    main(p.parse_args())
