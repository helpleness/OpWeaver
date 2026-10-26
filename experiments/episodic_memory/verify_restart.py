"""Fresh processes + identical question, different persisted experience.

The child query command receives only model, DB, owner and question, not labels/text.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .generate_data import FIELDS, question, render


def verify(run: Path, template=0):
    run = run.resolve()
    suffix = f"_template{template}" if template else ""
    db = run / f"restart_demo{suffix}.sqlite"
    if db.exists():
        raise FileExistsError("Use a fresh DB for this verification; preserving existing evidence")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def command(owner, action, flag, text):
        args = [
            sys.executable,
            "-m",
            "experiments.episodic_memory.demo",
            "--checkpoint",
            str(run / "model.pt"),
            "--db",
            str(db),
            "--owner",
            owner,
            action,
            flag,
            text,
        ]
        result = subprocess.run(
            args, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", check=True
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    records = []
    for owner, feeling, reason in (("case_a", "尴尬", "担心评价"), ("case_b", "安心", "获得支持")):
        episode = {
            "anchor": "画册项目第1次讨论",
            "values": {
                "person": 0,
                "day": 2,
                "event": 0,
                "emotion": FIELDS["emotion"].index(feeling),
                "reason": FIELDS["reason"].index(reason),
                "focus": 0,
            },
        }
        event = render(episode, template)
        written = command(owner, "write", "--text", event)
        q = question(episode, "emotion", template)
        first = command(owner, "query", "--question", q)
        second = command(owner, "query", "--question", q)
        records.append(
            {
                "owner": owner,
                "expected": feeling,
                "question": q,
                "event_text": event,
                "write": written,
                "first_process": first,
                "second_process": second,
                "correct": first["answer"] == feeling,
                "identical_across_processes": first == second,
            }
        )
    report = {
        "template": template,
        "records": records,
        "all_correct": all(r["correct"] for r in records),
        "exact_across_fresh_processes": all(r["identical_across_processes"] for r in records),
        "same_question_different_memories_change_answer": records[0]["first_process"]["answer"]
        != records[1]["first_process"]["answer"],
    }
    (run / f"restart_verification{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--template", type=int, choices=(0, 2, 3), default=0)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.template), ensure_ascii=False, indent=2))
