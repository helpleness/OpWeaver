"""Development regression and separate-process adapter integration checks."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from .evaluate_language import pipeline
from .generate_data import load_events
from .language_model import load_classifier
from .train import load_checkpoint, seed_everything
from .train_evidence import open_store


def main(trained, out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    seed_everything(64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory = root / "runs/typed_round_02"
    model, tok, _ = load_checkpoint(memory / "model.pt", device)
    model.eval()
    classifier, tokenizer, _ = load_classifier(trained / "pretrained_compositional", device)
    frontends = {"pretrained_compositional": (classifier, tokenizer)}
    source = root / "runs/typed_comparison_02"
    previous = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    report = {}
    for label, seed in (("fresh0_both", 90391), ("fresh1_both", 90392)):
        events = {
            f"fresh{seed}-" + e["id"]: {
                **e,
                "id": f"fresh{seed}-" + e["id"],
                "owner": f"fresh{seed}-" + e["owner"],
            }
            for e in load_events(source / f"data_{seed}", "test")
        }
        records = previous["conditions"][label]["typed"]["records"]
        store = open_store(source / f"memory_{seed}_1.sqlite", model, tok)
        print("DEVELOPMENT " + label, flush=True)
        report[label] = pipeline(
            model,
            tok,
            frontends,
            store,
            [(events[r["id"]], r["task"]) for r in records],
            [r["text"] for r in records],
        )
        store.close()
    # Query subprocess receives no truth or historical text, only model, DB and question.
    prior = json.loads((memory / "restart_verification_template3.json").read_text(encoding="utf-8"))
    restart = []
    for r in prior["records"]:
        args = [
            sys.executable,
            "-m",
            "experiments.episodic_memory.demo",
            "--checkpoint",
            str(memory / "model.pt"),
            "--intent-checkpoint",
            str(trained / "pretrained_compositional"),
            "--db",
            str(memory / "restart_demo_template3.sqlite"),
            "--owner",
            r["owner"],
            "query",
            "--question",
            r["question"],
        ]

        def run():
            p = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                check=True,
            )
            return json.loads(p.stdout.strip().splitlines()[-1])

        first, second = run(), run()
        restart.append(
            dict(
                owner=r["owner"],
                expected=r["expected"],
                first=first,
                second=second,
                correct=first["answer"] == r["expected"],
                identical=first == second,
            )
        )
    report["restart"] = restart
    (out / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.trained, a.out)
