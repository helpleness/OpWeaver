"""Separate language, routing and memory probes on OLD development failures only."""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .evidence_policy import infer
from .generate_data import TASKS, load_events
from .models import pad_tokens
from .train import load_checkpoint, seed_everything
from .train_evidence import open_store


@torch.no_grad()
def main():
    root = Path(__file__).parent
    seed_everything(61)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_checkpoint(root / "runs/typed_round_02/model.pt", device)
    model.eval()
    source = root / "runs/typed_comparison_02"
    previous = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    report = {"suites": {}, "interventions": []}
    for condition, seed in (("fresh0_both", 90391), ("fresh1_both", 90392)):
        events = {
            f"fresh{seed}-" + e["id"]: e for e in load_events(source / f"data_{seed}", "test")
        }
        store = open_store(source / f"memory_{seed}_1.sqlite", model, tok)
        counts, confusion, rows = Counter(), Counter(), []
        for r in previous["conditions"][condition]["typed"]["records"]:
            if not r["supported"]:
                continue
            e, task = events[r["id"]], TASKS.index(r["task"])
            owner = f"fresh{seed}-" + e["owner"]
            h, q = model.encode_questions(pad_tokens([tok.encode(r["text"])], device))
            p = model.intent_head(h).softmax(-1)
            predicted = int(p.argmax())
            onehot = F.one_hot(torch.tensor([task], device=device), len(TASKS)).float()
            oracle = infer(model, store, owner, h, q, intent_override=onehot)
            z, _ = store.fetch(owner, [r["id"]])
            value = model.auxiliary[task](
                torch.tensor(z[:, model.slot_for_task[task]], device=device)
            ).softmax(-1)
            counts["n"] += 1
            counts["language_correct"] += predicted == task
            counts["language_high_conf_wrong"] += predicted != task and float(p.max()) > 0.8
            counts["memory_oracle_argmax_correct"] += int(value.argmax()) == e["values"][r["task"]]
            counts["memory_oracle_confident_correct"] += (
                int(value.argmax()) == e["values"][r["task"]] and float(value.max()) > 0.8
            )
            counts["oracle_field_pipeline_correct"] += oracle["prediction"] == r["target"]
            counts["oracle_field_pipeline_unknown"] += oracle["prediction"] == 0
            counts["oracle_field_pipeline_wrong"] += oracle["prediction"] not in (0, r["target"])
            confusion[f"{r['task']}->{TASKS[predicted]}"] += 1
            rows.append(
                {
                    "id": r["id"],
                    "task": r["task"],
                    "intent": TASKS[predicted],
                    "confidence": float(p.max()),
                    "oracle_field_prediction": oracle["prediction"],
                    "slot_correct": int(value.argmax()) == e["values"][r["task"]],
                }
            )
        store.close()
        report["suites"][condition] = {
            "counts": dict(counts),
            "confusion": dict(confusion),
            "records": rows,
        }
        print(json.dumps({"condition": condition, **counts}), flush=True)
    probes = [
        ("person", "和我交流的人是哪位？"),
        ("person", "那件事发生时，和我交流的人是哪位？"),
        ("person", "和我交流的人是哪位？那件事发生时。"),
        ("reason", "为什么产生那种感受？"),
        ("reason", "回忆当时，我为什么产生那种感受？"),
        ("emotion", "回忆当时，我产生的感受是什么？"),
    ]
    for task, stem in probes:
        rows = []
        for anchor in ("画册项目第1次讨论", "花园项目第15次讨论", "软件项目第3次讨论"):
            for include in (False, True):
                text = anchor + "，" + stem if include else stem
                h, _ = model.encode_questions(pad_tokens([tok.encode(text)], device))
                p = model.intent_head(h).softmax(-1)
                rows.append(
                    dict(text=text, prediction=TASKS[int(p.argmax())], confidence=float(p.max()))
                )
        report["interventions"].append(dict(target=task, stem=stem, records=rows))
    (root / "runs/understanding_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
