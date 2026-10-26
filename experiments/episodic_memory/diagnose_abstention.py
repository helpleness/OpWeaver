"""Label-assisted interventions for diagnosis ONLY, never deployed inference."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from .generate_data import OFFSETS, TASKS, load_events, question
from .models import pad_tokens
from .store import MemoryStore, writer_version
from .train import load_checkpoint, seed_everything


def paired_queries(events, count=600, seed=901):
    rng = random.Random(seed)
    present = [e for e in events if e["stored"]]
    missing = [e for e in events if not e["stored"]]
    return [
        (rng.choice(missing if i % 5 == 0 else present), TASKS[i % len(TASKS)])
        for i in range(count)
    ]


@torch.no_grad()
def diagnose(run, changed_store, out, count=600):
    if out.exists():
        raise FileExistsError(out)
    seed_everything(901)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_checkpoint(run / "model.pt", device)
    model.eval()
    events = load_events(run / "data", "test")
    pairs = paired_queries(events, count)
    report = {"run": str(run), "count": count, "conditions": {}}
    for label, event_changed, query_changed in (
        ("original", False, False),
        ("question_only", False, True),
        ("event_only", True, False),
        ("both", True, True),
    ):
        store = MemoryStore(
            changed_store if event_changed else run / "memory_test.sqlite",
            writer_version(model, tok.spec),
            (model.config.slots, model.config.dim),
            model.config.key_dim,
        )
        counts, records = Counter(), []
        for e, task in pairs:
            h, q = model.encode_questions(
                pad_tokens(
                    [tok.encode(question(e, task, 3 if query_changed else e["query_template"]))],
                    device,
                )
            )
            ids, _ = store.retrieve(e["owner"], q[0].cpu().numpy(), top_clusters=2)
            flat, _ = store.retrieve(e["owner"], q[0].cpu().numpy(), flat=True)
            z, k = store.fetch(e["owner"], ids)
            logits, _ = model.read(
                h, q, torch.tensor(z[None], device=device), torch.tensor(k[None], device=device)
            )
            pred = int(logits.argmax(-1))
            forced = int(logits[:, 1:].argmax(-1)) + 1
            target = OFFSETS[task] + e["values"][task] if e["stored"] else 0
            row = {
                "id": e["id"],
                "task": task,
                "stored": e["stored"],
                "target": target,
                "prediction": pred,
                "conditional_answer": forced,
                "support_found": e["id"] in ids,
                "flat_support_found": e["id"] in flat,
                "unknown_margin": float(logits[0, 0] - logits[0, 1:].max()),
            }
            counts["present" if e["stored"] else "missing"] += 1
            counts["correct"] += pred == target
            if not e["stored"]:
                counts["missing_false_answer"] += pred != 0
            else:
                counts["present_correct"] += pred == target
                counts["support_found"] += e["id"] in ids
                counts["flat_support_found"] += e["id"] in flat
                if pred == 0:
                    counts["present_unknown"] += 1
                    if e["id"] not in ids:
                        bucket = "retrieval_miss"
                    elif forced == target:
                        bucket = "correct_conditional_answer_vetoed"
                    else:
                        true_z, true_k = store.fetch(e["owner"], [e["id"]])
                        single, _ = model.read(
                            h,
                            q,
                            torch.tensor(true_z[None], device=device),
                            torch.tensor(true_k[None], device=device),
                        )
                        row["support_only_conditional"] = int(single[:, 1:].argmax(-1)) + 1
                        bucket = (
                            "candidate_interference"
                            if row["support_only_conditional"] == target
                            else "reading_or_encoding_error"
                        )
                        j = TASKS.index(task)
                        value = model.auxiliary[j](
                            torch.tensor(true_z[:, model.slot_for_task[j]], device=device)
                        ).argmax(-1)
                        row["label_assisted_slot_decode_correct"] = int(value) == e["values"][task]
                    row["diagnosis"] = bucket
                    counts[bucket] += 1
                elif pred != target:
                    counts["wrong_nonunknown"] += 1
            records.append(row)
        store.close()
        report["conditions"][label] = {"counts": dict(counts), "records": records}
        print(json.dumps({"condition": label, **counts}), flush=True)
    report["warning"] = (
        "Conditional argmax, true-event isolation and true-task slot probing use diagnostic interventions. They are not safe answer policies or unique causal identifications."
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--changed-store", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    diagnose(args.run, args.changed_store, args.out)
