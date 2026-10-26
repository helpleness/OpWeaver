"""Held-out histories. Answer labels are confined to this evaluator, never the DB."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .generate_data import ANSWERS, OFFSETS, TASKS, load_events, question
from .models import pad_tokens
from .store import MemoryStore, writer_version
from .train import load_checkpoint, seed_everything


@torch.no_grad()
def run_evaluation(
    run: Path,
    split="test",
    count=600,
    checkpoint="model.pt",
    top_clusters=2,
    modes=("oracle", "clustered", "flat", "zero", "shuffled", "no_appraisal"),
    seed=901,
    events=None,
    store_path=None,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_checkpoint(run / checkpoint, device)
    model.eval()
    version = writer_version(model, tokenizer.spec)
    store_path = store_path or run / f"memory_{split}.sqlite"
    store = MemoryStore(
        store_path,
        version,
        (model.config.slots, model.config.dim),
        model.config.key_dim,
    )
    events = events if events is not None else load_events(run / "data", split)
    by_owner = defaultdict(list)
    for e in events:
        if e["stored"]:
            by_owner[e["owner"]].append(e["id"])
    rng = random.Random(seed)
    present = [e for e in events if e["stored"]]
    missing = [e for e in events if not e["stored"]]
    queries = []
    for i in range(count):
        # Fixed 20% genuinely unobserved events, balanced answer types.
        e = rng.choice(missing if i % 5 == 0 else present)
        task = TASKS[i % len(TASKS)]
        queries.append((e, task))
    summaries, qkeys = [], []
    for start in range(0, count, 32):
        subset = queries[start : start + 32]
        h, k = model.encode_questions(
            pad_tokens(
                [tokenizer.encode(question(e, t, e["query_template"])) for e, t in subset], device
            )
        )
        summaries.extend(h.unbind())
        qkeys.extend(k.unbind())
    reference_summaries, reference_keys = [], []
    if any(m.startswith("reference_") for m in modes):
        for start in range(0, count, 32):
            subset = queries[start : start + 32]
            h, k = model.encode_questions(
                pad_tokens(
                    [
                        tokenizer.encode(question(e, t, e["reference_query_template"]))
                        for e, t in subset
                    ],
                    device,
                )
            )
            reference_summaries.extend(h.unbind())
            reference_keys.extend(k.unbind())
    results = {
        "split": split,
        "checkpoint": checkpoint,
        "count": count,
        "seed": seed,
        "top_clusters": top_clusters,
        "store": store.counts(),
        "modes": {},
    }
    examples = []
    for mode in modes:
        mode_keys = reference_keys if mode == "reference_address" else qkeys
        mode_summaries = reference_summaries if mode == "reference_summary" else summaries
        totals = defaultdict(int)
        outcomes = []
        task_counts = defaultdict(lambda: [0, 0])
        elapsed = candidates_total = payload_total = 0
        mode_rng = random.Random(seed + 1)
        for i, (e, task) in enumerate(queries):
            query = mode_keys[i].cpu().numpy()
            started = time.perf_counter()
            selected, candidates = store.retrieve(
                e["owner"], query, top_clusters=top_clusters, flat=mode == "flat"
            )
            if mode == "oracle":
                others = [eid for eid in by_owner[e["owner"]] if eid != e["id"]]
                selected = mode_rng.sample(others, min(7 if e["stored"] else 8, len(others)))
                if e["stored"]:
                    selected.append(e["id"])
                mode_rng.shuffle(selected)
            z, keys = store.fetch(e["owner"], selected)
            # Ablations preserve routing and keys, changing only payload content.
            if mode == "zero":
                z[:] = 0
            elif mode == "shuffled":
                z = np.roll(z, 1, axis=0).copy()
            elif mode == "no_appraisal":
                z[:, 2:4] = 0
            logits, _ = model.read(
                mode_summaries[i][None],
                mode_keys[i][None],
                torch.tensor(z[None], device=device),
                torch.tensor(keys[None], device=device),
            )
            pred = int(logits.argmax(-1).item())  # synchronizes GPU before elapsed measurement
            elapsed += time.perf_counter() - started
            target = OFFSETS[task] + e["values"][task] if e["stored"] else 0
            correct = pred == target
            outcomes.append(
                {
                    "event_id": e["id"],
                    "task": task,
                    "stored": e["stored"],
                    "target": target,
                    "prediction": pred,
                    "support_found": e["id"] in selected,
                }
            )
            totals["correct"] += correct
            totals["present"] += e["stored"]
            totals["present_correct"] += correct and e["stored"]
            totals["missing"] += not e["stored"]
            totals["missing_false_answer"] += pred != 0 and not e["stored"]
            totals["support_found"] += e["stored"] and e["id"] in selected
            totals["support_found_correct"] += correct and e["stored"] and e["id"] in selected
            if e["stored"]:
                task_counts[task][0] += correct
                task_counts[task][1] += 1
            candidates_total += candidates
            payload_total += len(selected) * model.config.slots * model.config.dim * 2
            if mode == "clustered" and (i < 20 or (correct and e["stored"] and len(examples) < 40)):
                examples.append(
                    {
                        "query": question(e, task, e["query_template"]),
                        "target": ANSWERS[target],
                        "prediction": ANSWERS[pred],
                        "support_found": e["id"] in selected,
                        "correct": bool(correct),
                        "event_id": e["id"],
                        "selected_ids": selected,
                    }
                )
        results["modes"][mode] = {
            "accuracy": totals["correct"] / count,
            "present_accuracy": totals["present_correct"] / max(1, totals["present"]),
            "recall_at_8": totals["support_found"] / max(1, totals["present"]),
            "accuracy_when_found": totals["support_found_correct"]
            / max(1, totals["support_found"]),
            "missing_false_answer_rate": totals["missing_false_answer"] / max(1, totals["missing"]),
            "mean_read_and_answer_ms": elapsed / count * 1000,
            "mean_candidate_keys": candidates_total / count,
            "mean_payload_bytes": payload_total / count,
            "per_task_present_accuracy": {t: a / max(1, n) for t, (a, n) in task_counts.items()},
            "counts": dict(totals),
            "outcomes": outcomes,
        }
        print(
            json.dumps(
                {
                    "mode": mode,
                    **{k: v for k, v in results["modes"][mode].items() if k != "outcomes"},
                }
            ),
            flush=True,
        )
    # Actual close/reopen verification, retaining exactly the same model and query.
    e, _ = queries[-1]
    q = qkeys[-1].cpu().numpy()
    ids_before, _ = store.retrieve(e["owner"], q, top_clusters=top_clusters)
    z_before, k_before = store.fetch(e["owner"], ids_before)
    store.close()
    reopened = MemoryStore(
        store_path,
        version,
        (model.config.slots, model.config.dim),
        model.config.key_dim,
    )
    ids_after, _ = reopened.retrieve(e["owner"], q, top_clusters=top_clusters)
    z_after, k_after = reopened.fetch(e["owner"], ids_after)
    results["reopen_payload_exact"] = bool(
        ids_before == ids_after
        and np.array_equal(z_before, z_after)
        and np.array_equal(k_before, k_after)
    )
    reopened.close()
    results["examples"] = examples
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--count", type=int, default=600)
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--top-clusters", type=int, default=2)
    args = p.parse_args()
    result = run_evaluation(args.run, args.split, args.count, args.checkpoint, args.top_clusters)
    path = args.run / f"evaluation_{args.split}_{Path(args.checkpoint).stem}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")
