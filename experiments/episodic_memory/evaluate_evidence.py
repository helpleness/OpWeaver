"""Paired accuracy AND unsupported-answer stress tests; no truth enters the reader."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .diagnose_abstention import paired_queries
from .evidence_policy import infer
from .generate_data import OFFSETS, TASKS, generate, load_events, question, render
from .models import pad_tokens
from .train import Corpus, build_store, load_checkpoint, seed_everything
from .train_evidence import open_store


@torch.no_grad()
def suite(models, tokenizer, store, pairs, changed_questions=False, unsupported=None):
    if any(m.config.typed_reader for m, _ in models.values()):
        raise ValueError("Typed readers have their own question state; use evaluate_typed instead")
    device = next(models["old_cluster"][0].parameters()).device
    states = []
    for start in range(0, len(pairs), 64):
        chunk = pairs[start : start + 64]
        h, q = models["old_cluster"][0].encode_questions(
            pad_tokens(
                [
                    tokenizer.encode(
                        question(e, task, 3 if changed_questions else e["query_template"])
                    )
                    for e, task in chunk
                ],
                device,
            )
        )
        states.extend(zip(h.unbind(), q.unbind()))
    summaries = {}
    for name, (model, retry) in models.items():
        counts, records = Counter(), []
        for (e, task), (h, q) in zip(pairs, states):
            exclude = [e["id"]] if unsupported == "removed" else []
            r = infer(
                model,
                store,
                e["owner"],
                h[None],
                q[None],
                retry=retry,
                exclude=exclude,
                zero=unsupported == "zero",
            )
            supported = e["stored"] and not unsupported
            target = OFFSETS[task] + e["values"][task] if supported else 0
            pred = r["prediction"]
            found = any(e["id"] in a["ids"] for a in r["attempts"]) if supported else False
            counts["present" if supported else "unsupported"] += 1
            counts["correct"] += pred == target
            counts["retries"] += len(r["attempts"]) > 1
            if supported:
                counts["present_correct"] += pred == target
                counts["present_unknown"] += pred == 0
                counts["wrong_nonunknown"] += pred not in (0, target)
                counts["support_found"] += found
                if pred == 0:
                    counts["unknown_with_support" if found else "unknown_without_support"] += 1
                    counts["vetoed_correct_conditional"] += (
                        found and r["attempts"][-1]["conditional_answer"] == target
                    )
            else:
                counts["unsupported_false_answer"] += pred != 0
            records.append(
                {
                    "id": e["id"],
                    "task": task,
                    "target": target,
                    "supported": bool(supported),
                    "prediction": pred,
                    "support_found": found,
                    "attempts": r["attempts"],
                }
            )
        metrics = {
            "counts": dict(counts),
            "present_accuracy": counts["present_correct"] / max(1, counts["present"]),
            "unsupported_false_answer_rate": counts["unsupported_false_answer"]
            / max(1, counts["unsupported"]),
            "support_recall": counts["support_found"] / max(1, counts["present"]),
            "retry_rate": counts["retries"] / len(pairs),
        }
        print(json.dumps({"model": name, **metrics}), flush=True)
        summaries[name] = {**metrics, "records": records}
    return summaries


def evaluate(base, trained, changed_store, output, candidate=None, stress_seed=90271):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed_everything(901)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old, tok, _ = load_checkpoint(base / "model.pt", device)
    joint, _, _ = load_checkpoint(trained / "mixed_joint" / "model.pt", device)
    verified, _, _ = load_checkpoint(trained / "verified" / "model.pt", device)
    for m in (old, joint, verified):
        m.eval()
    models = {
        "old_cluster": (old, False),
        "old_retry": (old, True),
        "mixed_joint_retry": (joint, True),
        "verified_retry": (verified, True),
    }
    if candidate:
        semantic, _, _ = load_checkpoint(candidate / "model.pt", device)
        semantic.eval()
        models["semantic_verified_retry"] = (semantic, True)
    events = load_events(base / "data", "test")
    pairs = paired_queries(events)
    stores = [
        open_store(base / "memory_test.sqlite", old, tok),
        open_store(changed_store, old, tok),
    ]
    result = {
        "base": str(base),
        "trained": str(trained),
        "conditions": {},
        "negative_suites": {},
        "stress_seed": stress_seed,
        "thresholds": {
            "evidence": verified.config.evidence_threshold,
            "value": verified.config.value_threshold,
        },
    }
    if candidate:
        result["candidate"] = str(candidate)
        result["candidate_thresholds"] = {
            "evidence": semantic.config.evidence_threshold,
            "value": semantic.config.value_threshold,
        }
    for label, ei, qi in (
        ("original", 0, False),
        ("question_only", 0, True),
        ("event_only", 1, False),
        ("both", 1, True),
    ):
        print(f"CONDITION {label}", flush=True)
        result["conditions"][label] = suite(models, tok, stores[ei], pairs, qi)
    present_pairs = [(e, t) for e, t in pairs if e["stored"]]
    for kind in ("removed", "zero"):
        print(f"NEGATIVE {kind}", flush=True)
        result["negative_suites"][kind] = suite(models, tok, stores[1], present_pairs, True, kind)
    for store in stores:
        store.close()
    # New random facts and explicit owner namespace, untouched by training/calibration.
    generated = output / "stress_data"
    generate(
        generated,
        seed=stress_seed,
        train_histories=1,
        eval_histories=16,
        train_events=8,
        test_events=96,
    )
    stress = [
        {**e, "owner": "stress-" + e["owner"], "id": "stress-" + e["id"]}
        for e in load_events(generated, "test")
    ]
    negatives = [(e, task) for e in stress if not e["stored"] for task in TASKS]
    for changed in (False, True):
        rows = [{**e, "text": render(e, 3) if changed else e["text"]} for e in stress]
        path = output / f"stress_{int(changed)}.sqlite"
        build_store(path, old, tok, Corpus(rows, tok), device, max_members=32)
        store = open_store(path, old, tok)
        label = "new_owners_both" if changed else "new_owners_original"
        print(f"NEGATIVE {label}", flush=True)
        result["negative_suites"][label] = suite(models, tok, store, negatives, changed)
        if changed:
            print("FRESH POSITIVE AND NEGATIVE HISTORY", flush=True)
            result["fresh_history"] = suite(models, tok, store, paired_queries(stress), True)
        store.close()
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--changed-store", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--stress-seed", type=int, default=90271)
    a = p.parse_args()
    evaluate(a.base, a.trained, a.changed_store, a.out, a.candidate, a.stress_seed)
