"""Paired 2x2 language-shift evaluation: identical facts, owners and questions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .evaluate import run_evaluation
from .generate_data import TASKS, load_events, question, render
from .models import pad_tokens
from .train import Corpus, build_store, load_checkpoint, seed_everything


@torch.no_grad()
def representation_audit(model, tokenizer, original, changed, train, device, exposure=None):
    # Separate embedding tables: a character seen only by the writer is unseen by the reader.
    event_seen = Counter(t for e in train for t in tokenizer.encode(e["text"]))
    query_seen = Counter(
        t
        for e in train
        for task in TASKS
        for t in tokenizer.encode(question(e, task, e["query_template"]))
    )
    if exposure:
        event_seen = Counter(dict.fromkeys(exposure["event_token_ids"], 1))
        query_seen = Counter(dict.fromkeys(exposure["query_token_ids"], 1))
    examples = [e for e in original if e["stored"]][:192]
    by_id = {e["id"]: e for e in changed}
    key_cos, slot_cos, qkey_cos, summary_cos = [], [], [], []
    aux = {kind: {t: [0, 0] for t in TASKS} for kind in ("original", "changed")}
    absent_event, absent_query = Counter(), Counter()
    intent_counts = {kind: [0, 0] for kind in ("original", "changed")}
    for start in range(0, len(examples), 32):
        batch = examples[start : start + 32]
        alt = [by_id[e["id"]] for e in batch]
        z0, k0 = model.encode_events(
            pad_tokens([tokenizer.encode(e["text"]) for e in batch], device)
        )
        z1, k1 = model.encode_events(pad_tokens([tokenizer.encode(e["text"]) for e in alt], device))
        key_cos.extend(F.cosine_similarity(k0, k1).tolist())
        slot_cos.extend(F.cosine_similarity(z0, z1, dim=-1).mean(-1).tolist())
        for kind, z in (("original", z0), ("changed", z1)):
            for j, task in enumerate(TASKS):
                pred = model.auxiliary[j](z[:, model.slot_for_task[j]]).argmax(-1).tolist()
                aux[kind][task][0] += sum(p == e["values"][task] for p, e in zip(pred, batch))
                aux[kind][task][1] += len(batch)
        for task in TASKS:
            q0 = [tokenizer.encode(question(e, task, e["query_template"])) for e in batch]
            q1 = [tokenizer.encode(question(e, task, e["query_template"])) for e in alt]
            h0, qk0 = model.encode_questions(pad_tokens(q0, device))
            h1, qk1 = model.encode_questions(pad_tokens(q1, device))
            qkey_cos.extend(F.cosine_similarity(qk0, qk1).tolist())
            summary_cos.extend(F.cosine_similarity(h0, h1).tolist())
            if model.config.intent_bottleneck:
                for kind, h in (("original", h0), ("changed", h1)):
                    pred = model.intent_head(h).argmax(-1)
                    intent_counts[kind][0] += int((pred == TASKS.index(task)).sum())
                    intent_counts[kind][1] += len(batch)
            absent_query.update(t for row in q1 for t in row if not query_seen[t])
        absent_event.update(
            t for e in alt for t in tokenizer.encode(e["text"]) if not event_seen[t]
        )
    return {
        "audited_events": len(examples),
        "event_key_cosine": sum(key_cos) / len(key_cos),
        "event_slot_cosine": sum(slot_cos) / len(slot_cos),
        "query_key_cosine": sum(qkey_cos) / len(qkey_cos),
        "query_summary_cosine": sum(summary_cos) / len(summary_cos),
        "predicted_intent_accuracy": {
            kind: a / n if n else None for kind, (a, n) in intent_counts.items()
        },
        "event_aux_accuracy": {
            kind: {t: a / n for t, (a, n) in tasks.items()} for kind, tasks in aux.items()
        },
        "tokens_unseen_by_event_encoder": {
            tokenizer.spec["vocab"][i]: n for i, n in absent_event.items()
        },
        "tokens_unseen_by_query_encoder": {
            tokenizer.spec["vocab"][i]: n for i, n in absent_query.items()
        },
        "note": "Exposure uses materialized training views when language_exposure.json exists, otherwise canonical training data. Cosine is diagnostic, not causal proof.",
    }


def factorial(run, out, count=600, template=2, split="test"):
    if out.exists():
        raise FileExistsError("Choose a new evaluation directory")
    out.mkdir(parents=True)
    seed_everything(901)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_checkpoint(run / "model.pt", device)
    model.eval()
    base = load_events(run / "data", split)
    changed = [
        {
            **e,
            "text": render(e, template),
            "query_template": template,
            "reference_query_template": e["query_template"],
        }
        for e in base
    ]
    changed_store = out / "changed_events.sqlite"
    build_store(changed_store, model, tokenizer, Corpus(changed, tokenizer), device, max_members=32)
    exposure_path = run / "language_exposure.json"
    exposure = (
        json.loads(exposure_path.read_text(encoding="utf-8")) if exposure_path.exists() else None
    )
    audit = representation_audit(
        model, tokenizer, base, changed, load_events(run / "data", "train"), device, exposure
    )
    result = {
        "run": str(run.resolve()),
        "template": template,
        "split": split,
        "count": count,
        "audit": audit,
        "conditions": {},
    }
    for condition, change_event, change_query in (
        ("original", False, False),
        ("question_only", False, True),
        ("event_only", True, False),
        ("both", True, True),
    ):
        events = [
            {
                **e,
                "text": alt["text"] if change_event else e["text"],
                "query_template": template if change_query else e["query_template"],
                "reference_query_template": e["query_template"],
            }
            for e, alt in zip(base, changed)
        ]
        modes = ("clustered", "oracle", "zero")
        if condition == "question_only":
            modes += ("reference_summary", "reference_address")
        print(f"CONDITION {condition}", flush=True)
        evaluation = run_evaluation(
            run,
            split=split,
            count=count,
            events=events,
            store_path=changed_store if change_event else None,
            modes=modes,
        )
        result["conditions"][condition] = evaluation
        (out / f"{condition}.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (out / "factorial.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=600)
    p.add_argument("--template", type=int, default=2)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    a = p.parse_args()
    factorial(a.run, a.out, a.count, a.template, a.split)
