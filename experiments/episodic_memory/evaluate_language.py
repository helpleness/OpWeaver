"""Independent language tests + frozen-memory pipeline, on new structures and facts."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from .evaluate_typed import unique_pairs
from .evidence_policy import infer
from .generate_data import OFFSETS, TASKS, generate, load_events, question
from .language_data import anchor, compose, load, new_event
from .language_model import load_classifier, predict
from .models import pad_tokens
from .store import writer_version
from .train import Corpus, build_store, load_checkpoint, seed_everything
from .train_evidence import open_store


@torch.no_grad()
def language_suite(memory_model, old_tok, frontends, records):
    device = next(memory_model.parameters()).device
    texts = [r["text"] for r in records]
    labels = torch.tensor([r["label"] for r in records], device=device)
    probabilities = []
    for start in range(0, len(texts), 64):
        h, _ = memory_model.encode_questions(
            pad_tokens([old_tok.encode(t) for t in texts[start : start + 64]], device)
        )
        probabilities.append(memory_model.intent_head(h).softmax(-1))
    outputs = {"previous": torch.cat(probabilities)}
    outputs.update({name: predict(model, tok, texts) for name, (model, tok) in frontends.items()})
    results = {}
    for name, p in outputs.items():
        confidence, pred = p.max(-1)
        confusion = Counter(f"{TASKS[int(a)]}->{TASKS[int(b)]}" for a, b in zip(labels, pred))
        results[name] = dict(
            n=len(records),
            correct=int((pred == labels).sum()),
            high_conf_wrong=int(((pred != labels) & (confidence > 0.8)).sum()),
            low_confidence=int((confidence <= 0.8).sum()),
            confusion=dict(confusion),
            predictions=pred.tolist(),
            confidence=confidence.tolist(),
        )
        print(
            json.dumps(
                {
                    "model": name,
                    **{
                        k: v
                        for k, v in results[name].items()
                        if k not in ("confusion", "predictions", "confidence")
                    },
                }
            ),
            flush=True,
        )
    return {"records": records, "metrics": results}


@torch.no_grad()
def pipeline(memory_model, old_tok, frontends, store, pairs, texts, negative=None):
    device = next(memory_model.parameters()).device
    states = []
    for start in range(0, len(texts), 64):
        h, q = memory_model.encode_questions(
            pad_tokens([old_tok.encode(t) for t in texts[start : start + 64]], device)
        )
        states.extend(zip(h.unbind(), q.unbind()))
    options = {"previous": None}
    options.update({name: predict(model, tok, texts) for name, (model, tok) in frontends.items()})
    results = {}
    for name, p in options.items():
        counts = Counter(
            dict(
                present=0,
                correct=0,
                unknown=0,
                wrong=0,
                unsupported=0,
                false_answer=0,
                support_found=0,
                language_correct=0,
            )
        )
        records = []
        for i, ((e, task), (h, q), text) in enumerate(zip(pairs, states, texts)):
            intent = memory_model.intent_head(h[None]).softmax(-1) if p is None else p[i : i + 1]
            r = infer(
                memory_model,
                store,
                e["owner"],
                h[None],
                q[None],
                exclude=[e["id"]] if negative == "removed" else [],
                zero=negative == "zero",
                intent_override=None if p is None else intent,
            )
            supported = e["stored"] and negative is None
            target = OFFSETS[task] + e["values"][task] if supported else 0
            pred = r["prediction"]
            found = any(e["id"] in a["ids"] for a in r["attempts"])
            if supported:
                counts["present"] += 1
                counts["correct"] += pred == target
                counts["unknown"] += pred == 0
                counts["wrong"] += pred not in (0, target)
                counts["support_found"] += found
            else:
                counts["unsupported"] += 1
                counts["false_answer"] += pred != 0
            counts["language_correct"] += int(intent.argmax()) == TASKS.index(task)
            records.append(
                dict(
                    id=e["id"],
                    task=task,
                    text=text,
                    supported=supported,
                    target=target,
                    prediction=pred,
                    intent=int(intent.argmax()),
                    intent_confidence=float(intent.max()),
                    support_found=found,
                    attempts=r["attempts"],
                )
            )
        results[name] = dict(counts=dict(counts), records=records)
        print(json.dumps({"model": name, **counts}), flush=True)
    return results


@torch.no_grad()
def memory_probe(model, store, pairs):
    device = next(model.parameters()).device
    counts = Counter()
    for e, task in pairs:
        if not e["stored"]:
            continue
        j = TASKS.index(task)
        z, _ = store.fetch(e["owner"], [e["id"]])
        p = model.auxiliary[j](torch.tensor(z[:, model.slot_for_task[j]], device=device)).softmax(
            -1
        )
        correct = int(p.argmax()) == e["values"][task]
        counts["n"] += 1
        counts["correct"] += correct
        counts["confident_correct"] += correct and float(p.max()) > 0.8
        counts["confident_wrong"] += not correct and float(p.max()) > 0.8
    return dict(counts)


def main(memory, trained, out):
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(63)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, old_tok, _ = load_checkpoint(memory / "model.pt", device)
    model.eval()
    frontends = {}
    for name in (
        "random_narrow",
        "pretrained_narrow",
        "random_compositional",
        "pretrained_compositional",
    ):
        m, t, _ = load_classifier(trained / name, device)
        frontends[name] = (m, t)
    training_texts = set()
    for file in (trained / "data").glob("*.jsonl"):
        for row in load(file):
            training_texts.update((row["text"], row["positive"]))
    # Structural and lexical holdouts are evaluated only after all four models finish.
    report = dict(
        thresholds=dict(
            evidence=model.config.evidence_threshold, value=model.config.value_threshold
        ),
        writer_version=writer_version(model, old_tok.spec),
        language={},
        conditions={},
        negatives={},
        memory_probes={},
        old_tokenizer_unknowns={},
    )
    for family in (6, 7, 8, 9):
        rng = random.Random(6300 + family)
        records = []
        for i in range(1200):
            task = TASKS[i % 6]
            text = compose(task, anchor(rng), rng, family, True, rng.randrange(2))
            assert text not in training_texts
            records.append(dict(text=text, label=TASKS.index(task), family=family))
        print(f"LANGUAGE family{family}", flush=True)
        report["language"][f"family{family}"] = language_suite(model, old_tok, frontends, records)
    for view, seed in enumerate((90611, 90612)):
        data = out / f"data_{seed}"
        generate(
            data, seed=seed, train_histories=1, eval_histories=8, train_events=8, test_events=96
        )
        events = [
            {**e, "id": f"language{seed}-" + e["id"], "owner": f"language{seed}-" + e["owner"]}
            for e in load_events(data, "test")
        ]
        pairs = unique_pairs(events, seed + 2)
        rng = random.Random(seed + 10)
        changed_texts = [compose(task, e["anchor"], rng, 6 + view, True, view) for e, task in pairs]
        assert not set(changed_texts) & training_texts
        report["old_tokenizer_unknowns"][str(seed)] = sum(
            2 in old_tok.encode(t) for t in changed_texts
        )
        for changed in (False, True) if view == 0 else (True,):
            rows = [{**e, "text": new_event(e, view) if changed else e["text"]} for e in events]
            assert all(2 not in old_tok.encode(e["text"]) for e in rows)
            path = out / f"memory_{seed}_{int(changed)}.sqlite"
            build_store(path, model, old_tok, Corpus(rows, old_tok), device, max_members=32)
            store = open_store(path, model, old_tok)
            report["memory_probes"][f"{seed}_{int(changed)}"] = memory_probe(model, store, pairs)
            for changed_q in (False, True) if view == 0 else (True,):
                label = f"fresh{view}_" + (
                    "both"
                    if changed and changed_q
                    else "event_only"
                    if changed
                    else "question_only"
                    if changed_q
                    else "original"
                )
                texts = (
                    changed_texts
                    if changed_q
                    else [question(e, t, e["query_template"]) for e, t in pairs]
                )
                print(label, flush=True)
                report["conditions"][label] = pipeline(
                    model, old_tok, frontends, store, pairs, texts
                )
            if changed:
                # Negative controls compare the fixed main candidate and baseline.
                selected = {"pretrained_compositional": frontends["pretrained_compositional"]}
                missing = [(e, t) for e in events if not e["stored"] for t in TASKS]
                texts = [compose(t, e["anchor"], rng, 6 + view, True, view) for e, t in missing]
                print(f"NEGATIVE unobserved{view}", flush=True)
                report["negatives"][f"unobserved{view}"] = pipeline(
                    model, old_tok, selected, store, missing, texts
                )
                if view == 0:
                    present = [(e, t) for e, t in pairs if e["stored"]]
                    texts = [text for (e, t), text in zip(pairs, changed_texts) if e["stored"]]
                    for kind in ("removed", "zero"):
                        print("NEGATIVE " + kind, flush=True)
                        report["negatives"][kind] = pipeline(
                            model, old_tok, selected, store, present, texts, kind
                        )
            store.close()
    report["test_training_sentence_overlap"] = 0
    (out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--memory", type=Path, default=Path("experiments/episodic_memory/runs/typed_round_02")
    )
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.memory, a.trained, a.out)
