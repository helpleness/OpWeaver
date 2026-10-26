"""Sealed new-fact/new-language evaluation, plus a clearly labelled old regression."""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch

from .diagnose_abstention import paired_queries
from .evidence_policy import infer
from .generate_data import OFFSETS, TASKS, generate, load_events, question
from .models import pad_tokens
from .residual_language import fresh_event, fresh_question
from .store import writer_version
from .train import Corpus, build_store, load_checkpoint, seed_everything
from .train_evidence import open_store


def unique_pairs(events, seed):
    rng = random.Random(seed)
    pairs = []
    for task in TASKS:
        pairs.extend((e, task) for e in rng.sample([e for e in events if e["stored"]], 80))
        pairs.extend((e, task) for e in rng.sample([e for e in events if not e["stored"]], 20))
    rng.shuffle(pairs)
    assert len({(e["id"], t) for e, t in pairs}) == 600
    return pairs


@torch.no_grad()
def suite(models, tok, store, pairs, query_fn, negative=None):
    results = {}
    for name, model in models.items():
        device = next(model.parameters()).device
        states, texts = [], [query_fn(e, t) for e, t in pairs]
        encoded = [tok.encode(s) for s in texts]
        assert all(2 not in row for row in encoded), (
            "Unseen vocabulary is a separate test; do not silently use UNK"
        )
        for start in range(0, len(pairs), 64):
            h, q = model.encode_questions(pad_tokens(encoded[start : start + 64], device))
            states.extend(zip(h.unbind(), q.unbind()))
        counts = Counter(
            dict(
                present=0,
                correct=0,
                unknown=0,
                wrong=0,
                unsupported=0,
                false_answer=0,
                support_found=0,
                retries=0,
            )
        )
        records = []
        for (e, t), (h, q), text in zip(pairs, states, texts):
            r = infer(
                model,
                store,
                e["owner"],
                h[None],
                q[None],
                exclude=[e["id"]] if negative == "removed" else [],
                zero=negative == "zero",
            )
            present = e["stored"] and negative is None
            target = OFFSETS[t] + e["values"][t] if present else 0
            pred = r["prediction"]
            found = any(e["id"] in a["ids"] for a in r["attempts"])
            if present:
                counts["present"] += 1
                counts["correct"] += pred == target
                counts["unknown"] += pred == 0
                counts["wrong"] += pred not in (0, target)
                counts["support_found"] += found
            else:
                counts["unsupported"] += 1
                counts["false_answer"] += pred != 0
            counts["retries"] += len(r["attempts"]) > 1
            records.append(
                dict(
                    id=e["id"],
                    task=t,
                    text=text,
                    supported=present,
                    target=target,
                    prediction=pred,
                    support_found=found,
                    attempts=r["attempts"],
                )
            )
        results[name] = dict(
            counts=counts, accuracy=counts["correct"] / max(1, counts["present"]), records=records
        )
        print(json.dumps({"model": name, **counts}), flush=True)
    return results


def main(base, candidate, out, seeds=(90381, 90382), view_offset=0, development=None):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    seed_everything(45)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old, tok, _ = load_checkpoint(base / "model.pt", device)
    new, new_tok, _ = load_checkpoint(candidate / "model.pt", device)
    assert tok.spec == new_tok.spec
    assert writer_version(old, tok.spec) == writer_version(new, tok.spec)
    assert new.config.evidence_threshold >= old.config.evidence_threshold
    assert new.config.value_threshold >= old.config.value_threshold
    models = {"previous": old.eval(), "typed": new.eval()}
    report = dict(
        base=str(base),
        candidate=str(candidate),
        conditions={},
        negatives={},
        checkpoint_sha256=hashlib.sha256((candidate / "model.pt").read_bytes()).hexdigest(),
        thresholds={
            n: dict(evidence=m.config.evidence_threshold, value=m.config.value_threshold)
            for n, m in models.items()
        },
        seeds=list(seeds),
        view_offset=view_offset,
    )
    for group, seed in enumerate(seeds):
        view = group + view_offset
        data = out / f"data_{seed}"
        generate(
            data, seed=seed, train_histories=1, eval_histories=8, train_events=8, test_events=96
        )
        events = [
            {**e, "owner": f"fresh{seed}-" + e["owner"], "id": f"fresh{seed}-" + e["id"]}
            for e in load_events(data, "test")
        ]
        pairs = unique_pairs(events, seed + 2)
        negative_pairs = [(e, t) for e in events if not e["stored"] for t in TASKS]
        for changed in (False, True) if group == 0 else (True,):
            rows = [{**e, "text": fresh_event(e, view) if changed else e["text"]} for e in events]
            assert all(2 not in tok.encode(e["text"]) for e in rows)
            path = out / f"memory_{seed}_{int(changed)}.sqlite"
            build_store(path, old, tok, Corpus(rows, tok), device, max_members=32)
            store = open_store(path, old, tok)
            for changed_q in (False, True) if group == 0 else (True,):
                label = f"fresh{group}_" + (
                    "both"
                    if changed and changed_q
                    else "event_only"
                    if changed
                    else "question_only"
                    if changed_q
                    else "original"
                )
                query_fn = (
                    (lambda e, t: fresh_question(e, t, view))
                    if changed_q
                    else (lambda e, t: question(e, t, e["query_template"]))
                )
                print(label, flush=True)
                report["conditions"][label] = suite(models, tok, store, pairs, query_fn)
            if changed:

                def query_fn(e, t):
                    return fresh_question(e, t, view)

                print(f"NEGATIVE fresh{group}", flush=True)
                report["negatives"][f"unobserved{group}"] = suite(
                    models, tok, store, negative_pairs, query_fn
                )
                if group == 0:
                    for kind in ("removed", "zero"):
                        print("NEGATIVE " + kind, flush=True)
                        report["negatives"][kind] = suite(
                            models,
                            tok,
                            store,
                            [(e, t) for e, t in pairs if e["stored"]],
                            query_fn,
                            kind,
                        )
                    if not new.config.grounded_context:
                        typed_only = copy.deepcopy(new)
                        typed_only.config.refined_context = False
                        typed_only.verifier.load_state_dict(old.verifier.state_dict())
                        context_only = copy.deepcopy(new)
                        context_only.config.typed_reader = False
                        context_only.intent_head.load_state_dict(old.intent_head.state_dict())
                        print("ABLATIONS fresh0_both", flush=True)
                        report["ablations"] = suite(
                            {"typed_only": typed_only, "context_only": context_only},
                            tok,
                            store,
                            pairs,
                            query_fn,
                        )
            store.close()
    source = Path(__file__).parent / "runs/paraphrase_structured"
    store = open_store(
        Path(__file__).parent / "runs/paraphrase_structured_fresh/changed_events.sqlite", old, tok
    )
    print("OLD REGRESSION (development data)", flush=True)
    report["regression"] = suite(
        models,
        tok,
        store,
        paired_queries(load_events(source / "data", "test")),
        lambda e, t: question(e, t, 3),
    )
    store.close()
    if development is not None:
        events = [
            {**e, "owner": "fresh90381-" + e["owner"], "id": "fresh90381-" + e["id"]}
            for e in load_events(development / "data_90381", "test")
        ]
        pairs = unique_pairs(events, 90383)
        store = open_store(development / "memory_90381_1.sqlite", old, tok)
        print("FIRST STAGE DEVELOPMENT REGRESSION", flush=True)
        report["stage1_regression"] = {
            "both": suite(models, tok, store, pairs, lambda e, t: fresh_question(e, t, 0)),
            "removed": suite(
                models,
                tok,
                store,
                [(e, t) for e, t in pairs if e["stored"]],
                lambda e, t: fresh_question(e, t, 0),
                "removed",
            ),
        }
        store.close()
    (out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        type=Path,
        default=Path("experiments/episodic_memory/runs/evidence_round_03/verified"),
    )
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs=2, default=[90381, 90382])
    p.add_argument("--view-offset", type=int, default=0, choices=(0, 2))
    p.add_argument("--development-comparison", type=Path)
    a = p.parse_args()
    main(a.base, a.candidate, a.out, a.seeds, a.view_offset, a.development_comparison)
