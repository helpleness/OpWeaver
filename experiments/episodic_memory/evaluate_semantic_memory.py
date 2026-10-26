"""Fixed abstention matrix, paired baseline, calibration and held-out evaluation."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .generate_data import FIELDS, TASKS
from .qwen_memory import UNKNOWN, QwenMemory, save_json
from .semantic_data import INVALID
from .semantic_memory import SemanticMemory, parse_answer


def outcome(answer, expected, field):
    """LEGACY protocol score, not semantic correctness or hallucination detection.

    Retained for historical reproduction. See EVALUATION_CORRECTION.md and
    semantic_review.py; do not use this metric to claim general refusal ability.
    """
    if answer == UNKNOWN:
        return "refused"
    if expected is None:
        return "wrong"
    matches = [v for v in FIELDS[TASKS[field]] if v in answer]
    return "correct" if matches == [expected] else "wrong"


def summarize(rows):
    return {
        kind: dict(
            count=len(subset),
            outcomes=dict(Counter(r["outcome"] for r in subset)),
            reasons=dict(Counter(r["result"].get("reason", "legacy_gate") for r in subset)),
            intent_correct=sum(r.get("intent_correct", False) for r in subset),
        )
        for kind in sorted({r["kind"] for r in rows})
        for subset in [[r for r in rows if r["kind"] == kind]]
    }


@torch.no_grad()
def main(args):
    if not getattr(args, "legacy_protocol", False):
        raise ValueError(
            "Legacy scoring requires --legacy-protocol. Use semantic_review for evidence-bound review."
        )
    torch.set_num_threads(4)
    started = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    if args.baseline:
        model = QwenMemory(meta["backbone"], adapter_path=args.run / "adapter.pt")
    else:
        model = SemanticMemory.load(args.run)
    threshold = 0.8
    if args.split == "test" and not args.baseline:
        calibration = json.loads(
            (args.run / "calibration/calibration.json").read_text(encoding="utf-8")
        )
        assert calibration["adapter_sha256"] == meta["adapter_sha256"]
        assert calibration["semantic_sha256"] == meta["semantic_sha256"]
        threshold = calibration["threshold"]
    cases = json.loads((args.suite / f"{args.split}.json").read_text(encoding="utf-8"))
    all_events = [e for case in cases for e in case["events"]]
    store = model.store(out / "memories.sqlite", create=True)
    model.write_events(store, all_events)
    for case in cases:
        if case["damage_id"]:
            # Deliberate fault injection: valid address with zeroed continuous payload.
            with store.db:
                store.db.execute(
                    "UPDATE events SET memory=? WHERE event_id=?",
                    (np.zeros((8, 256), dtype="<f2").tobytes(), case["damage_id"]),
                )
    store.close()
    store = model.store(out / "memories.sqlite")
    rows = []
    for i, case in enumerate(cases):
        if args.baseline:
            result = model.ask(store, case["owner"], case["question"])
        else:
            result = model.ask_checked(store, case["owner"], case["question"], threshold=threshold)
        row = dict(
            id=case["id"],
            kind=case["kind"],
            question=case["question"],
            field=case["field"],
            expected=case["expected"],
            result=result,
            outcome=outcome(result["answer"], case["expected"], case["field"]),
            grammar_valid=parse_answer(result["answer"], case["field"]) is not None,
        )
        if not args.baseline:
            target = 6 if case["kind"] in INVALID[args.split] else case["field"]
            row["intent_correct"] = target == result["field"]
        rows.append(row)
        with (out / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 30 == 0:
            print(
                json.dumps(
                    dict(
                        stage=args.split,
                        baseline=args.baseline,
                        done=i + 1,
                        total=len(cases),
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    # The calibration set alone chooses the new score threshold. Raw results retain 0.8 trials.
    if args.split == "calibration" and not args.baseline:
        choices = []
        for t in (0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0):
            accepted = [r for r in rows if r["result"]["eligible"] and r["result"]["support"] > t]
            choices.append(
                dict(
                    threshold=t,
                    accepted=len(accepted),
                    correct=sum(r["outcome"] == "correct" for r in accepted),
                    wrong=sum(r["outcome"] == "wrong" for r in accepted),
                )
            )
        threshold = next(c["threshold"] for c in choices if c["wrong"] == 0)
        save_json(
            out / "calibration.json",
            dict(
                threshold=threshold,
                grid=choices,
                adapter_sha256=meta["adapter_sha256"],
                semantic_sha256=meta["semantic_sha256"],
                cases_sha256=hashlib.sha256(
                    (args.suite / "calibration.json").read_bytes()
                ).hexdigest(),
                note="Lowest allowed threshold with zero observed calibration false accepts; no finite sample guarantees future risk.",
            ),
        )
        for row in rows:
            if row["result"]["support"] <= threshold:
                row["outcome"] = "refused"
        save_json(out / "calibrated_outcomes.json", rows)
    probe = Counter()
    if not args.baseline:
        for e in all_events:
            if any(c["damage_id"] == e["id"] for c in cases):
                continue
            z, _ = store.fetch(e["owner"], [e["id"]])
            z = torch.tensor(z[0], device=model.device)
            for j, t in enumerate(TASKS):
                predicted = int(model.semantic.values[j](z[j]).argmax())
                target = e["values"][t] if e["values"][t] >= 0 else len(FIELDS[t])
                kind = "missing" if e["values"][t] < 0 else "known"
                probe[kind + "_total"] += 1
                probe[kind + "_correct"] += predicted == target
    store.close()
    summary = dict(
        metric_status="legacy_protocol_only_not_semantic_accuracy",
        run=str(args.run.resolve()),
        split=args.split,
        baseline=args.baseline,
        threshold=threshold,
        adapter_sha256=meta["adapter_sha256"],
        cases_sha256=hashlib.sha256((args.suite / f"{args.split}.json").read_bytes()).hexdigest(),
        conditions=summarize(rows),
        total=dict(Counter(r["outcome"] for r in rows)),
        probes=dict(probe),
        positive=dict(Counter(r["outcome"] for r in rows if r["expected"] is not None)),
        negative=dict(Counter(r["outcome"] for r in rows if r["expected"] is None)),
        seconds=time.perf_counter() - started,
    )
    save_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--suite", type=Path, default=Path("experiments/episodic_memory/runs/semantic_suite_01")
    )
    p.add_argument("--split", choices=["calibration", "test"], required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument(
        "--legacy-protocol", action="store_true", help="Explicitly reproduce deprecated scoring"
    )
    main(p.parse_args())
