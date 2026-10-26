"""Post-test diagnosis only: frozen models, unchanged thresholds, token traces and controls."""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from .evaluate_qwen_memory import status
from .generate_data import ORDINALS, PROJECTS, TASKS
from .qwen_memory import QwenMemory, save_json
from .train_qwen_memory import answer
from .train_qwen_structured import target_tables


def annotate(model, result, expected):
    ids = [t["token_id"] for t in result["trace"] if t["token_id"] != model.tokenizer.eos_token_id]
    encoded = model.tokenizer(result["raw"], add_special_tokens=False, return_offsets_mapping=True)
    canonical = ids == encoded["input_ids"]
    start = result["raw"].find(expected)
    span = (start, start + len(expected)) if start >= 0 else None
    offsets = iter(encoded["offset_mapping"])
    for t in result["trace"]:
        if t["token_id"] == model.tokenizer.eos_token_id:
            t["role"] = "end"
        elif canonical:
            a, b = next(offsets)
            t["offset"] = [a, b]
            if span and a < span[1] and b > span[0]:
                t["role"] = "fact"
            elif all(c.isspace() or c in '，。！？：；、,.!?;:「」“”"' for c in result["raw"][a:b]):
                t["role"] = "punctuation"
            else:
                t["role"] = "phrasing" if span else "unmatched_content"
        else:
            t["role"] = "unaligned"
    low = [t for t in result["trace"] if t["probability"] <= 0.8]
    minimum = min(result["trace"], key=lambda t: t["probability"])
    fact = [t["probability"] for t in result["trace"] if t["role"] == "fact"]
    return dict(
        canonical_tokenization=canonical,
        min_token=minimum,
        low_roles=dict(Counter(t["role"] for t in low)),
        fact_min=min(fact) if fact else None,
        low_indices=[t["index"] for t in low],
    )


@torch.no_grad()
def forced_opening(model, q, z, prefix):
    """Oracle task supplied by experimenter; never a deployed answer policy."""
    memory = model.memory_embeddings(z[None], torch.tensor([True], device=model.device))
    after = model.tokenizer.encode(
        "\n问题：" + q + "<|im_end|>\n<|im_start|>assistant\n" + prefix, add_special_tokens=False
    )
    embedding = model.base.get_input_embeddings()
    x = torch.cat(
        [
            embedding(torch.tensor(model.before, device=model.device))[None],
            memory,
            embedding(torch.tensor(after, device=model.device))[None],
        ],
        1,
    )
    out = model.base.generate(
        inputs_embeds=x,
        attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=model.device),
        do_sample=False,
        max_new_tokens=40,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    ids = out.sequences[0, -len(out.scores) :]
    ended = bool(int(ids[-1]) == model.tokenizer.eos_token_id)
    ps = [float(s[0].float().softmax(-1)[t]) for s, t in zip(out.scores, ids)]
    return dict(
        forced_prefix=prefix,
        raw=prefix + model.tokenizer.decode(ids, skip_special_tokens=True).strip(),
        continuation_min=min(ps) if ended else 0.0,
        ended=ended,
        note="Excludes forced prefix probability and uses oracle field; not comparable to production confidence or success rate.",
    )


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    started = time.perf_counter()
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    before = hashlib.sha256((args.run / "adapter.pt").read_bytes()).hexdigest()
    assert before == meta["adapter_sha256"]
    model = QwenMemory(meta["backbone"], adapter_path=args.run / "adapter.pt")
    old = [
        json.loads(s)
        for s in (args.suite / args.name / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = {
        e["id"]: e for e in json.loads((args.suite / "events.json").read_text(encoding="utf-8"))
    }
    store = model.store(args.suite / args.name / "memories.sqlite")
    records = []
    for i, row in enumerate(old):
        e = events[row["id"]]
        selected_id = row["memory"]["id"]
        z, _ = store.fetch(e["owner"], [selected_id])
        z = torch.tensor(z[0], device=model.device)
        generated = model.generate(
            row["question"], z, row["memory"]["accepted"], details=True, trace=True
        )
        # An instrumentation regression would invalidate this diagnosis.
        assert generated["raw"] == row["memory"]["raw"], (i, "generation changed")
        assert abs(generated["confidence"] - row["memory"]["confidence"]) < 1e-6, (
            i,
            "confidence changed",
        )
        detail = annotate(model, generated, row["expected"])
        slot = TASKS.index(row["task"])
        distributions = [head(z[j]).softmax(-1) for j, head in enumerate(model.adapter.value_heads)]
        labels = [e["values"][t] for t in TASKS] + [
            PROJECTS.index(e["project"]),
            ORDINALS.index(e["ordinal"]),
        ]
        record = dict(
            index=i,
            id=row["id"],
            task=row["task"],
            wording=row["wording"],
            question=row["question"],
            expected=row["expected"],
            evidence=row["memory"]["evidence"],
            selected_id=selected_id,
            raw_status=row["memory"]["raw_status"],
            final_status=row["memory"]["final_status"],
            generated=generated,
            diagnosis=detail,
            slot_probe=dict(
                predicted=int(distributions[slot].argmax()),
                expected=labels[slot],
                expected_probability=float(distributions[slot][labels[slot]]),
                all_fields_correct=all(
                    int(p.argmax()) == target for p, target in zip(distributions, labels)
                ),
            ),
        )
        if row["wording"] == "novel":
            prefix = answer(e, row["task"]).split(row["expected"])[0]
            control = forced_opening(model, row["question"], z, prefix)
            control["status"] = status(control["raw"], e, row["task"])
            record["forced_opening"] = control
        records.append(record)
        with (out / "traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            print(
                json.dumps(
                    dict(
                        model=args.name,
                        completed=i + 1,
                        total=len(old),
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    # All controlled substitutions are limited to actual wrong novel generations.
    interventions = []
    tables = target_tables(model)
    for row in records:
        if row["wording"] != "novel" or row["raw_status"] == "correct":
            continue
        e = events[row["id"]]
        z, _ = store.fetch(e["owner"], [row["selected_id"]])
        z = torch.tensor(z[0], device=model.device)
        labels = [e["values"][t] for t in TASKS] + [
            PROJECTS.index(e["project"]),
            ORDINALS.index(e["ordinal"]),
        ]
        gold = torch.cat([table[labels[j]] for j, table in enumerate(tables)], 0)[None]
        original_soft = model.adapter.soft_tokens
        for mode in ("gold_target_slot", "gold_all_slots"):
            slot = TASKS.index(row["task"])

            def replacement(values, present, *, mode=mode, slot=slot, gold=gold):
                x = original_soft(values, present).clone()
                if mode == "gold_all_slots":
                    return gold.expand(len(values), -1, -1)
                x[:, slot * 4 : (slot + 1) * 4] = gold[:, slot * 4 : (slot + 1) * 4]
                return x

            model.adapter.soft_tokens = replacement
            try:
                generated = model.generate(row["question"], z, True, details=True, trace=True)
            finally:
                model.adapter.soft_tokens = original_soft
            interventions.append(
                dict(
                    index=row["index"],
                    mode=mode,
                    question=row["question"],
                    expected=row["expected"],
                    status=status(generated["raw"], e, row["task"]),
                    generated=generated,
                    note="Oracle ground-truth embeddings substituted; diagnosis only, never used by inference.",
                )
            )
    store.close()
    summary = dict(
        model=args.name,
        adapter_sha256=before,
        unchanged_checkpoint=before
        == hashlib.sha256((args.run / "adapter.pt").read_bytes()).hexdigest(),
        replayed_exactly=len(records),
        wording={},
        seconds=time.perf_counter() - started,
    )
    for wording in ("original", "novel"):
        subset = [r for r in records if r["wording"] == wording]
        rejected = [r for r in subset if r["final_status"] == "unknown"]
        correct_rejected = [r for r in rejected if r["raw_status"] == "correct"]
        summary["wording"][wording] = dict(
            count=len(subset),
            correct_rejected=len(correct_rejected),
            wrong_rejected=sum(r["raw_status"] == "wrong" for r in rejected),
            min_role_all=dict(Counter(r["diagnosis"]["min_token"]["role"] for r in subset)),
            min_role_correct_rejected=dict(
                Counter(r["diagnosis"]["min_token"]["role"] for r in correct_rejected)
            ),
            only_nonfact_causes=sum(
                r["diagnosis"]["canonical_tokenization"]
                and r["diagnosis"]["fact_min"] is not None
                and r["diagnosis"]["fact_min"] > 0.8
                for r in correct_rejected
            ),
            fact_below_threshold=sum(
                r["diagnosis"]["fact_min"] is not None and r["diagnosis"]["fact_min"] <= 0.8
                for r in correct_rejected
            ),
            all_field_probes_correct=sum(r["slot_probe"]["all_fields_correct"] for r in subset),
            task_slot_correct=sum(
                r["slot_probe"]["predicted"] == r["slot_probe"]["expected"] for r in subset
            ),
            median_fact_probability=statistics.median(
                [
                    r["diagnosis"]["fact_min"]
                    for r in subset
                    if r["diagnosis"]["fact_min"] is not None
                ]
            ),
            ended=sum(r["generated"]["ended"] for r in subset),
            forced_prefix_status=dict(
                Counter(r["forced_opening"]["status"] for r in subset if "forced_opening" in r)
            ),
            correct_rejected_forced_continuation_pass=sum(
                r.get("forced_opening", {}).get("status") == "correct"
                and r["forced_opening"]["continuation_min"] > 0.8
                for r in correct_rejected
            ),
        )
    save_json(out / "summary.json", summary)
    save_json(out / "interventions.json", interventions)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--suite",
        type=Path,
        default=Path("experiments/episodic_memory/runs/qwen_scale_comparison_01"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/episodic_memory/runs/qwen_confidence_diagnosis_01"),
    )
    main(p.parse_args())
