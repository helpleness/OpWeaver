"""Paired, fixed intervention evaluation. Labels used for scoring only."""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from .generate_data import FIELDS, TASKS, load_events, question, render
from .qwen_memory import UNKNOWN, QwenMemory, save_json


def status(text, event, task):
    if text == UNKNOWN:
        return "unknown"
    # Strict unique-field-value matching, not lenient any-value containment.
    values = [v for v in FIELDS[task] if v in text]
    return "correct" if values == [FIELDS[task][event["values"][task]]] else "wrong"


def score(result, e, task):
    return dict(
        **result,
        raw_status=status(result["raw"], e, task),
        final_status=status(result["answer"], e, task),
    )


@torch.no_grad()
def main(args):
    started = time.perf_counter()
    out = args.run / "evaluation"
    out.mkdir(exist_ok=args.resume)
    metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    assert (
        hashlib.sha256((args.run / "adapter.pt").read_bytes()).hexdigest()
        == metadata["adapter_sha256"]
    )
    model = QwenMemory(metadata["backbone"], adapter_path=args.run / "adapter.pt")
    events = load_events(args.run / "data", "test")
    stored = [e for e in events if e["stored"]]
    missing = [e for e in events if not e["stored"]]
    rng = random.Random(7301)
    selected = [(e, task) for task in TASKS for e in rng.sample(stored, 10)]
    # Selection fixed before generating any output. No adapting to individual scores.
    manifest = [
        dict(id=e["id"], task=t, question=question(e, t, e["query_template"])) for e, t in selected
    ]
    if args.resume:
        assert json.loads((out / "selection.json").read_text(encoding="utf-8")) == manifest
    else:
        save_json(out / "selection.json", manifest)
    fake = copy.deepcopy(stored)
    cf_rng = random.Random(7302)
    for e in fake:
        for task in TASKS:
            original = e["values"][task]
            e["values"][task] = cf_rng.choice(
                [i for i in range(len(FIELDS[task])) if i != original]
            )
        e["text"] = render(e, 0)
    cf_by_id = {e["id"]: e for e in fake}
    # Counterfactual texts exist in evaluation provenance, not the inference database.
    save_json(out / "counterfactual_provenance.json", fake)
    normal = model.store(out / "memories.sqlite", create=not args.resume)
    counterfactual = model.store(out / "counterfactual.sqlite", create=not args.resume)
    if not args.resume:
        model.write_events(normal, stored)
        model.write_events(counterfactual, fake)
    for owner in {e["owner"] for e in stored}:
        normal._index(owner)
    counts = dict(normal=normal.counts(), counterfactual=counterfactual.counts())
    counts["normal"]["splits"] = (
        sum(c.splits for c in normal.cache.values()) if not args.resume else None
    )
    counts["normal"]["merges"] = (
        sum(c.merges for c in normal.cache.values()) if not args.resume else None
    )
    counts["normal"]["max_group_members"] = max(
        len(g) for c in normal.cache.values() for g in c.groups.values()
    )
    # Reopen to guarantee all scored reads use the persisted representation/index.
    normal.close()
    counterfactual.close()
    normal = model.store(out / "memories.sqlite")
    counterfactual = model.store(out / "counterfactual.sqlite")
    rows = []
    if args.resume:
        rows = [
            json.loads(line)
            for line in (out / "paired_results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == len(selected), "Resume expects all 60 main results already written"
        assert all(
            row["id"] == item["id"]
            and row["task"] == item["task"]
            and row["question"] == item["question"]
            for row, item in zip(rows, manifest)
        )
    for n, (e, task) in enumerate(selected):
        if args.resume:
            continue
        q = question(e, task, e["query_template"])
        actual = model.ask(normal, e["owner"], q)
        empty = model.generate(q, present=False, details=True)
        empty["answer"] = (
            UNKNOWN  # Report raw separately: empty-store guard is not learned refusal.
        )
        cf = model.ask(counterfactual, e["owner"], q)
        z, _ = normal.fetch(e["owner"], [e["id"]])
        oracle = model.generate(q, torch.tensor(z[0], device=model.device), True, details=True)
        # Wrong memory means replacing address AND content by a different real event.
        candidates = [
            x
            for x in stored
            if x["owner"] == e["owner"]
            and x["project"] != e["project"]
            and x["ordinal"] != e["ordinal"]
        ]
        wrong_event = candidates[0]
        wz, wk = normal.fetch(e["owner"], [wrong_event["id"]])
        h, mask = model.padded_features(model.features([q]))
        qkey = model.adapter.key(model.adapter.query_logits(h, mask))
        agreement = float(model.adapter.agreement(qkey, torch.tensor(wk, device=model.device))[0])
        wrong = (
            model.generate(q, torch.tensor(wz[0], device=model.device), True, details=True)
            if agreement > 0.99
            else dict(empty)
        )
        wrong.update(evidence=agreement, accepted=agreement > 0.99, injected_id=wrong_event["id"])
        if agreement <= 0.99:
            wrong["answer"] = UNKNOWN
        row = dict(
            id=e["id"],
            owner=e["owner"],
            task=task,
            question=q,
            expected=FIELDS[task][e["values"][task]],
            counterfactual_expected=FIELDS[task][cf_by_id[e["id"]]["values"][task]],
            normal=score(actual, e, task),
            no_memory=score(empty, e, task),
            counterfactual=score(cf, cf_by_id[e["id"]], task),
            unrelated=score(wrong, e, task),
            oracle=score(oracle, e, task),
        )
        row["counterfactual"]["original_truth_status"] = status(cf["answer"], e, task)
        rows.append(row)
        with (out / "paired_results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (n + 1) % 10 == 0:
            print(
                json.dumps(
                    dict(
                        stage="paired",
                        complete=n + 1,
                        total=len(selected),
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    missing_rows = []
    for i, e in enumerate(rng.sample(missing, 12)):
        task = TASKS[i % 6]
        q = question(e, task, e["query_template"])
        result = model.ask(normal, e["owner"], q)
        missing_rows.append(dict(expected_id=e["id"], task=task, question=q, **result))
    save_json(out / "missing_results.json", missing_rows)
    # Fixed held-out question and event rendering, 2 questions per field on paired test events.
    novel = copy.deepcopy(stored)
    for e in novel:
        e["text"] = render(e, 3)
    novel_store = model.store(out / "novel_events.sqlite", create=True)
    model.write_events(novel_store, novel)
    novel_rows = []
    for e, task in [item for t in TASKS for item in [x for x in selected if x[1] == t][:2]]:
        q = question(e, task, 3)
        canonical_q = question(e, task, e["query_template"])
        novel_rows.append(
            dict(
                id=e["id"],
                task=task,
                question=q,
                expected=FIELDS[task][e["values"][task]],
                question_only=score(model.ask(normal, e["owner"], q), e, task),
                event_only=score(model.ask(novel_store, e["owner"], canonical_q), e, task),
                both=score(model.ask(novel_store, e["owner"], q), e, task),
            )
        )
    save_json(out / "novel_results.json", novel_rows)
    summary = dict(
        selection_seed=7301,
        count=len(rows),
        conditions={},
        novel={},
        disk=counts,
        missing_count=len(missing_rows),
        missing_final_refused=sum(r["answer"] == UNKNOWN for r in missing_rows),
        missing_raw_refused=sum(r["raw"] == UNKNOWN for r in missing_rows),
        retrieval_top1_correct=sum(r["normal"]["id"] == r["id"] for r in rows),
        retrieval_accepted_correct=sum(
            r["normal"]["id"] == r["id"] and r["normal"]["accepted"] for r in rows
        ),
        retrieval_accepted_wrong=sum(
            r["normal"]["id"] != r["id"] and r["normal"]["accepted"] for r in rows
        ),
        backbone_frozen=metadata["backbone_frozen"],
        chat_exact=metadata["chat_exact"],
        chat_count=metadata["chat_count"],
        thresholds=metadata["thresholds"],
    )
    for name in ("normal", "no_memory", "counterfactual", "unrelated", "oracle"):
        summary["conditions"][name] = {
            metric: dict(Counter(r[name][metric] for r in rows))
            for metric in ("raw_status", "final_status")
        }
    for name in ("question_only", "event_only", "both"):
        summary["novel"][name] = {
            metric: dict(Counter(r[name][metric] for r in novel_rows))
            for metric in ("raw_status", "final_status")
        }
    summary["paired_correct_then_counterfactual_correct"] = sum(
        r["normal"]["final_status"] == "correct"
        and r["counterfactual"]["final_status"] == "correct"
        for r in rows
    )
    summary["normal_refusal_diagnosis"] = dict(
        address_rejected=sum(not r["normal"]["accepted"] for r in rows),
        accepted_but_generated_refusal=sum(
            r["normal"]["accepted"] and r["normal"]["raw"] == UNKNOWN for r in rows
        ),
        accepted_but_probability_rejected=sum(
            r["normal"]["accepted"]
            and r["normal"]["raw"] != UNKNOWN
            and r["normal"]["answer"] == UNKNOWN
            for r in rows
        ),
    )
    summary["database_columns"] = [r[1] for r in normal.db.execute("PRAGMA table_info(events)")]
    summary["by_field"] = {
        task: dict(Counter(r["normal"]["final_status"] for r in rows if r["task"] == task))
        for task in TASKS
    }
    summary["correct_raw_rejected"] = sum(
        r["normal"]["raw_status"] == "correct" and r["normal"]["final_status"] == "unknown"
        for r in rows
    )
    for store in (normal, counterfactual, novel_store):
        store.close()
    # Free GPU before an independent process proves restart persistence.
    del model
    torch.cuda.empty_cache()
    proofs = []
    chosen = next((r for r in rows if r["normal"]["final_status"] == "correct"), rows[0])
    for dbname, condition in (
        ("memories.sqlite", "normal"),
        ("counterfactual.sqlite", "counterfactual"),
    ):
        command = [
            sys.executable,
            "-m",
            "experiments.episodic_memory.qwen_demo",
            "--run",
            str(args.run),
            "--db",
            str(out / dbname),
            "--owner",
            chosen["owner"],
            "ask",
            "--question",
            chosen["question"],
        ]
        proc = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        proofs.append(
            dict(
                database=dbname,
                question=chosen["question"],
                owner=chosen["owner"],
                result=result,
                identical_to_parent=all(result[k] == chosen[condition][k] for k in result),
            )
        )
    summary["restart_proofs"] = proofs
    summary["elapsed_s"] = time.perf_counter() - started
    summary["adapter_sha256"] = hashlib.sha256((args.run / "adapter.pt").read_bytes()).hexdigest()
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume ancillary checks after all 60 paired results were saved",
    )
    main(p.parse_args())
