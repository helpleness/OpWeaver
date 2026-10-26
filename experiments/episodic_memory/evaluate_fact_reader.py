"""Small paired, evidence-audited experiment. No automatic language correctness score."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import torch

from .generate_data import FIELDS, TASKS
from .semantic_data import event
from .semantic_memory import SemanticMemory


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def narrative(e):
    styles = [
        "交谈对象记为{}",
        "发生日记为{}",
        "发生的事情记为{}",
        "那时我的心情是{}",
        "这份心情的缘由记为{}",
        "注意力主要落在{}",
    ]
    return (
        e["anchor"]
        + "。"
        + "。".join(
            styles[i].format(FIELDS[t][e["values"][t]])
            for i, t in enumerate(TASKS)
            if e["values"][t] >= 0
        )
        + "。"
    )


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(9211)
    questions = [
        "那次和我交谈的对象，名字是什么？",
        "这段经历记的是第几天？",
        "那次究竟发生了哪一件事情？",
        "当时的心情，记忆里怎么描述？",
        "那份感受的原因是什么？",
        "我的注意力主要落在哪个方面？",
    ]
    cases = []
    specs = [("known", f) for f in range(6)] + [
        ("missing", 0),
        ("missing", 4),
        ("conflict", 3),
        ("empty", 5),
        ("wrong_owner", 2),
        ("multi", 0),
    ]
    for i, (kind, field) in enumerate(specs):
        cid = f"fresh-{i:02d}"
        target = event(rng, cid + "-target", cid)
        distractor = event(rng, cid + "-distractor", cid, ((i + 7) % 16, (i + 3) % 16))
        if distractor["anchor"] == target["anchor"]:
            raise ValueError("Address collision")
        fields = [TASKS[field]] if kind != "multi" else ["person", "focus"]
        q = (
            target["anchor"]
            + "，"
            + (questions[field] if kind != "multi" else "请分别说出交谈对象和我关注的方面。")
        )
        records = [distractor, target]
        availability = "answerable"
        if kind == "missing":
            target["values"][TASKS[field]] = -1
            availability = "missing_field"
        elif kind == "conflict":
            other = copy.deepcopy(target)
            other["id"] += "-conflict"
            other["values"][TASKS[field]] = (other["values"][TASKS[field]] + 1) % len(
                FIELDS[TASKS[field]]
            )
            records.append(other)
            availability = "conflicting"
        elif kind == "empty":
            records = []
            availability = "no_evidence"
        elif kind == "wrong_owner":
            target["owner"] = cid + "-other"
            availability = "no_matching_event"
        for e in records:
            e["text"] = narrative(e)
        available = [e for e in records if e["owner"] == cid]
        cases.append(
            dict(
                id=cid,
                kind=kind,
                owner=cid,
                question=q,
                fields=fields,
                availability=availability,
                events=records,
                references={
                    t: sorted(
                        {
                            FIELDS[t][e["values"][t]]
                            for e in available
                            if e["anchor"] == target["anchor"] and e["values"][t] >= 0
                        }
                    )
                    for t in fields
                },
            )
        )
    save(out / "cases.json", cases)
    save(
        out / "protocol.json",
        dict(
            seed=9211,
            train_steps=200,
            alignment_steps=400,
            threshold=0.8,
            address_threshold=0.99,
            tune_on_test=False,
            cases_sha256=hashlib.sha256((out / "cases.json").read_bytes()).hexdigest(),
            evaluation="Review every final response against condition-specific evidence; no phrase scorer",
            conditions=["old_reader", "fact_only_reader", "base_no_context", "base_full_text"],
            limitation="Small synthetic functional test; not a generalization or safety certification",
        ),
    )


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    cases = json.loads((args.out / "cases.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.out / "protocol.json").read_text(encoding="utf-8"))
    assert (
        hashlib.sha256((args.out / "cases.json").read_bytes()).hexdigest()
        == protocol["cases_sha256"]
    )
    result_path = args.out / "outputs.jsonl"
    with result_path.open("x", encoding="utf-8") as output:
        for mode, path in [("old_reader", args.old), ("fact_only_reader", args.run)]:
            model = SemanticMemory.load(path)
            store = model.store(args.out / f"{mode}.sqlite", create=True)
            model.write_events(store, [e for c in cases for e in c["events"]])
            for c in cases:
                result = model.respond(
                    store, c["owner"], c["question"], protocol["threshold"], natural_status=True
                )
                row = dict(id=c["id"], mode=mode, question=c["question"], result=result)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps(dict(stage="evaluate", mode=mode, case=c["id"])), flush=True)
            store.close()
            if mode == "fact_only_reader":
                for mode2 in ("base_no_context", "base_full_text"):
                    for c in cases:
                        records = [e["text"] for e in c["events"] if e["owner"] == c["owner"]]
                        context = "\n".join(records) if mode2 == "base_full_text" else ""
                        prompt = (
                            "请根据下面提供的经历记录回答问题。没有足够信息时解释不确定性，"
                            "矛盾时说明冲突，不要猜测。可以自然表述，简短作答。\n"
                            "经历记录：\n"
                            + (context or "（没有提供经历记录）")
                            + "\n问题："
                            + c["question"]
                        )
                        # Explicit chat bypass uses the frozen official backbone only.
                        generated = model.chat(prompt, max_new_tokens=100, details=True)
                        row = dict(
                            id=c["id"],
                            mode=mode2,
                            question=c["question"],
                            prompt=prompt,
                            result=dict(
                                **generated,
                                response_source="bare_backbone",
                                semantic_correctness="not_automatically_judged",
                            ),
                        )
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        print(
                            json.dumps(dict(stage="evaluate", mode=mode2, case=c["id"])), flush=True
                        )
            del model
            torch.cuda.empty_cache()
    save(
        args.out / "output_manifest.json",
        dict(
            outputs_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
            cases_sha256=protocol["cases_sha256"],
            model_metadata={
                name: json.loads((path / "metadata.json").read_text(encoding="utf-8"))
                for name, path in (("old_reader", args.old), ("fact_only_reader", args.run))
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--old", type=Path, default=Path("experiments/episodic_memory/runs/semantic_memory_03")
    )
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/fact_reader_01")
    )
    a = p.parse_args()
    prepare(a.out) if a.prepare else run(a)
