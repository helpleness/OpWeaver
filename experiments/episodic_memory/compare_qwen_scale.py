"""Fixed new-event factorial: backbone size x input representation, no training here."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .evaluate_qwen_memory import score
from .generate_data import FIELDS, TASKS, generate, load_events, question
from .qwen_memory import MEMORY_SYSTEM, UNKNOWN, QwenMemory, save_json

NEW_QUESTIONS = {
    "person": ["当时和我进行这次交流的那个人叫什么？", "请只告诉我这次谈话的另一位参与者。"],
    "day": [
        "按经历里记录的天数，这件事是在第几天发生的？",
        "我想核对这次经历的发生日期，记录的是哪一天？",
    ],
    "event": [
        "这次经历里具体出了什么事？请说出发生的事情。",
        "把情绪和理由先放在一边，这次记录的事情本身是什么？",
    ],
    "emotion": ["事情发生时，我处于什么情绪状态？", "我问的是当时的心情：这次经历让我有什么感受？"],
    "reason": [
        "记录里是用什么理由解释当时那种感受的？",
        "我为什么会有那种心情？请回答记录中的原因。",
    ],
    "focus": ["那次我的注意力主要集中在什么方面？", "这段经历里，我最在意的具体方面是什么？"],
}


def prepare(suite):
    suite.mkdir(parents=True, exist_ok=False)
    generate(
        suite / "data",
        seed=8101,
        train_histories=0,
        eval_histories=4,
        train_events=16,
        test_events=32,
    )
    events = load_events(suite / "data", "test")
    for e in events:
        e["id"] = "scale-" + e["id"]
        e["owner"] = "scale-" + e["owner"]
    save_json(suite / "events.json", events)
    rng = random.Random(8102)
    stored = [e for e in events if e["stored"]]
    selected = [
        dict(
            id=e["id"],
            task=task,
            original=question(e, task, e["query_template"]),
            novel=e["anchor"] + "，" + NEW_QUESTIONS[task][j % 2],
        )
        for task in TASKS
        for j, e in enumerate(rng.sample(stored, 10))
    ]
    save_json(suite / "selection.json", selected)
    save_json(
        suite / "protocol.json",
        dict(
            event_seed=8101,
            selection_seed=8102,
            count=60,
            questions=NEW_QUESTIONS,
            sha256=hashlib.sha256((suite / "selection.json").read_bytes()).hexdigest(),
            thresholds=dict(evidence=0.99, min_token=0.8),
            note="Created before new backbone training results; no test-based selection. Text inputs are oracle diagnostic only.",
        ),
    )
    print(str(suite.resolve()), flush=True)


@torch.no_grad()
def text_read(model, text, q):
    inputs = model.tokenizer.apply_chat_template(
        [
            dict(role="system", content=MEMORY_SYSTEM),
            dict(role="user", content="内部记忆：" + text + "\n问题：" + q),
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output = model.base.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=40,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    ids = output.sequences[0, inputs.input_ids.shape[1] :]
    raw = model.tokenizer.decode(ids, skip_special_tokens=True).strip()
    ended = bool(len(ids) and int(ids[-1]) == model.tokenizer.eos_token_id)
    p = (
        min(float(s[0].float().softmax(-1)[t]) for s, t in zip(output.scores, ids))
        if ended
        else 0.0
    )
    return dict(
        raw=raw, confidence=p, ended=ended, tokens=len(ids), answer=raw if p > 0.8 else UNKNOWN
    )


def field_text(e):
    labels = ("人物", "时间", "事件", "感受", "原因", "关注")
    return "\n".join(
        [f"{name}：{FIELDS[task][e['values'][task]]}。" for name, task in zip(labels, TASKS)]
        + [f"场景：{e['project']}。", f"次数：{e['ordinal']}。"]
    )


@torch.no_grad()
def evaluate(suite, run, name):
    torch.set_num_threads(4)
    start = time.perf_counter()
    out = suite / name
    out.mkdir(exist_ok=False)
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((run / "adapter.pt").read_bytes()).hexdigest()
    assert digest == meta["adapter_sha256"]
    model = QwenMemory(meta["backbone"], adapter_path=run / "adapter.pt")
    events = json.loads((suite / "events.json").read_text(encoding="utf-8"))
    selected = json.loads((suite / "selection.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in events}
    store = model.store(out / "memories.sqlite", create=True)
    model.write_events(store, [e for e in events if e["stored"]])
    store.close()
    store = model.store(out / "memories.sqlite")
    rows = []
    for n, item in enumerate(selected):
        e = by_id[item["id"]]
        for wording in ("original", "novel"):
            q = item[wording]
            result = model.ask(store, e["owner"], q)
            if result["id"] == e["id"] and result["accepted"]:
                oracle = {k: result[k] for k in ("raw", "confidence", "ended", "tokens", "answer")}
            else:
                z, _ = store.fetch(e["owner"], [e["id"]])
                oracle = model.generate(
                    q, torch.tensor(z[0], device=model.device), True, details=True
                )
            empty = model.generate(q, present=False, details=True)
            empty["answer"] = UNKNOWN
            row = dict(
                id=e["id"],
                owner=e["owner"],
                task=item["task"],
                wording=wording,
                question=q,
                expected=FIELDS[item["task"]][e["values"][item["task"]]],
                memory=score(result, e, item["task"]),
                oracle_memory=score(oracle, e, item["task"]),
                text=score(text_read(model, e["text"], q), e, item["task"]),
                fields=score(text_read(model, field_text(e), q), e, item["task"]),
                empty=score(empty, e, item["task"]),
            )
            rows.append(row)
            with (out / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (n + 1) % 10 == 0:
            print(
                json.dumps(
                    dict(
                        stage="new_event_matrix",
                        model=name,
                        complete=n + 1,
                        total=60,
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
    summary = dict(
        model=name,
        run=str(run.resolve()),
        adapter_sha256=digest,
        selection_sha256=hashlib.sha256((suite / "selection.json").read_bytes()).hexdigest(),
        conditions={},
        retrieval={},
        by_field={},
        disk=store.counts(),
        seconds=time.perf_counter() - start,
    )
    for wording in ("original", "novel"):
        subset = [r for r in rows if r["wording"] == wording]
        summary["conditions"][wording] = {
            mode: {
                metric: dict(Counter(r[mode][metric] for r in subset))
                for metric in ("raw_status", "final_status")
            }
            for mode in ("text", "fields", "memory", "oracle_memory", "empty")
        }
        summary["retrieval"][wording] = dict(
            top1_correct=sum(r["memory"]["id"] == r["id"] for r in subset),
            correct_accepted=sum(
                r["memory"]["id"] == r["id"] and r["memory"]["accepted"] for r in subset
            ),
            wrong_accepted=sum(
                r["memory"]["id"] != r["id"] and r["memory"]["accepted"] for r in subset
            ),
        )
        summary["by_field"][wording] = {
            task: dict(Counter(r["memory"]["final_status"] for r in subset if r["task"] == task))
            for task in TASKS
        }
    store.close()
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite",
        type=Path,
        default=Path("experiments/episodic_memory/runs/qwen_scale_comparison_01"),
    )
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", type=Path)
    p.add_argument("--name")
    a = p.parse_args()
    if a.prepare:
        prepare(a.suite)
    else:
        if a.run is None or not a.name:
            p.error("--run and --name are required for evaluation")
        evaluate(a.suite, a.run, a.name)
