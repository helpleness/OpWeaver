"""Causal long-conversation evaluation: unchanged model, growing on-disk memory.

Every user turn (including chatter) is written as one chunk. Probe questions and
responses never enter the continuing conversation or memory. No oracle retrieval
is used by a model; relevant-turn annotations are for external evaluation only.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from .generate_data import FIELDS, TASKS
from .semantic_data import event
from .semantic_memory import SemanticMemory

SYSTEM = (
    "你是帮助用户回顾个人经历的助手。只依据本次提供的聊天历史回答最后的问题。"
    "区分不同项目和讨论次数，不补写人物、行为者或原因。明确更正作废了旧说法时采用更正；"
    "未经核实的矛盾不能任选一方。缺少证据就解释无法确定。自然简短作答。"
)
CHECKPOINTS = [8, 32, 64, 128]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe(e, style):
    values = {t: FIELDS[t][i] for t, i in e["values"].items() if i >= 0}
    formats = [
        [
            "和我交谈的是{}",
            "时间是{}",
            "发生的事情是{}",
            "那时我感到{}",
            "这种感受是因为{}",
            "我主要关注{}",
        ],
        [
            "对方叫{}",
            "这件事在{}发生",
            "那次发生了{}",
            "我的心情是{}",
            "这份心情源于{}",
            "我的注意力在{}",
        ],
        [
            "交流对象记作{}",
            "发生日是{}",
            "具体事情是{}",
            "当时我的感受为{}",
            "产生这种感觉的原因是{}",
            "我最在意{}",
        ],
    ]
    return (
        "接着说说"
        + e["anchor"]
        + "。"
        + "，".join(formats[style][i].format(values[t]) for i, t in enumerate(TASKS) if t in values)
        + "。"
    )


def prepare(out, seed=9213, checkpoints=None, fresh=False):
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    addresses = [(p, o) for p in range(16) for o in range(16)]
    rng.shuffle(addresses)
    turns, originals, ai = [], {}, 0
    for n in range(1, max(checkpoints) + 1):
        eid = f"T{n:03d}"
        prefix, kind = "", "event"
        if n == 20:
            e = copy.deepcopy(originals[1])
            e["id"] = eid
            e["values"]["reason"] = (e["values"]["reason"] + 3) % len(FIELDS["reason"])
            prefix = "我要更正早先这次经历的原因，之前的原因记错了，以本条为准。"
            if fresh:
                prefix = "这件事的缘由我已重新查证，旧的归因应撤回，现在确认的是下面这一项。"
            kind = "correction"
        elif n == 40:
            e = copy.deepcopy(originals[2])
            e["id"] = eid
            e["values"]["emotion"] = (e["values"]["emotion"] + 3) % len(FIELDS["emotion"])
            prefix = "关于下面这次经历，我另有一条心情不一样的回忆，和早先两种说法都还没核实，不能认定哪条对。"
            if fresh:
                prefix = "对于当时的心情，我又想到一种可能。它和上次不一致，目前两边都没有得到证实，请保留分歧。"
            kind = "unresolved_variant"
        elif n % 8 == 0:
            text = rng.choice(
                [
                    "先聊到这里，我想慢慢把经历理一理，稍后继续。",
                    "刚才说的先保留，暂时不用总结，我还想接着聊。",
                    "整理这些片段需要耐心，我们继续按顺序聊就好。",
                ]
            )
            turns.append(
                dict(
                    id=eid,
                    number=n,
                    kind="chatter",
                    anchor=None,
                    text=text,
                    assistant="好，我们继续。",
                )
            )
            continue
        else:
            e = event(rng, eid, "stream-user", addresses[ai])
            ai += 1
            if n == 4:
                e["values"]["person"] = -1
            originals[n] = copy.deepcopy(e)
        text = prefix + describe(e, (n - 1) % 3)
        turns.append(
            dict(
                id=eid,
                number=n,
                kind=kind,
                anchor=e["anchor"],
                values=e["values"],
                text=text,
                assistant="这段经历我了解了，请继续。",
            )
        )
    probes = []
    for cp in checkpoints:
        ordinary = [t for t in turns[:cp] if t["kind"] == "event"]
        targets = [
            ("early", turns[0], "和我交谈的对象叫什么名字？", "person"),
            ("middle", ordinary[len(ordinary) // 2], "那次具体发生了什么事？", "event"),
            ("recent", ordinary[-1], "我当时主要关注哪个方面？", "focus"),
            ("missing", turns[3], "和我交流的人叫什么名字？", "person"),
            ("correction", turns[0], "按目前明确确认的说法，那份感受的原因是什么？", "reason"),
            ("conflict", turns[1], "根据目前的记录，那时我的心情是什么？", "emotion"),
        ]
        for name, target, question, field in targets:
            if fresh:
                question = dict(
                    early="那回交流的另一方，姓名记下来了没有？请说出名字。",
                    middle="我想还原事情本身，那一回究竟出了什么事？",
                    recent="那次占据我注意的，主要是哪一方面？",
                    missing="这回交谈的另一位具体是谁？",
                    correction="以我已经核定的版本为准，那种心情到底源自什么？",
                    conflict="只看我当时内心的感受，能确定是哪一种吗？",
                )[name]
            relevant = [t["id"] for t in turns[:cp] if t["anchor"] == target["anchor"]]
            probes.append(
                dict(
                    id=f"C{cp:03d}-{name}",
                    checkpoint=cp,
                    probe=name,
                    question=target["anchor"] + "，" + question,
                    field=field,
                    relevant_turn_ids=relevant,
                )
            )
    save(out / "conversation.json", turns)
    save(out / "probes.json", probes)
    save(
        out / "protocol.json",
        dict(
            seed=seed,
            checkpoints=checkpoints,
            user_turns=max(checkpoints),
            conversation_messages=max(checkpoints) * 2,
            checkpoint_probes=6,
            planned_outputs=len(probes) * 3,
            conditions=["base_full_history", "base_recent_window", "disk_memory"],
            recent_input_token_budget=2048,
            max_new_tokens=100,
            threshold=0.8,
            address_threshold=0.99,
            training=False,
            probe_responses_reinserted=False,
            writer_policy="Every user message, including chatter, is independently encoded and appended in order; no summary or event-label filtering",
            scope="Controlled synthetic conversation, not open-domain natural chat. Per-user-turn chunking; no learned event boundary detector.",
            hashes={
                name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                for name in ("conversation.json", "probes.json")
            },
        ),
    )


def messages(turns, question):
    msgs = [dict(role="system", content=SYSTEM)]
    for turn in turns:
        msgs.extend(
            [
                dict(role="user", content=turn["text"]),
                dict(role="assistant", content=turn["assistant"]),
            ]
        )
    msgs.append(dict(role="user", content=question))
    return msgs


def tokenize(model, turns, question):
    return model.tokenizer.apply_chat_template(
        messages(turns, question),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def window(model, turns, question, budget):
    # Whole user+assistant turns only; no silent truncation within a fact.
    lo, hi = 0, len(turns)
    while lo < hi:
        mid = (lo + hi) // 2
        if tokenize(model, turns[mid:], question).input_ids.shape[1] <= budget:
            hi = mid
        else:
            lo = mid + 1
    selected = turns[lo:]
    assert tokenize(model, selected, question).input_ids.shape[1] <= budget
    return selected


@torch.no_grad()
def generate(model, turns, question, max_new_tokens):
    inputs = tokenize(model, turns, question).to(model.device)
    n = inputs.input_ids.shape[1]
    if n + max_new_tokens > model.base.config.max_position_embeddings:
        return dict(answer=None, status="context_limit_not_run", input_tokens=n, ended=None)
    model.base.eval()
    started = time.perf_counter()
    try:
        out = model.base.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
        ids = out[0, n:]
        return dict(
            answer=model.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            input_tokens=n,
            output_tokens=len(ids),
            ended=bool(len(ids) and int(ids[-1]) == model.tokenizer.eos_token_id),
            seconds=time.perf_counter() - started,
            status="generated",
        )
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(answer=None, status="oom_not_run", input_tokens=n, ended=None)


@torch.no_grad()
def run(out, run_path, bounded=False):
    torch.set_num_threads(4)
    if bounded:
        from .bounded_attention import install

        install()
    protocol = read(out / "protocol.json")
    for name, sha in protocol["hashes"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    turns, probes = read(out / "conversation.json"), read(out / "probes.json")
    model = SemanticMemory.load(run_path)
    feature_lengths = [len(model.tokenizer.encode(t["text"])) for t in turns]
    if max(feature_lengths) > 192:
        raise ValueError("A user turn exceeds the existing writer budget; do not silently truncate")
    db_path = out / "memories.sqlite"
    store = model.store(db_path, create=True)
    start, prior = time.perf_counter(), 0
    checkpoints = []
    with (out / "outputs.jsonl").open("x", encoding="utf-8") as f:
        for cp in CHECKPOINTS:
            new = turns[prior:cp]
            model.write_events(
                store,
                [
                    dict(id=t["id"], owner="stream-user", timestamp=t["number"], text=t["text"])
                    for t in new
                ],
            )
            store.close()
            store = model.store(db_path)  # Reconstruct disk payload access and cluster index.
            counts = store.counts()
            cp_info = dict(
                checkpoint=cp,
                user_turns=cp,
                chat_messages=cp * 2,
                stream_tokens=tokenize(model, turns[:cp], "继续。").input_ids.shape[1],
                store=counts,
                db_bytes=db_path.stat().st_size,
            )
            checkpoints.append(cp_info)
            print(json.dumps(dict(stage="checkpoint", **cp_info)), flush=True)
            for probe in [p for p in probes if p["checkpoint"] == cp]:
                for mode in protocol["conditions"]:
                    visible = turns[:cp]
                    if mode == "base_recent_window":
                        visible = window(
                            model, visible, probe["question"], protocol["recent_input_token_budget"]
                        )
                    if mode == "disk_memory":
                        before = time.perf_counter()
                        result = model.respond(
                            store, "stream-user", probe["question"], threshold=protocol["threshold"]
                        )
                        result["status"] = "memory_result"
                        result["seconds"] = time.perf_counter() - before
                    else:
                        result = generate(
                            model, visible, probe["question"], protocol["max_new_tokens"]
                        )
                    row = dict(
                        id=probe["id"],
                        checkpoint=cp,
                        mode=mode,
                        question=probe["question"],
                        visible_turn_ids=[t["id"] for t in visible],
                        result=result,
                    )
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(
                        json.dumps(
                            dict(
                                stage="answer",
                                checkpoint=cp,
                                probe=probe["probe"],
                                mode=mode,
                                elapsed_s=round(time.perf_counter() - start, 1),
                            )
                        ),
                        flush=True,
                    )
            prior = cp
    store.close()
    save(
        out / "run_manifest.json",
        dict(
            checkpoints=checkpoints,
            output_sha256=hashlib.sha256((out / "outputs.jsonl").read_bytes()).hexdigest(),
            model_metadata=read(run_path / "metadata.json"),
            model_max_context=model.base.config.max_position_embeddings,
            max_user_turn_tokens=max(feature_lengths),
            seconds=time.perf_counter() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            attention_backend="query_chunked_sdpa_128_full_keys" if bounded else "torch_sdpa",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--bounded-attention", action="store_true")
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/fact_reader_01")
    )
    a = p.parse_args()
    prepare(a.out) if a.prepare else run(a.out, a.run, a.bounded_attention)
