"""Matched growing-history versus persistent-memory experiment up to 32K tokens.

No training or judge heuristics. The original model receives ALL history inside
its window. Prefix KV reuse preserves all causal keys and is checked against an
uncached decode before evaluation. Probe answers never enter later history.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from . import episode_memory
from .bounded_attention import install
from .episode_memory import EpisodeMemory
from .semantic_data import event
from .stream_experiment import describe, read, save

SYSTEM = (
    "你是帮助用户回顾经历的助手。只依据本次提供的聊天历史或事件记忆回答最后的问题。"
    "区分不同项目和讨论次数，不补写人物、行为者、原因或未记录的事实。"
    "明确更正作废了旧说法时采用更正；未经核实且无法确定的矛盾不能任选一方。"
    "材料中的用户经历与阅读材料要分清，不把别人的文字当成用户经历。"
    "缺少证据就自然解释无法确定，不要求固定拒答句。简短回答所问内容。"
)
BUDGETS = [2048, 8192, 16384, 24576, 32000]
MAX_NEW = 100


def template(model, turns, question=None):
    msgs = [dict(role="system", content=SYSTEM)]
    for t in turns:
        msgs.extend(
            [dict(role="user", content=t["text"]), dict(role="assistant", content=t["assistant"])]
        )
    if question is not None:
        msgs.append(dict(role="user", content=question))
    encoded = model.tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=question is not None,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded.input_ids[0].tolist()


def prepare(model, corpus, out, seed):
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    pool = []
    with (corpus / "test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "fineweb2_zh":
                pool.append(r)
    rng.shuffle(pool)
    addresses = [(a, b) for a in range(16) for b in range(16)]
    rng.shuffle(addresses)
    reserved = addresses.pop()
    turns, originals, checkpoints, probes = [], [], [], []
    ai = fi = 0
    corrected = disputed = False

    def make_event(kind="event", original=None):
        nonlocal ai
        e = (
            copy.deepcopy(original)
            if original
            else event(rng, "unused", "long-user", addresses[ai])
        )
        if original is None:
            ai += 1
            # Coherent emotions and causes reduce a synthetic common-sense confound.
            pairs = [(0, 3), (1, 7), (2, 1), (3, 5), (4, 2), (5, 6), (6, 0), (7, 7)]
            e["values"]["emotion"], e["values"]["reason"] = rng.choice(pairs)
        prefix = ""
        if kind == "correction":
            e["values"]["reason"] = 0
            prefix = "我重新确认过了：早先对这件事的原因解释不准确，原来的原因请撤回，以下原因才是我确认的。"
        elif kind == "unresolved_variant":
            e["values"]["emotion"] = (e["values"]["emotion"] + 3) % 8
            prefix = "这件事的心情我又记起另一种说法，但原来那版和这一版都没核实，我现在确实无法判定哪种心情是真的。"
        elif len(originals) == 0:
            e["values"]["emotion"], e["values"]["reason"] = 0, 3
        elif len(originals) == 2:
            e["values"]["person"] = -1
        elif len(originals) == 3:
            e["values"]["emotion"], e["values"]["reason"] = 7, 7
        return dict(
            kind=kind,
            anchor=e["anchor"],
            values=e["values"],
            text=prefix + describe(e, ai % 3),
            assistant="这段经历我记下了，你可以继续。",
        ), e

    def append(t):
        t.update(id=f"T{len(turns) + 1:04d}", number=len(turns) + 1)
        if len(model.tokenizer.encode(t["text"], add_special_tokens=False)) > 192:
            raise ValueError("Do not silently truncate a memory fragment")
        turns.append(t)

    for _ in range(5):
        t, e = make_event()
        append(t)
        originals.append(e)
    for budget in BUDGETS:
        # Leave space for the longest question and generated answer.
        while True:
            n = len(template(model, turns))
            if not corrected and n > 5700:
                t, _ = make_event("correction", originals[0])
                corrected = True
            elif not disputed and n > 11700:
                t, _ = make_event("unresolved_variant", originals[1])
                disputed = True
            elif len(turns) % 4 == 0 and ai < 180:
                t, _ = make_event()
            else:
                r = pool[fi]
                fi += 1
                ids = model.tokenizer.encode(r["text"], add_special_tokens=False)
                body = model.tokenizer.decode(ids[:145], skip_special_tokens=True)
                t = dict(
                    kind="reading_distractor",
                    anchor=None,
                    text="另外，我读到一段材料，先作为阅读内容留在对话里：" + body,
                    assistant="收到这段阅读材料，我们接着聊。",
                    source_group_id=r["group_id"],
                    source_text_sha256=r["text_sha256"],
                )
            trial = turns + [t]
            if len(template(model, trial)) > budget - 80:
                # Don't discard an event update at a boundary.
                if t["kind"] == "correction":
                    corrected = False
                if t["kind"] == "unresolved_variant":
                    disputed = False
                break
            append(t)
        ordinary = [t for t in turns if t["kind"] == "event"]
        targets = [
            ("early", turns[0], "回想那次交谈，与我交流的那个人叫什么？", "person"),
            ("middle", ordinary[len(ordinary) // 2], "请说说那次实际发生的事情。", "event"),
            ("recent", ordinary[-1], "那时我把注意力放在哪方面？", "focus"),
            ("correction", turns[0], "根据截至现在我确认的叙述，那份心情是因为什么？", "reason"),
            (
                "conflict",
                turns[1],
                "综合已有叙述，当时我的心情是什么？无法确定就说明分歧。",
                "emotion",
            ),
            ("missing", turns[2], "那次和我交流的人的名字是什么？", "person"),
            ("stable", turns[3], "我当时的心情具体是什么？", "emotion"),
        ]
        missing = event(rng, "absent", "long-user", reserved)
        targets.append(("absent", missing, "那次与我交流的人叫什么名字？", "person"))
        cp = len(turns)
        lengths = []
        for name, t, q, field in targets:
            question = t["anchor"] + "，" + q
            ids = template(model, turns, question)
            assert len(ids) + MAX_NEW <= model.base.config.max_position_embeddings
            lengths.append(len(ids))
            probes.append(
                dict(
                    id=f"L{budget:05d}-{name}",
                    checkpoint=cp,
                    budget=budget,
                    probe=name,
                    field=field,
                    question=question,
                    relevant_turn_ids=[x["id"] for x in turns if x["anchor"] == t["anchor"]],
                )
            )
        checkpoints.append(
            dict(
                checkpoint=cp,
                budget=budget,
                prefix_tokens=len(template(model, turns)),
                question_input_token_range=[min(lengths), max(lengths)],
                user_turns=cp,
                chat_messages=2 * cp,
                event_fragments=sum(t["anchor"] is not None for t in turns),
                reading_fragments=sum(t["anchor"] is None for t in turns),
            )
        )
    save(out / "conversation.json", turns)
    save(out / "probes.json", probes)
    protocol = dict(
        seed=seed,
        checkpoints=[x["checkpoint"] for x in checkpoints],
        token_checkpoints=checkpoints,
        conditions=["base_full_history", "episode_open_grouped"],
        planned_outputs=len(probes) * 2,
        same_system_prompt=SYSTEM,
        max_new_tokens=MAX_NEW,
        model_context_limit=model.base.config.max_position_embeddings,
        full_history_truncation=False,
        probe_responses_reinserted=False,
        training=False,
        memory_policy="Every user turn, including distractors, encoded independently; acknowledgments are fixed and contain no new facts",
        scope="Controlled event facts interleaved with heldout Chinese reading material; not a fully natural human conversation",
        addressing="Existing closed 16 x 16 address space; automatic retrieval without gold event IDs",
        prefix_cache="All causal KV retained; incremental prefill; fork cache for each probe; no probe pollution",
        hashes={
            n: hashlib.sha256((out / n).read_bytes()).hexdigest()
            for n in ["conversation.json", "probes.json"]
        },
    )
    save(out / "protocol.json", protocol)
    return turns, probes, protocol


@torch.no_grad()
def extend(model, prefix, cache=None, consumed=0, chunk=256):
    started = time.perf_counter()
    for pos in range(consumed, len(prefix), chunk):
        end = min(pos + chunk, len(prefix))
        output = model.base(
            input_ids=torch.tensor([prefix[pos:end]], device=model.device),
            attention_mask=torch.ones(1, end, dtype=torch.long, device=model.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
    assert cache.get_seq_length() == len(prefix)
    return cache, time.perf_counter() - started


@torch.no_grad()
def answer(model, full, cache=None):
    torch.cuda.synchronize()
    start = time.perf_counter()
    generated = model.base.generate(
        input_ids=torch.tensor([full], device=model.device),
        attention_mask=torch.ones(1, len(full), dtype=torch.long, device=model.device),
        past_key_values=copy.deepcopy(cache) if cache is not None else None,
        do_sample=False,
        max_new_tokens=MAX_NEW,
        use_cache=True,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    ids = generated[0, len(full) :]
    return dict(
        answer=model.tokenizer.decode(ids, skip_special_tokens=True).strip(),
        status="generated",
        ended=bool(len(ids) and int(ids[-1]) == model.tokenizer.eos_token_id),
        input_tokens=len(full),
        output_tokens=len(ids),
        seconds=time.perf_counter() - start,
    )


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    install()
    episode_memory.EPISODE_SYSTEM = SYSTEM
    new = EpisodeMemory.load(args.run)
    base = new.model
    base.base.eval()
    args.out.mkdir(parents=True, exist_ok=False)
    save(
        args.out / "plan.json",
        dict(
            seeds=[57511, 57512],
            token_budgets=BUDGETS,
            probes_per_checkpoint=8,
            planned_outputs=160,
            main_comparison="unchanged full-history Qwen versus same Qwen with trained disk memory",
            model_run=str(args.run),
            weights=read(args.run / "metadata.json"),
            no_parameter_updates=True,
            beyond_context="Not run; all full-history inputs plus generation fit native 32768 positions",
        ),
    )
    # Method check independent of evaluation events.
    tiny = [
        dict(text="我今天读了一本书。", assistant="知道了。"),
        dict(text="书名是春日。", assistant="请继续。"),
    ]
    prefix = template(base, tiny)
    cache, _ = extend(base, prefix, chunk=17)
    checks = []
    for q in ["我读的书叫什么名字？", "我读书是哪一天？"]:
        full = template(base, tiny, q)
        assert full[: len(prefix)] == prefix
        uncached = answer(base, full)
        cached = answer(base, full, cache)
        checks.append(
            dict(
                question=q,
                uncached=uncached,
                cached=cached,
                identical=uncached["answer"] == cached["answer"],
            )
        )
    save(args.out / "cache_equivalence.json", checks)
    if not all(c["identical"] for c in checks):
        raise RuntimeError("Cached decode differs in method check; do not publish comparison")
    del cache
    for seed in [57511, 57512]:
        out = args.out / f"seed_{seed}"
        turns, probes, protocol = prepare(base, args.corpus, out, seed)
        print(
            json.dumps(
                dict(stage="prepared", seed=seed, checkpoints=protocol["token_checkpoints"])
            ),
            flush=True,
        )
        store = new.store(out / "memory.sqlite", create=True)
        cache = None
        consumed = prior = 0
        previous_prefix = []
        checkpoint_log = []
        with (out / "outputs.jsonl").open("x", encoding="utf-8") as f:
            for cp in protocol["checkpoints"]:
                torch.cuda.reset_peak_memory_stats()
                prefix = template(base, turns[:cp])
                assert prefix[: len(previous_prefix)] == previous_prefix
                cache, prefill_seconds = extend(base, prefix, cache, consumed)
                consumed = len(prefix)
                previous_prefix = prefix
                start = time.perf_counter()
                # Write in bounded batches rather than keeping all hidden features resident.
                for offset in range(prior, cp, 16):
                    new.write_events(
                        store,
                        [
                            dict(
                                id=t["id"], owner="long-user", timestamp=t["number"], text=t["text"]
                            )
                            for t in turns[offset : min(offset + 16, cp)]
                        ],
                    )
                store.close()
                store = new.store(out / "memory.sqlite")
                writing_seconds = time.perf_counter() - start
                for probe in [p for p in probes if p["checkpoint"] == cp]:
                    full = template(base, turns[:cp], probe["question"])
                    assert full[: len(prefix)] == prefix
                    original = answer(base, full, cache)
                    start = time.perf_counter()
                    modified = new.respond(store, "long-user", probe["question"])
                    torch.cuda.synchronize()
                    modified["seconds"] = time.perf_counter() - start
                    for mode, result in [
                        ("base_full_history", original),
                        ("episode_open_grouped", modified),
                    ]:
                        f.write(
                            json.dumps(
                                dict(
                                    id=probe["id"],
                                    checkpoint=cp,
                                    budget=probe["budget"],
                                    mode=mode,
                                    question=probe["question"],
                                    visible_turn_ids=[t["id"] for t in turns[:cp]],
                                    result=result,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    f.flush()
                    assert cache.get_seq_length() == len(prefix)
                    print(
                        json.dumps(
                            dict(
                                stage="probe",
                                seed=seed,
                                budget=probe["budget"],
                                probe=probe["probe"],
                                base_seconds=round(original["seconds"], 2),
                                memory_seconds=round(modified["seconds"], 2),
                            )
                        ),
                        flush=True,
                    )
                checkpoint_log.append(
                    dict(
                        checkpoint=cp,
                        input_tokens=len(prefix),
                        prefill_seconds=prefill_seconds,
                        writing_seconds=writing_seconds,
                        store=store.counts(),
                        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                    )
                )
                save(out / "checkpoint_runtime.json", checkpoint_log)
                prior = cp
        store.close()
        del cache
        save(
            out / "manifest.json",
            dict(
                outputs=80,
                checkpoints=checkpoint_log,
                output_sha256=hashlib.sha256((out / "outputs.jsonl").read_bytes()).hexdigest(),
                review="pending",
            ),
        )
    save(args.out / "completion.json", dict(outputs=160, training=False, review="pending"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--run",
        type=Path,
        default=Path("experiments/episodic_memory/runs/episode_memory_grouped_01"),
    )
    p.add_argument(
        "--corpus", type=Path, default=Path("experiments/episodic_memory/corpora/open_01/prepared")
    )
    run(p.parse_args())
