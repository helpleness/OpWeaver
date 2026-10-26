"""Paired long-stream controls; emits evidence and answers, no prose scoring."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from .bounded_attention import install
from .episode_memory import EPISODE_SYSTEM, EpisodeMemory
from .stream_experiment import generate, prepare, read, save, window


@torch.no_grad()
def event_text_control(model, ordered, question):
    body = "".join(
        f"\n片段{i + 1}（{'助手' if t.get('speaker') == 'assistant' else '用户'}）：{t['text']}"
        for i, t in enumerate(ordered)
    )
    if not ordered:
        body = "没有提供与本次问题匹配的事件记忆。"
    inputs = model.tokenizer.apply_chat_template(
        [
            dict(role="system", content=EPISODE_SYSTEM),
            dict(role="user", content=body + "\n问题：" + question),
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output = model.base.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=100,
        use_cache=True,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )
    ids = output[0, inputs.input_ids.shape[1] :]
    return dict(
        answer=model.tokenizer.decode(ids, skip_special_tokens=True).strip(),
        ended=bool(len(ids) and int(ids[-1]) == model.tokenizer.eos_token_id),
        input_tokens=inputs.input_ids.shape[1],
        status="generated",
        diagnostic="Identical retrieved fragments in original text; evaluator-only control",
    )


@torch.no_grad()
def run(out, run_path, regression_source=None):
    torch.set_num_threads(4)
    install()
    if regression_source:
        out.mkdir(parents=True, exist_ok=False)
        for name in ("conversation.json", "probes.json", "protocol.json"):
            shutil.copy2(regression_source / name, out / name)
    else:
        prepare(out, seed=57232, checkpoints=[8, 32, 64, 128], fresh=True)
    turns, probes, protocol = [
        read(out / name) for name in ("conversation.json", "probes.json", "protocol.json")
    ]
    modes = (
        ["episode_memory"]
        if regression_source
        else [
            "base_full_history",
            "base_recent_window",
            "old_memory",
            "episode_memory",
            "base_same_event_text",
        ]
    )
    protocol.update(
        conditions=modes,
        planned_outputs=len(probes) * len(modes),
        evaluation="Independent evidence review, no template/refusal/keyword semantic scorer",
        query_field_gate=False,
        revision_classifier=False,
        generation_template_gate=False,
        address_threshold=0.99,
        note="The old fact-support classifier is not part of the new architecture. Removing it is an architectural change, not proof of equal risk; hallucinations must be reviewed.",
        run=str(run_path),
    )
    save(out / "protocol.json", protocol)
    new = EpisodeMemory.load(run_path)
    old = new.model
    store = new.store(out / "episode.sqlite", create=True)
    old_store = old.store(out / "old.sqlite", create=True) if not regression_source else None
    start, prior = time.perf_counter(), 0
    checkpoints = []
    with (out / "outputs.jsonl").open("x", encoding="utf-8") as f:
        for cp in protocol["checkpoints"]:
            events = [
                dict(id=t["id"], owner="stream-user", timestamp=t["number"], text=t["text"])
                for t in turns[prior:cp]
            ]
            new.write_events(store, events)
            store.close()
            store = new.store(out / "episode.sqlite")
            if old_store:
                old.write_events(old_store, events)
                old_store.close()
                old_store = old.store(out / "old.sqlite")
            checkpoints.append(
                dict(
                    checkpoint=cp,
                    episode=store.counts(),
                    old=old_store.counts() if old_store else None,
                )
            )
            for probe in [p for p in probes if p["checkpoint"] == cp]:
                memory_result = None
                for mode in modes:
                    visible = turns[:cp]
                    if mode == "base_recent_window":
                        visible = window(old, visible, probe["question"], 2048)
                    if mode == "old_memory":
                        result = old.respond(
                            old_store, "stream-user", probe["question"], threshold=0.8
                        )
                        result["status"] = "memory_result"
                    elif mode == "episode_memory":
                        result = new.respond(store, "stream-user", probe["question"])
                        memory_result = result
                    elif mode == "base_same_event_text":
                        retrieved = set(memory_result["retrieval"]["ids"])
                        visible = [t for t in turns[:cp] if t["id"] in retrieved]
                        result = event_text_control(old, visible, probe["question"])
                    else:
                        result = generate(old, visible, probe["question"], 100)
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
                            checkpoint=cp,
                            probe=probe["probe"],
                            seconds=round(time.perf_counter() - start, 1),
                        )
                    ),
                    flush=True,
                )
            prior = cp
    store.close()
    if old_store:
        old_store.close()
    save(
        out / "manifest.json",
        dict(
            model=read(run_path / "metadata.json"),
            seconds=time.perf_counter() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            checkpoints=checkpoints,
            output_sha256=hashlib.sha256((out / "outputs.jsonl").read_bytes()).hexdigest(),
            review="pending",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--regression-source", type=Path)
    a = p.parse_args()
    run(a.out, a.run, a.regression_source)
