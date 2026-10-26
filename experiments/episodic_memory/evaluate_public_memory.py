"""Unseen real-dialogue fidelity diagnostic with explicitly provided grouping.

It tests disk codes/reading, not automatic open-domain event association.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .episode_memory import EpisodeMemory
from .evaluate_episode_memory import event_text_control
from .luna_review import digest, dump_rows
from .stream_experiment import save
from .stream_review import STREAM_RUBRIC


@torch.no_grad()
def run(corpus, small_path, public_path, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    public = EpisodeMemory.load(public_path)
    small = EpisodeMemory.load(small_path, public.model)
    candidates = []
    with (corpus / "test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["source"] == "oasst2" and len(row.get("turns", [])) >= 2:
                candidates.append(row)
    random.Random(57401).shuffle(candidates)
    cases, seen, language_counts = [], set(), {"zh": 0, "en": 0}
    question = "用户到底想做什么？请只概括用户已经明确说出的需求和限制，不要把助手的建议说成用户已决定或执行的事。"
    for row in candidates:
        if row["group_id"] in seen or language_counts[row["language"]] >= 4:
            continue
        turns = row["turns"][:4]
        lengths = [len(public.tokenizer.encode(t["text"], add_special_tokens=False)) for t in turns]
        if max(lengths) > 192 or sum(lengths) > 512:
            continue
        cases.append(
            dict(
                id=f"P{len(cases):02d}",
                source_group_id=row["group_id"],
                language=row["language"],
                question=question,
                turns=[
                    dict(id=f"E{i + 1}", text=t["text"], speaker=t["role"])
                    for i, t in enumerate(turns)
                ],
            )
        )
        language_counts[row["language"]] += 1
        seen.add(row["group_id"])
        if len(cases) == 8:
            break
    if len(cases) != 8:
        raise ValueError(f"Not enough short unseen dialogues: {language_counts}")
    save(out / "cases.json", cases)
    outputs = []
    for name, model in [("episode_small", small), ("episode_open", public)]:
        path = out / (name + ".sqlite")
        store = model.store(path, create=True)
        for c in cases:
            store.add_many(
                [
                    dict(
                        id=c["id"] + "-" + t["id"],
                        owner=c["id"],
                        timestamp=i,
                        speaker=t["speaker"],
                        memory=model.encode_text(t["text"]).cpu().numpy(),
                        key=np.ones(32, np.float32) / np.sqrt(32),
                    )
                    for i, t in enumerate(c["turns"])
                ]
            )
        store.close()
        store = model.store(path)
        for c in cases:
            ordered = store.ordered(c["id"], [c["id"] + "-" + t["id"] for t in c["turns"]])
            zs, _ = store.fetch(c["id"], [eid for eid, _ in ordered])
            fragments = [
                (torch.tensor(z, device=model.device), speaker)
                for z, (_, speaker) in zip(zs, ordered)
            ]
            result = model.generate(fragments, question)
            outputs.append(dict(id=c["id"], mode=name, result=result))
        store.close()
    for c in cases:
        outputs.append(
            dict(
                id=c["id"],
                mode="base_event_text",
                result=event_text_control(public.model, c["turns"], question),
            )
        )
    dump_rows(out / "outputs.jsonl", outputs)
    random.Random(57402).shuffle(outputs)
    packets, mapping = [], {}
    for i, r in enumerate(outputs):
        c = next(c for c in cases if c["id"] == r["id"])
        p = dict(
            review_id=f"P{i:03d}",
            question=question,
            answer=r["result"]["answer"],
            evidence=[
                dict(
                    id=t["id"], text=("用户：" if t["speaker"] == "user" else "助手：") + t["text"]
                )
                for t in c["turns"]
            ],
            execution_state="本次响应已完成。此项是已知同一对话的编码/读取诊断，未检验自动事件归组；没有后台任务。",
            generation_ended=r["result"]["ended"],
        )
        p["binding_sha256"] = digest(p)
        packets.append(p)
        mapping[p["review_id"]] = dict(mode=r["mode"], case_id=r["id"])
    dump_rows(out / "all_packets.jsonl", packets)
    for i, letter in enumerate("abc"):
        dump_rows(out / f"packets_{letter}.jsonl", packets[i * 8 : (i + 1) * 8])
    save(out / "private_mapping.json", mapping)
    (out / "rubric.txt").write_text(
        STREAM_RUBRIC
        + "\n本组是开放域真实对话的用户需求概括，不是六字段任务。逐条检查概括是否忠于用户说过的话，以及是否把助手建议误写成用户计划。允许合理概括，不要求逐字复述。证据中说话者标签必须尊重。\n",
        encoding="utf-8",
    )
    save(
        out / "manifest.json",
        dict(
            cases=len(cases),
            outputs=len(outputs),
            source_split="test",
            languages=language_counts,
            grouping="provided conversation tree; not automatic event retrieval",
            no_raw_text_in_memory=True,
            judgments="pending",
            model_paths=[str(small_path), str(public_path)],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--small", type=Path, default=Path("experiments/episodic_memory/runs/episode_memory_02")
    )
    p.add_argument(
        "--public",
        type=Path,
        default=Path("experiments/episodic_memory/runs/episode_memory_open_01"),
    )
    a = p.parse_args()
    run(a.corpus, a.small, a.public, a.out)
