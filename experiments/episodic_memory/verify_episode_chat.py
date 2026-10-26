"""Check that attaching the episode module preserves the explicit normal-chat path."""

import json
from pathlib import Path

import torch

from .episode_memory import EpisodeMemory
from .semantic_memory import SemanticMemory


def main():
    torch.set_num_threads(4)
    root = Path("experiments/episodic_memory/runs")
    output = root / "episode_chat_invariance_01.json"
    if output.exists():
        raise FileExistsError(output)
    base = SemanticMemory.load(root / "fact_reader_01")
    questions = [
        "你好，请用一句话介绍你能帮忙做什么。",
        "请用简单的话解释为什么会下雨。",
        "请把‘今天我想去公园散步’翻译成英文。",
        "计算17乘以6，并简短说明。",
    ]
    before = [base.chat(q) for q in questions]
    memory = EpisodeMemory.load(root / "episode_memory_grouped_01", backbone_model=base)
    after = [memory.model.chat(q) for q in questions]
    rows = [
        dict(question=q, before=b, after=a, identical=b == a)
        for q, b, a in zip(questions, before, after)
    ]
    output.write_text(
        json.dumps(
            dict(
                cases=rows,
                identical=sum(r["identical"] for r in rows),
                note="Explicit normal-chat bypass only, not a quality benchmark or automatic memory/chat switching test.",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    assert all(r["identical"] for r in rows)
    print(json.dumps(dict(identical=len(rows), output=str(output))))


if __name__ == "__main__":
    main()
