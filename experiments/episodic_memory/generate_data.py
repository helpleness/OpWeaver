"""Controlled Chinese episodes. Labels live ONLY in the experiment dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIELDS = {
    "person": ["小林", "小周", "小陈", "小何", "小许", "小唐", "小陆", "小杨"],
    "day": [f"第{i}天" for i in range(1, 17)],
    "event": [
        "叫错名字",
        "忘带资料",
        "提前完成",
        "弄坏杯子",
        "提出建议",
        "错过约定",
        "收到礼物",
        "解决故障",
    ],
    "emotion": ["尴尬", "安心", "高兴", "失望", "好奇", "生气", "紧张", "平静"],
    "reason": [
        "担心评价",
        "获得支持",
        "超出预期",
        "没有准备",
        "发生误会",
        "时间不足",
        "意见不同",
        "问题解决",
    ],
    "focus": [
        "他人反应",
        "任务结果",
        "时间安排",
        "自身表现",
        "对方需求",
        "环境变化",
        "后续计划",
        "具体细节",
    ],
}
TASKS = list(FIELDS)
ANSWERS = ["UNKNOWN"] + [v for values in FIELDS.values() for v in values]
OFFSETS = {}
for _task in TASKS:
    OFFSETS[_task] = 1 + sum(len(FIELDS[t]) for t in TASKS[: TASKS.index(_task)])
PROJECTS = [
    "画册",
    "机器人",
    "花园",
    "地图",
    "展览",
    "乐队",
    "课程",
    "旅行",
    "食谱",
    "摄影",
    "软件",
    "运动",
    "小说",
    "舞台",
    "模型",
    "天文",
]
ORDINALS = [f"第{i}次" for i in range(1, 17)]
QUESTIONS = {
    "person": ["当时和谁交流？", "参与的人是谁？", "回想起来，对方是哪位？"],
    "day": ["发生在第几天？", "那次是哪一天？", "回想起来，日期是什么？"],
    "event": ["当时发生了什么？", "那次的事件是什么？", "回想起来，发生的事情是哪件？"],
    "emotion": ["当时感受是什么？", "那次有什么情绪？", "回想起来，当时心情如何？"],
    "reason": [
        "当时评价的原因是什么？",
        "那次为什么有这种感受？",
        "回想起来，产生评价的缘由是什么？",
    ],
    "focus": ["当时最关注什么？", "那次注意力放在哪里？", "回想起来，最在意的方面是什么？"],
}


def render(event: dict, template: int) -> str:
    if template == 3:
        from .paraphrases import heldout_event

        return heldout_event(event)
    if template not in (0, 1, 2):
        raise ValueError(f"Unknown event template: {template}")
    f = {t: FIELDS[t][event["values"][t]] for t in TASKS}
    anchor = event["anchor"]
    if template == 0:
        return (
            f"{anchor}。时间{f['day']}，和{f['person']}交流。发生{f['event']}。"
            f"感受{f['emotion']}，因为{f['reason']}。关注{f['focus']}。"
        )
    if template == 1:
        return (
            f"记下{anchor}：在{f['day']}，参与者是{f['person']}。"
            f"事情是{f['event']}；我觉得{f['emotion']}，原因是{f['reason']}；在意{f['focus']}。"
        )
    return (
        f"回忆{anchor}。我在意{f['focus']}。那是{f['day']}，对方是{f['person']}。"
        f"发生的事为{f['event']}，当时心情{f['emotion']}，缘由是{f['reason']}。"
    )


def question(event: dict, task: str, template: int) -> str:
    if template == 3:
        from .paraphrases import heldout_question

        return heldout_question(event, task)
    return event["anchor"] + "，" + QUESTIONS[task][template]


def build_tokenizer(extended=False) -> dict:
    # Closed, declared experimental vocabulary; no test labels are inspected.
    atoms = sorted(set(ANSWERS[1:] + PROJECTS + ORDINALS), key=lambda x: (-len(x), x))
    probe = {"anchor": "画册项目第1次讨论", "values": dict.fromkeys(TASKS, 0)}
    chars = sorted(
        set(
            "".join(
                render(probe, i) + "".join(question(probe, t, i) for t in TASKS) for i in range(3)
            )
        )
    )
    if extended:
        from .paraphrases import vocabulary_text

        chars = sorted(set(chars) | set(vocabulary_text(probe)))
    vocab = ["<PAD>", "<CLS>", "<UNK>"] + sorted(set(atoms + chars))
    return {"vocab": vocab, "atoms": atoms, "max_length": 96}


class Tokenizer:
    def __init__(self, spec: dict):
        self.spec = spec
        self.ids = {s: i for i, s in enumerate(spec["vocab"])}
        self.atoms = spec["atoms"]

    def encode(self, text: str) -> list[int]:
        result, i = [1], 0
        while i < len(text):
            token = next((a for a in self.atoms if text.startswith(a, i)), text[i])
            result.append(self.ids.get(token, 2))
            i += len(token)
        if len(result) > self.spec["max_length"]:
            raise ValueError(
                "Event exceeds declared token budget; do not silently discard evidence"
            )
        return result


def generate(
    root: Path,
    seed: int = 17,
    train_histories: int = 192,
    eval_histories: int = 12,
    train_events: int = 32,
    test_events: int = 96,
    extended_vocabulary: bool = False,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    if max(train_events, test_events) + 8 > 256:
        raise ValueError("Only 256 distinct context anchors per history")
    rng = random.Random(seed)
    manifest = {
        "seed": seed,
        "schema": 1,
        "splits": {},
        "answers": ANSWERS,
        "note": "Random independent facts; held-out histories and template 2. No free-form chat claim.",
    }
    for split, histories, count, heldout in [
        ("train", train_histories, train_events, False),
        ("validation", eval_histories, train_events, False),
        ("test", eval_histories, test_events, False),
        ("test_template", eval_histories, test_events, True),
    ]:
        rows = []
        for h in range(histories):
            owner = f"{split}-{h:04d}"
            contexts = rng.sample([(p, o) for p in PROJECTS for o in ORDINALS], count + 8)
            for i, (project, ordinal) in enumerate(contexts):
                e = {
                    "id": f"{owner}-{i:04d}",
                    "owner": owner,
                    "timestamp": i,
                    "anchor": f"{project}项目{ordinal}讨论",
                    "project": project,
                    "ordinal": ordinal,
                    "values": {t: rng.randrange(len(FIELDS[t])) for t in TASKS},
                    "stored": i < count,
                    "query_template": 2 if heldout else rng.randrange(2),
                }
                e["text"] = render(e, 2 if heldout else rng.randrange(2))
                rows.append(e)
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "histories": histories,
            "stored_per_history": count,
            "unobserved_per_history": 8,
            "events": len(rows),
        }
    (root / "tokenizer.json").write_text(
        json.dumps(build_tokenizer(extended_vocabulary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_events(root: Path, split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    print(json.dumps(generate(args.out, args.seed), ensure_ascii=False, indent=2))
