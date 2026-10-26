"""Compositional intent data. Task, background, anchor and scope vary independently.

Training and test use disjoint structural families and disjoint question realizations.
The annotation comes from the generated question AST, not a keyword parser at inference.
"""

import hashlib
import json
import random
from pathlib import Path

from .generate_data import ORDINALS, PROJECTS, TASKS, load_events
from .residual_language import training_question_grounded
from .train_evidence import question_text

BACKGROUNDS = [
    "那件事情发生时",
    "当时产生了感受",
    "回想参与交流的人",
    "当时讨论了原因",
    "回想那次的时间",
    "那次关注了具体细节",
    "和对方交流时",
    "事情发生的那一天",
]
PHRASES = {
    "person": [
        "和谁交流",
        "参与者是谁",
        "对方是哪位",
        "交流的人是谁",
        "谁参与了那次讨论",
        "我和哪位参与者讨论",
    ],
    "day": [
        "日期是什么",
        "发生在第几天",
        "那次是哪一天",
        "事情的时间是什么",
        "哪一天发生了那件事",
        "交流的日期是哪天",
    ],
    "event": [
        "发生了什么",
        "那件事是什么",
        "发生的事情是哪件",
        "事情是怎样的",
        "当时有什么事情",
        "发生的是哪件事",
    ],
    "emotion": [
        "心情如何",
        "感受是什么",
        "有什么情绪",
        "我的心情是什么",
        "产生了什么感受",
        "我感到怎样的情绪",
    ],
    "reason": [
        "为什么有这种感受",
        "产生心情的原因是什么",
        "情绪的缘由是什么",
        "为什么产生这种心情",
        "因为什么产生感受",
        "是什么原因让我有这种情绪",
    ],
    "focus": [
        "最关注什么",
        "注意力放在哪里",
        "在意什么方面",
        "关注的是什么",
        "最在意的方面是什么",
        "我注意了哪个方面",
    ],
}
TEST_PHRASES = {
    "person": ["当时和我交流的是谁", "参与那次交流的人是哪一位"],
    "day": ["那次交流是在第几天", "那件事发生时是第几天"],
    "event": ["那次交流时发生了什么", "哪件事情发生在那次交流时"],
    "emotion": ["那次交流时我的心情如何", "当时产生了哪种感受"],
    "reason": ["那次交流时我为何有那种心情", "我为何产生了那种感受"],
    "focus": ["那次交流时我关注的是什么", "我当时最在意什么"],
}


def anchor(rng):
    return f"{rng.choice(PROJECTS)}项目{rng.choice(ORDINALS)}讨论"


def compose(task, anchor_text, rng, family=None, heldout=False, variant=0):
    # Background is independent of the asked field. It may deliberately contain
    # words like '发生' or '感受' while the question asks a person or a cause.
    background, second = rng.sample(BACKGROUNDS, 2)
    stem = TEST_PHRASES[task][variant] if heldout else rng.choice(PHRASES[task])
    other = rng.choice([t for t in TASKS if t != task])
    distractor = rng.choice(PHRASES[other])
    family = rng.randrange(6) if family is None else family
    if heldout:
        if family == 6:
            return f"回想{anchor_text}：{background}，{stem}？{second}。"
        if family == 7:
            return f"{background}，{anchor_text}。{stem}？回想当时，{second}。"
        if family == 8:
            return f"关于{anchor_text}，我不是问{distractor}，我想问的是{stem}？"
        if family == 9:
            return f"有人问过“{distractor}”，但这次请回答：{stem}？说的是{anchor_text}。"
        raise ValueError("Held-out structural family required")
    if family == 0:
        return f"{anchor_text}，{background}，{stem}？"
    if family == 1:
        return f"{stem}？回忆{anchor_text}，{background}。"
    if family == 2:
        return f"{background}。{stem}？那次是{anchor_text}。"
    if family == 3:
        return f"回忆{anchor_text}，我想知道{stem}，不用回答{distractor}。"
    if family == 4:
        return f"{anchor_text}。{distractor}不是我想问的，我的问题是{stem}？"
    if family == 5:
        return f"在{anchor_text}，先前的问题是“{distractor}”。现在问：{stem}？"
    raise ValueError("Training structural family required")


def create(out, source, count=12000, val_count=1200):
    out.mkdir(parents=True, exist_ok=False)
    events = load_events(source / "data", "train")
    manifest = {
        "train_families": list(range(6)),
        "test_families": list(range(6, 10)),
        "datasets": {},
    }
    for mode in ("narrow", "compositional"):
        hashes = set()
        for split, n, seed in (("train", count, 6101), ("validation", val_count, 6102)):
            rng = random.Random(seed)
            rows = []
            while len(rows) < n:
                # Six counterfactual questions share the exact same background seed,
                # structural family, and anchor, eliminating label/background correlations.
                group_seed = rng.randrange(2**31)
                group_anchor = anchor(rng)
                family = rng.randrange(6)
                for task in TASKS:
                    if len(rows) >= n:
                        break
                    if mode == "compositional":
                        text = compose(task, group_anchor, random.Random(group_seed), family)
                        positive = compose(task, anchor(rng), rng)
                    else:
                        e = rng.choice(events)
                        text = (
                            training_question_grounded if rng.random() < 0.5 else question_text
                        )(e, task, rng)
                        positive = (
                            training_question_grounded if rng.random() < 0.5 else question_text
                        )(e, task, rng)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    positive_digest = hashlib.sha256(positive.encode()).hexdigest()
                    if digest in hashes or positive_digest in hashes:
                        continue
                    hashes.add(digest)
                    hashes.add(positive_digest)
                    rows.append(
                        dict(
                            text=text,
                            positive=positive,
                            task=task,
                            label=TASKS.index(task),
                            family=family if mode == "compositional" else -1,
                            group=group_seed,
                        )
                    )
            path = out / f"{mode}_{split}.jsonl"
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
            manifest["datasets"][f"{mode}_{split}"] = dict(
                rows=len(rows),
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def new_event(e, view):
    from .paraphrases import values

    f = values(e)
    if view == 0:
        return (
            f"回忆{e['anchor']}，参与的人是{f['person']}。发生{f['event']}的时间是{f['day']}。"
            f"因为{f['reason']}，我当时的心情为{f['emotion']}。在意的是{f['focus']}。"
        )
    return (
        f"那是{e['anchor']}，那件事为{f['event']}。感受是{f['emotion']}，缘由为{f['reason']}。"
        f"在{f['day']}和{f['person']}交流，关注的方面是{f['focus']}。"
    )
