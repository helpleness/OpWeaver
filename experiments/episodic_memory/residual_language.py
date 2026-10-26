"""Separate train and sealed test sentence structures; existing vocabulary only."""

from .paraphrases import values

TRAIN_STEMS = {
    "person": ["那次交流，参与的人是哪位", "那位参与交流的人是谁"],
    "day": ["回忆那件事，发生的日期是什么", "那次事情发生的时间是哪一天"],
    "event": ["回忆那次，发生的事件是什么", "那件发生的事情是什么"],
    "emotion": ["回忆那次，产生的心情是什么", "当时的情绪是什么", "那种感受是什么"],
    "reason": ["那次心情产生的原因是什么", "当时为什么产生那种感受", "那种情绪的缘由是什么"],
    "focus": ["当时关注的方面是什么", "回忆那次，最在意什么"],
}

# Neither held-out event structure nor these question stems are sampled in training.
TEST_STEMS = {
    "person": ["回忆那件事：当时交流的对方是谁", "那次参与交流的是谁"],
    "day": ["回忆那件事：日期为哪一天", "那次事情是在第几天发生的"],
    "event": ["回忆那件事：当时发生的是什么事情", "那次发生的是哪件事情"],
    "emotion": ["回忆那件事：当时的心情是什么", "那次交流，我产生的情绪是什么"],
    "reason": ["回忆那件事：当时的情绪因为什么产生", "那次交流，为什么产生这种心情"],
    "focus": ["回忆那件事：当时关注的是什么方面", "那次交流，我的注意力放在哪里"],
}


def training_question(e, task, rng):
    stem = rng.choice(TRAIN_STEMS[task])
    return rng.choice(
        ["{anchor}。{stem}？", "{stem}？回想{anchor}。", "回忆{anchor}：{stem}？"]
    ).format(anchor=e["anchor"], stem=stem)


def training_question_grounded(e, task, rng):
    if task in ("emotion", "reason"):
        word = rng.choice(["心情", "感受", "情绪"])
        stem = (
            f"{word}是什么"
            if task == "emotion"
            else f"{word}产生的{rng.choice(['原因', '缘由'])}是什么"
        )
        stem = rng.choice(["当时", "那次", "回想起来，", "我当时的"]) + stem
    else:
        stem = rng.choice(TRAIN_STEMS[task])
    return rng.choice(
        [
            "{anchor}。{stem}？",
            "{stem}？我想回忆{anchor}。",
            "回想{anchor}，{stem}？",
            "{stem}？回忆{anchor}。",
            "回忆那次交流，{stem}？那次是{anchor}。",
        ]
    ).format(anchor=e["anchor"], stem=stem)


SECOND_TEST_STEMS = {
    "person": ["那件事发生时，和我交流的人是哪位", "回想那次，交流的对方是哪位参与者"],
    "day": ["那件事发生的那一天，是第几天", "回想那次，发生事情的时间是哪天"],
    "event": ["那次讨论发生了什么事情", "回想那次，发生的是哪件事件"],
    "emotion": ["回忆当时，我的感受是什么", "回想那次，我当时的心情如何"],
    "reason": ["回忆当时，我为什么产生那种感受", "回想那次，那种情绪产生的原因是什么"],
    "focus": ["回忆当时，我最在意什么方面", "回想那次，注意力放在什么方面"],
}


def fresh_question(e, task, view):
    if view >= 2:
        stem = SECOND_TEST_STEMS[task][view - 2]
        return f"{e['anchor']}，{stem}？" if view == 2 else f"{stem}？那件事是{e['anchor']}。"
    stem = TEST_STEMS[task][view]
    return f"{stem}？那是{e['anchor']}。" if view == 0 else f"回想{e['anchor']}，{stem}？"


def fresh_event(e, view):
    f = values(e)
    if view == 2:
        return (
            f"{f['day']}和{f['person']}交流，发生的事件是{f['event']}。"
            f"这是{e['anchor']}。当时心情是{f['emotion']}，因为{f['reason']}。"
            f"那次最在意的方面是{f['focus']}。"
        )
    if view == 3:
        return (
            f"记下那次交流：感受为{f['emotion']}，原因是{f['reason']}。"
            f"事情发生在{f['day']}，对方是{f['person']}，那件事是{f['event']}。"
            f"我关注的方面是{f['focus']}。回忆{e['anchor']}。"
        )
    if view == 0:
        return (
            f"那次交流的对方是{f['person']}，日期为{f['day']}。"
            f"我产生{f['emotion']}这种感受，因为{f['reason']}。"
            f"当时发生{f['event']}，注意力放在{f['focus']}。那是{e['anchor']}。"
        )
    return (
        f"{e['anchor']}。最在意的方面是{f['focus']}。"
        f"{f['day']}发生的事情为{f['event']}，参与交流的是{f['person']}。"
        f"产生的心情是{f['emotion']}。那种心情的缘由是{f['reason']}。"
    )
