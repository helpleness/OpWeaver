"""Compositional training language; diagnostic template 2 and holdout 3 stay excluded."""

from __future__ import annotations

import random

from .generate_data import FIELDS, TASKS

EVENT_CLAUSES = {
    "person": ["和{person}交流", "参与者是{person}", "对方是{person}", "那位交流的人是{person}"],
    "day": ["时间是{day}", "日期为{day}", "那是{day}", "事情发生在{day}"],
    "event": ["发生{event}", "那件事是{event}", "发生的事情为{event}", "事情是{event}"],
    "emotion": [
        "感受是{emotion}",
        "当时心情为{emotion}",
        "我觉得{emotion}",
        "产生的情绪是{emotion}",
    ],
    "reason": [
        "原因是{reason}",
        "这种感受的缘由是{reason}",
        "因为{reason}",
        "产生评价是因为{reason}",
    ],
    "focus": ["关注{focus}", "我在意{focus}", "注意力放在{focus}", "最在意的方面是{focus}"],
}
QUESTION_STEMS = {
    "person": ["和谁交流", "参与者是谁", "对方是哪位", "交流的人是谁", "那位参与者是谁"],
    "day": ["日期是什么", "发生在第几天", "那次是哪一天", "事情的时间是什么", "事情是哪天发生的"],
    "event": [
        "发生的事情是哪件",
        "那件事是什么",
        "当时发生了什么",
        "发生的事是什么",
        "那次有什么事情",
    ],
    "emotion": ["当时心情如何", "产生的感受是什么", "那次有什么情绪", "心情是什么", "当时感觉如何"],
    "reason": [
        "产生评价的缘由是什么",
        "为什么有这种感受",
        "产生感受的原因是什么",
        "为什么产生这种心情",
        "情绪的缘由是什么",
    ],
    "focus": [
        "最在意的方面是什么",
        "注意力放在哪里",
        "当时最关注什么",
        "在意什么方面",
        "那次关注的是什么",
    ],
}
WRAPPERS = [
    "{anchor}，{stem}？",
    "回想{anchor}：{stem}？",
    "{stem}？回忆{anchor}。",
    "我想回忆{anchor}，{stem}？",
    "{anchor}，回想起来，{stem}？",
]

# Entire sentence structures below are never sampled by training augmentation.
HELDOUT_QUESTIONS = {
    "person": "和我交流的那位是谁？",
    "day": "那件事情的日期是哪一天？",
    "event": "那次交流发生的是哪件事？",
    "emotion": "那次产生的心情是什么？",
    "reason": "那种心情产生的缘由是什么？",
    "focus": "我当时在意的是什么方面？",
}


def values(e):
    return {t: FIELDS[t][e["values"][t]] for t in TASKS}


def augmented_event(e, rng: random.Random):
    f = values(e)
    clauses = [rng.choice(EVENT_CLAUSES[t]).format(**f) for t in TASKS]
    rng.shuffle(clauses)
    clauses.insert(rng.randrange(len(clauses) + 1), rng.choice(["记下", "回忆", ""]) + e["anchor"])
    return "。".join(clauses) + "。"


def augmented_question(e, task, rng: random.Random):
    from .generate_data import question

    forbidden = {question(e, task, 2), heldout_question(e, task)}
    while True:
        text = rng.choice(WRAPPERS).format(
            anchor=e["anchor"], stem=rng.choice(QUESTION_STEMS[task])
        )
        if text not in forbidden:
            return text


def heldout_event(e):
    f = values(e)
    return (
        f"那件事情是{f['event']}。我在{f['day']}和{f['person']}交流。"
        f"当时的感受为{f['emotion']}，产生这种感受的缘由是{f['reason']}。"
        f"我最在意的是{f['focus']}。回想起来，那是{e['anchor']}。"
    )


def heldout_question(e, task):
    return e["anchor"] + "，回忆那次交流：" + HELDOUT_QUESTIONS[task]


def vocabulary_text(probe):
    # Declared symbols only; exposing characters is not exposing test sentences as examples.
    f = values(probe)
    return (
        "".join(s.format(**f) for options in EVENT_CLAUSES.values() for s in options)
        + "".join(
            s.format(anchor=probe["anchor"], stem=q)
            for s in WRAPPERS
            for qs in QUESTION_STEMS.values()
            for q in qs
        )
        + heldout_event(probe)
        + "".join(heldout_question(probe, task) for task in TASKS)
    )
