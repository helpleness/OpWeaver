"""Versioned, disjoint language families for semantic memory and abstention tests."""

import argparse
import copy
import hashlib
import random
from pathlib import Path

from .generate_data import FIELDS, ORDINALS, PROJECTS, TASKS
from .qwen_memory import save_json

QUESTIONS = {
    "train": [
        [
            "当时和谁交流？",
            "另一位参与者叫什么？",
            "这段经历中的对方是哪位？",
            "只回答人物，不用说事情。",
            "谈话对象的名字是什么？",
            "我是在跟哪个人说话？",
        ],
        [
            "是哪一天发生的？",
            "发生在第几天？",
            "只说日期，不说讨论次数。",
            "这件事的时间是哪天？",
            "记录的发生日是哪一天？",
            "我想知道事情发生的天数。",
        ],
        [
            "当时发生了什么？",
            "具体是哪件事？",
            "只说事情，不用说参与者。",
            "那次的事件内容是什么？",
            "发生的事情本身是什么？",
            "请描述那件事，不要描述心情。",
        ],
        [
            "当时是什么心情？",
            "我有怎样的感受？",
            "只说情绪，不说原因。",
            "那时的情绪状态是什么？",
            "我当时感觉如何？",
            "请描述我那时的心情。",
        ],
        [
            "为什么产生那种感受？",
            "这种情绪的原因是什么？",
            "只说缘由，不要说情绪名称。",
            "我为何会这样想？",
            "是什么理由导致那种心情？",
            "请说明当时感受背后的原因。",
        ],
        [
            "我当时最在意什么？",
            "注意力主要放在哪方面？",
            "只说关注点，不说事情。",
            "当时我最关注什么？",
            "哪方面最吸引我的注意？",
            "请说出我关心的方面。",
        ],
    ],
    "calibration": [
        ["请补全这次交流对象的姓名。", "这次谈话是我与哪位进行的？"],
        ["这段经历应当归到第几天？", "我需要给这件事标注日期，应填哪一天？"],
        ["如果给事情本身做个概括，该写什么？", "这一回究竟出现了什么情况？"],
        ["我那时内心的情绪可以用什么词形容？", "此事发生时我的心理感受是什么？"],
        ["这份感受是由什么引起的？", "当时产生这种感受的理由记录了什么？"],
        ["我最看重的方面在记录中是什么？", "这次经历中我的关注对象是什么方面？"],
    ],
    "test": [
        [
            "我要找的是那位与我交谈的人的名字。",
            "先不管谈了什么，和我对话的是谁？",
            "这次互动的另一方叫什么名字？",
        ],
        [
            "若沿时间线定位此事，它落在第几天？",
            "不问这是第几次讨论，我要问实际发生在哪天。",
            "请从这段回忆中找出对应的发生日。",
        ],
        [
            "剥离心情和理由后，留下的那件事是什么？",
            "这段回忆说的是哪件具体事情？",
            "不要列人名，告诉我实际出了什么事。",
        ],
        [
            "暂且不谈缘由，那会儿我的情绪是什么？",
            "请提取我经历这件事时的内心感受。",
            "这件事对应的心情记录是什么？",
        ],
        [
            "我问的不是情绪的名字，而是它从何而来。",
            "追溯那份心情，记录把它归因于什么？",
            "这次感受背后的缘故，在经历里是怎么记的？",
        ],
        [
            "当时究竟是哪一方面占据了我的注意？",
            "事情和感受之外，我的着眼点落在何处？",
            "回忆中我优先在意的那个方面是什么？",
        ],
    ],
}

INVALID = {
    "train": {
        "unsupported": [
            "花了多少钱？",
            "对方的电话号码是什么？",
            "地点的街道地址是什么？",
            "对方多大了？",
            "天气几度？",
            "是否发生了另外一件事？",
        ],
        "ambiguous": [
            "把人物和原因一起告诉我。",
            "请把整段经历全部描述一下。",
            "那件事的时间和感受分别是什么？",
            "日期、人物和关注点都是什么？",
        ],
        "no_anchor": [
            "上次和谁交流？",
            "那天为什么难过？",
            "我以前发生过什么？",
            "那一次我关心什么？",
        ],
        "false_premise": [
            "是不是和小林交谈的？",
            "能否确认那时发生了收到礼物？",
            "当时一定很高兴吧？",
        ],
    },
    "calibration": {
        "unsupported": ["对方出生在哪个城市？", "这场交谈持续几分钟？"],
        "ambiguous": ["请一并给出事情与发生日期。", "我想一次知道心情及其原因。"],
        "no_anchor": ["帮我想想从前那个人的姓名。", "之前的事情在哪一天？"],
        "false_premise": ["那是不是小周参加的交流？", "你能证明当时我很平静吗？"],
    },
    "test": {
        "unsupported": [
            "见面时对方穿什么颜色的衣服？",
            "这次活动的预算数额是多少？",
            "现场究竟有几个人？",
        ],
        "ambiguous": [
            "把交流对象与注意力所在都列给我。",
            "发生什么、为何有这种感觉，请同时说明。",
            "请给我这段经历各方面的完整概览。",
        ],
        "no_anchor": [
            "你记得我某次的感受吗？",
            "能找到我以前某件事的理由吗？",
            "当年有一个人，你记得名字吗？",
        ],
        "false_premise": [
            "我记得那回是小陈，你同意吗？",
            "请回答是或否：那次我是不是安心？",
            "那回肯定是解决故障了，对不对？",
        ],
    },
}

ANSWERS = [
    ["那次和{}交流。", "对方是{}。", "交流对象是{}。", "参与者是{}。", "那个人叫{}。", "{}。"],
    ["那件事发生在{}。", "时间是{}。", "发生日期是{}。", "记录的是{}。", "那是在{}。", "{}。"],
    [
        "那次的事情是{}。",
        "发生的事情是{}。",
        "事件是{}。",
        "那次发生了{}。",
        "记录的事件为{}。",
        "{}。",
    ],
    ["当时感到{}。", "我的心情是{}。", "当时的感受是{}。", "那时我很{}。", "情绪是{}。", "{}。"],
    ["因为{}。", "原因是{}。", "缘由是{}。", "那是由于{}。", "记录的理由是{}。", "{}。"],
    ["当时关注{}。", "我在意的是{}。", "关注点是{}。", "我主要关注{}。", "注意力放在{}。", "{}。"],
]


def question(e, field, split, rng):
    body = rng.choice(QUESTIONS[split][field])
    if split == "train":
        return rng.choice(["{a}，{q}", "关于{a}，{q}", "请回忆{a}。{q}"]).format(
            a=e["anchor"], q=body
        )
    return e["anchor"] + "，" + body


def render(e, split, rng):
    f = {t: FIELDS[t][e["values"][t]] for t in TASKS if e["values"][t] >= 0}
    formats = {
        "train": [
            ["交流对象是{}", "时间为{}", "事情是{}", "我的感受是{}", "感受的原因是{}", "我关注{}"],
            [
                "我与{}交流",
                "那是在{}",
                "当时发生{}",
                "我觉得{}",
                "因为{}才有这种心情",
                "在意的方面是{}",
            ],
            ["对方叫{}", "发生在{}", "事件记录：{}", "心情：{}", "原因：{}", "注意力在{}"],
            [
                "参与者为{}",
                "日期是{}",
                "具体事情为{}",
                "当时的情绪为{}",
                "这种感受源于{}",
                "我的关注点为{}",
            ],
        ],
        "calibration": [
            [
                "谈话对象：{}",
                "这件事在{}发生",
                "经历中的事情是{}",
                "当时感觉{}",
                "情绪背后的理由是{}",
                "我主要在意{}",
            ]
        ],
        "test": [
            [
                "与我交谈的人是{}",
                "这段回忆对应{}",
                "这次经历发生的事为{}",
                "我的内心感受为{}",
                "产生这份感受是由于{}",
                "我把注意力放在{}",
            ]
        ],
    }
    parts = [rng.choice(formats[split])[i].format(f[t]) for i, t in enumerate(TASKS) if t in f]
    rng.shuffle(parts)
    return e["anchor"] + "。" + "。".join(parts) + "。"


def event(rng, eid, owner, address=None):
    p, o = address or (rng.randrange(16), rng.randrange(16))
    return dict(
        id=eid,
        owner=owner,
        timestamp=0,
        project=PROJECTS[p],
        ordinal=ORDINALS[o],
        anchor=f"{PROJECTS[p]}项目{ORDINALS[o]}讨论",
        values={t: rng.randrange(len(FIELDS[t])) for t in TASKS},
    )


def prepare(out, calibration_seed=9202, calibration_family="calibration"):
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(9201)
    events, queries = [], []
    for i in range(720):
        e = event(rng, f"train-{i}", f"train-{i // 24}")
        if i % 3 == 0:
            for task in rng.sample(TASKS, rng.randint(1, 3)):
                e["values"][task] = -1
        for view in range(2):
            v = copy.deepcopy(e)
            v["id"] += f"-v{view}"
            v["text"] = render(v, "train", rng)
            events.append(v)
        for field in range(6):
            queries.append(
                dict(
                    text=question(e, field, "train", rng),
                    field=field,
                    event=i * 2,
                    address=[PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"])],
                )
            )
        kind = list(INVALID["train"])[i % 4]
        text = rng.choice(INVALID["train"][kind])
        queries.append(
            dict(
                text=text if kind == "no_anchor" else e["anchor"] + "，" + text,
                field=6,
                event=i * 2,
                address=None
                if kind == "no_anchor"
                else [PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"])],
            )
        )
    save_json(out / "train.json", dict(events=events, queries=queries))
    for split, seed, repeats in [("calibration", calibration_seed, 8), ("test", 9203, 16)]:
        family = calibration_family if split == "calibration" else split
        rng = random.Random(seed)
        cases = []
        kinds = [
            "normal",
            "counterfactual",
            "partial_known",
            "duplicate",
            "empty",
            "wrong_address",
            "wrong_owner",
            "missing_field",
            "unsupported",
            "ambiguous",
            "no_anchor",
            "false_premise",
            "conflict",
            "nonexistent",
            "damaged",
        ]
        for kind in kinds:
            for j in range(repeats):
                owner = f"{split}-{kind}-{j}"
                e = event(rng, owner + "-target", owner)
                field = j % 6
                task = TASKS[field]
                q = question(e, field, family, rng)
                records = []
                occupied = {(PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"]))}
                for d in range(3):
                    while True:
                        address = (rng.randrange(16), rng.randrange(16))
                        if address not in occupied:
                            occupied.add(address)
                            break
                    records.append(event(rng, owner + f"-d{d}", owner, address))
                expected = FIELDS[task][e["values"][task]]
                if kind == "counterfactual":
                    for t in TASKS:
                        e["values"][t] = (e["values"][t] + rng.randrange(1, len(FIELDS[t]))) % len(
                            FIELDS[t]
                        )
                    expected = FIELDS[task][e["values"][task]]
                if kind == "partial_known":
                    for t in rng.sample([t for t in TASKS if t != task], 2):
                        e["values"][t] = -1
                if kind == "missing_field":
                    e["values"][task] = -1
                if kind == "wrong_owner":
                    e["owner"] = owner + "-other"
                if kind == "wrong_address":
                    o = (ORDINALS.index(e["ordinal"]) + 1) % 16
                    e["ordinal"] = ORDINALS[o]
                    e["anchor"] = f"{e['project']}项目{e['ordinal']}讨论"
                if kind in INVALID[split]:
                    body = rng.choice(INVALID[family][kind])
                    q = body if kind == "no_anchor" else e["anchor"] + "，" + body
                if kind not in ("empty", "nonexistent"):
                    records.append(e)
                if kind == "empty":
                    records = []
                if kind in ("conflict", "duplicate"):
                    other = copy.deepcopy(e)
                    other["id"] += "-copy"
                    if kind == "conflict":
                        other["values"][task] = (other["values"][task] + 1) % len(FIELDS[task])
                    records.append(other)
                for record in records:
                    record["text"] = render(record, split, rng)
                cases.append(
                    dict(
                        id=owner,
                        kind=kind,
                        owner=owner,
                        field=field,
                        question=q,
                        events=records,
                        expected=expected
                        if kind in ("normal", "counterfactual", "partial_known", "duplicate")
                        else None,
                        damage_id=e["id"] if kind == "damaged" else None,
                    )
                )
        save_json(out / f"{split}.json", cases)
    save_json(
        out / "manifest.json",
        dict(
            seeds=[9201, 9202, 9203],
            train_events=len(events),
            train_queries=len(queries),
            question_families=QUESTIONS,
            invalid_families=INVALID,
            answer_styles=ANSWERS,
            sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.json")},
            note="Frozen before training; split-specific question/event families. Six finite-vocabulary fields; multi-field/yes-no requests are explicitly unsupported.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    prepare(p.parse_args().out)
