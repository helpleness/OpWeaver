"""Pre-test development revision: compositional query training and fresh calibration."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from .generate_data import ORDINALS, PROJECTS
from .qwen_memory import save_json
from .semantic_data import INVALID, QUESTIONS, prepare

FRESH = [
    ["这份经历的人物栏对应谁？", "我要确认谈话对面的姓名。"],
    ["这件事属于哪一天的经历？", "给这段记忆标上第几天才对？"],
    ["请给出那次遭遇的具体事件。", "记录中到底是发生了哪件事？"],
    ["那一刻我的情绪是怎样的？", "我内心当时留下了何种感受？"],
    ["那份情绪在记录中有什么解释？", "什么缘故让我有那样的感觉？"],
    ["当时我最关切的领域是哪方面？", "我把关注重点放在哪个方面了？"],
]
FRESH_INVALID = {
    "unsupported": ["对方从事什么职业？", "这场交流用了多少分钟？"],
    "ambiguous": ["请一起回答谁参加以及何时发生。", "我既要事情本身，也要你的感受。"],
    "no_anchor": ["过去某次谈话的日期，你能告诉我吗？", "找一个我曾经关注的事情。"],
    "false_premise": ["那位参与者是小何，对吧？", "你是否认同当时我很紧张？"],
}

FINAL = [
    ["这回是我和哪一位交谈的？", "请指出与我交流的人的姓名。"],
    ["事情出现在记录中的哪一天？", "这次经历发生的日子是第几天？"],
    ["请讲出那一回发生的事情。", "回忆描述的具体事件是什么？"],
    ["事情发生的时候我是什么感受？", "那时我内心是什么情绪？"],
    ["是什么造成了我的那种感受？", "请讲出当时这份心情的原因。"],
    ["我那时候在意的是哪个方面？", "请指出当时最让我关注的方面。"],
]
FINAL_INVALID = {
    "unsupported": ["这次见面的详细位置是什么？", "参与者的职业经历是什么？"],
    "ambiguous": ["发生日期和关注的方面，各是什么？", "是谁参与，以及我为什么有那种感受？"],
    "no_anchor": ["请回想以前一次交流的对象。", "能否告诉我有一天发生的事情？"],
    "false_premise": ["那时是小杨在场吗？", "请判断当时我是否高兴。"],
}


def main(args):
    fresh = FINAL if args.final_calibration else FRESH
    fresh_invalid = FINAL_INVALID if args.final_calibration else FRESH_INVALID
    calibration_seed = 9207 if args.final_calibration else 9205
    QUESTIONS["calibration_fresh"] = fresh
    INVALID["calibration_fresh"] = fresh_invalid
    prepare(args.out, calibration_seed=calibration_seed, calibration_family="calibration_fresh")
    data = json.loads((args.source / "train.json").read_text(encoding="utf-8"))
    rng = random.Random(9206)
    queries = []
    field_names = ["人物姓名", "发生日期", "具体事件", "我的情绪", "情绪的缘由", "注意力所在"]
    unsupported = [
        "对方的年龄是多少？",
        "对方的住址在哪里？",
        "参与者的工作单位是什么？",
        "这个人的联系方式是什么？",
        "那天是星期几？",
        "是上午还是下午发生的？",
        "讨论持续了多久？",
        "现场的温度是多少？",
        "后来还会发生什么？",
        "如果重来我应该怎么做？",
        "对方为什么会这样做？",
        "这件事对社会有什么意义？",
        "对方当时有什么情绪？",
        "其他人是不是也很紧张？",
        "请猜测我没有记录的想法。",
        "我的感受有多强烈？",
        "当时花费的金额是多少？",
        "我为什么关注那件事？",
        "下一次计划在哪见面？",
        "我关心的事情最后成功了吗？",
    ]
    wrappers = ["{a}，{q}", "关于{a}，{q}", "请回忆{a}。{q}", "在{a}这段记忆里，{q}"]
    for i, e in enumerate(data["events"][::2]):
        addr = [PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"])]

        def add(body, field, anchored=True):
            text = rng.choice(wrappers).format(a=e["anchor"], q=body) if anchored else body
            queries.append(
                dict(text=text, field=field, event=i * 2, address=addr if anchored else None)
            )

        for field in range(6):
            body = rng.choice(QUESTIONS["train"][field])
            if rng.random() < 0.5:
                other = rng.choice([k for k in range(6) if k != field])
                body = "不是想问" + field_names[other] + "，" + body
            add(body, field)
        add(rng.choice(unsupported), 6)
        a, b = rng.sample(range(6), 2)
        add(
            "请分别说明："
            + rng.choice(QUESTIONS["train"][a][:2])
            + rng.choice(QUESTIONS["train"][b][:2]),
            6,
        )
        kind = rng.choice(["no_anchor", "false_premise"])
        add(rng.choice(INVALID["train"][kind]), 6, kind != "no_anchor")
    data["queries"] = queries
    save_json(args.out / "train.json", data)
    shutil.copyfile(args.source / "test.json", args.out / "test.json")
    save_json(
        args.out / "manifest.json",
        dict(
            seeds=dict(train_queries=9206, calibration=calibration_seed, test=9203),
            train_events=len(data["events"]),
            train_queries=len(queries),
            development_set=str((args.source / "calibration.json").resolve()),
            changes="Canonical field query at reader; compositional single/multi-field and unsupported intents; old calibration is development-only",
            fresh_calibration_questions=fresh,
            fresh_calibration_invalid=fresh_invalid,
            unsupported_training=unsupported,
            sha256={
                name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                for name in ["train.json", "calibration.json", "test.json"]
            },
        ),
    )
    assert (args.out / "test.json").read_bytes() == (args.source / "test.json").read_bytes()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/semantic_suite_01")
    )
    p.add_argument(
        "--out", type=Path, default=Path("experiments/episodic_memory/runs/semantic_suite_02")
    )
    p.add_argument("--final-calibration", action="store_true")
    main(p.parse_args())
