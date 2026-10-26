"""Recorded post-test semantic annotations, separate from the fixed strict scores.

Annotations are supplied by the reviewing assistant after inspecting saved outputs.
This file never trains, changes a prediction, or changes the headline scoring rule.
"""

import argparse
import json
from pathlib import Path

from .qwen_memory import save_json

ANNOTATIONS = {
    "qwen_0_5b": {
        (83, "text"): ("equivalent_paraphrase", "害怕被批评表达担心评价；原严格短语匹配未识别。"),
        (117, "text"): ("equivalent_paraphrase", "注意自己表现与关注自身表现一致。"),
        (95, "fields"): ("equivalent_paraphrase", "时间不够与时间不足一致。"),
        (59, "text"): ("ambiguous", "回答担心杯子被弄坏，未清楚区分担忧与已经发生的弄坏事件。"),
        (41, "fields"): (
            "target_paraphrase_with_additions",
            "包含收到礼物的意思，但增加了人物归属和不自在等描述。",
        ),
        (49, "fields"): (
            "target_paraphrase_with_additions",
            "包含收到礼物，但新增参观、第一次、询问他人等未经支持的内容，且输出截断。",
        ),
        (53, "fields"): (
            "target_paraphrase_with_additions",
            "提出想法与建议接近，但大量具体花园设计细节不在事实中，且输出截断。",
        ),
        (57, "fields"): (
            "target_paraphrase_with_additions",
            "包含收到礼物，但添加赠送者等未经支持的细节，且输出截断。",
        ),
        (3, "fields"): ("refusal_paraphrase", "没有这个信息属于拒答，并非给出错误人物。"),
        (75, "fields"): (
            "refusal_paraphrase",
            "没有这段经历的记忆少了我字，严格拒答字符串未匹配。",
        ),
        (81, "text"): ("refusal_paraphrase", "没有这段经历的记忆少了我字。"),
        (81, "fields"): ("refusal_paraphrase", "没有这段经历的记忆少了我字。"),
        (89, "text"): ("refusal_paraphrase", "没有这段经历的记忆少了我字。"),
        (89, "fields"): ("refusal_paraphrase", "明确表示无法提供答案，属于拒答表述。"),
        (81, "memory"): ("wrong_field", "问原因，回答情绪生气。"),
        (85, "memory"): ("wrong_field", "问原因，回答情绪生气。"),
        (89, "memory"): ("wrong_field", "问原因，回答情绪好奇。"),
    },
    "qwen_1_5b": {
        (39, "text"): ("equivalent_paraphrase", "第十天与第10天一致。"),
        (59, "text"): ("equivalent_paraphrase", "弄坏了杯子与弄坏杯子一致。"),
        (83, "text"): ("equivalent_paraphrase", "担心被评价与担心评价一致。"),
        (91, "text"): ("equivalent_paraphrase", "获得了支持与获得支持一致。"),
        (45, "fields"): ("equivalent_paraphrase", "叫错了名字与叫错名字一致。"),
        (55, "fields"): ("equivalent_paraphrase", "叫错了名字与叫错名字一致。"),
        (113, "fields"): ("equivalent_paraphrase", "对方的需求与对方需求一致。"),
        (115, "fields"): ("equivalent_paraphrase", "对方的需求与对方需求一致。"),
        (41, "text"): (
            "target_paraphrase_with_additions",
            "有收到礼物的意思，但赠送者小林并非记录明确支持的事实。",
        ),
        (53, "text"): (
            "target_paraphrase_with_additions",
            "有提出建议的意思，但把自己的感受扩展成我们对建议的感受。",
        ),
        (53, "fields"): (
            "target_paraphrase_with_additions",
            "有提出想法的意思，但新奇想法、团队旧方案等细节没有支持，且截断。",
        ),
        (55, "text"): (
            "target_paraphrase_with_additions",
            "有叫错名字的意思，但将自己的状态扩大成双方都冷静，且截断。",
        ),
        (57, "text"): ("target_paraphrase_with_additions", "有收到礼物的意思，但新增赠送者。"),
        (99, "text"): (
            "target_paraphrase_with_additions",
            "有没有事先准备的意思，但额外添加与预期有出入的解释。",
        ),
        (109, "text"): (
            "target_paraphrase_with_additions",
            "提到自己的表现，但把关注范围进一步限定到解决问题。",
        ),
        (113, "text"): ("ambiguous", "提到对方的需求，也把发生的事件说成关注点，存在混合回答。"),
        (117, "text"): ("ambiguous", "表达方式接近自身表现，但新增沟通疏漏的解释且未完整结束。"),
        (53, "memory"): ("wrong_field", "问事件，回答提出任务结果，混入关注点并未读出提出建议。"),
        (33, "empty"): (
            "unsupported_guess",
            "空记忆下猜测第12天；原始置信度0.3172，最终门控拒答。",
        ),
    },
}

EXPECTED_ADAPTERS = {
    "qwen_0_5b": "642d3ff3ca06961218b06a68620ef43d79f50d57d0b8d5560e1814031e421d03",
    "qwen_1_5b": "f0833be5ced216178c9b0302b933755277b2d08ad2ebf9d6b8fdf8e8962d5e3c",
}


def main(suite):
    audits = {}
    for name, entries in ANNOTATIONS.items():
        source = suite / name / "results.jsonl"
        if not source.exists():
            continue
        meta = json.loads((suite / name / "summary.json").read_text(encoding="utf-8"))
        if (
            meta["adapter_sha256"] != EXPECTED_ADAPTERS[name]
            or meta["selection_sha256"]
            != "31a583a37d05c9d993ec1e63291b602e0ce20361f3923ff85b601a495a73db08"
        ):
            raise ValueError(
                "These post-test annotations only apply to the recorded checkpoints and question set"
            )
        rows = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()]
        annotated = []
        for (index, mode), (category, note) in entries.items():
            r = rows[index]
            assert r["wording"] == "novel" and r[mode]["raw_status"] == "wrong"
            annotated.append(
                dict(
                    row_index=index,
                    id=r["id"],
                    question=r["question"],
                    task=r["task"],
                    mode=mode,
                    expected=r["expected"],
                    raw=r[mode]["raw"],
                    category=category,
                    note=note,
                )
            )
        audits[name] = annotated
    save_json(
        suite / "semantic_audit.json",
        dict(
            note="Assistant review of saved outputs, not independent blinded human evaluation. Exceptions and representative failures only; original strict scores unchanged. Correct target paraphrase does not imply all generated details are supported.",
            models=audits,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite",
        type=Path,
        default=Path("experiments/episodic_memory/runs/qwen_scale_comparison_01"),
    )
    main(p.parse_args().suite)
