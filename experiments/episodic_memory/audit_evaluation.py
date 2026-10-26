"""Explicit, non-blind assistant audit of selected SAVED outputs, not a judge model.

Case selection targets known scoring failures and cannot estimate population rates.
Every label below was read against the source record. Exact response assertions
prevent these case-specific decisions being applied to a new model response.
"""

import argparse
from collections import Counter
from pathlib import Path

from .semantic_review import read_json, read_rows, summarize, write_json, write_rows

# (condition, case, exact response, behavior, target, instructions, format,
#  rationale, [(claim quote, verdict, evidence quote or None)])
AUDIT = [
    (
        "task_no_context",
        "test-counterfactual-6",
        "交流对象姓名：未知",
        "abstain",
        "not_answered",
        "no",
        "no",
        "没有获得任何经历证据；未知是合理表达不确定，违反固定措辞并非事实错误。",
        [],
    ),
    (
        "task_no_context",
        "test-wrong_owner-6",
        "交流对象姓名：没有提供\n\n我在第16次讨论中与您进行了对话。",
        "mixed",
        "not_applicable",
        "no",
        "no",
        "先说未提供姓名，又无依据地断言自己参与该次对话；不能只看到拒答词就算合理拒答。",
        [("我在第16次讨论中与您进行了对话。", "unsupported", None)],
    ),
    (
        "task_full_text",
        "test-ambiguous-8",
        "交流对象：小陈\n注意力所在：自身表现",
        "answer",
        "correct",
        "no",
        "no",
        "两项事实均正确。历史提示禁止多字段，所以是范围约定违规，不是事实错误。",
        [
            ("交流对象：小陈", "supported", "与我交谈的人是小陈。"),
            ("注意力所在：自身表现", "supported", "我把注意力放在自身表现。"),
        ],
    ),
    (
        "task_full_text",
        "test-ambiguous-3",
        "发生在第8天。产生这份感受是由于问题解决。我的内心感受为尴尬。我把注意力放在具体细节。",
        "answer",
        "partial",
        "no",
        "no",
        "四项事实都有依据，但完整概览漏了人物和事件。完整性与真实性应分开。",
        [
            ("发生在第8天。", "supported", "这段回忆对应第8天。"),
            ("产生这份感受是由于问题解决。", "supported", "产生这份感受是由于问题解决。"),
            ("我的内心感受为尴尬。", "supported", "我的内心感受为尴尬。"),
            ("我把注意力放在具体细节。", "supported", "我把注意力放在具体细节。"),
        ],
    ),
    (
        "task_full_text",
        "test-false_premise-5",
        "否",
        "answer",
        "correct",
        "no",
        "no",
        "问是否安心，记录为紧张，否是有依据的答案。历史范围约定要求拒答，不能据此判事实错。",
        [("否", "supported", "我的内心感受为紧张。")],
    ),
    (
        "task_full_text",
        "test-false_premise-4",
        "是",
        "answer",
        "incorrect",
        "no",
        "no",
        "问是否安心，记录为生气，肯定回答与证据冲突。这是事实错误，不是措辞问题。",
        [("是", "contradicted", "我的内心感受为生气。")],
    ),
    (
        "task_full_text",
        "test-counterfactual-14",
        "这段回忆中提到的“这件事”是指花园项目第3次讨论中提出的建议。",
        "answer",
        "correct",
        "yes",
        "yes",
        "提出的建议表达了记录中的提出建议；没有新增建议内容或行为者。",
        [("花园项目第3次讨论中提出的建议", "supported", "这次经历发生的事为提出建议。")],
    ),
    (
        "task_full_text",
        "test-duplicate-5",
        "我在地图项目第16次讨论中把注意力放在了他人的反应上。",
        "answer",
        "correct",
        "yes",
        "yes",
        "他人的反应是他人反应的正常语义改写，不能因插入的字而判错。",
        [("把注意力放在了他人的反应上", "supported", "我把注意力放在他人反应。")],
    ),
    (
        "task_full_text",
        "test-normal-15",
        "我在花园项目第14次讨论中，把注意力放在时间安排上，因此我的内心感受是生气。",
        "answer",
        "correct",
        "no",
        "no",
        "目标情绪正确，关注点也正确，但因此新增了证据未给出的因果关系；整句不能算完全有依据。",
        [
            ("把注意力放在时间安排上", "supported", "我把注意力放在时间安排。"),
            ("我的内心感受是生气", "supported", "我的内心感受为生气。"),
            ("把注意力放在时间安排上，因此我的内心感受是生气", "unsupported", None),
        ],
    ),
]

RAW_CORRECT = {
    "test-normal-11": ("当时最在意的是自身表现。", "我把注意力放在自身表现。"),
    "test-counterfactual-11": ("当时最在意的是对方需求。", "我把注意力放在对方需求。"),
    "test-partial_known-5": ("我关注具体细节。", "我把注意力放在具体细节。"),
    "test-duplicate-11": ("我最在意的是环境变化。", "我把注意力放在环境变化。"),
}


def main(out):
    packets = read_rows(out / "packets.jsonl")
    mapping = read_json(out / "private_mapping.json")
    lookup = {
        (mapping[p["review_id"]]["condition"], mapping[p["review_id"]]["case_id"]): p
        for p in packets
    }
    reviews = []
    entries = AUDIT + [
        (
            "memory_final",
            case,
            "我没有这段经历的记忆。",
            "abstain",
            "not_answered",
            "not_assessed",
            "not_assessed",
            "库中有目标事实，最终用户只收到控制器拒答。原始生成的正确内容没有交付，不能把最终回答改判正确。",
            [],
        )
        for case in RAW_CORRECT
    ]
    for mode, case, response, behavior, target, instruction, fmt, why, claims in entries:
        p = lookup[mode, case]
        if p["response"] != response:
            raise ValueError(f"Saved output changed: {mode}/{case}")
        reviews.append(
            dict(
                review_id=p["review_id"],
                binding_sha256=p["binding_sha256"],
                reviewer="assistant_in_this_thread_nonblind_not_independent",
                selection_reason="Purposive audit of known scoring/censoring failures, not random sampling",
                behavior=behavior,
                target=target,
                instruction_compliance=instruction,
                format_compliance=fmt,
                rationale=why,
                all_claims_checked=True,
                claims=[
                    dict(
                        quote=q,
                        verdict=v,
                        evidence=[dict(id=case + "-target", quote=e)] if e else [],
                    )
                    for q, v, e in claims
                ],
            )
        )
    write_rows(out / "reviews.jsonl", reviews)
    write_json(out / "summary.json", summarize(packets, mapping, reviews))
    source = Path(mapping[lookup["memory_final", "test-normal-11"]["review_id"]]["source"])
    rows = read_rows(source)
    diagnostics = []
    for row in rows:
        if row["id"] not in RAW_CORRECT:
            continue
        raw, quote = RAW_CORRECT[row["id"]]
        if row["result"]["raw"] != raw:
            raise ValueError("Changed raw generation")
        p = lookup["memory_final", row["id"]]
        assert any(quote in e["text"] for e in p["evidence"])
        diagnostics.append(
            dict(
                case_id=row["id"],
                raw=raw,
                final=row["result"]["answer"],
                evidence_quote=quote,
                diagnosis="Correct fact blocked by exact-template gate",
            )
        )
    write_json(
        out / "controller_audit.json",
        dict(
            source=str(source),
            total=len(rows),
            counts=dict(
                Counter(
                    "generation_delivered"
                    if r["result"]["eligible"]
                    else "generated_then_blocked"
                    if "token_min" in r["result"]
                    else "blocked_before_generation"
                    for r in rows
                )
            ),
            inference_basis="Historical ask_checked writes token_min only after generate_routed returns",
            rejected_correct_generations=diagnostics,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out)
