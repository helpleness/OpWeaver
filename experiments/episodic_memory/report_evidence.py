"""Measured rejection diagnostics, paired comparisons and unsupported-answer checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(runs, comparison, candidate, output):
    data = read(comparison / "comparison.json")
    diagnosis = read(runs / "abstention_diagnosis.json")
    calibration = read(candidate / "calibration.json")
    names = {
        "old_cluster": "旧模型",
        "old_retry": "旧模型+重试",
        "mixed_joint_retry": "混合库联合分类",
        "verified_retry": "首版证据头（失败）",
        "semantic_verified_retry": "语义证据验证",
    }
    labels = {
        "original": "原句",
        "question_only": "只换提问",
        "event_only": "只换事件叙述",
        "both": "两者都换",
    }
    lines = [
        "# 有记忆却拒答：本机对比实验",
        "",
        "使用同一组600个问题：480个已存事件、120个未存事件。所有编码器、地址和已有磁盘记忆保持一致；只训练读取器和验证器。模板3已经被用于诊断，以下是回归对比，不是全新句式成绩。",
        "",
        "## 旧模型的104次UNKNOWN是什么",
        "",
        "| 条件 | 没检索到 | 正确条件答案被拒答压过 | 候选干扰 | 单独给正确事件仍没读对 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        c = diagnosis["conditions"][key]["counts"]
        lines.append(
            f"| {label} | {c.get('retrieval_miss', 0)} | {c.get('correct_conditional_answer_vetoed', 0)} | {c.get('candidate_interference', 0)} | {c.get('reading_or_encoding_error', 0)} |"
        )
    lines += [
        "",
        "分类是有真值辅助的诊断；只能提示失败位置，不能据此直接取消拒答。读取/编码类别也可能包含问题意图解释错误。",
        "",
        "## 已存事件答对率",
        "",
        "| 条件 | " + " | ".join(names.values()) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for key, label in labels.items():
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{data['conditions'][key][n]['present_accuracy']:.2%}" for n in names)
            + " |"
        )
    lines += [
        "",
        "## 两者都换：拒答、错答与召回",
        "",
        "| 模型 | 答对/480 | UNKNOWN/480 | 错误非UNKNOWN/480 | 找回支持事件/480 | 未存事件错误作答/120 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n, label in names.items():
        c = data["conditions"]["both"][n]["counts"]
        lines.append(
            f"| {label} | {c.get('present_correct', 0)} | {c.get('present_unknown', 0)} | {c.get('wrong_nonunknown', 0)} | {c.get('support_found', 0)} | {c.get('unsupported_false_answer', 0)} |"
        )
    transition = {
        a: {b: 0 for b in ("答对", "UNKNOWN", "错答")} for a in ("答对", "UNKNOWN", "错答")
    }
    old_rows = data["conditions"]["both"]["old_cluster"]["records"]
    new_rows = data["conditions"]["both"]["semantic_verified_retry"]["records"]

    def status(row):
        return (
            "答对"
            if row["prediction"] == row["target"]
            else "UNKNOWN"
            if row["prediction"] == 0
            else "错答"
        )

    for old, new in zip(old_rows, new_rows):
        assert (old["id"], old["task"], old["target"]) == (new["id"], new["task"], new["target"])
        if old["supported"]:
            transition[status(old)][status(new)] += 1
    lines += [
        "",
        "同一问题的变化（仅已存事件）：",
        "",
        "| 旧结果→新结果 | 答对 | UNKNOWN | 错答 |",
        "|---|---:|---:|---:|",
    ]
    for label, cells in transition.items():
        lines.append(f"| {label} | {cells['答对']} | {cells['UNKNOWN']} | {cells['错答']} |")
    remaining = {"证据分数不足": 0, "内容置信不足": 0, "两项都不足": 0}
    for row in data["conditions"]["both"]["semantic_verified_retry"]["records"]:
        if row["supported"] and row["prediction"] == 0:
            a = row["attempts"][-1]
            low_e = a["evidence"] <= calibration["new_threshold"]
            low_v = a["value_probability"] <= calibration["value_threshold"]
            remaining[
                "两项都不足" if low_e and low_v else "证据分数不足" if low_e else "内容置信不足"
            ] += 1
    lines += [
        "",
        "最终模型剩余UNKNOWN的直接门槛原因（互斥计数）："
        + "；".join(f"{k}{v}次" for k, v in remaining.items())
        + "。这说明触发了哪项判断，不保证对应判断本身正确。",
    ]
    lines += [
        "",
        "## 没有证据时是否瞎猜",
        "",
        "表中每格为错误作答数/无证据问题数。支持移除是从两轮候选中排除正确事件，未物理删除SQLite记录或重算组中心；读取器不能通过重试偷看其内容。空载荷保留地址但清空内容。新owner使用独立随机事实、每个owner存96条事件，询问另8条从未写入的事件。",
        "",
        "| 无证据条件 | " + " | ".join(names.values()) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for kind, label in (
        ("removed", "移除正确事件"),
        ("zero", "清空内容"),
        ("new_owners_original", "新owner/原句"),
        ("new_owners_both", "新owner/改写"),
    ):
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{data['negative_suites'][kind][n]['counts'].get('unsupported_false_answer', 0)}/{data['negative_suites'][kind][n]['counts']['unsupported']}"
                for n in names
            )
            + " |"
        )
    if "fresh_history" in data:
        lines += [
            "",
            "## 新随机历史的改写回忆",
            "",
            f"随机种子{data['stress_seed']}；训练和校准结束后生成，另采样480个已存问题与120个未知问题。",
            "",
            "| 模型 | 已存答对率 | 已存却UNKNOWN | 错误非UNKNOWN | 未存错误作答 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, label in names.items():
            m = data["fresh_history"][name]
            c = m["counts"]
            lines.append(
                f"| {label} | {m['present_accuracy']:.2%} | {c.get('present_unknown', 0)} | {c.get('wrong_nonunknown', 0)} | {c.get('unsupported_false_answer', 0)}/{c['unsupported']} |"
            )
    lines += [
        "",
        "## 门槛与验证",
        "",
        f"- 证据门槛：{calibration['old_threshold']} → {calibration['new_threshold']}，内容置信门槛：{calibration['value_threshold']}。没有降低。",
        f"- 负例校准：{calibration['validation_queries']}个验证问题、两轮共{calibration['attempts']}次检查；错误放行{calibration['false_accepts_before']} → {calibration['false_accepts_after']}。",
        f"- 正例验证：{json.dumps(calibration['positive_validation'], ensure_ascii=False)}。",
        "- 分数是神经模型输出，不是经过充分校准的正确概率；0.99门槛不代表真实错误率保证。",
        "",
        "首版证据头虽然在负例验证上零错误，但正例验证只答对126/600，592/600的条件答案已经正确。这个失败说明只测负例会把‘全部拒答’误当成安全进步。其全部模型、测试和重启失败均保留在evidence_round_01及evidence_comparison_01。",
        "",
        "第二版删除高维地址坐标、加入双侧间隔损失后，正例验证仍全部拒答，未进入测试。最终版本另外平衡正负候选损失，并把新验证器的学习率设为1e-3以充分收敛；读取器仍1e-4。多项修改一起进行，不能把收益单独归因于其中一项。两个门槛未改，最终测试在训练和正负验证结束后执行。",
        "",
    ]
    restart = candidate / "restart_verification_template3.json"
    if restart.exists():
        r = read(restart)
        lines += [
            "## 跨进程改写回忆",
            "",
            f"新进程重复输出完全相同：{r['exact_across_fresh_processes']}。",
        ]
        for row in r["records"]:
            lines.append(f"- 目标“{row['expected']}”，实际“{row['first_process']['answer']}”。")
            attempt = row["first_process"]
            if "evidence_score" in attempt:
                lines.append(
                    f"  证据分数{attempt['evidence_score']:.6f}；最高条件答案概率{attempt['conditional_value_probability']:.6f}。"
                )
        lines.append("“安心”示例的事件匹配通过，但内容置信未过门槛；没有为了让演示答对而放宽门槛。")
        lines.append("")
    env = read(candidate / "environment.json")
    lines += [
        "## 实现与限制",
        "",
        f"最终模型{env['parameters']:,}参数，冻结编码器校验通过：{env['frozen_encoders_exact']}。训练和校准约{env['elapsed_s']:.1f}秒，不含建库和测试；GPU为{env['gpu']}。",
        "重试最多一次，仅扫描当前owner的地址并读取最多8条内容；这不是大规模磁盘索引方案。数据库和writer指纹不变，支持读取旧库。没有读取聊天原文或在推理中提供正确事件ID。",
        "这是单随机种子的有限词表合成实验。首版测试失败后的修正是探索性过程，已透明记录；不能声称一次预注册实验成功，也不能声称真实对话中不会幻觉。新owner负例分成共享事件的多类问题，不能视为完全独立的伯努利样本。",
        "",
        "## 原始证据",
        "",
        f"- [对比JSON]({comparison.resolve().as_posix()}/comparison.json)",
        f"- [诊断JSON]({runs.resolve().as_posix()}/abstention_diagnosis.json)",
        f"- [校准记录]({candidate.resolve().as_posix()}/calibration.json)",
        "- [实验方案与修正记录](EVIDENCE_PROTOCOL.md)",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("experiments/episodic_memory/runs"))
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, default=Path("experiments/episodic_memory/EVIDENCE_RESULTS.md")
    )
    a = p.parse_args()
    report(a.runs, a.comparison, a.candidate, a.out)
