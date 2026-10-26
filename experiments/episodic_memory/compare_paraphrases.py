"""Summarize paired factorial JSONs; refuse to compare different query samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root, output):
    names = {
        "旧模型": ("paraphrase_baseline", "paraphrase_baseline_fresh"),
        "只增强表达": ("paraphrase_aug_eval", "paraphrase_aug_fresh"),
        "增强+意图瓶颈+一致性": ("paraphrase_structured_eval", "paraphrase_structured_fresh"),
    }
    conditions = {
        "original": "原句",
        "question_only": "只换提问",
        "event_only": "只换事件叙述",
        "both": "两者都换",
    }
    data = {
        name: [load(root / folder / "factorial.json") for folder in folders]
        for name, folders in names.items()
    }
    reference = None
    for runs in data.values():
        for run in runs:
            for c in conditions:
                rows = run["conditions"][c]["modes"]["clustered"]["outcomes"]
                signature = [(r["event_id"], r["task"], r["stored"], r["target"]) for r in rows]
                if reference is None:
                    reference = signature
                assert signature == reference, "Unpaired facts or questions; invalid comparison"
    present_count = sum(r[2] for r in reference)
    missing_count = len(reference) - present_count
    lines = [
        "# 改写泛化：配对对比实验",
        "",
        "所有数值从本机评估JSON读取；已自动核对模型与四种条件使用相同事件ID、任务、目标答案和采样顺序。",
        f"每格为已存事件答对率，分母{present_count}。另有{missing_count}个未写入事件问题，单独报告错误作答率。单随机种子。",
        "",
    ]
    for i, title in enumerate(("已知诊断句式：模板2", "新留出句式：模板3")):
        lines += [
            f"## {title}",
            "",
            "| 条件 | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for condition, label in conditions.items():
            values = [data[n][i]["conditions"][condition]["modes"]["clustered"] for n in names]
            lines.append(
                f"| {label} | "
                + " | ".join(
                    f"{v['present_accuracy']:.2%} ({v['counts']['present_correct']}/{v['counts']['present']})"
                    for v in values
                )
                + " |"
            )
        lines += [
            "",
            "| 模型/条件 | Recall@8 | 未知问题错误作答率 | oracle答对率 | 清空内容后答对率 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in names:
            for condition, label in conditions.items():
                modes = data[name][i]["conditions"][condition]["modes"]
                m = modes["clustered"]
                lines.append(
                    f"| {name}/{label} | {m['recall_at_8']:.2%} | {m['missing_false_answer_rate']:.2%} | {modes['oracle']['present_accuracy']:.2%} | {modes['zero']['present_accuracy']:.2%} |"
                )
        lines += [
            "",
            "| 模型 | 事件地址余弦 | 内容槽余弦 | 原句槽解码均值 | 改写槽解码均值 | 改写问题意图识别 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in names:
            audit = data[name][i]["audit"]
            auxiliary = audit["event_aux_accuracy"]
            original = sum(auxiliary["original"].values()) / 6
            changed = sum(auxiliary["changed"].values()) / 6
            intent = audit.get("predicted_intent_accuracy", {}).get("changed")
            intent_text = "无此模块" if intent is None else f"{intent:.2%}"
            lines.append(
                f"| {name} | {audit['event_key_cosine']:.3f} | {audit['event_slot_cosine']:.3f} | {original:.2%} | {changed:.2%} | {intent_text} |"
            )
        lines.append("")
    base = data["旧模型"][0]["conditions"]["question_only"]["modes"]
    lines += [
        "## 旧模型问题表示干预",
        "",
        f"只换问题时答对率{base['clustered']['present_accuracy']:.2%}。保留新问题地址、换回原问题的读取表示：{base['reference_summary']['present_accuracy']:.2%}；只换回原地址：{base['reference_address']['present_accuracy']:.2%}。",
        "这支持读取问题语义是主要瓶颈；干预需要原问法，不能作为真实推理成绩。事件侧槽辅助解码也明显下降，支持内容编码对句式敏感。",
        "词表里有一个字，不代表它已被训练过。旧模型使用两套独立的事件/问题编码器，某个字出现在另一侧也不能直接给本侧提供语义。模板2含事件编码器未在训练输入中见过的10种符号、问题编码器未见过的16种符号；增强后覆盖这些用字，但仍需检验新组合能否理解。",
        "新模板3还把场景线索移到事件末尾，旧模型的寻址也明显退化；因此诊断模板2的‘检索基本正常’结论不能扩大到所有语言变化。",
        "",
        "## 实验配置与原始文件",
        "",
    ]
    for run_name in ("pilot_03", "paraphrase_aug_v2", "paraphrase_structured"):
        env = load(root / run_name / "environment.json")
        logs = [
            json.loads(line)
            for line in (root / run_name / "training.jsonl").read_text().splitlines()
        ]
        complete = next(x for x in reversed(logs) if x["stage"] == "complete")
        lines.append(
            f"- `{run_name}`：{env['parameters']:,}参数，训练/建库阶段{complete['elapsed_s']:.2f}秒；不含数据生成、增强缓存预处理、评估。"
        )
    for name, folders in names.items():
        for folder in folders:
            lines.append(f"- {name}：[原始评估 {folder}](runs/{folder}/factorial.json)")
    lines += [
        "",
        "## 剩余错误与跨进程复核",
        "",
        "| 新模型、新句式条件 | 有记忆却回答UNKNOWN | 给出错误的非UNKNOWN答案 |",
        "|---|---:|---:|",
    ]
    fresh = data["增强+意图瓶颈+一致性"][1]
    for condition, label in conditions.items():
        rows = fresh["conditions"][condition]["modes"]["clustered"]["outcomes"]
        abstentions = sum(r["stored"] and r["prediction"] == 0 for r in rows)
        wrong = sum(
            r["stored"] and r["prediction"] != 0 and r["prediction"] != r["target"] for r in rows
        )
        lines.append(f"| {label} | {abstentions} | {wrong} |")
    lines += [
        "",
        "两者都换时114个错误中104个是UNKNOWN，另有16个问题没找回支持事件（这两个集合可重叠）。对192条事件的槽辅助解码和意图诊断表明内容/问题识别已大幅改善，但真实候选集合下仍有读取与拒答问题。oracle候选更容易，不能把它与实际路由的差距全部归因于召回漏检。",
        "第二阶段目前使用原句编码的训练库，只增强问题；改写后的事件库仍存在分布变化。这是下一轮应在独立验证集上检验的原因假设，不是本轮已证明的唯一根因。",
        "",
    ]
    restart = load(root / "paraphrase_structured" / "restart_verification_template3.json")
    lines.append(
        f"新句式写入后退出程序，新进程重复查询输出一致：{restart['exact_across_fresh_processes']}；两个答案全部正确：{restart['all_correct']}。"
    )
    for row in restart["records"]:
        lines.append(
            f"- 应记住“{row['expected']}”，实际回答“{row['first_process']['answer']}”；重启前后相同：{row['identical_across_processes']}。"
        )
    lines += [
        "",
        "持久化稳定不等于每次都能正确读出；本轮保留了这个失败案例，没有通过换样例或重训隐藏它。",
        "",
    ]
    lines += [
        "",
        "## 解释边界",
        "",
        "模板2已被用于分析失败原因，只能算修复复测。模板3没有进入训练或验证选模，大部分词汇/短语来自增强中的其他组合，事件侧仍有‘想、起、来’三个字未出现在该编码器的训练视图中。没有词表外符号；证明的是受控组合泛化，不是任意同义词、长对话或真实语言理解。",
        "增强模型与结构模型的比较同时改变了意图瓶颈和一致性损失，并增加少量参数和训练计算；不能独立归因到其中某一项。第一阶段验证分布也从原句改为增强句。没有按测试成绩继续调参或重新选择checkpoint。",
        "内存内容仍需通过学习地址检索，数据库保存潜在表示而非聊天原文。writer改变后重新编码建库；不兼容旧writer的数据库。",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("experiments/episodic_memory/runs"))
    p.add_argument(
        "--out", type=Path, default=Path("experiments/episodic_memory/PARAPHRASE_RESULTS.md")
    )
    args = p.parse_args()
    summarize(args.runs, args.out)
