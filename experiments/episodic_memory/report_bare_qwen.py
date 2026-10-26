"""Saved-output audit: preserve strict metrics, disclose paraphrases and nonstandard refusals."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .qwen_memory import save_json


def main(args):
    raw = (args.results / "results.jsonl").read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "693d51964dd54b9e51ed5a06ac6f3ac2647b4196f537c05a470277a8c53c5e09"
    ):
        raise ValueError(
            "Saved-output audit is bound to the reviewed run. Re-audit changed outputs before producing semantic scores."
        )
    rows = [json.loads(s) for s in raw.decode("utf-8").splitlines()]
    assert len(rows) == 784
    bykey = {(r["mode"], r["id"]): r for r in rows}
    assert len(bykey) == 784
    audit = []

    def mark(mode, cid, label, note):
        r = bykey[mode, cid]
        audit.append(
            dict(
                mode=mode,
                id=cid,
                label=label,
                note=note,
                raw=r["result"]["raw"],
                expected=r["expected"],
            )
        )

    for cid in [
        "test-counterfactual-6",
        "test-duplicate-1",
        "test-duplicate-7",
        "test-duplicate-13",
        "test-empty-1",
        "test-missing_field-1",
        "test-nonexistent-7",
    ]:
        mark(
            "task_no_context",
            cid,
            "refused",
            "Explicitly says unknown/not provided/cannot determine; no unsupported factual answer.",
        )
    for cid in ["test-empty-1", "test-missing_field-14", "test-no_anchor-0"]:
        mark(
            "task_full_text",
            cid,
            "refused",
            "Clear refusal with different wording, not a fact error.",
        )
    for cid, note in {
        "test-counterfactual-14": "提出的建议 expresses 提出建议.",
        "test-duplicate-5": "他人的反应 is a paraphrase of 他人反应.",
    }.items():
        mark("task_full_text", cid, "target_fact_correct", note)
    for cid, note in {
        "test-normal-4": "问题得到了解决 expresses 问题解决.",
        "test-counterfactual-8": "叫错了名字 expresses 叫错名字.",
        "test-counterfactual-10": "获得了来自小杨的支持 contains 获得支持, but the source of support is not specified in the record; see additional-claim flag.",
        "test-counterfactual-11": "对方的需求 expresses 对方需求.",
        "test-counterfactual-14": "提出了建议 expresses 提出建议, but assigning the action to 我 is an unconfirmed detail.",
        "test-partial_known-11": "具体的细节 expresses 具体细节.",
        "test-duplicate-5": "他人的反应 expresses 他人反应.",
    }.items():
        mark("task_oracle_text", cid, "target_fact_correct", note)
    for mode, cid, note in [
        (
            "task_full_text",
            "test-normal-15",
            "Adds causation from focus on scheduling to anger; recorded reason is 时间不足, not the focus itself.",
        ),
        (
            "task_oracle_text",
            "test-normal-5",
            "Specifies that attention to details concerned the cup-breaking process and cause; the record only says 具体细节.",
        ),
        (
            "task_oracle_text",
            "test-counterfactual-3",
            "Narrows 担心评价 to evaluation of the band project; this object is not explicit in the record.",
        ),
        (
            "task_oracle_text",
            "test-counterfactual-10",
            "Attributes support to 小杨; only conversation partner and 获得支持 are recorded separately.",
        ),
        (
            "task_oracle_text",
            "test-counterfactual-14",
            "Attributes 提出建议 to 我; the record does not explicitly identify the actor.",
        ),
    ]:
        mark(mode, cid, "extra_unsupported_claim", note)
    overrides = {
        (a["mode"], a["id"]): a["label"] for a in audit if a["label"] != "extra_unsupported_claim"
    }
    counts = {}
    for mode in sorted({r["mode"] for r in rows}):
        selected = [r for r in rows if r["mode"] == mode]

        def label(r):
            return overrides.get(
                (mode, r["id"]),
                {
                    "correct": "target_fact_correct",
                    "refused": "refused",
                    "wrong": "unmatched",
                    "truncated_match": "truncated_match",
                }[r["strict_outcome"]],
            )

        counts[mode] = dict(
            positive=dict(Counter(label(r) for r in selected if r["expected"] is not None)),
            negative=dict(Counter(label(r) for r in selected if r["expected"] is None)),
            extra_claims=sum(
                a["mode"] == mode and a["label"] == "extra_unsupported_claim" for a in audit
            ),
        )
    save_json(
        args.results / "semantic_audit.json",
        dict(
            results_sha256=hashlib.sha256(raw).hexdigest(),
            method="Assistant post-hoc review of saved outputs, not independent blind evaluation. Target fact accuracy is distinct from whole-answer grounding. Plain-chat refusals were not exhaustively classified.",
            annotations=audit,
            counts=counts,
        ),
    )
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    memory = json.loads((args.run / "test/summary.json").read_text(encoding="utf-8"))
    assert summary["case_sha256"] == memory["cases_sha256"]
    names = {
        "chat_no_context": "原始普通聊天，无经历",
        "task_no_context": "原始Qwen，同规则、无经历",
        "task_full_text": "原始Qwen，同规则、全部可用经历文字",
        "task_oracle_text": "原始Qwen，同规则、指定正确事件文字（oracle）",
    }
    lines = [
        "# 未接记忆模块的原始 Qwen：同题补充对照",
        "\n> 历史协议分数与事后同义表达计数不等于完整语义评估。请先读 [评分方法纠正](EVALUATION_CORRECTION.md)。\n",
        "",
        "**先前33→55/64比较的是两套带记忆的系统，不能证明比原始Qwen更聪明。** 本轮直接加载未接任何适配器的官方Qwen2.5-1.5B-Instruct，补齐同题对照。",
        "",
        "## 同样64道有支持问题",
        "",
        "| 条件 | 原评分：目标事实匹配 | 同义表达核查后：目标事实正确 |",
        "|---|---:|---:|",
    ]
    for mode in names:
        lines.append(
            f"| {names[mode]} | {summary['modes'][mode]['positive'].get('correct', 0)}/64 | {counts[mode]['positive'].get('target_fact_correct', 0)}/64 |"
        )
    lines += [
        f"| 新记忆系统，磁盘连续向量、无历史原文 | {memory['positive'].get('correct', 0)}/64 | {memory['positive'].get('correct', 0)}/64 |",
        "",
        "同义核查只针对目标事实，不等于整句全部有依据：全部文字组有1条添加了未支持的因果；oracle组至少4条添加了未明确记录的注意对象、评价对象、支持者或行动主体。逐条注释在semantic_audit.json。核查由本助手完成，不是独立盲评。",
        "全部文字组严格评分55正确、7未匹配、2拒答；2个未匹配是同义表达，因此目标事实核查为57正确、5错误、2拒答。新记忆系统是55正确、0错误、9拒答。不能据此宣称新系统提高了基座阅读理解能力；它在这个样本上采取了更保守的回答策略。",
        "oracle组严格48正确、13未匹配、2拒答、1条目标事实命中但生成被截断；同义核查增加7条目标事实正确。这个诊断给模型提供了正确事件，却不能保证模型输出达到理论上界：上下文组织、措辞、错误字段选择和额外推断仍影响结果。",
        "",
        "## 没有经历信息时",
        "",
        "原始Qwen按相同任务规则、没有经历时，64题均没有正确事实匹配：核查后63条明确表示未知，1条无依据地声称“我在第16次讨论中与您进行了对话”。没有读过这些随机生成的私人经历，本来就不应知道答案；拒答不能算模型能力差。",
        f"普通聊天组也没有完整目标事实匹配，但240条中{summary['modes']['chat_no_context']['truncated']}条到达40-token上限，且大量回答使用自然语言说明缺少信息。旧评分器将它们标为wrong只是格式不匹配；本报告不把这些计为240次事实错误，也不据此评价普通聊天质量。",
        "",
        "## 拒答与范围限制",
        "",
        "176题涵盖缺失、错地址、错用户、矛盾等条件，也包含当前原型明确未支持的多字段和是非问题。原始模型直接读全部可用文字时，严格计94条标准拒答；另3条是自然语言拒答，合计97条拒答、79条未按当前任务规则拒答。新记忆系统176条全部拒答。",
        "79条不能全部称为幻觉：其中16条是多字段请求、7条是是非请求，原始模型的部分回答实际上提供了记录中的信息，只是超出这个有限原型的协议。其余情况也需区分错用另一条事件、未明确拒答和编造事实。比较的是整套系统行为，不能把差异都归因于记忆编码器。",
        "更直接的证据是：全部文字组在16个同地址矛盾问题中均给出了非拒答输出；在16个所问字段缺失问题中，核查后14个没有明确拒答。新系统这两类均拒答，但前一轮已说明并非每次都由正确的冲突识别原因触发。",
        "",
        "## 普通聊天是否保留",
        "",
        f"直接用原始Qwen回答原来的8个普通聊天问题，与新系统显式chat路径逐字一致{summary['chat_exact']}/{summary['chat_count']}。这与主干冻结、chat旁路一致；不能据此证明开启记忆的任意自由聊天没有退化。",
        "",
        "## 这次对照能支持什么",
        "",
        "- 原始Qwen一旦直接看到事件文字，本来就有相当的读取能力；目前没有证据表明新增模块让基座更聪明。",
        "- 向量记忆系统能在不把历史原文放进上下文的情况下利用已落盘经历；独立重启复现是此前已有的持久化证据。",
        "- 当前的低错误输出表现依赖字段路由、置信门槛、冲突检查和有限输出语法这一整套设计，仍需付出误拒代价。",
        "- 尚未验证超长历史、开放词表、多事件推理或与文本检索系统的长期可扩展性对比。短上下文文字对照不等于已经部署聊天原文检索。",
        "",
        "## 复现与边界",
        "",
        "240个问题与前轮逐字相同。每条独立会话；第2–4组共用任务提示，不提供字段真值或答案；文字只按owner和可用性元数据过滤。oracle明确使用实验者指定正确事件，仅作诊断。原始模型不加载adapter.pt或semantic.pt，不套外部门槛，不过滤生成答案。",
        "贪心生成、40新token、BF16、batch8；普通聊天复核为单条100token。完整文字组64道支持题没有截断；oracle有1条截断。首次顺序运行因低频GPU过慢而中止，完整正式运行重新开始，部分结果不参与评分。批量数值差异与提示差异意味着这不是严格单因素消融。",
        "本轮是已暴露测试集的事后补充控制，不训练或修改记忆系统。原始输出、完整提示、严格评分与语义核查均保留。",
        f"- 原始模型revision：`{summary['source_revision']}`。",
        f"- 测试清单SHA256：`{summary['case_sha256']}`。",
        f"- 完整输出目录：`{args.results.as_posix()}`。",
        "- 脚本：compare_bare_qwen.py、report_bare_qwen.py；边界测试：test_bare_qwen.py。",
        "- 3项上下文隔离/不泄漏标签测试通过，代码通过Ruff。",
        "",
        "```powershell",
        "C:\\Python313\\python.exe -m experiments.episodic_memory.compare_bare_qwen --out experiments/episodic_memory/runs/my_bare_qwen_control",
        "# report_bare_qwen的人工核查绑定已审阅输出哈希；重跑输出不同须重新核查，不能自动套用旧注释。",
        "C:\\Python313\\python.exe -m experiments.episodic_memory.report_bare_qwen",
        "```",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/episodic_memory/runs/bare_qwen_comparison_02"),
    )
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/semantic_memory_03")
    )
    p.add_argument(
        "--report", type=Path, default=Path("experiments/episodic_memory/BARE_QWEN_RESULTS.md")
    )
    main(p.parse_args())
