"""Create a reviewable report from measured JSON files, without invented metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_report(run: Path):
    env = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    logs = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    complete = next((r for r in reversed(logs) if r["stage"] == "complete"), {})
    evaluations = []
    for split in ("validation", "test", "test_template"):
        path = run / f"evaluation_{split}_model.json"
        if path.exists():
            evaluations.append(json.loads(path.read_text(encoding="utf-8")))
    lines = [
        "# 磁盘事件记忆：本机实验报告",
        "",
        f"实验目录：`{run.resolve()}`",
        "",
        "## 环境与训练",
        "",
        f"- GPU：{env['gpu']}；PyTorch：{env['torch']}。",
        f"- 参数量：{env['parameters']:,}。",
        f"- 第一阶段步数：{env['arguments']['steps']}；第二阶段：{env['arguments']['stage_b_steps']}。",
        f"- 脚本内总耗时：{complete.get('elapsed_s', float('nan')):.1f}秒。不是独占GPU性能基准。",
        f"- PyTorch峰值已分配显存：{complete.get('peak_cuda_allocated_mb', float('nan')):.1f}MiB；不包括驱动、显示桌面、其他进程及分配器保留空间。",
        "- 冻结事件编码器后建库；第二阶段writer指纹保持不变。",
        f"- 训练配置：`{json.dumps(env['config'], ensure_ascii=False)}`。",
        "",
        "## 数据隔离与结论边界",
        "",
        "每段历史的事实独立随机分配；训练/验证/测试owner隔离。每个测试集中20%问题来自未写入数据库的事件。",
        "输入为受控中文和有限短语词表，查询显式给出项目及讨论次数。训练使用场景属性辅助监督；推理只使用学习得到的向量，不传入正确事件ID或答案。",
        "本报告只支持受控事件回忆结论，不支持自由对话、终身学习、人格或主观情绪结论。",
        "",
    ]
    labels = {
        "oracle": "候选保证含支持事件",
        "clustered": "动态分组+磁盘读取",
        "flat": "全地址扫描",
        "zero": "清空内容向量",
        "shuffled": "错配地址与内容",
        "no_appraisal": "清空评价/关注槽",
    }
    for ev in evaluations:
        lines += [
            f"## {ev['split']}：{ev['count']}个问题",
            "",
            f"数据库：{ev['store']['events']}条事件，{ev['store']['groups']}组，{ev['store']['file_bytes']:,}字节。",
            "",
            "| 条件 | 已存事件答对率 | Recall@8 | 无证据错误作答率 | 总准确率 |",
            "|---|---:|---:|---:|---:|",
        ]
        for mode, m in ev["modes"].items():
            lines.append(
                f"| {labels[mode]} | {m['present_accuracy']:.1%} | {m['recall_at_8']:.1%} | {m['missing_false_answer_rate']:.1%} | {m['accuracy']:.1%} |"
            )
        m = ev["modes"]["clustered"]
        lines += [
            "",
            f"分组路由平均比较 {m['mean_candidate_keys']:.1f} 条组内候选地址；平均加载内容 {m['mean_payload_bytes']:.0f} 字节。此计数不含组中心扫描，不能用来声称总寻址计算减少相同比例。",
            f"缓存索引条件下，路由+内容读取+回答平均 {m['mean_read_and_answer_ms']:.2f} ms；不含问题编码，不能当成冷盘端到端延迟。",
            f"关闭重开数据库，索引与FP16载荷完全一致：{ev['reopen_payload_exact']}。",
            "",
            "各类已存事件答对率："
            + "；".join(f"{t}={v:.1%}" for t, v in m["per_task_present_accuracy"].items())
            + "。",
            "",
        ]
    restart = run / "restart_verification.json"
    if restart.exists():
        r = json.loads(restart.read_text(encoding="utf-8"))
        lines += [
            "## 独立进程持久化与反事实检查",
            "",
            f"- 两个不同owner的经历分别写入；写入后程序退出，查询启动新进程。两种答案均正确：{r['all_correct']}。",
            f"- 重复启动查询进程，输出完全相同：{r['exact_across_fresh_processes']}。",
            f"- 相同问题随保存的经历不同而改变答案：{r['same_question_different_memories_change_answer']}。",
            "",
        ]
        for row in r["records"]:
            lines.append(
                f"- {row['owner']}：应记住“{row['expected']}”，实际回答“{row['first_process']['answer']}”。"
            )
        lines.append("")
    if evaluations:
        ev = next((e for e in evaluations if e["split"] == "test"), evaluations[0])
        lines += [
            "## 示例（包含失败，不只展示成功）",
            "",
            "| 问题 | 目标 | 输出 | 找回支持事件 |",
            "|---|---|---|---|",
        ]
        for ex in ev["examples"][:12]:
            lines.append(
                f"| {ex['query']} | {ex['target']} | {ex['prediction']} | {ex['support_found']} |"
            )
    lines += [
        "",
        "## 解读与限制",
        "",
        f"测试使用{env['arguments']['eval_histories']}个独立owner，每个{env['arguments']['test_events']}条已存事件；一次查询仅访问当前owner。数据库总事件数不是单次检索规模。问题采样允许重复。此目录只记录一个训练随机种子。",
        "应将正常读取与内容清零、错配进行比较，判断是否使用所存内容；结合独立进程检查验证这种依赖是否跨程序重启保留。",
        "未见模板同时改变事件叙述和问题表达。若保证正确候选仍失败，问题不限于寻址，不能宣称自由中文表达泛化。",
        "需同时检查事件数和组数；大量微小组可能使组中心扫描成本接近逐事件扫描。分组不是记忆压缩或遗忘。",
        "首轮试验的失败尝试、最终模型的结论和边界见本目录上两级的RESULTS.md。",
        "",
        "## 可复核文件",
        "",
        "- `training.jsonl`：训练和验证曲线。",
        "- `environment.json`：配置、参数量、运行环境。",
        "- `evaluation_*_model.json`：计数、分任务指标与样本。",
        "- `model.pt` / `stage_a.pt`：最终与第一阶段选定checkpoint。",
        "- `memory_*.sqlite`：不含原文或答案标签的事件数据库。",
        "- `restart_verification.json`：独立进程检查。",
        "",
    ]
    path = run / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    print(make_report(p.parse_args().run))
