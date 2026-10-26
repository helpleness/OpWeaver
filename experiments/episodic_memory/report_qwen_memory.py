"""Render recorded measurements; never rerun or select test questions."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main(run, output):
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    meta = load(run / "metadata.json")
    s = load(run / "evaluation/summary.json")
    rows = [
        json.loads(line)
        for line in (run / "evaluation/paired_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    val = load(run / "validation_step_1200.json")
    novel_rows = load(run / "evaluation/novel_results.json")
    condition_names = dict(
        normal="真实记忆",
        no_memory="无记忆",
        counterfactual="同地址反事实记忆（按注入事实评分）",
        unrelated="不同场景错误记忆",
        oracle="指定正确事件ID：诊断，不是主结果",
    )
    lines = [
        "# Qwen2.5-0.5B-Instruct 磁盘连续记忆实验",
        "",
        "已在本机RTX4060 Laptop 8GB完成训练及固定测试。基座冻结；新增记忆模块可训练，部署时读取SQLite中的连续载荷。最终测试后未继续调参。",
        "",
        "结论：受控事件上已有根据持久化经历回答、随记忆替换而改变回答的证据；新句式泛化仍未通过，存在取对记忆却把原因回答成情绪的高置信错误。不能将本轮结果当作通用长期记忆已实现。",
        "",
        "## 配对测试结果",
        "",
        "同一组60个问题，每字段10题；四个未参与训练的owner、128条已存事件。下表为通过证据>0.99及所有生成token概率>0.8双门槛后的结果。反事实行按注入事实计分，不表示其符合原始真实经历。",
        "",
        "| 条件 | 正确 | 错误 | 拒答 | 原始生成正确/错误/拒答 |",
        "|---|---:|---:|---:|---|",
    ]
    for key, name in condition_names.items():
        f = s["conditions"][key]["final_status"]
        r = s["conditions"][key]["raw_status"]
        lines.append(
            f"| {name} | {f.get('correct', 0)} | {f.get('wrong', 0)} | {f.get('unknown', 0)} | {r.get('correct', 0)}/{r.get('wrong', 0)}/{r.get('unknown', 0)} |"
        )
    lines += [
        "",
        f"真实记忆与反事实记忆两边都正确跟随各自内容的配对题：{s['paired_correct_then_counterfactual_correct']}/60。无记忆/错误地址的最终拒答含外部证据门控，不能当作模型本身学会拒答的证明；原始生成列用于区分。",
        "",
        f"未写入的12条经历：原始拒答{s['missing_raw_refused']}/12，最终拒答{s['missing_final_refused']}/12。",
        "",
        "## 寻址与读取诊断",
        "",
        f"正常检索第一名为正确事件{s['retrieval_top1_correct']}/60；正确且通过证据门槛{s['retrieval_accepted_correct']}/60；错误地址被接受{s['retrieval_accepted_wrong']}/60。",
        "",
        f"拒答拆分：地址未通过{s['normal_refusal_diagnosis']['address_rejected']}；地址通过但原始生成拒答{s['normal_refusal_diagnosis']['accepted_but_generated_refusal']}；地址通过但生成概率不足{s['normal_refusal_diagnosis']['accepted_but_probability_rejected']}。原始内容正确却被概率门槛拒绝{s['correct_raw_rejected']}题。",
        "",
        "| 字段 | 正确 | 错误 | 拒答 |",
        "|---|---:|---:|---:|",
    ]
    for task, f in s["by_field"].items():
        lines.append(
            f"| {task} | {f.get('correct', 0)} | {f.get('wrong', 0)} | {f.get('unknown', 0)} |"
        )
    lines += [
        "",
        "## 固定改写配对集",
        "",
        "同样12个事件问题，各字段2题；仅换问法、仅换事件叙述、两者都换。heldout函数未参加本轮训练。样本小，不能据此推断开放中文泛化率。",
        "",
        "| 条件 | 正确 | 错误 | 拒答 | 原始生成正确/错误/拒答 |",
        "|---|---:|---:|---:|---|",
    ]
    for key, name in [
        ("question_only", "只换问题"),
        ("event_only", "只换事件"),
        ("both", "同时改写"),
    ]:
        f = s["novel"][key]["final_status"]
        r = s["novel"][key]["raw_status"]
        lines.append(
            f"| {name} | {f.get('correct', 0)} | {f.get('wrong', 0)} | {f.get('unknown', 0)} | {r.get('correct', 0)}/{r.get('wrong', 0)}/{r.get('unknown', 0)} |"
        )
    paired_ids = {(r["id"], r["task"]) for r in novel_rows}
    paired_baseline = Counter(
        r["normal"]["final_status"] for r in rows if (r["id"], r["task"]) in paired_ids
    )
    lines += [
        "",
        f"同样12题在原问法、原事件下：正确{paired_baseline.get('correct', 0)}、错误{paired_baseline.get('wrong', 0)}、拒答{paired_baseline.get('unknown', 0)}。",
    ]
    for row in novel_rows:
        failed = row["question_only"]
        if failed["final_status"] == "wrong":
            lines += [
                "",
                f"新问法的高置信错误：问“{row['question']}”，应答“{row['expected']}”，实际回答“{failed['raw']}”，最低token概率{failed['confidence']:.4f}。检索ID{'正确' if failed['id'] == row['id'] else '错误'}，证据{failed['evidence']:.6f}。同一事件用原问法仍能回答原因，错误位于生成式的问题理解/字段读取阶段；本轮不以补失败模板或降低门槛修饰该结果。",
            ]
    lines += ["", "## 具体输出（按固定测试顺序选例）", ""]
    examples = [
        next(
            (
                r
                for r in rows
                if r["normal"]["final_status"] == "correct"
                and r["counterfactual"]["final_status"] == "correct"
            ),
            rows[0],
        )
    ]
    fail = next((r for r in rows if r["normal"]["final_status"] != "correct"), None)
    if fail:
        examples.append(fail)
    for row in examples:
        lines += [
            f"- 问：{row['question']}",
            f"  原事实：{row['expected']}；写入反事实：{row['counterfactual_expected']}。",
            f"  真记忆回答：“{row['normal']['answer']}”；原始生成：“{row['normal']['raw']}”；最低token概率{row['normal']['confidence']:.4f}。",
            f"  反事实记忆回答：“{row['counterfactual']['answer']}”。",
        ]
    disk = s["disk"]["normal"]
    lines += [
        "",
        "## 实现与资源",
        "",
        "- 写入：冻结Qwen对当前事件生成上下文特征；可训练注意力池提取8×256潜变量，六个事实槽加项目/次数两个地址槽。训练字段辅助头与嵌入对齐监督，未在部署侧查答案表。",
        "- 存储：SQLite载荷为FP16，4 KiB/事件；另有32维FP32地址128字节、ID/owner/时间戳和索引。没有历史原文、历史token或答案列。训练原文只在训练/评估目录，查询程序不读取这些目录。潜变量可解码，不是加密。",
        "- 分堆：CPU余弦地址索引在线入堆，可属于相近两堆；超过8个成员时分裂，小而相似的簇可合并。分堆不合并、删除或改写事件内容。先查2堆取4条，证据不足再扫本owner的地址取4条。全部地址仍常驻CPU，尚非十亿级磁盘索引。",
        "- 读取：载荷投影成32个连续输入嵌入，与固定字段标记交错交给Qwen生成。问题只输入文本，不输入目标字段标签或正确事件ID。oracle仅独立诊断。",
        f"- 实际持久化{disk['events']}条事件，{disk['groups']}个簇；最大{disk['max_group_members']}成员；SQLite {disk['file_bytes']:,}字节。",
        f"- 基座{meta['backbone_parameters']:,}参数，适配器总计{meta['adapter_parameters']:,}参数，第二阶段训练{meta['trainable_parameters']:,}参数。最终训练用时{meta['elapsed_s']:.1f}秒（复用缓存；不含下载和首建特征），峰值CUDA allocated {meta['peak_allocated_gb']:.2f} GB，reserved {meta['peak_reserved_gb']:.2f} GB。不是整机内存峰值。",
        f"- 正常聊天前后贪心输出{s['chat_exact']}/{s['chat_count']}逐字一致。显式`chat`绕过记忆模块，且基座从未更新；这证明保留原有聊天路径，不证明自动判断何时回忆，也不证明带记忆的任意聊天无退化。",
        f"- 两个独立进程重新加载真实/反事实数据库，与原进程输出一致：{sum(p['identical_to_parent'] for p in s['restart_proofs'])}/2。重启命令只收到基座、适配器、数据库、owner和问题，未收到事件真值。",
        "- 自动检查19项通过：旧版14项，加补齐不变性、梯度链、空记忆隔离、地址不匹配、SQLite重启及owner隔离5项。",
        "",
        "## 失败候选与选择依据",
        "",
        "所有候选仅根据验证集修改；最终固定选择第三候选1200步。初版01用未分工的软记忆和均值地址，验证事件地址准确率45.8%，四条读取探针最终均拒答。候选02通过字段监督让槽位保存事实，但联合LM优化破坏地址置信度：验证支持通过率从98.96%降至22.92%。候选03先训练写入/地址，再固定它们，只训练读取投影和空记忆；batch4提高到16，学习率0.0003降到0.0001。所有失败checkpoint保留。",
        "",
        f"最终验证支持通过率{val['support_pass_rate'] * 100:.2f}%，8个槽的验证分类准确率最低值为{min(val['field_accuracy'].values()) * 100:.2f}%。分类只是辅助诊断，不能替代生成测试。训练与损失细节见[固定方案及修正](QWEN_MEMORY_PROTOCOL.md)。",
        "",
        "评估脚本在保存第一个未存事件结果时发生ID字段重名错误；主测试60题已完整落盘。修复记录字段并使用--resume完成后续测试；验证原题清单与adapter SHA256后复用全部主结果，没有重训、换题或挑选输出。重启读取2/2一致。分堆操作计数未在异常前持久化，因此不报告该次运行的分裂/合并次数；持久化簇数量与最大成员数直接从数据库核验。",
        "",
        "## 不能从本轮推出的结论",
        "",
        "- 这是受控有限字段的神经记忆原型。事实值来自预先声明的8/16类，地址16个项目×16个次数，跨owner隔离；尚未解决任意人物、任意日期、长篇经历、同锚点冲突、多事件推理及自动重要性/遗忘。当前输入特征预算192 token，超长显式报错。",
        "- 仍有向量寻址，但不检索聊天原文。以用户所说的“原生长期记忆”理解，它是参与模型前向的可训练外部状态，不是已证明与生物记忆等价的新基础架构。",
        "- 反事实记忆被可信地编码后，模型可能按错误经历回答；没有来源核验就不能识破任意伪造记忆。",
        "- 0.99和0.8数值未降低；新0.8是所有生成token的最小条件概率，旧0.8是57类分类概率，不能把数值相同当成相同校准或风险保证。",
        "- 未同旧小模型做相同数据、算力和预算的严格架构对照，不能把所有改善归因于Qwen本身。",
        "",
        "## 可复查文件",
        "",
        f"- 最终run：`{run.as_posix()}`；权重`adapter.pt`。",
        "- 主测试逐题结果：`evaluation/paired_results.jsonl`；改写：`evaluation/novel_results.json`；汇总/重启证据：`evaluation/summary.json`。",
        "- 聊天逐字输出：`chat_before.json` / `chat_after.json`；训练日志：`training.jsonl`；验证：`validation_step_1200.json`。",
        f"- 官方基座来源：[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，revision `{meta['source_revision']}`。",
        f"- 最终adapter SHA256：`{s['adapter_sha256']}`。",
        "",
        "运行命令见[README](README.md)。",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(str(output.resolve()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/qwen_memory_03")
    )
    p.add_argument(
        "--output", type=Path, default=Path("experiments/episodic_memory/QWEN_MEMORY_RESULTS.md")
    )
    args = p.parse_args()
    main(args.run, args.output)
