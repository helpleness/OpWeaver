"""Render measured results, keeping wrong answers separate from abstention."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .generate_data import ANSWERS, TASKS


def main(comparison, run):
    root = Path(__file__).parent
    result = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((root / "runs/residual_diagnosis.json").read_text(encoding="utf-8"))
    calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    first_path = root / "runs/typed_comparison_01/comparison.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    lines = [
        "# 残余拒答与错答：字段化读取实验",
        "",
        "**结论：这版候选未通过新句式泛化验收。** 原先的问题与无支持误匹配可以改善，但新的提问方式仍会引发高置信度字段误判。不能把拒答减少当作成功，也不能因为0.99/0.8门槛不变就宣称不会增加瞎猜。本报告保留所有退化，不将候选自动替换上一版。",
        "",
        "## 旧45次拒答、13次错答的诊断",
        "",
        "13次错答全都有正确事件，真实任务的记忆槽探针也全部正确：11次问情绪却答成原因，2次日期读错。11次情绪错误中，5次问题意图首选已经错成原因，6次意图首选仍是情绪，但读取器将不确定意图放大成原因答案。2次日期错误在单独给出正确事件时仍发生，属于读取退化。探针用到了真实任务，仅作为诊断证据，不用于推理。",
        "",
        "从前两版的10错到13错不是简单增加3个样本：原先拒答的5题变错，原先正确的2题变错，原先错误的4题转为拒答，6题仍错。上轮修复证据拒答后释放了原有的字段混淆，另外产生2次日期读取退化。旧回归集是抽样问题，包含重复；新集使用不重复的(事件,任务)问题。",
        "",
        "45次拒答的互斥、按顺序诊断：31次条件答案正确但证据判断拒绝，4次内容置信不足，6次正确事件在候选中但排序首选错误，1次候选干扰，1次槽可解但读取失败，2次读取和槽探针同时失败。后两种同时失败不能证明信息已经从编码中消失。",
        "",
        "## 修改与约束",
        "",
        "冻结写入器和查询地址编码器，沿用原SQLite文件与分堆。新增独立问题意图编码器；读取现有潜在槽的各字段分布，计算 P(字段|问题) × Σ事件 P(事件|查询)P(字段值|事件槽)。这使最终答案置信不超过对应字段意图置信，并保留候选事件的歧义；不会把不确定的情绪/原因判断再放大成高置信度输出。",
        "",
        "训练地址语义残差分类器来修正项目/次数匹配，训练槽解码器、意图编码器和证据验证器。训练只用train事实、原始和两份训练改写库，以及单列的训练问题结构。推理没有真实字段ID、事件ID或答案标签，也不解析原始历史。六字段与16×16地址监督是本原型的人工任务结构，不能称为无监督、开放领域记忆。",
        "",
        "第一阶段在新第一套同时改写中将99拒答/126错变为77拒答/10错，第二套11拒答/21错变为7拒答/0错；但屏蔽支持480题新增2次误答，原始句式的拒答也从2增至11。这一失败没有隐藏。首轮新测试从此作为开发集，独立最终复测换用另一组预先固定的事件种子和完整句式。",
        "",
        "第二阶段从问题编码器直接预测语义地址，证据分数限制为不高于项目分布一致度×次数分布一致度。这样验证器不能将模糊或矛盾的地址识别放大到0.99以上。训练增加只基于train事实的地址监督和情绪/原因最小对比问句；在第一阶段checkpoint上继续固定3000步。",
        "",
        f"上一版/本版门槛：{result['thresholds']}。校准前后证据门槛{calibration['old_threshold']}→{calibration['new_threshold']}；内容门槛{calibration['value_threshold']}。最大一次重试、每次最多8个内容载荷，与基线一致。门槛没有降低，但置信度函数改变，所以仍须看独立错误率，不能仅凭门槛数字声称安全性不变。",
        "",
        f"正例验证：{calibration['positive_validation']}。1800负例验证问题、3600次尝试，校准后错误接受{calibration['false_accepts_after']}次。",
        "",
        "## 新事件与全新完整句式",
        "",
        f"最终两个新事件种子{result.get('seeds')}，完整句式版本{result.get('view_offset', 0)}、{result.get('view_offset', 0) + 1}；每组8人×96条已存事件+每人8条未存事件。每条件480个已存问题、120个未存问题；按事件和字段去重。两组使用不同新叙述和新问题句式，未用于训练或校准。新事实仍来自已声明的实体、答案和地址词表。四条件在第一组保持事实和问题目标配对。",
        "",
        "| 条件 | 旧：正确/拒答/错答 | 新：正确/拒答/错答 | 旧→新正确率 | 未存误答：旧→新 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result["conditions"].items():
        a, b = [values[n]["counts"] for n in ("previous", "typed")]
        lines.append(
            f"| {name} | {a['correct']}/{a['unknown']}/{a['wrong']} | {b['correct']}/{b['unknown']}/{b['wrong']} | {a['correct'] / a['present']:.2%}→{b['correct'] / b['present']:.2%} | {a['false_answer']}→{b['false_answer']} / {a['unsupported']} |"
        )
    lines += [
        "",
        "## 负例压力测试",
        "",
        "| 条件 | 问题数 | 旧错误作答 | 新错误作答 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in result["negatives"].items():
        a, b = [values[n]["counts"] for n in ("previous", "typed")]
        lines.append(f"| {name} | {a['unsupported']} | {a['false_answer']} | {b['false_answer']} |")
    lines += [
        "",
        "removed在初次/重试检索都排除真实支持，不是物理删除并重算簇；zero将已读取载荷清零。unobserved为实际从未写入的新事件。多个任务共享同一事件，不能当成完全独立样本计算零风险保证。",
        "",
        "## 第一阶段保留的固定组件消融（90381，现为开发集）",
        "",
        "| 配置 | 正确 | 拒答 | 错答 | 未存误答 |",
        "|---|---:|---:|---:|---:|",
    ]
    ablations = {
        "previous": first["conditions"]["fresh0_both"]["previous"],
        **first["ablations"],
        "stage1_full": first["conditions"]["fresh0_both"]["typed"],
    }
    for name, values in ablations.items():
        c = values["counts"]
        lines.append(
            f"| {name} | {c['correct']} | {c['unknown']} | {c['wrong']} | {c['false_answer']} |"
        )
    lines += [
        "",
        "typed_only使用新字段/意图读取器、旧地址语义和旧验证器；context_only使用旧读取器/旧意图、新地址语义和新验证器。这是固定权重的组件替换，不是各自独立重训，不能将交互收益简单相加。",
        "",
        "## 旧回归集复测（开发集，不是独立泛化证据）",
        "",
    ]
    reg = result["regression"]
    for name in ("previous", "typed"):
        lines.append(f"- {name}: {dict(reg[name]['counts'])}")
    transition = Counter()
    for a, b in zip(reg["previous"]["records"], reg["typed"]["records"]):
        if not a["supported"]:
            continue

        def state(row):
            return (
                "拒答"
                if row["prediction"] == 0
                else "正确"
                if row["prediction"] == row["target"]
                else "错答"
            )

        transition[state(a) + "→" + state(b)] += 1
    lines += ["", f"配对转移：{dict(transition)}。", "", "### 新测试仍存在的错答", ""]
    wrong_count = 0
    wrong_causes = {}
    samples = []
    for condition, values in result["conditions"].items():
        causes = Counter()
        for r in values["typed"]["records"]:
            if r["supported"] and r["prediction"] not in (0, r["target"]):
                wrong_count += 1
                a = r["attempts"][-1]
                cause = (
                    f"字段误判:{r['task']}→{TASKS[a['predicted_field']]}"
                    if TASKS[a["predicted_field"]] != r["task"]
                    else "事件选择错误"
                    if a["selected_id"] != r["id"]
                    else "字段和事件正确但值读错"
                )
                causes[cause] += 1
                if len(samples) < 12:
                    samples.append(
                        f"- {condition} / {r['id']} / {r['task']}: 应为{ANSWERS[r['target']]}，回答{ANSWERS[r['prediction']]}；预测字段{TASKS[a['predicted_field']]}，字段置信{a['field_probability']:.4f}，事件选择正确={a['selected_id'] == r['id']}，内容置信{a['value_probability']:.4f}。"
                    )
        wrong_causes[condition] = dict(causes)
    lines += [f"- {name}: {causes}。" for name, causes in wrong_causes.items()]
    lines += ["", "前12条失败示例（完整列表在原始JSON）：", "", *samples]
    if not wrong_count:
        lines.append("本轮新测试未出现已存事件错答；这不代表开放对话没有错误。")
    lines += [
        "",
        "固定门槛不能解决分布外高置信度错误。显式边缘化能限制低置信度意图被读取器放大，但如果意图编码器本身就自信判错，后续字段概率仍会自信地回答错误字段。数据扩充与地址辅助监督没有建立稳定的自然语言任务理解；不能仅从训练/验证低损失判断泛化成功。最终保留失败，没有继续按最终句式调参。",
    ]
    lines += ["", "### 剩余拒答的直接分解", ""]
    for name, values in {**result["conditions"], "old_regression": reg}.items():
        counts = Counter()
        for r in values["typed"]["records"]:
            if not r["supported"] or r["prediction"]:
                continue
            a = r["attempts"][-1]
            label = (
                "未检索到"
                if not r["support_found"]
                else "候选中有支持但排序错误"
                if a["selected_id"] != r["id"]
                else "正确事件的证据不足"
                if a["evidence"] <= result["thresholds"]["typed"]["evidence"]
                else "内容或字段置信不足"
            )
            counts[label] += 1
        lines.append(f"- {name}: {dict(counts)}。")
    if "stage1_regression" in result:
        lines += ["", "### 第一阶段误匹配复查（开发集）", ""]
        for label, data in result["stage1_regression"].items():
            lines.append(f"- {label}: 新版{dict(data['typed']['counts'])}。")
    restart_path = run / "restart_verification_template3.json"
    if restart_path.exists():
        restart = json.loads(restart_path.read_text(encoding="utf-8"))
        lines += [
            "",
            f"跨进程旧示例：全部答对={restart['all_correct']}，重复新进程完全一致={restart['exact_across_fresh_processes']}。这只证明该示例的落盘与读取链条，不能抵消新句式的失败。",
        ]
    lines += [
        "",
        "## 运行与可复现证据",
        "",
        f"GPU={metadata['gpu']}；参数{metadata['parameters']:,}；本阶段3000步、batch32；本阶段训练与验证{metadata['elapsed_s']:.1f}秒（不含第一阶段、新测试和建库）。writer指纹={metadata['writer_version']}，冻结原查询地址编码器的逐参数一致性检查通过。12项单元测试覆盖不确定性传播、候选顺序/填充不变性、门槛、隔离、落盘和词表等。",
        "",
        "本轮有一次根据第一阶段测试失败进行的公开修正，不是一次预注册实验成功。最终复测之前固定追加训练步数、两套新句式、两组新事件和门槛，最终复测后没有再调参。只有一个训练种子；分组地址仍常驻CPU、重试扫描当前owner地址，这不是无限上下文或大规模磁盘索引结论。",
        "",
        f"- [完整预测]({(comparison / 'comparison.json').resolve().as_posix()})",
        f"- [第一阶段完整结果（含失败）]({first_path.resolve().as_posix()})",
        f"- [逐条残余诊断]({(root / 'runs/residual_diagnosis.json').resolve().as_posix()})",
        f"- [模型]({(run / 'model.pt').resolve().as_posix()})",
        f"- [校准]({(run / 'calibration.json').resolve().as_posix()})",
        "- [预定方案](RESIDUAL_PROTOCOL.md)",
    ]
    assert len([r for r in diagnosis["records"] if r["bucket"] == "wrong"]) == 13
    path = root / "RESIDUAL_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    main(a.comparison, a.run)
