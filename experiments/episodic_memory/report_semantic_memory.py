"""Build the comparison from saved measurements, including failed development candidate."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell(counts):
    return "/".join(str(counts.get(k, 0)) for k in ("correct", "wrong", "refused"))


def main(args):
    candidate = read(args.run / "test/summary.json")
    baseline = read(args.baseline / "summary.json")
    assert candidate["cases_sha256"] == baseline["cases_sha256"]
    meta = read(args.run / "metadata.json")
    cal = read(args.run / "calibration/calibration.json")
    regression = read(args.run / "verification/regression.json")
    restart = read(args.run / "verification/restart.json")
    chain = [meta]
    source = Path(meta["source_run"])
    while source.name.startswith("semantic_memory_"):
        previous = read(source / "metadata.json")
        chain.append(previous)
        source = Path(previous["source_run"])
    names = dict(
        normal="完整记忆",
        counterfactual="同地址替换事实",
        partial_known="其他字段缺失但所问字段存在",
        duplicate="重复且一致的记忆",
        empty="空记忆库",
        wrong_address="错误地址",
        wrong_owner="不同用户",
        missing_field="所问字段缺失",
        unsupported="超出字段范围",
        ambiguous="请求多个字段",
        no_anchor="未指定事件",
        false_premise="是非确认请求",
        conflict="同地址事实冲突",
        nonexistent="不存在的事件",
        damaged="载荷清零",
    )
    lines = [
        "# 语义读取、扩展训练与拒答测试结果",
        "\n> 历史协议分数，不能作为通用语义准确率；固定拒答格式、范围和事实被混合评价。请先读 [评分方法纠正](EVALUATION_CORRECTION.md)。\n",
        "",
        "本轮在本机训练并评估冻结 Qwen2.5-1.5B-Instruct 的新记忆模块。先保留初版开发失败，再修订问题理解与读取接口。旧模型、旧权重和旧默认入口保留。",
        "",
        "## 独立最终测试",
        "",
        "每格依次为正确/错误或不受支持输出/拒答。两个模型使用完全相同的240个问题和事件，64题有支持、176题要求拒答或超出当前单字段原型范围。最终测试只在训练与校准完成后运行，未用于选择checkpoint或调整阈值。",
        "",
        "| 条件 | 旧1.5B | 新候选 |",
        "|---|---:|---:|",
    ]
    for kind in names:
        lines.append(
            f"| {names[kind]}（16题） | {cell(baseline['conditions'][kind]['outcomes'])} | {cell(candidate['conditions'][kind]['outcomes'])} |"
        )
    lines += [
        "",
        f"有支持条件合计（64题）：旧版 {cell(baseline['positive'])}，新版 {cell(candidate['positive'])}。",
        f"拒答条件合计（176题）：旧版 {cell(baseline['negative'])}，新版 {cell(candidate['negative'])}。",
        "",
        "已暴露的上一轮60个新问法只作回归：旧版35正确、0错误、25拒答；新版 "
        + cell(regression["counts"])
        + "。这不是第二套独立盲测。",
        "旧版13次错误放行包括10次在矛盾记录中任取一条作答、3次对“是不是安心”回答“是”；这3条存储的情绪类别分别为高兴、生气、平静。逐例核查记录与剩余拒答保存在test/audit.json。",
        "",
        "## 实现变化",
        "",
        "1. 冻结基座提取问题特征，由可训练模块识别六个字段或“不支持”。用户原问法负责字段与地址判断；生成器使用预测字段的统一读取指令，读取对应4个连续token，避免其他字段和原问题措辞干扰。",
        "2. 写入器训练增加缺失字段样本；辅助事实头包含缺失类别。数据库仍只存潜变量、地址、owner与索引，没有聊天原文、答案或token序列。事实头不把解码事实文字传给Qwen。",
        "3. 新核验器要求地址证据、问题字段与潜变量事实支持度通过，并检查生成答案与潜变量预测一致。有限答案语法拒绝否定、附加事实、错字段输出；相同事实允许六种表达。措辞/标点的最低生成概率仅记录，不再充当事实正确概率。",
        "4. 发现同地址多条记录时读取有限载荷检查目标字段；冲突拒答，一致重复可答。owner隔离、空库、地址不符、缺失字段和损坏载荷各有独立拒答原因。",
        "",
        "## 数据与校准",
        "",
        "最终训练包含720组事件、1440个乱序叙述视图、6480个问题；240组事件含缺失字段。训练问法加入单字段排除干扰、多字段组合和近邻超范围问题，而不是只补之前失败的几句话。六类事实和原有有限取值词表仍保留。",
        "初版在原120题校准集上仅通过12/32个支持问题，主要因生成措辞超出核验语法；第二版在seed9205集仅10/32通过，因问题分类器过度预测不支持。两套结果保留并降级为开发集。最终版让冻结Qwen在明确字段定义的上下文中编码问题，再训练线性分类头；使用seed9207及新句式的120题作最终校准。最终测试seed9203和文件内容在修订前后完全不变。",
        f"地址门槛始终严格>0.99；新语义支持门槛由独立校准集选择为严格>{cal['threshold']}，候选范围下限0.8。语义分数是问题字段概率与对应潜变量事实概率的较小值，与旧版最低生成词概率不是同一种指标，不能仅凭数字不变声称风险相等。校准选择细节见calibration.json；风险表现应看上面的实际错答和错误放行。",
        "",
        "支持类评分沿用目标字段唯一值匹配；负例要求明确拒答。复合/是非题是此原型声明未支持的能力，不代表人类或通用模型无法回答。反事实输入应跟随注入内容，与同地址矛盾记录分开统计。",
        "",
        "## 训练与核验",
        "",
        f"首版训练3000步读写/意图监督与600步读取投影；第二版追加1500步问题模块与400步读取投影；最终版固定其他模块，训练上下文字段头1000步。三轮训练耗时合计约{sum(m['elapsed_s'] for m in chain):.1f}秒，不含特征提取和评估。",
        f"训练进程PyTorch显存峰值 allocated {max(m['peak_allocated_gb'] for m in chain):.2f}GB、reserved {max(m['peak_reserved_gb'] for m in chain):.2f}GB；不是整机显存峰值。",
        f"普通chat绕过记忆路径，训练前后{meta['chat_exact']}/{meta['chat_count']}逐字一致；独立进程重新打开磁盘库复现{sum(r['exact'] for r in restart)}/{len(restart)}（正确记忆与冲突拒答）。",
        f"测试库潜变量探针：{json.dumps(candidate['probes'], ensure_ascii=False)}。探针可解码不等于生成一定正确。",
        "27项单元测试通过，包含否定/附加事实拒绝、矛盾载荷、缺失字段、一致重复记录及语义核验与措辞概率分离；新增代码通过Ruff与格式检查。",
        "",
        "## 范围与剩余风险",
        "",
        "本轮仍有9次误拒：5次问题字段置信不足（其中3次最高概率字段本身判错），4次已生成正确事实但表达超出核验语法，均为关注字段。没有在看过测试后把这些失败表达补进允许列表，保留独立测试结果。",
        "16次矛盾记录拒答中，11次由冲突检查直接触发，2次被预测为字段缺失，3次因问题置信不足提前拒答；不能声称冲突检测器本身16/16识别正确。",
        "本轮同时改变数据、路由和核验，未做足够的单因素消融，不能把全部收益单独归因于某一个模块。训练和测试仍是合成的有限六字段任务，尚未验证新人物词表、任意自然经历、开放域多事实回答或自动在聊天与记忆模式间切换。小测试集零错误也不保证未来零错误。",
        "答案语法约束是主动限制输出范围，不是开放域语义核验器。新增事实头仍有分类误判的可能，生成器与核验器共享记忆表示，错误可能相关；需要更大独立负例集继续估计风险。",
        "",
        "## 可复现产物",
        "",
        f"- 候选运行目录：`{args.run.as_posix()}`。",
        f"- 新旧测试清单SHA256：`{candidate['cases_sha256']}`。",
        "- `semantic_suite_01`和`semantic_suite_02`保留开发过程；`semantic_suite_03`包含最终训练、独立校准与原样最终测试。",
        "- 每个评估目录有`results.jsonl`、`summary.json`和实际读取的`memories.sqlite`；校准目录另有阈值网格。",
        "- 训练日志、模型SHA256、普通聊天前后输出均在运行目录；开发失败没有删除。",
        "- 入口与复现命令见README；方案及开发修订见SEMANTIC_PROTOCOL.md。",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument(
        "--report", type=Path, default=Path("experiments/episodic_memory/SEMANTIC_RESULTS.md")
    )
    main(p.parse_args())
