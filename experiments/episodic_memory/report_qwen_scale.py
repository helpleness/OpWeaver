"""Report fixed scale experiment, separating oracle text from deployed latent reading."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def counts(item):
    return "/".join(str(item.get(k, 0)) for k in ("correct", "wrong", "unknown"))


def main(args):
    old = read(args.old / "metadata.json")
    new = read(args.new / "metadata.json")
    regress = [read(p / "evaluation/summary.json") for p in (args.old, args.new)]
    names = ["qwen_0_5b", "qwen_1_5b"]
    fresh = [read(args.suite / name / "summary.json") for name in names]
    rows = [
        [
            json.loads(s)
            for s in (args.suite / name / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in names
    ]
    assert fresh[0]["selection_sha256"] == fresh[1]["selection_sha256"]
    assert [(r["id"], r["task"], r["wording"], r["question"]) for r in rows[0]] == [
        (r["id"], r["task"], r["wording"], r["question"]) for r in rows[1]
    ]
    paired = {}
    for wording in ("original", "novel"):
        paired[wording] = {}
        for mode in ("text", "fields", "memory", "oracle_memory"):
            subset = [(a, b) for a, b in zip(*rows) if a["wording"] == wording]
            paired[wording][mode] = dict(
                improved=sum(
                    a[mode]["raw_status"] != "correct" and b[mode]["raw_status"] == "correct"
                    for a, b in subset
                ),
                regressed=sum(
                    a[mode]["raw_status"] == "correct" and b[mode]["raw_status"] != "correct"
                    for a, b in subset
                ),
                both_correct=sum(
                    a[mode]["raw_status"] == "correct" and b[mode]["raw_status"] == "correct"
                    for a, b in subset
                ),
            )
    (args.suite / "paired_changes.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    modes = dict(
        text="正确事件文字（oracle）",
        fields="正确字段列表（oracle）",
        memory="正常向量检索与读取",
        oracle_memory="指定正确事件的向量读取（oracle）",
        empty="无记忆",
    )
    left, right = (s["conditions"]["novel"] for s in fresh)
    rejected = [
        r for r in rows[1] if r["wording"] == "novel" and r["memory"]["final_status"] == "unknown"
    ]
    correct_rejected = sum(r["memory"]["raw_status"] == "correct" for r in rejected)
    empty_guesses = [
        r for r in rows[1] if r["wording"] == "novel" and r["empty"]["raw_status"] == "wrong"
    ]
    lines = [
        "# Qwen2.5：0.5B 与 1.5B 磁盘记忆对照",
        "",
        "在RTX4060 Laptop 8GB上完成。两者均为官方Instruct模型；1.5B重新提取特征、训练适配器并重建数据库。主干冻结，部署ask仍只读连续记忆，不读取历史原文。",
        "",
        f"本轮新事件、新问法结果（0.5B→1.5B）：向量原始正确数{left['memory']['raw_status'].get('correct', 0)}→{right['memory']['raw_status'].get('correct', 0)}/60，最终正确数{left['memory']['final_status'].get('correct', 0)}→{right['memory']['final_status'].get('correct', 0)}/60，拒答{left['memory']['final_status'].get('unknown', 0)}→{right['memory']['final_status'].get('unknown', 0)}，最终未匹配回答{left['memory']['final_status'].get('wrong', 0)}→{right['memory']['final_status'].get('wrong', 0)}。正确寻址且通过证据门槛{fresh[0]['retrieval']['novel']['correct_accepted']}→{fresh[1]['retrieval']['novel']['correct_accepted']}/60。不能仅凭原始正确数上升就称整体可用率提高。",
        "",
        f"新问法的文字能力对照（严格匹配）：自然事件文字{left['text']['raw_status'].get('correct', 0)}→{right['text']['raw_status'].get('correct', 0)}/60，清晰字段列表{left['fields']['raw_status'].get('correct', 0)}→{right['fields']['raw_status'].get('correct', 0)}/60。文字路径中有同义表述被漏判，详见后文核查；不能把这些数字直接当成通用语言理解率。仅一个种子和有限事件词表，不能证明更大模型普遍更好或更差。",
        "",
        "## 新事件、新问法：主要对照",
        "",
        "seed8101生成新事件，seed8102每字段固定10题，共60题。每题同时测原问法和新问法；所有题目在1.5B训练结果出现前固定。两个模型接收同样事实、问题与记忆库；没有根据测试调参。",
        "",
        "表格按“正确/未匹配/拒答”计数。正确使用预先固定的目标字段唯一值匹配；未匹配可能包括同义改写及不同措辞的拒答，不能全部解释成语义错误。门控前后分开，避免把输出措辞和置信度变化当作阅读能力变化。两者均使用贪心生成、40个新token上限；未正常结束的生成置信度记为0。",
        "",
        "| 问法 | 输入路径 | 0.5B 原始生成 | 1.5B 原始生成 | 0.5B 门控后 | 1.5B 门控后 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for wording, label in [("original", "原问法"), ("novel", "新问法")]:
        for mode, name in modes.items():
            a, b = (s["conditions"][wording][mode] for s in fresh)
            lines.append(
                f"| {label} | {name} | {counts(a['raw_status'])} | {counts(b['raw_status'])} | {counts(a['final_status'])} | {counts(b['final_status'])} |"
            )
    lines += [
        "",
        "文字与字段列表是能力诊断：评估器直接提供正确事实，不执行检索。它们不会进入正式ask接口。无记忆最终拒答含外部门控，原始生成单列。oracle向量只诊断读取上限，不计为正常部署成功率；当正常检索已正确且接受时，两条路径输入相同，复用同一确定性输出。",
        "",
        f"1.5B新问法下的{len(rejected)}次有记忆拒答中，{correct_rejected}次原始内容已经匹配正确。空记忆的新问法还有{len(empty_guesses)}次原始输出既非标准拒答，也未匹配目标事实，最终由门控拒答。具体输出：{'；'.join(r['empty']['raw'] + '（最低token概率' + format(r['empty']['confidence'], '.4f') + '）' for r in empty_guesses) or '无'}。阈值未降低，但生成token的确定性不是事实正确性的校准概率。",
        "",
        "## 新集寻址与字段诊断",
        "",
        "| 模型 | 问法 | 首位事件正确 | 正确且通过0.99 | 错地址被接受 |",
        "|---|---|---:|---:|---:|",
    ]
    for label, s in zip(("0.5B", "1.5B"), fresh):
        for wording in ("original", "novel"):
            r = s["retrieval"][wording]
            lines.append(
                f"| {label} | {wording} | {r['top1_correct']}/60 | {r['correct_accepted']}/60 | {r['wrong_accepted']}/60 |"
            )
    lines += [
        "",
        "新问法下，正常向量读取按字段（正确/未匹配/拒答）：",
        "",
        "| 字段 | 0.5B | 1.5B |",
        "|---|---:|---:|",
    ]
    for task in fresh[0]["by_field"]["novel"]:
        lines.append(
            f"| {task} | {counts(fresh[0]['by_field']['novel'][task])} | {counts(fresh[1]['by_field']['novel'][task])} |"
        )
    lines += [
        "",
        "同题原始输出的变化（严格匹配）：",
        "",
        "| 新问法输入 | 原来不匹配→现在正确 | 原来正确→现在不匹配 | 两者都正确 |",
        "|---|---:|---:|---:|",
    ]
    for mode, p in paired["novel"].items():
        lines.append(
            f"| {modes[mode]} | {p['improved']} | {p['regressed']} | {p['both_correct']} |"
        )
    lines += [
        "",
        "## 已暴露旧集：回归对照",
        "",
        "旧题用于连续性验证，明确不是本轮新盲测。每种主条件60题；反事实行按注入的新事实评分。",
        "",
        "| 条件 | 0.5B 门控后 | 1.5B 门控后 | 0.5B 原始生成 | 1.5B 原始生成 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in [
        ("normal", "真记忆"),
        ("no_memory", "无记忆"),
        ("counterfactual", "同场景替换事实"),
        ("unrelated", "场景不符记忆"),
        ("oracle", "正确事件ID诊断"),
    ]:
        a, b = (s["conditions"][mode] for s in regress)
        lines.append(
            f"| {label} | {counts(a['final_status'])} | {counts(b['final_status'])} | {counts(a['raw_status'])} | {counts(b['raw_status'])} |"
        )
    old_novel = read(args.old / "evaluation/novel_results.json")
    new_novel = {(r["id"], r["task"]): r for r in read(args.new / "evaluation/novel_results.json")}
    for r in old_novel:
        if r["question_only"]["final_status"] != "wrong":
            continue
        n = new_novel[(r["id"], r["task"])]
        lines += [
            "",
            f"旧高置信失败实例：问题“{r['question']}”，正确事实“{r['expected']}”。0.5B原始输出“{r['question_only']['raw']}”，最低token概率{r['question_only']['confidence']:.4f}；1.5B原始输出“{n['question_only']['raw']}”，最低token概率{n['question_only']['confidence']:.4f}，最终输出“{n['question_only']['answer']}”。",
        ]
    lines += [
        "",
        "旧集12题改写：",
        "",
        "| 条件 | 0.5B 门控后 | 1.5B 门控后 | 0.5B 原始生成 | 1.5B 原始生成 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, label in [
        ("question_only", "只换提问"),
        ("event_only", "只换事件"),
        ("both", "同时改写"),
    ]:
        a, b = (s["novel"][mode] for s in regress)
        lines.append(
            f"| {label} | {counts(a['final_status'])} | {counts(b['final_status'])} | {counts(a['raw_status'])} | {counts(b['raw_status'])} |"
        )
    lines += [
        "",
        f"1.5B未存事件最终拒答{regress[1]['missing_final_refused']}/12，原始拒答{regress[1]['missing_raw_refused']}/12；真记忆与反事实两边均正确跟随内容{regress[1]['paired_correct_then_counterfactual_correct']}/60。",
        "",
        "## 训练、内存、聊天与持久化",
        "",
        "| 指标 | 0.5B | 1.5B |",
        "|---|---:|---:|",
    ]
    for label, key, fmt in [
        ("基座参数", "backbone_parameters", ",d"),
        ("适配器总参数", "adapter_parameters", ",d"),
        ("读取阶段可训练参数", "trainable_parameters", ",d"),
        ("训练时间（秒）", "elapsed_s", ".1f"),
        ("训练进程CUDA allocated峰值GB", "peak_allocated_gb", ".2f"),
        ("训练进程CUDA reserved峰值GB", "peak_reserved_gb", ".2f"),
    ]:
        lines.append(f"| {label} | {format(old[key], fmt)} | {format(new[key], fmt)} |")
    lines += [
        "",
        f"新模型普通聊天前后{new['chat_exact']}/{new['chat_count']}逐字一致；独立进程重启真实/反事实数据库复现{sum(r['identical_to_parent'] for r in regress[1]['restart_proofs'])}/2。普通chat显式绕过记忆模块，不代表已经实现聊天时自动判断回忆或带记忆的任意聊天无退化。",
        "",
        "两者训练数据、训练问法、有效batch16、2000步写入预训练与1200步读取训练一致，学习率与损失不变。新特征缓存按模型revision及维度验证，旧缓存未误用。输入隐藏维度896→1536导致适配器参数数量改变；因此不能把所有变化严格归因于主干参数量。只跑一个种子，未做显著性估计。",
        "",
        "1.5B训练期间运行了旧模型的新集推理；时间包含共享GPU负载影响，不作为吞吐性能对比。显存指标是训练进程PyTorch统计，不是整个GPU或整机内存总峰值。",
        "",
        "规则始终为地址证据>0.99与所有生成token最小条件概率>0.8；不同模型概率未经过相同风险校准。读写架构仍限定六类事实、16×16场景地址、单支持事件与192-token特征预算。分堆和SQLite只保存地址与潜变量，没有历史文本/token/答案列；所有地址目前仍缓存CPU。",
        "",
        "## 评分核查",
        "",
        "严格匹配有已确认的局限，例如‘时间不够’未匹配‘时间不足’，‘注意自己表现’未匹配‘自身表现’。另有‘没有这段经历的记忆’因少了‘我’而被列入未匹配，而非标准拒答。部分长回答虽然包含目标事实的同义表述，却还编造了其他细节，不能简单改判整条正确。逐例记录保存在比较目录的semantic_audit.json；这是本助手对保存结果的核查，不是独立盲评。固定评分不因后验核查修改。",
        "",
        "## 文件与复现",
        "",
        f"- 最终1.5B运行目录：`{args.new.as_posix()}`；`adapter.pt`为最终权重。",
        f"- 1.5B官方模型：[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `{new['source_revision']}`。",
        f"- 新适配器SHA256：`{new['adapter_sha256']}`。",
        f"- 新集清单与配对结果：`{args.suite.as_posix()}`，每模型`results.jsonl`、`summary.json`。",
        "- 方案：[QWEN_SCALE_PROTOCOL.md](QWEN_SCALE_PROTOCOL.md)；运行命令：[README](README.md)。",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--old", type=Path, default=Path("experiments/episodic_memory/runs/qwen_memory_03")
    )
    p.add_argument(
        "--new", type=Path, default=Path("experiments/episodic_memory/runs/qwen_1_5b_memory_01")
    )
    p.add_argument(
        "--suite",
        type=Path,
        default=Path("experiments/episodic_memory/runs/qwen_scale_comparison_01"),
    )
    p.add_argument(
        "--output", type=Path, default=Path("experiments/episodic_memory/QWEN_SCALE_RESULTS.md")
    )
    main(p.parse_args())
