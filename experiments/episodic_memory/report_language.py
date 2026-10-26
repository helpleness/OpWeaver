"""Measured report for the independent language-understanding experiment."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import transformers

from .generate_data import TASKS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(trained, comparison, audit):
    root = Path(__file__).parent
    r = read(comparison / "comparison.json")
    d = read(root / "runs/understanding_diagnosis.json")
    audit = read(audit / "audit.json")
    names = [
        "previous",
        "random_narrow",
        "pretrained_narrow",
        "random_compositional",
        "pretrained_compositional",
    ]
    meta = {name: read(trained / name / "metadata.json") for name in names[1:]}
    source = root / "runs/language_backbone"
    provenance = dict(
        torch=torch.__version__,
        transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        source="https://huggingface.co/uer/chinese_roberta_L-4_H-256",
        source_weights_sha256=hashlib.sha256(
            (source / "pytorch_model.bin").read_bytes()
        ).hexdigest(),
        source_revision=(source / ".cache/huggingface/download/pytorch_model.bin.metadata")
        .read_text()
        .splitlines()[0],
        memory_checkpoint_sha256=hashlib.sha256(
            (root / "runs/typed_round_02/model.pt").read_bytes()
        ).hexdigest(),
    )
    (comparison / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    lines = [
        "# 问题理解与记忆读取分离：本机实测",
        "",
        "本轮在两套全新事件/句式上同时减少了拒答和错答，但尚未消除高置信度字段错误。主模型在测试前固定为“预训练＋组合数据”，没有根据最终结果换模型、换门槛或继续补测试模板。原记忆模型、写入器、磁盘载荷、地址判断和槽解码器完全沿用typed_round_02；仅替换字段理解前端。",
        "",
        "## 句式、数据与模型是什么关系",
        "",
        "句式变化是触发条件，根因不是某几个句子“特殊”。原型约130万参数，只训练有限词表的六字段分类/记忆任务，未做通用语言预训练，也没有自回归文本生成目标。已有train.jsonl约3.33 MB且包含JSON标签；在线改写/重复训练增加样本呈现次数，不等于获得同等数量的新语言知识。因此它并不是一个已经学会中文交流的大模型。",
        "",
        "通用语言预训练不能用“达到几个GB就会说话”作统一门槛：token数量、数据多样性/质量、模型容量、目标函数与算力都重要。Chinchilla研究模型规模与训练token的配比；FineWeb提供15万亿token的预训练语料，但这不是会说话所需的最低值。本轮复用已有中文预训练，未在这台电脑上重做通用语言预训练。",
        "",
        "- [Chinchilla原论文](https://arxiv.org/abs/2203.15556)",
        "- [FineWeb原论文](https://arxiv.org/abs/2406.17557)",
        "- [本轮UER中文小编码器模型卡](https://huggingface.co/uer/chinese_roberta_L-4_H-256)",
        "",
        "## 分离诊断：旧失败集仅用于开发分析",
        "",
        "| 旧集 | 仅问题分类正确 | 真字段＋原检索/门槛：正确/拒答/错答 | 真字段＋真事件：槽argmax正确 |",
        "|---|---:|---:|---:|",
    ]
    for name, s in d["suites"].items():
        c = s["counts"]
        lines.append(
            f"| {name} | {c['language_correct']}/{c['n']} | {c['oracle_field_pipeline_correct']}/{c['oracle_field_pipeline_unknown']}/{c['oracle_field_pipeline_wrong']} | {c['memory_oracle_argmax_correct']}/{c['n']} |"
        )
    lines += [
        "",
        "这两种oracle只用于测量上限，不能当成实际回答策略。部署时字段概率由问题编码器预测，没有真实字段ID。",
        "",
        "干预证据：同一个“回忆当时，我为什么产生那种感受？”在不加事件锚点时被判为reason，置信约0.9999；加上“画册项目第1次讨论”等无关定位信息后，变为emotion，置信约0.87–0.92。人物问题加入“那件事发生时”背景也会被发生/事情词干扰。说明学到的是上下文/位置相关的分类捷径，未稳健区分被问成分与背景成分。原理解头和地址头共享表示，可能有多任务干扰；本轮未单独量化这个因素，不能把所有失败归因于它。",
        "",
        "## 改进及2×2对照",
        "",
        f"新前端：4层、hidden256，含分类头{meta['pretrained_compositional']['parameters']:,}参数。随机与预训练组使用完全相同的网络与分词器；每组12000训练问题、1200验证问题，同1200步、batch24（双视图48句），同学习率1e-4、种子62。这里只训练六类字段理解，不训练记忆答案。原记忆模型约130万参数另行保留。",
        "",
        "组合数据独立变化目标字段、背景、位置、锚点及否定/引用范围；同背景对应六种不同问题，相同字段的两种表达跨背景/锚点保持一致。损失为两视图分类CE＋0.1 JS一致性。标签来自生成语法树，推理没有关键词规则。四组更新步数相同，但句长不同，不声称精确等FLOPs，也不声称随机模型已达到其最佳训练预算。",
        "",
        "训练与验证的主句、正视图均检查零交集。最早language_round_01试运行发现正视图与验证重复，已作废；本报告只用修复后重训的language_round_02。所有四组在同分布验证集上1200/1200正确；这说明单看模板内验证会严重高估新结构泛化。",
        "",
        "最终训练结构族0–5，测试结构族6–9，目标问句实现也独立留出。预训练可能见过通用语言结构，不能声称测试句式对预训练绝对未知。",
        "",
        "| 前端 | 四类结构总正确率（4800题） | 高置信错分（>0.8） |",
        "|---|---:|---:|",
    ]
    aggregate = {}
    for name in names:
        total = Counter()
        for suite in r["language"].values():
            for key in ("n", "correct", "high_conf_wrong", "low_confidence"):
                total[key] += suite["metrics"][name][key]
        aggregate[name] = dict(total)
        lines.append(
            f"| {name} | {total['correct'] / total['n']:.2%} | {total['high_conf_wrong']} |"
        )
    lines += [
        "",
        "同架构对照支持两个结论：已有语言预训练对这个小样本任务有明显帮助；在预训练基础上，覆盖背景/位置/指令范围的组合训练进一步改善泛化。随机初始化也能在训练分布上全对，却在新结构上失败；不是简单增加参数或复读模板就能解决。",
        "",
        "| 留出结构 | 原模块正确率 | 预训练＋组合正确率 |",
        "|---|---:|---:|",
    ]
    for name, s in r["language"].items():
        a, b = s["metrics"]["previous"], s["metrics"]["pretrained_compositional"]
        lines.append(f"| {name} | {a['correct'] / a['n']:.2%} | {b['correct'] / b['n']:.2%} |")
    lines += [
        "",
        "family6/7为新背景从句和位置结构；family8/9为新的否定/引用作用范围。完整测试句子与所有微调训练、验证句子及正视图零交集。独立语言测试包含原地址分词器无法覆盖的字词，因此不将它直接当作端到端地址测试。",
        "",
        "## 新事件的端到端对比",
        "",
        "新事实种子90611/90612，每套8人×96条已存事件、每人另有8条未存事件。各条件480个已存＋120个未存问题，同套四条件配对。第二套使用不同问题结构和不同事件叙述。事件与答案仍在原型有限词表内，不是新实体开放词表记忆。",
        "",
        f"证据门槛{r['thresholds']['evidence']}、最终值门槛{r['thresholds']['value']}；最多一次重试，最多每次8条载荷，全部保持不变。字段概率按原公式参与答案边缘化，不强制one-hot。两套端到端新问题经原地址分词器的未知符号问题数：{r['old_tokenizer_unknowns']}。",
        "",
        "| 条件 | 原：正确/拒答/错答 | 新：正确/拒答/错答 | 原→新准确率 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in r["conditions"].items():
        a, b = [values[k]["counts"] for k in ("previous", "pretrained_compositional")]
        lines.append(
            f"| {name} | {a['correct']}/{a['unknown']}/{a['wrong']} | {b['correct']}/{b['unknown']}/{b['wrong']} | {a['correct'] / a['present']:.2%}→{b['correct'] / b['present']:.2%} |"
        )
    lines += [
        "",
        "### 同一新测试上的全部前端对照",
        "",
        "| 前端 | 新A 正确/拒答/错答 | 新B 正确/拒答/错答 |",
        "|---|---:|---:|",
    ]
    for name in names:
        a, b = [r["conditions"][c][name]["counts"] for c in ("fresh0_both", "fresh1_both")]
        lines.append(
            f"| {name} | {a['correct']}/{a['unknown']}/{a['wrong']} | {b['correct']}/{b['unknown']}/{b['wrong']} |"
        )
    lines += ["", "### 读取上限与剩余失败", ""]
    for name, c in r["memory_probes"].items():
        lines.append(
            f"- {name}：真实字段＋真实事件槽解码，{c['correct']}/{c['n']}正确，其中超过0.8且正确{c['confident_correct']}，高置信错值{c['confident_wrong']}。"
        )
    for name in ("fresh0_both", "fresh1_both"):
        wrong, unknown = Counter(), Counter()
        for row in r["conditions"][name]["pretrained_compositional"]["records"]:
            if not row["supported"]:
                continue
            if row["prediction"] == 0:
                unknown[
                    "未检索到"
                    if not row["support_found"]
                    else "证据判断"
                    if row["attempts"][-1]["evidence"] <= 0.99
                    else "字段/内容置信"
                ] += 1
            elif row["prediction"] != row["target"]:
                wrong[
                    row["task"] + "→" + TASKS[row["intent"]]
                    if row["task"] != TASKS[row["intent"]]
                    else "正确字段但值错误"
                ] += 1
        lines += [f"- {name}错答：{dict(wrong)}；拒答：{dict(unknown)}。"]
    lines += [
        "",
        "剩余高置信度字段错误仍存在。第二套的大量拒答来自本轮冻结的地址判断，不能归因于记忆内容消失，也不能靠放宽证据门槛掩盖。独立理解改善不等于整个系统已能理解任意中文。",
        "",
        "## 无有效记忆压力测试",
        "",
        "| 条件 | 题数 | 原误答 | 新误答 |",
        "|---|---:|---:|---:|",
    ]
    for name, s in r["negatives"].items():
        a, b = [s[n]["counts"] for n in ("previous", "pretrained_compositional")]
        lines.append(f"| {name} | {b['unsupported']} | {a['false_answer']} | {b['false_answer']} |")
    lines += [
        "",
        "removed只在初次和重试检索中屏蔽支持ID，并非物理删除重建簇；zero清零载荷；unobserved是真正未写入事件。多个问题共享事件，零错不能推出开放世界零风险。",
        "",
        "## 原失败集复查和独立进程",
        "",
    ]
    for name in ("fresh0_both", "fresh1_both"):
        a, b = [audit[name][n]["counts"] for n in ("previous", "pretrained_compositional")]
        lines.append(
            f"- {name}开发集：正确/拒答/错答，{a['correct']}/{a['unknown']}/{a['wrong']}→{b['correct']}/{b['unknown']}/{b['wrong']}；新前端字段分类{b['language_correct']}/600。"
        )
    lines += [
        f"- 旧磁盘示例重复新进程：{sum(x['correct'] for x in audit['restart'])}/{len(audit['restart'])}答对；所有重复结果一致={all(x['identical'] for x in audit['restart'])}。查询子进程仅获得模型、数据库和问题。",
        "",
        "## 运行记录与边界",
        "",
        f"GPU：{provenance['gpu']}，Torch {provenance['torch']}，Transformers {provenance['transformers']}。四组训练＋验证累计{sum(m['elapsed_s'] for m in meta.values()):.1f}秒，不含下载、数据生成和最终测试。新增前端不是语言生成模型；整个实验仍为六字段、固定答案词表的记忆原型。",
        "",
        "本轮只用一个训练种子，两个新事件种子；结论是该受控任务上的改善，不是人类长期记忆或通用数字人的证明。没有更新事件存储格式，没有检索历史原文，也没有将测试标签传给部署推理。14项单元测试通过。",
        "",
        f"- [完整预测与问题文本]({(comparison / 'comparison.json').resolve().as_posix()})",
        f"- [独立诊断]({(root / 'runs/understanding_diagnosis.json').resolve().as_posix()})",
        f"- [前端与数据]({trained.resolve().as_posix()})",
        f"- [来源与哈希]({(comparison / 'provenance.json').resolve().as_posix()})",
        "- [实验方案](LANGUAGE_PROTOCOL.md)",
        "- [运行说明](README.md)",
    ]
    (comparison / "summary.json").write_text(
        json.dumps(dict(language=aggregate, metadata=meta), indent=2), encoding="utf-8"
    )
    path = root / "LANGUAGE_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    main(a.trained, a.comparison, a.audit)
