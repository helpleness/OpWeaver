"""Blind model-judge packets for causal checkpoints; relevant evidence is judge-only."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .luna_review import RUBRIC, digest, dump_rows, read_rows, validate

STREAM_RUBRIC = (
    RUBRIC.replace(
        "同一事件的冲突记录无优先级，不可任选一条。",
        "明确更正且声明旧说法作废时，应采用更正后的字段；仅新增未经核实的矛盾时不能任选一条，不能把较晚自动当正确。",
    )
    + """
本轮是连续长聊天检查点实验。evidence只摘录该条件当时可见历史中与所问事件同项目、同讨论次数的全部用户片段，按照轮次排列；更正和矛盾都保留。其他事件不能证明所问事件的事实，不得借用别条评审中的记录。没有相关片段时用no_matching_event；如果提供片段但目标字段缺少，用missing_field。
这些摘录来自评估用完整原文日志，不是给被测记忆模型的提示词。记忆模型只访问磁盘向量，摘录用于独立检验它是否记住/读对。不要把“相关事件的证据存在”自动当“系统成功读取”。
某些回复是answer:null附带系统状态说明，已经在answer显示为“未提供事实回答。系统状态：…”。这不是语言模型生成拒答句，behavior可为abstain/clarify，但仍须按实际证据判断是否未回答可答问题。
统一口径：当证据本就不足，准确说明不确定且不猜测，whole_reply=acceptable，target=not_answered；不要仅因没有具体事实就标honest_but_incomplete。证据足够但实现能力限制而没回答，才可用honest_but_incomplete。证据冲突而回复不选边、说明冲突时，availability=conflicting，target=not_answered；如果回复任选一边用target=unresolved_conflict。把这些标签分别当证据状态和回复完成情况，避免混淆。
逐项claims若判unsupported或contradicted，请检查whole_reply与event_claim_problem/status_problem是否相应，不能分析发现错误却汇总没有问题。若认为该断言无实质影响，必须解释理由。
上面的未回答规则适用于问题要求具体事实、而事实缺失或未决的情况。先理解实际问题：如果问题问是否记录、是否能确定等是非判断，正确表达记录缺失或尚无法确定可以是target=correct、behavior=answer，而不是拒答。若证据明确充分却回答不能确定，需标出证据状态判断错误。不要因没有姓名或情绪词就机械套用not_answered。
真实但无关的字段不是所问目标的部分正确；明确把该字段断言为错误的原因、主体或对象则是关系错误，不能只凭其中词语在记录中出现就判supported。
"""
)


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    turns = {
        t["id"]: t for t in json.loads((source / "conversation.json").read_text(encoding="utf-8"))
    }
    probes = {p["id"]: p for p in json.loads((source / "probes.json").read_text(encoding="utf-8"))}
    rows = read_rows(source / "outputs.jsonl")
    rng = random.Random(9214)
    rng.shuffle(rows)
    packets, mapping, skipped = [], {}, []
    for i, row in enumerate(rows):
        result, probe = row["result"], probes[row["id"]]
        if result["status"] in ("context_limit_not_run", "oom_not_run"):
            skipped.append(row)
            continue
        visible = set(row["visible_turn_ids"])
        evidence = [
            dict(id=tid, text=turns[tid]["text"])
            for tid in probe["relevant_turn_ids"]
            if tid in visible
        ]
        assert all(turns[e["id"]]["number"] <= row["checkpoint"] for e in evidence)
        state = "本次响应已结束；没有后台继续执行的任务。"
        if "decision" in result:
            d = result["decision"]
            state += (
                f"内部调用日志：reason={d['reason']}, model_generated={d['model_generated']}, "
                f"eligible={d['eligible']}。这些只证明执行分支，不证明经历事实正确。"
            )
            if d["reason"] == "unsupported_generation":
                state += "生成器已运行，但候选表述未通过接受检查。"
        if "retrieval" in result:
            state += (
                "执行日志：本次实际读取的片段ID为"
                + json.dumps(result["retrieval"]["ids"], ensure_ascii=False)
                + "。这只证明读取范围，不证明回答正确；没有操作分类或旧答案模板检查。"
            )
        answer = result["answer"]
        if answer is None:
            answer = "未提供事实回答。系统状态：" + result["status_explanation"]
            state += "没有交付语言模型事实回答，所显示的是程序状态说明，不代表模型生成了拒答句。"
        packet = dict(
            review_id=f"S{i:03d}",
            question=row["question"],
            answer=answer,
            evidence=evidence,
            execution_state=state,
            generation_ended=result.get("ended"),
            evidence_scope="所问事件的全部可见片段；原始日志其他事件不能替代该事件事实。",
        )
        packet["binding_sha256"] = digest(packet)
        packets.append(packet)
        mapping[packet["review_id"]] = dict(
            mode=row["mode"],
            checkpoint=row["checkpoint"],
            probe=probe["probe"],
            output_id=row["id"],
        )
    dump_rows(out / "all_packets.jsonl", packets)
    size = (len(packets) + 2) // 3
    for i, label in enumerate("abc"):
        dump_rows(out / f"packets_{label}.jsonl", packets[i * size : (i + 1) * size])
    (out / "rubric.txt").write_text(STREAM_RUBRIC, encoding="utf-8")
    (out / "private_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                packets=len(packets),
                skipped=len(skipped),
                requested_model="gpt-5.6-luna",
                seed=9214,
                source_hashes={
                    str((source / name).resolve()): hashlib.sha256(
                        (source / name).read_bytes()
                    ).hexdigest()
                    for name in ("conversation.json", "probes.json", "outputs.jsonl")
                },
                rubric_sha256=hashlib.sha256(STREAM_RUBRIC.encode()).hexdigest(),
                caveat="Model labels and checkpoint lengths hidden; evidence and structured status can reveal condition. No human gold labels.",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(out):
    packets = read_rows(out / "all_packets.jsonl")
    reviews = [r for part in "abc" for r in read_rows(out / f"reviews_{part}.jsonl")]
    validate(packets, reviews)
    mapping = json.loads((out / "private_mapping.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for r in reviews:
        m = mapping[r["review_id"]]
        groups[m["checkpoint"], m["mode"]].append(r)
    summary = []
    for (cp, mode), rs in sorted(groups.items()):
        summary.append(
            dict(
                checkpoint=cp,
                mode=mode,
                reviewed=len(rs),
                availability=dict(Counter(r["availability"] for r in rs)),
                target=dict(Counter(r["target"] for r in rs)),
                whole_reply=dict(Counter(r["whole_reply"] for r in rs)),
                event_claim_problems=sum(r["event_claim_problem"] for r in rs),
                status_problems=sum(r["status_problem"] for r in rs),
            )
        )
    dump_rows(out / "reviews_all.jsonl", reviews)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/stream_eval_01")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    summarize(a.out) if a.summarize else prepare(a.source, a.out)
