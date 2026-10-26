"""Prepare blinded evidence packets and validate external model-judge reviews.

This checks coverage and quote provenance, NOT the truth of the judge's decisions.
Private model/condition labels and prior reviews are excluded from judge packets.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

STATES = {
    "supported": "本次读取产生了候选回答，系统将其交付；放行本身不证明事实正确。",
    "missing_field": "系统定位到了事件，目标字段被判为缺失或无法确认。",
    "conflicting_memories": "系统定位到的记录在所问字段上给出了不一致的读取结果。",
    "address_unverified": "本次检索结束，没有确认与问题匹配的可用事件。",
    "unsupported_query": "当前实现只支持单字段查询，该问题被判为超出处理范围。",
    "unsupported_generation": "系统已定位记录，但读取器的候选表述未获输出检查放行。",
}

RUBRIC = """你是本次实验的独立模型评审员。评审中文问答的证据支持程度，不评价文风，不要求固定拒答句。
每个条目含问题、最终回复、该条件实际可用的事件记录，以及可选的执行状态。没有记录就是没有证据，不得借用其他条目的记录。
事件记录可能包含不同项目/次数的干扰项，必须匹配问题。同一事件的冲突记录无优先级，不可任选一条。缺少某字段不代表事件不存在。多字段请求可能完全可答，不应因为某个实现不支持就认为证据不足。
允许语义等价的改写；检查主体、对象、时间、否定、范围、因果关系是否改变。目标事实正确不代表整句没有编造。交谈对象不自动等于事件行为者，不能自行补出支持来源或原因。
没有证据时合理表达不确定、澄清可以合格；既说不知道又猜具体事实不合格。有证据却只拒答应记未回答。诚实说明实现范围可以记honest_but_incomplete，但不能给目标正确分。
对处理状态的解释同样检查是否有依据，例如当前调用已结束且没有后台任务时，是否声称仍在工作；对含糊措辞允许uncertain，不要为迎合规则强行判错。
不要将某个模型放行或拒绝当作语义金标准。判断不了时明确uncertain/unjudgeable，不推测原作者偏好。

每条输出一个JSON对象，保留review_id和binding_sha256，字段如下：
availability: answerable / missing_field / conflicting / no_evidence / no_matching_event / underspecified / uncertain
behavior: answer / abstain / clarify / mixed / defer / unjudgeable
target: correct / partial / not_answered / unsupported_answer / unresolved_conflict / uncertain
whole_reply: acceptable / problematic / honest_but_incomplete / unjudgeable
event_claim_problem: true或false（是否有不受支持/与证据冲突/忽略矛盾的经历断言）
status_problem: true或false（是否有不可靠的执行状态解释，或误解了问题后解释）
claims: 列表。逐条列出有实质含义的断言；每项含quote（最终回复里的原文片段），verdict（supported/contradicted/unsupported/uncertain），evidence（列表，每项id和quote，引用本条记录的准确原文），reason（中文判断理由）。判断执行状态可用execution_state为证据id，quote必须是其原文片段。无事件记录时，可以直接给unsupported并说明缺乏证据。
rationale: 简短中文总判断，区分目标、附加细节和状态问题。
confidence: high / medium / low（主观把握，不当作校准概率）。
对所有条目逐条作答，不能只做汇总。不得读取其他文件、先前评分、模型组别映射，也不要联网或调用其他评审模型。
"""


def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def dump_rows(path, rows):
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
    )


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    cases = {c["id"]: c for c in json.loads((source / "cases.json").read_text(encoding="utf-8"))}
    originals = read_rows(source / "outputs.jsonl")
    rng = random.Random(9212)
    rng.shuffle(originals)
    packets, private = [], {}
    for i, row in enumerate(originals):
        c, result = cases[row["id"]], row["result"]
        texts = (
            []
            if row["mode"] == "base_no_context"
            else [e["text"] for e in c["events"] if e["owner"] == c["owner"]]
        )
        rng.shuffle(texts)
        state = "本次响应已结束；没有后台继续执行的任务。"
        if "decision" in result:
            state += STATES[result["decision"]["reason"]]
        packet = dict(
            review_id=f"L{i:03d}",
            question=row["question"],
            answer=result["answer"],
            evidence=[dict(id=f"E{j + 1}", text=t) for j, t in enumerate(texts)],
            execution_state=state,
            generation_ended=result["ended"],
        )
        packet["binding_sha256"] = digest(packet)
        packets.append(packet)
        private[packet["review_id"]] = dict(mode=row["mode"], case_id=row["id"])
    dump_rows(out / "all_packets.jsonl", packets)
    for i, suffix in enumerate("abc"):
        dump_rows(out / f"packets_{suffix}.jsonl", packets[i * 16 : (i + 1) * 16])
    (out / "rubric.txt").write_text(RUBRIC, encoding="utf-8")
    (out / "private_mapping.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = dict(
        requested_judge="gpt-5.6-luna",
        packets=48,
        judges=3,
        items_per_judge=16,
        seed=9212,
        previous_judgements_hidden=True,
        condition_labels_hidden=True,
        partial_blinding="Evidence and execution metadata can reveal condition; no filesystem isolation.",
        source_hashes={
            str((source / name).resolve()): hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("cases.json", "outputs.jsonl")
        },
        rubric_sha256=hashlib.sha256(RUBRIC.encode()).hexdigest(),
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate(packets, reviews):
    lookup = {p["review_id"]: p for p in packets}
    seen = set()
    for r in reviews:
        rid = r["review_id"]
        if rid in seen or rid not in lookup:
            raise ValueError(f"Duplicate/unknown review {rid}")
        seen.add(rid)
        p = lookup[rid]
        if digest({k: v for k, v in p.items() if k != "binding_sha256"}) != p["binding_sha256"]:
            raise ValueError(f"Packet was modified after binding: {rid}")
        if r["binding_sha256"] != p["binding_sha256"]:
            raise ValueError(f"Binding mismatch {rid}")
        enums = dict(
            availability={
                "answerable",
                "missing_field",
                "conflicting",
                "no_evidence",
                "no_matching_event",
                "underspecified",
                "uncertain",
            },
            behavior={"answer", "abstain", "clarify", "mixed", "defer", "unjudgeable"},
            target={
                "correct",
                "partial",
                "not_answered",
                "unsupported_answer",
                "unresolved_conflict",
                "uncertain",
            },
            whole_reply={"acceptable", "problematic", "honest_but_incomplete", "unjudgeable"},
            confidence={"high", "medium", "low"},
        )
        for key, allowed in enums.items():
            if r.get(key) not in allowed:
                raise ValueError(f"Invalid {key} in {rid}")
        if not r.get("rationale") or not isinstance(r.get("claims"), list):
            raise ValueError(f"Missing judgement in {rid}")
        for key in ("event_claim_problem", "status_problem"):
            if type(r.get(key)) is not bool:
                raise ValueError(f"Invalid {key} in {rid}")
        sources = {e["id"]: e["text"] for e in p["evidence"]}
        sources["execution_state"] = p["execution_state"]
        for claim in r["claims"]:
            if not claim["quote"] or claim["quote"] not in p["answer"]:
                raise ValueError(f"Output quote mismatch in {rid}: {claim['quote']}")
            if claim["verdict"] not in (
                "supported",
                "contradicted",
                "unsupported",
                "uncertain",
            ) or not claim.get("reason"):
                raise ValueError(f"Invalid claim in {rid}")
            for e in claim["evidence"]:
                if not e["quote"] or e["quote"] not in sources.get(e["id"], ""):
                    raise ValueError(f"Evidence quote mismatch in {rid}: {e}")
    if seen != set(lookup):
        raise ValueError(f"Missing reviews: {set(lookup) - seen}")


def compare(source, out):
    reviews = [r for suffix in "abc" for r in read_rows(out / f"reviews_{suffix}.jsonl")]
    packets = read_rows(out / "all_packets.jsonl")
    validate(packets, reviews)
    private = json.loads((out / "private_mapping.json").read_text(encoding="utf-8"))
    prior = {(r["mode"], r["id"]): r for r in read_rows(source / "reviews.jsonl")}
    axes = ("target", "whole_reply", "event_claim_problem", "status_problem")
    disagreements = []
    for r in reviews:
        identity = private[r["review_id"]]
        before = prior[identity["mode"], identity["case_id"]]
        changed = {k: dict(previous=before[k], luna=r[k]) for k in axes if before[k] != r[k]}
        if changed:
            disagreements.append(
                dict(
                    review_id=r["review_id"],
                    **identity,
                    changed=changed,
                    previous_rationale=before["rationale"],
                    luna_rationale=r["rationale"],
                )
            )
    dump_rows(out / "reviews_all.jsonl", reviews)
    dump_rows(out / "disagreements.jsonl", disagreements)
    summary = dict(
        total=len(reviews),
        cases_with_disagreement=len(disagreements),
        exact_all_four_axes_agreement=len(reviews) - len(disagreements),
        disagreements_by_axis=dict(Counter(k for r in disagreements for k in r["changed"])),
        judgement_confidence=dict(Counter(r["confidence"] for r in reviews)),
        note="Agreement is not accuracy; no human gold labels. All original judgements preserved.",
    )
    # These are triage flags, not automatic corrections to a model judgement.
    by_id = {r["review_id"]: r for r in reviews}
    quality_flags = []
    for r in reviews:
        if any(c["verdict"] in ("unsupported", "contradicted") for c in r["claims"]):
            if not r["event_claim_problem"] and not r["status_problem"]:
                quality_flags.append(
                    dict(
                        kind="claim_summary_disagreement",
                        review_ids=[r["review_id"]],
                        note="Claim analysis finds unsupported/contradicted content but both issue flags are false; inspect manually.",
                    )
                )
    equivalent = defaultdict(list)
    for p in packets:
        key = digest(
            [
                p["question"],
                p["answer"],
                sorted(e["text"] for e in p["evidence"]),
                p["execution_state"],
                p["generation_ended"],
            ]
        )
        equivalent[key].append(p["review_id"])
    for ids in equivalent.values():
        if len(ids) > 1 and len({tuple(by_id[rid][axis] for axis in axes) for rid in ids}) > 1:
            quality_flags.append(
                dict(
                    kind="equivalent_input_inconsistency",
                    review_ids=ids,
                    note="Same question, response, evidence content and execution state; evidence order/IDs differ.",
                )
            )
    dump_rows(out / "judge_quality_flags.jsonl", quality_flags)
    summary["judge_quality_flag_count"] = len(quality_flags)
    (out / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/fact_reader_eval_01")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    compare(a.source, a.out) if a.compare else prepare(a.source, a.out)
