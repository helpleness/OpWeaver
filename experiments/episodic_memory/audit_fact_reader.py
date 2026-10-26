"""Non-blind assistant review of all 48 frozen pilot outputs, NOT a general scorer.

Judgements below were made by reading every output against its supplied evidence.
They include status-explanation failures as well as extra event claims. Do not
apply these annotations to newly generated outputs; the source hash is fixed.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .evaluate_fact_reader import save

SOURCE_SHA256 = "5471a8bcb5ad1c74efffd73c5b50a06335d1d93c19a944cf13ad6bb270f19872"
CASES_SHA256 = "68a1e6b24b6f1ada386c0a878945cc700d80dea7a3eb366ea988877243b2a466"


def audit(out):
    path = out / "outputs.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Outputs differ from the individually reviewed pilot")
    if hashlib.sha256((out / "cases.json").read_bytes()).hexdigest() != CASES_SHA256:
        raise ValueError("Evidence differs from the individually reviewed pilot")
    cases = {c["id"]: c for c in json.loads((out / "cases.json").read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 48 and len({(r["mode"], r["id"]) for r in rows}) == 48
    reviewed = []
    for row in rows:
        c, mode = cases[row["id"]], row["mode"]
        i = int(row["id"][-2:])
        evidence = (
            []
            if mode == "base_no_context"
            else [dict(id=e["id"], text=e["text"]) for e in c["events"] if e["owner"] == c["owner"]]
        )
        result = row["result"]
        r = dict(
            id=row["id"],
            mode=mode,
            question=c["question"],
            answer=result["answer"],
            evidence=evidence,
            controller=result.get("decision"),
            availability="no_evidence" if mode == "base_no_context" else c["availability"],
            target="not_answered",
            whole_reply="acceptable",
            event_claim_problem=False,
            status_problem=False,
            claim_checks=[],
            reviewer="assistant_in_this_thread_nonblind_not_independent",
            ended=result["ended"],
            all_claims_checked=True,
        )
        if mode == "base_no_context":
            r["rationale"] = "未获得任何经历记录；明确表示无法确定，没有猜测经历事实。"
            if i == 1:
                r.update(
                    whole_reply="problematic",
                    status_problem=True,
                    rationale="虽未编造经历，但把问第几天误解成第几段/第几次；问题已明确第13次，解释不贴合问题。",
                )
        elif i < 6:
            r.update(target="correct", rationale="所问字段与该事件证据一致；没有添加新的事实关系。")
            target = next(e for e in evidence if e["id"] == c["id"] + "-target")
            r["claim_checks"].append(
                dict(
                    quote=result["answer"],
                    verdict="supported",
                    evidence_id=target["id"],
                    evidence_quote=target["text"],
                )
            )
            if mode == "fact_only_reader" and i == 5:
                r.update(
                    target="not_answered",
                    whole_reply="problematic",
                    status_problem=True,
                    rationale="读取器原文‘关注他人反应’与证据一致，但模板门控拒绝交付。最终解释又把表述检查失败说成无法确定记忆是否存在，扩大了不确定范围。",
                    claim_checks=[
                        dict(
                            quote="无法确定记忆是否存在",
                            verdict="unsupported_status",
                            evidence_note="门控已找到同地址事件并通过事实支持检查，失败点仅在生成表达检查。",
                        )
                    ],
                )
                r["reader_review"] = dict(
                    raw=result["reader_raw"],
                    target="correct",
                    evidence_quote="注意力主要落在他人反应。",
                )
            elif mode == "base_full_text" and i == 2:
                r.update(
                    whole_reply="problematic",
                    event_claim_problem=True,
                    rationale="目标事件错过约定正确；但记录只说交谈对象为小许、我的心情平静和获得支持，没有说小许是行为者、平静是他的感受或支持来自小陆。",
                )
                r["claim_checks"] = [
                    dict(
                        quote="错过了约定",
                        verdict="supported",
                        evidence_id=target["id"],
                        evidence_quote="发生的事情记为错过约定。",
                    ),
                    dict(
                        quote="小许错过了约定",
                        verdict="unsupported",
                        evidence_note="交谈对象不自动等于事件行为者。",
                    ),
                    dict(
                        quote="他的心情是平静的",
                        verdict="unsupported",
                        evidence_note="原始记录的情绪主体是我。",
                    ),
                    dict(
                        quote="获得了来自小陆的支持",
                        verdict="unsupported",
                        evidence_note="目标记录没有支持来源；小陆来自不同事件。",
                    ),
                ]
        elif mode in ("old_reader", "fact_only_reader"):
            notes = {
                6: "识别到目标字段缺失/未确认并解释，没有猜人名；其他字段确实存在。",
                7: "解释原因字段缺失或不确定，没有编造原因。",
                8: "没有任选冲突值，但未明确说明已经发现冲突；‘正在检查’声称仍有处理在进行，实际本次检查已返回。",
                9: "没有猜关注点，但‘正在尝试查找’暗示后台仍在检索，实际本次检索已返回。",
                10: "没有挪用他人记忆或猜事件；但‘正在尝试查找’暗示后台仍在检索，实际本次检索已返回。",
                11: "如实解释当前单字段能力范围，但没有回答证据足够的两项事实，不能当成合理缺失信息拒答得分。",
            }
            r["rationale"] = notes[i]
            if i in (8, 9, 10):
                r.update(whole_reply="problematic", status_problem=True)
                r["claim_checks"] = [
                    dict(
                        quote="正在",
                        verdict="unsupported_status",
                        evidence_note="没有后台继续任务；状态是本次检查结束后的结果。",
                    )
                ]
            elif i == 11:
                r["whole_reply"] = "honest_but_incomplete"
        else:
            notes = {
                6: "人物字段缺失，却将另一个事件中的小陈移到乐队项目第5次讨论，无依据。",
                7: "原因字段缺失，答复编造小杨反应不符预期导致生气；只有生气和人物信息有依据。",
                8: "同一事件有平静和尴尬两条记录且无时间优先级，答复只选平静，忽略冲突。",
                9: "没有记录，解释无法确定并请求更多信息，未猜测。",
                10: "可用记录只有机器人项目第14次讨论，却把其提出建议移给课程项目第7次讨论，并新增行为主体我。",
                11: "人物小陆正确；他人反应被缩窄成那个人对我的反应，额外指定了反应者与对象，未被记录支持。",
            }
            r["rationale"] = notes[i]
            if i != 9:
                r.update(
                    whole_reply="problematic",
                    event_claim_problem=True,
                    target="partial" if i == 11 else "unsupported_answer",
                )
                if i == 8:
                    r["target"] = "unresolved_conflict"
                r["claim_checks"] = [
                    dict(
                        quote=result["answer"],
                        verdict="not_fully_supported",
                        evidence_note=notes[i],
                    )
                ]
        # Output excerpts are checked for traceability, not interpreted by substring scoring.
        for claim in r["claim_checks"]:
            assert claim["quote"] in result["answer"]
            if "evidence_id" in claim:
                source = next(e for e in evidence if e["id"] == claim["evidence_id"])
                assert claim["evidence_quote"] in source["text"]
        r["output_sha256"] = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        reviewed.append(r)
    with (out / "reviews.jsonl").open("w", encoding="utf-8") as f:
        for r in reviewed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for mode in sorted({r["mode"] for r in reviewed}):
        rs = [r for r in reviewed if r["mode"] == mode]
        known = [r for r in rs if cases[r["id"]]["kind"] == "known"]
        summary[mode] = dict(
            reviewed=len(rs),
            total=len(rs),
            truncated=sum(not r["ended"] for r in rs),
            six_single_field_target_correct=None
            if mode == "base_no_context"
            else sum(r["target"] == "correct" for r in known),
            six_single_field_whole_reply_supported=None
            if mode == "base_no_context"
            else sum(r["target"] == "correct" and r["whole_reply"] == "acceptable" for r in known),
            replies_with_unsupported_event_claims=sum(r["event_claim_problem"] for r in rs),
            replies_with_status_or_question_explanation_problems=sum(
                r["status_problem"] for r in rs
            ),
            multi_field_target=next(r["target"] for r in rs if r["id"] == "fresh-11"),
            outcomes=dict(Counter(r["whole_reply"] for r in rs)),
        )
    save(
        out / "review_summary.json",
        dict(
            source_sha256=SOURCE_SHA256,
            conditions=summary,
            limitation="48/48 checked by the same assistant, nonblind, no independent adjudication; 12 synthetic cases only",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    audit(p.parse_args().out)
