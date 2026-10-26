"""Evidence-bound review packets, NOT an automatic semantic language judge.

Only benchmark references are derived mechanically from the known generator.
Response semantics require explicit, attributable review. Unreviewed != wrong.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .generate_data import FIELDS, ORDINALS, PROJECTS, TASKS

MULTI = {
    "把交流对象与注意力所在都列给我。": ["person", "focus"],
    "发生什么、为何有这种感觉，请同时说明。": ["event", "reason"],
    "请给我这段经历各方面的完整概览。": list(TASKS),
}
YES_NO = {
    "我记得那回是小陈，你同意吗？": ("person", "小陈"),
    "请回答是或否：那次我是不是安心？": ("emotion", "安心"),
    "那回肯定是解决故障了，对不对？": ("event", "解决故障"),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def evidence(case, mode):
    # For the memory system this is the available store, not proof of retrieval.
    if mode in ("chat_no_context", "task_no_context"):
        return []
    return [
        e
        for e in case["events"]
        if e["owner"] == case["owner"]
        and e["id"] != case["damage_id"]
        and (mode != "task_oracle_text" or e["id"] == case["id"] + "-target")
    ]


def reference(case, records):
    """Describe known synthetic test semantics; fail closed for unknown templates."""
    kind = case["kind"]
    spec = dict(question_type="single_field", fields=[TASKS[case["field"]]], premise=None)
    if kind == "ambiguous":
        matches = [v for k, v in MULTI.items() if case["question"].endswith(k)]
        if len(matches) != 1:
            raise ValueError("Unannotated multi-field question")
        spec.update(question_type="multi_field", fields=matches[0])
    elif kind == "false_premise":
        matches = [v for k, v in YES_NO.items() if case["question"].endswith(k)]
        if len(matches) != 1:
            raise ValueError("Unannotated yes/no question")
        field, value = matches[0]
        spec.update(question_type="yes_no", fields=[field], premise=value)
    elif kind == "no_anchor":
        spec.update(question_type="underspecified", fields=[])
    elif kind == "unsupported":
        spec.update(question_type="unrecorded_information", fields=[])
    anchors = [
        f"{p}项目{o}讨论"
        for p in PROJECTS
        for o in ORDINALS
        if f"{p}项目{o}讨论" in case["question"]
    ]
    if len(anchors) > 1:
        raise ValueError("Multiple anchors need explicit annotation")
    matched = [e for e in records if e["anchor"] in anchors]
    values = {
        f: sorted({FIELDS[f][e["values"][f]] for e in matched if e["values"][f] >= 0})
        for f in spec["fields"]
    }
    if not records:
        availability = "no_evidence"
    elif kind == "no_anchor":
        availability = "underspecified"
    elif kind == "unsupported":
        availability = "unrecorded_information"
    elif not matched:
        availability = "no_matching_event"
    elif any(len(v) > 1 for v in values.values()):
        availability = "conflicting"
    elif any(not v for v in values.values()):
        availability = "missing_field"
    else:
        availability = "answerable"
    spec.update(
        availability=availability,
        reference_values=values,
        matched_records=[e["id"] for e in matched],
        legacy_scope_supported=kind
        not in ("ambiguous", "false_premise", "no_anchor", "unsupported"),
    )
    if spec["question_type"] == "yes_no" and availability == "answerable":
        spec["reference_boolean"] = values[spec["fields"][0]][0] == spec["premise"]
    return spec


def packet(case, row, mode):
    records = evidence(case, mode)
    result = row["result"]
    p = dict(
        question=case["question"],
        response=result["answer"],
        generation_ended=result.get("ended"),
        evidence=[dict(id=e["id"], text=e["text"]) for e in records],
        reference=reference(case, records),
    )
    # Preserve actual instructions, including historical fixed-phrase constraints.
    # Semantic grounding and compliance with these instructions are separate axes.
    p["instructions"] = next(
        (m["content"] for m in row.get("messages", []) if m["role"] == "system"), None
    )
    p["binding_sha256"] = digest(p)
    return p


def validate_review(p, review):
    bound_content = {k: v for k, v in p.items() if k not in ("binding_sha256", "review_id")}
    if digest(bound_content) != p["binding_sha256"]:
        raise ValueError("Packet content changed after binding")
    if review["binding_sha256"] != p["binding_sha256"]:
        raise ValueError("Review is bound to different evidence or output")
    enums = {
        "behavior": {"answer", "abstain", "clarify", "mixed", "unjudgeable"},
        "target": {
            "correct",
            "incorrect",
            "partial",
            "not_answered",
            "not_applicable",
            "uncertain",
        },
        "instruction_compliance": {"yes", "no", "uncertain", "not_assessed"},
        "format_compliance": {"yes", "no", "uncertain", "not_assessed"},
    }
    for key, allowed in enums.items():
        if review.get(key) not in allowed:
            raise ValueError(f"Invalid review {key}")
    for key in ("reviewer", "rationale", "selection_reason"):
        if not review.get(key):
            raise ValueError(f"Missing {key}")
    if review.get("all_claims_checked") is not True:
        raise ValueError("Reviewer must explicitly check all factual claims")
    if review["target"] == "correct" and p["reference"]["availability"] != "answerable":
        raise ValueError("Unavailable/ambiguous target cannot be scored as a definite correct fact")
    if review["behavior"] == "abstain" and review["target"] == "correct":
        raise ValueError("Pure abstention does not answer the target fact")
    sources = {e["id"]: e["text"] for e in p["evidence"]}
    for claim in review["claims"]:
        if not claim["quote"] or claim["quote"] not in p["response"]:
            raise ValueError("Claim must quote the actual output")
        if claim["verdict"] not in ("supported", "contradicted", "unsupported", "uncertain"):
            raise ValueError("Invalid entailment verdict")
        if claim["verdict"] in ("supported", "contradicted") and not claim["evidence"]:
            raise ValueError("Entailment/contradiction requires cited evidence")
        for cite in claim["evidence"]:
            if not cite["quote"] or cite["quote"] not in sources.get(cite["id"], ""):
                raise ValueError("Evidence quote not present in this condition")
    if review["target"] == "correct" and not review["claims"]:
        raise ValueError("Correct factual answer requires claim review")


def summarize(packets, mapping, reviews):
    by_id = {p["review_id"]: p for p in packets}
    checked = {}
    for r in reviews:
        if r["review_id"] in checked:
            raise ValueError("Duplicate review; adjudicate separately")
        validate_review(by_id[r["review_id"]], r)
        checked[r["review_id"]] = r
    result = {}
    for condition in sorted({v["condition"] for v in mapping.values()}):
        ps = [p for p in packets if mapping[p["review_id"]]["condition"] == condition]
        rs = [checked[p["review_id"]] for p in ps if p["review_id"] in checked]
        result[condition] = dict(
            total=len(ps),
            reviewed=len(rs),
            unreviewed=len(ps) - len(rs),
            availability=dict(Counter(p["reference"]["availability"] for p in ps)),
            reviewed_behavior=dict(Counter(r["behavior"] for r in rs)),
            reviewed_target=dict(Counter(r["target"] for r in rs)),
            reviewed_outputs_with_unsupported_claims=sum(
                any(c["verdict"] == "unsupported" for c in r["claims"]) for r in rs
            ),
            reviewed_outputs_with_contradictions=sum(
                any(c["verdict"] == "contradicted" for c in r["claims"]) for r in rs
            ),
            semantic_accuracy=None,
            note="No aggregate accuracy: targeted audit, not a representative or independently judged sample.",
        )
    return result


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=False)
    suite = root / "semantic_suite_03/test.json"
    cases = {c["id"]: c for c in read_json(suite)}
    sources = [
        (root / "bare_qwen_comparison_02/results.jsonl", None),
        (root / "semantic_memory_03/test/results.jsonl", "memory_final"),
        (root / "semantic_baseline_03/results.jsonl", "memory_previous"),
    ]
    pairs = []
    for path, condition in sources:
        for row in read_rows(path):
            mode = condition or row["mode"]
            p = packet(cases[row["id"]], row, mode)
            private = dict(
                condition=mode,
                case_id=row["id"],
                source=str(path.resolve()),
                legacy_outcome=row.get("outcome", row.get("strict_outcome")),
            )
            pairs.append((p, private))
    random.Random(9210).shuffle(pairs)
    mapping = {}
    for i, (p, private) in enumerate(pairs):
        p["review_id"] = f"R{i:04d}"
        mapping[p["review_id"]] = private
    write_rows(out / "packets.jsonl", [p for p, _ in pairs])
    write_json(out / "private_mapping.json", mapping)
    write_json(
        out / "manifest.json",
        dict(
            source_sha256={
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [suite] + [path for path, _ in sources]
            },
            count=len(pairs),
            shuffle_seed=9210,
            limitations="Model labels hidden, but evidence/instructions may reveal condition. Not double blind.",
        ),
    )
    write_json(out / "summary.json", summarize([p for p, _ in pairs], mapping, []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/episodic_memory/runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reviews",
        type=Path,
        help="Validate and summarize existing packets with attributed reviews",
    )
    args = parser.parse_args()
    if args.reviews:
        write_json(
            args.out / "summary.json",
            summarize(
                read_rows(args.out / "packets.jsonl"),
                read_json(args.out / "private_mapping.json"),
                read_rows(args.reviews),
            ),
        )
    else:
        prepare(args.runs, args.out)


if __name__ == "__main__":
    main()
