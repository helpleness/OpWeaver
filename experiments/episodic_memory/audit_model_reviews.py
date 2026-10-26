"""Audit quote provenance/coverage and triage model-judge disagreements.

Never decides answer correctness from phrases, keywords, or an acceptance gate.
Semantic labels are supplied by identified reviewers; flags need explicit review.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .luna_review import digest, dump_rows, read_rows, validate

AXES = ("target", "whole_reply", "event_claim_problem", "status_problem")


def audit(out):
    packets = read_rows(out / "all_packets.jsonl")
    reviews = [r for suffix in "abc" for r in read_rows(out / f"reviews_{suffix}.jsonl")]
    validate(packets, reviews)
    primary = {r["review_id"]: r for r in read_rows(out / "primary_reviews.jsonl")}
    by_id = {p["review_id"]: p for p in packets}
    assert set(primary) == set(by_id)
    disagreements, agree, flags = [], [], []
    for r in reviews:
        p = primary[r["review_id"]]
        assert p["binding_sha256"] == r["binding_sha256"]
        changed = {k: dict(primary=p[k], luna=r[k]) for k in AXES if p[k] != r[k]}
        if changed:
            disagreements.append(
                dict(
                    review_id=r["review_id"],
                    changed=changed,
                    primary=p,
                    luna=r,
                    packet=by_id[r["review_id"]],
                )
            )
        else:
            agree.append(r["review_id"])
        if any(c["verdict"] in ("contradicted", "unsupported") for c in r["claims"]) and not (
            r["event_claim_problem"] or r["status_problem"]
        ):
            flags.append(dict(kind="claim_summary_inconsistency", ids=[r["review_id"]]))
        if (r["event_claim_problem"] or r["status_problem"]) and r["whole_reply"] == "acceptable":
            flags.append(dict(kind="issue_but_acceptable", ids=[r["review_id"]]))
    equivalent = defaultdict(list)
    for p in packets:
        equivalent[
            digest([p["question"], p["answer"], sorted(e["text"] for e in p["evidence"])])
        ].append(p["review_id"])
    reviewer_lookup = {r["review_id"]: r for r in reviews}
    for ids in equivalent.values():
        if len(ids) > 1 and len({tuple(reviewer_lookup[i][a] for a in AXES) for i in ids}) > 1:
            flags.append(
                dict(
                    kind="equivalent_semantics_label_difference",
                    ids=ids,
                    note="Execution-state differences must be inspected before declaring inconsistency",
                )
            )
    sample = random.Random(57391).sample(sorted(agree), min(12, len(agree)))
    dump_rows(out / "reviews_all.jsonl", reviews)
    dump_rows(out / "disagreements.jsonl", disagreements)
    dump_rows(out / "quality_flags.jsonl", flags)
    dump_rows(
        out / "agreement_sample.jsonl",
        [
            dict(review_id=i, packet=by_id[i], luna=reviewer_lookup[i], primary=primary[i])
            for i in sample
        ],
    )
    summary = dict(
        reviewed=len(reviews),
        quoted_evidence_validation="passed",
        independent_primary=len(primary),
        disagreements=len(disagreements),
        agreement=len(agree),
        quality_flags=len(flags),
        random_seed=57391,
        agreement_sample=sample,
        adjudication="pending",
        human_gold=False,
        note="Agreement is not accuracy. Automated checks validate provenance/consistency, never semantic truth.",
    )
    (out / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    audit(p.parse_args().out)
