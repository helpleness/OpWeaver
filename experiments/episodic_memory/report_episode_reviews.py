"""Aggregate already adjudicated semantic reviews, without judging answers."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .luna_review import read_rows, validate


def summarize(root: Path):
    packets = read_rows(root / "all_packets.jsonl")
    reviews = read_rows(root / "final_reviews.jsonl")
    validate(packets, reviews)
    mapping = json.loads((root / "private_mapping.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for r in reviews:
        groups[mapping[r["review_id"]]["mode"]].append(r)
    summary = {}
    for mode, rows in sorted(groups.items()):
        summary[mode] = dict(
            reviewed=len(rows),
            target=dict(Counter(r["target"] for r in rows)),
            whole_reply=dict(Counter(r["whole_reply"] for r in rows)),
            event_claim_problems=sum(r["event_claim_problem"] for r in rows),
            status_problems=sum(r["status_problem"] for r in rows),
            human_gold=False,
        )
    (root / "adjudicated_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
