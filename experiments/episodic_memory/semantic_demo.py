"""Standalone disk memory commands. Query never loads experimental truth records."""

import argparse
import json
from pathlib import Path

from .semantic_memory import DECISION_MEANINGS, SemanticMemory


def main(args):
    model = SemanticMemory.load(args.run)
    if args.command == "chat":
        print(json.dumps(dict(answer=model.chat(args.text)), ensure_ascii=False))
        return
    if args.db is None:
        raise ValueError("--db is required for disk memory commands")
    store = model.store(args.db, create=args.command == "write" and not args.db.exists())
    try:
        if args.command == "write":
            model.write_events(
                store,
                [dict(id=args.id, owner=args.owner, text=args.text, timestamp=args.timestamp)],
            )
            result = dict(saved=args.id, owner=args.owner, counts=store.counts())
        else:
            metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("reader_objective") == "available_facts_only":
                threshold = 0.8
                policy_status = "fixed_threshold_not_recalibrated"
            else:
                calibration = json.loads(
                    (args.run / "calibration/calibration.json").read_text(encoding="utf-8")
                )
                assert all(
                    calibration[name + "_sha256"] == metadata[name + "_sha256"]
                    for name in ("adapter", "semantic")
                )
                threshold = calibration["threshold"]
                policy_status = "historical_protocol_calibration"
            result = model.respond(store, args.owner, args.question, threshold=threshold)
            result["policy"] = dict(threshold=threshold, status=policy_status)
            result["explanation"] = {
                **DECISION_MEANINGS,
                "supported": "通过了原型的保守门控；不等于通过独立语义正确性评审。",
            }[result["decision"]["reason"]]
        print(json.dumps(result, ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/fact_reader_01")
    )
    p.add_argument("--db", type=Path)
    p.add_argument("--owner", default="demo")
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--id", required=True)
    w.add_argument("--text", required=True)
    w.add_argument("--timestamp", type=int, default=0)
    q = sub.add_parser("ask")
    q.add_argument("--question", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--text", required=True)
    main(p.parse_args())
