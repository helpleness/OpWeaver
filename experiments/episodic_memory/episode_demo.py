"""Read/write ordered episode codes; normal chat bypasses memory entirely."""

import argparse
import json
from pathlib import Path

from .episode_memory import EpisodeMemory


def main(args):
    model = EpisodeMemory.load(args.run)
    if args.command == "chat":
        print(json.dumps(dict(answer=model.model.chat(args.text)), ensure_ascii=False))
        return
    if args.db is None:
        raise ValueError("--db required for write/ask")
    store = model.store(args.db, create=args.command == "write" and not args.db.exists())
    try:
        if args.command == "write":
            model.write_events(
                store,
                [
                    dict(
                        id=args.id,
                        owner=args.owner,
                        text=args.text,
                        timestamp=args.timestamp,
                        speaker=args.speaker,
                    )
                ],
            )
            result = dict(saved=args.id, counts=store.counts())
        else:
            result = model.respond(store, args.owner, args.question)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/episode_memory_02")
    )
    p.add_argument("--db", type=Path)
    p.add_argument("--owner", default="me")
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--id", required=True)
    w.add_argument("--text", required=True)
    w.add_argument("--timestamp", type=int, default=0)
    w.add_argument("--speaker", choices=["user", "assistant"], default="user")
    q = sub.add_parser("ask")
    q.add_argument("--question", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--text", required=True)
    main(p.parse_args())
