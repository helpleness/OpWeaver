"""Inference process: no training dataset, answer vocabulary, or label imports."""

import argparse
import json
import time
import uuid
from pathlib import Path

from .qwen_memory import QwenMemory


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, default=Path("experiments/episodic_memory/runs/qwen_memory_03")
    )
    p.add_argument("--db", type=Path)
    p.add_argument("--owner", default="demo")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    writer = sub.add_parser("write")
    writer.add_argument("--text", required=True)
    writer.add_argument("--id")
    ask = sub.add_parser("ask")
    ask.add_argument("--question", required=True)
    chat = sub.add_parser("chat")
    chat.add_argument("--text", required=True)
    a = p.parse_args()
    metadata = json.loads((a.run / "metadata.json").read_text(encoding="utf-8"))
    model = QwenMemory(metadata["backbone"], adapter_path=a.run / "adapter.pt")
    if a.command == "chat":
        result = dict(answer=model.chat(a.text))
    else:
        if a.db is None:
            p.error("--db is required for memory commands")
        store = model.store(a.db, create=a.command == "init")
        try:
            if a.command == "write":
                eid = a.id or str(uuid.uuid4())
                model.write_events(
                    store, [dict(id=eid, owner=a.owner, text=a.text, timestamp=int(time.time()))]
                )
                result = dict(written=eid, **store.counts())
            elif a.command == "ask":
                result = model.ask(store, a.owner, a.question)
            else:
                result = store.counts()
        finally:
            store.close()
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
