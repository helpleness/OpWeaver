"""Separate-process write/query CLI; query never reads the training dataset."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import torch

from .generate_data import ANSWERS
from .models import pad_tokens
from .store import MemoryStore, writer_version
from .train import load_checkpoint, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--owner", default="local")
    sub = p.add_subparsers(dest="action", required=True)
    write = sub.add_parser("write")
    write.add_argument("--text", required=True)
    write.add_argument("--event-id")
    read = sub.add_parser("query")
    read.add_argument("--question", required=True)
    read.add_argument("--flat", action="store_true")
    args = p.parse_args()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    text = args.text if args.action == "write" else args.question
    tokens = tokenizer.encode(text)
    if 2 in tokens:
        raise ValueError(
            "Text contains symbols outside this controlled prototype's vocabulary. See README examples."
        )
    version = writer_version(model, tokenizer.spec)
    store = MemoryStore(
        args.db,
        version,
        (model.config.slots, model.config.dim),
        model.config.key_dim,
        create=args.action == "write" and not args.db.exists(),
    )
    with torch.no_grad():
        if args.action == "write":
            z, key = model.encode_events(pad_tokens([tokens], device))
            eid = args.event_id or str(uuid.uuid4())
            store.add_many(
                [
                    {
                        "id": eid,
                        "owner": args.owner,
                        "timestamp": int(time.time()),
                        "memory": z[0].cpu().numpy(),
                        "key": key[0].cpu().numpy(),
                    }
                ]
            )
            result = {
                "written": eid,
                "raw_text_stored": False,
                "writer_version": version,
                **store.counts(),
            }
        else:
            summary, qkey = model.encode_questions(pad_tokens([tokens], device))
            ids, candidates = store.retrieve(args.owner, qkey[0].cpu().numpy(), flat=args.flat)
            z, keys = store.fetch(args.owner, ids)
            logits, _ = model.read(
                summary,
                qkey,
                torch.tensor(z[None], device=device),
                torch.tensor(keys[None], device=device),
            )
            probability = logits.softmax(-1)[0]
            pred = int(probability.argmax()) if ids else 0
            result = {
                "answer": ANSWERS[pred],
                "uncalibrated_probability": float(probability[pred]),
                "selected_ids": ids,
                "candidate_count": candidates,
                "writer_version": version,
                "note": "UNKNOWN means insufficient recalled evidence, not proof the event never happened.",
            }
    store.close()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
