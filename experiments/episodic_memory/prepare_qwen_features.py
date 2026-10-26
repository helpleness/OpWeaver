"""Recompute frozen features for a new backbone; never reuse another model's space."""

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch

from .generate_data import TASKS, load_events, question
from .paraphrases import augmented_event, augmented_question
from .qwen_memory import QwenMemory, save_json
from .train import seed_everything


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    seed_everything(72)
    start = time.perf_counter()
    model = QwenMemory(args.backbone, structured=True)
    shutil.copytree(args.source / "data", args.out / "data")
    cache, queries = {}, {}
    for split in ("train", "validation"):
        events = load_events(args.out / "data", split)
        rng = random.Random(7102)
        event_texts = [s for e in events for s in (e["text"], augmented_event(e, rng))]
        question_texts = [
            s
            for e in events
            for task in TASKS
            for s in (question(e, task, e["query_template"]), augmented_question(e, task, rng))
        ]
        event_features = model.features(event_texts, batch=args.batch)
        query_features = model.features(question_texts, batch=args.batch)
        cache[split] = dict(
            events=events,
            event_features=event_features,
            event_means=torch.stack([h.float().mean(0) for h in event_features]),
            query_means=torch.stack([h.float().mean(0) for h in query_features]),
            questions=question_texts,
        )
        queries[split] = query_features
        print(
            json.dumps(
                dict(
                    stage="features",
                    split=split,
                    events=len(events),
                    queries=len(question_texts),
                    elapsed_s=time.perf_counter() - start,
                )
            ),
            flush=True,
        )
    torch.save(cache, args.out / "training_features.pt")
    torch.save(queries, args.out / "query_token_features.pt")
    save_json(
        args.out / "features_metadata.json",
        dict(
            backbone=str(args.backbone.resolve()),
            hidden_size=model.base.config.hidden_size,
            source_revision=(
                args.backbone / ".cache/huggingface/download/model.safetensors.metadata"
            )
            .read_text()
            .splitlines()[0],
            data_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((args.out / "data").glob("*.jsonl"))
            },
            elapsed_s=time.perf_counter() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/qwen_smoke_01")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    main(p.parse_args())
