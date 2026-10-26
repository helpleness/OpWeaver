"""2x2 matched architecture/budget: random vs pretrained; narrow vs compositional data."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .language_data import create, load
from .language_model import IntentClassifier, predict
from .train import seed_everything


def train(backbone, source, out, steps=1200, batch=24):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    data = out / "data"
    manifest = create(data, source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    for dataset in ("narrow", "compositional"):
        rows = load(data / f"{dataset}_train.jsonl")
        val = load(data / f"{dataset}_validation.jsonl")
        # Tokenization is identical in the matched initialization comparison.
        texts = [r["text"] for r in rows] + [r["positive"] for r in rows]
        encoded = tokenizer(texts, padding=True, return_tensors="pt", truncation=False)
        assert encoded["input_ids"].shape[1] <= 128
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = torch.tensor([r["label"] for r in rows], device=device)
        for initialized in (False, True):
            seed_everything(62)
            name = f"{'pretrained' if initialized else 'random'}_{dataset}"
            target = out / name
            target.mkdir()
            model = IntentClassifier(backbone, initialized).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            rng = random.Random(62)
            started = time.perf_counter()
            model.train()
            for step in range(1, steps + 1):
                ids = torch.tensor(rng.sample(range(len(rows)), batch), device=device)
                indices = torch.cat([ids, ids + len(rows)])
                logits = model({k: v[indices] for k, v in encoded.items()})
                first, second = logits.chunk(2)
                ce = F.cross_entropy(logits, labels[ids].repeat(2))
                p, q = first.softmax(-1), second.softmax(-1)
                midpoint = ((p + q) / 2).clamp_min(1e-8)
                js = (
                    (p * (p.clamp_min(1e-8).log() - midpoint.log())).sum(-1)
                    + (q * (q.clamp_min(1e-8).log() - midpoint.log())).sum(-1)
                ).mean() / 2
                loss = ce + 0.1 * js
                lr = (
                    1e-4
                    * min(1.0, step / 100)
                    * (0.1 + 0.9 * (1 + math.cos(math.pi * step / steps)) / 2)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step == 1 or step % 200 == 0:
                    record = dict(
                        model=name,
                        step=step,
                        loss=float(loss.detach()),
                        elapsed_s=time.perf_counter() - started,
                    )
                    with (target / "training.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
            model.eval()
            probs = predict(model, tokenizer, [r["text"] for r in val])
            y = torch.tensor([r["label"] for r in val], device=device)
            confidence, pred = probs.max(-1)
            metrics = dict(
                n=len(val),
                correct=int((pred == y).sum()),
                high_conf_wrong=int(((pred != y) & (confidence > 0.8)).sum()),
                low_confidence=int((confidence <= 0.8).sum()),
            )
            tokenizer.save_pretrained(target)
            model.encoder.config.save_pretrained(target)
            torch.save(model.state_dict(), target / "weights.pt")
            metadata = dict(
                name=name,
                pretrained=initialized,
                dataset=dataset,
                steps=steps,
                batch=batch,
                seed=62,
                lr=1e-4,
                elapsed_s=time.perf_counter() - started,
                parameters=sum(p.numel() for p in model.parameters()),
                validation=metrics,
                backbone="uer/chinese_roberta_L-4_H-256",
                data=manifest["datasets"],
                max_tokens=encoded["input_ids"].shape[1],
                training_token_presentations=int(encoded["attention_mask"].sum())
                * steps
                * batch
                / len(rows),
                weights_sha256=hashlib.sha256((target / "weights.pt").read_bytes()).hexdigest(),
            )
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(json.dumps({"model": name, "validation": metrics}), flush=True)
            del model, optimizer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone", type=Path, default=Path("experiments/episodic_memory/runs/language_backbone")
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/episodic_memory/runs/paraphrase_structured"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    a = p.parse_args()
    train(a.backbone, a.source, a.out, a.steps)
