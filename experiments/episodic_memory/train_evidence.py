"""Freeze all encoders; compare mixed-store joint reader vs evidence-separated reader."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .evidence_policy import infer
from .generate_data import OFFSETS, TASKS, load_events, question
from .models import Config, MemoryModel, pad_tokens
from .paraphrases import augmented_event, augmented_question
from .store import MemoryStore, writer_version
from .train import Corpus, build_store, load_checkpoint, save_checkpoint, seed_everything


def open_store(path, model, tok):
    return MemoryStore(
        path,
        writer_version(model, tok.spec),
        (model.config.slots, model.config.dim),
        model.config.key_dim,
    )


def question_text(e, task, rng):
    return (
        question(e, task, e["query_template"])
        if rng.random() < 0.35
        else augmented_question(e, task, rng)
    )


def make_batch(model, tok, events, stores, rng, batch_size, device):
    stored = [e for e in events if e["stored"]]
    missing = [e for e in events if not e["stored"]]
    es = [rng.choice(missing if rng.random() < 0.2 else stored) for _ in range(batch_size)]
    tasks = [rng.choice(TASKS) for _ in es]
    with torch.no_grad():
        h, q = model.encode_questions(
            pad_tokens([tok.encode(question_text(e, t, rng)) for e, t in zip(es, tasks)], device)
        )
    zs, ks, masks, labels, pair_targets = [], [], [], [], []
    for i, (e, task) in enumerate(zip(es, tasks)):
        store = rng.choice(stores)
        exclude = [e["id"]] if e["stored"] and rng.random() < 0.25 else []
        ids, _ = store.retrieve(
            e["owner"], q[i].cpu().numpy(), flat=rng.random() < 0.5, top_clusters=2, exclude=exclude
        )
        z, k = store.fetch(e["owner"], ids)
        blank = rng.random() < 0.1
        if blank:
            z[:] = 0
        support = e["stored"] and e["id"] in ids and not blank
        n = len(z)
        zs.append(np.pad(z, ((0, 8 - n), (0, 0), (0, 0))))
        ks.append(np.pad(k, ((0, 8 - n), (0, 0))))
        masks.append([True] * n + [False] * (8 - n))
        labels.append(OFFSETS[task] + e["values"][task] if support else 0)
        targets = [float(eid == e["id"] and support) for eid in ids]
        pair_targets.append(targets + [0.0] * (8 - len(targets)))
    return (
        h,
        q,
        torch.tensor(np.stack(zs), device=device),
        torch.tensor(np.stack(ks), device=device),
        torch.tensor(masks, device=device),
        torch.tensor(labels, device=device),
        torch.tensor(pair_targets, device=device),
    )


@torch.no_grad()
def calibrate(model, tok, events, stores, device, count=1800):
    """Both attempts are inspected. No test template or test history is used."""
    rng = random.Random(8821)
    seen = [e for e in events if e["stored"]]
    unseen = [e for e in events if not e["stored"]]
    records = []
    for i in range(count):
        kind = ("unobserved", "removed_support", "blank_payload")[i % 3]
        e = rng.choice(unseen if kind == "unobserved" else seen)
        task = rng.choice(TASKS)
        h, q = model.encode_questions(pad_tokens([tok.encode(question_text(e, task, rng))], device))
        result = infer(
            model,
            rng.choice(stores),
            e["owner"],
            h,
            q,
            exclude=[e["id"]] if kind == "removed_support" else [],
            zero=kind == "blank_payload",
            audit_all=True,
        )
        records += [
            {"kind": kind, "evidence": a["evidence"], "value_probability": a["value_probability"]}
            for a in result["attempts"]
        ]
    relevant = [
        r["evidence"] for r in records if r["value_probability"] > model.config.value_threshold
    ]
    worst = max(relevant, default=0.0)
    before = model.config.evidence_threshold
    model.config.evidence_threshold = max(before, min(1.0, worst + 1e-6))
    return {
        "validation_queries": count,
        "attempts": len(records),
        "old_threshold": before,
        "new_threshold": model.config.evidence_threshold,
        "value_threshold": model.config.value_threshold,
        "max_unsupported_evidence_passing_value_test": worst,
        "false_accepts_before": sum(
            r["evidence"] > before and r["value_probability"] > model.config.value_threshold
            for r in records
        ),
        "false_accepts_after": sum(
            r["evidence"] > model.config.evidence_threshold
            and r["value_probability"] > model.config.value_threshold
            for r in records
        ),
        "records": records,
    }


@torch.no_grad()
def validate_positive(model, tok, events, stores, device, count=600):
    rng = random.Random(8822)
    seen = [e for e in events if e["stored"]]
    counts = dict(
        correct=0,
        unknown=0,
        wrong=0,
        conditional_correct=0,
        support_found=0,
        evidence_rejections=0,
        value_rejections=0,
    )
    for _ in range(count):
        e, task = rng.choice(seen), rng.choice(TASKS)
        h, q = model.encode_questions(pad_tokens([tok.encode(question_text(e, task, rng))], device))
        result = infer(model, rng.choice(stores), e["owner"], h, q)
        target = OFFSETS[task] + e["values"][task]
        a = result["attempts"][-1]
        counts["correct"] += result["prediction"] == target
        counts["unknown"] += result["prediction"] == 0
        counts["wrong"] += result["prediction"] not in (0, target)
        counts["conditional_correct"] += a["conditional_answer"] == target
        counts["support_found"] += any(e["id"] in attempt["ids"] for attempt in result["attempts"])
        if a["evidence"] is not None:
            counts["evidence_rejections"] += a["evidence"] <= model.config.evidence_threshold
            counts["value_rejections"] += a["value_probability"] <= model.config.value_threshold
    return {"count": count, **counts, "accuracy": counts["correct"] / count}


def train(base, out, steps=3000, batch_size=16, semantic=False, reuse_stores=None, balanced=False):
    if balanced and not semantic:
        raise ValueError("--balanced-evidence requires --semantic")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source, tok, _ = load_checkpoint(base / "model.pt", device)
    source.eval()
    version = writer_version(source, tok.spec)
    train_events = load_events(base / "data", "train")
    validation_events = load_events(base / "data", "validation")
    paths = {}
    for split, events in (("train", train_events), ("validation", validation_events)):
        paths[split] = [base / f"memory_{split}.sqlite"]
        for view in (1, 2):
            if reuse_stores:
                paths[split].append(reuse_stores / f"mixed_{split}_{view}.sqlite")
                continue
            rng = random.Random(711 + view)
            changed = [{**e, "text": augmented_event(e, rng)} for e in events]
            path = out / f"mixed_{split}_{view}.sqlite"
            build_store(path, source, tok, Corpus(changed, tok), device, max_members=32)
            paths[split].append(path)
    for kind in ("verified",) if semantic else ("mixed_joint", "verified"):
        seed_everything(42)
        target = out / kind
        target.mkdir()
        shutil.copytree(base / "data", target / "data")
        for split in ("test", "validation"):
            shutil.copy2(base / f"memory_{split}.sqlite", target / f"memory_{split}.sqlite")
        config = Config(**source.config.to_dict())
        config.verified_reader = kind == "verified"
        config.semantic_verifier = semantic
        model = MemoryModel(config).to(device)
        missing_keys, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        assert not unexpected and all(n.startswith("verifier.") for n in missing_keys)
        model.requires_grad_(False)
        for component in (model.read_layers, model.read_norm, model.answer):
            component.requires_grad_(True)
        if config.verified_reader:
            model.verifier.requires_grad_(True)
        stores = [open_store(p, model, tok) for p in paths["train"]]
        if balanced:
            verifier_params = list(model.verifier.parameters())
            verifier_ids = {id(p) for p in verifier_params}
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            p
                            for p in model.parameters()
                            if p.requires_grad and id(p) not in verifier_ids
                        ],
                        "lr": 1e-4,
                    },
                    {"params": verifier_params, "lr": 1e-3},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4
            )
        rng = random.Random(42)
        started = time.perf_counter()
        log_path = target / "training.jsonl"

        def log(row):
            row.update(model=kind, elapsed_s=round(time.perf_counter() - started, 2))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        model.train()
        for step in range(1, steps + 1):
            h, q, z, k, valid, labels, targets = make_batch(
                model, tok, train_events, stores, rng, batch_size, device
            )
            raw, _, pairs, valid = model.read_parts(h, q, z, k, valid)
            if config.verified_reader:
                supported = labels != 0
                content_loss = (
                    F.cross_entropy(raw[supported, 1:].float(), labels[supported] - 1)
                    if supported.any()
                    else raw.sum() * 0
                )
                per_pair = F.binary_cross_entropy_with_logits(
                    pairs[valid].float(), targets[valid], reduction="none"
                )
                positive_pairs = targets[valid] > 0

                def balanced_mean(values):
                    groups = [values[positive_pairs], values[~positive_pairs]]
                    return torch.stack([g.mean() for g in groups if g.numel()]).mean()

                evidence_loss = balanced_mean(per_pair) if balanced else per_pair.mean()
                selection_loss = (
                    F.cross_entropy(
                        pairs[supported].masked_fill(~valid[supported], -1e4).float(),
                        targets[supported].argmax(-1),
                    )
                    if supported.any()
                    else pairs.sum() * 0
                )
                loss = content_loss + evidence_loss + 0.2 * selection_loss
                if semantic:
                    # Push BOTH supported and unsupported cases away from the fixed decision boundary.
                    sign = targets[valid] * 2 - 1
                    margin = F.relu(6.0 - sign * pairs[valid])
                    loss = loss + 0.2 * (balanced_mean(margin) if balanced else margin.mean())
            else:
                loss = F.cross_entropy(raw.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reader training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 250 == 0:
                log({"step": step, "loss": float(loss.detach())})
        for store in stores:
            store.close()
        model.eval()
        assert writer_version(model, tok.spec) == version
        for name, value in source.state_dict().items():
            if name.startswith(
                (
                    "query_encoder.",
                    "query_key.",
                    "intent_head.",
                    "intent_embeddings",
                    "context_heads.",
                )
            ):
                assert torch.equal(value, model.state_dict()[name]), name
        if config.verified_reader:
            val_stores = [open_store(p, model, tok) for p in paths["validation"]]
            calibration = calibrate(model, tok, validation_events, val_stores, device)
            calibration["positive_validation"] = validate_positive(
                model, tok, validation_events, val_stores, device
            )
            (target / "calibration.json").write_text(
                json.dumps(calibration, indent=2), encoding="utf-8"
            )
            for store in val_stores:
                store.close()
            log(
                {"stage": "calibration", **{k: v for k, v in calibration.items() if k != "records"}}
            )
        metadata = {
            "base": str(base.resolve()),
            "kind": kind,
            "steps": steps,
            "batch_size": batch_size,
            "balanced_evidence": balanced,
            "verifier_lr": 1e-3 if balanced else 1e-4,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "writer_version": version,
            "frozen_encoders_exact": True,
            "elapsed_s": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": model.config.to_dict(),
        }
        save_checkpoint(target / "model.pt", model, tok, metadata)
        (target / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log({"stage": "complete", "writer_version": version})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--reuse-stores", type=Path)
    p.add_argument("--balanced-evidence", action="store_true")
    a = p.parse_args()
    train(a.base, a.out, a.steps, a.batch_size, a.semantic, a.reuse_stores, a.balanced_evidence)
