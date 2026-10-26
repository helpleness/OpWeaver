from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .generate_data import (
    OFFSETS,
    ORDINALS,
    PROJECTS,
    TASKS,
    Tokenizer,
    generate,
    load_events,
    question,
)
from .losses import routing_loss, total_loss
from .models import Config, MemoryModel, pad_tokens
from .store import MemoryStore, writer_version


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def precision(device, amp):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and amp
        else contextlib.nullcontext()
    )


def save_checkpoint(path, model, tokenizer, extra):
    torch.save(
        {
            "config": model.config.to_dict(),
            "model": model.state_dict(),
            "tokenizer": tokenizer.spec,
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MemoryModel(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, Tokenizer(ckpt["tokenizer"]), ckpt["extra"]


class Corpus:
    def __init__(self, events, tokenizer):
        self.events = {e["id"]: e for e in events}
        self.stored = [e for e in events if e["stored"]]
        self.by_owner = defaultdict(list)
        self.event_tokens = {}
        self.question_tokens = {}
        for e in events:
            if e["stored"]:
                self.by_owner[e["owner"]].append(e["id"])
            self.event_tokens[e["id"]] = tokenizer.encode(e["text"])
            for task in TASKS:
                self.question_tokens[e["id"], task] = tokenizer.encode(
                    question(e, task, e["query_template"])
                )

    def candidates(self, e, rng, count=8):
        other = [eid for eid in self.by_owner[e["owner"]] if eid != e["id"]]
        hard = [eid for eid in other if self.events[eid]["project"] == e["project"]]
        chosen = rng.sample(hard, min(len(hard), (count - 1) // 2))
        chosen += rng.sample([eid for eid in other if eid not in chosen], count - 1 - len(chosen))
        chosen += [e["id"]]
        rng.shuffle(chosen)
        return chosen

    def batch(self, batch_size, rng, device, missing_probability=0.2):
        selected = [rng.choice(self.stored) for _ in range(batch_size)]
        tasks = [rng.choice(TASKS) for _ in selected]
        candidates = [self.candidates(e, rng) for e in selected]
        positive = torch.tensor(
            [ids.index(e["id"]) for ids, e in zip(candidates, selected)], device=device
        )
        valid = torch.ones(batch_size, 8, dtype=torch.bool, device=device)
        labels = []
        for i, (e, task) in enumerate(zip(selected, tasks)):
            absent = rng.random() < missing_probability
            if absent:
                valid[i, positive[i]] = False
            labels.append(0 if absent else OFFSETS[task] + e["values"][task])
        return {
            "event_tokens": pad_tokens(
                [self.event_tokens[eid] for ids in candidates for eid in ids], device
            ),
            "questions": pad_tokens(
                [self.question_tokens[e["id"], t] for e, t in zip(selected, tasks)], device
            ),
            "aux": torch.tensor(
                [
                    [self.events[eid]["values"][t] for t in TASKS]
                    for ids in candidates
                    for eid in ids
                ],
                device=device,
            ),
            "positive": positive,
            "valid": valid,
            "labels": torch.tensor(labels, device=device),
            "event_context": torch.tensor(
                [context_labels(self.events[eid]) for ids in candidates for eid in ids],
                device=device,
            ),
            "query_context": torch.tensor([context_labels(e) for e in selected], device=device),
        }


def context_labels(e):
    # Optional supervised scaffold: both attributes are explicitly in the input text.
    # This is NOT a learned extraction claim from unrestricted language.
    ordinal = e.get("ordinal", e["anchor"].split("项目")[1].removesuffix("讨论"))
    return [PROJECTS.index(e["project"]), ORDINALS.index(ordinal)]


def forward_batch(model, batch):
    b = len(batch["labels"])
    z, keys = model.encode_events(batch["event_tokens"])
    summary, qkey = model.encode_questions(batch["questions"])
    zm = z.reshape(b, 8, model.config.slots, model.config.dim)
    km = keys.reshape(b, 8, model.config.key_dim)
    logits, _ = model.read(summary, qkey, zm, km, batch["valid"])
    route = routing_loss(qkey, km, batch["positive"])
    aux = model.auxiliary_loss(z, batch["aux"])
    loss, ans = total_loss(logits, batch["labels"], route, aux, model.config.route_weight)
    context = loss.new_zeros(())
    if model.config.context_weight:
        context = (
            model.context_loss(keys, batch["event_context"])
            + model.context_loss(qkey, batch["query_context"])
        ) / 2
        loss = loss + model.config.context_weight * context
    return (
        loss,
        logits,
        {
            "answer": float(ans.detach()),
            "route": float(route.detach()),
            "aux": float(aux.detach()),
            "context": float(context.detach()),
        },
    )


@torch.no_grad()
def validate_oracle(model, corpus, device, amp, batches=20):
    model.eval()
    rng = random.Random(918)
    correct = present_correct = present_count = total = 0
    losses = []
    for _ in range(batches):
        batch = corpus.batch(8, rng, device)
        with precision(device, amp):
            loss, logits, _ = forward_batch(model, batch)
        pred = logits.argmax(-1)
        correct += int((pred == batch["labels"]).sum())
        mask = batch["labels"] != 0
        present_correct += int(((pred == batch["labels"]) & mask).sum())
        present_count += int(mask.sum())
        total += len(pred)
        losses.append(float(loss))
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": correct / total,
        "present_accuracy": present_correct / max(present_count, 1),
    }


@torch.no_grad()
def build_store(path, model, tokenizer, corpus, device, max_members=64):
    model.eval()
    version = writer_version(model, tokenizer.spec)
    store = MemoryStore(
        path,
        version,
        (model.config.slots, model.config.dim),
        model.config.key_dim,
        create=True,
        max_members=max_members,
    )
    for ids in corpus.by_owner.values():
        records = []
        for start in range(0, len(ids), 32):
            subset = ids[start : start + 32]
            z, keys = model.encode_events(
                pad_tokens([corpus.event_tokens[eid] for eid in subset], device)
            )
            z, keys = z.float().cpu().numpy(), keys.cpu().numpy()
            for i, eid in enumerate(subset):
                e = corpus.events[eid]
                records.append(
                    {
                        "id": eid,
                        "owner": e["owner"],
                        "timestamp": e["timestamp"],
                        "memory": z[i],
                        "key": keys[i],
                    }
                )
        store.add_many(records)
    info = store.counts()
    store.close()
    return info


def stage_b(
    model, tokenizer, corpus, store, device, steps, batch_size, rng, log, amp, top_clusters
):
    model.freeze_writer()
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01
    )
    for step in range(1, steps + 1):
        es = [rng.choice(corpus.stored) for _ in range(batch_size)]
        tasks = [rng.choice(TASKS) for _ in es]
        qt = pad_tokens([corpus.question_tokens[e["id"], t] for e, t in zip(es, tasks)], device)
        with precision(device, amp):
            summary, qkeys = model.encode_questions(qt)
        detached_keys = qkeys.detach().float().cpu().numpy()
        zs, ks, supervised_keys, route_targets, labels, masks = [], [], [], [], [], []
        for i, (e, task) in enumerate(zip(es, tasks)):
            selected, _ = store.retrieve(e["owner"], detached_keys[i], top_clusters=top_clusters)
            if rng.random() < 0.2:
                selected = [eid for eid in selected if eid != e["id"]]
            z, key = store.fetch(e["owner"], selected)
            n = len(z)
            zs.append(np.pad(z, ((0, 8 - n), (0, 0), (0, 0))))
            ks.append(np.pad(key, ((0, 8 - n), (0, 0))))
            masks.append([True] * n + [False] * (8 - n))
            labels.append(OFFSETS[task] + e["values"][task] if e["id"] in selected else 0)
            # Supervised routing can see the target KEY, never its answer in the reader.
            route_ids = corpus.candidates(e, rng)
            _, route_k = store.fetch(e["owner"], route_ids)
            supervised_keys.append(route_k)
            route_targets.append(route_ids.index(e["id"]))
        labels_t = torch.tensor(labels, device=device)
        with precision(device, amp):
            logits, _ = model.read(
                summary,
                qkeys,
                torch.tensor(np.stack(zs), device=device),
                torch.tensor(np.stack(ks), device=device),
                torch.tensor(masks, device=device),
            )
            route = routing_loss(
                qkeys,
                torch.tensor(np.stack(supervised_keys), device=device),
                torch.tensor(route_targets, device=device),
            )
            loss = F.cross_entropy(logits.float(), labels_t) + model.config.route_weight * route
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage B loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            log(
                {
                    "stage": "B",
                    "step": step,
                    "loss": float(loss.detach()),
                    "batch_accuracy": float((logits.argmax(-1) == labels_t).float().mean()),
                }
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--stage-b-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--small", action="store_true")
    p.add_argument("--train-histories", type=int, default=192)
    p.add_argument("--eval-histories", type=int, default=12)
    p.add_argument("--test-events", type=int, default=96)
    p.add_argument("--max-members", type=int, default=32)
    p.add_argument("--top-clusters", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--address-source", choices=("slots", "cls"), default="cls")
    p.add_argument("--route-weight", type=float, default=1.0)
    p.add_argument(
        "--context-weight",
        type=float,
        default=1.0,
        help="Supervise explicit project/occurrence cues during training; 0 disables the scaffold",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new output directory; existing experiments are immutable")
    args.out.mkdir(parents=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    data = args.out / "data"
    generate(
        data,
        train_histories=args.train_histories,
        eval_histories=args.eval_histories,
        test_events=args.test_events,
    )
    tokenizer = Tokenizer(json.loads((data / "tokenizer.json").read_text(encoding="utf-8")))
    config = Config(len(tokenizer.ids))
    config.address_source, config.route_weight = args.address_source, args.route_weight
    config.context_weight = args.context_weight
    if args.small:
        config.dim, config.event_layers, config.query_layers, config.reader_layers = 128, 2, 1, 1
    model = MemoryModel(config).to(device)
    train = Corpus(load_events(data, "train"), tokenizer)
    validation = Corpus(load_events(data, "validation"), tokenizer)
    started = time.perf_counter()
    log_path = args.out / "training.jsonl"

    def log(record):
        record["elapsed_s"] = round(time.perf_counter() - started, 2)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "device": str(device),
        "amp_bfloat16": amp,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(x.numel() for x in model.parameters()),
        "config": config.to_dict(),
    }
    (args.out / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log({"stage": "setup", **metadata})
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(args.seed)
    best = float("inf")
    model.train()
    for step in range(1, args.steps + 1):
        batch = train.batch(args.batch_size, rng, device)
        with precision(device, amp):
            loss, logits, parts = forward_batch(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage A loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            log({"stage": "A", "step": step, "loss": float(loss.detach()), **parts})
        if step % 250 == 0 or step == args.steps:
            val = validate_oracle(model, validation, device, amp)
            log({"stage": "validation", "step": step, **val})
            if val["loss"] < best:
                best = val["loss"]
                save_checkpoint(
                    args.out / "stage_a.pt", model, tokenizer, {"step": step, "validation": val}
                )
    model, tokenizer, _ = load_checkpoint(args.out / "stage_a.pt", device)
    for split in ("train", "validation", "test", "test_template"):
        corpus = train if split == "train" else Corpus(load_events(data, split), tokenizer)
        info = build_store(
            args.out / f"memory_{split}.sqlite", model, tokenizer, corpus, device, args.max_members
        )
        log({"stage": "store", "split": split, **info})
    version = writer_version(model, tokenizer.spec)
    store = MemoryStore(
        args.out / "memory_train.sqlite", version, (config.slots, config.dim), config.key_dim
    )
    stage_b(
        model,
        tokenizer,
        train,
        store,
        device,
        args.stage_b_steps,
        args.batch_size,
        rng,
        log,
        amp,
        args.top_clusters,
    )
    store.close()
    if writer_version(model, tokenizer.spec) != version:
        raise AssertionError("Frozen writer changed during stage B")
    final_meta = {
        "writer_version": version,
        "elapsed_s": time.perf_counter() - started,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0,
    }
    save_checkpoint(args.out / "model.pt", model, tokenizer, final_meta)
    log({"stage": "complete", **final_meta})


if __name__ == "__main__":
    main()
