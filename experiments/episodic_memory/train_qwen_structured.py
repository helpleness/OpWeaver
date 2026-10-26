"""Supervised latent slots and continuous decoding for frozen Qwen backbones."""

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .generate_data import FIELDS, ORDINALS, PROJECTS, TASKS, question
from .qwen_memory import UNKNOWN, QwenMemory, save_json
from .train import seed_everything
from .train_qwen_memory import CHAT_PROMPTS, answer, context


def labels_for(events, device):
    return torch.tensor(
        [[e["values"][t] for t in TASKS] + context(e) for e in events], device=device
    )


def target_tables(model):
    # Training-only teacher embeddings. Never used by the deployed reader or disk store.
    tables = []
    pad = model.tokenizer.encode("。", add_special_tokens=False)
    assert len(pad) == 1
    for values in list(FIELDS.values()) + [PROJECTS, ORDINALS]:
        rows = []
        for value in values:
            ids = model.tokenizer.encode(value, add_special_tokens=False)
            assert len(ids) <= 4
            rows.append(ids + pad * (4 - len(ids)))
        tables.append(
            model.base.get_input_embeddings()(torch.tensor(rows, device=model.device))
            .float()
            .detach()
        )
    return tables


def auxiliary(model, z, labels, tables):
    a = model.adapter
    ce = torch.stack(
        [F.cross_entropy(head(z[:, i]), labels[:, i]) for i, head in enumerate(a.value_heads)]
    ).mean()
    target = torch.cat([table[labels[:, i]] for i, table in enumerate(tables)], 1)
    decoded = a.soft_tokens(z, torch.ones(len(z), dtype=torch.bool, device=model.device))
    alignment = F.mse_loss(decoded, target) / target.square().mean().clamp_min(1e-6)
    return ce, alignment


@torch.no_grad()
def validate(model, data):
    model.adapter.eval()
    labels = labels_for(data["events"], model.device)
    zs, keys, predictions, qkeys = [], [], [], []
    for start in range(0, len(labels), 16):
        h, mask = model.padded_features(data["event_features"][start * 2 : (start + 16) * 2 : 2])
        z, key = model.adapter.write(h, mask)
        zs.append(z)
        keys.append(key)
        predictions.append(
            torch.stack(
                [head(z[:, i]).argmax(-1) for i, head in enumerate(model.adapter.value_heads)], -1
            )
        )
    selected = data["query_features"][::2]
    for start in range(0, len(selected), 32):
        h, mask = model.padded_features(selected[start : start + 32])
        qkeys.append(model.adapter.key(model.adapter.query_logits(h, mask)))
    z, key, prediction, qkey = map(torch.cat, (zs, keys, predictions, qkeys))
    agreements = model.adapter.agreement(qkey, key.repeat_interleave(6, 0))
    result = dict(
        field_accuracy={
            name: float((prediction[:, i] == labels[:, i]).float().mean())
            for i, name in enumerate(TASKS + ["project", "ordinal"])
        },
        query_context_correct=float(
            (qkey.reshape(-1, 2, 16).argmax(-1) == labels[:, 6:].repeat_interleave(6, 0))
            .all(-1)
            .float()
            .mean()
        ),
        support_pass_rate=float((agreements > 0.99).float().mean()),
        samples=[],
    )
    for i, task in ((0, "person"), (1, "emotion"), (2, "event"), (3, "reason")):
        e = data["events"][i]
        result["samples"].append(
            dict(
                target=answer(e, task),
                **model.generate(question(e, task, 0), z[i].half().float(), True, details=True),
            )
        )
    return result


def main(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(72)
    start = time.perf_counter()
    model = QwenMemory(args.backbone, structured=True)
    a = model.adapter
    # These legacy modules only exist for checkpoint compatibility; the new path never uses them.
    a.event_address.requires_grad_(False)
    a.query_address.requires_grad_(False)
    before = {q: model.chat(q) for q in CHAT_PROMPTS}
    save_json(out / "chat_before.json", before)
    feature_metadata = args.reuse / "features_metadata.json"
    if feature_metadata.exists():
        feature_info = json.loads(feature_metadata.read_text(encoding="utf-8"))
        source_revision = (
            (args.backbone / ".cache/huggingface/download/model.safetensors.metadata")
            .read_text()
            .splitlines()[0]
        )
        if (
            feature_info["source_revision"] != source_revision
            or feature_info["hidden_size"] != a.hidden
        ):
            raise ValueError("Feature cache belongs to a different backbone; recompute it")
    shutil.copytree(args.reuse / "data", out / "data")
    cache = torch.load(args.reuse / "training_features.pt", weights_only=True)
    if any(row.shape[-1] != a.hidden for data in cache.values() for row in data["event_features"]):
        raise ValueError("Cached event width does not match backbone")
    query_cache = args.reuse / "query_token_features.pt"
    if query_cache.exists():
        queries = torch.load(query_cache, weights_only=True)
    else:
        queries = {
            split: model.features(data["questions"], batch=16) for split, data in cache.items()
        }
        torch.save(queries, query_cache)
    for split in cache:
        cache[split]["query_features"] = queries[split]
    data = cache["train"]
    labels = labels_for(data["events"], model.device)
    tables = target_tables(model)
    parameters = [p for p in a.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.001)
    rng = random.Random(72)
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.warmup + 1):
        ids = [rng.randrange(len(labels)) for _ in range(32)]
        views = [rng.randrange(2) for _ in ids]
        tasks = [rng.randrange(6) for _ in ids]
        h, mask = model.padded_features(
            [data["event_features"][i * 2 + v] for i, v in zip(ids, views)]
        )
        qh, qm = model.padded_features(
            [data["query_features"][i * 12 + t * 2 + v] for i, t, v in zip(ids, tasks, views)]
        )
        a.train()
        z, _ = a.write(h, mask)
        ce, alignment = auxiliary(model, z.half().float(), labels[ids], tables)
        ql = a.query_logits(qh, qm).reshape(-1, 16)
        qce = F.cross_entropy(ql, labels[ids, 6:].flatten())
        loss = ce + alignment + qce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        opt.step()
        if step % 250 == 0 or step == args.warmup:
            print(
                json.dumps(
                    dict(
                        stage="warmup",
                        step=step,
                        ce=float(ce.detach()),
                        alignment=float(alignment.detach()),
                        query_ce=float(qce.detach()),
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
    report = validate(model, cache["validation"])
    save_json(out / "validation_warmup.json", report)
    print(json.dumps(dict(stage="validation_warmup", **report), ensure_ascii=True), flush=True)
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.freeze_writer:
        a.requires_grad_(False)
        a.prefix.requires_grad_(True)
        a.null_memory.requires_grad_(True)
        a.prefix_position.requires_grad_(True)
        parameters = [p for p in a.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=args.reader_lr, weight_decay=0.001)
    stored = [i for i, e in enumerate(data["events"]) if e["stored"]]
    for step in range(1, args.steps + 1):
        ids = [rng.choice(stored) for _ in range(args.batch)]
        views = [rng.randrange(2) for _ in ids]
        tasks = [rng.randrange(6) for _ in ids]
        present = torch.tensor([rng.random() > 0.2 for _ in ids], device=model.device)
        h, mask = model.padded_features(
            [data["event_features"][i * 2 + v] for i, v in zip(ids, views)]
        )
        qs = [data["questions"][i * 12 + t * 2 + v] for i, t, v in zip(ids, tasks, views)]
        answers = [
            answer(data["events"][i], TASKS[t]) if keep else UNKNOWN
            for i, t, keep in zip(ids, tasks, present.tolist())
        ]
        a.train()
        model.base.train()
        z, _ = a.write(h, mask)
        z = z.half().float()
        ce, alignment = auxiliary(model, z, labels[ids], tables)
        lm = model.teacher_loss(z, present, qs, answers)
        loss = lm + ce + 0.5 * alignment
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            row = dict(
                stage="memory",
                step=step,
                lm=float(lm.detach()),
                ce=float(ce.detach()),
                alignment=float(alignment.detach()),
                seconds=time.perf_counter() - start,
            )
            with (out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % 400 == 0 or step == args.steps:
            report = validate(model, cache["validation"])
            save_json(out / f"validation_step_{step}.json", report)
            torch.save(a.state_dict(), out / f"adapter_step_{step}.pt")
            print(
                json.dumps(dict(stage="validation", step=step, **report), ensure_ascii=True),
                flush=True,
            )
    torch.save(a.state_dict(), out / "adapter.pt")
    after = {q: model.chat(q) for q in CHAT_PROMPTS}
    save_json(out / "chat_after.json", after)
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    save_json(
        out / "metadata.json",
        dict(
            backbone=str(args.backbone.resolve()),
            seed=72,
            steps=args.steps,
            warmup=args.warmup,
            batch=args.batch,
            freeze_writer=args.freeze_writer,
            reader_lr=args.reader_lr,
            source_revision=(
                args.backbone / ".cache/huggingface/download/model.safetensors.metadata"
            )
            .read_text()
            .splitlines()[0],
            writer_version=model.version(),
            backbone_frozen=True,
            backbone_parameters=sum(p.numel() for p in model.base.parameters()),
            adapter_parameters=sum(p.numel() for p in a.parameters()),
            trainable_parameters=sum(p.numel() for p in parameters),
            elapsed_s=time.perf_counter() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
            chat_exact=sum(before[q] == after[q] for q in before),
            chat_count=len(before),
            adapter_sha256=hashlib.sha256((out / "adapter.pt").read_bytes()).hexdigest(),
            thresholds=dict(evidence=0.99, min_generated_token_probability=0.8),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone", type=Path, default=Path("experiments/episodic_memory/runs/qwen_backbone")
    )
    p.add_argument(
        "--reuse", type=Path, default=Path("experiments/episodic_memory/runs/qwen_smoke_01")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--freeze-writer", action="store_true")
    p.add_argument("--reader-lr", type=float, default=0.0003)
    main(p.parse_args())
