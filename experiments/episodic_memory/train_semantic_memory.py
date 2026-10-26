"""Train semantic query/availability heads, then routed continuous Qwen reader."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .generate_data import FIELDS, ORDINALS, PROJECTS, TASKS
from .qwen_memory import save_json
from .reader_objective import reader_batch, reader_queries
from .semantic_memory import SemanticMemory
from .train import seed_everything
from .train_qwen_memory import CHAT_PROMPTS
from .train_qwen_structured import target_tables


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(args):
    seed_everything(9204)
    torch.set_num_threads(4)
    start = time.perf_counter()
    meta = load_json(args.source / "metadata.json")
    assert (
        hashlib.sha256((args.source / "adapter.pt").read_bytes()).hexdigest()
        == meta["adapter_sha256"]
    )
    model = SemanticMemory(
        meta["backbone"],
        adapter_path=args.source / "adapter.pt",
        semantic_path=args.source / "semantic.pt"
        if (args.source / "semantic.pt").exists()
        else None,
        canonical_reader=args.canonical_reader,
        contextual_query=args.contextual_query,
    )
    if not args.legacy_refusal_target:
        model.use_fact_reader_prompt()
    if args.reset_reader and args.legacy_refusal_target:
        raise ValueError("Reader reset experiment forbids fixed refusal targets")
    data = load_json(args.suite / "train.json")
    feature_file = args.suite / (
        "contextual_train_features.pt" if args.contextual_query else "train_features.pt"
    )
    data_hash = hashlib.sha256((args.suite / "train.json").read_bytes()).hexdigest()
    if args.prepare:
        if feature_file.exists():
            raise FileExistsError(feature_file)
        features = {}
        for key, texts in [
            ("event", [e["text"] for e in data["events"]]),
            ("query", [q["text"] for q in data["queries"]]),
        ]:
            rows = []
            for pos in range(0, len(texts), 64):
                extractor = model.query_features if key == "query" else model.features
                rows.extend(extractor(texts[pos : pos + 64], batch=8))
                if pos % 512 == 0:
                    print(
                        json.dumps(
                            dict(
                                stage="features",
                                kind=key,
                                done=min(pos + 64, len(texts)),
                                total=len(texts),
                            )
                        ),
                        flush=True,
                    )
            features[key] = rows
        features["data_sha256"] = data_hash
        features["backbone_revision"] = meta["source_revision"]
        features["contextual_query"] = args.contextual_query
        torch.save(features, feature_file)
        print(
            json.dumps(dict(stage="features_done", seconds=time.perf_counter() - start)), flush=True
        )
        return
    args.out.mkdir(parents=True, exist_ok=False)
    cache = torch.load(feature_file, weights_only=True)
    assert (
        cache["data_sha256"] == data_hash and cache["backbone_revision"] == meta["source_revision"]
    )
    assert cache.get("contextual_query", False) == args.contextual_query
    assert not args.contextual_query or args.warmup == 0
    a, s = model.adapter, model.semantic
    before = {q: model.chat(q) for q in CHAT_PROMPTS}
    save_json(args.out / "chat_before.json", before)
    labels = torch.tensor(
        [
            [e["values"][t] if e["values"][t] >= 0 else len(FIELDS[t]) for t in TASKS]
            for e in data["events"]
        ],
        device=model.device,
    )
    addresses = torch.tensor(
        [[PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"])] for e in data["events"]],
        device=model.device,
    )
    qlabels = torch.tensor([q["field"] for q in data["queries"]], device=model.device)
    tables = target_tables(model)
    # Teacher-only missing value embedding for alignment; never used during inference.
    missing = model.tokenizer.encode("未知", add_special_tokens=False)
    dot = model.tokenizer.encode("。", add_special_tokens=False)
    missing = (
        model.base.get_input_embeddings()(
            torch.tensor(missing + dot * (4 - len(missing)), device=model.device)
        )
        .float()
        .detach()
    )
    tables[:6] = [torch.cat([t, missing[None]], 0) for t in tables[:6]]
    a.event_address.requires_grad_(False)
    a.query_address.requires_grad_(False)
    a.prefix.requires_grad_(False)
    a.prefix_position.requires_grad_(False)
    a.null_memory.requires_grad_(False)
    params = [p for p in list(a.parameters()) + list(s.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.001)
    rng = random.Random(9204)
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.warmup + 1):
        ids = [rng.randrange(len(labels)) for _ in range(32)]
        qi = [rng.randrange(len(qlabels)) for _ in range(48)]
        h, mask = model.padded_features([cache["event"][i] for i in ids])
        qh, qm = model.padded_features([cache["query"][i] for i in qi])
        z, _ = a.write(h, mask)
        z = z.half().float()
        value_loss = torch.stack(
            [F.cross_entropy(head(z[:, j]), labels[ids, j]) for j, head in enumerate(s.values)]
        ).mean()
        address_loss = torch.stack(
            [
                F.cross_entropy(a.value_heads[j + 6](z[:, j + 6]), addresses[ids, j])
                for j in range(2)
            ]
        ).mean()
        intent = F.cross_entropy(s.query(qh, qm), qlabels[qi])
        valid = [j for j, i in enumerate(qi) if data["queries"][i]["address"] is not None]
        target = torch.tensor(
            [data["queries"][qi[j]]["address"] for j in valid], device=model.device
        )
        qaddress = F.cross_entropy(
            a.query_logits(qh[valid], qm[valid]).reshape(-1, 16), target.flatten()
        )
        teacher = torch.cat(
            [tables[j][labels[ids, j]] for j in range(6)]
            + [tables[j + 6][addresses[ids, j]] for j in range(2)],
            1,
        )
        decoded = a.soft_tokens(z, torch.ones(len(z), dtype=torch.bool, device=model.device))
        alignment = F.mse_loss(decoded, teacher) / teacher.square().mean().clamp_min(1e-6)
        loss = value_loss + address_loss + intent + qaddress + 0.2 * alignment
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 250 == 0 or step == 1:
            row = dict(
                stage="semantic",
                step=step,
                value=float(value_loss.detach()),
                intent=float(intent.detach()),
                address=float(qaddress.detach()),
                seconds=time.perf_counter() - start,
            )
            with (args.out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    if args.query_steps:
        a.requires_grad_(False)
        s.values.requires_grad_(False)
        if args.contextual_query:
            s.requires_grad_(False)
            s.context_head.requires_grad_(True)
        query_params = [p for p in s.parameters() if p.requires_grad]
        query_opt = torch.optim.AdamW(
            query_params, lr=0.001 if args.contextual_query else 0.0002, weight_decay=0.01
        )
        for step in range(1, args.query_steps + 1):
            qi = [rng.randrange(len(qlabels)) for _ in range(64)]
            qh, qm = model.padded_features([cache["query"][i] for i in qi])
            weights = (
                torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1 / 3], device=model.device)
                if args.contextual_query
                else None
            )
            loss = F.cross_entropy(model.query_distribution(qh, qm), qlabels[qi], weight=weights)
            query_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_params, 1.0)
            query_opt.step()
            if step % 250 == 0:
                print(
                    json.dumps(
                        dict(stage="query_refinement", step=step, loss=float(loss.detach()))
                    ),
                    flush=True,
                )
    a.eval()
    s.eval()
    with torch.no_grad():
        zs = []
        for pos in range(0, len(labels), 32):
            h, m = model.padded_features(cache["event"][pos : pos + 32])
            z, _ = a.write(h, m)
            zs.append(z.half().float())
        zs = torch.cat(zs)
    if args.reset_reader:
        with torch.no_grad():
            torch.nn.init.normal_(a.prefix.weight, std=0.02)
            a.prefix.bias.zero_()
            a.prefix_position.zero_()
            a.null_memory.zero_()
    # Fixed checkpoint boundary: questions and facts stay frozen during reader training.
    a.requires_grad_(False)
    s.requires_grad_(False)
    a.prefix.requires_grad_(True)
    a.prefix_position.requires_grad_(True)
    a.null_memory.requires_grad_(args.legacy_refusal_target)
    params = [p for p in a.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.001)
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    valid_queries = reader_queries(data, args.legacy_refusal_target)
    # Align only known factual slots. Missing-state words never become LM targets.
    if args.reader_alignment_steps:
        align_opt = torch.optim.AdamW(params, lr=0.001, weight_decay=0.001)
        for step in range(1, args.reader_alignment_steps + 1):
            qs, ids, present, _ = reader_batch(data, valid_queries, 64, rng)
            fields = torch.tensor([q["field"] for q in qs], device=model.device)
            decoded = a.soft_tokens(zs[ids], torch.tensor(present, device=model.device))
            decoded = decoded.reshape(len(ids), 8, 4, -1)[torch.arange(len(ids)), fields]
            target = torch.stack([tables[int(f)][labels[i, f]] for i, f in zip(ids, fields)])
            loss = F.mse_loss(decoded, target) / target.square().mean().clamp_min(1e-6)
            align_opt.zero_grad(set_to_none=True)
            loss.backward()
            align_opt.step()
        print(
            json.dumps(dict(stage="reader_alignment", steps=step, loss=float(loss.detach()))),
            flush=True,
        )
    target_audit = dict(examples=0, absent_memory=0, fixed_refusal_targets=0)
    for step in range(1, args.steps + 1):
        qs, ids, present, answers = reader_batch(
            data, valid_queries, args.batch, rng, args.legacy_refusal_target
        )
        fields = torch.tensor([q["field"] for q in qs], device=model.device)
        target_audit["examples"] += len(answers)
        target_audit["absent_memory"] += sum(not value for value in present)
        target_audit["fixed_refusal_targets"] += answers.count("我没有这段经历的记忆。")
        if not args.legacy_refusal_target:
            assert all(present) and not target_audit["fixed_refusal_targets"]
        present = torch.tensor(present, device=model.device)
        a.train()
        model.base.train()
        loss = model.teacher_routed(zs[ids], present, [q["text"] for q in qs], answers, fields)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reader loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            row = dict(
                stage="reader",
                step=step,
                loss=float(loss.detach()),
                seconds=time.perf_counter() - start,
            )
            with (args.out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    torch.save(a.state_dict(), args.out / "adapter.pt")
    torch.save(s.state_dict(), args.out / "semantic.pt")
    after = {q: model.chat(q) for q in CHAT_PROMPTS}
    save_json(args.out / "chat_after.json", after)
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    save_json(
        args.out / "metadata.json",
        dict(
            backbone=meta["backbone"],
            source_run=str(args.source.resolve()),
            source_revision=meta["source_revision"],
            seed=9204,
            warmup=args.warmup,
            steps=args.steps,
            query_steps=args.query_steps,
            canonical_reader=args.canonical_reader,
            contextual_query=args.contextual_query,
            reader_objective="legacy_fixed_refusal"
            if args.legacy_refusal_target
            else "available_facts_only",
            reader_query_count=len(valid_queries),
            reader_reset=args.reset_reader,
            reader_alignment_steps=args.reader_alignment_steps,
            target_audit=target_audit,
            batch=args.batch,
            backbone_frozen=True,
            data_sha256=data_hash,
            adapter_sha256=hashlib.sha256((args.out / "adapter.pt").read_bytes()).hexdigest(),
            semantic_sha256=hashlib.sha256((args.out / "semantic.pt").read_bytes()).hexdigest(),
            chat_exact=sum(before[q] == after[q] for q in before),
            chat_count=len(before),
            elapsed_s=time.perf_counter() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        ),
    )
    print(json.dumps(dict(stage="done", seconds=time.perf_counter() - start)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/qwen_1_5b_memory_01")
    )
    p.add_argument(
        "--suite", type=Path, default=Path("experiments/episodic_memory/runs/semantic_suite_01")
    )
    p.add_argument(
        "--out", type=Path, default=Path("experiments/episodic_memory/runs/semantic_memory_01")
    )
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--warmup", type=int, default=3000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--query-steps", type=int, default=0)
    p.add_argument("--canonical-reader", action="store_true")
    p.add_argument("--contextual-query", action="store_true")
    p.add_argument("--reset-reader", action="store_true")
    p.add_argument("--reader-alignment-steps", type=int, default=0)
    p.add_argument(
        "--legacy-refusal-target",
        action="store_true",
        help="Reproduce historical fixed-refusal reader targets; not semantic evaluation",
    )
    main(p.parse_args())
