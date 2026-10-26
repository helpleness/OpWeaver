"""Train only latent memory modules; Qwen stays frozen throughout."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .generate_data import FIELDS, ORDINALS, PROJECTS, TASKS, generate, load_events, question
from .paraphrases import augmented_event, augmented_question
from .qwen_memory import UNKNOWN, QwenMemory, save_json
from .train import seed_everything

CHAT_PROMPTS = [
    "你好，请介绍一下自己。",
    "用两句话解释为什么会下雨。",
    "17加25等于多少？",
    "请把“今天阳光很好”翻译成英文。",
    "请写一句给朋友的生日祝福。",
    "列出三个常见的水果。",
    "如果我有三颗糖，给朋友一颗，还剩几颗？",
    "帮我写一句礼貌的会议延期通知。",
]


def answer(e, task):
    value = FIELDS[task][e["values"][task]]
    return {
        "person": f"那次和{value}交流。",
        "day": f"那件事发生在{value}。",
        "event": f"那次的事情是{value}。",
        "emotion": f"当时感到{value}。",
        "reason": f"因为{value}。",
        "focus": f"当时关注{value}。",
    }[task]


def context(e):
    return [PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"])]


def prepare(model, out):
    generate(
        out / "data",
        seed=7101,
        train_histories=32,
        eval_histories=4,
        train_events=16,
        test_events=32,
    )
    cache = {}
    for split in ("train", "validation"):
        events = load_events(out / "data", split)
        rng = random.Random(7102)
        event_texts = [text for e in events for text in (e["text"], augmented_event(e, rng))]
        question_texts = [
            text
            for e in events
            for task in TASKS
            for text in (question(e, task, e["query_template"]), augmented_question(e, task, rng))
        ]
        event_features = model.features(event_texts)
        query_features = model.features(question_texts, batch=16)
        cache[split] = dict(
            events=events,
            event_features=event_features,
            event_means=torch.stack([h.float().mean(0) for h in event_features]),
            query_means=torch.stack([h.float().mean(0) for h in query_features]),
            questions=question_texts,
        )
        print(
            json.dumps(
                dict(stage="features", split=split, events=len(events), queries=len(question_texts))
            ),
            flush=True,
        )
    torch.save(cache, out / "training_features.pt")
    return cache


@torch.no_grad()
def address_metrics(model, cache):
    a = model.adapter
    events = cache["events"]
    labels = torch.tensor([context(e) for e in events], device=model.device)
    eh = cache["event_means"][::2].to(model.device)
    qh = (
        cache["query_means"]
        .reshape(len(events), 6, 2, -1)[:, :, 0]
        .reshape(-1, a.hidden)
        .to(model.device)
    )
    ep = a.event_address(eh).reshape(-1, 2, 16).softmax(-1)
    qp = a.query_address(qh).reshape(-1, 2, 16).softmax(-1)
    agreement = (ep[:, None] * qp.reshape(len(events), 6, 2, 16)).sum(-1).prod(-1)
    return dict(
        event_context_correct=float((ep.argmax(-1) == labels).all(-1).float().mean()),
        query_context_correct=float(
            (qp.argmax(-1) == labels.repeat_interleave(6, 0)).all(-1).float().mean()
        ),
        support_pass_rate=float((agreement > 0.99).float().mean()),
        min_agreement=float(agreement.min()),
    )


def main(backbone, out, steps=1200, batch=4, reuse=None, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(71)
    started = time.perf_counter()
    model = QwenMemory(backbone)
    source_revision = (
        (backbone / ".cache/huggingface/download/model.safetensors.metadata")
        .read_text()
        .splitlines()[0]
    )
    chat_before = {q: model.chat(q) for q in CHAT_PROMPTS}
    save_json(out / "chat_before.json", chat_before)
    if reuse:
        import shutil

        shutil.copytree(reuse / "data", out / "data")
        cache = torch.load(reuse / "training_features.pt", weights_only=True)
    else:
        cache = prepare(model, out)
    train = cache["train"]
    events = train["events"]
    labels = torch.tensor([context(e) for e in events], device=model.device)
    eh = train["event_means"].to(model.device)
    qh = train["query_means"].reshape(len(events), 6, 2, -1).to(model.device)
    adapter = model.adapter
    address_parameters = list(adapter.event_address.parameters()) + list(
        adapter.query_address.parameters()
    )
    opt = torch.optim.AdamW(address_parameters, lr=0.001)
    adapter.train()
    for step in range(1, 2001 if not smoke else 11):
        ids = torch.randint(len(events), (64,), device=model.device)
        view = torch.randint(2, (64,), device=model.device)
        task = torch.randint(6, (64,), device=model.device)
        el = adapter.event_address(eh[ids * 2 + view]).reshape(-1, 2, 16)
        ql = adapter.query_address(qh[ids, task, view]).reshape(-1, 2, 16)
        loss = F.cross_entropy(el.flatten(0, 1), labels[ids].flatten()) + F.cross_entropy(
            ql.flatten(0, 1), labels[ids].flatten()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(
                json.dumps(dict(stage="address", step=step, loss=float(loss.detach()))), flush=True
            )
    for component in (adapter.event_address, adapter.query_address):
        component.requires_grad_(False)
    save_json(out / "address_validation.json", address_metrics(model, cache["validation"]))
    opt = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad], lr=0.002, weight_decay=0.001
    )
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(71)
    present_ids = [i for i, e in enumerate(events) if e["stored"]]
    torch.cuda.reset_peak_memory_stats()
    steps = 3 if smoke else steps
    for step in range(1, steps + 1):
        ids = [rng.choice(present_ids) for _ in range(batch)]
        tasks = [rng.randrange(6) for _ in ids]
        views = [rng.randrange(2) for _ in ids]
        present = torch.tensor([rng.random() >= 0.2 for _ in ids], device=model.device)
        h, mask = model.padded_features(
            [train["event_features"][i * 2 + v] for i, v in zip(ids, views)]
        )
        questions = [train["questions"][i * 12 + t * 2 + v] for i, t, v in zip(ids, tasks, views)]
        answers = [
            answer(events[i], TASKS[t]) if keep else UNKNOWN
            for i, t, keep in zip(ids, tasks, present.tolist())
        ]
        adapter.train()
        model.base.train()
        z, _ = adapter.write(h, mask)
        z = z.half().float()  # Match FP16 disk payloads while keeping gradients.
        loss = model.teacher_loss(z, present, questions, answers)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite memory loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row = dict(
                stage="memory",
                step=step,
                loss=float(loss.detach()),
                elapsed_s=time.perf_counter() - started,
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
            )
            with (out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % 400 == 0 or step == steps:
            torch.save(adapter.state_dict(), out / f"adapter_step_{step}.pt")
            model.base.eval()
            adapter.eval()
            samples = []
            for i, task in ((0, "person"), (1, "emotion"), (2, "event"), (3, "reason")):
                e = cache["validation"]["events"][i]
                vh, vm = model.padded_features([cache["validation"]["event_features"][i * 2]])
                vz, _ = adapter.write(vh, vm)
                generated = model.generate(question(e, task, 0), vz, True, details=True)
                samples.append(dict(target=answer(e, task), **generated))
            save_json(out / f"validation_step_{step}.json", samples)
            print(
                json.dumps(dict(stage="validation", step=step, samples=samples), ensure_ascii=True),
                flush=True,
            )
    model.base.eval()
    adapter.eval()
    torch.save(adapter.state_dict(), out / "adapter.pt")
    chat_after = {q: model.chat(q) for q in CHAT_PROMPTS}
    save_json(out / "chat_after.json", chat_after)
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    metadata = dict(
        backbone=str(backbone.resolve()),
        source_revision=source_revision,
        steps=steps,
        batch=batch,
        seed=71,
        writer_version=model.version(),
        backbone_parameters=sum(p.numel() for p in model.base.parameters()),
        adapter_parameters=sum(p.numel() for p in adapter.parameters()),
        trainable_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        elapsed_s=time.perf_counter() - started,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        chat_exact=sum(chat_before[q] == chat_after[q] for q in CHAT_PROMPTS),
        chat_count=len(CHAT_PROMPTS),
        backbone_frozen=True,
        adapter_sha256=hashlib.sha256((out / "adapter.pt").read_bytes()).hexdigest(),
        thresholds=dict(evidence=0.99, min_generated_token_probability=0.8),
    )
    save_json(out / "metadata.json", metadata)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone", type=Path, default=Path("experiments/episodic_memory/runs/qwen_backbone")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--reuse", type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    main(a.backbone, a.out, a.steps, a.batch, a.reuse, a.smoke)
