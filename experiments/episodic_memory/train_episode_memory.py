"""Train an ordered latent channel, without operation labels or refusal targets."""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .episode_memory import EpisodeMemory
from .generate_data import FIELDS, TASKS
from .semantic_data import ANSWERS, QUESTIONS, event
from .semantic_memory import SemanticMemory
from .stream_experiment import describe, save


def data(seed=57201, count=512):
    rng = random.Random(seed)
    cases = []
    introductions = [
        "我刚核实了，先前关于{f}的说法记错了，以这次为准。",
        "这里订正一下{f}，旧说法撤回，下面是我确认的版本。",
        "我重新检查过了，只修正{f}，其余信息不变。",
        "我原来把{f}说错了，现在改正如下。",
        "关于{f}，这回已经确定，请按这份新的记。",
        "之前那份{f}不准确，我已经查证，现正式更正。",
    ]
    doubts = [
        "关于{f}我又有另一种回忆，但还不确定，两种说法都没有核实。",
        "下面只是另一种可能，{f}究竟是哪一种我仍拿不准。",
        "有人建议修改{f}，我没有同意，也没有确认，先记下这个疑点。",
        "我不确定以前的{f}有没有记错，现在这份也可能不对。",
        "关于{f}的矛盾请先留着，现在不能认定哪个正确。",
        "如果以后证实{f}有误再改，目前我只是提出疑问。",
    ]
    names = ["交流对象", "发生日期", "发生的事情", "当时的心情", "感受产生的原因", "关注的方面"]
    for i in range(count):
        e = event(rng, f"episode-{i}", "train", (rng.randrange(16), rng.randrange(16)))
        field = (i // 4) % 6
        task = TASKS[field]
        texts = [describe(e, rng.randrange(3))]
        target = e["values"][task]
        kind = ["single", "correction", "doubt", "supplement"][i % 4]
        if kind in ("correction", "doubt"):
            newer = copy.deepcopy(e)
            newer["values"][task] = (target + rng.randrange(1, len(FIELDS[task]))) % len(
                FIELDS[task]
            )
            intro = rng.choice(introductions if kind == "correction" else doubts).format(
                f=names[field]
            )
            texts.append(intro + describe(newer, rng.randrange(3)))
            target = newer["values"][task] if kind == "correction" else None
        elif kind == "supplement":
            old = copy.deepcopy(e)
            old["values"][task] = -1
            texts = [
                describe(old, rng.randrange(3)),
                "再补上一点，" + describe(e, rng.randrange(3)),
            ]
        question = e["anchor"] + "，" + rng.choice(QUESTIONS["train"][field])
        # Only positive available facts have QA targets. Uncertainty is learned by
        # preserving the statement's meaning, not copying a prescribed refusal.
        answer = (
            rng.choice(ANSWERS[field]).format(FIELDS[task][target]) if target is not None else None
        )
        cases.append(
            dict(
                id=e["id"],
                texts=texts,
                question=question,
                answer=answer,
                kind=kind,
                training_field=TASKS[field],
            )
        )
    return cases


def train(source, out, steps=120):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(57201)
    rng = random.Random(57201)
    cases = data()
    save(out / "train.json", cases)
    original = SemanticMemory.load(source)
    model = EpisodeMemory(original)
    start = time.perf_counter()
    texts = list(dict.fromkeys(text for c in cases for text in c["texts"]))
    embedding_rows = [model.embed_text(t).float() for t in texts]
    token_ids = sorted(
        {i for text in texts for i in model.tokenizer.encode(text, add_special_tokens=False)}
    )
    table = (
        model.base.get_input_embeddings()(torch.tensor(token_ids, device=model.device))
        .detach()
        .float()
    )
    # Training-only PCA initialization; no evaluation utterances or answers enter it.
    _, eigvecs = torch.linalg.eigh(table.T @ table / len(table))
    basis = eigvecs[:, -model.codec.width :]
    with torch.no_grad():
        model.codec.encoder.weight.copy_(basis.T)
        model.codec.decoder.weight.copy_(basis)
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.0002, weight_decay=0.001)
    scale = table.square().mean().detach()
    logs = []
    for step in range(400):
        loss = 0
        for _ in range(4):
            emb = rng.choice(embedding_rows)
            reconstructed = model.codec.decode(model.codec.encode(emb[None]))[0]
            loss = loss + F.mse_loss(reconstructed, emb) / scale
        loss = loss / 4
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logs.append(
        dict(
            stage="reconstruction",
            step=400,
            loss=float(loss.detach()),
            seconds=time.perf_counter() - start,
        )
    )
    print(json.dumps(logs[-1]), flush=True)
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eligible = [c for c in cases if c["answer"] is not None]
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.00002, weight_decay=0.001)
    # Preserve both input fidelity and positive-fact use. No operation classification,
    # field-routing objective, template gate, or fixed-refusal language supervision.
    for step in range(1, steps + 1):
        c = rng.choice(eligible)
        fragments, reconstruction = [], 0
        for text in c["texts"]:
            embeddings = model.embed_text(text).float()
            code = model.codec.encode(embeddings[None])[0]
            fragments.append((code, "user"))
            reconstruction = (
                reconstruction + F.mse_loss(model.codec.decode(code), embeddings) / scale
            )
        qa = model.training_loss(fragments, c["question"], c["answer"])
        loss = qa + 2 * reconstruction / len(fragments)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.codec.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            logs.append(
                dict(
                    stage="qa_and_reconstruction",
                    step=step,
                    loss=float(loss.detach()),
                    qa=float(qa.detach()),
                    reconstruction=float(reconstruction.detach()) / len(fragments),
                    seconds=time.perf_counter() - start,
                )
            )
            print(json.dumps(logs[-1]), flush=True)
    model.base.gradient_checkpointing_disable()
    torch.save(model.codec.state_dict(), out / "codec.pt")
    save(out / "training_log.json", logs)
    meta = dict(
        source_run=str(source.resolve()),
        backbone=original.backbone_path.as_posix(),
        width=model.codec.width,
        codec_sha256=hashlib.sha256((out / "codec.pt").read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((out / "train.json").read_bytes()).hexdigest(),
        training_cases=len(cases),
        training_fragments=len(texts),
        unique_training_tokens=len(token_ids),
        positive_qa_cases=len(eligible),
        qa_steps=steps,
        reconstruction_steps=400,
        parameters=sum(p.numel() for p in model.codec.parameters()),
        fixed_refusal_targets=0,
        backbone_frozen=True,
        encoder="learned_width_projection_and_local_context",
        storage="ordered variable-length float16 vectors; no raw text or token IDs",
        limitations="Sequence length is preserved; representation may reconstruct wording. Not human-like abstraction, not constant-size memory, not proven unlimited context.",
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        seconds=time.perf_counter() - start,
    )
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    save(out / "metadata.json", meta)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/fact_reader_01")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=120)
    a = p.parse_args()
    train(a.source, a.out, a.steps)
