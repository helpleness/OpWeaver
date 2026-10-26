"""Broaden the latent channel on a reproducibly sampled open corpus.

Reports corpus bytes, sampled tokens and optimization exposure separately.
Only memory codec weights change; dialogue answer strings are not SFT targets.
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .episode_memory import EpisodeMemory
from .stream_experiment import save


def reservoir(path, capacities, seed):
    rng = random.Random(seed)
    samples = {k: [] for k in capacities}
    seen = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            source = r["source"]
            seen[source] += 1
            bucket = samples[source]
            if len(bucket) < capacities[source]:
                bucket.append(r)
            else:
                j = rng.randrange(seen[source])
                if j < capacities[source]:
                    bucket[j] = r
    return [r for bucket in samples.values() for r in bucket], dict(seen)


def chunks(model, records, seed):
    rng = random.Random(seed)
    rows, provenance = [], []
    for start in range(0, len(records), 128):
        batch = records[start : start + 128]
        tokens = model.tokenizer([r["text"] for r in batch], add_special_tokens=False).input_ids
        for r, ids in zip(batch, tokens):
            if len(ids) < 16:
                continue
            offset = rng.randrange(max(1, len(ids) - 192 + 1))
            sample = ids[offset : offset + 192]
            rows.append(torch.tensor(sample, dtype=torch.int32))
            provenance.append(
                dict(
                    source=r["source"],
                    group_id=r["group_id"],
                    text_sha256=r["text_sha256"],
                    offset=offset,
                    tokens=len(sample),
                    language=r["language"],
                )
            )
    return rows, provenance


def train(corpus, source, out, steps=2400):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(57301)
    rng = random.Random(57301)
    model = EpisodeMemory.load(source)
    started = time.perf_counter()
    records, examined = reservoir(
        corpus / "train.jsonl", {"fineweb2_zh": 24000, "oasst2": 8000}, 57301
    )
    public, provenance = chunks(model, records, 57302)
    validation_records, _ = reservoir(
        corpus / "validation.jsonl", {"fineweb2_zh": 256, "oasst2": 128}, 57303
    )
    validation, validation_provenance = chunks(model, validation_records, 57304)
    train_groups = {r["group_id"] for r in provenance}
    assert not train_groups.intersection(r["group_id"] for r in validation_provenance)
    save(out / "training_sample.json", provenance)
    save(out / "validation_sample.json", validation_provenance)
    torch.save(public, out / "training_tokens.pt")
    torch.save(validation, out / "validation_tokens.pt")
    print(
        json.dumps(
            dict(
                stage="sampled",
                documents=len(records),
                chunks=len(public),
                sampled_tokens=sum(len(r) for r in public),
            )
        ),
        flush=True,
    )
    synthetic_cases = json.loads((source / "train.json").read_text(encoding="utf-8"))
    synthetic = [
        torch.tensor(model.tokenizer.encode(t, add_special_tokens=False), dtype=torch.int32)
        for c in synthetic_cases
        for t in c["texts"]
    ]

    def batch_embed(rows):
        length = max(len(r) for r in rows)
        ids = torch.full(
            (len(rows), length), model.tokenizer.pad_token_id, device=model.device, dtype=torch.long
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row.to(model.device).long()
            mask[i, : len(row)] = True
        embeddings = model.base.get_input_embeddings()(ids).detach().float()
        embeddings[~mask] = 0
        return embeddings, mask

    @torch.no_grad()
    def fidelity():
        err, scale, tokens = 0.0, 0.0, 0
        for i in range(0, len(validation), 16):
            emb, mask = batch_embed(validation[i : i + 16])
            pred = model.codec.decode(model.codec.encode(emb))
            err += float((pred[mask] - emb[mask]).square().sum())
            scale += float(emb[mask].square().sum())
            tokens += int(mask.sum())
        return dict(
            relative_embedding_mse=err / scale,
            tokens=tokens,
            note="Representation fidelity, not answer accuracy",
        )

    before = fidelity()
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.0001, weight_decay=0.001)
    exposure = Counter()
    logs = []
    for step in range(1, steps + 1):
        rows = [rng.choice(public) for _ in range(24)] + [rng.choice(synthetic) for _ in range(8)]
        emb, mask = batch_embed(rows)
        pred = model.codec.decode(model.codec.encode(emb))
        loss = F.mse_loss(pred[mask], emb[mask]) / emb[mask].square().mean().clamp_min(1e-8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exposure["public_reconstruction_tokens"] += sum(len(r) for r in rows[:24])
        exposure["synthetic_reconstruction_tokens"] += sum(len(r) for r in rows[24:])
        if step == 1 or step % 400 == 0:
            log = dict(
                stage="open_reconstruction",
                step=step,
                loss=float(loss.detach()),
                seconds=time.perf_counter() - started,
            )
            logs.append(log)
            print(json.dumps(log), flush=True)

    # Distill the frozen base's next-token distribution on public passages.
    # This encourages its ordinary language behavior after the latent bottleneck;
    # it does not assert that the teacher's factual answers are gold.
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.codec.parameters(), lr=0.00001, weight_decay=0.001)
    for step in range(1, 81):
        row = rng.choice(public)
        emb, mask = batch_embed([row])
        positions = torch.linspace(8, len(row) - 2, 12, device=model.device).long()
        with torch.no_grad():
            teacher_h = model.base.model(
                inputs_embeds=emb.to(model.base.dtype), attention_mask=mask, use_cache=False
            ).last_hidden_state[0, positions]
            teacher_p = model.base.lm_head(teacher_h).float().softmax(-1)
        decoded = model.codec.decode(model.codec.encode(emb))
        student_h = model.base.model(
            inputs_embeds=decoded.to(model.base.dtype), attention_mask=mask, use_cache=False
        ).last_hidden_state[0, positions]
        student_lp = model.base.lm_head(student_h).float().log_softmax(-1)
        kl = F.kl_div(student_lp, teacher_p, reduction="batchmean")
        reconstruction = F.mse_loss(decoded[mask], emb[mask]) / emb[mask].square().mean().clamp_min(
            1e-8
        )
        loss = kl + 2 * reconstruction
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.codec.parameters(), 1.0)
        optimizer.step()
        exposure["public_distillation_input_tokens"] += len(row)
        exposure["distillation_prediction_positions"] += len(positions)
        if step % 20 == 0:
            log = dict(
                stage="frozen_base_distillation",
                step=step,
                kl=float(kl.detach()),
                seconds=time.perf_counter() - started,
            )
            logs.append(log)
            print(json.dumps(log), flush=True)
    model.base.gradient_checkpointing_disable()
    after = fidelity()
    torch.save(model.codec.state_dict(), out / "codec.pt")
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    meta.update(
        codec_sha256=hashlib.sha256((out / "codec.pt").read_bytes()).hexdigest(),
        public_training_source=str(source.resolve()),
        public_corpus=str(corpus.resolve()),
        public_corpus_manifest=json.loads((corpus / "manifest.json").read_text(encoding="utf-8")),
        sampled_public_documents=len(records),
        sampled_public_chunks=len(public),
        sampled_public_tokens=sum(len(r) for r in public),
        sampled_sources=dict(Counter(r["source"] for r in provenance)),
        sampled_languages=dict(Counter(r["language"] for r in provenance)),
        examined_corpus_records=examined,
        optimization_exposure=dict(exposure),
        open_reconstruction_steps=steps,
        distillation_steps=80,
        validation_fidelity_before=before,
        validation_fidelity_after=after,
        fixed_refusal_targets=0,
        backbone_frozen=True,
        seconds=time.perf_counter() - started,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        corpus_consumption_warning="The multi-GB prepared corpus was sampled; it was NOT all used for gradient updates. Exposure counts include repeated draws.",
    )
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    save(out / "metadata.json", meta)
    save(out / "training_log.json", logs)
    print(json.dumps(dict(stage="done", before=before, after=after, **exposure)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument(
        "--source", type=Path, default=Path("experiments/episodic_memory/runs/episode_memory_02")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2400)
    a = p.parse_args()
    train(a.corpus, a.source, a.out, a.steps)
