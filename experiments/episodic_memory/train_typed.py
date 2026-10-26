"""Train explicit intent/value reading on disk latents; writer and routing stay frozen."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .generate_data import OFFSETS, ORDINALS, PROJECTS, TASKS, load_events
from .models import Config, MemoryModel, pad_tokens
from .residual_language import training_question, training_question_grounded
from .store import writer_version
from .train import load_checkpoint, save_checkpoint, seed_everything
from .train_evidence import calibrate, open_store, question_text, validate_positive


def train(base, source_data, stores_root, out, steps=3000, batch_size=32, grounded=False):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    seed_everything(44)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source, tok, _ = load_checkpoint(base / "model.pt", device)
    config = Config(**source.config.to_dict())
    config.typed_reader = True
    config.refined_context = True
    config.grounded_context = grounded
    model = MemoryModel(config).to(device)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    assert not unexpected and all(
        n.startswith(("typed_intent_encoder.", "context_refiners.", "query_context_heads."))
        for n in missing
    )
    if not source.config.typed_reader:
        model.typed_intent_encoder.load_state_dict(source.query_encoder.state_dict())
    model.requires_grad_(False)
    for component in (
        model.typed_intent_encoder,
        model.intent_head,
        model.auxiliary,
        model.context_refiners,
        model.verifier,
    ):
        component.requires_grad_(True)
    if grounded:
        model.query_context_heads.requires_grad_(True)
    verifier = list(model.verifier.parameters())
    ids = {id(p) for p in verifier}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad and id(p) not in ids],
                "lr": 3e-4,
            },
            {"params": verifier, "lr": 1e-4},
        ]
    )
    events = load_events(source_data / "data", "train")
    by_id = {e["id"]: e for e in events}
    stored, absent = [e for e in events if e["stored"]], [e for e in events if not e["stored"]]
    stores = [open_store(source_data / "memory_train.sqlite", model, tok)] + [
        open_store(stores_root / f"mixed_train_{v}.sqlite", model, tok) for v in (1, 2)
    ]
    rng = random.Random(44)
    started = time.perf_counter()

    def context(e):
        return [PROJECTS.index(e["project"]), ORDINALS.index(e["ordinal"])]

    def balanced_mean(loss, targets):
        groups = [loss[targets > 0], loss[targets == 0]]
        return torch.stack([g.mean() for g in groups if g.numel()]).mean()

    model.train()
    for step in range(1, steps + 1):
        es = [rng.choice(absent if rng.random() < 0.2 else stored) for _ in range(batch_size)]
        tasks = [rng.choice(TASKS) for _ in es]
        augment = training_question_grounded if grounded else training_question
        probability = 0.5 if grounded else 0.35
        texts = [
            (augment if rng.random() < probability else question_text)(e, t, rng)
            for e, t in zip(es, tasks)
        ]
        encoded = [tok.encode(text) for text in texts]
        assert all(2 not in row for row in encoded), (
            "Training text contains out-of-vocabulary symbols"
        )
        tokens = pad_tokens(encoded, device)
        h, q = model.encode_questions(tokens)
        zs, ks, masks, targets, labels, fields, kcontexts, payload_masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i, (e, t) in enumerate(zip(es, tasks)):
            store = rng.choice(stores)
            exclude = [e["id"]] if e["stored"] and rng.random() < 0.25 else []
            selected, _ = store.retrieve(
                e["owner"],
                q[i].detach().cpu().numpy(),
                flat=rng.random() < 0.5,
                top_clusters=2,
                exclude=exclude,
            )
            z, k = store.fetch(e["owner"], selected)
            blank = rng.random() < 0.1
            if blank:
                z[:] = 0
            n = len(selected)
            support = e["stored"] and e["id"] in selected and not blank
            zs.append(np.pad(z, ((0, 8 - n), (0, 0), (0, 0))))
            ks.append(np.pad(k, ((0, 8 - n), (0, 0))))
            masks.append([True] * n + [False] * (8 - n))
            payload_masks.append([not blank] * n + [False] * (8 - n))
            targets.append(
                [float(eid == e["id"] and support) for eid in selected] + [0.0] * (8 - n)
            )
            labels.append(OFFSETS[t] + e["values"][t] if support else 0)
            fields.append(
                [[by_id[eid]["values"][t] for t in TASKS] for eid in selected] + [[0] * 6] * (8 - n)
            )
            kcontexts.append([context(by_id[eid]) for eid in selected] + [[0, 0]] * (8 - n))
        z, k = torch.tensor(np.stack(zs), device=device), torch.tensor(np.stack(ks), device=device)
        valid, payload = (
            torch.tensor(masks, device=device),
            torch.tensor(payload_masks, device=device),
        )
        targets, labels = torch.tensor(targets, device=device), torch.tensor(labels, device=device)
        fields, kc = torch.tensor(fields, device=device), torch.tensor(kcontexts, device=device)
        qc = torch.tensor([context(e) for e in es], device=device)
        raw, _, pairs, _ = model.read_parts(h, q, z, k, valid)
        supported = labels != 0
        content = (
            F.cross_entropy(raw[supported, 1:], labels[supported] - 1)
            if supported.any()
            else raw.sum() * 0
        )
        intent = F.cross_entropy(
            model.intent_head(h), torch.tensor([TASKS.index(t) for t in tasks], device=device)
        )
        auxiliary = torch.stack(
            [
                F.cross_entropy(
                    head(z[:, :, model.slot_for_task[j]][payload]), fields[:, :, j][payload]
                )
                for j, head in enumerate(model.auxiliary)
            ]
        ).mean()
        contexts = torch.stack(
            [
                F.cross_entropy(
                    model.query_context_heads[j](h) if grounded else model.context_logits(q, j),
                    qc[:, j],
                )
                + F.cross_entropy(model.context_logits(k[valid], j), kc[:, :, j][valid])
                for j in range(2)
            ]
        ).mean()
        evidence = balanced_mean(
            F.binary_cross_entropy_with_logits(pairs[valid], targets[valid], reduction="none"),
            targets[valid],
        )
        margin = balanced_mean(F.relu(6 - (2 * targets[valid] - 1) * pairs[valid]), targets[valid])
        loss = content + intent + 0.5 * auxiliary + 0.3 * contexts + evidence + 0.2 * margin
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite typed-reader loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = {
                "step": step,
                "elapsed_s": time.perf_counter() - started,
                **{
                    name: float(value.detach())
                    for name, value in dict(
                        loss=loss,
                        content=content,
                        intent=intent,
                        auxiliary=auxiliary,
                        contexts=contexts,
                        evidence=evidence,
                    ).items()
                },
            }
            with (out / "training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    for store in stores:
        store.close()
    model.eval()
    assert writer_version(source, tok.spec) == writer_version(model, tok.spec)
    for name, value in source.state_dict().items():
        if name.startswith(("query_encoder.", "query_key.", "context_heads.")):
            assert torch.equal(value, model.state_dict()[name]), name
    validation = load_events(source_data / "data", "validation")
    stores = [open_store(source_data / "memory_validation.sqlite", model, tok)] + [
        open_store(stores_root / f"mixed_validation_{v}.sqlite", model, tok) for v in (1, 2)
    ]
    calibration = calibrate(model, tok, validation, stores, device)
    calibration["positive_validation"] = validate_positive(model, tok, validation, stores, device)
    for store in stores:
        store.close()
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    metadata = dict(
        base=str(base),
        steps=steps,
        batch_size=batch_size,
        seed=44,
        grounded_context=grounded,
        elapsed_s=time.perf_counter() - started,
        writer_version=writer_version(model, tok.spec),
        frozen_routing=True,
        parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    )
    save_checkpoint(out / "model.pt", model, tok, metadata)
    (out / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in calibration.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        type=Path,
        default=Path("experiments/episodic_memory/runs/evidence_round_03/verified"),
    )
    p.add_argument(
        "--source-data",
        type=Path,
        default=Path("experiments/episodic_memory/runs/paraphrase_structured"),
    )
    p.add_argument(
        "--stores", type=Path, default=Path("experiments/episodic_memory/runs/evidence_round_01")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grounded-context", action="store_true")
    a = p.parse_args()
    train(a.base, a.source_data, a.stores, a.out, a.steps, a.batch_size, a.grounded_context)
