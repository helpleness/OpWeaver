"""Bounded retry policy. No truth labels, event text or known support ID are inputs."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def infer(
    model,
    store,
    owner,
    h,
    q,
    retry=True,
    exclude=(),
    zero=False,
    audit_all=False,
    flat_only=False,
    intent_override=None,
):
    attempts = []
    for flat in [True] if flat_only else ([False, True] if retry else [False]):
        ids, candidate_count = store.retrieve(
            owner, q[0].cpu().numpy(), top_clusters=2, flat=flat, exclude=exclude
        )
        z, keys = store.fetch(owner, ids)
        if zero:
            z = np.zeros_like(z)
        zt, kt = torch.tensor(z[None], device=q.device), torch.tensor(keys[None], device=q.device)
        raw, _, pairs, valid = model.read_parts(h, q, zt, kt, intent_override=intent_override)
        if pairs is None:
            pred = int(raw.argmax(-1)) if ids else 0
            evidence = None
        else:
            evidence = float(pairs.masked_fill(~valid, -1e4).sigmoid().max())
            value_prob, value_id = raw[:, 1:].softmax(-1).max(-1)
            pred = (
                int(value_id) + 1
                if ids
                and evidence > model.config.evidence_threshold
                and float(value_prob) > model.config.value_threshold
                else 0
            )
        attempts.append(
            {
                "flat": flat,
                "ids": ids,
                "candidate_count": candidate_count,
                "prediction": pred,
                "conditional_answer": int(raw[:, 1:].argmax(-1)) + 1,
                "value_probability": float(raw[:, 1:].softmax(-1).max()),
                "evidence": evidence,
            }
        )
        if model.config.typed_reader:
            intent = (
                model.intent_head(h).softmax(-1) if intent_override is None else intent_override
            )
            weights = pairs.masked_fill(~valid, -1e4).softmax(-1)
            attempts[-1].update(
                predicted_field=int(intent.argmax(-1)),
                field_probability=float(intent.max()),
                selected_id=ids[int(weights.argmax(-1))] if ids else None,
                event_probability=float(weights.max()),
            )
        if pred and not audit_all:
            break
    return {"prediction": attempts[-1]["prediction"], "attempts": attempts}
