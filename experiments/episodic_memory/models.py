from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .generate_data import ANSWERS, FIELDS, TASKS


@dataclass
class Config:
    vocab_size: int
    dim: int = 192
    event_layers: int = 4
    query_layers: int = 2
    reader_layers: int = 2
    heads: int = 4
    ff_mult: int = 4
    slots: int = 4
    key_dim: int = 64
    max_length: int = 96
    dropout: float = 0.0
    address_source: str = "slots"
    route_weight: float = 0.2
    context_weight: float = 0.0
    intent_bottleneck: bool = False
    intent_weight: float = 0.0
    consistency_weight: float = 0.0
    verified_reader: bool = False
    evidence_threshold: float = 0.99
    value_threshold: float = 0.8
    semantic_verifier: bool = False
    typed_reader: bool = False
    refined_context: bool = False
    grounded_context: bool = False

    def to_dict(self):
        return asdict(self)


class TextEncoder(nn.Module):
    def __init__(self, c: Config, layers: int):
        super().__init__()
        self.embedding = nn.Embedding(c.vocab_size, c.dim, padding_idx=0)
        self.position = nn.Embedding(c.max_length, c.dim)
        layer = nn.TransformerEncoderLayer(
            c.dim, c.heads, c.dim * c.ff_mult, c.dropout, batch_first=True, norm_first=True
        )
        self.network = nn.TransformerEncoder(
            layer, layers, nn.LayerNorm(c.dim), enable_nested_tensor=False
        )

    def forward(self, tokens):
        pad = tokens.eq(0)
        x = self.embedding(tokens) + self.position(
            torch.arange(tokens.size(1), device=tokens.device)
        )
        return self.network(x, src_key_padding_mask=pad), pad


class MemoryModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        if c.verified_reader and not c.context_weight:
            raise ValueError("Verified reader requires the trained context prediction heads")
        if c.typed_reader and not (c.verified_reader and c.intent_bottleneck):
            raise ValueError("Typed reader requires verification and predicted intent")
        if c.grounded_context and not c.typed_reader:
            raise ValueError("Grounded context requires the separate question encoder")
        self.config = c
        self.event_encoder = TextEncoder(c, c.event_layers)
        self.query_encoder = TextEncoder(c, c.query_layers)
        if c.typed_reader:
            self.typed_intent_encoder = TextEncoder(c, c.query_layers)
        if c.grounded_context:
            self.query_context_heads = nn.ModuleList([nn.Linear(c.dim, 16), nn.Linear(c.dim, 16)])
        self.slot_queries = nn.Parameter(torch.randn(c.slots, c.dim) * 0.02)
        self.pool = nn.MultiheadAttention(c.dim, c.heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(c.dim)
        self.event_key = nn.Linear(
            c.dim * c.slots if c.address_source == "slots" else c.dim, c.key_dim
        )
        self.query_key = nn.Linear(c.dim, c.key_dim)
        if c.intent_bottleneck:
            self.intent_head = nn.Linear(c.dim, len(TASKS))
            self.intent_embeddings = nn.Parameter(torch.randn(len(TASKS), c.dim) * 0.02)
        if c.context_weight:
            self.context_heads = nn.ModuleList([nn.Linear(c.key_dim, 16), nn.Linear(c.key_dim, 16)])
        if c.refined_context:
            self.context_refiners = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(c.key_dim, 128), nn.GELU(), nn.Linear(128, 16))
                    for _ in range(2)
                ]
            )
            for refiner in self.context_refiners:
                nn.init.zeros_(refiner[-1].weight)
                nn.init.zeros_(refiner[-1].bias)
        self.read_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    c.dim, c.heads, c.dim * c.ff_mult, c.dropout, batch_first=True, norm_first=True
                )
                for _ in range(c.reader_layers)
            ]
        )
        self.read_norm = nn.LayerNorm(c.dim)
        self.answer = nn.Sequential(
            nn.Linear(c.dim + 2, c.dim), nn.GELU(), nn.Linear(c.dim, len(ANSWERS))
        )
        self.slot_for_task = [1, 1, 0, 2, 2, 3]
        self.auxiliary = nn.ModuleList([nn.Linear(c.dim, len(FIELDS[t])) for t in TASKS])
        if c.verified_reader:
            self.verifier = nn.Sequential(
                nn.Linear(5 if c.semantic_verifier else 2 * c.key_dim + 5, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    def encode_events(self, tokens):
        h, pad = self.event_encoder(tokens)
        seeds = self.slot_queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        z, _ = self.pool(seeds, h, h, key_padding_mask=pad, need_weights=False)
        z = self.slot_norm(z + seeds)
        address = z.flatten(1) if self.config.address_source == "slots" else h[:, 0]
        key = F.normalize(self.event_key(address).float(), dim=-1)
        return z, key

    def encode_questions(self, tokens):
        h, _ = self.query_encoder(tokens)
        summary = h[:, 0]
        key = F.normalize(self.query_key(summary).float(), dim=-1)
        if self.config.typed_reader:
            summary = self.typed_intent_encoder(tokens)[0][:, 0]
        return summary, key

    def context_logits(self, keys, index):
        logits = self.context_heads[index](10 * keys)
        if self.config.refined_context:
            logits = logits + self.context_refiners[index](10 * keys)
        return logits

    def verify_pairs(self, qkey, keys, memories, summary=None):
        q = qkey[:, None].float().expand_as(keys)
        k = keys.float()
        cos = (q * k).sum(-1, keepdim=True)
        agreement = [
            (
                (
                    self.query_context_heads[i](summary).float().softmax(-1)[:, None]
                    if self.config.grounded_context
                    else self.context_logits(q, i).float().softmax(-1)
                )
                * self.context_logits(k, i).float().softmax(-1)
            ).sum(-1, keepdim=True)
            for i in range(2)
        ]
        rms = memories.float().square().mean((-1, -2)).sqrt()[..., None]
        peak = memories.float().abs().amax((-1, -2))[..., None]
        features = torch.cat(
            [cos, *agreement, rms, peak]
            if self.config.semantic_verifier
            else [(q - k).abs(), q * k, cos, *agreement, rms, peak],
            -1,
        )
        pair_logits = self.verifier(features).squeeze(-1)
        if self.config.grounded_context:
            # Explicitly retain both address distributions' uncertainty. A neural
            # verifier cannot amplify weak project/ordinal agreement past the gate.
            joint = (agreement[0] * agreement[1]).squeeze(-1).clamp(1e-7, 1 - 1e-7)
            pair_logits = torch.minimum(pair_logits, torch.logit(joint))
        return pair_logits

    def read_parts(self, summary, qkey, memories, keys, valid=None, intent_override=None):
        # No raw events, answer labels, owner identifiers, or event IDs reach the reader.
        b, k, slots, dim = memories.shape
        scores = torch.einsum("bd,bkd->bk", qkey.float(), keys.float())
        if valid is None:
            valid = torch.ones_like(scores, dtype=torch.bool)
        if not valid.any(dim=1).all():
            # Empty stores have one dummy zero-memory entry.
            valid = valid.clone()
            valid[~valid.any(dim=1), 0] = True
        masked = scores.masked_fill(~valid, -1e4)
        pair_logits = (
            self.verify_pairs(qkey, keys, memories, summary)
            if self.config.verified_reader
            else None
        )
        attention_scores = (
            pair_logits.masked_fill(~valid, -1e4) if pair_logits is not None else masked / 0.07
        )
        log_weights = F.log_softmax(attention_scores, dim=-1)
        if self.config.typed_reader:
            # Marginalize event and field uncertainty instead of letting a decoder
            # amplify a weak intent prediction into a confident answer of another type.
            intent = (
                self.intent_head(summary).float().softmax(-1)
                if intent_override is None
                else intent_override
            )
            if intent.shape != (b, len(TASKS)):
                raise ValueError("Expected one probability per query and field")
            probabilities = torch.cat(
                [
                    (
                        log_weights.exp()[..., None]
                        * head(memories[:, :, self.slot_for_task[i]]).float().softmax(-1)
                    ).sum(1)
                    * intent[:, i : i + 1]
                    for i, head in enumerate(self.auxiliary)
                ],
                -1,
            )
            raw = torch.cat(
                [torch.zeros_like(probabilities[:, :1]), probabilities.clamp_min(1e-30).log()], -1
            )
            return raw, scores, pair_logits, valid
        bias = log_weights.repeat_interleave(slots, dim=-1)
        bias = (
            bias[:, None, None, :]
            .expand(b, self.config.heads, 1, k * slots)
            .reshape(b * self.config.heads, 1, k * slots)
        )
        memory = memories.reshape(b, k * slots, dim)
        if self.config.intent_bottleneck:
            # The task is predicted from text. Ground-truth task IDs never enter read().
            intent = self.intent_head(summary).float().softmax(-1)
            summary = (intent @ self.intent_embeddings.float()).to(summary.dtype)
        x = summary[:, None]
        for layer in self.read_layers:
            x = layer(x, memory, memory_mask=bias.to(x.dtype))
        probabilities = log_weights.exp()
        entropy = -(probabilities * log_weights).sum(-1)
        extra = torch.stack([masked.max(-1).values, entropy], -1)
        if self.config.verified_reader:
            extra = torch.zeros_like(extra)  # Answer identity does not decide evidence sufficiency.
        features = torch.cat([self.read_norm(x[:, 0]).float(), extra], -1)
        return self.answer(features), scores, pair_logits, valid

    def read(self, summary, qkey, memories, keys, valid=None, intent_override=None):
        raw, scores, pairs, valid = self.read_parts(
            summary, qkey, memories, keys, valid, intent_override
        )
        if pairs is None:
            return raw, scores
        values = F.log_softmax(raw[:, 1:].float(), -1)
        evidence = pairs.masked_fill(~valid, -1e4).sigmoid().max(-1).values
        best = values.max(-1).values
        # Argmax accepts only when BOTH predeclared tests pass. Calibration may only raise thresholds.
        refusal_margin = torch.maximum(
            torch.log(torch.tensor(self.config.evidence_threshold, device=raw.device))
            - evidence.clamp_min(1e-8).log(),
            torch.log(torch.tensor(self.config.value_threshold, device=raw.device)) - best,
        )
        return torch.cat([(best + refusal_margin)[:, None], values], -1), scores

    def auxiliary_loss(self, memories, labels):
        return torch.stack(
            [
                F.cross_entropy(head(memories[:, self.slot_for_task[i]]).float(), labels[:, i])
                for i, head in enumerate(self.auxiliary)
            ]
        ).mean()

    def context_loss(self, keys, labels):
        return torch.stack(
            [
                F.cross_entropy(head(10 * keys).float(), labels[:, i])
                for i, head in enumerate(self.context_heads)
            ]
        ).mean()

    def freeze_writer(self):
        for component in (self.event_encoder, self.pool, self.slot_norm, self.event_key):
            component.requires_grad_(False)
        self.slot_queries.requires_grad_(False)


def pad_tokens(rows, device):
    result = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        result[i, : len(row)] = torch.tensor(row, device=device)
    return result
