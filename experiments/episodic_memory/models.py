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
        self.config = c
        self.event_encoder = TextEncoder(c, c.event_layers)
        self.query_encoder = TextEncoder(c, c.query_layers)
        self.slot_queries = nn.Parameter(torch.randn(c.slots, c.dim) * 0.02)
        self.pool = nn.MultiheadAttention(c.dim, c.heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(c.dim)
        self.event_key = nn.Linear(
            c.dim * c.slots if c.address_source == "slots" else c.dim, c.key_dim
        )
        self.query_key = nn.Linear(c.dim, c.key_dim)
        if c.context_weight:
            self.context_heads = nn.ModuleList([nn.Linear(c.key_dim, 16), nn.Linear(c.key_dim, 16)])
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
        return summary, F.normalize(self.query_key(summary).float(), dim=-1)

    def read(self, summary, qkey, memories, keys, valid=None):
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
        log_weights = F.log_softmax(masked / 0.07, dim=-1)
        bias = log_weights.repeat_interleave(slots, dim=-1)
        bias = (
            bias[:, None, None, :]
            .expand(b, self.config.heads, 1, k * slots)
            .reshape(b * self.config.heads, 1, k * slots)
        )
        memory = memories.reshape(b, k * slots, dim)
        x = summary[:, None]
        for layer in self.read_layers:
            x = layer(x, memory, memory_mask=bias.to(x.dtype))
        probabilities = log_weights.exp()
        entropy = -(probabilities * log_weights).sum(-1)
        features = torch.cat(
            [self.read_norm(x[:, 0]).float(), masked.max(-1).values[:, None], entropy[:, None]], -1
        )
        return self.answer(features), scores

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
