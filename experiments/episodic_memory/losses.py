"""Explicit supervision, not answers passed to the inference network."""

import torch
from torch.nn import functional as F


def routing_loss(query_keys, event_keys, positive_indices, temperature=0.07):
    scores = torch.einsum("bd,bkd->bk", query_keys.float(), event_keys.float())
    return F.cross_entropy(scores / temperature, positive_indices)


def total_loss(answer_logits, labels, route_loss, auxiliary_loss, route_weight=0.2):
    answer_loss = F.cross_entropy(answer_logits.float(), labels)
    return answer_loss + route_weight * route_loss + 0.1 * auxiliary_loss, answer_loss
