"""Query-chunked SDPA for Windows builds without fused attention.

Keeps every key/value and the full causal receptive field. This changes execution
memory usage, not context length or weights. Only inference (dropout=0) is used.
"""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import repeat_kv


def chunked_sdpa(query, key, value, mask=None, causal=True, scale=None, chunk=128):
    length, keys = query.shape[-2], key.shape[-2]
    pieces = []
    key_positions = torch.arange(keys, device=query.device)
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        if mask is not None:
            part_mask = mask if mask.shape[-2] == 1 else mask[..., start:end, :]
        elif causal:
            positions = torch.arange(start, end, device=query.device) + keys - length
            part_mask = positions[:, None] >= key_positions[None]
        else:
            part_mask = None
        pieces.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[..., start:end, :],
                key,
                value,
                attn_mask=part_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        )
    return torch.cat(pieces, dim=-2)


def bounded_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs
):
    if query.shape[-2] <= 128:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
    if dropout:
        raise ValueError("Bounded attention is inference-only")
    groups = getattr(module, "num_key_value_groups", 1)
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    result = chunked_sdpa(query, key, value, attention_mask, causal=causal, scale=scaling)
    return result.transpose(1, 2).contiguous(), None


def install():
    # Preserve HF's sdpa mask construction, replacing only the attention execution.
    ALL_ATTENTION_FUNCTIONS.register("sdpa", bounded_forward)
