"""Numerical causal/padding equivalence, including a cached prefix."""

import unittest

import torch

from .bounded_attention import chunked_sdpa


class AttentionTests(unittest.TestCase):
    def test_full_causal_context_is_preserved(self):
        torch.manual_seed(1)
        q, k, v = [torch.randn(1, 2, 37, 8) for _ in range(3)]
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        actual = chunked_sdpa(q, k, v, chunk=7)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_explicit_padding_mask(self):
        torch.manual_seed(2)
        q, k, v = [torch.randn(2, 2, 17, 8) for _ in range(3)]
        mask = torch.ones(2, 1, 17, 17, dtype=torch.bool).tril()
        mask[:, :, :, -3:] = False
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(
            chunked_sdpa(q, k, v, mask, chunk=4), expected, rtol=1e-5, atol=1e-6
        )

    def test_cached_prefix_offsets_and_future_masking(self):
        torch.manual_seed(3)
        q = torch.randn(1, 2, 7, 8)
        k, v = [torch.randn(1, 2, 19, 8) for _ in range(2)]
        mask = torch.arange(7)[:, None] + 12 >= torch.arange(19)[None]
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(chunked_sdpa(q, k, v, chunk=3), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
