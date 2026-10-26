"""Small invariant checks; no pretrained model download or GPU required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .qwen_memory import MemoryAdapter
from .store import MemoryStore


class QwenMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.a = MemoryAdapter(hidden=32, dim=16, structured=True).eval()

    def test_padding_cannot_change_written_memory_or_query_address(self):
        h = torch.randn(2, 7, 32)
        mask = torch.ones(2, 7, dtype=torch.bool)
        padded = torch.cat([h, torch.randn(2, 3, 32) * 1000], 1)
        padded_mask = torch.cat([mask, torch.zeros(2, 3, dtype=torch.bool)], 1)
        for expected, actual in zip(self.a.write(h, mask), self.a.write(padded, padded_mask)):
            torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            self.a.query_logits(h, mask),
            self.a.query_logits(padded, padded_mask),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_reader_gradients_reach_writer_through_fp16_storage_simulation(self):
        z, _ = self.a.write(torch.randn(2, 7, 32), torch.ones(2, 7, dtype=torch.bool))
        soft = self.a.soft_tokens(z.half().float(), torch.ones(2, dtype=torch.bool))
        soft.square().mean().backward()
        for parameter in (
            self.a.input_projection.weight,
            self.a.slot_queries,
            self.a.pool.in_proj_weight,
            self.a.prefix.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_null_memory_does_not_leak_passed_event(self):
        present = torch.zeros(2, dtype=torch.bool)
        x = self.a.soft_tokens(torch.randn(2, 8, 16), present)
        y = self.a.soft_tokens(torch.randn(2, 8, 16) * 100, present)
        torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_address_mismatch_cannot_pass(self):
        q = torch.full((1, 2, 16), -20.0)
        q[:, :, 0] = 20
        wrong = q.clone()
        wrong[:, 1, :] = -20
        wrong[:, 1, 1] = 20
        self.assertGreater(float(self.a.agreement(self.a.key(q), self.a.key(q))), 0.99)
        self.assertLess(float(self.a.agreement(self.a.key(q), self.a.key(wrong))), 1e-8)

    def test_disk_reopen_contains_no_text_and_isolates_owner(self):
        z, k = self.a.write(torch.randn(1, 7, 32), torch.ones(1, 7, dtype=torch.bool))
        z, k = z[0].detach().numpy(), k[0].detach().numpy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite"
            s = MemoryStore(path, "test-writer", (8, 16), 32, create=True)
            s.add_many([dict(id="event", owner="alice", timestamp=1, memory=z, key=k)])
            s.close()
            s = MemoryStore(path, "test-writer", (8, 16), 32)
            self.assertEqual(
                [r[1] for r in s.db.execute("PRAGMA table_info(events)")],
                ["event_id", "owner", "timestamp", "memory", "address"],
            )
            restored, _ = s.fetch("alice", ["event"])
            np.testing.assert_array_equal(restored[0], z.astype(np.float16).astype(np.float32))
            with self.assertRaises(KeyError):
                s.fetch("bob", ["event"])
            self.assertEqual(s.retrieve("bob", k)[0], [])
            s.close()
            with self.assertRaises(ValueError):
                MemoryStore(path, "changed-writer", (8, 16), 32)


if __name__ == "__main__":
    unittest.main()
