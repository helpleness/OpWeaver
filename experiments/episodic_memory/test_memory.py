from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .cluster import OnlineClusters
from .generate_data import TASKS, Tokenizer, build_tokenizer, generate, load_events, question
from .models import Config, MemoryModel, pad_tokens
from .store import MemoryStore, writer_version


class MemoryTests(unittest.TestCase):
    def test_split_merge_bounds_and_coverage(self):
        rng = np.random.default_rng(4)
        c = OnlineClusters(threshold=-1, max_members=4)
        for i in range(35):
            c.add(str(i), rng.normal(size=8))
        c.merge_small(minimum=3, similarity=-1)
        self.assertGreater(c.splits, 0)
        self.assertEqual(set(c.keys), set().union(*c.groups.values()))
        self.assertTrue(all(len(v) <= 4 for v in c.groups.values()))
        self.assertTrue(all(sum(e in group for group in c.groups.values()) <= 2 for e in c.keys))
        for e, key in c.keys.items():
            ids, _ = c.select(key, flat=True)
            self.assertEqual(ids[0], e)

    def test_degenerate_keys_do_not_break_split(self):
        c = OnlineClusters(max_members=4)
        for i in range(20):
            c.add(str(i), np.ones(8))
        self.assertTrue(all(len(v) <= 4 for v in c.groups.values()))

    def test_store_isolation_atomicity_and_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.sqlite"
            store = MemoryStore(path, "v1", (4, 8), 8, create=True, max_members=4)
            z = np.arange(32, dtype=np.float32).reshape(4, 8) / 10
            record = {
                "id": "a",
                "owner": "owner1",
                "timestamp": 1,
                "memory": z,
                "key": np.ones(8) / np.sqrt(8),
            }
            store.add_many([record])
            before = store.fetch("owner1", ["a"])[0]
            with self.assertRaises(KeyError):
                store.fetch("owner2", ["a"])
            self.assertEqual(store.retrieve("owner2", np.ones(8))[0], [])
            with self.assertRaises(ValueError):
                store.add_many([{**record, "id": "b"}, record])
            self.assertEqual(store.counts()["events"], 1)
            self.assertEqual(set(store._index("owner1").keys), {"a"})
            columns = [r[1] for r in store.db.execute("PRAGMA table_info(events)")]
            self.assertEqual(columns, ["event_id", "owner", "timestamp", "memory", "address"])
            store.close()
            reopened = MemoryStore(path, "v1", (4, 8), 8)
            np.testing.assert_array_equal(before, reopened.fetch("owner1", ["a"])[0])
            np.testing.assert_array_equal(before[0], z.astype(np.float16).astype(np.float32))
            reopened.close()
            with self.assertRaises(ValueError):
                MemoryStore(path, "wrong-version", (4, 8), 8)

    def test_generated_histories_disjoint_and_tokenized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate(root, train_histories=2, eval_histories=1, train_events=8, test_events=8)
            tokenizer = Tokenizer(json.loads((root / "tokenizer.json").read_text(encoding="utf-8")))
            all_ids = set()
            for split in ("train", "validation", "test", "test_template"):
                events = load_events(root, split)
                ids = {e["id"] for e in events}
                self.assertFalse(ids & all_ids)
                all_ids |= ids
                for e in events:
                    self.assertNotIn(2, tokenizer.encode(e["text"]))
                    for t in TASKS:
                        self.assertNotIn(2, tokenizer.encode(question(e, t, e["query_template"])))

    def test_reader_gradient_reaches_writer_and_freezing_works(self):
        torch.set_num_threads(1)
        tok = Tokenizer(build_tokenizer())
        config = Config(len(tok.ids), dim=32, event_layers=1, query_layers=1, reader_layers=1)
        model = MemoryModel(config)
        tokens = pad_tokens([tok.encode("画册项目第1次讨论。感受高兴。")], "cpu")
        z, key = model.encode_events(tokens)
        h, q = model.encode_questions(tokens)
        logits, _ = model.read(h, q, z[:, None], key[:, None])
        logits.sum().backward()
        grad = model.event_encoder.embedding.weight.grad
        self.assertGreater(float(grad.abs().sum()), 0)
        original = writer_version(model, tok.spec)
        model.freeze_writer()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        h, q = model.encode_questions(tokens)
        logits, _ = model.read(h, q, z.detach()[:, None], key.detach()[:, None])
        logits.sum().backward()
        optimizer.step()
        self.assertEqual(original, writer_version(model, tok.spec))


if __name__ == "__main__":
    unittest.main()
