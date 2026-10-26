from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .cluster import OnlineClusters
from .generate_data import (
    FIELDS,
    TASKS,
    Tokenizer,
    build_tokenizer,
    generate,
    load_events,
    question,
    render,
)
from .models import Config, MemoryModel, pad_tokens
from .paraphrases import augmented_event, augmented_question
from .store import MemoryStore, writer_version


class MemoryTests(unittest.TestCase):
    def test_external_intent_preserves_uncertainty_and_legacy_parity(self):
        torch.set_num_threads(1)
        c = Config(
            184,
            dim=32,
            event_layers=1,
            query_layers=1,
            reader_layers=1,
            context_weight=1.0,
            intent_bottleneck=True,
            verified_reader=True,
            semantic_verifier=True,
            typed_reader=True,
        )
        model = MemoryModel(c).eval()
        h, q, z, k = (
            torch.randn(2, 32),
            torch.randn(2, 64),
            torch.randn(2, 3, 4, 32),
            torch.randn(2, 3, 64),
        )
        p = model.intent_head(h).softmax(-1)
        torch.testing.assert_close(
            model.read_parts(h, q, z, k)[0], model.read_parts(h, q, z, k, intent_override=p)[0]
        )
        uniform = torch.full((2, 6), 1 / 6)
        raw = model.read_parts(h, q, z, k, intent_override=uniform)[0]
        self.assertTrue((raw[:, 1:].softmax(-1).max(-1).values <= 1 / 6 + 1e-6).all())
        self.assertTrue((model.read(h, q, z, k, intent_override=uniform)[0].argmax(-1) == 0).all())
        with self.assertRaises(ValueError):
            model.read_parts(h, q, z, k, intent_override=torch.ones(2, 5))

    def test_language_views_are_disjoint_across_train_validation(self):
        from .language_data import compose, create, load

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate(
                root / "source" / "data",
                train_histories=1,
                eval_histories=1,
                train_events=8,
                test_events=8,
            )
            create(root / "language", root / "source", count=120, val_count=24)
            for mode in ("narrow", "compositional"):
                train = load(root / "language" / f"{mode}_train.jsonl")
                val = load(root / "language" / f"{mode}_validation.jsonl")
                left = {s for r in train for s in (r["text"], r["positive"])}
                right = {s for r in val for s in (r["text"], r["positive"])}
                self.assertFalse(left & right)
                for family in (6, 7, 8, 9):
                    for task in TASKS:
                        text = compose(task, "画册项目第1次讨论", random.Random(19), family, True)
                        self.assertNotIn(text, left | right)

    def test_fresh_language_is_in_vocabulary_and_separate(self):
        from .residual_language import (
            SECOND_TEST_STEMS,
            TEST_STEMS,
            TRAIN_STEMS,
            fresh_event,
            fresh_question,
            training_question_grounded,
        )

        tok = Tokenizer(build_tokenizer(True))
        e = {"anchor": "画册项目第1次讨论", "values": dict.fromkeys(TASKS, 0)}
        for task in TASKS:
            self.assertFalse(
                set(TEST_STEMS[task] + SECOND_TEST_STEMS[task]) & set(TRAIN_STEMS[task])
            )
            for view in range(4):
                self.assertNotIn(2, tok.encode(fresh_question(e, task, view)))
            rng = random.Random(463)
            for _ in range(256):
                text = training_question_grounded(e, task, rng)
                self.assertNotIn(2, tok.encode(text))
                self.assertNotIn(text, [fresh_question(e, task, v) for v in range(4)])
        for view in range(4):
            text = fresh_event(e, view)
            self.assertNotIn(2, tok.encode(text))
            self.assertNotIn(text, [render(e, v) for v in (0, 1, 2, 3)])
            for task in TASKS:
                self.assertIn(FIELDS[task][0], text)

    @torch.no_grad()
    def test_grounded_evidence_cannot_amplify_uncertain_or_mismatched_address(self):
        c = Config(
            184,
            dim=32,
            event_layers=1,
            query_layers=1,
            reader_layers=1,
            context_weight=1.0,
            intent_bottleneck=True,
            verified_reader=True,
            semantic_verifier=True,
            typed_reader=True,
            refined_context=True,
            grounded_context=True,
        )
        model = MemoryModel(c).eval()
        h, q, k, z = (
            torch.randn(1, 32),
            torch.randn(1, 64),
            torch.randn(1, 2, 64),
            torch.randn(1, 2, 4, 32),
        )
        with torch.no_grad():
            model.verifier[-1].weight.zero_()
            model.verifier[-1].bias.fill_(20)
            for head in model.query_context_heads:
                head.weight.zero_()
                head.bias.zero_()
            for head in model.context_heads:
                head.weight.zero_()
                head.bias.fill_(-20)
                head.bias[0] = 20
        # Uniform project and ordinal predictions must stay far below 0.99,
        # regardless of the neural verifier's confident yes.
        self.assertLess(float(model.verify_pairs(q, k, z, h).sigmoid().max()), 0.01)
        with torch.no_grad():
            for head in model.query_context_heads:
                head.bias.fill_(-20)
                head.bias[0] = 20
        self.assertGreater(float(model.verify_pairs(q, k, z, h).sigmoid().min()), 0.99)
        with torch.no_grad():
            model.query_context_heads[1].bias[0] = -20
            model.query_context_heads[1].bias[1] = 20
        self.assertLess(float(model.verify_pairs(q, k, z, h).sigmoid().max()), 0.01)

    def test_typed_reader_propagates_uncertainty_and_ignores_padding(self):
        torch.set_num_threads(1)
        c = Config(
            184,
            dim=32,
            event_layers=1,
            query_layers=1,
            reader_layers=1,
            context_weight=1.0,
            intent_bottleneck=True,
            verified_reader=True,
            semantic_verifier=True,
            typed_reader=True,
            refined_context=True,
        )
        model = MemoryModel(c).eval()
        h, q = torch.randn(2, 32), torch.nn.functional.normalize(torch.randn(2, 64), dim=-1)
        z, k = torch.randn(2, 3, 4, 32), torch.randn(2, 3, 64)
        valid = torch.tensor([[True, True, False]] * 2)
        raw = model.read_parts(h, q, z, k, valid)[0]
        prob = raw[:, 1:].softmax(-1)
        torch.testing.assert_close(prob.sum(-1), torch.ones(2))
        self.assertTrue(
            (prob.max(-1).values <= model.intent_head(h).softmax(-1).max(-1).values + 1e-6).all()
        )
        zp, kp = z.clone(), k.clone()
        zp[:, 2], kp[:, 2] = 1000, -1000
        torch.testing.assert_close(raw, model.read_parts(h, q, zp, kp, valid)[0])
        order = [1, 0, 2]
        torch.testing.assert_close(
            raw, model.read_parts(h, q, z[:, order], k[:, order], valid[:, order])[0]
        )
        with torch.no_grad():
            model.intent_head.weight.zero_()
            model.intent_head.bias.zero_()
            model.verifier[-1].weight.zero_()
            model.verifier[-1].bias.fill_(20.0)
        self.assertTrue((model.read(h, q, z, k, valid)[0].argmax(-1) == 0).all())
        raw = model.read_parts(h, q, z, k, valid)[0]
        (-raw[:, 1]).sum().backward()
        self.assertGreater(float(model.auxiliary[0].weight.grad.abs().sum()), 0)

    def test_excluded_support_cannot_reappear_on_flat_retry(self):
        index = OnlineClusters(max_members=4)
        for i in range(10):
            index.add(str(i), np.eye(10)[i])
        for flat in (False, True):
            ids, _ = index.select(np.eye(10)[0], flat=flat, exclude=["0"])
            self.assertNotIn("0", ids)

    def test_verified_reader_gate_and_gradient(self):
        torch.set_num_threads(1)
        c = Config(
            184,
            dim=32,
            event_layers=1,
            query_layers=1,
            reader_layers=1,
            address_source="cls",
            context_weight=1.0,
            intent_bottleneck=True,
            verified_reader=True,
            semantic_verifier=True,
        )
        model = MemoryModel(c)
        h, q = torch.randn(2, 32), torch.nn.functional.normalize(torch.randn(2, 64), dim=-1)
        z, k = (
            torch.randn(2, 3, 4, 32),
            torch.nn.functional.normalize(torch.randn(2, 3, 64), dim=-1),
        )
        raw, _, pairs, valid = model.read_parts(h, q, z, k)
        raw[:, 1:].sum().backward(retain_graph=True)
        self.assertGreater(float(model.verifier[0].weight.grad.abs().sum()), 0)
        scores, _ = model.read(h, q, z, k)
        confidence, candidate = raw[:, 1:].softmax(-1).max(-1)
        accept = (
            pairs.masked_fill(~valid, -1e4).sigmoid().max(-1).values > c.evidence_threshold
        ) & (confidence > c.value_threshold)
        expected = torch.where(accept, candidate + 1, 0)
        torch.testing.assert_close(scores.argmax(-1), expected)
        with torch.no_grad():
            model.verifier[-1].weight.zero_()
            model.verifier[-1].bias.fill_(10.0)
            model.answer[-1].weight.zero_()
            model.answer[-1].bias.fill_(-10.0)
            model.answer[-1].bias[1] = 10.0
        self.assertTrue((model.read(h, q, z, k)[0].argmax(-1) == 1).all())
        with torch.no_grad():
            model.verifier[-1].bias.fill_(-10.0)
        self.assertTrue((model.read(h, q, z, k)[0].argmax(-1) == 0).all())
        with torch.no_grad():
            model.verifier[-1].bias.fill_(10.0)
            model.answer[-1].bias.zero_()
        self.assertTrue((model.read(h, q, z, k)[0].argmax(-1) == 0).all())
        c.evidence_threshold = 1.0
        self.assertTrue((model.read(h, q, z, k)[0].argmax(-1) == 0).all())

    def test_paraphrases_preserve_facts_and_exclude_holdout_sentences(self):
        tok = Tokenizer(build_tokenizer(extended=True))
        rng = random.Random(1127)
        e = {
            "anchor": "画册项目第1次讨论",
            "values": {t: i % len(FIELDS[t]) for i, t in enumerate(TASKS)},
        }
        for _ in range(256):
            text = augmented_event(e, rng)
            self.assertNotIn(text, [render(e, i) for i in (2, 3)])
            self.assertNotIn(2, tok.encode(text))
            self.assertIn(e["anchor"], text)
            for task in TASKS:
                self.assertIn(FIELDS[task][e["values"][task]], text)
                q = augmented_question(e, task, rng)
                self.assertNotIn(q, [question(e, task, i) for i in (2, 3)])
                self.assertNotIn(2, tok.encode(q))
        for i in (2, 3):
            self.assertNotIn(2, tok.encode(render(e, i)))
            for task in TASKS:
                self.assertNotIn(2, tok.encode(question(e, task, i)))

    def test_intent_and_multiview_loss_backpropagate_without_reader_labels(self):
        from .train import Corpus, forward_batch

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate(
                root,
                train_histories=1,
                eval_histories=1,
                train_events=8,
                test_events=8,
                extended_vocabulary=True,
            )
            tok = Tokenizer(build_tokenizer(True))
            corpus = Corpus(load_events(root, "train"), tok, augment=True, consistency=True)
            config = Config(
                len(tok.ids),
                dim=32,
                event_layers=1,
                query_layers=1,
                reader_layers=1,
                address_source="cls",
                context_weight=1.0,
                intent_bottleneck=True,
                intent_weight=0.5,
                consistency_weight=0.2,
            )
            model = MemoryModel(config)
            batch = corpus.batch(2, random.Random(31), "cpu")
            loss, logits, parts = forward_batch(model, batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(parts["consistency"], 0)
            loss.backward()
            self.assertGreater(float(model.intent_head.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.event_encoder.embedding.weight.grad.abs().sum()), 0)
            self.assertEqual(tuple(logits.shape), (2, 57))

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
        config = Config(
            len(tok.ids),
            dim=32,
            event_layers=1,
            query_layers=1,
            reader_layers=1,
            address_source="cls",
            context_weight=1.0,
        )
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
