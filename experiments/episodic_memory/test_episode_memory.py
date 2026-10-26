import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .episode_memory import EpisodeStore, SequenceCodec
from .train_episode_memory import data


class EpisodeTests(unittest.TestCase):
    def test_fragments_keep_order_speaker_and_payload_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.sqlite"
            s = EpisodeStore(path, "v", 4, create=True)
            key = np.ones(32, dtype=np.float32)
            for eid, length, speaker, owner in [
                ("early", 3, "user", "a"),
                ("later", 5, "assistant", "a"),
                ("other", 2, "user", "b"),
            ]:
                s.add_many(
                    [
                        dict(
                            id=eid,
                            owner=owner,
                            timestamp=0,
                            memory=np.full((length, 4), length),
                            key=key,
                            speaker=speaker,
                        )
                    ]
                )
            s.close()
            s = EpisodeStore(path, "v", 4)
            self.assertEqual(
                s.ordered("a", ["later", "early", "other"]),
                [("early", "user"), ("later", "assistant")],
            )
            zs, _ = s.fetch("a", ["later", "early"])
            self.assertEqual([z.shape for z in zs], [(5, 4), (3, 4)])
            with self.assertRaises(KeyError):
                s.fetch("a", ["other"])
            with self.assertRaises(ValueError):
                s.add_many(
                    [
                        dict(
                            id="bad",
                            owner="a",
                            timestamp=0,
                            memory=np.ones((3, 4)),
                            key=key,
                            speaker="untrusted-role",
                        )
                    ]
                )
            self.assertEqual(s.counts()["events"], 3)
            self.assertEqual(s.db.execute("SELECT COUNT(*) FROM fragment_meta").fetchone()[0], 3)
            s.close()

    def test_codec_does_not_remove_negation_or_select_a_field_by_code(self):
        codec = SequenceCodec(12, 4)
        x = torch.randn(1, 17, 12)
        z = codec.encode(x)
        self.assertEqual(z.shape, (1, 17, 4))
        loss = codec.decode(z).square().mean()
        loss.backward()
        self.assertIsNotNone(codec.encoder.weight.grad)
        self.assertEqual(set(dict(codec.named_children())), {"encoder", "decoder", "context"})

    def test_training_has_no_fixed_refusal_targets_or_update_labels(self):
        rows = data(count=40)
        self.assertTrue(any(r["answer"] is None for r in rows))
        self.assertFalse(any(r["answer"] == "我没有这段经历的记忆。" for r in rows))
        self.assertTrue(all("operation_label" not in r for r in rows))
        self.assertEqual(len({(r["training_field"], r["kind"]) for r in rows}), 24)


if __name__ == "__main__":
    unittest.main()
