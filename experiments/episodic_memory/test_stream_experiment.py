"""Causal evaluation construction, not tests of language-model competence."""

import tempfile
import unittest
from pathlib import Path

from .stream_experiment import messages, prepare, read


class StreamTests(unittest.TestCase):
    def test_checkpoints_never_reference_future_or_insert_exam_answers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "experiment"
            prepare(root)
            turns = read(root / "conversation.json")
            probes = read(root / "probes.json")
            ids = {t["id"]: t for t in turns}
            self.assertEqual(len(turns), 128)
            self.assertEqual(len(probes), 24)
            for p in probes:
                self.assertTrue(
                    all(ids[i]["number"] <= p["checkpoint"] for i in p["relevant_turn_ids"])
                )
                msgs = messages(turns[: p["checkpoint"]], p["question"])
                self.assertEqual(len(msgs), 2 * p["checkpoint"] + 2)
                self.assertEqual(msgs[-1]["content"], p["question"])
            correction = [p for p in probes if p["probe"] == "correction"]
            self.assertEqual(correction[0]["relevant_turn_ids"], ["T001"])
            self.assertEqual(correction[-1]["relevant_turn_ids"], ["T001", "T020"])

    def test_correction_and_unresolved_variant_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "experiment"
            prepare(root)
            turns = read(root / "conversation.json")
            self.assertIn("以本条为准", turns[19]["text"])
            self.assertIn("都还没核实", turns[39]["text"])
            self.assertNotEqual(turns[0]["values"]["reason"], turns[19]["values"]["reason"])
            self.assertEqual(turns[0]["values"]["person"], turns[19]["values"]["person"])
            self.assertEqual(turns[3]["values"]["person"], -1)
            self.assertEqual(sum(t["kind"] == "chatter" for t in turns), 15)


if __name__ == "__main__":
    unittest.main()
