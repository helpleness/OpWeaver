"""Safety-relevant boundaries: no keyword-only acceptance and split leakage."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from .generate_data import FIELDS, TASKS
from .prepare_semantic_revision import FINAL, FINAL_INVALID, FRESH, FRESH_INVALID
from .semantic_data import ANSWERS, INVALID, QUESTIONS, prepare
from .semantic_memory import SemanticMemory, parse_answer


class SemanticTests(unittest.TestCase):
    def test_reader_uses_predicted_field_instruction_without_gold_fact(self):
        model = object.__new__(SemanticMemory)
        model.device = torch.device("cpu")
        model.canonical_reader = True
        model.generate = Mock(return_value={"raw": "example"})
        model.generate_routed("任意原始问法，不应直接进入读取器", None, 3)
        self.assertEqual(model.generate.call_args.args[0], QUESTIONS["train"][3][0])
        self.assertIsNone(model.route_tasks)

    def policy_fixture(self, values):
        # Deterministic network substitutes exercise policy boundaries, not learned accuracy.
        model = object.__new__(SemanticMemory)
        model.device = torch.device("cpu")
        model.contextual_query = False
        query_logits = torch.zeros(1, 7)
        query_logits[0, 2] = 20

        def head(z):
            logits = torch.full((len(z), 9), -20.0)
            logits.scatter_(1, z[:, :1].long(), 20.0)
            return logits

        model.semantic = SimpleNamespace(
            eval=lambda: None, query=lambda h, m: query_logits, values=[head] * 6
        )
        model.features = Mock(return_value=[])
        model.padded_features = Mock(
            return_value=(torch.ones(1, 1, 4), torch.ones(1, 1, dtype=torch.bool))
        )
        model.recall = Mock(return_value=dict(evidence=1.0, accepted=True, id="0"))
        model.adapter = SimpleNamespace(
            query_logits=lambda h, m: torch.ones(1, 32),
            key=lambda x: x,
            agreement=lambda q, k: torch.ones(len(k)),
        )
        model.generate_routed = Mock(
            return_value=dict(raw="那次的事情是忘带资料。", confidence=0.2, ended=True)
        )
        zs = np.ones((len(values), 8, 256), dtype=np.float32)
        zs[:, 2, 0] = values
        store = SimpleNamespace(
            _index=lambda owner: SimpleNamespace(
                keys={str(i): np.ones(32, dtype=np.float32) for i in range(len(values))}
            ),
            fetch=lambda owner, ids: (zs, None),
        )
        return model, store

    def test_conflicting_latents_block_generation(self):
        model, store = self.policy_fixture([1, 2])
        result = model.ask_checked(store, "owner", "query")
        self.assertEqual(result["reason"], "conflicting_memories")
        self.assertEqual(result["decision_source"], "controller")
        self.assertFalse(result["model_generated"])
        model.generate_routed.assert_not_called()

    def test_missing_target_blocks_generation(self):
        model, store = self.policy_fixture([8])
        result = model.ask_checked(store, "owner", "query")
        self.assertEqual(result["reason"], "missing_field")
        model.generate_routed.assert_not_called()

    def test_duplicates_allow_only_matching_generated_fact(self):
        model, store = self.policy_fixture([1, 1])
        result = model.ask_checked(store, "owner", "query")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["action"], "answer")
        self.assertTrue(result["model_generated"])
        # Low style-token probability alone does not reject a verified factual answer.
        model.generate_routed.return_value["raw"] = "那次的事情是提出建议。"
        result = model.ask_checked(store, "owner", "query")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "unsupported_generation")
        self.assertEqual(result["action"], "abstain")
        self.assertTrue(result["model_generated"])

    def test_negation_and_unsupported_extra_claims_are_rejected(self):
        self.assertIsNone(parse_answer("不是提出建议。", 2))
        self.assertIsNone(parse_answer("提出建议，但后来收到礼物。", 2))
        self.assertIsNone(parse_answer("小杨提出建议。", 2))
        self.assertIsNone(parse_answer("因为生气。", 4))

    def test_all_declared_paraphrases_preserve_fact_identity(self):
        for field, task in enumerate(TASKS):
            for value, word in enumerate(FIELDS[task]):
                for style in ANSWERS[field]:
                    self.assertEqual(parse_answer(style.format(word), field), value)

    def test_question_families_are_disjoint(self):
        groups = []
        for split in ("train", "calibration", "test"):
            group = {q for field in QUESTIONS[split] for q in field}
            group.update(q for family in INVALID[split].values() for q in family)
            groups.append(group)
        groups.append(
            {q for field in FRESH for q in field}
            | {q for field in FRESH_INVALID.values() for q in field}
        )
        groups.append(
            {q for field in FINAL for q in field}
            | {q for field in FINAL_INVALID.values() for q in field}
        )
        for i in range(len(groups)):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])

    def test_test_conditions_have_valid_oracle_contracts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "suite"
            prepare(p)
            cases = json.loads((p / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cases), 240)
            self.assertEqual(len({c["id"] for c in cases}), 240)
            for c in cases:
                targets = [e for e in c["events"] if e["id"].endswith("-target")]
                if c["kind"] == "missing_field":
                    self.assertEqual(targets[0]["values"][TASKS[c["field"]]], -1)
                if c["kind"] == "wrong_owner":
                    self.assertNotEqual(targets[0]["owner"], c["owner"])
                if c["kind"] == "empty":
                    self.assertEqual(c["events"], [])
                if c["kind"] in ("duplicate", "conflict"):
                    copies = [e for e in c["events"] if e["id"].endswith("-copy")]
                    equal = (
                        targets[0]["values"][TASKS[c["field"]]]
                        == copies[0]["values"][TASKS[c["field"]]]
                    )
                    self.assertEqual(equal, c["kind"] == "duplicate")


if __name__ == "__main__":
    unittest.main()
