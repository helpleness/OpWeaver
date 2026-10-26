"""The renderer must not invent evidence from a rejected candidate or score itself."""

import unittest
from unittest.mock import Mock

from .qwen_memory import UNKNOWN
from .semantic_memory import FACT_READER_SYSTEM, SemanticMemory, status_prompt


class FactResponseTests(unittest.TestCase):
    def model(self, generated=False, eligible=False):
        model = object.__new__(SemanticMemory)
        model.ask_checked = Mock(
            return_value=dict(
                answer="对方是小林。" if eligible else UNKNOWN,
                raw="对方是小林。" if generated else UNKNOWN,
                model_generated=generated,
                eligible=eligible,
                reason="supported" if eligible else "unsupported_generation",
                action="answer" if eligible else "abstain",
                support=0.95,
            )
        )
        model.chat = Mock(return_value=dict(answer="这次读取还无法确认，请再核实。", ended=True))
        return model

    def test_renderer_does_not_receive_unverified_candidate_or_refusal_target(self):
        model = self.model(generated=True)
        result = model.respond(None, "owner", "那次和谁交谈？", natural_status=True)
        self.assertEqual(result["reader_raw"], "对方是小林。")
        prompt = model.chat.call_args.args[0]
        self.assertNotIn("小林", prompt)
        self.assertNotIn(UNKNOWN, prompt)
        self.assertEqual(result["answer"], "这次读取还无法确认，请再核实。")
        self.assertEqual(result["semantic_correctness"], "not_automatically_judged")

    def test_no_generation_is_not_logged_as_a_generated_refusal(self):
        model = self.model()
        result = model.respond(None, "owner", "问题")
        self.assertIsNone(result["reader_raw"])
        self.assertFalse(result["decision"]["model_generated"])
        self.assertIsNone(result["answer"])
        model.chat.assert_not_called()
        self.assertNotIn(UNKNOWN, result["status_explanation"])

    def test_accepted_generation_still_requires_external_fact_evaluation(self):
        model = self.model(generated=True, eligible=True)
        result = model.respond(None, "owner", "问题")
        model.chat.assert_not_called()
        self.assertEqual(result["semantic_correctness"], "not_automatically_judged")
        self.assertEqual(model.ask_checked.call_args.args[-1], 0.8)

    def test_missing_field_does_not_mean_missing_event(self):
        prompt = status_prompt("问题", "missing_field")
        self.assertIn("其他内容可能仍然存在", prompt)
        self.assertNotIn(UNKNOWN, FACT_READER_SYSTEM)


if __name__ == "__main__":
    unittest.main()
