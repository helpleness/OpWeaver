"""Control construction must not leak truth, private records or damaged content."""

import copy
import unittest

from .compare_bare_qwen import context_text, messages


class BareQwenControlTests(unittest.TestCase):
    def fixture(self):
        return dict(
            id="case",
            owner="me",
            question="同一个问题",
            damage_id="damaged",
            expected="SECRET_ANSWER",
            field=0,
            events=[
                dict(id="case-target", owner="me", text="OWN_RECORD", values={"person": 99}),
                dict(id="other", owner="other-person", text="PRIVATE_SECRET"),
                dict(id="damaged", owner="me", text="DAMAGED_SECRET", anchor="已知事件地址"),
            ],
        )

    def test_full_context_preserves_owner_and_availability_boundaries(self):
        text = context_text(self.fixture())
        self.assertIn("OWN_RECORD", text)
        self.assertIn("已知事件地址", text)
        self.assertNotIn("PRIVATE_SECRET", text)
        self.assertNotIn("DAMAGED_SECRET", text)
        self.assertNotIn("SECRET_ANSWER", text)

    def test_no_context_modes_have_no_records(self):
        for mode in ("chat_no_context", "task_no_context"):
            rendered = str(messages(self.fixture(), mode))
            self.assertNotIn("OWN_RECORD", rendered)
            self.assertNotIn("PRIVATE_SECRET", rendered)
            self.assertNotIn("SECRET_ANSWER", rendered)
            self.assertIn("同一个问题", rendered)

    def test_truth_labels_cannot_change_prompt(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b["expected"] = "OTHER_TRUTH"
        b["field"] = 5
        b["events"][0]["values"] = {"person": 0}
        for mode in ("chat_no_context", "task_no_context", "task_full_text", "task_oracle_text"):
            self.assertEqual(messages(a, mode), messages(b, mode))
        self.assertEqual(context_text(a, oracle=True), "OWN_RECORD")


if __name__ == "__main__":
    unittest.main()
