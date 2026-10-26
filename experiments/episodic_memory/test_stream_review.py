"""Evidence packaging must preserve causality and include execution evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from .luna_review import read_rows
from .stream_review import prepare


class StreamReviewTests(unittest.TestCase):
    def test_relevant_excerpt_excludes_future_and_invisible_records_and_keeps_gate_log(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source"
            source.mkdir()
            (source / "conversation.json").write_text(
                json.dumps(
                    [
                        dict(id="T001", number=1, text="早先事件"),
                        dict(id="T003", number=3, text="未来更正"),
                    ]
                ),
                encoding="utf-8",
            )
            (source / "probes.json").write_text(
                json.dumps([dict(id="Q", probe="early", relevant_turn_ids=["T001", "T003"])]),
                encoding="utf-8",
            )
            rows = [
                dict(
                    id="Q",
                    checkpoint=1,
                    mode="disk_memory",
                    question="问题",
                    visible_turn_ids=["T001"],
                    result=dict(
                        status="memory_result",
                        answer=None,
                        ended=None,
                        status_explanation="候选表述未通过接受检查。",
                        decision=dict(
                            reason="unsupported_generation", model_generated=True, eligible=False
                        ),
                    ),
                ),
                dict(
                    id="Q",
                    checkpoint=1,
                    mode="base_recent_window",
                    question="问题",
                    visible_turn_ids=[],
                    result=dict(status="generated", answer="无法确定", ended=True),
                ),
            ]
            (source / "outputs.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
            out = Path(d) / "review"
            prepare(source, out)
            packets = read_rows(out / "all_packets.jsonl")
            for p in packets:
                self.assertNotIn("未来更正", json.dumps(p, ensure_ascii=False))
                if p["answer"] == "无法确定":
                    self.assertEqual(p["evidence"], [])
                else:
                    self.assertIn("reason=unsupported_generation", p["execution_state"])
                    self.assertIn("候选表述未通过接受检查", p["execution_state"])


if __name__ == "__main__":
    unittest.main()
