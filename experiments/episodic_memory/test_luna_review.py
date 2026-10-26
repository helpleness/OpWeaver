"""Review ingestion integrity; these tests do not prove a judge understands language."""

import copy
import unittest

from .luna_review import digest, validate


class JudgeIngestionTests(unittest.TestCase):
    def fixtures(self):
        packet = dict(
            review_id="L000",
            question="和谁交谈？",
            answer="小林。",
            evidence=[dict(id="E1", text="交谈对象是小林。")],
            execution_state="本次响应已结束。",
            generation_ended=True,
        )
        packet["binding_sha256"] = digest(packet)
        review = dict(
            review_id="L000",
            binding_sha256=packet["binding_sha256"],
            availability="answerable",
            behavior="answer",
            target="correct",
            whole_reply="acceptable",
            event_claim_problem=False,
            status_problem=False,
            confidence="high",
            rationale="所问人物与证据一致。",
            claims=[
                dict(
                    quote="小林",
                    verdict="supported",
                    reason="人物相同",
                    evidence=[dict(id="E1", quote="交谈对象是小林。")],
                )
            ],
        )
        return packet, review

    def test_changed_output_cannot_reuse_old_judgement(self):
        packet, review = self.fixtures()
        packet["answer"] = "小陈。"
        with self.assertRaises(ValueError):
            validate([packet], [review])

    def test_cannot_cite_evidence_not_supplied_to_judge(self):
        packet, review = self.fixtures()
        review["claims"][0]["evidence"][0]["quote"] = "交谈对象是小陈。"
        with self.assertRaises(ValueError):
            validate([packet], [review])

    def test_missing_or_duplicate_review_is_not_silently_accepted(self):
        packet, review = self.fixtures()
        with self.assertRaises(ValueError):
            validate([packet], [])
        with self.assertRaises(ValueError):
            validate([packet], [review, copy.deepcopy(review)])

    def test_valid_quotes_do_not_establish_the_semantic_verdict(self):
        packet, review = self.fixtures()
        # Ingestion validates attribution, not accuracy; a bad judge can still be wrong.
        review["target"] = "uncertain"
        review["confidence"] = "low"
        validate([packet], [review])


if __name__ == "__main__":
    unittest.main()
