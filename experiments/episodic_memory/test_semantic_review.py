"""Evaluation boundaries: evidence availability, label validity and audit integrity."""

import copy
import random
import unittest

from .generate_data import TASKS
from .reader_objective import reader_batch, reader_queries
from .semantic_review import evidence, packet, reference, summarize, validate_review


def example():
    event = dict(
        id="c-target",
        owner="u",
        anchor="花园项目第1次讨论",
        text="花园项目第1次讨论。与我交谈的人是小林。我的内心感受为生气。",
        values={t: 0 for t in TASKS},
    )
    event["values"]["emotion"] = 5
    return dict(
        id="c",
        owner="u",
        kind="normal",
        field=0,
        damage_id=None,
        question="花园项目第1次讨论，和谁交流？",
        events=[event],
        expected="小林",
    )


def review(p, **changes):
    r = dict(
        review_id="R0",
        binding_sha256=p["binding_sha256"],
        reviewer="test_fixture_not_real_review",
        selection_reason="test",
        rationale="test",
        all_claims_checked=True,
        behavior="abstain",
        target="not_answered",
        instruction_compliance="not_assessed",
        format_compliance="not_assessed",
        claims=[],
    )
    r.update(changes)
    return r


class ReviewTests(unittest.TestCase):
    def test_no_context_has_no_private_evidence_or_correct_fact_denominator(self):
        c = example()
        self.assertEqual(evidence(c, "task_no_context"), [])
        self.assertEqual(reference(c, [])["availability"], "no_evidence")
        p = packet(c, dict(result=dict(answer="小林")), "task_no_context")
        with self.assertRaises(ValueError):
            validate_review(p, review(p, behavior="answer", target="correct"))

    def test_owner_isolation_and_damaged_records(self):
        c = example()
        c["events"][0]["owner"] = "other"
        self.assertEqual(evidence(c, "task_full_text"), [])
        c["events"][0]["owner"] = "u"
        c["damage_id"] = "c-target"
        self.assertEqual(evidence(c, "memory_final"), [])

    def test_multifield_and_yes_no_are_answerable_not_automatic_negatives(self):
        c = example()
        c.update(
            kind="ambiguous",
            expected=None,
            question="花园项目第1次讨论，把交流对象与注意力所在都列给我。",
        )
        self.assertEqual(reference(c, c["events"])["availability"], "answerable")
        c.update(
            kind="false_premise", question="花园项目第1次讨论，请回答是或否：那次我是不是安心？"
        )
        ref = reference(c, c["events"])
        self.assertEqual(ref["fields"], ["emotion"])  # Ignore misleading old field=person.
        self.assertFalse(ref["reference_boolean"])

    def test_conflicts_and_duplicates_do_not_have_same_reference(self):
        c = example()
        c["events"].append(copy.deepcopy(c["events"][0]))
        self.assertEqual(reference(c, c["events"])["availability"], "answerable")
        c["events"][1]["values"]["person"] = 1
        self.assertEqual(reference(c, c["events"])["availability"], "conflicting")

    def test_unannotated_new_question_does_not_silently_receive_label(self):
        c = example()
        c.update(kind="ambiguous", question="新的多字段问题")
        with self.assertRaises(ValueError):
            reference(c, c["events"])

    def test_unreviewed_is_not_wrong_and_has_no_accuracy(self):
        p = packet(example(), dict(result=dict(answer="不知道")), "task_no_context")
        p["review_id"] = "R0"
        s = summarize([p], {"R0": dict(condition="bare")}, [])["bare"]
        self.assertEqual(s["unreviewed"], 1)
        self.assertEqual(s["reviewed_target"], {})
        self.assertIsNone(s["semantic_accuracy"])

    def test_review_is_bound_to_output_and_evidence(self):
        p = packet(example(), dict(result=dict(answer="不知道")), "task_no_context")
        r = review(p)
        p["response"] = "是小林"
        with self.assertRaises(ValueError):
            validate_review(p, r)

    def test_claim_cannot_cite_evidence_hidden_from_condition(self):
        p = packet(
            example(), dict(result=dict(answer="我不知道，但肯定是小林。")), "task_no_context"
        )
        r = review(
            p,
            behavior="mixed",
            claims=[
                dict(
                    quote="肯定是小林",
                    verdict="supported",
                    evidence=[dict(id="c-target", quote="与我交谈的人是小林。")],
                )
            ],
        )
        with self.assertRaises(ValueError):
            validate_review(p, r)

    def test_refusal_wording_has_no_automatic_semantic_verdict(self):
        for text in ("我不知道。", "现有记录没有这个信息。", "交流对象姓名：未知"):
            p = packet(example(), dict(result=dict(answer=text)), "task_no_context")
            validate_review(
                p, review(p)
            )  # Explicit judgement required for each; not a regex scorer.

    def test_default_reader_never_supervises_missing_facts_as_fixed_phrase(self):
        events = [copy.deepcopy(example()["events"][0]) for _ in range(4)]
        for e in events[2:]:
            e["values"]["person"] = -1
        data = dict(events=events, queries=[dict(event=i, field=0, text="人物？") for i in (0, 2)])
        qs = reader_queries(data)
        _, ids, present, answers = reader_batch(data, qs, 100, random.Random(1))
        self.assertTrue(all(i < 2 for i in ids))
        self.assertTrue(all(present))
        self.assertTrue(all("小林" in a for a in answers))
        _, _, _, old = reader_batch(data, reader_queries(data, True), 100, random.Random(1), True)
        self.assertIn("我没有这段经历的记忆。", old)


if __name__ == "__main__":
    unittest.main()
