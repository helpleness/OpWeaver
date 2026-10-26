"""Storage/policy invariants, not a scorer for free-form answers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .store import MemoryStore
from .test_semantic_memory import SemanticTests


class RevisionTests(unittest.TestCase):
    def test_revision_survives_restart_is_field_scoped_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(path, "v", (2, 2), 4, create=True)
            store.enable_revisions()

            def record(eid, owner="me", **kw):
                return dict(
                    id=eid, owner=owner, timestamp=0, memory=np.ones((2, 2)), key=np.ones(4), **kw
                )

            store.add_many([record("old"), record("other", "other")])
            store.add_many([record("new", operation=dict(field=4, applied=True), supersedes=["old"])])
            store.add_many([record("unverified")])
            store.close()
            store = MemoryStore(path, "v", (2, 2), 4)
            self.assertEqual(
                store.active_ids("me", ["old", "new", "unverified"], 4), ["new", "unverified"]
            )
            self.assertEqual(store.active_ids("me", ["old", "new"], 3), ["old"])
            self.assertEqual(store.counts()["events"], 4)
            with self.assertRaises(ValueError):
                store.add_many([record("bad", operation=dict(field=4), supersedes=["other"])])
            self.assertEqual(store.counts()["events"], 4)  # Transaction rolled back.
            with self.assertRaises(ValueError):
                store.add_many([record("self", operation=dict(field=4), supersedes=["self"])])
            store.add_many(
                [record("latest", operation=dict(field=4), supersedes=["new", "unverified"])]
            )
            self.assertEqual(
                store.active_ids("me", ["old", "new", "unverified", "latest"], 4), ["latest"]
            )
            store.close()

    def test_typed_fact_never_copies_wrong_negated_or_extra_generated_claims(self):
        model, store = SemanticTests().policy_fixture([1, 1])
        model.answer_policy = "typed_fact"
        store.active_ids = lambda owner, ids, field: ids
        for draft in [
            "不是忘带资料。",
            "小杨忘带资料，后来收到礼物。",
            "不认识的句式也不会决定放行",
            "",
        ]:
            model.generate_routed.return_value = dict(raw=draft, confidence=0.01, ended=False)
            result = model.ask_checked(store, "owner", "question")
            self.assertEqual(result["answer"], "事件：忘带资料。")
            self.assertEqual(result["raw"], draft)
            self.assertFalse(result["draft_used_as_answer"])
            self.assertEqual(result["fact"]["evidence_ids"], ["0", "1"])

    def test_typed_fact_keeps_conflict_and_missing_checks(self):
        for values, reason in [([1, 2], "conflicting_memories"), ([8], "missing_field")]:
            model, store = SemanticTests().policy_fixture(values)
            model.answer_policy = "typed_fact"
            store.active_ids = lambda owner, ids, field: ids
            result = model.ask_checked(store, "owner", "question")
            self.assertFalse(result["eligible"])
            self.assertEqual(result["reason"], reason)


if __name__ == "__main__":
    unittest.main()
