"""Online, overlapping index groups. Group edits NEVER merge event payloads."""

from __future__ import annotations

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


class OnlineClusters:
    def __init__(self, threshold=0.65, max_members=64, secondary_margin=0.05):
        if max_members < 2:
            raise ValueError("max_members must be >= 2")
        self.threshold = threshold
        self.max_members = max_members
        self.secondary_margin = secondary_margin
        self.keys: dict[str, np.ndarray] = {}
        self.groups: dict[int, set[str]] = {}
        self.centers: dict[int, np.ndarray] = {}
        self.next_id = 0
        self.splits = 0
        self.merges = 0

    def _refresh(self, cid):
        self.centers[cid] = unit(np.stack([self.keys[e] for e in sorted(self.groups[cid])]).sum(0))

    def _new(self, members):
        cid = self.next_id
        self.next_id += 1
        self.groups[cid] = set(members)
        self._refresh(cid)
        return cid

    def add(self, event_id, key):
        if event_id in self.keys:
            raise ValueError("Event IDs are immutable; use a new ID for a revision")
        self.keys[event_id] = unit(key)
        ranked = sorted(
            ((float(self.keys[event_id] @ c), cid) for cid, c in self.centers.items()), reverse=True
        )
        chosen = []
        if ranked and ranked[0][0] >= self.threshold:
            chosen.append(ranked[0][1])
            if (
                len(ranked) > 1
                and ranked[1][0] >= self.threshold
                and ranked[0][0] - ranked[1][0] <= self.secondary_margin
            ):
                chosen.append(ranked[1][1])
        if not chosen:
            self._new([event_id])
        for cid in chosen:
            self.groups[cid].add(event_id)
            self._refresh(cid)
            if len(self.groups[cid]) > self.max_members:
                self._split(cid)

    def _split(self, cid):
        ids = sorted(self.groups[cid])
        x = np.stack([self.keys[e] for e in ids])
        first = int(np.argmin(x @ self.centers[cid]))
        second = int(np.argmin(x @ x[first]))
        centers = np.stack([x[first], x[second]])
        assignment = None
        for _ in range(10):
            assignment = np.argmax(x @ centers.T, axis=1)
            counts = np.bincount(assignment, minlength=2)
            if min(counts) == 0:
                break
            centers = np.stack([unit(x[assignment == i].sum(0)) for i in range(2)])
        counts = np.bincount(assignment, minlength=2)
        if min(counts) == 0 or max(counts) > self.max_members:
            # Deterministic balanced fallback handles duplicate/degenerate keys.
            order = np.argsort(x @ (centers[0] - centers[1]), kind="stable")
            assignment = np.zeros(len(ids), dtype=int)
            assignment[order[len(ids) // 2 :]] = 1
        del self.groups[cid]
        del self.centers[cid]
        for i in range(2):
            self._new([eid for eid, label in zip(ids, assignment) if label == i])
        self.splits += 1

    def merge_small(self, minimum=16, similarity=0.90):
        # Called at a flush boundary; bounded groups and member de-duplication.
        for a in list(self.groups):
            if a not in self.groups or len(self.groups[a]) >= minimum:
                continue
            options = [
                (float(self.centers[a] @ self.centers[b]), b)
                for b in self.groups
                if b != a and len(self.groups[a] | self.groups[b]) <= self.max_members
            ]
            if options:
                score, b = max(options)
                if score >= similarity:
                    self.groups[b] |= self.groups[a]
                    del self.groups[a]
                    del self.centers[a]
                    self._refresh(b)
                    self.merges += 1

    def candidates(self, query, top_clusters=4):
        ranked = sorted(
            self.centers, key=lambda cid: float(query @ self.centers[cid]), reverse=True
        )
        return (
            sorted(set().union(*(self.groups[c] for c in ranked[:top_clusters]))) if ranked else []
        )

    def select(self, query, top_k=8, top_clusters=4, flat=False):
        candidates = sorted(self.keys) if flat else self.candidates(query, top_clusters)
        ranked = sorted(candidates, key=lambda e: (-float(query @ self.keys[e]), e))
        return ranked[:top_k], len(candidates)
