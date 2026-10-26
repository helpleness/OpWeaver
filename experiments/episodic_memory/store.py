"""Transactional disk payloads. Only address vectors and grouping are CPU-cached."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from .cluster import OnlineClusters


def writer_version(model, tokenizer_spec):
    digest = hashlib.sha256(json.dumps(tokenizer_spec, sort_keys=True).encode())
    for name, tensor in model.state_dict().items():
        if name.startswith(("event_encoder.", "slot_queries", "pool.", "slot_norm.", "event_key.")):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MemoryStore:
    def __init__(
        self,
        path: Path,
        version: str,
        shape: tuple[int, int],
        key_dim: int,
        create=False,
        threshold=0.65,
        max_members=64,
    ):
        self.path = Path(path)
        self.cache = {}
        if create and self.path.exists():
            raise FileExistsError(f"Refusing to overwrite memory: {path}")
        if not create and not self.path.exists():
            raise FileNotFoundError(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("PRAGMA foreign_keys=ON")
        if create:
            with self.db:
                self.db.executescript("""
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE events(event_id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                        timestamp INTEGER NOT NULL, memory BLOB NOT NULL, address BLOB NOT NULL);
                    CREATE INDEX events_owner ON events(owner);
                    CREATE TABLE groups(owner TEXT, cid INTEGER, centroid BLOB NOT NULL,
                        PRIMARY KEY(owner,cid));
                    CREATE TABLE memberships(owner TEXT, cid INTEGER, event_id TEXT,
                        PRIMARY KEY(owner,cid,event_id),
                        FOREIGN KEY(owner,cid) REFERENCES groups(owner,cid) ON DELETE CASCADE,
                        FOREIGN KEY(event_id) REFERENCES events(event_id));
                """)
                meta = {
                    "schema": 1,
                    "version": version,
                    "shape": list(shape),
                    "key_dim": key_dim,
                    "threshold": threshold,
                    "max_members": max_members,
                }
                self.db.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [(k, json.dumps(v)) for k, v in meta.items()],
                )
        self.meta = {k: json.loads(v) for k, v in self.db.execute("SELECT key,value FROM metadata")}
        if (
            self.meta["version"] != version
            or self.meta["shape"] != list(shape)
            or self.meta["key_dim"] != key_dim
        ):
            self.db.close()
            raise ValueError(
                "Writer/version/shape mismatch; rebuild with the matching event encoder"
            )
        self.shape = tuple(shape)
        self.key_dim = key_dim

    def enable_revisions(self):
        """Add an auditable field-level revision log without modifying latent history."""
        with self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS write_operations(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
                    decision TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS field_revisions(
                    old_id TEXT NOT NULL REFERENCES events(event_id),
                    new_id TEXT NOT NULL REFERENCES events(event_id),
                    field INTEGER NOT NULL CHECK(field BETWEEN 0 AND 5),
                    PRIMARY KEY(old_id,new_id,field));
            """)

    def active_ids(self, owner, event_ids, field):
        """Only explicit, owner-scoped edges supersede a field; time alone never does."""
        exists = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='field_revisions'"
        ).fetchone()
        if not exists:
            return list(event_ids)
        superseded = {
            row[0]
            for row in self.db.execute(
                "SELECT r.old_id FROM field_revisions r "
                "JOIN events old ON old.event_id=r.old_id "
                "JOIN events new ON new.event_id=r.new_id "
                "WHERE old.owner=? AND new.owner=? AND r.field=?",
                (owner, owner, field),
            )
        }
        # A single-field correction does not restate/erase the other five fields.
        out_of_scope = set()
        for eid, decision in self.db.execute(
            "SELECT o.event_id,o.decision FROM write_operations o "
            "JOIN events e ON e.event_id=o.event_id WHERE e.owner=?",
            (owner,),
        ):
            op = json.loads(decision)
            if op.get("applied") and op.get("field") != field:
                out_of_scope.add(eid)
        return [eid for eid in event_ids if eid not in superseded and eid not in out_of_scope]

    def _index(self, owner):
        if owner not in self.cache:
            c = OnlineClusters(self.meta["threshold"], self.meta["max_members"])
            for eid, key in self.db.execute(
                "SELECT event_id,address FROM events WHERE owner=?", (owner,)
            ):
                c.keys[eid] = np.frombuffer(key, dtype="<f4").copy()
            for cid, center in self.db.execute(
                "SELECT cid,centroid FROM groups WHERE owner=?", (owner,)
            ):
                c.groups[cid] = set()
                c.centers[cid] = np.frombuffer(center, dtype="<f4").copy()
                c.next_id = max(c.next_id, cid + 1)
            for cid, eid in self.db.execute(
                "SELECT cid,event_id FROM memberships WHERE owner=?", (owner,)
            ):
                c.groups[cid].add(eid)
            self.cache[owner] = c
        return self.cache[owner]

    def encode_memory(self, z):
        if z.shape != self.shape:
            raise ValueError("Bad payload shape")
        return z.tobytes()

    def decode_memory(self, raw):
        return np.frombuffer(raw, dtype="<f2").reshape(self.shape).astype(np.float32)

    def collate_memories(self, rows):
        return np.stack(rows)

    def add_metadata(self, record):
        """Subclass hook, executed in the same transaction as the payload."""

    def add_many(self, records):
        owners = {r["owner"] for r in records}
        try:
            with self.db:
                for r in records:
                    z = np.asarray(r["memory"], dtype="<f2")
                    key = np.asarray(r["key"], dtype="<f4")
                    if key.shape != (self.key_dim,):
                        raise ValueError("Bad payload shape")
                    if not np.isfinite(z).all() or not np.isfinite(key).all():
                        raise ValueError("Nonfinite memory")
                    index = self._index(r["owner"])
                    index.add(r["id"], key)
                    self.db.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?)",
                        (r["id"], r["owner"], r["timestamp"], self.encode_memory(z), key.tobytes()),
                    )
                    self.add_metadata(r)
                    if "operation" in r:
                        self.db.execute(
                            "INSERT INTO write_operations VALUES (?,?)",
                            (r["id"], json.dumps(r["operation"], ensure_ascii=False)),
                        )
                        field = r["operation"].get("field")
                        for old_id in r.get("supersedes", []):
                            if (
                                old_id == r["id"]
                                or self.db.execute(
                                    "SELECT 1 FROM events WHERE event_id=? AND owner=?",
                                    (old_id, r["owner"]),
                                ).fetchone()
                                is None
                            ):
                                raise ValueError(
                                    "Revision must reference a previous same-owner event"
                                )
                            self.db.execute(
                                "INSERT INTO field_revisions VALUES (?,?,?)",
                                (old_id, r["id"], field),
                            )
                for owner in owners:
                    index = self._index(owner)
                    index.merge_small()
                    self.db.execute("DELETE FROM groups WHERE owner=?", (owner,))
                    self.db.executemany(
                        "INSERT INTO groups VALUES (?,?,?)",
                        [
                            (owner, cid, np.asarray(c, dtype="<f4").tobytes())
                            for cid, c in index.centers.items()
                        ],
                    )
                    self.db.executemany(
                        "INSERT INTO memberships VALUES (?,?,?)",
                        [
                            (owner, cid, eid)
                            for cid, members in index.groups.items()
                            for eid in sorted(members)
                        ],
                    )
        except Exception:
            for owner in owners:
                self.cache.pop(owner, None)
            raise

    def retrieve(self, owner, query_key, top_k=8, top_clusters=4, flat=False, exclude=()):
        return self._index(owner).select(query_key, top_k, top_clusters, flat, exclude)

    def fetch(self, owner, event_ids):
        rows = []
        for eid in event_ids:
            row = self.db.execute(
                "SELECT memory,address FROM events WHERE event_id=? AND owner=?", (eid, owner)
            ).fetchone()
            if row is None:
                raise KeyError(f"No event in this owner's memory: {eid}")
            rows.append(
                (
                    self.decode_memory(row[0]),
                    np.frombuffer(row[1], dtype="<f4").copy(),
                )
            )
        if not rows:
            return np.zeros((1, *self.shape), np.float32), np.zeros((1, self.key_dim), np.float32)
        return self.collate_memories([r[0] for r in rows]), np.stack([r[1] for r in rows])

    def counts(self):
        return {
            "events": self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "groups": self.db.execute("SELECT COUNT(*) FROM groups").fetchone()[0],
            "file_bytes": self.path.stat().st_size,
        }

    def close(self):
        self.db.close()
