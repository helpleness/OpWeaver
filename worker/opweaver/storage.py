"""Small durable records in SQLite; tensors stay in filesystem artifacts."""

import json
import sqlite3
from pathlib import Path


class RunStore:
    def __init__(self, project: Path):
        self.path = project / "experiments.sqlite3"
        project.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, updated TEXT DEFAULT CURRENT_TIMESTAMP, record TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def save(self, record: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO runs(id, record) VALUES (?, ?)",
                (record["id"], json.dumps(record)),
            )

    def set_metadata(self, key: str, value: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def history(self) -> list[dict]:
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT record FROM runs ORDER BY updated DESC, rowid DESC LIMIT 100"
                )
            ]

    def metadata_list(self, prefix: str) -> list[dict]:
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT value FROM metadata WHERE substr(key, 1, ?) = ? ORDER BY rowid DESC",
                    (len(prefix), prefix),
                )
            ]

    def recover(self, live_ids: set[str]) -> list[dict]:
        records = self.history()
        for record in records:
            if record["id"] not in live_ids and record["state"] not in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            ):
                record.update(
                    state="FAILED",
                    error="Worker exited before this task completed. Resume the last durable checkpoint.",
                )
                self.save(record)
        return records
