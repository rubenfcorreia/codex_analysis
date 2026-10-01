from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from analysis.shared.state_utils import ensure_dir


class NoteStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                note_text TEXT NOT NULL,
                context_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notes_scope_key ON notes(scope, scope_key);
            """
        )
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add_note(self, *, scope: str, scope_key: str, note_text: str, context: Mapping[str, Any]) -> None:
        note = str(note_text).strip()
        if not note:
            return
        now = self._now()
        self.connection.execute(
            """
            INSERT INTO notes (scope, scope_key, note_text, context_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (scope, scope_key, note, json.dumps(dict(context), sort_keys=True), now, now),
        )
        self.connection.commit()

    def list_notes(self, *, scope: str, scope_key: str) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT id, scope, scope_key, note_text, context_json, created_at, updated_at
            FROM notes
            WHERE scope = ? AND scope_key = ?
            ORDER BY created_at DESC, id DESC
            """,
            (scope, scope_key),
        ).fetchall()
        notes: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["context"] = json.loads(payload.pop("context_json") or "{}")
            except json.JSONDecodeError:
                payload["context"] = {}
            notes.append(payload)
        return notes
