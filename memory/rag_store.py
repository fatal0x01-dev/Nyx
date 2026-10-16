"""SQLite-backed retrieval store for durable system observations."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator
from uuid import uuid4

import config.settings as cfg


class RAGStore:
    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or cfg.RAG_DB_PATH
        parent = os.path.dirname(os.path.abspath(self._db_path))
        os.makedirs(parent, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add(self, text: str, metadata: dict | None = None) -> str:
        document_id = f"observation:{uuid4().hex}"
        self._upsert(document_id, text, metadata or {})
        return document_id

    def _upsert(self, document_id: str, text: str, metadata: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (id, content, metadata, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    text,
                    json.dumps(metadata, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_system_context(self, environment_key: str, context: dict[str, str]) -> None:
        document_id = f"system-context:{environment_key}"
        content = "\n".join(
            f"{key}: {value}"
            for key, value in context.items()
            if value
        )
        self._upsert(
            document_id,
            content,
            {
                "kind": "system_context",
                "environment_key": environment_key,
                **context,
            },
        )

    def get_system_context(self, environment_key: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT metadata FROM documents WHERE id = ?",
                (f"system-context:{environment_key}",),
            ).fetchone()
        if row is None:
            return None
        metadata = json.loads(row["metadata"])
        return {
            str(key): str(value)
            for key, value in metadata.items()
            if key not in {"kind", "environment_key"} and value is not None
        }

    def query(self, query: str, top_k: int = 5) -> list[str]:
        if top_k <= 0:
            return []
        terms = set(re.findall(r"[a-z0-9./-]+", query.lower().replace("_", " ")))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT content FROM documents ORDER BY updated_at DESC"
            ).fetchall()
        scored = []
        for row in rows:
            content = str(row["content"])
            content_terms = set(
                re.findall(r"[a-z0-9./-]+", content.lower().replace("_", " "))
            )
            score = len(terms & content_terms)
            if score:
                scored.append((score, content))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [content for _, content in scored[:top_k]]
