"""Persistent fact store — cross-session key-value memory.

Facts survive session boundaries. Scope controls visibility:
  - 'global'     : shared across every session in this database
  - <session_id> : private to that specific session

Use lcm_remember / lcm_recall / lcm_forget to manage facts from agent tools.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import configure_connection

_FACTS_SELECT = "fact_id, scope, key, value, category, source_session_id, created_at, updated_at"


class FactStore:
    """SQLite-backed persistent fact store sharing the main LCM database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        configure_connection(self._conn)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS lcm_facts (
                    fact_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope             TEXT NOT NULL DEFAULT 'global',
                    key               TEXT NOT NULL,
                    value             TEXT NOT NULL,
                    category          TEXT NOT NULL DEFAULT 'fact',
                    source_session_id TEXT,
                    created_at        REAL NOT NULL,
                    updated_at        REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_scope_key
                    ON lcm_facts(scope, key);
                CREATE INDEX IF NOT EXISTS idx_facts_category
                    ON lcm_facts(category, scope);
                CREATE INDEX IF NOT EXISTS idx_facts_updated
                    ON lcm_facts(updated_at DESC);
            """)
            self._conn.commit()

    # ── Write operations ───────────────────────────────────────────────────

    def remember(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        category: str = "fact",
        source_session_id: str = "",
    ) -> int:
        """Store or update a fact. Returns fact_id."""
        now = time.time()
        key = key.strip()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO lcm_facts (scope, key, value, category, source_session_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, key) DO UPDATE SET
                    value             = excluded.value,
                    category          = excluded.category,
                    source_session_id = excluded.source_session_id,
                    updated_at        = excluded.updated_at
                """,
                (scope, key, value, category, source_session_id or None, now, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT fact_id FROM lcm_facts WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchone()
            return int(row[0]) if row else 0

    def forget(self, key: str, *, scope: str = "global") -> bool:
        """Delete a fact by key + scope. Returns True if a row was deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM lcm_facts WHERE scope = ? AND key = ?",
                (scope, key.strip()),
            )
            self._conn.commit()
            return bool(cur.rowcount)

    # ── Read operations ────────────────────────────────────────────────────

    def recall_exact(self, key: str, *, scope: str = "global") -> Optional[Dict[str, Any]]:
        """Retrieve one fact by exact (scope, key)."""
        row = self._conn.execute(
            f"SELECT {_FACTS_SELECT} FROM lcm_facts WHERE scope = ? AND key = ?",
            (scope, key.strip()),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def recall_query(
        self,
        query: str = "",
        *,
        scope: str | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search facts by LIKE match on key+value, optionally filtered."""
        where: list[str] = []
        args: list[Any] = []

        if scope is not None:
            where.append("scope = ?")
            args.append(scope)
        if category is not None:
            where.append("category = ?")
            args.append(category)
        if query:
            pat = f"%{query}%"
            where.append("(key LIKE ? OR value LIKE ?)")
            args.extend([pat, pat])

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        args.append(max(1, limit))

        rows = self._conn.execute(
            f"SELECT {_FACTS_SELECT} FROM lcm_facts {clause} ORDER BY updated_at DESC LIMIT ?",
            args,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn:
            conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _row_to_dict(row) -> Dict[str, Any]:
    cols = ["fact_id", "scope", "key", "value", "category", "source_session_id", "created_at", "updated_at"]
    return dict(zip(cols, row[: len(cols)]))
