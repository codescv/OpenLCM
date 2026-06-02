"""Persistent fact store — cross-session key-value memory.

Facts survive session boundaries. Scope controls visibility:
  - 'global'     : shared across every session in this database
  - <session_id> : private to that specific session

Facts can be tagged and linked to form a lightweight knowledge graph
without requiring a separate graph database.

Use lcm_remember / lcm_recall / lcm_forget / lcm_link from agent tools.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import configure_connection

_FACTS_SELECT = "fact_id, scope, key, value, category, tags, related_keys, source_session_id, created_at, updated_at"


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
                    tags              TEXT NOT NULL DEFAULT '[]',
                    related_keys      TEXT NOT NULL DEFAULT '[]',
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
            # Add tags/related_keys columns to existing tables that predate them
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(lcm_facts)").fetchall()}
            if "tags" not in cols:
                self._conn.execute("ALTER TABLE lcm_facts ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'")
            if "related_keys" not in cols:
                self._conn.execute("ALTER TABLE lcm_facts ADD COLUMN related_keys TEXT NOT NULL DEFAULT '[]'")
            self._conn.commit()

    # ── Write operations ───────────────────────────────────────────────────

    def remember(
        self,
        key: str,
        value: str,
        *,
        scope: str = "global",
        category: str = "fact",
        tags: list[str] | None = None,
        related_keys: list[str] | None = None,
        source_session_id: str = "",
    ) -> dict[str, Any]:
        """Store or update a fact. Returns a result dict including previous_value if updated."""
        now = time.time()
        key = key.strip()
        # None means "preserve existing value on conflict"; [] means "explicitly clear"
        tags_json = json.dumps(tags) if tags is not None else None
        related_json = json.dumps(related_keys) if related_keys is not None else None

        with self._lock:
            # Capture old value for contradiction detection
            old_row = self._conn.execute(
                "SELECT value FROM lcm_facts WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchone()
            old_value = old_row[0] if old_row else None

            # For INSERT: default to '[]' when tags not provided
            # For UPDATE on conflict: CASE preserves existing value when new value is NULL
            tags_insert = tags_json if tags_json is not None else '[]'
            related_insert = related_json if related_json is not None else '[]'
            self._conn.execute(
                """
                INSERT INTO lcm_facts
                    (scope, key, value, category, tags, related_keys, source_session_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, key) DO UPDATE SET
                    value             = excluded.value,
                    category          = excluded.category,
                    tags              = CASE WHEN ? IS NULL THEN tags ELSE ? END,
                    related_keys      = CASE WHEN ? IS NULL THEN related_keys ELSE ? END,
                    source_session_id = excluded.source_session_id,
                    updated_at        = excluded.updated_at
                """,
                (scope, key, value, category, tags_insert, related_insert,
                 source_session_id or None, now, now,
                 tags_json, tags_json, related_json, related_json),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT fact_id FROM lcm_facts WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchone()
            fact_id = int(row[0]) if row else 0

        result: dict[str, Any] = {
            "fact_id": fact_id,
            "key": key,
            "value": value,
            "scope": scope,
            "category": category,
            "status": "stored",
        }
        # Contradiction detection: surface old value when it differs substantially
        if old_value is not None and old_value != value and abs(len(old_value) - len(value)) > 10:
            result["updated"] = True
            result["previous_value"] = old_value
        return result

    def forget(self, key: str, *, scope: str = "global") -> bool:
        """Delete a fact by key + scope. Returns True if a row was deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM lcm_facts WHERE scope = ? AND key = ?",
                (scope, key.strip()),
            )
            self._conn.commit()
            return bool(cur.rowcount)

    def link(self, key1: str, key2: str, *, scope: str = "global") -> bool:
        """Bidirectionally add key2 to key1's related_keys and key1 to key2's."""
        updated = False
        for a, b in [(key1, key2), (key2, key1)]:
            row = self._conn.execute(
                "SELECT related_keys FROM lcm_facts WHERE scope = ? AND key = ?",
                (scope, a.strip()),
            ).fetchone()
            if row is None:
                continue
            try:
                existing: list = json.loads(row[0] or "[]")
            except (json.JSONDecodeError, TypeError):
                existing = []
            if b not in existing:
                existing.append(b)
                with self._lock:
                    self._conn.execute(
                        "UPDATE lcm_facts SET related_keys = ? WHERE scope = ? AND key = ?",
                        (json.dumps(existing), scope, a.strip()),
                    )
                    self._conn.commit()
                updated = True
        return updated

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
        tag: str | None = None,
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
        if tag is not None:
            # JSON contains the tag string
            where.append("tags LIKE ?")
            args.append(f'%"{tag}"%')
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

    def recall_related(self, key: str, *, scope: str = "global") -> List[Dict[str, Any]]:
        """Return facts connected to key via shared tags or explicit related_keys links."""
        source = self.recall_exact(key, scope=scope)
        if not source:
            return []

        related: list[Dict[str, Any]] = []
        seen_keys: set[str] = {key}

        # 1. Explicit related_keys links (already deserialized to list by _row_to_dict)
        raw_rkeys = source.get("related_keys")
        rkeys: list[str] = raw_rkeys if isinstance(raw_rkeys, list) else []
        for rk in rkeys:
            if rk in seen_keys:
                continue
            f = self.recall_exact(rk, scope=scope)
            if f:
                related.append(f)
                seen_keys.add(rk)

        # 2. Shared-tag facts (already deserialized)
        raw_tags = source.get("tags")
        tags: list[str] = raw_tags if isinstance(raw_tags, list) else []
        for t in tags:
            for f in self.recall_query(tag=t, scope=scope, limit=20):
                if f["key"] not in seen_keys:
                    related.append(f)
                    seen_keys.add(f["key"])

        return related

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
    cols = ["fact_id", "scope", "key", "value", "category", "tags", "related_keys",
            "source_session_id", "created_at", "updated_at"]
    d = dict(zip(cols, row[: len(cols)]))
    # Deserialize JSON arrays
    for field in ("tags", "related_keys"):
        try:
            d[field] = json.loads(d.get(field) or "[]")
        except (json.JSONDecodeError, TypeError):
            d[field] = []
    return d
