"""Optional semantic embedding store for LCM.

Stores vector embeddings of DAG summary nodes and facts in the same SQLite
database, enabling cosine-similarity search as a complement to FTS5 keyword
search.

Requires:
  - sqlite-vec extension: pip install sqlite-vec
  - An embedding model: LCM_EMBEDDING_MODEL=openai/text-embedding-3-small

If either is unavailable the store is a silent no-op and lcm_semantic_search
falls back gracefully to FTS5 via lcm_grep.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AVAILABLE: bool | None = None  # lazily determined


def _sqlite_vec_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import sqlite_vec  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


def _serialize_f32(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _deserialize_f32(b: bytes) -> list[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))


class EmbeddingStore:
    """Vector store backed by sqlite-vec in the main LCM SQLite database.

    All methods are no-ops when sqlite-vec is not installed or no embedding
    model is configured — callers never need to guard for that case.
    """

    def __init__(self, db_path: str | Path, *, embedding_model: str = "") -> None:
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._dim: int = 0
        self._enabled = bool(embedding_model) and _sqlite_vec_available()
        if self._enabled:
            self._init_db()

    def _init_db(self) -> None:
        try:
            import sqlite_vec
            self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            sqlite_vec.load(self._conn)
            # lcm_embeddings table is created by db_bootstrap migration
            logger.debug("EmbeddingStore ready with model=%s", self.embedding_model)
        except Exception as exc:
            logger.debug("EmbeddingStore disabled: %s", exc)
            self._enabled = False
            self._conn = None

    # ── Embedding generation ───────────────────────────────────────────────

    async def _get_embedding(self, text: str) -> list[float] | None:
        if not self._enabled or not text:
            return None
        try:
            import litellm
            resp = await litellm.aembedding(model=self.embedding_model, input=[text[:8000]])
            vec = resp.data[0]["embedding"]
            if self._dim == 0:
                self._dim = len(vec)
            return vec
        except Exception as exc:
            logger.debug("Embedding call failed: %s", exc)
            return None

    # ── Write ──────────────────────────────────────────────────────────────

    async def embed(self, content_type: str, content_id: int, text: str) -> bool:
        """Compute and store embedding for a DAG node or fact. Returns True on success."""
        if not self._enabled or not self._conn:
            return False
        vec = await self._get_embedding(text)
        if vec is None:
            return False
        blob = _serialize_f32(vec)
        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO lcm_embeddings (content_type, content_id, model, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(content_type, content_id, model) DO UPDATE SET
                        embedding  = excluded.embedding,
                        created_at = excluded.created_at
                    """,
                    (content_type, content_id, self.embedding_model, blob, now),
                )
                self._conn.commit()
            return True
        except Exception as exc:
            logger.debug("Embedding store write failed: %s", exc)
            return False

    def delete(self, content_type: str, content_id: int) -> None:
        if not self._enabled or not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM lcm_embeddings WHERE content_type = ? AND content_id = ?",
                (content_type, content_id),
            )
            self._conn.commit()

    # ── Search ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query_text: str,
        *,
        content_type: str | None = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar items by cosine similarity. Empty list if unavailable."""
        if not self._enabled or not self._conn or not query_text:
            return []
        query_vec = await self._get_embedding(query_text)
        if query_vec is None:
            return []
        query_blob = _serialize_f32(query_vec)
        dim = len(query_vec)

        try:
            where = f"AND content_type = ?" if content_type else ""
            args: list = [query_blob, dim, limit]
            if content_type:
                args = [query_blob, dim, content_type, limit]
            # sqlite-vec cosine distance: vec_distance_cosine(a, b) lower = more similar
            rows = self._conn.execute(
                f"""
                SELECT e.content_type, e.content_id,
                       vec_distance_cosine(e.embedding, vec_f32(?)) AS distance,
                       e.created_at
                FROM lcm_embeddings e, vec_f32(?, ?) v
                WHERE 1=1 {where}
                ORDER BY distance ASC
                LIMIT ?
                """,
                args,
            ).fetchall()
            return [
                {
                    "content_type": r[0],
                    "content_id": r[1],
                    "score": round(1.0 - float(r[2]), 4),  # similarity (higher = better)
                    "distance": round(float(r[2]), 4),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("Embedding search failed: %s", exc)
            return []

    @property
    def enabled(self) -> bool:
        return self._enabled

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
