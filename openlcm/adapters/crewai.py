"""CrewAI adapter — LCMStorage.

Implements CrewAI's storage backend interface so LCM can serve as the
persistent memory layer for CrewAI agents and crews.

Install: pip install openlcm[crewai]

Usage::

    from openlcm.core.engine import LCMEngine
    from openlcm.backends.anthropic import AnthropicBackend
    from openlcm.adapters.crewai import LCMStorage
    from crewai import Crew, Agent, Task
    from crewai.memory import LongTermMemory

    engine = LCMEngine(backend=AnthropicBackend(), db_path="~/.openlcm/crewai.db")
    engine.bind_session("my-crew")

    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=True,
        long_term_memory=LongTermMemory(storage=LCMStorage(engine)),
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .base import LCMAdapter

logger = logging.getLogger(__name__)


class LCMStorage(LCMAdapter):
    """CrewAI storage backend backed by LCMEngine.

    Satisfies the StorageBackend protocol used by CrewAI's memory system:
    - save()   → persists memory entries via MessageStore
    - search() → retrieves via lcm_grep FTS5 search
    - reset()  → clears session messages

    The memory entries are stored as assistant messages so they participate
    in LCM's compaction and summarization pipeline naturally.
    """

    def __init__(self, engine, session_id: str = "crewai") -> None:
        super().__init__(engine)
        if not engine._session_id:
            engine.bind_session(session_id, platform="crewai")

    def save(
        self,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: str = "",
        action: str = "",
    ) -> None:
        """Persist a memory entry.

        Args:
            value: The text content to remember.
            metadata: Optional additional metadata dict.
            agent: Agent name that produced this memory.
            action: Action or task description.
        """
        content_parts = [value]
        if agent:
            content_parts.append(f"[Agent: {agent}]")
        if action:
            content_parts.append(f"[Action: {action}]")
        if metadata:
            content_parts.append(f"[Metadata: {json.dumps(metadata, ensure_ascii=False)}]")

        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(content_parts),
        }
        session_id = self._engine._session_id or "crewai"
        try:
            from openlcm.core.tokens import count_message_tokens
            self._engine._store.append(
                session_id, msg, token_estimate=count_message_tokens(msg)
            )
        except Exception as exc:
            logger.warning("LCMStorage.save failed: %s", exc)

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search stored memories using FTS5.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            score_threshold: Minimum relevance score (0.0 = no filter).

        Returns:
            List of dicts with keys: content, metadata, score.
        """
        session_id = self._engine._session_id
        if not session_id:
            return []
        try:
            hits = self._engine._store.search(
                query,
                session_id=session_id,
                limit=limit,
                sort="relevance",
            )
            results = []
            for hit in hits:
                content = hit.get("content", "")
                score = abs(float(hit.get("search_rank") or 0))
                if score_threshold > 0 and score < score_threshold:
                    continue
                results.append({
                    "content": content,
                    "metadata": {"store_id": hit.get("store_id"), "timestamp": hit.get("timestamp")},
                    "score": score,
                })
            return results
        except Exception as exc:
            logger.warning("LCMStorage.search failed: %s", exc)
            return []

    def reset(self) -> None:
        """Clear all stored memories for the current session."""
        session_id = self._engine._session_id
        if session_id:
            try:
                self._engine._store.delete_session_messages(session_id)
                self._engine._dag.delete_session_nodes(session_id)
            except Exception as exc:
                logger.warning("LCMStorage.reset failed: %s", exc)
