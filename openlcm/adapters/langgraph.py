"""LangGraph adapter — LCMCheckpointer.

Implements LangGraph's BaseCheckpointSaver interface so LCM can act as
the persistence and context management layer for any LangGraph graph.

Install: pip install openlcm[langgraph]

Usage::

    from openlcm.core.engine import LCMEngine
    from openlcm.backends.anthropic import AnthropicBackend
    from openlcm.adapters.langgraph import LCMCheckpointer
    from langgraph.graph import StateGraph

    engine = LCMEngine(backend=AnthropicBackend())
    checkpointer = LCMCheckpointer(engine)

    graph = StateGraph(MyState).compile(checkpointer=checkpointer)

    # Run with a thread_id — LCM persists and compacts automatically
    config = {"configurable": {"thread_id": "user-123"}}
    result = await graph.ainvoke({"messages": [...]}, config=config)
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

from .base import LCMAdapter

logger = logging.getLogger(__name__)


class LCMCheckpointer(LCMAdapter):
    """LangGraph checkpointer backed by LCMEngine.

    Stores graph state in LCM's SQLite store and triggers compaction
    automatically when context pressure exceeds the configured threshold.

    LangGraph calls:
        - put() after each node execution to checkpoint state
        - get_tuple() to load the latest checkpoint before resuming
        - list() to enumerate past checkpoints (for time-travel)

    LCM maps:
        - thread_id → session_id
        - checkpoint messages → MessageStore
        - context pressure → compress() on next put()
    """

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self._serde = None  # Use LangGraph's default JSON serde

    # ── Required: get_tuple ───────────────────────────────────────────────

    def get_tuple(self, config: Dict[str, Any]):
        """Return the latest CheckpointTuple for the given config."""
        try:
            from langgraph.checkpoint.base import CheckpointTuple
        except ImportError:
            raise ImportError("pip install openlcm[langgraph]")

        thread_id = self._thread_id(config)
        if not thread_id:
            return None

        self._engine.bind_session(thread_id, platform="langgraph")
        rows = self._engine._store.get_session_messages(thread_id, limit=1000)
        if not rows:
            return None

        checkpoint = self._rows_to_checkpoint(rows, thread_id)
        metadata = {"thread_id": thread_id, "step": len(rows)}
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None,
        )

    async def aget_tuple(self, config: Dict[str, Any]):
        return self.get_tuple(config)

    # ── Required: put ─────────────────────────────────────────────────────

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist checkpoint state and trigger LCM compaction if needed."""
        thread_id = self._thread_id(config)
        if not thread_id:
            return config

        self._engine.bind_session(thread_id, platform="langgraph")
        messages = self._checkpoint_to_messages(checkpoint)
        if messages:
            self._engine._ingest_messages(messages)

        if self._engine.should_compress_preflight(messages):
            import asyncio
            try:
                asyncio.run(self._engine.compress(messages))
            except RuntimeError:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._engine.compress(messages))

        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": thread_id}}

    async def aput(self, config, checkpoint, metadata, new_versions):
        thread_id = self._thread_id(config)
        if not thread_id:
            return config

        self._engine.bind_session(thread_id, platform="langgraph")
        messages = self._checkpoint_to_messages(checkpoint)
        if messages:
            self._engine._ingest_messages(messages)

        if self._engine.should_compress_preflight(messages):
            await self._engine.compress(messages)

        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": thread_id}}

    # ── Required: put_writes ──────────────────────────────────────────────

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = self._thread_id(config)
        if not thread_id:
            return
        self._engine.bind_session(thread_id, platform="langgraph")
        for channel, value in writes:
            if channel == "messages":
                msgs = value if isinstance(value, list) else [value]
                self._engine._ingest_messages(msgs)

    async def aput_writes(self, config, writes, task_id):
        self.put_writes(config, writes, task_id)

    # ── Required: list ────────────────────────────────────────────────────

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator:
        try:
            from langgraph.checkpoint.base import CheckpointTuple
        except ImportError:
            return

        thread_id = self._thread_id(config or {})
        if not thread_id:
            return

        rows = self._engine._store.get_session_messages(thread_id, limit=limit or 1000)
        if not rows:
            return

        checkpoint = self._rows_to_checkpoint(rows, thread_id)
        yield CheckpointTuple(
            config=config or {},
            checkpoint=checkpoint,
            metadata={"thread_id": thread_id, "step": len(rows)},
            parent_config=None,
        )

    async def alist(self, config, *, filter=None, before=None, limit=None) -> AsyncIterator:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _thread_id(config: Dict[str, Any]) -> str:
        return str(config.get("configurable", {}).get("thread_id", "") or "")

    @staticmethod
    def _checkpoint_to_messages(checkpoint: Dict[str, Any]) -> list:
        """Extract messages from a LangGraph checkpoint dict."""
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        if isinstance(messages, list):
            result = []
            for m in messages:
                if hasattr(m, "dict"):
                    d = m.dict()
                    result.append({
                        "role": d.get("type", "user"),
                        "content": d.get("content", ""),
                    })
                elif isinstance(m, dict):
                    result.append(m)
            return result
        return []

    @staticmethod
    def _rows_to_checkpoint(rows: list, thread_id: str) -> Dict[str, Any]:
        """Convert stored rows back to a minimal checkpoint dict."""
        messages = [
            {"role": r.get("role", "user"), "content": r.get("content", "")}
            for r in rows
        ]
        return {
            "v": 1,
            "id": thread_id,
            "channel_values": {"messages": messages},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
