"""LangGraph adapter — LCMCheckpointer.

Implements LangGraph's BaseCheckpointSaver interface so LCM can act as
the persistence and context management layer for any LangGraph graph.

Install: pip install openlcm[langgraph]

If you already have a LangChain LLM defined, pass it directly — no need
to configure a separate model for LCM::

    from langchain_anthropic import ChatAnthropic
    from openlcm.adapters.langgraph import LCMCheckpointer

    llm = ChatAnthropic(model="claude-3-haiku-20240307")   # your existing LLM
    checkpointer = LCMCheckpointer(llm=llm)

    graph = StateGraph(MyState).compile(checkpointer=checkpointer)

Starting from scratch (no existing LLM)?::

    from openlcm import LCMEngine
    from openlcm.adapters.langgraph import LCMCheckpointer

    engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")
    checkpointer = LCMCheckpointer(engine)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

from .base import LCMAdapter, _resolve_engine

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

    def __init__(self, engine=None, *, llm=None, db_path: str = "") -> None:
        super().__init__(_resolve_engine(engine, llm=llm, db_path=db_path, platform="langgraph"))
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
        task_path: tuple = (),
    ) -> None:
        thread_id = self._thread_id(config)
        if not thread_id:
            return
        self._engine.bind_session(thread_id, platform="langgraph")
        for channel, value in writes:
            if channel == "messages":
                msgs = value if isinstance(value, list) else [value]
                self._engine._ingest_messages(msgs)

    async def aput_writes(self, config, writes, task_id, task_path: tuple = ()):
        self.put_writes(config, writes, task_id, task_path)

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
        """Extract and normalise messages from a LangGraph checkpoint dict.

        Uses LangChainMessages to properly serialize tool_calls / ToolMessage
        so they survive compression without data loss.
        """
        from .langchain import LangChainMessages

        channel_values = checkpoint.get("channel_values", {})
        raw = channel_values.get("messages", [])
        if not isinstance(raw, list):
            return []

        # Separate LangChain BaseMessage objects from plain dicts
        lc_objects = [m for m in raw if hasattr(m, "content") and not isinstance(m, dict)]
        plain_dicts = [m for m in raw if isinstance(m, dict)]

        if lc_objects:
            # Convert LangChain objects using the proper converter
            return LangChainMessages.to_lcm(raw)

        # Already plain dicts — normalize role names (LangChain uses "human"/"ai")
        _role_map = {"human": "user", "ai": "assistant", "chatbot": "assistant"}
        result = []
        for m in plain_dicts:
            role = _role_map.get(str(m.get("role", "user")).lower(), m.get("role", "user"))
            result.append({**m, "role": role})
        return result

    @staticmethod
    def _rows_to_checkpoint(rows: list, thread_id: str) -> Dict[str, Any]:
        """Convert stored LCM rows back to a minimal LangGraph checkpoint dict.

        Rows are kept as plain dicts in channel_values so LangGraph can store
        them.  When the graph resumes, _checkpoint_to_messages normalises them.
        """
        messages = [
            {"role": r.get("role", "user"), "content": r.get("content", "")}
            for r in rows
        ]
        return {
            "v": 1,
            "id": thread_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"messages": messages},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
