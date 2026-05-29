"""AutoGen adapter — LCMContext.

Subclasses AutoGen's ChatCompletionContext so LCM provides transparent
context management for AutoGen agents and multi-agent systems.

Install: pip install openlcm[autogen]

Usage::

    from openlcm.core.engine import LCMEngine
    from openlcm.backends.openai import OpenAIBackend
    from openlcm.adapters.autogen import LCMContext
    from autogen_agentchat.agents import AssistantAgent

    engine = LCMEngine(backend=OpenAIBackend(model="gpt-4o-mini"))
    engine.bind_session("autogen-session", context_length=128000)

    agent = AssistantAgent(
        "assistant",
        model_client=...,
        model_context=LCMContext(engine),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import LCMAdapter

logger = logging.getLogger(__name__)


class LCMContext(LCMAdapter):
    """AutoGen model context backed by LCMEngine.

    Integrates with AutoGen's ChatCompletionContext interface:
    - add_message()  → persist to MessageStore, compress if needed
    - get_messages() → return LCM-assembled (compressed) context
    - clear()        → reset session

    AutoGen reads `get_messages()` before each LLM call, so the model
    always sees the LCM-optimized context automatically.
    """

    def __init__(self, engine, session_id: str = "autogen") -> None:
        super().__init__(engine)
        if not engine._session_id:
            engine.bind_session(session_id, platform="autogen")
        self._messages: List[Dict[str, Any]] = []

    async def add_message(self, message: Any) -> None:
        """Persist a new message and trigger compaction if needed."""
        msg = self._normalize_message(message)
        self._messages.append(msg)
        self._engine._ingest_messages([msg])

        if self._engine.should_compress_preflight(self._messages):
            self._messages = await self._engine.compress(self._messages)

    async def get_messages(self) -> List[Dict[str, Any]]:
        """Return the LCM-assembled context for the next LLM call."""
        if self._engine.should_compress_preflight(self._messages):
            self._messages = await self._engine.compress(self._messages)
        return list(self._messages)

    async def clear(self) -> None:
        """Clear messages and reset the session."""
        self._messages = []
        session_id = self._engine._session_id
        if session_id:
            try:
                self._engine._store.delete_session_messages(session_id)
                self._engine._dag.delete_session_nodes(session_id)
            except Exception as exc:
                logger.warning("LCMContext.clear failed: %s", exc)

    def update_from_usage(self, usage: Dict[str, Any]) -> None:
        """Forward token usage to the engine for pressure tracking."""
        self._engine.update_from_response(usage)

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        """Convert AutoGen message types to the standard dict format."""
        if isinstance(message, dict):
            return message
        if hasattr(message, "to_dict"):
            return message.to_dict()
        if hasattr(message, "dict"):
            d = message.dict()
            return {"role": d.get("source", d.get("type", "user")), "content": d.get("content", str(message))}
        if hasattr(message, "content"):
            role = getattr(message, "source", getattr(message, "role", "user"))
            return {"role": str(role), "content": str(message.content)}
        return {"role": "user", "content": str(message)}
