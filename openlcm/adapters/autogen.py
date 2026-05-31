"""AutoGen adapter — LCMContext + AutoGenMessages converter.

Two things live here:

1. **AutoGenMessages** — converts between AutoGen's typed LLMMessage objects
   and LCM's internal dict format, including tool calls (FunctionCall /
   FunctionExecutionResult).

2. **LCMContext** — a ``ChatCompletionContext`` subclass that plugs LCM
   transparently into any AutoGen agent.  The agent calls ``get_messages()``
   before every LLM turn; LCMContext runs compression automatically when
   context pressure is high.

Install: pip install openlcm[autogen]

Usage (recommended — pass your existing model client)::

    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from openlcm.adapters.autogen import LCMContext

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        model_context=LCMContext(llm=model_client),   # reuses same client
    )

Starting from scratch::

    from openlcm import LCMEngine
    from openlcm.adapters.autogen import LCMContext

    engine = LCMEngine(model="openai/gpt-4o-mini")
    agent  = AssistantAgent("assistant", model_client=...,
                            model_context=LCMContext(engine))

Multi-agent system::

    # Each agent gets its own session-scoped LCMContext
    agent_a = AssistantAgent("planner",  model_context=LCMContext(llm=client, session_id="planner"))
    agent_b = AssistantAgent("executor", model_context=LCMContext(llm=client, session_id="executor"))

Message format converter (stand-alone)::

    from openlcm.adapters.autogen import AutoGenMessages

    lcm_msgs  = AutoGenMessages.to_lcm(autogen_messages)
    ag_msgs   = AutoGenMessages.from_lcm(lcm_msgs)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import LCMAdapter, _resolve_engine

logger = logging.getLogger(__name__)


# ── Message converter ─────────────────────────────────────────────────────────

class AutoGenMessages:
    """Convert between AutoGen LLMMessage types and LCM internal format.

    Handles all four AutoGen message types:
    - ``UserMessage``             → ``{"role":"user",      "content":"..."}``
    - ``AssistantMessage`` (text) → ``{"role":"assistant", "content":"..."}``
    - ``AssistantMessage`` (calls)→ ``{"role":"assistant", "content":JSON_tool_calls}``
    - ``SystemMessage``           → ``{"role":"system",    "content":"..."}``
    - ``FunctionExecutionResult`` → ``{"role":"tool",      "content":"...", ...}``

    Also accepts plain dicts in OpenAI format (pass-through / fall-back).

    All methods are static::

        lcm  = AutoGenMessages.to_lcm(autogen_messages)
        back = AutoGenMessages.from_lcm(lcm_messages)
    """

    @staticmethod
    def to_lcm(messages: list) -> list[dict]:
        """Convert a list of AutoGen LLMMessage objects to LCM internal format.

        Also accepts plain dicts (e.g. from older AutoGen versions).
        """
        # Import lazily so the module loads without autogen installed
        try:
            from autogen_core.models import (
                UserMessage, AssistantMessage, SystemMessage,
                FunctionExecutionResult, FunctionCall,
            )
            _have_types = True
        except ImportError:
            _have_types = False

        result: list[dict] = []

        for m in messages:
            # ── Already a plain dict ──────────────────────────────────────────
            if isinstance(m, dict):
                result.append(_normalise_dict(m))
                continue

            if not _have_types:
                # No autogen installed — best-effort extraction
                result.append(_normalise_obj(m))
                continue

            # ── UserMessage ───────────────────────────────────────────────────
            if isinstance(m, UserMessage):
                result.append({"role": "user", "content": str(m.content)})

            # ── SystemMessage ─────────────────────────────────────────────────
            elif isinstance(m, SystemMessage):
                result.append({"role": "system", "content": str(m.content)})

            # ── AssistantMessage ──────────────────────────────────────────────
            elif isinstance(m, AssistantMessage):
                content = m.content
                if isinstance(content, list):
                    # content is list[FunctionCall]
                    tool_calls = []
                    for fc in content:
                        if isinstance(fc, FunctionCall):
                            try:
                                args = json.loads(fc.arguments) if isinstance(fc.arguments, str) else dict(fc.arguments)
                            except (ValueError, TypeError):
                                args = {"_raw": str(fc.arguments)}
                            tool_calls.append({
                                "id":   fc.id,
                                "name": fc.name,
                                "args": args,
                                "type": "function",
                            })
                        else:
                            # Unknown item in list — stringify
                            tool_calls.append({"id": "", "name": str(fc), "args": {}, "type": "function"})
                    content_str = json.dumps(
                        {"text": "", "tool_calls": tool_calls},
                        ensure_ascii=False,
                    )
                else:
                    content_str = str(content) if content is not None else ""
                result.append({"role": "assistant", "content": content_str})

            # ── FunctionExecutionResult ───────────────────────────────────────
            elif isinstance(m, FunctionExecutionResult):
                result.append({
                    "role":         "tool",
                    "content":      str(m.content),
                    "tool_call_id": m.call_id,
                    "name":         getattr(m, "name", "") or "",
                })

            # ── Unknown / duck-typed fallback ─────────────────────────────────
            else:
                result.append(_normalise_obj(m))

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> list:
        """Convert LCM internal format back to AutoGen LLMMessage typed objects.

        Falls back to plain dicts if autogen_core is not installed, so the
        adapter degrades gracefully in environments that use a non-typed fork.
        """
        try:
            from autogen_core.models import (
                UserMessage, AssistantMessage, SystemMessage,
                FunctionExecutionResult, FunctionCall,
            )
            _have_types = True
        except ImportError:
            _have_types = False

        result: list = []

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")

            if not _have_types:
                result.append(m)
                continue

            if role == "system":
                result.append(SystemMessage(content=content))

            elif role == "user":
                result.append(UserMessage(content=content, source="user"))

            elif role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        fc_list = []
                        for tc in parsed["tool_calls"]:
                            fc_list.append(FunctionCall(
                                id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=json.dumps(tc.get("args", {}), ensure_ascii=False),
                            ))
                        result.append(AssistantMessage(content=fc_list, source="assistant"))
                        continue
                except (ValueError, TypeError):
                    pass
                result.append(AssistantMessage(content=content, source="assistant"))

            elif role == "tool":
                result.append(FunctionExecutionResult(
                    call_id=m.get("tool_call_id", ""),
                    content=content,
                    is_error=False,
                    name=m.get("name", ""),
                ))

            else:
                result.append(UserMessage(content=content, source=role))

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_dict(m: dict) -> dict:
    """Normalise an already-dict message (handles OpenAI-style tool_calls key)."""
    role = (m.get("role") or "user").lower()
    if role == "assistant" and m.get("tool_calls"):
        from .openai import OpenAIMessages
        return OpenAIMessages.to_lcm([m])[0]
    if role == "tool":
        return {
            "role":         "tool",
            "content":      str(m.get("content", "")),
            "tool_call_id": m.get("tool_call_id", ""),
            "name":         m.get("name", ""),
        }
    return {"role": role, "content": str(m.get("content", ""))}


def _normalise_obj(m: Any) -> dict:
    """Best-effort extraction from an unknown AutoGen message object."""
    if hasattr(m, "content"):
        content = m.content
        role    = str(getattr(m, "source", getattr(m, "role", "user")))
        # AutoGen uses "source" not "role" on message objects
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        role = role_map.get(role.lower(), "user")
        if isinstance(content, list):
            # Probably list[FunctionCall] — stringify gracefully
            return {"role": "assistant", "content": json.dumps(
                {"text": "", "tool_calls": [{"id": getattr(f, "id", ""), "name": getattr(f, "name", str(f)), "args": {}, "type": "function"} for f in content]},
                ensure_ascii=False,
            )}
        return {"role": role, "content": str(content) if content is not None else ""}
    return {"role": "user", "content": str(m)}


# ── LCMContext ────────────────────────────────────────────────────────────────

class LCMContext(LCMAdapter):
    """AutoGen ``ChatCompletionContext`` backed by LCMEngine.

    Satisfies the full ``ChatCompletionContext`` ABC:
    - ``add_message(message)``  → persist to store, compress if needed
    - ``get_messages()``        → return LCM-assembled context as typed LLMMessages
    - ``clear()``               → reset session
    - ``message_count()``       → number of messages currently held
    - ``save_state()``          → serialisable state dict
    - ``load_state(state)``     → restore from saved state

    The model always sees the LCM-optimised context — no changes to agent code.
    """

    def __init__(
        self,
        engine=None,
        session_id: str = "autogen",
        *,
        llm=None,
        db_path: str = "",
    ) -> None:
        super().__init__(_resolve_engine(engine, llm=llm, db_path=db_path, platform="autogen"))
        if not self._engine._session_id:
            self._engine.bind_session(session_id, platform="autogen")
        self._messages: List[Dict[str, Any]] = []

    # ── ChatCompletionContext interface ───────────────────────────────────────

    async def add_message(self, message: Any) -> None:
        """Persist a new message and trigger compaction if context pressure is high."""
        msg_dicts = AutoGenMessages.to_lcm([message])
        if not msg_dicts:
            return

        # Persist to store BEFORE updating in-memory list (safe on exception)
        self._engine._ingest_messages(msg_dicts)
        self._messages.extend(msg_dicts)

        if self._engine.should_compress_preflight(self._messages):
            self._messages = await self._engine.compress(self._messages)

    async def get_messages(self) -> list:
        """Return the LCM-assembled context as typed AutoGen LLMMessage objects."""
        if self._engine.should_compress_preflight(self._messages):
            self._messages = await self._engine.compress(self._messages)
        return AutoGenMessages.from_lcm(self._messages)

    async def clear(self) -> None:
        """Clear messages and reset the session store."""
        self._messages = []
        session_id = self._engine._session_id
        if session_id:
            try:
                self._engine._store.delete_session_messages(session_id)
                self._engine._dag.delete_session_nodes(session_id)
            except Exception as exc:
                logger.warning("LCMContext.clear failed: %s", exc)

    async def message_count(self) -> int:
        """Return the current number of messages held by the context."""
        return len(self._messages)

    async def save_state(self) -> Dict[str, Any]:
        """Return serialisable state for checkpointing."""
        return {"messages": list(self._messages), "session_id": self._engine._session_id or ""}

    async def load_state(self, state: Dict[str, Any]) -> None:
        """Restore context from a previously saved state dict."""
        self._messages = list(state.get("messages", []))
        sid = state.get("session_id", "")
        if sid and sid != self._engine._session_id:
            self._engine.bind_session(sid, platform="autogen")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def update_from_usage(self, usage: Dict[str, Any]) -> None:
        """Forward token usage metadata to the engine for pressure tracking."""
        self._engine.update_from_response(usage)
