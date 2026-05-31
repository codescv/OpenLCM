"""Google ADK adapter — LCMSessionService + lcm_compress_callback.

Two things live here:

1. **LCMSessionService** — drop-in replacement for ADK's ``InMemorySessionService``.
   Delegates all ADK type-compatibility work to an inner ``InMemorySessionService``
   while adding LCM persistence and dashboard visibility for every event.

2. **lcm_compress_callback** — a ready-made ADK ``before_model_callback`` that
   compresses the conversation context before it reaches the Gemini API.
   Attach it to any ``LlmAgent`` to get automatic context management.

Install: pip install openlcm[google-adk]

Recommended usage — use both together::

    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from openlcm import LCMEngine
    from openlcm.adapters.google_adk import LCMSessionService, lcm_compress_callback

    engine = LCMEngine(model="gemini/gemini-2.0-flash")
    engine.bind_session("my-session", context_length=1_000_000)

    agent = LlmAgent(
        name="my_agent",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant.",
        tools=[get_weather, web_search],
        before_model_callback=lcm_compress_callback(engine),  # compression
    )

    runner = Runner(
        agent=agent,
        app_name="demo",
        session_service=LCMSessionService(engine),            # persistence + dashboard
    )

    session = await runner.session_service.create_session(app_name="demo", user_id="u1")

    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Hello!")]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

Starting from scratch (no existing model config)::

    from openlcm import LCMEngine
    engine = LCMEngine(model="gemini/gemini-2.0-flash")

Already have a Gemini model? Pass it directly::

    import google.generativeai as genai
    model  = genai.GenerativeModel("gemini-2.0-flash")
    engine = LCMEngine(summarize_fn=model)

Architecture
------------
- ``LCMSessionService`` wraps ``InMemorySessionService`` so ADK gets the exact
  ``Session`` / ``Event`` typed objects it expects.  Every ``append_event`` call
  also persists the event to the LCM SQLite store so it appears in the dashboard.

- ``lcm_compress_callback`` intercepts ``LlmRequest.contents`` (the list of
  ``types.Content`` messages about to be sent to Gemini), runs LCM compression
  via ``GeminiMessages``, and replaces ``contents`` with the compressed version
  before the API call is made.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .base import _resolve_engine

logger = logging.getLogger(__name__)


def _get_base_session_service():
    try:
        from google.adk.sessions.base_session_service import BaseSessionService
        return BaseSessionService
    except ImportError:
        return object


# ── Helpers: event → LCM messages ────────────────────────────────────────────

def _event_to_messages(event: Any) -> list[dict]:
    """Convert one ADK Event to zero or more LCM internal message dicts.

    Handles text, function_call, and function_response parts.
    """
    # Extract author and parts
    author = getattr(event, "author", None) or (
        event.get("author", event.get("role", "user")) if isinstance(event, dict) else "user"
    )
    role_map = {"model": "assistant", "user": "user", "system": "system"}
    role = role_map.get(str(author).lower(), "assistant")

    content = (
        event.get("content") if isinstance(event, dict)
        else getattr(event, "content", None)
    )
    if content is None:
        return []

    parts = (
        content.get("parts", []) if isinstance(content, dict)
        else list(getattr(content, "parts", []) or [])
    )
    if not parts:
        raw = content if isinstance(content, str) else ""
        return [{"role": role, "content": raw}] if raw else []

    text_parts:   list[str]  = []
    tool_calls:   list[dict] = []
    tool_results: list[dict] = []

    for part in parts:
        # Detect part type
        if isinstance(part, dict):
            ptype = next((k for k in ("function_call", "function_response", "text") if k in part and part[k] is not None), "unknown")
        else:
            if getattr(part, "function_call", None) is not None:
                ptype = "function_call"
            elif getattr(part, "function_response", None) is not None:
                ptype = "function_response"
            else:
                ptype = "text" if getattr(part, "text", None) is not None else "unknown"

        if ptype == "text":
            t = part.get("text", "") if isinstance(part, dict) else (getattr(part, "text", "") or "")
            if t:
                text_parts.append(t)

        elif ptype == "function_call":
            fc = part.get("function_call") if isinstance(part, dict) else part.function_call
            if fc is not None:
                if isinstance(fc, dict):
                    tc_id, tc_name, tc_args = fc.get("id", fc.get("name", "")), fc.get("name", ""), dict(fc.get("args", {}))
                else:
                    tc_id   = getattr(fc, "id",   getattr(fc, "name", "")) or ""
                    tc_name = getattr(fc, "name", "") or ""
                    tc_args = dict(getattr(fc, "args", {}) or {})
                tool_calls.append({"id": tc_id, "name": tc_name, "args": tc_args, "type": "function"})

        elif ptype == "function_response":
            fr = part.get("function_response") if isinstance(part, dict) else part.function_response
            if fr is not None:
                if isinstance(fr, dict):
                    fr_id, fr_name, fr_resp = fr.get("id", fr.get("name", "")), fr.get("name", ""), dict(fr.get("response", {}))
                else:
                    fr_id   = getattr(fr, "id",       getattr(fr, "name", "")) or ""
                    fr_name = getattr(fr, "name",     "") or ""
                    fr_resp = dict(getattr(fr, "response", {}) or {})
                tool_results.append({
                    "role":         "tool",
                    "content":      json.dumps(fr_resp, ensure_ascii=False),
                    "tool_call_id": fr_id,
                    "name":         fr_name,
                })

    messages: list[dict] = []
    messages.extend(tool_results)

    text = "\n".join(text_parts)
    if tool_calls:
        messages.append({
            "role":    "assistant",
            "content": json.dumps({"text": text, "tool_calls": tool_calls}, ensure_ascii=False),
        })
    elif text:
        messages.append({"role": role, "content": text})

    return messages


# ── LCMSessionService ─────────────────────────────────────────────────────────

class LCMSessionService(_get_base_session_service()):
    """Google ADK session service that adds LCM persistence to the default
    ``InMemorySessionService``.

    Drop-in replacement::

        # Before:
        session_service = InMemorySessionService()
        # After:
        session_service = LCMSessionService(engine)   # or LCMSessionService(llm=my_model)

    Every event appended through the Runner is persisted to the LCM SQLite
    store automatically.  The live dashboard at http://localhost:7842 shows
    all tool calls, token pressure, and DAG summaries in real time.

    For automatic context compression, also attach ``lcm_compress_callback``
    to your ``LlmAgent`` — see module docstring.
    """

    def __init__(self, engine=None, *, llm=None, db_path: str = "") -> None:
        # Use BaseSessionService.__init__ (accepts *args/**kwargs), not LCMAdapter
        super().__init__()
        self._engine = _resolve_engine(engine, llm=llm, db_path=db_path, platform="google_adk")
        # Inner service handles all ADK type compatibility (Session / Event objects)
        try:
            from google.adk.sessions import InMemorySessionService
            self._inner = InMemorySessionService()
        except ImportError:
            raise ImportError("google-adk is required. Install with: pip install google-adk")

    # ── Public interface ──────────────────────────────────────────────────────

    async def create_session(
        self,
        *,
        app_name:   str = "",
        user_id:    str = "",
        state:      Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        """Create a session.  Returns a proper ADK ``Session`` object."""
        session = await self._inner.create_session(
            app_name=app_name,
            user_id=user_id,
            state=state,
            session_id=session_id,
        )
        self._engine.bind_session(session.id, platform="google-adk")
        return session

    async def get_session(
        self,
        *,
        app_name:   str = "",
        user_id:    str = "",
        session_id: str,
        config:     Any = None,
    ):
        """Return the ADK Session object from the inner service."""
        return await self._inner.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            config=config,
        )

    async def append_event(self, session: Any, event: Any) -> Any:
        """Persist event via the inner service AND write to the LCM store.

        The inner service updates the in-memory ADK session; LCM gets a
        persisted copy for dashboard visibility and future compression.
        """
        # Let ADK's inner service handle the event (updates session.events, etc.)
        result = await self._inner.append_event(session, event)

        # Mirror to LCM store
        session_id = getattr(session, "id", None) or (session.get("id") if isinstance(session, dict) else "")
        if session_id:
            if session_id != self._engine._session_id:
                self._engine.bind_session(session_id, platform="google-adk")
            msgs = _event_to_messages(event)
            if msgs:
                try:
                    self._engine._ingest_messages(msgs)
                except Exception as exc:
                    logger.warning("LCMSessionService.append_event ingest failed: %s", exc)

        return result

    async def delete_session(
        self,
        *,
        app_name:   str = "",
        user_id:    str = "",
        session_id: str,
    ) -> None:
        """Delete from both the inner service and the LCM store."""
        try:
            await self._inner.delete_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("LCMSessionService.delete_session (inner) failed: %s", exc)
        try:
            self._engine._store.delete_session_messages(session_id)
            self._engine._dag.delete_session_nodes(session_id)
        except Exception as exc:
            logger.warning("LCMSessionService.delete_session (lcm) failed: %s", exc)

    async def list_sessions(self, *, app_name: str = "", user_id: str = "") -> Any:
        """List sessions — combines inner service + LCM store."""
        return await self._inner.list_sessions(app_name=app_name, user_id=user_id)

    # ── Sync variants (ADK also calls these in some contexts) ─────────────────

    def create_session_sync(self, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.create_session(**kwargs))

    def get_session_sync(self, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.get_session(**kwargs))

    def list_sessions_sync(self, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.list_sessions(**kwargs))

    def delete_session_sync(self, **kwargs):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.delete_session(**kwargs))


# ── lcm_compress_callback ─────────────────────────────────────────────────────

def lcm_compress_callback(engine) -> Callable:
    """Return an ADK ``before_model_callback`` that compresses context via LCM.

    Attach to any ``LlmAgent`` to get automatic context management::

        agent = LlmAgent(
            ...,
            before_model_callback=lcm_compress_callback(engine),
        )

    When the conversation history in ``LlmRequest.contents`` exceeds the
    configured threshold, LCM compresses older messages into DAG summaries
    and replaces ``contents`` with the shorter, lossless equivalent.

    Args:
        engine: An ``LCMEngine`` instance (already bound to a session).

    Returns:
        An async callback compatible with ``LlmAgent.before_model_callback``.
    """
    from .gemini import GeminiMessages

    async def _callback(callback_context: Any, llm_request: Any) -> None:
        try:
            contents = list(llm_request.contents or [])
            if not contents:
                return None

            # Extract system instruction if present
            system_instruction = getattr(llm_request, "system_instruction", None)
            sys_text = ""
            if system_instruction:
                sys_parts = list(getattr(system_instruction, "parts", []) or [])
                sys_text = " ".join(getattr(p, "text", "") for p in sys_parts if getattr(p, "text", ""))

            lcm_msgs = GeminiMessages.to_lcm(contents, system=sys_text)

            if not engine.should_compress_preflight(lcm_msgs):
                return None

            compressed        = await engine.compress(lcm_msgs)
            sys_out, new_contents = GeminiMessages.from_lcm(compressed)
            llm_request.contents  = new_contents

            if sys_out and system_instruction is not None:
                try:
                    from google.genai import types
                    llm_request.system_instruction = types.Content(
                        parts=[types.Part(text=sys_out)]
                    )
                except Exception:
                    pass

        except Exception as exc:
            # Never break the agent — log and let it proceed with original context
            logger.warning("lcm_compress_callback failed: %s", exc)

        return None   # return None to continue with (now possibly compressed) request

    return _callback
