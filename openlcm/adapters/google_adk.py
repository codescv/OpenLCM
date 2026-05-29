"""Google ADK adapter — LCMSessionService.

Provides LCM-backed session management for Google ADK agents.

Install: pip install openlcm[google-adk]

Usage::

    from openlcm.core.engine import LCMEngine
    from openlcm.backends.litellm import LiteLLMBackend
    from openlcm.adapters.google_adk import LCMSessionService
    from google.adk.runners import Runner

    engine = LCMEngine(backend=LiteLLMBackend("gemini/gemini-1.5-flash"))

    session_service = LCMSessionService(engine)

    runner = Runner(
        agent=my_agent,
        app_name="my-app",
        session_service=session_service,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .base import LCMAdapter

logger = logging.getLogger(__name__)


class LCMSessionService(LCMAdapter):
    """Google ADK session service backed by LCMEngine.

    Mirrors the BaseSessionService interface:
    - create_session()  → bind LCM session
    - get_session()     → return session with LCM-assembled messages
    - append_event()    → persist event, compress if needed
    - delete_session()  → clear session data

    ADK events are translated to standard message dicts so they flow
    through LCM's compaction pipeline naturally.
    """

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self._sessions: Dict[str, Dict[str, Any]] = {}

    async def create_session(
        self,
        *,
        app_name: str = "",
        user_id: str = "",
        session_id: str = "",
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and bind a new session."""
        sid = session_id or f"{app_name}:{user_id}:{int(time.time())}"
        self._engine.bind_session(sid, platform="google-adk")
        session = {
            "id": sid,
            "app_name": app_name,
            "user_id": user_id,
            "state": state or {},
            "events": [],
            "last_update_time": time.time(),
        }
        self._sessions[sid] = session
        return session

    async def get_session(
        self,
        *,
        app_name: str = "",
        user_id: str = "",
        session_id: str,
        config: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return session with LCM-assembled messages."""
        if session_id not in self._sessions:
            self._engine.bind_session(session_id, platform="google-adk")
            rows = self._engine._store.get_session_messages(session_id, limit=1000)
            if not rows:
                return None
            self._sessions[session_id] = {
                "id": session_id,
                "app_name": app_name,
                "user_id": user_id,
                "state": {},
                "events": [self._row_to_event(r) for r in rows],
                "last_update_time": time.time(),
            }

        session = dict(self._sessions[session_id])
        messages = [self._row_to_message(r) for r in
                    self._engine._store.get_session_messages(session_id, limit=1000)]
        if messages and self._engine.should_compress_preflight(messages):
            import asyncio
            compressed = await self._engine.compress(messages)
            session["_lcm_context"] = compressed
        return session

    async def append_event(
        self,
        session: Dict[str, Any],
        event: Any,
    ) -> Any:
        """Persist an event and trigger compaction if context pressure builds."""
        session_id = session.get("id", "")
        if not session_id:
            return event

        if session_id != self._engine._session_id:
            self._engine.bind_session(session_id, platform="google-adk")

        msg = self._event_to_message(event)
        if msg:
            self._engine._ingest_messages([msg])
            all_messages = [self._row_to_message(r) for r in
                           self._engine._store.get_session_messages(session_id, limit=1000)]
            if all_messages and self._engine.should_compress_preflight(all_messages):
                import asyncio
                await self._engine.compress(all_messages)

        if "events" in session:
            session["events"].append(event)
        session["last_update_time"] = time.time()
        if session_id in self._sessions:
            self._sessions[session_id]["last_update_time"] = time.time()
        return event

    async def delete_session(
        self,
        *,
        app_name: str = "",
        user_id: str = "",
        session_id: str,
    ) -> None:
        """Remove a session and its LCM data."""
        self._sessions.pop(session_id, None)
        try:
            self._engine._store.delete_session_messages(session_id)
            self._engine._dag.delete_session_nodes(session_id)
        except Exception as exc:
            logger.warning("LCMSessionService.delete_session failed: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _event_to_message(event: Any) -> Optional[Dict[str, Any]]:
        if isinstance(event, dict):
            role = event.get("author", event.get("role", "user"))
            content = event.get("content", "")
            if isinstance(content, dict):
                parts = content.get("parts", [])
                text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
                content = "\n".join(text_parts)
            return {"role": str(role), "content": str(content)} if content else None
        if hasattr(event, "content"):
            role = getattr(event, "author", "user")
            content = getattr(event.content, "parts", None)
            if content:
                text = "\n".join(getattr(p, "text", "") for p in content if hasattr(p, "text"))
                return {"role": str(role), "content": text}
        return None

    @staticmethod
    def _row_to_message(row: Dict[str, Any]) -> Dict[str, Any]:
        return {"role": row.get("role", "user"), "content": row.get("content", "")}

    @staticmethod
    def _row_to_event(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "author": row.get("role", "user"),
            "content": {"parts": [{"text": row.get("content", "")}]},
            "timestamp": row.get("timestamp", 0),
        }
