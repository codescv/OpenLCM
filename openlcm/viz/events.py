"""EventBus — async pub/sub bridge between LCMEngine and SSE clients.

LCMEngine calls _emit(event_type, data) synchronously.
EventBus converts those synchronous calls into async SSE streams
for connected browser clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 200
_HISTORY_LIMIT = 500


class EventBus:
    """Thread-safe event bus connecting LCMEngine to SSE streams.

    Usage::

        bus = EventBus()
        engine.add_listener(bus.publish)

        # In FastAPI SSE endpoint:
        queue = bus.subscribe()
        async for event in bus.stream(queue):
            yield event
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: list[dict[str, Any]] = []
        # Protects _queues and _history from concurrent mutation between
        # the agent thread (publish) and asyncio threads (subscribe/unsubscribe).
        self._lock = threading.Lock()

    def _get_loop(self) -> asyncio.AbstractEventLoop | None:
        # Always prefer the registered uvicorn loop so cross-thread publish
        # wakes up the correct loop's queue consumers (not the caller's loop).
        if self._loop is not None:
            return self._loop
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop to use for thread-safe publishing."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE client and return its dedicated queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        with self._lock:
            self._queues.append(q)
            recent = list(self._history[-50:])
        for event in recent:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a disconnected client's queue."""
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def publish(self, event_type: str, data: dict) -> None:
        """Publish an event from LCMEngine (called synchronously).

        This method is safe to call from any thread. It enqueues the event
        into every connected client's asyncio.Queue using thread-safe methods.
        """
        event = {
            "type": event_type,
            "data": data,
            "ts": time.time(),
        }
        with self._lock:
            self._history.append(event)
            if len(self._history) > _HISTORY_LIMIT:
                self._history = self._history[-_HISTORY_LIMIT:]
            queues_snapshot = list(self._queues)

        loop = self._get_loop()
        for q in queues_snapshot:
            if loop and loop.is_running():
                try:
                    loop.call_soon_threadsafe(q.put_nowait, event)
                except (asyncio.QueueFull, RuntimeError):
                    pass
            else:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def stream(self, q: asyncio.Queue):
        """Async generator yielding SSE-formatted events from a queue."""
        try:
            while True:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield self._format_sse(event)
                q.task_done()
        except asyncio.TimeoutError:
            # Send keepalive ping
            yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(q)

    @staticmethod
    def _format_sse(event: dict) -> str:
        payload = json.dumps({"type": event["type"], "data": event["data"], "ts": event["ts"]}, ensure_ascii=False)
        return f"data: {payload}\n\n"

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._queues)

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)
