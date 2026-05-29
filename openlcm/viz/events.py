"""EventBus — async pub/sub bridge between LCMEngine and SSE clients.

LCMEngine calls _emit(event_type, data) synchronously.
EventBus converts those synchronous calls into async SSE streams
for connected browser clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
        # Keep recent events for new clients connecting mid-session
        self._history: list[dict[str, Any]] = []

    def _get_loop(self) -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return self._loop

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop to use for thread-safe publishing."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE client and return its dedicated queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._queues.append(q)
        # Replay recent events to new client
        for event in self._history[-50:]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a disconnected client's queue."""
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
        self._history.append(event)
        if len(self._history) > _HISTORY_LIMIT:
            self._history = self._history[-_HISTORY_LIMIT:]

        loop = self._get_loop()
        dead_queues: list[asyncio.Queue] = []
        for q in list(self._queues):
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
        for q in dead_queues:
            self.unsubscribe(q)

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
        return len(self._queues)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
