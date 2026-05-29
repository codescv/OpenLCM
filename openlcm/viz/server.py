"""FastAPI visualization server for OpenLCM.

Serves the live dashboard and streams LCM lifecycle events via SSE.

Start via CLI: openlcm viz
Or programmatically::

    from openlcm.core.engine import LCMEngine
    from openlcm.viz.server import create_app, serve

    engine = LCMEngine(backend=...)
    app = create_app(engine)
    serve(app, host="127.0.0.1", port=7842)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(engine=None, bus=None):
    """Create and return the FastAPI application.

    Args:
        engine: LCMEngine instance. If None, the server returns empty/demo data.
        bus: EventBus instance. Created automatically if not provided.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        raise ImportError("pip install openlcm[viz]")

    from .events import EventBus

    if bus is None:
        bus = EventBus()
    if engine is not None:
        engine.add_listener(bus.publish)

    app = FastAPI(title="OpenLCM Dashboard", version="0.1.0", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    async def _startup():
        bus.set_loop(asyncio.get_running_loop())
        logger.info("OpenLCM Dashboard ready")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = _STATIC_DIR / "index.html"
        return HTMLResponse(content=index.read_text(encoding="utf-8"))

    @app.get("/events")
    async def sse_events(request: Request):
        """Server-Sent Events stream of LCM lifecycle events."""
        try:
            from sse_starlette.sse import EventSourceResponse
        except ImportError:
            raise ImportError("pip install sse-starlette")

        async def event_generator():
            q = bus.subscribe()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=25.0)
                        payload = json.dumps({"type": event["type"], "data": event["data"], "ts": event["ts"]}, ensure_ascii=False)
                        yield {"data": payload}
                        q.task_done()
                    except asyncio.TimeoutError:
                        yield {"data": json.dumps({"type": "ping", "data": {}, "ts": 0})}
                    except asyncio.CancelledError:
                        break
            finally:
                bus.unsubscribe(q)

        return EventSourceResponse(event_generator())

    @app.get("/api/status")
    async def api_status():
        if engine is None:
            return JSONResponse({"error": "No engine connected"})
        return JSONResponse(engine.get_status())

    @app.get("/api/dag")
    async def api_dag():
        if engine is None:
            return JSONResponse({"nodes": []})
        session_id = engine._session_id
        if not session_id:
            return JSONResponse({"nodes": [], "session_id": ""})
        nodes = engine._dag.get_session_nodes(session_id, limit=500)
        return JSONResponse({
            "session_id": session_id,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "depth": n.depth,
                    "token_count": n.token_count,
                    "source_token_count": n.source_token_count,
                    "source_type": n.source_type,
                    "source_ids": n.source_ids,
                    "created_at": n.created_at,
                    "earliest_at": n.earliest_at,
                    "latest_at": n.latest_at,
                    "expand_hint": n.expand_hint,
                    "summary_preview": n.summary[:200] + "..." if len(n.summary) > 200 else n.summary,
                }
                for n in nodes
            ],
        })

    @app.get("/api/messages")
    async def api_messages(limit: int = 50):
        if engine is None:
            return JSONResponse({"messages": []})
        session_id = engine._session_id
        if not session_id:
            return JSONResponse({"messages": [], "session_id": ""})
        rows = engine._store.get_session_tail(session_id, limit=min(limit, 200))
        return JSONResponse({
            "session_id": session_id,
            "messages": [
                {
                    "store_id": r.get("store_id"),
                    "role": r.get("role"),
                    "content_preview": (r.get("content") or "")[:300],
                    "token_estimate": r.get("token_estimate", 0),
                    "timestamp": r.get("timestamp"),
                }
                for r in rows
            ],
        })

    @app.post("/api/grep")
    async def api_grep(body: Dict[str, Any]):
        if engine is None:
            return JSONResponse({"results": []})
        query = str(body.get("query", ""))
        limit = int(body.get("limit", 10))
        session_id = engine._session_id
        if not query or not session_id:
            return JSONResponse({"results": []})
        hits = engine._store.search(query, session_id=session_id, limit=limit)
        return JSONResponse({
            "query": query,
            "results": [
                {
                    "store_id": h.get("store_id"),
                    "role": h.get("role"),
                    "snippet": h.get("snippet", (h.get("content") or "")[:200]),
                    "timestamp": h.get("timestamp"),
                }
                for h in hits
            ],
        })

    @app.get("/api/sessions")
    async def api_sessions():
        if engine is None:
            return JSONResponse({"sessions": []})
        try:
            rows = engine._store._conn.execute(
                "SELECT session_id, COUNT(*) as msg_count, MIN(timestamp) as first_at, MAX(timestamp) as last_at "
                "FROM messages GROUP BY session_id ORDER BY last_at DESC LIMIT 50"
            ).fetchall()
            return JSONResponse({
                "sessions": [
                    {"session_id": r[0], "message_count": r[1], "first_at": r[2], "last_at": r[3]}
                    for r in rows
                ]
            })
        except Exception as exc:
            return JSONResponse({"error": str(exc)})

    @app.get("/api/events/history")
    async def api_event_history():
        return JSONResponse({"events": bus.history[-100:]})

    # Serve static files (dashboard.js, styles.css, etc.)
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    app.state.engine = engine
    app.state.bus = bus
    return app


def serve(
    app=None,
    engine=None,
    host: str = "127.0.0.1",
    port: int = 7842,
    open_browser: bool = True,
) -> None:
    """Start the visualization server (blocking).

    Args:
        app: FastAPI app (created if None).
        engine: LCMEngine to attach (ignored if app already has one).
        host: Bind host.
        port: Bind port.
        open_browser: Open the dashboard in the default browser.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("pip install openlcm[viz]")

    if app is None:
        app = create_app(engine)

    if open_browser:
        import threading
        import webbrowser
        def _open():
            import time
            time.sleep(1.2)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\nOpenLCM Dashboard → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
