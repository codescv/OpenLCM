"""FastAPI visualization server for OpenLCM — multi-session dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _node_to_dict(n) -> dict:
    return {
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
        "summary": n.summary,
        "summary_preview": n.summary[:200] + "..." if len(n.summary) > 200 else n.summary,
    }


def _session_stats(engine, session_id: str) -> dict:
    """Build a stats dict for any session_id (not just the active one)."""
    nodes = engine._dag.get_session_nodes(session_id, limit=1000)
    msg_count = engine._store.get_session_count(session_id)
    total_tokens = engine._store.get_session_token_total(session_id)
    tokens_freed = sum(max(0, n.source_token_count - n.token_count) for n in nodes)
    d0_count = sum(1 for n in nodes if n.depth == 0)
    last_at: float | None = None
    first_at: float | None = None
    try:
        sessions = engine._store.list_sessions()
        for s in sessions:
            if s["session_id"] == session_id:
                last_at = s.get("last_at")
                first_at = s.get("first_at")
                break
    except Exception:
        pass

    is_active = engine._session_id == session_id
    extra: dict = {}
    if is_active:
        extra = {
            "last_prompt_tokens": engine.last_prompt_tokens,
            "context_length": engine.context_length,
            "threshold_tokens": engine.threshold_tokens,
            "threshold_percent": engine.threshold_percent,
            "compression_count": engine.compression_count,
            "last_compression_status": engine._last_compression_status,
            "summary_model": engine._config.summary_model,
            "fresh_tail_count": engine._config.fresh_tail_count,
            "db_path": str(engine._store.db_path),
            "overflow_recovery_failed": engine._last_overflow_recovery_failed,
        }

    return {
        "session_id": session_id,
        "is_active": is_active,
        "message_count": msg_count,
        "total_tokens": total_tokens,
        "dag_nodes": len(nodes),
        "dag_d0": d0_count,
        "tokens_freed": tokens_freed,
        "first_at": first_at,
        "last_at": last_at,
        **extra,
    }


def create_app(engine=None, bus=None):
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        raise ImportError("pip install openlcm[viz]")

    from .events import EventBus

    if bus is None:
        bus = EventBus()
    if engine is not None:
        engine.add_listener(bus.publish)

    app = FastAPI(title="OpenLCM Dashboard", version="0.2.0", docs_url=None, redoc_url=None)

    # Restrict cross-origin requests to same-origin (localhost) only.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:7842", "http://localhost:7842"],
        allow_methods=["GET", "POST"],  # DELETE is intentionally excluded from CORS
        allow_headers=["*"],
    )

    def _require_local(request: Request) -> None:
        """Reject destructive requests that don't originate from localhost."""
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(status_code=403, detail="Dashboard write access is restricted to localhost.")

    @app.on_event("startup")
    async def _startup():
        bus.set_loop(asyncio.get_running_loop())
        logger.info("OpenLCM Dashboard ready")

    # ── Static pages ────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(content=(_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    # ── SSE ─────────────────────────────────────────────────────────────────

    @app.get("/events")
    async def sse_events(request: Request):
        """SSE stream — plain StreamingResponse, no extra dependencies."""
        from fastapi.responses import StreamingResponse as _SR

        async def event_generator():
            q = bus.subscribe()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=20.0)
                        if event.get("type") == "ping":
                            yield ": keepalive\n\n"
                            continue
                        payload = json.dumps(
                            {"type": event["type"], "data": event["data"], "ts": event["ts"]},
                            ensure_ascii=False,
                        )
                        yield f"data: {payload}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                    except (asyncio.CancelledError, GeneratorExit):
                        break
                    except Exception:
                        break
            finally:
                bus.unsubscribe(q)

        return _SR(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache, no-transform",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/events/history")
    async def api_event_history():
        return JSONResponse({"events": bus.history[-200:]})

    # ── Global overview ──────────────────────────────────────────────────────

    @app.get("/api/overview")
    async def api_overview():
        """Global stats across all sessions."""
        if engine is None:
            return JSONResponse({"sessions": [], "totals": {}})
        try:
            store_sessions = engine._store.list_sessions()
        except Exception:
            store_sessions = []

        sessions_out = []
        global_messages = 0
        global_tokens_freed = 0
        global_dag_nodes = 0
        global_compressions = 0

        for s in store_sessions:
            sid = s["session_id"]
            nodes = engine._dag.get_session_nodes(sid, limit=1000)
            freed = sum(max(0, n.source_token_count - n.token_count) for n in nodes)
            d0 = sum(1 for n in nodes if n.depth == 0)
            is_active = engine._session_id == sid

            global_messages += s.get("message_count", 0)
            global_tokens_freed += freed
            global_dag_nodes += len(nodes)
            global_compressions += d0

            sessions_out.append({
                "session_id": sid,
                "is_active": is_active,
                "message_count": s.get("message_count", 0),
                "total_tokens": s.get("total_tokens", 0),
                "dag_nodes": len(nodes),
                "compressions": d0,
                "tokens_freed": freed,
                "first_at": s.get("first_at"),
                "last_at": s.get("last_at"),
                # Active-session extras
                **({"last_prompt_tokens": engine.last_prompt_tokens,
                    "context_length": engine.context_length,
                    "threshold_tokens": engine.threshold_tokens,
                    "last_compression_status": engine._last_compression_status,
                    } if is_active else {}),
            })

        global_facts = 0
        try:
            facts_store = getattr(engine, "_facts", None)
            if facts_store is not None:
                global_facts = len(facts_store.recall_query(limit=10000))
        except Exception:
            pass

        return JSONResponse({
            "sessions": sessions_out,
            "active_session": engine._session_id,
            "totals": {
                "sessions": len(sessions_out),
                "messages": global_messages,
                "tokens_freed": global_tokens_freed,
                "dag_nodes": global_dag_nodes,
                "compressions": global_compressions,
                "facts": global_facts,
            },
        })

    # ── Per-session read endpoints ────────────────────────────────────────────

    @app.get("/api/status")
    async def api_status(session_id: str = ""):
        if engine is None:
            return JSONResponse({"error": "No engine connected"})
        sid = session_id or engine._session_id
        if not sid:
            return JSONResponse(engine.get_status())
        if sid == engine._session_id:
            return JSONResponse(engine.get_status())
        return JSONResponse(_session_stats(engine, sid))

    @app.get("/api/dag")
    async def api_dag(session_id: str = ""):
        if engine is None:
            return JSONResponse({"nodes": []})
        sid = session_id or engine._session_id
        if not sid:
            return JSONResponse({"nodes": [], "session_id": ""})
        nodes = engine._dag.get_session_nodes(sid, limit=500)
        return JSONResponse({
            "session_id": sid,
            "nodes": [_node_to_dict(n) for n in nodes],
        })

    @app.get("/api/messages")
    async def api_messages(limit: int = 200, session_id: str = ""):
        if engine is None:
            return JSONResponse({"messages": []})
        sid = session_id or engine._session_id
        if not sid:
            return JSONResponse({"messages": [], "session_id": ""})
        rows = engine._store.get_session_tail(sid, limit=min(limit, 500))
        return JSONResponse({
            "session_id": sid,
            "messages": [
                {
                    "store_id": r.get("store_id"),
                    "role": r.get("role"),
                    "content_preview": (r.get("content") or "")[:300],
                    "content_full": r.get("content") or "",
                    "token_estimate": r.get("token_estimate", 0),
                    "created_at": r.get("created_at"),
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
        session_id = str(body.get("session_id", "")) or engine._session_id
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
                    "created_at": h.get("created_at"),
                }
                for h in hits
            ],
        })

    # ── CRUD: Session-level ───────────────────────────────────────────────────

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        _require_local(request)
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        msg_count = engine._store.delete_session_messages(session_id)
        dag_count = engine._dag.delete_session_nodes(session_id)
        bus.publish("session_deleted", {"session_id": session_id, "messages": msg_count, "dag_nodes": dag_count})
        return JSONResponse({"deleted": True, "messages": msg_count, "dag_nodes": dag_count})

    @app.delete("/api/sessions/{session_id}/messages")
    async def clear_session_messages(session_id: str, request: Request):
        _require_local(request)
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        count = engine._store.delete_session_messages(session_id)
        bus.publish("messages_cleared", {"session_id": session_id, "count": count})
        return JSONResponse({"deleted": True, "count": count})

    @app.delete("/api/sessions/{session_id}/dag")
    async def clear_session_dag(session_id: str, request: Request):
        _require_local(request)
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        count = engine._dag.delete_session_nodes(session_id)
        bus.publish("dag_cleared", {"session_id": session_id, "count": count})
        return JSONResponse({"deleted": True, "count": count})

    # ── CRUD: Node-level ──────────────────────────────────────────────────────

    @app.delete("/api/sessions/{session_id}/dag/{node_id}")
    async def delete_dag_node(session_id: str, node_id: int, request: Request):
        _require_local(request)
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        ok = engine._dag.delete_node(node_id)
        if ok:
            bus.publish("node_deleted", {"session_id": session_id, "node_id": node_id})
        return JSONResponse({"deleted": ok})

    # ── CRUD: Message-level ───────────────────────────────────────────────────

    @app.delete("/api/sessions/{session_id}/messages/{store_id}")
    async def delete_message(session_id: str, store_id: int, request: Request):
        _require_local(request)
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        ok = engine._store.delete_message(store_id)
        if ok:
            bus.publish("message_deleted", {"session_id": session_id, "store_id": store_id})
        return JSONResponse({"deleted": ok})

    # ── Fact store ────────────────────────────────────────────────────────────

    @app.get("/api/facts")
    async def api_facts(scope: str = "", query: str = "", limit: int = 200):
        if engine is None:
            return JSONResponse({"facts": [], "total": 0})
        facts_store = getattr(engine, "_facts", None)
        if facts_store is None:
            return JSONResponse({"facts": [], "total": 0})
        rows = facts_store.recall_query(query, scope=scope or None, limit=min(limit, 500))
        return JSONResponse({"facts": rows, "total": len(rows)})

    @app.post("/api/facts")
    async def api_facts_upsert(body: Dict[str, Any]):
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        facts_store = getattr(engine, "_facts", None)
        if facts_store is None:
            return JSONResponse({"error": "Fact store not available"}, status_code=503)
        key = str(body.get("key") or "").strip()
        value = str(body.get("value") or "").strip()
        if not key or not value:
            return JSONResponse({"error": "key and value required"}, status_code=400)
        scope = str(body.get("scope") or "global").strip()
        category = str(body.get("category") or "fact").strip()
        fact_id = facts_store.remember(
            key, value, scope=scope, category=category,
            source_session_id=engine.current_session_id or "",
        )
        bus.publish("fact_stored", {"key": key, "scope": scope, "category": category})
        return JSONResponse({"fact_id": fact_id, "key": key, "value": value, "scope": scope, "category": category})

    @app.delete("/api/facts")
    async def api_facts_delete(request: Request, body: Dict[str, Any]):
        _require_local(request)
        if engine is None:
            return JSONResponse({"error": "No engine"}, status_code=503)
        facts_store = getattr(engine, "_facts", None)
        if facts_store is None:
            return JSONResponse({"error": "Fact store not available"}, status_code=503)
        key = str(body.get("key") or "").strip()
        scope = str(body.get("scope") or "global").strip()
        if not key:
            return JSONResponse({"error": "key required"}, status_code=400)
        deleted = facts_store.forget(key, scope=scope)
        if deleted:
            bus.publish("fact_deleted", {"key": key, "scope": scope})
        return JSONResponse({"deleted": deleted})

    # ── Static files ──────────────────────────────────────────────────────────

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
    try:
        import uvicorn
    except ImportError:
        raise ImportError("pip install openlcm[viz]")

    if app is None:
        app = create_app(engine)

    if open_browser:
        import threading, webbrowser
        def _open():
            import time as _t; _t.sleep(1.2)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\nOpenLCM Dashboard → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
