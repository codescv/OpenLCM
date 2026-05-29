"""OpenLCM CLI — openlcm command-line tool.

All read-only commands work without a running engine (just a DB path).
The viz command starts the live dashboard server.

Usage:
  openlcm status [--db PATH]
  openlcm grep QUERY [--limit N] [--db PATH]
  openlcm sessions [--db PATH]
  openlcm expand [--node-id N | --store-id N] [--db PATH]
  openlcm export SESSION_ID [--out FILE] [--db PATH]
  openlcm doctor [--db PATH]
  openlcm viz [--host HOST] [--port PORT] [--no-browser] [--db PATH]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("Install typer: pip install openlcm[viz]  or  pip install typer")
    sys.exit(1)

app = typer.Typer(
    name="openlcm",
    help="OpenLCM — Lossless Context Management CLI",
    add_completion=False,
)

_DEFAULT_DB = str(Path.home() / ".openlcm" / "lcm.db")


def _db_option():
    return typer.Option("", "--db", help="Path to lcm.db (default: ~/.openlcm/lcm.db)")


def _resolve_db(db: str) -> Path:
    return Path(db).expanduser().resolve() if db else Path.home() / ".openlcm" / "lcm.db"


def _open_store(db: str):
    from openlcm.core.store import MessageStore
    from openlcm.core.config import LCMConfig
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"Database not found: {db_path}", err=True)
        raise typer.Exit(1)
    config = LCMConfig()
    return MessageStore(db_path, ingest_protection_config=config)


def _open_dag(db: str):
    from openlcm.core.dag import SummaryDAG
    return SummaryDAG(_resolve_db(db))


def _open_lifecycle(db: str):
    from openlcm.core.lifecycle_state import LifecycleStateStore
    return LifecycleStateStore(_resolve_db(db))


# ── status ────────────────────────────────────────────────────────────────

@app.command()
def status(db: str = _db_option()):
    """Show current LCM database status."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}. Run an agent with LCM first.")
        raise typer.Exit(0)

    store = _open_store(db)
    dag = _open_dag(db)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()
    total_sessions = row[0] if row else 0
    row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
    total_messages = row[0] if row else 0
    row = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()
    total_nodes = row[0] if row else 0
    conn.close()

    db_size = db_path.stat().st_size if db_path.exists() else 0

    typer.echo(f"\nOpenLCM Status")
    typer.echo(f"  Database:       {db_path}")
    typer.echo(f"  Size:           {db_size / 1024:.1f} KB")
    typer.echo(f"  Sessions:       {total_sessions}")
    typer.echo(f"  Messages:       {total_messages}")
    typer.echo(f"  DAG nodes:      {total_nodes}")
    typer.echo("")


# ── sessions ──────────────────────────────────────────────────────────────

@app.command()
def sessions(db: str = _db_option()):
    """List all sessions in the database."""
    import sqlite3
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo(f"No database at {db_path}.")
        raise typer.Exit(0)

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT session_id, COUNT(*) as cnt, MIN(timestamp), MAX(timestamp) "
        "FROM messages GROUP BY session_id ORDER BY MAX(timestamp) DESC LIMIT 50"
    ).fetchall()
    conn.close()

    if not rows:
        typer.echo("No sessions found.")
        return

    typer.echo(f"\n{'SESSION ID':<50} {'MSGS':>6}  {'LAST ACTIVE':<20}")
    typer.echo("─" * 80)
    for r in rows:
        import datetime
        last = datetime.datetime.fromtimestamp(r[3]).strftime("%Y-%m-%d %H:%M") if r[3] else "—"
        typer.echo(f"{r[0]:<50} {r[1]:>6}  {last:<20}")
    typer.echo("")


# ── grep ──────────────────────────────────────────────────────────────────

@app.command()
def grep(
    query: str = typer.Argument(..., help="Search query (FTS5 syntax)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    session: str = typer.Option("", "--session", help="Restrict to session ID"),
    db: str = _db_option(),
):
    """Search conversation history with FTS5."""
    store = _open_store(db)
    session_id = session or None
    hits = store.search(query, session_id=session_id, limit=limit, sort="recency")
    if not hits:
        typer.echo(f"No results for '{query}'")
        return
    typer.echo(f"\n{len(hits)} result(s) for '{query}':\n")
    for h in hits:
        import datetime
        ts = h.get("timestamp", 0)
        time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
        typer.echo(f"  [{h.get('role','?').upper()}] #{h.get('store_id','')}  {time_str}")
        snippet = (h.get("snippet") or h.get("content") or "")[:200].replace("\n", " ")
        typer.echo(f"  {snippet}")
        typer.echo("")


# ── expand ────────────────────────────────────────────────────────────────

@app.command()
def expand(
    node_id: Optional[int] = typer.Option(None, "--node-id", help="Expand a summary node"),
    store_id: Optional[int] = typer.Option(None, "--store-id", help="Expand a raw message by store_id"),
    db: str = _db_option(),
):
    """Expand a summary node or raw message."""
    if node_id is None and store_id is None:
        typer.echo("Provide --node-id or --store-id", err=True)
        raise typer.Exit(1)

    if store_id is not None:
        store = _open_store(db)
        msg = store.get(store_id)
        if not msg:
            typer.echo(f"No message with store_id={store_id}")
            raise typer.Exit(1)
        typer.echo(f"\n[{msg.get('role','?').upper()}] store_id={store_id}  session={msg.get('session_id','')}\n")
        typer.echo(msg.get("content", "") or "(no content)")
        typer.echo("")
        return

    dag = _open_dag(db)
    node = dag.get_node(node_id)
    if not node:
        typer.echo(f"No DAG node with node_id={node_id}")
        raise typer.Exit(1)
    typer.echo(f"\nNode #{node_id} D{node.depth}  tokens={node.token_count}  src={node.source_token_count}\n")
    typer.echo(node.summary)
    typer.echo("")


# ── export ────────────────────────────────────────────────────────────────

@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    out: str = typer.Option("", "--out", "-o", help="Output file path (default: session_id.json)"),
    db: str = _db_option(),
):
    """Export full conversation history for a session as JSON."""
    store = _open_store(db)
    dag = _open_dag(db)

    messages = store.get_session_messages(session_id, limit=50000)
    nodes = dag.get_session_nodes(session_id, limit=10000)

    payload = {
        "session_id": session_id,
        "message_count": len(messages),
        "dag_node_count": len(nodes),
        "messages": messages,
        "dag_nodes": [
            {
                "node_id": n.node_id, "depth": n.depth, "summary": n.summary,
                "token_count": n.token_count, "source_token_count": n.source_token_count,
                "source_ids": n.source_ids, "source_type": n.source_type,
                "created_at": n.created_at, "earliest_at": n.earliest_at,
                "latest_at": n.latest_at, "expand_hint": n.expand_hint,
            }
            for n in nodes
        ],
    }

    output_path = out or f"{session_id[:30].replace('/', '-')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    typer.echo(f"Exported {len(messages)} messages + {len(nodes)} DAG nodes → {output_path}")


# ── doctor ────────────────────────────────────────────────────────────────

@app.command()
def doctor(db: str = _db_option()):
    """Run database integrity checks."""
    import sqlite3
    db_path = _resolve_db(db)
    typer.echo(f"\nOpenLCM Doctor — {db_path}\n")

    if not db_path.exists():
        typer.echo("✗ Database file not found")
        raise typer.Exit(1)

    conn = sqlite3.connect(str(db_path))
    checks = []

    # Quick check
    row = conn.execute("PRAGMA quick_check").fetchone()
    ok = row and row[0] == "ok"
    checks.append(("SQLite integrity", "ok" if ok else "FAIL", ok))

    # Table existence
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for t in ("messages", "summary_nodes"):
        checks.append((f"Table: {t}", "present" if t in tables else "MISSING", t in tables))

    # FTS
    fts_ok = "messages_fts" in tables and "nodes_fts" in tables
    checks.append(("FTS5 indexes", "present" if fts_ok else "MISSING", fts_ok))

    # Row counts
    msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] if "messages" in tables else 0
    nodes = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] if "summary_nodes" in tables else 0
    checks.append(("Message rows", str(msgs), True))
    checks.append(("DAG node rows", str(nodes), True))

    conn.close()

    for name, result, ok in checks:
        icon = "✓" if ok else "✗"
        typer.echo(f"  {icon} {name}: {result}")

    typer.echo(f"\n  DB size: {db_path.stat().st_size / 1024:.1f} KB\n")


# ── viz ───────────────────────────────────────────────────────────────────

@app.command()
def viz(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(7842, "--port", "-p", help="Bind port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
    db: str = _db_option(),
):
    """Start the live OpenLCM visualization dashboard."""
    try:
        from openlcm.viz.server import create_app, serve
    except ImportError:
        typer.echo("Install visualization deps: pip install openlcm[viz]", err=True)
        raise typer.Exit(1)

    db_path = _resolve_db(db)

    # Create a read-only engine for the dashboard (no summarization needed)
    engine = None
    if db_path.exists():
        try:
            from openlcm.core.engine import LCMEngine
            from openlcm.backends.base import SummaryBackend

            class _NullBackend(SummaryBackend):
                async def summarize(self, prompt, max_tokens, model="", timeout=None):
                    return None

            engine = LCMEngine(backend=_NullBackend(), db_path=str(db_path))
        except Exception as exc:
            typer.echo(f"Warning: could not initialize engine: {exc}", err=True)

    app = create_app(engine)
    serve(app, host=host, port=port, open_browser=not no_browser)


def main():
    app()


if __name__ == "__main__":
    main()
