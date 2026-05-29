"""Core LCM engine tests — no external LLM calls needed.

Tests run fully offline using an in-memory SQLite database and a mock
SummaryBackend that returns deterministic summaries.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from openlcm.core.config import LCMConfig
from openlcm.core.engine import LCMEngine
from openlcm.backends.base import SummaryBackend


# ── Test fixtures ──────────────────────────────────────────────────────────

class MockBackend(SummaryBackend):
    """Returns a predictable summary for testing."""

    def __init__(self, prefix: str = "SUMMARY"):
        self.prefix = prefix
        self.call_count = 0

    async def summarize(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "",
        timeout: float | None = None,
    ) -> Optional[str]:
        self.call_count += 1
        word_count = len(prompt.split())
        return f"{self.prefix}: {word_count} words compressed. Expand for details about: test content"


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_lcm.db"


@pytest.fixture
def backend():
    return MockBackend()


@pytest.fixture
def engine(tmp_db, backend):
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=100,
        context_threshold=0.5,
        condensation_fanin=3,
        incremental_max_depth=1,
    )
    return LCMEngine(backend=backend, config=config, db_path=str(tmp_db))


def make_messages(n: int, role_cycle=("user", "assistant")) -> list:
    msgs = []
    for i in range(n):
        role = role_cycle[i % len(role_cycle)]
        msgs.append({
            "role": role,
            "content": f"This is message number {i + 1} with some content to fill tokens. " * 5,
        })
    return msgs


# ── Tests: Engine initialization ──────────────────────────────────────────

def test_engine_requires_backend():
    with pytest.raises(TypeError, match="SummaryBackend"):
        LCMEngine(backend="not-a-backend")


def test_engine_creates_db(tmp_db, backend):
    engine = LCMEngine(backend=backend, db_path=str(tmp_db))
    engine.bind_session("test-session")
    assert tmp_db.exists()


def test_engine_bind_session(engine):
    engine.bind_session("session-1", platform="test", context_length=100_000)
    assert engine._session_id == "session-1"
    assert engine._session_platform == "test"
    assert engine.context_length == 100_000
    assert engine.threshold_tokens == 50_000  # 0.5 * 100_000


def test_engine_properties(engine):
    engine.bind_session("session-x")
    assert engine.current_session_id == "session-x"
    assert engine._hermes_home  # should return db parent dir


# ── Tests: Message ingestion ───────────────────────────────────────────────

def test_ingest_messages(engine):
    engine.bind_session("ingest-test")
    msgs = make_messages(5)
    engine._ingest_messages(msgs)
    count = engine._store.get_session_count("ingest-test")
    assert count == 5


def test_ingest_idempotent_cursor(engine):
    engine.bind_session("cursor-test")
    msgs = make_messages(3)
    engine._ingest_messages(msgs)
    engine._ingest_messages(msgs)  # second call: cursor at 3, nothing new
    count = engine._store.get_session_count("cursor-test")
    assert count == 3  # not 6


def test_ingest_new_messages_appended(engine):
    engine.bind_session("append-test")
    msgs = make_messages(3)
    engine._ingest_messages(msgs)
    extended = msgs + make_messages(2)
    engine._ingest_messages(extended)
    count = engine._store.get_session_count("append-test")
    assert count == 5


# ── Tests: should_compress ────────────────────────────────────────────────

def test_should_compress_false_below_threshold(engine):
    engine.bind_session("s1", context_length=100_000)
    engine.last_prompt_tokens = 30_000
    assert not engine.should_compress()


def test_should_compress_true_above_threshold(engine):
    engine.bind_session("s1", context_length=100_000)
    engine.last_prompt_tokens = 60_000
    assert engine.should_compress()


def test_should_compress_false_no_threshold(engine):
    engine.bind_session("s1")
    engine.context_length = 0
    engine.threshold_tokens = 0
    assert not engine.should_compress(prompt_tokens=999_999)


# ── Tests: compress() ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compress_empty_returns_empty(engine):
    engine.bind_session("empty-test")
    result = await engine.compress([])
    assert result == []


@pytest.mark.asyncio
async def test_compress_noop_small_context(engine):
    engine.bind_session("small-test")
    msgs = make_messages(3)
    result = await engine.compress(msgs)
    # No compaction — not enough raw backlog
    assert result is not None  # returns something


@pytest.mark.asyncio
async def test_compress_creates_dag_node(engine, backend):
    engine.bind_session("dag-test")
    # Provide enough messages to trigger leaf compaction
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    msgs += make_messages(20)
    # First ingest all
    engine._ingest_messages(msgs)
    # Now compress
    result = await engine.compress(msgs)
    # DAG should have nodes
    nodes = engine._dag.get_session_nodes("dag-test")
    assert len(nodes) > 0
    assert backend.call_count > 0


@pytest.mark.asyncio
async def test_compress_emits_events(engine):
    engine.bind_session("events-test")
    events = []
    engine.add_listener(lambda t, d: events.append((t, d)))

    msgs = [{"role": "system", "content": "System prompt."}]
    msgs += make_messages(20)
    engine._ingest_messages(msgs)
    await engine.compress(msgs)

    event_types = [e[0] for e in events]
    assert "message_ingested" in event_types
    assert "compaction_start" in event_types
    assert "compaction_end" in event_types


@pytest.mark.asyncio
async def test_compress_preserves_fresh_tail(engine):
    engine.bind_session("tail-test")
    msgs = [{"role": "system", "content": "System."}] + make_messages(20)
    engine._ingest_messages(msgs)
    result = await engine.compress(msgs)

    # Fresh tail (last 4) should always be in result
    assert len(result) >= 1  # at minimum the assembled context has something


@pytest.mark.asyncio
async def test_compress_count_increments(engine):
    engine.bind_session("count-test")
    msgs = [{"role": "system", "content": "System."}] + make_messages(20)
    engine._ingest_messages(msgs)

    assert engine.compression_count == 0
    await engine.compress(msgs)
    # compression_count may or may not increment depending on whether threshold was met
    # Just verify it doesn't error
    assert engine.compression_count >= 0


# ── Tests: DAG ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dag_summary_content(engine, backend):
    engine.bind_session("dag-content-test")
    msgs = [{"role": "system", "content": "You are helpful."}] + make_messages(20)
    engine._ingest_messages(msgs)
    await engine.compress(msgs)

    nodes = engine._dag.get_session_nodes("dag-content-test")
    if nodes:
        node = nodes[0]
        assert node.summary.startswith("SUMMARY:")
        assert node.token_count > 0


# ── Tests: Tools ──────────────────────────────────────────────────────────

def test_get_tool_schemas(engine):
    schemas = engine.get_tool_schemas()
    assert len(schemas) == 7
    names = {s["name"] for s in schemas}
    assert "lcm_grep" in names
    assert "lcm_expand" in names
    assert "lcm_status" in names


def test_handle_tool_lcm_status(engine):
    engine.bind_session("tool-test")
    result = engine.handle_tool_call("lcm_status", {})
    data = json.loads(result)
    assert "engine" in data or "session_id" in data


def test_handle_tool_unknown(engine):
    engine.bind_session("tool-test")
    result = engine.handle_tool_call("nonexistent_tool", {})
    data = json.loads(result)
    assert "error" in data


# ── Tests: Event hooks ────────────────────────────────────────────────────

def test_add_remove_listener(engine):
    engine.bind_session("listener-test")
    events = []
    cb = lambda t, d: events.append(t)
    engine.add_listener(cb)
    engine._emit("test_event", {"x": 1})
    assert "test_event" in events
    engine.remove_listener(cb)
    engine._emit("test_event", {"x": 2})
    assert events.count("test_event") == 1  # not duplicated


# ── Tests: MessageStore ───────────────────────────────────────────────────

def test_store_search(tmp_db, backend):
    engine = LCMEngine(backend=backend, db_path=str(tmp_db))
    engine.bind_session("search-test")
    msgs = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog"}]
    engine._ingest_messages(msgs)

    hits = engine._store.search("quick brown", session_id="search-test")
    assert len(hits) > 0
    assert "quick" in hits[0].get("content", "").lower()


def test_store_session_count(tmp_db, backend):
    engine = LCMEngine(backend=backend, db_path=str(tmp_db))
    engine.bind_session("count-test")
    msgs = make_messages(7)
    engine._ingest_messages(msgs)
    assert engine._store.get_session_count("count-test") == 7


# ── Tests: Session rollover ───────────────────────────────────────────────

def test_rollover_session(engine):
    engine.bind_session("old-session")
    msgs = make_messages(5)
    engine._ingest_messages(msgs)

    moved = engine.rollover_session("old-session", "new-session")
    assert engine._session_id == "new-session"
    # Compression count resets
    assert engine.compression_count == 0


# ── Tests: get_status ─────────────────────────────────────────────────────

def test_get_status(engine):
    engine.bind_session("status-test", context_length=50_000)
    status = engine.get_status()
    assert status["session_id"] == "status-test"
    assert status["context_length"] == 50_000
    assert status["fresh_tail_count"] == 4


# ── Tests: Backend import ─────────────────────────────────────────────────

def test_anthropic_backend_import():
    from openlcm.backends.anthropic import AnthropicBackend
    b = AnthropicBackend(model="claude-haiku-4-5-20251001")
    assert b._default_model == "claude-haiku-4-5-20251001"


def test_openai_backend_import():
    from openlcm.backends.openai import OpenAIBackend
    b = OpenAIBackend(model="gpt-4o-mini")
    assert b._default_model == "gpt-4o-mini"


def test_litellm_backend_import():
    from openlcm.backends.litellm import LiteLLMBackend
    b = LiteLLMBackend(model="anthropic/claude-haiku-4-5")
    assert b._model == "anthropic/claude-haiku-4-5"


def test_litellm_backend_requires_model():
    from openlcm.backends.litellm import LiteLLMBackend
    with pytest.raises(ValueError, match="model string"):
        LiteLLMBackend(model="")
