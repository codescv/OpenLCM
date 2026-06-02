"""LCMEngine — Lossless Context Management, framework-agnostic.

Architecture:
  1. Every message is persisted verbatim in an immutable MessageStore (SQLite)
  2. When context pressure builds, older messages outside the fresh tail
     are summarized into leaf nodes (D0) in a SummaryDAG
  3. When enough D0 nodes accumulate, they are condensed into D1, D2, ...
  4. The assembled active context = system prompt + DAG summaries + fresh tail
  5. Event hooks let the visualization layer observe every lifecycle transition

Usage::

    from openlcm.core.engine import LCMEngine
    from openlcm.core.config import LCMConfig
    from openlcm.backends.anthropic import AnthropicBackend

    engine = LCMEngine(
        backend=AnthropicBackend(model="claude-haiku-4-5-20251001"),
        db_path="~/.openlcm/myapp.db",
    )
    engine.bind_session("session-abc")

    # Each turn: ingest current messages, get back the compressed list
    compressed = await engine.compress(messages)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import LCMConfig
from .dag import SummaryDAG, SummaryNode
from .escalation import SummaryCircuitBreaker, _strip_reasoning_blocks, summarize_with_escalation
from .externalize import (
    build_transcript_gc_placeholder,
    extract_externalized_ref,
    find_externalized_payload_for_message,
    load_externalized_payload,
    maybe_externalize_tool_output,
    reassign_externalized_payloads,
)
from .extraction import extract_before_compaction, sanitize_pre_compaction_content, sanitize_pre_compaction_tool_arguments
from .ingest_protection import (
    assistant_output_quarantine_reason,
    extract_ingest_externalized_refs,
    protect_inline_payloads_in_text,
    protect_messages_for_ingest,
    quarantine_suspicious_assistant_messages,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
    sensitive_pattern_status,
)
from .facts import FactStore
from .lifecycle_state import LifecycleStateStore
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .message_patterns import compile_message_patterns, matches_message_pattern
from .session_patterns import build_session_match_keys, compile_session_patterns, matches_session_pattern
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

_PRESERVED_TODO_CONTEXT_PREFIX = "[Your active task list was preserved across context compression]"
_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"

_VISIBLE_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_INTERNAL_ASSISTANT_PART_TYPES = {
    "analysis", "chain_of_thought", "internal", "reasoning",
    "redacted_thinking", "scratchpad", "thought", "thinking",
}


def _tool_call_id(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return ""
    value = tool_call.get("id") or tool_call.get("tool_call_id")
    return str(value).strip() if value else ""


def _assistant_tool_call_ids(messages: List[Dict[str, Any]]) -> set[str]:
    call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            cid = _tool_call_id(tc)
            if cid:
                call_ids.add(cid)
    return call_ids


class LCMEngine:
    """Framework-agnostic Lossless Context Management engine.

    This class contains the full LCM algorithm extracted from hermes-lcm,
    with all Hermes-specific dependencies removed. It works standalone with
    any Python agent framework.

    Args:
        backend: A SummaryBackend implementation for LLM-powered summarization.
                 Required — there is no default. Use AnthropicBackend,
                 OpenAIBackend, or LiteLLMBackend from openlcm.backends.
        config: LCMConfig instance. Reads from LCM_* env vars if not provided.
        db_path: Path to the SQLite database file. Defaults to ~/.openlcm/lcm.db.
    """

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str = "",
        api_base: str = "",
        summarize_fn=None,
        backend=None,
        config: LCMConfig | None = None,
        db_path: str | Path = "",
        **litellm_kwargs,
    ) -> None:
        from openlcm.backends.base import SummaryBackend
        from openlcm.backends.litellm import LiteLLMBackend
        from openlcm.backends.callable import CallableBackend

        if backend is not None:
            # Explicit SummaryBackend instance — advanced / custom use
            if not isinstance(backend, SummaryBackend):
                raise TypeError(
                    f"backend must be a SummaryBackend instance, got {type(backend).__name__}."
                )
            self._backend = backend
        elif summarize_fn is not None:
            # Framework user path: pass your existing LLM or any callable.
            # Works with LangChain models, CrewAI LLMs, plain functions, etc.
            #   engine = LCMEngine(summarize_fn=my_langchain_llm)
            #   engine = LCMEngine(summarize_fn=lambda p, mt: my_llm.invoke(p).content)
            if isinstance(summarize_fn, SummaryBackend):
                self._backend = summarize_fn
            else:
                self._backend = CallableBackend(summarize_fn)
        elif model:
            # Vanilla / standalone path: just pass a LiteLLM model string.
            #   engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")
            #   engine = LCMEngine(model="azure/gpt-4o")
            #   engine = LCMEngine(model="ollama/llama3.2", api_base="http://localhost:11434")
            self._backend = LiteLLMBackend(
                model=model,
                api_key=api_key,
                api_base=api_base,
                **litellm_kwargs,
            )
        else:
            raise ValueError(
                "LCMEngine needs to know which LLM to use for summarization.\n"
                "\n"
                "Already using a framework? Pass your existing LLM:\n"
                "  engine = LCMEngine(summarize_fn=llm)          # LangChain/CrewAI model\n"
                "  engine = LCMEngine(summarize_fn=my_callable)  # any (prompt, max_tokens)->str\n"
                "\n"
                "Starting from scratch? Pass a LiteLLM model string:\n"
                "  engine = LCMEngine(model='anthropic/claude-haiku-4-5-20251001')\n"
                "  engine = LCMEngine(model='azure/gpt-4o')\n"
                "  engine = LCMEngine(model='bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')\n"
                "  engine = LCMEngine(model='ollama/llama3.2', api_base='http://localhost:11434')\n"
                "  engine = LCMEngine(model='gemini/gemini-2.0-flash')\n"
                "  Full list: https://docs.litellm.ai/docs/providers"
            )
        self._config = config or LCMConfig.from_env()

        resolved_db = self._resolve_db_path(db_path)
        self._store = MessageStore(resolved_db, ingest_protection_config=self._config)
        self._dag = SummaryDAG(resolved_db)
        self._lifecycle = LifecycleStateStore(resolved_db)
        self._facts = FactStore(resolved_db)

        # Session state
        self._session_id: str = ""
        self._conversation_id: str = ""
        self._session_platform: str = ""
        self._session_ignored: bool = False
        self._session_stateless: bool = False
        self._session_match_keys: list[str] = []

        # Session filters
        self._compiled_ignore_session_patterns = compile_session_patterns(self._config.ignore_session_patterns)
        self._compiled_stateless_session_patterns = compile_session_patterns(self._config.stateless_session_patterns)
        self._compiled_ignore_message_patterns = compile_message_patterns(self._config.ignore_message_patterns)
        self._ignored_message_count: int = 0

        # Compaction state
        self._last_compacted_store_id: int = 0
        self._ingest_cursor: int = 0
        self._ingest_cursor_needs_reconcile: bool = False

        # Runtime metrics
        self.context_length: int = 0
        self.threshold_tokens: int = 0
        self.threshold_percent: float = self._config.context_threshold
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_cache_read_tokens: int = 0
        self.last_cache_write_tokens: int = 0
        self.compression_count: int = 0
        self._last_compression_status: str = "idle"
        self._last_compression_noop_reason: str = ""
        self._last_condensation_suppressed_reason: str = ""
        self._last_overflow_recovery_failed: bool = False
        self._last_ingest_reconciliation: Dict[str, Any] = {"action": "none", "reason": "not run"}

        # Protect first/last N messages
        self.protect_first_n: int = 3
        self.protect_last_n: int = self._config.fresh_tail_count

        # Summary circuit breaker
        self._summary_circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=self._config.summary_circuit_breaker_failure_threshold,
            cooldown_seconds=self._config.summary_circuit_breaker_cooldown_seconds,
        )

        # Event hooks for visualization layer
        self._listeners: list[Callable[[str, dict], None]] = []

        self._pending_context_anchor_messages: Optional[List[Dict[str, Any]]] = None

    @property
    def current_session_id(self) -> str:
        """The active session ID (compatibility with tool layer)."""
        return self._session_id

    @property
    def side_channel_active(self) -> bool:
        """Always False in standalone engine (no Hermes side-channel concept)."""
        return False

    @property
    def current_session_platform(self) -> str:
        return self._session_platform

    @property
    def current_conversation_id(self) -> str:
        return self._conversation_id

    @property
    def current_session_ignored(self) -> bool:
        return self._session_ignored

    @property
    def current_session_stateless(self) -> bool:
        return self._session_stateless

    @property
    def last_reasoning_tokens(self) -> int:
        return 0

    @property
    def cache_metrics_available(self) -> bool:
        return self.last_cache_read_tokens > 0 or self.last_cache_write_tokens > 0

    @property
    def cache_read_ratio(self) -> float:
        if self.last_prompt_tokens <= 0:
            return 0.0
        return self.last_cache_read_tokens / self.last_prompt_tokens

    def rotate_backup_path(self) -> Path:
        """Return the path for rolling rotate backups."""
        return self._store.db_path.parent / "lcm-rotate-latest.sqlite3"

    @property
    def _hermes_home(self) -> str:
        """Compatibility shim — tools.py uses this to locate externalized payloads."""
        return str(self._store.db_path.parent)

    # ── Public event hook API ──────────────────────────────────────────────

    def add_listener(self, callback: Callable[[str, dict], None]) -> None:
        """Register a callback for LCM lifecycle events.

        The callback receives (event_type: str, data: dict). It is called
        synchronously from within compress() — keep it fast and non-blocking.

        Event types:
            session_bound       — new session started
            message_ingested    — message persisted to MessageStore
            compaction_start    — compress() beginning
            node_added          — new DAG leaf node created
            node_condensed      — DAG nodes merged into higher depth
            compaction_end      — compress() finished (includes before/after stats)
            token_pressure      — token usage updated after LLM response
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, dict], None]) -> None:
        self._listeners = [cb for cb in self._listeners if cb is not callback]

    def _emit(self, event_type: str, data: dict) -> None:
        for cb in self._listeners:
            try:
                cb(event_type, data)
            except Exception:
                pass

    # ── Session lifecycle ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_db_path(db_path: str | Path = "") -> Path:
        if db_path:
            return Path(db_path).expanduser().resolve()
        return Path.home() / ".openlcm" / "lcm.db"

    def bind_session(
        self,
        session_id: str,
        *,
        platform: str = "",
        conversation_id: str = "",
        context_length: int = 0,
    ) -> None:
        """Bind the engine to a new or existing session.

        Call this at the start of each conversation or agent run. Multiple calls
        with the same session_id are safe (cursor reconciliation handles restarts).

        Args:
            session_id: Unique identifier for this session.
            platform: Optional platform string (e.g. "langgraph", "crewai").
            conversation_id: Optional stable conversation/thread ID.
            context_length: Model's context window size in tokens.
        """
        previous = self._session_id
        if previous and previous != session_id:
            self._reset_session_runtime_state()

        self._session_id = session_id
        self._session_platform = platform
        self._refresh_session_filters()

        if context_length > 0:
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self._config.context_threshold)

        lifecycle = self._lifecycle.bind_session(session_id, conversation_id=conversation_id or None)
        self._conversation_id = lifecycle.conversation_id
        self._last_compacted_store_id = lifecycle.current_frontier_store_id

        if not self._session_ignored and not self._session_stateless:
            existing_count = self._store.get_session_count(session_id)
            self._ingest_cursor_needs_reconcile = existing_count > 0

        self._emit("session_bound", {
            "session_id": session_id,
            "platform": platform,
            "conversation_id": self._conversation_id,
            "context_length": self.context_length,
        })
        logger.info("LCM bound session %s (platform=%s)", session_id, platform or "unknown")

    def end_session(self, messages: List[Dict[str, Any]] | None = None) -> None:
        """Flush remaining messages and finalize the session."""
        if not self._session_id:
            return
        if messages and not self._session_ignored and not self._session_stateless:
            try:
                self._ingest_messages(messages)
            except Exception as exc:
                logger.warning("LCM session-end ingest failed: %s", exc)
        self._lifecycle.finalize_session(
            self._conversation_id,
            self._session_id,
            frontier_store_id=self._last_compacted_store_id,
        )

    def _reset_session_runtime_state(self) -> None:
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation = {"action": "none", "reason": "not run"}
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""

    # ── Token tracking ─────────────────────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Call after each LLM response to update token pressure metrics."""
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)
        self.last_input_tokens = int(usage.get("input_tokens", self.last_prompt_tokens) or 0)
        self.last_output_tokens = int(usage.get("output_tokens", self.last_completion_tokens) or 0)
        self.last_cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
        self.last_cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)

        if self.context_length > 0 and self.last_prompt_tokens > 0:
            ratio = self.last_prompt_tokens / self.context_length
            self._emit("token_pressure", {
                "prompt_tokens": self.last_prompt_tokens,
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "ratio": round(ratio, 4),
            })

    def set_context_length(self, context_length: int) -> None:
        """Update the model's context window size (triggers threshold recalculation)."""
        if context_length > 0:
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self._config.context_threshold)

    # ── Compaction decision ────────────────────────────────────────────────

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        """Return True if compaction should run now."""
        if self._session_ignored or self._session_stateless:
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Pre-flight check that also ingests messages into the store."""
        if self._session_ignored or self._session_stateless:
            return False
        if self._session_id and messages:
            try:
                self._ingest_messages(messages)
            except Exception as exc:
                logger.warning("LCM preflight ingest failed: %s", exc)
        rough = count_messages_tokens(messages)
        # Overflow recovery takes priority — always compress when over cap
        if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
            return True
        # Deferred maintenance — catch up on recorded backlog debt
        if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
            return True
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            eligible, _ = self._leaf_compaction_candidate_status(messages)
            return eligible
        return False

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[bool, str]:
        if not messages:
            return False, "empty message list"
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"
        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail"
        raw_tokens = count_messages_tokens(candidate_raw)
        if raw_tokens < self._config.leaf_chunk_tokens:
            return False, "raw backlog below leaf chunk threshold"
        return True, "eligible"

    # ── Main compaction entry point ────────────────────────────────────────

    async def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str = "",
    ) -> List[Dict[str, Any]]:
        """Lossless context compaction with overflow recovery and deferred maintenance.

        Ingests any new messages, summarizes old ones into DAG leaf nodes,
        condenses nodes when enough accumulate, and assembles the new active
        context as: [system prompt] + [summaries] + [fresh tail].

        Args:
            messages: Current full message list from the agent.
            current_tokens: Observed prompt token count (uses last_prompt_tokens if None).
            focus_topic: Optional topic hint to guide summarization focus.

        Returns:
            Compressed message list to replace the input with.
        """
        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        if not self._session_id:
            raise RuntimeError(
                "LCMEngine.compress() called before bind_session(). "
                "Call engine.bind_session('my-session', context_length=N) first."
            )

        if self._session_ignored or self._session_stateless:
            return messages

        observed_prompt_tokens = (
            current_tokens or self.last_prompt_tokens or count_messages_tokens(messages)
        )
        tokens_before = observed_prompt_tokens

        # Determine overflow state before ingesting (signals are pre-ingest)
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        self._emit("compaction_start", {
            "session_id": self._session_id,
            "messages_count": len(messages),
            "prompt_tokens": tokens_before,
        })
        self._last_compression_status = "running"

        # Step 1: Ingest new messages
        working_messages = self._ingest_messages(messages)

        # Overflow recovery: forced convergence when context already exceeds cap
        if force_overflow:
            leading_anchor_count = self._leading_anchor_count(working_messages)
            compressed = self._assemble_overflow_recovery_context(
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
            )
            return self._finalize_forced_overflow_result(
                working_messages,
                compressed,
                assembly_cap_override=recovery_assembly_cap,
            )

        # Deferred maintenance: run extra leaf passes when backlog debt is recorded
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = self._should_run_deferred_maintenance(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )

        leaf_compacted = False
        leaf_passes = 0
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)
        elif self._config.dynamic_leaf_chunk_enabled:
            max_leaf_passes = 4
        else:
            max_leaf_passes = 1
        noop_reason = "no eligible raw backlog outside fresh tail"

        # Step 2-5: Leaf compaction loop
        while leaf_passes < max_leaf_passes:
            n = len(working_messages)
            fresh_tail_start = max(0, n - self._config.fresh_tail_count)
            leading_anchor_count = self._leading_anchor_count(working_messages)

            if fresh_tail_start <= leading_anchor_count:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            raw_tokens = count_messages_tokens(candidate_raw)
            working_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens)

            if raw_tokens < working_chunk_tokens:
                noop_reason = "raw backlog below leaf chunk threshold"
                # Deferred maintenance under critical pressure pushes through anyway
                if not (deferred_maintenance_active and critical_budget_pressure):
                    break

            to_compact = (
                candidate_raw if not self._config.dynamic_leaf_chunk_enabled
                else self._select_oldest_leaf_chunk(candidate_raw, working_chunk_tokens)
            )
            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            # Pre-compaction extraction (best-effort)
            if self._config.extraction_enabled:
                self._run_pre_compaction_extraction(to_compact)

            # Summarize with rescue
            compacted_chunk, source_tokens, summary_text, _level, _attempts = \
                await self._summarize_leaf_chunk_with_rescue(to_compact, focus_topic=focus_topic)

            source_store_ids = self._get_store_ids_for_messages(compacted_chunk)
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            self._last_compacted_store_id = max(source_store_ids) if source_store_ids else 0
            self._persist_frontier_marker()

            self._emit("node_added", {
                "node_id": node.node_id,
                "depth": 0,
                "token_count": summary_tokens,
                "source_token_count": source_tokens,
                "source_ids_count": len(source_store_ids),
            })

            # Trim compacted messages from working set
            remaining = working_messages[leading_anchor_count + len(compacted_chunk):]
            working_messages = working_messages[:leading_anchor_count] + remaining
            leaf_compacted = True
            leaf_passes += 1

            if not self._config.dynamic_leaf_chunk_enabled and not deferred_maintenance_active:
                break

        # Persist or clear backlog debt based on remaining work
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )

        if not leaf_compacted:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = noop_reason
            logger.info("LCM compression no-op: %s", noop_reason)
            return working_messages

        # Step 6: Condense DAG nodes (cache-friendly suppression aware)
        await self._maybe_condense(
            focus_topic=focus_topic,
            leaf_compacted_this_turn=True,
            force_overflow=False,
            critical_budget_pressure=critical_budget_pressure,
        )

        # Step 7: Assemble new active context
        leading_anchor_count = self._leading_anchor_count(working_messages)
        compressed = self._assemble_context(
            working_messages[0] if leading_anchor_count else None,
            working_messages[leading_anchor_count:],
            assembly_cap_override=recovery_assembly_cap,
        )

        self.compression_count += 1
        self._last_compression_status = "compacted"
        self._ingest_cursor = len(compressed)

        tokens_after = count_messages_tokens(compressed)
        logger.info(
            "LCM compaction #%d: %d msgs → %d, %d→%d tokens, %d leaf pass(es), %d DAG nodes",
            self.compression_count,
            len(messages),
            len(compressed),
            tokens_before,
            tokens_after,
            leaf_passes,
            len(self._dag.get_session_nodes(self._session_id)),
        )
        self._emit("compaction_end", {
            "session_id": self._session_id,
            "messages_before": len(messages),
            "messages_after": len(compressed),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "compression_count": self.compression_count,
            "dag_nodes": len(self._dag.get_session_nodes(self._session_id)),
            "leaf_passes": leaf_passes,
        })
        return compressed

    # ── Leaf compaction helpers ────────────────────────────────────────────

    @staticmethod
    def _leading_anchor_count(messages: List[Dict[str, Any]]) -> int:
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return 1
        return 0

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    async def _summarize_leaf_chunk_with_rescue(
        self,
        initial_chunk: List[Dict[str, Any]],
        focus_topic: str = "",
    ) -> tuple[List[Dict[str, Any]], int, str, int, int]:
        attempt_chunk = list(initial_chunk)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            source_tokens = count_messages_tokens(attempt_chunk)
            serialized = self._serialize_messages(attempt_chunk)
            token_budget = max(2000, int(source_tokens * 0.20))
            token_budget = min(token_budget, 12000)
            try:
                summary_text, level = await summarize_with_escalation(
                    text=serialized,
                    source_tokens=source_tokens,
                    token_budget=token_budget,
                    backend=self._backend,
                    depth=0,
                    model=self._config.summary_model,
                    fallback_models=self._config.summary_fallback_models,
                    circuit_breaker=self._summary_circuit_breaker,
                    timeout=self._config.summary_timeout_ms / 1000,
                    l2_budget_ratio=self._config.l2_budget_ratio,
                    l3_truncate_tokens=self._config.l3_truncate_tokens,
                    focus_topic=focus_topic,
                    custom_instructions=self._config.custom_instructions,
                )
                return attempt_chunk, source_tokens, summary_text, level, attempt
            except Exception as exc:
                if attempt >= max_attempts:
                    raise
                error_msg = str(exc).lower()
                retryable = any(m in error_msg for m in (
                    "context length", "maximum context", "too many tokens",
                    "token limit", "prompt is too long", "timed out", "timeout",
                ))
                if not retryable:
                    raise
                smaller = attempt_chunk[:-max(1, len(attempt_chunk) // 4)]
                if not smaller:
                    raise
                logger.warning("LCM leaf summary retry with smaller chunk: %d→%d msgs", len(attempt_chunk), len(smaller))
                attempt_chunk = smaller
        raise RuntimeError("leaf rescue exhausted")

    # ── Overflow recovery ──────────────────────────────────────────────────

    def _effective_assembly_token_cap(self) -> Optional[int]:
        """Compute the active assembly cap from config knobs.

        Two settings can constrain the assembled context:
          * max_assembly_tokens  — explicit hard cap
          * reserve_tokens_floor — headroom kept inside context_length
        Returns None when both are disabled (no cap enforced).
        """
        caps: list[int] = []
        if self._config.max_assembly_tokens > 0:
            caps.append(self._config.max_assembly_tokens)
        if self.context_length > 0 and self._config.reserve_tokens_floor > 0:
            reserve_cap = self.context_length - self._config.reserve_tokens_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
            else:
                logger.warning(
                    "LCM reserve_tokens_floor=%d disables reserve-based cap because context_length=%d",
                    self._config.reserve_tokens_floor,
                    self.context_length,
                )
        return max(1, min(caps)) if caps else None

    def _overflow_recovery_signal_tokens(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        candidates: list[int] = []
        if observed_tokens is not None and observed_tokens > 0:
            candidates.append(observed_tokens)
        if messages is not None:
            candidates.append(count_messages_tokens(messages))
        return max(candidates) if candidates else None

    def _should_force_overflow_recovery(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Return True when observed context size meets or exceeds the assembly cap."""
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return False
        tokens = self._overflow_recovery_signal_tokens(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if tokens is None:
            return False
        return tokens >= assembly_cap

    def _overflow_recovery_assembly_cap(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """Return a cap adjusted for prompt overhead (tokens outside the message list)."""
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return None
        if messages is None or observed_tokens is None or observed_tokens <= 0:
            return assembly_cap
        message_tokens = count_messages_tokens(messages)
        overhead_tokens = max(0, observed_tokens - message_tokens)
        return max(1, assembly_cap - overhead_tokens)

    def _assemble_overflow_recovery_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Build minimal context under assembly cap for overflow recovery."""
        if tail_messages:
            first = tail_messages[0]
            role = first.get("role") or ""
            content = first.get("content") or ""
            if role == "assistant" and self._looks_like_active_summary_blob(content):
                # Try re-assembling without the stale summary blob
                candidate = self._assemble_context(
                    system_msg,
                    tail_messages[1:],
                    assembly_cap_override=assembly_cap_override,
                )
                if any(
                    (msg.get("content") or "") == content
                    for msg in (candidate[1:] if system_msg is not None else candidate)
                ):
                    return candidate

        candidate = self._assemble_context(
            system_msg,
            tail_messages,
            assembly_cap_override=assembly_cap_override,
        )
        min_len = 1 if system_msg is not None else 0
        if len(candidate) == min_len and tail_messages:
            # Absolute fallback: keep system + last message only
            fallback = ([system_msg] if system_msg is not None else []) + [tail_messages[-1]]
            return self._sanitize_active_context_messages(fallback)
        return candidate

    def _finalize_forced_overflow_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if compressed != original_messages:
            self._last_compression_status = "overflow_recovery"
            self._last_compression_noop_reason = ""
            self._ingest_cursor = len(compressed)
            self._ingest_cursor_needs_reconcile = False
            logger.info(
                "LCM overflow recovery: %d messages → %d (no new summary node)",
                len(original_messages),
                len(compressed),
            )
        else:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = (
                "overflow recovery found no droppable active-context messages"
            )

        effective_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )
        if effective_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            result_tokens = count_messages_tokens(compressed)
            self._last_overflow_recovery_failed = result_tokens > effective_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not reach cap=%d; returning best-effort (%d tokens)",
                    effective_cap,
                    result_tokens,
                )
        return compressed

    @staticmethod
    def _looks_like_active_summary_blob(content: str) -> bool:
        """Return True if content looks like a previously assembled LCM scaffold."""
        if not isinstance(content, str) or not content:
            return False
        block = (
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]\n"
            r".*?\n"
            r"\[(?:Expand for details.*?)\]"
        )
        pattern = rf"^{block}(?:\n\n---\n\n{block})*$"
        return re.fullmatch(pattern, content, flags=re.DOTALL) is not None

    # ── Deferred maintenance ───────────────────────────────────────────────

    def _budget_pressure_ratio(
        self,
        *,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[float]:
        if self.context_length <= 0:
            return None
        token_count: Optional[int] = None
        if observed_tokens is not None and observed_tokens > 0:
            token_count = observed_tokens
        elif messages is not None:
            token_count = count_messages_tokens(messages)
        elif self.last_prompt_tokens > 0:
            token_count = self.last_prompt_tokens
        if token_count is None or token_count <= 0:
            return None
        return token_count / self.context_length

    def _critical_budget_pressure_reached(
        self,
        *,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        threshold = self._config.critical_budget_pressure_ratio
        if threshold <= 0:
            return False
        pressure = self._budget_pressure_ratio(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        return pressure is not None and pressure >= threshold

    def _raw_backlog_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)
        leading = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading:
            return []
        return messages[leading:fresh_tail_start]

    def _raw_backlog_tokens(self, messages: List[Dict[str, Any]]) -> int:
        backlog = self._raw_backlog_messages(messages)
        return count_messages_tokens(backlog) if backlog else 0

    def _raw_backlog_threshold(self, raw_tokens: int) -> int:
        if self._config.dynamic_leaf_chunk_enabled:
            return self._working_leaf_chunk_tokens(raw_tokens)
        return max(1, self._config.leaf_chunk_tokens)

    def _has_raw_backlog_debt(self) -> bool:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return False
        state = self._lifecycle.get_by_conversation(self._conversation_id)
        return bool(state and state.debt_kind == "raw_backlog" and state.debt_size_estimate > 0)

    def _should_run_deferred_maintenance(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: Optional[int] = None,
    ) -> bool:
        if not self._has_raw_backlog_debt():
            return False
        raw_tokens = self._raw_backlog_tokens(messages)
        if raw_tokens <= 0:
            return False
        if raw_tokens >= self._raw_backlog_threshold(raw_tokens):
            return True
        return self._critical_budget_pressure_reached(
            observed_tokens=observed_tokens,
            messages=messages,
        )

    def _refresh_raw_backlog_debt(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: Optional[int] = None,
    ) -> None:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return
        raw_tokens = self._raw_backlog_tokens(messages)
        threshold = self._raw_backlog_threshold(raw_tokens) if raw_tokens > 0 else 0
        keep = (
            raw_tokens > 0
            and self._has_raw_backlog_debt()
            and self._critical_budget_pressure_reached(
                observed_tokens=observed_tokens,
                messages=messages,
            )
        )
        if raw_tokens > 0 and (raw_tokens >= threshold or keep):
            self._lifecycle.record_debt(
                self._conversation_id,
                kind="raw_backlog",
                size_estimate=raw_tokens,
            )
        elif self._has_raw_backlog_debt():
            self._lifecycle.clear_debt(self._conversation_id)

    # ── Active context sanitization ────────────────────────────────────────

    @staticmethod
    def _structured_part_text(part: Dict[str, Any]) -> str:
        for key in ("text", "content", "value"):
            value = part.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for nested_key in ("value", "content"):
                    nested = value.get(nested_key)
                    if isinstance(nested, str):
                        return nested
        return ""

    @classmethod
    def _structured_part_has_visible_assistant_content(cls, part: Any) -> bool:
        if part is None:
            return False
        if isinstance(part, str):
            return bool(_strip_reasoning_blocks(part).strip())
        if not isinstance(part, dict):
            return bool(str(part).strip())
        part_type = str(part.get("type") or "").strip().lower()
        if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
            return False
        if part_type in _VISIBLE_TEXT_PART_TYPES:
            return bool(_strip_reasoning_blocks(cls._structured_part_text(part)).strip())
        return True  # Unknown non-internal block — preserve conservatively

    @classmethod
    def _strip_structured_text_part(cls, part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cleaned = dict(part)
        for key in ("text", "content", "value"):
            value = cleaned.get(key)
            if isinstance(value, str):
                stripped = _strip_reasoning_blocks(value)
                if not stripped.strip():
                    return None
                cleaned[key] = stripped
                return cleaned
            if isinstance(value, dict):
                nested = dict(value)
                for nested_key in ("value", "content", "text"):
                    nested_value = nested.get(nested_key)
                    if isinstance(nested_value, str):
                        stripped = _strip_reasoning_blocks(nested_value)
                        if not stripped.strip():
                            return None
                        nested[nested_key] = stripped
                        cleaned[key] = nested
                        return cleaned
        return cleaned if cls._structured_part_has_visible_assistant_content(cleaned) else None

    @classmethod
    def _sanitize_active_assistant_content(cls, content: Any) -> Any:
        if content is None:
            return None
        if isinstance(content, str):
            stripped = _strip_reasoning_blocks(content)
            return stripped if stripped.strip() else None
        if isinstance(content, list):
            cleaned_parts: list[Any] = []
            for part in content:
                if isinstance(part, str):
                    stripped = _strip_reasoning_blocks(part)
                    if stripped.strip():
                        cleaned_parts.append(stripped)
                    continue
                if isinstance(part, dict):
                    part_type = str(part.get("type") or "").strip().lower()
                    if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
                        continue
                    if part_type in _VISIBLE_TEXT_PART_TYPES:
                        cleaned = cls._strip_structured_text_part(part)
                        if cleaned is not None:
                            cleaned_parts.append(cleaned)
                        continue
                if cls._structured_part_has_visible_assistant_content(part):
                    cleaned_parts.append(part)
            return cleaned_parts or None
        if isinstance(content, dict):
            part_type = str(content.get("type") or "").strip().lower()
            if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
                return None
            if part_type in _VISIBLE_TEXT_PART_TYPES:
                return cls._strip_structured_text_part(content)
            return content if cls._structured_part_has_visible_assistant_content(content) else None
        return content if str(content).strip() else None

    @classmethod
    def _clean_active_assistant_message(
        cls, msg: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if msg.get("role") != "assistant":
            return msg
        if "content" not in msg:
            return msg
        cleaned_content = cls._sanitize_active_assistant_content(msg.get("content"))
        if cleaned_content is None:
            if not msg.get("tool_calls"):
                return None
            cleaned_content = ""
        if cleaned_content == msg.get("content"):
            return msg
        cleaned = dict(msg)
        cleaned["content"] = cleaned_content
        return cleaned

    def _sanitize_active_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Drop empty/internal assistant messages, repair orphaned tool pairs."""
        cleaned: list[Dict[str, Any]] = []
        dropped = 0
        stripped = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                cleaned_msg = self._clean_active_assistant_message(msg)
                if cleaned_msg is None:
                    dropped += 1
                    continue
                if cleaned_msg is not msg:
                    stripped += 1
                cleaned.append(cleaned_msg)
            else:
                cleaned.append(msg)

        if dropped:
            logger.info("LCM sanitize: dropped %d empty assistant message(s)", dropped)
        if stripped:
            logger.info("LCM sanitize: stripped internal content from %d assistant message(s)", stripped)

        return self._sanitize_tool_pairs(cleaned, insert_missing_tool_stubs=insert_missing_tool_stubs)

    def _sanitize_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ensure every tool_call has a contiguous tool_result; drop/stub orphans.

        Providers (Anthropic, OpenAI) require that assistant tool-call messages
        be immediately followed by their matching tool-result messages. After
        context compression, tool-result messages can become separated from
        their call or disappear entirely. This guard:
          - drops tool-result messages that appear before any assistant call
          - drops late/duplicate/out-of-order results
          - inserts a synthetic stub result when the matching result is missing
        The raw SQLite store and DAG are never modified.
        """
        sanitized: list[Dict[str, Any]] = []
        dropped_results = 0
        inserted_stubs = 0

        i = 0
        while i < len(messages):
            msg = messages[i]

            # Orphan tool-result with no preceding assistant call — drop it
            if msg.get("role") == "tool":
                dropped_results += 1
                i += 1
                continue

            sanitized.append(msg)

            if msg.get("role") == "assistant":
                expected_ids = [
                    call_id
                    for call_id in (
                        _tool_call_id(tc) for tc in (msg.get("tool_calls") or [])
                    )
                    if call_id
                ]

                for expected_id in expected_ids:
                    matched = False
                    while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                        next_msg = messages[i + 1]
                        next_id = str(next_msg.get("tool_call_id") or "").strip()
                        if next_id == expected_id:
                            sanitized.append(next_msg)
                            i += 1
                            matched = True
                            break
                        dropped_results += 1
                        i += 1

                    if not matched and insert_missing_tool_stubs:
                        sanitized.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": expected_id,
                        })
                        inserted_stubs += 1

                # Drain any remaining tool messages that have no matching call
                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    dropped_results += 1
                    i += 1

            i += 1

        if dropped_results:
            logger.info("LCM tool-pair guardrail: dropped %d orphan/late tool result(s)", dropped_results)
        if inserted_stubs:
            logger.info("LCM tool-pair guardrail: inserted %d missing tool-result stub(s)", inserted_stubs)

        return sanitized

    # ── Cache-friendly condensation ────────────────────────────────────────

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
        critical_budget_pressure: bool = False,
    ) -> tuple[bool, str]:
        """Return (allow, reason) for condensation at a given depth."""
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._config.cache_friendly_condensation_enabled:
            return True, ""
        if force_overflow:
            return True, ""
        if critical_budget_pressure:
            return True, ""

        fanin = max(1, self._config.condensation_fanin)
        debt_threshold = fanin * max(1, self._config.cache_friendly_min_debt_groups)
        if uncondensed_count >= debt_threshold:
            return True, ""
        if uncondensed_count == fanin:
            return False, "cache_friendly_single_group"
        return False, "cache_friendly_low_debt"

    def _run_pre_compaction_extraction(self, messages: List[Dict[str, Any]]) -> None:
        try:
            serialized = self._serialize_messages(messages)
            output_path = (
                self._config.extraction_output_path
                or str(Path.home() / ".openlcm" / "extractions")
            )
            extract_before_compaction(
                serialized,
                output_path=output_path,
                session_id=self._session_id,
                model=self._config.extraction_model or self._config.summary_model,
                timeout=self._config.summary_timeout_ms / 1000,
                backend=self._backend,
            )
        except Exception as exc:
            logger.debug("Pre-compaction extraction failed (non-blocking): %s", exc)

    # ── DAG condensation ───────────────────────────────────────────────────

    async def _maybe_condense(
        self,
        focus_topic: str = "",
        *,
        leaf_compacted_this_turn: bool = False,
        force_overflow: bool = False,
        critical_budget_pressure: bool = False,
    ) -> None:
        """Condense DAG nodes upward, respecting cache-friendly suppression."""
        self._last_condensation_suppressed_reason = ""

        max_depth = self._config.incremental_max_depth
        if max_depth == 0:
            return

        # -1 means unlimited: derive upper bound from deepest existing node + 1
        if max_depth < 0:
            all_nodes = self._dag.get_session_nodes(self._session_id)
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        suppression_reason = ""
        for depth in range(upper):
            uncondensed = self._dag.get_uncondensed_at_depth(self._session_id, depth)
            if len(uncondensed) < self._config.condensation_fanin:
                continue

            allow, reason = self._should_allow_follow_on_condensation(
                uncondensed_count=len(uncondensed),
                leaf_compacted_this_turn=leaf_compacted_this_turn,
                force_overflow=force_overflow,
                critical_budget_pressure=critical_budget_pressure,
            )
            if not allow:
                suppression_reason = reason or suppression_reason
                continue

            # Condense in fanin-sized groups, oldest first
            to_condense = uncondensed[:self._config.condensation_fanin]
            await self._condense_nodes(to_condense, target_depth=depth + 1, focus_topic=focus_topic)

        if suppression_reason:
            self._last_condensation_suppressed_reason = suppression_reason

    async def _condense_nodes(
        self,
        nodes: List[SummaryNode],
        target_depth: int,
        focus_topic: str = "",
    ) -> None:
        combined_text = "\n\n".join(n.summary for n in nodes)
        source_tokens = sum(n.token_count for n in nodes)
        token_budget = max(2000, int(source_tokens * 0.40))
        token_budget = min(token_budget, 16000)

        try:
            summary_text, _level = await summarize_with_escalation(
                text=combined_text,
                source_tokens=source_tokens,
                token_budget=token_budget,
                backend=self._backend,
                depth=target_depth,
                model=self._config.summary_model,
                fallback_models=self._config.summary_fallback_models,
                circuit_breaker=self._summary_circuit_breaker,
                timeout=self._config.summary_timeout_ms / 1000,
                focus_topic=focus_topic,
                custom_instructions=self._config.custom_instructions,
            )
        except Exception as exc:
            logger.warning("LCM condensation at depth %d failed: %s", target_depth, exc)
            self._last_condensation_suppressed_reason = str(exc)
            return

        node_ids = [n.node_id for n in nodes]
        earliest_at = min((n.earliest_at or n.created_at for n in nodes), default=None)
        latest_at = max((n.latest_at or n.created_at for n in nodes), default=None)

        condensed = SummaryNode(
            session_id=self._session_id,
            depth=target_depth,
            summary=summary_text,
            token_count=count_tokens(summary_text),
            source_token_count=source_tokens,
            source_ids=node_ids,
            source_type="nodes",
            created_at=time.time(),
            earliest_at=earliest_at,
            latest_at=latest_at,
            expand_hint=self._extract_expand_hint(summary_text),
        )
        self._dag.add_node(condensed)
        self._emit("node_condensed", {
            "depth": target_depth,
            "input_node_ids": node_ids,
            "output_node_id": condensed.node_id,
            "token_count": condensed.token_count,
            "source_token_count": source_tokens,
        })

    # ── Context assembly ───────────────────────────────────────────────────

    def _assemble_context(
        self,
        system_message: Optional[Dict[str, Any]],
        remaining_messages: List[Dict[str, Any]],
        *,
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Build active context: [system] + [summaries] + [fresh tail].

        When assembly_cap_override is set, the fresh tail is trimmed from the
        oldest end until the assembled result fits under the cap.
        """
        from collections import defaultdict

        session_id = self._session_id
        n = len(remaining_messages)
        fresh_tail_count = self._config.fresh_tail_count

        # Determine effective assembly cap
        cap = assembly_cap_override if assembly_cap_override is not None else self._effective_assembly_token_cap()

        dag_nodes = self._dag.get_session_nodes(session_id)

        if not dag_nodes:
            result: list[Dict[str, Any]] = []
            if system_message:
                result.append(system_message)
            result.extend(remaining_messages)
            result = self._sanitize_active_context_messages(result)
            if cap is not None:
                result = self._trim_to_cap(result, cap, system_message)
            return result

        # Group nodes by depth, highest first
        by_depth: dict[int, list[SummaryNode]] = defaultdict(list)
        for node in dag_nodes:
            by_depth[node.depth].append(node)

        # Build LCM scaffold message
        scaffold_parts: list[str] = [
            "[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep, lcm_expand, or lcm_expand_query to recall specifics.]\n"
        ]
        max_dag_depth = max(by_depth.keys())
        for depth in range(max_dag_depth, -1, -1):
            nodes_at_depth = sorted(by_depth.get(depth, []), key=lambda nd: nd.created_at)
            depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(depth, f"Depth-{depth}")
            for node in nodes_at_depth:
                scaffold_parts.append(
                    f"\n[{depth_label} Summary (d{depth}, node {node.node_id})]"
                    f"\n{node.summary}"
                    f"\n[{node.expand_hint or 'Expand for details'}]"
                )
        scaffold_content = "\n".join(scaffold_parts)

        fresh_tail_start = max(0, n - fresh_tail_count)
        fresh_tail = remaining_messages[fresh_tail_start:]

        result = []
        if system_message:
            result.append(system_message)
        result.append({"role": "user", "content": scaffold_content})
        result.append({"role": "assistant", "content": "Understood. I have access to the full conversation history through LCM tools."})
        result.extend(fresh_tail)

        result = self._sanitize_active_context_messages(result)

        if cap is not None:
            result = self._trim_to_cap(result, cap, system_message)

        return result

    def _trim_to_cap(
        self,
        messages: List[Dict[str, Any]],
        cap: int,
        system_message: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Trim messages from the oldest non-anchor position until under cap."""
        if count_messages_tokens(messages) <= cap:
            return messages

        # Identify the first non-anchor index (after system + scaffold pair)
        anchor_count = 0
        if system_message:
            anchor_count += 1
        # scaffold user + assistant stub are also protected anchors
        if len(messages) > anchor_count and messages[anchor_count].get("role") == "user":
            anchor_count += 1
        if len(messages) > anchor_count and messages[anchor_count].get("role") == "assistant":
            content = messages[anchor_count].get("content") or ""
            if isinstance(content, str) and "LCM tools" in content:
                anchor_count += 1

        result = list(messages)
        while len(result) > anchor_count + 1 and count_messages_tokens(result) > cap:
            result.pop(anchor_count)

        return result

    # ── Message store helpers ──────────────────────────────────────────────

    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist new messages, return replay-safe copy."""
        if not self._session_id or self._session_ignored or self._session_stateless:
            return list(messages)

        n = len(messages)
        cursor = min(max(self._ingest_cursor, 0), n)

        if self._ingest_cursor_needs_reconcile and self._session_id:
            try:
                session_count = self._store.get_session_count(self._session_id)
                if session_count > 0:
                    tail_rows = self._store.get_session_tail(self._session_id, limit=n * 2)
                    # Simple reconciliation: if incoming message count matches store, advance cursor
                    if len(tail_rows) >= n:
                        cursor = n
                    elif tail_rows:
                        cursor = max(0, n - len(tail_rows))
            except Exception as exc:
                logger.debug("LCM cursor reconciliation failed: %s", exc)
            self._ingest_cursor_needs_reconcile = False

        replay_messages = list(messages)
        new_messages = messages[cursor:]
        if not new_messages:
            self._ingest_cursor = n
            return replay_messages

        for msg in new_messages:
            if self._compiled_ignore_message_patterns:
                text = text_content_for_pattern_matching(msg.get("content")) or ""
                if matches_message_pattern(text, self._compiled_ignore_message_patterns):
                    self._ignored_message_count += 1
                    logger.debug("LCM ignore_message_patterns dropped %s message", msg.get("role", "unknown"))
                    continue

            try:
                store_id = self._store.append(
                    self._session_id,
                    msg,
                    token_estimate=count_message_tokens(msg),
                )
                self._emit("message_ingested", {
                    "store_id": store_id,
                    "role": msg.get("role", "unknown"),
                    "token_estimate": count_message_tokens(msg),
                })
            except Exception as exc:
                logger.warning("LCM message ingest failed: %s", exc)

        self._ingest_cursor = n
        return replay_messages

    def _get_store_ids_for_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[int]:
        """Map a slice of messages to their store_ids."""
        if not self._session_id:
            return []
        try:
            tail = self._store.get_session_tail(self._session_id, limit=len(messages) * 2 + 10)
            result_ids: list[int] = []
            for msg in messages:
                role = msg.get("role", "")
                content = normalize_content_value(msg.get("content")) or ""
                for row in tail:
                    if row.get("role") == role:
                        stored_content = normalize_content_value(row.get("content")) or ""
                        if stored_content == content or stored_content.startswith(content[:80]):
                            sid = row.get("store_id")
                            if sid and sid not in result_ids:
                                result_ids.append(int(sid))
                                break
            return result_ids
        except Exception:
            return []

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        try:
            self._lifecycle.advance_frontier(
                self._conversation_id,
                self._session_id,
                self._last_compacted_store_id,
            )
        except Exception as exc:
            logger.debug("LCM frontier persist failed: %s", exc)

    # ── Serialization ──────────────────────────────────────────────────────

    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize a list of messages to text for summarization."""
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "unknown")).upper()
            content = normalize_content_value(msg.get("content")) or ""
            content = sanitize_pre_compaction_content(content)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_text = sanitize_pre_compaction_tool_arguments(tool_calls)
                parts.append(f"{role}: [tool_calls] {tc_text}")
            elif content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    # ── DAG helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_expand_hint(summary_text: str) -> str:
        match = re.search(r"Expand for details(?:\s+about)?:?\s*(.{0,120})", summary_text, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:160]
        return "Expand for details"

    # ── Session filters ────────────────────────────────────────────────────

    def _refresh_session_filters(self) -> None:
        self._session_match_keys = build_session_match_keys(
            self._session_id,
            platform=self._session_platform,
        )
        self._session_ignored = matches_session_pattern(
            self._session_match_keys,
            self._compiled_ignore_session_patterns,
        )
        self._session_stateless = (
            not self._session_ignored
            and matches_session_pattern(
                self._session_match_keys,
                self._compiled_stateless_session_patterns,
            )
        )

    # ── Status & tools ─────────────────────────────────────────────────────

    def get_runtime_identity(self) -> Dict[str, Any]:
        """Return identity info for lcm_status / lcm_doctor."""
        return {
            "engine": "openlcm",
            "version": "0.1.0",
            "session_id": self._session_id,
            "database_path": str(self._store.db_path),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return a status dict for lcm_status and CLI display."""
        session_id = self._session_id
        dag_nodes = self._dag.get_session_nodes(session_id) if session_id else []
        by_depth: dict[int, int] = {}
        for node in dag_nodes:
            by_depth[node.depth] = by_depth.get(node.depth, 0) + 1

        source_lineage = {}
        try:
            source_lineage = self._store.get_source_stats(session_id or None)
        except Exception:
            pass

        return {
            "engine": "openlcm",
            "session_id": session_id,
            "conversation_id": self._conversation_id,
            "platform": self._session_platform,
            "context_length": self.context_length,
            "threshold_tokens": self.threshold_tokens,
            "threshold_percent": self.threshold_percent,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "last_cache_write_tokens": self.last_cache_write_tokens,
            "compression_count": self.compression_count,
            "last_compression_status": self._last_compression_status,
            "last_compression_noop_reason": self._last_compression_noop_reason,
            "store_messages": self._store.get_session_count(session_id) if session_id else 0,
            "dag_nodes": len(dag_nodes),
            "dag_by_depth": by_depth,
            "session_ignored": self._session_ignored,
            "session_stateless": self._session_stateless,
            "ignore_session_patterns": list(self._config.ignore_session_patterns),
            "stateless_session_patterns": list(self._config.stateless_session_patterns),
            "ignore_message_patterns": list(self._config.ignore_message_patterns),
            "ignore_session_patterns_source": self._config.ignore_session_patterns_source,
            "stateless_session_patterns_source": self._config.stateless_session_patterns_source,
            "ignore_message_patterns_source": self._config.ignore_message_patterns_source,
            "ignored_message_count": self._ignored_message_count,
            "fresh_tail_count": self._config.fresh_tail_count,
            "leaf_chunk_tokens": self._config.leaf_chunk_tokens,
            "summary_model": self._config.summary_model,
            "db_path": str(self._store.db_path),
            "source_lineage": source_lineage,
            "ingest_reconciliation": dict(self._last_ingest_reconciliation),
            "overflow_recovery_failed": self._last_overflow_recovery_failed,
            "condensation_suppressed_reason": self._last_condensation_suppressed_reason,
            "runtime_identity": self.get_runtime_identity(),
            "lifecycle": None,
            "lifecycle_fragmentation": {},
            "ingest_protection": {},
        }

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return LCM tool schemas to register with the agent's tool system."""
        from .tools import get_tool_schemas
        return get_tool_schemas()

    def handle_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        messages: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Dispatch an LCM tool call and return JSON result string."""
        from . import tools as lcm_tools
        if messages and self._session_id and not self._session_ignored and not self._session_stateless:
            try:
                self._ingest_messages(messages)
            except Exception as exc:
                logger.warning("LCM tool-call ingest failed: %s", exc)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_load_session": lcm_tools.lcm_load_session,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_doctor": lcm_tools.lcm_doctor,
            "lcm_remember": lcm_tools.lcm_remember,
            "lcm_recall": lcm_tools.lcm_recall,
            "lcm_forget": lcm_tools.lcm_forget,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    # ── New-session rollover ───────────────────────────────────────────────

    def rollover_session(
        self,
        old_session_id: str,
        new_session_id: str,
        previous_messages: List[Dict[str, Any]] | None = None,
    ) -> int:
        """Complete a /new-style session rollover.

        Flushes the old session, retains configured DAG depth, and binds
        the engine to the new session ID. Returns count of carried-over nodes.
        """
        if previous_messages:
            self.end_session(previous_messages)

        retain = self._config.new_session_retain_depth
        if old_session_id and retain != -1:
            if retain == 0:
                self._dag.delete_session_nodes(old_session_id)
            else:
                self._dag.delete_below_depth(old_session_id, retain)

        self._reset_session_runtime_state()
        self.bind_session(new_session_id)

        if old_session_id and new_session_id and old_session_id != new_session_id:
            moved = self._dag.reassign_session_nodes(old_session_id, new_session_id)
            logger.info("LCM rollover: moved %d nodes from %s → %s", moved, old_session_id, new_session_id)
            return moved
        return 0
