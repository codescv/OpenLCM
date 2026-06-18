"""Three-level summarization escalation.

Level 1 (Normal):     LLM summary preserving details
Level 2 (Aggressive): LLM bullet-point summary at half the token budget
Level 3 (Fallback):   Deterministic truncation — no LLM, guaranteed convergence

Each level checks if Tokens(summary) < Tokens(source). If not, escalates.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from .tokens import count_tokens

if TYPE_CHECKING:
    from openlcm.backends.base import SummaryBackend

logger = logging.getLogger(__name__)


# Strip inline reasoning blocks emitted by thinking models before persisting
# summary text. Without this, reasoning content — which often quotes the
# summarizer system prompt verbatim — gets stored as the summary and later
# confuses lcm_expand_query.
_THINK_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\s*>"
    r".*?"
    r"</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)

_DEFAULT_ROUTE_KEY = "<task-default>"


@dataclass
class SummaryCircuitBreaker:
    """In-process circuit breaker for summary model routes.

    Prevents a hot compression loop from repeatedly hitting a failing backend
    while preserving deterministic L3 truncation as the final fallback.
    """

    failure_threshold: int = 2
    cooldown_seconds: int = 300
    _failures: dict[str, int] = field(default_factory=dict)
    _open_until: dict[str, float] = field(default_factory=dict)

    def _key(self, model: str | None) -> str:
        return (model or "").strip() or _DEFAULT_ROUTE_KEY

    def allows(self, model: str | None, *, now: float | None = None) -> bool:
        key = self._key(model)
        current_time = time.monotonic() if now is None else now
        opened_until = self._open_until.get(key, 0.0)
        if opened_until <= current_time:
            if key in self._open_until:
                self._open_until.pop(key, None)
            return True
        return False

    def record_success(self, model: str | None) -> None:
        key = self._key(model)
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def record_failure(self, model: str | None, *, now: float | None = None) -> None:
        key = self._key(model)
        failures = self._failures.get(key, 0) + 1
        self._failures[key] = failures
        threshold = max(1, int(self.failure_threshold or 1))
        if failures >= threshold:
            current_time = time.monotonic() if now is None else now
            cooldown = max(0, int(self.cooldown_seconds or 0))
            self._open_until[key] = current_time + cooldown
            logger.warning(
                "LCM summary circuit opened for %s after %d failure(s); cooldown=%ss",
                key, failures, cooldown,
            )


def _strip_reasoning_blocks(text: str) -> str:
    """Remove think/thinking/reasoning blocks from text. Idempotent."""
    if not text or "<" not in text:
        return text
    return _THINK_BLOCK_RE.sub("", text)


async def _call_backend_for_summary(
    prompt: str,
    max_tokens: int,
    backend: "SummaryBackend",
    model: str = "",
    timeout: float | None = None,
) -> Optional[str]:
    """Call the SummaryBackend and return stripped text, or None on failure."""
    try:
        result = await backend.summarize(prompt, max_tokens, model=model, timeout=timeout)
        if not result:
            return None
        return _strip_reasoning_blocks(result).strip() or None
    except Exception as exc:
        logger.warning("LCM summary backend call failed: %s", exc)
        return None


def _summary_model_chain(
    primary_model: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    chain: list[str] = []
    for model in [primary_model, *(fallback_models or [])]:
        normalized = (model or "").strip()
        if normalized not in chain:
            chain.append(normalized)
    if not chain:
        chain.append("")
    return chain


async def _invoke_summary_backend_chain(
    prompt: str,
    max_tokens: int,
    backend: "SummaryBackend",
    *,
    model: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
    timeout: float | None = None,
    circuit_breaker: SummaryCircuitBreaker | None = None,
    accepts_result: Callable[[str], bool] | None = None,
) -> Optional[str]:
    chain = _summary_model_chain(model, fallback_models)
    skipped = 0
    for candidate_model in chain:
        if circuit_breaker is not None and not circuit_breaker.allows(candidate_model):
            skipped += 1
            logger.warning("LCM summary route skipped by open circuit: %s", candidate_model or _DEFAULT_ROUTE_KEY)
            continue
        result = await _call_backend_for_summary(prompt, max_tokens, backend, model=candidate_model, timeout=timeout)
        if result and (accepts_result is None or accepts_result(result)):
            if circuit_breaker is not None:
                circuit_breaker.record_success(candidate_model)
            return result
        if circuit_breaker is not None:
            circuit_breaker.record_failure(candidate_model)
    if skipped == len(chain):
        logger.warning("LCM summary chain exhausted: all routes are temporarily open")
    return None


def _normalized_focus_topic(focus_topic: str, max_chars: int = 160) -> str:
    normalized = " ".join(str(focus_topic or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "…"


def _build_l1_focus_brief(focus_topic: str) -> str:
    topic = _normalized_focus_topic(focus_topic)
    if not topic:
        return ""
    return (
        "Focus brief:\n"
        f"Primary focus: {topic}\n"
        "Preserve concrete decisions, constraints, files, commands, identifiers, and current state for this focus.\n"
        "Spend roughly 60-70% of the summary budget on the focus when relevant.\n"
        "Do not discard unrelated blockers or active tasks just because they are off-focus.\n"
    )


def _build_l2_focus_brief(focus_topic: str) -> str:
    topic = _normalized_focus_topic(focus_topic)
    if not topic:
        return ""
    return (
        "Focus brief:\n"
        f"Primary focus: {topic}\n"
        "Prefer bullets that preserve decisions, blockers, files, commands, identifiers, and current state for this focus.\n"
        "Keep other active tasks only when they are current blockers or handoff state.\n"
    )


def _build_l1_prompt(text: str, token_budget: int, depth: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    depth_guidance = {
        0: "Preserve decisions, rationale, constraints, active tasks, file paths, commands, and specific values.",
        1: "Distill into arc-level outcomes: what evolved, what was decided, current state. Drop per-turn detail.",
        2: "Capture durable narrative: decisions in effect, completed milestones, timeline. Drop process detail.",
    }
    guidance = depth_guidance.get(depth, depth_guidance[2])
    focus_guidance = _build_l1_focus_brief(focus_topic)
    custom_block = f"\nAdditional instructions:\n{custom_instructions}\n" if custom_instructions else ""
    return (
        f"You are a summarizer assistant. Below is a conversation segment wrapped in <content> tags:\n\n"
        f"<content>\n{text}\n</content>\n\n"
        f"Please summarize the above conversation segment for future turns.\n"
        f"Instructions:\n"
        f"- {guidance}\n"
        f"- Remove repetition and conversational filler.\n"
        f"- Target ~{token_budget} tokens.\n"
        f"- End the summary with: \"Expand for details about: <what was compressed>\"\n"
        f"{focus_guidance}{custom_block}"
    )


def _build_l2_prompt(text: str, token_budget: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    focus_guidance = _build_l2_focus_brief(focus_topic)
    custom_block = f"\nAdditional instructions:\n{custom_instructions}\n" if custom_instructions else ""
    return (
        f"Below is a conversation segment wrapped in <content> tags:\n\n"
        f"<content>\n{text}\n</content>\n\n"
        f"Please compress the above content into bullet points. Maximum {token_budget} tokens.\n"
        f"Instructions:\n"
        f"- Keep only: decisions made, files changed, errors hit, current state.\n"
        f"- Drop all reasoning, alternatives considered, and process detail.\n"
        f"{focus_guidance}{custom_block}"
    )


def _deterministic_truncate(text: str, max_tokens: int) -> str:
    """Level 3: no LLM, guaranteed convergence.

    Keeps the first 40% and last 40% of the character budget, with a
    visible ellipsis marker in the middle.
    """
    if count_tokens(text) <= max_tokens:
        return text
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    head_budget = int(char_budget * 0.4)
    tail_budget = int(char_budget * 0.4)
    middle = "\n\n[...deterministic truncation — details available via lcm_expand...]\n\n"
    return text[:head_budget] + middle + text[-tail_budget:]


async def summarize_with_escalation(
    text: str,
    source_tokens: int,
    token_budget: int,
    backend: "SummaryBackend",
    depth: int = 0,
    model: str = "",
    timeout: float | None = None,
    l2_budget_ratio: float = 0.50,
    l3_truncate_tokens: int = 512,
    focus_topic: str = "",
    custom_instructions: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
    circuit_breaker: SummaryCircuitBreaker | None = None,
) -> tuple[str, int]:
    """Run 3-level escalation. Returns (summary_text, level_used).

    Guarantees convergence: level 3 is deterministic and always produces
    output shorter than the source.
    """
    # Level 1: detailed summary
    l1_prompt = _build_l1_prompt(text, token_budget, depth,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l1_result = await _invoke_summary_backend_chain(
        l1_prompt,
        token_budget * 2,
        backend,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        circuit_breaker=circuit_breaker,
        accepts_result=lambda result: count_tokens(result) < source_tokens,
    )
    if l1_result:
        logger.debug("L1 summarization succeeded (%d tokens)", count_tokens(l1_result))
        return l1_result, 1

    # Level 2: aggressive bullets at reduced budget
    l2_budget = int(token_budget * l2_budget_ratio)
    l2_prompt = _build_l2_prompt(text, l2_budget,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l2_result = await _invoke_summary_backend_chain(
        l2_prompt,
        l2_budget * 2,
        backend,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        circuit_breaker=circuit_breaker,
        accepts_result=lambda result: count_tokens(result) < source_tokens,
    )
    if l2_result:
        logger.debug("L2 summarization succeeded (%d tokens)", count_tokens(l2_result))
        return l2_result, 2

    # Level 3: deterministic truncation — always converges
    l3_result = _deterministic_truncate(text, l3_truncate_tokens)
    logger.debug("L3 deterministic truncation (%d tokens)", count_tokens(l3_result))
    return l3_result, 3
