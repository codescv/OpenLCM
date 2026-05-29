"""SummaryBackend — abstract interface for LLM-powered summarization.

Users must supply a concrete implementation when constructing LCMEngine.
Three ready-made backends are provided: AnthropicBackend, OpenAIBackend,
and LiteLLMBackend (which covers 100+ providers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SummaryBackend(ABC):
    """Abstract base for all summarization backends.

    A backend receives a plain-text prompt and a token budget and returns
    a summary string (or None on failure). The engine's 3-level escalation
    logic handles retries and fallback to deterministic truncation — the
    backend only needs to make one best-effort call.
    """

    @abstractmethod
    async def summarize(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "",
        timeout: float | None = None,
    ) -> Optional[str]:
        """Call the LLM and return the summary text, or None on failure.

        Args:
            prompt: The full summarization prompt (already built by the engine).
            max_tokens: Maximum tokens the summary should consume.
            model: Optional model override (empty = use backend's default).
            timeout: Optional wall-clock timeout in seconds.

        Returns:
            Summary text, or None if the call failed or timed out.
        """
        ...
