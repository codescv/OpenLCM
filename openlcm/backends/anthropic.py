"""Anthropic SDK backend for LCM summarization.

Install: pip install openlcm[anthropic]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import SummaryBackend

logger = logging.getLogger(__name__)


class AnthropicBackend(SummaryBackend):
    """Summarization backend using the Anthropic SDK directly.

    Args:
        model: Anthropic model ID. Defaults to claude-haiku-4-5 (fast + cheap).
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY env var if not set.
        temperature: Sampling temperature for summaries (default 0.3).

    Example::

        from openlcm.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(model="claude-haiku-4-5-20251001")
        engine = LCMEngine(backend=backend)
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str = "",
        temperature: float = 0.3,
    ) -> None:
        self._default_model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicBackend. "
                    "Install with: pip install openlcm[anthropic]"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def summarize(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "",
        timeout: float | None = None,
    ) -> Optional[str]:
        effective_model = model or self._default_model
        client = self._get_client()
        try:
            kwargs: dict = {
                "model": effective_model,
                "max_tokens": max(64, min(max_tokens, 8096)),
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if timeout is not None:
                import httpx
                kwargs["timeout"] = httpx.Timeout(timeout)

            response = await client.messages.create(**kwargs)
            content = response.content[0].text if response.content else ""
            return content.strip() or None
        except Exception as exc:
            logger.warning("AnthropicBackend summarize failed: %s", exc)
            return None
