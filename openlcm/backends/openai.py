"""OpenAI SDK backend for LCM summarization.

Install: pip install openlcm[openai]
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import SummaryBackend

logger = logging.getLogger(__name__)


class OpenAIBackend(SummaryBackend):
    """Summarization backend using the OpenAI SDK directly.

    Also works with any OpenAI-compatible API (Ollama, Together, Groq, etc.)
    by supplying a custom ``base_url``.

    Args:
        model: OpenAI model ID. Defaults to gpt-4o-mini.
        api_key: OpenAI API key. Reads OPENAI_API_KEY env var if not set.
        base_url: Optional base URL override for OpenAI-compatible APIs.
        temperature: Sampling temperature (default 0.3).

    Example::

        from openlcm.backends.openai import OpenAIBackend
        # Standard OpenAI
        backend = OpenAIBackend(model="gpt-4o-mini")
        # Ollama local
        backend = OpenAIBackend(model="llama3.2", base_url="http://localhost:11434/v1", api_key="ollama")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.3,
    ) -> None:
        self._default_model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIBackend. "
                    "Install with: pip install openlcm[openai]"
                )
            kwargs: dict = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
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
                "max_tokens": max(64, min(max_tokens, 16384)),
                "temperature": self._temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if timeout is not None:
                kwargs["timeout"] = timeout

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return content.strip() or None
        except Exception as exc:
            logger.warning("OpenAIBackend summarize failed: %s", exc)
            return None
