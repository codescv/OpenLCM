"""LiteLLM backend for LCM summarization.

Covers 100+ providers: Anthropic, OpenAI, Gemini, Bedrock, Ollama, Groq,
Together, Mistral, Cohere, Azure, Vertex AI, and more.

Install: pip install openlcm[litellm]

Model string format follows LiteLLM conventions:
  - "anthropic/claude-haiku-4-5"
  - "openai/gpt-4o-mini"
  - "gemini/gemini-1.5-flash"
  - "ollama/llama3.2"
  - "bedrock/anthropic.claude-3-haiku"
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import SummaryBackend

logger = logging.getLogger(__name__)


class LiteLLMBackend(SummaryBackend):
    """Summarization backend using LiteLLM (100+ providers, one interface).

    Args:
        model: LiteLLM model string, e.g. "anthropic/claude-haiku-4-5".
        temperature: Sampling temperature (default 0.3).
        api_key: Optional API key override (most providers read from env vars).
        api_base: Optional base URL override (for custom endpoints / Ollama).
        extra_kwargs: Additional kwargs passed through to litellm.acompletion().

    Example::

        from openlcm.backends.litellm import LiteLLMBackend
        backend = LiteLLMBackend(model="anthropic/claude-haiku-4-5-20251001")
        engine = LCMEngine(backend=backend)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        api_key: str = "",
        api_base: str = "",
        **extra_kwargs,
    ) -> None:
        if not model:
            raise ValueError("LiteLLMBackend requires a model string, e.g. 'anthropic/claude-haiku-4-5'")
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or None
        self._api_base = api_base or None
        self._extra_kwargs = extra_kwargs

    async def summarize(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "",
        timeout: float | None = None,
    ) -> Optional[str]:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm package is required for LiteLLMBackend. "
                "Install with: pip install openlcm[litellm]"
            )
        effective_model = model or self._model
        kwargs: dict = {
            "model": effective_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(64, max_tokens),
            "temperature": self._temperature,
            **self._extra_kwargs,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content or ""
            return content.strip() or None
        except Exception as exc:
            logger.warning("LiteLLMBackend summarize failed (%s): %s", effective_model, exc)
            return None
