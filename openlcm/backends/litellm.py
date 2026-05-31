"""LiteLLM backend for LCM summarization.

Covers every major provider through a single model-string interface:

  Provider          Model string example
  ──────────────    ────────────────────────────────────────────────────
  Anthropic         anthropic/claude-haiku-4-5-20251001
  OpenAI            openai/gpt-4o-mini
  Azure OpenAI      azure/gpt-4o
  AWS Bedrock       bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
  Google Gemini     gemini/gemini-2.0-flash
  Google Vertex AI  vertex_ai/gemini-pro
  Ollama (local)    ollama/llama3.2
  WatsonX           watsonx/ibm/granite-13b-chat-v2
  Groq              groq/llama-3.1-8b-instant
  Custom endpoint   openai/my-model  + api_base="http://..."

Full provider list: https://docs.litellm.ai/docs/providers
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

    Most users never need to instantiate this directly — pass a model string
    to LCMEngine and it creates this backend automatically::

        engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")
        engine = LCMEngine(model="azure/gpt-4o")
        engine = LCMEngine(model="ollama/llama3.2", api_base="http://localhost:11434")

    Instantiate directly only when you need to share a backend across engines::

        backend = LiteLLMBackend(model="gemini/gemini-2.0-flash")
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
        import litellm
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
