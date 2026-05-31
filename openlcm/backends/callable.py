"""CallableBackend — wrap any sync or async callable as a SummaryBackend.

This is the bridge for framework users who already have an LLM configured.
Instead of specifying a separate model string, they hand their existing LLM
to the adapter (or directly to LCMEngine) and LCM reuses it for summarization.

Supported callable shapes
──────────────────────────
  async def fn(prompt: str, max_tokens: int) -> str   # preferred
  async def fn(prompt: str) -> str
  def fn(prompt: str, max_tokens: int) -> str
  def fn(prompt: str) -> str

LangChain / LangGraph users
─────────────────────────────
The adapter constructors accept llm= directly and do this wrapping for you.
If you want to wire it manually::

    from langchain_anthropic import ChatAnthropic
    from openlcm import LCMEngine

    llm = ChatAnthropic(model="claude-3-haiku-20240307")
    engine = LCMEngine(summarize_fn=llm)   # LangChain models are auto-detected

    # Or wrap it explicitly:
    engine = LCMEngine(summarize_fn=lambda prompt, max_tokens: llm.invoke(prompt).content)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Optional

from .base import SummaryBackend

logger = logging.getLogger(__name__)


def _is_langchain_model(obj: Any) -> bool:
    """Duck-type check: does this look like a LangChain BaseChatModel?"""
    return hasattr(obj, "invoke") and hasattr(obj, "ainvoke")


def _is_crewai_llm(obj: Any) -> bool:
    """Duck-type check: CrewAI LLM wraps LangChain — same interface."""
    return hasattr(obj, "invoke") and hasattr(obj, "model_name")


async def _call_langchain(llm: Any, prompt: str, max_tokens: int) -> Optional[str]:
    """Invoke a LangChain BaseChatModel for summarization."""
    from langchain_core.messages import HumanMessage
    try:
        response = await llm.ainvoke(
            [HumanMessage(content=prompt)],
            config={"max_tokens": max_tokens},
        )
        return (response.content or "").strip() or None
    except Exception:
        # Retry without max_tokens config — some models don't accept it this way
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return (response.content or "").strip() or None
        except Exception as exc:
            logger.warning("CallableBackend LangChain call failed: %s", exc)
            return None


class CallableBackend(SummaryBackend):
    """SummaryBackend that delegates to a user-supplied callable or LLM object.

    LCMEngine creates this automatically when you pass summarize_fn= or when
    a framework adapter receives llm=.

    Args:
        fn: Any of the following:
            - A LangChain/LangGraph BaseChatModel (ChatAnthropic, ChatOpenAI, etc.)
            - A CrewAI LLM instance
            - An async callable: async def(prompt, max_tokens) -> str
            - A sync callable:   def(prompt, max_tokens) -> str
            - A callable with only one arg: def(prompt) -> str
    """

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self._is_langchain = _is_langchain_model(fn)

    async def summarize(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "",
        timeout: float | None = None,
    ) -> Optional[str]:
        try:
            if self._is_langchain:
                coro = _call_langchain(self._fn, prompt, max_tokens)
            elif asyncio.iscoroutinefunction(self._fn):
                sig = inspect.signature(self._fn)
                if len(sig.parameters) >= 2:
                    coro = self._fn(prompt, max_tokens)
                else:
                    coro = self._fn(prompt)
            else:
                # Sync callable — run in thread pool so we don't block the event loop
                sig = inspect.signature(self._fn)
                if len(sig.parameters) >= 2:
                    coro = asyncio.get_event_loop().run_in_executor(
                        None, self._fn, prompt, max_tokens
                    )
                else:
                    coro = asyncio.get_event_loop().run_in_executor(
                        None, self._fn, prompt
                    )

            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro

            if result is None:
                return None
            # Handle LangChain AIMessage returned by mistake
            if hasattr(result, "content"):
                return (result.content or "").strip() or None
            return str(result).strip() or None

        except Exception as exc:
            logger.warning("CallableBackend.summarize failed: %s", exc)
            return None
