"""Base adapter ABC and engine-resolution helper for framework integrations.

Each adapter is a thin bridge that translates framework-specific lifecycle
events (checkpoint saves, memory writes, etc.) into LCMEngine calls.
The core LCM algorithm stays entirely in LCMEngine.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openlcm.core.engine import LCMEngine


def _resolve_engine(
    engine: Any,
    *,
    llm: Any = None,
    db_path: str = "",
    platform: str = "",
) -> "LCMEngine":
    """Return an LCMEngine from an explicit engine, an existing LLM, or raise.

    Priority:
      1. explicit engine passed → use as-is
      2. llm= provided → wrap in CallableBackend, create engine automatically
      3. neither → raise with a helpful message
    """
    from openlcm.core.engine import LCMEngine

    if engine is not None:
        return engine

    if llm is not None:
        return LCMEngine(summarize_fn=llm, db_path=db_path)

    raise ValueError(
        "Pass either an LCMEngine instance or your existing LLM via llm=.\n"
        "\n"
        "  # Reuse your existing LLM (recommended for framework users):\n"
        "  adapter = Adapter(llm=my_langchain_llm)\n"
        "\n"
        "  # Or build an engine explicitly:\n"
        "  from openlcm import LCMEngine\n"
        "  engine = LCMEngine(model='anthropic/claude-haiku-4-5-20251001')\n"
        "  adapter = Adapter(engine)\n"
    )


class LCMAdapter(ABC):
    """Base class for all framework adapters.

    Subclasses wrap an LCMEngine and expose the integration surface
    specific to each framework (LangGraph, CrewAI, AutoGen, Google ADK).
    """

    def __init__(self, engine: "LCMEngine") -> None:
        self._engine = engine

    @property
    def engine(self) -> "LCMEngine":
        return self._engine
