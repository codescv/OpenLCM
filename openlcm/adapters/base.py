"""Base adapter ABC for framework integrations.

Each adapter is a thin bridge that translates framework-specific lifecycle
events (checkpoint saves, memory writes, etc.) into LCMEngine calls.
The core LCM algorithm stays entirely in LCMEngine.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openlcm.core.engine import LCMEngine


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
