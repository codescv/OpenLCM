"""OpenLCM — Framework-agnostic Lossless Context Management SDK.

Quick start::

    from openlcm import LCMEngine
    from openlcm.backends.anthropic import AnthropicBackend

    engine = LCMEngine(
        backend=AnthropicBackend(model="claude-haiku-4-5-20251001"),
        db_path="~/.openlcm/myapp.db",
    )
    engine.bind_session("session-abc", context_length=200_000)

    # Each agent turn — returns LCM-compressed messages
    compressed = await engine.compress(messages)

Framework adapters::

    from openlcm.adapters.langgraph import LCMCheckpointer
    from openlcm.adapters.crewai import LCMStorage
    from openlcm.adapters.autogen import LCMContext
    from openlcm.adapters.google_adk import LCMSessionService

Visualization::

    openlcm viz   # starts dashboard at http://localhost:7842
"""

from .core.engine import LCMEngine
from .core.config import LCMConfig
from .backends.base import SummaryBackend

__version__ = "0.1.0"
__all__ = ["LCMEngine", "LCMConfig", "SummaryBackend"]
