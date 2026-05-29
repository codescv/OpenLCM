"""LCM summarization backends.

Import the backend matching your LLM provider:

    from openlcm.backends.anthropic import AnthropicBackend
    from openlcm.backends.openai import OpenAIBackend
    from openlcm.backends.litellm import LiteLLMBackend

All backends implement the SummaryBackend ABC defined in openlcm.backends.base.
"""

from .base import SummaryBackend

__all__ = ["SummaryBackend"]
