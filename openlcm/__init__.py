"""OpenLCM — Framework-agnostic Lossless Context Management SDK.

One install, every provider::

    pip install openlcm

Quick start — pass any LiteLLM model string::

    from openlcm import LCMEngine

    # Anthropic
    engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")

    # Azure OpenAI
    engine = LCMEngine(model="azure/gpt-4o")

    # AWS Bedrock
    engine = LCMEngine(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Google Gemini
    engine = LCMEngine(model="gemini/gemini-2.0-flash")

    # Vertex AI
    engine = LCMEngine(model="vertex_ai/gemini-pro")

    # Ollama (local)
    engine = LCMEngine(model="ollama/llama3.2", api_base="http://localhost:11434")

    # WatsonX
    engine = LCMEngine(model="watsonx/ibm/granite-13b-chat-v2")

    # Custom OpenAI-compatible endpoint
    engine = LCMEngine(model="openai/my-model", api_base="http://my-server/v1")

Bind a session and compress each turn::

    engine.bind_session("session-abc", context_length=200_000)
    compressed = await engine.compress(messages)   # call before every LLM turn

Already using a framework LLM? Pass it directly — no separate model config needed::

    from langchain_anthropic import ChatAnthropic
    from openlcm.adapters.langgraph import LCMCheckpointer

    llm = ChatAnthropic(model="claude-3-haiku-20240307")  # your existing LLM
    checkpointer = LCMCheckpointer(llm=llm)               # reuses it for summarization

    # Works the same way for CrewAI, AutoGen, Google ADK:
    LCMStorage(llm=my_crewai_llm)
    LCMContext(llm=my_autogen_client)
    LCMSessionService(llm=my_gemini_model)

    # Or pass any callable to LCMEngine directly:
    engine = LCMEngine(summarize_fn=llm)
    engine = LCMEngine(summarize_fn=lambda prompt, max_tokens: llm.invoke(prompt).content)

Live dashboard::

    openlcm viz   # http://localhost:7842

Advanced — bring your own backend (custom inference server, etc.)::

    from openlcm.backends.base import SummaryBackend

    class MyBackend(SummaryBackend):
        async def summarize(self, prompt, max_tokens, model="", timeout=None):
            ...

    engine = LCMEngine(backend=MyBackend())

Full provider list: https://docs.litellm.ai/docs/providers
"""

from .core.engine import LCMEngine
from .core.config import LCMConfig
from .backends.base import SummaryBackend
from .backends.callable import CallableBackend

__version__ = "0.1.0"
__all__ = ["LCMEngine", "LCMConfig", "SummaryBackend", "CallableBackend"]
