"""Framework adapters for OpenLCM.

Two types of adapter live here:

1. **Message converters** — translate message formats between frameworks and
   LCM's internal dict format.  Use these when you manage the conversation
   history yourself (raw SDK, custom agent loop, etc.).

2. **Framework lifecycle adapters** — plug LCM into a framework's built-in
   memory / checkpointing / context system.  Use these when you want LCM to
   work *transparently* inside an existing framework graph or crew.

──────────────────────────────────────────────────────────────────────────────
Message converters  (``to_lcm`` / ``from_lcm``)
──────────────────────────────────────────────────────────────────────────────

OpenAI (and every OpenAI-compatible API)::

    from openlcm.adapters.openai import OpenAIMessages

    lcm_msgs = OpenAIMessages.to_lcm(openai_messages)
    if engine.should_compress_preflight(lcm_msgs):
        lcm_msgs     = await engine.compress(lcm_msgs)
        openai_msgs  = OpenAIMessages.from_lcm(lcm_msgs)

    # Compatible with: OpenAI, Groq, Together, Mistral, Perplexity, Fireworks,
    #                  Azure OpenAI, Ollama, vLLM, LM Studio, OpenRouter, Anyscale

Anthropic::

    from openlcm.adapters.anthropic import AnthropicMessages

    lcm_msgs           = AnthropicMessages.to_lcm(messages, system=system_prompt)
    system_out, an_msgs = AnthropicMessages.from_lcm(lcm_msgs)
    # Note: Anthropic returns (system_str, messages) because system is a separate param

LangChain (any backend)::

    from openlcm.adapters.langchain import LangChainMessages

    lcm_msgs  = LangChainMessages.to_lcm(lc_messages)   # list[BaseMessage]
    lc_msgs   = LangChainMessages.from_lcm(lcm_msgs)    # list[BaseMessage]

    # Compatible with: langchain_openai, langchain_anthropic, langchain_google_genai,
    #                  langchain_cohere, langchain_mistralai, langchain_groq,
    #                  langchain_ollama, langchain_aws (Bedrock), langchain_together, …

LlamaIndex::

    from openlcm.adapters.llamaindex import LlamaIndexMessages

    lcm_msgs = LlamaIndexMessages.to_lcm(chat_messages)  # list[ChatMessage]
    li_msgs  = LlamaIndexMessages.from_lcm(lcm_msgs)

    # Compatible with: llama_index.llms.openai, llama_index.llms.anthropic,
    #                  llama_index.llms.gemini, llama_index.llms.ollama, …

Haystack v2::

    from openlcm.adapters.haystack import HaystackMessages

    lcm_msgs = HaystackMessages.to_lcm(hs_messages)  # list[ChatMessage]
    hs_msgs  = HaystackMessages.from_lcm(lcm_msgs)

    # Compatible with: OpenAIChatGenerator, AnthropicChatGenerator,
    #                  HuggingFaceAPIChatGenerator, AzureOpenAIChatGenerator, …

Auto-detect the right converter from your message list::

    from openlcm.adapters import auto_detect

    converter = auto_detect(messages)
    lcm_msgs  = converter.to_lcm(messages)

──────────────────────────────────────────────────────────────────────────────
Framework lifecycle adapters
──────────────────────────────────────────────────────────────────────────────

LangGraph — transparent checkpointing::

    from openlcm.adapters.langgraph import LCMCheckpointer   # pip install openlcm[langgraph]

    graph = StateGraph(MyState).compile(checkpointer=LCMCheckpointer(llm=my_llm))

CrewAI — long-term memory storage::

    from openlcm.adapters.crewai import LCMStorage           # pip install openlcm[crewai]

    crew = Crew(..., long_term_memory=LongTermMemory(storage=LCMStorage(llm=my_llm)))

AutoGen — model context::

    from openlcm.adapters.autogen import LCMContext           # pip install openlcm[autogen]

    agent = AssistantAgent("bot", model_client=client, model_context=LCMContext(llm=client))

Google ADK — session service::

    from openlcm.adapters.google_adk import LCMSessionService  # pip install openlcm[google-adk]

    runner = Runner(agent=my_agent, session_service=LCMSessionService(llm=model))
"""

from __future__ import annotations

from typing import Any

# Top-level imports so `from openlcm.adapters import X` works directly.
# Each import is wrapped so the package loads even if a framework isn't installed.
try:
    from .openai import OpenAIMessages
except ImportError:
    OpenAIMessages = None  # type: ignore[assignment,misc]

try:
    from .anthropic import AnthropicMessages
except ImportError:
    AnthropicMessages = None  # type: ignore[assignment,misc]

try:
    from .langchain import LangChainMessages
except ImportError:
    LangChainMessages = None  # type: ignore[assignment,misc]

try:
    from .llamaindex import LlamaIndexMessages
except ImportError:
    LlamaIndexMessages = None  # type: ignore[assignment,misc]

try:
    from .haystack import HaystackMessages
except ImportError:
    HaystackMessages = None  # type: ignore[assignment,misc]

try:
    from .gemini import GeminiMessages
except ImportError:
    GeminiMessages = None  # type: ignore[assignment,misc]

try:
    from .autogen import AutoGenMessages, LCMContext
except ImportError:
    AutoGenMessages = None  # type: ignore[assignment,misc]
    LCMContext = None  # type: ignore[assignment,misc]

try:
    from .langgraph import LCMCheckpointer
except ImportError:
    LCMCheckpointer = None  # type: ignore[assignment,misc]

try:
    from .crewai import LCMStorage
except ImportError:
    LCMStorage = None  # type: ignore[assignment,misc]

try:
    from .google_adk import LCMSessionService, lcm_compress_callback
except ImportError:
    LCMSessionService = None  # type: ignore[assignment,misc]
    lcm_compress_callback = None  # type: ignore[assignment,misc]


def auto_detect(messages: list) -> Any:
    """Return the right message converter for the given message list.

    Inspects the first non-empty element and returns the appropriate converter
    class (with static ``to_lcm`` / ``from_lcm`` methods).

    Supported detection:
    - LangChain ``BaseMessage`` subclasses → ``LangChainMessages``
    - Anthropic content-block dicts (``type: tool_use``) → ``AnthropicMessages``
    - Haystack ``ChatMessage`` objects → ``HaystackMessages``
    - LlamaIndex ``ChatMessage`` objects → ``LlamaIndexMessages``
    - OpenAI dicts (``tool_calls`` key / plain role+content) → ``OpenAIMessages``

    Falls back to ``OpenAIMessages`` if the format cannot be determined.

    Args:
        messages: The conversation list whose format you want to detect.

    Returns:
        A converter class with static ``to_lcm()`` and ``from_lcm()`` methods.

    Example::

        from openlcm.adapters import auto_detect

        conv = openai_client.chat.completions.create(...).choices[0].message
        converter = auto_detect(my_messages)
        lcm = converter.to_lcm(my_messages)
    """
    if not messages:
        from .openai import OpenAIMessages
        return OpenAIMessages

    first = next((m for m in messages if m is not None), None)
    if first is None:
        from .openai import OpenAIMessages
        return OpenAIMessages

    # ── LangChain BaseMessage ─────────────────────────────────────────────────
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(first, BaseMessage):
            from .langchain import LangChainMessages
            return LangChainMessages
    except ImportError:
        pass

    # ── Haystack ChatMessage ──────────────────────────────────────────────────
    try:
        from haystack.dataclasses import ChatMessage as HaystackCM
        if isinstance(first, HaystackCM):
            from .haystack import HaystackMessages
            return HaystackMessages
    except ImportError:
        pass

    # ── LlamaIndex ChatMessage ────────────────────────────────────────────────
    try:
        from llama_index.core.llms import ChatMessage as LlamaCM
        if isinstance(first, LlamaCM):
            from .llamaindex import LlamaIndexMessages
            return LlamaIndexMessages
    except ImportError:
        pass

    # ── Anthropic content-block style ─────────────────────────────────────────
    if isinstance(first, dict):
        content = first.get("content", "")
        if isinstance(content, list) and content:
            block = content[0]
            if isinstance(block, dict) and block.get("type") in ("text", "tool_use", "tool_result", "image"):
                from .anthropic import AnthropicMessages
                return AnthropicMessages

    # ── Default: OpenAI dict format ───────────────────────────────────────────
    from .openai import OpenAIMessages
    return OpenAIMessages


__all__ = [
    # Auto-detection
    "auto_detect",
    # Message converters
    "OpenAIMessages",
    "AnthropicMessages",
    "LangChainMessages",
    "LlamaIndexMessages",
    "HaystackMessages",
    # Framework lifecycle adapters
    "LCMCheckpointer",
    "LCMStorage",
    "LCMContext",
    "LCMSessionService",
]
