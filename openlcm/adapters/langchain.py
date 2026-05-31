"""LangChain message converter for LCM.

Handles LangChain's BaseMessage subclasses (HumanMessage, AIMessage,
SystemMessage, ToolMessage, FunctionMessage, etc.) and converts to/from
the LCM internal dict format.

Works with every LangChain chat model backend:
    langchain_openai, langchain_anthropic, langchain_google_genai,
    langchain_cohere, langchain_mistralai, langchain_groq, langchain_ollama,
    langchain_aws (Bedrock), langchain_together, langchain_fireworks, …

Install: pip install langchain-core  (already a dependency of any LangChain model)

Quick start — drop-in replacement for manual _to_dicts/_from_dicts::

    from langchain_openai import ChatOpenAI
    from openlcm import LCMEngine
    from openlcm.adapters.langchain import LangChainMessages

    llm    = ChatOpenAI(model="gpt-4o-mini")
    engine = LCMEngine(summarize_fn=llm)
    engine.bind_session("my-session", context_length=128_000)

    conv = []   # list[BaseMessage]

    async def chat(user_msg: str) -> str:
        from langchain_core.messages import HumanMessage
        conv.append(HumanMessage(content=user_msg))

        lcm_msgs = LangChainMessages.to_lcm(conv)
        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs = await engine.compress(lcm_msgs)
            conv[:] = LangChainMessages.from_lcm(lcm_msgs)

        response = await llm.ainvoke(conv)
        conv.append(response)
        return response.content

With tools (LangGraph ToolNode / manual)::

    from langchain_core.messages import ToolMessage

    # After LLM returns AIMessage with tool_calls, run the tools:
    for tc in response.tool_calls:
        result = run_tool(tc["name"], tc["args"])
        conv.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))

    # Then call llm.ainvoke(conv) again — LCM compresses across tool boundaries.
"""

from __future__ import annotations

import json
from typing import Any


def _content_to_str(content: Any) -> str:
    """Flatten LangChain list-content (vision, multi-modal) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


class LangChainMessages:
    """Convert between LangChain BaseMessage objects and LCM internal format.

    All methods are static — no instantiation needed::

        lcm  = LangChainMessages.to_lcm(langchain_messages)
        back = LangChainMessages.from_lcm(lcm_messages)

    Supported message types
    -----------------------
    - ``SystemMessage``
    - ``HumanMessage``      (including multi-modal list content)
    - ``AIMessage``         (with or without ``tool_calls``)
    - ``ToolMessage``       (tool execution results)
    - ``FunctionMessage``   (legacy, treated as tool result)
    - Any ``BaseMessage``   subclass (falls back to role/content)
    """

    @staticmethod
    def to_lcm(messages: list) -> list[dict]:
        """Convert a list of LangChain BaseMessage objects to LCM internal format.

        Args:
            messages: ``list[BaseMessage]`` — the conversation history.

        Returns:
            LCM internal message list ready for ``engine.compress()``.
        """
        try:
            from langchain_core.messages import (
                AIMessage, HumanMessage, SystemMessage,
                ToolMessage, FunctionMessage,
            )
        except ImportError:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            )

        result: list[dict] = []

        for m in messages:
            # ── SystemMessage ─────────────────────────────────────────────────
            if isinstance(m, SystemMessage):
                result.append({"role": "system", "content": _content_to_str(m.content)})

            # ── HumanMessage ──────────────────────────────────────────────────
            elif isinstance(m, HumanMessage):
                result.append({"role": "user", "content": _content_to_str(m.content)})

            # ── AIMessage ─────────────────────────────────────────────────────
            elif isinstance(m, AIMessage):
                tool_calls = getattr(m, "tool_calls", None) or []
                if tool_calls:
                    normalized = [
                        {
                            "id":   tc.get("id", "")   if isinstance(tc, dict) else getattr(tc, "id",   ""),
                            "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            "args": tc.get("args", {}) if isinstance(tc, dict) else dict(getattr(tc, "args", {})),
                            "type": "function",
                        }
                        for tc in tool_calls
                    ]
                    content_str = json.dumps(
                        {"text": _content_to_str(m.content), "tool_calls": normalized},
                        ensure_ascii=False,
                    )
                else:
                    content_str = _content_to_str(m.content)
                result.append({"role": "assistant", "content": content_str})

            # ── ToolMessage ───────────────────────────────────────────────────
            elif isinstance(m, ToolMessage):
                try:
                    raw = json.loads(m.content) if isinstance(m.content, str) else m.content
                    content_str = json.dumps(raw, ensure_ascii=False, indent=2)
                except (ValueError, TypeError):
                    content_str = str(m.content)
                result.append({
                    "role":         "tool",
                    "content":      content_str,
                    "tool_call_id": m.tool_call_id or "",
                    "name":         getattr(m, "name", "") or "",
                })

            # ── FunctionMessage (legacy) ──────────────────────────────────────
            elif isinstance(m, FunctionMessage):
                result.append({
                    "role":         "tool",
                    "content":      _content_to_str(m.content),
                    "tool_call_id": "",
                    "name":         getattr(m, "name", "") or "",
                })

            # ── Any other BaseMessage subclass ────────────────────────────────
            else:
                # Use .type which is "human"/"ai"/"system"/"tool" in LangChain
                lc_type = getattr(m, "type", None) or getattr(m, "role", "user")
                role_map = {"human": "user", "ai": "assistant", "chatbot": "assistant"}
                role = role_map.get(str(lc_type).lower(), str(lc_type))
                result.append({"role": role, "content": _content_to_str(m.content)})

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> list:
        """Convert LCM internal format back to LangChain BaseMessage objects.

        Args:
            messages: LCM internal message list (output of ``engine.compress()``).

        Returns:
            ``list[BaseMessage]`` ready to pass directly to any LangChain LLM.
        """
        try:
            from langchain_core.messages import (
                AIMessage, HumanMessage, SystemMessage, ToolMessage,
            )
        except ImportError:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            )

        result: list = []

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")

            if role == "system":
                result.append(SystemMessage(content=content))

            elif role == "user":
                result.append(HumanMessage(content=content))

            elif role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        result.append(AIMessage(
                            content=parsed.get("text", ""),
                            tool_calls=parsed["tool_calls"],
                        ))
                        continue
                except (ValueError, TypeError):
                    pass
                result.append(AIMessage(content=content))

            elif role == "tool":
                result.append(ToolMessage(
                    content=content,
                    tool_call_id=m.get("tool_call_id", ""),
                    name=m.get("name", ""),
                ))

            else:
                # Unknown role — default to HumanMessage to avoid breaking the LLM call
                result.append(HumanMessage(content=content))

        return result
