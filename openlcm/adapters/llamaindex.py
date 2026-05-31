"""LlamaIndex message converter for LCM.

Handles LlamaIndex's ``ChatMessage`` format and converts to/from the
LCM internal dict format.

Works with every LlamaIndex LLM integration:
    llama_index.llms.openai, llama_index.llms.anthropic,
    llama_index.llms.gemini, llama_index.llms.ollama,
    llama_index.llms.mistral, llama_index.llms.groq, …

Install: pip install llama-index-core

Quick start::

    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI
    from openlcm import LCMEngine
    from openlcm.adapters.llamaindex import LlamaIndexMessages

    llm    = OpenAI(model="gpt-4o-mini")
    engine = LCMEngine(model="openai/gpt-4o-mini")
    engine.bind_session("my-session", context_length=128_000)

    conv = []   # list[ChatMessage]

    async def chat(user_msg: str) -> str:
        conv.append(ChatMessage.from_str(user_msg, role="user"))

        lcm_msgs = LlamaIndexMessages.to_lcm(conv)
        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs = await engine.compress(lcm_msgs)
            conv[:] = LlamaIndexMessages.from_lcm(lcm_msgs)

        response = await llm.achat(conv)
        conv.append(response.message)
        return response.message.content

Tool calling (via additional_kwargs)::

    # LlamaIndex stores OpenAI-style tool_calls in additional_kwargs.
    # LlamaIndexMessages serializes / restores them transparently.
    msg = response.message
    if msg.additional_kwargs.get("tool_calls"):
        for tc in msg.additional_kwargs["tool_calls"]:
            result = run_tool(tc["function"]["name"], tc["function"]["arguments"])
            conv.append(ChatMessage(
                role="tool",
                content=str(result),
                additional_kwargs={
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                },
            ))
"""

from __future__ import annotations

import json
from typing import Any


def _content_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


class LlamaIndexMessages:
    """Convert between LlamaIndex ChatMessage objects and LCM internal format.

    All methods are static — no instantiation needed::

        lcm  = LlamaIndexMessages.to_lcm(chat_messages)
        back = LlamaIndexMessages.from_lcm(lcm_messages)

    Tool calls
    ----------
    LlamaIndex stores tool calls in ``ChatMessage.additional_kwargs["tool_calls"]``
    using the OpenAI tool_call dict format.  This converter serializes them into
    the LCM ``{"text":...,"tool_calls":[...]}`` JSON convention so they survive
    compression intact.
    """

    @staticmethod
    def to_lcm(messages: list) -> list[dict]:
        """Convert a list of LlamaIndex ChatMessage objects to LCM internal format.

        Args:
            messages: ``list[ChatMessage]`` — the conversation history.

        Returns:
            LCM internal message list ready for ``engine.compress()``.
        """
        try:
            from llama_index.core.llms import MessageRole
        except ImportError:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )

        # Normalize MessageRole → string
        def _role(msg) -> str:
            r = msg.role
            r_str = r.value if hasattr(r, "value") else str(r)
            _map = {
                "user":      "user",
                "human":     "user",
                "assistant": "assistant",
                "ai":        "assistant",
                "chatbot":   "assistant",
                "model":     "assistant",
                "system":    "system",
                "tool":      "tool",
            }
            return _map.get(r_str.lower(), r_str.lower())

        result: list[dict] = []

        for m in messages:
            role    = _role(m)
            content = _content_str(m.content)
            kwargs  = m.additional_kwargs or {}

            # ── assistant with tool_calls ─────────────────────────────────────
            if role == "assistant" and kwargs.get("tool_calls"):
                normalized: list[dict] = []
                for tc in kwargs["tool_calls"]:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except (ValueError, TypeError):
                        args = {"_raw": raw_args}
                    normalized.append({
                        "id":   tc.get("id", "") if isinstance(tc, dict) else "",
                        "name": fn.get("name", ""),
                        "args": args,
                        "type": "function",
                    })
                content_str = json.dumps(
                    {"text": content, "tool_calls": normalized},
                    ensure_ascii=False,
                )
                result.append({"role": "assistant", "content": content_str})

            # ── tool result ───────────────────────────────────────────────────
            elif role == "tool":
                result.append({
                    "role":         "tool",
                    "content":      content,
                    "tool_call_id": kwargs.get("tool_call_id", ""),
                    "name":         kwargs.get("name", ""),
                })

            # ── user / system / assistant (no tools) ─────────────────────────
            else:
                result.append({"role": role, "content": content})

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> list:
        """Convert LCM internal format back to LlamaIndex ChatMessage objects.

        Args:
            messages: LCM internal message list (output of ``engine.compress()``).

        Returns:
            ``list[ChatMessage]`` ready to pass to any LlamaIndex LLM's ``chat()``
            or ``achat()`` method.
        """
        try:
            from llama_index.core.llms import ChatMessage, MessageRole
        except ImportError:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )

        _role_map = {
            "user":      MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system":    MessageRole.SYSTEM,
        }
        # Some versions of LlamaIndex have TOOL role, others don't
        _tool_role = getattr(MessageRole, "TOOL", MessageRole.USER)

        result: list = []

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")
            kwargs:  dict = {}

            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        content = parsed.get("text", "")
                        # Reconstruct OpenAI-style tool_calls for additional_kwargs
                        kwargs["tool_calls"] = [
                            {
                                "id":   tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name":      tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                                },
                            }
                            for tc in parsed["tool_calls"]
                        ]
                except (ValueError, TypeError):
                    pass

            elif role == "tool":
                tc_id = m.get("tool_call_id", "")
                name  = m.get("name", "")
                if tc_id:
                    kwargs["tool_call_id"] = tc_id
                if name:
                    kwargs["name"] = name

            li_role = (
                _tool_role if role == "tool"
                else _role_map.get(role, MessageRole.USER)
            )
            result.append(ChatMessage(
                role=li_role,
                content=content,
                additional_kwargs=kwargs or None,
            ))

        return result
