"""Haystack v2 message converter for LCM.

Handles Haystack v2's ``ChatMessage`` dataclass and converts to/from the
LCM internal dict format.

Works with every Haystack v2 generator component:
    OpenAIChatGenerator, AnthropicChatGenerator, HuggingFaceAPIChatGenerator,
    AzureOpenAIChatGenerator, GoogleAIStudioGeminiChatGenerator, OllamaChatGenerator,
    CohereChatGenerator, MistralChatGenerator, …

Install: pip install haystack-ai

Quick start::

    from haystack import Pipeline
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from openlcm import LCMEngine
    from openlcm.adapters.haystack import HaystackMessages

    generator = OpenAIChatGenerator(model="gpt-4o-mini")
    engine    = LCMEngine(model="openai/gpt-4o-mini")
    engine.bind_session("my-session", context_length=128_000)

    conv = []   # list[ChatMessage]

    async def chat(user_msg: str) -> str:
        conv.append(ChatMessage.from_user(user_msg))

        lcm_msgs = HaystackMessages.to_lcm(conv)
        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs = await engine.compress(lcm_msgs)
            conv[:] = HaystackMessages.from_lcm(lcm_msgs)

        result = generator.run(messages=conv)
        reply  = result["replies"][0]
        conv.append(reply)
        return reply.text or ""

Tool calling (Haystack v2.3+)::

    # Haystack v2.3+ added ToolCall and ChatMessage.from_tool_result()
    # The converter handles both the legacy additional_kwargs style and the
    # new ToolCall dataclass style transparently.
"""

from __future__ import annotations

import json
from typing import Any


def _hs_text(msg: Any) -> str:
    """Extract text from a Haystack ChatMessage (handles both text and .content)."""
    # Haystack >= 2.3: msg.text
    text = getattr(msg, "text", None)
    if text is not None:
        return str(text)
    # Older Haystack: msg.content
    content = getattr(msg, "content", None)
    if content is not None:
        return str(content)
    return ""


def _hs_role(msg: Any) -> str:
    """Normalize Haystack ChatRole to a plain string."""
    role = getattr(msg, "role", "user")
    r_str = role.value if hasattr(role, "value") else str(role)
    _map = {
        "user":      "user",
        "assistant": "assistant",
        "system":    "system",
        "tool":      "tool",
        "function":  "tool",
    }
    return _map.get(r_str.lower(), r_str.lower())


class HaystackMessages:
    """Convert between Haystack v2 ChatMessage objects and LCM internal format.

    All methods are static — no instantiation needed::

        lcm  = HaystackMessages.to_lcm(haystack_messages)
        back = HaystackMessages.from_lcm(lcm_messages)

    Tool calls
    ----------
    Haystack v2.3+ stores tool calls via ``ToolCall`` dataclasses in
    ``ChatMessage.tool_calls``.  Earlier versions used ``additional_kwargs``.
    Both styles are supported.
    """

    @staticmethod
    def to_lcm(messages: list) -> list[dict]:
        """Convert a list of Haystack ChatMessage objects to LCM internal format.

        Args:
            messages: ``list[ChatMessage]`` from a Haystack pipeline.

        Returns:
            LCM internal message list ready for ``engine.compress()``.
        """
        result: list[dict] = []

        for m in messages:
            role = _hs_role(m)

            # ── assistant ─────────────────────────────────────────────────────
            if role == "assistant":
                tool_calls: list[dict] = []

                # Haystack >= 2.3: ToolCall dataclasses
                hs_tcs = getattr(m, "tool_calls", None) or []
                for tc in hs_tcs:
                    tc_id   = getattr(tc, "id",        None) or getattr(tc, "tool_call_id", "") or ""
                    tc_name = getattr(tc, "tool_name",  None) or getattr(tc, "name", "") or ""
                    tc_args = getattr(tc, "arguments",  None) or getattr(tc, "args", {}) or {}
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except (ValueError, TypeError):
                            tc_args = {"_raw": tc_args}
                    tool_calls.append({"id": str(tc_id), "name": str(tc_name), "args": dict(tc_args), "type": "function"})

                # Older Haystack: additional_kwargs["tool_calls"] (OpenAI format)
                if not tool_calls:
                    kwargs = getattr(m, "additional_kwargs", None) or {}
                    for tc in kwargs.get("tool_calls", []):
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except (ValueError, TypeError):
                            args = {}
                        tool_calls.append({
                            "id":   tc.get("id", "") if isinstance(tc, dict) else "",
                            "name": fn.get("name", ""),
                            "args": args,
                            "type": "function",
                        })

                text = _hs_text(m)
                if tool_calls:
                    content_str = json.dumps({"text": text, "tool_calls": tool_calls}, ensure_ascii=False)
                else:
                    content_str = text
                result.append({"role": "assistant", "content": content_str})

            # ── tool result ───────────────────────────────────────────────────
            elif role == "tool":
                # Haystack >= 2.3: ChatMessage.from_tool_result(result, origin=ToolCall(...))
                origin   = getattr(m, "tool_call_result", None) or getattr(m, "origin", None)
                tc_id    = ""
                tc_name  = ""
                if origin is not None:
                    orig_call = getattr(origin, "origin", origin)
                    tc_id   = getattr(orig_call, "id",        "") or getattr(orig_call, "tool_call_id", "") or ""
                    tc_name = getattr(orig_call, "tool_name",  "") or getattr(orig_call, "name", "") or ""
                # Fallback: additional_kwargs
                kwargs = getattr(m, "additional_kwargs", None) or {}
                tc_id   = tc_id   or kwargs.get("tool_call_id", "")
                tc_name = tc_name or kwargs.get("name", "")

                result.append({
                    "role":         "tool",
                    "content":      _hs_text(m),
                    "tool_call_id": str(tc_id),
                    "name":         str(tc_name),
                })

            # ── user / system ─────────────────────────────────────────────────
            else:
                result.append({"role": role, "content": _hs_text(m)})

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> list:
        """Convert LCM internal format back to Haystack ChatMessage objects.

        Args:
            messages: LCM internal message list (output of ``engine.compress()``).

        Returns:
            ``list[ChatMessage]`` ready to feed into a Haystack generator component.
        """
        try:
            from haystack.dataclasses import ChatMessage
        except ImportError:
            raise ImportError(
                "haystack-ai is required. Install with: pip install haystack-ai"
            )

        result: list = []

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")

            if role == "system":
                result.append(ChatMessage.from_system(content))

            elif role == "user":
                result.append(ChatMessage.from_user(content))

            elif role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        # Try Haystack >= 2.3 ToolCall API first
                        try:
                            from haystack.dataclasses import ToolCall
                            tcs = [
                                ToolCall(
                                    id=tc.get("id", ""),
                                    tool_name=tc.get("name", ""),
                                    arguments=tc.get("args", {}),
                                )
                                for tc in parsed["tool_calls"]
                            ]
                            result.append(ChatMessage.from_assistant(
                                parsed.get("text", ""),
                                tool_calls=tcs,
                            ))
                        except (ImportError, TypeError):
                            # Fallback: store tool_calls in additional_kwargs
                            kwargs = {
                                "tool_calls": [
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
                            }
                            msg = ChatMessage.from_assistant(parsed.get("text", ""))
                            msg.additional_kwargs = kwargs
                            result.append(msg)
                        continue
                except (ValueError, TypeError):
                    pass
                result.append(ChatMessage.from_assistant(content))

            elif role == "tool":
                try:
                    from haystack.dataclasses import ToolCall, ChatMessage as CM
                    origin = ToolCall(
                        id=m.get("tool_call_id", ""),
                        tool_name=m.get("name", ""),
                        arguments={},
                    )
                    # Haystack >= 2.3 from_tool_result
                    result.append(CM.from_tool_result(content, origin=origin))
                except (ImportError, TypeError, AttributeError):
                    # Fallback for older Haystack
                    msg = ChatMessage.from_user(content)
                    msg.additional_kwargs = {
                        "tool_call_id": m.get("tool_call_id", ""),
                        "name":         m.get("name", ""),
                    }
                    result.append(msg)

            else:
                result.append(ChatMessage.from_user(content))

        return result
