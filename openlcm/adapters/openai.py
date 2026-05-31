"""OpenAI message converter for LCM.

Handles the OpenAI chat completion message format and converts to/from the
LCM internal dict format.  Because OpenAI's API is the de-facto standard,
this adapter also works unchanged with every OpenAI-compatible endpoint:

    - Groq        (groq.com)
    - Together AI (together.ai)
    - Mistral     (mistral.ai)
    - Perplexity  (pplx.ai)
    - Fireworks   (fireworks.ai)
    - Anyscale
    - Azure OpenAI
    - Ollama      (local, via /v1 compat)
    - vLLM        (local, via /v1 compat)
    - LM Studio   (local)
    - OpenRouter
    - Any other server that speaks the OpenAI chat completions spec

Install: pip install openai  (or the compatible client of your choice)

Quick start::

    from openai import AsyncOpenAI
    from openlcm import LCMEngine
    from openlcm.adapters.openai import OpenAIMessages

    client = AsyncOpenAI()
    engine = LCMEngine(model="openai/gpt-4o-mini")
    engine.bind_session("my-session", context_length=128_000)

    conv = []   # your running conversation

    async def chat(user_msg: str) -> str:
        conv.append({"role": "user", "content": user_msg})
        lcm_msgs = OpenAIMessages.to_lcm(conv)

        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs = await engine.compress(lcm_msgs)
            conv[:] = OpenAIMessages.from_lcm(lcm_msgs)

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conv,
            tools=MY_TOOLS,
        )
        msg = resp.choices[0].message
        conv.append(msg.model_dump())    # append raw dict back to history
        return msg.content or ""

Tool-calling round-trip::

    # After the model returns tool_calls, run the tools and append results:
    for tc in msg.tool_calls:
        result = run_tool(tc.function.name, tc.function.arguments)
        conv.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
    # Then call chat() again — LCM will compress across tool boundaries.
"""

from __future__ import annotations

import json
from typing import Any


def _flatten_content(content: Any) -> str:
    """Flatten OpenAI list-content (vision / multi-part) to a plain string."""
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
                    # image_url, audio, etc. — keep as JSON so it survives round-trip
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


class OpenAIMessages:
    """Convert between OpenAI message dicts and LCM internal format.

    All methods are static — no instantiation needed::

        lcm  = OpenAIMessages.to_lcm(openai_messages)
        back = OpenAIMessages.from_lcm(lcm_messages)

    LCM internal format
    -------------------
    Plain message::

        {"role": "user"|"assistant"|"system", "content": "string"}

    Assistant message with tool calls (content is JSON)::

        {
            "role": "assistant",
            "content": '{"text":"...","tool_calls":[{"id":"...","name":"...","args":{}}]}'
        }

    Tool result::

        {"role": "tool", "content": "string", "tool_call_id": "string", "name": "string"}
    """

    @staticmethod
    def to_lcm(messages: list[dict]) -> list[dict]:
        """Convert a list of OpenAI-format message dicts to LCM internal format.

        Args:
            messages: OpenAI ``chat.completions.create(messages=...)`` list.

        Returns:
            List of LCM internal dicts ready for ``engine.compress()``.
        """
        result: list[dict] = []

        for m in messages:
            role = (m.get("role") or "user").lower()

            # ── assistant ────────────────────────────────────────────────────
            if role == "assistant":
                tool_calls_raw = m.get("tool_calls") or []
                if tool_calls_raw:
                    normalized: list[dict] = []
                    for tc in tool_calls_raw:
                        # tc may be a dict or an openai SDK object
                        tc_dict = tc if isinstance(tc, dict) else (tc.model_dump() if hasattr(tc, "model_dump") else vars(tc))
                        fn = tc_dict.get("function") or {}
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except (ValueError, TypeError):
                            args = {"_raw": raw_args}
                        normalized.append({
                            "id":   tc_dict.get("id", ""),
                            "name": fn.get("name", ""),
                            "args": args,
                            "type": tc_dict.get("type", "function"),
                        })
                    content_str = json.dumps(
                        {"text": _flatten_content(m.get("content")), "tool_calls": normalized},
                        ensure_ascii=False,
                    )
                    result.append({"role": "assistant", "content": content_str})
                else:
                    result.append({"role": "assistant", "content": _flatten_content(m.get("content"))})

            # ── tool result ──────────────────────────────────────────────────
            elif role == "tool":
                result.append({
                    "role":         "tool",
                    "content":      _flatten_content(m.get("content")),
                    "tool_call_id": m.get("tool_call_id", ""),
                    "name":         m.get("name", ""),
                })

            # ── function (legacy) ────────────────────────────────────────────
            elif role == "function":
                result.append({
                    "role":         "tool",
                    "content":      _flatten_content(m.get("content")),
                    "tool_call_id": "",
                    "name":         m.get("name", ""),
                })

            # ── user / system / developer ────────────────────────────────────
            else:
                result.append({
                    "role":    role,
                    "content": _flatten_content(m.get("content")),
                })

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> list[dict]:
        """Convert LCM internal format back to OpenAI-compatible message dicts.

        Args:
            messages: List of LCM internal dicts (output of ``engine.compress()``).

        Returns:
            List of dicts suitable for ``client.chat.completions.create(messages=...)``.
        """
        result: list[dict] = []

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")

            # ── assistant ────────────────────────────────────────────────────
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        tool_calls = []
                        for tc in parsed["tool_calls"]:
                            tool_calls.append({
                                "id":   tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name":      tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                                },
                            })
                        result.append({
                            "role":       "assistant",
                            "content":    parsed.get("text") or None,
                            "tool_calls": tool_calls,
                        })
                        continue
                except (ValueError, TypeError):
                    pass
                result.append({"role": "assistant", "content": content})

            # ── tool result ──────────────────────────────────────────────────
            elif role == "tool":
                result.append({
                    "role":         "tool",
                    "tool_call_id": m.get("tool_call_id", ""),
                    "name":         m.get("name", ""),
                    "content":      content,
                })

            # ── everything else ──────────────────────────────────────────────
            else:
                result.append({"role": role, "content": content})

        return result
