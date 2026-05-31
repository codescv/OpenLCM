"""Anthropic message converter for LCM.

Handles Anthropic Claude's message format, including multi-part content blocks,
tool_use / tool_result pairs, and the separate system parameter.

Install: pip install anthropic

Quick start::

    import anthropic
    from openlcm import LCMEngine
    from openlcm.adapters.anthropic import AnthropicMessages

    client = anthropic.AsyncAnthropic()
    engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")
    engine.bind_session("my-session", context_length=200_000)

    system = "You are a helpful assistant."
    conv   = []   # only user / assistant messages (no system in Anthropic conv list)

    async def chat(user_msg: str) -> str:
        conv.append({"role": "user", "content": user_msg})

        # Anthropic keeps system separate — AnthropicMessages handles that
        lcm_msgs = AnthropicMessages.to_lcm(conv, system=system)

        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs = await engine.compress(lcm_msgs)
            system_out, conv[:] = AnthropicMessages.from_lcm(lcm_msgs)
            # system_out is the same system string passed in (preserved through LCM)

        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system,
            messages=conv,
            tools=MY_TOOLS,
        )

        # Append assistant message from response
        conv.append({"role": "assistant", "content": resp.content})
        return next((b.text for b in resp.content if hasattr(b, "text")), "")

Tool-calling::

    # After the model returns tool_use blocks, execute them and append results:
    for block in resp.content:
        if block.type == "tool_use":
            result = run_tool(block.name, block.input)
            conv.append({
                "role": "user",
                "content": [{
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     [{"type": "text", "text": json.dumps(result)}],
                }],
            })
    # Then call chat() again.

Notes
-----
- Anthropic does NOT use a ``system`` role in the messages list.  Pass it via
  ``AnthropicMessages.to_lcm(conv, system="...")`` and recover it with the
  second element of ``AnthropicMessages.from_lcm()``'s return value.
- ``from_lcm()`` returns ``(system_str, messages_list)`` — a 2-tuple — because
  the Anthropic API requires system as a separate top-level parameter.
- Tool results in Anthropic go back as user-role messages with ``tool_result``
  content blocks.  The converter handles this automatically.
"""

from __future__ import annotations

import json
from typing import Any


def _text_from_block(block: Any) -> str:
    """Extract plain text from an Anthropic content block (dict or SDK object)."""
    if isinstance(block, dict):
        return block.get("text", "")
    return getattr(block, "text", "") or ""


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "") or ""


class AnthropicMessages:
    """Convert between Anthropic SDK message format and LCM internal format.

    All methods are static — no instantiation needed::

        lcm                  = AnthropicMessages.to_lcm(messages, system="You are...")
        system_str, messages = AnthropicMessages.from_lcm(lcm_messages)

    LCM internal format
    -------------------
    See ``openlcm.adapters.openai.OpenAIMessages`` for the canonical definition.
    Anthropic-specific mappings:

    - ``tool_use`` content block  → assistant message with JSON ``tool_calls``
    - ``tool_result`` user block  → ``{"role":"tool", ...}`` message
    - Multi-part text blocks      → joined with newline into a single string
    - System message extracted to the ``system=`` kwarg of ``to_lcm()``
    """

    @staticmethod
    def to_lcm(messages: list, system: str = "") -> list[dict]:
        """Convert Anthropic messages to LCM internal format.

        Args:
            messages: The ``messages`` list passed to ``client.messages.create()``.
                      Each entry is either a dict or an Anthropic SDK Message object.
            system:   The ``system`` parameter from ``client.messages.create()``.
                      If provided, it is prepended as a ``{"role":"system"}`` entry.

        Returns:
            LCM internal message list.
        """
        result: list[dict] = []

        if system:
            result.append({"role": "system", "content": str(system)})

        for m in messages:
            role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")) or "user"
            content_raw = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")

            # ── Simple string content ────────────────────────────────────────
            if isinstance(content_raw, str):
                result.append({"role": str(role), "content": content_raw})
                continue

            # ── SDK Message / ContentBlock objects ───────────────────────────
            if not isinstance(content_raw, list):
                # e.g. Anthropic SDK response object — try to get .content
                content_raw = getattr(content_raw, "content", []) or []

            # ── List of content blocks ───────────────────────────────────────
            if role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict] = []

                for block in content_raw:
                    btype = _block_type(block)
                    if btype == "text":
                        text_parts.append(_text_from_block(block))
                    elif btype == "tool_use":
                        bid   = (block.get("id")    if isinstance(block, dict) else getattr(block, "id",    "")) or ""
                        bname = (block.get("name")  if isinstance(block, dict) else getattr(block, "name",  "")) or ""
                        binp  = (block.get("input") if isinstance(block, dict) else getattr(block, "input", {})) or {}
                        tool_calls.append({"id": bid, "name": bname, "args": dict(binp), "type": "function"})

                if tool_calls:
                    content_str = json.dumps(
                        {"text": "\n".join(text_parts), "tool_calls": tool_calls},
                        ensure_ascii=False,
                    )
                else:
                    content_str = "\n".join(text_parts)
                result.append({"role": "assistant", "content": content_str})

            elif role == "user":
                text_parts = []
                tool_results: list[dict] = []

                for block in content_raw:
                    btype = _block_type(block)
                    if btype == "text":
                        text_parts.append(_text_from_block(block))
                    elif btype == "tool_result":
                        tr_id   = (block.get("tool_use_id") if isinstance(block, dict) else getattr(block, "tool_use_id", "")) or ""
                        tr_cont = (block.get("content")     if isinstance(block, dict) else getattr(block, "content",     "")) or ""
                        # content may itself be a list of text blocks
                        if isinstance(tr_cont, list):
                            tr_text = "\n".join(
                                _text_from_block(b) for b in tr_cont if _block_type(b) == "text"
                            )
                        elif isinstance(tr_cont, str):
                            tr_text = tr_cont
                        else:
                            tr_text = str(tr_cont)
                        tool_results.append({
                            "role":         "tool",
                            "content":      tr_text,
                            "tool_call_id": tr_id,
                            "name":         "",
                        })

                # Tool results come first; any remaining text becomes a user message
                result.extend(tool_results)
                if text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})

            else:
                # system role inside messages list (uncommon but handle it)
                text = "\n".join(
                    _text_from_block(b) for b in content_raw if _block_type(b) == "text"
                ) or str(content_raw)
                result.append({"role": str(role), "content": text})

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert LCM internal format to Anthropic-ready messages.

        Args:
            messages: LCM internal message list (from ``engine.compress()``).

        Returns:
            ``(system_str, anthropic_messages)`` where:
            - ``system_str``        is the value for ``client.messages.create(system=...)``.
            - ``anthropic_messages`` is the value for ``client.messages.create(messages=...)``.

        Usage::

            system, msgs = AnthropicMessages.from_lcm(lcm_messages)
            resp = await client.messages.create(system=system, messages=msgs, ...)
        """
        system = ""
        result: list[dict] = []
        pending_tool_results: list[dict] = []

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")

            # ── system ───────────────────────────────────────────────────────
            if role == "system":
                system = content
                continue

            # Flush accumulated tool_result blocks when a non-tool message arrives
            if pending_tool_results and role != "tool":
                result.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

            # ── assistant ────────────────────────────────────────────────────
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        blocks: list[dict] = []
                        text = parsed.get("text", "")
                        if text:
                            blocks.append({"type": "text", "text": text})
                        for tc in parsed["tool_calls"]:
                            blocks.append({
                                "type":  "tool_use",
                                "id":    tc.get("id", ""),
                                "name":  tc.get("name", ""),
                                "input": tc.get("args", {}),
                            })
                        result.append({"role": "assistant", "content": blocks})
                        continue
                except (ValueError, TypeError):
                    pass
                result.append({"role": "assistant", "content": content})

            # ── tool result ──────────────────────────────────────────────────
            elif role == "tool":
                # Anthropic wraps tool_result blocks inside a user-role message
                pending_tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content":     [{"type": "text", "text": content}],
                })

            # ── user ─────────────────────────────────────────────────────────
            elif role == "user":
                result.append({"role": "user", "content": content})

            else:
                result.append({"role": role, "content": content})

        # Flush any remaining tool_results
        if pending_tool_results:
            result.append({"role": "user", "content": pending_tool_results})

        return system, result
