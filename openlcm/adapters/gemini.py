"""Google Gemini message converter for LCM.

Handles the native ``google.genai.types.Content`` format used by:
- google-genai SDK (direct)
- Google ADK before_model_callback (LlmRequest.contents)
- Vertex AI SDK

Install: pip install google-genai   (or google-adk, which includes it)

Quick start — compress in an ADK before_model_callback::

    from google.adk.agents import LlmAgent
    from openlcm import LCMEngine
    from openlcm.adapters.gemini import GeminiMessages

    engine = LCMEngine(model="gemini/gemini-2.0-flash")
    engine.bind_session("my-session", context_length=1_000_000)

    async def lcm_compress(callback_context, llm_request):
        lcm_msgs = GeminiMessages.to_lcm(llm_request.contents, system=llm_request.system_instruction)
        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs          = await engine.compress(lcm_msgs)
            sys_out, contents = GeminiMessages.from_lcm(lcm_msgs)
            llm_request.contents = contents
            if sys_out:
                from google.genai import types
                llm_request.system_instruction = types.Content(
                    parts=[types.Part(text=sys_out)]
                )
        return None   # return None to continue with the (now compressed) request

    agent = LlmAgent(
        name="my_agent",
        model="gemini-2.0-flash",
        before_model_callback=lcm_compress,
        tools=[...],
    )

Stand-alone usage with raw google-genai SDK::

    import google.generativeai as genai
    from openlcm.adapters.gemini import GeminiMessages

    client = genai.GenerativeModel("gemini-2.0-flash")
    history = []   # list[types.Content]

    async def chat(user_msg: str) -> str:
        from google.genai import types
        history.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))

        lcm_msgs = GeminiMessages.to_lcm(history)
        if engine.should_compress_preflight(lcm_msgs):
            lcm_msgs = await engine.compress(lcm_msgs)
            _, history[:] = GeminiMessages.from_lcm(lcm_msgs)

        response = client.generate_content(history)
        history.append(response.candidates[0].content)
        return response.text
"""

from __future__ import annotations

import json
from typing import Any, Optional


def _part_text(part: Any) -> str:
    if isinstance(part, dict):
        return part.get("text", "")
    return getattr(part, "text", "") or ""


def _part_type(part: Any) -> str:
    if isinstance(part, dict):
        for k in ("function_call", "function_response", "text", "inline_data"):
            if k in part and part[k] is not None:
                return k
        return "unknown"
    for attr in ("function_call", "function_response"):
        if getattr(part, attr, None) is not None:
            return attr
    if getattr(part, "text", None) is not None:
        return "text"
    if getattr(part, "inline_data", None) is not None:
        return "inline_data"
    return "unknown"


def _fc_dict(fc: Any) -> dict:
    if isinstance(fc, dict):
        return {"id": fc.get("id", fc.get("name", "")), "name": fc.get("name", ""), "args": dict(fc.get("args", {}))}
    return {
        "id":   getattr(fc, "id",   getattr(fc, "name", "")) or "",
        "name": getattr(fc, "name", "") or "",
        "args": dict(getattr(fc, "args", {}) or {}),
    }


def _fr_dict(fr: Any) -> dict:
    if isinstance(fr, dict):
        return {"id": fr.get("id", fr.get("name", "")), "name": fr.get("name", ""), "response": dict(fr.get("response", {}))}
    return {
        "id":       getattr(fr, "id",       getattr(fr, "name", "")) or "",
        "name":     getattr(fr, "name",     "") or "",
        "response": dict(getattr(fr, "response", {}) or {}),
    }


def _content_role(content: Any) -> str:
    if isinstance(content, dict):
        return (content.get("role") or "user").lower()
    return (getattr(content, "role", "user") or "user").lower()


def _content_parts(content: Any) -> list:
    if isinstance(content, dict):
        return content.get("parts", [])
    return list(getattr(content, "parts", []) or [])


class GeminiMessages:
    """Convert between Gemini ``types.Content`` objects and LCM internal format.

    Handles:
    - Plain text parts
    - ``function_call`` parts  (model requesting tool execution)
    - ``function_response`` parts (tool results sent back to model)
    - Multi-part contents (text + function_call in same turn)
    - System instructions (extracted / re-injected separately)
    - Inline data / images are preserved as JSON so they survive round-trips

    All methods are static::

        lcm_msgs           = GeminiMessages.to_lcm(contents, system="Be helpful.")
        system_str, contents = GeminiMessages.from_lcm(lcm_msgs)

    Return convention
    -----------------
    ``from_lcm`` returns ``(system_str, contents_list)`` because Gemini
    uses system as a separate parameter (same convention as AnthropicMessages).
    """

    @staticmethod
    def to_lcm(contents: list, system: Any = None) -> list[dict]:
        """Convert a list of Gemini Content objects to LCM internal format.

        Args:
            contents: ``list[types.Content]`` — the conversation history.
            system:   Optional system instruction (``types.Content`` or ``str``).
                      If provided, prepended as ``{"role":"system",...}``.

        Returns:
            LCM internal message list.
        """
        result: list[dict] = []

        # Prepend system instruction if provided
        if system is not None:
            if isinstance(system, str):
                if system:
                    result.append({"role": "system", "content": system})
            else:
                # types.Content object
                sys_parts = _content_parts(system)
                sys_text = " ".join(_part_text(p) for p in sys_parts if _part_type(p) == "text")
                if sys_text:
                    result.append({"role": "system", "content": sys_text})

        for content in contents:
            role  = _content_role(content)
            # Gemini uses "model" for assistant turns
            if role == "model":
                role = "assistant"

            parts = _content_parts(content)
            if not parts:
                continue

            text_parts:   list[str] = []
            tool_calls:   list[dict] = []
            tool_results: list[dict] = []

            for part in parts:
                ptype = _part_type(part)
                if ptype == "text":
                    t = _part_text(part)
                    if t:
                        text_parts.append(t)
                elif ptype == "function_call":
                    fc = _fc_dict(part.get("function_call") if isinstance(part, dict) else part.function_call)
                    tool_calls.append({
                        "id":   fc["id"] or fc["name"],
                        "name": fc["name"],
                        "args": fc["args"],
                        "type": "function",
                    })
                elif ptype == "function_response":
                    fr = _fr_dict(part.get("function_response") if isinstance(part, dict) else part.function_response)
                    tool_results.append({
                        "role":         "tool",
                        "content":      json.dumps(fr["response"], ensure_ascii=False),
                        "tool_call_id": fr["id"] or fr["name"],
                        "name":         fr["name"],
                    })
                elif ptype == "inline_data":
                    # Preserve image/blob as JSON placeholder so it survives round-trip
                    if isinstance(part, dict):
                        text_parts.append(f"[inline_data: {json.dumps(part.get('inline_data',{}), ensure_ascii=False)}]")
                    else:
                        text_parts.append("[inline_data]")

            # function_response parts → tool messages
            result.extend(tool_results)

            # Text + optional tool_calls → single message
            text = "\n".join(text_parts)
            if tool_calls:
                content_str = json.dumps({"text": text, "tool_calls": tool_calls}, ensure_ascii=False)
                result.append({"role": "assistant", "content": content_str})
            elif text:
                result.append({"role": role, "content": text})

        return result

    @staticmethod
    def from_lcm(messages: list[dict]) -> tuple[str, list]:
        """Convert LCM internal format back to Gemini Content objects.

        Args:
            messages: LCM internal message list (from ``engine.compress()``).

        Returns:
            ``(system_str, contents)`` where:
            - ``system_str`` — the text to pass to ``GenerativeModel(system_instruction=...)``.
            - ``contents``   — ``list[types.Content]`` for the messages parameter.

        Usage::

            system, contents = GeminiMessages.from_lcm(lcm_messages)
            model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system)
            response = model.generate_content(contents)
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai is required. Install with: pip install google-genai")

        system = ""
        contents: list = []
        pending_fr: list[Any] = []   # accumulated function_response parts

        def _flush_fr():
            if pending_fr:
                contents.append(types.Content(role="user", parts=list(pending_fr)))
                pending_fr.clear()

        for m in messages:
            role    = (m.get("role") or "user").lower()
            content = m.get("content", "")

            if role == "system":
                system = content
                continue

            # Flush pending function_response when we hit a non-tool message
            if pending_fr and role != "tool":
                _flush_fr()

            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        parts: list[Any] = []
                        if parsed.get("text"):
                            parts.append(types.Part(text=parsed["text"]))
                        for tc in parsed["tool_calls"]:
                            parts.append(types.Part(
                                function_call=types.FunctionCall(
                                    id=tc.get("id", ""),
                                    name=tc.get("name", ""),
                                    args=tc.get("args", {}),
                                )
                            ))
                        contents.append(types.Content(role="model", parts=parts))
                        continue
                except (ValueError, TypeError):
                    pass
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

            elif role == "tool":
                try:
                    response = json.loads(m["content"]) if m["content"].startswith("{") or m["content"].startswith("[") else {"result": m["content"]}
                except (ValueError, AttributeError, KeyError):
                    response = {"result": m.get("content", "")}
                pending_fr.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=m.get("tool_call_id", ""),
                        name=m.get("name", ""),
                        response=response,
                    )
                ))

            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))

            else:
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))

        _flush_fr()
        return system, contents
