"""Prefix Canonicalizer and KV-Cache Normalizer.

Transforms non-deterministic system prompts, tool schemas, and chat histories
into byte-deterministic prefixes for the provider's prefix cache.
Allowed dependencies per wiring.yaml: ["core"].
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.canonicalizer.ast_tools import PromptASTSegmenter
from src.core.models import (
    ChatCompletionRequest,
    ChatMessage,
    PrefixMetrics,
    ToolDefinition,
)


def canonicalize_json_value(val: Any) -> Any:
    """Recursively canonicalize dictionaries and lists for stable serialization."""
    if isinstance(val, dict):
        return {k: canonicalize_json_value(v) for k, v in sorted(val.items())}
    if isinstance(val, list):
        return [canonicalize_json_value(item) for item in val]
    return val


def canonical_json_dump(data: Any) -> str:
    """Serialize data into deterministic, whitespace-compact JSON with sorted keys."""
    canonical_obj = canonicalize_json_value(data)
    return json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_whitespace_and_newlines(text: str) -> str:
    """Normalize CRLF to LF, strip trailing whitespace per line, and collapse excessive blank lines."""
    if not text:
        return ""
    # Normalize newlines
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on each line
    lines = [re.sub(r"[ \t]+$", "", line) for line in normalized.split("\n")]
    result = "\n".join(lines)
    # Collapse 3 or more consecutive newlines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def canonicalize_tool_schema(tool: ToolDefinition | dict[str, Any]) -> dict[str, Any]:
    """Deterministically order and clean an individual tool function schema."""
    if isinstance(tool, ToolDefinition):
        tool_dict = tool.model_dump(exclude_none=True)
    else:
        tool_dict = dict(tool)

    fn = tool_dict.get("function")
    if isinstance(fn, dict):
        clean_fn = dict(fn)
        fn_name = clean_fn.get("name", "").strip()
        fn_desc = clean_fn.get("description", "")
        if isinstance(fn_desc, str):
            fn_desc = fn_desc.strip()

        params = clean_fn.get("parameters")
        if isinstance(params, dict):
            clean_params = dict(params)
            if "required" in clean_params and isinstance(clean_params["required"], list):
                clean_params["required"] = sorted(clean_params["required"])
            clean_fn["parameters"] = canonicalize_json_value(clean_params)

        clean_fn["name"] = fn_name
        if fn_desc:
            clean_fn["description"] = fn_desc
        tool_dict["function"] = clean_fn

    # Recursively sort all dictionary keys across every nested schema block
    return canonicalize_json_value(tool_dict)


def canonicalize_tools(
    tools: list[ToolDefinition] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Deterministically sort all tool definitions alphabetically by function name.

    Eliminates KV-cache thrashing caused by arbitrary tool array ordering.
    """
    if not tools:
        return []

    processed = [canonicalize_tool_schema(t) for t in tools]
    # Sort strictly by tool function name, then by canonical JSON
    return sorted(
        processed,
        key=lambda t: (
            t.get("function", {}).get("name", ""),
            canonical_json_dump(t),
        ),
    )


def canonicalize_message(msg: ChatMessage | dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a single chat message, ordering keys and preserving message content."""
    if isinstance(msg, ChatMessage):
        msg_dict = msg.model_dump(exclude_none=True)
    else:
        msg_dict = dict(msg)

    # Normalize role
    role = msg_dict.get("role", "user").lower()
    msg_dict["role"] = role

    # NOTE: Do NOT strip or collapse whitespace on non-system message content (user, assistant, tool).
    # Preserves code indentation, Markdown hard line breaks ('  \n'), and unified diff context lines (' ').
    content = msg_dict.get("content")
    if isinstance(content, str):
        normalized_content = content.replace("\r\n", "\n")
        if role == "system":
            # For static system instructions, normalize trailing whitespace per line to ensure prefix cache invariance
            lines = [line.rstrip() for line in normalized_content.splitlines()]
            msg_dict["content"] = "\n".join(lines).strip()
        else:
            msg_dict["content"] = normalized_content

    # Canonicalize tool calls if present (e.g. in assistant messages)
    tool_calls = msg_dict.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        clean_calls = []
        for tc in tool_calls:
            tc_dict = dict(tc) if isinstance(tc, dict) else tc.model_dump(exclude_none=True)
            fn = dict(tc_dict.get("function", {}))
            raw_args = fn.get("arguments", "")
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    fn["arguments"] = json.dumps(
                        canonicalize_json_value(parsed),
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Preserve raw string untouched if invalid JSON or contains NaN/Infinity
                    fn["arguments"] = raw_args
            tc_dict["function"] = fn
            clean_calls.append(tc_dict)
        msg_dict["tool_calls"] = clean_calls

    return msg_dict


def compute_prefix_hash(
    system_prompt: str,
    canonical_tools: list[dict[str, Any]],
    first_user_turn: str | None = None,
) -> str:
    """Computes a 64-character SHA-256 fingerprint of the invariant prefix.

    This digest serves as the cache alignment key sent in gateway telemetry.
    """
    hasher = hashlib.sha256()
    hasher.update(b"RUNSIFT_PREFIX_V1:")
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(b"::TOOLS::")
    hasher.update(canonical_json_dump(canonical_tools).encode("utf-8"))
    if first_user_turn:
        hasher.update(b"::INIT_USER::")
        hasher.update(first_user_turn.encode("utf-8"))
    return hasher.hexdigest()


class PrefixCanonicalizer:
    """High-throughput prefix canonicalizer engine for OpenAI-compatible requests."""

    def __init__(self, hoist_dynamic_system_context: bool = True):
        self.hoist_dynamic_system_context = hoist_dynamic_system_context
        self.segmenter = PromptASTSegmenter()

    def canonicalize_request(
        self,
        request: ChatCompletionRequest | dict[str, Any],
    ) -> tuple[dict[str, Any], PrefixMetrics]:
        """Processes request payload into cache-aligned format and computes telemetry."""
        if isinstance(request, ChatCompletionRequest):
            payload = request.model_dump(exclude_none=True)
        else:
            payload = dict(request)

        # 1. Canonicalize Tools
        raw_tools = payload.get("tools")
        canonical_tools = canonicalize_tools(raw_tools)
        if canonical_tools:
            payload["tools"] = canonical_tools
        elif "tools" in payload and not canonical_tools:
            payload.pop("tools", None)

        # 2. Canonicalize Messages
        raw_messages = payload.get("messages", [])
        canonical_messages: list[dict[str, Any]] = []
        system_prompt = ""
        first_user_turn = ""

        hoisted_dynamic_suffix = ""
        for _idx, m in enumerate(raw_messages):
            clean_m = canonicalize_message(m)
            role = clean_m.get("role")
            if role == "system" and not system_prompt:
                raw_content = clean_m.get("content", "") or ""
                if self.hoist_dynamic_system_context:
                    segmented = self.segmenter.segment_prompt(raw_content)
                    clean_m["content"] = segmented.static_prefix
                    system_prompt = segmented.static_prefix
                    hoisted_dynamic_suffix = segmented.dynamic_suffix
                else:
                    system_prompt = raw_content
            elif role == "user" and not first_user_turn:
                first_user_turn = clean_m.get("content", "") or ""
            canonical_messages.append(clean_m)

        if hoisted_dynamic_suffix and canonical_messages:
            # Target the latest user message so assistant prefills and tool returns remain clean
            target_msg = None
            for m in reversed(canonical_messages):
                if m.get("role") == "user":
                    target_msg = m
                    break
            if target_msg is None:
                target_msg = canonical_messages[-1]

            content = target_msg.get("content")
            if isinstance(content, str):
                target_msg["content"] = (
                    f"{content}\n\n[Runtime Context]\n{hoisted_dynamic_suffix}".strip()
                )
            elif isinstance(content, list):
                # Multimodal list: append text dictionary, preserving image_url / audio items bit-for-bit
                target_msg["content"] = list(content) + [
                    {"type": "text", "text": f"\n\n[Runtime Context]\n{hoisted_dynamic_suffix}"}
                ]

        payload["messages"] = canonical_messages

        # 3. Compute Prefix Fingerprint & Metrics on static invariant prefix
        p_hash = compute_prefix_hash(system_prompt, canonical_tools)

        # Rough token estimation (1 token ~= 4 chars)
        prefix_chars = len(system_prompt) + len(canonical_json_dump(canonical_tools))
        estimated_tokens = max(1, prefix_chars // 4)

        metrics = PrefixMetrics(
            prefix_hash=p_hash,
            canonical_tokens_estimated=estimated_tokens,
            tools_count=len(canonical_tools),
            system_prompt_length=len(system_prompt),
            turns_count=len(canonical_messages),
            is_cache_aligned=True,
            divergence_turn_index=None,
        )

        return payload, metrics
