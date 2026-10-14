"""Unit and invariant tests for RunSift Canonicalizer subsystem.

Verifies deterministic tool sorting, whitespace normalization, prompt AST reordering,
and prefix hash stability across multi-turn permutations.
"""

from src.canonicalizer.ast_tools import PromptASTSegmenter
from src.canonicalizer.normalizer import (
    PrefixCanonicalizer,
    canonical_json_dump,
    canonicalize_tools,
    normalize_whitespace_and_newlines,
)
from src.core.models import (
    ChatCompletionRequest,
    ChatMessage,
    ToolDefinition,
    ToolFunctionSchema,
)


def test_tool_sorting_deterministic():
    """Verify that tools passed in arbitrary orders produce bitwise identical canonical outputs."""
    tool_z = {
        "type": "function",
        "function": {
            "name": "zip_files",
            "description": "Zips up files",
            "parameters": {
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["output", "files"],
            },
        },
    }

    tool_a = {
        "type": "function",
        "function": {
            "name": "analyze_repo",
            "description": "Performs code analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "depth": {"type": "integer"},
                },
                "required": ["repo_url"],
            },
        },
    }

    tool_m = {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Stores state in memory",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["value", "key"],
            },
        },
    }

    # Permutation 1: [z, m, a]
    order_1 = canonicalize_tools([tool_z, tool_m, tool_a])
    # Permutation 2: [a, z, m]
    order_2 = canonicalize_tools([tool_a, tool_z, tool_m])
    # Permutation 3: [m, a, z]
    order_3 = canonicalize_tools([tool_m, tool_a, tool_z])

    # Assert names are strictly sorted alphabetically: analyze_repo, memory_store, zip_files
    names_1 = [t["function"]["name"] for t in order_1]
    assert names_1 == ["analyze_repo", "memory_store", "zip_files"]

    # Assert serialized strings are bitwise identical across all permutations
    assert (
        canonical_json_dump(order_1) == canonical_json_dump(order_2) == canonical_json_dump(order_3)
    )

    # Assert required array is sorted: ["key", "value"] instead of ["value", "key"]
    mem_tool = next(t for t in order_1 if t["function"]["name"] == "memory_store")
    assert mem_tool["function"]["parameters"]["required"] == ["key", "value"]


def test_tool_arguments_canonicalization():
    """Verify tool call arguments JSON strings are re-ordered into canonical key ordering."""
    canonicalizer = PrefixCanonicalizer()

    # Request with assistant tool_call whose arguments have keys {"query": "test", "limit": 5}
    req1 = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "You are a code agent."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test", "limit": 5}',
                        },
                    }
                ],
            },
        ],
    }

    # Identical semantic request, but with reversed JSON keys: {"limit": 5, "query": "test"}
    req2 = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "You are a code agent."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"limit": 5, "query": "test"}',
                        },
                    }
                ],
            },
        ],
    }

    res1, metrics1 = canonicalizer.canonicalize_request(req1)
    res2, metrics2 = canonicalizer.canonicalize_request(req2)

    # Tool call arguments must be identical after canonicalization
    call_args_1 = res1["messages"][1]["tool_calls"][0]["function"]["arguments"]
    call_args_2 = res2["messages"][1]["tool_calls"][0]["function"]["arguments"]

    assert call_args_1 == '{"limit":5,"query":"test"}'
    assert call_args_1 == call_args_2
    assert metrics1.prefix_hash == metrics2.prefix_hash


def test_whitespace_and_crlf_normalization():
    """Verify that CRLF line breaks and irregular trailing whitespaces are flattened."""
    messy_text = "Line 1   \r\nLine 2\t\r\n\r\n\r\n\r\nLine 3  \r\n"
    clean = normalize_whitespace_and_newlines(messy_text)
    assert clean == "Line 1\nLine 2\n\nLine 3"


def test_ast_prompt_segmentation_hoists_static_prefix():
    """Verify that dynamic session headers at the beginning of system prompts are hoisted."""
    segmenter = PromptASTSegmenter()

    prompt_with_timestamp = (
        "Current Date: 2026-09-19 18:40:00 UTC\n"
        "Session ID: sess-8914-alpha\n"
        "You are an NVIDIA Nemotron expert autonomous coding assistant.\n"
        "Always follow constitutional invariants and produce high performance code."
    )

    segmented = segmenter.segment_prompt(prompt_with_timestamp)

    assert segmented.is_reordered is True
    # The static prefix must start with the invariant instructions, not the timestamp
    assert segmented.static_prefix.startswith("You are an NVIDIA Nemotron expert")
    assert "Current Date" in segmented.dynamic_suffix
    assert "Session ID" in segmented.dynamic_suffix

    # Check unified prompt preserves both
    unified = segmented.unified_prompt
    assert "You are an NVIDIA Nemotron expert" in unified
    assert "[Runtime Context]" in unified
    assert "Current Date" in unified


def test_prefix_hash_invariance():
    """Verify prefix hash matches across variations that normalize to the same canonical representation."""
    canonicalizer = PrefixCanonicalizer()

    req_a = ChatCompletionRequest(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            ChatMessage(role="system", content="System invariant prompt.  \r\nSecond line."),
            ChatMessage(role="user", content="Build an agent proxy."),
        ],
        tools=[
            ToolDefinition(
                function=ToolFunctionSchema(
                    name="tool_b",
                    description="Tool B",
                    parameters={"type": "object", "properties": {"b": {"type": "string"}}},
                )
            ),
            ToolDefinition(
                function=ToolFunctionSchema(
                    name="tool_a",
                    description="Tool A",
                    parameters={"type": "object", "properties": {"a": {"type": "string"}}},
                )
            ),
        ],
    )

    req_b = ChatCompletionRequest(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            ChatMessage(role="system", content="System invariant prompt.\nSecond line."),
            ChatMessage(role="user", content="Build an agent proxy."),
        ],
        tools=[
            ToolDefinition(
                function=ToolFunctionSchema(
                    name="tool_a",
                    description="Tool A",
                    parameters={"type": "object", "properties": {"a": {"type": "string"}}},
                )
            ),
            ToolDefinition(
                function=ToolFunctionSchema(
                    name="tool_b",
                    description="Tool B",
                    parameters={"type": "object", "properties": {"b": {"type": "string"}}},
                )
            ),
        ],
    )

    _, metrics_a = canonicalizer.canonicalize_request(req_a)
    _, metrics_b = canonicalizer.canonicalize_request(req_b)

    assert metrics_a.prefix_hash == metrics_b.prefix_hash
    assert len(metrics_a.prefix_hash) == 64
    assert metrics_a.is_cache_aligned is True
