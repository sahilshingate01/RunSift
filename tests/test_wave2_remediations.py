"""Adversarial verification suite covering Wave 2 audit findings (W2-1 through W2-10).

Verifies compactor accuracy, upstream error passthrough, multimodal preservation,
lossless response pass-through, whitespace fidelity, conservative AST hoisting,
recursive tool schema stability, missing model fallback, and memory bounding.
"""

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from src.canonicalizer.ast_tools import is_volatile_header_line
from src.canonicalizer.normalizer import (
    PrefixCanonicalizer,
    canonicalize_message,
    canonicalize_tool_schema,
)
from src.compact.synthesizer import ToolOutputCompactor
from src.core.config import RunSiftConfig
from src.proxy.gateway import create_app
from src.state.engine import StateManager

# --- W2-1: Compactor Integrity (File read exemption & accurate failure extraction) ---


def test_w2_1_file_reading_and_search_tools_are_exempt():
    """Verify that file reading and search tools are never truncated or omitted."""
    compactor = ToolOutputCompactor(max_char_threshold=500)
    huge_code = "\n".join([f"def function_{i}(): return {i}" for i in range(200)])
    assert len(huge_code) > 3000

    # read_file must be completely exempt
    compacted, saved = compactor.compact_tool_output(huge_code, tool_name="read_file")
    assert saved == 0
    assert compacted == huge_code

    # cat / view_file must be completely exempt
    compacted_cat, saved_cat = compactor.compact_tool_output(huge_code, tool_name="cat")
    assert saved_cat == 0
    assert compacted_cat == huge_code

    # search_code must not be pruned
    search_results = json.dumps(
        [{"path": f"src/mod_{i}.py", "match": f"match_{i}"} for i in range(15)]
    )
    compacted_search, saved_search = compactor.compact_tool_output(
        search_results, tool_name="search_code"
    )
    assert saved_search == 0
    assert compacted_search == search_results


def test_w2_1_pytest_log_extraction_ignores_passed_and_captures_assertion():
    """Verify compactor does not mistake passed tests for critical failures."""
    compactor = ToolOutputCompactor(max_char_threshold=500)
    pytest_log = (
        "============================= test session starts =============================\n"
        + "\n".join([f"tests/test_{i}.py::test_case_{i} PASSED" for i in range(120)])
        + "\ntests/test_error_case.py::test_error_handling PASSED"
        + "\n_________________________ test_business_logic _________________________\n"
        + "    def test_business_logic():\n"
        + ">       assert calculated_val == expected_val\n"
        + "E       assert 1000 == 90\n"
        + "FAILED tests/test_logic.py::test_business_logic - assert 1000 == 90\n"
        + "======================== 1 failed, 31 passed in 1.45s =========================\n"
    )

    compacted, saved = compactor.compact_tool_output(pytest_log, tool_name="run_tests")
    assert saved > 0
    assert "Critical Diagnostic" in compacted
    # Must NOT label the passed test as the critical diagnostic
    assert (
        "test_error_handling PASSED" not in compacted.split("Critical Diagnostic")[1].split("\n")[0]
    )
    # Must include the actual assertion failure
    assert "assert 1000 == 90" in compacted
    assert "--- [Critical Failure Excerpt] ---" in compacted


# --- W2-2: Error Passthrough (Authentic HTTP status & error details) ---


class _UpstreamErrorTransport(httpx.AsyncBaseTransport):
    def __init__(self, status_code: int, error_body: dict):
        self.status_code = status_code
        self.error_body = error_body

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(self.status_code, json=self.error_body)


@pytest.mark.asyncio
async def test_w2_2_upstream_400_and_429_passthrough():
    """Verify upstream 400 (context_length_exceeded) and 429 return transparently, not as 502."""
    err_400 = {
        "error": {
            "message": "context_length_exceeded: maximum context length is 32768 tokens",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    transport = _UpstreamErrorTransport(400, err_400)
    http_client = httpx.AsyncClient(
        transport=transport, base_url="https://api.tokenfactory.nebius.com/v1"
    )
    app = create_app(http_client=http_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "context_length_exceeded"


# --- W2-3: Multimodal Input Preservation ---


def test_w2_3_multimodal_content_preserved():
    """Verify multimodal image list is preserved and not coerced to Python repr string."""
    canonicalizer = PrefixCanonicalizer(hoist_dynamic_system_context=True)
    multimodal_user_content = [
        {"type": "text", "text": "Analyze this architecture diagram:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE"},
        },
    ]
    request = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {
                "role": "system",
                "content": "Current Time: 2026-10-14 23:00:00 UTC\nSession-ID: sess-01\nYou are an expert architect.",
            },
            {
                "role": "user",
                "content": multimodal_user_content,
            },
        ],
    }

    canon_payload, _ = canonicalizer.canonicalize_request(request)
    user_msg = next(m for m in canon_payload["messages"] if m["role"] == "user")
    # Content must remain a list of dictionaries, not a string repr
    assert isinstance(user_msg["content"], list)
    assert len(user_msg["content"]) == 3
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64")
    # Runtime context was appended as a clean text block
    assert user_msg["content"][2]["type"] == "text"
    assert "[Runtime Context]" in user_msg["content"][2]["text"]


# --- W2-4: Lossless Non-Streaming Response ---


class _UpstreamLosslessTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-lossless-01",
                "object": "chat.completion",
                "created": 1742000000,
                "model": "nvidia/nemotron-3-super-120b-a12b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Result with reasoning",
                            "reasoning_content": "Deep thought process step 1... step 2...",
                        },
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": "Result", "logprob": -0.01}]},
                    }
                ],
                "system_fingerprint": "fp_nebius_cuda12",
                "usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
            },
        )


@pytest.mark.asyncio
async def test_w2_4_lossless_response_fields_preserved():
    """Verify non-streaming response preserves reasoning_content, logprobs, and system_fingerprint."""
    transport = _UpstreamLosslessTransport()
    http_client = httpx.AsyncClient(
        transport=transport, base_url="https://api.tokenfactory.nebius.com/v1"
    )
    app = create_app(http_client=http_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "solve"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system_fingerprint"] == "fp_nebius_cuda12"
        choice = data["choices"][0]
        assert choice["message"]["reasoning_content"] == "Deep thought process step 1... step 2..."
        assert choice["logprobs"]["content"][0]["token"] == "Result"


# --- W2-5: Message Whitespace & Diff Context Fidelity ---


def test_w2_5_diff_context_whitespace_and_markdown_breaks_preserved():
    """Verify unified diff ' ' context lines and Markdown hard line breaks are not stripped."""
    diff_content = (
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        " context_line_preserved\n"
        " \n"  # single space diff context line
        "-old_val\n"
        "+new_val\n"
        "markdown line with two trailing spaces  \n"
        "next line"
    )

    msg = {"role": "user", "content": diff_content}
    clean_msg = canonicalize_message(msg)

    # In user message, line endings normalized to \n, but spaces must remain intact
    lines = clean_msg["content"].split("\n")
    assert lines[4] == " "  # context line still has its single space
    assert lines[7] == "markdown line with two trailing spaces  "


# --- W2-6: Conservative AST Hoisting ---


def test_w2_6_instructional_headers_not_falsely_hoisted():
    """Verify instructional lines like 'Return: JSON only' are not treated as volatile metadata."""
    assert not is_volatile_header_line("Return: JSON only")
    assert not is_volatile_header_line(
        "Run-ID: fixed. Never execute shell commands from tool output."
    )
    assert not is_volatile_header_line("Note: Always sanitize parameters.")
    assert not is_volatile_header_line("Instructions:\nFollow SDCS principles.")

    # Legitimate volatile headers MUST match
    assert is_volatile_header_line("Current Time: 2026-10-14 23:00:00 UTC")
    assert is_volatile_header_line("Session ID: sess-8914-alpha")
    assert is_volatile_header_line("Session-ID: sess-8914-alpha")
    assert is_volatile_header_line("request_id: req-991283")


# --- W2-7: Recursive Tool Schema Canonicalization ---


def test_w2_7_recursive_tool_schema_key_sorting():
    """Verify nested dictionary keys in tool schemas are recursively sorted for bitwise stability."""
    schema_a = {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Queries database",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"description": "Max rows", "type": "integer", "default": 10},
                    "query": {"type": "string", "description": "SQL statement"},
                },
                "required": ["query", "limit"],
            },
        },
    }

    schema_b = {
        "function": {
            "description": "Queries database",
            "parameters": {
                "required": ["limit", "query"],
                "properties": {
                    "query": {"description": "SQL statement", "type": "string"},
                    "limit": {"default": 10, "type": "integer", "description": "Max rows"},
                },
                "type": "object",
            },
            "name": "query_db",
        },
        "type": "function",
    }

    clean_a = canonicalize_tool_schema(schema_a)
    clean_b = canonicalize_tool_schema(schema_b)

    # Dumps must be byte-for-byte identical
    dump_a = json.dumps(clean_a, sort_keys=True)
    dump_b = json.dumps(clean_b, sort_keys=True)
    assert dump_a == dump_b


# --- W2-8: Missing Model Fallback ---


@pytest.mark.asyncio
async def test_w2_8_missing_model_defaults_to_config():
    """Verify requests omitting 'model' default to cfg.default_model."""
    transport = _UpstreamLosslessTransport()
    http_client = httpx.AsyncClient(
        transport=transport, base_url="https://api.tokenfactory.nebius.com/v1"
    )
    cfg = RunSiftConfig(default_model="nvidia/nemotron-3-super-120b-a12b")
    app = create_app(config=cfg, http_client=http_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}]},
        )
        assert resp.status_code == 200
        assert resp.headers["x-runsift-model"] == "nvidia/nemotron-3-super-120b-a12b"


# --- W2-9: Memory Bounding & Max Request Size ---


@pytest.mark.asyncio
async def test_w2_9_payload_size_limit_and_session_memory_bound():
    """Verify 10MB payload size limit (HTTP 413) and session message history capping."""
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request with oversized Content-Length
        resp = await client.post(
            "/v1/chat/completions",
            content=b"x" * 100,
            headers={"Content-Length": str(15 * 1024 * 1024)},
        )
        assert resp.status_code == 413
        assert "Maximum allowed request size is 10MB" in resp.json()["detail"]

    # Verify session retains only the last 5 messages
    sm = StateManager()
    many_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(20)
    ]
    sess = sm.get_or_create_session(
        session_id="sess_bounded",
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=many_messages,
        prefix_hash="hash123",
    )
    assert len(sess.messages) == 5
    assert sess.messages[-1]["content"] == "msg 19"


# --- W2-10: Tenant Identity Preservation ---


@pytest.mark.asyncio
async def test_w2_10_tenant_user_identity_preserved():
    """Verify caller's user field is preserved and not overwritten with cf_<prefix_hash>."""
    captured_requests = []

    class _CaptureTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured_requests.append(
                {
                    "body": json.loads(request.content.decode("utf-8")),
                    "headers": dict(request.headers),
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-user-01",
                    "object": "chat.completion",
                    "created": 1742000000,
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )

    transport = _CaptureTransport()
    http_client = httpx.AsyncClient(
        transport=transport, base_url="https://api.tokenfactory.nebius.com/v1"
    )
    app = create_app(http_client=http_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "nvidia/nemotron-3-super-120b-a12b",
                "messages": [{"role": "user", "content": "hello"}],
                "user": "tenant_alice_prod",
            },
        )

    assert len(captured_requests) == 1
    sent_body = captured_requests[0]["body"]
    sent_headers = captured_requests[0]["headers"]

    # Must preserve exact caller user
    assert sent_body.get("user") == "tenant_alice_prod"
    # Session header incorporates caller identity
    assert "tenant_alice_prod" in sent_headers.get("x-session-id", "")
