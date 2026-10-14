"""End-to-end integration tests for RunSift FastAPI proxy gateway.

Verifies OpenAI wire parity, prefix hashing headers, KV-cache hit rate telemetry,
dashboard HTML rendering, and Server-Sent Events (SSE) streaming compatibility.
"""

import json
import time

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from src.proxy.gateway import create_app


class _TestUpstreamTransport(httpx.AsyncBaseTransport):
    """Deterministic ASGI/HTTP wire transport for testing NebiusClient HTTP & SSE parsing."""

    def __init__(self) -> None:
        self.seen_sessions: set[str] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "nvidia/nemotron-3-super-120b-a12b",
                            "object": "model",
                            "created": 1742000000,
                            "owned_by": "nvidia",
                        },
                        {
                            "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                            "object": "model",
                            "created": 1742000000,
                            "owned_by": "nvidia",
                        },
                    ],
                },
            )

        if path.endswith("/chat/completions") and request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            session_key = request.headers.get("x-session-id", "")
            is_warm = session_key in self.seen_sessions if session_key else False
            if session_key:
                self.seen_sessions.add(session_key)

            prompt_tokens = max(32, len(request.content) // 4)
            cached_tokens = int(prompt_tokens * 0.85) if is_warm else 0
            model = body.get("model", "nvidia/nemotron-3-super-120b-a12b")

            if body.get("stream"):
                chunk_1 = {
                    "id": "chatcmpl-stream-01",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "1, 2, 3."},
                            "finish_reason": "stop",
                        }
                    ],
                }
                chunk_usage = {
                    "id": "chatcmpl-stream-01",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 8,
                        "total_tokens": prompt_tokens + 8,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                }
                sse_bytes = (
                    f"data: {json.dumps(chunk_1)}\n\n"
                    f"data: {json.dumps(chunk_usage)}\n\n"
                    "data: [DONE]\n\n"
                ).encode()
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_bytes,
                )

            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test-01",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Verified canonical KV-cache alignment.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 12,
                        "total_tokens": prompt_tokens + 12,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
            )

        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def app_instance():
    transport = _TestUpstreamTransport()
    http_client = httpx.AsyncClient(
        transport=transport,
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    return create_app(http_client=http_client)


@pytest.mark.asyncio
async def test_health_endpoint(app_instance):
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] in ("0.1.0", "1.4.0", "2.0.0")
        assert "git_commit" in data
        assert "SDCS SPEC-001 v2.0.0" in data["spec"]
        assert "nemotron" in data["default_model"].lower()


@pytest.mark.asyncio
async def test_dashboard_endpoint(app_instance):
    """Verify /dashboard returns self-contained HTML with Tailwind, Chart.js, and honest labels."""
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "RunSift" in resp.text
        assert "ttftChart" in resp.text
        assert "tailwindcss" in resp.text
        assert "▶ Demo: 3 live turns with a synthetic test log" in resp.text
        assert "MOCK(M-013)" in resp.text
        assert "Est. tokens removed (chars ÷ 4)" in resp.text
        assert "Est. without RunSift:" in resp.text
        assert "est. input cost avoided:" in resp.text
        assert "TTFT: cache hits vs misses (this session)" in resp.text
        assert "condenses tool output over 4,000 chars" in resp.text
        assert "formatCacheStatusLabel" in resp.text
        assert "Warm KV Hit" not in resp.text
        assert "8K Threshold" not in resp.text
        assert "Failure isolated in radix_tree.py:142." not in resp.text


@pytest.mark.asyncio
async def test_api_telemetry_endpoint(app_instance):
    """Verify /api/telemetry returns full structured metrics for live monitoring."""
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.get("/api/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert "radix_cache_hit_rate" in data
        assert "mean_ttft_cold_ms" in data
        assert "mean_ttft_warm_ms" in data
        assert "speedup_ratio" in data
        assert "total_suspended_tool_seconds" in data
        assert "estimated_input_cost_avoided_usd" in data
        assert data.get("condenser_char_threshold") == 4000
        assert "prefill_flops_avoided_teraflops" not in data
        assert "estimated_nebius_cost_savings_usd" not in data
        assert "recent_requests" in data
        assert isinstance(data["recent_requests"], list)


@pytest.mark.asyncio
async def test_models_endpoint(app_instance):
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "nvidia/nemotron-3-super-120b-a12b" in model_ids


@pytest.mark.asyncio
async def test_chat_completions_non_streaming_cache_alignment(app_instance):
    """Verify first request performs prefill, and second identical prefix request hits cache."""
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "You are a specialized code optimizer."},
            {"role": "user", "content": "Explain KV-cache prefixing."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_docs",
                    "description": "Searches documentation",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    }

    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        # First turn: cold prefill
        resp1 = await client.post("/v1/chat/completions", json=payload)
        assert resp1.status_code == 200
        assert "x-cacheforge-prefix-hash" in resp1.headers
        prefix_hash = resp1.headers["x-cacheforge-prefix-hash"]
        assert len(prefix_hash) == 64

        # Second turn: identical prompt & tools with different user query tail
        payload2 = dict(payload)
        payload2["messages"] = [
            payload["messages"][0],
            payload["messages"][1],
            {"role": "assistant", "content": "KV-cache prefixing enables prompt sharing."},
            {"role": "user", "content": "How does RunSift accelerate this?"},
        ]

        resp2 = await client.post("/v1/chat/completions", json=payload2)
        assert resp2.status_code == 200
        # The prefix hash of the static system prompt + tools must remain bitwise identical
        assert resp2.headers["x-cacheforge-prefix-hash"] == prefix_hash
        assert resp2.headers.get("x-cacheforge-cache-hit") == "true"
        assert resp2.headers.get("x-cacheforge-kv-hit") == "true"
        assert "x-cacheforge-prefill-tokens-avoided" in resp2.headers
        assert "x-cacheforge-ttft-speedup" in resp2.headers


@pytest.mark.asyncio
async def test_chat_completions_streaming(app_instance):
    """Verify Server-Sent Events (SSE) streaming compatibility."""
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": "Stream assistant."},
            {"role": "user", "content": "Count to 3."},
        ],
        "stream": True,
    }

    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "x-cacheforge-prefix-warm" in resp.headers
        assert "x-cacheforge-tokens-compacted" in resp.headers
        text = resp.text
        assert "data:" in text
        assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_metrics_telemetry(app_instance):
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        metrics = resp.json()
        assert "requests_total" in metrics
        assert "cache_hit_rate" in metrics
        assert metrics["requests_total"] > 0


@pytest.mark.asyncio
async def test_demo_run_and_telemetry_reset(app_instance):
    """Verify one-click /api/demo/run 3-turn agent loop and /api/telemetry/reset."""
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        reset_resp = await client.post("/api/telemetry/reset")
        assert reset_resp.status_code == 200
        assert reset_resp.json()["requests_total"] == 0

        demo_resp = await client.post("/api/demo/run")
        assert demo_resp.status_code == 200
        data = demo_resp.json()
        assert data["status"] == "completed"
        assert len(data["turns"]) == 3
        assert data["turns"][0]["cache_hit"] is False
        assert data["turns"][1]["cache_hit"] is True
        assert data["turns"][2]["cache_hit"] is True
        assert data["turns"][1]["tokens_saved"] > 0
        assert (
            data["turns"][0]["prefix_hash"]
            == data["turns"][1]["prefix_hash"]
            == data["turns"][2]["prefix_hash"]
        )


@pytest.mark.asyncio
async def test_model_selection_and_demo_override(app_instance):
    """Verify POST /api/model/select switches active model and POST /api/demo/run honors model override."""
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        select_resp = await client.post(
            "/api/model/select",
            json={"model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"},
        )
        assert select_resp.status_code == 200
        sel_data = select_resp.json()
        assert sel_data["status"] == "updated"
        assert sel_data["active_model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        assert "nvidia/Nemotron-3_5-Lightning" in sel_data["supported_models"]

        telem_resp = await client.get("/api/telemetry")
        assert telem_resp.status_code == 200
        assert telem_resp.json()["upstream_model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

        demo_resp = await client.post(
            "/api/demo/run",
            json={"model": "nvidia/Nemotron-3_5-Lightning"},
        )
        assert demo_resp.status_code == 200
        assert demo_resp.json()["model"] == "nvidia/Nemotron-3_5-Lightning"


@pytest.mark.asyncio
async def test_session_collision_defense_and_headers(app_instance):
    """Verify distinct uncoordinated requests receive unique session IDs and headers."""
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "Hello agent"}],
    }

    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        r1 = await client.post("/v1/chat/completions", json=payload)
        r2 = await client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200

        sid1 = r1.headers.get("x-cacheforge-session-id")
        sid2 = r2.headers.get("x-cacheforge-session-id")
        assert sid1 and sid2
        # Without explicit session headers, requests must not collide
        assert sid1 != sid2


@pytest.mark.asyncio
async def test_cors_security_configuration(app_instance):
    """Verify CORS middleware rejects unlisted origins and allows configured origins."""
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        # Hostile / unlisted origin should not be allowed
        resp_evil = await client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp_evil.headers.get("access-control-allow-origin") != "*"
        assert resp_evil.headers.get("access-control-allow-origin") != "http://evil.com"

        # Explicitly allowed origin succeeds
        resp_good = await client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:8765",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp_good.status_code == 200
        assert resp_good.headers.get("access-control-allow-origin") == "http://localhost:8765"


@pytest.mark.asyncio
async def test_streaming_upstream_http_error_handling():
    """Verify that an upstream 401/502 error on streaming returns the error status code, not HTTP 200."""

    class _ErrorTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "Invalid Nebius Token"})

    http_client = httpx.AsyncClient(
        transport=_ErrorTransport(),
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    app = create_app(http_client=http_client)

    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "Test streaming error"}],
        "stream": True,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        # Must return upstream error code (401), NOT 200 with broken stream
        assert resp.status_code == 401
        assert "Invalid Nebius Token" in resp.text


@pytest.mark.asyncio
async def test_api_key_auth_enforcement():
    """Verify RUNSIFT_API_KEY restricts access to /v1/chat/completions."""
    from src.core.config import RunSiftConfig

    transport = _TestUpstreamTransport()
    http_client = httpx.AsyncClient(
        transport=transport,
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    cfg = RunSiftConfig(api_key="forge-secret-999")
    app = create_app(config=cfg, http_client=http_client)

    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "ping"}],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Unauthenticated request must be rejected
        unauth_resp = await client.post("/v1/chat/completions", json=payload)
        assert unauth_resp.status_code == 401

        # Request with valid Bearer token succeeds
        auth_resp = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer forge-secret-999"},
        )
        assert auth_resp.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_contains_xss_protection():
    """Verify /dashboard embeds escapeHtml sanitizer and avoids unescaped innerHTML sinks."""
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "function escapeHtml" in resp.text
        assert "escapeHtml(req.model)" in resp.text


@pytest.mark.asyncio
async def test_condenser_exception_fails_open(monkeypatch, caplog):
    """Verify that when condensing raises ValueError/TypeError/RecursionError, the gateway forwards the unmodified request upstream and sets x-cacheforge-fallback: true."""
    from src.compact.condenser_v2 import CondenserV2

    captured_payloads = []

    class _CaptureTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fallback-01",
                    "object": "chat.completion",
                    "created": 1742000000,
                    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Fallback succeeded."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
                },
            )

    def _boom_compact_messages(self, messages):
        raise ValueError("simulated condenser parser failure")

    monkeypatch.setattr(CondenserV2, "compact_messages", _boom_compact_messages)

    http_client = httpx.AsyncClient(
        transport=_CaptureTransport(),
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    app = create_app(http_client=http_client)

    raw_tool_text = (
        "FAILED tests/test_radix.py::test_evict - AssertionError: ref_count mismatch\n" * 100
    )
    payload = {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "messages": [
            {"role": "system", "content": "Current Time: 2026-10-18 12:00:00 UTC\nStatic prompt."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_pytest", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "run_pytest",
                "content": raw_tool_text,
            },
        ],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("x-cacheforge-fallback") == "true"
        assert resp.json()["choices"][0]["message"]["content"] == "Fallback succeeded."

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["messages"] == payload["messages"]
    assert "simulated condenser parser failure" in caplog.text


@pytest.mark.asyncio
async def test_upstream_402_and_429_passthrough():
    """Verify Token Factory 402 and 429 errors pass through with clear status and message."""
    for status_code, raw_msg in [
        (402, "Insufficient credit balance"),
        (429, "Rate limit reached for requests"),
    ]:

        class _StatusTransport(httpx.AsyncBaseTransport):
            def __init__(self, code: int, msg: str):
                self.code = code
                self.msg = msg

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    self.code,
                    json={"error": {"message": self.msg, "type": "upstream_error"}},
                )

        http_client = httpx.AsyncClient(
            transport=_StatusTransport(status_code, raw_msg),
            base_url="https://api.tokenfactory.nebius.com/v1",
        )
        app = create_app(http_client=http_client)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            assert resp.status_code == status_code
            err_msg = resp.json()["error"]["message"]
            assert raw_msg in err_msg
            assert f"HTTP {status_code}" in err_msg


@pytest.mark.asyncio
async def test_ttft_speedup_header_na_when_no_cold_ttft_measured():
    """When no cold TTFT has been measured, x-cacheforge-ttft-speedup must be 'n/a' instead of using a 1200.0ms fallback."""

    class _ImmediateWarmHitTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-warm-01",
                    "object": "chat.completion",
                    "created": 1742000000,
                    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Warm hit."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 128,
                        "completion_tokens": 4,
                        "total_tokens": 132,
                        "prompt_tokens_details": {"cached_tokens": 64},
                    },
                },
            )

    http_client = httpx.AsyncClient(
        transport=_ImmediateWarmHitTransport(),
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    app = create_app(http_client=http_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/api/telemetry/reset")
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-cacheforge-cache-hit") == "true"
        assert resp.headers.get("x-cacheforge-ttft-speedup") == "n/a"


@pytest.mark.asyncio
async def test_t5_telemetry_fields_streaming_and_non_streaming() -> None:
    """T5: tool_chars_before, tool_chars_after, condenser, and fallback appear in recent_requests on both paths."""
    http_client = httpx.AsyncClient(
        transport=_TestUpstreamTransport(),
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    app = create_app(http_client=http_client)
    raw_tool_text = (
        "FAILED tests/test_radix.py::test_evict - AssertionError: ref_count mismatch\n" * 120
    )

    def _make_payload(stream: bool) -> dict[str, object]:
        return {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "stream": stream,
            "messages": [
                {"role": "system", "content": "You are a debugging assistant."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_pytest", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "run_pytest",
                    "content": raw_tool_text,
                },
            ],
        }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/api/telemetry/reset")

        # 1. Non-streaming request
        r1 = await client.post("/v1/chat/completions", json=_make_payload(stream=False))
        assert r1.status_code == 200

        # 2. Streaming request
        async with client.stream(
            "POST", "/v1/chat/completions", json=_make_payload(stream=True)
        ) as r2:
            assert r2.status_code == 200
            _ = await r2.aread()

        telem_resp = await client.get("/api/telemetry")
        assert telem_resp.status_code == 200
        recent = telem_resp.json()["recent_requests"]
        assert len(recent) == 2

        for entry in recent:
            assert entry["tool_chars_before"] == len(raw_tool_text)
            assert 0 < entry["tool_chars_after"] < entry["tool_chars_before"]
            assert entry["condenser"] == "v2"
            assert entry["fallback"] is False


@pytest.mark.asyncio
async def test_t6_gateway_request_lines_and_handler_dedup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """T6: One line per request on stderr, no content or keys, RUNSIFT_REQUEST_LOG=off silences, no duplicate handlers."""

    from src.core.config import RunSiftConfig
    from src.proxy.gateway import request_logger

    monkeypatch.delenv("RUNSIFT_REQUEST_LOG", raising=False)
    monkeypatch.delenv("CACHEFORGE_REQUEST_LOG", raising=False)
    # Remove existing handlers to test handler registration cleanly
    for h in list(request_logger.handlers):
        request_logger.removeHandler(h)

    secret_api_key = "cf-secret-test-key-99999"
    secret_nebius_key = "nebius-secret-key-88888"
    secret_user_text = "SUPER_SECRET_USER_PROMPT_DO_NOT_LOG"

    cfg = RunSiftConfig(
        nebius_api_key=secret_nebius_key,
        api_key=secret_api_key,
    )
    http_client = httpx.AsyncClient(
        transport=_TestUpstreamTransport(),
        base_url="https://api.tokenfactory.nebius.com/v1",
    )

    # Creating multiple apps must not attach duplicate handlers
    _ = create_app(config=cfg, http_client=http_client)
    _ = create_app(config=cfg, http_client=http_client)
    app = create_app(config=cfg, http_client=http_client)

    cf_handlers = [
        h for h in request_logger.handlers if getattr(h, "_runsift_request_handler", False) or getattr(h, "_cacheforge_request_handler", False)
    ]
    assert len(cf_handlers) == 1
    # Rebind the stream handler to current sys.stderr so capsys captures it
    import sys

    cf_handlers[0].stream = sys.stderr  # type: ignore[attr-defined]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        reset_resp = await client.post(
            "/api/telemetry/reset",
            headers={"Authorization": f"Bearer {secret_api_key}"},
        )
        assert reset_resp.status_code == 200
        capsys.readouterr()

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret_api_key}"},
            json={
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                "messages": [{"role": "user", "content": secret_user_text}],
            },
        )
        assert resp.status_code == 200

    captured = capsys.readouterr()
    err_lines = [ln for ln in captured.err.splitlines() if ln.strip()]
    assert len(err_lines) == 1
    line = err_lines[0]
    assert line.startswith("req-01  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B  v2  tool output ")
    assert "prompt " in line and "cached " in line and "first token " in line
    assert secret_api_key not in line
    assert secret_nebius_key not in line
    assert secret_user_text not in line

    # Verify RUNSIFT_REQUEST_LOG=off silences request logging
    monkeypatch.setenv("RUNSIFT_REQUEST_LOG", "off")
    app_off = create_app(config=cfg, http_client=http_client)
    async with AsyncClient(transport=ASGITransport(app=app_off), base_url="http://test") as client:
        capsys.readouterr()
        resp2 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret_api_key}"},
            json={
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                "messages": [{"role": "user", "content": secret_user_text}],
            },
        )
        assert resp2.status_code == 200

    captured_off = capsys.readouterr()
    assert [ln for ln in captured_off.err.splitlines() if ln.strip()] == []


@pytest.mark.asyncio
async def test_dotenv_discovery_and_hot_reload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-location .env discovery (cwd, Downloads, Desktop, home) and live hot-reload on /health."""
    import os
    from pathlib import Path

    from src.core.config import RunSiftConfig, _load_dotenv_if_present, candidate_dotenv_paths

    fake_home = tmp_path / "home"
    fake_downloads = fake_home / "Downloads"
    fake_cwd = tmp_path / "cwd"
    fake_downloads.mkdir(parents=True)
    fake_cwd.mkdir(parents=True)

    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.chdir(fake_cwd)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    # 1. Saving a template with placeholder in cwd should not shadow a real key saved in Downloads
    (fake_cwd / ".env").write_text("NEBIUS_API_KEY=your-token-factory-key\n", encoding="utf-8")
    (fake_downloads / ".env").write_text(
        "NEBIUS_API_KEY=nebius-from-downloads-123\nRUNSIFT_CONDENSER=v2.1\n",
        encoding="utf-8",
    )

    cands = candidate_dotenv_paths()
    assert fake_cwd / ".env" in cands
    assert fake_downloads / ".env" in cands

    cfg = RunSiftConfig(nebius_api_key="")
    http_client = httpx.AsyncClient(
        transport=_TestUpstreamTransport(),
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    app = create_app(config=cfg, http_client=http_client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        h_resp = await client.get("/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["upstream_configured"] is True
        assert cfg.nebius_api_key == "nebius-from-downloads-123"
        assert cfg.condenser_version == "v2.1"

    # 2. Editing .env in cwd overrides the key on next _load_dotenv_if_present(override=True)
    (fake_cwd / ".env").write_text("NEBIUS_API_KEY=nebius-from-cwd-456\n", encoding="utf-8")
    loaded_from = _load_dotenv_if_present(override=True)
    assert loaded_from == fake_cwd / ".env"
    assert os.environ["NEBIUS_API_KEY"] == "nebius-from-cwd-456"
