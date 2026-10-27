"""Unit and Textual pilot tests for RunSift Watch mode (T4)."""

from __future__ import annotations

import httpx
import pytest
from textual.widgets import DataTable, Static

from src.tui.watch_app import WatchApp, format_estimated_request_cost


@pytest.mark.asyncio
async def test_t4_watch_displays_fake_telemetry_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """T4: Watch renders fake telemetry payload, sends Bearer token when RUNSIFT_API_KEY is set, and formats Nano vs other models."""
    monkeypatch.setenv("RUNSIFT_API_KEY", "test-watch-token-123")
    seen_auth_headers: list[str] = []

    fake_payload = {
        "upstream_status": "CONFIGURED (Nebius Token Factory)",
        "requests_total": 60,
        "cumulative_prompt_tokens": 125000,
        "cumulative_tokens_saved": 7565,
        "cumulative_tool_chars_removed": 90000,
        "estimated_input_cost_avoided_usd": 0.000454,
        "recent_requests": [
            {
                "id": "req-60",
                "timestamp": "12:01:05",
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                "condenser": "v2",
                "tool_chars_before": 39855,
                "tool_chars_after": 9594,
                "fallback": False,
                "tokens_in": 3288,
                "cached_tokens": 0,
                "tokens_out": 120,
                "ttft_ms": 432.0,
            },
            {
                "id": "req-59",
                "timestamp": "12:00:10",
                "model": "nvidia/nemotron-3-super-120b-a12b",
                "condenser": "",
                "tool_chars_before": 0,
                "tool_chars_after": 0,
                "fallback": False,
                "tokens_in": 500,
                "cached_tokens": 400,
                "tokens_out": 50,
                "ttft_ms": 210.0,
            },
        ],
    }

    class _FakeTelemetryTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen_auth_headers.append(request.headers.get("authorization", ""))
            return httpx.Response(200, json=fake_payload)

    client = httpx.AsyncClient(transport=_FakeTelemetryTransport())
    app = WatchApp(base_url="http://127.0.0.1:8765", http_client=client, poll_interval=None)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        assert seen_auth_headers == ["Bearer test-watch-token-123"]
        status_text = str(app.query_one("#watch-status", Static).render())
        assert "CONFIGURED (Nebius Token Factory)" in status_text

        totals_text = str(app.query_one("#watch-totals", Static).render())
        assert "60" in totals_text
        assert "125,000" in totals_text  # cumulative_prompt_tokens
        assert "90,000" in totals_text  # cumulative_tool_chars_removed
        assert "estimated input cost avoided (list price, chars ÷ 4)" in totals_text
        assert "$0.000454" in totals_text

        table = app.query_one("#watch-table", DataTable)
        assert table.row_count == 2
        row0 = table.get_row_at(0)
        row1 = table.get_row_at(1)
        assert str(row0[2]) == "v2"
        assert str(row1[2]) == "—"  # no default to "v2" when condenser is empty
        assert "39,855 → 9,594 (-76%)" in str(row0[3])
        assert str(row0[8]).startswith("$")
        assert str(row1[8]) == "—"

    await client.aclose()


@pytest.mark.asyncio
async def test_t4_watch_unreachable_gateway_shows_message_and_does_not_crash() -> None:
    """T4: When the gateway is unreachable, Watch shows a clear message and does not crash."""

    class _UnreachableTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=_UnreachableTransport())
    app = WatchApp(base_url="http://127.0.0.1:8765", http_client=client, poll_interval=None)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        status_text = str(app.query_one("#watch-status", Static).render())
        assert "Gateway unreachable at http://127.0.0.1:8765" in status_text
        assert "Retrying" in status_text

        # Trigger manual retry (`r`) and confirm it stays alive without crashing
        await pilot.press("r")
        await pilot.pause()
        assert "Gateway unreachable" in str(app.query_one("#watch-status", Static).render())

    await client.aclose()


def test_t4_cost_helper_nano_only() -> None:
    """Nano gets estimated list cost; any other model shows '—'."""
    assert (
        format_estimated_request_cost("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 1_000_000, 0)
        == "$0.06000"
    )
    assert format_estimated_request_cost("nvidia/nemotron-3-super-120b-a12b", 1_000_000, 0) == "—"


@pytest.mark.asyncio
async def test_demo_route_records_v1_condenser_and_tool_chars() -> None:
    """/api/demo/run passes condenser='v1' and tool character counts to telemetry, and record_request_telemetry has no 'v2' default."""
    from src.core.config import RunSiftConfig
    from src.core.invariants import GatewayTelemetry, telemetry
    from src.proxy.gateway import create_app

    # 1. Verify GatewayTelemetry.record_request_telemetry has no 'v2' default
    t = GatewayTelemetry()
    t.record_request_telemetry(
        turn_id="req-01",
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        prefix_hash="abc",
        cache_hit=False,
        ttft_ms=100.0,
        tokens_in=50,
    )
    assert t.recent_requests[0]["condenser"] == ""

    # 2. Verify /api/demo/run records condenser="v1" and tool_chars_before/after > 0 on turns 2 and 3
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Healthy."},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":200,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":150}}}\n\n'
        "data: [DONE]\n\n"
    )

    class _DemoUpstreamTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )

    telemetry.reset()
    cfg = RunSiftConfig(nebius_api_key="test-key", api_key="")
    upstream_http = httpx.AsyncClient(
        transport=_DemoUpstreamTransport(), base_url="https://api.tokenfactory.nebius.com/v1"
    )
    gw_app = create_app(config=cfg, http_client=upstream_http)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gw_app), base_url="http://testserver"
    ) as client:
        resp = await client.post("/api/demo/run")
        assert resp.status_code == 200
        tel_resp = await client.get("/api/telemetry")
        assert tel_resp.status_code == 200
        tel_data = tel_resp.json()

    await upstream_http.aclose()

    assert tel_data["requests_total"] == 3
    assert tel_data["cumulative_prompt_tokens"] == 600
    assert tel_data["cumulative_tool_chars_removed"] > 0
    recent = tel_data["recent_requests"]
    assert len(recent) == 3
    assert all(r["condenser"] == "v1" for r in recent)
    # Turn 1 (recent[2]) had no prior tool messages; Turns 2 and 3 (recent[1], recent[0]) compacted tool logs
    assert recent[2]["tool_chars_before"] == 0
    assert recent[1]["tool_chars_before"] > recent[1]["tool_chars_after"] > 0
    assert recent[0]["tool_chars_before"] > recent[0]["tool_chars_after"] > 0
