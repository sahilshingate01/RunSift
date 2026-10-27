"""Textual live watch UI for RunSift (`python -m src.tui watch`).

Implements Section 5 of `docs/specs/TERMINAL_APP_SPEC.md`:
- Polls `GET /api/telemetry` once a second, sending `RUNSIFT_API_KEY` (or `CACHEFORGE_API_KEY`) as a Bearer token if set.
- Live table of recent requests: time, model, condenser, tool output (before → after chars),
  prompt/cached/completion tokens, time to first token, and estimated list cost (Nano only;
  other models show "—").
- Running totals: requests, prompt tokens sent, tool characters removed, and
  "estimated input cost avoided (list price, chars ÷ 4)".
- Upstream status: shows `upstream_status` as it is (CONFIGURED or UNCONFIGURED).
- When the gateway can't be reached: shows a clear message and keeps retrying without crashing.
Allowed dependencies per wiring.yaml: ["core", "compact"].
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import httpx
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import DataTable, Label, Static

from src.tui.data import compute_session_cost_usd

NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def format_estimated_request_cost(model: str, tokens_in: int, tokens_out: int) -> str:
    """Formats estimated list cost for Nano ($0.06/1M in, $0.24/1M out); other models show '—'."""
    if model.strip().lower() == NANO_MODEL_ID.lower():
        cost = compute_session_cost_usd(tokens_in, tokens_out)
        return f"${cost:.5f}"
    return "—"


def format_tool_chars_cell(before: int, after: int, fallback: bool = False) -> str:
    """Formats tool output (before → after chars) for a watch table row."""
    if fallback:
        return f"{before:,} → {after:,} (FALLBACK)"
    if before > 0:
        pct = round(((after - before) / before) * 100)
        pct_str = f"{pct:+d}%" if pct != 0 else "0%"
        return f"{before:,} → {after:,} ({pct_str})"
    return "0 → 0"


def format_running_totals(payload: dict[str, Any]) -> str:
    """Formats the running totals banner from cumulative `/api/telemetry` fields."""
    requests_total = int(payload.get("requests_total", 0))
    recent = payload.get("recent_requests") or []
    if "cumulative_prompt_tokens" in payload:
        prompt_tokens_sent = int(payload["cumulative_prompt_tokens"])
    else:
        prompt_tokens_sent = sum(int(r.get("tokens_in", 0)) for r in recent)

    if "cumulative_tool_chars_removed" in payload:
        tool_chars_removed = int(payload["cumulative_tool_chars_removed"])
    else:
        tool_chars_removed = sum(
            max(0, int(r.get("tool_chars_before", 0)) - int(r.get("tool_chars_after", 0)))
            for r in recent
        )

    cost_avoided = float(payload.get("estimated_input_cost_avoided_usd", 0.0))
    return (
        f"[b]Requests:[/b] {requests_total:,}   |   "
        f"[b]Prompt tokens sent:[/b] {prompt_tokens_sent:,}   |   "
        f"[b]Tool chars removed:[/b] {tool_chars_removed:,}   |   "
        f"[b]estimated input cost avoided (list price, chars ÷ 4):[/b] ${cost_avoided:.6f}"
    )


class WatchApp(App[None]):
    """Live terminal monitor polling `/api/telemetry` once per second."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #watch-header {
        padding: 0 1;
        background: $boost;
        color: $text;
        height: 1;
        overflow: hidden;
    }
    #watch-status {
        padding: 0 1;
        height: 1;
        overflow: hidden;
    }
    #watch-totals {
        padding: 0 1;
        border: solid $primary;
        height: auto;
    }
    #watch-table {
        height: 1fr;
    }
    #watch-footer {
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        height: 1;
        overflow: hidden;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh_now", "Refresh", priority=True),
    ]

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        http_client: httpx.AsyncClient | None = None,
        poll_interval: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._external_client = http_client is not None
        self._http_client = http_client
        self.poll_interval = poll_interval

    def compose(self) -> ComposeResult:
        yield Label(
            f"[b]RunSift Watch — Live Gateway Traffic[/b] ({self.base_url})",
            id="watch-header",
        )
        yield Static("Connecting to gateway...", id="watch-status")
        yield Static("", id="watch-totals")
        yield DataTable(id="watch-table", cursor_type="row")
        yield Label(
            "Polling GET /api/telemetry every 1s  |  r: refresh now  |  q: quit", id="watch-footer"
        )

    async def on_mount(self) -> None:
        table = self.query_one("#watch-table", DataTable)
        table.add_columns(
            "Time",
            "Model",
            "Condenser",
            "Tool output (before → after chars)",
            "Prompt",
            "Cached",
            "Completion",
            "First token",
            "Estimated list cost",
        )
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=2.5)
        await self.poll_once()
        if self.poll_interval is not None and self.poll_interval > 0:
            self.set_interval(self.poll_interval, self.poll_once)

    async def on_unmount(self) -> None:
        if self._http_client is not None and not self._external_client:
            await self._http_client.aclose()

    async def action_refresh_now(self) -> None:
        await self.poll_once()

    async def poll_once(self) -> None:
        """Fetches `/api/telemetry` once and updates the live UI without crashing on errors."""
        if self._http_client is None:
            return
        headers: dict[str, str] = {}
        api_key = (
            os.environ.get("RUNSIFT_API_KEY", "").strip()
            or os.environ.get("CACHEFORGE_API_KEY", "").strip()
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self.base_url}/api/telemetry"
        try:
            resp = await self._http_client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self.query_one("#watch-status", Static).update(
                f"Gateway unreachable at {self.base_url} ({exc}). Retrying every 1s..."
            )
            return

        self.apply_telemetry_payload(payload)

    def apply_telemetry_payload(self, payload: dict[str, Any]) -> None:
        """Applies a `/api/telemetry` JSON dictionary to the status, totals, and table widgets."""
        upstream_status = str(payload.get("upstream_status") or "UNCONFIGURED")
        self.query_one("#watch-status", Static).update(f"[b]Upstream status:[/b] {upstream_status}")
        self.query_one("#watch-totals", Static).update(format_running_totals(payload))

        table = self.query_one("#watch-table", DataTable)
        table.clear()

        recent = payload.get("recent_requests") or []
        for req in recent:
            ts = str(req.get("timestamp") or "—")
            model = str(req.get("model") or "—")
            condenser = str(req.get("condenser") or "—")
            before = int(req.get("tool_chars_before", 0))
            after = int(req.get("tool_chars_after", 0))
            fallback = bool(req.get("fallback", False))
            tokens_in = int(req.get("tokens_in", 0))
            cached = int(req.get("cached_tokens", 0))
            tokens_out = int(req.get("tokens_out", 0))
            ttft_ms = float(req.get("ttft_ms", 0.0))

            table.add_row(
                ts,
                model,
                condenser,
                format_tool_chars_cell(before, after, fallback=fallback),
                f"{tokens_in:,}",
                f"{cached:,}",
                f"{tokens_out:,}",
                f"{round(ttft_ms)} ms",
                format_estimated_request_cost(model, tokens_in, tokens_out),
            )
