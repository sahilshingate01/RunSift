"""Constitutional invariants and telemetry definitions for RunSift.

Enforces SDCS SPEC-001 v2.0.0 and gateway integrity checks.
Leaf module in wiring.yaml.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

# Non-negotiable constitutional constants
CANONICAL_SEPARATOR = "\n---\n"
PREFIX_HASH_ALGORITHM = "sha256"
DEFAULT_CACHE_ALIGNMENT_TOLERANCE = 0.90  # 90% KV-cache target


@dataclass
class GatewayTelemetry:
    """In-memory telemetry recording live KV-cache hits, TTFT reductions, and token savings."""

    requests_total: int = 0
    cache_hits_total: int = 0
    prompt_tokens_original: int = 0
    prompt_tokens_canonical: int = 0
    tokens_saved_by_compaction: int = 0
    tool_chars_removed_total: int = 0
    cached_tokens_reused: int = 0
    total_tool_wait_seconds_suspended: float = 0.0
    start_time: float = field(default_factory=time.time)
    cold_ttfts: list[float] = field(default_factory=list)
    warm_ttfts: list[float] = field(default_factory=list)
    recent_requests: list[dict[str, Any]] = field(default_factory=list)
    _id_counter: Any = field(default_factory=lambda: itertools.count(1))

    def next_turn_id(self) -> str:
        """Returns monotonically increasing unique request/turn ID safe for concurrent calls."""
        return f"req-{next(self._id_counter):02d}"

    def reset(self) -> None:
        """Resets all live telemetry counters and request history to initial zero state."""
        self.requests_total = 0
        self.cache_hits_total = 0
        self.prompt_tokens_original = 0
        self.prompt_tokens_canonical = 0
        self.tokens_saved_by_compaction = 0
        self.tool_chars_removed_total = 0
        self.cached_tokens_reused = 0
        self.total_tool_wait_seconds_suspended = 0.0
        self.start_time = time.time()
        self.cold_ttfts.clear()
        self.warm_ttfts.clear()
        self.recent_requests.clear()
        self._id_counter = itertools.count(1)

    def record_request_telemetry(
        self,
        turn_id: str,
        model: str,
        prefix_hash: str,
        cache_hit: bool,
        ttft_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cached_tokens: int = 0,
        suspended_seconds: float = 0.0,
        compacted_saved_chars: int = 0,
        tool_chars_before: int = 0,
        tool_chars_after: int = 0,
        condenser: str = "",
        fallback: bool = False,
    ) -> None:
        """Records detailed per-request live telemetry for the dashboard and analytics engine."""
        self.requests_total += 1
        saved_tokens = compacted_saved_chars // 4
        orig_tokens_in = tokens_in + saved_tokens

        if cache_hit:
            self.cache_hits_total += 1
            self.warm_ttfts.append(ttft_ms)
            if len(self.warm_ttfts) > 1000:
                self.warm_ttfts = self.warm_ttfts[-1000:]
            reused = cached_tokens if cached_tokens > 0 else tokens_in
            self.cached_tokens_reused += reused
        else:
            reused = 0
            self.cold_ttfts.append(ttft_ms)
            if len(self.cold_ttfts) > 1000:
                self.cold_ttfts = self.cold_ttfts[-1000:]

        removed_chars = max(compacted_saved_chars, max(0, tool_chars_before - tool_chars_after))
        self.prompt_tokens_original += orig_tokens_in
        self.prompt_tokens_canonical += tokens_in
        self.tokens_saved_by_compaction += saved_tokens
        self.tool_chars_removed_total += removed_chars
        self.total_tool_wait_seconds_suspended += suspended_seconds

        entry = {
            "id": turn_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "model": model,
            "prefix_hash": (
                f"{prefix_hash[:16]}...{prefix_hash[-8:]}"
                if len(prefix_hash) > 24
                else (prefix_hash or "None")
            ),
            "cache_hit": cache_hit,
            "status": "HIT" if cache_hit else "MISS",
            "ttft_ms": round(ttft_ms, 1),
            "suspended_seconds": round(suspended_seconds, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_original": orig_tokens_in,
            "tokens_saved": saved_tokens,
            "cached_tokens": reused,
            "tokens": tokens_in + tokens_out,
            "tool_chars_before": tool_chars_before,
            "tool_chars_after": tool_chars_after,
            "condenser": condenser,
            "fallback": fallback,
        }
        self.recent_requests.insert(0, entry)
        if len(self.recent_requests) > 50:
            self.recent_requests.pop()

    @property
    def cache_hit_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.cache_hits_total / self.requests_total

    @property
    def mean_ttft_cold_ms(self) -> float:
        if self.cold_ttfts:
            return sum(self.cold_ttfts) / len(self.cold_ttfts)
        return 0.0

    @property
    def mean_ttft_warm_ms(self) -> float:
        if self.warm_ttfts:
            return sum(self.warm_ttfts) / len(self.warm_ttfts)
        return 0.0

    @property
    def speedup_ratio(self) -> float:
        if not self.cold_ttfts or not self.warm_ttfts:
            return 1.0
        warm = max(1.0, self.mean_ttft_warm_ms)
        return round(self.mean_ttft_cold_ms / warm, 2)

    @property
    def ttft_reduction_pct(self) -> float:
        if not self.cold_ttfts or not self.warm_ttfts:
            return 0.0
        cold = max(1.0, self.mean_ttft_cold_ms)
        return round(((cold - self.mean_ttft_warm_ms) / cold) * 100, 2)

    @property
    def estimated_input_cost_avoided_usd(self) -> float:
        # tokens_saved_by_compaction is estimated as characters removed // 4.
        # Priced at $0.06 per 1M input tokens (Token Factory's list input price for Nano).
        return round(self.tokens_saved_by_compaction * 0.00000006, 6)


# Global live telemetry singleton (starts at zero; populated strictly by real gateway traffic)
telemetry = GatewayTelemetry()
