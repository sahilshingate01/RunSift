"""Deterministic repository builder for TensorRoute v2 (coupled architecture).

Materializes the clean reference codebase:
- 7 modules: units, radix, scheduler, limiter, codec, telemetry, pipeline
- 56 unit/integration tests across 7 test files in tests/
- Optional SDCS 7-pillar scaffolding and wiring.yaml
"""

from __future__ import annotations

from pathlib import Path

INIT_PY = '''"""TensorRoute: GPU Inference Prefix Router, Batch Scheduler, and Telemetry Engine."""

from tensorroute.codec import SSEChunkAssembler
from tensorroute.limiter import TokenBucketLimiter
from tensorroute.pipeline import Pipeline, build_pipeline
from tensorroute.radix import PrefixRadixTree, RadixNode
from tensorroute.scheduler import BatchRequest, ContinuousBatchScheduler
from tensorroute.telemetry import LatencySample, LatencyWindow
from tensorroute.units import safe_ratio, to_ms

__all__ = [
    "BatchRequest",
    "ContinuousBatchScheduler",
    "LatencySample",
    "LatencyWindow",
    "Pipeline",
    "PrefixRadixTree",
    "RadixNode",
    "SSEChunkAssembler",
    "TokenBucketLimiter",
    "build_pipeline",
    "safe_ratio",
    "to_ms",
]
'''

UNITS_PY = '''"""Unit conversion and bounded numerical ratio helpers.

Subsystem: units
Allowed imports: [] (Leaf module)
"""

from __future__ import annotations


def to_ms(seconds: float) -> float:
    """Converts a duration in seconds to milliseconds."""
    return float(seconds) * 1000.0


def safe_ratio(part: float, whole: float) -> float:
    """Computes part / whole as a dimensionless fraction in [0.0, 1.0].

    Returns a fraction in [0, 1]. Returns 0.0 when whole <= 0.0 or part <= 0.0.
    """
    if whole <= 0.0 or part <= 0.0:
        return 0.0
    return min(1.0, float(part) / float(whole))
'''

RADIX_PY = '''"""Prefix Radix Tree for GPU KV-Cache Block Routing and TTL/LRU Eviction.

Subsystem: radix
Allowed imports: [] (Leaf module)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RadixNode:
    """Represents a token-chunk node in the KV-cache prefix radix tree."""

    token_chunk: tuple[int, ...]
    worker_id: str = ""
    ref_count: int = 0
    last_access_ts: float = 0.0
    ttl_seconds: float = 300.0
    children: dict[tuple[int, ...], RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = None

    def is_expired(self, now: float) -> bool:
        return (now - self.last_access_ts) > self.ttl_seconds


class PrefixRadixTree:
    """Manages token prefix routing to sticky GPU workers with reference counting."""

    def __init__(self, chunk_size: int = 4, default_ttl: float = 300.0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.default_ttl = default_ttl
        self.root = RadixNode(token_chunk=())

    def _chunk_tokens(self, tokens: list[int]) -> list[tuple[int, ...]]:
        n = (len(tokens) // self.chunk_size) * self.chunk_size
        return [
            tuple(tokens[i : i + self.chunk_size])
            for i in range(0, n, self.chunk_size)
        ]

    def insert_prefix(
        self,
        tokens: list[int],
        worker_id: str,
        now: float,
        ttl_seconds: float | None = None,
    ) -> int:
        """Inserts a token sequence into the tree and increments ref_count along the path."""
        chunks = self._chunk_tokens(tokens)
        if not chunks:
            return 0
        ttl = self.default_ttl if ttl_seconds is None else ttl_seconds
        curr = self.root
        for chunk in chunks:
            nxt = curr.children.get(chunk)
            if nxt is None:
                nxt = RadixNode(
                    token_chunk=chunk,
                    worker_id=worker_id,
                    ref_count=0,
                    last_access_ts=now,
                    ttl_seconds=ttl,
                    parent=curr,
                )
                curr.children[chunk] = nxt
            nxt.worker_id = worker_id
            nxt.last_access_ts = now
            nxt.ref_count += 1
            curr = nxt
        return len(chunks) * self.chunk_size

    def match_prefix(self, tokens: list[int], now: float) -> tuple[str | None, int]:
        """Finds the longest non-expired prefix and returns (worker_id, matched_tokens)."""
        chunks = self._chunk_tokens(tokens)
        if not chunks:
            return None, 0
        curr = self.root
        matched_tokens = 0
        matched_worker: str | None = None

        for chunk in chunks:
            nxt = curr.children.get(chunk)
            if nxt is None or nxt.is_expired(now):
                break
            nxt.last_access_ts = now
            matched_tokens += self.chunk_size
            matched_worker = nxt.worker_id
            curr = nxt

        return matched_worker, matched_tokens

    def release_prefix(self, tokens: list[int]) -> int:
        """Decrements ref_count along the prefix path down to zero."""
        chunks = self._chunk_tokens(tokens)
        if not chunks:
            return 0
        curr = self.root
        nodes: list[RadixNode] = []
        for chunk in chunks:
            nxt = curr.children.get(chunk)
            if nxt is None:
                break
            nodes.append(nxt)
            curr = nxt

        decremented = 0
        for node in nodes:
            if node.ref_count > 0:
                node.ref_count -= 1
                decremented += 1
        return decremented

    def evict_expired_leaves(self, now: float) -> int:
        """Evicts expired leaf nodes (ref_count == 0 or expired) and adjusts ancestor counts."""
        leaves: list[RadixNode] = []
        self._collect_leaves(self.root, leaves)
        evicted_count = 0

        for leaf in leaves:
            if leaf is self.root:
                continue
            if leaf.is_expired(now) and not leaf.children:
                parent = leaf.parent
                if parent is not None and leaf.token_chunk in parent.children:
                    del parent.children[leaf.token_chunk]
                    evicted_count += 1
                    anc = parent
                    while anc is not None and anc is not self.root:
                        anc.ref_count = max(0, anc.ref_count - leaf.ref_count)
                        anc = anc.parent

        return evicted_count

    def _collect_leaves(self, node: RadixNode, out: list[RadixNode]) -> None:
        if not node.children:
            if node is not self.root:
                out.append(node)
            return
        for child in list(node.children.values()):
            self._collect_leaves(child, out)
'''

SCHEDULER_PY = '''"""Continuous Batching Request Scheduler with Priority Preemption.

Subsystem: scheduler
Allowed imports: [] (Leaf module)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchRequest:
    """Represents a pending inference request in the continuous batch queue."""

    request_id: str
    prompt_tokens: int
    priority: int = 0
    prefix_hash: str = ""


class ContinuousBatchScheduler:
    """Forms token-budgeted and count-budgeted GPU inference batches."""

    def __init__(self, max_batch_size: int = 4, max_batch_tokens: int = 4096) -> None:
        if max_batch_size <= 0 or max_batch_tokens <= 0:
            raise ValueError("Batch limits must be positive integers")
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self._queue: list[BatchRequest] = []
        self._seq_counter: int = 0
        self._arrival_order: dict[str, int] = {}

    def enqueue(self, req: BatchRequest) -> None:
        """Enqueues a request, preserving FIFO tie-breaking within equal priority."""
        if req.prompt_tokens > self.max_batch_tokens:
            raise ValueError(
                f"Request {req.request_id} ({req.prompt_tokens} tokens) exceeds max_batch_tokens ({self.max_batch_tokens})"
            )
        self._seq_counter += 1
        self._arrival_order[req.request_id] = self._seq_counter
        self._queue.append(req)
        self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))

    def pending_count(self) -> int:
        return len(self._queue)

    def pop_next_batch(self) -> list[BatchRequest]:
        """Greedily packs highest-priority requests that fit within size and token budgets."""
        if not self._queue:
            return []

        batch: list[BatchRequest] = []
        remaining: list[BatchRequest] = []
        running_tokens = 0

        for req in self._queue:
            if (
                len(batch) < self.max_batch_size
                and (running_tokens + req.prompt_tokens) <= self.max_batch_tokens
            ):
                batch.append(req)
                running_tokens += req.prompt_tokens
            else:
                remaining.append(req)

        self._queue = remaining
        return batch

    def drain_all_batches(self, flush_partial: bool = True) -> list[list[BatchRequest]]:
        """Drains the queue into consecutive batches.

        If `flush_partial=True`, all batches including a final batch smaller than
        `max_batch_size` are returned and removed from the queue.
        If `flush_partial=False`, only full batches (`len(batch) == max_batch_size`)
        are emitted, and any partial remainder batch is re-queued in order.
        """
        batches: list[list[BatchRequest]] = []
        while self._queue:
            batch = self.pop_next_batch()
            if not batch:
                break
            if not flush_partial and len(batch) < self.max_batch_size:
                for req in batch:
                    self._queue.append(req)
                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))
                break
            batches.append(batch)
        return batches
'''

LIMITER_PY = '''"""Token-Bucket Rate Limiter with Fractional Refill and Burst Accounting.

Subsystem: limiter
Allowed imports: [units]
"""

from __future__ import annotations

from tensorroute.units import safe_ratio


class TokenBucketLimiter:
    """Enforces token-per-second rate limits with sub-second precision."""

    def __init__(
        self,
        capacity: float,
        refill_rate_per_sec: float,
        initial_tokens: float | None = 0.0,
        start_ts: float = 0.0,
    ) -> None:
        if capacity <= 0 or refill_rate_per_sec <= 0:
            raise ValueError("capacity and refill_rate_per_sec must be positive")
        self.capacity = float(capacity)
        self.refill_rate_per_sec = float(refill_rate_per_sec)
        self.tokens = 0.0 if initial_tokens is None else min(self.capacity, float(initial_tokens))
        self.last_refill_ts = float(start_ts)
        self.total_consumed: float = 0.0
        self.total_rejected: int = 0

    def _refill(self, now: float) -> None:
        """Refills bucket based on exact fractional elapsed time since last_refill_ts."""
        if now < self.last_refill_ts:
            raise ValueError("Monotonic clock violation: now < last_refill_ts")
        elapsed = now - self.last_refill_ts
        if elapsed > 0.0:
            added = elapsed * self.refill_rate_per_sec
            self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill_ts = now

    def available_tokens(self, now: float) -> float:
        self._refill(now)
        return self.tokens

    def utilization(self, now: float) -> float:
        """Returns current bucket utilization as a fraction in [0.0, 1.0]."""
        self._refill(now)
        return safe_ratio(self.capacity - self.tokens, self.capacity)

    def try_acquire(self, requested_tokens: float, now: float) -> tuple[bool, float]:
        """Attempts to consume `requested_tokens` at timestamp `now`.

        Returns `(allowed, wait_seconds)`:
        - If sufficient tokens exist, deducts `requested_tokens` and returns `(True, 0.0)`.
        - Otherwise, leaves token balance unchanged, increments `total_rejected`,
          and returns `(False, deficit / refill_rate_per_sec)`.
        """
        if requested_tokens < 0:
            raise ValueError("requested_tokens must be non-negative")
        self._refill(now)

        if requested_tokens <= self.capacity and self.tokens >= requested_tokens:
            self.tokens -= requested_tokens
            self.total_consumed += requested_tokens
            return True, 0.0

        self.total_rejected += 1
        if requested_tokens > self.capacity:
            return False, float("inf")
        deficit = requested_tokens - self.tokens
        wait_sec = deficit / self.refill_rate_per_sec
        return False, round(wait_sec, 6)
'''

CODEC_PY = '''"""Server-Sent Events (SSE) Chunk Parser and Multi-Chunk Tool Call Assembler.

Subsystem: codec
Allowed imports: [] (Leaf module)
"""

from __future__ import annotations

import json
from typing import Any


class SSEChunkAssembler:
    """Reassembles streamed OpenAI-compatible SSE chunks into complete messages and tool calls."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls_by_index: dict[int, dict[str, Any]] = {}
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cached_tokens: int = 0
        self.done: bool = False

    def feed_line(self, raw_line: str) -> bool:
        """Parses a single SSE wire line. Returns True if [DONE] was received."""
        line = raw_line.strip()
        if not line or line.startswith(":"):
            return False
        if not line.startswith("data:"):
            return False

        payload_str = line[5:].strip()
        if payload_str == "[DONE]":
            self.done = True
            return True

        chunk = json.loads(payload_str)
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = int(usage.get("prompt_tokens") or self.prompt_tokens)
            self.completion_tokens = int(usage.get("completion_tokens") or self.completion_tokens)
            details = usage.get("prompt_tokens_details") or {}
            self.cached_tokens = int(details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or self.cached_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            return False

        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            self.content_parts.append(str(delta["content"]))
        if delta.get("reasoning_content"):
            self.reasoning_parts.append(str(delta["reasoning_content"]))

        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0))
            entry = self.tool_calls_by_index.get(idx)
            if entry is None:
                entry = {
                    "id": tc.get("id") or f"call_{idx}",
                    "type": tc.get("type") or "function",
                    "function": {"name": "", "arguments": ""},
                }
                self.tool_calls_by_index[idx] = entry
            elif tc.get("id"):
                entry["id"] = tc["id"]

            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] += str(fn["name"])
            if fn.get("arguments"):
                entry["function"]["arguments"] += str(fn["arguments"])

        return False

    def finalize(self) -> dict[str, Any]:
        """Returns the assembled assistant message and usage dictionary."""
        ordered_tools = [self.tool_calls_by_index[k] for k in sorted(self.tool_calls_by_index.keys())]
        return {
            "content": "".join(self.content_parts),
            "reasoning_content": "".join(self.reasoning_parts),
            "tool_calls": ordered_tools,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens,
            },
            "done": self.done,
        }
'''

TELEMETRY_PY = '''"""Rolling Latency Window and Percentile Histogram Aggregator.

Subsystem: telemetry
Allowed imports: [units]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tensorroute.units import safe_ratio


@dataclass(frozen=True)
class LatencySample:
    """Single request telemetry measurement."""

    ttft_ms: float
    prompt_tokens: int
    cached_tokens: int
    cache_hit: bool


class LatencyWindow:
    """Maintains a bounded FIFO ring window of latency and cache samples."""

    def __init__(self, max_samples: int = 100) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self._samples: list[LatencySample] = []

    def record(
        self,
        ttft_ms: float,
        prompt_tokens: int,
        cached_tokens: int = 0,
        cache_hit: bool = False,
    ) -> None:
        if ttft_ms < 0 or prompt_tokens < 0 or cached_tokens < 0:
            raise ValueError("Telemetry metrics must be non-negative")
        self._samples.append(
            LatencySample(
                ttft_ms=float(ttft_ms),
                prompt_tokens=int(prompt_tokens),
                cached_tokens=min(int(prompt_tokens), int(cached_tokens)),
                cache_hit=bool(cache_hit),
            )
        )
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples :]

    def count(self) -> int:
        return len(self._samples)

    def percentile_ttft(self, q: float, min_ttft_ms: float = 0.0) -> float:
        """Computes the q-th percentile (0.0 <= q <= 100.0) of TTFT samples >= min_ttft_ms.

        Uses nearest-rank method: `idx = max(0, math.ceil((q / 100.0) * n) - 1)`.
        Returns 0.0 if no samples match the filter.
        """
        if not (0.0 <= q <= 100.0):
            raise ValueError("Percentile q must be between 0.0 and 100.0")
        filtered = sorted(s.ttft_ms for s in self._samples if s.ttft_ms >= min_ttft_ms)
        n = len(filtered)
        if n == 0:
            return 0.0
        if q == 0.0:
            return filtered[0]
        idx = max(0, min(n - 1, math.ceil((q / 100.0) * n) - 1))
        return round(filtered[idx], 3)

    def summary(self, min_ttft_ms: float = 0.0) -> dict[str, float | int]:
        """Returns aggregate window statistics for samples with ttft_ms >= min_ttft_ms."""
        matching = [s for s in self._samples if s.ttft_ms >= min_ttft_ms]
        n = len(matching)
        if n == 0:
            return {
                "count": 0,
                "mean_ttft_ms": 0.0,
                "p50_ttft_ms": 0.0,
                "p95_ttft_ms": 0.0,
                "p99_ttft_ms": 0.0,
                "hit_rate_pct": 0.0,
                "token_reuse_pct": 0.0,
            }

        total_ttft = sum(s.ttft_ms for s in matching)
        hits = sum(1 for s in matching if s.cache_hit)
        total_prompt = sum(s.prompt_tokens for s in matching)
        total_cached = sum(s.cached_tokens for s in matching)

        return {
            "count": n,
            "mean_ttft_ms": round(total_ttft / n, 3),
            "p50_ttft_ms": self.percentile_ttft(50.0, min_ttft_ms=min_ttft_ms),
            "p95_ttft_ms": self.percentile_ttft(95.0, min_ttft_ms=min_ttft_ms),
            "p99_ttft_ms": self.percentile_ttft(99.0, min_ttft_ms=min_ttft_ms),
            "hit_rate_pct": round(safe_ratio(hits, n) * 100.0, 2),
            "token_reuse_pct": round(safe_ratio(total_cached, total_prompt) * 100.0, 2),
        }
'''

PIPELINE_PY = '''"""End-to-End Inference Gateway Pipeline Orchestrator.

Subsystem: pipeline
Allowed imports: [radix, scheduler, limiter, codec, telemetry, units]
"""

from __future__ import annotations

from typing import Any

from tensorroute.codec import SSEChunkAssembler
from tensorroute.limiter import TokenBucketLimiter
from tensorroute.radix import PrefixRadixTree
from tensorroute.scheduler import BatchRequest, ContinuousBatchScheduler
from tensorroute.telemetry import LatencyWindow
from tensorroute.units import to_ms


class Pipeline:
    """Glues prefix routing, continuous scheduling, token limiting, and telemetry."""

    def __init__(
        self,
        limiter: TokenBucketLimiter,
        scheduler: ContinuousBatchScheduler,
        tree: PrefixRadixTree,
        window: LatencyWindow,
    ) -> None:
        self.limiter = limiter
        self.scheduler = scheduler
        self.tree = tree
        self.window = window

    def admit_all(
        self,
        requests: list[BatchRequest],
        start_ts: float,
        tick: float = 0.25,
        deadline_s: float = 30.0,
    ) -> list[str]:
        """Enqueues requests, drains batches, and admits each via limiter within deadline."""
        for req in requests:
            self.scheduler.enqueue(req)

        batches = self.scheduler.drain_all_batches(flush_partial=True)
        admitted: list[str] = []
        now = float(start_ts)
        deadline = start_ts + deadline_s

        for batch in batches:
            for req in batch:
                admitted_req = False
                while now <= deadline:
                    ok, _ = self.limiter.try_acquire(req.prompt_tokens, now)
                    if ok:
                        admitted.append(req.request_id)
                        admitted_req = True
                        break
                    now += tick
                if not admitted_req or now > deadline:
                    break
        return admitted

    def record_completion(self, result: dict[str, Any], ttft_seconds: float) -> None:
        """Reads assembler usage dict and records latency in milliseconds."""
        usage = result.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_tokens = int(usage.get("cached_tokens") or 0)
        self.window.record(
            ttft_ms=to_ms(ttft_seconds),
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            cache_hit=cached_tokens > 0,
        )

    def route(self, tokens: list[int], now: float, min_cached_tokens: int = 8) -> str | None:
        """Returns sticky worker ID if matched tokens >= min_cached_tokens, else None."""
        worker, matched_tokens = self.tree.match_prefix(tokens, now=now)
        if worker is not None and matched_tokens >= min_cached_tokens:
            return worker
        return None


def build_pipeline(
    capacity: float = 1000.0,
    refill_rate: float = 500.0,
    max_batch_size: int = 4,
    max_batch_tokens: int = 4096,
    chunk_size: int = 4,
) -> Pipeline:
    """Constructs a fully configured Pipeline instance."""
    limiter = TokenBucketLimiter(
        capacity=capacity,
        refill_rate_per_sec=refill_rate,
        initial_tokens=capacity,
        start_ts=0.0,
    )
    scheduler = ContinuousBatchScheduler(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
    )
    tree = PrefixRadixTree(chunk_size=chunk_size)
    window = LatencyWindow()
    return Pipeline(
        limiter=limiter,
        scheduler=scheduler,
        tree=tree,
        window=window,
    )
'''

TEST_UNITS_PY = '''"""Unit tests for tensorroute.units (4 tests)."""

from __future__ import annotations

import pytest
from tensorroute.units import safe_ratio, to_ms


def test_to_ms_conversion() -> None:
    assert to_ms(0.2) == pytest.approx(200.0)
    assert to_ms(1.5) == pytest.approx(1500.0)
    assert to_ms(0.0) == 0.0


def test_safe_ratio_valid_fractions() -> None:
    assert safe_ratio(1, 4) == pytest.approx(0.25)
    assert safe_ratio(3, 4) == pytest.approx(0.75)
    assert safe_ratio(4, 4) == pytest.approx(1.0)


def test_safe_ratio_zero_denominator() -> None:
    assert safe_ratio(5, 0) == 0.0
    assert safe_ratio(0, 0) == 0.0


def test_safe_ratio_bounds() -> None:
    assert safe_ratio(-5, 10) == 0.0
    assert safe_ratio(20, 10) == 1.0
'''

TEST_RADIX_PY = '''"""Unit test suite for PrefixRadixTree (10 tests)."""

from __future__ import annotations

import pytest
from tensorroute.radix import PrefixRadixTree


def test_radix_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        PrefixRadixTree(chunk_size=0)


def test_radix_insert_and_exact_match() -> None:
    tree = PrefixRadixTree(chunk_size=1, default_ttl=100.0)
    inserted = tree.insert_prefix([1, 2], worker_id="gpu-0", now=10.0)
    assert inserted == 2
    worker, matched = tree.match_prefix([1, 2], now=20.0)
    assert worker == "gpu-0"
    assert matched == 2


def test_radix_partial_chunk_truncation() -> None:
    tree = PrefixRadixTree(chunk_size=4)
    inserted = tree.insert_prefix([10, 20, 30, 40, 50, 60], worker_id="gpu-1", now=1.0)
    assert inserted == 4
    worker, matched = tree.match_prefix([10, 20, 30, 40, 99, 99], now=2.0)
    assert worker == "gpu-1"


def test_radix_short_sequence_below_chunk_size() -> None:
    tree = PrefixRadixTree(chunk_size=4)
    assert tree.insert_prefix([1, 2], worker_id="gpu-0", now=1.0) == 0
    worker, matched = tree.match_prefix([1, 2], now=2.0)
    assert worker is None
    assert matched == 0


def test_radix_ttl_expiration_blocks_match() -> None:
    tree = PrefixRadixTree(chunk_size=1, default_ttl=50.0)
    tree.insert_prefix([1, 2, 3, 4], worker_id="gpu-2", now=10.0)
    worker_ok, matched_ok = tree.match_prefix([1, 2, 3, 4], now=55.0)
    assert worker_ok == "gpu-2"
    assert matched_ok == 4
    worker_exp, matched_exp = tree.match_prefix([1, 2, 3, 4], now=110.0)
    assert worker_exp is None
    assert matched_exp == 0


def test_radix_release_prefix_decrements_ref_count() -> None:
    tree = PrefixRadixTree(chunk_size=1)
    seq = [1, 2]
    tree.insert_prefix(seq, worker_id="gpu-0", now=1.0)
    tree.insert_prefix(seq, worker_id="gpu-0", now=2.0)
    first_node = tree.root.children[(1,)]
    assert first_node.ref_count == 2
    released = tree.release_prefix(seq)
    assert released == 2
    assert first_node.ref_count == 1


def test_radix_release_floors_at_zero() -> None:
    tree = PrefixRadixTree(chunk_size=1)
    seq = [1]
    tree.insert_prefix(seq, worker_id="gpu-0", now=1.0)
    tree.release_prefix(seq)
    tree.release_prefix(seq)
    assert tree.root.children[(1,)].ref_count == 0


def test_radix_evict_expired_leaf_preserves_shared_parent_ref_count() -> None:
    tree = PrefixRadixTree(chunk_size=1, default_ttl=100.0)
    tree.insert_prefix([1, 2], worker_id="gpu-a", now=10.0, ttl_seconds=20.0)
    tree.insert_prefix([1, 3], worker_id="gpu-b", now=15.0, ttl_seconds=200.0)

    shared_parent = tree.root.children[(1,)]
    assert shared_parent.ref_count == 2

    evicted = tree.evict_expired_leaves(now=50.0)
    assert evicted == 1
    assert (2,) not in shared_parent.children
    assert (3,) in shared_parent.children
    assert shared_parent.ref_count == 1


def test_radix_evict_deep_chain_updates_all_ancestors_once() -> None:
    tree = PrefixRadixTree(chunk_size=1, default_ttl=100.0)
    tree.insert_prefix([1, 2], worker_id="gpu-0", now=10.0, ttl_seconds=100.0)
    tree.insert_prefix([1, 2], worker_id="gpu-0", now=10.0, ttl_seconds=100.0)
    tree.insert_prefix([1, 2, 3], worker_id="gpu-0", now=10.0, ttl_seconds=10.0)

    n1 = tree.root.children[(1,)]
    n2 = n1.children[(2,)]
    assert n1.ref_count == 3
    assert n2.ref_count == 3

    evicted = tree.evict_expired_leaves(now=30.0)
    assert evicted == 1
    assert n1.ref_count == 2
    assert n2.ref_count == 2


def test_radix_match_counts_tokens() -> None:
    tree = PrefixRadixTree(chunk_size=4, default_ttl=100.0)
    tree.insert_prefix([1, 2, 3, 4, 5, 6, 7, 8], worker_id="gpu-0", now=10.0)
    worker, matched = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8], now=20.0)
    assert worker == "gpu-0"
    assert matched == 8
'''

TEST_SCHEDULER_PY = '''"""Unit test suite for ContinuousBatchScheduler (8 tests)."""

from __future__ import annotations

import pytest
from tensorroute.scheduler import BatchRequest, ContinuousBatchScheduler


def test_scheduler_invalid_limits() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ContinuousBatchScheduler(max_batch_size=0, max_batch_tokens=100)


def test_scheduler_request_exceeding_max_batch_tokens_raises() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=4, max_batch_tokens=500)
    with pytest.raises(ValueError, match="exceeds max_batch_tokens"):
        sched.enqueue(BatchRequest("big", prompt_tokens=600))


def test_scheduler_fifo_order_for_equal_priority() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=4, max_batch_tokens=2000)
    sched.enqueue(BatchRequest("req-1", prompt_tokens=100, priority=0))
    sched.enqueue(BatchRequest("req-2", prompt_tokens=100, priority=0))
    sched.enqueue(BatchRequest("req-3", prompt_tokens=100, priority=0))
    batch = sched.pop_next_batch()
    assert [r.request_id for r in batch] == ["req-1", "req-2", "req-3"]


def test_scheduler_priority_preemption_over_fifo() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=2, max_batch_tokens=2000)
    sched.enqueue(BatchRequest("low-1", prompt_tokens=100, priority=1))
    sched.enqueue(BatchRequest("high-1", prompt_tokens=100, priority=10))
    sched.enqueue(BatchRequest("low-2", prompt_tokens=100, priority=1))
    batch = sched.pop_next_batch()
    assert [r.request_id for r in batch] == ["high-1", "low-1"]


def test_scheduler_batch_token_budget_packing() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=4, max_batch_tokens=500)
    sched.enqueue(BatchRequest("r1", prompt_tokens=300, priority=10))
    sched.enqueue(BatchRequest("r2", prompt_tokens=300, priority=9))
    sched.enqueue(BatchRequest("r3", prompt_tokens=150, priority=8))
    b1 = sched.pop_next_batch()
    assert [r.request_id for r in b1] == ["r1", "r3"]


def test_scheduler_drain_all_exact_multiple() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=2, max_batch_tokens=2000)
    for i in range(4):
        sched.enqueue(BatchRequest(f"r{i}", prompt_tokens=200))
    batches = sched.drain_all_batches(flush_partial=True)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert sched.pending_count() == 0


def test_scheduler_drain_all_retains_partial_remainder_when_false() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)
    for i in range(5):
        sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=10 - i))
    batches = sched.drain_all_batches(flush_partial=False)
    assert len(batches) == 1
    assert [r.request_id for r in batches[0]] == ["req-0", "req-1", "req-2"]
    assert sched.pending_count() == 2
    tail = sched.pop_next_batch()
    assert [r.request_id for r in tail] == ["req-3", "req-4"]


def test_scheduler_flushes_partial_tail() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)
    for i in range(5):
        sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=5 - i))
    batches = sched.drain_all_batches(flush_partial=True)
    assert len(batches) == 2
    assert [r.request_id for r in batches[0]] == ["req-0", "req-1", "req-2"]
    assert [r.request_id for r in batches[1]] == ["req-3", "req-4"]
    assert sched.pending_count() == 0
'''

TEST_LIMITER_PY = '''"""Unit test suite for TokenBucketLimiter (9 tests)."""

from __future__ import annotations

import pytest
from tensorroute.limiter import TokenBucketLimiter


def test_limiter_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TokenBucketLimiter(capacity=0, refill_rate_per_sec=10)


def test_limiter_initial_full_capacity_acquire() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=50.0, initial_tokens=100.0, start_ts=0.0)
    ok, wait = lim.try_acquire(60.0, now=0.0)
    assert ok is True
    assert wait == 0.0
    assert lim.available_tokens(now=0.0) == 40.0


def test_limiter_monotonic_clock_guard() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=10.0, initial_tokens=100.0, start_ts=10.0)
    with pytest.raises(ValueError, match="Monotonic clock violation"):
        lim.available_tokens(now=9.5)


def test_limiter_caps_at_capacity() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=50.0, initial_tokens=80.0, start_ts=0.0)
    assert lim.available_tokens(now=10.0) == 100.0


def test_limiter_insufficient_tokens_returns_exact_wait_time() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=40.0, initial_tokens=10.0, start_ts=0.0)
    ok, wait = lim.try_acquire(30.0, now=0.0)
    assert ok is False
    assert wait == pytest.approx(0.5)
    assert lim.total_rejected == 1
    assert lim.available_tokens(now=0.0) == 10.0


def test_limiter_request_exceeding_capacity_returns_inf_wait() -> None:
    lim = TokenBucketLimiter(capacity=50.0, refill_rate_per_sec=10.0, initial_tokens=50.0, start_ts=0.0)
    ok, wait = lim.try_acquire(75.0, now=0.0)
    assert ok is False
    assert wait == float("inf")
    assert lim.total_rejected == 1


def test_limiter_negative_request_raises() -> None:
    lim = TokenBucketLimiter(capacity=50.0, refill_rate_per_sec=10.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        lim.try_acquire(-5.0, now=1.0)


def test_utilization_is_fraction() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=10.0, initial_tokens=75.0, start_ts=0.0)
    assert lim.utilization(now=0.0) == pytest.approx(0.25)


def test_limiter_starts_empty_by_default() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=10.0)
    assert lim.available_tokens(now=0.0) == 0.0
'''

TEST_CODEC_PY = '''"""Unit test suite for SSEChunkAssembler (8 tests)."""

from __future__ import annotations

from tensorroute.codec import SSEChunkAssembler


def test_codec_ignores_comments_and_blank_lines() -> None:
    asm = SSEChunkAssembler()
    assert asm.feed_line("") is False
    assert asm.feed_line(": keep-alive") is False
    assert asm.feed_line("event: message") is False


def test_codec_handles_done_sentinel() -> None:
    asm = SSEChunkAssembler()
    assert asm.feed_line("data: [DONE]") is True
    assert asm.finalize()["done"] is True


def test_codec_assembles_content_and_reasoning_deltas() -> None:
    asm = SSEChunkAssembler()
    asm.feed_line('data: {"choices": [{"delta": {"reasoning_content": "Step 1. "}}]}')
    asm.feed_line('data: {"choices": [{"delta": {"reasoning_content": "Step 2."}}]}')
    asm.feed_line('data: {"choices": [{"delta": {"content": "Hello "}}]}')
    asm.feed_line('data: {"choices": [{"delta": {"content": "world!"}}]}')
    res = asm.finalize()
    assert res["reasoning_content"] == "Step 1. Step 2."
    assert res["content"] == "Hello world!"


def test_codec_concatenates_single_tool_call_arguments_across_chunks() -> None:
    asm = SSEChunkAssembler()
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_abc", "function": {"name": "read_file", "arguments": '{"path": '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"src/radix.py", '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"limit": 50}'}}]}}]},
    ]
    for c in chunks:
        import json
        asm.feed_line(f"data: {json.dumps(c)}")
    res = asm.finalize()
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["function"]["name"] == "read_file"
    assert res["tool_calls"][0]["function"]["arguments"] == '{"path": "src/radix.py", "limit": 50}'


def test_codec_handles_interleaved_multi_tool_calls() -> None:
    asm = SSEChunkAssembler()
    import json
    asm.feed_line(f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "f0", "arguments": "a"}}]}}]})}')
    asm.feed_line(f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 2, "id": "c2", "function": {"name": "f2", "arguments": "b"}}]}}]})}')
    res = asm.finalize()
    assert [t["id"] for t in res["tool_calls"]] == ["c0", "c2"]


def test_codec_extracts_nested_cached_tokens_usage() -> None:
    asm = SSEChunkAssembler()
    asm.feed_line(
        'data: {"choices": [], "usage": {"prompt_tokens": 2048, "completion_tokens": 32, "prompt_tokens_details": {"cached_tokens": 1792}}}'
    )
    res = asm.finalize()
    assert res["usage"] == {"prompt_tokens": 2048, "completion_tokens": 32, "cached_tokens": 1792}


def test_codec_extracts_top_level_prompt_cache_hit_tokens() -> None:
    asm = SSEChunkAssembler()
    asm.feed_line(
        'data: {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "prompt_cache_hit_tokens": 800}}'
    )
    assert asm.finalize()["usage"]["cached_tokens"] == 800


def test_codec_preserves_usage_when_later_chunk_has_no_usage() -> None:
    asm = SSEChunkAssembler()
    asm.feed_line('data: {"choices": [], "usage": {"prompt_tokens": 512, "completion_tokens": 10}}')
    asm.feed_line('data: {"choices": [{"delta": {"content": "end"}}]}')
    assert asm.finalize()["usage"]["prompt_tokens"] == 512
'''

TEST_TELEMETRY_PY = '''"""Unit test suite for LatencyWindow (10 tests)."""

from __future__ import annotations

import pytest
from tensorroute.telemetry import LatencyWindow


def test_telemetry_invalid_max_samples() -> None:
    with pytest.raises(ValueError, match="max_samples must be positive"):
        LatencyWindow(max_samples=0)


def test_telemetry_rejects_negative_metrics() -> None:
    win = LatencyWindow()
    with pytest.raises(ValueError, match="must be non-negative"):
        win.record(ttft_ms=-1.0, prompt_tokens=100)


def test_telemetry_empty_window_summary_and_percentile() -> None:
    win = LatencyWindow()
    assert win.percentile_ttft(99.0) == 0.0
    s = win.summary()
    assert s["count"] == 0
    assert s["mean_ttft_ms"] == 0.0
    assert s["p99_ttft_ms"] == 0.0


def test_telemetry_single_sample_percentiles_do_not_overflow() -> None:
    win = LatencyWindow()
    win.record(ttft_ms=425.5, prompt_tokens=1000, cached_tokens=800, cache_hit=True)
    assert win.percentile_ttft(0.0) == 425.5
    assert win.percentile_ttft(50.0) == 425.5
    assert win.percentile_ttft(95.0) == 425.5
    assert win.percentile_ttft(99.0) == 425.5
    assert win.percentile_ttft(100.0) == 425.5


def test_telemetry_filtered_empty_window_returns_zero_without_division_error() -> None:
    win = LatencyWindow()
    win.record(ttft_ms=50.0, prompt_tokens=500, cached_tokens=400, cache_hit=True)
    win.record(ttft_ms=80.0, prompt_tokens=500, cached_tokens=400, cache_hit=True)
    s = win.summary(min_ttft_ms=200.0)
    assert s["count"] == 0
    assert s["mean_ttft_ms"] == 0.0
    assert s["p99_ttft_ms"] == 0.0


def test_telemetry_ten_sample_nearest_rank_percentiles() -> None:
    win = LatencyWindow()
    for i in range(1, 11):
        win.record(ttft_ms=float(i * 100), prompt_tokens=100, cached_tokens=50, cache_hit=(i % 2 == 0))
    assert win.percentile_ttft(50.0) == 500.0
    assert win.percentile_ttft(90.0) == 900.0
    assert win.percentile_ttft(99.0) == 1000.0


def test_telemetry_ring_buffer_evicts_oldest() -> None:
    win = LatencyWindow(max_samples=3)
    for val in [100.0, 200.0, 300.0, 400.0]:
        win.record(ttft_ms=val, prompt_tokens=100)
    assert win.count() == 3
    assert win.percentile_ttft(0.0) == 200.0


def test_telemetry_invalid_percentile_range() -> None:
    win = LatencyWindow()
    with pytest.raises(ValueError, match="between 0.0 and 100.0"):
        win.percentile_ttft(105.0)


def test_telemetry_hit_rate_is_percentage() -> None:
    win = LatencyWindow()
    win.record(ttft_ms=100.0, prompt_tokens=1000, cached_tokens=0, cache_hit=False)
    win.record(ttft_ms=200.0, prompt_tokens=1000, cached_tokens=500, cache_hit=True)
    s = win.summary()
    assert s["hit_rate_pct"] == 50.0


def test_telemetry_token_reuse_is_percentage() -> None:
    win = LatencyWindow()
    win.record(ttft_ms=100.0, prompt_tokens=1000, cached_tokens=0, cache_hit=False)
    win.record(ttft_ms=200.0, prompt_tokens=1000, cached_tokens=900, cache_hit=True)
    s = win.summary()
    assert s["token_reuse_pct"] == 45.0
'''

TEST_PIPELINE_PY = '''"""Integration test suite for Pipeline (7 tests)."""

from __future__ import annotations

import pytest
from tensorroute.pipeline import build_pipeline
from tensorroute.scheduler import BatchRequest


def test_pipeline_records_latency_in_ms() -> None:
    pipeline = build_pipeline()
    pipeline.record_completion(
        result={"usage": {"prompt_tokens": 100, "cached_tokens": 50}},
        ttft_seconds=0.2,
    )
    s = pipeline.window.summary()
    assert s["p50_ttft_ms"] == pytest.approx(200.0)
    assert s["mean_ttft_ms"] == pytest.approx(200.0)


def test_pipeline_admits_all_requests_including_tail() -> None:
    pipeline = build_pipeline(
        capacity=400.0,
        refill_rate=200.0,
        max_batch_size=4,
        max_batch_tokens=1000,
        chunk_size=4,
    )
    reqs = [
        BatchRequest("r0", prompt_tokens=100, priority=5),
        BatchRequest("r1", prompt_tokens=100, priority=4),
        BatchRequest("r2", prompt_tokens=100, priority=3),
        BatchRequest("r3", prompt_tokens=100, priority=2),
        BatchRequest("r4", prompt_tokens=50, priority=1),
    ]
    admitted = pipeline.admit_all(reqs, start_ts=0.0, tick=0.25, deadline_s=2.0)
    assert admitted == ["r0", "r1", "r2", "r3", "r4"]


def test_pipeline_immediate_startup_admission() -> None:
    pipeline = build_pipeline(
        capacity=500.0,
        refill_rate=100.0,
        max_batch_size=4,
        max_batch_tokens=1000,
        chunk_size=4,
    )
    reqs = [
        BatchRequest("init_0", prompt_tokens=100),
        BatchRequest("init_1", prompt_tokens=100),
        BatchRequest("init_2", prompt_tokens=100),
        BatchRequest("init_3", prompt_tokens=100),
    ]
    admitted = pipeline.admit_all(reqs, start_ts=0.0, tick=0.25, deadline_s=0.1)
    assert admitted == ["init_0", "init_1", "init_2", "init_3"]


def test_pipeline_routes_to_warm_worker() -> None:
    pipeline = build_pipeline(chunk_size=4)
    pipeline.tree.insert_prefix([1, 2, 3, 4, 5, 6, 7, 8], worker_id="gpu-warm", now=10.0)
    worker = pipeline.route([1, 2, 3, 4, 5, 6, 7, 8], now=20.0, min_cached_tokens=8)
    assert worker == "gpu-warm"


def test_pipeline_does_not_route_below_threshold() -> None:
    pipeline = build_pipeline(chunk_size=4)
    pipeline.tree.insert_prefix([1, 2, 3, 4], worker_id="gpu-warm", now=10.0)
    worker = pipeline.route([1, 2, 3, 4], now=20.0, min_cached_tokens=8)
    assert worker is None


def test_pipeline_cold_route_returns_none() -> None:
    pipeline = build_pipeline(chunk_size=4)
    worker = pipeline.route([99, 98, 97, 96], now=10.0, min_cached_tokens=4)
    assert worker is None


def test_pipeline_empty_admission() -> None:
    pipeline = build_pipeline()
    assert pipeline.admit_all([], start_ts=0.0) == []
'''

WIRING_YAML = """version: "1.9.0"

subsystems:
  units:
    path: "src/tensorroute/units.py"
    description: "Unit conversion and bounded numerical ratio helpers."
    allowed_imports: []

  radix:
    path: "src/tensorroute/radix.py"
    description: "Prefix Radix Tree for GPU KV-Cache routing and leaf TTL eviction."
    allowed_imports: []

  scheduler:
    path: "src/tensorroute/scheduler.py"
    description: "Continuous batching request scheduler with priority preemption."
    allowed_imports: []

  limiter:
    path: "src/tensorroute/limiter.py"
    description: "Token-bucket rate limiter with fractional sub-second refill."
    allowed_imports:
      - units
      - tensorroute.units

  codec:
    path: "src/tensorroute/codec.py"
    description: "SSE chunk parser and multi-chunk tool-call delta assembler."
    allowed_imports: []

  telemetry:
    path: "src/tensorroute/telemetry.py"
    description: "Rolling latency window and nearest-rank percentile aggregator."
    allowed_imports:
      - units
      - tensorroute.units

  pipeline:
    path: "src/tensorroute/pipeline.py"
    description: "End-to-End Inference Gateway Pipeline Orchestrator."
    allowed_imports:
      - radix
      - scheduler
      - limiter
      - codec
      - telemetry
      - units
      - tensorroute.radix
      - tensorroute.scheduler
      - tensorroute.limiter
      - tensorroute.codec
      - tensorroute.telemetry
      - tensorroute.units
"""

SPINE_MD = """# Constitutional Invariants (The Spine)
<!-- SPEC-001 Pillar 1 | Mutability: IMMUTABLE -->

## 1. Domain Axioms
- `tensorroute` is a deterministic, zero-external-dependency Python library for GPU prefix routing, continuous batch scheduling, token-bucket rate limiting, SSE chunk assembly, rolling telemetry, and pipeline orchestration.
- Import boundaries are strictly governed by `wiring.yaml`.
"""

APP_MAP_MD = """# Topographical System Atlas (The Map)
<!-- SPEC-001 Pillar 2 -->

- `src/tensorroute/units.py`: Unit conversions and bounded ratio helper functions.
- `src/tensorroute/radix.py`: PrefixRadixTree and RadixNode for KV-cache prefix routing.
- `src/tensorroute/scheduler.py`: ContinuousBatchScheduler and BatchRequest queue.
- `src/tensorroute/limiter.py`: TokenBucketLimiter with fractional refill.
- `src/tensorroute/codec.py`: SSEChunkAssembler for streaming tool calls and usage.
- `src/tensorroute/telemetry.py`: LatencyWindow for rolling percentiles and summaries.
- `src/tensorroute/pipeline.py`: Pipeline orchestrator and build_pipeline factory.
- `tests/test_units.py`: Unit tests for units module.
- `tests/test_radix.py`: Unit tests for radix module.
- `tests/test_scheduler.py`: Unit tests for scheduler module.
- `tests/test_limiter.py`: Unit tests for limiter module.
- `tests/test_codec.py`: Unit tests for codec module.
- `tests/test_telemetry.py`: Unit tests for telemetry module.
- `tests/test_pipeline.py`: Integration tests for pipeline module.
"""

DECISIONS_MD = """# Architectural Decision Log (Decisions)
<!-- SPEC-001 Pillar 5 -->

### [ADR-001] Token-Bucket Rate Limiting vs Fixed-Window Counter
- **Status:** ACCEPTED
- **Rationale:** Smooth fractional burst absorption without fixed-boundary spikes.

### [ADR-002] Prefix Radix Tree vs Flat Hash Indexing
- **Status:** ACCEPTED
- **Rationale:** Enables longest-prefix matching and shared KV-cache branch reuse.

### [ADR-003] Pipeline Factory Encapsulation
- **Status:** ACCEPTED
- **Rationale:** Centralizes component wiring so client code and tests build unified pipelines.
"""

ROADMAP_MD = """# Teleological North Star (The Roadmap)
<!-- SPEC-001 Pillar 3 -->

## Milestone 1: Zero-Defect Core Routing, Pipeline & Telemetry
- **[INTENT]:** All 56 unit/integration tests pass (`56 passed, 0 failed`).
- **[MEASURED]:** Verified via `pytest -o pythonpath=src -vv`.
"""

EVALS_MD = """# Positive Episodic Memory (Baseline Scorecard)
<!-- SPEC-001 Pillar 7 -->

- Full Test Suite Command: `pytest -o pythonpath=src -vv`
- Baseline Target: `56 passed` across 7 test modules.
"""

STATE_MD = """# Dynamic Working Memory (The Blackboard)
<!-- SPEC-001 Pillar 4 | Budget: <= 350 Tokens -->

## Current Objective
- Maintain zero defect state across all 7 modules and verify all 56 tests pass.

## Status & Gate Verification
- Topology: 7 modules (`units`, `radix`, `scheduler`, `limiter`, `codec`, `telemetry`, `pipeline`).

## Immediate Next Action (Post-Compact)
- Run `pytest`, inspect failing tests, apply minimal surgical edits, and verify.
"""

AGENTS_MD = """# Agent Instructions (SDCS Standard)
1. Tests are read-only: do not edit files in `tests/`.
2. Inspect failing tests with `run_pytest`, locate relevant source files in `src/tensorroute/`.
3. Apply minimal edits with `edit_file` and verify with `run_pytest`.
"""

NOTES_MD = f"""# Project Notes: TensorRoute v2 (Coupled)

{SPINE_MD}

{APP_MAP_MD}

{DECISIONS_MD}

{EVALS_MD}
"""


def materialize_clean_repo(dest_dir: Path, arm: str = "arm_b", include_sdcs: bool = False) -> None:
    """Writes the clean TensorRoute v2 codebase into dest_dir."""
    src_pkg = dest_dir / "src" / "tensorroute"
    tests_dir = dest_dir / "tests"
    src_pkg.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (src_pkg / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (src_pkg / "units.py").write_text(UNITS_PY, encoding="utf-8")
    (src_pkg / "radix.py").write_text(RADIX_PY, encoding="utf-8")
    (src_pkg / "scheduler.py").write_text(SCHEDULER_PY, encoding="utf-8")
    (src_pkg / "limiter.py").write_text(LIMITER_PY, encoding="utf-8")
    (src_pkg / "codec.py").write_text(CODEC_PY, encoding="utf-8")
    (src_pkg / "telemetry.py").write_text(TELEMETRY_PY, encoding="utf-8")
    (src_pkg / "pipeline.py").write_text(PIPELINE_PY, encoding="utf-8")

    (tests_dir / "test_units.py").write_text(TEST_UNITS_PY, encoding="utf-8")
    (tests_dir / "test_radix.py").write_text(TEST_RADIX_PY, encoding="utf-8")
    (tests_dir / "test_scheduler.py").write_text(TEST_SCHEDULER_PY, encoding="utf-8")
    (tests_dir / "test_limiter.py").write_text(TEST_LIMITER_PY, encoding="utf-8")
    (tests_dir / "test_codec.py").write_text(TEST_CODEC_PY, encoding="utf-8")
    (tests_dir / "test_telemetry.py").write_text(TEST_TELEMETRY_PY, encoding="utf-8")
    (tests_dir / "test_pipeline.py").write_text(TEST_PIPELINE_PY, encoding="utf-8")

    if include_sdcs:
        (dest_dir / "wiring.yaml").write_text(WIRING_YAML, encoding="utf-8")
        (dest_dir / "notes.md").write_text(NOTES_MD, encoding="utf-8")
