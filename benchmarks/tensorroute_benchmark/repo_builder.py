"""Deterministic repository builder for the 100% original `tensorroute` package.

Materializes the clean reference codebase (5 modules + 45 unit tests across 5 test files)
plus SDCS 7-pillar scaffolding (`arm_b`) or mechanically compiled flat notes (`arm_a`).
"""

from __future__ import annotations

from pathlib import Path

INIT_PY = '''"""TensorRoute: GPU Inference Prefix Router, Batch Scheduler, and Telemetry Engine."""

from tensorroute.codec import SSEChunkAssembler
from tensorroute.limiter import TokenBucketLimiter
from tensorroute.radix import PrefixRadixTree, RadixNode
from tensorroute.scheduler import BatchRequest, ContinuousBatchScheduler
from tensorroute.telemetry import LatencyWindow

__all__ = [
    "BatchRequest",
    "ContinuousBatchScheduler",
    "LatencyWindow",
    "PrefixRadixTree",
    "RadixNode",
    "SSEChunkAssembler",
    "TokenBucketLimiter",
]
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
            nxt.ttl_seconds = ttl
            nxt.ref_count += 1
            curr = nxt
        return len(chunks) * self.chunk_size

    def match_prefix(self, tokens: list[int], now: float) -> tuple[str | None, int]:
        """Finds the longest non-expired prefix match and updates last_access_ts."""
        chunks = self._chunk_tokens(tokens)
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
        """Decrements reference counts along the prefix path by 1 (floor at 0)."""
        chunks = self._chunk_tokens(tokens)
        curr = self.root
        released_nodes = 0
        for chunk in chunks:
            nxt = curr.children.get(chunk)
            if nxt is None:
                break
            nxt.ref_count = max(0, nxt.ref_count - 1)
            released_nodes += 1
            curr = nxt
        return released_nodes

    def evict_expired_leaves(self, now: float) -> int:
        """Evicts expired leaf nodes that have no remaining child branches.

        When an expired leaf is removed from its parent's children map,
        its own path references are no longer held by the leaf, so each
        ancestor up to (but excluding) root has its ref_count decremented
        by `leaf.ref_count` (clamped at 0).
        """
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
Allowed imports: [] (Leaf module)
"""

from __future__ import annotations


class TokenBucketLimiter:
    """Enforces token-per-second rate limits with sub-second precision."""

    def __init__(
        self,
        capacity: float,
        refill_rate_per_sec: float,
        initial_tokens: float | None = None,
        start_ts: float = 0.0,
    ) -> None:
        if capacity <= 0 or refill_rate_per_sec <= 0:
            raise ValueError("capacity and refill_rate_per_sec must be positive")
        self.capacity = float(capacity)
        self.refill_rate_per_sec = float(refill_rate_per_sec)
        self.tokens = self.capacity if initial_tokens is None else min(self.capacity, float(initial_tokens))
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
Allowed imports: [] (Leaf module)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
            "hit_rate_pct": round((hits / n) * 100.0, 2),
            "token_reuse_pct": (
                round((total_cached / total_prompt) * 100.0, 2)
                if total_prompt > 0
                else 0.0
            ),
        }
'''

TEST_RADIX_PY = '''"""Unit test suite for PrefixRadixTree (9 tests)."""

from __future__ import annotations

import pytest
from tensorroute.radix import PrefixRadixTree


def test_radix_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        PrefixRadixTree(chunk_size=0)


def test_radix_insert_and_exact_match() -> None:
    tree = PrefixRadixTree(chunk_size=4, default_ttl=100.0)
    inserted = tree.insert_prefix([1, 2, 3, 4, 5, 6, 7, 8], worker_id="gpu-0", now=10.0)
    assert inserted == 8
    worker, matched = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8], now=20.0)
    assert worker == "gpu-0"
    assert matched == 8


def test_radix_partial_chunk_truncation() -> None:
    tree = PrefixRadixTree(chunk_size=4)
    inserted = tree.insert_prefix([10, 20, 30, 40, 50, 60], worker_id="gpu-1", now=1.0)
    assert inserted == 4
    worker, matched = tree.match_prefix([10, 20, 30, 40, 99, 99], now=2.0)
    assert worker == "gpu-1"
    assert matched == 4


def test_radix_short_sequence_below_chunk_size() -> None:
    tree = PrefixRadixTree(chunk_size=4)
    assert tree.insert_prefix([1, 2], worker_id="gpu-0", now=1.0) == 0
    worker, matched = tree.match_prefix([1, 2], now=2.0)
    assert worker is None
    assert matched == 0


def test_radix_ttl_expiration_blocks_match() -> None:
    tree = PrefixRadixTree(chunk_size=4, default_ttl=50.0)
    tree.insert_prefix([1, 2, 3, 4], worker_id="gpu-2", now=10.0)
    worker_ok, matched_ok = tree.match_prefix([1, 2, 3, 4], now=55.0)
    assert worker_ok == "gpu-2"
    assert matched_ok == 4
    # Last access was updated to 55.0; at now=110.0 (elapsed 55 > 50) it expires
    worker_exp, matched_exp = tree.match_prefix([1, 2, 3, 4], now=110.0)
    assert worker_exp is None
    assert matched_exp == 0


def test_radix_release_prefix_decrements_ref_count() -> None:
    tree = PrefixRadixTree(chunk_size=4)
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    tree.insert_prefix(seq, worker_id="gpu-0", now=1.0)
    tree.insert_prefix(seq, worker_id="gpu-0", now=2.0)
    first_node = tree.root.children[(1, 2, 3, 4)]
    assert first_node.ref_count == 2
    released = tree.release_prefix(seq)
    assert released == 2
    assert first_node.ref_count == 1


def test_radix_release_floors_at_zero() -> None:
    tree = PrefixRadixTree(chunk_size=4)
    seq = [1, 2, 3, 4]
    tree.insert_prefix(seq, worker_id="gpu-0", now=1.0)
    tree.release_prefix(seq)
    tree.release_prefix(seq)
    assert tree.root.children[(1, 2, 3, 4)].ref_count == 0


def test_radix_evict_expired_leaf_preserves_shared_parent_ref_count() -> None:
    tree = PrefixRadixTree(chunk_size=4, default_ttl=100.0)
    # Branch A: [1,2,3,4, 5,6,7,8] inserted at t=10 with short TTL=20
    tree.insert_prefix([1, 2, 3, 4, 5, 6, 7, 8], worker_id="gpu-a", now=10.0, ttl_seconds=20.0)
    # Branch B shares [1,2,3,4] and extends to [9,10,11,12] at t=15 with long TTL=200
    tree.insert_prefix([1, 2, 3, 4, 9, 10, 11, 12], worker_id="gpu-b", now=15.0, ttl_seconds=200.0)

    shared_parent = tree.root.children[(1, 2, 3, 4)]
    assert shared_parent.ref_count == 2

    # At t=50, leaf (5,6,7,8) is expired (50 - 10 = 40 > 20), while (9,10,11,12) is active
    evicted = tree.evict_expired_leaves(now=50.0)
    assert evicted == 1
    assert (5, 6, 7, 8) not in shared_parent.children
    assert (9, 10, 11, 12) in shared_parent.children
    # Shared parent must retain ref_count == 1 for active Branch B (not underflow to 0!)
    assert shared_parent.ref_count == 1


def test_radix_evict_deep_chain_updates_all_ancestors_once() -> None:
    tree = PrefixRadixTree(chunk_size=2, default_ttl=100.0)
    # Insert [1,2, 3,4] twice (active) and [1,2, 3,4, 5,6] once (short ttl=10)
    tree.insert_prefix([1, 2, 3, 4], worker_id="gpu-0", now=10.0, ttl_seconds=100.0)
    tree.insert_prefix([1, 2, 3, 4], worker_id="gpu-0", now=10.0, ttl_seconds=100.0)
    tree.insert_prefix([1, 2, 3, 4, 5, 6], worker_id="gpu-0", now=10.0, ttl_seconds=10.0)

    n1 = tree.root.children[(1, 2)]
    n2 = n1.children[(3, 4)]
    assert n1.ref_count == 3
    assert n2.ref_count == 3

    evicted = tree.evict_expired_leaves(now=30.0)
    assert evicted == 1
    assert n1.ref_count == 2
    assert n2.ref_count == 2
'''

TEST_SCHEDULER_PY = '''"""Unit test suite for ContinuousBatchScheduler (9 tests)."""

from __future__ import annotations

import pytest
from tensorroute.scheduler import BatchRequest, ContinuousBatchScheduler


def test_scheduler_invalid_limits() -> None:
    with pytest.raises(ValueError, match="Batch limits must be positive"):
        ContinuousBatchScheduler(max_batch_size=0, max_batch_tokens=1024)


def test_scheduler_rejects_oversized_single_request() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=4, max_batch_tokens=512)
    with pytest.raises(ValueError, match="exceeds max_batch_tokens"):
        sched.enqueue(BatchRequest(request_id="r1", prompt_tokens=600))


def test_scheduler_priority_ordering_and_fifo_tiebreak() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)
    sched.enqueue(BatchRequest("low-1", prompt_tokens=100, priority=1))
    sched.enqueue(BatchRequest("high-1", prompt_tokens=100, priority=10))
    sched.enqueue(BatchRequest("high-2", prompt_tokens=100, priority=10))
    batch = sched.pop_next_batch()
    assert [r.request_id for r in batch] == ["high-1", "high-2", "low-1"]


def test_scheduler_token_budget_packing() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=4, max_batch_tokens=1000)
    sched.enqueue(BatchRequest("r1", prompt_tokens=600, priority=5))
    sched.enqueue(BatchRequest("r2", prompt_tokens=600, priority=4))
    sched.enqueue(BatchRequest("r3", prompt_tokens=300, priority=3))
    # r1 (600) + r3 (300) <= 1000 fit together; r2 (600) stays for next batch
    batch1 = sched.pop_next_batch()
    assert [r.request_id for r in batch1] == ["r1", "r3"]
    assert sched.pending_count() == 1


def test_scheduler_pop_empty_returns_empty_list() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=2, max_batch_tokens=1000)
    assert sched.pop_next_batch() == []


def test_scheduler_drain_all_exact_multiple() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=2, max_batch_tokens=2000)
    for i in range(4):
        sched.enqueue(BatchRequest(f"r{i}", prompt_tokens=200))
    batches = sched.drain_all_batches(flush_partial=True)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert sched.pending_count() == 0


def test_scheduler_drain_all_flushes_partial_remainder_when_true() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)
    for i in range(5):
        sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=5 - i))
    batches = sched.drain_all_batches(flush_partial=True)
    assert len(batches) == 2
    assert [r.request_id for r in batches[0]] == ["req-0", "req-1", "req-2"]
    assert [r.request_id for r in batches[1]] == ["req-3", "req-4"]
    assert sched.pending_count() == 0


def test_scheduler_drain_all_retains_partial_remainder_when_false() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)
    for i in range(5):
        sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=10 - i))
    batches = sched.drain_all_batches(flush_partial=False)
    assert len(batches) == 1
    assert [r.request_id for r in batches[0]] == ["req-0", "req-1", "req-2"]
    assert sched.pending_count() == 2
    # Remaining 2 requests must still be in queue in priority order
    tail = sched.pop_next_batch()
    assert [r.request_id for r in tail] == ["req-3", "req-4"]


def test_scheduler_drain_single_item_with_flush_partial_true() -> None:
    sched = ContinuousBatchScheduler(max_batch_size=4, max_batch_tokens=4096)
    sched.enqueue(BatchRequest("solo", prompt_tokens=128))
    batches = sched.drain_all_batches(flush_partial=True)
    assert len(batches) == 1
    assert [r.request_id for r in batches[0]] == ["solo"]
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
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=50.0, start_ts=0.0)
    ok, wait = lim.try_acquire(60.0, now=0.0)
    assert ok is True
    assert wait == 0.0
    assert lim.available_tokens(now=0.0) == 40.0


def test_limiter_monotonic_clock_guard() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=10.0, start_ts=10.0)
    with pytest.raises(ValueError, match="Monotonic clock violation"):
        lim.available_tokens(now=9.5)


def test_limiter_subsecond_fractional_refill() -> None:
    lim = TokenBucketLimiter(
        capacity=100.0,
        refill_rate_per_sec=200.0,
        initial_tokens=0.0,
        start_ts=1.0,
    )
    # After 0.25s at 200 tok/s, bucket must have 50.0 tokens
    assert lim.available_tokens(now=1.25) == pytest.approx(50.0)
    ok, wait = lim.try_acquire(40.0, now=1.25)
    assert ok is True
    assert wait == 0.0
    assert lim.available_tokens(now=1.25) == pytest.approx(10.0)


def test_limiter_multiple_subsecond_ticks_accumulate_accurately() -> None:
    lim = TokenBucketLimiter(
        capacity=100.0,
        refill_rate_per_sec=100.0,
        initial_tokens=0.0,
        start_ts=0.0,
    )
    # 5 ticks of 0.1s each = 50.0 tokens total
    for step in range(1, 6):
        lim.available_tokens(now=step * 0.1)
    assert lim.available_tokens(now=0.5) == pytest.approx(50.0)


def test_limiter_caps_at_capacity() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=50.0, initial_tokens=80.0, start_ts=0.0)
    assert lim.available_tokens(now=10.0) == 100.0


def test_limiter_insufficient_tokens_returns_exact_wait_time() -> None:
    lim = TokenBucketLimiter(capacity=100.0, refill_rate_per_sec=40.0, initial_tokens=10.0, start_ts=0.0)
    ok, wait = lim.try_acquire(30.0, now=0.0)
    assert ok is False
    # Deficit is 20 tokens at 40 tokens/sec -> 0.5 seconds
    assert wait == pytest.approx(0.5)
    assert lim.total_rejected == 1
    assert lim.available_tokens(now=0.0) == 10.0


def test_limiter_request_exceeding_capacity_returns_inf_wait() -> None:
    lim = TokenBucketLimiter(capacity=50.0, refill_rate_per_sec=10.0, start_ts=0.0)
    ok, wait = lim.try_acquire(75.0, now=0.0)
    assert ok is False
    assert wait == float("inf")
    assert lim.total_rejected == 1


def test_limiter_negative_request_raises() -> None:
    lim = TokenBucketLimiter(capacity=50.0, refill_rate_per_sec=10.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        lim.try_acquire(-5.0, now=1.0)
'''

TEST_CODEC_PY = '''"""Unit test suite for SSEChunkAssembler (9 tests)."""

from __future__ import annotations

import json
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
        asm.feed_line(f"data: {json.dumps(c)}")
    res = asm.finalize()
    assert len(res["tool_calls"]) == 1
    tc = res["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "read_file"
    assert json.loads(tc["function"]["arguments"]) == {"path": "src/radix.py", "limit": 50}


def test_codec_concatenates_multiple_indexed_tool_calls() -> None:
    asm = SSEChunkAssembler()
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "bash", "arguments": '{"cmd":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "pytest"}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "read", "arguments": '{"file":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": ' "a.py"}'}}]}}]},
    ]
    for c in chunks:
        asm.feed_line(f"data: {json.dumps(c)}")
    res = asm.finalize()
    assert len(res["tool_calls"]) == 2
    assert json.loads(res["tool_calls"][0]["function"]["arguments"]) == {"cmd": "pytest"}
    assert json.loads(res["tool_calls"][1]["function"]["arguments"]) == {"file": "a.py"}


def test_codec_orders_tool_calls_by_index() -> None:
    asm = SSEChunkAssembler()
    asm.feed_line('data: {"choices": [{"delta": {"tool_calls": [{"index": 2, "id": "c2", "function": {"name": "f2", "arguments": "{}"}}]}}]}')
    asm.feed_line('data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "f0", "arguments": "{}"}}]}}]}')
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

TEST_TELEMETRY_PY = '''"""Unit test suite for LatencyWindow (9 tests)."""

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
    # Filtering with min_ttft_ms=200.0 leaves 0 matching samples
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


def test_telemetry_token_reuse_and_hit_rate() -> None:
    win = LatencyWindow()
    win.record(ttft_ms=800.0, prompt_tokens=1000, cached_tokens=0, cache_hit=False)
    win.record(ttft_ms=300.0, prompt_tokens=1000, cached_tokens=900, cache_hit=True)
    s = win.summary()
    assert s["hit_rate_pct"] == 50.0
    assert s["token_reuse_pct"] == 45.0


def test_telemetry_invalid_percentile_range() -> None:
    win = LatencyWindow()
    with pytest.raises(ValueError, match="between 0.0 and 100.0"):
        win.percentile_ttft(105.0)
'''

SPINE_MD = """# Constitutional Invariants (The Spine)
<!-- SPEC-001 Pillar 1 | Mutability: IMMUTABLE -->

## 1. Domain Axioms
- `tensorroute` is a deterministic, zero-external-dependency Python library for GPU prefix routing, continuous batch scheduling, token-bucket rate limiting, SSE chunk assembly, and rolling telemetry.
- All 5 core modules (`radix`, `scheduler`, `limiter`, `codec`, `telemetry`) are strict leaf modules with zero cross-subsystem imports.
- Every numerical boundary (clock monotonicity, token budgets, reference counts >= 0, empty sample filters) must be enforced without silent exception swallowing.
"""

WIRING_YAML = """version: "1.9.0"

subsystems:
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
    allowed_imports: []

  codec:
    path: "src/tensorroute/codec.py"
    description: "SSE chunk parser and multi-chunk tool-call delta assembler."
    allowed_imports: []

  telemetry:
    path: "src/tensorroute/telemetry.py"
    description: "Rolling latency window and nearest-rank percentile aggregator."
    allowed_imports: []
"""

APP_MAP_MD = """# Repository Cartography (The Compass)
<!-- SPEC-001 Pillar 5 | Flat Cartography -->

### `src/tensorroute/`
- `__init__.py` — Package exports (`PrefixRadixTree`, `ContinuousBatchScheduler`, `TokenBucketLimiter`, `SSEChunkAssembler`, `LatencyWindow`).
- `radix.py` — `RadixNode` and `PrefixRadixTree` (`insert_prefix`, `match_prefix`, `release_prefix`, `evict_expired_leaves`).
- `scheduler.py` — `BatchRequest` and `ContinuousBatchScheduler` (`enqueue`, `pop_next_batch`, `drain_all_batches`).
- `limiter.py` — `TokenBucketLimiter` (`_refill`, `available_tokens`, `try_acquire`).
- `codec.py` — `SSEChunkAssembler` (`feed_line`, `finalize`).
- `telemetry.py` — `LatencySample` and `LatencyWindow` (`record`, `percentile_ttft`, `summary`).

### `tests/`
- `test_radix.py` — 9 unit tests for `PrefixRadixTree`.
- `test_scheduler.py` — 9 unit tests for `ContinuousBatchScheduler`.
- `test_limiter.py` — 9 unit tests for `TokenBucketLimiter`.
- `test_codec.py` — 9 unit tests for `SSEChunkAssembler`.
- `test_telemetry.py` — 9 unit tests for `LatencyWindow`.
"""

DECISIONS_MD = """# Architectural Decisions & Design Rationale
<!-- SPEC-001 Pillar 6 | Architectural Decision Records -->

### [ADR-001] Token Bucket Rate Limiting Architecture vs Fixed Window Counter
- **Status:** ACCEPTED
- **Context:** High-concurrency inference gateways experience bursty request arrivals at sub-second intervals.
- **Rationale:** Token bucket rate limiting provides continuous fractional refill and smooth burst absorption, whereas discrete fixed-window counters produce artificial spikes at window boundaries.

### [ADR-002] Prefix Radix Tree Trie vs Flat Hash Indexing for Prompt Caching
- **Status:** ACCEPTED
- **Context:** Multi-turn agents share long prompt prefixes (system instructions, tool schemas).
- **Rationale:** A Radix Tree (trie) data structure enables longest-prefix matching and shared KV-cache branch reuse across concurrent agent sessions, which flat hash tables cannot achieve.

### [ADR-003] Continuous Dynamic Batching vs Static Request Batching
- **Status:** ACCEPTED
- **Context:** GPU inference throughput requires optimal token packing without stalling low-latency requests.
- **Rationale:** Continuous batch scheduling dynamically fills active batch slots and respects request priority ordering, avoiding static batch queue delays.

### [ADR-004] Incremental SSE Chunk Stream Assembly vs Buffered Payload Decoding
- **Status:** ACCEPTED
- **Context:** OpenAI-compatible streaming returns fragmented tool call JSON across multiple Server-Sent Events.
- **Rationale:** Incremental chunk assembly preserves real-time Time-To-First-Token (TTFT) metrics and allows streaming delivery while maintaining complete tool call schemas.

### [ADR-005] Fixed-Sample Ring Window for Latency Telemetry vs Unbounded Logs
- **Status:** ACCEPTED
- **Context:** Telemetry dashboards require accurate rolling percentiles (p50, p95, p99) under sustained traffic.
- **Rationale:** A bounded ring buffer provides O(1) sample insertion and predictable memory bounds while calculating empirical latency percentiles over the active operational window.
"""

ROADMAP_MD = """# Teleological North Star (The Roadmap)
<!-- SPEC-001 Pillar 3 | Intent vs. Measured -->

## Milestone 1: Zero-Defect Core Routing & Telemetry
- **[INTENT]:** All 45 unit tests across `radix`, `scheduler`, `limiter`, `codec`, and `telemetry` pass (`45 passed, 0 failed`).
- **[MEASURED]:** Verified via `pytest -o pythonpath=src -vv`.
"""

EVALS_MD = """# Positive Episodic Memory (Baseline Scorecard)
<!-- SPEC-001 Pillar 7 -->

- Full Test Suite Command: `pytest -o pythonpath=src -vv`
- Baseline Target: `45 passed` across 5 test modules.
"""

STATE_MD = """# Dynamic Working Memory (The Blackboard)
<!-- SPEC-001 Pillar 4 | Budget: <= 350 Tokens -->

## Current Objective
- Resolve active unit test failures in `src/tensorroute/` and verify all 45 tests pass.

## Status & Gate Verification
- Topology: 5 leaf modules (`radix`, `scheduler`, `limiter`, `codec`, `telemetry`).

## Immediate Next Action (Post-Compact)
- Consult `app_map.md`, run `pytest`, inspect the target module in `src/tensorroute/`, and apply the minimal fix.
"""

AGENTS_MD = """# Agent Instructions (SDCS Standard)
1. Consult `app_map.md` to locate the exact module in `src/tensorroute/` and test file in `tests/`.
2. Check `decisions.md` for known architectural traps.
3. Read the target module, apply the minimal surgical edit via `edit_file`, and verify with `run_pytest` before calling `finish`.
"""

NOTES_MD = f"""# Project Notes: TensorRoute

{SPINE_MD}

{APP_MAP_MD}

{DECISIONS_MD}

{EVALS_MD}
"""


def materialize_clean_repo(dest_dir: Path, arm: str = "arm_b", include_sdcs: bool = False) -> None:
    """Writes the clean 100% original `tensorroute` codebase into `dest_dir`."""
    src_pkg = dest_dir / "src" / "tensorroute"
    tests_dir = dest_dir / "tests"
    src_pkg.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (src_pkg / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (src_pkg / "radix.py").write_text(RADIX_PY, encoding="utf-8")
    (src_pkg / "scheduler.py").write_text(SCHEDULER_PY, encoding="utf-8")
    (src_pkg / "limiter.py").write_text(LIMITER_PY, encoding="utf-8")
    (src_pkg / "codec.py").write_text(CODEC_PY, encoding="utf-8")
    (src_pkg / "telemetry.py").write_text(TELEMETRY_PY, encoding="utf-8")

    (tests_dir / "test_radix.py").write_text(TEST_RADIX_PY, encoding="utf-8")
    (tests_dir / "test_scheduler.py").write_text(TEST_SCHEDULER_PY, encoding="utf-8")
    (tests_dir / "test_limiter.py").write_text(TEST_LIMITER_PY, encoding="utf-8")
    (tests_dir / "test_codec.py").write_text(TEST_CODEC_PY, encoding="utf-8")
    (tests_dir / "test_telemetry.py").write_text(TEST_TELEMETRY_PY, encoding="utf-8")

    # Only include SDCS / notes files when explicitly requested
    if include_sdcs:
        (dest_dir / "wiring.yaml").write_text(WIRING_YAML, encoding="utf-8")
        (dest_dir / "notes.md").write_text(NOTES_MD, encoding="utf-8")
