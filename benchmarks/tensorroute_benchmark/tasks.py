"""100% Original Task Definitions (`OB-01` through `OB-15`) for `TensorRoute-Bench` v2.

Provides 15 deterministic, non-overlapping systems software bugs across the 5 modules of
`src/tensorroute/` (3 bugs per subsystem: `radix`, `scheduler`, `limiter`, `codec`, `telemetry`):
- Wave 1 (`OB-01` to `OB-05`): Primary 5-module bug wave (also used in 5-bug continuous `--mode gauntlet`).
- Wave 2 (`OB-06` to `OB-10`): Secondary 5-module bug wave.
- Wave 3 (`OB-11` to `OB-15`): Tertiary 5-module bug wave.

Because all 3 bugs within each module target disjoint methods/blocks, they can be injected
individually (`--mode suite`, N=15 tasks) or simultaneously in multi-bug waves (`--mode gauntlet`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OriginalBenchmarkTask:
    task_id: str
    subsystem: str
    target_file: str
    test_file: str
    title: str
    problem_statement: str
    symptom_prompt: str
    clean_snippet: str
    buggy_snippet: str
    full_test_command: str


BENCHMARK_TASKS: list[OriginalBenchmarkTask] = [
    # =========================================================================
    # WAVE 1: OB-01 through OB-05 (1 per subsystem)
    # =========================================================================
    OriginalBenchmarkTask(
        task_id="OB-01-radix-evict-refcount",
        subsystem="radix",
        target_file="src/tensorroute/radix.py",
        test_file="tests/test_radix.py",
        title="Fix double ancestor reference decrement during expired leaf eviction",
        problem_statement=(
            "When `PrefixRadixTree.evict_expired_leaves(now)` evicts an expired leaf node, "
            "the ancestor traversal loop subtracts `2 * leaf.ref_count` instead of `leaf.ref_count`. "
            "This causes shared parent prefixes that still have active child branches to underflow "
            "their `ref_count`."
        ),
        symptom_prompt=(
            "Production KV-cache routing reports premature eviction of shared parent prefixes "
            "after expired leaf cleanup. Run `run_pytest` to locate the failing radix tree unit "
            "tests, inspect the implementation, and fix the reference count accounting bug."
        ),
        clean_snippet=(
            "                    anc = parent\n"
            "                    while anc is not None and anc is not self.root:\n"
            "                        anc.ref_count = max(0, anc.ref_count - leaf.ref_count)\n"
            "                        anc = anc.parent"
        ),
        buggy_snippet=(
            "                    anc = parent\n"
            "                    while anc is not None and anc is not self.root:\n"
            "                        anc.ref_count = max(0, anc.ref_count - (2 * leaf.ref_count))\n"
            "                        anc = anc.parent"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-02-scheduler-partial-drain",
        subsystem="scheduler",
        target_file="src/tensorroute/scheduler.py",
        test_file="tests/test_scheduler.py",
        title="Fix ContinuousBatchScheduler.drain_all_batches dropping partial tail batches",
        problem_statement=(
            "In `ContinuousBatchScheduler.drain_all_batches(flush_partial=True)`, the partial-batch "
            "re-queue guard uses `if not flush_partial or len(batch) < self.max_batch_size:` instead of "
            "`if not flush_partial and len(batch) < self.max_batch_size:`, causing partial tail "
            "batches to be re-queued and dropped even when `flush_partial=True`."
        ),
        symptom_prompt=(
            "Batch queue draining silently withholds tail requests whenever the final batch is smaller "
            "than `max_batch_size` with `flush_partial=True`. Run `run_pytest`, inspect the batch "
            "scheduler, and fix `drain_all_batches` so partial tail batches are emitted when requested."
        ),
        clean_snippet=(
            "            if not flush_partial and len(batch) < self.max_batch_size:\n"
            "                for req in batch:\n"
            "                    self._queue.append(req)\n"
            "                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))\n"
            "                break\n"
            "            batches.append(batch)"
        ),
        buggy_snippet=(
            "            if not flush_partial or len(batch) < self.max_batch_size:\n"
            "                for req in batch:\n"
            "                    self._queue.append(req)\n"
            "                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))\n"
            "                break\n"
            "            batches.append(batch)"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-03-limiter-fractional-refill",
        subsystem="limiter",
        target_file="src/tensorroute/limiter.py",
        test_file="tests/test_limiter.py",
        title="Fix integer second truncation starving sub-second token bucket refills",
        problem_statement=(
            "`TokenBucketLimiter._refill(now)` truncates elapsed time using "
            "`float(int(now - self.last_refill_ts))` while advancing `self.last_refill_ts = now`, "
            "zeroing out all sub-second token refills (e.g. 0.1s or 0.25s intervals)."
        ),
        symptom_prompt=(
            "High-frequency callers polling the token-bucket rate limiter at sub-second intervals "
            "(100ms-250ms) receive zero token refills. Run `run_pytest`, inspect the limiter "
            "refill arithmetic, and restore exact fractional elapsed time computation."
        ),
        clean_snippet=(
            "        elapsed = now - self.last_refill_ts\n"
            "        if elapsed > 0.0:\n"
            "            added = elapsed * self.refill_rate_per_sec\n"
            "            self.tokens = min(self.capacity, self.tokens + added)\n"
            "            self.last_refill_ts = now"
        ),
        buggy_snippet=(
            "        elapsed = float(int(now - self.last_refill_ts))\n"
            "        if now > self.last_refill_ts:\n"
            "            added = elapsed * self.refill_rate_per_sec\n"
            "            self.tokens = min(self.capacity, self.tokens + added)\n"
            "            self.last_refill_ts = now"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-04-codec-toolcall-concat",
        subsystem="codec",
        target_file="src/tensorroute/codec.py",
        test_file="tests/test_codec.py",
        title="Fix SSEChunkAssembler overwriting streamed tool_call arguments across chunks",
        problem_statement=(
            "When `SSEChunkAssembler.feed_line` processes multi-chunk streamed tool calls, "
            "`entry['function']['arguments']` is overwritten (`=`) instead of concatenated (`+=`) "
            "on subsequent SSE delta chunks."
        ),
        symptom_prompt=(
            "Streamed tool calls arriving across multiple SSE chunks fail JSON parsing when assembling "
            "delta chunks. Run `run_pytest` to reproduce the failure, inspect the SSE codec assembler, "
            "and fix the tool call argument assembly bug."
        ),
        clean_snippet=(
            '            if fn.get("name"):\n'
            '                entry["function"]["name"] += str(fn["name"])\n'
            '            if fn.get("arguments"):\n'
            '                entry["function"]["arguments"] += str(fn["arguments"])'
        ),
        buggy_snippet=(
            '            if fn.get("name"):\n'
            '                entry["function"]["name"] += str(fn["name"])\n'
            '            if fn.get("arguments"):\n'
            '                entry["function"]["arguments"] = str(fn["arguments"])'
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-05-telemetry-percentile-bounds",
        subsystem="telemetry",
        target_file="src/tensorroute/telemetry.py",
        test_file="tests/test_telemetry.py",
        title="Fix nearest-rank 1-based percentile index overflow in LatencyWindow.percentile_ttft",
        problem_statement=(
            "In `LatencyWindow.percentile_ttft(q)`, the nearest-rank index is computed as "
            "`idx = math.ceil((q / 100.0) * n)` without subtracting 1 (`math.ceil((q / 100.0) * n) - 1`), "
            "raising `IndexError: list index out of range` whenever `q == 100.0` or `n == 1` and "
            "shifting all percentiles up by one rank."
        ),
        symptom_prompt=(
            "`LatencyWindow.percentile_ttft` and `summary()` crash with `IndexError: list index out of range` "
            "on single-sample windows and boundary percentiles. Run `run_pytest`, inspect `percentile_ttft` "
            "in `src/tensorroute/telemetry.py`, and fix the nearest-rank index bounds calculation."
        ),
        clean_snippet=(
            "        if q == 0.0:\n"
            "            return filtered[0]\n"
            "        idx = max(0, min(n - 1, math.ceil((q / 100.0) * n) - 1))\n"
            "        return round(filtered[idx], 3)"
        ),
        buggy_snippet=(
            "        if q == 0.0:\n"
            "            return filtered[0]\n"
            "        idx = math.ceil((q / 100.0) * n)\n"
            "        return round(filtered[idx], 3)"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    # =========================================================================
    # WAVE 2: OB-06 through OB-10 (1 per subsystem, disjoint from Wave 1 & 3)
    # =========================================================================
    OriginalBenchmarkTask(
        task_id="OB-06-radix-match-chunk-count",
        subsystem="radix",
        target_file="src/tensorroute/radix.py",
        test_file="tests/test_radix.py",
        title="Fix PrefixRadixTree.match_prefix counting matched chunks instead of matched tokens",
        problem_statement=(
            "In `PrefixRadixTree.match_prefix(tokens, now)`, each matched node increments "
            "`matched_tokens += 1` instead of `matched_tokens += self.chunk_size`, under-reporting "
            "the number of matched prefix tokens by a factor of `chunk_size`."
        ),
        symptom_prompt=(
            "`PrefixRadixTree.match_prefix` returns the matched worker ID but under-reports "
            "the matched token count whenever `chunk_size > 1`. Run `run_pytest` to reproduce "
            "the failure and fix the token count accumulation in `match_prefix`."
        ),
        clean_snippet=(
            "            nxt.last_access_ts = now\n"
            "            matched_tokens += self.chunk_size\n"
            "            matched_worker = nxt.worker_id"
        ),
        buggy_snippet=(
            "            nxt.last_access_ts = now\n"
            "            matched_tokens += 1\n"
            "            matched_worker = nxt.worker_id"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-07-scheduler-priority-direction",
        subsystem="scheduler",
        target_file="src/tensorroute/scheduler.py",
        test_file="tests/test_scheduler.py",
        title="Fix ContinuousBatchScheduler.enqueue sorting priorities ascending instead of descending",
        problem_statement=(
            "In `ContinuousBatchScheduler.enqueue(req)`, `self._queue.sort` sorts by "
            "`(r.priority, self._arrival_order.get(r.request_id, 0))` instead of `(-r.priority, ...)`, "
            "scheduling lowest-priority requests ahead of highest-priority requests."
        ),
        symptom_prompt=(
            "High-priority inference requests are being scheduled behind low-priority requests "
            "in `ContinuousBatchScheduler`. Run `run_pytest`, inspect queue sorting in the "
            "scheduler, and restore descending priority order with FIFO tie-breaking."
        ),
        clean_snippet=(
            "        self._queue.append(req)\n"
            "        self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))"
        ),
        buggy_snippet=(
            "        self._queue.append(req)\n"
            "        self._queue.sort(key=lambda r: (r.priority, self._arrival_order.get(r.request_id, 0)))"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-08-limiter-deficit-wait-calc",
        subsystem="limiter",
        target_file="src/tensorroute/limiter.py",
        test_file="tests/test_limiter.py",
        title="Fix TokenBucketLimiter.try_acquire ignoring current token balance in wait-time calculation",
        problem_statement=(
            "When `TokenBucketLimiter.try_acquire(requested_tokens, now)` rejects a request due to "
            "insufficient tokens, it computes `deficit = requested_tokens` instead of "
            "`deficit = requested_tokens - self.tokens`, overestimating the required wait time."
        ),
        symptom_prompt=(
            "Rejected rate-limit requests receive an inflated `wait_seconds` estimate that ignores "
            "tokens already available in the bucket. Run `run_pytest`, inspect `try_acquire`, "
            "and fix the token deficit calculation."
        ),
        clean_snippet=(
            "        deficit = requested_tokens - self.tokens\n"
            "        wait_sec = deficit / self.refill_rate_per_sec\n"
            "        return False, round(wait_sec, 6)"
        ),
        buggy_snippet=(
            "        deficit = requested_tokens\n"
            "        wait_sec = deficit / self.refill_rate_per_sec\n"
            "        return False, round(wait_sec, 6)"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-09-codec-toolcall-index-sort",
        subsystem="codec",
        target_file="src/tensorroute/codec.py",
        test_file="tests/test_codec.py",
        title="Fix SSEChunkAssembler.finalize reversing tool_calls index order",
        problem_statement=(
            "In `SSEChunkAssembler.finalize()`, `ordered_tools` sorts `self.tool_calls_by_index.keys()` "
            "with `reverse=True`, emitting parallel tool calls in reverse index order."
        ),
        symptom_prompt=(
            "Multi-tool SSE responses return assembled `tool_calls` in reversed index order "
            "upon finalization. Run `run_pytest`, inspect `SSEChunkAssembler.finalize`, and "
            "restore ascending tool call index ordering."
        ),
        clean_snippet=(
            "        ordered_tools = [self.tool_calls_by_index[k] for k in sorted(self.tool_calls_by_index.keys())]"
        ),
        buggy_snippet=(
            "        ordered_tools = [self.tool_calls_by_index[k] for k in sorted(self.tool_calls_by_index.keys(), reverse=True)]"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-10-telemetry-ring-eviction",
        subsystem="telemetry",
        target_file="src/tensorroute/telemetry.py",
        test_file="tests/test_telemetry.py",
        title="Fix LatencyWindow.record retaining oldest samples instead of newest on ring overflow",
        problem_statement=(
            "When `LatencyWindow.record` exceeds `self.max_samples`, it slices "
            "`self._samples[:self.max_samples]` (retaining the oldest samples and discarding the "
            "newly recorded sample) instead of `self._samples[-self.max_samples:]`."
        ),
        symptom_prompt=(
            "Once `LatencyWindow` reaches `max_samples`, subsequent latency measurements are not "
            "reflected in window queries. Run `run_pytest`, inspect sample retention in `LatencyWindow.record`, "
            "and ensure the ring buffer retains the newest samples on overflow."
        ),
        clean_snippet=(
            "        if len(self._samples) > self.max_samples:\n"
            "            self._samples = self._samples[-self.max_samples :]"
        ),
        buggy_snippet=(
            "        if len(self._samples) > self.max_samples:\n"
            "            self._samples = self._samples[: self.max_samples]"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    # =========================================================================
    # WAVE 3: OB-11 through OB-15 (1 per subsystem, disjoint from Wave 1 & 2)
    # =========================================================================
    OriginalBenchmarkTask(
        task_id="OB-11-radix-release-underflow",
        subsystem="radix",
        target_file="src/tensorroute/radix.py",
        test_file="tests/test_radix.py",
        title="Fix PrefixRadixTree.release_prefix allowing negative reference count underflow",
        problem_statement=(
            "In `PrefixRadixTree.release_prefix(tokens)`, decrementing `nxt.ref_count` uses "
            "`nxt.ref_count = nxt.ref_count - 1` without clamping at `0` via `max(0, ...)`, "
            "causing duplicate releases to drive `ref_count` negative."
        ),
        symptom_prompt=(
            "Releasing an already-unreferenced prefix in `PrefixRadixTree.release_prefix` drives "
            "node `ref_count` below zero. Run `run_pytest`, inspect `release_prefix`, and floor "
            "reference counts at 0."
        ),
        clean_snippet=(
            "            nxt.ref_count = max(0, nxt.ref_count - 1)\n"
            "            released_nodes += 1\n"
            "            curr = nxt"
        ),
        buggy_snippet=(
            "            nxt.ref_count = nxt.ref_count - 1\n"
            "            released_nodes += 1\n"
            "            curr = nxt"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-12-scheduler-token-budget-check",
        subsystem="scheduler",
        target_file="src/tensorroute/scheduler.py",
        test_file="tests/test_scheduler.py",
        title="Fix ContinuousBatchScheduler.pop_next_batch checking individual request tokens instead of cumulative batch tokens",
        problem_statement=(
            "In `ContinuousBatchScheduler.pop_next_batch()`, the packing condition checks "
            "`req.prompt_tokens <= self.max_batch_tokens` instead of "
            "`(running_tokens + req.prompt_tokens) <= self.max_batch_tokens`, allowing batches "
            "to exceed `self.max_batch_tokens`."
        ),
        symptom_prompt=(
            "`ContinuousBatchScheduler.pop_next_batch` packs multiple requests whose combined "
            "prompt tokens exceed `max_batch_tokens`. Run `run_pytest`, inspect `pop_next_batch`, "
            "and fix the cumulative token budget check."
        ),
        clean_snippet=(
            "            if (\n"
            "                len(batch) < self.max_batch_size\n"
            "                and (running_tokens + req.prompt_tokens) <= self.max_batch_tokens\n"
            "            ):\n"
            "                batch.append(req)\n"
            "                running_tokens += req.prompt_tokens"
        ),
        buggy_snippet=(
            "            if (\n"
            "                len(batch) < self.max_batch_size\n"
            "                and req.prompt_tokens <= self.max_batch_tokens\n"
            "            ):\n"
            "                batch.append(req)\n"
            "                running_tokens += req.prompt_tokens"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-13-limiter-acquire-deduction",
        subsystem="limiter",
        target_file="src/tensorroute/limiter.py",
        test_file="tests/test_limiter.py",
        title="Fix TokenBucketLimiter.try_acquire deducting from capacity instead of current token balance",
        problem_statement=(
            "In `TokenBucketLimiter.try_acquire(requested_tokens, now)`, when a request is allowed, "
            "it sets `self.tokens = self.capacity - requested_tokens` instead of deducting from "
            "the current balance (`self.tokens -= requested_tokens`)."
        ),
        symptom_prompt=(
            "Acquiring tokens from a partially filled `TokenBucketLimiter` results in unexpected "
            "remaining token balances on subsequent calls. Run `run_pytest`, inspect `try_acquire`, "
            "and fix the token deduction accounting."
        ),
        clean_snippet=(
            "        if requested_tokens <= self.capacity and self.tokens >= requested_tokens:\n"
            "            self.tokens -= requested_tokens\n"
            "            self.total_consumed += requested_tokens\n"
            "            return True, 0.0"
        ),
        buggy_snippet=(
            "        if requested_tokens <= self.capacity and self.tokens >= requested_tokens:\n"
            "            self.tokens = self.capacity - requested_tokens\n"
            "            self.total_consumed += requested_tokens\n"
            "            return True, 0.0"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-14-codec-usage-cached-tokens",
        subsystem="codec",
        target_file="src/tensorroute/codec.py",
        test_file="tests/test_codec.py",
        title="Fix SSEChunkAssembler dropping top-level prompt_cache_hit_tokens and prior cached_tokens",
        problem_statement=(
            "In `SSEChunkAssembler.feed_line`, parsing `usage` only reads "
            "`details.get('cached_tokens') or 0`, ignoring top-level `usage.get('prompt_cache_hit_tokens')` "
            "and overwriting `self.cached_tokens` when a trailing chunk omits cache hit details."
        ),
        symptom_prompt=(
            "`SSEChunkAssembler` reports `cached_tokens == 0` when providers emit top-level "
            "`prompt_cache_hit_tokens` in SSE usage chunks. Run `run_pytest`, inspect `feed_line` "
            "in `codec.py`, and fix cached token extraction."
        ),
        clean_snippet=(
            '            details = usage.get("prompt_tokens_details") or {}\n'
            '            self.cached_tokens = int(details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or self.cached_tokens)'
        ),
        buggy_snippet=(
            '            details = usage.get("prompt_tokens_details") or {}\n'
            '            self.cached_tokens = int(details.get("cached_tokens") or 0)'
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
    OriginalBenchmarkTask(
        task_id="OB-15-telemetry-token-reuse-ratio",
        subsystem="telemetry",
        target_file="src/tensorroute/telemetry.py",
        test_file="tests/test_telemetry.py",
        title="Fix LatencyWindow.summary dividing total_cached by sample count instead of total_prompt",
        problem_statement=(
            "In `LatencyWindow.summary()`, `token_reuse_pct` is computed as "
            "`round((total_cached / n) * 100.0, 2)` instead of "
            "`round((total_cached / total_prompt) * 100.0, 2)`."
        ),
        symptom_prompt=(
            "`LatencyWindow.summary()` computes an invalid `token_reuse_pct` (sometimes exceeding 100%) "
            "in telemetry reports. Run `run_pytest`, inspect `summary()` in `src/tensorroute/telemetry.py`, "
            "and fix the token reuse ratio calculation."
        ),
        clean_snippet=(
            '            "token_reuse_pct": (\n'
            "                round((total_cached / total_prompt) * 100.0, 2)\n"
            "                if total_prompt > 0\n"
            "                else 0.0\n"
            "            ),"
        ),
        buggy_snippet=(
            '            "token_reuse_pct": (\n'
            "                round((total_cached / n) * 100.0, 2)\n"
            "                if total_prompt > 0\n"
            "                else 0.0\n"
            "            ),"
        ),
        full_test_command="pytest -o pythonpath=src -vv",
    ),
]


def inject_bug(workspace_dir: Path, task: OriginalBenchmarkTask) -> None:
    """Replaces `task.clean_snippet` with `task.buggy_snippet` in `workspace_dir / task.target_file`."""
    target_path = workspace_dir / task.target_file
    content = target_path.read_text(encoding="utf-8")
    if task.clean_snippet not in content:
        raise RuntimeError(f"Clean snippet not found in {target_path} for task {task.task_id}")
    mutated = content.replace(task.clean_snippet, task.buggy_snippet, 1)
    target_path.write_text(mutated, encoding="utf-8")


def apply_reference_fix(workspace_dir: Path, task: OriginalBenchmarkTask) -> None:
    """Restores `task.clean_snippet` in `workspace_dir / task.target_file`."""
    target_path = workspace_dir / task.target_file
    content = target_path.read_text(encoding="utf-8")
    if task.buggy_snippet not in content:
        raise RuntimeError(f"Buggy snippet not found in {target_path} for task {task.task_id}")
    restored = content.replace(task.buggy_snippet, task.clean_snippet, 1)
    target_path.write_text(restored, encoding="utf-8")


def inject_gauntlet_bugs(
    workspace_dir: Path,
    tasks: list[OriginalBenchmarkTask] | None = None,
) -> list[OriginalBenchmarkTask]:
    """Injects multiple disjoint bugs simultaneously into `workspace_dir` for Gauntlet mode.

    Defaults to Wave 1 (`OB-01` through `OB-05`, 1 bug in each of the 5 core modules),
    causing 10 unit tests across all 5 test files to fail simultaneously (`35 passed, 10 failed`).
    """
    selected = tasks if tasks is not None else BENCHMARK_TASKS[:5]
    for task in selected:
        inject_bug(workspace_dir, task)
    return selected
