"""Deterministic task definitions for TensorRoute v2 coupled gauntlet.

Defines 5 architectural coupled traps (T1 through T5) with:
- Strict symptom-only prompts (no file or function names, no module cross-references)
- Fail-to-pass (F2P) test targets
- Architectural guard tests broken by naive fixes
- Reference fixes (correct) and naive patches (wrong)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Mutation:
    target_file: str
    clean_snippet: str
    buggy_snippet: str


@dataclass(frozen=True)
class CoupledBenchmarkTask:
    task_id: str
    title: str
    symptom_prompt: str
    f2p_tests: list[str]
    guard_tests: list[str]
    mutations: list[Mutation]
    naive_mutations: list[Mutation] = field(default_factory=list)


COUPLED_TASKS: list[CoupledBenchmarkTask] = [
    CoupledBenchmarkTask(
        task_id="T1-shared-helper",
        title="Telemetry percentage summary vs dimensionless ratio contract",
        symptom_prompt="The dashboard shows a cache hit rate of 0.5 when half the requests hit; it should read 50.",
        f2p_tests=[
            "tests/test_telemetry.py::test_telemetry_hit_rate_is_percentage",
            "tests/test_telemetry.py::test_telemetry_token_reuse_is_percentage",
        ],
        guard_tests=[
            "tests/test_limiter.py::test_utilization_is_fraction",
            "tests/test_units.py::test_safe_ratio_valid_fractions",
        ],
        mutations=[
            Mutation(
                target_file="src/tensorroute/telemetry.py",
                clean_snippet=(
                    '            "hit_rate_pct": round(safe_ratio(hits, n) * 100.0, 2),\n'
                    '            "token_reuse_pct": round(safe_ratio(total_cached, total_prompt) * 100.0, 2),'
                ),
                buggy_snippet=(
                    '            "hit_rate_pct": round(safe_ratio(hits, n), 2),\n'
                    '            "token_reuse_pct": round(safe_ratio(total_cached, total_prompt), 2),'
                ),
            ),
        ],
        naive_mutations=[
            Mutation(
                target_file="src/tensorroute/units.py",
                clean_snippet=(
                    "def safe_ratio(part: float, whole: float) -> float:\n"
                    '    """Computes part / whole as a dimensionless fraction in [0.0, 1.0].\n'
                    "\n"
                    "    Returns a fraction in [0, 1]. Returns 0.0 when whole <= 0.0 or part <= 0.0.\n"
                    '    """\n'
                    "    if whole <= 0.0 or part <= 0.0:\n"
                    "        return 0.0\n"
                    "    return min(1.0, float(part) / float(whole))"
                ),
                buggy_snippet=(
                    "def safe_ratio(part: float, whole: float) -> float:\n"
                    '    """Computes part / whole as a percentage in [0.0, 100.0]."""\n'
                    "    if whole <= 0.0 or part <= 0.0:\n"
                    "        return 0.0\n"
                    "    return min(100.0, (float(part) / float(whole)) * 100.0)"
                ),
            ),
        ],
    ),
    CoupledBenchmarkTask(
        task_id="T2-units-boundary",
        title="Latency window millisecond unit conversion at pipeline completion boundary",
        symptom_prompt="Recorded p95 time-to-first-token is about 0.2 ms; real requests take about 200 ms.",
        f2p_tests=[
            "tests/test_pipeline.py::test_pipeline_records_latency_in_ms",
        ],
        guard_tests=[
            "tests/test_telemetry.py::test_telemetry_ten_sample_nearest_rank_percentiles",
            "tests/test_telemetry.py::test_telemetry_single_sample_percentiles_do_not_overflow",
        ],
        mutations=[
            Mutation(
                target_file="src/tensorroute/pipeline.py",
                clean_snippet=(
                    "        self.window.record(\n"
                    "            ttft_ms=to_ms(ttft_seconds),\n"
                    "            prompt_tokens=prompt_tokens,\n"
                    "            cached_tokens=cached_tokens,\n"
                    "            cache_hit=cached_tokens > 0,\n"
                    "        )"
                ),
                buggy_snippet=(
                    "        self.window.record(\n"
                    "            ttft_ms=ttft_seconds,\n"
                    "            prompt_tokens=prompt_tokens,\n"
                    "            cached_tokens=cached_tokens,\n"
                    "            cache_hit=cached_tokens > 0,\n"
                    "        )"
                ),
            ),
        ],
        naive_mutations=[
            Mutation(
                target_file="src/tensorroute/telemetry.py",
                clean_snippet="                ttft_ms=float(ttft_ms),",
                buggy_snippet="                ttft_ms=float(ttft_ms) * 1000.0,",
            ),
        ],
    ),
    CoupledBenchmarkTask(
        task_id="T3-masked-bug",
        title="Partial batch tail emission and sub-second rate limiter refill",
        symptom_prompt="In each admission wave, the last few queued requests are never admitted.",
        f2p_tests=[
            "tests/test_scheduler.py::test_scheduler_flushes_partial_tail",
            "tests/test_pipeline.py::test_pipeline_admits_all_requests_including_tail",
        ],
        guard_tests=[
            "tests/test_scheduler.py::test_scheduler_drain_all_exact_multiple",
            "tests/test_scheduler.py::test_scheduler_drain_all_retains_partial_remainder_when_false",
        ],
        mutations=[
            Mutation(
                target_file="src/tensorroute/scheduler.py",
                clean_snippet=(
                    "            if not flush_partial and len(batch) < self.max_batch_size:\n"
                    "                for req in batch:\n"
                    "                    self._queue.append(req)\n"
                    "                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))\n"
                    "                break\n"
                    "            batches.append(batch)"
                ),
                buggy_snippet=(
                    "            if len(batch) < self.max_batch_size:\n"
                    "                for req in batch:\n"
                    "                    self._queue.append(req)\n"
                    "                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))\n"
                    "                break\n"
                    "            batches.append(batch)"
                ),
            ),
            Mutation(
                target_file="src/tensorroute/limiter.py",
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
            ),
        ],
        naive_mutations=[
            # Scheduler-only fix: restores scheduler but leaves limiter refill truncated
            Mutation(
                target_file="src/tensorroute/scheduler.py",
                clean_snippet=(
                    "            if len(batch) < self.max_batch_size:\n"
                    "                for req in batch:\n"
                    "                    self._queue.append(req)\n"
                    "                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))\n"
                    "                break\n"
                    "            batches.append(batch)"
                ),
                buggy_snippet=(
                    "            if not flush_partial and len(batch) < self.max_batch_size:\n"
                    "                for req in batch:\n"
                    "                    self._queue.append(req)\n"
                    "                self._queue.sort(key=lambda r: (-r.priority, self._arrival_order.get(r.request_id, 0)))\n"
                    "                break\n"
                    "            batches.append(batch)"
                ),
            ),
        ],
    ),
    CoupledBenchmarkTask(
        task_id="T4-changed-default",
        title="Empty initial rate limiter default with pipeline factory configuration",
        symptom_prompt="A newly created rate limiter starts full, allowing a burst at startup. It should start empty unless told otherwise.",
        f2p_tests=[
            "tests/test_limiter.py::test_limiter_starts_empty_by_default",
        ],
        guard_tests=[
            "tests/test_pipeline.py::test_pipeline_immediate_startup_admission",
        ],
        mutations=[
            Mutation(
                target_file="src/tensorroute/limiter.py",
                clean_snippet=(
                    "    def __init__(\n"
                    "        self,\n"
                    "        capacity: float,\n"
                    "        refill_rate_per_sec: float,\n"
                    "        initial_tokens: float | None = 0.0,\n"
                    "        start_ts: float = 0.0,\n"
                    "    ) -> None:\n"
                    "        if capacity <= 0 or refill_rate_per_sec <= 0:\n"
                    '            raise ValueError("capacity and refill_rate_per_sec must be positive")\n'
                    "        self.capacity = float(capacity)\n"
                    "        self.refill_rate_per_sec = float(refill_rate_per_sec)\n"
                    "        self.tokens = 0.0 if initial_tokens is None else min(self.capacity, float(initial_tokens))"
                ),
                buggy_snippet=(
                    "    def __init__(\n"
                    "        self,\n"
                    "        capacity: float,\n"
                    "        refill_rate_per_sec: float,\n"
                    "        initial_tokens: float | None = None,\n"
                    "        start_ts: float = 0.0,\n"
                    "    ) -> None:\n"
                    "        if capacity <= 0 or refill_rate_per_sec <= 0:\n"
                    '            raise ValueError("capacity and refill_rate_per_sec must be positive")\n'
                    "        self.capacity = float(capacity)\n"
                    "        self.refill_rate_per_sec = float(refill_rate_per_sec)\n"
                    "        self.tokens = self.capacity if initial_tokens is None else min(self.capacity, float(initial_tokens))"
                ),
            ),
            Mutation(
                target_file="src/tensorroute/pipeline.py",
                clean_snippet=(
                    "    limiter = TokenBucketLimiter(\n"
                    "        capacity=capacity,\n"
                    "        refill_rate_per_sec=refill_rate,\n"
                    "        initial_tokens=capacity,\n"
                    "        start_ts=0.0,\n"
                    "    )"
                ),
                buggy_snippet=(
                    "    limiter = TokenBucketLimiter(\n"
                    "        capacity=capacity,\n"
                    "        refill_rate_per_sec=refill_rate,\n"
                    "        start_ts=0.0,\n"
                    "    )"
                ),
            ),
        ],
        naive_mutations=[
            # Changing only limiter default to 0.0 without updating build_pipeline to pass initial_tokens=capacity
            Mutation(
                target_file="src/tensorroute/limiter.py",
                clean_snippet=(
                    "    def __init__(\n"
                    "        self,\n"
                    "        capacity: float,\n"
                    "        refill_rate_per_sec: float,\n"
                    "        initial_tokens: float | None = None,\n"
                    "        start_ts: float = 0.0,\n"
                    "    ) -> None:\n"
                    "        if capacity <= 0 or refill_rate_per_sec <= 0:\n"
                    '            raise ValueError("capacity and refill_rate_per_sec must be positive")\n'
                    "        self.capacity = float(capacity)\n"
                    "        self.refill_rate_per_sec = float(refill_rate_per_sec)\n"
                    "        self.tokens = self.capacity if initial_tokens is None else min(self.capacity, float(initial_tokens))"
                ),
                buggy_snippet=(
                    "    def __init__(\n"
                    "        self,\n"
                    "        capacity: float,\n"
                    "        refill_rate_per_sec: float,\n"
                    "        initial_tokens: float | None = 0.0,\n"
                    "        start_ts: float = 0.0,\n"
                    "    ) -> None:\n"
                    "        if capacity <= 0 or refill_rate_per_sec <= 0:\n"
                    '            raise ValueError("capacity and refill_rate_per_sec must be positive")\n'
                    "        self.capacity = float(capacity)\n"
                    "        self.refill_rate_per_sec = float(refill_rate_per_sec)\n"
                    "        self.tokens = 0.0 if initial_tokens is None else min(self.capacity, float(initial_tokens))"
                ),
            ),
        ],
    ),
    CoupledBenchmarkTask(
        task_id="T5-workaround-root-cause",
        title="Radix prefix token match counting vs chunk multiplier workaround",
        symptom_prompt="Prefix-aware routing sends requests to cold workers even when a warm prefix exists.",
        f2p_tests=[
            "tests/test_radix.py::test_radix_match_counts_tokens",
            "tests/test_pipeline.py::test_pipeline_routes_to_warm_worker",
        ],
        guard_tests=[
            "tests/test_pipeline.py::test_pipeline_does_not_route_below_threshold",
        ],
        mutations=[
            Mutation(
                target_file="src/tensorroute/radix.py",
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
            ),
        ],
        naive_mutations=[
            # Workaround in Pipeline.route (multiply matched_tokens by chunk_size) PLUS radix fix
            Mutation(
                target_file="src/tensorroute/pipeline.py",
                clean_snippet=(
                    "    def route(self, tokens: list[int], now: float, min_cached_tokens: int = 8) -> str | None:\n"
                    '        """Returns sticky worker ID if matched tokens >= min_cached_tokens, else None."""\n'
                    "        worker, matched_tokens = self.tree.match_prefix(tokens, now=now)\n"
                    "        if worker is not None and matched_tokens >= min_cached_tokens:\n"
                    "            return worker\n"
                    "        return None"
                ),
                buggy_snippet=(
                    "    def route(self, tokens: list[int], now: float, min_cached_tokens: int = 8) -> str | None:\n"
                    '        """Returns sticky worker ID if matched tokens >= min_cached_tokens, else None."""\n'
                    "        worker, matched_tokens = self.tree.match_prefix(tokens, now=now)\n"
                    "        if worker is not None and (matched_tokens * self.tree.chunk_size) >= min_cached_tokens:\n"
                    "            return worker\n"
                    "        return None"
                ),
            ),
            Mutation(
                target_file="src/tensorroute/radix.py",
                clean_snippet=(
                    "            nxt.last_access_ts = now\n"
                    "            matched_tokens += 1\n"
                    "            matched_worker = nxt.worker_id"
                ),
                buggy_snippet=(
                    "            nxt.last_access_ts = now\n"
                    "            matched_tokens += self.chunk_size\n"
                    "            matched_worker = nxt.worker_id"
                ),
            ),
        ],
    ),
]


def inject_task_bug(ws: Path, task: CoupledBenchmarkTask) -> None:
    """Injects mutations for a single task."""
    for mut in task.mutations:
        fpath = ws / mut.target_file
        content = fpath.read_text(encoding="utf-8")
        if mut.clean_snippet not in content:
            raise RuntimeError(f"Clean snippet not found in {fpath} for {task.task_id}")
        mutated = content.replace(mut.clean_snippet, mut.buggy_snippet, 1)
        fpath.write_text(mutated, encoding="utf-8")


def apply_task_reference_fix(ws: Path, task: CoupledBenchmarkTask) -> None:
    """Restores clean snippets for a single task."""
    for mut in task.mutations:
        fpath = ws / mut.target_file
        content = fpath.read_text(encoding="utf-8")
        if mut.buggy_snippet not in content:
            raise RuntimeError(f"Buggy snippet not found in {fpath} for {task.task_id}")
        restored = content.replace(mut.buggy_snippet, mut.clean_snippet, 1)
        fpath.write_text(restored, encoding="utf-8")


def inject_all_coupled_bugs(ws: Path) -> None:
    """Simultaneously injects all 5 coupled traps into the workspace."""
    for task in COUPLED_TASKS:
        inject_task_bug(ws, task)


def apply_all_coupled_fixes(ws: Path) -> None:
    """Applies reference fixes for all 5 coupled traps."""
    for task in COUPLED_TASKS:
        apply_task_reference_fix(ws, task)


def apply_naive_fix(ws: Path, task: CoupledBenchmarkTask) -> None:
    """Applies the naive (flawed) fix for task."""
    for mut in task.naive_mutations:
        fpath = ws / mut.target_file
        content = fpath.read_text(encoding="utf-8")
        if mut.clean_snippet not in content:
            raise RuntimeError(
                f"Snippet to mutate not found in {fpath} for naive fix of {task.task_id}"
            )
        mutated = content.replace(mut.clean_snippet, mut.buggy_snippet, 1)
        fpath.write_text(mutated, encoding="utf-8")
