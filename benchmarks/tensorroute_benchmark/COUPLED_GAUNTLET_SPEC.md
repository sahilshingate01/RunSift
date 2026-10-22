# Coupled-Bug Gauntlet Spec ("TensorRoute v2, coupled")

Build, verify, pilot, then stop for review. Everything here runs locally: no Nebius tokens.

**Status of prerequisites:** protocol v3 is verified and committed (7ba271d, 4309e81, 690e642). Build on the v3 harness: native tool calls, raw tool output from the harness, and compaction done only by the RunSift gateway on `role="tool"` messages. Use the manifest-driven scoring and the JUnit XML guard that are already in the harness.

**First:** commit this spec file on its own, so the working tree is clean before any build or run.

## Goal

Measure whether agents can fix bugs without breaking other modules, and whether RunSift's gateway-side compaction helps or hurts that. Leave the existing 5-bug gauntlet unchanged. This is a separate task set with its own pre-registration.

## 1. Build the v2 repo

Create a new builder and task file. Don't modify the existing gauntlet's sources.
- `benchmarks/tensorroute_benchmark/repo_builder_coupled.py`
- `benchmarks/tensorroute_benchmark/tasks_coupled.py`

The v2 package is the current five modules (`radix`, `scheduler`, `limiter`, `codec`, `telemetry`) plus the following.

**`src/tensorroute/units.py`**
- `to_ms(seconds: float) -> float`
- `safe_ratio(part: float, whole: float) -> float`
  - Returns a **fraction** in [0, 1].
  - Returns 0.0 when `whole == 0`.
  - The docstring states the fraction contract explicitly.

**Changes to the existing modules**
- **limiter:** add `utilization(now) -> float`, equal to `units.safe_ratio(capacity - tokens, capacity)`. This is a fraction.
- **telemetry:** `summary()` computes `hit_rate_pct = safe_ratio(hits, n) * 100` and `token_reuse_pct = safe_ratio(total_cached, total_prompt) * 100`.

**`src/tensorroute/pipeline.py`**
- `build_pipeline(capacity, refill_rate, max_batch_size, max_batch_tokens, chunk_size) -> Pipeline`
  - This factory constructs the `TokenBucketLimiter`, `ContinuousBatchScheduler`, `PrefixRadixTree` and `LatencyWindow`.
- `Pipeline.admit_all(requests, start_ts, tick=0.25, deadline_s=30) -> list[str]`
  - Enqueues the requests and drains batches with `flush_partial=True`.
  - Admits each request via `limiter.try_acquire(req.prompt_tokens, now)`, advancing `now` by `tick` until the request is admitted or the deadline passes.
  - Returns the admitted request ids.
- `Pipeline.record_completion(result: dict, ttft_seconds: float) -> None`
  - Reads `result["usage"]` (the output of `SSEChunkAssembler.finalize()`).
  - Calls `window.record(ttft_ms=units.to_ms(ttft_seconds), prompt_tokens=…, cached_tokens=…, cache_hit=cached_tokens > 0)`.
- `Pipeline.route(tokens, now, min_cached_tokens=8) -> str | None`
  - Uses `tree.match_prefix`.
  - Returns the worker if `matched_tokens >= min_cached_tokens`, otherwise `None` ("cold").

**Tests:** `tests/test_units.py` and `tests/test_pipeline.py` (integration tests), plus the existing per-module tests.
- Add `tests/test_limiter.py::test_utilization_is_fraction`, which expects for example 0.25.
- **Remove** the limiter's sub-second refill unit tests in v2. That way trap 3's limiter bug is only visible through the pipeline.
- Every pipeline test builds its objects via `build_pipeline()`. No test constructs a limiter directly.

## 2. The five traps

Inject all five at once, as in the existing gauntlet. For each trap the spec gives:
- **Bug:** what's injected.
- **Symptom:** the only text the agent sees about it.
- **F2P:** the fail-to-pass tests, which the fix must make pass.
- **Naive fix:** the obvious-but-wrong fix. Save it as a patch fixture.
- **Guard tests:** passing tests that the naive fix breaks.
- **Correct fix.**

### T1: shared helper
- **Bug:** `telemetry.summary()` omits the `* 100`, so `*_pct` fields come back as fractions.
- **Symptom:** "The dashboard shows a cache hit rate of 0.5 when half the requests hit; it should read 50."
- **F2P:** `test_telemetry_hit_rate_is_percentage`, `test_telemetry_token_reuse_is_percentage`.
- **Naive fix:** make `units.safe_ratio` return a percentage.
- **Guards broken by the naive fix:** `tests/test_limiter.py::test_utilization_is_fraction`, plus the `safe_ratio` tests in `tests/test_units.py`.
- **Correct fix:** multiply by 100 at the telemetry call sites.

### T2: units at the boundary
- **Bug:** `Pipeline.record_completion` passes `ttft_seconds` straight into `window.record`, with no `to_ms`.
- **Symptom:** "Recorded p95 time-to-first-token is about 0.2 ms; real requests take about 200 ms."
- **F2P:** `test_pipeline_records_latency_in_ms`.
- **Naive fix:** multiply by 1000 inside `LatencyWindow.record`.
- **Guards broken by the naive fix:** the telemetry unit tests that pass milliseconds directly (the percentile and mean tests).
- **Correct fix:** call `units.to_ms()` in `record_completion`.

### T3: masked bug
- **Bugs (two):**
  - `drain_all_batches` drops the partial tail batch when `flush_partial=True` (OB-02 style).
  - `_refill` truncates sub-second elapsed time (OB-03 style).
- **Symptom:** "In each admission wave, the last few queued requests are never admitted."
- **F2P:** `test_scheduler_flushes_partial_tail`, `test_pipeline_admits_all_requests_including_tail`.
  - The pipeline test needs **both** fixes.
  - After a scheduler-only fix, the pipeline test must still fail, with a *different* assertion (requests rejected or timing out).
- **"Naive" fix:** fixing only the scheduler. This surfaces the second failure.
- **Correct fix:** fix both.

### T4: changed default
This one is a spec change delivered as a bug report: the limiter must start empty by default.
- **Symptom:** "A newly created rate limiter starts full, allowing a burst at startup. It should start empty unless told otherwise."
- **F2P:** `test_limiter_starts_empty_by_default`.
- **Naive fix:** change only the default of `initial_tokens`.
- **Guards broken by the naive fix:** the pipeline tests that expect immediate admission at startup.
- **Correct fix:** change the default **and** pass `initial_tokens=capacity` explicitly in `build_pipeline()`.
- **Composition check:** the correct T3 and T4 fixes applied together must pass everything.

### T5: workaround vs root cause
- **Bug:** `radix.match_prefix` counts matched chunks instead of matched tokens (OB-06 style).
- **Symptom:** "Prefix-aware routing sends requests to cold workers even when a warm prefix exists."
- **F2P:** `test_radix_match_counts_tokens`, `test_pipeline_routes_to_warm_worker`.
- **Guard:** `test_pipeline_does_not_route_below_threshold`, which passes at the start.
- **Naive path:**
  1. Patch `Pipeline.route` to multiply the matched count by `chunk_size`. The pipeline F2P test now passes, but the radix F2P test still fails.
  2. Then fix radix as well. The count is now doubled, so the guard fails.
- **Correct fix:** fix `radix.match_prefix` only.

### Rules for the symptom text
- Describe user-visible behaviour only.
- No file names, function names, or mentions of other modules.

## 3. Prove the traps

Extend `verify_tasks.py` with `--task-set coupled`. The build fails unless all of the following hold:

1. The clean repo passes every test.
2. The buggy repo, with all five bugs injected, fails exactly the union of the F2P sets.
3. Each correct fix passes its F2P tests without breaking anything.
4. Each naive patch behaves as designed:
   - **T1, T2, T4:** passes its F2P tests but breaks at least one of its guards.
   - **T3:** the scheduler-only fix leaves the pipeline F2P test failing.
   - **T5:** the workaround plus the radix fix fails the guard.
5. All five correct fixes applied together pass everything.

**Output:** compute the F2P and guard sets from these actual runs. Write them, with `test_count` and `trap_count`, to `results/tasks_coupled_manifest.json`, along with the SHA-256 of every source file. **This manifest is generated, never hand-written.** The harness reads `test_count` and `trap_count` from it.

## 4. Harness changes

These apply to task set `coupled`, identically for both arms.

- **`tests/` is read-only** for `edit_file` in this task set. Return `Error: tests are read-only.` This prevents "fixing" the tests instead of the code.
- **Per-test logging:** run pytest with `--junitxml` as well as the normal verbose output, and record per-test pass/fail at every pytest run.
  - This is logging only. The model still sees the normal pytest text, raw from the harness. Any compaction is done by the gateway.
- **Scoring:** if the final JUnit XML is missing or incomplete (timeout or crash), score that repeat as 0 traps solved.
- **Recorded per repeat:**
  - `traps_solved`: a trap counts as solved when all its F2P tests **and** all its guard tests pass at the end. Record the count (0–`trap_count`) and the list of which traps.
  - `regressions_introduced`: tests from the initial passing set that failed at any later pytest run.
  - `regressions_repaired`: how many of those pass at the end.
  - `turns_to_repair`: for each regression, the turn it appeared and the turn it was fixed.
  - `final_pass_to_pass_intact`: bool.
  - Everything the v3 harness already records: turns, overflow, nudges, tokens, server cached tokens, TTFT, `per_turn_tools`, raw replies, tool feedback.

## 5. Pilot

This is calibration only. Pilot runs are not counted.

- **Runs:** 3 repeats, **baseline arm only**, 40 turns, temperature 0.6, on each local model.
- **Order:** one server at a time.
  1. Gemma 4 26B on port 8083.
  2. Then Nemotron 30B on port 8082.
- **Launch commands:** use the exact commands recorded in the v3 runs' provenance (`server_launch_command`). Restart the gateway from the current commit, pointed at the active server, and record `gateway_commit`.
- **Target:** a baseline mean of 1.0 to 3.5 traps solved.
- **If outside that range:** adjust only the symptom wording, never the code or the tests, then re-pilot. Note every adjustment.
- **Save as:** `diag_coupled_pilot_*`.

**STOP after the pilot and report:**
1. The full `verify_tasks --task-set coupled` output, including every trap proof.
2. `tasks_coupled_manifest.json`.
3. The pilot table: traps solved per repeat (and which traps), regressions introduced and repaired, turns, overflow, and nudges.

Do **not** pre-register or start counted runs until that report has been checked.

## 6. Pre-registration (later, after the pilot is reviewed)

Add a dated section to `PRE_REGISTRATION.md`: "Coupled-bug gauntlet v1". Commit it before the first counted run. It must include:
- The task set and the manifest hash.
- The models and their launch commands.
- Run configuration: 40 turns, temperature 0.6, 15 repeats, arms baseline vs `both`, alternating arm order.
- **Primary metric:** traps solved with guards intact (0–5 per repeat).
- **Test:** two-sided exact paired permutation test on per-repeat differences (RunSift − baseline), alpha 0.05, over all repeats.
- **Secondary metrics:**
  - The valid-only subset, with the caveat stated up front that it can be biased when exclusions fall mostly on one arm.
  - Regressions introduced and repaired, `turns_to_repair`, overflows, tokens, TTFT.
- A statement that the results will be reported whatever they show.

## 7. Counted runs (later)

**Before starting:**
- Commit first, so `git_dirty` is false at launch.
- Restart the gateway from that commit and record `gateway_commit`.

**Order and naming:** Gemma first, then Nemotron. Name the results `local_coupled_gemma4_26b_40turn*` and `local_coupled_nemotron_30b_40turn*`.

**Watchdog (every 20 minutes):**
- Is a new repeat file appearing?
- Do the model server and the gateway health checks both answer?
- Is the harness process alive?

**If something goes wrong:**
- **Infrastructure failure with no code change:** restart the service, resume with `--repeat-offset`, and note it in the report.
- **A code fix is needed:** stop, commit the fix, add a dated amendment, and restart that model's batch from repeat 1 under a new name.
- **Never edit `src/` or the harness during a run.**
- **The same failure happens twice:** stop and report.

## 8. Report (later)

- **Per repeat, both arms:**
  - traps solved, and which ones
  - regressions introduced and repaired
  - `turns_to_repair`
  - `final_pass_to_pass_intact`
  - turns, overflow, nudges, valid
- **Totals** and the pre-registered p-value for each model.
- **Side by side** with the v3 independent-bug results.
- **Sources:** name the file behind every number, and quote only text that appears in logged files.

## Out of scope for now (planned)

- **Ablation:** the gateway digest flags tests "newly failing since the last run".
- **Phase 2: the SDCS 2×2 on this task set.**
  - Arms: RunSift on/off × SDCS on/off.
  - The SDCS arms get an `app_map.md` and `wiring.yaml` describing these module dependencies, with no answers in them.

## Done when (for this build-and-pilot phase)

- `verify_tasks --task-set coupled` passes every trap proof. Paste the output.
- The harness unit tests cover the read-only-tests rule and the regression bookkeeping.
- The following all pass, with their final lines quoted:
  - `python -m pytest -q`
  - `ruff check .`
  - `ruff format --check .`
  - `sdcs verify --topology`
  - `sdcs verify --mocks`
- The manifest and the pilot results are committed.
- The pilot report is delivered, and work stops there.
