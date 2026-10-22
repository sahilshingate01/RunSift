# TensorRoute-Bench: 100% Original Autonomous Multi-Turn Bug-Fix Benchmark
<!-- SPEC-001 / SPEC-002 Verification Standard | Zero Third-Party Code -->

**Date:** 2026-10-14  
**Origin:** 100% Original Self-Contained Codebase (`tensorroute` — 5 subsystems, 45 unit tests).  
**Tasks:** 5 Deterministic Multi-Turn Bug-Fix Tasks (`OB-01` through `OB-05`).  

---

## 1. Design Principles & Zero-Third-Party Guarantee
- **100% Original Target Codebase:** Every module (`radix.py`, `scheduler.py`, `limiter.py`, `codec.py`, `telemetry.py`) and all 45 unit tests (`test_radix.py`, `test_scheduler.py`, `test_limiter.py`, `test_codec.py`, `test_telemetry.py`) are authored from scratch within this benchmark. Zero code or issues from external open-source repositories are used.
- **Zero Pre-Training Memorization:** Because `tensorroute` is an original benchmark suite created specifically for this evaluation, no LLM has memorized its source files or patches during pre-training.
- **Real Tool Execution & Binary Verification:** In every run, the autonomous agent executes real tools (`run_pytest`, `read_file`, `edit_file`, `search_code`, `finish`) inside an isolated workspace. A task is scored as **Solved ($R_{\text{solve}} = 1$)** if and only if the agent's code edit causes the full test suite (`pytest -o pythonpath=src -vv`) to transition from failing (`43 passed, 2 failed` or `41 passed, 4 failed`) to **100% passing (`45 passed, 0 failed`)**.

---

## 2. The 5 Original Benchmark Tasks (`OB-01` – `OB-05`)

| Task ID | Subsystem | Target File | Bug Description | Initial State | Target State |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`OB-01-radix-evict-refcount`** | `radix` | `src/tensorroute/radix.py` | `evict_expired_leaves()` decrements immediate parent `ref_count` twice when evicting an expired leaf, underflowing shared active prefixes to `0`. | `2 FAIL, 43 PASS` (`8,731` chars) | `45/45 PASS` |
| **`OB-02-scheduler-partial-drain`** | `scheduler` | `src/tensorroute/scheduler.py` | `drain_all_batches(flush_partial=True)` breaks and drops tail batches when `len(batch) < max_batch_size`. | `2 FAIL, 43 PASS` (`6,436` chars) | `45/45 PASS` |
| **`OB-03-limiter-fractional-refill`** | `limiter` | `src/tensorroute/limiter.py` | `_refill()` truncates elapsed seconds via `int(now - last_refill_ts)`, starving sub-second token refills. | `2 FAIL, 43 PASS` (`6,304` chars) | `45/45 PASS` |
| **`OB-04-codec-toolcall-concat`** | `codec` | `src/tensorroute/codec.py` | `SSEChunkAssembler.feed_line()` overwrites (`=`) streamed `function.arguments` across SSE chunks instead of concatenating (`+=`). | `2 FAIL, 43 PASS` (`8,873` chars) | `45/45 PASS` |
| **`OB-05-telemetry-percentile-bounds`** | `telemetry` | `src/tensorroute/telemetry.py` | `percentile_ttft()` off-by-one `IndexError` on 1-sample windows and `summary()` `ZeroDivisionError` on filtered empty windows. | `4 FAIL, 41 PASS` (`11,353` chars) | `45/45 PASS` |

---

## 3. Comparative Arms (`Baseline` vs. `RunSift + SDCS`)

1. **`Baseline` (Unoptimized Autonomous Agent Loop):**
   - Volatile timestamp & session UUID headers injected at token index 0 of the system prompt on every turn (breaking Radix KV-cache prefix matching).
   - Non-deterministic tool schema serialization across turns.
   - Raw verbose `pytest -vv --tb=long` stdout (`6,300–11,350` chars per run) appended directly into conversation history.
2. **`RunSift` (Canonicalized Prefix + Compacted Tool Memory + SDCS Cartography):**
   - **Prefix Canonicalizer:** Hoists volatile timestamp/session headers out of the system prompt prefix and deterministically sorts tool schemas so the SHA-256 `prefix_hash` remains 100% invariant across all turns.
   - **Negative Memory Compactor:** Compresses verbose `pytest -vv` logs (`~8,500` chars) down to a `<600`-char structured failure digest containing only failing test names, file/line coordinates, assertion errors, and summary counts—preserving 100% of the bug-fix signal while cutting prompt token growth by >60%.

---

## 4. Pre-Registration: 40-Turn 15-Repeat Gauntlet (Dated 2026-10-15)

**Date:** 2026-10-15  
**Configuration:** Multi-bug continuous gauntlet (`--mode gauntlet`), 40 max turns (`--max-turns 40`), 15 repeats (`--repeats 15`), temperature = 0.6 (`--temperature 0.6`), local Nemotron 30B GGUF on `llama-server`. Alternating arm order across repeats (baseline vs both).

### 4.1 Primary Metric & Hypothesis Test
- **Primary Metric:** Modules fixed per repeat ($0$ to $5$ modules fixed per repeat session).
- **Statistical Test:** Paired permutation test on per-repeat differences ($\text{RunSift} - \text{Baseline}$), two-sided exact permutation (or Monte Carlo $B = 100,000$), $\alpha = 0.05$.
- **Primary Analysis Dataset:** All $N = 15$ repeats unconditionally.
- **Secondary Analysis Dataset:** Valid-only repeats subset (repeats with $\le 20\%$ nudge rate across both arms).

### 4.2 Secondary Metrics
- **Context Overflows:** Count of sessions truncated by upstream context window exhaustion.
- **Turns Completed:** Total turns survived before termination or resolution.
- **TTFT (Time-To-First-Token):** Mean $\pm$ stdev across turns with signed deltas.
- **Uncached Tokens:** Total prompt tokens minus upstream server-reported cached tokens ($\text{Prompt} - \text{Cached}$), mean $\pm$ stdev with signed deltas.

### 4.3 Amendment (Dated 2026-10-15)
- **Change:** Added a 35-second process execution timeout (`timeout=35`) to `run_workspace_pytest()` in `benchmarks/tensorroute_benchmark/harness.py` (commit `1fece8a`), introduced after pre-registration commit `6968cab` and before launching the 15-repeat run.
- **Rationale:** Guard against agent-written infinite loops and deadlocks (e.g. while modifying loop conditions in `scheduler.py` or `limiter.py`), which would otherwise hang the benchmark subprocess indefinitely.
- **Symmetry & Impact:** The timeout applies identically and symmetrically to both arms (`Baseline` and `RunSift`). Audit of the 15-repeat execution traces confirms that no final result hit the 35-second timeout limit; all runs completed within normal execution bounds.

---

## 5. Pre-Registration: Gemma 4 26B Replication (Dated 2026-10-16)

**Date:** 2026-10-16  
**Model:** Gemma 4 26B-A4B-it GGUF (`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` on `llama-server` port 8083).  
**Configuration:** Multi-bug continuous gauntlet (`--mode gauntlet`), 40 max turns (`--max-turns 40`), 15 repeats (`--repeats 15`), temperature = 0.6 (`--temperature 0.6`), upstream kind `llamacpp` (`--upstream-kind llamacpp --base-url http://127.0.0.1:8083/v1 --gateway-url http://127.0.0.1:8765/v1`). Alternating arm order across repeats (baseline vs both).  
**Reporting Commitment:** Results will be reported whatever they show.

### 5.1 Primary Metric & Hypothesis Test
- **Primary Metric:** Modules fixed per repeat ($0$ to $5$ modules fixed per repeat session).
- **Statistical Test:** Paired permutation test on per-repeat differences ($\text{RunSift} - \text{Baseline}$), two-sided exact paired permutation (or Monte Carlo $B = 100,000$), $\alpha = 0.05$.
- **Primary Analysis Dataset:** All $N = 15$ repeats unconditionally.
- **Secondary Analysis Dataset:** Valid-only repeats subset (repeats with $\le 20\%$ nudge rate across both arms).

### 5.2 Secondary Metrics
- **Context Overflows:** Count of sessions truncated by upstream context window exhaustion.
- **Turns Completed:** Total turns survived before termination or resolution.
- **TTFT (Time-To-First-Token):** Mean $\pm$ stdev across turns with signed deltas.
- **Uncached Tokens:** Total prompt tokens minus upstream server-reported cached tokens ($\text{Prompt} - \text{Cached}$), mean $\pm$ stdev with signed deltas.

---

## 6. Pre-Registration: Protocol v3 (Native Tool Calling & Gateway-Side Compaction) (Dated 2026-10-16)

**Date:** 2026-10-16  
**Models:**
1. Gemma 4 26B-A4B-it GGUF (`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` on `llama-server` port 8083).
2. Nemotron-3-Nano-30B-A3B GGUF (`Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf` on `llama-server` port 8082).

**Protocol Amendments (Protocol v3):**
1. **Native OpenAI Tool Calling:** Harness passes `AGENT_TOOLS_SCHEMA` as `tools` to both arms. Models invoke tools natively via OpenAI `tool_calls` wire format (e.g., Gemma `<|tool_call|>` and Nemotron `<tool_call>` parsed by `llama-server`). Text-JSON parsing is retained strictly as fallback for responses without `tool_calls`.
2. **Product-Side Compaction:** Compaction is performed entirely by RunSift reverse proxy gateway (`src/compact/synthesizer.py`), operating on `role="tool"` messages. Pytest compaction extracts per-file counts, explicit requested target status (without hints or next-target spoilers), and assertion failure excerpts.
3. **Zero Harness Compaction:** Both arms receive 100% raw tool output from the harness. For the baseline arm, raw tool outputs flow uncompacted directly to the model server. For the RunSift arm, compaction is applied solely through the product gateway.

**Configuration:** Multi-bug continuous gauntlet (`--mode gauntlet`), 40 max turns (`--max-turns 40`), 15 repeats (`--repeats 15`), temperature = 0.6 (`--temperature 0.6`), upstream kind `llamacpp` (`--upstream-kind llamacpp`), arms: `baseline` vs `both` with alternating arm order across repeats.
- Gemma run: `--base-url http://127.0.0.1:8083/v1 --gateway-url http://127.0.0.1:8765/v1`, saving to `local_gauntlet_gemma4_26b_40turn_v3.json` (and `..._rep1.json` through `..._rep15.json`).
- Nemotron run: `--base-url http://127.0.0.1:8082/v1 --gateway-url http://127.0.0.1:8765/v1`, saving to `local_gauntlet_nemotron_30b_40turn_v3.json` (and `..._rep1.json` through `..._rep15.json`).

**Reporting Commitment:** Results will be reported whatever they show, for both models side by side.

### 6.1 Primary Metric & Hypothesis Test
- **Primary Metric:** Modules fixed per repeat ($0$ to $5$ modules fixed per repeat session).
- **Statistical Test:** Paired permutation test on per-repeat differences ($\text{RunSift} - \text{Baseline}$), two-sided exact paired permutation (or Monte Carlo $B = 100,000$), $\alpha = 0.05$.
- **Primary Analysis Dataset:** All $N = 15$ repeats unconditionally.
- **Secondary Analysis Dataset:** Valid-only repeats subset (repeats with $\le 20\%$ nudge rate across both arms).

### 6.2 Secondary Metrics
- **Context Overflows:** Count of sessions truncated by upstream context window exhaustion.
- **Turns Completed:** Total turns survived before termination or resolution.
- **TTFT (Time-To-First-Token):** Mean $\pm$ stdev across turns with signed deltas.
- **Uncached Tokens:** Total prompt tokens minus upstream server-reported cached tokens ($\text{Prompt} - \text{Cached}$), mean $\pm$ stdev with signed deltas.

### 6.3 Amendment: Final Evaluation Scoring Guard for Timed-Out and Incomplete Test Runs (Dated 2026-10-16)
- **Change:** Updated `score_final_evaluation()` and `extract_subsystem_status()` in `benchmarks/tensorroute_benchmark/harness.py` to explicitly guard against timed-out or incomplete pytest runs. If the final evaluation run hits the 35s timeout (`[TIMEOUT]` in output), or if the sum of passed and failed tests does not equal the expected suite count (45 tests for TensorRoute Gauntlet), the run is scored `modules_fixed = "0/5"` (0 modules fixed) and assigned `final_eval_status = "timeout"` or `"incomplete"` with `solved = False`.
- **Rationale:** Prior parser logic in `extract_subsystem_status()` only looked for `FAILED ` and `ERROR ` lines in raw pytest output. If pytest timed out (e.g. after a hang or infinite loop introduced by agent code edits) or crashed during collection, no `FAILED tests/test_*.py` lines were printed; as a result, `failing_subsystems` remained empty and the session was erroneously scored as 5/5 passing despite failing tests.
- **Empirical Impact & Rescoring:**
  - Audit scan of all 40-turn execution files identified exactly one affected repeat: `local_gauntlet_nemotron_30b_40turn_v3_rep5.json` (RunSift arm), where the agent timed out on turn 40 with 0 passed / 45 failed.
  - The repeat has been rescored from `5/5` to `0/5` modules fixed (`final_eval_status = "timeout"`). All original files are preserved intact; corrected results are written as `*_rescored.json`.
  - Corrected Nemotron 30B Protocol v3 result: RunSift `58/75` vs Baseline `37/75` modules fixed (mean difference $+1.4000$), paired exact permutation test $p = 0.047363$ ($\alpha = 0.05$). Gemma 4 26B is unaffected (RunSift `71/75` vs Baseline `30/75`, $p = 0.000122$).

---

## 7. Pilot Section & Pre-Registration: TensorRoute v2 Coupled-Bug Gauntlet (Dated 2026-10-16)

**Date:** 2026-10-16  
**Suite:** TensorRoute-Bench v2 Coupled-Bug Gauntlet (`--mode gauntlet --task-set coupled`).  
**Codebase:** 6 subsystems (`radix`, `scheduler`, `limiter`, `codec`, `telemetry`, `units`), 56 unit tests, 5 coupled traps (`T1` through `T5`).

### 7.1 Pilot Verification & Calibration Disclosure

A 3-repeat baseline pilot was conducted across models at expanded context windows to avoid artificial context exhaustion and establish baseline difficulty:
1. **Gemma 4 26B Pilot (98k context, `--n-cpu-moe 22`):**
   - Result: 6 / 15 traps solved (mean: **2.00 traps/repeat**). Overflows: 0 / 3 (0%).
   - All 3 repeats completed 40 turns without overflow.
   - Sits squarely within the pre-set calibration target of 1.0–3.5 traps.
2. **Nemotron 30B Pilot (131k context, `--n-cpu-moe 40`):**
   - Result: 5 / 15 traps solved (mean: **1.67 traps/repeat**). Overflows: 1 / 3.
   - Sits squarely within the pre-set calibration target of 1.0–3.5 traps.
3. **Excluded Gemma 131k Pilot Runs:**
   - The pilot for Gemma 4 26B at 131k context (`diag_coupled_pilot_gemma4_26b_131k.json`) ran at `--n-cpu-moe 22`. In Repeat 1, it completed 40 turns (3 traps solved, 54/56 tests passing).
   - In Repeats 2 and 3, `llama-server` crashed/died ("connection refused" on turn 1), and the harness (prior to upstream error classification) recorded them as context overflows at 781 tokens with `is_valid: true`.
   - The files `diag_coupled_pilot_gemma4_26b_131k_rep2.json`, `diag_coupled_pilot_gemma4_26b_131k_rep3.json`, and the aggregate `diag_coupled_pilot_gemma4_26b_131k.json` are preserved unedited on disk, but are explicitly **excluded** from calibration and validity because the server was down.
4. **Trap Difficulty Distribution in Pilot:**
   - `T1-shared-helper`: Solved in 5 of 6 valid pilot sessions (failed in Nemotron 131k rep 3).
   - `T2-units-boundary`: Solved in 3 of 6 valid pilot sessions.
   - `T5-workaround-root-cause`: Solved in 3 of 6 valid pilot sessions.
   - `T3-masked-bug` and `T4-changed-default`: Solved 0 times in 6 valid pilot sessions.
   - `T4-changed-default` architectural guard test (`tests/test_pipeline.py::test_pipeline_immediate_startup_admission`): broken by naive edits and never repaired in 5 of 6 valid pilot sessions.

### 7.2 Experimental Configuration (Counted Runs)

- **Models & Infrastructure:**
  - **Gemma 4 26B-A4B-it GGUF:** Port 8083, context window 98,304 (`-c 98304`), `--n-cpu-moe 22`, `--cache-reuse 64`, `--reasoning off`.
  - **Nemotron-3-Nano-30B-A3B GGUF:** Port 8082, context window 131,072 (`-c 131072`), `--n-cpu-moe 40`, `--cache-reuse 64`, `--reasoning off`.
  - Exactly one local model server runs at any given time.
- **Harness & Gateway:**
  - Protocol v3 harness with native OpenAI tool calling (`AGENT_TOOLS_SCHEMA`).
  - Gateway-side compaction (`src/compact/synthesizer.py`) on `role="tool"` messages for the RunSift arm.
  - Raw uncompacted pytest outputs flow to the baseline arm.
  - Upstream error classification active: connection refused, 5xx after 5 retries, timeouts, and network drops are classified as infrastructure errors (`is_valid: false`, `context_overflow: false`).
  - Keep-alive reconnect retry covers `ConnectionResetError`, `BrokenPipeError`, and `ConnectionAbortedError` (`WinError 10053`), and the local RunSift gateway runs with `--timeout-keep-alive 120`.
  - Auth/quota failures (401, 402, 403) abort the run immediately, preserving prior completed repeats on disk.
- **Batch Execution & Commit Freezing Rules:**
  - **Frozen Commit:** All counted batches must share one identical git commit (`git_dirty: false`) frozen after the Nebius Token Factory rescoring and `ConnectionAbortedError` reconnect fix (recorded in the result files' `provenance.git_commit`). Zero harness or scoring code modifications are permitted once local coupled Batch 1 starts.
  - **Execution Plan:** **15 paired repeats per model**, executed in **three batches of five** repeats (Batch 1: reps 1–5, Batch 2: reps 6–10, Batch 3: reps 11–15).
  - **Inter-Batch Inspection Guard:** Between batches, check system health only (infrastructure errors, process crashes, GPU memory and temperatures). Never inspect or compare arm deltas or interim effect sizes between batches, eliminating selective termination or protocol-drift bias.
  - Alternating arm order across repeats to eliminate thermal or sequence bias.
  - Max turns: 40 (`--max-turns 40`).
  - Temperature: 0.6 (`--temperature 0.6`).
  - Analyzed independently per model.
- **Infrastructure Error Replacement & Re-Run Procedure:**
  - Only an infrastructure error in either arm (`task_results[0].infra_error == true`, e.g. process crash, connection refused, unrecoverable network drop) makes a repeat replaceable. A stuck-agent repeat (`nudge_rate > 20%`, which also sets top-level `is_valid: false`) is model behavior, not infrastructure: it must never be re-run, stays in the primary analysis with its real score, and is excluded only from the valid-only secondary analysis.
  - Because `harness.py` (`raw_repeats[:repeat_offset]`) truncates an aggregate file if a middle repeat is re-run into the same `--result-name`, a re-run of repeat $N$ must be written to a separate `--result-name <prefix>_rerun<K>` with `--repeat-offset N-1 --repeats 1` (producing `<prefix>_rerun<K>_rep<N>.json` and preserving the original `<prefix>_rep<N>.json` untouched).
  - If more than 3 repeat re-runs fail ($>3$ infrastructure errors across a 15-repeat run), the run must halt and report the infrastructure failure for investigation.
  - Analysis is computed directly from the per-repeat files via `benchmarks/tensorroute_benchmark/scripts/analyze_gauntlet_results.py`, which selects exactly one non-infra-error file per repeat number ($1 \dots 15$) across `<prefix>_rep<N>.json` and `<prefix>_rerun*_rep<N>.json`, and fails immediately on any gap, duplicate non-infra-error file, or $>3$ skipped infrastructure-error files.

### 7.3 Statistical Hypothesis & Metrics

- **Primary Metric:** Traps solved per repeat session ($0$ to $5$ coupled traps solved, where a trap is scored solved if and only if all its F2P tests and guard tests pass in the final evaluation).
- **Statistical Test:** Paired exact permutation test on per-repeat differences ($\text{RunSift} - \text{Baseline}$), two-sided, $\alpha = 0.05$.
- **Primary Analysis Dataset:** All $N = 15$ complete non-infra-error paired repeats unconditionally (including any stuck-agent repeats with $> 20\%$ nudge rate).
- **Secondary Analysis Dataset:** Valid-only repeats subset (repeats with $\le 20\%$ nudge rate across both arms).
- **Secondary Metrics:**
  - Regressions introduced: count of passing tests broken by agent edits during the session.
  - Regressions repaired: count of introduced regressions successfully restored to passing by the end of the session.
  - Final pass-to-pass intact: binary indicator of whether all initial passing tests remained passing.
  - Context overflows: count of sessions truncated by upstream context window exhaustion.
  - Tokens per turn: mean and peak prompt tokens per turn.
  - Tokens per trap solved: total session prompt tokens divided by traps solved.

### 7.4 Reporting Commitment
Results will be reported whatever they show, for both models side by side, cited directly from committed result files.

### 7.5 Amendment: `exceed_context_size_error` Classifier Bug, `rep15` Reclassification, and Coupled Patch-Replay Rescoring (Dated 2026-10-17)

1. **`llama.cpp` Context Overflow Classifier Bug (`classify_upstream_error`):**
   - In `local_coupled_nemotron_30b_40turn_rep15.json` (collected on commit `2d64007`), the Baseline arm reached `129,845` prompt tokens on Turn 28 and on Turn 29 received from `llama-server`:
     ```json
     HTTP 400: {"error":{"code":400,"message":"request (137848 tokens) exceeds the available context size (131072 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":137848,"n_ctx":131072}}
     ```
   - Although this is a genuine 131,072-token context window overflow (`137,848 > 131,072` tokens), `classify_upstream_error` in `harness.py` matched `r"exceeds\s*context\s*size"` without `"the available"` (`exceeds the available context size`) and did not check `"exceed_context_size_error"`. As a result, `_execute_agent_session` fell through to the `else` branch, setting `infra_error: true`, `context_overflow: false`, and skipping final pytest evaluation (`passed_tests: 0`, `failed_tests: 56`, `traps_solved: 0`).
   - `classify_upstream_error` has been fixed to match `r"exceeds\s*(?:the\s*available\s*)?context\s*size"` and `r"exceed_context_size_error"`.
2. **Reclassification & Deterministic Coupled Patch-Replay Rescoring (`rescore_coupled_results.py`):**
   - Genuine context overflows are model/context-growth behavior, not infrastructure failures: **they are scored from the workspace state at the time of overflow and are never re-run.**
   - Because `harness.py` always captures `patch_diff` (`git diff src/`) and `regressions_introduced` even when `infra_error` is set, `benchmarks/tensorroute_benchmark/scripts/rescore_coupled_results.py` deterministically rescores coupled sessions by:
     1. Rebuilding the clean coupled repository (`materialize_clean_repo_coupled`), injecting all 5 coupled bugs (`inject_all_coupled_bugs`), and initializing git (`git init`, `git add`).
     2. Applying the session's recorded `patch_diff` via `git apply`.
     3. Running `pytest` with `--junitxml` (`run_workspace_pytest`) and scoring via `score_coupled_evaluation` using the session's recorded `regressions_introduced`.
   - Prior to rescoring `rep15`, `rescore_coupled_results.py` was validated on 6 normal coupled repeat files (`12` sessions across Gemma 4 26B and Nemotron 30B: `local_coupled_gemma4_26b_40turn_rep5`, `rep6`, `rep15`, and `local_coupled_nemotron_30b_40turn_rep1`, `rep7`, `rep8`) and reproduced `traps_solved`, `solved_traps`, `regressions_repaired`, `final_pass_to_pass_intact`, and `passed_tests`/`failed_tests` with 100% exact agreement.
   - Applying `rescore_coupled_results.py` to `local_coupled_nemotron_30b_40turn_rep15.json` produced `local_coupled_nemotron_30b_40turn_rep15_rescored.json` (leaving the original file untouched), reclassifying Baseline `rep15` as `context_overflow: true`, `infra_error: false`, appending the `137,848`-token overflow turn to `per_turn_prompt_tokens` (`peak_turn_prompt_tokens: 137848`, `total_prompt_tokens: 1604111`), and scoring its workspace state at Turn 29 via patch replay (`51` passed, `5` failed, `traps_solved: 2/5` `['T1-shared-helper', 'T5-workaround-root-cause']`, `regressions_introduced: 2`, `regressions_repaired: 0`, `final_pass_to_pass_intact: false`).

---

## 8. Pre-Registration: Nebius Token Factory Protocol v3 Gauntlet (Dated 2026-10-17)

**Date:** 2026-10-17  
**Suite:** TensorRoute-Bench v2 Original 5-Bug Gauntlet (`--mode gauntlet --task-set original`).  
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory (`--upstream-kind nebius`).

### 8.1 Experimental Configuration

- **Protocol v3:**
  - Original task set (`--task-set original`, 5 independent module bugs, 45 unit tests).
  - Native OpenAI tool calling (`AGENT_TOOLS_SCHEMA`) in both arms.
  - Gateway-side compaction (`src/compact/synthesizer.py`) on `role="tool"` messages for the RunSift arm; raw uncompacted pytest output for the baseline arm.
  - Max turns: 40 (`--max-turns 40`).
  - Temperature: 0.6 (`--temperature 0.6`).
  - 15 paired repeats executed in three batches of five (Batch 1: reps 1–5, Batch 2: reps 6–10, Batch 3: reps 11–15), alternating arm order across repeats (`--result-name nebius_gauntlet_nemotron_30b_40turn_v3`).
- **Context Window & Reasoning:**
  - Uses Token Factory's full context window (exact length is unknown because `/v1/models` does not report it; any upstream context-length errors are recorded as `context_overflow: true`).
  - Reasoning disabled symmetrically via `chat_template_kwargs: {"enable_thinking": false}` in both arms.
- **Batch Execution, Frozen Commit, and Re-Run Rules:**
  - Identical to Section 7.2: all batches share one frozen git commit (`git_dirty: false`), inter-batch checks are restricted to system/API health only, and auth/quota errors (401, 402, 403) abort immediately while preserving completed repeats.
  - Only an infrastructure error in either arm (`task_results[0].infra_error == true`, e.g. 5xx after 5 retries, timeout, or connection drop) makes a repeat replaceable. A stuck-agent repeat (`nudge_rate > 20%`, which also sets top-level `is_valid: false`) is model behavior, not infrastructure: it must never be re-run, stays in the primary analysis with its real score, and is excluded only from the valid-only secondary analysis.
  - Any repeat encountering an infrastructure error is re-run under the same repeat number into a separate `--result-name nebius_gauntlet_nemotron_30b_40turn_v3_rerun<K>` (`--repeat-offset N-1 --repeats 1`), halting if $>3$ infrastructure errors occur.
  - Final analysis is computed from the per-repeat files via `benchmarks/tensorroute_benchmark/scripts/analyze_gauntlet_results.py`, taking exactly one non-infra-error file per repeat number ($1 \dots 15$) and failing on any gap or duplicate.

### 8.2 Statistical Hypothesis & Metrics

- **Primary Metric:** Modules fixed per repeat session ($0$ to $5$ modules fixed).
- **Statistical Test:** Paired exact permutation test on per-repeat differences ($\text{RunSift} - \text{Baseline}$), two-sided, $\alpha = 0.05$.
- **Primary Analysis Dataset:** All $N = 15$ complete non-infra-error paired repeats unconditionally (including any stuck-agent repeats with $> 20\%$ nudge rate).
- **Secondary Analysis Dataset:** Valid-only repeats subset (repeats with $\le 20\%$ nudge rate across both arms).
- **Secondary Metrics:**
  - Context overflows: count of sessions hitting upstream context-length errors.
  - Tokens per session: total prompt tokens and total completion tokens per session.
  - Tokens per turn: mean prompt tokens per turn.
  - Tokens per fixed bug: total prompt tokens divided by modules fixed.
  - Cached tokens: upstream cached tokens (`usage.prompt_tokens_details.cached_tokens`) as reported by Nebius Token Factory.

### 8.3 Reporting Commitment
Results will be reported whatever they show, cited directly from committed result files.

### 8.4 Amendment: JUnit XML Classname Scoring Bug & Deterministic Patch-Replay Rescoring (Dated 2026-10-17)

1. **Harness JUnit XML Classname Mapping Bug (`score_junit_xml`):**
   - Since commit `690e642`, `score_junit_xml` in `harness.py` mapped failing `<testcase>` elements to modules by checking `"test_{s}.py"` or `".{s}."` against the `file` and `classname` XML attributes. Standard `pytest --junitxml` output does not emit a `file` attribute and formats `classname` as `tests.test_<module>` (ending in `test_<module>` with no trailing dot). Consequently, no failing testcase matched any module, and every non-timeout original-task session (`--task-set original`) scored `modules_fixed: "5/5"` regardless of how many tests failed.
   - In addition, `analyze_gauntlet_results.py` checked `"traps_solved" in b_res`, which is always `True` on `TaskArmResult` dicts (`traps_solved: 0` by default), causing original-task runs (`task_id: "GAUNTLET-WAVE-1"`) to read `traps_solved` (`0`) instead of `modules_fixed`.
2. **Deterministic Patch-Replay Rescoring (`rescore_v3_results.py`):**
   - Because each repeat JSON records the exact final `patch_diff` produced by each arm's workspace at session end, all 15 selected repeats of `nebius_gauntlet_nemotron_30b_40turn_v3` (`rep1`–`rep7`, `rerun1_rep8`, `rep9`–`rep15`, all collected on commit `c8a7a04`) were rescored deterministically without re-running any model turns:
     - Rebuild the clean repo (`materialize_clean_repo`), inject the 5 original gauntlet bugs (`inject_gauntlet_bugs`), initialize git, and apply the recorded `patch_diff` via `git apply`.
     - Run the 45 tests with `--junitxml` and compute `modules_fixed` using the corrected `score_junit_xml` module mapping.
     - Assert that the replayed `passed_tests` and `failed_tests` counts match the originally recorded `passed_tests` and `failed_tests` in every session (`0` mismatches across all 30 sessions).
     - Write the rescored results to `*_rescored.json` files alongside the untouched original JSON files.
   - Prior to rescoring the Nebius files, `rescore_v3_results.py` was validated against `local_gauntlet_nemotron_30b_40turn_v3_rep1`–`5_rescored.json` (which were scored before `690e642`) and reproduced their `modules_fixed` and pass/fail counts with 100% agreement across all 5 repeats (both arms).
   - The experimental protocol, model interactions, and collected trajectory data are completely unchanged.

---

## 9. Pre-Registration: RunSift Condenser v2 Evaluation (Dated 2026-10-17)

**Date:** 2026-10-17  
**Spec:** `docs/specs/CONDENSER_V2_SPEC.md` (revision 2)  
**Pre-Registered Optional Run C Decision:** **`Run C = YES`**

### 9.1 Questions

1. **Decisive:** on the coupled task where v1 failed, does v2 bring Nemotron's trap-solving back to the no-RunSift baseline while still cutting tokens?
2. **Platform:** does the same hold on Nebius Token Factory, the hackathon's required platform?

### 9.2 Run A, Decisive: Local Nemotron, Coupled Task

- **Setup:** identical to the v1 coupled run in pre-registration Section 7, so v1's 37-vs-13 result is directly comparable:
  - `llama.cpp` Nemotron 3 Nano 30B (`Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf`) with `-c 131072 -np 1 -ngl 99 --n-cpu-moe 40 --load-mode none -fa on --cache-reuse 64 --reasoning off`, on port `8082`;
  - the coupled task set (`--task-set coupled`), 40 turns (`--max-turns 40`), temperature `0.6` (`--temperature 0.6`);
  - the gateway with `--timeout-keep-alive 120` and `RUNSIFT_CONDENSER=v2`.
- **Arms:** baseline vs RunSift with `RUNSIFT_CONDENSER=v2`.
- **Size:** 15 paired repeats, in 3 batches of 5 (`--result-name local_coupled_nemotron_30b_40turn_v2`), with arm order alternating by repeat. Log GPU health to `benchmarks/tensorroute_benchmark/results/gpu_health_nemotron_30b_v2.csv` and stop at 85 C or on thermal slowdown, as before.
- **Primary metric:** traps solved per session, with guards included. Paired exact permutation test, two-sided, $\alpha = 0.05$.
- **Decision rule, fixed in advance:** we may say "v2 closes the v1 gap" only if both of these hold:
  - the v2-vs-baseline test is **not** significant ($p \ge 0.05$);
  - v2's total traps solved is at least **80%** of the baseline's total in this run.
  
  Otherwise we report what happened and make no recovery claim. v1's result (`13` vs `37`) is shown alongside for comparison; it is not part of a test.

### 9.3 Run B, Platform: Token Factory, Coupled Task

- **Setup:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory at its full context, with reasoning off via `chat_template_kwargs` in both arms. Coupled task set (`--task-set coupled`), 40 turns (`--max-turns 40`), temperature `0.6` (`--temperature 0.6`), 15 pairs in batches of 5.
- **Arms and metrics:** the same as Run A (`RUNSIFT_CONDENSER=v2`).
- **Commitment:** Run B is done **whatever Run A shows**, to avoid selective reporting. Its pre-registration is in the same commit as Run A's.
- **Budget:** about 2.5M prompt tokens per pair, based on the v1 coupled volumes. 15 pairs comes to about \$2–3 at list price. Check the credit balance before batch 1.

### 9.4 Optional Run C (`Run C = YES`)

- **Decision (fixed before Run A starts):** **`Run C = YES`**
- **Setup:** Nebius Token Factory (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`), original task set (`--task-set original`), baseline vs RunSift with `RUNSIFT_CONDENSER=v2`, 40 turns, temperature `0.6`, 15 pairs in batches of 5. It tests whether v2 keeps v1's tie on the original task set with similar token savings.

### 9.5 Secondary Metrics

These are the same for every run, reported whatever they show, with each delta's sign:

1. Prompt tokens per session, per turn and per trap/bug solved; completion tokens per session.
2. `wall_clock_s` per session (median and mean), with model time and tool time.
3. Calls per session of `run_pytest`, `search_code` and `read_file`. These are v1's failure signature: fewer test runs and more searching.
4. Sessions that never got more tests passing than at the start (v1 Nemotron: `11/15`).
5. Regressions introduced, regressions repaired, final pass-to-pass intact, and overflows.
6. Cached-token share. For Run B (and Run C): cost per session at list price, and under a hypothetical cached-input price of $1/10$.
7. Infra errors, with which arm each hit.

### 9.6 Rules

These are the same as pre-registration Sections 7 and 8, as amended:

- One frozen commit with `git_dirty: false`. Health-only checks between batches.
- **Infra errors** (connection loss, server down, 5xx after retries) get re-run under the same repeat number into `_rerun<K>`, at most 3 per run.
- **Genuine context overflows,** including `llama.cpp`'s `"exceeds the available context size"`, are model behavior. Score them; never re-run them.
- **Stuck-agent repeats** stay in the primary analysis.
- **401, 402 or 403** aborts the run.
- **Scoring check:** `analyze_gauntlet_results.py` produces the primary analysis. Also spot-check 3 randomly chosen sessions per arm by patch replay with the committed coupled rescorer (`rescore_coupled_results.py`), and report whether they match.

---

## 10. Pre-Registration: RunSift Condenser v2.1 Evaluation (Dated 2026-10-17)

**Date:** 2026-10-17  
**Spec:** `docs/specs/CONDENSER_V2_1_ADDENDUM.md`  
**Pre-Registered Run E Decision:** **`Run E = YES`**

### 10.1 Run D, decisive: local Nemotron, coupled task, baseline vs v2.1

- **Setup:** identical to Run A, so Run A's v2 numbers can sit beside it. That means llama.cpp Nemotron 3 Nano 30B with `-c 131072 -np 1 -ngl 99 --n-cpu-moe 40 --load-mode none -fa on --cache-reuse 64 --reasoning off` on port 8082; the coupled task set, 40 turns, temperature 0.6; the gateway with `--timeout-keep-alive 120` and `RUNSIFT_CONDENSER=v2.1`; 15 paired repeats in batches of 5 with alternating arm order; GPU logging and the same hard stops.
- **Quality, decision rule fixed in advance:** the metric is traps solved per session, using a paired exact permutation test, two-sided, α = 0.05. We may say "v2.1 keeps quality" only if p ≥ 0.05 **and** v2.1's total traps are at least 80% of the baseline total in this run.
- **Speed, decision rule fixed in advance:** the metric is `wall_clock_s` per session, using a paired exact permutation test on the per-repeat difference (v2.1 minus baseline), two-sided, α = 0.05.
  - We may say "v2.1 is faster than no RunSift" only if the mean difference is negative and p < 0.05.
  - We may say "no measurable slowdown" only if p ≥ 0.05.
  - If the difference is positive and p < 0.05, we report a slowdown.
  - These are two separate claims, each tested at 0.05. Both p-values are reported whatever they show.
- **Comparison with Run A** (descriptive only, because it was run at a different time): v2.1's cached share, uncached tokens, time to first token and wall-clock, next to Run A's v2 numbers. The two runs' baseline arms are a stability check. Report how much they differ.

### 10.2 Run E, platform: Token Factory, coupled task, baseline vs v2.1

- **Decision (fixed before Run D starts):** **`Run E = YES`**
- Same metrics and rules as Run D, on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` at full context, with reasoning off in both arms. **Decide yes or no in the section 10 commit, before Run D starts, and write the decision down.** The decision may take Run B's committed results into account, because they are known before Run D starts. It may not depend on Run D. Budget: about \$2–3 at list price.

### 10.3 Secondary metrics for every run

Report these whatever they show, each with the sign of its delta:
- prompt tokens per session, per turn and per trap solved;
- uncached tokens per session and cached share;
- median time to first token overall, and on the turns right after a test run (the mechanism);
- wall-clock, model time and tool time;
- `run_pytest`, `search_code` and `read_file` calls per session;
- never-improved sessions;
- scheduler-failed sessions;
- regressions introduced and repaired, final pass-to-pass intact, and overflows;
- peak prompt tokens;
- infra errors by arm;
- for Run E: cost at list price, and under a hypothetical cached-input price of 1/10.

### 10.4 Rules

Unchanged from sections 7–9:
- one frozen commit, `git_dirty: false`;
- health-only checks between batches;
- infra errors re-run into `_rerun<K>`, at most 3;
- genuine overflows are scored, never re-run;
- stuck-agent repeats stay in;
- 401, 402 or 403 aborts;
- the analysis comes from `analyze_gauntlet_results.py`, plus a 3-per-arm patch-replay spot check.

**New rule:** every batch log is saved in `results/` and committed. Only numbers that appear in committed files may be reported.

### 10.5 Which version ships (decision rule fixed now)

The default for the submission is the version with the strongest pre-registered evidence:

- **v2.1** becomes the default if Run D meets its quality rule **and** its speed result is either "faster" or "no measurable slowdown".
- **v2** stays the default if v2.1 fails the quality rule.
- If both fail, the write-up says so plainly. The default is whichever version the evidence supports, and every version's results are reported.

The README and Devpost cite the runs behind whichever version ships.

---

## 11. Pre-Registration: Run F — Standard Task on Nebius Token Factory with Condenser v2.1 (Dated 2026-10-18)

**Date:** 2026-10-18  
**Spec:** `docs/specs/RUN_F_PREREGISTRATION_ADDENDUM.md`

## 11.1 Why Run F

Under the pre-registered ship rule (section 10), **condenser v2.1 becomes the default**:
- **Run D** (local, hard task): 30 vs 24 traps, p = 0.43. That's exactly 80.0% of baseline, so the quality rule is met. No measurable slowdown (p = 0.41).
- **Run E** (Token Factory, hard task): 21 vs 20, p = 1.0.

v2.1 has no platform result on the **standard task**. The strongest Token Factory results so far come from v2: Run B (hard task, 29 vs 32) and Run C (standard task, 40 vs 65, p = 0.028). Run F gives the shipped default its own standard-task platform result, the counterpart to Run C, so every headline number describes the shipped version.

## 11.2 Setup

The setup is identical to Run C (section 9.4), except for the condenser:
- **Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory, full context, reasoning off via `chat_template_kwargs` in both arms.
- **Task:** the original task set (`--task-set original`), 40 turns, temperature 0.6.
- **Arms:** baseline vs RunSift with `RUNSIFT_CONDENSER=v2.1`. Start the gateway from the section 11 commit with `--timeout-keep-alive 120`.
- **Size:** 15 paired repeats in 3 batches of 5, with arm order alternating by repeat.
- **Procedure:** run a 10-turn smoke test first, as in Run E.

## 11.3 Decision rules (fixed in advance)

- **Quality:** the metric is modules fixed per session, using a paired exact permutation test, two-sided, α = 0.05. We may say "v2.1 keeps quality on the standard task" only if p ≥ 0.05 **and** v2.1's total is at least 80% of the baseline total in this run.
  - If v2.1 is **significantly better** (p < 0.05, total above baseline), we may say "v2.1 fixed more bugs than no RunSift".
  - If v2.1 is **significantly worse** (p < 0.05, total below baseline), see 11.5.
- **Speed:** the metric is `wall_clock_s` per session, using a paired exact permutation test on the per-repeat difference (v2.1 minus baseline), two-sided, α = 0.05, with the same wording rules as section 10:
  - "faster" only if the mean difference is negative and p < 0.05;
  - "no measurable slowdown" if p ≥ 0.05;
  - otherwise, report a slowdown.
- **Comparison with Run C** (descriptive only, run at a different time): show v2.1's numbers next to v2's in Run C. Only within-run comparisons are tested.

## 11.4 Secondary metrics

Report all of these whatever they show, each with the sign of its delta:
- prompt tokens per session, per turn and per fixed bug; completion tokens per session;
- uncached tokens per session and cached share;
- cost per session at list price ($0.06 / $0.24 per 1M), and under a hypothetical cached-input price of 1/10;
- median time to first token, overall and on turns right after a test run;
- wall-clock, model time and tool time;
- `run_pytest`, `search_code` and `read_file` calls per session;
- **never-improved sessions** (never more tests passing than at the start). This is the watch item from Run E (4/15 for v2.1 vs 0/15 for baseline);
- scheduler-failed sessions;
- overflows, sessions where the test suite hung, and early full solves;
- infra errors by arm.

## 11.5 What Run F can change (fixed in advance)

- Run F is reported **whatever it shows**, in the README and the Devpost write-up.
- Run F does **not** re-open the section 10 ship decision, **with one exception**: if v2.1 is **significantly worse** than the baseline on the standard task (p < 0.05, total below baseline), the default reverts to **v2**. v2 is the version with platform evidence on both tasks (Runs B and C), and the README and Devpost must say plainly why.
- The headline numbers for the shipped version come from the runs of that version: Runs E and F for v2.1, or Runs B and C for v2 if the exception applies. Results from other versions are shown labeled by version.

## 11.6 Rules

These are unchanged from sections 7–10:
- one frozen commit, `git_dirty: false`;
- health-only checks between batches; auto-continue if clean;
- infra errors are re-run under the same repeat number into `_rerun<K>`, at most 3;
- genuine overflows and hung test suites are model behavior: score them, never re-run them;
- stuck-agent repeats stay in;
- 401, 402 or 403 aborts the run;
- the analysis comes from `analyze_gauntlet_results.py`, plus a 3-per-arm patch-replay spot check with `rescore_v3_results.py`;
- every batch log is saved in `results/` and committed;
- only numbers that appear in committed files are reported.

**Budget:** about $2–3 at list price, based on Run C's volumes, and about 40–60 minutes. Check the credit balance before batch 1.

---

## 12. Turn-2 Probe: Pre-Registration Addendum (Dated 2026-10-18)

**Status:** committed as section 12 before the probe runs. The probe runs on that commit or a later clean one.
**Save in the repo as:** `docs/specs/TURN2_PROBE_PREREGISTRATION.md`, committed unchanged, in the same commit as section 12.
**Written:** 2026-10-18.

---

## 12.1 Why

In Run F, 4 of 15 RunSift sessions (reps 8, 10, 11 and 13) stopped calling tools on their **first reply after the first test run**. Instead the model wrote JSON plans (`"analysis"`, `"plan"`, `"commands"`). Those four sessions account for 11 of the 19-module gap.

This never happened in the other 195 sessions checked: Run F's baseline arm and both arms of v3, A, B, C, D and E. At that turn, v2 and v2.1 send **byte-identical** input (recomputed from the sidecars), so the v2.1 change can't be the cause. The question this probe answers is whether **condensed test output**, which both versions send, makes this failure more likely than raw output does.

## 12.2 Setup

- **Request:** Run F rep 8's second request, rebuilt with the harness's own request code: the same system prompt, tool schemas and user message, and the same assistant `run_pytest` call (`{"target": "None"}`), followed by the raw first test output from rep 8's sidecar (40,141 characters).
- **Arms:**
  - **raw:** sent directly to Token Factory, as the baseline arm does;
  - **condensed:** sent through the gateway with `RUNSIFT_CONDENSER=v2`, as the RunSift arm does.

  The two arms differ only in the tool message the model receives.
- **Integrity check (before counting):** the gateway's condensed tool message must equal the one recomputed from rep 8's sidecar byte for byte, which is 9,662 characters (`x-runsift-tool-chars-after` or `x-cacheforge-tool-chars-after`). If it doesn't match, stop and report.
- **Parameters:** model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, temperature 0.6, reasoning off via `chat_template_kwargs`, and the same `max_tokens` as Run F, in both arms.
- **Size:** 40 requests per arm, alternating raw, condensed, raw, condensed, and so on. No retries except for infra errors (HTTP 5xx or connection errors), which are re-sent and logged.

## 12.3 Metric and test (fixed in advance)

- **Primary:** the share of replies with **no tool call**, which is the event that triggers a nudge in the harness. Fisher's exact test, two-sided, α = 0.05.
- **Secondary (reported, not tested):** replies whose content starts with `{` and contains `"analysis"`; calls to tools that don't exist, such as `str_replace_editor`.

## 12.4 What the result can change

- If the condensed arm skips tool calls **more often, with p < 0.05**, the README says: "On the first reply after a condensed test log, the model skipped tool calls in x of 40 requests, against y of 40 with the raw log. This applies to v2 and v2.1 alike."
- Otherwise the README says: "Not reproduced in a 40-vs-40 probe (x of 40 vs y of 40)."
- **Either way,** the probe doesn't change the ship decision (sections 10 and 11.5) or any run's score. It tests one turn now, not the conditions on the day of Run F.

## 12.5 Rules

- Commit before running, so the result file records `git_dirty: false`.
- Save the raw reply of every request, all 80 of them, plus the log in `results/`, and commit them.
- 401, 402 or 403 aborts the probe. Never print the key.
- **Budget:** about 80 requests and about 0.7M prompt tokens, so under $0.10 at list price. About 10 minutes.

---

# Run G: Pre-Registration Addendum (PRE_REGISTRATION.md section 13)

**Status:** final. It is committed as section 13 before Run G starts, and Run G starts on that commit.
**Save in the repo as:** `docs/specs/RUN_G_PREREGISTRATION_ADDENDUM.md`, committed unchanged, in the same commit as section 13.
**Written:** 2026-10-18, after the turn-2 probe (section 12) reported.

---

## 13.1 Why Run G

- **v2 and v2.1 have never been compared directly.** Every run so far compared one of them with no RunSift, at different times, and the baseline swings between runs: on the standard task it scored 57, 40 and 59.
- **v2 is the default only because of the section 11.5 fallback.** v2.1 lost Run F (59 vs 39 after rescoring, p = 0.035). But three sessions that went wrong before the two versions could send different input account for 12 of that 20-bug gap. So Run F can't tell whether v2.1 is worse than v2.
- **v2's standard-task win rests on one run.** Run C: 40 vs 65, p = 0.028.
- **The probe didn't reproduce the failure.** Run F's JSON-plan failure was not reproduced in the turn-2 probe (section 12): 5 of 40 condensed replies vs 1 of 40 raw replies had no tool call, p = 0.20, and none of the 80 replies was a JSON plan. So Run G tests v2 and v2.1 as they are.

Run G runs all three arms at the same time, on the same task and the same platform state.

## 13.2 Setup

The setup is identical to Runs C and F except for the arms:
- **Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory, full context, reasoning off via `chat_template_kwargs` in every arm.
- **Task:** the original task set (`--task-set original`), 40 turns, temperature 0.6.
- **Arms:**
  - **baseline:** direct to Token Factory;
  - **v2:** through gateway A on port 8765, started with `RUNSIFT_CONDENSER=v2`;
  - **v2.1:** through gateway B on port 8766, started with `RUNSIFT_CONDENSER=v2.1`.

  Start both gateways from the section 13 commit with `--timeout-keep-alive 120`. The repo `.env` doesn't override a variable already set in the process, so each gateway keeps its own version.
- **Code under test:** the condensers as of the section 13 commit, including the fix that keeps `[TIMEOUT]` lines (78df5c0). It changes output only for test runs that don't finish.
- **Size:** 15 repeats per arm, 45 sessions, in 3 batches of 5.
- **Smoke test first:** one 10-turn repeat of each arm. Before Run G starts, the smoke files must pass the integrity checks in 13.7 and load in `analyze_three_arm.py` with one repeat.

## 13.3 How to run it

**Analysis.** It uses `benchmarks/tensorroute_benchmark/scripts/analyze_three_arm.py`, committed in 0c72973. Its test reproduces Run F: 59 vs 39, p = 0.0352. Pairs are matched by repeat number, and `_rescored` files are preferred.

**Order.** For each repeat k = 1 to 15, run two harness commands back to back. Odd k run the v2 command first; even k run the v2.1 command first.
- **Baseline and v2:** `--arm both --gateway-url http://127.0.0.1:8765/v1 --repeats 1 --repeat-offset (k−1) --result-name nebius_gauntlet_nemotron_30b_40turn_runG_v2`
- **v2.1:** `--arm runsift --gateway-url http://127.0.0.1:8766/v1 --repeats 1 --repeat-offset (k−1) --result-name nebius_gauntlet_nemotron_30b_40turn_runG_v2_1`

Both commands also take every other flag exactly as in Run F (`--nebius --mode gauntlet --task-set original --upstream-kind nebius`, the base URL, the model, `--temperature 0.6 --max-turns 40`).

**Batches.** After repeats 5 and 10, run the usual health-only checks, and continue automatically if they're clean.

## 13.4 Comparisons and decision rules (fixed in advance)

The metric is modules fixed per session, after the full patch-replay rescore (13.7). Each test is a two-sided paired exact permutation test over the 15 repeat pairs.

| Test | Comparison | Role |
|---|---|---|
| **G1** | v2.1 vs v2 | primary: the head-to-head |
| **G2** | v2 vs baseline | replication of Run C |
| **G3** | v2.1 vs baseline | replication of Run F |

- **Multiple tests:** "significant" means significant after a Holm–Bonferroni correction across G1–G3 at family α = 0.05. Raw p-values are reported too.
- **Wording, applied to each test:**
  - "**better**" or "**worse**": the Holm-adjusted p < 0.05;
  - "**no significant difference**": otherwise, reported with both totals and the raw p. With 15 pairs this is weak evidence, so the README must never call it "the same" or "equivalent".
- **Prompt tokens and cost (G4, not in the Holm family):**
  - the metric is prompt tokens per session, v2.1 vs v2, with the same test;
  - "v2 sends fewer tokens per session than v2.1" may be claimed only if p < 0.05;
  - cost per session at list price is reported next to it.

## 13.5 What Run G can change (fixed in advance)

**The Token Factory default stays v2,** unless G1 finds v2.1 significantly better than v2. In that case the default becomes v2.1, and the README says it costs more per session (G4).

**What the README recommends for local GPUs:**
- **v2.1, citing Run D's speed result,** if G1 doesn't find v2.1 worse **and** v2.1's total is at least 80% of v2's;
- **v2,** if G1 finds v2.1 worse. The README then says v2.1 lost to v2 head-to-head, and v2.1 stays only as a documented option.

**G2 changes the headline wording:**
- **v2 better than baseline:** "v2 fixed more bugs than no RunSift in both standard-task runs (C and G)";
- **no significant difference:** the README shows Run C's win and Run G's result side by side, and states no win claim beyond Run C;
- **v2 worse than baseline:** the README states plainly that Run C's win did not replicate.

**What always applies:**
- **Every result goes in the README and the Devpost write-up,** next to Runs C and F.
- **No earlier result is withdrawn or re-scored** because of Run G.
- **The headline cost claims don't depend on Run G.** If Run G's costs differ, they're added to the table like every other run.

## 13.6 Secondary metrics (reported for all three arms, with signs)

- prompt tokens per session, per turn and per fixed bug; completion tokens;
- uncached tokens and cached share; cost per session at list price, and under a hypothetical cached-input price of 1/10;
- median time to first token, overall and on the turns right after a test run;
- wall-clock, model time and tool time per session;
- overflows, hung test suites, early full solves;
- sessions with JSON-plan replies, and the number of such replies;
- sessions whose first reply after the first test run had no tool call, by arm. This follows up the probe; it is reported, not tested;
- sessions that never improved; `test_files_edited` per session;
- infra errors by arm.

## 13.7 Rules

These are unchanged from sections 7–12:
- **Frozen code:** one frozen commit, `git_dirty: false`.
- **Integrity checks before counting any result:**
  - the v2 arm carries `x-runsift-condenser: v2` (or `x-cacheforge-condenser: v2`) on 100% of turns;
  - the v2.1 arm carries `x-runsift-condenser: v2.1` (or `x-cacheforge-condenser: v2.1`) on 100% of turns;
  - every sidecar recomputes exactly;
  - no API key appears in any file.
- **Infra errors** are re-run under the same repeat number into `_rerun<K>`, at most 3 per arm. Genuine overflows and hung test suites are scored, never re-run. Stuck-agent repeats stay in. 401, 402 or 403 aborts the run.
- **Scoring:**
  - before its final evaluation, the harness restores edited files under `tests/` and removes new ones, recording both in `test_files_edited` (1b31b50, 9ffaee1);
  - then every one of the 45 sessions is replayed with `rescore_v3_results.py`, and the analysis uses the `_rescored` files;
  - any session whose replay differs from the harness score is listed with the reason.
- **Logs and citations:** every batch log is saved in `results/` and committed (logs are git-ignored, so add them with `git add -f`). Only numbers that appear in committed files are reported.

**Budget:** about $0.19 per repeat at list price ($0.12 baseline, about $0.03 for v2, about $0.04 for v2.1, from Runs C and F). That's about $3 for 15 repeats plus the smoke test, and roughly 1 to 1.5 hours. Check the credit balance before the smoke test.
