# RunSift Condenser v2: Specification and Pre-Registration Draft

**Status:** design, revision 2. Ready to implement once the coupled-run bookkeeping (the classifier fix, the rep15 rescore, the section 7 amendment) is committed and reviewed.
**Save in the repo as:** `docs/specs/CONDENSER_V2_SPEC.md`, in the first commit of this work.
**Written:** 2026-10-17. Revision 2 adds the coupled-gauntlet result and makes local Nemotron on the coupled task the decisive test. Every evidence number below was checked against result files or the cited papers.

---

## 1. Why v2

Condenser v1 (`src/compact/synthesizer.py`) delivered the headline result. On Nebius Token Factory (Nemotron 3 Nano, full 262k context, 15 pairs), it fixed the same number of bugs (57 vs 57 of 75) with 74% fewer prompt tokens per session.

**Then the pre-registered coupled gauntlet** (section 7 of the pre-registration, commit `2d64007`, 15 pairs per model) showed where v1 fails:

| Model, context | Traps solved, baseline vs v1 | p | Prompt tokens per session |
|---|---|---|---|
| Gemma 4 26B, 98k | 39 vs 39 of 75 | 1.0 | −65% |
| Nemotron 30B, 131k | **37 vs 13** of 75 | **0.0044** | −68% |

The Nemotron figure counts rep15's baseline as a genuine overflow, rescored by patch replay. Excluding rep15 gives p = 0.0063; as recorded, p = 0.0105.

**Mechanism:** in v1's Nemotron sessions, the agent:
- ran pytest about half as often (73 vs 153 calls);
- searched about twice as much (231 vs 122 `search_code` calls);
- in 11 of 15 sessions, never got more tests passing than at the start.

v1 keeps each failing assertion but drops the test body that shows *which code is under test*. Two real blocks from the coupled log:

```
def test_pipeline_routes_to_warm_worker() -> None:
    pipeline = build_pipeline(chunk_size=4)                        # dropped by v1
    pipeline.tree.insert_prefix([1, ..., 8], worker_id="gpu-warm", now=10.0)   # dropped
    worker = pipeline.route([1, ..., 8], now=20.0, min_cached_tokens=8)        # dropped
>   assert worker == "gpu-warm"                                    # kept

def test_radix_match_counts_tokens() -> None:
    tree = PrefixRadixTree(chunk_size=4, default_ttl=100.0)        # dropped
    tree.insert_prefix([1, ..., 8], worker_id="gpu-0", now=10.0)   # dropped
    worker, matched = tree.match_prefix([1, ..., 8], now=20.0)     # dropped
>   assert matched == 8                                            # kept
```

Without those lines, the model can't tell which function returned the wrong value. Section 4.4's rule keeps all of them.

More findings, from the v1 Token Factory run:

| Finding (exploratory, from the v1 Token Factory run) | Source |
|---|---|
| RunSift sessions left `scheduler` unfixed in 7/15 sessions, vs 2/15 for the baseline (Fisher p = 0.109) | rescored `nebius_gauntlet_nemotron_30b_40turn_v3_*_rescored.json` |
| v1 drops each failing test's setup lines, including `batches = sched.drain_all_batches(flush_partial=...)` | `_extract_failure_excerpts`, synthesizer.py:250-255 |
| v1 cuts every kept line at 300 chars from the head, losing the end of long values (`req-3`/`req-4`) | synthesizer.py:255 |
| RunSift sessions made 20% more `read_file` calls (139 vs 116), mostly re-opening test files | per-turn tools in the same files |
| The harness records no wall-clock time, no tool-call arguments, and only the first 1,500 chars of raw tool output (`harness.py:1699`) | harness code |

Evidence from published studies (verified in the papers' text):

- **Lost in Context Compaction** (Penn State + IBM, arXiv 2608.11242): compactors keep 17% of standing instructions on average. → Touch only tool output.
- **CliffCompaction** (arXiv 2609.26779): "We never compact a compaction"; it only truncates or drops content, never rewrites. → Extract verbatim, always from the original.
- **Beyond Token Savings** (arXiv 2609.32961): fewer tokens can mean longer runs and lower success; history rewrites break cache pricing. → Measure wall-clock and success, and stay cache-stable.
- **Fan et al., harness study on Nemotron-3** (arXiv 2609.20804): staged handling of stale observations was cheapest in 7 of 8 panels; a recall tool was never called in 36 of 64 settings. → Condense stale output harder; no recall tool.
- **SDCS Gate D distiller** (the author's own earlier work): keep what the failing line references. The idea is re-implemented here; no SDCS code is imported.

## 2. Scope

**In scope:**
- A new condenser module in `src/compact/`.
- A config flag that selects v1 or v2.
- Gateway wiring and a response header.
- Harness instrumentation needed to evaluate v2.
- Unit tests.
- Smoke tests and two pre-registered runs: local Nemotron on the coupled task (decisive), then Token Factory on the coupled task.

**Out of scope:**
- Tool-schema filtering.
- A recall tool.
- LLM summarization.
- Model routing.
- Any change to how system, user or assistant messages are handled.

**Boundaries:**
- The `compact` subsystem may depend only on `core` and `compact` (`wiring.yaml`). If the implementation seems to need anything else, stop and ask. Don't edit `wiring.yaml` or `spine.md`.
- No SDCS code or SDCS files in the product or in either benchmark arm.

## 3. Invariants

Each invariant has at least one test in section 7.

- **I1. Tool output only.** System, user and assistant messages pass through byte-identical. Only `role: "tool"` messages with string content are condensed.
- **I2. Exempt tools untouched.** Outputs of tools in `EXEMPT_TOOL_NAMES` (`read_file`, `grep`, `cat`, `search_code` and the rest) pass through byte-identical, so the code the agent edits is never altered.
- **I3. Extract, never rewrite.** Every content line in a v2 digest is copied verbatim from the original output. There are two exceptions, and both are clearly marked:
  - generated lines start with `[cf:` and end with `]`;
  - clipped lines contain `…[cf: N chars clipped]…`.
- **I4. Deterministic and stateless.** The output depends only on the message list in the request. The same request always produces identical bytes. No state is stored between requests.
- **I5. Idempotent.** Condensing a v2 digest returns it unchanged, detected by the `<!-- cf:v2 -->` marker. A digest is never condensed again.
- **I6. Always from the original.** Clients re-send raw tool output on every turn, and v2 derives each digest from that raw text, never from an earlier digest.
- **I7. v1 is preserved.** With `RUNSIFT_CONDENSER=v1` (or legacy `CACHEFORGE_CONDENSER=v1`), output is byte-identical to commit `2d64007`, so earlier results stay reproducible.

## 4. Design

### 4.1 Message pass

For each request:

1. Index the assistant `tool_calls` by id, recording function name and raw `arguments`. v1's `compact_messages` already does this; reuse the approach.
2. For each `role: "tool"` message:
   - resolve its tool name and arguments;
   - skip it if the tool is exempt, the content isn't a string, or the content is at or below the threshold (`max_tool_output_tokens * 4` chars, as in v1).
3. Classify each remaining message:
   - **Pytest output:** use v1's detection (`_is_pytest_output`).
   - **Other:** everything else.

### 4.2 Freshness (pytest only)

- A pytest output's **run key** is `(tool name, normalized target)`, where the target is the `target` argument, or `""` for a full-suite run.
- For each run key, the **latest** output in the message list is **fresh**. Every earlier output with the same run key is **stale**.
- A targeted run never makes a full-suite run stale, and the reverse is also true.

Freshness depends only on the message list, so it is deterministic (I4). When a new test run arrives, the previous run with the same key turns stale. Expect exactly one cache break per new test run; section 8 measures the cost.

### 4.3 Fresh pytest digest

Output, in order:

1. The first line: `[cf: condensed pytest output, v2]`.
2. If a target was given, v1's target status line, prefixed `[cf: ...]`.
3. Per-file counts: `[cf: tests/test_x.py: 7 passed, 2 failed]`, one line per file, sorted.
4. For each failure block in the FAILURES section, in order:
   - its `____ test_name ____` separator line, verbatim;
   - the **referenced setup lines** from 4.4;
   - the `>` line(s) and every `E` line of the test frame;
   - for any deeper frame inside the project (`src/`): its `>` line and its `E` lines;
   - the location line (`tests\test_x.py:61: AssertionError`).
5. The `short test summary info` lines (`FAILED ...`), verbatim.
6. The final `=== N failed, M passed in Xs ===` line, verbatim.
7. The last line: `<!-- cf:v2 -->`.

Everything else is dropped: the session header, PASSED lines, frames outside the project, the slowest-durations section and blank lines.

### 4.4 Referenced setup lines

This is the scheduler fix. It is deterministic, uses only Python's `tokenize`, and never runs code.

Within the failing test's frame, the source lines between `def test_...` (exclusive) and the first `>` line are its **body**.

1. **Wanted names:** the NAME tokens on the `>` line, including continuation lines. Exclude Python keywords and builtins (`len`, `range`, `isinstance` and so on).
2. **Walk the body upward from the `>` line.** Keep a line if either:
   - (a) it **binds** a wanted name: assignment target including tuple unpacking, augmented assignment, `for NAME in`, `with ... as NAME`, or `import`; or
   - (b) it is an **expression statement on a wanted object**: its first NAME token is wanted, as in `sched.enqueue(...)`.
3. When a binding line is kept, add the names on its right-hand side to the wanted set. Lines further up that bind them are then caught too (`sched = ContinuousBatchScheduler(...)`).
4. When a kept line is indented deeper than the body's first line, also keep the block header that opens it (`for i in range(5):`).
5. Keep at most **8** lines per failure. Prefer the lines closest to `>`, and output them in original order with original indentation.
6. Never keep the `def test_...` line. The separator line already names the test.
7. **Fallback:** if tokenizing fails or no body is found, keep the 3 source lines directly above `>`. Never raise an exception.

Worked example: the real `test_scheduler_drain_all_flushes_partial_remainder_when_true` block. The `>` line is `assert len(batches) == 2`.

- The wanted set starts as `{batches}`.
- `batches = sched.drain_all_batches(flush_partial=True)` is kept, which adds `sched` to the wanted set.
- `sched.enqueue(...)` is kept, along with its `for i in range(5):` header.
- `sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)` is kept.

The result is the whole short body, and `flush_partial=True` is visible.

### 4.5 Long lines

Any kept line longer than 600 chars becomes its first 290 chars, then `…[cf: N chars clipped]…`, then its last 290 chars. This keeps the end of long values: the `req-3`/`req-4` batch in the scheduler case.

### 4.6 Stale pytest digest

Replace the whole output with:

```
[cf: earlier test run, superseded by a later run of the same tests]
<final "=== N failed, M passed in Xs ===" line, verbatim>
<the FAILED lines from the short test summary, verbatim, at most 20; then "[cf: K more failures]" if needed>
<!-- cf:v2 -->
```

### 4.7 Non-pytest output

Use v1's JSON and text-log handling unchanged, except that every kept line follows 4.5 and the v2 marker is appended. No fresh/stale handling for these outputs in v2.

### 4.8 Configuration and visibility

- `RUNSIFT_CONDENSER` (or `CACHEFORGE_CONDENSER`) takes `v2` (the default) or `v1`. It is read in `src/core/config.py`.
- The gateway adds response headers:
  - `x-runsift-condenser: v1|v2` (along with backward-compatible `x-cacheforge-condenser`);
  - `x-runsift-tool-chars-before`, the total chars of condensable tool content in the request;
  - `x-runsift-tool-chars-after`.

## 5. Harness instrumentation

This is needed to evaluate v2. It applies to both arms.

Add fields; keep every existing field unchanged.

1. **Per session:** `wall_clock_s`, from the first model request to the start of the final evaluation.
2. **Per turn:**
   - `per_turn_model_ms`: from request sent to stream end, including generation, not just time to first token;
   - `per_turn_tool_ms`: tool execution time;
   - `per_turn_tool_args`: the full JSON arguments of the tool call. Edits must be recoverable.
3. **Sidecar file per repeat:** `<result>_rep<N>_tool_outputs.jsonl.gz`. It holds one record per turn per arm, `{arm, turn, tool, args, raw_output}`, with the full raw tool output. v2 is deterministic, so this lets anyone recompute exactly what the RunSift model saw. The sidecar goes in the results directory, which the dirty check already ignores.
4. **For the RunSift arm:** record the `x-runsift-condenser` (or `x-cacheforge-condenser`) and `x-runsift-tool-chars-*` headers on every turn.
5. **Analysis script:** extend `analyze_gauntlet_results.py` to report `wall_clock_s`, model and tool time, `read_file` calls per session, and sessions with any failing `tests/test_scheduler.py` test in the final evaluation. Leave the existing outputs unchanged.

## 6. Hygiene

- Commit before any run, and don't use `--no-verify`. Gates: `pytest -q` and `ruff check .` / `ruff format --check .`, both under `.venv\Scripts\python.exe`, plus `sdcs verify --topology`, `sdcs verify --mocks` and `sdcs map --check`.
- No new mocks. If simulated behavior seems needed, stop and ask.
- Don't edit `wiring.yaml` or `spine.md` without approval.

## 7. Tests

New file: `tests/test_condenser_v2.py`.

1. **Determinism (I4):** the same message list condensed 100 times gives identical bytes.
2. **Idempotence (I5):** `condense(condense(x)) == condense(x)` for every fixture.
3. **Untouched roles (I1):** system, user and assistant messages are byte-identical after a pass. Include instruction-like text such as "Never delete files without asking" and "Do not touch tests/".
4. **Exempt tools (I2):** `read_file` and `grep` outputs above the threshold are byte-identical.
5. **Scheduler fixture (4.4, 4.5):** build a real log. Replay the RunSift arm's `patch_diff` from `nebius_gauntlet_nemotron_30b_40turn_v3_rerun1_rep8.json` onto a fresh wave-1 workspace (the rescorer already does this), then run pytest with the harness flags. Save the log under `tests/fixtures/`. Assert:
   - the digest contains the test's `drain_all_batches(flush_partial=...)` line, using the exact line from the fixture;
   - it contains the `sched = ...` line and the `enqueue` loop;
   - it contains `req-4` (the clipped tail);
   - it does not contain `def test_`.
6. **Transitive names:** a synthetic test body where the assertion uses `b`, `b = f(a)`, and `a = A()`. Both binding lines are kept, and an unrelated `c = 3` is not.
7. **Fresh and stale (4.2, 4.6):**
   - two full-suite `run_pytest` outputs: the first becomes the stale form, with its final summary line verbatim, and the second is fresh;
   - a targeted run doesn't make a full-suite run stale;
   - the same message list gives identical bytes on every pass.
8. **Fallback (4.4.7):** a malformed traceback returns the 3-line fallback without an exception.
9. **Size:** the fresh digest of the initial 45-test log is no more than 25% of the raw chars. Record the real ratio in the test output.
10. **v1 golden (I7):** with `RUNSIFT_CONDENSER=v1` (or legacy `CACHEFORGE_CONDENSER=v1`), the existing fixtures produce exactly the bytes of commit `2d64007`. Generate the golden files from the unchanged v1 code as the very first step, before editing any file in `src/compact/`, and commit them on their own.
11. **Verbatim check (I3):** every line of every digest is either an exact line, or the exact head and tail of a line, from the raw input, or it starts with `[cf:` or is the marker.
12. **Coupled fixtures (the decisive mechanism):** build the starting coupled log (`materialize_clean_repo_coupled`, `inject_all_coupled_bugs`, pytest with the harness flags) and save it under `tests/fixtures/`. Assert, using exact lines from the fixture:
    - the digest for `test_pipeline_routes_to_warm_worker` contains `pipeline = build_pipeline(chunk_size=4)`, the `pipeline.tree.insert_prefix(...)` line, and `worker = pipeline.route(...)`;
    - the digest for `test_radix_match_counts_tokens` contains `tree = PrefixRadixTree(...)`, `tree.insert_prefix(...)`, and `worker, matched = tree.match_prefix(...)`. This checks that tuple unpacking binds `matched`;
    - neither digest contains a `def test_` line;
    - the whole fresh digest of this log is no more than 30% of the raw chars. Record the actual ratio.

## 8. Pre-registration (draft for PRE_REGISTRATION.md section 9)

Write it before any counted run, in a commit made after the code and harness work.

### 8.1 Questions

1. **Decisive:** on the coupled task where v1 failed, does v2 bring Nemotron's trap-solving back to the no-RunSift baseline while still cutting tokens?
2. **Platform:** does the same hold on Nebius Token Factory, the hackathon's required platform?

### 8.2 Run A, decisive: local Nemotron, coupled task

- **Setup:** identical to the v1 coupled run in pre-registration section 7, so v1's 37-vs-13 result is directly comparable:
  - llama.cpp Nemotron 3 Nano 30B with `-c 131072 -np 1 -ngl 99 --n-cpu-moe 40 --load-mode none -fa on --cache-reuse 64 --reasoning off`, on port 8082;
  - the coupled task set, 40 turns, temperature 0.6;
  - the gateway with `--timeout-keep-alive 120`.
- **Arms:** baseline vs RunSift with `RUNSIFT_CONDENSER=v2`.
- **Size:** 15 paired repeats, in 3 batches of 5, with arm order alternating by repeat. Log GPU health and stop at 85 C or on thermal slowdown, as before.
- **Primary metric:** traps solved per session, with guards included. Paired exact permutation test, two-sided, α = 0.05.
- **Decision rule, fixed in advance:** we may say "v2 closes the v1 gap" only if both of these hold:
  - the v2-vs-baseline test is **not** significant (p ≥ 0.05);
  - v2's total traps solved is at least **80%** of the baseline's total in this run.
  
  Otherwise we report what happened and make no recovery claim. v1's result (13 vs 37) is shown alongside for comparison; it is not part of a test.

### 8.3 Run B, platform: Token Factory, coupled task

- **Setup:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Token Factory at its full context, with reasoning off via `chat_template_kwargs` in both arms. Coupled task set, 40 turns, temperature 0.6, 15 pairs in batches of 5.
- **Arms and metrics:** the same as Run A.
- **Commitment:** Run B is done **whatever Run A shows**, to avoid selective reporting. Its pre-registration is in the same commit as Run A's.
- **Budget:** about 2.5M prompt tokens per pair, based on the v1 coupled volumes. 15 pairs comes to about $2–3 at list price. Check the credit balance before batch 1.

### 8.4 Optional Run C

Token Factory, original task set, baseline vs v2, 15 pairs. It confirms that v2 keeps v1's tie there with similar savings. Decide whether to run it **before** Run A starts, and write the decision down. It may not be added after seeing results.

### 8.5 Secondary metrics

These are the same for every run, reported whatever they show, with each delta's sign:

1. Prompt tokens per session, per turn and per trap solved; completion tokens per session.
2. `wall_clock_s` per session (median and mean), with model time and tool time.
3. Calls per session of `run_pytest`, `search_code` and `read_file`. These are v1's failure signature: fewer test runs and more searching.
4. Sessions that never got more tests passing than at the start (v1 Nemotron: 11/15).
5. Regressions introduced, regressions repaired, final pass-to-pass intact, and overflows.
6. Cached-token share. For Run B: cost per session at list price, and under a hypothetical cached-input price of 1/10.
7. Infra errors, with which arm each hit.

### 8.6 Rules

These are the same as pre-registration sections 7 and 8, as amended:

- One frozen commit with `git_dirty: false`. Health-only checks between batches.
- **Infra errors** (connection loss, server down, 5xx after retries) get re-run under the same repeat number into `_rerun<K>`, at most 3 per run.
- **Genuine context overflows,** including llama.cpp's "exceeds the available context size", are model behavior. Score them; never re-run them.
- **Stuck-agent repeats** stay in the primary analysis.
- **401, 402 or 403** aborts the run.
- **Scoring check:** `analyze_gauntlet_results.py` produces the primary analysis. Also spot-check 3 randomly chosen sessions per arm by patch replay with the committed coupled rescorer, and report whether they match.

## 9. Order of work

0. **Prerequisite:** the coupled bookkeeping is committed and reviewed. That means:
   - the 30 result files and GPU logs;
   - the `classify_upstream_error` fix for llama.cpp overflows, with its test;
   - the committed coupled rescorer;
   - `local_coupled_nemotron_30b_40turn_rep15_rescored.json`;
   - the section 7 amendment.
1. Commit this spec as `docs/specs/CONDENSER_V2_SPEC.md`.
2. Generate the v1 golden files from the unchanged code (test 10), and commit them.
3. Implement condenser v2, the config flag and the gateway headers, with tests 1–12. Run the gates, then commit.
4. Implement the harness instrumentation from section 5, with tests. Run the gates, then commit.
5. **Local smoke test:** Nemotron 131k, coupled task, 1 repeat, both arms, 10 turns. Report:
   - that every RunSift turn carries `x-runsift-condenser: v2` (or `x-cacheforge-condenser: v2`);
   - before/after char totals;
   - that the sidecar file exists, and that recomputing v2 from it matches the recorded "after" sizes;
   - that the timing fields are present and plausible.
   
   Stop for review.
6. After approval, commit pre-registration section 9 (Runs A, B and the Run C decision). Then run **Run A**'s counted batches.
7. Report Run A's verbatim analysis and stop. The reviewer checks it independently.
8. **Token Factory smoke test:** 1 repeat, both arms, 10 turns, coupled task. Report the same items plus cached tokens and the key-leak count. Then run **Run B**'s counted batches, and **Run C** if it was chosen in step 6.
9. Report verbatim and stop.

**Context budget for the agent:** send each batch's harness output to a log file in `results/` and never read the full log into the conversation. Check progress at most every 20–30 minutes, using only the new result files, the last GPU CSV line and the last 3 log lines. Stop and report if your own output starts repeating.
