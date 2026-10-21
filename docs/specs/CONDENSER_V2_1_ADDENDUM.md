# RunSift Condenser v2.1: Addendum to CONDENSER_V2_SPEC.md

**Status:** draft. Implement only after Runs B and C (pre-registration section 9) are finished, committed and reviewed. Don't add this file to the repo while B or C is running: any new file outside `results/` marks their batches as dirty.
**Save in the repo as:** `docs/specs/CONDENSER_V2_1_ADDENDUM.md`, in the first commit of the v2.1 work.
**Written:** 2026-10-17. Every number below comes from committed Run A files (`local_coupled_nemotron_30b_40turn_v2_*`, commit `4661dd5`) and was recomputed independently. Numbers labeled as replay estimates come from an offline replay of Run A's sidecars, not from a run.

---

## 1. Why v2.1

**Run A** (local Nemotron 3 Nano 30B, 131k context, coupled task, 15 pairs, baseline vs v2) met v2's pre-registered quality rule. It solved 32 vs 39 traps (p = 0.203), which is 82% of the baseline. v1 managed 13 vs 37 in the same setup. But v2 is **slower than running without RunSift:**

| Per session | Baseline | v1 (section 7 run) | v2 (Run A) |
|---|---|---|---|
| Prompt tokens | 1,711,391 | 566,640 | 543,367 |
| Share of prompt tokens served from cache | 95.4% | 96.0% | **78.7%** |
| Uncached prompt tokens | 78,952 | 22,598 | **115,622** |
| Median time to first token | 3.3 s | 1.3 s | **4.8 s** |
| Median wall-clock | 229 s | (not recorded) | **331 s** |

The v1 column comes from the section 7 run, which recorded no wall-clock time.

**Mechanism** (verified on rep9 by replaying the sidecar through `CondenserV2`): v2 rewrites the previous pytest digest from fresh (about 6,150 chars) to stale (about 1,020 chars) whenever a newer run with the same file set arrives. That rewrite changes a message in the middle of the history. A prefix cache then has to re-read everything from that message onward: 57–68% of the prompt by mid-session (turns 10 and 14 of rep9).

| Median first-token wait on the turn after a test run | Baseline | v1 | v2 |
|---|---|---|---|
| Wait | 10.0 s | 2.2 s | **15.5 s** |

The baseline waits on reading the new raw log. v1 never rewrites history. v2 re-reads most of the prompt.

**Replay estimate** (Run A sidecars, all 15 RunSift sessions, characters not tokens): removing the stale rewrite sends **+31% characters** compared with v2. That is still well below the baseline. It also cuts cache-miss characters by **35%**, and turns after a test run stop re-reading history. This is an estimate, not a result. The run below measures it.

**What v2.1 does not address:** `scheduler`-failed sessions were 6/15 for v2 vs 2/15 for the baseline in Run A. That's a separate, open finding.

## 2. The change

v2.1 is v2 with the stale rewrite removed. Everything else stays exactly as in v2:
- referenced setup lines;
- head+tail clipping;
- verbatim extraction;
- exemptions;
- determinism;
- idempotence.

**New invariant I8, history is never rewritten.** The condensed form of a tool message is a pure function of that message's own content and its own tool-call arguments. It never depends on any other message in the request. So for consecutive requests in a session, every message that existed in request *t* is byte-identical in request *t+1*, and the condensed request *t* is a message-level prefix of the condensed request *t+1*. Prefix caches then only process what's new.

**Implementation notes:**
- Add `RUNSIFT_CONDENSER=v2.1` (or legacy `CACHEFORGE_CONDENSER=v2.1`) as a third value. `v1` and `v2` remain selectable and byte-identical to their current output. The default stays `v2` until v2.1 has its own result; choosing the shipped default is section 6.
- In v2.1 mode, every pytest output gets v2's **fresh** digest, permanently. The run-key and freshness logic doesn't run in v2.1 mode.
- Gateway header: `x-runsift-condenser: v2.1` (along with `x-cacheforge-condenser: v2.1`).
- Growth check: v2 fresh digests in Run A were about 6k chars each, with about 9 test runs per session. Keeping every one adds tens of thousands of chars by turn 40, well inside a 131k window. No cap in v2.1; record peak prompt tokens.

## 3. Tests

These go in `tests/test_condenser_v2.py`, or a new `tests/test_condenser_v2_1.py`.

13. **Prefix stability (I8), end to end.** Replay a real Run A RunSift session from its sidecar (`local_coupled_nemotron_30b_40turn_v2_rep9_tool_outputs.jsonl.gz`) turn by turn. Push each request through the **full gateway transformation** (v2.1 condensing, then the canonicalizer, as `gateway.py` applies them). Assert that every message of condensed request *t* is byte-identical in condensed request *t+1* for all 40 turns. Run the same replay through v2 and assert it does **not** hold, to prove the test can detect the problem.
14. **v2 unchanged.** Before editing anything, generate v2 golden outputs from the current code for the existing fixtures and for the rep9 replay, and commit them on their own. Then assert that v2 mode reproduces them byte for byte. Also re-run the existing v1 golden test.
15. **Fresh digest equality.** For any single pytest message, the v2.1 digest equals v2's fresh digest of the same message.
16. **The v2 invariants in v2.1 mode:** determinism, idempotence, verbatim lines (I3), untouched roles, exempt tools, and the coupled and scheduler fixtures from tests 5, 11 and 12.

## 4. Pre-registration (draft for PRE_REGISTRATION.md section 10)

Write it before any counted run, in a commit made after the code.

### 4.1 Run D, decisive: local Nemotron, coupled task, baseline vs v2.1

- **Setup:** identical to Run A, so Run A's v2 numbers can sit beside it. That means llama.cpp Nemotron 3 Nano 30B with `-c 131072 -np 1 -ngl 99 --n-cpu-moe 40 --load-mode none -fa on --cache-reuse 64 --reasoning off` on port 8082; the coupled task set, 40 turns, temperature 0.6; the gateway with `--timeout-keep-alive 120` and `RUNSIFT_CONDENSER=v2.1`; 15 paired repeats in batches of 5 with alternating arm order; GPU logging and the same hard stops.
- **Quality, decision rule fixed in advance:** the metric is traps solved per session, using a paired exact permutation test, two-sided, α = 0.05. We may say "v2.1 keeps quality" only if p ≥ 0.05 **and** v2.1's total traps are at least 80% of the baseline total in this run.
- **Speed, decision rule fixed in advance:** the metric is `wall_clock_s` per session, using a paired exact permutation test on the per-repeat difference (v2.1 minus baseline), two-sided, α = 0.05.
  - We may say "v2.1 is faster than no RunSift" only if the mean difference is negative and p < 0.05.
  - We may say "no measurable slowdown" only if p ≥ 0.05.
  - If the difference is positive and p < 0.05, we report a slowdown.
  - These are two separate claims, each tested at 0.05. Both p-values are reported whatever they show.
- **Comparison with Run A** (descriptive only, because it was run at a different time): v2.1's cached share, uncached tokens, time to first token and wall-clock, next to Run A's v2 numbers. The two runs' baseline arms are a stability check. Report how much they differ.

### 4.2 Run E, platform: Token Factory, coupled task, baseline vs v2.1

Same metrics and rules as Run D, on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` at full context, with reasoning off in both arms. **Decide yes or no in the section 10 commit, before Run D starts, and write the decision down.** The decision may take Run B's committed results into account, because they are known before Run D starts. It may not depend on Run D. Budget: about $2–3 at list price.

### 4.3 Secondary metrics for every run

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

### 4.4 Rules

Unchanged from sections 7–9:
- one frozen commit, `git_dirty: false`;
- health-only checks between batches;
- infra errors re-run into `_rerun<K>`, at most 3;
- genuine overflows are scored, never re-run;
- stuck-agent repeats stay in;
- 401, 402 or 403 aborts;
- the analysis comes from `analyze_gauntlet_results.py`, plus a 3-per-arm patch-replay spot check.

**New rule:** every batch log is saved in `results/` and committed. Only numbers that appear in committed files may be reported.

## 5. Order of work

0. **Prerequisite:** Runs B and C finished, committed and reviewed.
1. Commit this addendum unchanged as `docs/specs/CONDENSER_V2_1_ADDENDUM.md`.
2. Generate the v2 golden outputs from the unchanged code (test 14) and commit them on their own.
3. Implement v2.1, the config value and the header, with tests 13–16. Run the gates, then commit.
4. **Local smoke test:** Nemotron 131k, coupled task, 1 repeat, both arms, 10 turns. Report:
   - `x-runsift-condenser: v2.1` (or `x-cacheforge-condenser: v2.1`) on every RunSift turn;
   - that the I8 prefix property holds when you replay this session's sidecar through the full gateway pipeline;
   - the RunSift arm's first-token wait on the turns right after a test run, next to the baseline's.

   Stop for review.
5. After approval, commit pre-registration section 10, including the Run E decision. Then run **Run D**. Commit the results and batch logs, report verbatim, and stop.
6. After review: run **Run E** if it was chosen, under the same procedure.

## 6. Which version ships (decision rule fixed now)

The default for the submission is the version with the strongest pre-registered evidence:

- **v2.1** becomes the default if Run D meets its quality rule **and** its speed result is either "faster" or "no measurable slowdown".
- **v2** stays the default if v2.1 fails the quality rule.
- If both fail, the write-up says so plainly. The default is whichever version the evidence supports, and every version's results are reported.

The README and Devpost cite the runs behind whichever version ships.
