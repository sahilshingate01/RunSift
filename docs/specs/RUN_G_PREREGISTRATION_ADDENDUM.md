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
  - the v2 arm carries `x-runsift-condenser: v2` on 100% of turns;
  - the v2.1 arm carries `x-runsift-condenser: v2.1` on 100% of turns;
  - every sidecar recomputes exactly;
  - no API key appears in any file.
- **Infra errors** are re-run under the same repeat number into `_rerun<K>`, at most 3 per arm. Genuine overflows and hung test suites are scored, never re-run. Stuck-agent repeats stay in. 401, 402 or 403 aborts the run.
- **Scoring:**
  - before its final evaluation, the harness restores edited files under `tests/` and removes new ones, recording both in `test_files_edited` (1b31b50, 9ffaee1);
  - then every one of the 45 sessions is replayed with `rescore_v3_results.py`, and the analysis uses the `_rescored` files;
  - any session whose replay differs from the harness score is listed with the reason.
- **Logs and citations:** every batch log is saved in `results/` and committed (logs are git-ignored, so add them with `git add -f`). Only numbers that appear in committed files are reported.

**Budget:** about $0.19 per repeat at list price ($0.12 baseline, about $0.03 for v2, about $0.04 for v2.1, from Runs C and F). That's about $3 for 15 repeats plus the smoke test, and roughly 1 to 1.5 hours. Check the credit balance before the smoke test.
