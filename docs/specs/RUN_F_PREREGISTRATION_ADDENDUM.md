# Run F: Pre-Registration Addendum (draft for PRE_REGISTRATION.md section 11)

**Status:** draft. It is committed as section 11 before Run F starts. Run F starts on that commit.
**Save in the repo as:** `docs/specs/RUN_F_PREREGISTRATION_ADDENDUM.md`, committed unchanged, in the same commit as section 11.
**Written:** 2026-10-18.

---

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
