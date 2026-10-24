# Turn-2 Probe: Pre-Registration Addendum (PRE_REGISTRATION.md section 12)

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
- **Integrity check (before counting):** the gateway's condensed tool message must equal the one recomputed from rep 8's sidecar byte for byte, which is 9,662 characters (`x-runsift-tool-chars-after`). If it doesn't match, stop and report.
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
