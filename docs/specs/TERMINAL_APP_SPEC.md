# RunSift Terminal App: Design

**Status:** design, written 2026-10-18. Start only after the shell-tests work (`CONDENSER_V2_SHELL_TESTS_ADDENDUM.md`) is committed and reviewed. Commit this file alone, as the first commit of this work.
**Owner decisions (2026-10-18):**
- this app replaces the planned dashboard-tile and no-key demo-script items;
- the owner approves the new `tui` subsystem in `wiring.yaml` (section 8) and the Textual dependency (section 9).

**Measured on:** the committed result files and sidecars at `66959a4`. The header-reproduction counts in T2 come from a scratch replay of all six replayable runs.

---

## 1. Purpose

The app has three jobs:
1. **Let anyone see the product work in about 10 seconds,** with no API key, no network and no login. It replays recorded Token Factory sessions through the real condenser.
2. **Let a developer running the gateway watch their own traffic** in the terminal.
3. **Give the demo video clear footage of both.**

The web dashboard stays as it is.

## 2. Commands

| Command | Needs | What it does |
|---|---|---|
| `python -m src.tui replay` | nothing (no key, no network) | Browse every pre-registered Token Factory run and step through recorded sessions |
| `python -m src.tui watch [--url http://127.0.0.1:8765]` | a running gateway | Live view of requests passing through the gateway |
| (gateway terminal) | a running gateway | One readable line per request, printed by the gateway itself (section 6) |

**Use `python -m src.tui`, which matches the existing `uvicorn src.proxy:app`.** setuptools' automatic package discovery doesn't install a top-level `src` package, so a console script isn't reliable. Add one only if it works from a fresh clone in a fresh virtual environment, and say so in the report.

## 3. Data (read-only)

The app reads `benchmarks/tensorroute_benchmark/results/`. It finds the directory relative to the repo root (the parent of `src/`). `--results-dir` overrides it.

| Run | Task | Condenser | File prefix | Replayable |
|---|---|---|---|---|
| v3 | standard | v1 | `nebius_gauntlet_nemotron_30b_40turn_v3` (score from `_rescored`) | no: no recorded tool outputs |
| B | hard | v2 | `nebius_coupled_nemotron_30b_40turn_v2` | yes |
| C | standard | v2 | `nebius_gauntlet_nemotron_30b_40turn_v2` | yes |
| E | hard | v2.1 | `nebius_coupled_nemotron_30b_40turn_v2_1` | yes |
| F | standard | v2.1 | `nebius_gauntlet_nemotron_30b_40turn_v2_1` (score from `_rescored`) | yes |
| G | standard | v2 | `nebius_gauntlet_nemotron_30b_40turn_runG_v2` (score from `_rescored`) | yes |
| G | standard | v2.1 | `nebius_gauntlet_nemotron_30b_40turn_runG_v2_1` (score from `_rescored`; has the RunSift arm only) | yes |

- **Per session,** `{prefix}_rep{N}.json` holds both arms. Use `baseline` and `runsift` (or legacy `cacheforge`), then `task_results[0]`:
  - outcome: `modules_fixed` (standard task) or `traps_solved` (hard task), `context_overflow`;
  - totals: `total_prompt_tokens`, `total_completion_tokens`, `peak_turn_prompt_tokens`, `turns`, `wall_clock_s`;
  - per turn: `per_turn_prompt_tokens`, `per_turn_tools`, `per_turn_tool_args`, `per_turn_tests_passing`, `per_turn_condenser`, `per_turn_tool_chars_before`, `per_turn_tool_chars_after`;
  - provenance: `provenance.git_commit`, `git_dirty`, `launch_timestamp`.
- **Scores:** when a `_rescored` file exists, the score comes from it and is labeled "(rescored)". Everything else comes from the original file.
- **Run G's baseline** comes from the `runG_v2` files.
- **For v3,** pick files exactly as `analyze_gauntlet_results.py nebius_gauntlet_nemotron_30b_40turn_v3_rescored` does. Repeat 8 was replaced by the `_rerun1` file.
- **Raw tool outputs** come from `{prefix}_rep{N}_tool_outputs.jsonl.gz`. Each record has `arm`, `turn`, `tool`, `args` and `raw_output`.
- **Cost** is list price, $0.06 per million input tokens and $0.24 per million output tokens. Show it with 3 decimals per session.

## 4. Replay screens

### 4.1 Runs (the first screen)

- **Rows:** the seven rows of the README results table. For each row:
  - run, task and condenser;
  - prompt tokens per session, cost per session (without → with) and sessions per dollar;
  - fixed (without → with), with the two-sided exact paired permutation p-value, computed in the app over the 15 pairs.
- **Every row is listed,** including Run F's loss.
- **Footer:** "Every number here is computed from the committed result files. Costs are at list price."
- **No session is pre-selected.**

### 4.2 Sessions (after picking a run)

- **One row per repeat (15),** each showing:
  - without and with: fixed, cost, turns and largest request;
  - whether either arm ran out of context.
- **Header:** the run's commit and launch date.

### 4.3 Session (after picking a repeat)

**Header:** run, repeat, task, condenser, commit, date.

**Two summary panels** side by side (without RunSift / with RunSift):
- bugs fixed, or traps solved on the hard task;
- turns;
- prompt tokens;
- cost;
- largest request;
- wall-clock time;
- ran out of context (yes/no).

**Request-size chart:** `per_turn_prompt_tokens` for both arms on the same scale, labeled with each arm's peak. For example, in Run C repeat 3 the baseline peaks at 185,961 tokens and RunSift at 20,723.

**Turn stepper,** for one arm at a time (default: with RunSift). It shows turn *k* of *N*, the tool and its arguments, and the tests passing after the turn.
- **Left pane, "What the tool returned":** the raw output from the sidecar.
- **Right pane, "What the model saw":**
  - with RunSift: the output condensed now by `CondenserV2` (with `enable_stale_rewrite` true for v2, false for v2.1), as it was first sent (the fresh digest);
  - without RunSift: "Sent unchanged (no RunSift)".
- **Status line:** "This tool output: X → Y chars (−Z%)".
- **Badge:** compare the next request's condensed tool characters, recomputed with `compact_messages` over the history rebuilt from the sidecar, with the recorded `per_turn_tool_chars_before/after` from the gateway's headers. Show one of:
  - "✓ matches what the gateway sent";
  - "≠ differs: the condenser changed after this run (timeout fix, 78df5c0)", when the history contains a `[TIMEOUT]` log and the run's commit predates `78df5c0`;
  - "≠ differs", for anything else. Never hidden.
- **Rebuilding the history:** each sidecar record becomes an assistant message with one tool call (name = `tool`, arguments = `json.dumps(args)`) plus a tool message (`content` = `raw_output`), in turn order. Records whose tool is `none`, `overflow`, `infra_error` or `error` add nothing.

**Keys:** ←/→ turn, ↑/↓ and PgUp/PgDn scroll the focused pane, Tab switches panes, `a` switches arm, Esc goes back, `?` opens help, `q` quits.

**Help** includes: "One session is an example, not a result. The Runs screen has the results."

## 5. Watch mode

- **Polling:** it polls `GET /api/telemetry` once a second. If `RUNSIFT_API_KEY` is set, it sends it as a Bearer token.
- **Live table** of recent requests. Columns:
  - time, model and condenser;
  - tool output (before → after chars);
  - prompt, cached and completion tokens;
  - time to first token;
  - estimated list cost (Nano only; other models show "—").
- **Running totals:**
  - requests and prompt tokens sent;
  - tool characters removed;
  - "estimated input cost avoided (list price, chars ÷ 4)".
- **Upstream status:** shows the `upstream_status` field as it is (CONFIGURED or UNCONFIGURED).
- **When the gateway can't be reached:** show a clear message and keep retrying. Don't crash.
- **Telemetry change this needs:** `record_request_telemetry` gains optional `tool_chars_before`, `tool_chars_after`, `condenser` and `fallback`. The gateway passes them on both the streaming and non-streaming paths, and they appear in `recent_requests`. Existing fields and tests are unchanged.

## 6. Gateway request lines

- **What prints:** one line per completed request, on stderr, through logger `runsift.requests` at INFO.
- **Handler:** `create_app` adds a handler unless `RUNSIFT_REQUEST_LOG=off`, and never adds a second one when the app is created again (the tests create many apps).
- **Format,** for example from a real request on 2026-10-18:
  ```
  req-02  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B  v2  tool output 39,855 → 9,594 chars (-76%)  prompt 3,288  cached 0  first token 432 ms
  ```
- **Variants:** a fallback request ends with `FALLBACK: condenser error, request sent unchanged`. An upstream error prints `upstream HTTP <code>`.
- **Never logged:** message content, keys or headers.

## 7. Honesty rules (from AGENTS.md section 3)

- Every number is read from a result file or computed live from committed data by product code. Estimates say "estimated".
- The outcome (fixed or traps) is shown next to every cost or token figure. Deltas carry their sign.
- All runs and all repeats are listed. The app opens on the Runs screen, not on a chosen session.
- There is no simulated or demo data mode.

## 8. `wiring.yaml` (owner-approved)

Add:
```yaml
  tui:
    path: "src/tui"
    responsibilities: "Terminal app: replays recorded benchmark sessions through the condenser without an API key, and shows live gateway traffic."
    allowed_dependencies: ["core", "compact"]
```
The app must not import `proxy`, `upstream`, `state` or `canonicalizer`. It reads files from disk and talks to the gateway only over HTTP (`httpx`).

## 9. Dependency (owner-approved)

- Add `textual` to `[project] dependencies`, with a lower bound equal to the version you test with, and update `uv.lock`. Add nothing else.
- Check it in Windows Terminal and in a plain PowerShell window, at 120×30 and 80×24.

## 10. Acceptance checks

- **T1. Runs table:** the app's table model equals the README results table for all seven rows. The test parses the README table, so no value is typed into it by hand: prompt-token change, cost per session, sessions per dollar, fixed counts and p-values, at README's precision.
- **T2. Header reproduction:** across the six replayable runs, today's condenser reproduces **3,185 of 3,239** recorded requests exactly.
  - All **54** differences contain a `[TIMEOUT]` log, in runs recorded before `78df5c0`: B 8 of 600, E 19 of 600, F 27 of 564.
  - C 486 of 486, G v2 459 of 459 and G v2.1 530 of 530 match.
  - The badge logic must give exactly these counts. Implement it as a script (`benchmarks/tensorroute_benchmark/scripts/verify_replay_headers.py`) plus a test on a fixed sample of sessions.
- **T3. Replay needs nothing:** it works with no `NEBIUS_API_KEY` and makes no network calls. In the test, make any `httpx` call raise.
- **T4. Watch:** unit tests with a fake telemetry payload. When the gateway is unreachable, it shows the message and doesn't crash.
- **T5. Telemetry fields:** the new fields appear in `recent_requests` on both the streaming and non-streaming paths, and all existing tests pass unchanged.
- **T6. Request lines:**
  - exactly one line per request;
  - no content or keys;
  - `RUNSIFT_REQUEST_LOG=off` silences them;
  - no duplicate handlers.
- **T7. UI:** Textual `run_test()` pilot tests:
  - the Runs screen loads with seven rows;
  - Enter opens a run, then a session;
  - → steps a turn;
  - `a` switches arm;
  - Esc goes back.
- **T8. Gates:** all seven gates pass.

## 11. README

Add a section after "Quickstart":

> ## Try it without a key
>
> ```bash
> python -m src.tui replay
> ```
>
> Browse every pre-registered Token Factory run, then step through a recorded session. You see what the agent's tools returned and what RunSift sent to Nemotron, next to the paired session without RunSift. The condensed side is recomputed on your machine by the same code, and each turn shows whether it matches what the gateway sent during the run. With the gateway running, `python -m src.tui watch` shows your own traffic live.

**Optional verification line:** use only the counts printed by the T2 script:
> Today's condenser reproduces 3,185 of the 3,239 recorded requests exactly. The other 54 contain a test-run timeout, which the condenser has handled differently since the timeout fix.

## 12. Out of scope

- changes to result files;
- benchmark runs and Token Factory calls;
- changes to the web dashboard;
- the local runs (A, D).

## 13. Order of work and cut line

Commit each step separately, and pass the gates at each step.
1. `wiring.yaml`, the dependency and an empty `src/tui` package.
2. The data layer: runs, sessions, turns and header verification, with T1–T3. This is the core.
3. The replay UI, with T7.
4. Telemetry fields and gateway request lines, with T5 and T6.
5. Watch mode, with T4.
6. README section and `app_map.md`.

**Cut line:** if step 5 isn't done by the end of Oct 25, ship steps 1–4 and 6, and leave watch mode out. The code freeze is Oct 26.
