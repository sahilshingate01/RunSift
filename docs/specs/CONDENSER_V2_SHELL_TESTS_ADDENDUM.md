# RunSift Condenser v2: Addendum for Test Output from Shell Tools

**Status:** design, written 2026-10-18. The owner chose this work; implement it as written, and raise any rule that turns out to be unworkable before changing it.
**Save in the repo as:** `docs/specs/CONDENSER_V2_SHELL_TESTS_ADDENDUM.md`, committed on its own before any code.
**Measured on:** the 126 committed `*_tool_outputs.jsonl.gz` sidecars at `66959a4`, and on pytest output captured from the 5-bug gauntlet workspace (`materialize_clean_repo` + `inject_gauntlet_bugs(BENCHMARK_TASKS[0:5])`; pytest 9.1.1, Python 3.12.8, Windows).

---

## 1. The problem

The pytest digest (spec sections 4.3 and 4.6) runs only when the tool is named `run_pytest` or `pytest` (`synthesizer.py:147–149`). That is the benchmark harness's tool. Real coding agents run tests through a general shell tool: OpenCode's `bash`, OpenHands' `execute_bash`, Goose's `developer__shell`. Their test logs get the generic text-log summary (section 4.7): up to 6 failure-pattern lines plus the first 12 and last 16 lines. That summary keeps the failing test names but loses the setup lines and most assertion lines, and it has never been benchmarked.

Forcing today's digest onto other pytest formats isn't safe either. Same workspace, 45 tests, 12 failing:

| pytest flags | Raw chars | Today, as `bash` (generic) | Today's digest, forced | What goes wrong when forced |
|---|---|---|---|---|
| `-vv -ra --tb=long` (harness) | 40,091 | 4,072 | 9,594 | nothing |
| none | 16,015 | 2,781 | 8,781 | per-file counts wrong: "0 passed, 1 failed" for files with 7 passed, 2 failed |
| `-q` | 15,168 | 2,619 | 8,843 | says the run did not finish; counts wrong |
| `-x -q` | 2,466 | 2,466 (not condensed) | 1,098 | says the run did not finish |
| `-q --tb=line` | 5,320 | 3,191 | 1,478 | says the run did not finish |
| a missing test file | 611 | 611 (not condensed) | 140 | drops `ERROR: file or directory not found: …` and says the run did not finish |

Causes:
1. The final-summary check requires a line wrapped in `=`, but `-q` prints a plain one (`12 failed, 33 passed in 0.22s`).
2. `no tests ran in …` isn't recognized as a summary.
3. Per-file counts come only from verbose result lines. Otherwise each file with a `FAILED` line is reported as "0 passed, 1 failed".
4. Everything outside the parsed pytest sections is dropped, including lines the shell or the agent's tool adds.

## 2. Goal and hard requirements

**Goal:** test output from a shell tool gets the digest the benchmarks measured, with correct handling of common pytest formats.

- **H1:** for every test log the benchmarks recorded, the shell path produces output byte-identical to the `run_pytest` path. Proven by replay (A1, A2).
- **H2:** no other recorded harness tool output changes, so every benchmarked session would be condensed byte-identically by the new code (A3).
- **H3:** the new path never sends anything the input didn't contain, except `[cf: …]` lines (I1).

**Not in scope:**
- v1 (`synthesizer.py`) and the `run_pytest`/`pytest` path are unchanged.
- Other test runners (jest, `go test`, cargo) keep the generic summary.
- No new benchmark claims.

## 3. Which outputs count as pytest output

- **D1.** Tool named `run_pytest` or `pytest`: unchanged.
- **D2. Shell-like tool name.** Split the name into lowercase tokens on non-alphanumerics and camelCase boundaries. It's shell-like if any token is in {`bash`, `sh`, `zsh`, `shell`, `terminal`, `console`, `command`, `cmd`, `exec`, `execute`, `run`, `powershell`, `pwsh`, `process`, `test`, `tests`, `pytest`}.
  - Yes: `bash`, `Bash`, `execute_bash`, `developer__shell`, `run_terminal_cmd`, `execute_command`, `run_shell_command`, `run_tests`.
  - No: `finish`, `read_file`, `edit_file`, `search_code`, `str_replace_editor`. This matters: the harness's `finish` output contains pytest output in all 326 recorded `finish` records.
- **D3. Format evidence**, required in addition to D2. At least one of:
  - the session banner `^=+ test session starts =+$`;
  - a decorated summary line `^=+ (.+) in \d+(\.\d+)?s( \([^)]*\))? =+$` whose inner text contains an outcome word (`passed`, `failed`, `error`, `errors`, `skipped`, `xfailed`, `xpassed`, `deselected`, `warning`, `warnings`) or `no tests ran`;
  - a plain summary line, `^((\d+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)(, )?)+|no tests ran) in \d+(\.\d+)?s`. This one counts **only together with a second marker:**
    - a progress line ending in `[ NN%]`;
    - a `FAILURES` or `ERRORS` header;
    - a `short test summary info` header;
    - a line starting with `FAILED ` or `ERROR ` followed by a test id.
- **D4. Exemptions still win.** Exempt tool names and reader commands in the arguments (`cat `, `head `, `tail `, `grep `, …) pass through unchanged as today, so `pytest … | tail -50` isn't touched.
- **D5. Kill switch.** `RUNSIFT_SHELL_PYTEST=off` disables D2 and D3 and restores today's generic path. It's on by default.

Measured with these markers: of the 6,809 recorded non-`run_pytest` tool outputs, the only ones with pytest markers are the 326 `finish` records, and D2 excludes them. The implementation must re-check this with its own detector (A3).

## 4. Splitting a shell output

- **S1. Region start:** the first banner line. If there is none, the first pytest marker line: a progress line, a verbose result line (`path.py::test STATUS`), a `FAILURES`/`ERRORS` header, a short-summary header, a `FAILED`/`ERROR` test line, or a summary line.
- **S2. Region end:** the last summary line, decorated or plain. If there is none, the region runs to the end of the output.
- **S3.** The **prefix** is the lines before the region, and the **suffix** is the lines after it.

Measured on the 1,761 recorded `run_pytest` outputs:
- 1,725 start with the banner, and none has a line after its last summary line, so prefix and suffix are always empty for them;
- the other 36 are the one-line `[TASK FINISHED - ALL TESTS PASSED]`, which D3 doesn't match and which is too short to condense on either path.

## 5. The digest

- **G1. Verbose output** (the region has at least one verbose result line matching `\S+\.py::\S+\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)`, after normalizing `\` to `/`): run today's `condense_pytest_fresh` or `condense_pytest_stale` on the region text, unchanged. This is the benchmarked digest. All 1,725 recorded logs that contain pytest output are verbose.
- **G2. Other formats** (default, `-q`, `--tb=line`, 0 tests collected): the same functions with three behavior switches. Their defaults keep today's behavior exactly.
  1. Accept a plain summary line, and `no tests ran in …` (plain or decorated), as the final summary line.
  2. **Per-file counts from default-mode progress lines,** `^(\S+\.py) ([.FEsxX]+)\s+\[\s*\d+%\]$`: `.` counts as passed, `F` and `E` as failed, `s` as skipped; `x` and `X` aren't counted. A progress line that has no filename continues the previous file.
  3. **If neither verbose nor progress lines name files, emit no per-file count lines.** The final summary line carries the totals. Never emit a count that wasn't derived from the output.
- **G3. Incomplete runs.** If the region has no summary line and no `[TIMEOUT]` line:
  - keep today's `[cf: test run did not finish (timed out or interrupted); …]` label;
  - also keep the region's last 3 non-empty lines verbatim, skipping any already kept.

  With a `[TIMEOUT]` line (the harness's own marker), behavior is today's.
- **G4. Prefix and suffix are kept verbatim,** each line through `clip_long_line`:
  - The prefix goes under `[cf: output before the test run]`, before the digest's first line. It's capped at its first 4 and last 4 lines.
  - The suffix goes under `[cf: output after the test run]`, just before the `<!-- cf:v2 -->` marker. It's capped at its first 10 and last 10 lines.
  - Omitted lines are replaced by `[cf: N lines omitted]`.
  - When the prefix or suffix is empty, its label isn't added.
- **G5.** If the condensed text isn't shorter than the input, send the input unchanged. This applies to the shell path only.
- **G6. Target from a shell command.** If the arguments have a `target` key, use it as today. Otherwise:
  - if a string value in the arguments JSON contains `pytest`, the target is the first token after the pytest token (`pytest`, `py.test`, or `-m pytest`) that doesn't start with `-`;
  - skip the value of options that take a separate value: `-k`, `-m`, `-o`, `-p`, `-c`, `--ignore`, `--deselect`, `--rootdir`, `--junitxml`, `--maxfail`, `--durations`;
  - tokenize with `shlex` in posix mode, and on a parse error fall back to splitting on whitespace.

  The result feeds today's "Requested target" line exactly as the `target` argument does.
- **G7. Stale rewrite** (v2 only; v2.1 never rewrites history): the rules are unchanged. The run key is the tool name plus the file set from G1's or G2's per-file extraction, else the target.

## 6. Invariants (unit-tested on every fixture)

- **I1 Verbatim:** every output line is a `[cf: …]` line, the marker, or an input line (possibly clipped by `clip_long_line`).
- **I2:** every failing test id in the input's `FAILED`/`ERROR` lines appears in the output.
- **I3:** each failure's `E` lines appear, in formats that print them.
- **I4:** the final summary line appears verbatim when present. The "did not finish" label appears only when there is no summary line.
- **I5:** per-file count lines appear only when derived from verbose or progress lines, and they match them.
- **I6:** prefix and suffix lines appear verbatim, within the caps.
- **I7 Idempotent:** condensing the output again returns it unchanged.
- **I8 Deterministic.**

## 7. Fixtures

**Generator.** `scripts/gen_pytest_format_fixtures.py` builds the 5-bug gauntlet workspace in a temporary directory and captures real pytest output for:
- the harness flags `-vv -ra --tb=long`;
- no flags; `-q`; `-x -q`; `-v`; `--tb=short`; `-q --tb=line`;
- a missing test file;
- a run without `pythonpath`, which produces collection errors;
- an all-passing run (`materialize_clean_repo` without injected bugs), with no flags and with `-q`.

**Wrapped variants** built from those captures:
- a command-echo prefix line;
- shell suffix lines (`exit code: 1`; `Command timed out after 120 seconds`);
- a killed run: the no-flags output cut after its third progress line, plus a timeout line;
- forward-slash (Linux-style) paths.

Replace the `rootdir:` path with `<workspace>` before saving. Commit the files under `tests/fixtures/pytest_formats/`.

## 8. Acceptance checks (all must pass before the code commit)

- **A1. Log-level replay,** in `benchmarks/tensorroute_benchmark/scripts/replay_shell_equivalence.py`. Run it for v2 and for v2.1.
  - **Check:** for every `run_pytest` record in every committed sidecar, compare two condensed versions:
    - the record condensed as `run_pytest` with its recorded arguments;
    - the record condensed as `bash` with `{"command": "python -m pytest -o pythonpath=src -vv -ra --tb=long --durations=0", **recorded_args}`. The recorded `target` is carried through unchanged; 504 recorded targets aren't file paths (`full`, `tests/`, `--tb=short`).
  - **Required:** 1,761 of 1,761 byte-identical.
  - **Also report, not required:** the same check with the target only inside the command string, for the 1,039 records whose target is a `.py` path or `.py::` test id and the 218 with no target.
- **A2. Session-level replay.**
  - Rebuild each sidecar session's tool-call and tool-message history.
  - Rename `run_pytest` to `bash`, with arguments as in A1.
  - Run `compact_messages` on the full history for v2 and for v2.1.
  - **Required:** every tool message is byte-identical to the run without renaming, so 0 mismatches.
- **A3. No change to benchmarked sessions.**
  - **Required:** none of the 6,809 recorded non-`run_pytest` outputs is detected by D2+D3 under its recorded tool name.
  - Report the count.
- **A4. Unit tests:**
  - I1–I8 on every fixture;
  - the D2 name examples;
  - D3 negatives: `echo "3 passed in 0.1s"`; a source file that contains the banner text, printed by a shell command; a `finish`-style message;
  - G5;
  - G6 with realistic commands (`python -m pytest -q tests/test_codec.py`, `pytest -k "radix and not slow" tests`, `cd /repo && pytest -x`);
  - the kill switch;
  - A1 on a fixed sample of 20 sidecar records, so it runs inside `pytest`.
- **A5.** The golden tests and all 101 existing tests pass unchanged.

## 9. README

Add to "How it works":

> **Test output is recognized** whether it comes from a dedicated `run_pytest` or `pytest` tool or from a shell tool (`bash`, `execute_bash`, …), by pytest's own output format. The benchmarks used a dedicated `run_pytest` tool with `-vv -ra --tb=long`. For that format the shell path gives byte-identical output, checked by replaying every recorded test log. Other pytest formats are covered by unit tests, not benchmarks. Other test runners get the generic summary.

## 10. What this does not show

The same condensed input doesn't mean the same results with a different agent. Nothing in the README, slides or Devpost text may claim the benchmark results for OpenCode or any other agent.

## 11. Order of work

1. Commit this addendum alone.
2. Add the fixture script and the fixtures.
3. Implement D1–D5, S1–S3 and G1–G7 in `src/compact/condenser_v2.py`. Leave `synthesizer.py` unchanged.
4. Add the tests and the replay script, then run A1–A3 and record the counts.
5. Add the README paragraph, and update the repository map and `MOCK(M-NNN)` markers if needed. There should be no new mock.
6. Run all seven gates, push (no force), report the A1–A3 counts with the final line of each gate, and stop.
