"""Unit tests for the RunSift Terminal App data layer (T1, T2, T3)."""

from __future__ import annotations

import re

import httpx
import pytest

from src.tui.data import (
    BADGE_BASELINE,
    BADGE_MATCH,
    BADGE_NOT_RECORDED,
    BADGE_TIMEOUT_FIX,
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    RUN_SPECS,
    build_session_turn_steps,
    load_all_runs,
    load_run_row,
)


def _parse_readme_results_table() -> list[dict[str, str]]:
    """Parses the 7-row Token Factory results table from README.md without hardcoding numbers."""
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    in_table = False
    rows: list[dict[str, str]] = []
    for line in readme_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| Run | Task | Condenser |"):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("|---"):
                continue
            if not stripped.startswith("|"):
                break
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 7:
                # Strip markdown bold ** around values for clean comparison
                clean_cells = [re.sub(r"\*\*(.*?)\*\*", r"\1", c) for c in cells]
                rows.append(
                    {
                        "run": clean_cells[0],
                        "task": clean_cells[1],
                        "condenser": clean_cells[2],
                        "prompt_tokens": clean_cells[3],
                        "cost": clean_cells[4],
                        "sessions_per_dollar": clean_cells[5],
                        "fixed": clean_cells[6],
                    }
                )
    return rows


def test_t1_runs_table_equals_readme_table() -> None:
    """T1: The app's table model equals the README results table for all seven rows."""
    readme_rows = _parse_readme_results_table()
    assert len(readme_rows) == 7

    app_rows = load_all_runs(results_dir=DEFAULT_RESULTS_DIR)
    assert len(app_rows) == len(readme_rows)

    for idx, (app_row, readme_row) in enumerate(zip(app_rows, readme_rows, strict=True)):
        assert app_row.spec.run_label == readme_row["run"], f"Row {idx} run label mismatch"
        assert app_row.spec.task == readme_row["task"], f"Row {idx} task mismatch"
        assert app_row.spec.condenser == readme_row["condenser"], f"Row {idx} condenser mismatch"
        assert app_row.prompt_tokens_cell == readme_row["prompt_tokens"], (
            f"Row {idx} prompt_tokens mismatch"
        )
        assert app_row.cost_cell == readme_row["cost"], f"Row {idx} cost mismatch"
        assert app_row.sessions_per_dollar_cell == readme_row["sessions_per_dollar"], (
            f"Row {idx} sessions_per_dollar mismatch"
        )
        # In the app, fixed_cell appends " (rescored)" when _rescored files exist (Section 3 of spec);
        # compare the base fixed + p-value string against the README cell.
        app_fixed_base = app_row.fixed_cell.removesuffix(" (rescored)")
        assert app_fixed_base == readme_row["fixed"], f"Row {idx} fixed mismatch"


def test_t2_header_reproduction_fixed_sample() -> None:
    """T2: Fixed sample of sessions checking exact header reproduction and timeout-fix badges."""
    spec_map = {s.key: s for s in RUN_SPECS}

    # 1. Run B rep 15: 8 timeout diffs (requests for turns 33..40 after turn 32's [TIMEOUT])
    row_b = load_run_row(spec_map["B"], results_dir=DEFAULT_RESULTS_DIR)
    sess_b15 = row_b.sessions[14]
    steps_b15 = build_session_turn_steps(sess_b15, arm="runsift")
    assert len(steps_b15) == 40
    assert sum(1 for st in steps_b15 if st.request_reproduced) == 32
    assert sum(1 for st in steps_b15 if st.request_has_timeout_diff) == 8
    # Turn 32 had [TIMEOUT], so the badge for the next request is BADGE_TIMEOUT_FIX
    assert steps_b15[31].badge == BADGE_TIMEOUT_FIX
    assert steps_b15[0].badge == BADGE_MATCH

    # 2. Run E rep 3: 19 timeout diffs (turns 22..40 after turn 21's [TIMEOUT])
    row_e = load_run_row(spec_map["E"], results_dir=DEFAULT_RESULTS_DIR)
    sess_e3 = row_e.sessions[2]
    steps_e3 = build_session_turn_steps(sess_e3, arm="runsift")
    assert len(steps_e3) == 40
    assert sum(1 for st in steps_e3 if st.request_reproduced) == 21
    assert sum(1 for st in steps_e3 if st.request_has_timeout_diff) == 19
    assert steps_e3[20].badge == BADGE_TIMEOUT_FIX

    # 3. Run F rep 4: 27 timeout diffs (turns 14..40 after turn 13's [TIMEOUT])
    row_f = load_run_row(spec_map["F"], results_dir=DEFAULT_RESULTS_DIR)
    sess_f4 = row_f.sessions[3]
    steps_f4 = build_session_turn_steps(sess_f4, arm="runsift")
    assert len(steps_f4) == 40
    assert sum(1 for st in steps_f4 if st.request_reproduced) == 13
    assert sum(1 for st in steps_f4 if st.request_has_timeout_diff) == 27
    assert steps_f4[12].badge == BADGE_TIMEOUT_FIX

    # 4. Run C rep 3: peak tokens 185,961 baseline vs 20,723 RunSift, 100% matched
    row_c = load_run_row(spec_map["C"], results_dir=DEFAULT_RESULTS_DIR)
    sess_c3 = row_c.sessions[2]
    assert sess_c3.baseline.peak_turn_prompt_tokens == 185961
    assert sess_c3.runsift.peak_turn_prompt_tokens == 20723
    steps_c3 = build_session_turn_steps(sess_c3, arm="runsift")
    assert all(st.request_reproduced for st in steps_c3)
    assert all(st.badge == BADGE_MATCH for st in steps_c3[:-1])
    assert steps_c3[-1].badge == BADGE_NOT_RECORDED

    # 5. Run G v2 rep 2: has a [TIMEOUT] in turn 19, and since Run G is post-78df5c0, it matches!
    row_g_v2 = load_run_row(spec_map["G_v2"], results_dir=DEFAULT_RESULTS_DIR)
    sess_g2 = row_g_v2.sessions[1]
    steps_g2 = build_session_turn_steps(sess_g2, arm="runsift")
    assert all(st.request_reproduced for st in steps_g2)
    assert all(st.badge == BADGE_MATCH for st in steps_g2[:-1])
    assert steps_g2[-1].badge == BADGE_NOT_RECORDED


def test_no_baseline_step_carries_badge_match() -> None:
    """No baseline step ever carries BADGE_MATCH, and unrecorded RunSift turns show BADGE_NOT_RECORDED."""
    runs = load_all_runs(results_dir=DEFAULT_RESULTS_DIR)
    for run in runs:
        if not run.spec.replayable:
            continue
        for sess in run.sessions:
            base_steps = build_session_turn_steps(sess, arm="baseline")
            assert len(base_steps) > 0
            for st in base_steps:
                assert st.badge != BADGE_MATCH
                assert st.badge == BADGE_BASELINE
                assert not st.request_reproduced
                assert st.chars_after == st.chars_before

            cf_steps = build_session_turn_steps(sess, arm="runsift")
            assert len(cf_steps) > 0
            # Final turn has no subsequent request, so its next-request header is not recorded
            assert cf_steps[-1].badge == BADGE_NOT_RECORDED
            for st in cf_steps:
                assert st.chars_after == len(st.model_saw)


def test_chars_after_equals_len_model_saw_on_run_c_rep3_turn4() -> None:
    """On Run C rep 3 turn 4, chars_after == len(model_saw) == 66 (0% change, not 66 -> 0)."""
    spec_map = {s.key: s for s in RUN_SPECS}
    row_c = load_run_row(spec_map["C"], results_dir=DEFAULT_RESULTS_DIR)
    sess_c3 = row_c.sessions[2]
    steps_c3 = build_session_turn_steps(sess_c3, arm="runsift")
    st_turn4 = steps_c3[3]
    assert st_turn4.turn == 4
    assert st_turn4.chars_before == 66
    assert st_turn4.chars_after == len(st_turn4.model_saw) == 66
    assert st_turn4.pct_change == 0


def test_t3_replay_needs_no_key_and_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3: Replay works with no NEBIUS_API_KEY and makes no network calls."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("RUNSIFT_API_KEY", raising=False)
    monkeypatch.delenv("CACHEFORGE_API_KEY", raising=False)

    def _forbid_network(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Network call forbidden during replay!")

    monkeypatch.setattr(httpx.Client, "send", _forbid_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _forbid_network)

    runs = load_all_runs(results_dir=DEFAULT_RESULTS_DIR)
    assert len(runs) == 7
    replayable_run = next(r for r in runs if r.spec.key == "C")
    steps_cf = build_session_turn_steps(replayable_run.sessions[0], arm="runsift")
    steps_base = build_session_turn_steps(replayable_run.sessions[0], arm="baseline")
    assert len(steps_cf) > 0
    assert len(steps_base) > 0
    assert steps_base[0].model_saw == "Sent unchanged (no RunSift)"
    assert steps_base[0].badge == BADGE_BASELINE


def test_cli_condense_command_outputs_headers_and_digest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`runsift condense` condenses the default benchmark specimen and prints x-runsift-* headers."""
    from src.tui.__main__ import main

    rc = main(["condense"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "x-runsift-condenser: v2" in captured
    assert "x-runsift-tool-chars-before: 40141" in captured
    assert "x-runsift-tool-chars-after: 9662 (-75.9%)" in captured
    assert "[cf: tests/" in captured


def test_serve_mask_and_persist_env_key(tmp_path) -> None:
    """`_mask_key` masks secrets and `_persist_env_key` writes/updates .env cleanly."""
    from src.tui.__main__ import _mask_key, _persist_env_key

    assert _mask_key("short") == "****"
    assert _mask_key("v1.abcdefghijklmnop9876") == "v1.ab...9876"

    env_file = tmp_path / ".env"
    _persist_env_key(env_file, "nebius-key-111", "v2")
    text1 = env_file.read_text(encoding="utf-8")
    assert "NEBIUS_API_KEY=nebius-key-111" in text1
    assert "RUNSIFT_CONDENSER=v2" in text1

    _persist_env_key(env_file, "nebius-key-222", "v2.1")
    text2 = env_file.read_text(encoding="utf-8")
    assert "NEBIUS_API_KEY=nebius-key-222" in text2
    assert "RUNSIFT_CONDENSER=v2.1" in text2
    assert "nebius-key-111" not in text2
