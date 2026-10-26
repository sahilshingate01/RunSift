"""Textual pilot UI tests for the RunSift replay application (T7)."""

from __future__ import annotations

import pytest
from textual.widgets import DataTable, Static

from src.tui.replay_app import HelpModalScreen, ReplayApp, RunsScreen, SessionScreen, SessionsScreen


@pytest.mark.asyncio
@pytest.mark.parametrize("term_size", [(120, 30), (80, 24)])
async def test_t7_replay_ui_navigation_and_controls(term_size: tuple[int, int]) -> None:
    """T7: Runs screen loads with 7 rows; Enter opens run then session; -> steps turn; a switches arm; Esc goes back."""
    app = ReplayApp()
    async with app.run_test(size=term_size) as pilot:
        await pilot.pause()

        # 1. Runs screen loads with 7 rows, no session pre-selected, and Fixed column right after Condenser
        assert isinstance(app.screen, RunsScreen)
        assert len(app.screen.query("#runs-banner")) == 1
        runs_table = app.screen.query_one("#runs-table", DataTable)
        assert runs_table.row_count == 7
        col_labels = [str(col.label) for col in runs_table.columns.values()]
        assert col_labels[:4] == [
            "Run",
            "Task",
            "Condenser",
            "Fixed, without → with",
        ]
        footer_text = str(app.screen.query_one("#runs-footer", Static).render())
        assert "Every number here is computed from the committed result files." in footer_text
        assert "Costs are at list price." in footer_text

        # Move down to row 1 (Run B, which is replayable; row 0 is v3)
        await pilot.press("down")
        await pilot.pause()

        # 2. Enter opens the Sessions screen (15 repeats, no banner, all 15 visible at 80x24 without scrolling)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        assert len(app.screen.query("#sessions-banner")) == 0
        sessions_table = app.screen.query_one("#sessions-table", DataTable)
        assert sessions_table.row_count == 15
        assert sessions_table.max_scroll_y == 0

        # 3. Enter opens a Session screen (Repeat 1 of Run B)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionScreen)
        sess_screen = app.screen
        assert sess_screen.turn_index == 0
        assert sess_screen.active_arm in ("runsift", "cacheforge")

        turn_header_1 = str(sess_screen.query_one("#turn-header", Static).render())
        assert "Turn 1/40" in turn_header_1
        assert "with RunSift" in turn_header_1
        assert "| Tests: " in turn_header_1

        # 4. -> steps a turn
        await pilot.press("right")
        await pilot.pause()
        assert sess_screen.turn_index == 1
        turn_header_2 = str(sess_screen.query_one("#turn-header", Static).render())
        assert "Turn 2/40" in turn_header_2

        # 5. `a` switches arm to baseline ("without RunSift")
        await pilot.press("a")
        await pilot.pause()
        assert sess_screen.active_arm == "baseline"
        turn_header_base = str(sess_screen.query_one("#turn-header", Static).render())
        assert "without RunSift" in turn_header_base
        right_body_base = str(sess_screen.query_one("#pane-right-body", Static).render())
        assert "Sent unchanged (no RunSift)" in right_body_base

        # 6. `?` opens Help modal with required sentence, Esc closes it
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpModalScreen)
        help_rendered = str(app.screen.query_one("#help-text", Static).render())
        assert (
            "One session is an example, not a result. The Runs screen has the results."
            in help_rendered
        )

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SessionScreen)

        # 7. Esc goes back to Sessions screen, then Runs screen
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RunsScreen)


@pytest.mark.asyncio
@pytest.mark.parametrize("term_size", [(120, 30), (80, 24)])
async def test_run_c_rep3_turn4_panes_content_height_and_single_line_header(
    term_size: tuple[int, int],
) -> None:
    """On Run C rep 3 turn 4 at 120x30 and 80x24:
    - #turn-header fits in a single line (height == 1 and len <= cols - 2) and includes the tests-passing value before Args
    - #turn-status shows '66 → 66 chars (0%)' (no minus sign on 0%)
    - #request-size-chart uses Lucida Console compatible characters (.░▒▓█)
    - #turn-badge and both panes are visible, and each pane's content height is at least 5 rows.
    """
    from textual.containers import VerticalScroll

    cols, rows = term_size
    app = ReplayApp()
    async with app.run_test(size=term_size) as pilot:
        await pilot.pause()

        # Select Run C (row index 2: row 0=v3, row 1=B, row 2=C)
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        assert app.screen.run_row.spec.key == "C"
        sessions_table = app.screen.query_one("#sessions-table", DataTable)
        assert sessions_table.row_count == 15
        assert sessions_table.max_scroll_y == 0

        # Select Repeat 3 (row index 2: rep 1, rep 2, rep 3)
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionScreen)
        sess_screen = app.screen
        assert sess_screen.session.rep == 3

        # Step to Turn 4 (turn_index == 3: press right 3 times)
        await pilot.press("right", "right", "right")
        await pilot.pause()
        assert sess_screen.turn_index == 3

        turn_header = sess_screen.query_one("#turn-header", Static)
        header_text = str(turn_header.render())
        assert "Turn 4/24 (with RunSift) | edit_file | Tests: — | Args: " in header_text
        assert "..." in header_text
        assert "\n" not in header_text
        assert len(header_text) <= cols - 2
        assert turn_header.region.height == 1

        turn_status = sess_screen.query_one("#turn-status", Static)
        status_text = str(turn_status.render())
        assert status_text == "This tool output: 66 → 66 chars (0%)"

        pane_left = sess_screen.query_one("#pane-left", VerticalScroll)
        pane_right = sess_screen.query_one("#pane-right", VerticalScroll)
        turn_badge = sess_screen.query_one("#turn-badge", Static)
        chart_text = str(sess_screen.query_one("#request-size-chart", Static).render())
        assert "shared scale: 0 .. 185,961 tok" in chart_text
        assert "peak 185,961 tok" in chart_text
        assert "peak 20,723 tok" in chart_text
        assert "█" in chart_text
        assert "░" in chart_text or "." in chart_text
        for bad_ch in "▁▂▃▅▆▇✓≠":
            assert bad_ch not in chart_text
            assert bad_ch not in str(turn_badge.render())

        for widget in (pane_left, pane_right, turn_badge):
            assert widget.visible
            assert widget.region.width > 0
            assert widget.region.height > 0
            assert widget.region.y >= 0
            assert widget.region.bottom <= rows

        # Both turn panes must have at least 5 rows of inner content height (excluding borders)
        assert pane_left.content_region.height >= 5
        assert pane_right.content_region.height >= 5

        # Also step to a turn with a numeric tests_passing count (Turn 5 of Run C rep 3 has tests_passing == 35)
        await pilot.press("right")
        await pilot.pause()
        assert sess_screen.turn_index == 4
        header_t5 = str(sess_screen.query_one("#turn-header", Static).render())
        assert "Turn 5/24 (with RunSift) | run_pytest | Tests: 35 | Args: " in header_t5
        assert len(header_t5) <= cols - 2
        assert sess_screen.query_one("#turn-header", Static).region.height == 1
