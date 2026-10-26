"""Textual replay UI for RunSift (`python -m src.tui replay`).

Implements Section 4 of `docs/specs/TERMINAL_APP_SPEC.md`:
- RunsScreen (4.1): seven rows of the README results table, computed from committed result files.
- SessionsScreen (4.2): 15 repeats for the selected run, showing without vs. with RunSift.
- SessionScreen (4.3): two side-by-side summary panels, request-size chart, and turn stepper
  with raw tool output, condensed model view, status line, and header-reproduction badge.
Allowed dependencies per wiring.yaml: ["core", "compact"].
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Label, Static

from src.tui.data import (
    DEFAULT_RESULTS_DIR,
    RunRowSummary,
    SessionPairSummary,
    TurnStepView,
    build_session_turn_steps,
    load_all_runs,
)

RUNSIFT_BANNER = """\
╔═════╗╔═╗ ╔═╗╔═╗ ╔═╗╔═════╗╔═════╗╔═════╗╔═════╗
║ ╔═╗ ║║ ║ ║ ║║ ╚╗║ ║║ ╔═══╝╚═╗ ╔═╝║ ╔═══╝╚═╗ ╔═╝
║ ╚═╝ ║║ ║ ║ ║║ ╔╗║ ║║ ╚═══╗  ║ ║  ║ ╚══╗    ║ ║  
║ ╔╗ ╔╝║ ║ ║ ║║ ║╚╗ ║╚═══╗ ║  ║ ║  ║ ╔══╝    ║ ║  
║ ║║ ╚╗║ ╚═╝ ║║ ║ ║ ║╔═══╝ ║╔═╝ ╚═╗║ ║       ║ ║  
╚═╝╚══╝╚═════╝╚═╝ ╚═╝╚═════╝╚═════╝╚═╝       ╚═╝  """

# Backward compatibility alias
CACHEFORGE_BANNER = RUNSIFT_BANNER

HELP_TEXT = """\
RunSift Replay — Keyboard Controls

  Enter       Open selected run or session
  ← / →       Previous / next turn in Session screen
  ↑ / ↓       Move table cursor or scroll focused pane
  PgUp / PgDn Scroll focused pane by page
  Tab         Switch focus between panes
  a           Switch arm (with RunSift ↔ without RunSift)
  Esc         Go back to previous screen (or close help)
  ?           Open this help screen
  q           Quit

Note: One session is an example, not a result. The Runs screen has the results.
"""


class HelpModalScreen(ModalScreen[None]):
    """Modal help dialog shown when pressing `?`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss_help", "Close"),
        Binding("question_mark", "dismiss_help", "Close"),
        Binding("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-modal"):
            yield Static(HELP_TEXT, id="help-text")

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


def _format_summary_panel(title: str, task: str, arm_summary: object) -> str:
    from src.tui.data import ArmSessionSummary

    assert isinstance(arm_summary, ArmSessionSummary)
    unit_label = "Traps solved" if task == "hard" else "Bugs fixed"
    resc = " (rescored)" if arm_summary.rescored else ""
    ovf = "yes" if arm_summary.context_overflow else "no"
    return (
        f"[b]{title}[/b]: {unit_label} {arm_summary.fixed}/5{resc} · "
        f"Turns {arm_summary.turns} · ${arm_summary.cost_usd:.3f}\n"
        f"Prompt {arm_summary.prompt_tokens:,} · Peak {arm_summary.peak_turn_prompt_tokens:,} · "
        f"{arm_summary.wall_clock_s:.1f}s · Ran out of context: {ovf}"
    )


_SPARK_BLOCKS = ".░▒▓█"


def _render_sparkline(values: list[int], scale_max: int) -> str:
    if not values or scale_max <= 0:
        return "—"
    chars: list[str] = []
    n_levels = len(_SPARK_BLOCKS)
    for v in values:
        if v <= 0:
            chars.append(_SPARK_BLOCKS[0])
        else:
            ratio = min(1.0, v / scale_max)
            idx = min(n_levels - 1, max(0, round(ratio * (n_levels - 1))))
            chars.append(_SPARK_BLOCKS[idx])
    return "".join(chars)


def _format_request_size_chart(session: SessionPairSummary) -> str:
    b_toks = session.baseline.per_turn_prompt_tokens
    c_toks = session.runsift.per_turn_prompt_tokens
    b_peak = session.baseline.peak_turn_prompt_tokens
    c_peak = session.runsift.peak_turn_prompt_tokens
    scale_max = max(1, b_peak, c_peak)

    pad_width = max(len(b_toks), len(c_toks), 1)
    b_spark = _render_sparkline(b_toks, scale_max)
    c_spark = _render_sparkline(c_toks, scale_max)

    return (
        f"[b]Request-size chart (shared scale: 0 .. {scale_max:,} tok)[/b]\n"
        f"Without: {b_spark:<{pad_width}}  peak {b_peak:,} tok ({len(b_toks)}t)\n"
        f"With:    {c_spark:<{pad_width}}  peak {c_peak:,} tok ({len(c_toks)}t)"
    )


class SessionScreen(Screen[None]):
    """Screen 4.3: Single paired session with summary panels, chart, and turn stepper."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("left", "prev_turn", "Prev Turn", priority=True),
        Binding("right", "next_turn", "Next Turn", priority=True),
        Binding("a", "toggle_arm", "Switch Arm", priority=True),
        Binding("escape", "go_back", "Back", priority=True),
        Binding("question_mark", "open_help", "Help", priority=True),
        Binding("q", "app.quit", "Quit", priority=True),
    ]

    def __init__(self, session: SessionPairSummary) -> None:
        super().__init__()
        self.session = session
        self.active_arm: str = "runsift"
        self.turn_index: int = 0
        self._steps_cache: dict[str, list[TurnStepView]] = {
            "runsift": build_session_turn_steps(session, arm="runsift"),
            "cacheforge": build_session_turn_steps(session, arm="runsift"),
            "baseline": build_session_turn_steps(session, arm="baseline"),
        }

    @property
    def current_steps(self) -> list[TurnStepView]:
        return self._steps_cache.get(self.active_arm, [])

    @property
    def current_step(self) -> TurnStepView | None:
        steps = self.current_steps
        if not steps:
            return None
        idx = max(0, min(self.turn_index, len(steps) - 1))
        return steps[idx]

    def compose(self) -> ComposeResult:
        s = self.session
        commit_short = s.git_commit[:7]
        date_str = s.launch_timestamp[:10] if len(s.launch_timestamp) >= 10 else s.launch_timestamp
        header_text = (
            f"[b]Session — Run {s.spec.run_label} · Repeat {s.rep}[/b]  |  "
            f"task: {s.spec.task}  |  condenser: {s.spec.condenser}  |  "
            f"commit: {commit_short}  |  date: {date_str}"
        )
        yield Static(header_text, id="session-header")

        with Horizontal(id="session-summaries"):
            yield Static(
                _format_summary_panel("Without RunSift", s.spec.task, s.baseline),
                id="panel-without",
                classes="summary-panel",
            )
            yield Static(
                _format_summary_panel("With RunSift", s.spec.task, s.runsift),
                id="panel-with",
                classes="summary-panel",
            )

        yield Static(_format_request_size_chart(s), id="request-size-chart")
        yield Static("", id="turn-header", markup=False)

        with Horizontal(id="turn-panes"):
            with VerticalScroll(id="pane-left", can_focus=True):
                yield Label("[b]What the tool returned[/b]", id="pane-left-title")
                yield Static("", id="pane-left-body", markup=False)
            with VerticalScroll(id="pane-right", can_focus=True):
                yield Label("[b]What the model saw[/b]", id="pane-right-title")
                yield Static("", id="pane-right-body", markup=False)

        yield Static("", id="turn-status")
        yield Static("", id="turn-badge")
        yield Static(
            "Keys: ←/→ turn · ↑/↓ PgUp/PgDn scroll · Tab switch pane · a switch arm · Esc back · ? help · q quit",
            id="session-footer",
        )

    def on_mount(self) -> None:
        self._refresh_turn_view()
        self.query_one("#pane-right", VerticalScroll).focus()

    def on_resize(self) -> None:
        self._refresh_turn_view()

    def _refresh_turn_view(self) -> None:
        step = self.current_step
        arm_label = "with RunSift" if self.active_arm in ("runsift", "cacheforge") else "without RunSift"
        if step is None:
            self.query_one("#turn-header", Static).update(
                f"Arm: {arm_label} — No recorded tool outputs for this run."
            )
            self.query_one("#pane-left-body", Static).update("")
            self.query_one("#pane-right-body", Static).update("")
            self.query_one("#turn-status", Static).update("This tool output: 0 → 0 chars (0%)")
            self.query_one("#turn-badge", Static).update("")
            return

        passing_str = str(step.tests_passing) if step.tests_passing is not None else "—"
        prefix = (
            f"Turn {step.turn}/{step.total_turns} ({arm_label}) | "
            f"{step.tool} | Tests: {passing_str} | Args: "
        )
        term_width = self.app.size.width if self.app.size.width > 0 else 120
        avail_width = max(40, term_width - 2)  # account for horizontal padding 0 1
        max_args = max(6, avail_width - len(prefix))

        args_one_line = " ".join(step.args_str.splitlines()).strip()
        if len(args_one_line) > max_args:
            args_one_line = args_one_line[: max(1, max_args - 3)] + "..."

        header_line = f"{prefix}{args_one_line}"
        if len(header_line) > avail_width:
            header_line = header_line[: max(1, avail_width - 3)] + "..."

        self.query_one("#turn-header", Static).update(header_line)
        self.query_one("#pane-left-body", Static).update(step.raw_output)
        self.query_one("#pane-right-body", Static).update(step.model_saw)
        if step.pct_change > 0:
            pct_str = f"−{step.pct_change}%"
        elif step.pct_change < 0:
            pct_str = f"+{abs(step.pct_change)}%"
        else:
            pct_str = "0%"
        self.query_one("#turn-status", Static).update(
            f"This tool output: {step.chars_before:,} → {step.chars_after:,} chars ({pct_str})"
        )
        self.query_one("#turn-badge", Static).update(f"Next request header check: {step.badge}")

    def action_prev_turn(self) -> None:
        if self.turn_index > 0:
            self.turn_index -= 1
            self._refresh_turn_view()

    def action_next_turn(self) -> None:
        steps = self.current_steps
        if self.turn_index + 1 < len(steps):
            self.turn_index += 1
            self._refresh_turn_view()

    def action_toggle_arm(self) -> None:
        self.active_arm = "baseline" if self.active_arm in ("runsift", "cacheforge") else "runsift"
        steps = self.current_steps
        if steps:
            self.turn_index = min(self.turn_index, len(steps) - 1)
        else:
            self.turn_index = 0
        self._refresh_turn_view()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_open_help(self) -> None:
        self.app.push_screen(HelpModalScreen())


class SessionsScreen(Screen[None]):
    """Screen 4.2: List of 15 repeats for a chosen run."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "go_back", "Back", priority=True),
        Binding("question_mark", "open_help", "Help", priority=True),
        Binding("q", "app.quit", "Quit", priority=True),
    ]

    def __init__(self, run_row: RunRowSummary) -> None:
        super().__init__()
        self.run_row = run_row

    def compose(self) -> ComposeResult:
        r = self.run_row
        header_text = (
            f"[b]Sessions — Run {r.spec.run_label} ({r.spec.task}, condenser {r.spec.condenser})[/b]  |  "
            f"commit: {r.git_commit}  |  launch date: {r.launch_date}"
        )
        yield Static(header_text, id="sessions-header")
        yield DataTable(id="sessions-table", cursor_type="row")
        replay_note = (
            "Enter: open session turn stepper"
            if r.spec.replayable
            else "Note: Run v3 has no recorded tool-output sidecars (summary only)"
        )
        yield Static(
            f"{replay_note}  |  Esc: back to Runs  |  ?: help  |  q: quit",
            id="sessions-footer",
        )

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns(
            "Repeat",
            "Fixed (without → with)",
            "Cost (without → with)",
            "Turns (without → with)",
            "Largest request (without → with)",
            "Ran out of context",
        )
        for sess in self.run_row.sessions:
            b = sess.baseline
            c = sess.runsift
            resc = " (rescored)" if (b.rescored or c.rescored) else ""
            if b.context_overflow and c.context_overflow:
                ovf = "both"
            elif b.context_overflow:
                ovf = "without RunSift"
            elif c.context_overflow:
                ovf = "with RunSift"
            else:
                ovf = "no"
            table.add_row(
                str(sess.rep),
                f"{b.fixed} → {c.fixed}{resc}",
                f"${b.cost_usd:.3f} → ${c.cost_usd:.3f}",
                f"{b.turns} → {c.turns}",
                f"{b.peak_turn_prompt_tokens:,} → {c.peak_turn_prompt_tokens:,}",
                ovf,
                key=str(sess.rep),
            )
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self.run_row.sessions):
            sess = self.run_row.sessions[idx]
            if sess.spec.replayable:
                self.app.push_screen(SessionScreen(sess))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_open_help(self) -> None:
        self.app.push_screen(HelpModalScreen())


class RunsScreen(Screen[None]):
    """Screen 4.1: The first screen showing all seven pre-registered Token Factory runs."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("question_mark", "open_help", "Help", priority=True),
        Binding("q", "app.quit", "Quit", priority=True),
    ]

    def __init__(self, runs: list[RunRowSummary]) -> None:
        super().__init__()
        self.runs = runs

    def compose(self) -> ComposeResult:
        yield Static(RUNSIFT_BANNER, id="runs-banner", markup=False)
        yield Static(
            "[b]RunSift Replay — Pre-registered Nebius Token Factory Runs (NVIDIA Nemotron 3 Nano)[/b]",
            id="runs-header",
        )
        yield DataTable(id="runs-table", cursor_type="row")
        yield Static(
            "Every number here is computed from the committed result files. Costs are at list price.",
            id="runs-footer",
        )

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns(
            "Run",
            "Task",
            "Condenser",
            "Fixed, without → with",
            "Prompt tokens / session",
            "Cost / session",
            "Sessions / dollar",
        )
        for r in self.runs:
            table.add_row(
                r.spec.run_label,
                r.spec.task,
                r.spec.condenser,
                r.fixed_cell,
                r.prompt_tokens_cell,
                r.cost_cell,
                r.sessions_per_dollar_cell,
                key=r.spec.key,
            )
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self.runs):
            self.app.push_screen(SessionsScreen(self.runs[idx]))

    def action_open_help(self) -> None:
        self.app.push_screen(HelpModalScreen())


class ReplayApp(App[None]):
    """Textual application for browsing and replaying recorded Token Factory sessions."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #runs-banner {
        padding: 0 1;
        color: $accent;
        height: auto;
    }
    #runs-header, #sessions-header, #session-header {
        width: 100%;
        padding: 0 1;
        background: $boost;
        color: $text;
        height: auto;
    }
    #runs-table, #sessions-table {
        height: 1fr;
    }
    #runs-footer, #sessions-footer {
        width: 100%;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        height: auto;
    }
    #session-footer {
        width: 100%;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        height: 1;
        overflow: hidden;
    }
    #session-summaries {
        height: auto;
        padding: 0 1;
    }
    .summary-panel {
        width: 1fr;
        padding: 0 1;
        height: auto;
    }
    #request-size-chart {
        padding: 0 1;
        height: auto;
    }
    #turn-header {
        padding: 0 1;
        background: $boost;
        height: 1;
        overflow: hidden;
    }
    #turn-panes {
        height: 1fr;
    }
    #pane-left, #pane-right {
        width: 1fr;
        height: 1fr;
        border: solid $surface-lighten-2;
        padding: 0 1;
    }
    #pane-left:focus, #pane-right:focus {
        border: double $accent;
    }
    #turn-status, #turn-badge {
        padding: 0 1;
        height: 1;
        overflow: hidden;
    }
    #help-modal {
        width: 72;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
        margin: 2 4;
    }
    """

    def __init__(self, results_dir: Path = DEFAULT_RESULTS_DIR) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.runs: list[RunRowSummary] = load_all_runs(results_dir=results_dir)

    def on_mount(self) -> None:
        self.push_screen(RunsScreen(self.runs))
