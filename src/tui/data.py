"""Read-only data layer for the RunSift Terminal App.

Reads committed benchmark result files and sidecars from `benchmarks/tensorroute_benchmark/results/`,
computes run and session metrics from the source files, and replays tool outputs through `CondenserV2`.
Allowed dependencies per wiring.yaml: ["core", "compact"].
"""

from __future__ import annotations

import gzip
import json
import re
import statistics
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from src.compact import CondenserV2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "tensorroute_benchmark" / "results"

# List pricing on Nebius Token Factory for NVIDIA Nemotron 3 Nano ($0.06/1M in, $0.24/1M out)
INPUT_PRICE_PER_M = 0.06
OUTPUT_PRICE_PER_M = 0.24

# Commits of Runs B, C, E, F recorded before the timeout fix (78df5c0)
PRE_TIMEOUT_FIX_COMMITS = frozenset(
    {
        "c8a7a04a652bac9ed711e46444a5cabecf895fcb",  # v3
        "4d820a062c085810d0944c473116eb2c236a2a1e",  # Runs B & C
        "7e05884c9cc77736d034f355d9f1f5a536f1cf5d",  # Run E
        "07a77963f2121e15dc0b2e5ca79a3203109444e2",  # Run F
    }
)

BADGE_MATCH = "√ matches what the gateway sent"
BADGE_TIMEOUT_FIX = "!= differs: the condenser changed after this run (timeout fix, 78df5c0)"
BADGE_DIFFERS = "!= differs"
BADGE_BASELINE = "No gateway on this arm (sent directly)"
BADGE_NOT_RECORDED = "not recorded"

NON_HISTORY_TOOLS = frozenset({"none", "overflow", "infra_error", "error"})


@dataclass(frozen=True)
class RunSpec:
    """Specification of a pre-registered Token Factory run row."""

    key: str
    run_label: str
    task: str  # "standard" or "hard"
    condenser: str  # "v1", "v2", or "v2.1"
    baseline_prefix: str
    runsift_prefix: str
    replayable: bool
    use_rescored_scores: bool

    @property
    def cacheforge_prefix(self) -> str:
        return self.runsift_prefix


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        key="v3",
        run_label="v3",
        task="standard",
        condenser="v1",
        baseline_prefix="nebius_gauntlet_nemotron_30b_40turn_v3",
        runsift_prefix="nebius_gauntlet_nemotron_30b_40turn_v3",
        replayable=False,
        use_rescored_scores=True,
    ),
    RunSpec(
        key="B",
        run_label="B",
        task="hard",
        condenser="v2",
        baseline_prefix="nebius_coupled_nemotron_30b_40turn_v2",
        runsift_prefix="nebius_coupled_nemotron_30b_40turn_v2",
        replayable=True,
        use_rescored_scores=False,
    ),
    RunSpec(
        key="C",
        run_label="C",
        task="standard",
        condenser="v2",
        baseline_prefix="nebius_gauntlet_nemotron_30b_40turn_v2",
        runsift_prefix="nebius_gauntlet_nemotron_30b_40turn_v2",
        replayable=True,
        use_rescored_scores=False,
    ),
    RunSpec(
        key="E",
        run_label="E",
        task="hard",
        condenser="v2.1",
        baseline_prefix="nebius_coupled_nemotron_30b_40turn_v2_1",
        runsift_prefix="nebius_coupled_nemotron_30b_40turn_v2_1",
        replayable=True,
        use_rescored_scores=False,
    ),
    RunSpec(
        key="F",
        run_label="F",
        task="standard",
        condenser="v2.1",
        baseline_prefix="nebius_gauntlet_nemotron_30b_40turn_v2_1",
        runsift_prefix="nebius_gauntlet_nemotron_30b_40turn_v2_1",
        replayable=True,
        use_rescored_scores=True,
    ),
    RunSpec(
        key="G_v2",
        run_label="G",
        task="standard",
        condenser="v2",
        baseline_prefix="nebius_gauntlet_nemotron_30b_40turn_runG_v2",
        runsift_prefix="nebius_gauntlet_nemotron_30b_40turn_runG_v2",
        replayable=True,
        use_rescored_scores=True,
    ),
    RunSpec(
        key="G_v2_1",
        run_label="G",
        task="standard",
        condenser="v2.1",
        baseline_prefix="nebius_gauntlet_nemotron_30b_40turn_runG_v2",
        runsift_prefix="nebius_gauntlet_nemotron_30b_40turn_runG_v2_1",
        replayable=True,
        use_rescored_scores=True,
    ),
)


@dataclass(frozen=True)
class ArmSessionSummary:
    """Metrics for one arm (without or with RunSift) of a single repeat session."""

    arm: str  # "baseline", "runsift", or "cacheforge"
    fixed: int
    rescored: bool
    turns: int
    prompt_tokens: int
    completion_tokens: int
    peak_turn_prompt_tokens: int
    cost_usd: float
    wall_clock_s: float
    context_overflow: bool
    per_turn_prompt_tokens: list[int]
    per_turn_tools: list[str]
    per_turn_tool_args: list[str]
    per_turn_tests_passing: list[int | None]
    per_turn_condenser: list[str]
    per_turn_tool_chars_before: list[int]
    per_turn_tool_chars_after: list[int]


@dataclass(frozen=True)
class SessionPairSummary:
    """Paired summary for one repeat (1..15) in a run."""

    spec: RunSpec
    rep: int
    git_commit: str
    git_dirty: bool
    launch_timestamp: str
    baseline: ArmSessionSummary
    runsift: ArmSessionSummary
    sidecar_path: Path | None
    baseline_sidecar_path: Path | None

    @property
    def cacheforge(self) -> ArmSessionSummary:
        return self.runsift


@dataclass(frozen=True)
class RunRowSummary:
    """Computed row for the Runs screen (matching the README results table)."""

    spec: RunSpec
    repeats: int
    baseline_prompt_tokens_mean: float
    runsift_prompt_tokens_mean: float
    prompt_tokens_change_pct: int  # rounded integer e.g. -74
    baseline_cost_mean: float
    runsift_cost_mean: float
    sessions_per_dollar_ratio: float
    baseline_fixed_total: int
    runsift_fixed_total: int
    rescored: bool
    p_value: float
    git_commit: str
    launch_date: str
    sessions: list[SessionPairSummary]

    @property
    def cacheforge_prompt_tokens_mean(self) -> float:
        return self.runsift_prompt_tokens_mean

    @property
    def cacheforge_cost_mean(self) -> float:
        return self.runsift_cost_mean

    @property
    def cacheforge_fixed_total(self) -> int:
        return self.runsift_fixed_total

    @property
    def prompt_tokens_cell(self) -> str:
        sign = "−" if self.prompt_tokens_change_pct < 0 else "+"
        return f"{sign}{abs(self.prompt_tokens_change_pct)}%"

    @property
    def cost_cell(self) -> str:
        return f"${self.baseline_cost_mean:.3f} → ${self.runsift_cost_mean:.3f}"

    @property
    def sessions_per_dollar_cell(self) -> str:
        return f"{self.sessions_per_dollar_ratio:.1f}×"

    @property
    def p_value_formatted(self) -> str:
        return format_readme_p_value(self.p_value)

    @property
    def fixed_cell(self) -> str:
        resc = " (rescored)" if self.rescored else ""
        return (
            f"{self.baseline_fixed_total} → {self.runsift_fixed_total} "
            f"(p = {self.p_value_formatted}){resc}"
        )


@dataclass(frozen=True)
class TurnStepView:
    """Data for a single turn in the Session turn stepper."""

    turn: int
    total_turns: int
    arm: str  # "baseline" or "cacheforge"
    tool: str
    args_str: str
    tests_passing: int | None
    raw_output: str
    model_saw: str
    chars_before: int
    chars_after: int
    pct_change: int
    badge: str
    request_reproduced: bool
    request_has_timeout_diff: bool


def format_readme_p_value(p_val: float) -> str:
    """Formats a p-value to the precision used in the README results table."""
    if p_val >= 0.995:
        return "1.0"
    if p_val >= 0.1:
        return f"{p_val:.2f}"
    return f"{p_val:.3f}"


def exact_permutation_test(diffs: list[int | float]) -> tuple[float, float]:
    """Two-sided exact paired permutation test over 15 pairs."""
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    observed_diff = float(statistics.mean(diffs))
    abs_obs = abs(observed_diff)
    count = 0
    total = 2**n
    for signs in product([-1, 1], repeat=n):
        stat = sum(s * d for s, d in zip(signs, diffs, strict=True)) / n
        if abs(stat) >= abs_obs - 1e-12:
            count += 1
    return observed_diff, count / total


def compute_session_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Computes list-price session cost ($0.06/1M input, $0.24/1M output)."""
    return (
        prompt_tokens * INPUT_PRICE_PER_M + completion_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000.0


def _parse_mod(val: Any) -> int:
    if isinstance(val, str) and "/" in val:
        return int(val.split("/")[0])
    return int(val)


def _is_valid_arm_data(data: dict[str, Any], require_baseline: bool = True) -> bool:
    c_tr = data.get("runsift", data.get("cacheforge", {})).get("task_results", [])
    if not c_tr or bool(c_tr[0].get("infra_error", False)):
        return False
    if require_baseline:
        b_tr = data.get("baseline", {}).get("task_results", [])
        if not b_tr or bool(b_tr[0].get("infra_error", False)):
            return False
    return True


def _resolve_prefix_repeats(
    prefix: str,
    results_dir: Path,
    require_baseline: bool = True,
    expected_repeats: int = 15,
) -> dict[int, tuple[Path, dict[str, Any], Path | None, dict[str, Any] | None]]:
    """Resolves for each repeat 1..expected_repeats: (orig_path, orig_data, rescored_path, rescored_data).

    Matches `analyze_gauntlet_results.py` (including `_rerun1` substitution on v3 repeat 8),
    while keeping both the original file and any `_rescored` file so scores come from `_rescored`
    when available and everything else comes from the original file.
    """
    pat_orig = re.compile(rf"^{re.escape(prefix)}(?:_rerun\d+)?_rep(\d+)\.json$")
    pat_resc = re.compile(rf"^{re.escape(prefix)}(?:_rerun\d+)?_rep(\d+)_rescored\.json$")

    orig_cands: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        r: [] for r in range(1, expected_repeats + 1)
    }
    resc_cands: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        r: [] for r in range(1, expected_repeats + 1)
    }

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    for path in sorted(results_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        m_resc = pat_resc.match(path.name)
        if m_resc:
            rep_num = int(m_resc.group(1))
            if 1 <= rep_num <= expected_repeats:
                data = json.loads(path.read_text(encoding="utf-8"))
                if _is_valid_arm_data(data, require_baseline=require_baseline):
                    resc_cands[rep_num].append((path, data))
            continue

        m_orig = pat_orig.match(path.name)
        if m_orig:
            rep_num = int(m_orig.group(1))
            if 1 <= rep_num <= expected_repeats:
                data = json.loads(path.read_text(encoding="utf-8"))
                if _is_valid_arm_data(data, require_baseline=require_baseline):
                    orig_cands[rep_num].append((path, data))

    resolved: dict[int, tuple[Path, dict[str, Any], Path | None, dict[str, Any] | None]] = {}
    for r in range(1, expected_repeats + 1):
        if not orig_cands[r]:
            raise ValueError(f"Missing valid original repeat file for {prefix} rep {r}")
        orig_path, orig_data = orig_cands[r][-1]
        resc_path: Path | None = None
        resc_data: dict[str, Any] | None = None
        if resc_cands[r]:
            resc_path, resc_data = resc_cands[r][-1]
        resolved[r] = (orig_path, orig_data, resc_path, resc_data)

    return resolved


def _extract_arm_summary(
    arm: str,
    task: str,
    orig_data: dict[str, Any],
    resc_data: dict[str, Any] | None,
) -> ArmSessionSummary:
    arm_key = arm
    if arm_key not in orig_data and arm_key in ("runsift", "cacheforge"):
        arm_key = "runsift" if "runsift" in orig_data else "cacheforge"
    orig_tr = orig_data[arm_key]["task_results"][0]
    if resc_data is not None:
        resc_arm_key = arm if arm in resc_data else ("runsift" if "runsift" in resc_data else "cacheforge")
        score_tr = resc_data.get(resc_arm_key, {}).get("task_results", [orig_tr])[0]
    else:
        score_tr = orig_tr
    is_coupled = task == "hard" or orig_tr.get("task_id") == "GAUNTLET-COUPLED"

    if is_coupled:
        fixed = int(score_tr.get("traps_solved", _parse_mod(score_tr["modules_fixed"])))
    else:
        fixed = _parse_mod(score_tr["modules_fixed"])

    prompt_tokens = int(orig_tr["total_prompt_tokens"])
    completion_tokens = int(orig_tr.get("total_completion_tokens", 0))
    cost_usd = compute_session_cost_usd(prompt_tokens, completion_tokens)

    return ArmSessionSummary(
        arm=arm,
        fixed=fixed,
        rescored=resc_data is not None,
        turns=int(orig_tr["turns"]),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        peak_turn_prompt_tokens=int(orig_tr["peak_turn_prompt_tokens"]),
        cost_usd=cost_usd,
        wall_clock_s=float(orig_tr.get("wall_clock_s", 0.0)),
        context_overflow=bool(orig_tr.get("context_overflow", False)),
        per_turn_prompt_tokens=[int(x) for x in orig_tr.get("per_turn_prompt_tokens", [])],
        per_turn_tools=[str(x) for x in orig_tr.get("per_turn_tools", [])],
        per_turn_tool_args=[str(x) for x in orig_tr.get("per_turn_tool_args", [])],
        per_turn_tests_passing=[
            int(x) if x is not None else None for x in orig_tr.get("per_turn_tests_passing", [])
        ],
        per_turn_condenser=[str(x) for x in orig_tr.get("per_turn_condenser", [])],
        per_turn_tool_chars_before=[int(x) for x in orig_tr.get("per_turn_tool_chars_before", [])],
        per_turn_tool_chars_after=[int(x) for x in orig_tr.get("per_turn_tool_chars_after", [])],
    )


def _sidecar_path_for(orig_path: Path, results_dir: Path) -> Path | None:
    stem = orig_path.name[:-5]  # strip .json
    sc = results_dir / f"{stem}_tool_outputs.jsonl.gz"
    return sc if sc.exists() else None


def load_run_row(spec: RunSpec, results_dir: Path = DEFAULT_RESULTS_DIR) -> RunRowSummary:
    """Loads all 15 repeats for a RunSpec and computes the RunRowSummary."""
    base_map = _resolve_prefix_repeats(spec.baseline_prefix, results_dir, require_baseline=True)
    if spec.runsift_prefix == spec.baseline_prefix:
        cf_map = base_map
    else:
        cf_map = _resolve_prefix_repeats(
            spec.runsift_prefix, results_dir, require_baseline=False
        )

    sessions: list[SessionPairSummary] = []
    for r in range(1, 16):
        b_orig_path, b_orig_data, _b_resc_path, b_resc_data = base_map[r]
        c_orig_path, c_orig_data, _c_resc_path, c_resc_data = cf_map[r]

        b_arm = _extract_arm_summary("baseline", spec.task, b_orig_data, b_resc_data)
        c_arm = _extract_arm_summary("runsift", spec.task, c_orig_data, c_resc_data)

        prov = c_orig_data.get("provenance") or b_orig_data.get("provenance") or {}
        git_commit = str(prov.get("git_commit") or "unknown")
        git_dirty = bool(prov.get("git_dirty", False))
        launch_ts = str(
            prov.get("launch_timestamp")
            or c_orig_data.get("launch_timestamp")
            or b_orig_data.get("launch_timestamp")
            or ""
        )

        sc_cf = _sidecar_path_for(c_orig_path, results_dir) if spec.replayable else None
        sc_base = _sidecar_path_for(b_orig_path, results_dir) if spec.replayable else None

        sessions.append(
            SessionPairSummary(
                spec=spec,
                rep=r,
                git_commit=git_commit,
                git_dirty=git_dirty,
                launch_timestamp=launch_ts,
                baseline=b_arm,
                runsift=c_arm,
                sidecar_path=sc_cf,
                baseline_sidecar_path=sc_base,
            )
        )

    b_prompts = [s.baseline.prompt_tokens for s in sessions]
    c_prompts = [s.runsift.prompt_tokens for s in sessions]
    b_costs = [s.baseline.cost_usd for s in sessions]
    c_costs = [s.runsift.cost_usd for s in sessions]
    diffs = [s.runsift.fixed - s.baseline.fixed for s in sessions]

    b_prompt_mean = float(statistics.mean(b_prompts))
    c_prompt_mean = float(statistics.mean(c_prompts))
    pct_change = round(((c_prompt_mean - b_prompt_mean) / max(1.0, b_prompt_mean)) * 100)

    b_cost_mean = float(statistics.mean(b_costs))
    c_cost_mean = float(statistics.mean(c_costs))
    ratio = b_cost_mean / max(1e-9, c_cost_mean)

    _, p_val = exact_permutation_test(diffs)

    first_launch = sessions[0].launch_timestamp if sessions else ""
    launch_date = first_launch[:10] if len(first_launch) >= 10 else first_launch
    commit_short = sessions[0].git_commit[:7] if sessions else "unknown"

    return RunRowSummary(
        spec=spec,
        repeats=len(sessions),
        baseline_prompt_tokens_mean=b_prompt_mean,
        runsift_prompt_tokens_mean=c_prompt_mean,
        prompt_tokens_change_pct=int(pct_change),
        baseline_cost_mean=b_cost_mean,
        runsift_cost_mean=c_cost_mean,
        sessions_per_dollar_ratio=ratio,
        baseline_fixed_total=sum(s.baseline.fixed for s in sessions),
        runsift_fixed_total=sum(s.runsift.fixed for s in sessions),
        rescored=any(s.baseline.rescored or s.runsift.rescored for s in sessions),
        p_value=p_val,
        git_commit=commit_short,
        launch_date=launch_date,
        sessions=sessions,
    )


def load_all_runs(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[RunRowSummary]:
    """Loads all seven rows of the README results table from committed result files."""
    return [load_run_row(spec, results_dir=results_dir) for spec in RUN_SPECS]


def _load_sidecar_records(sidecar_path: Path, arm: str) -> list[dict[str, Any]]:
    raw_bytes = gzip.decompress(sidecar_path.read_bytes())
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if arm in ("runsift", "cacheforge"):
        return [r for r in rows if str(r.get("arm") or "") in ("runsift", "cacheforge")]
    return [r for r in rows if str(r.get("arm") or "") == arm]


def _is_pre_timeout_fix(git_commit: str) -> bool:
    for c in PRE_TIMEOUT_FIX_COMMITS:
        if c.startswith(git_commit) or git_commit.startswith(c[:7]):
            return True
    return False


def build_session_turn_steps(
    session: SessionPairSummary,
    arm: str = "runsift",
) -> list[TurnStepView]:
    """Builds the turn-by-turn stepper views and header-reproduction badges for one arm of a session.

    Implements Section 4.3 of `docs/specs/TERMINAL_APP_SPEC.md`:
    - Rebuilds history from the sidecar in turn order (assistant tool_call + tool message,
      skipping `none`, `overflow`, `infra_error`, `error`).
    - Right pane ("What the model saw"):
      - with RunSift: condensed now by `CondenserV2(enable_stale_rewrite=(condenser == "v2"))`
        as it was first sent (the fresh digest).
      - without RunSift: `"Sent unchanged (no RunSift)"`.
    - Badge: compares the next request's condensed tool characters (`compact_messages` over the
      rebuilt history) against recorded `per_turn_tool_chars_before/after`.
    """
    if not session.spec.replayable:
        return []

    is_optimized_arm = arm in ("runsift", "cacheforge")
    sc_path = session.sidecar_path if is_optimized_arm else session.baseline_sidecar_path
    if sc_path is None or not sc_path.exists():
        return []

    arm_summary = session.runsift if is_optimized_arm else session.baseline
    records = _load_sidecar_records(sc_path, arm=arm)
    enable_stale = session.spec.condenser == "v2"
    condenser = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=enable_stale)

    rec_before_list = arm_summary.per_turn_tool_chars_before
    rec_after_list = arm_summary.per_turn_tool_chars_after
    tests_list = arm_summary.per_turn_tests_passing
    pre_timeout_commit = _is_pre_timeout_fix(session.git_commit)

    # First compute per-request header verification across all turns 0..N-1
    # Request k (0-indexed, sent at the start of turn k+1) contains history from turns 1..k.
    history: list[dict[str, Any]] = []
    history_has_timeout = False

    req_badges: list[tuple[str, bool, bool]] = []
    num_turns = len(records)

    for k in range(num_turns):
        if not is_optimized_arm:
            req_badges.append((BADGE_BASELINE, False, False))
        elif k < len(rec_before_list) and k < len(rec_after_list):
            _, _, comp_before, comp_after = condenser.compact_messages(history)
            if comp_before == rec_before_list[k] and comp_after == rec_after_list[k]:
                req_badges.append((BADGE_MATCH, True, False))
            elif history_has_timeout and pre_timeout_commit:
                req_badges.append((BADGE_TIMEOUT_FIX, False, True))
            else:
                req_badges.append((BADGE_DIFFERS, False, False))
        else:
            req_badges.append((BADGE_NOT_RECORDED, False, False))

        # Append turn k+1's tool output to history for the next request (k+1)
        rec = records[k]
        t_name = str(rec.get("tool") or "")
        if t_name not in NON_HISTORY_TOOLS:
            t_args = rec.get("args") if isinstance(rec.get("args"), dict) else {}
            raw_out = str(rec.get("raw_output") or "")
            if "[TIMEOUT]" in raw_out:
                history_has_timeout = True
            tc_id = f"call_{k + 1}"
            history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": t_name,
                                "arguments": json.dumps(t_args),
                            },
                        }
                    ],
                }
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": t_name,
                    "content": raw_out,
                }
            )

    steps: list[TurnStepView] = []
    for k, rec in enumerate(records):
        turn_num = int(rec.get("turn") or (k + 1))
        t_name = str(rec.get("tool") or "")
        t_args = rec.get("args") if isinstance(rec.get("args"), dict) else {}
        args_str = json.dumps(t_args) if t_args else "{}"
        raw_out = str(rec.get("raw_output") or "")
        tests_passing = tests_list[k] if k < len(tests_list) else None

        chars_before = len(raw_out)
        if is_optimized_arm:
            if t_name in NON_HISTORY_TOOLS:
                model_saw = raw_out
            else:
                single_hist = [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": t_name, "arguments": args_str},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": t_name,
                        "content": raw_out,
                    },
                ]
                comp_single, _, _, _ = condenser.compact_messages(single_hist)
                model_saw = str(comp_single[1].get("content") or "")
            chars_after = len(model_saw)
        else:
            model_saw = "Sent unchanged (no RunSift)"
            chars_after = chars_before

        if chars_before > 0:
            pct_change = round(((chars_before - chars_after) / chars_before) * 100)
        else:
            pct_change = 0

        if not is_optimized_arm:
            ui_badge = BADGE_BASELINE
        elif k + 1 < len(req_badges):
            ui_badge, _, _ = req_badges[k + 1]
        else:
            ui_badge = BADGE_NOT_RECORDED

        _, req_repro, req_timeout_diff = req_badges[k]

        steps.append(
            TurnStepView(
                turn=turn_num,
                total_turns=num_turns,
                arm=arm,
                tool=t_name,
                args_str=args_str,
                tests_passing=tests_passing,
                raw_output=raw_out,
                model_saw=model_saw,
                chars_before=chars_before,
                chars_after=chars_after,
                pct_change=pct_change,
                badge=ui_badge,
                request_reproduced=req_repro,
                request_has_timeout_diff=req_timeout_diff,
            )
        )

    return steps


def verify_all_replay_headers(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Any]:
    """Verifies header reproduction across all six replayable runs (T2)."""
    runs = [r for r in load_all_runs(results_dir=results_dir) if r.spec.replayable]
    per_run: dict[str, dict[str, Any]] = {}
    total_requests = 0
    total_matched = 0
    total_diffs = 0
    total_timeout_diffs = 0
    unexpected_diffs: list[dict[str, Any]] = []

    for run_row in runs:
        run_key = run_row.spec.key
        r_total = 0
        r_matched = 0
        r_diffs = 0
        r_timeout_diffs = 0

        for sess in run_row.sessions:
            steps = build_session_turn_steps(sess, arm="cacheforge")
            for idx, st in enumerate(steps):
                r_total += 1
                if st.request_reproduced:
                    r_matched += 1
                else:
                    r_diffs += 1
                    if st.request_has_timeout_diff:
                        r_timeout_diffs += 1
                    else:
                        unexpected_diffs.append(
                            {
                                "run": run_key,
                                "rep": sess.rep,
                                "request_index": idx,
                                "turn": st.turn,
                            }
                        )

        per_run[run_key] = {
            "total_requests": r_total,
            "matched_requests": r_matched,
            "diff_requests": r_diffs,
            "timeout_diff_requests": r_timeout_diffs,
        }
        total_requests += r_total
        total_matched += r_matched
        total_diffs += r_diffs
        total_timeout_diffs += r_timeout_diffs

    return {
        "total_requests": total_requests,
        "matched_requests": total_matched,
        "diff_requests": total_diffs,
        "timeout_diff_requests": total_timeout_diffs,
        "unexpected_diffs": unexpected_diffs,
        "per_run": per_run,
    }
