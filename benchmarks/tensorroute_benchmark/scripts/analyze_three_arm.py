"""Three-arm analysis script for Run G (baseline, v2, v2.1) on the original task.

Inputs:
- `prefix_base_v2`: result prefix containing the `baseline` and `runsift` (v2) arms.
- `prefix_v2_1`: result prefix containing the `runsift` (v2.1) arm (with or without a `baseline` arm).
Pairs are matched by repeat number (1..repeats) and `_rescored.json` files are preferred when available.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
    RESULTS_DIR,
    _has_scheduler_failure,
    _parse_mod,
    exact_permutation_test,
)


def is_valid_single_or_dual_repeat_data(
    data: dict[str, Any], require_baseline: bool = True
) -> bool:
    """Returns True if the required arm(s) exist and have no infrastructure error."""
    c_tr = data.get("runsift", data.get("cacheforge", {})).get("task_results", [])
    if not c_tr or bool(c_tr[0].get("infra_error", False)):
        return False
    if require_baseline:
        b_tr = data.get("baseline", {}).get("task_results", [])
        if not b_tr or bool(b_tr[0].get("infra_error", False)):
            return False
    return True


def resolve_arm_repeat_files(
    prefix: str,
    results_dir: Path = RESULTS_DIR,
    expected_repeats: int = 15,
    max_reruns: int | None = 3,
    extra_prefixes: list[str] | None = None,
    require_baseline: bool = True,
) -> tuple[list[tuple[int, Path, dict[str, Any]]], list[Path]]:
    """Resolves per-repeat JSON files for a prefix, preferring `_rescored.json` files."""
    prefixes = [prefix]
    if extra_prefixes:
        for ep in extra_prefixes:
            if ep not in prefixes:
                prefixes.append(ep)

    patterns: list[re.Pattern[str]] = []
    for pfx in prefixes:
        if pfx.endswith("_rescored"):
            base_pfx = pfx[:-9]
            patterns.append(
                re.compile(rf"^{re.escape(base_pfx)}(?:_rerun\d+)?_rep(\d+)_rescored\.json$")
            )
            patterns.append(re.compile(rf"^{re.escape(pfx)}(?:_rerun\d+)?_rep(\d+)\.json$"))
        else:
            patterns.append(
                re.compile(rf"^{re.escape(pfx)}(?:_rerun\d+)?_rep(\d+)(?:_rescored)?\.json$")
            )

    candidates: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        r: [] for r in range(1, expected_repeats + 1)
    }

    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    for path in sorted(results_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        matched_rep: int | None = None
        for pat in patterns:
            m = pat.match(path.name)
            if m:
                matched_rep = int(m.group(1))
                break
        if matched_rep is None:
            continue

        if matched_rep < 1 or matched_rep > expected_repeats:
            raise ValueError(
                f"Unexpected repeat number {matched_rep} outside 1..{expected_repeats} in {path.name}"
            )

        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)

        if "rep" in data and int(data["rep"]) != matched_rep:
            raise ValueError(
                f"Repeat number mismatch in {path.name}: filename indicates rep {matched_rep}, "
                f"JSON payload has rep {data['rep']}"
            )

        candidates[matched_rep].append((path, data))

    selected: list[tuple[int, Path, dict[str, Any]]] = []
    skipped_invalid: list[Path] = []

    for r in range(1, expected_repeats + 1):
        rep_cands = candidates[r]
        rescored_cands = [(p, d) for p, d in rep_cands if p.name.endswith("_rescored.json")]
        orig_cands = [(p, d) for p, d in rep_cands if not p.name.endswith("_rescored.json")]

        valid_rescored = [
            (p, d)
            for p, d in rescored_cands
            if is_valid_single_or_dual_repeat_data(d, require_baseline=require_baseline)
        ]
        invalid_rescored = [
            p
            for p, d in rescored_cands
            if not is_valid_single_or_dual_repeat_data(d, require_baseline=require_baseline)
        ]
        valid_orig = [
            (p, d)
            for p, d in orig_cands
            if is_valid_single_or_dual_repeat_data(d, require_baseline=require_baseline)
        ]
        invalid_orig = [
            p
            for p, d in orig_cands
            if not is_valid_single_or_dual_repeat_data(d, require_baseline=require_baseline)
        ]

        if valid_rescored:
            skipped_invalid.extend(invalid_rescored)
            skipped_invalid.extend(invalid_orig)
            if len(valid_rescored) > 1:
                val_names = [p.name for p, _ in valid_rescored]
                raise ValueError(
                    f"Duplicate at repeat {r}: found {len(valid_rescored)} valid rescored files ({val_names}); expected 1"
                )
            chosen_path, chosen_data = valid_rescored[0]
            selected.append((r, chosen_path, chosen_data))
        else:
            skipped_invalid.extend(invalid_rescored)
            skipped_invalid.extend(invalid_orig)
            if len(valid_orig) == 0:
                inv_names = [p.name for p in (invalid_rescored + invalid_orig)]
                raise ValueError(
                    f"Gap at repeat {r}: found 0 valid files (invalid candidates: {inv_names})"
                )
            if len(valid_orig) > 1:
                val_names = [p.name for p, _ in valid_orig]
                raise ValueError(
                    f"Duplicate at repeat {r}: found {len(valid_orig)} valid files ({val_names}); expected 1"
                )
            chosen_path, chosen_data = valid_orig[0]
            selected.append((r, chosen_path, chosen_data))

    if max_reruns is not None and len(skipped_invalid) > max_reruns:
        inv_names = [p.name for p in skipped_invalid]
        raise ValueError(
            f"Exceeded max infrastructure error re-runs: {len(skipped_invalid)} invalid files "
            f"({inv_names}) > max_reruns={max_reruns}"
        )

    return selected, skipped_invalid


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Computes step-down Holm-Bonferroni adjusted p-values for a family of hypotheses."""
    m = len(p_values)
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for idx, (name, p_raw) in enumerate(sorted_items):
        adj = min(1.0, p_raw * (m - idx))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def _extract_arm_metrics(tr: dict[str, Any], is_coupled: bool = False) -> dict[str, Any]:
    fixed = (
        int(tr.get("traps_solved", _parse_mod(tr["modules_fixed"])))
        if is_coupled
        else _parse_mod(tr["modules_fixed"])
    )
    prompt_tokens = int(tr.get("total_prompt_tokens", 0))
    completion_tokens = int(tr.get("total_completion_tokens", 0))
    cached_tokens = int(tr.get("upstream_cached_tokens", 0))
    uncached_tokens = prompt_tokens - cached_tokens
    turns = int(tr.get("turns", 0))
    tokens_per_turn = prompt_tokens / max(1, turns)

    # Cost at list price ($0.06 / 1M input, $0.24 / 1M output) and 1/10 cached input ($0.006 / 1M cached)
    cost_list = (prompt_tokens * 0.06 + completion_tokens * 0.24) / 1_000_000.0
    cost_cached_1_10 = (
        uncached_tokens * 0.06 + cached_tokens * 0.006 + completion_tokens * 0.24
    ) / 1_000_000.0

    per_turn_ttft = [float(x) for x in tr.get("per_turn_ttft_ms", [])]
    tools = list(tr.get("per_turn_tools", []))
    post_pytest_ttft = [
        per_turn_ttft[i]
        for i in range(1, min(len(tools), len(per_turn_ttft)))
        if tools[i - 1] == "run_pytest"
    ]

    tests_passing = [x for x in tr.get("per_turn_tests_passing", []) if x is not None]
    init_pass = 48 if is_coupled else 33
    passed_final = int(tr.get("passed_tests", 0))
    never_improved = (max(tests_passing, default=init_pass) <= init_pass) and (
        passed_final <= init_pass
    )

    feedbacks = [str(fb) for fb in tr.get("per_turn_tool_feedbacks", [])]
    hung_suite = (
        any("[TIMEOUT]" in fb for fb in feedbacks) or tr.get("final_eval_status") == "timeout"
    )
    early_full_solve = bool(tr.get("solved", False)) and turns < 40

    raw_replies = [str(r) for r in tr.get("per_turn_raw_replies", [])]
    json_plan_replies = sum(
        1
        for idx, r in enumerate(raw_replies)
        if (idx < len(tools) and tools[idx] == "none")
        and r.lstrip().startswith("{")
        and any(k in r for k in ('"analysis"', '"plan"', '"commands"'))
    )

    first_pytest_idx = tools.index("run_pytest") if "run_pytest" in tools else -1
    first_post_pytest_no_tool = (
        first_pytest_idx >= 0
        and (first_pytest_idx + 1) < len(tools)
        and tools[first_pytest_idx + 1] == "none"
    )

    test_files_edited = list(tr.get("test_files_edited", []))
    if not test_files_edited:
        # Also detect from per_turn_tool_args if recorded prior to test_files_edited field
        edited_from_args: set[str] = set()
        for t_name, t_arg_str in zip(tools, tr.get("per_turn_tool_args", []), strict=False):
            if t_name == "edit_file" and t_arg_str:
                try:
                    arg_obj = json.loads(t_arg_str)
                    path_val = str(arg_obj.get("path", "")).strip().replace("\\", "/")
                    if path_val.startswith("tests/"):
                        edited_from_args.add(path_val)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
        test_files_edited = sorted(edited_from_args)

    return {
        "fixed": fixed,
        "turns": turns,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_turn": tokens_per_turn,
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "cost_list": cost_list,
        "cost_cached_1_10": cost_cached_1_10,
        "per_turn_ttft": per_turn_ttft,
        "post_pytest_ttft": post_pytest_ttft,
        "wall_clock_s": float(tr.get("wall_clock_s", 0.0)),
        "model_s": round(sum(float(x) for x in tr.get("per_turn_model_ms", [])) / 1000.0, 3),
        "tool_s": round(sum(float(x) for x in tr.get("per_turn_tool_ms", [])) / 1000.0, 3),
        "run_pytest_calls": sum(1 for t in tools if t == "run_pytest"),
        "search_code_calls": sum(1 for t in tools if t == "search_code"),
        "read_file_calls": sum(1 for t in tools if t == "read_file"),
        "never_improved": never_improved,
        "scheduler_failed": _has_scheduler_failure(tr, is_coupled),
        "overflow": bool(tr.get("context_overflow", False)),
        "hung_suite": hung_suite,
        "early_full_solve": early_full_solve,
        "infra_error": bool(tr.get("infra_error", False)),
        "json_plan_replies": json_plan_replies,
        "first_post_pytest_no_tool": first_post_pytest_no_tool,
        "test_files_edited": test_files_edited,
    }


def _summarize_arm(arm_records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(arm_records)
    total_fixed = sum(r["fixed"] for r in arm_records)
    total_prompt = sum(r["prompt_tokens"] for r in arm_records)
    total_cached = sum(r["cached_tokens"] for r in arm_records)
    all_ttft = [ms for r in arm_records for ms in r["per_turn_ttft"]]
    all_post_pytest_ttft = [ms for r in arm_records for ms in r["post_pytest_ttft"]]

    return {
        "n_sessions": n,
        "total_modules_fixed": total_fixed,
        "max_modules": n * 5,
        "mean_modules_fixed": statistics.mean(r["fixed"] for r in arm_records),
        "sd_modules_fixed": statistics.stdev(r["fixed"] for r in arm_records) if n > 1 else 0.0,
        "prompt_tokens_per_session_mean": statistics.mean(r["prompt_tokens"] for r in arm_records),
        "prompt_tokens_per_turn_mean": statistics.mean(r["tokens_per_turn"] for r in arm_records),
        "prompt_tokens_per_fixed_bug": (total_prompt / total_fixed) if total_fixed > 0 else None,
        "completion_tokens_per_session_mean": statistics.mean(
            r["completion_tokens"] for r in arm_records
        ),
        "uncached_tokens_per_session_mean": statistics.mean(
            r["uncached_tokens"] for r in arm_records
        ),
        "cached_share_pct": (total_cached / max(1, total_prompt)) * 100.0,
        "cost_list_per_session_mean": statistics.mean(r["cost_list"] for r in arm_records),
        "cost_cached_1_10_per_session_mean": statistics.mean(
            r["cost_cached_1_10"] for r in arm_records
        ),
        "median_ttft_ms_overall": statistics.median(all_ttft) if all_ttft else 0.0,
        "median_ttft_ms_post_pytest": (
            statistics.median(all_post_pytest_ttft) if all_post_pytest_ttft else 0.0
        ),
        "wall_clock_s_mean": statistics.mean(r["wall_clock_s"] for r in arm_records),
        "wall_clock_s_median": statistics.median(r["wall_clock_s"] for r in arm_records),
        "model_s_mean": statistics.mean(r["model_s"] for r in arm_records),
        "tool_s_mean": statistics.mean(r["tool_s"] for r in arm_records),
        "run_pytest_calls_mean": statistics.mean(r["run_pytest_calls"] for r in arm_records),
        "search_code_calls_mean": statistics.mean(r["search_code_calls"] for r in arm_records),
        "read_file_calls_mean": statistics.mean(r["read_file_calls"] for r in arm_records),
        "never_improved_sessions": sum(1 for r in arm_records if r["never_improved"]),
        "scheduler_failed_sessions": sum(1 for r in arm_records if r["scheduler_failed"]),
        "overflow_sessions": sum(1 for r in arm_records if r["overflow"]),
        "hung_suite_sessions": sum(1 for r in arm_records if r["hung_suite"]),
        "early_full_solve_sessions": sum(1 for r in arm_records if r["early_full_solve"]),
        "infra_errors": sum(1 for r in arm_records if r["infra_error"]),
        "json_plan_replies_total": sum(r["json_plan_replies"] for r in arm_records),
        "json_plan_sessions": sum(1 for r in arm_records if r["json_plan_replies"] > 0),
        "first_post_pytest_no_tool_sessions": sum(
            1 for r in arm_records if r["first_post_pytest_no_tool"]
        ),
        "first_post_pytest_no_tool_reps": [
            idx + 1 for idx, r in enumerate(arm_records) if r["first_post_pytest_no_tool"]
        ],
        "test_files_edited_sessions": sum(1 for r in arm_records if r["test_files_edited"]),
        "test_files_edited_by_rep": {
            idx + 1: r["test_files_edited"]
            for idx, r in enumerate(arm_records)
            if r["test_files_edited"]
        },
    }


def analyze_three_arm(
    prefix_base_v2: str,
    prefix_v2_1: str,
    results_dir: Path = RESULTS_DIR,
    expected_repeats: int = 15,
    max_reruns: int | None = 3,
    extra_prefixes_base_v2: list[str] | None = None,
    extra_prefixes_v2_1: list[str] | None = None,
) -> dict[str, Any]:
    """Analyzes a three-arm run (baseline, v2, v2.1) matched by repeat number."""
    sel_base_v2, skipped_base_v2 = resolve_arm_repeat_files(
        prefix=prefix_base_v2,
        results_dir=results_dir,
        expected_repeats=expected_repeats,
        max_reruns=max_reruns,
        extra_prefixes=extra_prefixes_base_v2,
        require_baseline=True,
    )
    sel_v2_1, skipped_v2_1 = resolve_arm_repeat_files(
        prefix=prefix_v2_1,
        results_dir=results_dir,
        expected_repeats=expected_repeats,
        max_reruns=max_reruns,
        extra_prefixes=extra_prefixes_v2_1,
        require_baseline=False,
    )

    map_base_v2 = {r: (p, d) for r, p, d in sel_base_v2}
    map_v2_1 = {r: (p, d) for r, p, d in sel_v2_1}

    base_recs: list[dict[str, Any]] = []
    v2_recs: list[dict[str, Any]] = []
    v2_1_recs: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for r in range(1, expected_repeats + 1):
        p_bv2, d_bv2 = map_base_v2[r]
        p_v21, d_v21 = map_v2_1[r]

        b_tr = d_bv2["baseline"]["task_results"][0]
        v2_tr = d_bv2.get("runsift", d_bv2.get("cacheforge", {}))["task_results"][0]
        v2_1_tr = d_v21.get("runsift", d_v21.get("cacheforge", {}))["task_results"][0]

        is_coupled = b_tr.get("task_id") == "GAUNTLET-COUPLED"

        b_m = _extract_arm_metrics(b_tr, is_coupled=is_coupled)
        v2_m = _extract_arm_metrics(v2_tr, is_coupled=is_coupled)
        v2_1_m = _extract_arm_metrics(v2_1_tr, is_coupled=is_coupled)

        base_recs.append(b_m)
        v2_recs.append(v2_m)
        v2_1_recs.append(v2_1_m)

        paired_rows.append(
            {
                "rep": r,
                "file_base_v2": p_bv2.name,
                "file_v2_1": p_v21.name,
                "baseline_fixed": b_m["fixed"],
                "v2_fixed": v2_m["fixed"],
                "v2_1_fixed": v2_1_m["fixed"],
                "diff_g1_v2_1_minus_v2": v2_1_m["fixed"] - v2_m["fixed"],
                "diff_g2_v2_minus_base": v2_m["fixed"] - b_m["fixed"],
                "diff_g3_v2_1_minus_base": v2_1_m["fixed"] - b_m["fixed"],
                "diff_g4_prompt_v2_1_minus_v2": v2_1_m["prompt_tokens"] - v2_m["prompt_tokens"],
            }
        )

    diffs_g1 = [row["diff_g1_v2_1_minus_v2"] for row in paired_rows]
    diffs_g2 = [row["diff_g2_v2_minus_base"] for row in paired_rows]
    diffs_g3 = [row["diff_g3_v2_1_minus_base"] for row in paired_rows]
    diffs_g4 = [row["diff_g4_prompt_v2_1_minus_v2"] for row in paired_rows]

    mean_g1, p_g1_raw = exact_permutation_test(diffs_g1)
    mean_g2, p_g2_raw = exact_permutation_test(diffs_g2)
    mean_g3, p_g3_raw = exact_permutation_test(diffs_g3)
    mean_g4, p_g4_raw = exact_permutation_test(diffs_g4)

    adj_p = holm_bonferroni({"G1": p_g1_raw, "G2": p_g2_raw, "G3": p_g3_raw})

    summary = {
        "repeats": expected_repeats,
        "arms": {
            "baseline": _summarize_arm(base_recs),
            "v2": _summarize_arm(v2_recs),
            "v2_1": _summarize_arm(v2_1_recs),
        },
        "tests": {
            "G1_v2_1_vs_v2_modules": {
                "v2_1_total": sum(r["fixed"] for r in v2_1_recs),
                "v2_total": sum(r["fixed"] for r in v2_recs),
                "mean_diff": mean_g1,
                "raw_p": p_g1_raw,
                "holm_bonferroni_p": adj_p["G1"],
            },
            "G2_v2_vs_baseline_modules": {
                "v2_total": sum(r["fixed"] for r in v2_recs),
                "baseline_total": sum(r["fixed"] for r in base_recs),
                "mean_diff": mean_g2,
                "raw_p": p_g2_raw,
                "holm_bonferroni_p": adj_p["G2"],
            },
            "G3_v2_1_vs_baseline_modules": {
                "v2_1_total": sum(r["fixed"] for r in v2_1_recs),
                "baseline_total": sum(r["fixed"] for r in base_recs),
                "mean_diff": mean_g3,
                "raw_p": p_g3_raw,
                "holm_bonferroni_p": adj_p["G3"],
            },
            "G4_v2_1_vs_v2_prompt_tokens": {
                "v2_1_mean": statistics.mean(r["prompt_tokens"] for r in v2_1_recs),
                "v2_mean": statistics.mean(r["prompt_tokens"] for r in v2_recs),
                "mean_diff": mean_g4,
                "raw_p": p_g4_raw,
            },
        },
        "skipped_infra_files": {
            "base_v2": [p.name for p in skipped_base_v2],
            "v2_1": [p.name for p in skipped_v2_1],
        },
        "paired_rows": paired_rows,
    }
    return summary


def print_three_arm_report(summary: dict[str, Any]) -> None:
    n = summary["repeats"]
    max_m = n * 5
    b = summary["arms"]["baseline"]
    v2 = summary["arms"]["v2"]
    v21 = summary["arms"]["v2_1"]
    t = summary["tests"]

    print(f"=== THREE-ARM RUN G ANALYSIS ({n} paired repeats) ===")
    print("Rep | Baseline |   v2   |  v2.1  | v2.1-v2 | v2-Base | v2.1-Base")
    print("-" * 68)
    for row in summary["paired_rows"]:
        print(
            f"{row['rep']:3d} | {row['baseline_fixed']:8d} | {row['v2_fixed']:6d} | "
            f"{row['v2_1_fixed']:6d} | {row['diff_g1_v2_1_minus_v2']:+7d} | "
            f"{row['diff_g2_v2_minus_base']:+7d} | {row['diff_g3_v2_1_minus_base']:+9d}"
        )

    print("\n=== PRIMARY & TOKEN HYPOTHESIS TESTS (Paired Exact Permutation, Two-Sided) ===")
    g1 = t["G1_v2_1_vs_v2_modules"]
    g2 = t["G2_v2_vs_baseline_modules"]
    g3 = t["G3_v2_1_vs_baseline_modules"]
    g4 = t["G4_v2_1_vs_v2_prompt_tokens"]

    print(
        f"G1 (v2.1 vs v2 modules):       {g1['v2_1_total']}/{max_m} vs {g1['v2_total']}/{max_m} "
        f"(mean diff {g1['mean_diff']:+.4f}) | raw p = {g1['raw_p']:.4f} ({g1['raw_p']:.6f}) | "
        f"Holm-Bonferroni adjusted p = {g1['holm_bonferroni_p']:.4f} ({g1['holm_bonferroni_p']:.6f})"
    )
    print(
        f"G2 (v2 vs baseline modules):   {g2['baseline_total']} vs {g2['v2_total']} "
        f"({g2['v2_total']}/{max_m} vs {g2['baseline_total']}/{max_m}, mean diff {g2['mean_diff']:+.4f}) | "
        f"raw p = {g2['raw_p']:.4f} ({g2['raw_p']:.6f}) | "
        f"Holm-Bonferroni adjusted p = {g2['holm_bonferroni_p']:.4f} ({g2['holm_bonferroni_p']:.6f})"
    )
    print(
        f"G3 (v2.1 vs baseline modules): {g3['baseline_total']} vs {g3['v2_1_total']} "
        f"({g3['v2_1_total']}/{max_m} vs {g3['baseline_total']}/{max_m}, mean diff {g3['mean_diff']:+.4f}) | "
        f"raw p = {g3['raw_p']:.4f} ({g3['raw_p']:.6f}) | "
        f"Holm-Bonferroni adjusted p = {g3['holm_bonferroni_p']:.4f} ({g3['holm_bonferroni_p']:.6f})"
    )
    print(
        f"G4 (v2.1 vs v2 prompt tokens): {g4['v2_1_mean']:.1f} vs {g4['v2_mean']:.1f} "
        f"(mean diff {g4['mean_diff']:+.1f}) | p = {g4['raw_p']:.4f} ({g4['raw_p']:.6f}) (unadjusted)"
    )

    print("\n=== SECTION 13.6 SECONDARY METRICS, JSON-PLAN REPLIES & TEST FILES EDITED ===")
    for label, arm_d in (("Baseline", b), ("v2", v2), ("v2.1", v21)):
        pt_bug = (
            f"{arm_d['prompt_tokens_per_fixed_bug']:.1f}"
            if arm_d["prompt_tokens_per_fixed_bug"] is not None
            else "N/A"
        )
        print(
            f"[{label:8s}] Modules: {arm_d['total_modules_fixed']}/{max_m} "
            f"(mean {arm_d['mean_modules_fixed']:.2f} ± {arm_d['sd_modules_fixed']:.2f}) | "
            f"PromptTok/Sess: {arm_d['prompt_tokens_per_session_mean']:.1f} | "
            f"PromptTok/Turn: {arm_d['prompt_tokens_per_turn_mean']:.1f} | "
            f"PromptTok/Bug: {pt_bug} | "
            f"CompTok/Sess: {arm_d['completion_tokens_per_session_mean']:.1f}"
        )
        print(
            f"           UncachedTok/Sess: {arm_d['uncached_tokens_per_session_mean']:.1f} | "
            f"CachedShare: {arm_d['cached_share_pct']:.1f}% | "
            f"Cost(list): ${arm_d['cost_list_per_session_mean']:.4f} | "
            f"Cost(1/10 cached): ${arm_d['cost_cached_1_10_per_session_mean']:.4f}"
        )
        print(
            f"           MedianTTFT(overall): {arm_d['median_ttft_ms_overall']:.1f}ms | "
            f"MedianTTFT(post-pytest): {arm_d['median_ttft_ms_post_pytest']:.1f}ms | "
            f"WallClock: {arm_d['wall_clock_s_mean']:.2f}s (med {arm_d['wall_clock_s_median']:.2f}s) | "
            f"ModelTime: {arm_d['model_s_mean']:.2f}s | ToolTime: {arm_d['tool_s_mean']:.2f}s"
        )
        print(
            f"           Calls/Sess (pytest/search/read): {arm_d['run_pytest_calls_mean']:.2f} / "
            f"{arm_d['search_code_calls_mean']:.2f} / {arm_d['read_file_calls_mean']:.2f} | "
            f"NeverImproved: {arm_d['never_improved_sessions']}/{n} | "
            f"SchedFailed: {arm_d['scheduler_failed_sessions']}/{n} | "
            f"Overflows: {arm_d['overflow_sessions']}/{n} | "
            f"HungSuite: {arm_d['hung_suite_sessions']}/{n} | "
            f"EarlyFullSolve: {arm_d['early_full_solve_sessions']}/{n} | "
            f"InfraErrors: {arm_d['infra_errors']}"
        )
        print(
            f"           JSON-plan replies: {arm_d['json_plan_replies_total']} "
            f"across {arm_d['json_plan_sessions']}/{n} sessions | "
            f"FirstPostPytestNoTool: {arm_d['first_post_pytest_no_tool_sessions']}/{n} sessions "
            f"{arm_d['first_post_pytest_no_tool_reps']} | "
            f"test_files_edited: {arm_d['test_files_edited_sessions']}/{n} sessions "
            f"{arm_d['test_files_edited_by_rep']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Run G three-arm results (baseline, v2, v2.1)."
    )
    parser.add_argument(
        "prefix_base_v2",
        help="Result prefix containing baseline and v2 arms",
    )
    parser.add_argument(
        "prefix_v2_1",
        help="Result prefix containing v2.1 runsift arm",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=15,
        help="Expected number of paired repeats (default: 15)",
    )
    parser.add_argument(
        "--max-reruns",
        type=int,
        default=3,
        help="Maximum allowed invalid infrastructure-error files skipped (default: 3)",
    )
    parser.add_argument(
        "--extra-prefix-base-v2",
        action="append",
        default=[],
        help="Additional rerun prefix(es) for baseline+v2",
    )
    parser.add_argument(
        "--extra-prefix-v2-1",
        action="append",
        default=[],
        help="Additional rerun prefix(es) for v2.1",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    args = parser.parse_args()

    summary = analyze_three_arm(
        prefix_base_v2=args.prefix_base_v2,
        prefix_v2_1=args.prefix_v2_1,
        results_dir=args.results_dir,
        expected_repeats=args.repeats,
        max_reruns=args.max_reruns,
        extra_prefixes_base_v2=args.extra_prefix_base_v2,
        extra_prefixes_v2_1=args.extra_prefix_v2_1,
    )
    print_three_arm_report(summary)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
