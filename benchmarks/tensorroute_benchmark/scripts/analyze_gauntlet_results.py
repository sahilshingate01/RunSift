from __future__ import annotations

import argparse
import json
import re
import statistics
from itertools import product
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("benchmarks/tensorroute_benchmark/results")


def is_valid_repeat_data(data: dict[str, Any]) -> bool:
    """Returns True if the per-repeat JSON payload has no infrastructure error in either arm.

    Note: Top-level `is_valid` is NOT checked here because the harness also sets
    `is_valid = False` when `nudge_rate_pct > 20.0` (a stuck agent). Stuck-agent
    sessions are model behavior, not infrastructure failures: they must never be
    re-run, must remain in the primary analysis with their real scores, and are
    excluded only from the valid-only secondary analysis.
    """
    b_tr = data.get("baseline", {}).get("task_results", [])
    c_tr = data.get("runsift", data.get("cacheforge", {})).get("task_results", [])
    if not b_tr or not c_tr:
        return False
    return not (b_tr[0].get("infra_error", False) or c_tr[0].get("infra_error", False))


def resolve_repeat_files(
    prefix: str,
    results_dir: Path = RESULTS_DIR,
    expected_repeats: int = 15,
    max_reruns: int | None = 3,
    extra_prefixes: list[str] | None = None,
    print_substitutions: bool = True,
) -> tuple[list[tuple[int, Path, dict[str, Any]]], list[Path]]:
    """Resolves per-repeat JSON files for a given prefix and its re-runs.

    Enforces:
    - Every repeat number in 1..expected_repeats has at least one non-infra-error file (fails on gap).
    - Every repeat number in 1..expected_repeats has at most one non-infra-error file (fails on duplicate).
    - When both `_repN.json` and `_repN_rescored.json` exist for repeat N, prefers `_repN_rescored.json`.
    - No matched file has a repeat number outside 1..expected_repeats.
    - If a file specifies internal 'rep', it must match the filename repeat number.
    - Total invalid (infra-error) files skipped across the run must not exceed max_reruns.
    """
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
    substituted_rescored: list[tuple[int, str, str]] = []

    for r in range(1, expected_repeats + 1):
        rep_cands = candidates[r]
        rescored_cands = [(p, d) for p, d in rep_cands if p.name.endswith("_rescored.json")]
        orig_cands = [(p, d) for p, d in rep_cands if not p.name.endswith("_rescored.json")]

        valid_rescored = [(p, d) for p, d in rescored_cands if is_valid_repeat_data(d)]
        invalid_rescored = [p for p, d in rescored_cands if not is_valid_repeat_data(d)]
        valid_orig = [(p, d) for p, d in orig_cands if is_valid_repeat_data(d)]
        invalid_orig = [p for p, d in orig_cands if not is_valid_repeat_data(d)]

        if valid_rescored:
            skipped_invalid.extend(invalid_rescored)
            skipped_invalid.extend(invalid_orig)
            if len(valid_rescored) > 1:
                val_names = [p.name for p, _ in valid_rescored]
                raise ValueError(
                    f"Duplicate at repeat {r}: found {len(valid_rescored)} valid rescored files ({val_names}); "
                    "expected exactly 1"
                )
            chosen_path, chosen_data = valid_rescored[0]
            orig_names = [p.name for p, _ in orig_cands]
            if orig_names:
                substituted_rescored.append((r, orig_names[-1], chosen_path.name))
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
                    f"Duplicate at repeat {r}: found {len(valid_orig)} valid files ({val_names}); "
                    "expected exactly 1"
                )
            chosen_path, chosen_data = valid_orig[0]
            selected.append((r, chosen_path, chosen_data))

    if print_substitutions and substituted_rescored:
        sub_str = ", ".join(f"rep{r} ({orig} -> {resc})" for r, orig, resc in substituted_rescored)
        print(f"Substituted rescored repeat files ({len(substituted_rescored)}): {sub_str}")

    if max_reruns is not None and len(skipped_invalid) > max_reruns:
        inv_names = [p.name for p in skipped_invalid]
        raise ValueError(
            f"Exceeded max infrastructure error re-runs: {len(skipped_invalid)} invalid files "
            f"({inv_names}) > max_reruns={max_reruns}"
        )

    return selected, skipped_invalid


def _parse_mod(s: Any) -> int:
    if isinstance(s, str) and "/" in s:
        return int(s.split("/")[0])
    return int(s)


def _has_scheduler_failure(res: dict[str, Any], is_coupled: bool) -> bool:
    """Returns True if the session had any failing tests/test_scheduler.py test in final evaluation."""
    if "failing_subsystems" in res and isinstance(res["failing_subsystems"], list):
        return "scheduler" in res["failing_subsystems"]
    if is_coupled:
        solved_traps = res.get("solved_traps", [])
        return "T3" not in solved_traps
    if bool(res.get("solved", False)):
        return False
    for fb in reversed(res.get("per_turn_tool_feedbacks", [])):
        if "passed" in fb or "failed" in fb or "FAILED" in fb:
            if "test_scheduler.py" in fb:
                return True
            if "passed" in fb and "failed" not in fb and "FAILED" not in fb:
                return False
            break
    return _parse_mod(res.get("modules_fixed", 0)) < 5


def analyze(
    prefix: str = "local_gauntlet_gemma4_26b_40turn",
    results_dir: Path = RESULTS_DIR,
    expected_repeats: int = 15,
    max_reruns: int | None = 3,
    extra_prefixes: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected, _skipped = resolve_repeat_files(
        prefix=prefix,
        results_dir=results_dir,
        expected_repeats=expected_repeats,
        max_reruns=max_reruns,
        extra_prefixes=extra_prefixes,
        print_substitutions=False,
    )

    records: list[dict[str, Any]] = []
    for r, f, data in selected:
        b = data["baseline"]
        c = data.get("runsift", data.get("cacheforge", {}))
        b_res = b["task_results"][0]
        c_res = c["task_results"][0]

        is_coupled = b_res.get("task_id") == "GAUNTLET-COUPLED"

        if is_coupled:
            b_fixed = int(b_res.get("traps_solved", _parse_mod(b_res["modules_fixed"])))
            c_fixed = int(c_res.get("traps_solved", _parse_mod(c_res["modules_fixed"])))
        else:
            b_fixed = _parse_mod(b_res["modules_fixed"])
            c_fixed = _parse_mod(c_res["modules_fixed"])

        b_nudges = int(b_res.get("nudge_count", b.get("nudge_count", 0)))
        c_nudges = int(c_res.get("nudge_count", c.get("nudge_count", 0)))
        b_nudge_rate = float(b_res.get("nudge_rate_pct", b.get("nudge_rate_pct", 0.0)))
        c_nudge_rate = float(c_res.get("nudge_rate_pct", c.get("nudge_rate_pct", 0.0)))

        is_valid = is_valid_repeat_data(data) and (b_nudge_rate <= 20.0) and (c_nudge_rate <= 20.0)

        def count_fixes_by_turn(tests_list: list[int | None], init_pass: int) -> tuple[int, int]:
            fixes_le_20 = 0
            fixes_gt_20 = 0
            prev = init_pass
            for turn, val in enumerate(tests_list, start=1):
                if val is not None and val > prev:
                    delta_tests = val - prev
                    bugs = 1 if delta_tests <= 2 else round(delta_tests / 2)
                    if turn <= 20:
                        fixes_le_20 += bugs
                    else:
                        fixes_gt_20 += bugs
                    prev = val
            return fixes_le_20, fixes_gt_20

        init_pass = 48 if is_coupled else 33
        b_le_20, b_gt_20 = count_fixes_by_turn(b_res.get("per_turn_tests_passing", []), init_pass)
        c_le_20, c_gt_20 = count_fixes_by_turn(c_res.get("per_turn_tests_passing", []), init_pass)

        b_prompt = int(b_res["total_prompt_tokens"])
        c_prompt = int(c_res["total_prompt_tokens"])
        b_comp = int(b_res.get("total_completion_tokens", 0))
        c_comp = int(c_res.get("total_completion_tokens", 0))
        b_cached = int(b_res["upstream_cached_tokens"])
        c_cached = int(c_res["upstream_cached_tokens"])
        b_uncached = b_prompt - b_cached
        c_uncached = c_prompt - c_cached
        b_turns = int(b_res["turns"])
        c_turns = int(c_res["turns"])

        b_tools = list(b_res.get("per_turn_tools", []))
        c_tools = list(c_res.get("per_turn_tools", []))
        b_read_file_calls = sum(1 for t in b_tools if t == "read_file")
        c_read_file_calls = sum(1 for t in c_tools if t == "read_file")
        b_wall_clock_s = float(b_res.get("wall_clock_s", 0.0))
        c_wall_clock_s = float(c_res.get("wall_clock_s", 0.0))
        b_model_s = round(sum(float(x) for x in b_res.get("per_turn_model_ms", [])) / 1000.0, 3)
        c_model_s = round(sum(float(x) for x in c_res.get("per_turn_model_ms", [])) / 1000.0, 3)
        b_tool_s = round(sum(float(x) for x in b_res.get("per_turn_tool_ms", [])) / 1000.0, 3)
        c_tool_s = round(sum(float(x) for x in c_res.get("per_turn_tool_ms", [])) / 1000.0, 3)
        b_scheduler_failed = _has_scheduler_failure(b_res, is_coupled)
        c_scheduler_failed = _has_scheduler_failure(c_res, is_coupled)

        rec: dict[str, Any] = {
            "rep": r,
            "file": f.name,
            "is_coupled": is_coupled,
            "b_fixed": b_fixed,
            "c_fixed": c_fixed,
            "diff": c_fixed - b_fixed,
            "b_turns": b_turns,
            "c_turns": c_turns,
            "b_overflow": bool(b_res["context_overflow"]),
            "c_overflow": bool(c_res["context_overflow"]),
            "b_prompt_tokens": b_prompt,
            "c_prompt_tokens": c_prompt,
            "b_completion_tokens": b_comp,
            "c_completion_tokens": c_comp,
            "b_tokens_per_turn": b_prompt / max(1, b_turns),
            "c_tokens_per_turn": c_prompt / max(1, c_turns),
            "b_peak_tokens": int(b_res["peak_turn_prompt_tokens"]),
            "c_peak_tokens": int(c_res["peak_turn_prompt_tokens"]),
            "b_cached_tokens": b_cached,
            "c_cached_tokens": c_cached,
            "b_uncached": b_uncached,
            "c_uncached": c_uncached,
            "b_nudges": b_nudges,
            "c_nudges": c_nudges,
            "b_nudge_rate": b_nudge_rate,
            "c_nudge_rate": c_nudge_rate,
            "valid": is_valid,
            "b_ttft": float(b_res["avg_ttft_ms"]),
            "c_ttft": float(c_res["avg_ttft_ms"]),
            "b_le_20": b_le_20,
            "b_gt_20": b_gt_20,
            "c_le_20": c_le_20,
            "c_gt_20": c_gt_20,
            "b_wall_clock_s": b_wall_clock_s,
            "c_wall_clock_s": c_wall_clock_s,
            "b_model_s": b_model_s,
            "c_model_s": c_model_s,
            "b_tool_s": b_tool_s,
            "c_tool_s": c_tool_s,
            "b_read_file_calls": b_read_file_calls,
            "c_read_file_calls": c_read_file_calls,
            "b_scheduler_failed": b_scheduler_failed,
            "c_scheduler_failed": c_scheduler_failed,
        }

        if is_coupled:
            rec["b_solved_traps"] = list(b_res.get("solved_traps", []))
            rec["c_solved_traps"] = list(c_res.get("solved_traps", []))
            rec["b_reg_intro"] = len(b_res.get("regressions_introduced", []))
            rec["c_reg_intro"] = len(c_res.get("regressions_introduced", []))
            rec["b_reg_rep"] = int(b_res.get("regressions_repaired", 0))
            rec["c_reg_rep"] = int(c_res.get("regressions_repaired", 0))
            rec["b_p2p_intact"] = bool(b_res.get("final_pass_to_pass_intact", False))
            rec["c_p2p_intact"] = bool(c_res.get("final_pass_to_pass_intact", False))

        records.append(rec)
    return records


def exact_permutation_test(diffs: list[int | float]) -> tuple[float, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-repeat TensorRoute-Bench results with gap/duplicate validation."
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default="local_gauntlet_gemma4_26b_40turn",
        help="Result file prefix (e.g. nebius_gauntlet_nemotron_30b_v3)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing per-repeat result JSON files",
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
        "--extra-prefix",
        action="append",
        default=[],
        help="Additional rerun prefix(es) to include when resolving repeats",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write computed aggregate metrics as JSON",
    )
    args = parser.parse_args()

    selected, skipped = resolve_repeat_files(
        prefix=args.prefix,
        results_dir=args.results_dir,
        expected_repeats=args.repeats,
        max_reruns=args.max_reruns,
        extra_prefixes=args.extra_prefix,
    )
    records = analyze(
        prefix=args.prefix,
        results_dir=args.results_dir,
        expected_repeats=args.repeats,
        max_reruns=args.max_reruns,
        extra_prefixes=args.extra_prefix,
    )

    is_coupled = records[0]["is_coupled"] if records else False
    unit_label = "Traps Solved" if is_coupled else "Modules Fixed"
    n_reps = len(records)

    print(
        f"=== ANALYZING: {args.prefix} ({n_reps} repeats, task_set={'coupled' if is_coupled else 'original'}) ==="
    )
    if skipped:
        print(f"Skipped invalid (infra-error) files ({len(skipped)}): {[p.name for p in skipped]}")
    print("=== PER REPEAT RESULTS ===")
    print(
        f"{'Rep':<4} | {'File':<44} | {'Base':<5} | {'Turns':<5} | {'Ovfl':<5} | {'PeakTok':<8} | {'CF':<5} | {'Turns':<5} | {'Ovfl':<5} | {'PeakTok':<8} | {'Diff':<5} | {'Valid'}"
    )
    print("-" * 133)
    for r in records:
        print(
            f"{r['rep']:<4} | {r['file']:<44} | {r['b_fixed']:<5} | {r['b_turns']:<5} | {r['b_overflow']!s:<5} | {r['b_peak_tokens']:<8} | {r['c_fixed']:<5} | {r['c_turns']:<5} | {r['c_overflow']!s:<5} | {r['c_peak_tokens']:<8} | {r['diff']:+d}    | {r['valid']!s}"
        )

    print(f"\n=== TOTALS & PRIMARY TEST (ALL {n_reps} REPEATS) ===")
    b_total = sum(r["b_fixed"] for r in records)
    c_total = sum(r["c_fixed"] for r in records)
    diffs_all = [r["diff"] for r in records]
    mean_diff_all, p_val_all = exact_permutation_test(diffs_all)
    b_std = float(statistics.stdev([r["b_fixed"] for r in records])) if n_reps > 1 else 0.0
    c_std = float(statistics.stdev([r["c_fixed"] for r in records])) if n_reps > 1 else 0.0
    print(
        f"Baseline total {unit_label.lower()}:   {b_total}/{n_reps * 5} (mean {statistics.mean([r['b_fixed'] for r in records]):.2f} ± {b_std:.2f})"
    )
    print(
        f"RunSift total {unit_label.lower()}:   {c_total}/{n_reps * 5} (mean {statistics.mean([r['c_fixed'] for r in records]):.2f} ± {c_std:.2f})"
    )
    print(f"Mean Difference (RunSift - Baseline): {mean_diff_all:+.4f}")
    print(f"Pre-registered Paired Permutation Test p-value: {p_val_all:.6f} (alpha = 0.05)")

    valid_records = [r for r in records if r["valid"]]
    diffs_valid = [r["diff"] for r in valid_records]
    mean_diff_valid, p_val_valid = exact_permutation_test(diffs_valid)
    print(f"\n=== SECONDARY TEST (VALID-ONLY REPEATS: {len(valid_records)}/{n_reps}) ===")
    print(
        f"Baseline {unit_label.lower()} (valid-only):   {sum(r['b_fixed'] for r in valid_records)}/{len(valid_records) * 5}"
    )
    print(
        f"RunSift {unit_label.lower()} (valid-only):    {sum(r['c_fixed'] for r in valid_records)}/{len(valid_records) * 5}"
    )
    print(f"Valid-only Mean Difference: {mean_diff_valid:+.4f}, p-value: {p_val_valid:.6f}")

    print("\n=== SECONDARY METRICS: TOKENS, CACHING, TTFT, OVERFLOWS ===")
    b_prompt = [r["b_prompt_tokens"] for r in records]
    c_prompt = [r["c_prompt_tokens"] for r in records]
    b_comp = [r["b_completion_tokens"] for r in records]
    c_comp = [r["c_completion_tokens"] for r in records]
    b_tpt = [r["b_tokens_per_turn"] for r in records]
    c_tpt = [r["c_tokens_per_turn"] for r in records]
    b_cached = [r["b_cached_tokens"] for r in records]
    c_cached = [r["c_cached_tokens"] for r in records]
    b_unc = [r["b_uncached"] for r in records]
    c_unc = [r["c_uncached"] for r in records]
    b_ttft = [r["b_ttft"] for r in records]
    c_ttft = [r["c_ttft"] for r in records]
    b_ovf = sum(1 for r in records if r["b_overflow"])
    c_ovf = sum(1 for r in records if r["c_overflow"])
    b_wc = [r["b_wall_clock_s"] for r in records]
    c_wc = [r["c_wall_clock_s"] for r in records]
    b_mod_s = [r["b_model_s"] for r in records]
    c_mod_s = [r["c_model_s"] for r in records]
    b_tl_s = [r["b_tool_s"] for r in records]
    c_tl_s = [r["c_tool_s"] for r in records]
    b_rf = [r["b_read_file_calls"] for r in records]
    c_rf = [r["c_read_file_calls"] for r in records]
    b_sf = sum(1 for r in records if r["b_scheduler_failed"])
    c_sf = sum(1 for r in records if r["c_scheduler_failed"])

    b_tokens_per_solve = sum(b_prompt) / b_total if b_total > 0 else float("inf")
    c_tokens_per_solve = sum(c_prompt) / c_total if c_total > 0 else float("inf")

    print(f"Context Overflows:          Baseline {b_ovf}/{n_reps} | RunSift {c_ovf}/{n_reps}")
    print(
        f"Prompt Tokens / Session:    Baseline {statistics.mean(b_prompt):.1f} ± {statistics.stdev(b_prompt) if n_reps > 1 else 0.0:.1f} | "
        f"RunSift {statistics.mean(c_prompt):.1f} ± {statistics.stdev(c_prompt) if n_reps > 1 else 0.0:.1f} "
        f"({(statistics.mean(c_prompt) - statistics.mean(b_prompt)) / max(1.0, float(statistics.mean(b_prompt))) * 100:+.2f}%)"
    )
    print(
        f"Completion Tokens / Sess:   Baseline {statistics.mean(b_comp):.1f} ± {statistics.stdev(b_comp) if n_reps > 1 else 0.0:.1f} | "
        f"RunSift {statistics.mean(c_comp):.1f} ± {statistics.stdev(c_comp) if n_reps > 1 else 0.0:.1f}"
    )
    print(
        f"Prompt Tokens / Turn:       Baseline {statistics.mean(b_tpt):.1f} ± {statistics.stdev(b_tpt) if n_reps > 1 else 0.0:.1f} | "
        f"RunSift {statistics.mean(c_tpt):.1f} ± {statistics.stdev(c_tpt) if n_reps > 1 else 0.0:.1f}"
    )
    print(
        f"Prompt Tokens / Solved Bug: Baseline {b_tokens_per_solve:.1f} | RunSift {c_tokens_per_solve:.1f}"
    )
    print(
        f"Server Cached Tokens / Sess:Baseline {statistics.mean(b_cached):.1f} ({sum(b_cached) / max(1, sum(b_prompt)) * 100:.1f}% of prompt) | "
        f"RunSift {statistics.mean(c_cached):.1f} ({sum(c_cached) / max(1, sum(c_prompt)) * 100:.1f}% of prompt)"
    )
    print(
        f"Uncached Tokens / Session:  Baseline {statistics.mean(b_unc):.1f} ± {statistics.stdev(b_unc) if n_reps > 1 else 0.0:.1f} | "
        f"RunSift {statistics.mean(c_unc):.1f} ± {statistics.stdev(c_unc) if n_reps > 1 else 0.0:.1f}"
    )
    print(
        f"Wire TTFT (ms):             Baseline {statistics.mean(b_ttft):.2f} ± {statistics.stdev(b_ttft) if n_reps > 1 else 0.0:.2f} | "
        f"RunSift {statistics.mean(c_ttft):.2f} ± {statistics.stdev(c_ttft) if n_reps > 1 else 0.0:.2f}"
    )
    print(
        f"Wall Clock / Session (s):   Baseline {statistics.mean(b_wc):.2f} ± {statistics.stdev(b_wc) if n_reps > 1 else 0.0:.2f} (median {statistics.median(b_wc):.2f}) | "
        f"RunSift {statistics.mean(c_wc):.2f} ± {statistics.stdev(c_wc) if n_reps > 1 else 0.0:.2f} (median {statistics.median(c_wc):.2f})"
    )
    print(
        f"Model Time / Session (s):   Baseline {statistics.mean(b_mod_s):.2f} ± {statistics.stdev(b_mod_s) if n_reps > 1 else 0.0:.2f} | "
        f"RunSift {statistics.mean(c_mod_s):.2f} ± {statistics.stdev(c_mod_s) if n_reps > 1 else 0.0:.2f}"
    )
    print(
        f"Tool Time / Session (s):    Baseline {statistics.mean(b_tl_s):.2f} ± {statistics.stdev(b_tl_s) if n_reps > 1 else 0.0:.2f} | "
        f"RunSift {statistics.mean(c_tl_s):.2f} ± {statistics.stdev(c_tl_s) if n_reps > 1 else 0.0:.2f}"
    )
    print(
        f"read_file Calls / Session:  Baseline {statistics.mean(b_rf):.2f} ± {statistics.stdev(b_rf) if n_reps > 1 else 0.0:.2f} | "
        f"RunSift {statistics.mean(c_rf):.2f} ± {statistics.stdev(c_rf) if n_reps > 1 else 0.0:.2f}"
    )
    print(f"Scheduler Failed Sessions:  Baseline {b_sf}/{n_reps} | RunSift {c_sf}/{n_reps}")

    if is_coupled:
        print("\n=== COUPLED REGRESSION METRICS ===")
        b_ri = sum(r["b_reg_intro"] for r in records)
        c_ri = sum(r["c_reg_intro"] for r in records)
        b_rr = sum(r["b_reg_rep"] for r in records)
        c_rr = sum(r["c_reg_rep"] for r in records)
        b_p2p = sum(1 for r in records if r["b_p2p_intact"])
        c_p2p = sum(1 for r in records if r["c_p2p_intact"])
        print(f"Regressions Introduced:     Baseline {b_ri} | RunSift {c_ri}")
        print(f"Regressions Repaired:       Baseline {b_rr} | RunSift {c_rr}")
        print(
            f"Final Pass-to-Pass Intact:  Baseline {b_p2p}/{n_reps} | RunSift {c_p2p}/{n_reps}"
        )

    if args.json_out:
        out_payload = {
            "prefix": args.prefix,
            "repeats": n_reps,
            "task_set": "coupled" if is_coupled else "original",
            "selected_files": [p.name for _, p, _ in selected],
            "skipped_invalid_files": [p.name for p in skipped],
            "baseline_total_solved": b_total,
            "runsift_total_solved": c_total,
            "cacheforge_total_solved": c_total,
            "total_possible": n_reps * 5,
            "mean_diff": round(mean_diff_all, 6),
            "p_value": round(p_val_all, 6),
            "valid_only_repeats": len(valid_records),
            "valid_only_baseline_total_solved": sum(r["b_fixed"] for r in valid_records),
            "valid_only_runsift_total_solved": sum(r["c_fixed"] for r in valid_records),
            "valid_only_cacheforge_total_solved": sum(r["c_fixed"] for r in valid_records),
            "valid_only_mean_diff": round(mean_diff_valid, 6),
            "valid_only_p_value": round(p_val_valid, 6),
            "baseline_overflows": b_ovf,
            "runsift_overflows": c_ovf,
            "cacheforge_overflows": c_ovf,
            "baseline_prompt_tokens_mean": round(float(statistics.mean(b_prompt)), 2),
            "runsift_prompt_tokens_mean": round(float(statistics.mean(c_prompt)), 2),
            "cacheforge_prompt_tokens_mean": round(float(statistics.mean(c_prompt)), 2),
            "baseline_completion_tokens_mean": round(float(statistics.mean(b_comp)), 2),
            "runsift_completion_tokens_mean": round(float(statistics.mean(c_comp)), 2),
            "cacheforge_completion_tokens_mean": round(float(statistics.mean(c_comp)), 2),
            "baseline_tokens_per_turn_mean": round(float(statistics.mean(b_tpt)), 2),
            "runsift_tokens_per_turn_mean": round(float(statistics.mean(c_tpt)), 2),
            "cacheforge_tokens_per_turn_mean": round(float(statistics.mean(c_tpt)), 2),
            "baseline_tokens_per_solve": round(b_tokens_per_solve, 2) if b_total > 0 else None,
            "runsift_tokens_per_solve": round(c_tokens_per_solve, 2) if c_total > 0 else None,
            "cacheforge_tokens_per_solve": round(c_tokens_per_solve, 2) if c_total > 0 else None,
            "baseline_cached_tokens_mean": round(float(statistics.mean(b_cached)), 2),
            "runsift_cached_tokens_mean": round(float(statistics.mean(c_cached)), 2),
            "cacheforge_cached_tokens_mean": round(float(statistics.mean(c_cached)), 2),
            "baseline_wall_clock_s_mean": round(float(statistics.mean(b_wc)), 3),
            "runsift_wall_clock_s_mean": round(float(statistics.mean(c_wc)), 3),
            "cacheforge_wall_clock_s_mean": round(float(statistics.mean(c_wc)), 3),
            "baseline_wall_clock_s_median": round(float(statistics.median(b_wc)), 3),
            "runsift_wall_clock_s_median": round(float(statistics.median(c_wc)), 3),
            "cacheforge_wall_clock_s_median": round(float(statistics.median(c_wc)), 3),
            "baseline_model_s_mean": round(float(statistics.mean(b_mod_s)), 3),
            "runsift_model_s_mean": round(float(statistics.mean(c_mod_s)), 3),
            "cacheforge_model_s_mean": round(float(statistics.mean(c_mod_s)), 3),
            "baseline_tool_s_mean": round(float(statistics.mean(b_tl_s)), 3),
            "runsift_tool_s_mean": round(float(statistics.mean(c_tl_s)), 3),
            "cacheforge_tool_s_mean": round(float(statistics.mean(c_tl_s)), 3),
            "baseline_read_file_calls_mean": round(float(statistics.mean(b_rf)), 2),
            "runsift_read_file_calls_mean": round(float(statistics.mean(c_rf)), 2),
            "cacheforge_read_file_calls_mean": round(float(statistics.mean(c_rf)), 2),
            "baseline_scheduler_failed_sessions": b_sf,
            "runsift_scheduler_failed_sessions": c_sf,
            "cacheforge_scheduler_failed_sessions": c_sf,
            "records": records,
        }
        args.json_out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
