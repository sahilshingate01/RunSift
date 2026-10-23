"""Script to rescore Protocol v3 benchmark result JSON files via deterministic patch replay and JUnit XML evaluation.

Rebuilds the initial 5-bug workspace (materialize_clean_repo, inject_gauntlet_bugs with BENCHMARK_TASKS[0:5],
git init, git add), applies the saved patch_diff via git apply, runs pytest with JUnit XML, verifies that
passed/failed test counts match the recorded session counts, and writes *_rescored.json next to each original
file without modifying the originals.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tensorroute_benchmark.harness import (
    _compute_gauntlet_variance,
    inject_gauntlet_bugs,
    run_workspace_pytest,
    score_final_evaluation,
)
from benchmarks.tensorroute_benchmark.repo_builder import materialize_clean_repo
from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
    resolve_repeat_files,
)
from benchmarks.tensorroute_benchmark.tasks import BENCHMARK_TASKS

RESULTS_DIR = Path("benchmarks/tensorroute_benchmark/results")


def rescore_task_result_via_workspace(
    tr: dict[str, Any], arm: str, label: str = ""
) -> dict[str, Any]:
    """Rebuilds the bugged workspace, applies patch_diff, and scores with JUnit XML."""
    tr = copy.deepcopy(tr)
    if tr.get("infra_error", False):
        tr["modules_fixed"] = "0/5"
        tr["final_eval_status"] = "infra_error"
        tr["solved"] = False
        return tr

    recorded_passed = int(tr.get("passed_tests", 0))
    recorded_failed = int(tr.get("failed_tests", 0))
    patch_diff = tr.get("patch_diff", "") or ""

    tmp_ws = Path(tempfile.mkdtemp(prefix=f"tr_rescore_{arm}_"))
    try:
        materialize_clean_repo(tmp_ws, arm="arm_b" if arm in ("runsift", "cacheforge", "arm_b") else "arm_a")
        inject_gauntlet_bugs(tmp_ws, BENCHMARK_TASKS[0:5])
        subprocess.run(["git", "init", "-q"], cwd=tmp_ws, check=True)
        subprocess.run(["git", "add", "-A"], cwd=tmp_ws, check=True)

        if patch_diff.strip():
            patch_file = tmp_ws / "_saved_session.patch"
            text_to_write = patch_diff if patch_diff.endswith("\n") else patch_diff + "\n"
            with open(patch_file, "w", encoding="utf-8", newline="\n") as fp:
                fp.write(text_to_write)
            apply_res = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                cwd=tmp_ws,
                capture_output=True,
                text=True,
                check=False,
            )
            patch_file.unlink(missing_ok=True)
            if apply_res.returncode != 0:
                raise RuntimeError(
                    f"git apply failed for {label} ({arm}): {apply_res.stderr.strip()}"
                )

        junit_path = tmp_ws / "final_junit.xml"
        rc_final, raw_final, passed_final, failed_final = run_workspace_pytest(
            tmp_ws, junit_xml_path=str(junit_path)
        )

        if passed_final != recorded_passed or failed_final != recorded_failed:
            print(
                f"  [!] Pass/fail count corrected in {label} ({arm}): "
                f"rescored ({passed_final} passed, {failed_final} failed) != "
                f"recorded ({recorded_passed} passed, {recorded_failed} failed)"
            )
            tr["passed_tests"] = passed_final
            tr["failed_tests"] = failed_final

        final_eval_status, modules_fixed_str, solved, _passing_subs = score_final_evaluation(
            raw_output=raw_final,
            passed=passed_final,
            failed=failed_final,
            rc=rc_final,
            expected_tests=45,
            trap_count=5,
            junit_xml_path=junit_path,
        )

        tr["modules_fixed"] = modules_fixed_str
        tr["final_eval_status"] = final_eval_status
        tr["solved"] = solved
        return tr
    finally:
        shutil.rmtree(tmp_ws, ignore_errors=True)


def _update_arm_summary(arm_dict: dict[str, Any]) -> None:
    trs = arm_dict.get("task_results", [])
    if not trs:
        return
    n_tasks = len(trs)
    solved_cnt = sum(1 for r in trs if r.get("solved", False))
    arm_dict["tasks_solved"] = f"{solved_cnt}/{n_tasks}"
    arm_dict["solve_rate_pct"] = round((solved_cnt / max(1, n_tasks)) * 100.0, 2)


def rescore_repeat_dict(data: dict[str, Any], label: str = "") -> dict[str, Any]:
    """Rescores both arms of a repeat payload dict and updates arm/improvement solve rates."""
    rescored = copy.deepcopy(data)
    for arm in ("baseline", "runsift", "cacheforge"):
        if arm in rescored and "task_results" in rescored[arm]:
            new_trs = []
            for tr in rescored[arm]["task_results"]:
                new_trs.append(rescore_task_result_via_workspace(tr, arm, label=label))
            rescored[arm]["task_results"] = new_trs
            _update_arm_summary(rescored[arm])

    c_key = "runsift" if "runsift" in rescored else ("cacheforge" if "cacheforge" in rescored else None)
    if "improvements" in rescored and "baseline" in rescored and c_key:
        b_sr = float(rescored["baseline"].get("solve_rate_pct", 0.0))
        c_sr = float(rescored[c_key].get("solve_rate_pct", 0.0))
        rescored["improvements"]["solve_rate_delta_pct"] = round(c_sr - b_sr, 2)

    return rescored


def rescore_repeat_file(file_path: Path) -> tuple[Path, dict[str, Any]]:
    """Rescores a single repeat JSON file and writes out <name>_rescored.json."""
    stem = file_path.stem
    while stem.endswith("_rescored"):
        stem = stem[: -len("_rescored")]
    base_path = file_path.with_name(f"{stem}.json")
    data = json.loads(base_path.read_text(encoding="utf-8"))
    rescored = rescore_repeat_dict(data, label=base_path.name)
    out_path = file_path.with_name(f"{stem}_rescored.json")
    out_path.write_text(json.dumps(rescored, indent=2), encoding="utf-8")
    return out_path, rescored


def validate_against_nemotron_v3_reps_1_to_5(results_dir: Path = RESULTS_DIR) -> None:
    """Validates the workspace patch-replay rescorer on local_gauntlet_nemotron_30b_40turn_v3_rep1-5."""
    print("=== VALIDATING RESCORER ON local_gauntlet_nemotron_30b_40turn_v3_rep1..5 ===")
    for r in range(1, 6):
        orig_path = results_dir / f"local_gauntlet_nemotron_30b_40turn_v3_rep{r}.json"
        expected_path = results_dir / f"local_gauntlet_nemotron_30b_40turn_v3_rep{r}_rescored.json"
        orig_data = json.loads(orig_path.read_text(encoding="utf-8"))
        expected_data = json.loads(expected_path.read_text(encoding="utf-8"))

        rescored_data = rescore_repeat_dict(orig_data, label=orig_path.name)
        for arm in [a for a in ("baseline", "runsift", "cacheforge") if a in rescored_data]:
            got_tr = rescored_data[arm]["task_results"][0]
            exp_tr = expected_data[arm]["task_results"][0]
            assert got_tr["modules_fixed"] == exp_tr["modules_fixed"], (
                f"Rep {r} {arm} modules_fixed mismatch: got {got_tr['modules_fixed']} != "
                f"expected {exp_tr['modules_fixed']}"
            )
            assert got_tr["passed_tests"] == exp_tr["passed_tests"]
            assert got_tr["failed_tests"] == exp_tr["failed_tests"]
            assert got_tr["final_eval_status"] == exp_tr["final_eval_status"]
        b_mf = rescored_data["baseline"]["task_results"][0]["modules_fixed"]
        c_mf = rescored_data.get("runsift", rescored_data.get("cacheforge", {}))["task_results"][0]["modules_fixed"]
        print(
            f"  Rep {r}: baseline={b_mf}, runsift={c_mf} -> EXACT MATCH with {expected_path.name}"
        )
    print(
        "✓ Validation passed: all 5 repeats (both arms) reproduced modules_fixed and pass/fail counts 100%.\n"
    )


def rescore_prefix(prefix: str, results_dir: Path = RESULTS_DIR, repeats: int = 15) -> list[Path]:
    """Rescores all 15 selected valid repeat files for `prefix` and writes *_rescored.json files."""
    selected, _skipped = resolve_repeat_files(
        prefix=prefix, results_dir=results_dir, expected_repeats=repeats
    )
    out_paths: list[Path] = []
    rescored_runs: list[dict[str, Any]] = []

    print(f"=== RESCORING {len(selected)} SELECTED REPEAT FILES FOR {prefix} ===")
    for r, file_path, _data in selected:
        out_path, rescored = rescore_repeat_file(file_path)
        out_paths.append(out_path)
        rescored_runs.append(rescored)
        b_tr = rescored["baseline"]["task_results"][0]
        c_tr = rescored.get("runsift", rescored.get("cacheforge", {}))["task_results"][0]
        print(
            f"  Rep {r:2d} ({file_path.name} -> {out_path.name}): "
            f"Baseline {b_tr['modules_fixed']} ({b_tr['passed_tests']}/{b_tr['passed_tests'] + b_tr['failed_tests']} pass) | "
            f"RunSift {c_tr['modules_fixed']} ({c_tr['passed_tests']}/{c_tr['passed_tests'] + c_tr['failed_tests']} pass)"
        )

    agg_orig = results_dir / f"{prefix}.json"
    if agg_orig.exists() and rescored_runs:
        agg_rescored = copy.deepcopy(rescored_runs[-1])
        agg_rescored["raw_repeats"] = rescored_runs
        agg_rescored["repeats"] = len(rescored_runs)
        agg_rescored["variance_analysis"] = _compute_gauntlet_variance(rescored_runs)
        agg_out = results_dir / f"{prefix}_rescored.json"
        agg_out.write_text(json.dumps(agg_rescored, indent=2), encoding="utf-8")
        out_paths.append(agg_out)
        print(f"  Wrote aggregate rescored file -> {agg_out.name}")

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore Protocol v3 results via git apply and JUnit XML."
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default="nebius_gauntlet_nemotron_30b_40turn_v3",
        help="Prefix to rescore (default: nebius_gauntlet_nemotron_30b_40turn_v3)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation against local_gauntlet_nemotron_30b_40turn_v3_rep1..5_rescored.json",
    )
    args = parser.parse_args()

    validate_against_nemotron_v3_reps_1_to_5(RESULTS_DIR)
    if not args.validate_only:
        rescore_prefix(args.prefix, RESULTS_DIR)


if __name__ == "__main__":
    main()
