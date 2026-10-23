"""Script to rescore coupled benchmark result JSON files via deterministic patch replay and JUnit XML evaluation.

Rebuilds the initial 5-trap coupled workspace (materialize_clean_repo_coupled, inject_all_coupled_bugs,
git init, git add), applies the saved patch_diff via git apply, runs pytest with JUnit XML, and scores
using score_coupled_evaluation with the session's recorded regressions_introduced.
Also reclassifies misclassified context-overflow errors (per classify_upstream_error) so genuine
overflows are evaluated by patch replay rather than zeroed as infrastructure errors.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
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
    classify_upstream_error,
    inject_all_coupled_bugs,
    materialize_clean_repo_coupled,
    parse_junit_xml_test_results,
    run_workspace_pytest,
    score_coupled_evaluation,
    score_final_evaluation,
)

RESULTS_DIR = Path("benchmarks/tensorroute_benchmark/results")
MANIFEST_PATH = RESULTS_DIR / "tasks_coupled_manifest.json"


def _load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def rescore_coupled_task_result(
    tr: dict[str, Any],
    arm: str,
    manifest: dict[str, Any] | None = None,
    label: str = "",
    verify_counts: bool = False,
) -> dict[str, Any]:
    """Rebuilds the coupled workspace, applies patch_diff, runs pytest with JUnit XML, and scores."""
    tr = copy.deepcopy(tr)
    if manifest is None:
        manifest = _load_manifest()

    total_expected_tests = int(manifest.get("test_count", 56))
    total_expected_traps = int(manifest.get("trap_count", 5))

    # Reclassify misclassified context-overflow errors before checking infra_error
    if tr.get("infra_error", False):
        err_msg = str(tr.get("infra_error_msg", "") or "")
        if classify_upstream_error(err_msg) == "context_overflow":
            tr["infra_error"] = False
            tr["infra_error_msg"] = ""
            tr["context_overflow"] = True
            m_ovf = re.search(r"request\s*\((\d+)\s*tokens\)", err_msg)
            if m_ovf:
                ovf_tokens = int(m_ovf.group(1))
                pt_list = list(tr.get("per_turn_prompt_tokens", []))
                if not pt_list or pt_list[-1] != ovf_tokens:
                    pt_list.append(ovf_tokens)
                tr["per_turn_prompt_tokens"] = pt_list
                tr["total_prompt_tokens"] = sum(pt_list)
                tr["peak_turn_prompt_tokens"] = max(pt_list) if pt_list else 0
            tools_list = list(tr.get("per_turn_tools", []))
            if tools_list and tools_list[-1] == "error":
                tools_list[-1] = "overflow"
                tr["per_turn_tools"] = tools_list
            replies_list = list(tr.get("per_turn_raw_replies", []))
            if replies_list and replies_list[-1].startswith("[ERROR: "):
                replies_list[-1] = "[EXCEPTION: " + replies_list[-1][len("[ERROR: ") :]
                tr["per_turn_raw_replies"] = replies_list
            fb_list = list(tr.get("per_turn_tool_feedbacks", []))
            if fb_list and fb_list[-1] == "[UPSTREAM ERROR - NO FEEDBACK]":
                fb_list[-1] = "[CONTEXT OVERFLOW - NO FEEDBACK]"
                tr["per_turn_tool_feedbacks"] = fb_list

    if tr.get("infra_error", False):
        tr["final_eval_status"] = "infra_error"
        tr["modules_fixed"] = f"0/{total_expected_traps}"
        tr["solved"] = False
        tr["traps_solved"] = 0
        tr["solved_traps"] = []
        tr["regressions_repaired"] = 0
        tr["final_pass_to_pass_intact"] = False
        tr["passed_tests"] = 0
        tr["failed_tests"] = total_expected_tests
        return tr

    recorded_passed = int(tr.get("passed_tests", 0))
    recorded_failed = int(tr.get("failed_tests", 0))
    patch_diff = tr.get("patch_diff", "") or ""
    regressions_introduced = list(tr.get("regressions_introduced", []))

    tmp_ws = Path(tempfile.mkdtemp(prefix=f"tr_rescore_coupled_{arm}_"))
    try:
        materialize_clean_repo_coupled(
            tmp_ws, arm="arm_b" if arm in ("runsift", "cacheforge", "arm_b") else "arm_a"
        )
        inject_all_coupled_bugs(tmp_ws)
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
            tmp_ws, expected_tests=total_expected_tests, junit_xml_path=junit_path
        )

        if verify_counts and (passed_final != recorded_passed or failed_final != recorded_failed):
            raise ValueError(
                f"Pass/fail count mismatch in {label} ({arm}): "
                f"rescored ({passed_final} passed, {failed_final} failed) != "
                f"recorded ({recorded_passed} passed, {recorded_failed} failed)"
            )

        final_eval_status, _mod_str, _solved, _passing_subs = score_final_evaluation(
            raw_output=raw_final,
            passed=passed_final,
            failed=failed_final,
            rc=rc_final,
            expected_tests=total_expected_tests,
            trap_count=total_expected_traps,
            manifest=manifest,
            junit_xml_path=junit_path,
        )

        final_test_results = parse_junit_xml_test_results(junit_path)
        traps_solved_cnt, solved_traps, regressions_repaired, final_pass_to_pass_intact = (
            score_coupled_evaluation(
                final_test_results=final_test_results,
                manifest=manifest,
                regressions_introduced=regressions_introduced,
                final_eval_status=final_eval_status,
            )
        )

        tr["passed_tests"] = passed_final
        tr["failed_tests"] = failed_final
        tr["final_eval_status"] = final_eval_status
        tr["traps_solved"] = traps_solved_cnt
        tr["solved_traps"] = solved_traps
        tr["regressions_repaired"] = regressions_repaired
        tr["final_pass_to_pass_intact"] = final_pass_to_pass_intact
        tr["modules_fixed"] = f"{traps_solved_cnt}/{total_expected_traps}"
        tr["solved"] = traps_solved_cnt == total_expected_traps
        return tr
    finally:
        shutil.rmtree(tmp_ws, ignore_errors=True)


def _sync_arm_summary(arm_dict: dict[str, Any]) -> None:
    trs = arm_dict.get("task_results", [])
    if not trs:
        return
    n_tasks = len(trs)
    solved_cnt = sum(1 for r in trs if r.get("solved", False))
    arm_dict["tasks_solved"] = f"{solved_cnt}/{n_tasks}"
    arm_dict["solve_rate_pct"] = round((solved_cnt / max(1, n_tasks)) * 100.0, 2)
    arm_dict["context_overflow_count"] = sum(
        1 for r in trs if bool(r.get("context_overflow", False))
    )
    all_tokens = [x for r in trs for x in r.get("per_turn_prompt_tokens", [])]
    arm_dict["total_prompt_tokens"] = sum(all_tokens)
    arm_dict["peak_turn_prompt_tokens"] = max(all_tokens) if all_tokens else 0
    upstream_cached = sum(int(r.get("upstream_cached_tokens", 0)) for r in trs)
    arm_dict["upstream_cached_tokens"] = upstream_cached
    arm_dict["uncached_prefill_tokens"] = sum(all_tokens) - upstream_cached
    arm_dict["per_turn_prompt_tokens"] = all_tokens
    arm_dict["per_turn_tools"] = [x for r in trs for x in r.get("per_turn_tools", [])]
    arm_dict["per_turn_tests_passing"] = [
        x for r in trs for x in r.get("per_turn_tests_passing", [])
    ]


def rescore_coupled_repeat_dict(
    data: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    label: str = "",
    verify_counts: bool = False,
) -> dict[str, Any]:
    """Rescores both arms of a coupled repeat dict and updates scorecard summary fields."""
    rescored = copy.deepcopy(data)
    if manifest is None:
        manifest = _load_manifest()

    for arm in ("baseline", "runsift", "cacheforge"):
        if arm in rescored and "task_results" in rescored[arm]:
            new_trs = []
            for tr in rescored[arm]["task_results"]:
                new_trs.append(
                    rescore_coupled_task_result(
                        tr,
                        arm=arm,
                        manifest=manifest,
                        label=label,
                        verify_counts=verify_counts,
                    )
                )
            rescored[arm]["task_results"] = new_trs
            _sync_arm_summary(rescored[arm])

    b_trs = rescored.get("baseline", {}).get("task_results", [])
    c_trs = rescored.get("runsift", rescored.get("cacheforge", {})).get("task_results", [])
    invalidation_reasons: list[str] = []
    for r in b_trs:
        if r.get("infra_error", False):
            msg = str(r.get("infra_error_msg", "server unreachable / down"))
            invalidation_reasons.append(
                f"Baseline infrastructure error (connection refused / server down): {msg[:100]}"
            )
    for r in c_trs:
        if r.get("infra_error", False):
            msg = str(r.get("infra_error_msg", "server unreachable / down"))
            invalidation_reasons.append(
                f"RunSift infrastructure error (connection refused / server down): {msg[:100]}"
            )
    b_nudge = float(rescored.get("baseline", {}).get("nudge_rate_pct", 0.0))
    c_nudge = float(rescored.get("runsift", rescored.get("cacheforge", {})).get("nudge_rate_pct", 0.0))
    if b_trs and b_nudge > 20.0:
        invalidation_reasons.append(
            f"Baseline nudge rate {b_nudge}% exceeds 20.0% threshold (stuck agent)"
        )
    if c_trs and c_nudge > 20.0:
        invalidation_reasons.append(
            f"RunSift nudge rate {c_nudge}% exceeds 20.0% threshold (stuck agent)"
        )

    rescored["is_valid"] = len(invalidation_reasons) == 0
    rescored["invalidation_reason"] = (
        "; ".join(invalidation_reasons) if invalidation_reasons else None
    )

    c_arm_key = "runsift" if "runsift" in rescored else ("cacheforge" if "cacheforge" in rescored else None)
    if "improvements" in rescored and "baseline" in rescored and c_arm_key:
        b = rescored["baseline"]
        c = rescored[c_arm_key]
        b_prompt = int(b.get("total_prompt_tokens", 0))
        c_prompt = int(c.get("total_prompt_tokens", 0))
        b_unc = int(b.get("uncached_prefill_tokens", 0))
        c_unc = int(c.get("uncached_prefill_tokens", 0))
        b_peak = int(b.get("peak_turn_prompt_tokens", 0))
        c_peak = int(c.get("peak_turn_prompt_tokens", 0))
        b_sr = float(b.get("solve_rate_pct", 0.0))
        c_sr = float(c.get("solve_rate_pct", 0.0))
        rescored["improvements"]["solve_rate_delta_pct"] = round(c_sr - b_sr, 2)
        rescored["improvements"]["token_savings_pct"] = round(
            ((b_prompt - c_prompt) / max(1, b_prompt)) * 100.0, 2
        )
        rescored["improvements"]["uncached_prefill_savings_pct"] = round(
            ((b_unc - c_unc) / max(1, b_unc)) * 100.0, 2
        )
        rescored["improvements"]["peak_token_reduction_pct"] = round(
            ((b_peak - c_peak) / max(1, b_peak)) * 100.0, 2
        )
        rescored["improvements"]["tokens_saved_total"] = b_prompt - c_prompt

    return rescored


def rescore_coupled_file(
    file_path: Path, verify_counts: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Rescores a single coupled repeat file and writes `<stem>_rescored.json`."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    rescored = rescore_coupled_repeat_dict(data, label=file_path.name, verify_counts=verify_counts)
    out_path = file_path.with_name(f"{file_path.stem}_rescored.json")
    out_path.write_text(json.dumps(rescored, indent=2), encoding="utf-8")
    return out_path, rescored


def validate_normal_coupled_sessions(results_dir: Path = RESULTS_DIR) -> None:
    """Validates coupled workspace replay on 6 normal repeats (12 sessions across both models)."""
    manifest = _load_manifest(results_dir / "tasks_coupled_manifest.json")
    test_files = [
        "local_coupled_gemma4_26b_40turn_rep5.json",
        "local_coupled_gemma4_26b_40turn_rep6.json",
        "local_coupled_gemma4_26b_40turn_rep15.json",
        "local_coupled_nemotron_30b_40turn_rep1.json",
        "local_coupled_nemotron_30b_40turn_rep7.json",
        "local_coupled_nemotron_30b_40turn_rep8.json",
    ]
    print(f"=== VALIDATING COUPLED RESCORER ON {len(test_files)} NORMAL REPEAT FILES ===")
    for fname in test_files:
        fpath = results_dir / fname
        orig = json.loads(fpath.read_text(encoding="utf-8"))
        rescored = rescore_coupled_repeat_dict(
            orig, manifest=manifest, label=fname, verify_counts=True
        )
        for arm in [a for a in ("baseline", "runsift", "cacheforge") if a in orig]:
            got = rescored[arm]["task_results"][0]
            exp = orig[arm]["task_results"][0]
            assert got["traps_solved"] == exp["traps_solved"], (
                f"{fname} ({arm}) traps_solved mismatch: {got['traps_solved']} != {exp['traps_solved']}"
            )
            assert got["solved_traps"] == exp["solved_traps"], (
                f"{fname} ({arm}) solved_traps mismatch: {got['solved_traps']} != {exp['solved_traps']}"
            )
            assert got["regressions_repaired"] == exp["regressions_repaired"], (
                f"{fname} ({arm}) regressions_repaired mismatch: {got['regressions_repaired']} != {exp['regressions_repaired']}"
            )
            assert got["final_pass_to_pass_intact"] == exp["final_pass_to_pass_intact"], (
                f"{fname} ({arm}) final_pass_to_pass_intact mismatch: {got['final_pass_to_pass_intact']} != {exp['final_pass_to_pass_intact']}"
            )
            assert got["passed_tests"] == exp["passed_tests"], (
                f"{fname} ({arm}) passed_tests mismatch: {got['passed_tests']} != {exp['passed_tests']}"
            )
            assert got["failed_tests"] == exp["failed_tests"], (
                f"{fname} ({arm}) failed_tests mismatch: {got['failed_tests']} != {exp['failed_tests']}"
            )
            assert got["modules_fixed"] == exp["modules_fixed"]
        b_tr = rescored["baseline"]["task_results"][0]
        c_tr = rescored.get("runsift", rescored.get("cacheforge", {}))["task_results"][0]
        print(
            f"  {fname}: Baseline traps={b_tr['traps_solved']} {b_tr['solved_traps']} "
            f"(reg_rep={b_tr['regressions_repaired']}, intact={b_tr['final_pass_to_pass_intact']}, "
            f"{b_tr['passed_tests']}/{b_tr['failed_tests']}) | "
            f"RunSift traps={c_tr['traps_solved']} {c_tr['solved_traps']} "
            f"(reg_rep={c_tr['regressions_repaired']}, intact={c_tr['final_pass_to_pass_intact']}, "
            f"{c_tr['passed_tests']}/{c_tr['failed_tests']}) -> EXACT MATCH"
        )
    print(
        f"✓ Validation passed: all {len(test_files)} normal repeats (12 sessions) reproduced "
        "traps_solved, solved_traps, regressions_repaired, final_pass_to_pass_intact, and passed/failed counts 100%.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore coupled benchmark session files via workspace replay and JUnit XML."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=[
            "benchmarks/tensorroute_benchmark/results/local_coupled_nemotron_30b_40turn_rep15.json"
        ],
        help="Path(s) to coupled repeat JSON files to rescore",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on normal coupled sessions without writing rescored files",
    )
    args = parser.parse_args()

    validate_normal_coupled_sessions(RESULTS_DIR)
    if args.validate_only:
        return

    for f_str in args.files:
        fpath = Path(f_str)
        out_path, rescored = rescore_coupled_file(fpath, verify_counts=False)
        b_tr = rescored["baseline"]["task_results"][0]
        c_tr = rescored.get("runsift", rescored.get("cacheforge", {}))["task_results"][0]
        print(
            f"Wrote {out_path.name}:\n"
            f"  Baseline:   traps={b_tr['traps_solved']}/5 {b_tr['solved_traps']}, "
            f"passed={b_tr['passed_tests']}, failed={b_tr['failed_tests']}, "
            f"reg_intro={len(b_tr['regressions_introduced'])}, reg_rep={b_tr['regressions_repaired']}, "
            f"intact={b_tr['final_pass_to_pass_intact']}, overflow={b_tr['context_overflow']}, "
            f"infra_error={b_tr['infra_error']}, peak_tokens={b_tr['peak_turn_prompt_tokens']}, "
            f"total_prompt_tokens={b_tr['total_prompt_tokens']}\n"
            f"  RunSift:    traps={c_tr['traps_solved']}/5 {c_tr['solved_traps']}, "
            f"passed={c_tr['passed_tests']}, failed={c_tr['failed_tests']}, "
            f"reg_intro={len(c_tr['regressions_introduced'])}, reg_rep={c_tr['regressions_repaired']}, "
            f"intact={c_tr['final_pass_to_pass_intact']}, overflow={c_tr['context_overflow']}, "
            f"infra_error={c_tr['infra_error']}, peak_tokens={c_tr['peak_turn_prompt_tokens']}, "
            f"total_prompt_tokens={c_tr['total_prompt_tokens']}"
        )


if __name__ == "__main__":
    main()
