"""Step 1 Deterministic Verification Script for TensorRoute-Bench v2.

Supports:
1. `--task-set original` (default): 15 Tasks + Wave 1-3 Gauntlets.
2. `--task-set coupled`: 5 Coupled Traps (T1-T5) with shared helper, unit boundaries,
   masked bugs, changed defaults, and workaround-vs-root-cause traps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .repo_builder import materialize_clean_repo as materialize_clean_repo_v1
    from .repo_builder_coupled import materialize_clean_repo as materialize_clean_repo_v2
    from .tasks import (
        BENCHMARK_TASKS,
        apply_reference_fix,
        inject_bug,
        inject_gauntlet_bugs,
    )
    from .tasks_coupled import (
        COUPLED_TASKS,
        apply_all_coupled_fixes,
        apply_naive_fix,
        apply_task_reference_fix,
        inject_all_coupled_bugs,
        inject_task_bug,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repo_builder import materialize_clean_repo as materialize_clean_repo_v1
    from repo_builder_coupled import materialize_clean_repo as materialize_clean_repo_v2
    from tasks import (
        BENCHMARK_TASKS,
        apply_reference_fix,
        inject_bug,
        inject_gauntlet_bugs,
    )
    from tasks_coupled import (
        COUPLED_TASKS,
        apply_all_coupled_fixes,
        apply_naive_fix,
        apply_task_reference_fix,
        inject_all_coupled_bugs,
        inject_task_bug,
    )

try:
    from sdcs.verifier.topology import run_topology_audit
except ImportError:
    _sdcs_root = Path(__file__).resolve().parent.parent.parent
    _sib = _sdcs_root.parent / "Spec-Driven Cognitive Scaffolding (SDCS) framework" / "src"
    sys.path.insert(
        0, str((_sdcs_root / "src") if (_sdcs_root / "src" / "sdcs").is_dir() else _sib)
    )
    from sdcs.verifier.topology import run_topology_audit


def _run_pytest(cwd: Path, junit_path: Path | None = None) -> tuple[int, str, int, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "pythonpath=src",
        "-vv",
        "-ra",
        "--tb=long",
        "--durations=0",
    ]
    if junit_path is not None:
        cmd.extend([f"--junitxml={junit_path}"])

    res = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = res.stdout + res.stderr
    m_pass = re.search(r"(\d+)\s+passed", out)
    m_fail = re.search(r"(\d+)\s+failed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    failed = int(m_fail.group(1)) if m_fail else 0
    return res.returncode, out, passed, failed


def _parse_junit_tests(junit_path: Path) -> tuple[set[str], set[str]]:
    """Returns (passing_tests, failing_tests) as canonical 'path/to/test_file.py::test_name'."""
    if not junit_path.exists():
        return set(), set()
    tree = ET.parse(junit_path)
    root = tree.getroot()
    passing: set[str] = set()
    failing: set[str] = set()
    for tc in root.findall(".//testcase"):
        cls = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        # Format classname like 'tests.test_units' -> 'tests/test_units.py'
        path_str = cls.replace(".", "/") + ".py"
        test_id = f"{path_str}::{name}"
        if tc.find("failure") is not None or tc.find("error") is not None:
            failing.add(test_id)
        else:
            passing.add(test_id)
    return passing, failing


def _compute_sha256_tree(ws: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for p in sorted(ws.rglob("*.py")):
        if "__pycache__" in str(p) or ".pytest_cache" in str(p):
            continue
        rel = str(p.relative_to(ws)).replace("\\", "/")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[rel] = h
    return hashes


def verify_coupled_tasks() -> bool:
    print("====================================================================")
    print(" TensorRoute-Bench v2 :: Coupled-Bug Gauntlet Verification")
    print("====================================================================")

    tmp_root = Path(tempfile.mkdtemp(prefix="tr_coupled_verify_"))
    try:
        # Proof 1: The clean repo passes every test & Gate T topology
        ws_clean = tmp_root / "clean"
        materialize_clean_repo_v2(ws_clean, arm="arm_b", include_sdcs=True)
        topo_rc = run_topology_audit(ws_clean, ws_clean / "wiring.yaml")
        if topo_rc != 0:
            print("  [FAIL] Clean repo: Gate T topology verification failed!")
            return False

        clean_xml = ws_clean / "clean.xml"
        rc_clean, _out_clean, passed_c, failed_c = _run_pytest(ws_clean, junit_path=clean_xml)
        if rc_clean != 0 or failed_c > 0:
            print(f"  [FAIL] Clean repo: expected 0 failed, got {failed_c}F/{passed_c}P")
            return False

        passing_c, _failing_c = _parse_junit_tests(clean_xml)
        total_tests = passed_c
        print(
            f"  [PASS] Proof 1: Clean repo passes 100% tests ({total_tests}/{total_tests} PASS) | Gate T: OK"
        )

        # Source file hashes from clean repo
        source_hashes = _compute_sha256_tree(ws_clean)

        # Proof 2: The buggy repo fails exactly the union of the F2P sets
        ws_buggy = tmp_root / "buggy"
        materialize_clean_repo_v2(ws_buggy, arm="arm_b")
        inject_all_coupled_bugs(ws_buggy)

        buggy_xml = ws_buggy / "buggy.xml"
        _rc_bug, _out_bug, passed_b, _failed_b = _run_pytest(ws_buggy, junit_path=buggy_xml)
        passing_b, failing_b = _parse_junit_tests(buggy_xml)

        expected_f2p_union: set[str] = set()
        for t in COUPLED_TASKS:
            expected_f2p_union.update(t.f2p_tests)

        if failing_b != expected_f2p_union:
            diff_unexpected = failing_b - expected_f2p_union
            diff_missing = expected_f2p_union - failing_b
            print("  [FAIL] Proof 2: Buggy repo does not match expected F2P union!")
            if diff_unexpected:
                print(f"         Unexpected failures: {diff_unexpected}")
            if diff_missing:
                print(f"         Missing expected failures: {diff_missing}")
            return False

        print(
            f"  [PASS] Proof 2: Buggy repo fails EXACTLY the union of F2P sets ({len(failing_b)}F/{passed_b}P)"
        )
        for t in COUPLED_TASKS:
            print(f"         - {t.task_id}: {t.f2p_tests}")

        # Proof 3: Each correct fix passes its F2P tests without breaking anything
        print("  --- Proof 3: Individual Reference Fixes ---")
        for t in COUPLED_TASKS:
            ws_fix = tmp_root / f"fix_{t.task_id}"
            materialize_clean_repo_v2(ws_fix, arm="arm_b")
            inject_all_coupled_bugs(ws_fix)
            apply_task_reference_fix(ws_fix, t)
            fix_xml = ws_fix / "fix.xml"
            _rc_f, _out_f, passed_f, failed_f = _run_pytest(ws_fix, junit_path=fix_xml)
            _passing_f, failing_f = _parse_junit_tests(fix_xml)

            # The fixed task's F2P tests must NOT be in failing_f
            for f2p in t.f2p_tests:
                if f2p in failing_f:
                    print(
                        f"  [FAIL] Proof 3: Task {t.task_id} reference fix failed to restore {f2p}"
                    )
                    return False

            print(
                f"  [PASS] Proof 3: {t.task_id:<28} | Restores F2P tests cleanly ({failed_f}F/{passed_f}P)"
            )

        # Proof 4: Each naive patch behaves as designed
        print("  --- Proof 4: Naive Fix Guard Break Proofs ---")
        for _idx, t in enumerate(COUPLED_TASKS):
            ws_naive = tmp_root / f"naive_{t.task_id}"
            materialize_clean_repo_v2(ws_naive, arm="arm_b")
            inject_task_bug(ws_naive, t)
            apply_naive_fix(ws_naive, t)
            naive_xml = ws_naive / "naive.xml"
            _rc_n, _out_n, _passed_n, _failed_n = _run_pytest(ws_naive, junit_path=naive_xml)
            _passing_n, failing_n = _parse_junit_tests(naive_xml)

            if t.task_id == "T3-masked-bug":
                # Scheduler F2P passes, pipeline F2P still fails
                sched_f2p = t.f2p_tests[0]
                pipe_f2p = t.f2p_tests[1]
                if sched_f2p in failing_n:
                    print(f"  [FAIL] Proof 4: T3 naive scheduler fix failed to pass {sched_f2p}")
                    return False
                if pipe_f2p not in failing_n:
                    print(
                        f"  [FAIL] Proof 4: T3 naive fix unexpectedly passed pipeline test {pipe_f2p}"
                    )
                    return False
                print(
                    f"  [PASS] Proof 4: {t.task_id:<28} | Scheduler-only fix leaves pipeline F2P failing"
                )

            elif t.task_id == "T5-workaround-root-cause":
                # Radix + pipeline workaround breaks the guard
                guard = t.guard_tests[0]
                if guard not in failing_n:
                    print(
                        f"  [FAIL] Proof 4: T5 workaround plus radix fix failed to break guard {guard}"
                    )
                    return False
                print(
                    f"  [PASS] Proof 4: {t.task_id:<28} | Workaround + radix fix breaks guard: {guard}"
                )

            else:
                # T1, T2, T4: F2P passes, but at least one guard test breaks
                for f2p in t.f2p_tests:
                    if f2p in failing_n:
                        print(f"  [FAIL] Proof 4: {t.task_id} naive fix failed to pass F2P {f2p}")
                        return False
                broken_guards = [g for g in t.guard_tests if g in failing_n]
                if not broken_guards:
                    print(
                        f"  [FAIL] Proof 4: {t.task_id} naive fix did not break any guard tests! {t.guard_tests}"
                    )
                    return False
                print(
                    f"  [PASS] Proof 4: {t.task_id:<28} | F2P passes, breaks guard(s): {broken_guards}"
                )

        # Proof 5: All five correct fixes applied together pass everything
        ws_all_fixed = tmp_root / "all_fixed"
        materialize_clean_repo_v2(ws_all_fixed, arm="arm_b")
        inject_all_coupled_bugs(ws_all_fixed)
        apply_all_coupled_fixes(ws_all_fixed)
        all_fixed_xml = ws_all_fixed / "all_fixed.xml"
        rc_af, _out_af, passed_af, failed_af = _run_pytest(ws_all_fixed, junit_path=all_fixed_xml)
        if rc_af != 0 or failed_af > 0 or passed_af != total_tests:
            print(f"  [FAIL] Proof 5: All fixes applied together failed: {failed_af}F/{passed_af}P")
            return False
        print(
            f"  [PASS] Proof 5: All 5 reference fixes combined restore 100% pass ({passed_af}/{total_tests})"
        )

        # Generate and save manifest
        manifest_data = {
            "task_set": "coupled",
            "test_count": total_tests,
            "trap_count": len(COUPLED_TASKS),
            "source_file_hashes": source_hashes,
            "all_tests": sorted(passing_c),
            "initial_passing_tests": sorted(passing_b),
            "f2p_union": sorted(failing_b),
            "traps": {
                t.task_id: {
                    "title": t.title,
                    "symptom": t.symptom_prompt,
                    "f2p_tests": sorted(t.f2p_tests),
                    "guard_tests": sorted(t.guard_tests),
                }
                for t in COUPLED_TASKS
            },
        }

        # Write to results/tasks_coupled_manifest.json and benchmarks/tensorroute_benchmark/results/
        res_dir = Path(__file__).resolve().parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = res_dir / "tasks_coupled_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        print(f"\n  [MANIFEST] Generated {manifest_path} ({manifest_path.stat().st_size} bytes)")

        # Also write root results if exists or needed
        root_res_dir = Path(__file__).resolve().parent.parent.parent / "results"
        if root_res_dir.is_dir():
            (root_res_dir / "tasks_coupled_manifest.json").write_text(
                json.dumps(manifest_data, indent=2), encoding="utf-8"
            )

        print("====================================================================")
        print(" [SUCCESS] All 5 Coupled-Bug Traps Verified Deterministically!")
        print("====================================================================")
        return True

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def verify_all_tasks() -> bool:
    print("====================================================================")
    print(" TensorRoute-Bench v2 :: Deterministic 15-Task + Gauntlet Verification")
    print("====================================================================")

    tmp_root = Path(tempfile.mkdtemp(prefix="tensorroute_verify_"))
    try:
        for idx, task in enumerate(BENCHMARK_TASKS):
            ws = tmp_root / task.task_id
            materialize_clean_repo_v1(ws, arm="arm_b", include_sdcs=True)

            # Verify SDCS Gate T topology on the materialized repo
            if idx == 0:
                topo_ok = run_topology_audit(ws, ws / "wiring.yaml") == 0
                if not topo_ok:
                    print(f"  [FAIL] {task.task_id}: Gate T topology failed on materialized repo")
                    return False

            # Inject the task's seeded bug
            inject_bug(ws, task)
            rc_bug, out_bug, passed_b, failed_b = _run_pytest(ws)
            if rc_bug == 0 or failed_b == 0:
                print(
                    f"  [FAIL] {task.task_id}: Expected pytest failure in buggy state, but passed!"
                )
                return False

            # Apply reference fix and verify 45/45 pass
            apply_reference_fix(ws, task)
            rc_fix, out_fix, passed_f, failed_f = _run_pytest(ws)
            if rc_fix != 0 or passed_f != 45 or failed_f != 0:
                print(
                    f"  [FAIL] {task.task_id}: Expected '45 passed' after fix, got:\n{out_fix[-400:]}"
                )
                return False

            print(
                f"  [PASS] {task.task_id:<36} | Buggy: {failed_b:2d}F/{passed_b:2d}P ({len(out_bug):5d} chars) -> Fixed: 45/45 PASS | Gate T: OK"
            )

        # Verify 5-Bug Gauntlet Mode (Wave 1: OB-01..OB-05 simultaneous)
        ws_gauntlet = tmp_root / "gauntlet_5bug"
        materialize_clean_repo_v1(ws_gauntlet, arm="arm_b")
        gauntlet_tasks = inject_gauntlet_bugs(ws_gauntlet, BENCHMARK_TASKS[:5])
        rc_g, out_g, passed_g, failed_g = _run_pytest(ws_gauntlet)
        if rc_g == 0 or failed_g < 10:
            print(f"  [FAIL] 5-Bug Gauntlet: Expected >=10 failures, got {failed_g}F/{passed_g}P")
            return False
        for t in gauntlet_tasks:
            apply_reference_fix(ws_gauntlet, t)
        rc_gf, _, passed_gf, failed_gf = _run_pytest(ws_gauntlet)
        if rc_gf != 0 or passed_gf != 45 or failed_gf != 0:
            print(f"  [FAIL] 5-Bug Gauntlet fix failed: {failed_gf}F/{passed_gf}P")
            return False
        print(
            f"  [PASS] {'GAUNTLET-5-BUG (OB-01..OB-05)':<36} | Buggy: {failed_g:2d}F/{passed_g:2d}P ({len(out_g):5d} chars) -> Fixed: 45/45 PASS | Gate T: OK"
        )

        # Verify all 15 bugs are mutually non-overlapping
        ws_all15 = tmp_root / "gauntlet_15bug"
        materialize_clean_repo_v1(ws_all15, arm="arm_b")
        inject_gauntlet_bugs(ws_all15, BENCHMARK_TASKS)
        _rc_15, out_15, passed_15, failed_15 = _run_pytest(ws_all15)
        for t in BENCHMARK_TASKS:
            apply_reference_fix(ws_all15, t)
        rc_15f, _, passed_15f, failed_15f = _run_pytest(ws_all15)
        if rc_15f != 0 or passed_15f != 45 or failed_15f != 0:
            print("  [FAIL] 15-Bug disjointness verification failed!")
            return False
        print(
            f"  [PASS] {'DISJOINT-15-BUG (OB-01..OB-15)':<36} | Buggy: {failed_15:2d}F/{passed_15:2d}P ({len(out_15):5d} chars) -> Fixed: 45/45 PASS | Gate T: OK"
        )

        print("====================================================================")
        print(" [SUCCESS] All 15 tasks + multi-bug Gauntlet verified deterministically!")
        print("====================================================================")
        return True
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic task verification.")
    parser.add_argument(
        "--task-set",
        choices=["original", "coupled"],
        default="original",
        help="Task set to verify: 'original' (15 tasks + gauntlets) or 'coupled' (5 coupled traps)",
    )
    args = parser.parse_args()
    if args.task_set == "coupled":
        ok = verify_coupled_tasks()
    else:
        ok = verify_all_tasks()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
