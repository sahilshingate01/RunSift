"""Generates real pytest output fixtures across formats for Condenser v2 shell-test recognition.

Builds the 5-bug gauntlet workspace (`materialize_clean_repo` + `inject_gauntlet_bugs(BENCHMARK_TASKS[0:5])`)
and an all-passing clean workspace in temporary directories, runs real pytest with various flags,
normalizes `rootdir:` paths to `<workspace>`, and writes fixtures to `tests/fixtures/pytest_formats/`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "tensorroute_benchmark"))

from repo_builder import materialize_clean_repo
from tasks import BENCHMARK_TASKS, inject_gauntlet_bugs

OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "pytest_formats"


def _normalize_workspace(text: str, ws: Path) -> str:
    """Replaces rootdir paths and temporary workspace paths with <workspace> and ensures LF endings."""
    text = text.replace("\r\n", "\n")
    ws_str = str(ws)
    ws_posix = ws_str.replace("\\", "/")
    text = text.replace(ws_str, "<workspace>")
    text = text.replace(ws_posix, "<workspace>")
    text = re.sub(r"^rootdir:\s+.*$", "rootdir: <workspace>", text, flags=re.MULTILINE)
    return text


def _run_pytest_capture(
    ws: Path,
    args: list[str],
    include_pythonpath: bool = True,
) -> str:
    env = os.environ.copy()
    if include_pythonpath:
        env["PYTHONPATH"] = str(ws / "src")
        cmd = [sys.executable, "-m", "pytest", "-o", "pythonpath=src", *args]
    else:
        env.pop("PYTHONPATH", None)
        cmd = [sys.executable, "-m", "pytest", *args]

    res = subprocess.run(
        cmd,
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=35,
    )
    raw = res.stdout + res.stderr
    return _normalize_workspace(raw, ws)


def _write_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    path.write_bytes(normalized.encode("utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_buggy = Path(tempfile.mkdtemp(prefix="cf_pytest_fmt_buggy_"))
    tmp_clean = Path(tempfile.mkdtemp(prefix="cf_pytest_fmt_clean_"))

    try:
        materialize_clean_repo(tmp_buggy)
        inject_gauntlet_bugs(tmp_buggy, BENCHMARK_TASKS[0:5])

        materialize_clean_repo(tmp_clean)

        # 1. Captures on 5-bug gauntlet workspace
        harness_vv = _run_pytest_capture(tmp_buggy, ["-vv", "-ra", "--tb=long", "--durations=0"])
        no_flags = _run_pytest_capture(tmp_buggy, [])
        quiet_q = _run_pytest_capture(tmp_buggy, ["-q"])
        failfast_xq = _run_pytest_capture(tmp_buggy, ["-x", "-q"])
        verbose_v = _run_pytest_capture(tmp_buggy, ["-v"])
        tb_short = _run_pytest_capture(tmp_buggy, ["--tb=short"])
        q_tb_line = _run_pytest_capture(tmp_buggy, ["-q", "--tb=line"])
        missing_file = _run_pytest_capture(tmp_buggy, ["tests/test_missing_nonexistent.py"])
        no_pythonpath = _run_pytest_capture(tmp_buggy, [], include_pythonpath=False)

        # 2. Captures on all-passing clean workspace
        clean_no_flags = _run_pytest_capture(tmp_clean, [])
        clean_q = _run_pytest_capture(tmp_clean, ["-q"])

        # Save base captures
        _write_lf(OUT_DIR / "gauntlet_harness_vv_ra_tblong.txt", harness_vv)
        _write_lf(OUT_DIR / "gauntlet_no_flags.txt", no_flags)
        _write_lf(OUT_DIR / "gauntlet_q.txt", quiet_q)
        _write_lf(OUT_DIR / "gauntlet_x_q.txt", failfast_xq)
        _write_lf(OUT_DIR / "gauntlet_v.txt", verbose_v)
        _write_lf(OUT_DIR / "gauntlet_tb_short.txt", tb_short)
        _write_lf(OUT_DIR / "gauntlet_q_tb_line.txt", q_tb_line)
        _write_lf(OUT_DIR / "gauntlet_missing_file.txt", missing_file)
        _write_lf(OUT_DIR / "gauntlet_no_pythonpath_collection_errors.txt", no_pythonpath)
        _write_lf(OUT_DIR / "clean_passing_no_flags.txt", clean_no_flags)
        _write_lf(OUT_DIR / "clean_passing_q.txt", clean_q)

        # 3. Wrapped variants built from those captures
        # (a) command-echo prefix line
        wrapped_prefix = "$ python -m pytest -q\n" + quiet_q
        _write_lf(OUT_DIR / "wrapped_command_echo_prefix.txt", wrapped_prefix)

        # (b) shell suffix lines (exit code: 1; Command timed out after 120 seconds)
        wrapped_suffix_exit = no_flags.rstrip("\n") + "\nexit code: 1\n"
        _write_lf(OUT_DIR / "wrapped_shell_suffix_exit_code.txt", wrapped_suffix_exit)

        wrapped_prefix_and_suffix = (
            "$ pytest --tb=short\n"
            + tb_short.rstrip("\n")
            + "\nexit code: 1\nCommand timed out after 120 seconds\n"
        )
        _write_lf(OUT_DIR / "wrapped_prefix_and_suffix.txt", wrapped_prefix_and_suffix)

        # (c) killed run: the no-flags output cut after its third progress line, plus a timeout line
        progress_re = re.compile(r"\[\s*\d+%\]\s*$")
        killed_lines: list[str] = []
        progress_seen = 0
        for line in no_flags.splitlines():
            killed_lines.append(line)
            if progress_re.search(line):
                progress_seen += 1
                if progress_seen == 3:
                    break
        killed_lines.append("Command timed out after 120 seconds")
        _write_lf(OUT_DIR / "wrapped_killed_after_3_progress_lines.txt", "\n".join(killed_lines))

        # (d) forward-slash (Linux-style) paths
        linux_paths = no_flags.replace("\\", "/").replace("platform win32", "platform linux")
        _write_lf(OUT_DIR / "wrapped_linux_forward_slash_paths.txt", linux_paths)

        print(f"Generated {len(list(OUT_DIR.glob('*.txt')))} fixtures in {OUT_DIR}")
    finally:
        shutil.rmtree(tmp_buggy, ignore_errors=True)
        shutil.rmtree(tmp_clean, ignore_errors=True)


if __name__ == "__main__":
    main()
