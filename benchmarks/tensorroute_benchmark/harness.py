"""Live Multi-Turn Agent Evaluation Harness for `TensorRoute-Bench` v2.

Supports two benchmark execution geometries on the 100% original `tensorroute` codebase:
1. `--mode suite` (15-Task Isolated Suite, `OB-01` through `OB-15`, 3 bugs per subsystem):
   - Evaluates `baseline` vs `runsift` across N=15 distinct systems bugs (or 5-task wave via `--wave 1`).
2. `--mode gauntlet` (Continuous 5-Bug Multi-Module Session, 15-22 turns without context reset):
   - Injects all 5 Wave-1 bugs (`OB-01`..`OB-05`, 1 per subsystem, 12 failing tests / 33 passing)
     into the SAME workspace simultaneously.
   - Forces the agent to diagnose and repair all 5 modules sequentially in one continuous
     multi-turn conversation without resetting `history`, directly testing Turn-15 Context Collapse.

Scientific Validity Guarantees (v2):
- 100% Authentic Pytest Output: Runs unmodified `pytest -o pythonpath=src -vv -ra --tb=long --durations=0`
  with zero synthetic log lines.
- Zero-Spoiler Prompts: User prompts describe symptoms without revealing target line diffs;
  the `runsift` arm hydrates SDCS `APP_MAP_MD` (Pillar 5) and `DECISIONS_MD` (Pillar 6)
  into the canonicalized static prefix and appends a deterministic `[SDCS Blackboard State]`
  module status footer to compacted pytest digests.
- Dual Cache & Percentile Telemetry: Records SHA-256 prefix invariance hits, upstream server-reported
  `cached_tokens` (`timings.cache_n` / `prompt_tokens_details.cached_tokens`), mean/p50/p95 TTFT,
  peak single-turn prompt tokens, and per-module completion (`modules_fixed` out of 5).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import datetime
import email.utils
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

try:
    from .repo_builder import APP_MAP_MD, DECISIONS_MD, materialize_clean_repo
    from .repo_builder_coupled import materialize_clean_repo as materialize_clean_repo_coupled
    from .tasks import (
        BENCHMARK_TASKS,
        OriginalBenchmarkTask,
        inject_bug,
        inject_gauntlet_bugs,
    )
    from .tasks_coupled import (
        COUPLED_TASKS,
        inject_all_coupled_bugs,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repo_builder import APP_MAP_MD, DECISIONS_MD, materialize_clean_repo
    from repo_builder_coupled import materialize_clean_repo as materialize_clean_repo_coupled
    from tasks import (
        BENCHMARK_TASKS,
        OriginalBenchmarkTask,
        inject_bug,
        inject_gauntlet_bugs,
    )
    from tasks_coupled import (
        COUPLED_TASKS,
        inject_all_coupled_bugs,
    )

BENCH_DIR = Path(__file__).resolve().parent
SDCS_ROOT = BENCH_DIR.parent.parent

ENC = tiktoken.get_encoding("cl100k_base")

ALL_SUBSYSTEMS = ("radix", "scheduler", "limiter", "codec", "telemetry")

VOLATILE_HEADER_RE = re.compile(
    r"^\s*(?:Current Time|Timestamp|Date|Session-ID|Request-ID|Trace-ID)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

AGENT_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_pytest",
            "description": "Runs the pytest suite in the workspace and returns test results and tracebacks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Optional test file path (default runs full 45-test suite)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads a source or documentation file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path (e.g. 'src/tensorroute/radix.py')",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replaces exact `old_text` with `new_text` in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "old_text": {
                        "type": "string",
                        "description": "Exact existing lines to replace",
                    },
                    "new_text": {"type": "string", "description": "Replacement lines"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Searches for a text pattern across `src/tensorroute/` and `tests/`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Substring or regex to search"}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declares the task complete after verifying all 45 unit tests pass.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of root cause and fix",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

BASE_AGENT_CONSTITUTION = """detailed thinking off
You are an autonomous Principal Python Systems Engineering Agent debugging the `tensorroute` library (`src/tensorroute/` and `tests/`).

Operating Protocol:
1. Use the provided tools (run_pytest, read_file, edit_file, search_code, finish) to inspect, edit, and verify code. Call one tool per response.
2. Do not guess line contents: read the target file first, apply the minimal fix with `edit_file` (do NOT include line numbers in `old_text` or `new_text`), verify with `run_pytest` that all 45 unit tests pass, and then call `finish`."""

SDCS_PILLAR_CONSTITUTION = f"""{BASE_AGENT_CONSTITUTION}

{APP_MAP_MD.strip()}

{DECISIONS_MD.strip()}"""


@dataclass
class TaskArmResult:
    task_id: str
    arm: str
    model: str
    solved: bool
    modules_fixed: str
    passed_tests: int
    failed_tests: int
    turns: int
    total_prompt_tokens: int
    peak_turn_prompt_tokens: int
    total_completion_tokens: int
    upstream_cached_tokens: int
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    per_turn_ttft_ms: list[float]
    per_turn_prompt_tokens: list[int]
    cache_hits: int
    cache_hit_rate_pct: float
    prefix_matches: int
    prefix_match_rate_pct: float
    context_overflow: bool
    final_eval_status: str = "complete"
    per_turn_tools: list[str] = field(default_factory=list)
    per_turn_tests_passing: list[int | None] = field(default_factory=list)
    patch_diff: str = ""
    nudge_count: int = 0
    nudge_rate_pct: float = 0.0
    hints_count: int = 0
    per_turn_raw_replies: list[str] = field(default_factory=list)
    per_turn_tool_feedbacks: list[str] = field(default_factory=list)
    traps_solved: int = 0
    solved_traps: list[str] = field(default_factory=list)
    regressions_introduced: list[str] = field(default_factory=list)
    regressions_repaired: int = 0
    turns_to_repair: dict[str, dict[str, int]] = field(default_factory=dict)
    final_pass_to_pass_intact: bool = True
    infra_error: bool = False
    infra_error_msg: str = ""
    wall_clock_s: float = 0.0
    per_turn_model_ms: list[float] = field(default_factory=list)
    per_turn_tool_ms: list[float] = field(default_factory=list)
    per_turn_tool_args: list[str] = field(default_factory=list)
    per_turn_condenser: list[str] = field(default_factory=list)
    per_turn_tool_chars_before: list[int] = field(default_factory=list)
    per_turn_tool_chars_after: list[int] = field(default_factory=list)
    failing_subsystems: list[str] = field(default_factory=list)
    test_files_edited: list[str] = field(default_factory=list)


class AuthQuotaAbortError(RuntimeError):
    """Raised when upstream returns HTTP 401, 402, or 403, indicating auth or quota exhaustion."""


_RETRY_BASE_BACKOFF_S: float = 1.0
_MAX_RETRY_WAIT_S: float = 60.0
_MAX_RETRIES: int = 5


def parse_retry_after(header_val: str | None) -> float | None:
    """Parses Retry-After header as seconds or RFC 7231 HTTP-date."""
    if not header_val:
        return None
    header_val = header_val.strip()
    try:
        val = float(header_val)
        return max(0.0, val)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(header_val)
        if dt is not None:
            now = datetime.timezone.utc
            diff = (dt - datetime.datetime.now(now)).total_seconds()
            return max(0.0, diff)
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def classify_upstream_error(exc: Exception | str) -> str:
    """Classifies an upstream failure into 'context_overflow', 'infra_error', or 'other'.

    'context_overflow': Upstream explicit context length / window exceeded error.
    'infra_error': Connection refused, server down, remote disconnected, broken pipe,
                   network timeout/connection error, 5xx server error,
                   502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout.
    'other': Any other client/API error.
    """
    err_str = str(exc)
    err_str_lower = err_str.lower()

    # 1. Check for true context overflow FIRST
    overflow_patterns = [
        r"request\s*\((\d+)\s*tokens\)\s*exceeds\s*(?:the\s*available\s*)?context\s*size",
        r"exceeds\s*(?:the\s*available\s*)?context\s*size",
        r"exceed_context_size_error",
        r"context_length_exceeded",
        r"maximum\s*context\s*length",
        r"context\s*window\s*exceeded",
        r"prompt\s*is\s*too\s*long",
    ]
    for pat in overflow_patterns:
        if re.search(pat, err_str_lower):
            return "context_overflow"

    # 2. Check for infrastructure errors (connection refused, server down, timeouts, 5xx)
    infra_patterns = [
        r"connection\s*refused",
        r"winerror\s*10061",
        r"errno\s*111",
        r"target machine actively refused it",
        r"remote\s*disconnected",
        r"remotedisconnected",
        r"connection\s*reset",
        r"broken\s*pipe",
        r"connection\s*aborted",
        r"connecterror",
        r"connecttimeout",
        r"timed?\s*out",
        r"timeout",
        r"http\s*429",
        r"http\s*5\d\d",
        r"5\d\d\s*server\s*error",
        r"internal\s*server\s*error",
        r"bad gateway",
        r"service unavailable",
        r"gateway timeout",
        r"unable to connect to upstream",
    ]
    for pat in infra_patterns:
        if re.search(pat, err_str_lower):
            return "infra_error"

    if isinstance(
        exc, (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, TimeoutError)
    ):
        return "infra_error"

    return "other"


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def compute_percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    n = len(ordered)
    if q <= 0.0:
        return round(ordered[0], 2)
    idx = max(0, min(n - 1, math.ceil((q / 100.0) * n) - 1))
    return round(ordered[idx], 2)


def compute_prefix_hash(system_text: str, tools: list[dict[str, Any]]) -> str:
    blob = json.dumps(
        {"system": system_text, "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonicalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Applies RunSift prefix canonicalization: hoists volatile headers and sorts tools."""
    out = json.loads(json.dumps(payload))
    messages = out.get("messages", [])
    extracted_headers: list[str] = []

    tools = out.get("tools", [])
    sorted_tools = sorted(
        tools,
        key=lambda t: str(t.get("function", {}).get("name", "")),
    )
    out["tools"] = sorted_tools
    canonical_tools_block = json.dumps(sorted_tools, sort_keys=True, separators=(",", ":"))

    for msg in messages:
        if msg.get("role") == "system":
            content = str(msg.get("content", ""))
            content = re.sub(
                r"\[Registered Tool Schemas\]:.*?\n\n",
                "",
                content,
                flags=re.DOTALL,
            )
            found = VOLATILE_HEADER_RE.findall(content)
            if found:
                extracted_headers.extend(h.strip() for h in found)
                cleaned = VOLATILE_HEADER_RE.sub("", content).strip()
            else:
                cleaned = content.strip()
            msg["content"] = f"[Registered Tool Schemas]: {canonical_tools_block}\n\n{cleaned}"

    if extracted_headers and messages:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["content"] = (
                    f"{messages[i]['content']}\n\n[Session Context: {' | '.join(extracted_headers)}]"
                )
                break

    sys_text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    p_hash = compute_prefix_hash(sys_text, sorted_tools)
    return out, p_hash


def extract_subsystem_status(
    raw_output: str,
    passed: int | None = None,
    failed: int | None = None,
    expected_tests: int = 45,
    subsystems: tuple[str, ...] | list[str] = ALL_SUBSYSTEMS,
) -> tuple[list[str], list[str]]:
    """Deterministically parses `pytest` FAILED/ERROR summary lines to identify passing vs failing modules."""
    if "[TIMEOUT]" in raw_output:
        return [], list(subsystems)
    if passed is not None and failed is not None and (passed + failed) != expected_tests:
        return [], list(subsystems)
    failed_lines = [
        ln.strip().replace("\\", "/")
        for ln in raw_output.splitlines()
        if ln.strip().startswith(("FAILED ", "ERROR "))
        or "FAILED tests/test_" in ln.replace("\\", "/")
        or "ERROR tests/test_" in ln.replace("\\", "/")
    ]
    failed_blob = "\n".join(failed_lines)
    failing_subsystems = [
        s
        for s in subsystems
        if f"tests/test_{s}.py" in failed_blob or f"test_{s}.py" in failed_blob
    ]
    passing_subsystems = [s for s in subsystems if s not in failing_subsystems]
    return passing_subsystems, failing_subsystems


def score_junit_xml(
    junit_path: Path | str,
    expected_tests: int,
    trap_count: int,
    subsystems: tuple[str, ...] | list[str] = ALL_SUBSYSTEMS,
) -> tuple[str, str, bool, list[str]]:
    """Evaluates final test results from a JUnit XML report.

    If the JUnit XML file is missing, malformed, or has a test count different from expected_tests
    (e.g., due to timeout, crash, or early abort), scores 0 traps solved and reports
    final_eval_status as 'incomplete' or 'timeout'.
    """
    path = Path(junit_path)
    if not path.exists():
        return "incomplete", f"0/{trap_count}", False, []

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()
    except (ET.ParseError, OSError, ValueError):
        return "incomplete", f"0/{trap_count}", False, []

    suite_elems = root.findall(".//testsuite") if root.tag == "testsuites" else [root]
    total_tests = 0
    total_failures = 0
    total_errors = 0
    failing_subsystems = set()

    for suite in suite_elems:
        total_tests += int(suite.attrib.get("tests", 0))
        total_failures += int(suite.attrib.get("failures", 0))
        total_errors += int(suite.attrib.get("errors", 0))

        for tc in suite.findall("testcase"):
            tc_file = tc.attrib.get("file", "").replace("\\", "/")
            tc_class = tc.attrib.get("classname", "")
            tc_name = tc.attrib.get("name", "")
            tc_parts = set(tc_class.split(".")) | set(tc_name.split("."))
            has_fail = tc.find("failure") is not None or tc.find("error") is not None
            if has_fail:
                for s in subsystems:
                    if (
                        f"test_{s}.py" in tc_file
                        or f"tests/test_{s}.py" in tc_file
                        or f"test_{s}" in tc_parts
                        or s in tc_parts
                        or f".{s}." in tc_class
                    ):
                        failing_subsystems.add(s)

    if total_tests != expected_tests or total_tests == 0:
        return "incomplete", f"0/{trap_count}", False, []

    passing_subs = [s for s in subsystems if s not in failing_subsystems]
    solved = (total_failures == 0) and (total_errors == 0) and (total_tests == expected_tests)
    return "complete", f"{len(passing_subs)}/{trap_count}", solved, passing_subs


def score_final_evaluation(
    raw_output: str,
    passed: int,
    failed: int,
    rc: int,
    expected_tests: int | None = None,
    trap_count: int | None = None,
    manifest: dict[str, Any] | Any | None = None,
    subsystems: tuple[str, ...] | list[str] = ALL_SUBSYSTEMS,
    junit_xml_path: Path | str | None = None,
) -> tuple[str, str, bool, list[str]]:
    """Scores the final pytest evaluation with explicit guards for timeouts and incomplete runs.

    Takes test count and trap count from manifest (if provided) or explicit parameters.
    If junit_xml_path is provided, inspects the JUnit XML; if missing or incomplete,
    scores 0 traps solved.

    Returns:
        tuple of (final_eval_status, modules_fixed, solved, passing_subsystems)
    """
    total_expected_tests = 45
    total_expected_traps = 5

    if manifest is not None:
        if isinstance(manifest, dict):
            if "test_count" in manifest:
                total_expected_tests = manifest["test_count"]
            elif "expected_tests" in manifest:
                total_expected_tests = manifest["expected_tests"]
            if "trap_count" in manifest:
                total_expected_traps = manifest["trap_count"]
            elif "expected_traps" in manifest:
                total_expected_traps = manifest["expected_traps"]
            if "subsystems" in manifest:
                subsystems = manifest["subsystems"]
        else:
            total_expected_tests = getattr(
                manifest, "test_count", getattr(manifest, "expected_tests", total_expected_tests)
            )
            total_expected_traps = getattr(
                manifest, "trap_count", getattr(manifest, "expected_traps", total_expected_traps)
            )
            if hasattr(manifest, "subsystems"):
                subsystems = manifest.subsystems

    if expected_tests is not None:
        total_expected_tests = expected_tests
    if trap_count is not None:
        total_expected_traps = trap_count

    if "[TIMEOUT]" in raw_output:
        return "timeout", f"0/{total_expected_traps}", False, []

    if junit_xml_path is not None:
        status, mod_str, solved, passing = score_junit_xml(
            junit_path=junit_xml_path,
            expected_tests=total_expected_tests,
            trap_count=total_expected_traps,
            subsystems=subsystems,
        )
        if status != "complete":
            return status, f"0/{total_expected_traps}", False, []
        return status, mod_str, solved, passing

    if (passed + failed) != total_expected_tests:
        return "incomplete", f"0/{total_expected_traps}", False, []

    passing_subs, _ = extract_subsystem_status(
        raw_output,
        passed=passed,
        failed=failed,
        expected_tests=total_expected_tests,
        subsystems=subsystems,
    )
    solved = (rc == 0) and (passed == total_expected_tests) and (failed == 0)
    return "complete", f"{len(passing_subs)}/{total_expected_traps}", solved, passing_subs


def parse_junit_xml_test_results(junit_xml_path: Path | str) -> dict[str, bool]:
    """Parses JUnit XML and returns mapping of test_id -> passed (True/False)."""
    p = Path(junit_xml_path)
    if not p.is_file():
        return {}
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(p)
        root = tree.getroot()
        results: dict[str, bool] = {}
        for tc in root.findall(".//testcase"):
            cls = tc.attrib.get("classname", "")
            name = tc.attrib.get("name", "")
            path_str = cls.replace(".", "/") + ".py"
            test_id = f"{path_str}::{name}"
            failed = (tc.find("failure") is not None) or (tc.find("error") is not None)
            results[test_id] = not failed
        return results
    except (ET.ParseError, OSError, ValueError, AttributeError):
        return {}


def update_regression_bookkeeping(
    current_test_results: dict[str, bool],
    initial_passing_tests: set[str],
    current_turn: int,
    regressions_introduced: list[str],
    turns_to_repair: dict[str, dict[str, int]],
) -> None:
    """Updates regression tracking when a test in initial_passing_tests fails or recovers."""
    for test_id, passed in current_test_results.items():
        if test_id in initial_passing_tests:
            if not passed:
                if test_id not in regressions_introduced:
                    regressions_introduced.append(test_id)
                if test_id not in turns_to_repair:
                    turns_to_repair[test_id] = {"introduced_turn": current_turn}
            else:
                if test_id in turns_to_repair and "repaired_turn" not in turns_to_repair[test_id]:
                    turns_to_repair[test_id]["repaired_turn"] = current_turn


def score_coupled_evaluation(
    final_test_results: dict[str, bool],
    manifest: dict[str, Any],
    regressions_introduced: list[str],
    final_eval_status: str,
) -> tuple[int, list[str], int, bool]:
    """Scores coupled traps: a trap is solved when all its F2P and guard tests pass.

    Returns:
        (traps_solved_count, solved_traps, regressions_repaired, final_pass_to_pass_intact)
    """
    if (
        final_eval_status in ("timeout", "incomplete", "infra_error", "error")
        or not final_test_results
    ):
        return 0, [], 0, False

    traps = manifest.get("traps", {})
    solved_traps: list[str] = []

    for trap_id, trap_info in traps.items():
        f2p = trap_info.get("f2p_tests", [])
        guards = trap_info.get("guard_tests", [])
        all_passed = True
        for t in f2p:
            if not final_test_results.get(t, False):
                all_passed = False
                break
        if all_passed:
            for g in guards:
                if not final_test_results.get(g, False):
                    all_passed = False
                    break
        if all_passed:
            solved_traps.append(trap_id)

    regressions_repaired = sum(
        1 for t in regressions_introduced if final_test_results.get(t, False)
    )
    initial_passing = set(manifest.get("initial_passing_tests", []))
    final_pass_to_pass_intact = (
        all(final_test_results.get(t, False) for t in initial_passing) if initial_passing else True
    )

    return len(solved_traps), solved_traps, regressions_repaired, final_pass_to_pass_intact


def run_workspace_pytest(
    ws: Path,
    target: str = "",
    expected_tests: int = 45,
    junit_xml_path: str = "",
) -> tuple[int, str, int, int]:
    """Runs 100% authentic pytest in the isolated workspace with zero synthetic log padding.

    Uses standard verbose diagnostic flags (`-vv -ra --tb=long --durations=0`).
    If a single-file `target` passes with 0 failures (`passed < expected_tests`), automatically runs
    the full test suite so multi-module Gauntlet sessions always report remaining failures.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ws / "src")
    base_cmd = [
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
    if junit_xml_path:
        base_cmd.extend(["--junitxml", str(junit_xml_path)])

    cmd = [*base_cmd, target] if (target and (ws / target).exists()) else list(base_cmd)

    try:
        res = subprocess.run(
            cmd, cwd=ws, env=env, capture_output=True, text=True, check=False, timeout=35
        )
        raw = res.stdout + res.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        raw = (
            f"{out}\n{err}\n\n[TIMEOUT] pytest exceeded 35s limit "
            f"(possible infinite loop or hang in workspace code)."
        )
        return 1, raw, 0, expected_tests

    m_pass = re.search(r"(\d+)\s+passed", raw)
    m_fail = re.search(r"(\d+)\s+failed", raw)
    passed = int(m_pass.group(1)) if m_pass else 0
    failed = int(m_fail.group(1)) if m_fail else 0

    # If the agent ran a single test file that passed, check the full test suite
    if target and failed == 0 and passed < expected_tests:
        try:
            res_full = subprocess.run(
                base_cmd, cwd=ws, env=env, capture_output=True, text=True, check=False, timeout=35
            )
            raw_full = res_full.stdout + res_full.stderr
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            raw_full = (
                f"{out}\n{err}\n\n[TIMEOUT] pytest full suite exceeded 35s limit "
                f"(possible infinite loop or hang in workspace code)."
            )
            return 1, raw_full, 0, expected_tests

        mp_f = re.search(r"(\d+)\s+passed", raw_full)
        mf_f = re.search(r"(\d+)\s+failed", raw_full)
        p_full = int(mp_f.group(1)) if mp_f else 0
        f_full = int(mf_f.group(1)) if mf_f else 0
        if (p_full + f_full) < expected_tests and res_full.returncode != 0:
            f_full = max(f_full, expected_tests - p_full)
        return (
            res_full.returncode,
            raw_full,
            p_full,
            f_full,
        )

    if target and (passed + failed) < expected_tests and res.returncode != 0:
        failed = max(failed, expected_tests - passed)

    return res.returncode, raw, passed, failed


def preflight_workspace_check(task_set: str = "original") -> None:
    """Preflight test environment verification.

    Materializes an isolated workspace and runs run_workspace_pytest.
    For original: reports exactly 12 failed, 33 passed (45 tests).
    For coupled: reports exactly 8 failed, 48 passed (56 tests).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="tensorroute_preflight_"))
    try:
        if task_set == "coupled":
            materialize_clean_repo_coupled(tmp_dir, arm="arm_b")
            inject_all_coupled_bugs(tmp_dir)
            rc, out, passed, failed = run_workspace_pytest(tmp_dir, expected_tests=56)
            if passed != 48 or failed != 8 or (passed + failed) != 56:
                raise RuntimeError(
                    f"[PREFLIGHT FAILED] Coupled test environment broken. Expected '8 failed, 48 passed' "
                    f"(total 56 tests), but run_workspace_pytest reported rc={rc}, passed={passed}, "
                    f"failed={failed}. Output:\n{out}"
                )
            print(
                f"[PREFLIGHT OK] Coupled test environment verified: {failed} failed, {passed} passed "
                f"(56/56 total tests accounted for)."
            )
        else:
            materialize_clean_repo(tmp_dir, arm="arm_b")
            inject_gauntlet_bugs(tmp_dir, BENCHMARK_TASKS[:5])
            rc, out, passed, failed = run_workspace_pytest(tmp_dir)
            if passed != 33 or failed != 12 or (passed + failed) != 45:
                raise RuntimeError(
                    f"[PREFLIGHT FAILED] Test environment broken. Expected '12 failed, 33 passed' "
                    f"(total 45 tests), but run_workspace_pytest reported rc={rc}, passed={passed}, "
                    f"failed={failed}. Output:\n{out}"
                )
            print(
                f"[PREFLIGHT OK] Test environment verified: {failed} failed, {passed} passed "
                f"(45/45 total tests accounted for)."
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


KNOWN_TOOL_NAMES = {"run_pytest", "read_file", "edit_file", "search_code", "finish"}


def _normalize_tool_dict(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if "tool" in obj:
        t_name = str(obj.get("tool") or "")
        if t_name not in KNOWN_TOOL_NAMES:
            return None
        raw_a = obj.get("args") or obj.get("arguments") or obj.get("parameters")
        if isinstance(raw_a, dict):
            return {"tool": t_name, "args": raw_a}
        extra = {
            k: v
            for k, v in obj.items()
            if k not in ("tool", "name", "args", "arguments", "parameters")
        }
        return {"tool": t_name, "args": extra}
    for k in KNOWN_TOOL_NAMES:
        if k in obj and isinstance(obj[k], dict):
            return {"tool": k, "args": obj[k]}
    return None


def _extract_json_tool_from_text(text: str) -> dict[str, Any] | None:
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        return None

    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    for b in blocks:
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            norm = _normalize_tool_dict(json.loads(b, strict=False))
            if norm is not None:
                return norm
        with contextlib.suppress(Exception):
            norm = _normalize_tool_dict(ast.literal_eval(b))
            if norm is not None:
                return norm

    idx = cleaned.find("{")
    while idx != -1:
        for end_idx in range(len(cleaned), idx, -1):
            sub = cleaned[idx:end_idx]
            if sub.endswith("}"):
                with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                    norm = _normalize_tool_dict(json.loads(sub, strict=False))
                    if norm is not None:
                        return norm
                with contextlib.suppress(Exception):
                    norm = _normalize_tool_dict(ast.literal_eval(sub))
                    if norm is not None:
                        return norm
        idx = cleaned.find("{", idx + 1)

    m_edit = re.search(
        r'["\']tool["\']\s*:\s*["\']edit_file["\'].*?["\']path["\']\s*:\s*["\']([^"\']+)["\'].*?["\']old_text["\']\s*:\s*["\']?(.*?)["\']?\s*,\s*["\']new_text["\']\s*:\s*["\']?(.*?)["\']?\s*\}',
        cleaned,
        flags=re.DOTALL,
    )
    if m_edit:
        return {
            "tool": "edit_file",
            "args": {
                "path": m_edit.group(1),
                "old_text": m_edit.group(2).replace("\\n", "\n").replace('\\"', '"'),
                "new_text": m_edit.group(3).replace("\\n", "\n").replace('\\"', '"'),
            },
        }
    return None


def parse_agent_tool_call(
    content: str,
    native_tool_calls: list[dict[str, Any]],
    reasoning_text: str = "",
) -> dict[str, Any] | None:
    """Extracts a tool call from native OpenAI `tool_calls` or markdown JSON blocks."""
    if native_tool_calls:
        tc = native_tool_calls[0]
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        raw_args = fn.get("arguments") or tc.get("args") or "{}"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args, strict=False)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {}
        else:
            args = dict(raw_args)
        if name:
            tc_id = str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}")
            return {"tool": name, "args": args, "id": tc_id}

    res = _extract_json_tool_from_text(content)
    if res is not None:
        res["id"] = f"call_{uuid.uuid4().hex[:8]}"
        return res

    if reasoning_text:
        tail_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```", reasoning_text[-1500:], flags=re.DOTALL
        )
        for b in reversed(tail_blocks):
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                obj = json.loads(b, strict=False)
                if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                    return {
                        "tool": obj.get("tool") or obj.get("name"),
                        "args": obj.get("args")
                        or obj.get("arguments")
                        or obj.get("parameters")
                        or {},
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                    }

    return None


def execute_tool_on_workspace(
    ws: Path,
    tool_name: str,
    args: dict[str, Any],
    compact: bool = False,
    include_sdcs_footer: bool = False,
    task_set: str = "original",
    expected_tests: int = 45,
    junit_xml_path: Path | str = "",
) -> tuple[str, bool, int | None]:
    """Executes the requested tool against the isolated workspace. Returns (output, finished, tests_passing)."""
    if tool_name == "run_pytest":
        target = str(args.get("target") or "")
        rc, full_log, passed, failed = run_workspace_pytest(
            ws, target=target, expected_tests=expected_tests, junit_xml_path=junit_xml_path
        )
        tests_passing = passed
        if target and (passed + failed) < expected_tests:
            _, _, p_full, _ = run_workspace_pytest(
                ws, expected_tests=expected_tests, junit_xml_path=junit_xml_path
            )
            tests_passing = p_full
        is_finished = rc == 0 and passed == expected_tests and failed == 0
        return full_log, is_finished, tests_passing

    if tool_name == "read_file":
        rel_path = str(args.get("path") or "").lstrip("/\\")
        fpath = (ws / rel_path).resolve()
        if not str(fpath).startswith(str(ws.resolve())) or not fpath.is_file():
            return f"Error: File '{rel_path}' not found.", False, None
        text = fpath.read_text(encoding="utf-8")
        lines = text.splitlines()
        return f"File `{rel_path}` ({len(lines)} lines):\n```python\n{text}\n```", False, None

    if tool_name == "edit_file":
        import ast as _ast

        rel_path = str(args.get("path") or "").lstrip("/\\")
        norm_path = rel_path.replace("\\", "/")
        if task_set == "coupled" and (norm_path.startswith("tests/") or norm_path == "tests"):
            return "Error: tests are read-only.", False, None

        old_text = str(args.get("old_text") or "")
        new_text = str(args.get("new_text") or "")
        fpath = (ws / rel_path).resolve()
        if not str(fpath).startswith(str(ws.resolve())) or not fpath.is_file():
            return f"Error: File '{rel_path}' not found.", False, None

        old_clean = "\n".join(re.sub(r"^\s*\d+:\s?", "", ln) for ln in old_text.splitlines())
        new_clean = "\n".join(re.sub(r"^\s*\d+:\s?", "", ln) for ln in new_text.splitlines())

        content = fpath.read_text(encoding="utf-8")
        candidates: list[str] = []

        if old_clean and old_clean in content:
            idx_pos = content.find(old_clean)
            line_start = content.rfind("\n", 0, idx_pos) + 1
            prefix_ws = content[line_start:idx_pos]
            if not prefix_ws:
                candidates.append(content.replace(old_clean, new_clean, 1))
            else:
                if new_clean.startswith(prefix_ws):
                    candidates.append(
                        content[:line_start] + new_clean + content[idx_pos + len(old_clean) :]
                    )
                candidates.append(
                    content[:idx_pos] + new_clean.lstrip(" ") + content[idx_pos + len(old_clean) :]
                )
                candidates.append(
                    content[:line_start] + new_clean + content[idx_pos + len(old_clean) :]
                )

        if old_clean.strip():
            file_lines = content.splitlines()
            old_lines = [ln for ln in old_clean.splitlines() if ln.strip()]
            old_norm = [ln.strip().replace('"', "'") for ln in old_lines]
            n_old = len(old_norm)
            for i in range(len(file_lines)):
                window = file_lines[i : i + n_old]
                if (
                    len(window) == n_old
                    and [w.strip().replace('"', "'") for w in window] == old_norm
                ):
                    base_indent_match = re.match(r"^(\s*)", window[0])
                    base_indent = base_indent_match.group(1) if base_indent_match else ""
                    min_win_indent_len = min(
                        (len(w) - len(w.lstrip()) for w in window if w.strip()),
                        default=0,
                    )
                    min_win_indent = " " * min_win_indent_len
                    new_raw_lines = new_clean.splitlines()
                    non_empty_new = [ln for ln in new_raw_lines if ln.strip()]
                    first_indent = (
                        len(new_raw_lines[0]) - len(new_raw_lines[0].lstrip())
                        if new_raw_lines and new_raw_lines[0].strip()
                        else 0
                    )
                    rest_indents = [
                        len(ln) - len(ln.lstrip()) for ln in new_raw_lines[1:] if ln.strip()
                    ]
                    min_new_indent = (
                        min(len(ln) - len(ln.lstrip()) for ln in non_empty_new)
                        if non_empty_new
                        else 0
                    )

                    rep_options: list[list[str]] = []
                    if (
                        first_indent == 0
                        and rest_indents
                        and min(rest_indents) >= min_win_indent_len
                        and len(base_indent) > 0
                    ):
                        rep_options.append(
                            [base_indent + new_raw_lines[0].lstrip(), *new_raw_lines[1:]]
                        )
                    if first_indent == len(base_indent):
                        rep_options.append(list(new_raw_lines))
                    rep_options.append(
                        [
                            (min_win_indent + ln[min_new_indent:]) if ln.strip() else ""
                            for ln in new_raw_lines
                        ]
                    )
                    rep_options.append(
                        [
                            (base_indent + ln[min_new_indent:]) if ln.strip() else ""
                            for ln in new_raw_lines
                        ]
                    )
                    for rep in rep_options:
                        updated_lines = file_lines[:i] + rep + file_lines[i + n_old :]
                        candidates.append("\n".join(updated_lines) + "\n")
                    break

        if not candidates:
            return (
                f"Error: `old_text` not found in `{rel_path}`. Copy exact lines from `{rel_path}`.",
                False,
                None,
            )

        selected: str | None = None
        last_syntax_err: SyntaxError | None = None
        for cand in candidates:
            if fpath.suffix == ".py":
                try:
                    _ast.parse(cand)
                    compile(cand, str(fpath), "exec")
                    selected = cand
                    break
                except SyntaxError as exc:
                    last_syntax_err = exc
            else:
                selected = cand
                break

        if selected is None:
            lineno = last_syntax_err.lineno if last_syntax_err else 0
            msg = last_syntax_err.msg if last_syntax_err else "invalid syntax"
            return (
                f"Error: Edit rejected due to Python SyntaxError on line {lineno}: {msg}. Check indentation.",
                False,
                None,
            )

        fpath.write_text(selected, encoding="utf-8")
        return f"Successfully updated `{rel_path}` (syntax verified).", False, None

    if tool_name == "search_code":
        pattern = str(args.get("pattern") or "")
        matches: list[str] = []
        for py in sorted(ws.rglob("*.py")):
            for idx, ln in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if pattern in ln:
                    matches.append(f"{py.relative_to(ws).as_posix()}:{idx}: {ln.strip()}")
        return "\n".join(matches[:30]) if matches else "No matches found.", False, None

    if tool_name == "finish":
        return f"Task finished: {args.get('summary', '')}", True, None

    return (
        f"Unknown tool '{tool_name}'. Use run_pytest, read_file, edit_file, search_code, or finish.",
        False,
        None,
    )


_PERSISTENT_CONNS: dict[str, Any] = {}


def _get_persistent_conn(base_url: str) -> tuple[Any, str]:
    """Returns a persistent HTTP/1.1 keep-alive connection and base path for `base_url`."""
    import http.client
    import urllib.parse

    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    scheme = parsed.scheme or "http"
    host = parsed.netloc
    key = f"{scheme}://{host}"
    conn = _PERSISTENT_CONNS.get(key)
    if conn is None:
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, timeout=180)
        else:
            conn = http.client.HTTPConnection(host, timeout=180)
        _PERSISTENT_CONNS[key] = conn
    return conn, parsed.path


def _reset_persistent_conn(base_url: str) -> None:
    import urllib.parse

    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    key = f"{parsed.scheme or 'http'}://{parsed.netloc}"
    conn = _PERSISTENT_CONNS.pop(key, None)
    if conn is not None:
        with contextlib.suppress(OSError):
            conn.close()


_RECORDED_TURN1_REQUESTS: dict[str, dict[str, Any]] = {}


def verify_and_log_turn1_diff(base_req: dict[str, Any], cf_req: dict[str, Any]) -> dict[str, Any]:
    """Audits turn-1 request payloads between baseline and RunSift arms."""
    diffs: dict[str, Any] = {}
    all_keys = set(base_req.keys()) | set(cf_req.keys())
    for k in sorted(all_keys):
        v1 = base_req.get(k)
        v2 = cf_req.get(k)
        if v1 != v2:
            diffs[k] = {"baseline": v1, "runsift": v2, "cacheforge": v2}

    execution_flags = [
        "model",
        "temperature",
        "stream",
        "max_tokens",
        "stream_options",
        "chat_template_kwargs",
        "cache_prompt",
    ]
    flag_mismatches = [f for f in execution_flags if f in diffs]
    if flag_mismatches:
        print(f"    [TURN-1 DIFF AUDIT] ❌ Flag mismatch between arms: {flag_mismatches}")
        for f in flag_mismatches:
            print(f"        {f}: baseline={base_req.get(f)!r} vs runsift={cf_req.get(f)!r}")
    else:
        print("    [TURN-1 DIFF AUDIT] ✓ Request execution flags match identically between arms.")
        print(
            f"        Shared execution flags: stream={base_req.get('stream')}, max_tokens={base_req.get('max_tokens')}, "
            f"temperature={base_req.get('temperature')}, chat_template_kwargs={base_req.get('chat_template_kwargs')}, "
            f"stream_options={base_req.get('stream_options')}"
        )
    return diffs


def stream_chat_completion_sse(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    prefix_hash: str,
    seen_prefixes: set[str],
    extra_headers: dict[str, str] | None = None,
    upstream_kind: str = "nebius",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Streams `/v1/chat/completions` over persistent HTTP/1.1 Keep-Alive SSE."""
    req_payload = dict(payload)
    # Native OpenAI tool calling: preserve tools schema in payload
    req_payload["stream"] = True
    req_payload["temperature"] = payload.get("temperature", 0.0)
    req_payload.setdefault("max_tokens", 1024)

    if upstream_kind == "llamacpp":
        req_payload["cache_prompt"] = True
        req_payload.setdefault("stream_options", {"include_usage": True})
    elif upstream_kind == "nebius":
        req_payload.setdefault("stream_options", {"include_usage": True})
        req_payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
    else:
        req_payload.setdefault("stream_options", {"include_usage": True})

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Connection": "keep-alive",
        "User-Agent": "RunSift-TensorRouteBench/2.0.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if prefix_hash:
        headers["x-session-id"] = prefix_hash[:16]
    if extra_headers:
        headers.update(extra_headers)

    body_bytes = json.dumps(req_payload).encode("utf-8")
    headers["Content-Length"] = str(len(body_bytes))

    t0 = time.perf_counter()
    ttft_ms = 0.0
    ttft_recorded = False
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_map: dict[int, dict[str, Any]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    upstream_cached = 0
    is_loop_hint = False

    for attempt in range(1, _MAX_RETRIES + 1):
        conn, base_path = _get_persistent_conn(base_url)
        endpoint_path = f"{base_path.rstrip('/')}/chat/completions"
        try:
            t0 = time.perf_counter()
            conn.request("POST", endpoint_path, body=body_bytes, headers=headers)
            resp = conn.getresponse()

            if resp.status in (401, 402, 403):
                err_body = resp.read().decode("utf-8", errors="replace")
                _reset_persistent_conn(base_url)
                raise AuthQuotaAbortError(f"HTTP {resp.status}: {err_body[:400]}")

            if resp.status == 429 or (500 <= resp.status < 600):
                err_body = resp.read().decode("utf-8", errors="replace")
                _reset_persistent_conn(base_url)
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"HTTP {resp.status} after {_MAX_RETRIES} attempts: {err_body[:400]}"
                    )
                wait_sec = parse_retry_after(resp.getheader("Retry-After"))
                if wait_sec is None:
                    wait_sec = min(_MAX_RETRY_WAIT_S, _RETRY_BASE_BACKOFF_S * (2 ** (attempt - 1)))
                else:
                    wait_sec = min(_MAX_RETRY_WAIT_S, wait_sec)
                print(
                    f"    [Retry {attempt}/{_MAX_RETRIES}] HTTP {resp.status} - waiting {wait_sec:.2f}s before retry..."
                )
                time.sleep(wait_sec)
                continue

            if resp.status >= 400:
                err_body = resp.read().decode("utf-8", errors="replace")
                _reset_persistent_conn(base_url)
                raise RuntimeError(f"HTTP {resp.status}: {err_body[:400]}")

            cf_p_hash = resp.getheader("x-runsift-prefix-hash") or resp.getheader("x-cacheforge-prefix-hash")
            if cf_p_hash:
                prefix_hash = cf_p_hash
            cf_loop_hint = resp.getheader("x-runsift-loop-hint") or resp.getheader("x-cacheforge-loop-hint")
            is_loop_hint = cf_loop_hint in ("true", "True", "1")
            cf_condenser = resp.getheader("x-runsift-condenser") or resp.getheader("x-cacheforge-condenser") or ""
            cf_chars_before_raw = resp.getheader("x-runsift-tool-chars-before") or resp.getheader("x-cacheforge-tool-chars-before")
            cf_chars_after_raw = resp.getheader("x-runsift-tool-chars-after") or resp.getheader("x-cacheforge-tool-chars-after")
            cf_chars_before = (
                int(cf_chars_before_raw)
                if cf_chars_before_raw is not None and cf_chars_before_raw.strip().isdigit()
                else 0
            )
            cf_chars_after = (
                int(cf_chars_after_raw)
                if cf_chars_after_raw is not None and cf_chars_after_raw.strip().isdigit()
                else 0
            )

            while True:
                raw_bytes = resp.readline()
                if not raw_bytes:
                    break
                line = raw_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    upstream_cached = int(
                        details.get("cached_tokens")
                        or usage.get("prompt_cache_hit_tokens")
                        or upstream_cached
                    )

                timings = chunk.get("timings")
                if isinstance(timings, dict):
                    cache_n = int(timings.get("cache_n") or 0)
                    prompt_n = int(timings.get("prompt_n") or 0)
                    if cache_n > 0:
                        upstream_cached = max(upstream_cached, cache_n)
                    if prompt_tokens == 0 and (cache_n + prompt_n) > 0:
                        prompt_tokens = cache_n + prompt_n

                choices = chunk.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    delta = choices[0].get("delta") or {}
                    has_token = bool(
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("tool_calls")
                    )
                    if has_token and not ttft_recorded:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                        ttft_recorded = True
                    if delta.get("content"):
                        content_parts.append(str(delta["content"]))
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(str(delta["reasoning_content"]))
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        entry = tool_calls_map.setdefault(
                            idx,
                            {
                                "id": tc.get("id") or f"call_{idx}",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc.get("id"):
                            entry["id"] = str(tc["id"])
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["function"]["name"] += str(fn["name"])
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += str(fn["arguments"])
            resp.read()
            break
        except AuthQuotaAbortError:
            raise
        except Exception as exc:
            _reset_persistent_conn(base_url)
            is_timeout = (
                isinstance(exc, (TimeoutError, socket.timeout))
                or "timed out" in str(exc).lower()
                or "timeout" in str(exc).lower()
            )
            if is_timeout:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"Upstream timeout after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                wait_sec = min(_MAX_RETRY_WAIT_S, _RETRY_BASE_BACKOFF_S * (2 ** (attempt - 1)))
                print(
                    f"    [Retry {attempt}/{_MAX_RETRIES}] Timeout ({exc}) - waiting {wait_sec:.2f}s before retry..."
                )
                time.sleep(wait_sec)
                continue

            # Transient keep-alive connection reset/disconnect
            is_conn_drop = (
                isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError))
                or "remotedisconnected" in str(exc).lower()
                or "10053" in str(exc)
            )
            if is_conn_drop and attempt < 2:
                print(
                    f"    [Retry {attempt}/{_MAX_RETRIES}] Connection reset on keep-alive socket - retrying with fresh connection..."
                )
                continue
            raise

    model_ms = (time.perf_counter() - t0) * 1000.0
    if not ttft_recorded:
        ttft_ms = model_ms

    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)

    if prompt_tokens == 0:
        prompt_tokens = sum(
            count_tokens(str(m.get("content", ""))) for m in req_payload.get("messages", [])
        )
    if completion_tokens == 0:
        completion_tokens = max(1, count_tokens(full_content) + count_tokens(full_reasoning))

    is_prefix_matched = prefix_hash in seen_prefixes if prefix_hash else False
    if prefix_hash:
        seen_prefixes.add(prefix_hash)

    is_server_cache_hit = upstream_cached > 0

    ordered_tool_calls = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())]
    meta = {
        "ttft_ms": round(ttft_ms, 2),
        "model_ms": round(model_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "upstream_cached_tokens": upstream_cached,
        "cache_hit": is_server_cache_hit,
        "prefix_match": is_prefix_matched,
        "loop_hint": is_loop_hint,
        "condenser": cf_condenser,
        "tool_chars_before": cf_chars_before,
        "tool_chars_after": cf_chars_after,
    }
    return full_content, ordered_tool_calls, meta


def _execute_agent_session(
    session_id: str,
    arm: str,
    tmp_ws: Path,
    initial_user_prompt: str,
    base_url: str,
    api_key: str,
    model_name: str,
    max_turns: int,
    seen_prefixes: set[str],
    gateway_url: str = "",
    ablation: str = "both",
    upstream_kind: str = "nebius",
    include_sdcs_footer: bool = False,
    temperature: float = 0.0,
    expected_tests: int | None = None,
    trap_count: int | None = None,
    manifest: dict[str, Any] | Any | None = None,
    task_set: str = "original",
    sidecar_records: list[dict[str, Any]] | None = None,
) -> TaskArmResult:
    """Runs the multi-turn agent loop on an already-materialized workspace `tmp_ws`."""
    _rng = random.Random(f"seed:{arm}:{session_id}")
    is_cf = arm in ("runsift", "cacheforge", "arm_b")
    # Both arms receive the plain BASE_AGENT_CONSTITUTION: no app map, no decisions text
    constitution = BASE_AGENT_CONSTITUTION

    total_expected_tests = 45
    total_expected_traps = 5
    if manifest is not None and isinstance(manifest, dict):
        total_expected_tests = manifest.get("test_count", total_expected_tests)
        total_expected_traps = manifest.get("trap_count", total_expected_traps)
    if expected_tests is not None:
        total_expected_tests = expected_tests
    if trap_count is not None:
        total_expected_traps = trap_count

    initial_passing_tests: set[str] = set()
    if task_set == "coupled" and manifest and isinstance(manifest, dict):
        initial_passing_tests = set(manifest.get("initial_passing_tests", []))

    regressions_introduced: list[str] = []
    turns_to_repair: dict[str, dict[str, int]] = {}

    history: list[dict[str, Any]] = [{"role": "user", "content": initial_user_prompt}]

    ttft_list: list[float] = []
    model_ms_list: list[float] = []
    tool_ms_list: list[float] = []
    tool_args_list: list[str] = []
    condenser_list: list[str] = []
    tool_chars_before_list: list[int] = []
    tool_chars_after_list: list[int] = []
    prompt_tokens_list: list[int] = []
    total_completion_tokens = 0
    upstream_cached_total = 0
    cache_hits = 0
    prefix_matches = 0
    turns_taken = 0
    nudge_count = 0
    hints_count = 0
    context_overflow = False
    infra_error = False
    infra_error_msg = ""
    tools_list: list[str] = []
    tests_passing_list: list[int | None] = []
    raw_replies_list: list[str] = []
    tool_feedbacks_list: list[str] = []

    t_session_start: float | None = None

    for turn in range(max_turns):
        turns_taken = turn + 1
        # Realistic client baseline: stable daily timestamp, session-scoped ID, and stable tool order
        stable_tools = list(AGENT_TOOLS_SCHEMA)
        tools_block = json.dumps(stable_tools, separators=(",", ":"))

        sys_prompt = (
            f"Current Date: 2026-10-15 UTC\n"
            f"Session-ID: sess-{session_id}\n"
            f"[Registered Tool Schemas]: {tools_block}\n\n"
            f"{constitution}"
        )

        raw_payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "system", "content": sys_prompt}, *history],
            "tools": stable_tools,
            "temperature": temperature,
            "max_tokens": 1024,
            "stream": True,
        }
        if upstream_kind == "llamacpp":
            raw_payload["cache_prompt"] = True
            raw_payload.setdefault("stream_options", {"include_usage": True})
        elif upstream_kind == "nebius":
            raw_payload["stream_options"] = {"include_usage": True}
            raw_payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            raw_payload["stream_options"] = {"include_usage": True}

        if turn == 0:
            _RECORDED_TURN1_REQUESTS[arm] = copy.deepcopy(raw_payload)
            if arm in ("runsift", "cacheforge", "arm_b") and "baseline" in _RECORDED_TURN1_REQUESTS:
                verify_and_log_turn1_diff(
                    _RECORDED_TURN1_REQUESTS["baseline"], _RECORDED_TURN1_REQUESTS[arm]
                )

        do_canonicalize = is_cf and (ablation in ("both", "canonicalize-only"))
        do_compact = is_cf and (ablation in ("both", "compact-only"))

        extra_hdrs: dict[str, str] = {}
        if is_cf and gateway_url:
            # Route payload directly through live RunSift proxy gateway
            target_url = gateway_url
            dispatch_payload = raw_payload
            p_hash = ""
            if not do_canonicalize:
                extra_hdrs["x-runsift-disable-canonicalizer"] = "true"
                extra_hdrs["x-cacheforge-disable-canonicalizer"] = "true"
            if not do_compact:
                extra_hdrs["x-runsift-disable-compaction"] = "true"
                extra_hdrs["x-cacheforge-disable-compaction"] = "true"
        elif do_canonicalize:
            target_url = base_url
            dispatch_payload, p_hash = canonicalize_payload(raw_payload)
        else:
            target_url = base_url
            dispatch_payload = raw_payload
            p_hash = compute_prefix_hash(sys_prompt, stable_tools)

        if t_session_start is None:
            t_session_start = time.perf_counter()

        t_req_start = time.perf_counter()
        try:
            resp_text, native_tcs, meta = stream_chat_completion_sse(
                base_url=target_url,
                api_key=api_key,
                payload=dispatch_payload,
                prefix_hash=p_hash,
                seen_prefixes=seen_prefixes,
                extra_headers=extra_hdrs,
                upstream_kind=upstream_kind,
            )
        except AuthQuotaAbortError:
            raise
        except (RuntimeError, OSError, ValueError) as exc:
            err_model_ms = round((time.perf_counter() - t_req_start) * 1000.0, 2)
            model_ms_list.append(err_model_ms)
            tool_ms_list.append(0.0)
            tool_args_list.append("")
            if is_cf:
                condenser_list.append("")
                tool_chars_before_list.append(0)
                tool_chars_after_list.append(0)
            err_str = str(exc)
            err_kind = classify_upstream_error(exc)
            if err_kind == "infra_error":
                print(
                    f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken}/{max_turns}] "
                    f"INFRASTRUCTURE ERROR / SERVER UNREACHABLE: {err_str:.120s}"
                )
                infra_error = True
                infra_error_msg = err_str
                context_overflow = False
                tools_list.append("infra_error")
                tests_passing_list.append(None)
                raw_replies_list.append(f"[INFRA_ERROR: {err_str[:800]}]")
                tool_feedbacks_list.append("[INFRASTRUCTURE ERROR - SERVER UNREACHABLE]")
                if sidecar_records is not None:
                    sidecar_records.append(
                        {
                            "arm": arm,
                            "turn": turns_taken,
                            "tool": "infra_error",
                            "args": {},
                            "raw_output": "",
                        }
                    )
                break
            elif err_kind == "context_overflow":
                m_ovf = re.search(r"request\s*\((\d+)\s*tokens\)", err_str)
                ovf_tokens = (
                    int(m_ovf.group(1))
                    if m_ovf
                    else sum(
                        count_tokens(str(m.get("content", "")))
                        for m in dispatch_payload.get("messages", [])
                    )
                )
                if ovf_tokens > 0:
                    prompt_tokens_list.append(ovf_tokens)
                print(
                    f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken}/{max_turns}] "
                    f"UPSTREAM ERROR / CONTEXT OVERFLOW ({ovf_tokens} tokens): {err_str:.120s}"
                )
                context_overflow = True
                tools_list.append("overflow")
                tests_passing_list.append(None)
                raw_replies_list.append(f"[EXCEPTION: {err_str[:800]}]")
                tool_feedbacks_list.append("[CONTEXT OVERFLOW - NO FEEDBACK]")
                if sidecar_records is not None:
                    sidecar_records.append(
                        {
                            "arm": arm,
                            "turn": turns_taken,
                            "tool": "overflow",
                            "args": {},
                            "raw_output": "",
                        }
                    )
                break
            else:
                print(
                    f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken}/{max_turns}] "
                    f"UPSTREAM RUNTIME ERROR: {err_str:.120s}"
                )
                infra_error = True
                infra_error_msg = err_str
                context_overflow = False
                tools_list.append("error")
                tests_passing_list.append(None)
                raw_replies_list.append(f"[ERROR: {err_str[:800]}]")
                tool_feedbacks_list.append("[UPSTREAM ERROR - NO FEEDBACK]")
                if sidecar_records is not None:
                    sidecar_records.append(
                        {
                            "arm": arm,
                            "turn": turns_taken,
                            "tool": "error",
                            "args": {},
                            "raw_output": "",
                        }
                    )
                break

        ttft_list.append(float(meta["ttft_ms"]))
        model_ms_list.append(float(meta.get("model_ms", meta["ttft_ms"])))
        if is_cf:
            condenser_list.append(str(meta.get("condenser", "")))
            tool_chars_before_list.append(int(meta.get("tool_chars_before", 0)))
            tool_chars_after_list.append(int(meta.get("tool_chars_after", 0)))
        prompt_tokens_list.append(int(meta["prompt_tokens"]))
        total_completion_tokens += int(meta["completion_tokens"])
        upstream_cached_total += int(meta.get("upstream_cached_tokens", 0))
        if meta["cache_hit"]:
            cache_hits += 1
        if meta.get("prefix_match"):
            prefix_matches += 1
        if meta.get("loop_hint"):
            hints_count += 1

        raw_reply_trunc = resp_text[:800]
        raw_replies_list.append(raw_reply_trunc)

        parsed_tc = parse_agent_tool_call(resp_text, native_tcs)
        loop_str = " | LoopHint: TRUE" if meta.get("loop_hint") else ""
        print(
            f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d}] "
            f"TTFT: {meta['ttft_ms']:6.1f}ms | PromptTok: {meta['prompt_tokens']:6d} | "
            f"CachedTok: {meta['upstream_cached_tokens']:6d} | ServerHit: {meta['cache_hit']!s:5s} | "
            f"PrefixMatch: {meta.get('prefix_match')!s:5s}{loop_str} | Tool: {parsed_tc['tool'] if parsed_tc else 'none'}"
        )

        if not parsed_tc:
            nudge_count += 1
            tools_list.append("none")
            tool_ms_list.append(0.0)
            tool_args_list.append("")
            tests_passing_list.append(None)
            history.append({"role": "assistant", "content": resp_text})
            nudge_msg = (
                "Please invoke one of the provided tools to inspect files, run tests, or edit code."
            )
            tool_feedback_trunc = nudge_msg[:1500]
            tool_feedbacks_list.append(tool_feedback_trunc)
            if sidecar_records is not None:
                sidecar_records.append(
                    {
                        "arm": arm,
                        "turn": turns_taken,
                        "tool": "none",
                        "args": {},
                        "raw_output": "",
                    }
                )
            print(
                f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | MODEL RAW REPLY (first 800 chars)]:\n"
                f"{raw_reply_trunc}\n"
                f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | TEXT SHOWN TO MODEL (first 1500 chars)]:\n"
                f"{tool_feedback_trunc}\n"
            )
            history.append(
                {
                    "role": "user",
                    "content": nudge_msg,
                }
            )
            continue

        tc_id = str(parsed_tc.get("id") or f"call_{uuid.uuid4().hex[:8]}")
        tool_used = str(parsed_tc["tool"])
        tool_args = dict(parsed_tc.get("args") or {})
        tool_args_json = json.dumps(tool_args)
        tool_args_list.append(tool_args_json)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": resp_text if resp_text else None,
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tool_used,
                        "arguments": tool_args_json,
                    },
                }
            ],
        }
        history.append(assistant_msg)
        tools_list.append(tool_used)

        t_tool_start = time.perf_counter()
        turn_junit = tmp_ws / f"pytest_turn_{turns_taken}.xml"
        tool_out, is_finished, passed_cnt = execute_tool_on_workspace(
            tmp_ws,
            tool_name=tool_used,
            args=tool_args,
            task_set=task_set,
            expected_tests=total_expected_tests,
            junit_xml_path=turn_junit if tool_used == "run_pytest" else "",
        )
        if task_set == "coupled" and tool_used == "run_pytest" and turn_junit.is_file():
            test_res = parse_junit_xml_test_results(turn_junit)
            update_regression_bookkeeping(
                test_res,
                initial_passing_tests,
                turns_taken,
                regressions_introduced,
                turns_to_repair,
            )

        if is_finished:
            finish_junit = tmp_ws / f"pytest_finish_{turns_taken}.xml"
            rc_check, full_check_log, p_cnt, f_cnt = run_workspace_pytest(
                tmp_ws, expected_tests=total_expected_tests, junit_xml_path=finish_junit
            )
            tool_ms_list.append(round((time.perf_counter() - t_tool_start) * 1000.0, 2))
            tests_passing_list.append(p_cnt)
            if rc_check == 0 and p_cnt == total_expected_tests and f_cnt == 0:
                finished_msg = "[TASK FINISHED - ALL TESTS PASSED]"
                tool_feedback_trunc = finished_msg[:1500]
                tool_feedbacks_list.append(tool_feedback_trunc)
                if sidecar_records is not None:
                    sidecar_records.append(
                        {
                            "arm": arm,
                            "turn": turns_taken,
                            "tool": tool_used,
                            "args": tool_args,
                            "raw_output": finished_msg,
                        }
                    )
                print(
                    f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | MODEL RAW REPLY (first 800 chars)]:\n"
                    f"{raw_reply_trunc}\n"
                    f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | TEXT SHOWN TO MODEL (first 1500 chars)]:\n"
                    f"{tool_feedback_trunc}\n"
                )
                break
            rejected_msg = (
                f"`finish` rejected because pytest still has {f_cnt} failure(s) ({p_cnt}/{total_expected_tests} passed). "
                f"Fix the remaining failing module(s) before calling `finish`:\n{full_check_log}"
            )
            tool_feedback_trunc = rejected_msg[:1500]
            tool_feedbacks_list.append(tool_feedback_trunc)
            if sidecar_records is not None:
                sidecar_records.append(
                    {
                        "arm": arm,
                        "turn": turns_taken,
                        "tool": tool_used,
                        "args": tool_args,
                        "raw_output": rejected_msg,
                    }
                )
            print(
                f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | MODEL RAW REPLY (first 800 chars)]:\n"
                f"{raw_reply_trunc}\n"
                f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | TEXT SHOWN TO MODEL (first 1500 chars)]:\n"
                f"{tool_feedback_trunc}\n"
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": "finish",
                    "content": rejected_msg,
                }
            )
            continue

        tool_ms_list.append(round((time.perf_counter() - t_tool_start) * 1000.0, 2))
        tests_passing_list.append(passed_cnt)
        tool_feedback_trunc = tool_out[:1500]
        tool_feedbacks_list.append(tool_feedback_trunc)
        if sidecar_records is not None:
            sidecar_records.append(
                {
                    "arm": arm,
                    "turn": turns_taken,
                    "tool": tool_used,
                    "args": tool_args,
                    "raw_output": tool_out,
                }
            )
        print(
            f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | MODEL RAW REPLY (first 800 chars)]:\n"
            f"{raw_reply_trunc}\n"
            f"    [{arm.upper():10s} | {session_id} | Turn {turns_taken:2d}/{max_turns:2d} | TEXT SHOWN TO MODEL (first 1500 chars)]:\n"
            f"{tool_feedback_trunc}\n"
        )
        history.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": tool_used,
                "content": tool_out,
            }
        )

    wall_clock_s = (
        round(time.perf_counter() - t_session_start, 3) if t_session_start is not None else 0.0
    )

    diff_tests_res = subprocess.run(
        ["git", "diff", "--name-only", "tests/"],
        cwd=tmp_ws,
        capture_output=True,
        text=True,
        check=False,
    )
    edited_set = {
        p.strip().replace("\\", "/") for p in diff_tests_res.stdout.splitlines() if p.strip()
    }
    status_tests_res = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "tests/"],
        cwd=tmp_ws,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in status_tests_res.stdout.splitlines():
        if line.startswith("?? ") and len(line) > 3:
            path_part = line[3:].strip().replace("\\", "/")
            if path_part and "__pycache__" not in path_part and not path_part.endswith(".pyc"):
                edited_set.add(path_part)
    test_files_edited = sorted(edited_set)
    subprocess.run(["git", "checkout", "--", "tests/"], cwd=tmp_ws, check=False)
    subprocess.run(["git", "clean", "-fd", "tests/"], cwd=tmp_ws, check=False)

    failing_subsystems: list[str] = []
    if infra_error:
        final_eval_status = "infra_error"
        modules_fixed_str = f"0/{total_expected_traps}"
        solved = False
        traps_solved_cnt = 0
        solved_traps = []
        regressions_repaired = 0
        final_pass_to_pass_intact = False
        passed_final = 0
        failed_final = total_expected_tests
        failing_subsystems = list(ALL_SUBSYSTEMS)
    else:
        final_xml = tmp_ws / "pytest_final.xml"
        rc_final, raw_final, passed_final, failed_final = run_workspace_pytest(
            tmp_ws, expected_tests=total_expected_tests, junit_xml_path=final_xml
        )
        final_eval_status, modules_fixed_str, solved, _passing_subs = score_final_evaluation(
            raw_output=raw_final,
            passed=passed_final,
            failed=failed_final,
            rc=rc_final,
            expected_tests=total_expected_tests,
            trap_count=total_expected_traps,
            manifest=manifest,
            junit_xml_path=final_xml,
        )
        failing_subsystems = [s for s in ALL_SUBSYSTEMS if s not in _passing_subs]

        traps_solved_cnt = 0
        solved_traps = []
        regressions_repaired = 0
        final_pass_to_pass_intact = True

        if task_set == "coupled":
            final_test_results = parse_junit_xml_test_results(final_xml)
            traps_solved_cnt, solved_traps, regressions_repaired, final_pass_to_pass_intact = (
                score_coupled_evaluation(
                    final_test_results=final_test_results,
                    manifest=manifest if isinstance(manifest, dict) else {},
                    regressions_introduced=regressions_introduced,
                    final_eval_status=final_eval_status,
                )
            )
            modules_fixed_str = f"{traps_solved_cnt}/{total_expected_traps}"
            solved = traps_solved_cnt == total_expected_traps
            if (
                any(
                    k.startswith("tests/test_scheduler.py::") and not v
                    for k, v in final_test_results.items()
                )
                and "scheduler" not in failing_subsystems
            ):
                failing_subsystems.append("scheduler")

    diff_res = subprocess.run(
        ["git", "diff", "src/"],
        cwd=tmp_ws,
        capture_output=True,
        text=True,
        check=False,
    )
    patch_diff = diff_res.stdout

    nudge_rate = round((nudge_count / max(1, turns_taken)) * 100.0, 2)
    return TaskArmResult(
        task_id=session_id,
        arm=arm,
        model=model_name,
        solved=solved,
        modules_fixed=modules_fixed_str,
        passed_tests=passed_final,
        failed_tests=failed_final,
        turns=turns_taken,
        total_prompt_tokens=sum(prompt_tokens_list),
        peak_turn_prompt_tokens=max(prompt_tokens_list) if prompt_tokens_list else 0,
        total_completion_tokens=total_completion_tokens,
        upstream_cached_tokens=upstream_cached_total,
        avg_ttft_ms=round(sum(ttft_list) / max(1, len(ttft_list)), 2),
        p50_ttft_ms=compute_percentile(ttft_list, 50.0),
        p95_ttft_ms=compute_percentile(ttft_list, 95.0),
        per_turn_ttft_ms=ttft_list,
        per_turn_prompt_tokens=prompt_tokens_list,
        per_turn_tools=tools_list,
        per_turn_tests_passing=tests_passing_list,
        cache_hits=cache_hits,
        cache_hit_rate_pct=round((cache_hits / max(1, len(ttft_list))) * 100.0, 2),
        prefix_matches=prefix_matches,
        prefix_match_rate_pct=round((prefix_matches / max(1, len(ttft_list))) * 100.0, 2),
        context_overflow=context_overflow,
        final_eval_status=final_eval_status,
        patch_diff=patch_diff,
        nudge_count=nudge_count,
        nudge_rate_pct=nudge_rate,
        hints_count=hints_count,
        per_turn_raw_replies=raw_replies_list,
        per_turn_tool_feedbacks=tool_feedbacks_list,
        traps_solved=traps_solved_cnt,
        solved_traps=solved_traps,
        regressions_introduced=regressions_introduced,
        regressions_repaired=regressions_repaired,
        turns_to_repair=turns_to_repair,
        final_pass_to_pass_intact=final_pass_to_pass_intact,
        infra_error=infra_error,
        infra_error_msg=infra_error_msg,
        wall_clock_s=wall_clock_s,
        per_turn_model_ms=model_ms_list,
        per_turn_tool_ms=tool_ms_list,
        per_turn_tool_args=tool_args_list,
        per_turn_condenser=condenser_list,
        per_turn_tool_chars_before=tool_chars_before_list,
        per_turn_tool_chars_after=tool_chars_after_list,
        failing_subsystems=failing_subsystems,
        test_files_edited=test_files_edited,
    )


def run_task_arm(
    task: OriginalBenchmarkTask,
    arm: str,
    base_url: str,
    api_key: str,
    model_name: str,
    max_turns: int = 6,
    seen_prefixes: set[str] | None = None,
    gateway_url: str = "",
    ablation: str = "both",
    upstream_kind: str = "nebius",
    include_sdcs_footer: bool = False,
    temperature: float = 0.0,
) -> TaskArmResult:
    """Runs an autonomous multi-turn agent session on a single `task` under `arm`."""
    if seen_prefixes is None:
        seen_prefixes = set()

    tmp_ws = Path(tempfile.mkdtemp(prefix=f"tr_{arm}_{task.task_id}_"))
    try:
        materialize_clean_repo(tmp_ws, arm="arm_b" if arm in ("runsift", "cacheforge", "arm_b") else "arm_a")
        inject_bug(tmp_ws, task)
        subprocess.run(["git", "init", "-q"], cwd=tmp_ws, check=False)
        subprocess.run(["git", "add", "-A"], cwd=tmp_ws, check=False)

        user_prompt = (
            f"Issue Report `{task.task_id}`: {task.title}\n\n"
            f"Symptom:\n{task.symptom_prompt}\n\n"
            f"Start by running `run_pytest` to inspect the failing unit test(s), read the relevant "
            f"module under `src/tensorroute/`, apply the minimal fix with `edit_file`, and verify "
            f"all 45 unit tests pass with `run_pytest`."
        )

        return _execute_agent_session(
            session_id=task.task_id,
            arm=arm,
            tmp_ws=tmp_ws,
            initial_user_prompt=user_prompt,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            max_turns=max_turns,
            seen_prefixes=seen_prefixes,
            gateway_url=gateway_url,
            ablation=ablation,
            upstream_kind=upstream_kind,
            include_sdcs_footer=include_sdcs_footer,
            temperature=temperature,
        )
    finally:
        shutil.rmtree(tmp_ws, ignore_errors=True)


def run_gauntlet_arm(
    arm: str,
    base_url: str,
    api_key: str,
    model_name: str,
    max_turns: int = 20,
    wave: int = 1,
    seen_prefixes: set[str] | None = None,
    gateway_url: str = "",
    ablation: str = "both",
    upstream_kind: str = "nebius",
    include_sdcs_footer: bool = False,
    temperature: float = 0.0,
    task_set: str = "original",
    sidecar_records: list[dict[str, Any]] | None = None,
) -> TaskArmResult:
    """Runs a continuous multi-bug Gauntlet session in a single workspace.

    If task_set == 'coupled', runs the 5-trap coupled gauntlet (56 tests, 5 traps)
    with read-only tests and regression bookkeeping.
    Otherwise, injects all 5 bugs from the selected wave (1 bug in each of radix.py, scheduler.py,
    limiter.py, codec.py, and telemetry.py) simultaneously.
    """
    if seen_prefixes is None:
        seen_prefixes = set()

    if task_set == "coupled":
        manifest_path = BENCH_DIR / "results" / "tasks_coupled_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                manifest = {}
        total_expected_tests = manifest.get("test_count", 56)
        total_expected_traps = manifest.get("trap_count", 5)

        session_id = "GAUNTLET-COUPLED"
        tmp_ws = Path(tempfile.mkdtemp(prefix=f"tr_{arm}_coupled_"))
        try:
            materialize_clean_repo_coupled(
                tmp_ws, arm="arm_b" if arm in ("runsift", "cacheforge", "arm_b") else "arm_a"
            )
            inject_all_coupled_bugs(tmp_ws)
            subprocess.run(["git", "init", "-q"], cwd=tmp_ws, check=False)
            subprocess.run(["git", "add", "-A"], cwd=tmp_ws, check=False)

            bullet_list = "\n".join(
                f"{i}. [{t.task_id}] {t.symptom_prompt}" for i, t in enumerate(COUPLED_TASKS, 1)
            )
            user_prompt = (
                f"Multi-Module Production Incident `{session_id}` (Continuous 5-Trap Coupled Gauntlet):\n"
                f"The codebase currently has active regressions causing unit test failures across the {total_expected_tests}-test suite:\n\n"
                f"{bullet_list}\n\n"
                f"Instructions:\n"
                f"1. Run `run_pytest` first to inspect the test failures.\n"
                f"2. Diagnose and repair broken modules under `src/tensorroute/` using `read_file` and `edit_file`.\n"
                f"   Note: tests are read-only (`tests/` cannot be edited).\n"
                f"3. Run `run_pytest` after your edits to track progress until ALL {total_expected_tests} unit tests pass (`{total_expected_tests} passed, 0 failed`)."
            )

            return _execute_agent_session(
                session_id=session_id,
                arm=arm,
                tmp_ws=tmp_ws,
                initial_user_prompt=user_prompt,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                max_turns=max_turns,
                seen_prefixes=seen_prefixes,
                gateway_url=gateway_url,
                ablation=ablation,
                upstream_kind=upstream_kind,
                include_sdcs_footer=include_sdcs_footer,
                temperature=temperature,
                expected_tests=total_expected_tests,
                trap_count=total_expected_traps,
                manifest=manifest,
                task_set=task_set,
                sidecar_records=sidecar_records,
            )
        finally:
            shutil.rmtree(tmp_ws, ignore_errors=True)

    start_idx = max(0, (wave - 1) * 5)
    gauntlet_tasks = BENCHMARK_TASKS[start_idx : start_idx + 5]
    session_id = f"GAUNTLET-WAVE-{wave}"

    tmp_ws = Path(tempfile.mkdtemp(prefix=f"tr_{arm}_gauntlet_w{wave}_"))
    try:
        materialize_clean_repo(tmp_ws, arm="arm_b" if arm in ("runsift", "cacheforge", "arm_b") else "arm_a")
        inject_gauntlet_bugs(tmp_ws, gauntlet_tasks)
        subprocess.run(["git", "init", "-q"], cwd=tmp_ws, check=False)
        subprocess.run(["git", "add", "-A"], cwd=tmp_ws, check=False)

        bullet_list = "\n".join(
            f"{i}. [{t.task_id}] {t.symptom_prompt}" for i, t in enumerate(gauntlet_tasks, 1)
        )
        user_prompt = (
            f"Multi-Module Production Incident `{session_id}` (Continuous 5-Bug Gauntlet):\n"
            f"All 5 core subsystems of `tensorroute` (`radix`, `scheduler`, `limiter`, `codec`, `telemetry`) "
            f"currently have active regressions causing unit test failures across the 45-test suite:\n\n"
            f"{bullet_list}\n\n"
            f"Instructions:\n"
            f"1. Run `run_pytest` first to inspect the test failures.\n"
            f"2. Diagnose and repair each broken module in `src/tensorroute/` one by one using `read_file` and `edit_file`.\n"
            f"3. Run `run_pytest` after your edits to track progress until ALL 45 unit tests pass (`45 passed, 0 failed`)."
        )

        return _execute_agent_session(
            session_id=session_id,
            arm=arm,
            tmp_ws=tmp_ws,
            initial_user_prompt=user_prompt,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            max_turns=max_turns,
            seen_prefixes=seen_prefixes,
            gateway_url=gateway_url,
            ablation=ablation,
            upstream_kind=upstream_kind,
            include_sdcs_footer=include_sdcs_footer,
            temperature=temperature,
            task_set=task_set,
            sidecar_records=sidecar_records,
        )
    finally:
        shutil.rmtree(tmp_ws, ignore_errors=True)


def wait_for_local_server(base_url: str, timeout_s: int = 3, upstream_kind: str = "nebius") -> bool:
    """If targeting a local llama-server endpoint, checks if `/health` reports `status == 'ok'`."""
    if upstream_kind != "llamacpp":
        return True
    if "127.0.0.1" not in base_url and "localhost" not in base_url:
        return True
    root_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    health_url = f"{root_url}/health"
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    if data.get("status") == "ok":
                        return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    return False


_LAUNCH_TIMESTAMP: str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _capture_launch_git_metadata() -> tuple[str, bool]:
    """Captures git commit and dirty status once at process start."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        commit = "unknown"
    try:
        raw_status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8")
        dirty_lines = []
        for line in raw_status.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(maxsplit=1)
            if len(parts) < 2:
                dirty_lines.append(line_str)
                continue
            fpath = parts[1].replace("\\", "/").strip('"')
            if " -> " in fpath:
                fpath = fpath.split(" -> ")[-1].strip('"')
            if "benchmarks/tensorroute_benchmark/results/" in fpath or fpath.startswith(
                ("benchmarks/results/", "benchmarks/latest_")
            ):
                continue
            dirty_lines.append(line_str)
        dirty = len(dirty_lines) > 0
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        dirty = False
    return commit, dirty


_LAUNCH_GIT_COMMIT, _LAUNCH_GIT_DIRTY = _capture_launch_git_metadata()


def capture_runtime_provenance(
    base_url: str,
    api_key: str = "",
    upstream_kind: str = "nebius",
    server_launch_command: str = "",
    gateway_url: str = "",
) -> dict[str, Any]:
    """Dynamically collects live environment provenance at benchmark execution time."""
    # 1. Git metadata captured once at process launch
    git_commit = _LAUNCH_GIT_COMMIT
    git_dirty = _LAUNCH_GIT_DIRTY

    # 1b. Gateway git commit
    gateway_commit = "not configured"
    target_gw = gateway_url or os.environ.get("RUNSIFT_GATEWAY_URL", os.environ.get("CACHEFORGE_GATEWAY_URL", ""))
    if target_gw:
        clean_gw = target_gw.rstrip("/")
        if clean_gw.endswith("/v1"):
            health_url = clean_gw[:-3] + "/health"
        else:
            health_url = f"{clean_gw}/health"
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    health_data = json.loads(resp.read().decode("utf-8"))
                    gateway_commit = (
                        health_data.get("git_commit") or health_data.get("commit") or "unknown"
                    )
                else:
                    gateway_commit = f"http_error_{resp.status}"
        except (
            urllib.error.URLError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            TimeoutError,
        ):
            gateway_commit = "unreachable"

    # 2. CLI arguments & timestamp
    cli_args = list(sys.argv)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 3. Models endpoint query
    models_resp: Any = "not exposed by provider"
    clean_base = base_url.rstrip("/")
    models_url = f"{clean_base}/models" if clean_base.endswith("/v1") else f"{clean_base}/v1/models"
    try:
        req = urllib.request.Request(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                models_resp = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError):
        models_resp = "not exposed by provider"

    # 4. Hardware and engine diagnostics
    if not server_launch_command:
        server_launch_command = os.environ.get("LLAMA_SERVER_LAUNCH_CMD", "")

    is_local = upstream_kind == "llamacpp"
    if is_local:
        try:
            smi_out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
            nvidia_smi = smi_out if smi_out else "not available"
        except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
            nvidia_smi = "not available"

        engine_version: Any = "local server"
        try:
            props_url = clean_base.replace("/v1", "") + "/props"
            with urllib.request.urlopen(props_url, timeout=3) as resp:
                if resp.status == 200:
                    engine_version = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError):
            engine_version = "not available"

        if not server_launch_command:
            server_launch_command = "local process"
    else:
        nvidia_smi = "not exposed by provider"
        engine_version = "not exposed by provider"
        if not server_launch_command:
            server_launch_command = "not exposed by provider"

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gateway_commit": gateway_commit,
        "launch_timestamp": _LAUNCH_TIMESTAMP,
        "cli_args": cli_args,
        "timestamp": ts,
        "base_url": base_url,
        "server_launch_command": server_launch_command,
        "engine_version": engine_version,
        "nvidia_smi": nvidia_smi,
        "models_endpoint_response": models_resp,
    }


def _build_scorecard_dict(
    mode_label: str,
    model_name: str,
    base_url: str,
    baseline_results: list[TaskArmResult],
    cacheforge_results: list[TaskArmResult],
    api_key: str = "",
    upstream_kind: str = "nebius",
    server_launch_command: str = "",
    gateway_url: str = "",
) -> dict[str, Any]:
    n_tasks = max(len(baseline_results), len(cacheforge_results), 1)
    base_all_ttfts = [x for r in baseline_results for x in r.per_turn_ttft_ms]
    cf_all_ttfts = [x for r in cacheforge_results for x in r.per_turn_ttft_ms]
    base_all_tokens = [x for r in baseline_results for x in r.per_turn_prompt_tokens]
    cf_all_tokens = [x for r in cacheforge_results for x in r.per_turn_prompt_tokens]
    base_all_tools = [x for r in baseline_results for x in getattr(r, "per_turn_tools", [])]
    cf_all_tools = [x for r in cacheforge_results for x in getattr(r, "per_turn_tools", [])]
    base_all_tests_passing = [
        x for r in baseline_results for x in getattr(r, "per_turn_tests_passing", [])
    ]
    cf_all_tests_passing = [
        x for r in cacheforge_results for x in getattr(r, "per_turn_tests_passing", [])
    ]

    base_total_turns = sum(r.turns for r in baseline_results)
    cf_total_turns = sum(r.turns for r in cacheforge_results)
    base_total_hits = sum(r.cache_hits for r in baseline_results)
    cf_total_hits = sum(r.cache_hits for r in cacheforge_results)
    base_total_prefix = sum(r.prefix_matches for r in baseline_results)
    cf_total_prefix = sum(r.prefix_matches for r in cacheforge_results)

    base_hit_rate = round((base_total_hits / max(1, len(base_all_ttfts))) * 100.0, 2)
    cf_hit_rate = round((cf_total_hits / max(1, len(cf_all_ttfts))) * 100.0, 2)
    base_prefix_rate = round((base_total_prefix / max(1, len(base_all_ttfts))) * 100.0, 2)
    cf_prefix_rate = round((cf_total_prefix / max(1, len(cf_all_ttfts))) * 100.0, 2)

    base_avg_ttft = round(sum(base_all_ttfts) / max(1, len(base_all_ttfts)), 2)
    cf_avg_ttft = round(sum(cf_all_ttfts) / max(1, len(cf_all_ttfts)), 2)
    base_p50_ttft = compute_percentile(base_all_ttfts, 50.0)
    cf_p50_ttft = compute_percentile(cf_all_ttfts, 50.0)
    base_p95_ttft = compute_percentile(base_all_ttfts, 95.0)
    cf_p95_ttft = compute_percentile(cf_all_ttfts, 95.0)

    base_total_prompt = sum(base_all_tokens)
    cf_total_prompt = sum(cf_all_tokens)
    base_peak_prompt = max(base_all_tokens) if base_all_tokens else 0
    cf_peak_prompt = max(cf_all_tokens) if cf_all_tokens else 0
    base_upstream_cached = sum(r.upstream_cached_tokens for r in baseline_results)
    cf_upstream_cached = sum(r.upstream_cached_tokens for r in cacheforge_results)
    base_uncached_prefill = base_total_prompt - base_upstream_cached
    cf_uncached_prefill = cf_total_prompt - cf_upstream_cached

    base_solved = sum(1 for r in baseline_results if r.solved)
    cf_solved = sum(1 for r in cacheforge_results if r.solved)

    base_nudges = sum(r.nudge_count for r in baseline_results)
    cf_nudges = sum(r.nudge_count for r in cacheforge_results)
    base_nudge_rate = round((base_nudges / max(1, base_total_turns)) * 100.0, 2)
    cf_nudge_rate = round((cf_nudges / max(1, cf_total_turns)) * 100.0, 2)

    base_hints = sum(getattr(r, "hints_count", 0) for r in baseline_results)
    cf_hints = sum(getattr(r, "hints_count", 0) for r in cacheforge_results)

    base_overflows = sum(1 for r in baseline_results if getattr(r, "context_overflow", False))
    cf_overflows = sum(1 for r in cacheforge_results if getattr(r, "context_overflow", False))

    if not baseline_results or not cacheforge_results:
        ttft_reduction_pct = 0.0
        token_savings_pct = 0.0
        uncached_prefill_savings_pct = 0.0
        peak_token_reduction_pct = 0.0
        tokens_saved_total = 0
        solve_rate_delta_pct = 0.0
        nudge_rate_delta_pct = 0.0
        hints_per_session_delta = 0.0
        cache_hit_rate_delta_pct = 0.0
        prefix_match_rate_delta_pct = 0.0
    else:
        ttft_reduction_pct = round(
            ((base_avg_ttft - cf_avg_ttft) / max(0.001, base_avg_ttft)) * 100.0, 2
        )
        token_savings_pct = round(
            ((base_total_prompt - cf_total_prompt) / max(1, base_total_prompt)) * 100.0, 2
        )
        uncached_prefill_savings_pct = round(
            ((base_uncached_prefill - cf_uncached_prefill) / max(1, max(1, base_uncached_prefill)))
            * 100.0,
            2,
        )
        peak_token_reduction_pct = round(
            ((base_peak_prompt - cf_peak_prompt) / max(1, base_peak_prompt)) * 100.0, 2
        )
        tokens_saved_total = base_total_prompt - cf_total_prompt
        solve_rate_delta_pct = round(((cf_solved - base_solved) / max(1, n_tasks)) * 100.0, 2)
        nudge_rate_delta_pct = round(cf_nudge_rate - base_nudge_rate, 2)
        hints_per_session_delta = round(
            (cf_hints / max(1, len(cacheforge_results)))
            - (base_hints / max(1, len(baseline_results))),
            2,
        )
        cache_hit_rate_delta_pct = round(cf_hit_rate - base_hit_rate, 2)
        prefix_match_rate_delta_pct = round(cf_prefix_rate - base_prefix_rate, 2)

    is_valid = True
    invalidation_reasons: list[str] = []
    for r in baseline_results:
        if getattr(r, "infra_error", False):
            is_valid = False
            msg = getattr(r, "infra_error_msg", "server unreachable / down")
            invalidation_reasons.append(
                f"Baseline infrastructure error (connection refused / server down): {msg[:100]}"
            )
    for r in cacheforge_results:
        if getattr(r, "infra_error", False):
            is_valid = False
            msg = getattr(r, "infra_error_msg", "server unreachable / down")
            invalidation_reasons.append(
                f"RunSift infrastructure error (connection refused / server down): {msg[:100]}"
            )
    if baseline_results and base_nudge_rate > 20.0:
        is_valid = False
        invalidation_reasons.append(
            f"Baseline nudge rate {base_nudge_rate}% exceeds 20.0% threshold (stuck agent)"
        )
    if cacheforge_results and cf_nudge_rate > 20.0:
        is_valid = False
        invalidation_reasons.append(
            f"RunSift nudge rate {cf_nudge_rate}% exceeds 20.0% threshold (stuck agent)"
        )

    return {
        "is_valid": is_valid,
        "invalidation_reason": "; ".join(invalidation_reasons) if invalidation_reasons else None,
        "launch_timestamp": _LAUNCH_TIMESTAMP,
        "provenance": capture_runtime_provenance(
            base_url=base_url,
            api_key=api_key,
            upstream_kind=upstream_kind,
            server_launch_command=server_launch_command,
            gateway_url=gateway_url,
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_suite": f"TensorRoute-Bench v2 ({mode_label})",
        "model": model_name,
        "upstream_base_url": base_url,
        "tasks_evaluated": n_tasks,
        "turns": cf_total_turns,
        "baseline": {
            "tasks_solved": f"{base_solved}/{n_tasks}",
            "solve_rate_pct": round((base_solved / max(1, n_tasks)) * 100.0, 2),
            "context_overflow_count": base_overflows,
            "nudge_count": base_nudges,
            "nudge_rate_pct": base_nudge_rate,
            "hints_count": base_hints,
            "hints_per_session": round(base_hints / max(1, len(baseline_results)), 2),
            "cache_hit_rate_pct": base_hit_rate,
            "prefix_match_rate_pct": base_prefix_rate,
            "upstream_cached_tokens": base_upstream_cached,
            "uncached_prefill_tokens": base_uncached_prefill,
            "avg_ttft_ms": base_avg_ttft,
            "p50_ttft_ms": base_p50_ttft,
            "p95_ttft_ms": base_p95_ttft,
            "total_prompt_tokens": base_total_prompt,
            "peak_turn_prompt_tokens": base_peak_prompt,
            "total_turns": base_total_turns,
            "per_turn_ttft_ms": base_all_ttfts,
            "per_turn_prompt_tokens": base_all_tokens,
            "per_turn_tools": base_all_tools,
            "per_turn_tests_passing": base_all_tests_passing,
            "task_results": [asdict(r) for r in baseline_results],
        },
        "runsift": {
            "tasks_solved": f"{cf_solved}/{n_tasks}",
            "solve_rate_pct": round((cf_solved / max(1, n_tasks)) * 100.0, 2),
            "context_overflow_count": cf_overflows,
            "nudge_count": cf_nudges,
            "nudge_rate_pct": cf_nudge_rate,
            "hints_count": cf_hints,
            "hints_per_session": round(cf_hints / max(1, len(cacheforge_results)), 2),
            "cache_hit_rate_pct": cf_hit_rate,
            "prefix_match_rate_pct": cf_prefix_rate,
            "upstream_cached_tokens": cf_upstream_cached,
            "uncached_prefill_tokens": cf_uncached_prefill,
            "avg_ttft_ms": cf_avg_ttft,
            "p50_ttft_ms": cf_p50_ttft,
            "p95_ttft_ms": cf_p95_ttft,
            "total_prompt_tokens": cf_total_prompt,
            "peak_turn_prompt_tokens": cf_peak_prompt,
            "total_turns": cf_total_turns,
            "per_turn_ttft_ms": cf_all_ttfts,
            "per_turn_prompt_tokens": cf_all_tokens,
            "per_turn_tools": cf_all_tools,
            "per_turn_tests_passing": cf_all_tests_passing,
            "task_results": [asdict(r) for r in cacheforge_results],
        },
        "cacheforge": {
            "tasks_solved": f"{cf_solved}/{n_tasks}",
            "solve_rate_pct": round((cf_solved / max(1, n_tasks)) * 100.0, 2),
            "context_overflow_count": cf_overflows,
            "nudge_count": cf_nudges,
            "nudge_rate_pct": cf_nudge_rate,
            "hints_count": cf_hints,
            "hints_per_session": round(cf_hints / max(1, len(cacheforge_results)), 2),
            "cache_hit_rate_pct": cf_hit_rate,
            "prefix_match_rate_pct": cf_prefix_rate,
            "upstream_cached_tokens": cf_upstream_cached,
            "uncached_prefill_tokens": cf_uncached_prefill,
            "avg_ttft_ms": cf_avg_ttft,
            "p50_ttft_ms": cf_p50_ttft,
            "p95_ttft_ms": cf_p95_ttft,
            "total_prompt_tokens": cf_total_prompt,
            "peak_turn_prompt_tokens": cf_peak_prompt,
            "total_turns": cf_total_turns,
            "per_turn_ttft_ms": cf_all_ttfts,
            "per_turn_prompt_tokens": cf_all_tokens,
            "per_turn_tools": cf_all_tools,
            "per_turn_tests_passing": cf_all_tests_passing,
            "task_results": [asdict(r) for r in cacheforge_results],
        },
        "improvements": {
            "solve_rate_delta_pct": solve_rate_delta_pct,
            "nudge_rate_delta_pct": nudge_rate_delta_pct,
            "hints_per_session_delta": hints_per_session_delta,
            "cache_hit_rate_delta_pct": cache_hit_rate_delta_pct,
            "prefix_match_rate_delta_pct": prefix_match_rate_delta_pct,
            "ttft_reduction_pct": ttft_reduction_pct,
            "token_savings_pct": token_savings_pct,
            "uncached_prefill_savings_pct": uncached_prefill_savings_pct,
            "peak_token_reduction_pct": peak_token_reduction_pct,
            "tokens_saved_total": tokens_saved_total,
        },
    }


def _compute_gauntlet_variance(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    if n == 0:
        return {}
    base_solves = [r["baseline"]["solve_rate_pct"] for r in runs]
    cf_solves = [r["cacheforge"]["solve_rate_pct"] for r in runs]
    base_ttfts = [r["baseline"]["avg_ttft_ms"] for r in runs]
    cf_ttfts = [r["cacheforge"]["avg_ttft_ms"] for r in runs]
    base_prompts = [r["baseline"]["total_prompt_tokens"] for r in runs]
    cf_prompts = [r["cacheforge"]["total_prompt_tokens"] for r in runs]
    base_uncached = [r["baseline"]["uncached_prefill_tokens"] for r in runs]
    cf_uncached = [r["cacheforge"]["uncached_prefill_tokens"] for r in runs]
    base_cached = [r["baseline"]["upstream_cached_tokens"] for r in runs]
    cf_cached = [r["cacheforge"]["upstream_cached_tokens"] for r in runs]
    base_nudges = [r["baseline"]["nudge_rate_pct"] for r in runs]
    cf_nudges = [r["cacheforge"]["nudge_rate_pct"] for r in runs]
    base_hints = [r["baseline"].get("hints_count", 0) for r in runs]
    cf_hints = [r["cacheforge"].get("hints_count", 0) for r in runs]
    base_overflows = [r["baseline"].get("context_overflow_count", 0) for r in runs]
    cf_overflows = [r["cacheforge"].get("context_overflow_count", 0) for r in runs]

    base_mods: list[int] = []
    cf_mods: list[int] = []
    for r in runs:
        b_tr = r.get("baseline", {}).get("task_results", [])
        if b_tr and isinstance(b_tr[0], dict) and "modules_fixed" in b_tr[0]:
            mf = b_tr[0]["modules_fixed"]
            if isinstance(mf, str) and "/" in mf:
                try:
                    base_mods.append(int(mf.split("/")[0]))
                except ValueError:
                    base_mods.append(0)
            elif isinstance(mf, (int, float)):
                base_mods.append(int(mf))
            else:
                base_mods.append(0)
        else:
            base_mods.append(0)

        c_tr = r.get("cacheforge", {}).get("task_results", [])
        if c_tr and isinstance(c_tr[0], dict) and "modules_fixed" in c_tr[0]:
            mf = c_tr[0]["modules_fixed"]
            if isinstance(mf, str) and "/" in mf:
                try:
                    cf_mods.append(int(mf.split("/")[0]))
                except ValueError:
                    cf_mods.append(0)
            elif isinstance(mf, (int, float)):
                cf_mods.append(int(mf))
            else:
                cf_mods.append(0)
        else:
            cf_mods.append(0)

    return {
        "runs_count": n,
        "baseline_solve_rate_mean": round(statistics.mean(base_solves), 2),
        "baseline_solve_rate_stdev": round(statistics.stdev(base_solves), 2) if n > 1 else 0.0,
        "cacheforge_solve_rate_mean": round(statistics.mean(cf_solves), 2),
        "cacheforge_solve_rate_stdev": round(statistics.stdev(cf_solves), 2) if n > 1 else 0.0,
        "baseline_modules_fixed_counts": base_mods,
        "baseline_modules_fixed_sum": sum(base_mods),
        "baseline_modules_fixed_total_possible": 5 * n,
        "cacheforge_modules_fixed_counts": cf_mods,
        "cacheforge_modules_fixed_sum": sum(cf_mods),
        "cacheforge_modules_fixed_total_possible": 5 * n,
        "baseline_ttft_mean": round(statistics.mean(base_ttfts), 2),
        "baseline_ttft_stdev": round(statistics.stdev(base_ttfts), 2) if n > 1 else 0.0,
        "cacheforge_ttft_mean": round(statistics.mean(cf_ttfts), 2),
        "cacheforge_ttft_stdev": round(statistics.stdev(cf_ttfts), 2) if n > 1 else 0.0,
        "baseline_prompt_tokens_mean": round(statistics.mean(base_prompts), 2),
        "baseline_prompt_tokens_stdev": round(statistics.stdev(base_prompts), 2) if n > 1 else 0.0,
        "cacheforge_prompt_tokens_mean": round(statistics.mean(cf_prompts), 2),
        "cacheforge_prompt_tokens_stdev": round(statistics.stdev(cf_prompts), 2) if n > 1 else 0.0,
        "baseline_uncached_tokens_mean": round(statistics.mean(base_uncached), 2),
        "baseline_uncached_tokens_stdev": round(statistics.stdev(base_uncached), 2)
        if n > 1
        else 0.0,
        "cacheforge_uncached_tokens_mean": round(statistics.mean(cf_uncached), 2),
        "cacheforge_uncached_tokens_stdev": round(statistics.stdev(cf_uncached), 2)
        if n > 1
        else 0.0,
        "baseline_cached_tokens_mean": round(statistics.mean(base_cached), 2),
        "baseline_cached_tokens_stdev": round(statistics.stdev(base_cached), 2) if n > 1 else 0.0,
        "cacheforge_cached_tokens_mean": round(statistics.mean(cf_cached), 2),
        "cacheforge_cached_tokens_stdev": round(statistics.stdev(cf_cached), 2) if n > 1 else 0.0,
        "baseline_nudge_rate_mean": round(statistics.mean(base_nudges), 2),
        "baseline_nudge_rate_stdev": round(statistics.stdev(base_nudges), 2) if n > 1 else 0.0,
        "cacheforge_nudge_rate_mean": round(statistics.mean(cf_nudges), 2),
        "cacheforge_nudge_rate_stdev": round(statistics.stdev(cf_nudges), 2) if n > 1 else 0.0,
        "baseline_hints_mean": round(statistics.mean(base_hints), 2),
        "baseline_hints_stdev": round(statistics.stdev(base_hints), 2) if n > 1 else 0.0,
        "cacheforge_hints_mean": round(statistics.mean(cf_hints), 2),
        "cacheforge_hints_stdev": round(statistics.stdev(cf_hints), 2) if n > 1 else 0.0,
        "baseline_overflows_mean": round(statistics.mean(base_overflows), 2),
        "baseline_overflows_stdev": round(statistics.stdev(base_overflows), 2) if n > 1 else 0.0,
        "baseline_overflows_count": sum(base_overflows),
        "cacheforge_overflows_mean": round(statistics.mean(cf_overflows), 2),
        "cacheforge_overflows_stdev": round(statistics.stdev(cf_overflows), 2) if n > 1 else 0.0,
        "cacheforge_overflows_count": sum(cf_overflows),
    }


def _print_gauntlet_variance(v: dict[str, Any], title: str) -> None:
    print("\n====================================================================")
    print(f" VARIANCE ANALYSIS ({title})")
    print("====================================================================")
    print(
        f" Baseline Solve Rate:   {v['baseline_solve_rate_mean']:.1f}% ± {v['baseline_solve_rate_stdev']:.2f}%"
    )
    print(
        f" RunSift Solve Rate:    {v.get('runsift_solve_rate_mean', v.get('cacheforge_solve_rate_mean')):.1f}% ± {v.get('runsift_solve_rate_stdev', v.get('cacheforge_solve_rate_stdev')):.2f}%"
    )
    print(
        f" Baseline Modules Fixed: {v['baseline_modules_fixed_sum']}/{v['baseline_modules_fixed_total_possible']} "
        f"(counts per repeat: {v['baseline_modules_fixed_counts']})"
    )
    print(
        f" RunSift Modules Fixed: {v.get('runsift_modules_fixed_sum', v.get('cacheforge_modules_fixed_sum'))}/{v.get('runsift_modules_fixed_total_possible', v.get('cacheforge_modules_fixed_total_possible'))} "
        f"(counts per repeat: {v.get('runsift_modules_fixed_counts', v.get('cacheforge_modules_fixed_counts'))})"
    )
    print(
        f" Baseline Wire TTFT:    {v['baseline_ttft_mean']:.1f}ms ± {v['baseline_ttft_stdev']:.2f}ms"
    )
    print(
        f" RunSift Wire TTFT:     {v.get('runsift_ttft_mean', v.get('cacheforge_ttft_mean')):.1f}ms ± {v.get('runsift_ttft_stdev', v.get('cacheforge_ttft_stdev')):.2f}ms"
    )
    print(
        f" Baseline Prompt Tok:   {v['baseline_prompt_tokens_mean']:.1f} ± {v['baseline_prompt_tokens_stdev']:.1f}"
    )
    print(
        f" RunSift Prompt Tok:    {v.get('runsift_prompt_tokens_mean', v.get('cacheforge_prompt_tokens_mean')):.1f} ± {v.get('runsift_prompt_tokens_stdev', v.get('cacheforge_prompt_tokens_stdev')):.1f}"
    )
    print(
        f" Baseline Nudge Rate:   {v['baseline_nudge_rate_mean']:.1f}% ± {v['baseline_nudge_rate_stdev']:.2f}%"
    )
    print(
        f" RunSift Nudge Rate:    {v.get('runsift_nudge_rate_mean', v.get('cacheforge_nudge_rate_mean')):.1f}% ± {v.get('runsift_nudge_rate_stdev', v.get('cacheforge_nudge_rate_stdev')):.2f}%"
    )
    print(
        f" Baseline Loop Hints:   {v['baseline_hints_mean']:.1f} ± {v['baseline_hints_stdev']:.2f}"
    )
    print(
        f" RunSift Loop Hints:    {v.get('runsift_hints_mean', v.get('cacheforge_hints_mean')):.1f} ± {v.get('runsift_hints_stdev', v.get('cacheforge_hints_stdev')):.2f}"
    )
    print(
        f" Baseline Overflows:    {v['baseline_overflows_count']}/{v['runs_count']} "
        f"({v['baseline_overflows_mean']:.1f} ± {v['baseline_overflows_stdev']:.2f})"
    )
    print(
        f" RunSift Overflows:     {v.get('runsift_overflows_count', v.get('cacheforge_overflows_count'))}/{v['runs_count']} "
        f"({v.get('runsift_overflows_mean', v.get('cacheforge_overflows_mean')):.1f} ± {v.get('runsift_overflows_stdev', v.get('cacheforge_overflows_stdev')):.2f})"
    )
    print("====================================================================\n")


def _compute_suite_variance(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    if n == 0:
        return {}
    base_solves = [r["baseline"]["solve_rate_pct"] for r in runs]
    cf_solves = [r["cacheforge"]["solve_rate_pct"] for r in runs]
    base_ttfts = [r["baseline"]["avg_ttft_ms"] for r in runs]
    cf_ttfts = [r["cacheforge"]["avg_ttft_ms"] for r in runs]
    base_prompts = [r["baseline"]["total_prompt_tokens"] for r in runs]
    cf_prompts = [r["cacheforge"]["total_prompt_tokens"] for r in runs]
    base_uncached = [r["baseline"]["uncached_prefill_tokens"] for r in runs]
    cf_uncached = [r["cacheforge"]["uncached_prefill_tokens"] for r in runs]
    base_cached = [r["baseline"]["upstream_cached_tokens"] for r in runs]
    cf_cached = [r["cacheforge"]["upstream_cached_tokens"] for r in runs]
    base_nudges = [r["baseline"]["nudge_rate_pct"] for r in runs]
    cf_nudges = [r["cacheforge"]["nudge_rate_pct"] for r in runs]
    base_hints = [r["baseline"].get("hints_count", 0) for r in runs]
    cf_hints = [r["cacheforge"].get("hints_count", 0) for r in runs]

    return {
        "runs_count": n,
        "baseline_solve_rate_mean": round(statistics.mean(base_solves), 2),
        "baseline_solve_rate_stdev": round(statistics.stdev(base_solves), 2) if n > 1 else 0.0,
        "cacheforge_solve_rate_mean": round(statistics.mean(cf_solves), 2),
        "cacheforge_solve_rate_stdev": round(statistics.stdev(cf_solves), 2) if n > 1 else 0.0,
        "baseline_ttft_mean": round(statistics.mean(base_ttfts), 2),
        "baseline_ttft_stdev": round(statistics.stdev(base_ttfts), 2) if n > 1 else 0.0,
        "cacheforge_ttft_mean": round(statistics.mean(cf_ttfts), 2),
        "cacheforge_ttft_stdev": round(statistics.stdev(cf_ttfts), 2) if n > 1 else 0.0,
        "baseline_prompt_tokens_mean": round(statistics.mean(base_prompts), 2),
        "baseline_prompt_tokens_stdev": round(statistics.stdev(base_prompts), 2) if n > 1 else 0.0,
        "cacheforge_prompt_tokens_mean": round(statistics.mean(cf_prompts), 2),
        "cacheforge_prompt_tokens_stdev": round(statistics.stdev(cf_prompts), 2) if n > 1 else 0.0,
        "baseline_uncached_tokens_mean": round(statistics.mean(base_uncached), 2),
        "baseline_uncached_tokens_stdev": round(statistics.stdev(base_uncached), 2)
        if n > 1
        else 0.0,
        "cacheforge_uncached_tokens_mean": round(statistics.mean(cf_uncached), 2),
        "cacheforge_uncached_tokens_stdev": round(statistics.stdev(cf_uncached), 2)
        if n > 1
        else 0.0,
        "baseline_cached_tokens_mean": round(statistics.mean(base_cached), 2),
        "baseline_cached_tokens_stdev": round(statistics.stdev(base_cached), 2) if n > 1 else 0.0,
        "cacheforge_cached_tokens_mean": round(statistics.mean(cf_cached), 2),
        "cacheforge_cached_tokens_stdev": round(statistics.stdev(cf_cached), 2) if n > 1 else 0.0,
        "baseline_nudge_rate_mean": round(statistics.mean(base_nudges), 2),
        "baseline_nudge_rate_stdev": round(statistics.stdev(base_nudges), 2) if n > 1 else 0.0,
        "cacheforge_nudge_rate_mean": round(statistics.mean(cf_nudges), 2),
        "cacheforge_nudge_rate_stdev": round(statistics.stdev(cf_nudges), 2) if n > 1 else 0.0,
        "baseline_hints_mean": round(statistics.mean(base_hints), 2),
        "baseline_hints_stdev": round(statistics.stdev(base_hints), 2) if n > 1 else 0.0,
        "cacheforge_hints_mean": round(statistics.mean(cf_hints), 2),
        "cacheforge_hints_stdev": round(statistics.stdev(cf_hints), 2) if n > 1 else 0.0,
    }


def _print_suite_variance(v: dict[str, Any], title: str) -> None:
    print("\n====================================================================")
    print(f" VARIANCE ANALYSIS ({title})")
    print("====================================================================")
    print(
        f" Baseline Solve Rate:   {v['baseline_solve_rate_mean']:.1f}% ± {v['baseline_solve_rate_stdev']:.2f}%"
    )
    print(
        f" RunSift Solve Rate:    {v.get('runsift_solve_rate_mean', v.get('cacheforge_solve_rate_mean')):.1f}% ± {v.get('runsift_solve_rate_stdev', v.get('cacheforge_solve_rate_stdev')):.2f}%"
    )
    print(
        f" Baseline Avg TTFT:     {v['baseline_ttft_mean']:.1f}ms ± {v['baseline_ttft_stdev']:.2f}ms"
    )
    print(
        f" RunSift Avg TTFT:      {v.get('runsift_ttft_mean', v.get('cacheforge_ttft_mean')):.1f}ms ± {v.get('runsift_ttft_stdev', v.get('cacheforge_ttft_stdev')):.2f}ms"
    )
    print(
        f" Baseline Prompt Tok:   {v['baseline_prompt_tokens_mean']:.1f} ± {v['baseline_prompt_tokens_stdev']:.1f}"
    )
    print(
        f" RunSift Prompt Tok:    {v.get('runsift_prompt_tokens_mean', v.get('cacheforge_prompt_tokens_mean')):.1f} ± {v.get('runsift_prompt_tokens_stdev', v.get('cacheforge_prompt_tokens_stdev')):.1f}"
    )
    print(
        f" Baseline Nudge Rate:   {v['baseline_nudge_rate_mean']:.1f}% ± {v['baseline_nudge_rate_stdev']:.2f}%"
    )
    print(
        f" RunSift Nudge Rate:    {v.get('runsift_nudge_rate_mean', v.get('cacheforge_nudge_rate_mean')):.1f}% ± {v.get('runsift_nudge_rate_stdev', v.get('cacheforge_nudge_rate_stdev')):.2f}%"
    )
    print(
        f" Baseline Loop Hints:   {v['baseline_hints_mean']:.1f} ± {v['baseline_hints_stdev']:.2f}"
    )
    print(
        f" RunSift Loop Hints:    {v.get('runsift_hints_mean', v.get('cacheforge_hints_mean')):.1f} ± {v.get('runsift_hints_stdev', v.get('cacheforge_hints_stdev')):.2f}"
    )
    print("====================================================================\n")


def run_gauntlet_comparison(
    base_url: str,
    api_key: str,
    model_name: str,
    max_turns: int = 20,
    wave: int = 1,
    gateway_url: str = "",
    ablation: str = "both",
    repeats: int = 1,
    temperature: float = 0.0,
    upstream_kind: str = "nebius",
    server_launch_command: str = "",
    include_sdcs_footer: bool = False,
    repeat_offset: int = 0,
    result_name: str = "",
    arm: str = "both",
    task_set: str = "original",
) -> dict[str, Any]:
    """Runs the continuous 5-bug Gauntlet comparison (`baseline` vs `runsift`)."""
    print("====================================================================")
    if task_set == "coupled":
        print(" TensorRoute-Bench v2 :: Continuous 5-Trap COUPLED GAUNTLET Mode")
    else:
        print(" TensorRoute-Bench v2 :: Continuous 5-Bug GAUNTLET Mode")
    print(f" Endpoint:      {base_url}")
    if gateway_url:
        print(f" Gateway URL:   {gateway_url} (True Proxy Routing for RunSift Arm)")
    print(f" Upstream Kind: {upstream_kind}")
    if task_set == "coupled":
        print(
            f" Model:         {model_name} | Task Set: coupled (5 Traps) | Max Turns: {max_turns}"
        )
    else:
        print(
            f" Model:         {model_name} | Wave: {wave} (5 Simultaneous Bugs) | Max Turns: {max_turns}"
        )
    total_planned = repeat_offset + repeats
    print(
        f" Ablation:      {ablation} | Repeats: {repeats} (Total: {total_planned}) | Temp: {temperature} | Arm: {arm}"
    )
    print("====================================================================\n")

    server_ready = wait_for_local_server(base_url, timeout_s=3, upstream_kind=upstream_kind)
    if not server_ready:
        print(f"[Offline Notice] Endpoint {base_url} is not responding.")
        print(
            f"Validated gauntlet configuration: ablation='{ablation}', repeats={repeats}, wave={wave}."
        )
        print(
            "To run live benchmark execution, launch llama-server or provide --base-url / --nebius."
        )
        return {}

    preflight_workspace_check(task_set=task_set)

    results_dir = BENCH_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    is_local = upstream_kind == "llamacpp" or "127.0.0.1" in base_url or "localhost" in base_url
    if "gemma" in model_name.lower():
        model_slug = "gemma4_26b"
    elif "30b" in model_name.lower():
        model_slug = "nemotron_30b"
    else:
        model_slug = "qwen25_7b"

    if result_name:
        out_name = result_name if result_name.endswith(".json") else f"{result_name}.json"
        out_file = results_dir / out_name
    elif task_set == "coupled":
        turn_suffix = "_40turn" if max_turns == 40 else ""
        prefix = "local_coupled" if is_local else "nebius_coupled"
        out_file = results_dir / f"{prefix}_{model_slug}{turn_suffix}.json"
    elif is_local:
        turn_suffix = "_40turn" if max_turns == 40 else ""
        if ablation in ("none", "canonicalize-only", "compact-only"):
            ablation_slug = ablation.replace("+", "_").replace("-", "_")
            out_file = (
                results_dir / f"local_gauntlet_{model_slug}_{ablation_slug}{turn_suffix}.json"
            )
        elif ablation == "both":
            out_file = results_dir / f"local_gauntlet_{model_slug}{turn_suffix}.json"
        else:
            out_file = results_dir / f"local_gauntlet_{model_slug}_{ablation}{turn_suffix}.json"
    elif ablation == "canonicalize-only":
        out_file = results_dir / "latest_gauntlet_canonicalize_only.json"
    elif ablation == "compact-only":
        out_file = results_dir / "latest_gauntlet_compact_only.json"
    elif ablation == "none":
        out_file = results_dir / "latest_gauntlet_none.json"
    elif "30b" in model_name.lower():
        out_file = results_dir / "nebius_gauntlet_nemotron_30b.json"
    else:
        out_file = results_dir / "latest_gauntlet_benchmark.json"

    all_runs: list[dict[str, Any]] = []
    if repeat_offset > 0 and out_file.exists():
        try:
            prev_data = json.loads(out_file.read_text(encoding="utf-8"))
            if "raw_repeats" in prev_data and isinstance(prev_data["raw_repeats"], list):
                all_runs = list(prev_data["raw_repeats"][:repeat_offset])
                print(f"[Repeat Offset] Loaded {len(all_runs)} prior repeats from {out_file.name}")
        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"[Warning] Failed to load prior repeats: {e}")
            all_runs = []

    last_res_base: TaskArmResult | None = None
    last_res_cf: TaskArmResult | None = None

    for i in range(1, repeats + 1):
        rep = repeat_offset + i
        total_reps = repeat_offset + repeats
        if total_reps > 1:
            print(f"\n>>> Running Gauntlet Repetition {rep}/{total_reps} <<<\n")
        baseline_prefixes: set[str] = set()
        cf_prefixes: set[str] = set()

        if arm in ("runsift", "cacheforge"):
            arm_order = [("runsift", cf_prefixes)]
        elif arm == "baseline":
            arm_order = [("baseline", baseline_prefixes)]
        elif rep % 2 == 1:
            arm_order = [("baseline", baseline_prefixes), ("runsift", cf_prefixes)]
        else:
            arm_order = [("runsift", cf_prefixes), ("baseline", baseline_prefixes)]

        rep_results: dict[str, TaskArmResult] = {}
        sidecar_records: list[dict[str, Any]] = []
        for phase_idx, (arm_name, prefix_set) in enumerate(arm_order, 1):
            phase_label = arm_name.upper()
            if arm_name in ("runsift", "cacheforge"):
                phase_label = f"RUNSIFT Arm ({ablation})"
            print(f"--- Phase {phase_idx}: Running {phase_label} (Continuous 5-Bug Gauntlet) ---")
            _reset_persistent_conn(base_url)
            if gateway_url:
                _reset_persistent_conn(gateway_url)

            try:
                res = run_gauntlet_arm(
                    arm=arm_name,
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    max_turns=max_turns,
                    wave=wave,
                    seen_prefixes=prefix_set,
                    gateway_url=gateway_url,
                    ablation=ablation,
                    upstream_kind=upstream_kind,
                    include_sdcs_footer=include_sdcs_footer,
                    temperature=temperature,
                    task_set=task_set,
                    sidecar_records=sidecar_records,
                )
            except AuthQuotaAbortError as exc:
                print(f"\n[FATAL AUTH/QUOTA ERROR] {exc}. Aborting benchmark run immediately.")
                if all_runs:
                    summary = copy.deepcopy(all_runs[-1])
                    summary["raw_repeats"] = all_runs
                    summary["repeats"] = len(all_runs)
                    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    print(
                        f"[Results Preserved] Saved {len(all_runs)} completed repeats to {out_file.name}"
                    )
                raise
            rep_results[arm_name] = res
            if task_set == "coupled":
                print(
                    f"  -> [{arm_name.upper()} GAUNTLET] Solved: {res.solved} | Traps Solved: {res.traps_solved}/5 "
                    f"({res.passed_tests}/56 tests) | Regressions: {len(res.regressions_introduced)} intro, "
                    f"{res.regressions_repaired} rep | Turns: {res.turns} | "
                    f"PromptTok: {res.total_prompt_tokens} (Peak: {res.peak_turn_prompt_tokens}) | "
                    f"AvgTTFT: {res.avg_ttft_ms}ms | ServerHit: {res.cache_hit_rate_pct}% | "
                    f"PrefixMatch: {res.prefix_match_rate_pct}% | Hints: {res.hints_count}\n"
                )
            else:
                print(
                    f"  -> [{arm_name.upper()} GAUNTLET] Solved: {res.solved} | Modules Fixed: {res.modules_fixed} "
                    f"({res.passed_tests}/45 tests) | Turns: {res.turns} | "
                    f"PromptTok: {res.total_prompt_tokens} (Peak: {res.peak_turn_prompt_tokens}) | "
                    f"AvgTTFT: {res.avg_ttft_ms}ms | ServerHit: {res.cache_hit_rate_pct}% | "
                    f"PrefixMatch: {res.prefix_match_rate_pct}% | Hints: {res.hints_count}\n"
                )

        res_base = rep_results.get("baseline")
        res_cf = rep_results.get("runsift", rep_results.get("cacheforge"))
        last_res_base = res_base
        last_res_cf = last_res_cf or res_cf
        _reset_persistent_conn(base_url)
        if gateway_url:
            _reset_persistent_conn(gateway_url)

        mode_label = (
            f"Continuous 5-Trap Coupled Gauntlet ({ablation})"
            if task_set == "coupled"
            else f"Continuous 5-Bug Multi-Module Gauntlet - Wave {wave} ({ablation})"
        )
        summary = _build_scorecard_dict(
            mode_label=mode_label,
            model_name=model_name,
            base_url=base_url,
            baseline_results=[res_base] if res_base else [],
            cacheforge_results=[res_cf] if res_cf else [],
            api_key=api_key,
            upstream_kind=upstream_kind,
            server_launch_command=server_launch_command,
            gateway_url=gateway_url,
        )
        summary["ablation"] = ablation
        summary["rep"] = rep
        all_runs.append(summary)
        rep_file = results_dir / f"{out_file.stem}_rep{rep}.json"
        rep_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        sidecar_file = results_dir / f"{out_file.stem}_rep{rep}_tool_outputs.jsonl.gz"
        with gzip.open(sidecar_file, "wt", encoding="utf-8") as gz_fp:
            for s_rec in sidecar_records:
                gz_fp.write(json.dumps(s_rec, ensure_ascii=False) + "\n")

    summary = copy.deepcopy(all_runs[-1])
    summary["raw_repeats"] = all_runs
    total_completed = len(all_runs)
    summary["repeats"] = total_completed
    if total_completed > 1:
        invalid_reps = [
            f"Rep {r.get('rep', idx + 1)}: {r.get('invalidation_reason')}"
            for idx, r in enumerate(all_runs)
            if not r.get("is_valid", True)
        ]
        if invalid_reps:
            summary["is_valid"] = False
            summary["invalidation_reason"] = "; ".join(invalid_reps)
        else:
            summary["is_valid"] = True
            summary["invalidation_reason"] = None

        summary["variance_analysis"] = _compute_gauntlet_variance(all_runs)
        _print_gauntlet_variance(
            summary["variance_analysis"], f"{total_completed} Repeats (All), Ablation: {ablation}"
        )

        valid_runs = [r for r in all_runs if r.get("is_valid", True)]
        if valid_runs and len(valid_runs) < len(all_runs):
            summary["variance_analysis_valid_only"] = _compute_gauntlet_variance(valid_runs)
            _print_gauntlet_variance(
                summary["variance_analysis_valid_only"],
                f"Valid Repeats Only: {len(valid_runs)}/{total_completed}, Ablation: {ablation}",
            )
        else:
            summary["variance_analysis_valid_only"] = None

    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_main_file = (
        results_dir / "latest_coupled_benchmark.json"
        if task_set == "coupled"
        else results_dir / "latest_gauntlet_benchmark.json"
    )
    latest_main_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if ("nebius" in base_url.lower() or upstream_kind == "nebius") and not is_local:
        if "30b" in model_name.lower():
            nebius_g_file = results_dir / "nebius_gauntlet_nemotron_30b.json"
            nebius_g_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        elif ablation == "both":
            nebius_g_file = results_dir / "nebius_gauntlet_nemotron_120b.json"
            nebius_g_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            nebius_main_file = results_dir / "nebius_gauntlet_benchmark.json"
            nebius_main_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validity_str = (
        "VALID"
        if summary.get("is_valid", True)
        else f"INVALID ({summary.get('invalidation_reason')})"
    )
    b_mod = last_res_base.modules_fixed if last_res_base else "N/A"
    b_pass = last_res_base.passed_tests if last_res_base else 0
    cf_mod = last_res_cf.modules_fixed if last_res_cf else "N/A"
    cf_pass = last_res_cf.passed_tests if last_res_cf else 0
    b_peak = last_res_base.peak_turn_prompt_tokens if last_res_base else 0
    cf_peak = last_res_cf.peak_turn_prompt_tokens if last_res_cf else 0

    print("====================================================================")
    print(" EMPIRICAL SCORECARD :: TensorRoute-Bench GAUNTLET (5 Bugs / 1 Session)")
    print("====================================================================")
    cf_summary = summary.get("runsift", summary.get("cacheforge", {}))
    print(f" Run Validity:          {validity_str}")
    print(
        f" Solve Rate (5/5):      {summary['baseline']['tasks_solved']} ({summary['baseline']['solve_rate_pct']}%) -->  "
        f"{cf_summary['tasks_solved']} ({cf_summary['solve_rate_pct']}%) "
        f"({summary['improvements']['solve_rate_delta_pct']:+.1f}%)"
    )
    mod_label = "Traps Repaired:     " if task_set == "coupled" else "Modules Repaired:   "
    total_expected = 56 if task_set == "coupled" else 45
    print(
        f" {mod_label}   {b_mod} ({b_pass}/{total_expected}) -->  {cf_mod} ({cf_pass}/{total_expected})"
    )
    print(
        f" Context Overflows:     {summary['baseline'].get('context_overflow_count', 0)}/{len(summary['baseline']['task_results'])}           -->  "
        f"{cf_summary.get('context_overflow_count', 0)}/{len(cf_summary['task_results'])}"
    )
    print(
        f" Loop Hints Injected:   {summary['baseline'].get('hints_count', 0)} ({summary['baseline'].get('hints_per_session', 0.0)}/sess) -->  "
        f"{cf_summary.get('hints_count', 0)} ({cf_summary.get('hints_per_session', 0.0)}/sess) "
        f"({summary['improvements'].get('hints_per_session_delta', 0.0):+.2f}/sess)"
    )
    print(
        f" Nudge Rate (Stuck):    {summary['baseline']['nudge_rate_pct']:5.1f}%       -->  {cf_summary['nudge_rate_pct']:5.1f}% ({summary['improvements']['nudge_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Server KV Cache Hit:   {summary['baseline']['cache_hit_rate_pct']:5.1f}%       -->  {cf_summary['cache_hit_rate_pct']:5.1f}% ({summary['improvements']['cache_hit_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Prefix Match Rate:     {summary['baseline']['prefix_match_rate_pct']:5.1f}%       -->  {cf_summary['prefix_match_rate_pct']:5.1f}% ({summary['improvements']['prefix_match_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Mean / p50 Wire TTFT:  {summary['baseline']['avg_ttft_ms']:5.1f}ms ({summary['baseline']['p50_ttft_ms']:5.1f}ms) -->  {cf_summary['avg_ttft_ms']:5.1f}ms ({cf_summary['p50_ttft_ms']:5.1f}ms) ({summary['improvements']['ttft_reduction_pct']:+.1f}%)"
    )
    print(
        f" Peak Turn Context Tok: {b_peak:6d}       -->  {cf_peak:6d} ({summary['improvements']['peak_token_reduction_pct']:+.1f}%)"
    )
    print(
        f" Uncached Prefill Tok:  {summary['baseline']['uncached_prefill_tokens']:6d}       -->  {cf_summary['uncached_prefill_tokens']:6d} ({summary['improvements']['uncached_prefill_savings_pct']:+.1f}%)"
    )
    print(
        f" Cumulative Prompt Tok: {summary['baseline']['total_prompt_tokens']:6d}       -->  {cf_summary['total_prompt_tokens']:6d} ({summary['improvements']['token_savings_pct']:+.1f}%, saved {summary['improvements']['tokens_saved_total']:+d} tokens)"
    )
    print("====================================================================")
    print(f" Saved Gauntlet artifact to: {out_file}")
    return summary


def run_comparative_suite(
    base_url: str,
    api_key: str,
    model_name: str,
    task_filter: str = "",
    wave: int = 0,
    max_turns: int = 6,
    gateway_url: str = "",
    ablation: str = "both",
    repeats: int = 1,
    temperature: float = 0.0,
    upstream_kind: str = "nebius",
    server_launch_command: str = "",
    include_sdcs_footer: bool = False,
) -> dict[str, Any]:
    tasks = list(BENCHMARK_TASKS)
    if wave in (1, 2, 3):
        start_idx = (wave - 1) * 5
        tasks = tasks[start_idx : start_idx + 5]
    if task_filter:
        tasks = [t for t in tasks if task_filter in t.task_id]

    print("====================================================================")
    print(" TensorRoute-Bench v2 :: Live Autonomous Agent Comparative Suite")
    print(f" Endpoint:      {base_url}")
    if gateway_url:
        print(f" Gateway URL:   {gateway_url} (True Proxy Routing for RunSift Arm)")
    print(f" Upstream Kind: {upstream_kind}")
    print(f" Model:         {model_name} | Tasks: {len(tasks)} | Max Turns/Task: {max_turns}")
    print(f" Ablation:      {ablation} | Repeats: {repeats} | Temp: {temperature}")
    print("====================================================================\n")

    server_ready = wait_for_local_server(base_url, timeout_s=3, upstream_kind=upstream_kind)
    if not server_ready:
        print(f"[Offline Notice] Endpoint {base_url} is not responding.")
        print(
            f"Validated harness configuration: ablation='{ablation}', repeats={repeats}, tasks={len(tasks)}."
        )
        print(
            "To run live benchmark execution, launch llama-server or provide --base-url / --nebius."
        )
        return {}

    preflight_workspace_check()

    all_runs: list[dict[str, Any]] = []
    for rep in range(1, repeats + 1):
        if repeats > 1:
            print(f"\n>>> Running Repetition {rep}/{repeats} <<<\n")

        baseline_prefixes: set[str] = set()
        cf_prefixes: set[str] = set()

        if rep % 2 == 1:
            arm_order = [("baseline", baseline_prefixes), ("cacheforge", cf_prefixes)]
        else:
            arm_order = [("cacheforge", cf_prefixes), ("baseline", baseline_prefixes)]

        rep_task_results: dict[str, list[TaskArmResult]] = {"baseline": [], "cacheforge": []}
        for phase_idx, (arm_name, prefix_set) in enumerate(arm_order, 1):
            phase_label = arm_name.upper()
            if arm_name == "cacheforge":
                phase_label = f"CACHEFORGE Arm ({ablation})"
            print(f"--- Phase {phase_idx}: Running {phase_label} across all tasks ---")
            _reset_persistent_conn(base_url)
            if gateway_url:
                _reset_persistent_conn(gateway_url)

            for idx, task in enumerate(tasks, 1):
                print(f"[{idx}/{len(tasks)}] {arm_name.upper()} :: {task.task_id}: {task.title}")
                res = run_task_arm(
                    task=task,
                    arm=arm_name,
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    max_turns=max_turns,
                    seen_prefixes=prefix_set,
                    gateway_url=gateway_url,
                    ablation=ablation,
                    upstream_kind=upstream_kind,
                    include_sdcs_footer=include_sdcs_footer,
                    temperature=temperature,
                )
                rep_task_results[arm_name].append(res)
                print(
                    f"  -> [{arm_name.upper()}]   Solved: {res.solved} ({res.passed_tests}/45) | "
                    f"Turns: {res.turns} | PromptTok: {res.total_prompt_tokens} | "
                    f"AvgTTFT: {res.avg_ttft_ms}ms | ServerHit: {res.cache_hit_rate_pct}% | "
                    f"PrefixMatch: {res.prefix_match_rate_pct}% | Hints: {res.hints_count}\n"
                )
            _reset_persistent_conn(base_url)
            if gateway_url:
                _reset_persistent_conn(gateway_url)

        baseline_results = rep_task_results["baseline"]
        cacheforge_results = rep_task_results["cacheforge"]

        summary = _build_scorecard_dict(
            mode_label=f"100% Original {len(tasks)}-Task Autonomous Bug-Fix Suite ({ablation})",
            model_name=model_name,
            base_url=base_url,
            baseline_results=baseline_results,
            cacheforge_results=cacheforge_results,
            api_key=api_key,
            upstream_kind=upstream_kind,
            server_launch_command=server_launch_command,
            gateway_url=gateway_url,
        )
        summary["ablation"] = ablation
        summary["rep"] = rep
        all_runs.append(summary)

    summary = copy.deepcopy(all_runs[-1])
    summary["raw_repeats"] = all_runs
    summary["repeats"] = repeats
    if repeats > 1:
        invalid_reps = [
            f"Rep {r.get('rep', i + 1)}: {r.get('invalidation_reason')}"
            for i, r in enumerate(all_runs)
            if not r.get("is_valid", True)
        ]
        if invalid_reps:
            summary["is_valid"] = False
            summary["invalidation_reason"] = "; ".join(invalid_reps)
        else:
            summary["is_valid"] = True
            summary["invalidation_reason"] = None

        summary["variance_analysis"] = _compute_suite_variance(all_runs)
        _print_suite_variance(
            summary["variance_analysis"], f"{repeats} Repeats (All), Ablation: {ablation}"
        )

        valid_runs = [r for r in all_runs if r.get("is_valid", True)]
        if valid_runs and len(valid_runs) < len(all_runs):
            summary["variance_analysis_valid_only"] = _compute_suite_variance(valid_runs)
            _print_suite_variance(
                summary["variance_analysis_valid_only"],
                f"Valid Repeats Only: {len(valid_runs)}/{repeats}, Ablation: {ablation}",
            )
        else:
            summary["variance_analysis_valid_only"] = None

    results_dir = BENCH_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "latest_tensorroute_benchmark.json"
    if repeats > 1:
        for r in all_runs:
            rep_file = results_dir / f"{out_file.stem}_rep{r.get('rep', 1)}.json"
            rep_file.write_text(json.dumps(r, indent=2), encoding="utf-8")

    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if "nebius" in base_url.lower() and len(tasks) >= 5:
        nebius_file = results_dir / "nebius_nemotron_120b_benchmark.json"
        nebius_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validity_str = (
        "VALID"
        if summary.get("is_valid", True)
        else f"INVALID ({summary.get('invalidation_reason')})"
    )

    print("====================================================================")
    print(" EMPIRICAL SCORECARD :: TensorRoute-Bench v2 (Autonomous Bug-Fix Suite)")
    print("====================================================================")
    cf_summary = summary.get("runsift", summary.get("cacheforge", {}))
    print(f" Run Validity:          {validity_str}")
    print(
        f" Tasks Solved (45/45):  {summary['baseline']['tasks_solved']} ({summary['baseline']['solve_rate_pct']}%) -->  "
        f"{cf_summary['tasks_solved']} ({cf_summary['solve_rate_pct']}%) "
        f"({summary['improvements']['solve_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Loop Hints Injected:   {summary['baseline'].get('hints_count', 0)} ({summary['baseline'].get('hints_per_session', 0.0)}/sess) -->  "
        f"{cf_summary.get('hints_count', 0)} ({cf_summary.get('hints_per_session', 0.0)}/sess) "
        f"({summary['improvements'].get('hints_per_session_delta', 0.0):+.2f}/sess)"
    )
    print(
        f" Nudge Rate (Stuck):    {summary['baseline']['nudge_rate_pct']:5.1f}%  -->  "
        f"{cf_summary['nudge_rate_pct']:5.1f}%  ({summary['improvements']['nudge_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Server KV Cache Hit:   {summary['baseline']['cache_hit_rate_pct']:5.1f}%  -->  "
        f"{cf_summary['cache_hit_rate_pct']:5.1f}%  ({summary['improvements']['cache_hit_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Prefix Match Rate:     {summary['baseline']['prefix_match_rate_pct']:5.1f}%  -->  "
        f"{cf_summary['prefix_match_rate_pct']:5.1f}%  ({summary['improvements']['prefix_match_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Mean SSE Wire TTFT:    {summary['baseline']['avg_ttft_ms']:5.1f}ms -->  "
        f"{cf_summary['avg_ttft_ms']:5.1f}ms ({summary['improvements']['ttft_reduction_pct']:+.1f}%)"
    )
    print(
        f" Uncached Prefill Tok:  {summary['baseline']['uncached_prefill_tokens']:6d}  -->  "
        f"{cf_summary['uncached_prefill_tokens']:6d}  ({summary['improvements']['uncached_prefill_savings_pct']:+.1f}%)"
    )
    print(
        f" Cumulative Prompt Tok: {summary['baseline']['total_prompt_tokens']:6d}  -->  "
        f"{cf_summary['total_prompt_tokens']:6d}  ({summary['improvements']['token_savings_pct']:+.1f}%, saved {summary['improvements']['tokens_saved_total']:+d} tokens)"
    )
    print("====================================================================")
    print(f" Saved benchmark artifact to: {out_file}")
    return summary


def _load_dotenv_vars() -> dict[str, str]:
    loaded: dict[str, str] = {}
    candidates = [
        SDCS_ROOT / ".env",
        SDCS_ROOT.parent / "RunSift" / ".env",
        SDCS_ROOT.parent / "CacheForge" / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            for raw_ln in env_path.read_text(encoding="utf-8").splitlines():
                ln = raw_ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    loaded[k.strip()] = v.strip().strip('"').strip("'")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TensorRoute-Bench v2 live autonomous agent evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["suite", "gauntlet", "both"],
        default="suite",
        help="Execution mode: 'suite' (15 isolated tasks), 'gauntlet' (5 simultaneous bugs in 1 continuous session), or 'both'",
    )
    parser.add_argument(
        "--wave",
        type=int,
        default=0,
        help="Optional 5-task wave selector (1=OB-01..05, 2=OB-06..10, 3=OB-11..15, 0=all 15 tasks in suite mode)",
    )
    parser.add_argument(
        "--nebius",
        action="store_true",
        help="Use Nebius Token Factory endpoint and credentials from .env",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenAI-compatible /v1 base URL",
    )
    parser.add_argument(
        "--upstream-kind",
        choices=["nebius", "llamacpp", "generic"],
        default="",
        help="Upstream inference provider kind for parameter selection (default: auto-detected)",
    )
    parser.add_argument(
        "--ablation",
        choices=["both", "canonicalize-only", "compact-only", "none"],
        default="both",
        help="Ablation configuration: 'both' (default), 'canonicalize-only', 'compact-only', or 'none'",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repetitions per task/arm to measure variance and report standard deviations (N>=3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for chat completion requests (default: 0.0)",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.environ.get("RUNSIFT_GATEWAY_URL", os.environ.get("CACHEFORGE_GATEWAY_URL", "http://127.0.0.1:8765/v1")),
        help="RunSift gateway proxy base URL (default: http://127.0.0.1:8765/v1) for live proxy routing",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for upstream endpoint",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model identifier",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Optional substring filter for task ID (e.g. OB-01)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Maximum autonomous agent turns (default: 6 for suite, 20 for gauntlet)",
    )
    parser.add_argument(
        "--server-launch-command",
        default=os.environ.get("LLAMA_SERVER_LAUNCH_CMD", ""),
        help="Exact launch command for the inference server to record in provenance",
    )
    parser.add_argument(
        "--include-sdcs-footer",
        action="store_true",
        help="Include SDCS module status bar footer in compacted pytest digest (ablation)",
    )
    parser.add_argument(
        "--repeat-offset",
        type=int,
        default=0,
        help="Optional offset for repetition counter when extending an existing benchmark run (default: 0)",
    )
    parser.add_argument(
        "--result-name",
        default="",
        help="Optional explicit filename (stem) for saving gauntlet results",
    )
    parser.add_argument(
        "--arm",
        choices=["both", "runsift", "cacheforge", "baseline"],
        default="both",
        help="Arm to evaluate: 'both' (default), 'runsift', 'cacheforge', or 'baseline'",
    )
    parser.add_argument(
        "--task-set",
        choices=["original", "coupled"],
        default="original",
        help="Task set to evaluate ('original' or 'coupled')",
    )
    args = parser.parse_args()
    dotenv_vars = _load_dotenv_vars()

    preflight_workspace_check(task_set=args.task_set)

    if args.nebius:
        base_url = (
            args.base_url
            or os.environ.get("NEBIUS_BASE_URL")
            or dotenv_vars.get("NEBIUS_BASE_URL")
            or "https://api.tokenfactory.nebius.com/v1"
        )
        api_key = (
            args.api_key
            or os.environ.get("NEBIUS_API_KEY")
            or dotenv_vars.get("NEBIUS_API_KEY")
            or ""
        )
        model_name = (
            args.model
            or os.environ.get("UPSTREAM_MODEL")
            or dotenv_vars.get("UPSTREAM_MODEL")
            or "nvidia/nemotron-3-super-120b-a12b"
        )
    else:
        base_url = args.base_url or os.environ.get("BENCHMARK_BASE_URL", "http://127.0.0.1:8081/v1")
        api_key = args.api_key or os.environ.get("BENCHMARK_API_KEY", "")
        if not args.model:
            model_name = os.environ.get("BENCHMARK_MODEL", "")
            if not model_name:
                with contextlib.suppress(
                    OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError
                ):
                    m_clean = base_url.rstrip("/")
                    m_url = (
                        f"{m_clean}/models" if m_clean.endswith("/v1") else f"{m_clean}/v1/models"
                    )
                    req_m = urllib.request.Request(m_url)
                    with urllib.request.urlopen(req_m, timeout=2) as resp_m:
                        m_data = json.loads(resp_m.read().decode("utf-8")).get("data", [])
                        if m_data and "id" in m_data[0]:
                            model_name = str(m_data[0]["id"])
            if not model_name:
                model_name = "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf"
        else:
            model_name = args.model

    if args.upstream_kind:
        upstream_kind = args.upstream_kind
    elif args.nebius or "nebius" in base_url.lower():
        upstream_kind = "nebius"
    elif any(h in base_url.lower() for h in ("8080", "8081", "llama")):
        upstream_kind = "llamacpp"
    else:
        upstream_kind = "generic"

    gateway_url = args.gateway_url

    if args.mode in ("gauntlet", "both"):
        default_turns = 40 if args.task_set == "coupled" else 20
        g_turns = args.max_turns if args.max_turns > 0 else default_turns
        g_wave = args.wave if args.wave in (1, 2, 3) else 1
        run_gauntlet_comparison(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            max_turns=g_turns,
            wave=g_wave,
            gateway_url=gateway_url,
            ablation=args.ablation,
            repeats=args.repeats,
            temperature=args.temperature,
            upstream_kind=upstream_kind,
            server_launch_command=args.server_launch_command,
            include_sdcs_footer=args.include_sdcs_footer,
            repeat_offset=args.repeat_offset,
            result_name=args.result_name,
            arm=args.arm,
            task_set=args.task_set,
        )

    if args.mode in ("suite", "both"):
        s_turns = args.max_turns if args.max_turns > 0 else 6
        run_comparative_suite(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            task_filter=args.task,
            wave=args.wave,
            max_turns=s_turns,
            gateway_url=gateway_url,
            ablation=args.ablation,
            repeats=args.repeats,
            temperature=args.temperature,
            upstream_kind=upstream_kind,
            server_launch_command=args.server_launch_command,
            include_sdcs_footer=args.include_sdcs_footer,
        )


if __name__ == "__main__":
    main()
