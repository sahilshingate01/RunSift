"""Replays committed sidecars to verify shell-tool pytest equivalence (A1, A2, A3).

Implements Section 8 of `docs/specs/CONDENSER_V2_SHELL_TESTS_ADDENDUM.md`:
- A1: Log-level replay of all 1,761 `run_pytest` records across all committed sidecars
  under both `v2` and `v2.1`, comparing `run_pytest` with recorded arguments against
  `bash` with `{"command": "python -m pytest -o pythonpath=src -vv -ra --tb=long --durations=0", **recorded_args}`.
  Also reports the optional command-only check for the 1,039 records whose target is a `.py` path
  or `.py::` test id and the 218 with no target (1,257 total).
- A2: Session-level replay of all sidecar sessions (by file + arm), renaming `run_pytest` to `bash`
  with arguments as in A1 and running `compact_messages` on the full history for `v2` and `v2.1`.
- A3: Checks all 6,809 recorded non-`run_pytest` tool outputs under their recorded tool name
  against D2+D3 (`CondenserV2._is_shell_pytest_output`).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compact.condenser_v2 import CondenserV2

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BASE_PYTEST_CMD = "python -m pytest -o pythonpath=src -vv -ra --tb=long --durations=0"


def _load_all_sidecars() -> list[tuple[Path, list[dict[str, Any]]]]:
    sidecar_paths = sorted(RESULTS_DIR.glob("*_tool_outputs.jsonl.gz"))
    loaded: list[tuple[Path, list[dict[str, Any]]]] = []
    for sp in sidecar_paths:
        raw_bytes = gzip.decompress(sp.read_bytes())
        rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
        loaded.append((sp, rows))
    return loaded


def _is_py_target(target: Any) -> bool:
    if not isinstance(target, str) or not target:
        return False
    return target.endswith(".py") or ".py::" in target


def run_equivalence_checks() -> dict[str, Any]:
    sidecars = _load_all_sidecars()

    condensers = {
        "v2": CondenserV2(max_char_threshold=4000, enable_stale_rewrite=True),
        "v2.1": CondenserV2(max_char_threshold=4000, enable_stale_rewrite=False),
    }

    # --- A1 & A3 ---
    a1_total = 0
    a1_matches = {"v2": 0, "v2.1": 0}
    a1_mismatches: dict[str, list[dict[str, Any]]] = {"v2": [], "v2.1": []}

    a1_cmd_only_total = 0
    a1_cmd_only_py_count = 0
    a1_cmd_only_no_target_count = 0
    a1_cmd_only_matches = {"v2": 0, "v2.1": 0}

    a3_non_pytest_total = 0
    a3_detected_count = 0
    a3_detected_examples: list[dict[str, Any]] = []

    for sp, rows in sidecars:
        for row_idx, r in enumerate(rows):
            tool_name = str(r.get("tool") or "")
            raw_output = str(r.get("raw_output") or "")
            recorded_args = r.get("args") if isinstance(r.get("args"), dict) else {}

            if tool_name == "run_pytest":
                a1_total += 1
                args_run_pytest = json.dumps(recorded_args)
                bash_args_dict = {"command": BASE_PYTEST_CMD, **recorded_args}
                args_bash = json.dumps(bash_args_dict)

                for ver, cond in condensers.items():
                    out_rp, _ = cond.compact_tool_output(
                        raw_output, tool_name="run_pytest", tool_args=args_run_pytest
                    )
                    out_bash, _ = cond.compact_tool_output(
                        raw_output, tool_name="bash", tool_args=args_bash
                    )
                    if out_rp.encode("utf-8") == out_bash.encode("utf-8"):
                        a1_matches[ver] += 1
                    elif len(a1_mismatches[ver]) < 5:
                        a1_mismatches[ver].append(
                            {
                                "sidecar": sp.name,
                                "row": row_idx,
                                "turn": r.get("turn"),
                                "args": recorded_args,
                                "run_pytest_len": len(out_rp),
                                "bash_len": len(out_bash),
                            }
                        )

                # Optional command-only check for .py / .py:: targets (1,039) + no target (218)
                target_val = recorded_args.get("target")
                is_empty_target = not target_val
                is_py_t = _is_py_target(target_val)
                if is_empty_target or is_py_t:
                    a1_cmd_only_total += 1
                    if is_empty_target:
                        a1_cmd_only_no_target_count += 1
                        cmd_str = BASE_PYTEST_CMD
                    else:
                        a1_cmd_only_py_count += 1
                        cmd_str = f"{BASE_PYTEST_CMD} {target_val}"
                    args_cmd_only = json.dumps({"command": cmd_str})
                    for ver, cond in condensers.items():
                        out_rp, _ = cond.compact_tool_output(
                            raw_output, tool_name="run_pytest", tool_args=args_run_pytest
                        )
                        out_cmd, _ = cond.compact_tool_output(
                            raw_output, tool_name="bash", tool_args=args_cmd_only
                        )
                        if out_rp.encode("utf-8") == out_cmd.encode("utf-8"):
                            a1_cmd_only_matches[ver] += 1
            else:
                a3_non_pytest_total += 1
                if condensers["v2"]._is_shell_pytest_output(raw_output, tool_name=tool_name):
                    a3_detected_count += 1
                    if len(a3_detected_examples) < 5:
                        a3_detected_examples.append(
                            {
                                "sidecar": sp.name,
                                "row": row_idx,
                                "tool": tool_name,
                                "raw_head": raw_output[:120],
                            }
                        )

    # --- A2: Session-level replay ---
    a2_sessions_total = 0
    a2_tool_msgs_total = {"v2": 0, "v2.1": 0}
    a2_mismatches = {"v2": 0, "v2.1": 0}
    a2_mismatch_examples: dict[str, list[dict[str, Any]]] = {"v2": [], "v2.1": []}

    for sp, rows in sidecars:
        arms_present = sorted({str(r.get("arm") or "") for r in rows})
        for arm in arms_present:
            arm_rows = [r for r in rows if str(r.get("arm") or "") == arm]
            if not arm_rows:
                continue
            a2_sessions_total += 1

            orig_msgs: list[dict[str, Any]] = [
                {"role": "system", "content": "You are an autonomous debugging agent."},
                {"role": "user", "content": "Fix all bugs in the repository."},
            ]
            bash_msgs: list[dict[str, Any]] = [
                {"role": "system", "content": "You are an autonomous debugging agent."},
                {"role": "user", "content": "Fix all bugs in the repository."},
            ]

            for idx, r in enumerate(arm_rows, start=1):
                t_name = str(r.get("tool") or "")
                t_args = r.get("args") if isinstance(r.get("args"), dict) else {}
                raw_out = str(r.get("raw_output") or "")
                tc_id = f"call_{idx}"

                orig_msgs.append(
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
                orig_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": t_name,
                        "content": raw_out,
                    }
                )

                if t_name == "run_pytest":
                    b_name = "bash"
                    b_args = {"command": BASE_PYTEST_CMD, **t_args}
                else:
                    b_name = t_name
                    b_args = t_args

                bash_msgs.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": b_name,
                                    "arguments": json.dumps(b_args),
                                },
                            }
                        ],
                    }
                )
                bash_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": b_name,
                        "content": raw_out,
                    }
                )

            for ver, cond in condensers.items():
                comp_orig, _, _, _ = cond.compact_messages(orig_msgs)
                comp_bash, _, _, _ = cond.compact_messages(bash_msgs)

                orig_tools = [m for m in comp_orig if m.get("role") == "tool"]
                bash_tools = [m for m in comp_bash if m.get("role") == "tool"]
                assert len(orig_tools) == len(bash_tools)

                for m_idx, (mo, mb) in enumerate(zip(orig_tools, bash_tools, strict=True)):
                    a2_tool_msgs_total[ver] += 1
                    if str(mo.get("content")).encode("utf-8") != str(mb.get("content")).encode(
                        "utf-8"
                    ):
                        a2_mismatches[ver] += 1
                        if len(a2_mismatch_examples[ver]) < 5:
                            a2_mismatch_examples[ver].append(
                                {
                                    "sidecar": sp.name,
                                    "arm": arm,
                                    "tool_msg_index": m_idx,
                                    "orig_tool": mo.get("name"),
                                    "bash_tool": mb.get("name"),
                                }
                            )

    return {
        "sidecar_files": len(sidecars),
        "A1": {
            "total_run_pytest_records": a1_total,
            "v2_byte_identical": a1_matches["v2"],
            "v2_1_byte_identical": a1_matches["v2.1"],
            "v2_mismatches": a1_mismatches["v2"],
            "v2_1_mismatches": a1_mismatches["v2.1"],
            "command_only_optional": {
                "eligible_records": a1_cmd_only_total,
                "py_target_records": a1_cmd_only_py_count,
                "no_target_records": a1_cmd_only_no_target_count,
                "v2_byte_identical": a1_cmd_only_matches["v2"],
                "v2_1_byte_identical": a1_cmd_only_matches["v2.1"],
            },
        },
        "A2": {
            "sessions_replayed": a2_sessions_total,
            "v2_tool_messages": a2_tool_msgs_total["v2"],
            "v2_1_tool_messages": a2_tool_msgs_total["v2.1"],
            "v2_mismatches": a2_mismatches["v2"],
            "v2_1_mismatches": a2_mismatches["v2.1"],
            "v2_mismatch_examples": a2_mismatch_examples["v2"],
            "v2_1_mismatch_examples": a2_mismatch_examples["v2.1"],
        },
        "A3": {
            "total_non_run_pytest_records": a3_non_pytest_total,
            "detected_by_d2_d3": a3_detected_count,
            "detected_examples": a3_detected_examples,
        },
    }


def main() -> None:
    res = run_equivalence_checks()
    print(json.dumps(res, indent=2))

    a1 = res["A1"]
    a2 = res["A2"]
    a3 = res["A3"]

    if (
        a1["v2_byte_identical"] != a1["total_run_pytest_records"]
        or a1["v2_1_byte_identical"] != a1["total_run_pytest_records"]
        or a2["v2_mismatches"] != 0
        or a2["v2_1_mismatches"] != 0
        or a3["detected_by_d2_d3"] != 0
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
