"""Turn-2 Probe (PRE_REGISTRATION.md Section 12).

Rebuilds Run F rep 8's second request (first reply after the first test run)
using the harness's own request code and sends 40 alternating raw vs condensed
requests (80 total) to Nebius Token Factory (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`).
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tensorroute_benchmark.harness import (
    AGENT_TOOLS_SCHEMA,
    BASE_AGENT_CONSTITUTION,
    BENCHMARK_TASKS,
    AuthQuotaAbortError,
    parse_agent_tool_call,
    stream_chat_completion_sse,
)
from src.compact import CondenserV2

RESULTS_DIR = REPO_ROOT / "benchmarks" / "tensorroute_benchmark" / "results"
REP8_JSON_PATH = RESULTS_DIR / "nebius_gauntlet_nemotron_30b_40turn_v2_1_rep8.json"
REP8_SIDECAR_PATH = (
    RESULTS_DIR / "nebius_gauntlet_nemotron_30b_40turn_v2_1_rep8_tool_outputs.jsonl.gz"
)

KNOWN_TOOL_NAMES = {
    t["function"]["name"] for t in AGENT_TOOLS_SCHEMA if "function" in t and "name" in t["function"]
}


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher's exact test p-value for a 2x2 table [[a, b], [c, d]]."""
    r1 = a + b
    r2 = c + d
    c1 = a + c
    n = r1 + r2

    def hypergeom_pmf(x: int) -> float:
        return (math.comb(c1, x) * math.comb(n - c1, r1 - x)) / math.comb(n, r1)

    p_obs = hypergeom_pmf(a)
    lo = max(0, r1 + c1 - n)
    hi = min(r1, c1)
    p_val = sum(p for x in range(lo, hi + 1) if (p := hypergeom_pmf(x)) <= p_obs + 1e-12)
    return min(1.0, p_val)


def build_rep8_turn2_payload(
    model_name: str, temperature: float = 0.6
) -> tuple[dict[str, Any], str]:
    """Rebuilds Run F rep 8's second request using the harness's exact request construction."""
    rep8_data = json.loads(REP8_JSON_PATH.read_text(encoding="utf-8"))
    cf_tr = rep8_data.get("runsift", rep8_data.get("cacheforge", {}))["task_results"][0]
    turn1_reply_text = cf_tr["per_turn_raw_replies"][0]
    turn1_tool_args_json = cf_tr["per_turn_tool_args"][0]  # '{"target": "None"}'

    sc_rows = [
        json.loads(ln)
        for ln in gzip.decompress(REP8_SIDECAR_PATH.read_bytes()).decode("utf-8").splitlines()
        if ln.strip()
    ]
    cf_turn1_sc = next(r for r in sc_rows if r.get("arm") in ("runsift", "cacheforge") and r.get("turn") == 1)
    raw_first_test_output = cf_turn1_sc["raw_output"]
    if len(raw_first_test_output) != 40141:
        raise RuntimeError(
            f"Expected 40,141 chars in rep 8 sidecar turn 1 raw_output, got {len(raw_first_test_output)}"
        )

    wave = 1
    start_idx = max(0, (wave - 1) * 5)
    gauntlet_tasks = BENCHMARK_TASKS[start_idx : start_idx + 5]
    session_id = f"GAUNTLET-WAVE-{wave}"

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

    stable_tools = list(AGENT_TOOLS_SCHEMA)
    tools_block = json.dumps(stable_tools, separators=(",", ":"))
    sys_prompt = (
        f"Current Date: 2026-10-15 UTC\n"
        f"Session-ID: sess-{session_id}\n"
        f"[Registered Tool Schemas]: {tools_block}\n\n"
        f"{BASE_AGENT_CONSTITUTION}"
    )

    tc_id = "call_0"
    history: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": turn1_reply_text if turn1_reply_text else None,
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "run_pytest",
                        "arguments": turn1_tool_args_json,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "run_pytest",
            "content": raw_first_test_output,
        },
    ]

    raw_payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "system", "content": sys_prompt}, *history],
        "tools": stable_tools,
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return raw_payload, raw_first_test_output


def verify_gateway_condenser_integrity(
    raw_payload: dict[str, Any], raw_first_test_output: str
) -> str:
    """Verifies Section 12.2 Integrity Check before counting:
    the gateway's condensed tool message must equal the one recomputed from rep 8's
    sidecar byte for byte, which is 9,662 characters (`x-runsift-tool-chars-after`).
    """
    c2 = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=True)
    recomputed_digest, _ = c2.compact_tool_output(
        raw_first_test_output,
        tool_name="run_pytest",
        tool_args='{"target": "None"}',
        is_stale=False,
    )
    if len(recomputed_digest) != 9662:
        raise RuntimeError(
            f"Integrity check failed: recomputed digest length {len(recomputed_digest)} != 9662"
        )

    compacted_msgs, _saved, chars_before, chars_after = c2.compact_messages(raw_payload["messages"])
    gw_tool_msg = compacted_msgs[-1]["content"]
    if (
        chars_before != 40141
        or chars_after != 9662
        or gw_tool_msg.encode("utf-8") != recomputed_digest.encode("utf-8")
    ):
        raise RuntimeError(
            f"Integrity check failed: gateway compact_messages mismatch "
            f"(before={chars_before}, after={chars_after}, len={len(gw_tool_msg)})"
        )
    return recomputed_digest


def get_git_provenance() -> dict[str, Any]:
    commit_res = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    status_res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    dirty_lines = [
        ln for ln in status_res.stdout.splitlines() if ln.strip() and "turn2_probe" not in ln
    ]
    return {
        "git_commit": commit_res.stdout.strip(),
        "git_dirty": len(dirty_lines) > 0,
        "dirty_files": dirty_lines,
    }


def main() -> None:
    api_key = (os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        env_path = REPO_ROOT / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("NEBIUS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not api_key:
        raise RuntimeError("Missing NEBIUS_API_KEY in environment or .env")

    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").strip()
    gateway_url = os.environ.get("RUNSIFT_GATEWAY_URL", os.environ.get("CACHEFORGE_GATEWAY_URL", "http://127.0.0.1:8765/v1")).strip()
    model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    temperature = 0.6
    n_pairs = 40

    prov = get_git_provenance()
    print(
        f"=== TURN-2 PROBE (Section 12) | commit={prov['git_commit']} | git_dirty={prov['git_dirty']} ===",
        flush=True,
    )
    if prov["git_dirty"]:
        raise RuntimeError(f"Refusing to run on dirty git tree: {prov['dirty_files']}")

    raw_payload, raw_first_test_output = build_rep8_turn2_payload(
        model_name=model_name, temperature=temperature
    )
    recomputed_digest = verify_gateway_condenser_integrity(raw_payload, raw_first_test_output)
    print(
        f"✓ Local CondenserV2 integrity check passed: raw=40,141 chars -> condensed={len(recomputed_digest)} chars (byte-identical to rep 8 sidecar).",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    infra_retries_logged: list[dict[str, Any]] = []

    for pair_idx in range(1, n_pairs + 1):
        for arm in ("raw", "condensed"):
            req_index = len(records) + 1
            target_url = base_url if arm == "raw" else gateway_url
            payload_copy = copy.deepcopy(raw_payload)
            seen_prefixes: set[str] = set()

            while True:
                t0 = time.perf_counter()
                try:
                    resp_text, native_tcs, meta = stream_chat_completion_sse(
                        base_url=target_url,
                        api_key=api_key,
                        payload=payload_copy,
                        prefix_hash="",
                        seen_prefixes=seen_prefixes,
                        extra_headers={},
                        upstream_kind="nebius",
                    )
                    break
                except AuthQuotaAbortError as auth_exc:
                    print(f"ABORT: Auth/Quota error (401/402/403): {auth_exc}", flush=True)
                    raise
                except (RuntimeError, OSError, ValueError) as exc:
                    elapsed_s = round(time.perf_counter() - t0, 3)
                    err_msg = str(exc)
                    infra_retries_logged.append(
                        {
                            "req_index": req_index,
                            "pair": pair_idx,
                            "arm": arm,
                            "error": err_msg,
                            "elapsed_s": elapsed_s,
                        }
                    )
                    print(
                        f"  [!] Infra error on req {req_index} (pair {pair_idx}, {arm}): {err_msg:.120s} — re-sending...",
                        flush=True,
                    )
                    time.sleep(2.0)

            if arm == "condensed":
                hdr_cond = str(meta.get("condenser", ""))
                hdr_before = int(meta.get("tool_chars_before", 0))
                hdr_after = int(meta.get("tool_chars_after", 0))
                if hdr_cond != "v2" or hdr_before != 40141 or hdr_after != 9662:
                    raise RuntimeError(
                        f"Live gateway integrity check failed on req {req_index}: "
                        f"condenser={hdr_cond!r}, before={hdr_before}, after={hdr_after} (expected 'v2', 40141, 9662)"
                    )

            parsed_tc = parse_agent_tool_call(resp_text, native_tcs)
            has_tool_call = parsed_tc is not None
            no_tool_call = not has_tool_call

            stripped_content = resp_text.lstrip()
            is_json_analysis_plan = stripped_content.startswith("{") and ('"analysis"' in resp_text)

            raw_tool_names = [
                str((tc.get("function") or {}).get("name") or "") for tc in (native_tcs or [])
            ]
            unknown_tool_calls = [tn for tn in raw_tool_names if tn and tn not in KNOWN_TOOL_NAMES]
            # Also check if resp_text or parsed failed due to unknown tool name in JSON text
            if not unknown_tool_calls and "str_replace_editor" in resp_text:
                unknown_tool_calls.append("str_replace_editor")

            rec = {
                "req_index": req_index,
                "pair": pair_idx,
                "arm": arm,
                "no_tool_call": no_tool_call,
                "parsed_tool": parsed_tc["tool"] if parsed_tc else None,
                "parsed_tool_args": parsed_tc["args"] if parsed_tc else None,
                "native_tool_calls": native_tcs,
                "raw_reply_content": resp_text,
                "is_json_analysis_plan": is_json_analysis_plan,
                "unknown_tool_calls": unknown_tool_calls,
                "prompt_tokens": int(meta.get("prompt_tokens", 0)),
                "completion_tokens": int(meta.get("completion_tokens", 0)),
                "upstream_cached_tokens": int(meta.get("upstream_cached_tokens", 0)),
                "ttft_ms": float(meta.get("ttft_ms", 0.0)),
                "model_ms": float(meta.get("model_ms", 0.0)),
                "condenser_header": str(meta.get("condenser", "")),
                "tool_chars_before": int(meta.get("tool_chars_before", 0)),
                "tool_chars_after": int(meta.get("tool_chars_after", 0)),
            }
            records.append(rec)

            print(
                f"  [{req_index:2d}/80 | Pair {pair_idx:2d}/40 | {arm:9s}] "
                f"no_tool={no_tool_call!s:5s} | tool={rec['parsed_tool']!s:12s} | "
                f"json_analysis={is_json_analysis_plan!s:5s} | unknown_tools={unknown_tool_calls} | "
                f"prompt_tok={rec['prompt_tokens']:5d} | comp_tok={rec['completion_tokens']:4d} | "
                f"ttft={rec['ttft_ms']:6.1f}ms",
                flush=True,
            )

    raw_recs = [r for r in records if r["arm"] == "raw"]
    cond_recs = [r for r in records if r["arm"] == "condensed"]

    y_raw_no_tool = sum(1 for r in raw_recs if r["no_tool_call"])
    x_cond_no_tool = sum(1 for r in cond_recs if r["no_tool_call"])

    raw_with_tool = len(raw_recs) - y_raw_no_tool
    cond_with_tool = len(cond_recs) - x_cond_no_tool

    fisher_p = fisher_exact_two_sided(x_cond_no_tool, cond_with_tool, y_raw_no_tool, raw_with_tool)

    raw_json_analysis = sum(1 for r in raw_recs if r["is_json_analysis_plan"])
    cond_json_analysis = sum(1 for r in cond_recs if r["is_json_analysis_plan"])

    raw_unknown_tools = sum(1 for r in raw_recs if r["unknown_tool_calls"])
    cond_unknown_tools = sum(1 for r in cond_recs if r["unknown_tool_calls"])

    if x_cond_no_tool > y_raw_no_tool and fisher_p < 0.05:
        readme_wording = (
            f"On the first reply after a condensed test log, the model skipped tool calls in "
            f"{x_cond_no_tool} of 40 requests, against {y_raw_no_tool} of 40 with the raw log. "
            f"This applies to v2 and v2.1 alike."
        )
    else:
        readme_wording = (
            f"Not reproduced in a 40-vs-40 probe ({x_cond_no_tool} of 40 vs {y_raw_no_tool} of 40)."
        )

    summary_payload = {
        "experiment": "turn2_probe_section12",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": prov,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": 1024,
        "requests_per_arm": n_pairs,
        "total_requests": len(records),
        "integrity_check": {
            "raw_first_test_output_chars": len(raw_first_test_output),
            "recomputed_condensed_chars": len(recomputed_digest),
            "gateway_header_chars_after": 9662,
            "byte_identical_verified": True,
        },
        "primary_metric": {
            "condensed_no_tool_call_x": x_cond_no_tool,
            "condensed_total": len(cond_recs),
            "raw_no_tool_call_y": y_raw_no_tool,
            "raw_total": len(raw_recs),
            "fisher_exact_two_sided_p": round(fisher_p, 6),
            "alpha": 0.05,
        },
        "secondary_metrics": {
            "condensed_json_analysis_plan_count": cond_json_analysis,
            "raw_json_analysis_plan_count": raw_json_analysis,
            "condensed_unknown_tool_calls_count": cond_unknown_tools,
            "raw_unknown_tool_calls_count": raw_unknown_tools,
        },
        "section_12_4_wording": readme_wording,
        "infra_retries": infra_retries_logged,
        "replies": records,
    }

    out_json = RESULTS_DIR / "nebius_turn2_probe_v2.json"
    out_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n=== TURN-2 PROBE SUMMARY (Section 12) ===", flush=True)
    print(
        f"Condensed skipped tool calls (x): {x_cond_no_tool} / 40 ({x_cond_no_tool / 40 * 100:.1f}%)",
        flush=True,
    )
    print(
        f"Raw skipped tool calls (y):       {y_raw_no_tool} / 40 ({y_raw_no_tool / 40 * 100:.1f}%)",
        flush=True,
    )
    print(f"Fisher's exact test (two-sided) p-value: {fisher_p:.6f}", flush=True)
    print(
        f"Secondary — JSON 'analysis' plans: condensed={cond_json_analysis}/40, raw={raw_json_analysis}/40",
        flush=True,
    )
    print(
        f"Secondary — Non-existent tool calls: condensed={cond_unknown_tools}/40, raw={raw_unknown_tools}/40",
        flush=True,
    )
    print(f"Section 12.4 Wording: {readme_wording}", flush=True)
    print(f"Saved all 80 raw replies to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
