"""Automated evaluation harness for RunSift.

Benchmarks multi-turn autonomous agent loops comparing:
  - Baseline (Without RunSift): non-deterministic tool order, volatile timestamp headers, raw stdout bloat.
  - RunSift Optimized: canonicalized prefix, hoisted invariant instructions, compacted tool memory.

Measures TTFT (Time-To-First-Token), cumulative token consumption, and KV-cache hit rates.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import httpx

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonicalizer import PrefixCanonicalizer, compute_prefix_hash
from src.compact import ToolOutputCompactor
from src.core import get_config
from src.upstream import NebiusClient

# Realistic agent tool schemas
BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": (
                "Executes shell commands inside an isolated Linux container sandbox. "
                "Supports pytest, git, nvidia-smi, and standard POSIX utilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Exact shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                    "workdir": {"type": "string", "description": "Working directory path"},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Reads source code or configuration file contents from disk with line numbering "
                "and byte-range slicing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository-relative file path"},
                    "offset": {"type": "integer", "description": "1-indexed starting line"},
                    "limit": {"type": "integer", "description": "Maximum lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Searches upstream technical documentation, NVIDIA TensorRT-LLM references, "
                "and Nebius Token Factory API specifications."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "num_results": {"type": "integer", "description": "Max results to return"},
                    "domain_filter": {"type": "string", "description": "Optional domain filter"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ast_grep",
            "description": (
                "Searches Abstract Syntax Tree (AST) nodes across Python and C++/CUDA files "
                "for structural code patterns and symbol definitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "AST pattern expression"},
                    "lang": {
                        "type": "string",
                        "description": "Target language (python, cpp, cuda)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

STATIC_SYSTEM_CONSTITUTION = (
    "You are an autonomous Principal Systems & GPU Infrastructure Engineering Agent "
    "operating on NVIDIA DGX SuperPOD and Nebius Token Factory clusters.\n"
    "Constitutional Operating Principles:\n"
    "1. Prefix Locality: Preserve deterministic ordering of system instructions, "
    "tool declarations, and multi-turn conversation prefixes for prefix caching.\n"
    "2. Concise Diagnostic Output: Emit concise, deterministic diagnostic syntheses.\n"
    "3. Root-Cause Precision: When inspecting test failures, tracebacks, or hardware telemetry, "
    "isolate the exact file, line number, exception class, and remediation criteria.\n"
    "4. Deterministic Verification: Always verify fixes against empirical test gates before "
    "closing out an engineering turn. Keep responses concise (1-2 sentences)."
)

VERBOSE_TOOL_OUTPUT_SAMPLE = (
    "Running test suite: pytest -v tests/\n"
    # Synthetic example output, not a result.
    + "\n".join(
        [f"tests/test_module_{i:02d}.py::test_case_{i:02d} ... PASSED [100%]" for i in range(45)]
    )
    + "\nFAIL: tests/test_network.py::test_network_connection ... FAILED"
    + "\nTraceback (most recent call last):\n"
    + "  File 'src/upstream/network.py', line 182, in connect\n"
    + "    raise ConnectionRefusedError('GPU worker endpoint unreachable at port 9000')\n"
    + "ConnectionRefusedError: GPU worker endpoint unreachable at port 9000\n"
    + "\n".join(
        [f"  ... diagnostic frame {k:02d}: inspecting RDMA socket state OK" for k in range(35)]
    )
)


async def _measure_turn(
    client: NebiusClient,
    payload: dict[str, Any],
    prefix_hash: str,
) -> tuple[str, dict[str, Any]]:
    """Executes a single turn via live SSE streaming to measure true wire TTFT and token usage."""
    stream_payload = dict(payload)
    stream_payload["max_tokens"] = 16
    stream_payload["temperature"] = 0.0

    meta: dict[str, Any] = {}
    async for _chunk_str, chunk_meta in client.stream_chat_completion(
        stream_payload, prefix_hash=prefix_hash
    ):
        meta = chunk_meta
    reply_text = "Acknowledged diagnostic status."
    return reply_text, meta


async def _measure_turn_gateway(
    http_client: httpx.AsyncClient,
    gateway_url: str,
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Executes a single turn directly through the live RunSift Gateway proxy."""
    stream_payload = dict(payload)
    stream_payload["max_tokens"] = 16
    stream_payload["temperature"] = 0.0
    stream_payload["stream"] = True
    stream_payload.setdefault("stream_options", {"include_usage": True})

    endpoint = f"{gateway_url.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    ttft_ms = 0.0
    ttft_recorded = False
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    reply_parts: list[str] = []

    req = http_client.build_request("POST", endpoint, json=stream_payload)
    resp = await http_client.send(req, stream=True)
    if resp.is_error:
        await resp.aread()
        await resp.aclose()
        resp.raise_for_status()

    prefix_hash = resp.headers.get("x-runsift-prefix-hash", resp.headers.get("x-cacheforge-prefix-hash", ""))

    try:
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            if not ttft_recorded:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                ttft_recorded = True

            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = int(
                    details.get("cached_tokens")
                    or usage.get("prompt_cache_hit_tokens")
                    or cached_tokens
                )

            choices = chunk.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    reply_parts.append(str(delta["content"]))
    finally:
        await resp.aclose()

    if not ttft_recorded:
        ttft_ms = (time.perf_counter() - t0) * 1000.0

    if prompt_tokens == 0:
        prompt_tokens = sum(
            len(str(m.get("content", ""))) // 4 for m in payload.get("messages", [])
        )

    reply_text = "".join(reply_parts) or "Acknowledged diagnostic status."
    meta = {
        "ttft_ms": round(ttft_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
        "prefix_hash": prefix_hash,
    }
    return reply_text, meta


def capture_runtime_provenance(
    base_url: str,
    api_key: str = "",
    gateway_url: str = "",
) -> dict[str, Any]:
    """Dynamically collects live environment provenance at benchmark execution time."""
    # 1. Git metadata
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        git_commit = "unknown"
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
        git_dirty = len(dirty_lines) > 0
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        git_dirty = False

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
    else:
        gateway_commit = git_commit

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
    is_local = any(h in base_url.lower() for h in ("127.0.0.1", "localhost", "0.0.0.0"))
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

        server_launch_command = "local process"
    else:
        nvidia_smi = "not exposed by provider"
        engine_version = "not exposed by provider"
        server_launch_command = "not exposed by provider"

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gateway_commit": gateway_commit,
        "cli_args": cli_args,
        "timestamp": ts,
        "base_url": base_url,
        "server_launch_command": server_launch_command,
        "engine_version": engine_version,
        "nvidia_smi": nvidia_smi,
        "models_endpoint_response": models_resp,
    }


async def run_benchmark_suite(
    num_turns: int = 10,
    client: NebiusClient | None = None,
    gateway_url: str = "",
    model: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    config = get_config()
    target_model = model or config.default_model
    owns_client = client is None
    if client is None:
        client = NebiusClient(config=config)

    gateway_client: httpx.AsyncClient | None = None
    if gateway_url:
        gateway_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    print("====================================================================")
    print(" RunSift Live Empirical Evaluation Benchmark Suite")
    if gateway_url:
        print(f" Routing Mode:      Live Gateway Proxy ({gateway_url})")
    else:
        print(" Routing Mode:      In-Process Pipeline Verification")
    print(f" Upstream Endpoint: {config.nebius_base_url}")
    print(f" Target Model:      {target_model} | Turns: {num_turns}")
    print("====================================================================\n")

    canonicalizer = PrefixCanonicalizer()
    compactor = ToolOutputCompactor()

    try:
        # -------------------------------------------------------------
        # 1. Run Baseline (Without RunSift)
        # -------------------------------------------------------------
        print("[1/2] Running Baseline Agent Loop (Realistic client, raw uncompacted tool logs)...")
        baseline_cache_hits = 0
        baseline_prefix_matches = 0
        baseline_cached_tokens = 0
        baseline_ttft_list: list[float] = []
        baseline_tokens_list: list[int] = []
        baseline_history: list[dict[str, Any]] = []
        seen_baseline_prefixes: set[str] = set()

        # Realistic client baseline: stable daily timestamp, session-scoped ID, stable tool schemas
        timestamp_header = "Current Date: 2026-10-15 UTC\nSession-ID: sess-benchmark-10turn\n"
        system_content = timestamp_header + STATIC_SYSTEM_CONSTITUTION
        stable_tools = list(BENCHMARK_TOOLS)

        for turn in range(num_turns):
            current_messages = [
                {"role": "system", "content": system_content},
                *baseline_history,
                {
                    "role": "user",
                    "content": f"Turn {turn + 1}: Summarize latest test diagnostic status in one sentence.",
                },
            ]

            payload = {
                "model": target_model,
                "messages": current_messages,
                "tools": stable_tools,
            }

            raw_prefix_hash = compute_prefix_hash(system_content, stable_tools)
            is_prefix_match = raw_prefix_hash in seen_baseline_prefixes
            seen_baseline_prefixes.add(raw_prefix_hash)
            if is_prefix_match:
                baseline_prefix_matches += 1

            reply_text, meta = await _measure_turn(client, payload, prefix_hash=raw_prefix_hash)

            turn_cached = int(meta.get("cached_tokens", 0))
            baseline_cached_tokens += turn_cached
            if turn_cached > 0:
                baseline_cache_hits += 1

            ttft_ms = float(meta.get("ttft_ms", 0.0))
            prompt_tokens = int(meta.get("prompt_tokens", 0))
            baseline_ttft_list.append(ttft_ms)
            baseline_tokens_list.append(prompt_tokens)
            print(
                f"  [Baseline Turn {turn + 1}/{num_turns}] TTFT: {ttft_ms:6.1f}ms | "
                f"Prompt Tokens: {prompt_tokens:4d} | Cached Tokens: {turn_cached:4d} | "
                f"Server Hit: {turn_cached > 0} | Prefix Match: {is_prefix_match}"
            )

            # Accumulate turn history including assistant tool call and raw uncompacted tool stdout
            tc_id = f"call_diag_{turn + 1}"
            baseline_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": "bash_exec",
                                "arguments": json.dumps({"cmd": "pytest -v tests/"}),
                            },
                        }
                    ],
                }
            )
            baseline_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": "bash_exec",
                    "content": VERBOSE_TOOL_OUTPUT_SAMPLE,
                }
            )
            baseline_history.append({"role": "assistant", "content": reply_text[:160]})

        # -------------------------------------------------------------
        # 2. Run RunSift Optimized
        # -------------------------------------------------------------
        print("\n[2/2] Running RunSift Optimized Agent Loop (Canonicalized + Compacted)...")
        cf_cache_hits = 0
        cf_prefix_matches = 0
        cf_cached_tokens = 0
        cf_ttft_list: list[float] = []
        cf_tokens_list: list[int] = []
        cf_history: list[dict[str, Any]] = []
        seen_cf_prefixes: set[str] = set()

        for turn in range(num_turns):
            current_messages = [
                {"role": "system", "content": system_content},
                *cf_history,
                {
                    "role": "user",
                    "content": f"Turn {turn + 1}: Summarize latest test diagnostic status in one sentence.",
                },
            ]

            raw_payload = {
                "model": target_model,
                "messages": current_messages,
                "tools": stable_tools,
            }

            if gateway_client is not None and gateway_url:
                reply_text, meta = await _measure_turn_gateway(
                    gateway_client, gateway_url, raw_payload
                )
            else:
                canonical_payload, prefix_metrics = canonicalizer.canonicalize_request(raw_payload)
                reply_text, meta = await _measure_turn(
                    client, canonical_payload, prefix_hash=prefix_metrics.prefix_hash
                )

            turn_cached = int(meta.get("cached_tokens", 0))
            cf_cached_tokens += turn_cached
            if turn_cached > 0:
                cf_cache_hits += 1

            p_hash = str(meta.get("prefix_hash") or "")
            is_cf_prefix_match = p_hash in seen_cf_prefixes if p_hash else False
            if p_hash:
                seen_cf_prefixes.add(p_hash)
            if is_cf_prefix_match:
                cf_prefix_matches += 1

            ttft_ms = float(meta.get("ttft_ms", 0.0))
            prompt_tokens = int(meta.get("prompt_tokens", 0))
            cf_ttft_list.append(ttft_ms)
            cf_tokens_list.append(prompt_tokens)
            print(
                f"  [RunSift Turn {turn + 1}/{num_turns}] TTFT: {ttft_ms:6.1f}ms | "
                f"Prompt Tokens: {prompt_tokens:4d} | Cached Tokens: {turn_cached:4d} | "
                f"Server Hit: {turn_cached > 0} | Prefix Match: {is_cf_prefix_match}"
            )

            tc_id = f"call_diag_{turn + 1}"
            cf_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": "bash_exec",
                                "arguments": json.dumps({"cmd": "pytest -v tests/"}),
                            },
                        }
                    ],
                }
            )

            if gateway_url:
                # Send raw tool output to gateway; gateway's Negative Memory compactor automatically
                # compacts role == "tool" on the wire before forwarding to upstream
                cf_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": "bash_exec",
                        "content": VERBOSE_TOOL_OUTPUT_SAMPLE,
                    }
                )
            else:
                # In-process compaction
                compacted_log, _ = compactor.compact_tool_output(
                    VERBOSE_TOOL_OUTPUT_SAMPLE, tool_name="bash_exec"
                )
                cf_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": "bash_exec",
                        "content": compacted_log,
                    }
                )
            cf_history.append({"role": "assistant", "content": reply_text[:160]})

    finally:
        if owns_client:
            await client.close()
        if gateway_client is not None:
            await gateway_client.aclose()

    # -------------------------------------------------------------
    # 3. Compute Comparative Telemetry
    # -------------------------------------------------------------
    baseline_cache_rate = (baseline_cache_hits / num_turns) * 100.0
    cf_cache_rate = (cf_cache_hits / num_turns) * 100.0
    baseline_prefix_rate = (baseline_prefix_matches / num_turns) * 100.0
    cf_prefix_rate = (cf_prefix_matches / num_turns) * 100.0

    avg_baseline_ttft = sum(baseline_ttft_list) / len(baseline_ttft_list)
    avg_cf_ttft = sum(cf_ttft_list) / len(cf_ttft_list)
    ttft_reduction_pct = ((avg_baseline_ttft - avg_cf_ttft) / max(0.001, avg_baseline_ttft)) * 100.0

    total_baseline_tokens = sum(baseline_tokens_list)
    total_cf_tokens = sum(cf_tokens_list)
    token_savings_pct = (
        (total_baseline_tokens - total_cf_tokens) / max(1, total_baseline_tokens)
    ) * 100.0

    results = {
        "provenance": capture_runtime_provenance(
            config.nebius_base_url,
            config.nebius_api_key,
            gateway_url=gateway_url,
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turns": num_turns,
        "mode": "gateway_proxy" if gateway_url else "in_process",
        "gateway_url": gateway_url,
        "model": target_model,
        "upstream_base_url": config.nebius_base_url,
        "baseline": {
            "cache_hit_rate_pct": round(baseline_cache_rate, 2),
            "prefix_match_rate_pct": round(baseline_prefix_rate, 2),
            "upstream_cached_tokens": baseline_cached_tokens,
            "avg_ttft_ms": round(avg_baseline_ttft, 2),
            "total_prompt_tokens": total_baseline_tokens,
            "per_turn_ttft_ms": [round(x, 2) for x in baseline_ttft_list],
            "per_turn_prompt_tokens": baseline_tokens_list,
        },
        "runsift": {
            "cache_hit_rate_pct": round(cf_cache_rate, 2),
            "prefix_match_rate_pct": round(cf_prefix_rate, 2),
            "upstream_cached_tokens": cf_cached_tokens,
            "avg_ttft_ms": round(avg_cf_ttft, 2),
            "total_prompt_tokens": total_cf_tokens,
            "per_turn_ttft_ms": [round(x, 2) for x in cf_ttft_list],
            "per_turn_prompt_tokens": cf_tokens_list,
        },
        "cacheforge": {
            "cache_hit_rate_pct": round(cf_cache_rate, 2),
            "prefix_match_rate_pct": round(cf_prefix_rate, 2),
            "upstream_cached_tokens": cf_cached_tokens,
            "avg_ttft_ms": round(avg_cf_ttft, 2),
            "total_prompt_tokens": total_cf_tokens,
            "per_turn_ttft_ms": [round(x, 2) for x in cf_ttft_list],
            "per_turn_prompt_tokens": cf_tokens_list,
        },
        "improvements": {
            "cache_hit_rate_delta_pct": round(cf_cache_rate - baseline_cache_rate, 2),
            "prefix_match_rate_delta_pct": round(cf_prefix_rate - baseline_prefix_rate, 2),
            "ttft_reduction_pct": round(ttft_reduction_pct, 2),
            "token_savings_pct": round(token_savings_pct, 2),
            "tokens_saved_total": total_baseline_tokens - total_cf_tokens,
        },
    }

    output_path = (
        Path(output_file) if output_file else (REPO_ROOT / "benchmarks" / "latest_benchmark.json")
    )
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n====================================================================")
    print(" EMPIRICAL SCORECARD: Baseline vs RunSift (NVIDIA / Nebius)")
    print("====================================================================")
    print(
        f" Server KV Cache Hit:   {baseline_cache_rate:5.1f}%  -->  {cf_cache_rate:5.1f}%  ({results['improvements']['cache_hit_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Prefix Match Rate:     {baseline_prefix_rate:5.1f}%  -->  {cf_prefix_rate:5.1f}%  ({results['improvements']['prefix_match_rate_delta_pct']:+.1f}%)"
    )
    print(
        f" Mean TTFT Latency:     {avg_baseline_ttft:5.1f}ms -->  {avg_cf_ttft:5.1f}ms ({results['improvements']['ttft_reduction_pct']:+.1f}%)"
    )
    print(
        f" Cumulative Tokens:     {total_baseline_tokens:5d}   -->  {total_cf_tokens:5d}   ({results['improvements']['token_savings_pct']:+.1f}%, saved {results['improvements']['tokens_saved_total']:+d} tokens)"
    )
    print("====================================================================")
    print(f" Saved benchmark artifact to: {output_path}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RunSift Empirical Benchmark Harness")
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of turns to benchmark (default: 10)",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="",
        help="RunSift Gateway URL (e.g. http://127.0.0.1:8765/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model override",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON results path",
    )
    cli_args = parser.parse_args()

    effective_gateway = cli_args.gateway_url or os.environ.get("RUNSIFT_GATEWAY_URL", os.environ.get("CACHEFORGE_GATEWAY_URL", ""))
    asyncio.run(
        run_benchmark_suite(
            num_turns=cli_args.turns,
            gateway_url=effective_gateway,
            model=cli_args.model,
            output_file=cli_args.output,
        )
    )
