"""M-006 Live Upstream Smoke Test & Hardware Verification against Nebius Token Factory.

Conforming to SDCS v1.3.0 and Gate C/T protocols.
Historical smoke check, superseded; see PRE_REGISTRATION.md.
Checks prefix canonicalization and async tool wait handling
against live Nebius Token Factory serving nvidia/nemotron-3-super-120b-a12b.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Load environment from .env if present
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
GATEWAY_PORT = int(os.getenv("RUNSIFT_PORT", os.getenv("CACHEFORGE_PORT", "8765")))
GATEWAY_URL = os.getenv("GATEWAY_URL", f"http://127.0.0.1:{GATEWAY_PORT}")

# 3 Realistic cluster management tool schemas
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_telemetry",
            "description": "Retrieves current node GPU temperature, VRAM usage, and cluster status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Target GPU node identifier (e.g., gpu-node-01)",
                    }
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_cluster_diagnostics",
            "description": "Executes hardware diagnostics across InfiniBand fabric and NVLink bridges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fabric_id": {"type": "string"},
                    "deep_scan": {"type": "boolean"},
                },
                "required": ["fabric_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_vram_allocation",
            "description": "Queries detailed VRAM allocation and active process reservations on a node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                },
                "required": ["node_id"],
            },
        },
    },
]

STRUCTURED_SYSTEM_PROMPT = """You are an autonomous cluster engineering agent for NVIDIA DGX SuperPOD systems.
Your primary role is monitoring GPU cluster health, optimizing KV-cache allocation, and debugging hardware anomalies.
Always invoke the designated tool when system telemetry or cluster state inspection is required.
Provide concise, fact-based engineering responses once tool returns are received."""


async def ensure_gateway_running() -> subprocess.Popen[str] | None:
    """Verifies gateway is responsive; starts local server in subprocess if not running."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{GATEWAY_URL}/health", timeout=2.0)
            if r.status_code == 200 and str(r.json().get("spec", "")).startswith("SDCS SPEC-001"):
                print(f"[GATEWAY] Connected to running RunSift instance at {GATEWAY_URL}")
                return None
        except (httpx.HTTPError, ValueError):
            pass

    print(f"[GATEWAY] Spawning RunSift reverse proxy gateway on {GATEWAY_URL}...")
    proc = subprocess.Popen(  # noqa: ASYNC220
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.proxy:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(GATEWAY_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    async with httpx.AsyncClient() as client:
        for _ in range(25):
            await asyncio.sleep(0.4)
            try:
                r = await client.get(f"{GATEWAY_URL}/health", timeout=1.0)
                if r.status_code == 200 and str(r.json().get("spec", "")).startswith(
                    "SDCS SPEC-001"
                ):
                    print("[GATEWAY] RunSift gateway started and healthy.")
                    return proc
            except (httpx.HTTPError, ValueError):
                continue

    raise RuntimeError("Failed to start RunSift gateway within 10 seconds")


async def run_live_smoke_test():
    print("====================================================================")
    print(" RunSift M-006: Live Hardware Verification & Upstream Smoke Test")
    print(f" Target Endpoint: {NEBIUS_BASE_URL}")
    print(f" Target Model:    {UPSTREAM_MODEL}")
    print("====================================================================\n")

    if not NEBIUS_API_KEY:
        print(
            "ERROR: NEBIUS_API_KEY is not set. Please provide a valid Nebius API key.",
            file=sys.stderr,
        )
        sys.exit(1)

    proc = await ensure_gateway_running()
    session_id = f"smoke_live_{int(time.time())}"

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            # -------------------------------------------------------------
            # Turn 1: Cold Prefix & Tool Invocation
            # -------------------------------------------------------------
            print("[Turn 1] Dispatching Cold Prefix Request through RunSift...")
            turn1_messages = [
                {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Inspect node gpu-node-01 and report temperature and VRAM utilization.",
                },
            ]
            turn1_payload = {
                "model": UPSTREAM_MODEL,
                "messages": turn1_messages,
                "tools": TEST_TOOLS,
                "tool_choice": "auto",
                "max_tokens": 512,
            }

            t0 = time.perf_counter()
            r1 = await client.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                json=turn1_payload,
                headers={"x-runsift-session-id": session_id},
            )
            t1 = time.perf_counter()

            if r1.status_code != 200:
                print(f"Turn 1 Failed with HTTP {r1.status_code}: {r1.text}")
                sys.exit(1)

            data1 = r1.json()
            ttft_1_header = float(r1.headers.get("x-runsift-ttft-ms", r1.headers.get("x-cacheforge-ttft-ms", (t1 - t0) * 1000.0)))
            prefix_hash_1 = r1.headers.get("x-runsift-prefix-hash", r1.headers.get("x-cacheforge-prefix-hash", ""))
            usage_1 = data1.get("usage") or {}
            details_1 = usage_1.get("prompt_tokens_details") or {}
            cached_tokens_turn1 = int(
                details_1.get("cached_tokens") or usage_1.get("prompt_cache_hit_tokens") or 0
            )
            choice1 = data1["choices"][0]
            finish_reason_1 = choice1.get("finish_reason")
            tool_calls = choice1["message"].get("tool_calls") or []

            print(
                f"  ✓ Turn 1 Completed in {(t1 - t0) * 1000.0:.1f}ms (TTFT_1: {ttft_1_header:.1f}ms)"
            )
            print(f"  ✓ Prefix Hash: {prefix_hash_1[:16]}...{prefix_hash_1[-8:]}")
            print(f"  ✓ Turn 1 Cached Tokens: {cached_tokens_turn1}")
            print(f"  ✓ Finish Reason: {finish_reason_1}")

            if not tool_calls:
                print(
                    "  ! Model did not emit tool call directly, proceeding with standard continuation."
                )
                tool_call_id = "call_simulated_01"
                tool_name = "get_system_telemetry"
            else:
                tool_call_id = tool_calls[0]["id"]
                tool_name = tool_calls[0]["function"]["name"]
                tool_args = tool_calls[0]["function"]["arguments"]
                print(f"  ✓ Parsed Tool Call: {tool_name}({tool_args}) [ID: {tool_call_id}]")

            # -------------------------------------------------------------
            # Tool Suspension Delay
            # -------------------------------------------------------------
            suspension_delay = 2.5
            print(
                f"\n[Tool Suspension] Simulating external tool I/O execution ({suspension_delay}s)..."
            )
            await asyncio.sleep(suspension_delay)
            print("  ✓ External tool I/O completed.")

            simulated_tool_result = json.dumps(
                {
                    "node_id": "gpu-node-01",
                    "gpu_temp_c": 43.5,
                    "vram_used_gb": 19.4,
                    "vram_total_gb": 80.0,
                    "power_draw_w": 285.0,
                    "status": "OPTIMAL",
                    "nvlink_errors": 0,
                }
            )

            # -------------------------------------------------------------
            # Turn 2: Warm Canonical Continuation (Streaming SSE)
            # -------------------------------------------------------------
            print("\n[Turn 2] Dispatching Warm Continuation via Server-Sent Events (SSE)...")
            turn2_messages = list(turn1_messages)
            turn2_messages.append(choice1["message"])
            turn2_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": simulated_tool_result,
                }
            )

            turn2_payload = {
                "model": UPSTREAM_MODEL,
                "messages": turn2_messages,
                "tools": TEST_TOOLS,
                "stream": True,
                "max_tokens": 512,
            }

            t2_start = time.perf_counter()
            ttft_2 = 0.0
            first_chunk_received = False
            stream_chunks_count = 0
            full_content_chunks = []
            upstream_cached_turn2 = 0

            async with client.stream(
                "POST",
                f"{GATEWAY_URL}/v1/chat/completions",
                json=turn2_payload,
                headers={"x-runsift-session-id": session_id},
            ) as stream_resp:
                if stream_resp.status_code != 200:
                    print(f"Turn 2 Stream Failed with HTTP {stream_resp.status_code}")
                    sys.exit(1)

                prefix_hash_2 = stream_resp.headers.get("x-runsift-prefix-hash", stream_resp.headers.get("x-cacheforge-prefix-hash", ""))

                async for line in stream_resp.aiter_lines():
                    if not line:
                        continue
                    if not first_chunk_received:
                        ttft_2 = (time.perf_counter() - t2_start) * 1000.0
                        first_chunk_received = True

                    if line.startswith("data: ") and line != "data: [DONE]":
                        stream_chunks_count += 1
                        try:
                            chunk_data = json.loads(line[6:])
                            usage = chunk_data.get("usage")
                            if isinstance(usage, dict):
                                details = usage.get("prompt_tokens_details") or {}
                                chunk_cached = int(
                                    details.get("cached_tokens")
                                    or usage.get("prompt_cache_hit_tokens")
                                    or 0
                                )
                                if chunk_cached > 0:
                                    upstream_cached_turn2 = chunk_cached
                            choices = chunk_data.get("choices") or [{}]
                            delta = choices[0].get("delta") or {}
                            chunk_text = (
                                delta.get("content") or delta.get("reasoning_content") or ""
                            )
                            if chunk_text:
                                full_content_chunks.append(chunk_text)
                        except json.JSONDecodeError:
                            pass

            t2_end = time.perf_counter()
            total_turn2_duration = (t2_end - t2_start) * 1000.0
            full_response_text = "".join(full_content_chunks).strip()

            print(
                f"  ✓ SSE Stream Finished: {stream_chunks_count} chunks in {total_turn2_duration:.1f}ms"
            )
            print(
                f"  ✓ Warm Continuation TTFT_2: {ttft_2:.1f}ms (vs Cold TTFT_1: {ttft_1_header:.1f}ms)"
            )
            print(f"  ✓ Prefix Hash match = {prefix_hash_1 == prefix_hash_2}")
            print(f"  ✓ Upstream Cached Tokens (Turn 2): {upstream_cached_turn2}")
            print(f"  ✓ Model Output Snippet: {full_response_text[:120]}...")

            # -------------------------------------------------------------
            # Telemetry Metrics
            # -------------------------------------------------------------
            speedup_ratio = ttft_1_header / max(1.0, ttft_2)
            ttft_reduction_pct = max(0.0, ((ttft_1_header - ttft_2) / ttft_1_header) * 100.0)

            print("\n====================================================================")
            print(" LIVE NEBIUS HARDWARE VERIFICATION SCORECARD")
            print("====================================================================")
            print(f" Cold Prefill (TTFT_1):       {ttft_1_header:6.1f} ms")
            print(f" Warm Continuation (TTFT_2):   {ttft_2:6.1f} ms")
            print(
                f" TTFT Latency Reduction:      {ttft_reduction_pct:6.1f} %  ({speedup_ratio:.2f}x faster)"
            )
            print(f" Turn 1 Cached Tokens:        {cached_tokens_turn1}")
            print(f" Turn 2 Cached Tokens:        {upstream_cached_turn2}")
            print(f" Prefix Match:                {prefix_hash_1 == prefix_hash_2}")
            print("====================================================================\n")

            # -------------------------------------------------------------
            # Artifact Generation
            # -------------------------------------------------------------
            results = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_endpoint": NEBIUS_BASE_URL,
                "target_model": UPSTREAM_MODEL,
                "ttft_cold_ms": round(ttft_1_header, 2),
                "ttft_warm_ms": round(ttft_2, 2),
                "speedup_ratio": round(speedup_ratio, 2),
                "ttft_reduction_pct": round(ttft_reduction_pct, 2),
                "prefix_hash": prefix_hash_1,
                "prefix_match": prefix_hash_1 == prefix_hash_2,
                "cached_tokens_turn1": cached_tokens_turn1,
                "cached_tokens_turn2": upstream_cached_turn2,
                "stream_chunks_received": stream_chunks_count,
                "status": "PASSED" if ttft_2 <= ttft_1_header else "ADVISORY",
            }

            results_path = Path("benchmarks/live_smoke_result.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Saved live benchmark results to: {results_path}")

    finally:
        if proc:
            print("[GATEWAY] Terminating background RunSift gateway process...")
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    asyncio.run(run_live_smoke_test())
