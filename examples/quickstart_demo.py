"""RunSift 60-Second Standalone Quickstart Demo.

Run this script to experience RunSift's wire-level optimizations in under 60 seconds:
    python examples/quickstart_demo.py

Features demonstrated:
  1. Deterministic Prefix Canonicalization (tool sorting & volatile header hoisting)
  2. Async Connection Decoupling (no GPU connection hold during tool I/O)
  3. Negative Memory Compaction (synthesizes verbose diagnostic traces into <300 tokens)
  4. Developer Delight HTTP Response Headers (X-RunSift-KV-Hit, TTFT speedup, avoided prefill)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from httpx import ASGITransport, AsyncClient

from src.core import get_config
from src.proxy.gateway import create_app

# ANSI Color Codes for terminal delight
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def run_demo() -> None:
    print(
        f"\n{BOLD}{CYAN}======================================================================{RESET}"
    )
    print(f"{BOLD}{GREEN} 🔥 RunSift: 60-Second Autonomous Agent Wire Demo{RESET}")
    print(" Target Infrastructure: Nebius Token Factory (NVIDIA Open Models)")
    print(" Target Architecture:   Provider Prefix Cache & Async Connection Decoupling")
    print(
        f"{BOLD}{CYAN}======================================================================{RESET}\n"
    )

    cfg = get_config()
    # Instantiate in-process gateway (self-contained, no external ports required)
    app = create_app(config=cfg)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://runsift-local"
    ) as client:
        # Check health
        health_resp = await client.get("/health")
        health_data = health_resp.json()
        print(
            f"[{GREEN}OK{RESET}] RunSift Gateway Online (v{health_data['version']}, {health_data['spec']})"
        )
        print(
            f"[{GREEN}OK{RESET}] Default Upstream Model: {BOLD}{health_data['default_model']}{RESET}\n"
        )

        session_id = f"sess_demo_{int(time.time())}"

        # -----------------------------------------------------------------
        # TURN 1: Cold Prefill with Volatile Header & Tools
        # -----------------------------------------------------------------
        print(f"{BOLD}{YELLOW}▶ [TURN 1: Cold Agent Dispatch]{RESET}")
        print(
            f"  • Injecting dynamic microsecond timestamp: {time.time()} (breaks uncanonicalized prefix matching)"
        )
        print("  • Supplying 2 tool definitions in arbitrary order")

        turn1_payload = {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Current Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                        f"Session UUID: {session_id}\n"
                        "You are a Principal GPU Systems Reliability Agent on Nebius Token Factory. "
                        "Monitor NVLink error rates, inspect cluster state, and optimize throughput."
                    ),
                },
                {
                    "role": "user",
                    "content": "Perform pre-flight diagnostics on cluster node gpu-node-01.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_gpu_cluster_telemetry",
                        "description": "Inspects GPU VRAM utilization, temperature, and NVLink fabric integrity.",
                        "parameters": {
                            "type": "object",
                            "properties": {"node_id": {"type": "string"}},
                            "required": ["node_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "audit_kv_cache_pages",
                        "description": "Audits prefix cache page allocations.",
                        "parameters": {
                            "type": "object",
                            "properties": {"pool_id": {"type": "string"}},
                            "required": ["pool_id"],
                        },
                    },
                },
            ],
        }

        t0 = time.perf_counter()
        resp1 = await client.post(
            "/v1/chat/completions",
            json=turn1_payload,
            headers={"x-runsift-session-id": session_id},
        )
        t1_elapsed = (time.perf_counter() - t0) * 1000

        if resp1.status_code != 200:
            print(
                f"  [{YELLOW}FAIL{RESET}] Upstream inference endpoint unavailable (HTTP {resp1.status_code})"
            )
            print(f"    Detail: {resp1.text}")
            print(
                f"\n[{YELLOW}NOTICE{RESET}] Set a valid NEBIUS_API_KEY in your .env file or environment to dispatch live upstream completions."
            )
            return

        print(f"  {GREEN}✓ Turn 1 Dispatched & Canonicalized ({t1_elapsed:.1f}ms roundtrip){RESET}")
        prefix_hash = resp1.headers.get("x-runsift-prefix-hash", resp1.headers.get("x-cacheforge-prefix-hash", ""))
        if prefix_hash:
            print(f"    {MAGENTA}X-RunSift-Prefix-Hash:{RESET}            {prefix_hash[:24]}...")
        kv_hit1 = resp1.headers.get("x-runsift-kv-hit", resp1.headers.get("x-cacheforge-kv-hit"))
        if kv_hit1 is not None:
            print(f"    {MAGENTA}X-RunSift-KV-Hit:{RESET}                 {kv_hit1}")
        ttft1 = resp1.headers.get("x-runsift-ttft-ms", resp1.headers.get("x-cacheforge-ttft-ms"))
        if ttft1 is not None:
            print(
                f"    {MAGENTA}X-RunSift-TTFT-Ms:{RESET}                {ttft1} ms (Cold Prefill)"
            )

        # -----------------------------------------------------------------
        # TOOL SUSPENSION: Async Connection Decoupling
        # -----------------------------------------------------------------
        print(f"\n{BOLD}{YELLOW}▶ [EXTERNAL TOOL EXECUTION: Async Connection Decoupling]{RESET}")
        print(
            "  • Agent invokes external diagnostic tool: get_gpu_cluster_telemetry(node_id='gpu-node-01')"
        )
        print(
            f"  • {GREEN}RunSift terminates active GPU connection during tool wait (zero idle connection hold!){RESET}"
        )

        # Generate a verbose 1,500+ character hardware diagnostic trace
        verbose_tool_output = (
            "PCIe Gen5 / NVLink 4.0 Subsystem Diagnostics:\n"
            + "\n".join(
                [
                    f"  Lane {i:02d}: CRC OK, 900 GB/s bandwidth, 0 dropped frames"
                    for i in range(1, 25)
                ]
            )
            + "\nThermal Sensors: GPU 0: 42C, GPU 1: 44C, GPU 2: 41C, GPU 3: 43C. All healthy."
        )
        raw_char_len = len(verbose_tool_output)
        print(
            f"  • Tool returns raw diagnostic dump ({raw_char_len} chars, ~{raw_char_len // 4} tokens)"
        )

        # -----------------------------------------------------------------
        # TURN 2: Warm Continuation & Negative Memory Compaction
        # -----------------------------------------------------------------
        print(f"\n{BOLD}{YELLOW}▶ [TURN 2: Cache-Aligned Warm Continuation]{RESET}")

        turn2_payload = {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "messages": [
                turn1_payload["messages"][0],
                turn1_payload["messages"][1],
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_diag_01",
                            "type": "function",
                            "function": {
                                "name": "get_gpu_cluster_telemetry",
                                "arguments": '{"node_id": "gpu-node-01"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_diag_01",
                    "name": "get_gpu_cluster_telemetry",
                    "content": verbose_tool_output,
                },
                {
                    "role": "user",
                    "content": "Summarize node health in 1 sentence.",
                },
            ],
            "tools": turn1_payload["tools"],
        }

        t1 = time.perf_counter()
        resp2 = await client.post(
            "/v1/chat/completions",
            json=turn2_payload,
            headers={"x-runsift-session-id": session_id},
        )
        t2_elapsed = (time.perf_counter() - t1) * 1000

        if resp2.status_code != 200:
            print(
                f"  [{YELLOW}FAIL{RESET}] Upstream inference endpoint unavailable on Turn 2 (HTTP {resp2.status_code})"
            )
            print(f"    Detail: {resp2.text}")
            return

        print(f"  {GREEN}✓ Turn 2 Dispatched & Synthesized ({t2_elapsed:.1f}ms roundtrip){RESET}")
        prefix_hash2 = resp2.headers.get("x-runsift-prefix-hash", resp2.headers.get("x-cacheforge-prefix-hash", ""))
        if prefix_hash2:
            print(
                f"    {MAGENTA}X-RunSift-Prefix-Hash:{RESET}            {prefix_hash2[:24]}..."
            )
        kv_hit2 = resp2.headers.get("x-runsift-kv-hit", resp2.headers.get("x-cacheforge-kv-hit"))
        if kv_hit2 is not None:
            kv_hit = kv_hit2.lower() == "true"
            kv_tag = (
                f"{GREEN}(CACHE HIT!){RESET}"
                if kv_hit
                else "(Prefix matched; pending upstream cache warmup)"
            )
            print(
                f"    {BOLD}{GREEN}X-RunSift-KV-Hit:{RESET}                 {kv_hit2} {kv_tag}"
            )
        speedup = resp2.headers.get("x-runsift-ttft-speedup", resp2.headers.get("x-cacheforge-ttft-speedup"))
        if speedup is not None:
            print(f"    {BOLD}{GREEN}X-RunSift-TTFT-Speedup:{RESET}           {speedup}")
        prefill_avoided = resp2.headers.get("x-runsift-prefill-tokens-avoided", resp2.headers.get("x-cacheforge-prefill-tokens-avoided"))
        if prefill_avoided is not None:
            print(
                f"    {BOLD}{GREEN}X-RunSift-Prefill-Avoided:{RESET}        {prefill_avoided} tokens"
            )
        tokens_compacted = resp2.headers.get("x-runsift-tokens-compacted", resp2.headers.get("x-cacheforge-tokens-compacted"))
        if tokens_compacted is not None:
            print(
                f"    {BOLD}{GREEN}X-RunSift-Tokens-Compacted:{RESET}       {tokens_compacted} tokens (Negative Memory)"
            )

        # -----------------------------------------------------------------
        # EMPIRICAL BENCHMARK HEADLINE (from README.md / PRE_REGISTRATION.md)
        # -----------------------------------------------------------------
        print(
            f"\n{BOLD}{CYAN}======================================================================{RESET}"
        )
        print(f"{BOLD}{GREEN} 📊 Pre-Registered Token Factory Headline (7 Comparisons){RESET}")
        print(
            f"{BOLD}{CYAN}======================================================================{RESET}"
        )
        print("  In every Token Factory run, the same budget covered 2.2–4.6× as many")
        print("  agent sessions. Cost per session fell 55–78%. The median time to first")
        print("  token fell 17–33%. See README.md and")
        print("  benchmarks/tensorroute_benchmark/PRE_REGISTRATION.md for details.")
        print(
            f"{BOLD}{CYAN}======================================================================{RESET}\n"
        )
        print(f"👉 Open live dashboard: {BOLD}http://localhost:8765/dashboard{RESET}")
        print(
            f"👉 Run full benchmark:  {BOLD}python -u benchmarks/tensorroute_benchmark/harness.py --mode gauntlet{RESET}\n"
        )


if __name__ == "__main__":
    asyncio.run(run_demo())
