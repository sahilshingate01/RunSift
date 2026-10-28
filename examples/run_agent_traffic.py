"""RunSift Demo Agent Adapter — Drives Live Multi-Turn Traffic into Dashboard.

Usage:
  1. Ensure RunSift gateway is running:
       uvicorn src.proxy:app --port 8765
  2. Open the real-time telemetry dashboard in your browser:
       http://localhost:8765/dashboard
  3. In another terminal, run this script:
       python examples/run_agent_traffic.py
"""

import json
import time

import httpx

GATEWAY_URL = "http://localhost:8765/v1"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SESSION_ID = f"sess_agent_{int(time.time())}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_gpu_cluster_telemetry",
            "description": "Returns current VRAM utilization, temperature, and active compute jobs for a GPU node.",
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
            "name": "reschedule_batch_slot",
            "description": "Evicts idle hold sockets and reschedules pending inference batches.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an Autonomous Site Reliability Agent deployed across Nebius GPU clusters.
Monitor node health, invoke diagnostic tools, and optimize inference throughput."""


def print_banner():
    print("=" * 70)
    print(" 🔥 RunSift Live Agent Traffic Generator")
    print(f" Target Gateway:   {GATEWAY_URL}")
    print(f" Target Model:     {MODEL}")
    print(f" Agent Session ID: {SESSION_ID}")
    print(" Live Dashboard:   http://localhost:8765/dashboard")
    print("=" * 70 + "\n")


def execute_diagnostic_tool(tool_name: str, raw_args: str) -> str:
    """Formats a verbose diagnostic report for the demo tool call.

    Only the platform, CPU count and the optional nvidia-smi probe are read from this host.
    The VRAM, temperature and cache-status fields and the [CHECK] lines are fixed sample values.
    """
    import os
    import platform
    import shutil
    import subprocess

    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        args = {}
    node_id = args.get("node_id") or args.get("job_id") or "gpu-node-08"

    gpu_info = "nvidia-smi not present on this host."
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpu_info = proc.stdout.strip()
        except OSError:
            pass

    header = {
        "node_id": node_id,
        "tool_executed": tool_name,
        "client_platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_cores": os.cpu_count() or 1,
        "gpu_probe": gpu_info,
        "sample_values": ["vram_util_pct", "temp_c", "prefix_cache_status", "CHECK lines"],
        "vram_util_pct": 82.4,
        "temp_c": 58.2,
        "prefix_cache_status": "ONLINE_OPTIMAL",
    }
    # Include detailed diagnostic verification lines (>9,500 chars) so Condenser v2 (8K threshold) is exercised
    verification_lines = [
        f"[CHECK {i:03d}] PCIe Gen5 / NVLink 4.0 lane integrity check on {node_id} ... PASSED (0 CRC errors, 900 GB/s)"
        for i in range(1, 96)
    ]
    return (
        f"TELEMETRY_SUMMARY: {json.dumps(header)}\n"
        + "\n".join(verification_lines)
        + f"\nFINAL_STATUS: Node {node_id} healthy; KV-cache prefix blocks aligned."
    )


def cache_status(usage: dict | None) -> tuple[int | None, str]:
    """Reads cached prompt tokens the way the gateway does; a hit needs upstream cached_tokens > 0."""
    if not isinstance(usage, dict):
        return None, "⚪ CACHE UNKNOWN (no usage reported)"
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
    if cached > 0:
        return cached, f"🟢 CACHE HIT ({cached} cached prompt tokens, upstream-reported)"
    return cached, "🔴 CACHE MISS (upstream reported 0 cached tokens)"


def run_turn(
    client: httpx.Client,
    turn_name: str,
    messages: list[dict],
    stream: bool = False,
):
    print(f"[{turn_name}] Dispatching through RunSift gateway...")
    t0 = time.perf_counter()
    if not stream:
        resp = client.post(
            f"{GATEWAY_URL}/chat/completions",
            headers={"x-runsift-session-id": SESSION_ID},
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 256,
            },
            timeout=60.0,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            print(f"  ❌ Request failed ({resp.status_code}): {resp.text}")
            return None, None

        prefix_hash = resp.headers.get("x-runsift-prefix-hash", resp.headers.get("x-cacheforge-prefix-hash", "N/A"))
        data = resp.json()
        cached, status_str = cache_status(data.get("usage"))
        print(f"  ✓ Full response in {duration_ms:.1f}ms (non-streaming) | {status_str}")
        print(f"  ✓ Prefix Hash: {prefix_hash[:20]}...{prefix_hash[-8:]}")
        return data, cached

    # Streaming SSE turn; usage is requested so the upstream cached-token count comes back
    first_chunk_ms = 0.0
    first_chunk = False
    usage: dict | None = None
    content_parts: list[str] = []
    with client.stream(
        "POST",
        f"{GATEWAY_URL}/chat/completions",
        headers={"x-runsift-session-id": SESSION_ID},
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 256,
        },
        timeout=60.0,
    ) as stream_resp:
        if stream_resp.status_code != 200:
            print(f"  ❌ Stream failed ({stream_resp.status_code})")
            return None, None
        prefix_hash = stream_resp.headers.get("x-runsift-prefix-hash", stream_resp.headers.get("x-cacheforge-prefix-hash", "N/A"))
        chars_before = stream_resp.headers.get("x-runsift-tool-chars-before", stream_resp.headers.get("x-cacheforge-tool-chars-before", "0"))
        chars_after = stream_resp.headers.get("x-runsift-tool-chars-after", stream_resp.headers.get("x-cacheforge-tool-chars-after", "0"))
        for line in stream_resp.iter_lines():
            if not line:
                continue
            if line.startswith("data: ") and line != "data: [DONE]":
                if not first_chunk:
                    first_chunk_ms = (time.perf_counter() - t0) * 1000.0
                    first_chunk = True
                try:
                    chunk = json.loads(line[6:])
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        content_parts.append(text)
                except json.JSONDecodeError:
                    pass
    duration_ms = (time.perf_counter() - t0) * 1000.0
    cached, status_str = cache_status(usage)
    print(
        f"  ✓ SSE stream finished in {duration_ms:.1f}ms (first chunk at {first_chunk_ms:.1f}ms) | {status_str}"
    )
    print(f"  ✓ Tool output: {chars_before} chars raw -> {chars_after} chars sent to the model")
    print(f"  ✓ Prefix Hash: {prefix_hash[:20]}...{prefix_hash[-8:]}")
    return {
        "choices": [{"message": {"role": "assistant", "content": "".join(content_parts)}}]
    }, cached


def main():
    print_banner()

    with httpx.Client() as client:
        # Check gateway health
        try:
            health = client.get("http://localhost:8765/health", timeout=3.0).json()
            print(f"  ✓ Gateway Healthy | Spec: {health.get('spec')}")
            print(f"  ✓ Upstream Model:  {health.get('default_model')}\n")
        except (httpx.HTTPError, OSError) as e:
            print(f"❌ Could not connect to RunSift gateway at http://localhost:8765: {e}")
            print("Please ensure the gateway is running: uvicorn src.proxy:app --port 8765")
            return

        # Turn 1: Cold Prefill & Tool Call Generation
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Inspect node gpu-node-08 using get_gpu_cluster_telemetry and report status.",
            },
        ]
        res1, cached1 = run_turn(client, "Turn 1", messages, stream=False)
        if not res1:
            return

        assistant_msg = res1["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        if tool_calls:
            tc = tool_calls[0]
            tool_call_id = tc["id"]
            tool_name = tc["function"]["name"]
            tool_args = tc["function"].get("arguments", "{}")
            print(f"\n[Tool Call] Model requested {tool_name}({tool_args}) [id={tool_call_id}]")
            print("  ✓ Running the tool locally (sample diagnostic data)...")
            t_tool = time.perf_counter()
            tool_output = execute_diagnostic_tool(tool_name, tool_args)
            tool_elapsed = time.perf_counter() - t_tool
            print(f"  ✓ Tool completed in {tool_elapsed:.2f}s ({len(tool_output)} bytes captured)")

            messages.append(assistant_msg)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_output,
                }
            )
        else:
            messages.append(assistant_msg)
            messages.append(
                {
                    "role": "user",
                    "content": "Summarize the node health and confirm KV-cache readiness.",
                }
            )

        # Turn 2: continuation with the tool result (streaming SSE)
        res2, cached2 = run_turn(client, "Turn 2 (Streaming)", messages, stream=True)

        if res2:
            # Turns 1 and 2 are timed differently (full response vs first chunk), so no speedup ratio is printed
            shown = ["not reported" if c is None else c for c in (cached1, cached2)]
            print(
                f"\n📊 Upstream-reported cached prompt tokens: turn 1 = {shown[0]}, turn 2 = {shown[1]}"
            )
            print(
                "👉 Check the live update on your dashboard at: http://localhost:8765/dashboard\n"
            )


if __name__ == "__main__":
    main()
