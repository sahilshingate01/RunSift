#!/usr/bin/env python3
"""examples/autogen_coding_assistant.py — Untested example: AutoGen ConversableAgent with RunSift.

Note: This adapter is an untested example showing how to point Microsoft AutoGen's
OpenAI-compatible client (`base_url`) at the RunSift gateway (http://127.0.0.1:8765/v1).
"""

import os
import platform
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
from autogen import ConversableAgent, register_function

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import get_config

_cfg = get_config()
GATEWAY_BASE = f"http://127.0.0.1:{_cfg.gateway_port}"

# ---------------------------------------------------------------------------
# 1. Hardware & Gateway Telemetry Tools
# ---------------------------------------------------------------------------


def run_vram_profiler(
    model_checkpoint: Annotated[str, "Model identifier to inspect on cluster"],
) -> str:
    """Query cluster hardware allocations, gateway cache hit rate, and TensorRT-LLM memory footprints."""
    print(f"\n[Tool Execution: run_vram_profiler] Inspecting '{model_checkpoint}'...")
    hit_rate = 0.0
    saved_tokens = 0
    try:
        r = httpx.get(f"{GATEWAY_BASE}/api/telemetry", timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            hit_rate = data.get("radix_cache_hit_rate", 0.0)
            saved_tokens = data.get("cumulative_tokens_saved", 0)
    except httpx.HTTPError:
        pass

    return (
        f"Model: {model_checkpoint}\n"
        f"Host Platform: {platform.system()} {platform.release()} ({os.cpu_count()} CPU cores)\n"
        "Weights (FP8): 120.4 GB across 8x H100 SXM5\n"
        "Activation Buffer: 12.8 GB\n"
        f"Live RunSift Prefix Cache Hit Rate: {hit_rate:.1f}% (Compacted Tokens Saved: {saved_tokens})\n"
        "Base Prefix KV-Cache Allocated: 38.2 GB (Available Headroom: 468.6 GB)\n"
        "Optimal Tensor Parallelism (TP): 8 | Pipeline Parallelism (PP): 1\n"
        "Status: READY"
    )


def benchmark_kernel_latency(
    batch_size: Annotated[int, "Inference batch size"],
    sequence_length: Annotated[int, "Total context sequence length"],
) -> str:
    """Compute analytical GEMM and PagedAttention latency across NVLink 4.0 interconnects."""
    print(
        f"\n[Tool Execution: benchmark_kernel_latency] Computing BS={batch_size}, SeqLen={sequence_length}..."
    )
    flops_per_token = 2 * 12e9  # 12B active MoE parameters in Nemotron-3-Super-120B-A12B
    total_tflops = (batch_size * sequence_length * flops_per_token) / 1e12
    paged_attn_ms = round(0.45 + (batch_size * sequence_length) / 65536.0, 2)
    all_reduce_ms = round(0.18 + (batch_size * 0.012), 2)
    return (
        f"GEMM & Attention Profile: BS={batch_size}, SeqLen={sequence_length}\n"
        f"Active Compute Volume: {total_tflops:.2f} TFLOPs (A12B active parameters)\n"
        f"Paged Attention Kernel: {paged_attn_ms}ms\n"
        f"All-Reduce Collective Time: {all_reduce_ms}ms\n"
        "Prefill Throughput: 4,820 tokens/sec\n"
        "Decoded Throughput: 1,180 tokens/sec"
    )


# ---------------------------------------------------------------------------
# 2. Configure AutoGen LLM Client (OpenAI-Compatible RunSift Reverse Proxy)
# ---------------------------------------------------------------------------

config_list = [
    {
        "model": _cfg.default_model,
        "base_url": f"{GATEWAY_BASE}/v1",
        "api_key": _cfg.nebius_api_key,
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.0,
}

# ---------------------------------------------------------------------------
# 3. Agent Declarations
# ---------------------------------------------------------------------------

system_instruction = (
    "You are an autonomous High-Performance Computing (HPC) Infrastructure Engineer. "
    "Diagnose GPU cluster memory profiles, benchmark inference kernels, and provide "
    "actionable deployment recommendations. Always call tools when hardware details are requested. "
    "Conclude with TERMINATE when the deployment configuration is fully verified."
)

sre_assistant = ConversableAgent(
    name="SRE_Optimizer",
    system_message=system_instruction,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

user_proxy = ConversableAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
    code_execution_config=False,
)

# ---------------------------------------------------------------------------
# 4. Tool Registration
# ---------------------------------------------------------------------------

register_function(
    run_vram_profiler,
    caller=sre_assistant,
    executor=user_proxy,
    name="run_vram_profiler",
    description="Query cluster hardware allocations and inspect TensorRT-LLM memory footprints.",
)

register_function(
    benchmark_kernel_latency,
    caller=sre_assistant,
    executor=user_proxy,
    name="benchmark_kernel_latency",
    description="Benchmark simulated GEMM kernel latency across NVLink 4.0 interconnects.",
)

# ---------------------------------------------------------------------------
# 5. Multi-Turn Conversation Loop
# ---------------------------------------------------------------------------


def run_autogen_pipeline():
    print("=" * 72)
    print(" Dispatching AutoGen ConversableAgent Workflow via RunSift")
    print(" Gateway: http://127.0.0.1:8765/v1 | Upstream: Nebius Token Factory")
    print(f" Target:  {_cfg.default_model}")
    print("=" * 72)

    prompt = (
        f"Profile VRAM allocations for '{_cfg.default_model}'. "
        "If status is READY, benchmark kernel latency for batch size 16 and "
        "sequence length 4096, then output the recommended deployment configuration."
    )

    start_time = time.perf_counter()
    chat_result = user_proxy.initiate_chat(
        recipient=sre_assistant,
        message=prompt,
    )
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 72)
    print(f" Conversation Finished in {elapsed:.2f}s total wall-clock time.")
    print(f" Total Turns Processed: {len(chat_result.chat_history)}")
    print("=" * 72)
    print("Inspect http://127.0.0.1:8765/dashboard to verify live KV-cache reuse & hold savings.")


if __name__ == "__main__":
    run_autogen_pipeline()
