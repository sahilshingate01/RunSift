#!/usr/bin/env python3
"""examples/crewai_researcher.py — Untested example: Multi-Agent CrewAI with RunSift.

Note: This adapter is an untested example showing how to point CrewAI's OpenAI-compatible
LLM configuration (`base_url`) at the RunSift gateway (http://127.0.0.1:8765/v1).
"""

import os
import platform
import sys
import time
from pathlib import Path

import httpx
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import get_config

_cfg = get_config()
GATEWAY_BASE = f"http://127.0.0.1:{_cfg.gateway_port}"

# ---------------------------------------------------------------------------
# 1. Cluster Topology & Kernel Memory Profiling Tools
# ---------------------------------------------------------------------------


@tool("query_cluster_topology")
def query_cluster_topology(cluster_name: str) -> str:
    """Retrieve NVLink mesh topology, GPU interconnect bandwidth, and live RunSift gateway status."""
    print(f"\n[Tool I/O: query_cluster_topology] Querying topology for '{cluster_name}'...")
    gateway_status = "UNREACHABLE"
    try:
        r = httpx.get(f"{GATEWAY_BASE}/health", timeout=2.0)
        if r.status_code == 200:
            gateway_status = r.json().get("status", "healthy").upper()
    except httpx.HTTPError:
        pass

    return (
        f"Cluster: {cluster_name} (Gateway Status: {gateway_status})\n"
        f"Client Host: {platform.system()} {platform.release()} ({os.cpu_count()} cores)\n"
        "Architecture: 8x NVIDIA H100 SXM5 (80GB HBM3)\n"
        "Interconnect: 900 GB/s bidirectional NVLink 4.0\n"
        "Network Fabric: 400 Gb/s Quantum-2 NDR InfiniBand per node\n"
        "Topology: Full mesh NVSwitch; Zero cross-NUMA contention observed."
    )


@tool("validate_kernel_memory_profile")
def validate_kernel_memory_profile(layer_name: str, tensor_parallel_size: int) -> str:
    """Compute memory allocation footprints and activation memory under Tensor Parallelism."""
    print(
        f"\n[Tool I/O: validate_kernel_memory_profile] Computing profile on '{layer_name}' (TP={tensor_parallel_size})..."
    )
    tp = max(1, int(tensor_parallel_size))
    weight_gb_per_gpu = round(120.0 / tp, 2)
    activation_gb_per_gpu = round(38.4 / tp, 2)
    kv_cache_gb = 12.6
    headroom_gb = round(max(0.0, 80.0 - weight_gb_per_gpu - activation_gb_per_gpu - kv_cache_gb), 2)
    return (
        f"Layer: {layer_name} | TP: {tp}\n"
        f"Weight Memory: {weight_gb_per_gpu} GB / GPU\n"
        f"Activation Memory: {activation_gb_per_gpu} GB / GPU\n"
        f"KV-Cache Overhead: {kv_cache_gb} GB @ 8k context\n"
        f"Total Headroom Remaining: {headroom_gb} GB HBM3 available."
    )


# ---------------------------------------------------------------------------
# 2. Configure OpenAI-Compatible LLM Gateway
# ---------------------------------------------------------------------------

# Point directly at the local RunSift proxy gateway
runsift_llm = LLM(
    model=f"openai/{_cfg.default_model}",
    base_url=f"{GATEWAY_BASE}/v1",
    api_key=_cfg.nebius_api_key,
    temperature=0.0,
)
cacheforge_llm = runsift_llm

# ---------------------------------------------------------------------------
# 3. Agent Definitions
# ---------------------------------------------------------------------------

systems_researcher = Agent(
    role="Principal GPU Systems Researcher",
    goal="Profile cluster architecture bottlenecks and determine optimal communication topology.",
    backstory=(
        "You are an expert in high-performance computing clusters and NVIDIA DGX architecture. "
        "You inspect hardware fabric parameters and formulate memory distribution blueprints."
    ),
    tools=[query_cluster_topology],
    llm=runsift_llm,
    verbose=True,
)

cuda_architect = Agent(
    role="Lead Distributed Kernel Architect",
    goal="Translate systems architecture constraints into concrete Tensor Parallelism configurations.",
    backstory=(
        "You specialize in TensorRT-LLM and Megatron-LM memory layouts. You validate activation "
        "footprints and deliver concrete, production-ready inference deployment configs."
    ),
    tools=[validate_kernel_memory_profile],
    llm=runsift_llm,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 4. Task Specifications
# ---------------------------------------------------------------------------

task_research = Task(
    description=(
        "Query the hardware topology for cluster 'dgx-superpod-east'. "
        "Analyze interconnect bandwidth limits and specify the recommended Tensor Parallel (TP) "
        "and Pipeline Parallel (PP) split for serving a 120B parameter dense model."
    ),
    expected_output=(
        "A structured assessment of cluster interconnect capabilities with explicit recommendations "
        "for TP and PP degrees."
    ),
    agent=systems_researcher,
)

task_implementation = Task(
    description=(
        "Review the researcher's topology findings. Validate the memory profile for "
        "layer 'transformer.decoder.block.0' using the recommended TP configuration. "
        "Provide a concise JSON deployment configuration for TensorRT-LLM specifying: "
        "tp_size, pp_size, kv_cache_free_gpu_mem_fraction, and max_batch_size."
    ),
    expected_output=(
        "Validated memory profile summary followed by a valid JSON deployment block for TensorRT-LLM."
    ),
    agent=cuda_architect,
)

# ---------------------------------------------------------------------------
# 5. Execution Pipeline
# ---------------------------------------------------------------------------


def run_crew_pipeline():
    print("=" * 72)
    print(" Dispatching Multi-Agent CrewAI Workflow via RunSift")
    print(" Gateway: http://127.0.0.1:8765/v1 | Upstream: Nebius Token Factory")
    print(f" Model:   {_cfg.default_model}")
    print("=" * 72)

    crew = Crew(
        agents=[systems_researcher, cuda_architect],
        tasks=[task_research, task_implementation],
        process=Process.sequential,
        verbose=True,
    )

    wall_clock_start = time.perf_counter()
    result = crew.kickoff()
    elapsed = time.perf_counter() - wall_clock_start

    print("\n" + "=" * 72)
    print(f" Workflow Execution Completed in {elapsed:.2f}s total wall-clock time.")
    print(" Final Deliverable:")
    print("=" * 72)
    print(result)
    print(
        "\nCheck http://127.0.0.1:8765/dashboard to inspect prefix cache reuse & session telemetry."
    )


if __name__ == "__main__":
    run_crew_pipeline()
