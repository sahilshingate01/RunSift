# TensorRoute-Bench v2: 100% Original Autonomous Systems Bug-Fix Benchmark

`TensorRoute-Bench` v2 is a 100% original, deterministic multi-turn coding agent benchmark built around **`tensorroute`** — a self-contained, zero-external-dependency Python systems library (~500 LOC across 5 leaf modules + 45 unit tests) implementing:
1. **`radix.py`**: Prefix Radix Tree for GPU KV-Cache Block Routing & TTL/LRU Eviction (`9` tests)
2. **`scheduler.py`**: Continuous Batching Request Scheduler with Priority Preemption (`9` tests)
3. **`limiter.py`**: Token-Bucket Rate Limiter with Fractional Sub-Second Refill (`9` tests)
4. **`codec.py`**: SSE Chunk Parser & Multi-Chunk Tool-Call Delta Assembler (`9` tests)
5. **`telemetry.py`**: Rolling Latency Window & Nearest-Rank Percentile Aggregator (`9` tests)

## Two Execution Geometries

1. **Continuous 5-Bug Multi-Module Gauntlet (`--mode gauntlet`, 16–20 Turns in 1 Continuous Session):**
   - Injects all 5 Wave-1 bugs (`OB-01`..`OB-05`, 1 bug per module, `12 failed / 33 passed`) into the **same workspace simultaneously**.
   - Forces the agent to diagnose and repair all 5 modules sequentially in **one continuous multi-turn conversation without resetting `history`**, directly testing the **Turn-15 Context Collapse** boundary.
2. **15-Task Isolated Bug-Fix Suite (`--mode suite`, `OB-01` through `OB-15`, 3 Bugs per Subsystem):**
   - Evaluates `baseline` vs `runsift` across $N=15$ mutually disjoint systems bugs (`~60+` agent turns per arm).

## Scientific Validity Guarantees
- **100% Authentic Pytest Output:** Runs unmodified `pytest -o pythonpath=src -vv -ra --tb=long --durations=0` with zero synthetic log padding.
- **SDCS Pillar 4/5/6 Cognitive Hydration:** The `runsift` arm hydrates `app_map.md` (Pillar 5 Cartography) and `decisions.md` (Pillar 6 Negative Memory ADRs) into the canonicalized static prefix and appends a deterministic `[SDCS Blackboard State]` module status bar (`passing_modules`, `failing_modules`, `next_target`) to compacted pytest digests.
- **Dual Cache & Percentile Telemetry:** Tracks both SHA-256 prefix invariance hit rate and upstream hardware-reported `cached_tokens` (`timings.cache_n` on `llama.cpp` / `prompt_tokens_details.cached_tokens` on Nebius Token Factory), `uncached_prefill_tokens`, mean/p50/p95 SSE wire TTFT, and `peak_turn_prompt_tokens`.

---

## Historical Single-Session Gauntlet Tables (Superseded; see [`PRE_REGISTRATION.md`](./PRE_REGISTRATION.md) and [`results/superseded/README.md`](./results/superseded/README.md))

> **Note:** The tables below record early single-session (`N=1`) runs conducted prior to pre-registered multi-repeat (`N=15`) evaluation and gateway-side condensation (Protocol v3). They are retained for historical reference; see [`PRE_REGISTRATION.md`](./PRE_REGISTRATION.md) and the root [`README.md`](../../README.md) for current pre-registered multi-repeat evaluations.

### 1. Local GPU (`Qwen2.5-Coder-7B-Instruct-Q8_0.gguf` on RTX 3080 Ti, `32k` Context Window)
Artifact: [`results/superseded/local_gauntlet_qwen25_7b.json`](./results/superseded/local_gauntlet_qwen25_7b.json)

| Metric | Unoptimized Baseline | RunSift + SDCS | Delta |
| :--- | :---: | :---: | :---: |
| **Modules Repaired (`45` Tests)** | `0/5` (`34/45`, **Crashed Turn 7: HTTP 400 Overflow**) | **`5/5` (`45/45` Solved in 16 Turns)** | **`+5/5` Modules (`+100%`)** |
| **KV-Cache Hit Rate** | `0.0%` (`0` cached tok) | **`93.75%`** (`104,798` GPU-cached tok) | **`+93.75%`** |
| **Mean / p95 SSE Wire TTFT** | `3,596.3 ms` (`p95: 6,486.4 ms`) | **`191.0 ms` (`p95: 432.5 ms`)** | **`-94.69%` (`18.8x` faster)** |
| **Peak Single-Turn Context** | `35,696` tokens *(exceeded 32,768 limit)* | **`11,123` tokens** | **`-68.84%`** |
| **Uncached GPU Prefill Tokens** | `155,751` *(before Turn 7 crash)* | **`13,734` *(across all 16 turns)*** | **`-91.18%`** |

### 2. Nebius Token Factory (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`)
Artifact: [`results/superseded/nebius_gauntlet_nemotron_30b.json`](./results/superseded/nebius_gauntlet_nemotron_30b.json)

| Metric | Unoptimized Baseline | RunSift + SDCS | Delta |
| :--- | :---: | :---: | :---: |
| **Modules Repaired (`45` Tests)** | `4/5` (`41/45`, stalled at Turn 20) | **`5/5` (`45/45` Solved in 16 Turns)** | **`4/5 -> 5/5` (`100%` Solved)** |
| **KV-Cache Hit Rate** | `0.0%` (`0` cached tok) | **`93.75%`** (`105,600` server-cached tok) | **`+93.75%`** |
| **Mean / p50 SSE Wire TTFT** | `1,592.2 ms` (`p50: 1,463.7 ms`) | **`541.0 ms` (`p50: 539.0 ms`)** | **`-66.02%` (`2.94x` faster)** |
| **Peak Single-Turn Context** | `62,622` tokens | **`14,402` tokens** | **`-77.00%`** |
| **Uncached Prefill Tokens** | `780,477` tokens | **`46,926` tokens** | **`-93.99%`** |
| **Cumulative Session Tokens** | `780,477` tokens | **`152,526` tokens** | **`-80.46%` (saved `627,951` tok)** |

### 3. Nebius Token Factory (`nvidia/nemotron-3-super-120b-a12b`)
Artifact: [`results/superseded/nebius_gauntlet_nemotron_120b.json`](./results/superseded/nebius_gauntlet_nemotron_120b.json)

| Metric | Unoptimized Baseline | RunSift + SDCS | Delta |
| :--- | :---: | :---: | :---: |
| **Modules Repaired (`45` Tests)** | `5/5` (`45/45` in 18 Turns) | **`5/5` (`45/45` in 16 Turns)** | **`-2` Turns** |
| **KV-Cache Hit Rate** | `0.0%` | **`93.75%`** | **`+93.75%`** |
| **Mean / p95 SSE Wire TTFT** | `1,580.3 ms` (`p95: 2,108.3 ms`) | **`758.0 ms` (`p95: 908.3 ms`)** | **`-52.03%` (`2.08x` faster)** |
| **Peak Single-Turn Context** | `56,547` tokens | **`12,914` tokens** | **`-77.16%`** |
| **Cumulative Session Tokens** | `647,650` tokens | **`131,681` tokens** | **`-79.67%` (saved `515,969` tok)** |

---

## Historical 15-Task Bug-Fix Suite (`--mode suite`, `OB-01`..`OB-15`, Superseded)

| Model & Endpoint | Baseline Solved | RunSift + SDCS Solved | KV-Cache Hit Rate | Mean TTFT Delta | Prompt Token Savings | Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Local `Qwen2.5-Coder-7B` (`llama-server` RTX 3080 Ti)** | `13/15 (86.7%)` | **`15/15 (100.0%)`** | `0% -> 98.33%` | `1,109.4ms -> 119.2ms` (**`-89.25%`**) | `325,418 -> 158,220` (**`-51.38%`**) | [`local_suite15_qwen25_7b.json`](./results/superseded/local_suite15_qwen25_7b.json) |
| **Nebius `NVIDIA-Nemotron-3-Nano-30B-A3B`** | `14/15 (93.3%)` | **`15/15 (100.0%)`** | `0% -> 98.39%` | `553.8ms -> 565.1ms` *(short 4-turn)* | `419,120 -> 182,882` (**`-56.37%`**) | [`nebius_suite15_nemotron_30b.json`](./results/superseded/nebius_suite15_nemotron_30b.json) |
| **Nebius `nemotron-3-super-120b-a12b`** | `15/15 (100.0%)` | **`15/15 (100.0%)`** | `0% -> 98.36%` | `757.5ms -> 756.8ms` *(short 4-turn)* | `350,963 -> 175,611` (**`-49.96%`**) | [`nebius_suite15_nemotron_120b.json`](./results/superseded/nebius_suite15_nemotron_120b.json) |

---

## Reproducing the Benchmarks

```bash
# 1. Deterministically verify all 15 tasks + 5-bug Gauntlet + 15-bug disjointness (no LLM required)
python experiments/tensorroute_benchmark/verify_tasks.py

# 2. Run the Continuous 5-Bug Multi-Module Gauntlet on Nebius Token Factory
python -u experiments/tensorroute_benchmark/harness.py --mode gauntlet --nebius --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
python -u experiments/tensorroute_benchmark/harness.py --mode gauntlet --nebius --model nvidia/nemotron-3-super-120b-a12b

# 3. Run the 15-Task Suite (OB-01..OB-15) on Nebius Token Factory
python -u experiments/tensorroute_benchmark/harness.py --mode suite --nebius --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
python -u experiments/tensorroute_benchmark/harness.py --mode suite --nebius --model nvidia/nemotron-3-super-120b-a12b

# 4. Run locally against llama-server (--port 8081 --cache-reuse 64 -c 32768)
python -u experiments/tensorroute_benchmark/harness.py --mode both
```
