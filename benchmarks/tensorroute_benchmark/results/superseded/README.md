# Superseded Benchmark Results Archive

> [!WARNING]
> These are historical logs and result files; several claims from earlier runs were later corrected. Current results are in [`PRE_REGISTRATION.md`](../../PRE_REGISTRATION.md) and the committed result files in [`benchmarks/tensorroute_benchmark/results/`](../).
> The JSON files in this directory are **superseded and invalid for head-to-head comparison**.
> They are preserved strictly for historical lineage and audit trail integrity.

## Why These Results Are Superseded

1. **Harness Asymmetry & Prompt Contamination:**
   These runs were captured with the pre-audit version of `harness.py` and `repo_builder.py`, where system prompts contained asymmetric architectural records (`DECISIONS_MD`) detailing bug fixes, hint-laden symptom descriptions, and unnatural baseline variations (such as per-turn tool schema shuffling).

2. **Retroactive / Synthetic Provenance Blocks:**
   Prior commit `83842dd` retroactively appended manual `provenance` blocks claiming git commit `d130f14`. However, those runs were collected hours prior to that commit, and contained inferred hardware/runtime specs (e.g. `8x HGX H100`, specific vLLM versions) not exposed by upstream provider APIs.

3. **Active Benchmark Baseline:**
   Current benchmarks run through `harness.py` and `run_benchmark.py` with:
   - Zero-spoiler symptom prompts (no code solutions or arithmetic hints).
   - Strict prompt parity (both baseline and CacheForge arms receive identical system instructions and stable tool schemas).
   - Live, programmatic runtime provenance capture (`git rev-parse`, live `/v1/models` API responses, real local GPU metrics from `nvidia-smi`, or explicit `"not exposed by provider"` markers for cloud endpoints).
