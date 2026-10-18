"""Live Interactive Telemetry Dashboard for RunSift Gateway.

Serves a zero-dependency, self-contained HTML/CSS/JS dashboard via FastAPI HTMLResponse.
Uses CDN-loaded Tailwind CSS and Chart.js.
Allowed dependencies per wiring.yaml: ["core", "canonicalizer", "compact", "state", "upstream"].
"""

from __future__ import annotations


def render_dashboard_html() -> str:
    """Renders the standalone RunSift real-time telemetry dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RunSift 🔥 | Live Savings &amp; Gateway Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace'],
          },
          colors: {
            nvidia: '#76b900',
            nebius: '#6366f1',
            obsidian: '#090d16',
            cardbg: '#111827',
            cardborder: '#1f2937',
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #090d16; font-family: 'Inter', sans-serif; }
    .glow-nvidia { box-shadow: 0 0 22px -5px rgba(118, 185, 0, 0.32); }
    .glow-nebius { box-shadow: 0 0 22px -5px rgba(99, 102, 241, 0.32); }
    .status-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner-ring {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255,255,255,0.25);
      border-top-color: #76b900;
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
      vertical-align: middle;
    }
  </style>
</head>
<body class="text-slate-100 min-h-screen p-4 md:p-8">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Top Navigation / Header -->
    <header class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 pb-5 border-b border-gray-800">
      <div class="flex items-center space-x-3">
        <div class="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-500 via-emerald-600 to-indigo-600 p-0.5 glow-nvidia shrink-0">
          <div class="w-full h-full bg-gray-950 rounded-[10px] flex items-center justify-center text-xl">
            🔥
          </div>
        </div>
        <div>
          <div class="flex flex-wrap items-center gap-2">
            <h1 class="text-xl md:text-2xl font-bold tracking-tight text-white">RunSift Live Gateway Monitor</h1>
            <span class="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono whitespace-nowrap">
              http://127.0.0.1:8765/v1
            </span>
          </div>
          <p class="text-xs text-slate-400 mt-1">
            Your agent runs in the background without needing this page open — use this monitor to watch live token savings &amp; KV-cache hits over time, or click <strong class="text-[#76b900]">▶ Demo: 3 live turns with a synthetic test log</strong> (real model calls, simulated pytest log &amp; diff; MOCK M-013) to see it in action.
          </p>
        </div>
      </div>

      <!-- Interactive Controls -->
      <div class="flex flex-wrap items-center gap-2 text-xs font-mono shrink-0">
        <div class="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 text-slate-300">
          <span class="text-slate-400 whitespace-nowrap">Model:</span>
          <select id="model-selector" onchange="selectActiveModel(this.value)" class="bg-gray-900 text-[#76b900] font-semibold focus:outline-none cursor-pointer">
            <option value="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B">Nemotron-3-Nano-30B (Fast)</option>
            <option value="nvidia/nemotron-3-super-120b-a12b">Nemotron-3-Super-120B (MoE)</option>
            <option value="nvidia/Nemotron-3_5-Lightning">Nemotron-3.5-Lightning</option>
            <option value="nvidia/Nemotron-3-Ultra-550b-a55b">Nemotron-3-Ultra-550B</option>
          </select>
        </div>
        <button id="btn-run-live-v2" onclick="runReal3TurnLoop()" class="px-4 py-1.5 rounded-lg bg-gradient-to-r from-emerald-500 to-[#76b900] hover:from-emerald-400 hover:to-[#84cf00] text-gray-950 font-extrabold transition shadow-lg cursor-pointer whitespace-nowrap" title="Sends 3 real model calls to Token Factory using a simulated pytest log and git diff (MOCK M-013)">
          ▶ Demo: 3 live turns with a synthetic test log
        </button>
        <button id="btn-reset" onclick="resetTelemetry()" class="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-red-950/80 hover:text-red-300 hover:border-red-800 border border-gray-700 text-slate-300 transition cursor-pointer whitespace-nowrap">
          ↺ Reset
        </button>
      </div>
    </header>

    <!-- Status Strip -->
    <div class="bg-gray-900/90 border border-gray-800 rounded-xl px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
      <div class="flex flex-wrap items-center gap-3">
        <span class="flex items-center gap-2 text-emerald-400 font-semibold whitespace-nowrap">
          <span class="w-2 h-2 rounded-full bg-emerald-500 status-pulse"></span>
          <span id="upstream-status-badge">CONFIGURED (Nebius Token Factory)</span>
        </span>
        <span class="text-gray-700">|</span>
        <span id="condenser-badge" class="text-indigo-300 whitespace-nowrap">Condenser v2 · condenses tool output over 4,000 chars</span>
      </div>
      <div class="text-slate-400 whitespace-nowrap">
        Active Model: <span id="model-name" class="text-[#76b900] font-semibold">nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B</span>
      </div>
    </div>

    <!-- Step-by-Step Live Progress Tracker (Appears when sending 3 live test turns) -->
    <div id="live-progress-panel" class="hidden bg-gray-900/95 border border-indigo-500/40 rounded-xl p-4 space-y-3 glow-nebius">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div class="flex items-center space-x-2 text-xs font-mono">
          <span id="progress-spinner" class="spinner-ring"></span>
          <span id="progress-title" class="text-white font-bold">Dispatching 3-Turn Live Agent Loop to Nebius Token Factory...</span>
        </div>
        <span id="progress-pct-label" class="text-xs font-mono text-[#76b900] font-bold whitespace-nowrap">0% (0 / 3 turns)</span>
      </div>
      <div class="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
        <div id="progress-bar-fill" class="bg-gradient-to-r from-indigo-500 via-emerald-500 to-[#76b900] h-full rounded-full transition-all duration-300" style="width: 8%;"></div>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
        <div id="step-card-1" class="p-3 rounded-lg bg-gray-950/80 border border-gray-800 space-y-1">
          <div class="flex items-center justify-between gap-2">
            <span class="text-slate-300 font-semibold whitespace-nowrap">Turn 1 • Cold Prefill</span>
            <span id="step-badge-1" class="px-2 py-0.5 rounded bg-gray-800 text-slate-400 text-[10px] whitespace-nowrap">Pending</span>
          </div>
          <p id="step-desc-1" class="text-[11px] text-slate-400">Canonicalize system prompt &amp; pin prefix on Nebius</p>
        </div>
        <div id="step-card-2" class="p-3 rounded-lg bg-gray-950/80 border border-gray-800 space-y-1">
          <div class="flex items-center justify-between gap-2">
            <span class="text-slate-300 font-semibold whitespace-nowrap">Turn 2 • Condense Pytest</span>
            <span id="step-badge-2" class="px-2 py-0.5 rounded bg-gray-800 text-slate-400 text-[10px] whitespace-nowrap">Pending</span>
          </div>
          <p id="step-desc-2" class="text-[11px] text-slate-400">Strip 9.4 KB passing test noise + hit warm KV cache</p>
        </div>
        <div id="step-card-3" class="p-3 rounded-lg bg-gray-950/80 border border-gray-800 space-y-1">
          <div class="flex items-center justify-between gap-2">
            <span class="text-slate-300 font-semibold whitespace-nowrap">Turn 3 • Stale Log Rewrite</span>
            <span id="step-badge-3" class="px-2 py-0.5 rounded bg-gray-800 text-slate-400 text-[10px] whitespace-nowrap">Pending</span>
          </div>
          <p id="step-desc-3" class="text-[11px] text-slate-400">Fold Turn 2 log to 1 line + condense Turn 3 diff</p>
        </div>
      </div>
    </div>

    <!-- Live Status Banner -->
    <div id="demo-status-banner" class="hidden rounded-xl px-4 py-3 border text-xs font-mono flex flex-col sm:flex-row sm:items-center justify-between gap-2 transition-all">
      <span id="demo-status-text"></span>
      <span id="demo-status-meta" class="text-slate-400 whitespace-nowrap"></span>
    </div>

    <!-- Top Key Metrics Cards (4 Columns) -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">

      <!-- Metric 1: Est. tokens removed (chars ÷ 4) -->
      <div class="bg-gray-900/90 rounded-xl p-5 border border-emerald-700/60 glow-nvidia relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-medium gap-2">
          <span class="whitespace-nowrap">Est. tokens removed (chars ÷ 4)</span>
          <span class="px-1.5 py-0.5 rounded bg-emerald-950 text-[#76b900] text-[10px] font-mono border border-emerald-800/60 whitespace-nowrap" id="val-savings-pct">-0.0%</span>
        </div>
        <div class="mt-3 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-white font-mono" id="val-tokens-saved">0</span>
          <span class="text-xs text-[#76b900] font-semibold font-mono whitespace-nowrap">est. removed</span>
        </div>
        <div class="mt-2 text-xs text-slate-400 flex items-center justify-between font-mono">
          <span class="whitespace-nowrap">Est. without RunSift: <span id="val-tokens-raw" class="text-amber-400">0</span></span>
          <span>→</span>
          <span class="whitespace-nowrap">Sent: <span id="val-tokens-sent" class="text-emerald-400">0</span></span>
        </div>
        <p class="text-[11px] text-slate-500 mt-2">
          Chars removed: <span id="val-chars-removed" class="text-slate-300 font-mono">0 chars</span> · est. input cost avoided: <span id="val-cost-saved" class="text-[#76b900] font-mono">$0.00000</span>
        </p>
      </div>

      <!-- Metric 2: Prefix cache hit rate -->
      <div class="bg-gray-900/90 rounded-xl p-5 border border-gray-800/80 glow-nvidia relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-medium gap-2">
          <span class="whitespace-nowrap">Prefix Cache Hit Rate</span>
          <span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 text-[10px] font-mono border border-emerald-800/50 whitespace-nowrap">cached_tokens &gt; 0</span>
        </div>
        <div class="mt-3 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-white font-mono" id="val-hit-rate">0.0%</span>
          <span class="text-xs text-emerald-400 font-medium font-mono whitespace-nowrap" id="val-hit-count">0 / 0 hits</span>
        </div>
        <div class="w-full bg-gray-800 h-1.5 rounded-full mt-3 overflow-hidden">
          <div id="bar-hit-rate" class="bg-gradient-to-r from-emerald-500 to-[#76b900] h-full rounded-full transition-all duration-500" style="width: 0%;"></div>
        </div>
        <p class="text-[11px] text-slate-500 mt-2">Canonical prefix hashing keeps system &amp; tool schemas pinned</p>
      </div>

      <!-- Metric 3: TTFT: cache hits vs misses (this session) -->
      <div class="bg-gray-900/90 rounded-xl p-5 border border-gray-800/80 glow-nebius relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-medium gap-2">
          <span class="whitespace-nowrap">TTFT: cache hits vs misses (this session)</span>
        </div>
        <div class="mt-3 flex items-baseline space-x-2">
          <span class="text-2xl font-extrabold text-emerald-400 font-mono" id="val-ttft-warm">0.0 ms</span>
          <span class="text-xs text-slate-400 font-mono whitespace-nowrap">hits vs</span>
          <span class="text-lg font-bold text-amber-400 font-mono" id="val-ttft-cold">0.0 ms</span>
          <span class="text-xs text-slate-400 font-mono whitespace-nowrap">misses</span>
        </div>
        <div class="mt-2 text-xs text-slate-400 flex items-center justify-between font-mono">
          <span class="whitespace-nowrap">Mean hit TTFT: <span id="val-ttft-warm-sub" class="text-emerald-400">0.0 ms</span></span>
          <span class="whitespace-nowrap">Mean miss TTFT: <span id="val-ttft-cold-sub" class="text-amber-400">0.0 ms</span></span>
        </div>
        <p class="text-[11px] text-slate-500 mt-2">Within-session cache hit vs. miss mean TTFT (not a paired baseline)</p>
      </div>

      <!-- Metric 4: Cached tokens reported by Token Factory -->
      <div class="bg-gray-900/90 rounded-xl p-5 border border-gray-800/80 glow-nvidia relative overflow-hidden">
        <div class="flex items-center justify-between text-xs text-slate-400 font-medium gap-2">
          <span class="whitespace-nowrap">KV-Cached Tokens Reused</span>
        </div>
        <div class="mt-3 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-white font-mono" id="val-cached-tokens">0</span>
          <span class="text-xs text-emerald-400 font-medium font-mono whitespace-nowrap">Tokens Reused</span>
        </div>
        <div class="mt-2 text-xs text-slate-400 flex items-center justify-between font-mono">
          <span class="whitespace-nowrap">Requests: <span id="val-requests-total" class="text-slate-200">0</span></span>
          <span class="whitespace-nowrap">Cache hits: <span id="val-hits-reused-sub" class="text-emerald-400">0</span></span>
        </div>
        <p class="text-[11px] text-slate-500 mt-2">Cumulative cached_tokens reported by Nebius Token Factory</p>
      </div>

    </div>

    <!-- Charts Section (2 Columns) -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

      <!-- Chart 1: Measured TTFT per Request -->
      <div class="bg-gray-900/90 rounded-xl p-5 border border-gray-800">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-sm font-semibold text-white">Measured Time-To-First-Token (TTFT) per Request (ms)</h2>
            <p class="text-xs text-slate-400">Measured per-request TTFT colored by cache hit (green) vs. cache miss (amber)</p>
          </div>
          <span class="px-2 py-0.5 text-[10px] font-mono rounded bg-gray-800 text-slate-300 whitespace-nowrap">Live Telemetry</span>
        </div>
        <div class="h-60">
          <canvas id="ttftChart"></canvas>
        </div>
      </div>

      <!-- Chart 2: Cumulative Token & Context Savings -->
      <div class="bg-gray-900/90 rounded-xl p-5 border border-gray-800">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-sm font-semibold text-white">Cumulative Prompt Tokens: Est. without RunSift vs. Sent</h2>
            <p class="text-xs text-slate-400">Sent prompt tokens + est. tokens removed (chars ÷ 4) vs. actual prompt tokens sent</p>
          </div>
          <span class="px-2 py-0.5 text-[10px] font-mono rounded bg-gray-800 text-slate-300 whitespace-nowrap">Token Savings</span>
        </div>
        <div class="h-60">
          <canvas id="contextChart"></canvas>
        </div>
      </div>

    </div>

    <!-- Live Request Ledger Table -->
    <div class="bg-gray-900/90 rounded-xl border border-gray-800 overflow-hidden">
      <div class="p-4 border-b border-gray-800 flex items-center justify-between gap-2">
        <div>
          <h2 class="text-sm font-semibold text-white">Live Agent Turn Ledger</h2>
          <p class="text-xs text-slate-400">Every request sent through <code class="text-indigo-300">http://127.0.0.1:8765/v1/chat/completions</code></p>
        </div>
        <div class="flex items-center space-x-2 text-xs shrink-0">
          <span class="w-2 h-2 rounded-full bg-emerald-500 status-pulse"></span>
          <span class="text-slate-400 font-mono whitespace-nowrap">Auto-Updating (2.5s)</span>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs font-mono">
          <thead class="bg-gray-950/60 text-slate-400 uppercase tracking-wider text-[10px] border-b border-gray-800 whitespace-nowrap">
            <tr>
              <th class="px-4 py-3">Turn</th>
              <th class="px-4 py-3">Time</th>
              <th class="px-4 py-3">Cache Status</th>
              <th class="px-4 py-3">Tool Output Condensed</th>
              <th class="px-4 py-3">Prompt Tokens</th>
              <th class="px-4 py-3">First Token (TTFT)</th>
              <th class="px-4 py-3 text-right">Model</th>
            </tr>
          </thead>
          <tbody id="ledger-body" class="divide-y divide-gray-800/60 text-slate-300">
            <!-- Populated dynamically via /api/telemetry -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Footer -->
    <footer class="pt-2 pb-6 text-center text-xs text-slate-500 border-t border-gray-800/60">
      RunSift 🔥 • Live Local Gateway on <code class="text-slate-400">127.0.0.1:8765</code> • Powered by Nebius Token Factory &amp; NVIDIA Nemotron
    </footer>

  </div>

  <script>
    let ttftChartInstance = null;
    let contextChartInstance = null;

    // MOCK(M-013): Synthetic pytest log and git diff in dashboard 3-turn demo button
    function buildPytestTrace(turnLabel) {
      const passLines = [];
      for (let i = 1; i <= 115; i++) {
        const mod = i < 60 ? 'test_prefix_canonicalizer.py' : 'test_condenser_v2_8k.py';
        passLines.push(`tests/${mod}::test_case_${String(i).padStart(3, '0')}_invariant_check PASSED [${Math.min(99, Math.round((i / 116) * 100))}%]`);
      }
      return [
        `============================= test session starts ==============================`,
        `platform linux -- Python 3.12.4, pytest-8.3.2, pluggy-1.5.0 -- /venv/bin/python`,
        `rootdir: /workspace/RunSift (${turnLabel})`,
        `collected 116 items`,
        ``,
        ...passLines,
        `tests/test_radix_eviction.py::test_evict_stale_pages FAILED [100%]`,
        ``,
        `=================================== FAILURES ===================================`,
        `____________________________ test_evict_stale_pages ____________________________`,
        `src/state/radix_tree.py:142: in evict_lru_blocks`,
        `    assert node.ref_count == 0, f"Cannot evict pinned KV block {node.block_id} (ref_count={node.ref_count})"`,
        `E   AssertionError: Cannot evict pinned KV block blk_09f4a (ref_count=1)`,
        `=========================== short test summary info ============================`,
        `FAILED tests/test_radix_eviction.py::test_evict_stale_pages - AssertionError: Cannot evict pinned KV block blk_09f4a (ref_count=1)`,
        `======================== 1 failed, 115 passed in 4.18s =========================`
      ].join('\\n');
    }

    // MOCK(M-013): Synthetic git diff and passing test run in dashboard 3-turn demo button
    function buildGitDiffTrace() {
      const ctxLines = [];
      for (let i = 1; i <= 120; i++) {
        ctxLines.push(` context_line_${String(i).padStart(3, '0')}: page_table.verify_block_alignment(slot=${i}, stride=64, radix_warm=True)`);
      }
      return [
        `diff --git a/src/state/radix_tree.py b/src/state/radix_tree.py`,
        `index 84a91bc..f92d01e 100644`,
        `--- a/src/state/radix_tree.py`,
        `+++ b/src/state/radix_tree.py`,
        `@@ -118,126 +118,128 @@ class RadixPrefixTree:`,
        ...ctxLines,
        `-    if node.last_access_ts < cutoff_ts:`,
        `+    if node.last_access_ts < cutoff_ts and node.ref_count == 0:`,
        `         self._free_block(node.block_id)`,
        ``,
        `============================= test session starts ==============================`,
        `collected 116 items`,
        `tests/test_radix_eviction.py::test_evict_stale_pages PASSED [100%]`,
        `======================== 116 passed in 3.82s ===================================`
      ].join('\\n');
    }

    function formatCacheStatusLabel(responsePayload) {
      const usage = (responsePayload && responsePayload.usage) || {};
      const details = usage.prompt_tokens_details || {};
      const cached = Number(
        details.cached_tokens ||
        usage.cache_hit_tokens ||
        usage.prompt_cache_hit_tokens ||
        0
      );
      if (cached > 0) {
        return `cache hit (${cached.toLocaleString()} cached tokens)`;
      }
      return 'cache miss';
    }

    function initCharts() {
      const ctx1 = document.getElementById('ttftChart').getContext('2d');
      ttftChartInstance = new Chart(ctx1, {
        type: 'bar',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Measured TTFT (ms) — green: cache hit, amber: cache miss',
              data: [],
              backgroundColor: [],
              borderColor: [],
              borderWidth: 1.5,
              borderRadius: 4,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
          },
          scales: {
            x: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } },
            y: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }, title: { display: true, text: 'Milliseconds (ms)', color: '#64748b' } }
          }
        }
      });

      const ctx2 = document.getElementById('contextChart').getContext('2d');
      contextChartInstance = new Chart(ctx2, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Est. without RunSift (sent + chars removed ÷ 4)',
              data: [],
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245, 158, 11, 0.1)',
              fill: true,
              tension: 0.3,
              borderWidth: 2,
            },
            {
              label: 'Sent Prompt Tokens (with RunSift)',
              data: [],
              borderColor: '#6366f1',
              backgroundColor: 'rgba(99, 102, 241, 0.15)',
              fill: true,
              tension: 0.3,
              borderWidth: 2,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
          },
          scales: {
            x: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } },
            y: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }, title: { display: true, text: 'Cumulative Prompt Tokens', color: '#64748b' } }
          }
        }
      });
    }

    function updateModelProfileUI(modelId) {
      const sel = document.getElementById('model-selector');
      if (sel && sel.value !== modelId) {
        sel.value = modelId;
      }
      document.getElementById('model-name').textContent = modelId;
    }

    async function selectActiveModel(modelId) {
      updateModelProfileUI(modelId);
      try {
        const res = await fetch('/api/model/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelId })
        });
        if (res.ok) {
          showBanner(`✓ Switched active gateway target model to ${modelId}`, 'Ready for next agent turn', 'success');
        }
      } catch (err) {
        console.error('Failed to switch model:', err);
      }
    }

    function showBanner(text, metaText, tone) {
      const banner = document.getElementById('demo-status-banner');
      const txt = document.getElementById('demo-status-text');
      const meta = document.getElementById('demo-status-meta');
      banner.className = 'rounded-xl px-4 py-3 border text-xs font-mono flex flex-col sm:flex-row sm:items-center justify-between gap-2 transition-all ';
      if (tone === 'running') {
        banner.className += 'bg-indigo-950/80 border-indigo-700 text-indigo-200';
      } else if (tone === 'success') {
        banner.className += 'bg-emerald-950/80 border-emerald-700 text-emerald-300';
      } else {
        banner.className += 'bg-amber-950/80 border-amber-700 text-amber-300';
      }
      txt.textContent = text;
      meta.textContent = metaText || '';
    }

    function setStepCardState(stepNum, state, badgeText, descText) {
      const card = document.getElementById(`step-card-${stepNum}`);
      const badge = document.getElementById(`step-badge-${stepNum}`);
      const desc = document.getElementById(`step-desc-${stepNum}`);
      if (!card || !badge) return;
      if (state === 'running') {
        card.className = 'p-3 rounded-lg bg-indigo-950/50 border border-indigo-500/70 space-y-1';
        badge.className = 'px-2 py-0.5 rounded bg-indigo-900 text-indigo-200 text-[10px] font-bold whitespace-nowrap';
      } else if (state === 'done') {
        card.className = 'p-3 rounded-lg bg-emerald-950/40 border border-emerald-600/60 space-y-1';
        badge.className = 'px-2 py-0.5 rounded bg-emerald-900 text-[#76b900] text-[10px] font-bold whitespace-nowrap';
      } else {
        card.className = 'p-3 rounded-lg bg-gray-950/80 border border-gray-800 space-y-1';
        badge.className = 'px-2 py-0.5 rounded bg-gray-800 text-slate-400 text-[10px] whitespace-nowrap';
      }
      badge.textContent = badgeText;
      if (descText && desc) desc.textContent = descText;
    }

    async function runReal3TurnLoop() {
      const btn = document.getElementById('btn-run-live-v2');
      const selectedModel = document.getElementById('model-selector').value;
      const panel = document.getElementById('live-progress-panel');
      const spinner = document.getElementById('progress-spinner');
      const titleEl = document.getElementById('progress-title');
      const pctEl = document.getElementById('progress-pct-label');
      const barEl = document.getElementById('progress-bar-fill');

      btn.disabled = true;
      btn.textContent = '⏳ Running 3 Live Turns...';
      panel.classList.remove('hidden');
      spinner.style.display = 'inline-block';

      setStepCardState(1, 'running', 'Running...', 'Sending Turn 1 cold prefill to Nebius Token Factory...');
      setStepCardState(2, 'pending', 'Pending', 'Waiting for Turn 1...');
      setStepCardState(3, 'pending', 'Pending', 'Waiting for Turn 2...');
      barEl.style.width = '15%';
      pctEl.textContent = '15% (Turn 1 / 3)';
      titleEl.textContent = `Turn 1/3: Pinning canonical prefix on ${selectedModel}...`;

      const sessionId = 'live_dash_' + Date.now();
      const sysPrompt = 'Current Time: ' + new Date().toISOString() + '\\n' +
        'You are a Principal Systems Debugging Agent running on Nebius Token Factory with NVIDIA Nemotron. ' +
        'Always answer in 1 concise sentence citing exact file names, line numbers, and test counts.';
      const tools = [
        {
          type: 'function',
          function: {
            name: 'run_pytest',
            description: 'Executes pytest suite and returns full console output.',
            parameters: { type: 'object', properties: { target: { type: 'string' } }, required: ['target'] }
          }
        }
      ];

      try {
        // Turn 1: Cold Prefill
        const t1Messages = [
          { role: 'system', content: sysPrompt },
          { role: 'user', content: 'Turn 1: Ready to audit src/state/radix_tree.py KV-cache block eviction?' }
        ];
        const r1 = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-runsift-session-id': sessionId, 'x-cacheforge-session-id': sessionId },
          body: JSON.stringify({ model: selectedModel, messages: t1Messages, tools, max_tokens: 48, temperature: 0.0 })
        });
        const j1 = await r1.json();
        if (!r1.ok) {
          throw new Error((j1.error && (j1.error.message || j1.error)) || j1.detail || ('HTTP ' + r1.status));
        }
        await fetchTelemetry();
        const t1Ttft = Number(r1.headers.get('x-runsift-ttft-ms') || r1.headers.get('x-cacheforge-ttft-ms') || 0).toFixed(0);
        const t1Status = formatCacheStatusLabel(j1);
        setStepCardState(1, 'done', `Done (${t1Ttft}ms)`, `Cold prefill complete • ${t1Status}`);

        // Turn 2: 9.4 KB Pytest Trace -> Condenser v2
        setStepCardState(2, 'running', 'Condensing...', 'Condensing synthetic 9.4 KB pytest output via Condenser v2...');
        barEl.style.width = '55%';
        pctEl.textContent = '55% (Turn 2 / 3)';
        titleEl.textContent = `Turn 2/3: Condensing synthetic 9.4 KB pytest trace & sending to ${selectedModel}...`;

        const rawPytest = buildPytestTrace('turn-2');
        const t2Messages = [
          ...t1Messages,
          { role: 'assistant', content: null, tool_calls: [{ id: 'call_pytest_1', type: 'function', function: { name: 'run_pytest', arguments: '{"target":"tests/"}' } }] },
          { role: 'tool', tool_call_id: 'call_pytest_1', name: 'run_pytest', content: rawPytest },
          { role: 'user', content: 'Turn 2: Identify the exact failing test and AssertionError in 1 sentence.' }
        ];
        const r2 = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-runsift-session-id': sessionId, 'x-cacheforge-session-id': sessionId },
          body: JSON.stringify({ model: selectedModel, messages: t2Messages, tools, max_tokens: 64, temperature: 0.0 })
        });
        const j2 = await r2.json();
        if (!r2.ok) {
          throw new Error((j2.error && (j2.error.message || j2.error)) || j2.detail || ('HTTP ' + r2.status));
        }
        await fetchTelemetry();
        const b2 = Number(r2.headers.get('x-runsift-tool-chars-before') || r2.headers.get('x-cacheforge-tool-chars-before') || 0);
        const a2 = Number(r2.headers.get('x-runsift-tool-chars-after') || r2.headers.get('x-cacheforge-tool-chars-after') || 0);
        const pct2 = b2 > 0 ? Math.round(((a2 - b2) / b2) * 100) : 0;
        const t2Status = formatCacheStatusLabel(j2);
        setStepCardState(2, 'done', `${pct2}% Chars`, `${b2.toLocaleString()} → ${a2.toLocaleString()} chars • ${t2Status}`);

        // Turn 3: Stale Turn 2 Rewrite + 9.6 KB Git Diff Trace
        setStepCardState(3, 'running', 'Condensing...', 'Rewriting stale Turn 2 log + condensing synthetic Turn 3 git diff...');
        barEl.style.width = '85%';
        pctEl.textContent = '85% (Turn 3 / 3)';
        titleEl.textContent = `Turn 3/3: Rewriting stale tool output & verifying fix on ${selectedModel}...`;

        const rawDiff = buildGitDiffTrace();
        const t2AssistantContent = (j2.choices && j2.choices[0] && j2.choices[0].message && j2.choices[0].message.content) || '';
        const t3Messages = [
          ...t2Messages,
          { role: 'assistant', content: t2AssistantContent, tool_calls: [{ id: 'call_pytest_2', type: 'function', function: { name: 'run_pytest', arguments: '{"target":"tests/test_radix_eviction.py"}' } }] },
          { role: 'tool', tool_call_id: 'call_pytest_2', name: 'run_pytest', content: rawDiff },
          { role: 'user', content: 'Turn 3: Confirm the fix and test suite status in 1 sentence.' }
        ];
        const r3 = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-runsift-session-id': sessionId, 'x-cacheforge-session-id': sessionId },
          body: JSON.stringify({ model: selectedModel, messages: t3Messages, tools, max_tokens: 64, temperature: 0.0 })
        });
        const j3 = await r3.json();
        if (!r3.ok) {
          throw new Error((j3.error && (j3.error.message || j3.error)) || j3.detail || ('HTTP ' + r3.status));
        }
        await fetchTelemetry();
        const b3 = Number(r3.headers.get('x-runsift-tool-chars-before') || r3.headers.get('x-cacheforge-tool-chars-before') || 0);
        const a3 = Number(r3.headers.get('x-runsift-tool-chars-after') || r3.headers.get('x-cacheforge-tool-chars-after') || 0);
        const pct3 = b3 > 0 ? Math.round(((a3 - b3) / b3) * 100) : 0;
        const t3Status = formatCacheStatusLabel(j3);
        setStepCardState(3, 'done', `${pct3}% Chars`, `${b3.toLocaleString()} → ${a3.toLocaleString()} chars • ${t3Status}`);

        barEl.style.width = '100%';
        pctEl.textContent = '100% (3 / 3 turns complete)';
        spinner.style.display = 'none';
        titleEl.textContent = `✓ Completed 3 Live Nemotron Turns on ${selectedModel} (synthetic test log & diff, MOCK M-013)`;
        showBanner(
          `✓ 3 Live Turns Complete — Turn 3 Tool Output Condensed: ${b3.toLocaleString()} → ${a3.toLocaleString()} chars (${pct3}%) • ${t3Status}`,
          `Session: ${sessionId}`,
          'success'
        );
      } catch (err) {
        spinner.style.display = 'none';
        titleEl.textContent = `❌ Live loop error: ${err.message || err}`;
        showBanner(`❌ Live turn failed: ${err.message || err}`, 'Check NEBIUS_API_KEY in terminal', 'warn');
      } finally {
        btn.disabled = false;
        btn.textContent = '▶ Demo: 3 live turns with a synthetic test log';
      }
    }

    async function resetTelemetry() {
      await fetch('/api/telemetry/reset', { method: 'POST' });
      document.getElementById('live-progress-panel').classList.add('hidden');
      await fetchTelemetry();
      showBanner('↺ Telemetry counters, active sessions, and prefix cache reset to zero.', '', 'warn');
    }

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const data = await res.json();

        // Update Top Badges & Metric Cards
        const rawTok = data.cumulative_prompt_tokens_original || ((data.cumulative_prompt_tokens || 0) + (data.cumulative_tokens_saved || 0));
        const sentTok = data.cumulative_prompt_tokens || 0;
        const savedTok = data.cumulative_tokens_saved || 0;
        const savPct = rawTok > 0 ? ((savedTok / rawTok) * 100).toFixed(1) : '0.0';

        document.getElementById('val-tokens-saved').textContent = savedTok.toLocaleString();
        document.getElementById('val-tokens-raw').textContent = rawTok.toLocaleString();
        document.getElementById('val-tokens-sent').textContent = sentTok.toLocaleString();
        document.getElementById('val-savings-pct').textContent = `-${savPct}%`;
        document.getElementById('val-chars-removed').textContent = `${(data.cumulative_tool_chars_removed || 0).toLocaleString()} chars`;

        document.getElementById('val-hit-rate').textContent = data.radix_cache_hit_rate.toFixed(1) + '%';
        document.getElementById('val-hit-count').textContent = `${data.cache_hits_total || 0} / ${data.requests_total || 0} hits`;
        document.getElementById('bar-hit-rate').style.width = Math.min(100, data.radix_cache_hit_rate) + '%';
        const coldMeanText = data.mean_ttft_cold_ms.toFixed(1) + ' ms';
        const warmMeanText = data.mean_ttft_warm_ms.toFixed(1) + ' ms';
        document.getElementById('val-ttft-cold').textContent = coldMeanText;
        document.getElementById('val-ttft-warm').textContent = warmMeanText;
        const coldSubEl = document.getElementById('val-ttft-cold-sub');
        const warmSubEl = document.getElementById('val-ttft-warm-sub');
        if (coldSubEl) coldSubEl.textContent = coldMeanText;
        if (warmSubEl) warmSubEl.textContent = warmMeanText;
        document.getElementById('val-requests-total').textContent = (data.requests_total || 0).toLocaleString();
        const hitsReusedSubEl = document.getElementById('val-hits-reused-sub');
        if (hitsReusedSubEl) hitsReusedSubEl.textContent = (data.cache_hits_total || 0).toLocaleString();
        const cachedTokens = data.cached_tokens_reused || 0;
        document.getElementById('val-cached-tokens').textContent = cachedTokens.toLocaleString();
        const costSaved = Number(data.estimated_input_cost_avoided_usd || 0);
        document.getElementById('val-cost-saved').textContent = '$' + costSaved.toFixed(5);

        if (data.condenser_version) {
          const threshChars = Number(data.condenser_char_threshold || 4000).toLocaleString();
          document.getElementById('condenser-badge').textContent = `Condenser ${data.condenser_version} · condenses tool output over ${threshChars} chars`;
        }
        if (data.upstream_model) {
          updateModelProfileUI(data.upstream_model);
        }
        if (data.upstream_status) {
          document.getElementById('upstream-status-badge').textContent = data.upstream_status;
        }

        const recent = data.recent_requests || [];
        const chronological = recent.slice().reverse();

        // Dynamically update Measured TTFT & Context Compaction Charts from live request telemetry
        if (ttftChartInstance && contextChartInstance) {
          const labels = chronological.map(r => r.id);
          const actualSeries = chronological.map(r => r.ttft_ms);
          const barBgColors = chronological.map(r =>
            (r.status === 'HIT' || r.cache_hit) ? 'rgba(16, 185, 129, 0.55)' : 'rgba(245, 158, 11, 0.55)'
          );
          const barBorderColors = chronological.map(r =>
            (r.status === 'HIT' || r.cache_hit) ? 'rgba(118, 185, 0, 1.0)' : 'rgba(245, 158, 11, 1.0)'
          );

          ttftChartInstance.data.labels = labels;
          ttftChartInstance.data.datasets[0].data = actualSeries;
          ttftChartInstance.data.datasets[0].backgroundColor = barBgColors;
          ttftChartInstance.data.datasets[0].borderColor = barBorderColors;
          ttftChartInstance.update();

          let cumOrig = 0;
          let cumComp = 0;
          const origSeries = [];
          const compSeries = [];
          chronological.forEach(r => {
            const tComp = r.tokens_in || r.tokens || 0;
            const tOrig = r.tokens_original || tComp;
            cumOrig += tOrig;
            cumComp += tComp;
            origSeries.push(cumOrig);
            compSeries.push(cumComp);
          });

          contextChartInstance.data.labels = labels;
          contextChartInstance.data.datasets[0].data = origSeries;
          contextChartInstance.data.datasets[1].data = compSeries;
          contextChartInstance.update();
        }

        // Render Recent Requests Ledger (Single-line non-wrapping cells)
        const tbody = document.getElementById('ledger-body');
        if (recent.length > 0) {
          tbody.innerHTML = recent.map(req => {
            const isHit = req.status === 'HIT' || req.cache_hit;
            const statusBadge = isHit
              ? '<span class="inline-block whitespace-nowrap px-2.5 py-0.5 rounded bg-emerald-950/90 text-emerald-400 border border-emerald-800 font-bold">CACHE HIT</span>'
              : '<span class="inline-block whitespace-nowrap px-2.5 py-0.5 rounded bg-amber-950/90 text-amber-400 border border-amber-800 font-semibold">COLD MISS</span>';
            const ttftColor = isHit ? 'text-emerald-400 font-bold' : 'text-slate-300';
            const savedBadge = req.tokens_saved > 0 ? ` <span class="text-emerald-400 font-bold">(~${Number(req.tokens_saved).toLocaleString()} est. removed)</span>` : '';

            const cBefore = Number(req.tool_chars_before || 0);
            const cAfter = Number(req.tool_chars_after || 0);
            let toolCompText = '<span class="text-slate-500">No tool output</span>';
            if (cBefore > 0) {
              const pct = Math.round(((cAfter - cBefore) / cBefore) * 100);
              const pctBadge = pct < 0
                ? `<span class="text-[#76b900] font-bold">(${pct}%)</span>`
                : `<span class="text-slate-400">(0%)</span>`;
              toolCompText = `<span class="text-slate-200">${cBefore.toLocaleString()} → ${cAfter.toLocaleString()} chars</span> ${pctBadge}`;
            }

            return `
              <tr class="hover:bg-gray-800/40 transition whitespace-nowrap">
                <td class="px-4 py-3 text-white font-semibold whitespace-nowrap">${escapeHtml(req.id)}</td>
                <td class="px-4 py-3 text-slate-400 whitespace-nowrap">${escapeHtml(req.timestamp)}</td>
                <td class="px-4 py-3 whitespace-nowrap">${statusBadge}</td>
                <td class="px-4 py-3 whitespace-nowrap">${toolCompText}</td>
                <td class="px-4 py-3 text-slate-300 whitespace-nowrap">${req.tokens_in || req.tokens}${savedBadge}</td>
                <td class="px-4 py-3 ${ttftColor} whitespace-nowrap">${req.ttft_ms.toFixed(1)} ms</td>
                <td class="px-4 py-3 text-right text-slate-400 whitespace-nowrap">${escapeHtml(req.model)}</td>
              </tr>
            `;
          }).join('');
        } else {
          tbody.innerHTML = `
            <tr>
              <td colspan="7" class="px-4 py-8 text-center text-slate-400">
                No agent turns recorded yet — run your agent against <code class="text-indigo-300">http://127.0.0.1:8765/v1</code> or click <span class="text-[#76b900] font-bold">▶ Demo: 3 live turns with a synthetic test log</span> above.
              </td>
            </tr>
          `;
        }
      } catch (err) {
        console.error("Failed to fetch live telemetry:", err);
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      initCharts();
      fetchTelemetry();
      setInterval(fetchTelemetry, 2500);
    });
  </script>
</body>
</html>
"""
