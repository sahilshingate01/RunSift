"""CLI entrypoint for `runsift` and `python -m src.tui` (replay, watch, condense, serve)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from src.compact import CondenserV2, ToolOutputCompactor
from src.core import _load_dotenv_if_present
from src.core.models import ChatMessage
from src.tui.data import DEFAULT_RESULTS_DIR, RUN_SPECS, build_session_turn_steps, load_run_row
from src.tui.replay_app import ReplayApp

_BUILTIN_SAMPLE_PYTEST_LOG = """============================= test session starts =============================
platform linux -- Python 3.11.9, pytest-8.3.2, pluggy-1.5.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /workspace/tensorroute
collected 45 items

tests/test_codec.py::test_encode_header PASSED                           [  2%]
tests/test_codec.py::test_decode_header PASSED                           [  4%]
tests/test_codec.py::test_roundtrip_frame PASSED                         [  6%]
tests/test_codec.py::test_crc32_checksum_valid PASSED                    [  8%]
tests/test_codec.py::test_crc32_checksum_corrupt PASSED                  [ 11%]
tests/test_codec.py::test_varint_bounds_small PASSED                     [ 13%]
tests/test_codec.py::test_varint_bounds_large PASSED                     [ 15%]
tests/test_codec.py::test_zero_length_payload PASSED                     [ 17%]
tests/test_codec.py::test_max_frame_split PASSED                         [ 20%]
tests/test_codec.py::test_header_flags_bitmask PASSED                    [ 22%]
tests/test_codec.py::test_stream_id_monotonic PASSED                     [ 24%]
tests/test_codec.py::test_malformed_magic_rejected PASSED                [ 26%]
tests/test_codec.py::test_protocol_version_negotiation PASSED            [ 28%]
tests/test_codec.py::test_batch_frame_decode PASSED                      [ 31%]
tests/test_codec.py::test_keepalive_ping_ack PASSED                      [ 33%]
tests/test_router.py::test_route_table_lookup PASSED                     [ 35%]
tests/test_router.py::test_route_fallback_weight PASSED                  [ 37%]
tests/test_router.py::test_consistent_hash_ring PASSED                   [ 40%]
tests/test_router.py::test_node_drain_rebalance PASSED                   [ 42%]
tests/test_router.py::test_canary_traffic_split PASSED                   [ 44%]
tests/test_router.py::test_circuit_breaker_trip PASSED                   [ 46%]
tests/test_router.py::test_circuit_breaker_half_open PASSED              [ 48%]
tests/test_router.py::test_GeoIP_region_affinity PASSED                  [ 51%]
tests/test_router.py::test_retry_budget_exhaustion PASSED                [ 53%]
tests/test_router.py::test_hedged_request_cancellation PASSED            [ 55%]
tests/test_router.py::test_sticky_session_cookie PASSED                  [ 57%]
tests/test_router.py::test_wildcard_prefix_matching PASSED               [ 60%]
tests/test_router.py::test_healthcheck_probe_timeout PASSED              [ 62%]
tests/test_router.py::test_dead_letter_sink_routing PASSED               [ 64%]
tests/test_router.py::test_concurrent_table_swap PASSED                  [ 66%]
tests/test_scheduler.py::test_fifo_single_queue PASSED                   [ 68%]
tests/test_scheduler.py::test_weighted_fair_share PASSED                 [ 71%]
tests/test_scheduler.py::test_priority_aging_monotonic FAILED            [ 73%]
tests/test_scheduler.py::test_batch_drain_order PASSED                   [ 75%]
tests/test_scheduler.py::test_starvation_guard PASSED                    [ 77%]
tests/test_scheduler.py::test_quantum_slice_accounting PASSED            [ 80%]
tests/test_scheduler.py::test_burst_credit_replenish PASSED              [ 82%]
tests/test_scheduler.py::test_deadline_first_preemption PASSED           [ 84%]
tests/test_scheduler.py::test_idle_worker_stealing PASSED                [ 86%]
tests/test_scheduler.py::test_backpressure_watermark PASSED              [ 88%]
tests/test_scheduler.py::test_graceful_shutdown_drain PASSED             [ 91%]
tests/test_scheduler.py::test_priority_clamp_bounds PASSED               [ 93%]
tests/test_scheduler.py::test_Duplicate_packet_id_rejected PASSED        [ 95%]
tests/test_scheduler.py::test_metrics_snapshot_reset PASSED              [ 97%]
tests/test_scheduler.py::test_empty_queue_poll_noop PASSED               [100%]

================================== FAILURES ===================================
________________________ test_priority_aging_monotonic ________________________

    def test_priority_aging_monotonic():
        sched = PacketScheduler(quantum_ms=10, aging_factor=2.0)
        sched.enqueue("pkt-low", base_priority=1, waited_ms=25)
        sched.enqueue("pkt-high", base_priority=4, waited_ms=0)
        eff_low = sched.effective_priority("pkt-low")
        eff_high = sched.effective_priority("pkt-high")
>       assert eff_low == 6.0, f"expected aged priority 6.0, got {eff_low}"
E       AssertionError: expected aged priority 6.0, got -4.0
E       assert -4.0 == 6.0

tests/test_scheduler.py:42: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_scheduler.py::test_priority_aging_monotonic - AssertionError: expected aged priority 6.0, got -4.0
======================== 1 failed, 44 passed in 0.42s =========================
"""


def _load_default_specimen() -> tuple[str, str, str]:
    """Loads Run C repeat 3 turn 1 raw pytest output if present on disk, else built-in sample."""
    if DEFAULT_RESULTS_DIR.exists():
        spec_c = next(s for s in RUN_SPECS if s.key == "C")
        row = load_run_row(spec_c, results_dir=DEFAULT_RESULTS_DIR)
        sess = next(s for s in row.sessions if s.rep == 3)
        step = next(st for st in build_session_turn_steps(sess, arm="runsift") if st.turn == 1)
        if step.raw_output:
            return (
                step.raw_output,
                step.args_str,
                "Run C, repeat 3, turn 1 (committed benchmark sidecar)",
            )
    return _BUILTIN_SAMPLE_PYTEST_LOG, "{}", "built-in pytest failure specimen"


def run_condense_cli(
    input_path: str | None,
    condenser_version: str = "v2",
    tool_name: str = "run_pytest",
) -> int:
    """Condenses a tool log from file, stdin ('-'), or the default specimen and prints output."""
    tool_args = "{}"
    if input_path == "-":
        raw_text = sys.stdin.read()
        source_label = "stdin"
    elif input_path:
        path = Path(input_path)
        if not path.exists():
            sys.stderr.write(f"error: file not found: {input_path}\n")
            return 2
        raw_text = path.read_text(encoding="utf-8")
        source_label = str(path)
    else:
        raw_text, tool_args, source_label = _load_default_specimen()

    threshold = 4000 if len(raw_text) >= 4000 else 100
    if condenser_version == "v1":
        msg = ChatMessage(
            role="tool", content=raw_text, name=tool_name, tool_call_id="call_cli_condense"
        )
        compactor = ToolOutputCompactor(max_tokens=max(25, threshold // 4))
        compacted_msgs, _ = compactor.compact_messages([msg])
        condensed_text = str(compacted_msgs[0].content or "")
    else:
        condenser = CondenserV2(
            max_char_threshold=threshold,
            enable_stale_rewrite=(condenser_version == "v2"),
        )
        single_hist = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_args},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": tool_name,
                "content": raw_text,
            },
        ]
        comp_single, _, _, _ = condenser.compact_messages(single_hist)
        condensed_text = str(comp_single[1].get("content") or "")

    before = len(raw_text)
    after = len(condensed_text)
    pct = round((after / before - 1) * 100, 1) if before > 0 else 0.0
    sign_pct = f"{pct}%" if pct <= 0 else f"+{pct}%"
    before_lines = len(raw_text.splitlines())
    after_lines = len(condensed_text.splitlines())

    if input_path is None:
        sys.stdout.write("=" * 72 + "\n")
        sys.stdout.write(" RunSift Condenser v2 — Live Demo (No API Key Needed)\n")
        sys.stdout.write("=" * 72 + "\n")
        sys.stdout.write(f"# Specimen      : {source_label}\n")
        sys.stdout.write(
            "# Scenario      : A coding agent ran pytest on a buggy repo. Instead of\n"
            "#                 re-sending every passing test line to the LLM on every\n"
            "#                 turn, RunSift collapses passing tests into file\n"
            "#                 counts and keeps only the failing test's traceback.\n"
        )
        sys.stdout.write(f"x-runsift-condenser: {condenser_version}\n")
        sys.stdout.write(
            f"x-runsift-tool-chars-before: {before} ({before_lines} lines raw tool output)\n"
        )
        sys.stdout.write(
            f"x-runsift-tool-chars-after: {after} ({sign_pct}) ({after_lines} lines sent to model)\n"
        )
        sys.stdout.write("-" * 72 + "\n")
        sys.stdout.write("▼ CONDENSED PAYLOAD SENT TO THE MODEL ▼\n")
        sys.stdout.write("-" * 72 + "\n")
        sys.stdout.write(condensed_text)
        if not condensed_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.write("-" * 72 + "\n")
        sys.stdout.write(
            f"✓ Condenser v2 demo complete: {before:,} -> {after:,} chars ({sign_pct}).\n"
            "  Try piping your own test output:\n"
            "  pytest | uvx --from git+https://github.com/sahilshingate01/RunSift.git runsift condense -\n"
        )
        sys.stdout.write("=" * 72 + "\n")
    else:
        sys.stdout.write(f"# Source: {source_label}\n")
        sys.stdout.write(f"x-runsift-condenser: {condenser_version}\n")
        sys.stdout.write(f"x-runsift-tool-chars-before: {before}\n")
        sys.stdout.write(f"x-runsift-tool-chars-after: {after} ({sign_pct})\n")
        sys.stdout.write("-" * 72 + "\n")
        sys.stdout.write(condensed_text)
        if not condensed_text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runsift",
        description="RunSift CLI & Terminal App: replay recorded Token Factory sessions, condense tool logs, watch live traffic, or serve the gateway.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_replay = sub.add_parser(
        "replay",
        help="Browse every pre-registered Token Factory run and step through recorded sessions (no API key or network needed).",
    )
    p_replay.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing committed benchmark result files and sidecars.",
    )

    p_watch = sub.add_parser(
        "watch",
        help="Live view of requests passing through a running RunSift gateway.",
    )
    p_watch.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8765",
        help="Base URL of the running RunSift gateway (default: http://127.0.0.1:8765).",
    )

    p_condense = sub.add_parser(
        "condense",
        help="Condense a pytest/tool log from a file, stdin ('-'), or a real recorded benchmark specimen (no API key needed).",
    )
    p_condense.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a log file, '-' for stdin, or omit to run on a real benchmark pytest log.",
    )
    p_condense.add_argument(
        "--condenser",
        choices=["v1", "v2", "v2.1"],
        default="v2",
        help="Condenser version to apply (default: v2).",
    )
    p_condense.add_argument(
        "--tool",
        default="run_pytest",
        help="Tool name metadata for the message (default: run_pytest).",
    )

    p_serve = sub.add_parser(
        "serve",
        help="Start the OpenAI-compatible RunSift gateway (uvicorn src.proxy:app).",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1).",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port (default: 8765).",
    )
    p_serve.add_argument(
        "--no-wizard",
        action="store_true",
        help="Start the gateway immediately without interactive terminal setup prompts.",
    )

    return parser


def _mask_key(raw_key: str) -> str:
    key = raw_key.strip()
    if len(key) <= 10:
        return "****"
    return f"{key[:5]}...{key[-4:]}"


def _persist_env_key(env_path: Path, nebius_key: str, condenser_ver: str = "v2") -> None:
    lines: list[str] = []
    found_key = False
    found_cond = False
    if env_path.is_file():
        try:
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = raw_line.strip()
                if stripped.startswith("NEBIUS_API_KEY="):
                    lines.append(f"NEBIUS_API_KEY={nebius_key}")
                    found_key = True
                elif stripped.startswith("RUNSIFT_CONDENSER="):
                    lines.append(f"RUNSIFT_CONDENSER={condenser_ver}")
                    found_cond = True
                elif stripped.startswith("CACHEFORGE_CONDENSER="):
                    lines.append(f"CACHEFORGE_CONDENSER={condenser_ver}")
                    found_cond = True
                else:
                    lines.append(raw_line)
        except OSError:
            lines = []
    if not found_key:
        lines.append(f"NEBIUS_API_KEY={nebius_key}")
    if not found_cond:
        lines.append(f"RUNSIFT_CONDENSER={condenser_ver}")
    try:
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def _post_startup_worker(
    host: str,
    port: int,
    action: str,
    agent_cmd: str = "",
) -> None:
    import json
    import time
    import urllib.error
    import urllib.request
    import webbrowser

    base_url = f"http://{host}:{port}"
    # Poll /health until gateway is listening
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.25)
    else:
        return

    if action == "2":
        # Open a second pre-configured agent terminal window and optionally run the user's agent command
        run_tail = f" ^& {agent_cmd}" if agent_cmd else ""
        if sys.platform.startswith("win"):
            cmd_str = (
                f"title RunSift Agent Shell ^& "
                f'set "OPENAI_BASE_URL={base_url}/v1" ^& '
                f'set "OPENAI_API_KEY=local-gateway" ^& '
                f"echo ======================================================================== ^& "
                f"echo  RunSift Agent Terminal Connected to {base_url}/v1 ^& "
                f"echo  OPENAI_BASE_URL={base_url}/v1 ^& "
                f"echo  OPENAI_API_KEY=local-gateway ^& "
                f"echo ========================================================================"
                f"{run_tail}"
            )
            subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", cmd_str])
            webbrowser.open(f"{base_url}/dashboard")
        return

    if action == "1":
        sys.stdout.write("\n" + "=" * 72 + "\n")
        sys.stdout.write(f" ▶ Running Live 3-Turn Nemotron Agent Verification via {base_url} ...\n")
        sys.stdout.write("=" * 72 + "\n")
        sys.stdout.flush()
        try:
            req = urllib.request.Request(
                f"{base_url}/api/demo/run",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            turns = payload.get("turns") or []
            for t in turns:
                t_idx = t.get("turn", "?")
                saved = t.get("tokens_saved", 0)
                ttft = round(float(t.get("ttft_ms", 0.0)))
                hit = "CACHE HIT" if t.get("cache_hit") else "COLD PREFILL"
                summary = str(t.get("assistant_summary", "")).strip()
                sys.stdout.write(
                    f"  Turn {t_idx} [{hit}] · saved ~{saved:,} tokens · TTFT {ttft} ms\n"
                )
                if summary:
                    sys.stdout.write(f"    Nemotron: {summary[:110]}\n")
            sys.stdout.write("-" * 72 + "\n")
            sys.stdout.write(
                f" ✓ Live 3-Turn Verification Complete! Opening {base_url}/dashboard ...\n"
            )
            sys.stdout.write(f"   Point any OpenAI agent at: OPENAI_BASE_URL={base_url}/v1\n")
            sys.stdout.write("=" * 72 + "\n\n")
            sys.stdout.flush()
            webbrowser.open(f"{base_url}/dashboard")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            sys.stdout.write(f" ! Live demo notice: {exc}\n")
            sys.stdout.flush()


def _run_interactive_serve(host: str, port: int, no_wizard: bool = False) -> int:
    import getpass
    import os
    import threading

    loaded_env = _load_dotenv_if_present(override=True)
    current_key = os.getenv("NEBIUS_API_KEY", "").strip()
    local_env = Path.cwd() / ".env"

    sys.stdout.write("=" * 72 + "\n")
    sys.stdout.write(" RunSift — Interactive Gateway Setup & Launcher\n")
    sys.stdout.write("=" * 72 + "\n")

    is_tty = sys.stdin is not None and sys.stdin.isatty() and not no_wizard
    action = "1"

    if is_tty:
        # Question 1: Nebius API Key
        if current_key:
            source_desc = str(loaded_env) if loaded_env else "environment"
            sys.stdout.write(
                f" [1/2] Nebius Token Factory API Key\n"
                f"       ✓ Found key in {source_desc} ({_mask_key(current_key)})\n"
            )
            try:
                entered = getpass.getpass(
                    "       Press ENTER to keep it, or paste a new key (hidden): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                entered = ""
            if entered:
                current_key = entered
                os.environ["NEBIUS_API_KEY"] = current_key
                _persist_env_key(loaded_env or local_env, current_key)
                sys.stdout.write(f"       ✓ Saved new NEBIUS_API_KEY ({_mask_key(current_key)})\n")
        else:
            sys.stdout.write(
                " [1/2] Nebius Token Factory API Key (saved locally to .env; hidden on screen)\n"
            )
            try:
                entered = getpass.getpass(
                    "       Paste your NEBIUS_API_KEY and press ENTER (input is hidden): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                entered = ""
            if entered:
                current_key = entered
                os.environ["NEBIUS_API_KEY"] = current_key
                _persist_env_key(local_env, current_key)
                sys.stdout.write(
                    f"       ✓ Saved NEBIUS_API_KEY to {local_env} ({_mask_key(current_key)})\n"
                )
            else:
                sys.stdout.write(
                    "       ! Continuing without key (offline/demo mode; set in .env anytime)\n"
                )

        # Question 2: Agent & Verification Action
        sys.stdout.write("\n [2/2] Connect your agent or run live demo:\n")
        sys.stdout.write(
            "       [1] Run Live 3-Turn Nemotron Agent Demo + Open Live Dashboard (Recommended)\n"
        )
        sys.stdout.write(
            f"       [2] Launch your own agent in a 2nd terminal pointed at http://{host}:{port}/v1\n"
        )
        sys.stdout.write(f"       [3] Just start the gateway server on http://{host}:{port}\n")
        try:
            choice = input("       Enter choice [1/2/3] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"
        action = choice if choice in ("1", "2", "3") else "1"
        agent_cmd = ""
        if action == "2":
            try:
                agent_cmd = input(
                    "       Optional — enter agent command to run (or press ENTER for an interactive shell): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                agent_cmd = ""
    else:
        if current_key:
            sys.stdout.write(f" ✓ Loaded NEBIUS_API_KEY ({_mask_key(current_key)})\n")
        else:
            sys.stdout.write(
                f" ! NEBIUS_API_KEY not set — edit {local_env} and save to hot-reload.\n"
            )
        action = "3"
        agent_cmd = ""

    sys.stdout.write("=" * 72 + "\n")
    sys.stdout.write(f" Starting RunSift Gateway on http://{host}:{port} ...\n")
    sys.stdout.write("=" * 72 + "\n")
    sys.stdout.flush()

    if action in ("1", "2"):
        t = threading.Thread(
            target=_post_startup_worker,
            args=(host, port, action, agent_cmd),
            daemon=True,
        )
        t.start()

    env = os.environ.copy()
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.proxy:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        env=env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd is None:
        results_dir = getattr(args, "results_dir", DEFAULT_RESULTS_DIR)
        cmd = "replay" if results_dir.exists() else "condense"

    if cmd == "replay":
        results_dir = getattr(args, "results_dir", DEFAULT_RESULTS_DIR)
        app = ReplayApp(results_dir=results_dir)
        app.run()
        return 0

    if cmd == "watch":
        from src.tui.watch_app import WatchApp

        watch_app = WatchApp(base_url=args.url)
        watch_app.run()
        return 0

    if cmd == "condense":
        return run_condense_cli(
            input_path=getattr(args, "path", None),
            condenser_version=getattr(args, "condenser", "v2"),
            tool_name=getattr(args, "tool", "run_pytest"),
        )

    if cmd == "serve":
        return _run_interactive_serve(
            host=args.host,
            port=args.port,
            no_wizard=bool(getattr(args, "no_wizard", False)),
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
