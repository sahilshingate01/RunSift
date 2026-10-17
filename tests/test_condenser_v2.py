"""Unit and fixture tests for RunSift Condenser v2 (spec section 7, tests 1-12)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from src.compact import (
    CondenserV2,
    condense_tool_output,
    extract_referenced_setup_lines,
)
from src.core.config import RunSiftConfig
from src.proxy.gateway import create_app

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Determinism (I4)
# ---------------------------------------------------------------------------
def test_01_determinism_i4() -> None:
    """The same message list condensed 100 times gives identical bytes."""
    condenser = CondenserV2()
    raw_wave1 = _read_fixture("pytest_wave1_initial_raw.txt")
    raw_rep8 = _read_fixture("pytest_scheduler_rep8_raw.txt")

    messages = [
        {"role": "system", "content": "You are an autonomous debugging agent."},
        {"role": "user", "content": "Run the test suite and fix all bugs."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "run_pytest", "content": raw_wave1},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "run_pytest", "content": raw_rep8},
    ]

    baseline_msgs, baseline_saved, baseline_before, baseline_after = condenser.compact_messages(
        messages
    )
    baseline_bytes = json.dumps(baseline_msgs, sort_keys=True).encode("utf-8")

    for _ in range(100):
        msgs_i, saved_i, before_i, after_i = condenser.compact_messages(messages)
        assert json.dumps(msgs_i, sort_keys=True).encode("utf-8") == baseline_bytes
        assert saved_i == baseline_saved
        assert before_i == baseline_before
        assert after_i == baseline_after


# ---------------------------------------------------------------------------
# 2. Idempotence (I5)
# ---------------------------------------------------------------------------
def test_02_idempotence_i5() -> None:
    """condense(condense(x)) == condense(x) for every fixture."""
    condenser = CondenserV2(max_char_threshold=200)
    fixture_specs = [
        ("pytest_scheduler_rep8_raw.txt", "run_pytest", "{}", False),
        ("pytest_scheduler_rep8_raw.txt", "run_pytest", '{"target": "tests/test_codec.py"}', False),
        ("pytest_scheduler_rep8_raw.txt", "run_pytest", "{}", True),
        ("pytest_wave1_initial_raw.txt", "run_pytest", "{}", False),
        ("pytest_coupled_initial_raw.txt", "run_pytest", "{}", False),
        ("json_array_raw.json", "database_query", "{}", False),
        ("text_log_raw.txt", "bash_exec", "{}", False),
    ]

    for fname, tool_name, tool_args, is_stale in fixture_specs:
        raw = _read_fixture(fname)
        first, saved_1 = condenser.compact_tool_output(
            raw, tool_name=tool_name, tool_args=tool_args, is_stale=is_stale
        )
        assert "<!-- cf:v2 -->" in first
        assert saved_1 > 0
        second, saved_2 = condenser.compact_tool_output(
            first, tool_name=tool_name, tool_args=tool_args, is_stale=is_stale
        )
        assert second == first
        assert saved_2 == 0


# ---------------------------------------------------------------------------
# 3. Untouched roles (I1)
# ---------------------------------------------------------------------------
def test_03_untouched_roles_i1() -> None:
    """System, user and assistant messages are byte-identical after a pass."""
    condenser = CondenserV2(max_char_threshold=100)
    sys_text = (
        "CONSTITUTIONAL RULES:\n"
        "- Never delete files without asking.\n"
        "- Do not touch tests/.\n" + ("Rule detail line with important constraints.\n" * 150)
    )
    user_text = (
        "Please fix the repository.\n"
        "Remember: Never delete files without asking, and Do not touch tests/ under any circumstances.\n"
        + ("Additional context paragraph from user.\n" * 150)
    )
    asst_text = (
        "I will inspect the failure logs while obeying: Never delete files without asking "
        "and Do not touch tests/.\n" + ("Reasoning step detail.\n" * 150)
    )
    raw_pytest = _read_fixture("pytest_coupled_initial_raw.txt")

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
        {
            "role": "assistant",
            "content": asst_text,
            "tool_calls": [
                {
                    "id": "call_p1",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_p1", "name": "run_pytest", "content": raw_pytest},
    ]

    out_msgs, saved, _, _ = condenser.compact_messages(messages)
    assert saved > 0
    assert out_msgs[0]["content"].encode("utf-8") == sys_text.encode("utf-8")
    assert out_msgs[1]["content"].encode("utf-8") == user_text.encode("utf-8")
    assert out_msgs[2]["content"].encode("utf-8") == asst_text.encode("utf-8")
    assert "Never delete files without asking" in out_msgs[0]["content"]
    assert "Do not touch tests/" in out_msgs[1]["content"]


# ---------------------------------------------------------------------------
# 4. Exempt tools (I2)
# ---------------------------------------------------------------------------
def test_04_exempt_tools_i2() -> None:
    """read_file and grep outputs above the threshold are byte-identical."""
    condenser = CondenserV2(max_char_threshold=200)
    large_file_content = "\n".join(
        [f"def func_{i}(x: int) -> int:\n    return x + {i}" for i in range(300)]
    )
    large_grep_output = "\n".join(
        [
            f"src/tensorroute/scheduler.py:{i}:    batches = sched.drain_all_batches(flush_partial=False)"
            for i in range(1, 200)
        ]
    )
    assert len(large_file_content) > 5000
    assert len(large_grep_output) > 5000

    out_read, saved_read = condenser.compact_tool_output(large_file_content, tool_name="read_file")
    assert saved_read == 0
    assert out_read.encode("utf-8") == large_file_content.encode("utf-8")

    out_grep, saved_grep = condenser.compact_tool_output(large_grep_output, tool_name="grep")
    assert saved_grep == 0
    assert out_grep.encode("utf-8") == large_grep_output.encode("utf-8")

    out_search, saved_search = condenser.compact_tool_output(
        large_grep_output, tool_name="search_code"
    )
    assert saved_search == 0
    assert out_search.encode("utf-8") == large_grep_output.encode("utf-8")


# ---------------------------------------------------------------------------
# 5. Scheduler fixture (4.4, 4.5)
# ---------------------------------------------------------------------------
def test_05_scheduler_fixture_setup_and_clipping() -> None:
    """Verify scheduler fixture keeps drain_all_batches, sched = ..., enqueue loop, and clipped tail req-4."""
    condenser = CondenserV2()
    # 1. Check rep8 replayed log (where test_scheduler_drain_all_retains_partial_remainder_when_false fails with req-3/req-4)
    raw_rep8 = _read_fixture("pytest_scheduler_rep8_raw.txt")
    digest_rep8 = condenser.condense_pytest_fresh(raw_rep8)

    assert (
        "        sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)"
        in digest_rep8
    )
    assert "        for i in range(5):" in digest_rep8
    assert (
        '            sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=10 - i))'
        in digest_rep8
    )
    assert "        batches = sched.drain_all_batches(flush_partial=False)" in digest_rep8
    assert "req-4" in digest_rep8
    assert "def test_" not in digest_rep8

    # Also verify >600-char line clipping (4.5) preserves tail req-4
    long_line = (
        "E        +  where 2 = len([[BatchRequest(request_id='req-0', prompt_tokens=100, "
        + ("padding='x', " * 45)
        + "BatchRequest(request_id='req-4', prompt_tokens=100, priority=6, prefix_hash='')]])"
    )
    assert len(long_line) > 600
    clipped_digest = condenser.condense_pytest_fresh(
        raw_rep8.replace(
            "E        +  where 2 = len([[BatchRequest(request_id='req-0', prompt_tokens=100, priority=10, prefix_hash=''), BatchRequest(request_id='req-1', prompt_tokens=100, priority=9, prefix_hash=''), BatchRequest(request_id='req-2', prompt_tokens=100, priority=8, prefix_hash='')], [BatchRequest(request_id='req-3', prompt_tokens=100, priority=7, prefix_hash=''), BatchRequest(request_id='req-4', prompt_tokens=100, priority=6, prefix_hash='')]])",
            long_line,
        )
    )
    assert "chars clipped]…" in clipped_digest
    assert "req-4" in clipped_digest

    # 2. Check wave1 initial log (where test_scheduler_drain_all_flushes_partial_remainder_when_true also fails)
    raw_wave1 = _read_fixture("pytest_wave1_initial_raw.txt")
    digest_wave1 = condenser.condense_pytest_fresh(raw_wave1)
    assert "        batches = sched.drain_all_batches(flush_partial=True)" in digest_wave1
    assert (
        '            sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=5 - i))'
        in digest_wave1
    )
    assert "def test_" not in digest_wave1


# ---------------------------------------------------------------------------
# 6. Transitive names
# ---------------------------------------------------------------------------
def test_06_transitive_names() -> None:
    """Synthetic test body where assertion uses b, b = f(a), and a = A(). Both kept; c = 3 is not."""
    frame_source = [
        "    def test_synthetic_transitive() -> None:",
        "        a = A()",
        "        c = 3",
        "        b = f(a)",
    ]
    gt_lines = [">       assert b == 42"]

    kept = extract_referenced_setup_lines(frame_source, gt_lines)
    assert kept == [
        "        a = A()",
        "        b = f(a)",
    ]
    assert all("c = 3" not in line for line in kept)
    assert all("def test_" not in line for line in kept)


# ---------------------------------------------------------------------------
# 7. Fresh and stale (4.2, 4.6)
# ---------------------------------------------------------------------------
def test_07_fresh_and_stale() -> None:
    """Verify two full-suite runs (even with different target strings) make the first stale and second fresh; single-file run doesn't stale full-suite."""
    condenser = CondenserV2()
    raw_first = _read_fixture("pytest_wave1_initial_raw.txt")
    raw_second = _read_fixture("pytest_scheduler_rep8_raw.txt")

    # 1. Two full-suite outputs with empty target strings
    messages = [
        {"role": "user", "content": "Run tests."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "run_pytest", "content": raw_first},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "run_pytest", "content": raw_second},
    ]

    compacted, _, _, _ = condenser.compact_messages(messages)
    first_digest = compacted[2]["content"]
    second_digest = compacted[4]["content"]

    assert first_digest.startswith(
        "[cf: earlier test run, superseded by a later run of the same tests]\n"
        "======================== 12 failed, 33 passed in 0.21s ========================"
    )
    assert first_digest.endswith("<!-- cf:v2 -->")
    assert "____" not in first_digest

    assert second_digest.startswith("[cf: condensed pytest output, v2]")
    assert "======================== 8 failed, 37 passed in 0.25s =========================" in (
        second_digest
    )

    # 2. Two full-suite outputs with DIFFERENT target strings (e.g. "/workspace/tensorroute" vs "{}")
    #    where the first becomes stale because both outputs contain the same sorted test-file summary
    messages_diff_targets = [
        {"role": "user", "content": "Run tests with different target strings."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c_ws",
                    "type": "function",
                    "function": {
                        "name": "run_pytest",
                        "arguments": '{"target": "/workspace/tensorroute"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c_ws", "name": "run_pytest", "content": raw_first},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c_empty",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c_empty", "name": "run_pytest", "content": raw_second},
    ]
    compacted_dt, _, _, _ = condenser.compact_messages(messages_diff_targets)
    assert compacted_dt[2]["content"].startswith(
        "[cf: earlier test run, superseded by a later run of the same tests]\n"
        "======================== 12 failed, 33 passed in 0.21s ========================"
    )
    assert compacted_dt[4]["content"].startswith("[cf: condensed pytest output, v2]")

    # 3. A single-file run after a full-suite run must NOT make the full-suite run stale
    single_file_lines = [
        line
        for line in raw_second.splitlines()
        if (not line.startswith("tests/test_") or line.startswith("tests/test_radix.py::"))
        and (
            not line.startswith("FAILED tests/test_")
            or line.startswith("FAILED tests/test_radix.py::")
        )
    ]
    raw_single_file = "\n".join(single_file_lines)
    messages_targeted = [
        {"role": "user", "content": "Run full suite then single-file targeted."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c_full",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c_full", "name": "run_pytest", "content": raw_first},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c_targ",
                    "type": "function",
                    "function": {
                        "name": "run_pytest",
                        "arguments": '{"target": "tests/test_radix.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c_targ",
            "name": "run_pytest",
            "content": raw_single_file,
        },
    ]
    compacted_t, _, _, _ = condenser.compact_messages(messages_targeted)
    assert compacted_t[2]["content"].startswith("[cf: condensed pytest output, v2]")
    assert compacted_t[4]["content"].startswith("[cf: condensed pytest output, v2]")
    assert (
        "[cf: Requested target 'tests/test_radix.py': PASSED (9 passed, 0 failed)]"
        in compacted_t[4]["content"]
    )

    # Same message list gives identical bytes on every pass
    compacted_t2, _, _, _ = condenser.compact_messages(messages_targeted)
    assert json.dumps(compacted_t, sort_keys=True).encode("utf-8") == json.dumps(
        compacted_t2, sort_keys=True
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# 8. Fallback (4.4.7)
# ---------------------------------------------------------------------------
def test_08_fallback_malformed_traceback() -> None:
    """A malformed traceback returns the 3-line fallback without raising an exception."""
    # Case A: missing `def test_` line
    no_def_lines = [
        "        line_1 = setup_alpha()",
        "        line_2 = setup_beta()",
        "        line_3 = setup_gamma()",
        "        line_4 = setup_delta()",
    ]
    res_a = extract_referenced_setup_lines(no_def_lines, [">       assert line_4 == 1"])
    assert res_a == [
        "        line_2 = setup_beta()",
        "        line_3 = setup_gamma()",
        "        line_4 = setup_delta()",
    ]

    # Case B: unclosed string/bracket causing tokenize.TokenError
    malformed_body = [
        "    def test_broken_syntax():",
        "        x = 1",
        "        y = 'unterminated multiline",
        "        z = (",
        "        w = 4",
    ]
    res_b = extract_referenced_setup_lines(malformed_body, [">       assert w == 4"])
    assert len(res_b) == 3
    assert res_b == [
        "        y = 'unterminated multiline",
        "        z = (",
        "        w = 4",
    ]


# ---------------------------------------------------------------------------
# 9. Size (<= 25% on initial 45-test log)
# ---------------------------------------------------------------------------
def test_09_size_wave1_initial_log() -> None:
    """The fresh digest of the initial 45-test log is no more than 25% of the raw chars."""
    condenser = CondenserV2()
    raw = _read_fixture("pytest_wave1_initial_raw.txt")
    digest = condenser.condense_pytest_fresh(raw)
    ratio = len(digest) / len(raw)
    print(f"Wave-1 45-test fresh digest ratio: {len(digest)} / {len(raw)} = {ratio:.4f}")
    assert ratio <= 0.25, f"Expected ratio <= 0.25, got {ratio:.4f}"


# ---------------------------------------------------------------------------
# 10. v1 golden (I7)
# ---------------------------------------------------------------------------
def test_10_v1_golden_i7(monkeypatch: pytest.MonkeyPatch) -> None:
    """With RUNSIFT_CONDENSER=v1, the existing fixtures produce exactly the bytes of commit 2d64007."""
    from src.compact import ToolOutputCompactor

    monkeypatch.setenv("RUNSIFT_CONDENSER", "v1")

    cases = [
        (
            "pytest_scheduler_rep8_raw.txt",
            "pytest_scheduler_rep8_v1_golden.txt",
            "run_pytest",
            "",
            4000,
            12,
            16,
        ),
        (
            "pytest_scheduler_rep8_raw.txt",
            "pytest_scheduler_rep8_targeted_v1_golden.txt",
            "run_pytest",
            '{"target": "tests/test_codec.py"}',
            4000,
            12,
            16,
        ),
        (
            "pytest_wave1_initial_raw.txt",
            "pytest_wave1_initial_v1_golden.txt",
            "run_pytest",
            "",
            4000,
            12,
            16,
        ),
        (
            "pytest_coupled_initial_raw.txt",
            "pytest_coupled_initial_v1_golden.txt",
            "run_pytest",
            "",
            4000,
            12,
            16,
        ),
        (
            "json_array_raw.json",
            "json_array_v1_golden.txt",
            "database_query",
            "",
            200,
            12,
            16,
        ),
        (
            "text_log_raw.txt",
            "text_log_v1_golden.txt",
            "bash_exec",
            "",
            300,
            3,
            3,
        ),
    ]

    for raw_name, golden_name, tool_name, tool_args, threshold, head_b, tail_b in cases:
        raw_text = _read_fixture(raw_name)
        expected_bytes = (FIXTURES_DIR / golden_name).read_bytes()
        if head_b == 12 and tail_b == 16:
            actual = condense_tool_output(
                raw_text,
                tool_name=tool_name,
                tool_args=tool_args,
                max_char_threshold=threshold,
            )
        else:
            c1 = ToolOutputCompactor(
                max_char_threshold=threshold,
                head_lines_budget=head_b,
                tail_lines_budget=tail_b,
            )
            actual, _ = c1.compact_tool_output(raw_text, tool_name=tool_name, tool_args=tool_args)
        assert actual.encode("utf-8") == expected_bytes, f"Mismatch on {golden_name}"


# ---------------------------------------------------------------------------
# 11. Verbatim check (I3)
# ---------------------------------------------------------------------------
def _assert_verbatim_i3(raw_text: str, digest: str) -> None:
    raw_lines = set(raw_text.splitlines())
    clip_re = re.compile(r"^(.*)…\[cf: (\d+) chars clipped\]…(.*)$")

    for line in digest.splitlines():
        if line == "<!-- cf:v2 -->":
            continue
        if line.startswith("[cf:") and line.endswith("]"):
            continue
        m = clip_re.match(line)
        if m:
            head, count_str, tail = m.group(1), m.group(2), m.group(3)
            clipped_n = int(count_str)
            assert len(head) == 290
            assert len(tail) == 290
            assert any(
                rline.startswith(head) and rline.endswith(tail) and len(rline) == 580 + clipped_n
                for rline in raw_lines
            ), f"Clipped line does not match any raw line: {line!r}"
        else:
            assert line in raw_lines, f"Non-generated line not verbatim in raw input: {line!r}"


def test_11_verbatim_check_i3() -> None:
    """Every line of every digest is either an exact line, or head+tail clipped, or [cf:...], or marker."""
    condenser = CondenserV2(max_char_threshold=200)
    raw_rep8 = _read_fixture("pytest_scheduler_rep8_raw.txt")
    raw_wave1 = _read_fixture("pytest_wave1_initial_raw.txt")
    raw_coupled = _read_fixture("pytest_coupled_initial_raw.txt")
    raw_json = _read_fixture("json_array_raw.json")
    raw_log = _read_fixture("text_log_raw.txt")

    _assert_verbatim_i3(raw_rep8, condenser.condense_pytest_fresh(raw_rep8))
    _assert_verbatim_i3(
        raw_rep8,
        condenser.condense_pytest_fresh(raw_rep8, tool_args='{"target": "tests/test_codec.py"}'),
    )
    _assert_verbatim_i3(raw_rep8, condenser.condense_pytest_stale(raw_rep8))
    _assert_verbatim_i3(raw_wave1, condenser.condense_pytest_fresh(raw_wave1))
    _assert_verbatim_i3(raw_coupled, condenser.condense_pytest_fresh(raw_coupled))

    json_digest, _ = condenser.compact_tool_output(raw_json, tool_name="database_query")
    _assert_verbatim_i3(raw_json, json_digest)

    log_digest, _ = condenser.compact_tool_output(raw_log, tool_name="bash_exec")
    _assert_verbatim_i3(raw_log, log_digest)


# ---------------------------------------------------------------------------
# 12. Coupled fixtures (the decisive mechanism)
# ---------------------------------------------------------------------------
def test_12_coupled_fixtures_decisive_mechanism() -> None:
    """Verify coupled log keeps pipeline & radix setup lines (including tuple unpacking) and ratio <= 30%."""
    condenser = CondenserV2()
    raw_coupled = _read_fixture("pytest_coupled_initial_raw.txt")
    digest = condenser.condense_pytest_fresh(raw_coupled)

    # 1. test_pipeline_routes_to_warm_worker exact lines from fixture
    assert "        pipeline = build_pipeline(chunk_size=4)" in digest
    assert (
        '        pipeline.tree.insert_prefix([1, 2, 3, 4, 5, 6, 7, 8], worker_id="gpu-warm", now=10.0)'
        in digest
    )
    assert (
        "        worker = pipeline.route([1, 2, 3, 4, 5, 6, 7, 8], now=20.0, min_cached_tokens=8)"
        in digest
    )

    # 2. test_radix_match_counts_tokens exact lines from fixture (tuple unpacking binds `matched`)
    assert "        tree = PrefixRadixTree(chunk_size=4, default_ttl=100.0)" in digest
    assert (
        '        tree.insert_prefix([1, 2, 3, 4, 5, 6, 7, 8], worker_id="gpu-0", now=10.0)'
        in digest
    )
    assert (
        "        worker, matched = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8], now=20.0)" in digest
    )

    # 3. Neither digest contains a `def test_` line
    assert "def test_" not in digest

    # 4. Whole fresh digest is no more than 30% of the raw chars
    ratio = len(digest) / len(raw_coupled)
    print(f"Coupled 56-test fresh digest ratio: {len(digest)} / {len(raw_coupled)} = {ratio:.4f}")
    assert ratio <= 0.30, f"Expected ratio <= 0.30, got {ratio:.4f}"


# ---------------------------------------------------------------------------
# Gateway header integration check (Section 4.8)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gateway_condenser_v2_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gateway emits x-runsift-condenser and x-runsift-tool-chars-before/after headers."""
    monkeypatch.setenv("RUNSIFT_CONDENSER", "v2")
    raw_coupled = _read_fixture("pytest_coupled_initial_raw.txt")

    class _DummyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-v2",
                    "object": "chat.completion",
                    "created": 1742000000,
                    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
                },
            )

    cfg = RunSiftConfig(condenser_version="v2")
    http_client = httpx.AsyncClient(
        transport=_DummyTransport(), base_url="https://api.tokenfactory.nebius.com/v1"
    )
    app = create_app(config=cfg, http_client=http_client)

    payload = {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "messages": [
            {"role": "user", "content": "Run tests"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "run_pytest", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "name": "run_pytest", "content": raw_coupled},
        ],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert resp.headers["x-runsift-condenser"] == "v2"
        before = int(resp.headers["x-runsift-tool-chars-before"])
        after = int(resp.headers["x-runsift-tool-chars-after"])
        assert before == len(raw_coupled)
        assert 0 < after < before
        assert after / before <= 0.30


# ---------------------------------------------------------------------------
# 14. v2 unchanged (CONDENSER_V2_1_ADDENDUM.md section 3, test 14)
# ---------------------------------------------------------------------------
def test_14_v2_unchanged_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert that v2 mode reproduces all v2 golden fixtures and the rep9 replay byte for byte."""
    import gzip

    cases = [
        (
            "pytest_scheduler_rep8_raw.txt",
            "pytest_scheduler_rep8_v2_fresh_golden.txt",
            "run_pytest",
            "",
            4000,
            False,
        ),
        (
            "pytest_scheduler_rep8_raw.txt",
            "pytest_scheduler_rep8_target_codec_v2_fresh_golden.txt",
            "run_pytest",
            '{"target": "tests/test_codec.py"}',
            4000,
            False,
        ),
        (
            "pytest_scheduler_rep8_raw.txt",
            "pytest_scheduler_rep8_v2_stale_golden.txt",
            "run_pytest",
            "",
            4000,
            True,
        ),
        (
            "pytest_wave1_initial_raw.txt",
            "pytest_wave1_initial_v2_fresh_golden.txt",
            "run_pytest",
            "",
            4000,
            False,
        ),
        (
            "pytest_coupled_initial_raw.txt",
            "pytest_coupled_initial_v2_fresh_golden.txt",
            "run_pytest",
            "",
            4000,
            False,
        ),
        ("json_array_raw.json", "json_array_v2_golden.txt", "database_query", "", 200, False),
        ("text_log_raw.txt", "text_log_v2_golden.txt", "bash_exec", "", 300, False),
    ]

    for raw_name, golden_name, tool_name, tool_args, threshold, is_stale in cases:
        raw_text = _read_fixture(raw_name)
        expected_bytes = (FIXTURES_DIR / golden_name).read_bytes()
        c2 = CondenserV2(max_char_threshold=threshold)
        actual, _ = c2.compact_tool_output(
            raw_text, tool_name=tool_name, tool_args=tool_args, is_stale=is_stale
        )
        assert actual.encode("utf-8") == expected_bytes, f"Mismatch on {golden_name}"

    # Verify rep9 sidecar replay through v2 matches rep9_v2_replay_golden.json byte for byte
    sc_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "tensorroute_benchmark"
        / "results"
        / "local_coupled_nemotron_30b_40turn_v2_rep9_tool_outputs.jsonl.gz"
    )
    sc_rows = [
        json.loads(ln)
        for ln in gzip.decompress(sc_path.read_bytes()).decode("utf-8").splitlines()
        if ln.strip()
    ]
    cf_rows = [r for r in sc_rows if r["arm"] in ("runsift", "cacheforge")]

    msgs: list[dict[str, object]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    c2_replay = CondenserV2(max_char_threshold=4000)
    turns_actual = []
    for turn_idx in range(1, 41):
        comp_msgs, saved, b_chars, a_chars = c2_replay.compact_messages(msgs)
        condensable_contents = [
            {"tool_call_id": m.get("tool_call_id"), "name": m.get("name"), "content": m["content"]}
            for m in comp_msgs
            if m.get("role") == "tool" and m.get("name") in ("run_pytest", "search_code")
        ]
        turns_actual.append(
            {
                "turn": turn_idx,
                "saved": saved,
                "chars_before": b_chars,
                "chars_after": a_chars,
                "condensable_contents": condensable_contents,
            }
        )
        row = next((r for r in cf_rows if r["turn"] == turn_idx), None)
        if row and row["tool"] in ("run_pytest", "search_code", "read_file", "edit_file"):
            tc_id = f"call_{turn_idx}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": row["tool"], "arguments": json.dumps(row["args"])},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": row["tool"],
                    "content": row["raw_output"],
                }
            )

    expected_rep9_bytes = (FIXTURES_DIR / "rep9_v2_replay_golden.json").read_bytes()
    actual_rep9_bytes = json.dumps(turns_actual, indent=2, sort_keys=True).encode("utf-8")
    assert actual_rep9_bytes == expected_rep9_bytes

    # Also re-run the existing v1 golden test
    test_10_v1_golden_i7(monkeypatch)


# ---------------------------------------------------------------------------
# 13. Prefix stability (I8), end to end (CONDENSER_V2_1_ADDENDUM.md test 13)
# ---------------------------------------------------------------------------
def _replay_rep9_through_full_gateway_pipeline(
    enable_stale_rewrite: bool,
) -> tuple[int, list[int]]:
    """Replays rep9 sidecar turn-by-turn through CondenserV2 + PrefixCanonicalizer (as gateway.py applies them).

    Returns (rewrite_violation_turns_count, violated_turn_numbers).
    """
    import gzip

    from src.canonicalizer import PrefixCanonicalizer

    sc_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "tensorroute_benchmark"
        / "results"
        / "local_coupled_nemotron_30b_40turn_v2_rep9_tool_outputs.jsonl.gz"
    )
    sc_rows = [
        json.loads(ln)
        for ln in gzip.decompress(sc_path.read_bytes()).decode("utf-8").splitlines()
        if ln.strip()
    ]
    cf_rows = [r for r in sc_rows if r["arm"] in ("runsift", "cacheforge")]
    assert len(cf_rows) == 40

    condenser = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=enable_stale_rewrite)
    canonicalizer = PrefixCanonicalizer()

    raw_messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are an autonomous debugging agent."},
        {"role": "user", "content": "Fix all coupled bugs in the repository."},
    ]

    prev_canonical_msgs_bytes: list[bytes] | None = None
    violated_turns: list[int] = []

    for turn_idx in range(1, 41):
        # 1. Gateway condensing step
        compacted_msgs, _, _, _ = condenser.compact_messages(raw_messages)
        # 2. Gateway canonicalization step
        body = {
            "model": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf",
            "messages": compacted_msgs,
            "temperature": 0.6,
            "stream": True,
        }
        canonical_body, _ = canonicalizer.canonicalize_request(body)
        cur_msgs = canonical_body["messages"]
        cur_msgs_bytes = [
            json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            for m in cur_msgs
        ]

        if prev_canonical_msgs_bytes is not None:
            prefix_len = len(prev_canonical_msgs_bytes)
            assert len(cur_msgs_bytes) >= prefix_len
            if cur_msgs_bytes[:prefix_len] != prev_canonical_msgs_bytes:
                violated_turns.append(turn_idx)

        prev_canonical_msgs_bytes = cur_msgs_bytes

        # Append this turn's assistant tool call + raw tool output for turn t+1
        row = next((r for r in cf_rows if r["turn"] == turn_idx), None)
        assert row is not None
        tc_id = f"call_{turn_idx}"
        raw_messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": row["tool"], "arguments": json.dumps(row["args"])},
                    }
                ],
            }
        )
        raw_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": row["tool"],
                "content": row["raw_output"],
            }
        )

    return len(violated_turns), violated_turns


def test_13_prefix_stability_i8_end_to_end() -> None:
    """Replay rep9 sidecar through full gateway pipeline: passes for v2.1 (0 rewrites) and fails for v2."""
    v2_1_violations, v2_1_turns = _replay_rep9_through_full_gateway_pipeline(
        enable_stale_rewrite=False
    )
    assert v2_1_violations == 0, f"v2.1 violated I8 prefix stability on turns {v2_1_turns}"

    v2_violations, v2_turns = _replay_rep9_through_full_gateway_pipeline(enable_stale_rewrite=True)
    assert v2_violations > 0, (
        "Expected v2 to violate prefix stability on rep9 when stale rewrites occur"
    )
    # Specifically rep9 has multiple full-suite pytest runs (e.g. turns 10 and 14)
    assert 10 in v2_turns and 14 in v2_turns


# ---------------------------------------------------------------------------
# 15. Fresh digest equality (CONDENSER_V2_1_ADDENDUM.md test 15)
# ---------------------------------------------------------------------------
def test_15_fresh_digest_equality() -> None:
    """For any single pytest message, the v2.1 digest equals v2's fresh digest of the same message."""
    c_v2 = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=True)
    c_v2_1 = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=False)

    pytest_fixtures = [
        ("pytest_scheduler_rep8_raw.txt", "{}"),
        ("pytest_scheduler_rep8_raw.txt", '{"target": "tests/test_codec.py"}'),
        ("pytest_wave1_initial_raw.txt", "{}"),
        ("pytest_coupled_initial_raw.txt", "{}"),
    ]

    for fname, t_args in pytest_fixtures:
        raw = _read_fixture(fname)
        v2_fresh, v2_saved = c_v2.compact_tool_output(
            raw, tool_name="run_pytest", tool_args=t_args, is_stale=False
        )
        v2_1_out, v2_1_saved = c_v2_1.compact_tool_output(
            raw, tool_name="run_pytest", tool_args=t_args, is_stale=False
        )
        v2_1_stale_flag, _ = c_v2_1.compact_tool_output(
            raw, tool_name="run_pytest", tool_args=t_args, is_stale=True
        )
        assert v2_1_out.encode("utf-8") == v2_fresh.encode("utf-8")
        assert v2_1_stale_flag.encode("utf-8") == v2_fresh.encode("utf-8")
        assert v2_1_saved == v2_saved
        assert condense_tool_output(
            raw, tool_name="run_pytest", tool_args=t_args, version="v2.1"
        ).encode("utf-8") == v2_fresh.encode("utf-8")


# ---------------------------------------------------------------------------
# 16. The v2 invariants in v2.1 mode (CONDENSER_V2_1_ADDENDUM.md test 16)
# ---------------------------------------------------------------------------
def test_16_v2_invariants_in_v2_1_mode() -> None:
    """Verify determinism, idempotence, verbatim lines (I3), untouched roles, exempt tools, and fixtures in v2.1 mode."""
    c_v2_1 = CondenserV2(max_char_threshold=200, enable_stale_rewrite=False)
    raw_wave1 = _read_fixture("pytest_wave1_initial_raw.txt")
    raw_rep8 = _read_fixture("pytest_scheduler_rep8_raw.txt")
    raw_coupled = _read_fixture("pytest_coupled_initial_raw.txt")

    # 1. Determinism (100x) and permanent fresh digests across multiple pytest messages
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User prompt."},
        {
            "role": "assistant",
            "content": "Running first pytest.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "run_pytest", "content": raw_wave1},
        {
            "role": "assistant",
            "content": "Running second pytest.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "run_pytest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "run_pytest", "content": raw_rep8},
    ]
    base_msgs, base_saved, base_before, base_after = c_v2_1.compact_messages(messages)
    base_bytes = json.dumps(base_msgs, sort_keys=True).encode("utf-8")
    # First pytest message must still be fresh (not stale) even though a second full-suite run followed it
    assert base_msgs[3]["content"] == c_v2_1.condense_pytest_fresh(raw_wave1)
    assert base_msgs[5]["content"] == c_v2_1.condense_pytest_fresh(raw_rep8)

    for _ in range(100):
        m_i, s_i, b_i, a_i = c_v2_1.compact_messages(messages)
        assert json.dumps(m_i, sort_keys=True).encode("utf-8") == base_bytes
        assert (s_i, b_i, a_i) == (base_saved, base_before, base_after)

    # 2. Untouched roles (I1)
    assert base_msgs[0]["content"] == "System prompt."
    assert base_msgs[1]["content"] == "User prompt."
    assert base_msgs[2]["content"] == "Running first pytest."
    assert base_msgs[4]["content"] == "Running second pytest."

    # 3. Idempotence (I5) & Verbatim check (I3) across all fixtures
    for fname, t_name, t_args in [
        ("pytest_scheduler_rep8_raw.txt", "run_pytest", "{}"),
        ("pytest_scheduler_rep8_raw.txt", "run_pytest", '{"target": "tests/test_codec.py"}'),
        ("pytest_wave1_initial_raw.txt", "run_pytest", "{}"),
        ("pytest_coupled_initial_raw.txt", "run_pytest", "{}"),
        ("json_array_raw.json", "database_query", "{}"),
        ("text_log_raw.txt", "bash_exec", "{}"),
    ]:
        raw = _read_fixture(fname)
        first, saved_1 = c_v2_1.compact_tool_output(raw, tool_name=t_name, tool_args=t_args)
        assert "<!-- cf:v2 -->" in first
        assert saved_1 > 0
        second, saved_2 = c_v2_1.compact_tool_output(first, tool_name=t_name, tool_args=t_args)
        assert second == first
        assert saved_2 == 0
        _assert_verbatim_i3(raw, first)

    # 4. Exempt tools (I2)
    large_code = "\n".join([f"def f_{i}():\n    return {i}" for i in range(400)])
    for exempt_name in ("read_file", "grep", "search_code"):
        out_ex, saved_ex = c_v2_1.compact_tool_output(large_code, tool_name=exempt_name)
        assert saved_ex == 0
        assert out_ex == large_code

    # 5. Scheduler & Coupled fixtures (tests 5 & 12 checks in v2.1 mode)
    sched_digest, _ = c_v2_1.compact_tool_output(raw_rep8, tool_name="run_pytest")
    assert (
        "        sched = ContinuousBatchScheduler(max_batch_size=3, max_batch_tokens=4096)"
        in sched_digest
    )
    assert "        for i in range(5):" in sched_digest
    assert (
        '            sched.enqueue(BatchRequest(f"req-{i}", prompt_tokens=100, priority=10 - i))'
        in sched_digest
    )
    assert "        batches = sched.drain_all_batches(flush_partial=False)" in sched_digest
    assert "req-4" in sched_digest

    coupled_digest, _ = c_v2_1.compact_tool_output(raw_coupled, tool_name="run_pytest")
    assert "        pipeline = build_pipeline(chunk_size=4)" in coupled_digest
    assert (
        "        worker, matched = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8], now=20.0)"
        in coupled_digest
    )
    assert "def test_" not in coupled_digest
    assert len(coupled_digest) / len(raw_coupled) <= 0.30


@pytest.mark.asyncio
async def test_gateway_condenser_v2_1_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gateway emits x-runsift-condenser: v2.1 when RUNSIFT_CONDENSER=v2.1."""
    monkeypatch.setenv("RUNSIFT_CONDENSER", "v2.1")
    raw_coupled = _read_fixture("pytest_coupled_initial_raw.txt")

    class _DummyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-v21",
                    "object": "chat.completion",
                    "created": 1742000000,
                    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
                },
            )

    cfg = RunSiftConfig(condenser_version="v2.1")
    http_client = httpx.AsyncClient(
        transport=_DummyTransport(), base_url="https://api.tokenfactory.nebius.com/v1"
    )
    app = create_app(config=cfg, http_client=http_client)

    payload = {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "messages": [
            {"role": "user", "content": "Run tests"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "run_pytest", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "name": "run_pytest", "content": raw_coupled},
        ],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert resp.headers["x-runsift-condenser"] == "v2.1"


def test_17_pytest_timeout_and_incomplete_run_golden() -> None:
    """When pytest output has no summary line or contains [TIMEOUT], digest states the run did not finish, keeps the last started test line and timeout line verbatim, and labels per-file counts as partial."""
    raw_timeout = _read_fixture("pytest_timeout_synthetic_raw.txt")
    expected_bytes = (FIXTURES_DIR / "pytest_timeout_synthetic_v2_fresh_golden.txt").read_bytes()

    c2 = CondenserV2(max_char_threshold=100)
    actual_fresh, saved = c2.compact_tool_output(
        raw_timeout, tool_name="run_pytest", is_stale=False
    )
    assert saved > 0
    assert actual_fresh.encode("utf-8") == expected_bytes
    _assert_verbatim_i3(raw_timeout, actual_fresh)

    # Also verify stale digest preserves partial counts, last started test line, and timeout line
    actual_stale, _ = c2.compact_tool_output(raw_timeout, tool_name="run_pytest", is_stale=True)
    assert "did not finish" in actual_stale
    assert "(partial)" in actual_stale
    assert "tests/test_beta.py::test_beta_drain_retains_partial_when_false " in actual_stale
    assert (
        "[TIMEOUT] pytest exceeded 35s limit (possible infinite loop or hang in workspace code)."
        in actual_stale
    )
    _assert_verbatim_i3(raw_timeout, actual_stale)

    # Also verify missing summary line without [TIMEOUT] line
    raw_no_summary = "\n".join(line for line in raw_timeout.splitlines() if "[TIMEOUT]" not in line)
    digest_no_summary = c2.condense_pytest_fresh(raw_no_summary)
    assert "did not finish" in digest_no_summary
    assert "[cf: tests/test_alpha.py: 2 passed, 1 failed (partial)]" in digest_no_summary
    assert "[cf: tests/test_beta.py: 2 passed, 0 failed (partial)]" in digest_no_summary
    assert "tests/test_beta.py::test_beta_drain_retains_partial_when_false " in digest_no_summary
    _assert_verbatim_i3(raw_no_summary, digest_no_summary)


# ---------------------------------------------------------------------------
# 18. Shell-test Addendum A4 tests (CONDENSER_V2_SHELL_TESTS_ADDENDUM.md)
# ---------------------------------------------------------------------------
PYTEST_FORMATS_DIR = FIXTURES_DIR / "pytest_formats"


def test_18_shell_pytest_invariants_i1_to_i8_on_all_format_fixtures() -> None:
    """Checks I1-I8 across every fixture in tests/fixtures/pytest_formats/ for both fresh and stale."""
    from src.compact.condenser_v2 import (
        _extract_progress_per_file_counts,
        _is_any_summary_line,
        _region_has_verbose_lines,
        split_shell_pytest_region,
    )

    c2 = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=True)
    fixture_paths = sorted(PYTEST_FORMATS_DIR.glob("*.txt"))
    assert len(fixture_paths) >= 16

    for fp in fixture_paths:
        raw = fp.read_text(encoding="utf-8")
        prefix_lines, region_text, suffix_lines = split_shell_pytest_region(raw)

        for is_stale in (False, True):
            out, _saved = c2.compact_tool_output(
                raw,
                tool_name="bash",
                tool_args='{"command": "pytest"}',
                is_stale=is_stale,
            )
            assert len(out) <= len(raw)

            # I1: Verbatim
            _assert_verbatim_i3(raw, out)

            # I2: Every failing test id in the input's FAILED/ERROR lines appears in the output
            for line in region_text.splitlines():
                s = line.strip()
                if s.startswith(("FAILED ", "ERROR ")):
                    test_id = s.split()[1]
                    assert test_id in out, f"Missing failing test id {test_id} in {fp.name}"

            # I3: Each failure's E lines appear, in formats that print them (fresh mode)
            if not is_stale:
                for line in region_text.splitlines():
                    s = line.strip()
                    if line.lstrip().startswith("E ") and s != "E":
                        assert s in out, f"Missing E line {s!r} in {fp.name}"

            # I4: Final summary line appears verbatim when present; "did not finish" only when absent
            summary_lines = [
                ln for ln in region_text.splitlines() if _is_any_summary_line(ln.strip())
            ]
            if summary_lines:
                last_summary = summary_lines[-1]
                assert last_summary in out, f"Missing summary line {last_summary!r} in {fp.name}"
                assert "test run did not finish" not in out
            else:
                assert "test run did not finish" in out

            # I5: Per-file count lines appear only when derived from verbose or progress lines, and match
            has_verbose = _region_has_verbose_lines(region_text)
            prog_counts = _extract_progress_per_file_counts(region_text)
            per_file_cf_lines = [
                ln
                for ln in out.splitlines()
                if re.match(r"^\[cf: \S+\.py: \d+ passed, \d+ failed", ln)
            ]
            if out != raw:
                if not is_stale:
                    if has_verbose:
                        expected_fc = c2._v1._extract_per_file_counts(region_text)
                        assert len(per_file_cf_lines) == len(expected_fc)
                        for fpath, stats in sorted(expected_fc.items()):
                            assert (
                                f"[cf: {fpath}: {stats['passed']} passed, {stats['failed']} failed"
                                in out
                            )
                    elif prog_counts:
                        assert len(per_file_cf_lines) == len(prog_counts)
                        for fpath, stats in sorted(prog_counts.items()):
                            assert (
                                f"[cf: {fpath}: {stats['passed']} passed, {stats['failed']} failed"
                                in out
                            )
                    else:
                        assert per_file_cf_lines == [], (
                            f"Unexpected per-file counts in {fp.name}: {per_file_cf_lines}"
                        )
                elif not summary_lines:
                    # Stale incomplete run
                    if prog_counts and not has_verbose:
                        assert len(per_file_cf_lines) == len(prog_counts)
                else:
                    assert per_file_cf_lines == []

            # I6: Prefix and suffix lines appear verbatim, within the caps
            if out != raw:
                if prefix_lines:
                    assert "[cf: output before the test run]" in out
                    for pl in prefix_lines[:4]:
                        assert pl in out
                    for pl in prefix_lines[-4:]:
                        assert pl in out
                else:
                    assert "[cf: output before the test run]" not in out

                if suffix_lines:
                    assert "[cf: output after the test run]" in out
                    for sl in suffix_lines[:10]:
                        assert sl in out
                    for sl in suffix_lines[-10:]:
                        assert sl in out
                else:
                    assert "[cf: output after the test run]" not in out

            # I7: Idempotent
            out_again, saved_again = c2.compact_tool_output(
                out,
                tool_name="bash",
                tool_args='{"command": "pytest"}',
                is_stale=is_stale,
            )
            assert out_again == out
            assert saved_again == 0

            # I8: Deterministic
            for _ in range(10):
                out_det, _ = c2.compact_tool_output(
                    raw,
                    tool_name="bash",
                    tool_args='{"command": "pytest"}',
                    is_stale=is_stale,
                )
                assert out_det.encode("utf-8") == out.encode("utf-8")


def test_19_d2_shell_tool_names_and_d3_negatives() -> None:
    """Tests D2 tool-name matching and D3 negative cases."""
    from src.compact.condenser_v2 import has_pytest_format_evidence, is_shell_like_tool_name

    # D2 positive examples from spec
    for yes_name in (
        "bash",
        "Bash",
        "execute_bash",
        "developer__shell",
        "run_terminal_cmd",
        "execute_command",
        "run_shell_command",
        "run_tests",
    ):
        assert is_shell_like_tool_name(yes_name), f"Expected D2 positive for {yes_name!r}"

    # D2 negative examples from spec
    for no_name in (
        "finish",
        "read_file",
        "edit_file",
        "search_code",
        "str_replace_editor",
    ):
        assert not is_shell_like_tool_name(no_name), f"Expected D2 negative for {no_name!r}"

    c2 = CondenserV2(max_char_threshold=4000)

    # D3 negative 1: echo "3 passed in 0.1s" (plain summary without a second marker)
    echo_out = "3 passed in 0.1s\n"
    assert not has_pytest_format_evidence(echo_out)
    assert not c2._is_shell_pytest_output(echo_out, "bash")
    out_echo, saved_echo = c2.compact_tool_output(
        echo_out, tool_name="bash", tool_args='{"command": "echo \\"3 passed in 0.1s\\""}'
    )
    assert saved_echo == 0
    assert out_echo == echo_out

    # D3 negative 2: source file containing the banner text, printed by a shell command (`cat `)
    source_with_banner = (
        "def parse_banner(line: str) -> bool:\n"
        '    """Checks for ============================= test session starts ============================="""\n'
        '    return line.startswith("=====")\n' + ("# source code line\n" * 300)
    )
    out_cat, saved_cat = c2.compact_tool_output(
        source_with_banner,
        tool_name="bash",
        tool_args='{"command": "cat src/parser.py"}',
    )
    assert saved_cat == 0
    assert out_cat == source_with_banner

    # D3 negative 3: finish-style message containing full pytest output
    raw_gauntlet = (PYTEST_FORMATS_DIR / "gauntlet_harness_vv_ra_tblong.txt").read_text(
        encoding="utf-8"
    )
    finish_msg = (
        "`finish` rejected because pytest still has 12 failure(s) (33/45 passed).\n\n"
        + raw_gauntlet
    )
    assert not c2._is_shell_pytest_output(finish_msg, "finish")
    out_fin, _ = c2.compact_tool_output(finish_msg, tool_name="finish", tool_args="{}")
    assert out_fin.startswith("[cf: TOOL EXECUTION SUMMARY :: finish,")


def test_20_g5_never_expands_short_shell_output() -> None:
    """G5: If the condensed text isn't shorter than the input, send the input unchanged."""
    c2 = CondenserV2(max_char_threshold=4000)
    # Matches D3 (progress marker + plain summary)
    tiny_d3 = ".. [100%]\n2 passed in 0.01s\n"
    assert c2._is_shell_pytest_output(tiny_d3, "bash")
    out, saved = c2.compact_tool_output(
        tiny_d3, tool_name="bash", tool_args='{"command": "pytest -q"}'
    )
    assert saved == 0
    assert out == tiny_d3


def test_21_g6_extract_target_from_shell_commands() -> None:
    """G6: Realistic shell commands extract the target token accurately."""
    from src.compact.condenser_v2 import extract_target_from_shell_args

    assert (
        extract_target_from_shell_args('{"command": "python -m pytest -q tests/test_codec.py"}')
        == "tests/test_codec.py"
    )
    assert (
        extract_target_from_shell_args('{"command": "pytest -k \\"radix and not slow\\" tests"}')
        == "tests"
    )
    assert extract_target_from_shell_args('{"command": "cd /repo && pytest -x"}') is None
    assert (
        extract_target_from_shell_args(
            '{"command": "python -m pytest -o pythonpath=src --maxfail 1 tests/test_radix.py::test_a"}'
        )
        == "tests/test_radix.py::test_a"
    )
    # Explicit target key still takes precedence
    assert (
        extract_target_from_shell_args(
            '{"command": "python -m pytest -q", "target": "tests/test_scheduler.py"}'
        )
        == "tests/test_scheduler.py"
    )


def test_22_d5_kill_switch_runsift_shell_pytest_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """D5: RUNSIFT_SHELL_PYTEST=off disables D2+D3 and restores the generic path."""
    raw = (PYTEST_FORMATS_DIR / "gauntlet_no_flags.txt").read_text(encoding="utf-8")
    c2 = CondenserV2(max_char_threshold=4000)

    out_on, _ = c2.compact_tool_output(raw, tool_name="bash", tool_args='{"command": "pytest"}')
    assert out_on.startswith("[cf: condensed pytest output, v2]")

    monkeypatch.setenv("RUNSIFT_SHELL_PYTEST", "off")
    out_off, _ = c2.compact_tool_output(raw, tool_name="bash", tool_args='{"command": "pytest"}')
    assert out_off.startswith("[cf: TOOL EXECUTION SUMMARY :: bash,")

    # Dedicated run_pytest is unaffected by kill switch
    out_rp, _ = c2.compact_tool_output(raw, tool_name="run_pytest", tool_args="{}")
    assert out_rp.startswith("[cf: condensed pytest output, v2]")


def test_23_a1_fixed_sample_of_20_sidecar_records() -> None:
    """A1 on a fixed sample of 20 sidecar run_pytest records inside pytest for v2 and v2.1."""
    import gzip

    sc_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "tensorroute_benchmark"
        / "results"
        / "local_coupled_nemotron_30b_40turn_v2_rep9_tool_outputs.jsonl.gz"
    )
    sc_2_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "tensorroute_benchmark"
        / "results"
        / "nebius_coupled_nemotron_30b_40turn_v2_rep1_tool_outputs.jsonl.gz"
    )
    records = []
    for p in (sc_path, sc_2_path):
        rows = [
            json.loads(ln)
            for ln in gzip.decompress(p.read_bytes()).decode("utf-8").splitlines()
            if ln.strip()
        ]
        records.extend(r for r in rows if r.get("tool") == "run_pytest")
        if len(records) >= 20:
            break

    sample_20 = records[:20]
    assert len(sample_20) == 20

    base_cmd = "python -m pytest -o pythonpath=src -vv -ra --tb=long --durations=0"
    for enable_stale in (True, False):
        cond = CondenserV2(max_char_threshold=4000, enable_stale_rewrite=enable_stale)
        for rec in sample_20:
            raw_out = rec["raw_output"]
            rec_args = rec.get("args") if isinstance(rec.get("args"), dict) else {}
            out_rp, _ = cond.compact_tool_output(
                raw_out, tool_name="run_pytest", tool_args=json.dumps(rec_args)
            )
            out_bash, _ = cond.compact_tool_output(
                raw_out,
                tool_name="bash",
                tool_args=json.dumps({"command": base_cmd, **rec_args}),
            )
            assert out_bash.encode("utf-8") == out_rp.encode("utf-8")
