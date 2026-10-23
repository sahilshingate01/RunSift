"""Unit tests for benchmark final evaluation scoring and timeout/incomplete guards."""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
from dataclasses import asdict

import pytest

from benchmarks.tensorroute_benchmark import harness
from benchmarks.tensorroute_benchmark.harness import (
    ALL_SUBSYSTEMS,
    AuthQuotaAbortError,
    TaskArmResult,
    _build_scorecard_dict,
    _execute_agent_session,
    _reset_persistent_conn,
    classify_upstream_error,
    execute_tool_on_workspace,
    extract_subsystem_status,
    score_coupled_evaluation,
    score_final_evaluation,
    stream_chat_completion_sse,
    update_regression_bookkeeping,
)


def test_score_final_evaluation_timeout_yields_zero_modules():
    """Verify that a 35s pytest timeout scores modules_fixed = 0/5 and records final_eval_status = 'timeout'."""
    raw_timeout = (
        "============================= test session starts =============================\n"
        "collecting ... collected 45 items\n\n"
        "[TIMEOUT] pytest exceeded 35s limit (possible infinite loop or hang in workspace code)."
    )
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw_timeout,
        passed=0,
        failed=45,
        rc=1,
        expected_tests=45,
    )

    assert status == "timeout"
    assert modules_fixed == "0/5"
    assert solved is False
    assert passing == []


def test_score_final_evaluation_incomplete_test_count_yields_zero_modules():
    """Verify that an interrupted or crashed pytest run with passed + failed != 45 scores 0/5 and status = 'incomplete'."""
    raw_incomplete = (
        "============================= test session starts =============================\n"
        "tests/test_codec.py::test_codec_ignores_comments_and_blank_lines PASSED  [  2%]\n"
        "tests/test_codec.py::test_codec_handles_done_sentinel PASSED             [  4%]\n"
        "=== 2 passed in 0.12s ==="
    )
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw_incomplete,
        passed=2,
        failed=0,
        rc=1,
        expected_tests=45,
    )

    assert status == "incomplete"
    assert modules_fixed == "0/5"
    assert solved is False
    assert passing == []


def test_score_final_evaluation_collection_error_yields_zero_modules():
    """Verify that a collection failure (0 passed, 0 failed, rc=2) scores 0/5 and status = 'incomplete'."""
    raw_collection_error = (
        "============================= test session starts =============================\n"
        "ERROR collecting tests/test_radix.py\n"
        "SyntaxError: invalid syntax\n"
        "=== 1 error in 0.05s ==="
    )
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw_collection_error,
        passed=0,
        failed=0,
        rc=2,
        expected_tests=45,
    )

    assert status == "incomplete"
    assert modules_fixed == "0/5"
    assert solved is False
    assert passing == []


def test_score_final_evaluation_complete_full_pass():
    """Verify that a 45/45 pass with rc=0 scores 5/5, solved=True, and status = 'complete'."""
    raw_full = "============================= 45 passed in 0.85s =============================\n"
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw_full,
        passed=45,
        failed=0,
        rc=0,
        expected_tests=45,
    )

    assert status == "complete"
    assert modules_fixed == "5/5"
    assert solved is True
    assert set(passing) == set(ALL_SUBSYSTEMS)


def test_score_final_evaluation_complete_partial_failures():
    """Verify that a run with 43 passed and 2 failed in codec scores 4/5 and status = 'complete'."""
    raw_partial = (
        "FAILED tests/test_codec.py::test_codec_concatenates_single_tool_call_arguments_across_chunks\n"
        "FAILED tests/test_codec.py::test_codec_concatenates_multiple_indexed_tool_calls\n"
        "============================= 2 failed, 43 passed in 0.90s =============================\n"
    )
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw_partial,
        passed=43,
        failed=2,
        rc=1,
        expected_tests=45,
    )

    assert status == "complete"
    assert modules_fixed == "4/5"
    assert solved is False
    assert "codec" not in passing
    assert len(passing) == 4


def test_extract_subsystem_status_timeout_guard():
    """Verify extract_subsystem_status returns no passing modules when output contains [TIMEOUT]."""
    passing, failing = extract_subsystem_status(
        "[TIMEOUT] pytest exceeded 35s limit",
        passed=0,
        failed=45,
    )
    assert passing == []
    assert set(failing) == set(ALL_SUBSYSTEMS)


def test_extract_subsystem_status_incomplete_guard():
    """Verify extract_subsystem_status returns no passing modules when passed + failed != 45."""
    passing, failing = extract_subsystem_status(
        "no explicit failed lines here",
        passed=10,
        failed=5,
        expected_tests=45,
    )
    assert passing == []
    assert set(failing) == set(ALL_SUBSYSTEMS)


def test_task_arm_result_serialization_contains_final_eval_status():
    """Verify TaskArmResult includes final_eval_status and serializes properly to dict."""
    res = TaskArmResult(
        task_id="TEST-1",
        arm="cacheforge",
        model="test-model",
        solved=False,
        modules_fixed="0/5",
        passed_tests=0,
        failed_tests=45,
        turns=40,
        total_prompt_tokens=1000,
        peak_turn_prompt_tokens=500,
        total_completion_tokens=100,
        upstream_cached_tokens=800,
        avg_ttft_ms=500.0,
        p50_ttft_ms=500.0,
        p95_ttft_ms=500.0,
        per_turn_ttft_ms=[500.0],
        per_turn_prompt_tokens=[500],
        cache_hits=1,
        cache_hit_rate_pct=100.0,
        prefix_matches=1,
        prefix_match_rate_pct=100.0,
        context_overflow=False,
        final_eval_status="timeout",
    )
    d = asdict(res)
    assert d["final_eval_status"] == "timeout"
    assert d["modules_fixed"] == "0/5"


def test_score_final_evaluation_non_45_count():
    """Verify that score_final_evaluation correctly handles non-45 test counts and non-5 trap counts."""
    raw = "============================= 20 passed in 0.50s =============================\n"
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw,
        passed=20,
        failed=0,
        rc=0,
        expected_tests=20,
        trap_count=4,
        subsystems=("alpha", "beta", "gamma", "delta"),
    )
    assert status == "complete"
    assert modules_fixed == "4/4"
    assert solved is True
    assert len(passing) == 4


def test_score_final_evaluation_from_manifest_dict():
    """Verify that test count and trap count are extracted from a manifest dictionary."""
    manifest = {
        "test_count": 60,
        "trap_count": 6,
        "subsystems": ["a", "b", "c", "d", "e", "f"],
    }
    # 1. Complete pass
    raw_pass = "============================= 60 passed in 1.20s =============================\n"
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw_pass,
        passed=60,
        failed=0,
        rc=0,
        manifest=manifest,
    )
    assert status == "complete"
    assert modules_fixed == "6/6"
    assert solved is True
    assert len(passing) == 6

    # 2. Incomplete run (passed + failed != test_count)
    status_inc, modules_inc, solved_inc, _ = score_final_evaluation(
        raw_output="58 passed in 1.0s",
        passed=58,
        failed=0,
        rc=1,
        manifest=manifest,
    )
    assert status_inc == "incomplete"
    assert modules_inc == "0/6"
    assert solved_inc is False

    # 3. Timeout run
    status_to, modules_to, solved_to, _ = score_final_evaluation(
        raw_output="[TIMEOUT] exceeded limit",
        passed=0,
        failed=60,
        rc=1,
        manifest=manifest,
    )
    assert status_to == "timeout"
    assert modules_to == "0/6"
    assert solved_to is False


def test_score_final_evaluation_from_manifest_object():
    """Verify that test count and trap count are extracted from a manifest object/dataclass."""

    class CustomManifest:
        def __init__(self) -> None:
            self.test_count = 100
            self.trap_count = 10
            self.subsystems = [f"mod_{i}" for i in range(10)]

    manifest_obj = CustomManifest()
    raw = "============================= 100 passed in 2.0s =============================\n"
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output=raw,
        passed=100,
        failed=0,
        rc=0,
        manifest=manifest_obj,
    )
    assert status == "complete"
    assert modules_fixed == "10/10"
    assert solved is True
    assert len(passing) == 10


def test_score_junit_xml_missing_scores_zero_traps(tmp_path):
    """Verify that a missing JUnit XML file scores 0 traps and returns 'incomplete'."""
    missing_xml = tmp_path / "does_not_exist.xml"
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output="some output",
        passed=45,
        failed=0,
        rc=0,
        expected_tests=45,
        trap_count=5,
        junit_xml_path=missing_xml,
    )
    assert status == "incomplete"
    assert modules_fixed == "0/5"
    assert solved is False
    assert passing == []


def test_score_junit_xml_truncated_scores_zero_traps(tmp_path):
    """Verify that a truncated/malformed JUnit XML file scores 0 traps and returns 'incomplete'."""
    bad_xml = tmp_path / "malformed.xml"
    bad_xml.write_text("<testsuite tests='45'", encoding="utf-8")
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output="some output",
        passed=45,
        failed=0,
        rc=0,
        expected_tests=45,
        trap_count=5,
        junit_xml_path=bad_xml,
    )
    assert status == "incomplete"
    assert modules_fixed == "0/5"
    assert solved is False
    assert passing == []


def test_score_junit_xml_incomplete_count_scores_zero_traps(tmp_path):
    """Verify that a JUnit XML reporting fewer tests than expected scores 0 traps."""
    inc_xml = tmp_path / "incomplete.xml"
    inc_xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuite name="pytest" tests="20" failures="0" errors="0" skipped="0">\n'
        '  <testcase classname="tests.test_radix" name="test_1" />\n'
        "</testsuite>",
        encoding="utf-8",
    )
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output="some output",
        passed=20,
        failed=0,
        rc=0,
        expected_tests=45,
        trap_count=5,
        junit_xml_path=inc_xml,
    )
    assert status == "incomplete"
    assert modules_fixed == "0/5"
    assert solved is False
    assert passing == []


def test_score_junit_xml_complete_pass(tmp_path):
    """Verify that a valid complete pass in JUnit XML scores 5/5 and solved=True."""
    full_xml = tmp_path / "full_pass.xml"
    full_xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuite name="pytest" tests="45" failures="0" errors="0" skipped="0">\n'
        '  <testcase classname="tests.test_radix" name="test_1" file="tests/test_radix.py" />\n'
        "</testsuite>",
        encoding="utf-8",
    )
    status, modules_fixed, solved, passing = score_final_evaluation(
        raw_output="some output",
        passed=45,
        failed=0,
        rc=0,
        expected_tests=45,
        trap_count=5,
        junit_xml_path=full_xml,
    )
    assert status == "complete"
    assert modules_fixed == "5/5"
    assert solved is True
    assert set(passing) == set(ALL_SUBSYSTEMS)


def test_coupled_edit_file_tests_read_only(tmp_path):
    """Verify that editing tests/ is rejected with 'Error: tests are read-only.' in coupled task_set."""
    test_file = tmp_path / "tests" / "test_radix.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_dummy(): pass\n", encoding="utf-8")

    out, finished, _tests_passing = execute_tool_on_workspace(
        tmp_path,
        tool_name="edit_file",
        args={
            "path": "tests/test_radix.py",
            "old_text": "def test_dummy(): pass",
            "new_text": "def test_dummy(): assert True",
        },
        task_set="coupled",
    )
    assert out == "Error: tests are read-only."
    assert finished is False


def test_update_regression_bookkeeping():
    """Verify regression tracking correctly records turn introduced and turn repaired."""
    initial_passing = {
        "tests/test_units.py::test_safe_ratio",
        "tests/test_radix.py::test_radix_match",
    }
    regressions_introduced: list[str] = []
    turns_to_repair: dict[str, dict[str, int]] = {}

    # Turn 1: all passing
    results_t1 = {
        "tests/test_units.py::test_safe_ratio": True,
        "tests/test_radix.py::test_radix_match": True,
    }
    update_regression_bookkeeping(
        results_t1, initial_passing, 1, regressions_introduced, turns_to_repair
    )
    assert regressions_introduced == []
    assert turns_to_repair == {}

    # Turn 2: test_safe_ratio regresses (fails)
    results_t2 = {
        "tests/test_units.py::test_safe_ratio": False,
        "tests/test_radix.py::test_radix_match": True,
    }
    update_regression_bookkeeping(
        results_t2, initial_passing, 2, regressions_introduced, turns_to_repair
    )
    assert regressions_introduced == ["tests/test_units.py::test_safe_ratio"]
    assert turns_to_repair["tests/test_units.py::test_safe_ratio"] == {"introduced_turn": 2}

    # Turn 3: test_safe_ratio is repaired (passes)
    results_t3 = {
        "tests/test_units.py::test_safe_ratio": True,
        "tests/test_radix.py::test_radix_match": True,
    }
    update_regression_bookkeeping(
        results_t3, initial_passing, 3, regressions_introduced, turns_to_repair
    )
    assert turns_to_repair["tests/test_units.py::test_safe_ratio"] == {
        "introduced_turn": 2,
        "repaired_turn": 3,
    }


def test_score_coupled_evaluation():
    """Verify coupled trap scoring requires all F2P and guard tests to pass."""
    manifest = {
        "initial_passing_tests": ["tests/test_units.py::test_guard"],
        "traps": {
            "T1": {
                "f2p_tests": ["tests/test_telemetry.py::test_f2p"],
                "guard_tests": ["tests/test_units.py::test_guard"],
            },
            "T2": {
                "f2p_tests": ["tests/test_pipeline.py::test_f2p"],
                "guard_tests": [],
            },
        },
    }

    # Case 1: Incomplete or timeout status -> 0 traps solved
    cnt, solved, _rep, intact = score_coupled_evaluation(
        final_test_results={"tests/test_telemetry.py::test_f2p": True},
        manifest=manifest,
        regressions_introduced=[],
        final_eval_status="incomplete",
    )
    assert cnt == 0 and solved == [] and not intact

    # Case 2: T1 F2P passes, but its guard fails -> T1 NOT solved
    cnt2, solved2, rep2, intact2 = score_coupled_evaluation(
        final_test_results={
            "tests/test_telemetry.py::test_f2p": True,
            "tests/test_units.py::test_guard": False,
            "tests/test_pipeline.py::test_f2p": True,
        },
        manifest=manifest,
        regressions_introduced=["tests/test_units.py::test_guard"],
        final_eval_status="complete",
    )
    assert cnt2 == 1
    assert solved2 == ["T2"]
    assert rep2 == 0
    assert intact2 is False

    # Case 3: All F2P and guard tests pass -> Both T1 and T2 solved, initial pass intact
    cnt3, solved3, rep3, intact3 = score_coupled_evaluation(
        final_test_results={
            "tests/test_telemetry.py::test_f2p": True,
            "tests/test_units.py::test_guard": True,
            "tests/test_pipeline.py::test_f2p": True,
        },
        manifest=manifest,
        regressions_introduced=["tests/test_units.py::test_guard"],
        final_eval_status="complete",
    )
    assert cnt3 == 2
    assert set(solved3) == {"T1", "T2"}
    assert rep3 == 1
    assert intact3 is True


def test_classify_upstream_error_distinguishes_infra_from_overflow():
    """Verify that connection refused, timeouts, and 502s are classified as infra_error, not overflow."""
    # Infrastructure error cases
    assert (
        classify_upstream_error(
            ConnectionRefusedError(
                "[WinError 10061] No connection could be made because the target machine actively refused it"
            )
        )
        == "infra_error"
    )
    assert (
        classify_upstream_error(
            "HTTP 502: Bad Gateway: Unable to connect to upstream inference server: [WinError 10061]"
        )
        == "infra_error"
    )
    assert classify_upstream_error("HTTP 503: Service Unavailable") == "infra_error"
    assert classify_upstream_error("HTTP 504: Gateway Timeout") == "infra_error"
    assert (
        classify_upstream_error(
            "RemoteDisconnected('Remote end closed connection without response')"
        )
        == "infra_error"
    )
    assert (
        classify_upstream_error("ConnectionResetError: Connection reset by peer") == "infra_error"
    )

    # Context overflow cases
    assert (
        classify_upstream_error(
            "HTTP 400: request (135000 tokens) exceeds context size (131072 tokens)"
        )
        == "context_overflow"
    )
    assert (
        classify_upstream_error(
            "HTTP 400: {'error': {'code': 'context_length_exceeded', 'message': 'Maximum context length exceeded'}}"
        )
        == "context_overflow"
    )

    # Other API errors
    assert classify_upstream_error("HTTP 401: Unauthorized") == "other"
    assert classify_upstream_error("HTTP 404: Not Found") == "other"


def test_score_coupled_evaluation_infra_error_scores_zero():
    """Verify that final_eval_status='infra_error' scores 0 traps and intact=False."""
    manifest = {"traps": {"T1": {"f2p_tests": ["tests/test_t1.py::test_pass"], "guard_tests": []}}}
    cnt, solved, _rep, intact = score_coupled_evaluation(
        final_test_results={"tests/test_t1.py::test_pass": True},
        manifest=manifest,
        regressions_introduced=[],
        final_eval_status="infra_error",
    )
    assert cnt == 0
    assert solved == []
    assert intact is False


def test_comparison_result_invalidated_by_infra_error():
    """Verify that an infra_error in any arm sets is_valid=False in comparison summary."""
    bad_result = TaskArmResult(
        task_id="GAUNTLET-COUPLED",
        arm="baseline",
        model="dummy-model",
        solved=False,
        modules_fixed="0/5",
        passed_tests=0,
        failed_tests=56,
        turns=1,
        total_prompt_tokens=781,
        peak_turn_prompt_tokens=781,
        total_completion_tokens=0,
        upstream_cached_tokens=0,
        avg_ttft_ms=0.0,
        p50_ttft_ms=0.0,
        p95_ttft_ms=0.0,
        per_turn_ttft_ms=[],
        per_turn_prompt_tokens=[781],
        cache_hits=0,
        cache_hit_rate_pct=0.0,
        prefix_matches=0,
        prefix_match_rate_pct=0.0,
        context_overflow=False,
        final_eval_status="infra_error",
        infra_error=True,
        infra_error_msg="[WinError 10061] No connection could be made",
    )
    summary = _build_scorecard_dict(
        mode_label="Continuous 5-Trap Coupled Gauntlet",
        model_name="dummy-model",
        base_url="http://127.0.0.1:8083/v1",
        baseline_results=[bad_result],
        cacheforge_results=[],
    )
    assert summary["is_valid"] is False
    assert "infrastructure error" in summary["invalidation_reason"].lower()
    assert summary["baseline"]["context_overflow_count"] == 0


class _MockUpstreamServer(http.server.HTTPServer):
    def __init__(self, addr, handler_cls, responses):
        super().__init__(addr, handler_cls)
        self.responses = responses
        self.req_index = 0
        self.handled_requests = []


class _MockUpstreamHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        server: _MockUpstreamServer = self.server  # type: ignore
        server.handled_requests.append(body)

        idx = min(server.req_index, len(server.responses) - 1)
        status, headers, resp_body = server.responses[idx]
        server.req_index += 1

        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        if isinstance(resp_body, str):
            resp_body = resp_body.encode("utf-8")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def start_mock_upstream(responses: list[tuple[int, dict[str, str], bytes | str]]):
    server = _MockUpstreamServer(("127.0.0.1", 0), _MockUpstreamHandler, responses)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield url, server
    finally:
        server.shutdown()
        server.server_close()
        _reset_persistent_conn(url)


def test_mock_upstream_429_then_200_completes_turn(monkeypatch):
    """Verify that a 429 followed by 200 retries with backoff and completes the turn."""
    monkeypatch.setattr(harness, "_RETRY_BASE_BACKOFF_S", 0.001)
    responses = [
        (
            429,
            {"Content-Type": "application/json", "Retry-After": "0.01"},
            b'{"error": "rate limit"}',
        ),
        (
            200,
            {"Content-Type": "text/event-stream"},
            b'data: {"choices": [{"delta": {"content": "turn completed successfully"}}]}\n\ndata: [DONE]\n\n',
        ),
    ]
    with start_mock_upstream(responses) as (url, server):
        resp_text, _tcs, _meta = stream_chat_completion_sse(
            base_url=url,
            api_key="",
            payload={"messages": [{"role": "user", "content": "hello"}]},
            prefix_hash="",
            seen_prefixes=set(),
        )
        assert resp_text == "turn completed successfully"
        assert server.req_index == 2


def test_mock_upstream_402_aborts_run(monkeypatch):
    """Verify that HTTP 402 raises AuthQuotaAbortError immediately without retry."""
    monkeypatch.setattr(harness, "_RETRY_BASE_BACKOFF_S", 0.001)
    responses = [
        (
            402,
            {"Content-Type": "application/json"},
            b'{"error": "payment required: quota exhausted"}',
        ),
    ]
    with start_mock_upstream(responses) as (url, server):
        with pytest.raises(AuthQuotaAbortError) as exc_info:
            stream_chat_completion_sse(
                base_url=url,
                api_key="",
                payload={"messages": [{"role": "user", "content": "hello"}]},
                prefix_hash="",
                seen_prefixes=set(),
            )
        assert "402" in str(exc_info.value)
        assert server.req_index == 1


def test_mock_upstream_context_length_400_is_overflow(tmp_path, monkeypatch):
    """Verify that an HTTP 400 context length exceeded error is classified as context_overflow."""
    monkeypatch.setattr(harness, "_RETRY_BASE_BACKOFF_S", 0.001)
    responses = [
        (
            400,
            {"Content-Type": "application/json"},
            b'{"error": {"code": "context_length_exceeded", "message": "request (35000 tokens) exceeds context size (32768)"}}',
        ),
    ]
    with start_mock_upstream(responses) as (url, _server):
        res = _execute_agent_session(
            session_id="test_ovf",
            arm="baseline",
            tmp_ws=tmp_path,
            initial_user_prompt="Run pytest",
            base_url=url,
            api_key="",
            model_name="test-model",
            max_turns=1,
            seen_prefixes=set(),
        )
        assert res.context_overflow is True
        assert res.infra_error is False


def test_mock_upstream_five_500s_make_session_infra_error(tmp_path, monkeypatch):
    """Verify that five 500s in a row make the session an infra_error, not a context_overflow."""
    monkeypatch.setattr(harness, "_RETRY_BASE_BACKOFF_S", 0.001)
    responses = [
        (500, {"Content-Type": "application/json"}, b'{"error": "internal server error"}'),
    ]
    with start_mock_upstream(responses) as (url, server):
        res = _execute_agent_session(
            session_id="test_500",
            arm="baseline",
            tmp_ws=tmp_path,
            initial_user_prompt="Run pytest",
            base_url=url,
            api_key="",
            model_name="test-model",
            max_turns=1,
            seen_prefixes=set(),
        )
        assert res.infra_error is True
        assert res.context_overflow is False
        assert server.req_index == 5


def _make_repeat_json(
    rep: int,
    b_fixed: int = 2,
    c_fixed: int = 4,
    is_valid: bool = True,
    infra_error: bool = False,
) -> dict:
    return {
        "rep": rep,
        "is_valid": is_valid,
        "baseline": {
            "task_results": [
                {
                    "task_id": "GAUNTLET-5BUG",
                    "modules_fixed": f"{b_fixed}/5",
                    "turns": 20,
                    "context_overflow": False,
                    "infra_error": infra_error,
                    "total_prompt_tokens": 100000,
                    "total_completion_tokens": 500,
                    "peak_turn_prompt_tokens": 10000,
                    "upstream_cached_tokens": 80000,
                    "avg_ttft_ms": 500.0,
                }
            ]
        },
        "cacheforge": {
            "task_results": [
                {
                    "task_id": "GAUNTLET-5BUG",
                    "modules_fixed": f"{c_fixed}/5",
                    "turns": 20,
                    "context_overflow": False,
                    "infra_error": False,
                    "total_prompt_tokens": 40000,
                    "total_completion_tokens": 450,
                    "peak_turn_prompt_tokens": 5000,
                    "upstream_cached_tokens": 30000,
                    "avg_ttft_ms": 300.0,
                }
            ]
        },
    }


def test_analyze_repeat_files_replaces_infra_error_with_rerun(tmp_path):
    """Verify that an invalid infra-error repeat is replaced by its valid _rerun1_repN file."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
        analyze,
        resolve_repeat_files,
    )

    (tmp_path / "run_rep1.json").write_text(
        json.dumps(_make_repeat_json(1, b_fixed=1, c_fixed=3)), encoding="utf-8"
    )
    (tmp_path / "run_rep2.json").write_text(
        json.dumps(_make_repeat_json(2, b_fixed=0, c_fixed=0, is_valid=False, infra_error=True)),
        encoding="utf-8",
    )
    (tmp_path / "run_rerun1_rep2.json").write_text(
        json.dumps(_make_repeat_json(2, b_fixed=2, c_fixed=5)), encoding="utf-8"
    )
    (tmp_path / "run_rep3.json").write_text(
        json.dumps(_make_repeat_json(3, b_fixed=3, c_fixed=4)), encoding="utf-8"
    )

    selected, skipped = resolve_repeat_files("run", results_dir=tmp_path, expected_repeats=3)
    assert [p.name for _, p, _ in selected] == [
        "run_rep1.json",
        "run_rerun1_rep2.json",
        "run_rep3.json",
    ]
    assert [p.name for p in skipped] == ["run_rep2.json"]

    records = analyze("run", results_dir=tmp_path, expected_repeats=3)
    assert [r["b_fixed"] for r in records] == [1, 2, 3]
    assert [r["c_fixed"] for r in records] == [3, 5, 4]


def test_analyze_repeat_files_fails_on_gap(tmp_path):
    """Verify that missing or un-replaced invalid repeat files fail with a gap ValueError."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
        resolve_repeat_files,
    )

    (tmp_path / "run_rep1.json").write_text(json.dumps(_make_repeat_json(1)), encoding="utf-8")
    (tmp_path / "run_rep2.json").write_text(
        json.dumps(_make_repeat_json(2, is_valid=False, infra_error=True)), encoding="utf-8"
    )
    (tmp_path / "run_rep3.json").write_text(json.dumps(_make_repeat_json(3)), encoding="utf-8")

    with pytest.raises(ValueError, match="Gap at repeat 2"):
        resolve_repeat_files("run", results_dir=tmp_path, expected_repeats=3)


def test_analyze_repeat_files_fails_on_duplicate(tmp_path):
    """Verify that two valid files for the same repeat number fail with a duplicate ValueError."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
        resolve_repeat_files,
    )

    (tmp_path / "run_rep1.json").write_text(json.dumps(_make_repeat_json(1)), encoding="utf-8")
    (tmp_path / "run_rep2.json").write_text(json.dumps(_make_repeat_json(2)), encoding="utf-8")
    (tmp_path / "run_rerun1_rep2.json").write_text(
        json.dumps(_make_repeat_json(2)), encoding="utf-8"
    )
    (tmp_path / "run_rep3.json").write_text(json.dumps(_make_repeat_json(3)), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate at repeat 2"):
        resolve_repeat_files("run", results_dir=tmp_path, expected_repeats=3)


def test_analyze_nudge_invalid_repeat_kept_in_primary_excluded_from_valid_only(tmp_path):
    """Verify that a nudge-invalid repeat (stuck agent) resolves without error, is counted in primary analysis, and is excluded from valid-only analysis."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
        analyze,
        resolve_repeat_files,
    )

    rep1 = _make_repeat_json(1, b_fixed=2, c_fixed=4, is_valid=True, infra_error=False)
    rep2 = _make_repeat_json(2, b_fixed=0, c_fixed=3, is_valid=False, infra_error=False)
    rep2["baseline"]["nudge_count"] = 6
    rep2["baseline"]["nudge_rate_pct"] = 30.0
    rep2["baseline"]["task_results"][0]["nudge_count"] = 6
    rep2["baseline"]["task_results"][0]["nudge_rate_pct"] = 30.0
    rep3 = _make_repeat_json(3, b_fixed=3, c_fixed=5, is_valid=True, infra_error=False)

    (tmp_path / "run_rep1.json").write_text(json.dumps(rep1), encoding="utf-8")
    (tmp_path / "run_rep2.json").write_text(json.dumps(rep2), encoding="utf-8")
    (tmp_path / "run_rep3.json").write_text(json.dumps(rep3), encoding="utf-8")

    selected, skipped = resolve_repeat_files("run", results_dir=tmp_path, expected_repeats=3)
    assert len(selected) == 3
    assert len(skipped) == 0

    records = analyze("run", results_dir=tmp_path, expected_repeats=3)
    assert len(records) == 3
    assert [r["b_fixed"] for r in records] == [2, 0, 3]
    assert [r["c_fixed"] for r in records] == [4, 3, 5]

    valid_only = [r for r in records if r["valid"]]
    assert len(valid_only) == 2
    assert [r["rep"] for r in valid_only] == [1, 3]


def test_score_junit_xml_with_real_pytest_execution(tmp_path):
    """Generate JUnit XML by running real pytest on a workspace with 3 of 5 module bugs injected."""
    from benchmarks.tensorroute_benchmark.harness import (
        inject_gauntlet_bugs,
        run_workspace_pytest,
        score_final_evaluation,
    )
    from benchmarks.tensorroute_benchmark.repo_builder import materialize_clean_repo
    from benchmarks.tensorroute_benchmark.tasks import BENCHMARK_TASKS

    materialize_clean_repo(tmp_path, arm="arm_a")
    # Inject bugs in first 3 modules (radix, scheduler, limiter), leaving codec and telemetry passing
    inject_gauntlet_bugs(tmp_path, BENCHMARK_TASKS[0:3])

    xml_path = tmp_path / "real_pytest_junit.xml"
    rc, raw, passed, failed = run_workspace_pytest(tmp_path, junit_xml_path=str(xml_path))
    assert xml_path.is_file()
    assert passed + failed == 45
    assert failed > 0

    status, mod_str, solved, passing_subs = score_final_evaluation(
        raw_output=raw,
        passed=passed,
        failed=failed,
        rc=rc,
        expected_tests=45,
        trap_count=5,
        junit_xml_path=xml_path,
    )
    assert status == "complete"
    assert mod_str == "2/5"
    assert solved is False
    assert set(passing_subs) == {"codec", "telemetry"}


def test_analyze_original_task_file_with_default_traps_solved_zero(tmp_path):
    """Verify that original-task files carrying traps_solved=0 use modules_fixed, not traps_solved."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import analyze

    rep1 = _make_repeat_json(1, b_fixed=3, c_fixed=5, is_valid=True, infra_error=False)
    rep1["baseline"]["task_results"][0]["task_id"] = "GAUNTLET-WAVE-1"
    rep1["baseline"]["task_results"][0]["traps_solved"] = 0
    rep1["cacheforge"]["task_results"][0]["task_id"] = "GAUNTLET-WAVE-1"
    rep1["cacheforge"]["task_results"][0]["traps_solved"] = 0

    (tmp_path / "orig_rep1.json").write_text(json.dumps(rep1), encoding="utf-8")

    records = analyze("orig", results_dir=tmp_path, expected_repeats=1)
    assert len(records) == 1
    assert records[0]["is_coupled"] is False
    assert records[0]["b_fixed"] == 3
    assert records[0]["c_fixed"] == 5


def test_stream_chat_completion_sse_retries_connection_aborted_error(monkeypatch):
    """Verify that ConnectionAbortedError (WinError 10053) on keep-alive socket triggers reconnect retry."""
    monkeypatch.setattr(harness, "_RETRY_BASE_BACKOFF_S", 0.001)
    responses = [
        (
            200,
            {"Content-Type": "text/event-stream"},
            b'data: {"choices": [{"delta": {"content": "recovered after 10053"}}]}\n\ndata: [DONE]\n\n',
        ),
    ]
    orig_get_conn = harness._get_persistent_conn
    call_count = {"n": 0}

    class _AbortingConn:
        def request(self, *_args, **_kwargs):
            raise ConnectionAbortedError(
                10053,
                "An established connection was aborted by the software in your host machine",
            )

    def _mock_get_conn(base_url: str):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _AbortingConn(), "/v1"
        return orig_get_conn(base_url)

    monkeypatch.setattr(harness, "_get_persistent_conn", _mock_get_conn)

    with start_mock_upstream(responses) as (url, server):
        resp_text, _tcs, _meta = stream_chat_completion_sse(
            base_url=url,
            api_key="",
            payload={"messages": [{"role": "user", "content": "hello"}]},
            prefix_hash="",
            seen_prefixes=set(),
        )
        assert resp_text == "recovered after 10053"
        assert call_count["n"] == 2
        assert server.req_index == 1


def test_classify_upstream_error_rep15_llamacpp_exceed_context_size(tmp_path, monkeypatch):
    """Verify that the exact rep15 llama.cpp exceed_context_size_error string is classified as context_overflow."""
    rep15_err_msg = (
        'HTTP 400: {"error":{"code":400,"message":"request (137848 tokens) exceeds the available context size '
        '(131072 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":137848,"n_ctx":131072}}'
    )
    assert classify_upstream_error(rep15_err_msg) == "context_overflow"

    monkeypatch.setattr(harness, "_RETRY_BASE_BACKOFF_S", 0.001)
    raw_body = (
        '{"error":{"code":400,"message":"request (137848 tokens) exceeds the available context size '
        '(131072 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":137848,"n_ctx":131072}}'
    )
    responses = [
        (400, {"Content-Type": "application/json"}, raw_body),
    ]
    with start_mock_upstream(responses) as (url, _server):
        res = _execute_agent_session(
            session_id="GAUNTLET-COUPLED",
            arm="baseline",
            tmp_ws=tmp_path,
            initial_user_prompt="Run pytest",
            base_url=url,
            api_key="",
            model_name="test-model",
            max_turns=1,
            seen_prefixes=set(),
        )
        assert res.context_overflow is True
        assert res.infra_error is False
        assert res.peak_turn_prompt_tokens == 137848
        assert res.per_turn_tools == ["overflow"]


def test_resolve_repeat_files_prefers_rescored_all_and_single(tmp_path, capsys):
    """Verify that when both _repN.json and _repN_rescored.json exist, resolve_repeat_files uses the rescored file and prints which repeats were substituted."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import (
        analyze,
        resolve_repeat_files,
    )

    # Case 1: All-rescored case (valid originals AND valid rescored files for all 3 repeats)
    dir_all = tmp_path / "all_rescored"
    dir_all.mkdir()
    for r in (1, 2, 3):
        (dir_all / f"run_rep{r}.json").write_text(
            json.dumps(_make_repeat_json(r, b_fixed=5, c_fixed=5)), encoding="utf-8"
        )
        (dir_all / f"run_rep{r}_rescored.json").write_text(
            json.dumps(_make_repeat_json(r, b_fixed=r, c_fixed=r + 1)), encoding="utf-8"
        )

    sel_all, skip_all = resolve_repeat_files("run", results_dir=dir_all, expected_repeats=3)
    out_all = capsys.readouterr().out
    assert [p.name for _, p, _ in sel_all] == [
        "run_rep1_rescored.json",
        "run_rep2_rescored.json",
        "run_rep3_rescored.json",
    ]
    assert skip_all == []
    assert "Substituted rescored repeat files (3):" in out_all
    assert "rep1 (run_rep1.json -> run_rep1_rescored.json)" in out_all
    assert "rep2 (run_rep2.json -> run_rep2_rescored.json)" in out_all
    assert "rep3 (run_rep3.json -> run_rep3_rescored.json)" in out_all

    recs_all = analyze("run", results_dir=dir_all, expected_repeats=3)
    assert [r["b_fixed"] for r in recs_all] == [1, 2, 3]
    assert [r["c_fixed"] for r in recs_all] == [2, 3, 4]

    # Case 2: Single-rescored case (rep1 and rep2 have only originals; rep3 has original + rescored)
    dir_single = tmp_path / "single_rescored"
    dir_single.mkdir()
    (dir_single / "run_rep1.json").write_text(
        json.dumps(_make_repeat_json(1, b_fixed=2, c_fixed=3)), encoding="utf-8"
    )
    (dir_single / "run_rep2.json").write_text(
        json.dumps(_make_repeat_json(2, b_fixed=3, c_fixed=3)), encoding="utf-8"
    )
    (dir_single / "run_rep3.json").write_text(
        json.dumps(_make_repeat_json(3, b_fixed=0, c_fixed=1, is_valid=False, infra_error=True)),
        encoding="utf-8",
    )
    (dir_single / "run_rep3_rescored.json").write_text(
        json.dumps(_make_repeat_json(3, b_fixed=2, c_fixed=1, is_valid=True, infra_error=False)),
        encoding="utf-8",
    )

    capsys.readouterr()
    sel_single, skip_single = resolve_repeat_files(
        "run", results_dir=dir_single, expected_repeats=3
    )
    out_single = capsys.readouterr().out
    assert [p.name for _, p, _ in sel_single] == [
        "run_rep1.json",
        "run_rep2.json",
        "run_rep3_rescored.json",
    ]
    assert [p.name for p in skip_single] == ["run_rep3.json"]
    assert (
        "Substituted rescored repeat files (1): rep3 (run_rep3.json -> run_rep3_rescored.json)"
        in out_single
    )


def test_harness_section5_instrumentation_and_sidecar(tmp_path):
    """Verify Section 5 instrumentation: wall_clock_s, per_turn_model_ms, per_turn_tool_ms, per_turn_tool_args, RunSift headers, and sidecar records."""
    from benchmarks.tensorroute_benchmark.repo_builder import materialize_clean_repo
    from benchmarks.tensorroute_benchmark.tasks import BENCHMARK_TASKS, inject_gauntlet_bugs

    materialize_clean_repo(tmp_path, arm="arm_b")
    inject_gauntlet_bugs(tmp_path, BENCHMARK_TASKS[0:5])

    responses = [
        (
            200,
            {
                "Content-Type": "text/event-stream",
                "x-cacheforge-condenser": "v2",
                "x-cacheforge-tool-chars-before": "0",
                "x-cacheforge-tool-chars-after": "0",
            },
            b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\\"path\\": \\"src/tensorroute/scheduler.py\\"}"}}]}}]}\n\ndata: [DONE]\n\n',
        ),
        (
            200,
            {
                "Content-Type": "text/event-stream",
                "x-cacheforge-condenser": "v2",
                "x-cacheforge-tool-chars-before": "25487",
                "x-cacheforge-tool-chars-after": "5615",
            },
            b'data: {"choices": [{"delta": {"content": "No tool call on turn 2"}}]}\n\ndata: [DONE]\n\n',
        ),
    ]

    sidecar: list[dict] = []
    with start_mock_upstream(responses) as (url, _server):
        res = _execute_agent_session(
            session_id="GAUNTLET-WAVE-1",
            arm="cacheforge",
            tmp_ws=tmp_path,
            initial_user_prompt="Fix bugs",
            base_url=url,
            api_key="",
            model_name="test-model",
            max_turns=2,
            seen_prefixes=set(),
            gateway_url=url,
            sidecar_records=sidecar,
        )

    assert res.wall_clock_s > 0.0
    assert len(res.per_turn_model_ms) == 2
    assert all(ms >= 0.0 for ms in res.per_turn_model_ms)
    assert len(res.per_turn_tool_ms) == 2
    assert res.per_turn_tool_ms[0] >= 0.0
    assert res.per_turn_tool_ms[1] == 0.0
    assert len(res.per_turn_tool_args) == 2
    assert json.loads(res.per_turn_tool_args[0]) == {"path": "src/tensorroute/scheduler.py"}
    assert res.per_turn_tool_args[1] == ""
    assert res.per_turn_condenser == ["v2", "v2"]
    assert res.per_turn_tool_chars_before == [0, 25487]
    assert res.per_turn_tool_chars_after == [0, 5615]
    assert "scheduler" in res.failing_subsystems

    assert len(sidecar) == 2
    assert sidecar[0]["arm"] == "cacheforge"
    assert sidecar[0]["turn"] == 1
    assert sidecar[0]["tool"] == "read_file"
    assert sidecar[0]["args"] == {"path": "src/tensorroute/scheduler.py"}
    assert "class ContinuousBatchScheduler" in sidecar[0]["raw_output"]
    assert sidecar[1]["turn"] == 2
    assert sidecar[1]["tool"] == "none"
    assert sidecar[1]["raw_output"] == ""


def test_analyze_reports_section5_timing_read_file_and_scheduler_metrics(tmp_path):
    """Verify analyze_gauntlet_results.py computes wall_clock_s, model/tool time, read_file calls, and scheduler failures."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_gauntlet_results import analyze

    rep1 = _make_repeat_json(1, b_fixed=3, c_fixed=5, is_valid=True, infra_error=False)
    b_tr = rep1["baseline"]["task_results"][0]
    c_tr = rep1["cacheforge"]["task_results"][0]

    b_tr["wall_clock_s"] = 42.5
    b_tr["per_turn_model_ms"] = [10000.0, 15000.0, 5000.0]
    b_tr["per_turn_tool_ms"] = [2000.0, 500.0, 1500.0]
    b_tr["per_turn_tools"] = ["run_pytest", "read_file", "read_file"]
    b_tr["failing_subsystems"] = ["scheduler", "limiter"]

    c_tr["wall_clock_s"] = 28.0
    c_tr["per_turn_model_ms"] = [8000.0, 9000.0, 6000.0]
    c_tr["per_turn_tool_ms"] = [1800.0, 400.0, 1200.0]
    c_tr["per_turn_tools"] = ["run_pytest", "read_file", "edit_file"]
    c_tr["failing_subsystems"] = []

    (tmp_path / "sec5_rep1.json").write_text(json.dumps(rep1), encoding="utf-8")

    records = analyze("sec5", results_dir=tmp_path, expected_repeats=1)
    assert len(records) == 1
    r = records[0]
    assert r["b_wall_clock_s"] == 42.5
    assert r["c_wall_clock_s"] == 28.0
    assert r["b_model_s"] == 30.0
    assert r["c_model_s"] == 23.0
    assert r["b_tool_s"] == 4.0
    assert r["c_tool_s"] == 3.4
    assert r["b_read_file_calls"] == 2
    assert r["c_read_file_calls"] == 1
    assert r["b_scheduler_failed"] is True
    assert r["c_scheduler_failed"] is False


def test_harness_restores_edited_tests_before_final_eval_and_records_test_files_edited(tmp_path):
    """Regression test: if the agent edits or creates files under tests/, the harness records test_files_edited and restores/cleans tests/ before final eval."""
    import subprocess

    from benchmarks.tensorroute_benchmark.repo_builder import materialize_clean_repo
    from benchmarks.tensorroute_benchmark.tasks import BENCHMARK_TASKS, inject_gauntlet_bugs

    materialize_clean_repo(tmp_path, arm="arm_b")
    inject_gauntlet_bugs(tmp_path, BENCHMARK_TASKS[0:5])
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    test_codec_path = tmp_path / "tests" / "test_codec.py"
    original_test_codec = test_codec_path.read_text(encoding="utf-8")
    new_test_path = tmp_path / "tests" / "test_extra_untracked.py"
    new_test_path.write_text("def test_extra():\n    assert False\n", encoding="utf-8")
    tampered = "\n".join(f"def test_fake_{i}(): assert True\n" for i in range(9))
    edit_args = json.dumps(
        {"path": "tests/test_codec.py", "old_text": original_test_codec, "new_text": tampered}
    )
    sse_edit = f'data: {{"choices": [{{"delta": {{"tool_calls": [{{"index": 0, "id": "call_1", "type": "function", "function": {{"name": "edit_file", "arguments": {json.dumps(edit_args)}}}}}]}}}}]}}\n\ndata: [DONE]\n\n'.encode()
    sse_fin = b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_2", "type": "function", "function": {"name": "finish", "arguments": "{\\"summary\\": \\"done\\"}"}}]}}]}\n\ndata: [DONE]\n\n'
    responses = [
        (200, {"Content-Type": "text/event-stream"}, sse_edit),
        (200, {"Content-Type": "text/event-stream"}, sse_fin),
    ]
    with start_mock_upstream(responses) as (url, _server):
        res = _execute_agent_session(
            "GAUNTLET-WAVE-1",
            "cacheforge",
            tmp_path,
            "Fix bugs",
            url,
            "",
            "test-model",
            2,
            set(),
            gateway_url=url,
        )
    assert res.test_files_edited == ["tests/test_codec.py", "tests/test_extra_untracked.py"]
    assert test_codec_path.read_text(encoding="utf-8") == original_test_codec
    assert not new_test_path.exists()
    assert "codec" in res.failing_subsystems
    assert res.modules_fixed == "0/5"
    assert res.solved is False


def test_analyze_three_arm_run_f_rescored(capsys):
    """Given Run F's rescored files as the baseline and v2.1 arms, analyze_three_arm prints 59 vs 39 and p = 0.0352."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_three_arm import (
        analyze_three_arm,
        print_three_arm_report,
    )

    summary = analyze_three_arm(
        prefix_base_v2="nebius_gauntlet_nemotron_30b_40turn_v2_1",
        prefix_v2_1="nebius_gauntlet_nemotron_30b_40turn_v2_1",
    )
    capsys.readouterr()
    print_three_arm_report(summary)
    out = capsys.readouterr().out

    assert summary["arms"]["baseline"]["total_modules_fixed"] == 59
    assert summary["arms"]["v2_1"]["total_modules_fixed"] == 39
    assert round(summary["tests"]["G3_v2_1_vs_baseline_modules"]["raw_p"], 4) == 0.0352
    assert summary["arms"]["baseline"]["first_post_pytest_no_tool_sessions"] == 0
    assert summary["arms"]["v2_1"]["first_post_pytest_no_tool_sessions"] == 5
    assert summary["arms"]["v2_1"]["first_post_pytest_no_tool_reps"] == [7, 8, 10, 11, 13]
    assert "59 vs 39" in out
    assert "p = 0.0352" in out
    assert "FirstPostPytestNoTool: 5/15 sessions [7, 8, 10, 11, 13]" in out


def test_analyze_three_arm_first_post_pytest_no_tool(tmp_path):
    """Verify Section 13.6 metric: sessions whose first reply after the first run_pytest had no tool call."""
    from benchmarks.tensorroute_benchmark.scripts.analyze_three_arm import analyze_three_arm

    bv2_rep1 = _make_repeat_json(1, b_fixed=3, c_fixed=4)
    # Baseline: first run_pytest on turn 1, followed by read_file on turn 2 (False)
    bv2_rep1["baseline"]["task_results"][0]["per_turn_tools"] = [
        "run_pytest",
        "read_file",
        "none",
    ]
    # v2: read_file on turn 1, first run_pytest on turn 2, followed by "none" on turn 3 (True)
    bv2_rep1["cacheforge"]["task_results"][0]["per_turn_tools"] = [
        "read_file",
        "run_pytest",
        "none",
    ]

    v21_rep1 = {
        "rep": 1,
        "cacheforge": {
            "task_results": [
                {
                    "task_id": "GAUNTLET-5BUG",
                    "modules_fixed": "2/5",
                    "turns": 3,
                    "context_overflow": False,
                    "infra_error": False,
                    "total_prompt_tokens": 30000,
                    "total_completion_tokens": 400,
                    "upstream_cached_tokens": 20000,
                    # First run_pytest is the last turn, so no subsequent turn (False)
                    "per_turn_tools": ["read_file", "edit_file", "run_pytest"],
                }
            ]
        },
    }

    (tmp_path / "bv2_rep1.json").write_text(json.dumps(bv2_rep1), encoding="utf-8")
    (tmp_path / "v21_rep1.json").write_text(json.dumps(v21_rep1), encoding="utf-8")

    summary = analyze_three_arm("bv2", "v21", results_dir=tmp_path, expected_repeats=1)
    assert summary["arms"]["baseline"]["first_post_pytest_no_tool_sessions"] == 0
    assert summary["arms"]["v2"]["first_post_pytest_no_tool_sessions"] == 1
    assert summary["arms"]["v2"]["first_post_pytest_no_tool_reps"] == [1]
    assert summary["arms"]["v2_1"]["first_post_pytest_no_tool_sessions"] == 0
