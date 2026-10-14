"""Unit tests for RunSift Compact and State subsystems.

Verifies negative memory compaction, JSON pruning, and async tool suspension state machine.
"""

import json
import time

from src.compact import ToolOutputCompactor
from src.state import SessionStatus, StateManager


def test_tool_output_compaction_passthrough():
    """Verify that outputs below threshold are untouched."""
    compactor = ToolOutputCompactor(max_char_threshold=500)
    short_text = "File successfully written to /tmp/output.txt"
    compacted, saved = compactor.compact_tool_output(short_text, tool_name="write_file")
    assert compacted == short_text
    assert saved == 0


def test_tool_output_compactor_large_log():
    """Verify large stdout/stderr with stack traces is compacted into head/tail summary."""
    compactor = ToolOutputCompactor(
        max_char_threshold=300, head_lines_budget=3, tail_lines_budget=3
    )
    noisy_log = "\n".join([f"Step {i}: processing item {i * 10}..." for i in range(100)])
    noisy_log += "\nFATAL ERROR: NullPointerException at module.py line 402\n"
    noisy_log += "\n".join([f"Traceback frame {j}" for j in range(20)])

    compacted, saved = compactor.compact_tool_output(noisy_log, tool_name="bash_exec")

    assert saved > 0
    assert "[TOOL EXECUTION SUMMARY :: bash_exec]" in compacted
    assert "Critical Diagnostic" in compacted
    assert "NullPointerException" in compacted
    assert "lines omitted by RunSift" in compacted


def test_tool_output_compactor_large_json():
    """Verify large JSON arrays are summarized with sample and count."""
    compactor = ToolOutputCompactor(max_char_threshold=200)
    huge_data = [{"id": i, "data": f"record_{i}", "extra": "x" * 50} for i in range(50)]
    raw_json = json.dumps(huge_data)

    compacted, saved = compactor.compact_tool_output(raw_json, tool_name="database_query")

    assert saved > 0
    assert "[COMPACTED TOOL OUTPUT :: database_query]" in compacted
    assert "Array of 50 items (compacted)" in compacted
    assert "47 items omitted to preserve KV-cache" in compacted


def test_state_manager_lifecycle():
    """Verify full state suspension, tool return recording, and continuation resumption."""
    manager = StateManager(session_ttl_seconds=3600)

    # 1. Create session
    session = manager.get_or_create_session(
        session_id="test_sess_01",
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[{"role": "user", "content": "Fetch database table."}],
        prefix_hash="hash_abc123",
    )
    assert session.status == SessionStatus.ACTIVE
    assert len(session.messages) == 1

    # 2. Suspend for tools
    tool_calls = [
        {
            "id": "call_db_1",
            "type": "function",
            "function": {"name": "query_db", "arguments": '{"query": "SELECT * FROM users"}'},
        }
    ]
    suspended = manager.suspend_for_tools(session_id="test_sess_01", tool_calls=tool_calls)
    assert suspended.status == SessionStatus.SUSPENDED_TOOL_WAIT
    assert "call_db_1" in suspended.pending_tool_calls

    # Simulate tool wait duration
    time.sleep(0.05)

    # 3. Record tool return
    tool_output = "User 1, User 2, User 3"
    updated = manager.record_tool_return(
        session_id="test_sess_01",
        tool_call_id="call_db_1",
        raw_output=tool_output,
    )
    assert updated.status == SessionStatus.READY
    assert len(updated.messages) == 2
    assert updated.messages[-1]["role"] == "tool"
    assert updated.messages[-1]["content"] == tool_output
    assert updated.total_suspended_seconds > 0.04

    # 4. Resume continuation
    messages, suspended_time = manager.resume_continuation("test_sess_01")
    assert len(messages) == 2
    assert suspended_time > 0.04
    assert updated.status == SessionStatus.ACTIVE


def test_compact_messages_user_preserved_and_tool_compacted():
    """Verify user messages are never truncated while tool outputs are compacted idempotently."""
    compactor = ToolOutputCompactor(
        max_char_threshold=300, head_lines_budget=3, tail_lines_budget=3
    )
    verbose_user_code = "\n".join([f"def user_function_{i:02d}(): pass" for i in range(40)])
    verbose_tool_log = "\n".join(
        [f"[TEST-LINE-{i:02d}] test_worker_{i:02d} ... PASSED" for i in range(40)]
        + ["ERROR: GPU worker connection refused on port 9000"]
        + [f"[STACK-{j:02d}] frame_{j:02d} in run_loop" for j in range(10)]
    )
    messages = [
        {"role": "system", "content": "You are an SRE assistant."},
        {"role": "user", "content": verbose_user_code},
        {"role": "tool", "name": "bash", "tool_call_id": "call_1", "content": verbose_tool_log},
        {"role": "user", "content": "Short follow-up question."},
    ]

    first_pass, saved_first = compactor.compact_messages(messages)
    assert saved_first > 0
    # User messages must remain 100% intact and uncompacted
    assert first_pass[1]["content"] == verbose_user_code
    assert first_pass[3]["content"] == "Short follow-up question."
    # Tool output must be compacted
    assert "[TOOL EXECUTION SUMMARY :: bash]" in first_pass[2]["content"]
    assert "ERROR: GPU worker connection refused on port 9000" in first_pass[2]["content"]

    # Second pass must be byte-for-byte identical (0 additional chars saved) to preserve KV-cache
    second_pass, saved_second = compactor.compact_messages(first_pass)
    assert saved_second == 0
    assert second_pass[2]["content"] == first_pass[2]["content"]


def test_compaction_spoofing_defense():
    """Verify that oversized payloads masquerading as already compacted are forced to compact."""
    compactor = ToolOutputCompactor(max_char_threshold=300)
    fake_header_payload = "[COMPACTED TOOL OUTPUT :: fake_tool]\n" + ("x" * 2000)
    compacted, saved = compactor.compact_tool_output(fake_header_payload, tool_name="bash")
    assert saved > 0
    assert len(compacted) < len(fake_header_payload)


def test_state_manager_lru_capacity_bound():
    """Verify that StateManager evicts oldest sessions when max_sessions capacity is exceeded."""
    manager = StateManager(session_ttl_seconds=3600, max_sessions=3)
    # Create 3 sessions
    for i in range(1, 4):
        manager.get_or_create_session(
            session_id=f"sess_{i}",
            model="nvidia/nemotron",
            messages=[],
        )
    assert len(manager._sessions) == 3
    assert "sess_1" in manager._sessions

    # Create 4th session -> sess_1 should be evicted as oldest
    manager.get_or_create_session(
        session_id="sess_4",
        model="nvidia/nemotron",
        messages=[],
    )
    assert len(manager._sessions) == 3
    assert "sess_1" not in manager._sessions
    assert "sess_4" in manager._sessions


def test_pytest_compaction_requested_target_passed_other_files_failing():
    """Verify that when a requested test target passes but other files in the suite fail,

    the compactor explicitly declares the target PASSED without emitting any hints.
    """
    compactor = ToolOutputCompactor()
    raw_pytest = (
        "============================= test session starts =============================\n"
        "rootdir: /workspace\n"
        "collected 21 items\n\n"
        "tests/test_radix.py::test_radix_insert PASSED [  5%]\n"
        "tests/test_radix.py::test_radix_evict PASSED [ 10%]\n"
        "tests/test_radix.py::test_radix_prefix PASSED [ 15%]\n"
        "tests/test_codec.py::test_codec_round_trip PASSED [ 20%]\n"
        "tests/test_codec.py::test_codec_corrupt FAILED [ 25%]\n"
        "tests/test_scheduler.py::test_schedule_drain FAILED [ 30%]\n\n"
        "================================== FAILURES ===================================\n"
        "______________________________ test_codec_corrupt ______________________________\n"
        "    def test_codec_corrupt():\n"
        ">       assert codec.decode(b'bad') == {}\n"
        "E       ValueError: Corrupted magic header\n"
        "tests/test_codec.py:45: ValueError\n"
        "_____________________________ test_schedule_drain ______________________________\n"
        "    def test_schedule_drain():\n"
        ">       assert len(queue) == 0\n"
        "E       assert 1 == 0\n"
        "tests/test_scheduler.py:88: AssertionError\n"
        "=========================== short test summary info ===========================\n"
        "FAILED tests/test_codec.py::test_codec_corrupt - ValueError: Corrupted magic header\n"
        "FAILED tests/test_scheduler.py::test_schedule_drain - assert 1 == 0\n"
        "======================== 2 failed, 4 passed in 0.15s =========================\n"
    )

    tool_args = '{"target": "tests/test_radix.py"}'
    compacted, saved = compactor.compact_tool_output(
        raw_pytest, tool_name="run_pytest", tool_args=tool_args
    )

    assert saved > 0
    # Explicit target result verification
    assert "Requested target 'tests/test_radix.py': PASSED (3 passed, 0 failed)" in compacted
    # Per-test-file summary counts
    assert "Test Files Summary:" in compacted
    assert "- tests/test_radix.py: 3 passed, 0 failed" in compacted
    assert "- tests/test_codec.py: 1 passed, 1 failed" in compacted
    assert "- tests/test_scheduler.py: 0 passed, 1 failed" in compacted
    # Failures preserved
    assert "ValueError: Corrupted magic header" in compacted
    assert "assert 1 == 0" in compacted
    # Summary line preserved
    assert (
        "======================== 2 failed, 4 passed in 0.15s ========================="
        in compacted
    )
    # No hints and no next target
    assert "hint" not in compacted.lower()
    assert "next target" not in compacted.lower()
    assert "<!-- cf:compacted -->" in compacted

    # Second pass must be idempotent
    second_pass, saved_second = compactor.compact_tool_output(
        compacted, tool_name="run_pytest", tool_args=tool_args
    )
    assert saved_second == 0
    assert second_pass == compacted


def test_pytest_compaction_requested_target_failing():
    """Verify that when a requested test target fails, it is explicitly reported as FAILED."""
    compactor = ToolOutputCompactor()
    raw_pytest = (
        "============================= test session starts =============================\n"
        "tests/test_radix.py::test_radix_insert PASSED [ 50%]\n"
        "tests/test_radix.py::test_radix_evict FAILED [100%]\n\n"
        "================================== FAILURES ===================================\n"
        "______________________________ test_radix_evict _______________________________\n"
        ">       assert trie.size == 0\n"
        "E       assert 1 == 0\n"
        "tests/test_radix.py:30: AssertionError\n"
        "=========================== short test summary info ===========================\n"
        "FAILED tests/test_radix.py::test_radix_evict - assert 1 == 0\n"
        "======================== 1 failed, 1 passed in 0.08s =========================\n"
    )

    tool_args = '{"target": "tests/test_radix.py"}'
    compacted, saved = compactor.compact_tool_output(
        raw_pytest, tool_name="run_pytest", tool_args=tool_args
    )

    assert saved > 0
    assert "Requested target 'tests/test_radix.py': FAILED (1 failed, 1 passed)" in compacted
    assert "assert 1 == 0" in compacted
    assert (
        "======================== 1 failed, 1 passed in 0.08s ========================="
        in compacted
    )


def test_pytest_compaction_messages_integration():
    """Verify compact_messages extracts tool_args from preceding assistant tool_calls and compacts role=tool."""
    compactor = ToolOutputCompactor()
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix radix.py."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_pytest_42",
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
            "tool_call_id": "call_pytest_42",
            "name": "run_pytest",
            "content": (
                "============================= test session starts =============================\n"
                "platform win32 -- Python 3.13.6 -- pytest-8.4.2\n"
                "rootdir: /workspace\n"
                "collected 10 items\n\n"
                + "\n".join(
                    [
                        f"tests/test_radix.py::test_radix_{i} PASSED [{i * 10}%]"
                        for i in range(1, 11)
                    ]
                )
                + "\n======================== 10 passed in 0.05s =========================\n"
            ),
        },
    ]

    compacted_msgs, total_saved = compactor.compact_messages(messages)
    assert total_saved > 0
    tool_msg = compacted_msgs[3]
    assert tool_msg["role"] == "tool"
    assert (
        "Requested target 'tests/test_radix.py': PASSED (10 passed, 0 failed)"
        in tool_msg["content"]
    )
    assert "<!-- cf:compacted -->" in tool_msg["content"]
