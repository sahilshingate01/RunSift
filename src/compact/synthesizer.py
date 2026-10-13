"""Negative memory synthesizer and tool output compactor.

Condenses verbose tool execution logs, terminal stdout/stderr dumps,
and large JSON payloads into high-density invariant summaries (<300 tokens),
stopping runaway context compounding in autonomous agent loops.
Allowed dependencies per wiring.yaml: ["core"].
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from src.core.models import ChatMessage

EXEMPT_TOOL_NAMES = {
    "read_file",
    "view_file",
    "get_file",
    "cat",
    "read",
    "fetch_file",
    "download_file",
    "open_file",
    "git_diff",
    "diff",
    "show_diff",
    "apply_patch",
    "ast_grep",
    "search_code",
    "grep",
    "search",
    "web_search",
    "search_web",
    "bing_search",
    "google_search",
    "file_search",
    "find_files",
    "list_dir",
    "ls",
}


class ToolOutputCompactor:
    """Invariant-based compactor condensing voluminous tool returns into cache-preserving summaries."""

    def __init__(
        self,
        max_char_threshold: int = 4000,  # ~1000 tokens
        head_lines_budget: int = 12,
        tail_lines_budget: int = 16,
    ):
        self.max_char_threshold = max_char_threshold
        self.head_lines_budget = head_lines_budget
        self.tail_lines_budget = tail_lines_budget

    def is_compaction_needed(self, text: str, tool_name: str = "tool", tool_args: str = "") -> bool:
        t_clean = (tool_name or "").lower().strip()
        if t_clean in EXEMPT_TOOL_NAMES or any(
            t_clean.startswith(p) for p in ("read_", "view_", "cat_", "search_", "find_", "web_")
        ):
            return False

        # If a shell tool (bash, run_shell_command) was used to inspect a file (cat, head, tail, etc.)
        args_clean = (tool_args or "").lower()
        if any(
            reader_cmd in args_clean
            for reader_cmd in ("cat ", "head ", "tail ", "less ", "more ", "type ", "grep ")
        ):
            return False

        if self._is_pytest_output(text, tool_name):
            if "<!-- cf:compacted -->" in text:
                return False
            return len(text.strip().splitlines()) > 1 or len(text) > 200

        # If output consists of source code without test/build failure indicators, preserve intact
        has_code_markers = any(
            m in text for m in ("def ", "class ", "import ", "return ", "function ", "#include ")
        )
        has_error_markers = any(
            err in text.lower()
            for err in (
                "failed",
                "failure",
                "traceback (most recent call last)",
                "error:",
                "exception:",
                "assertionerror",
                "fatal:",
            )
        )
        if has_code_markers and not has_error_markers:
            return False

        return len(text) > self.max_char_threshold

    def compact_tool_output(
        self,
        raw_output: str,
        tool_name: str = "tool",
        tool_call_id: str | None = None,
        tool_args: str = "",
    ) -> tuple[str, int]:
        """Condenses raw tool output if it exceeds threshold budget.

        Returns (compacted_text, characters_saved).
        """
        if not self.is_compaction_needed(raw_output, tool_name=tool_name, tool_args=tool_args):
            return raw_output, 0

        # Avoid re-compacting legitimately compacted output within budget
        if "<!-- cf:compacted -->" in raw_output and len(raw_output) <= self.max_char_threshold * 4:
            return raw_output, 0

        if raw_output.startswith(("[COMPACTED TOOL OUTPUT ::", "[TOOL EXECUTION SUMMARY ::")):
            if (
                "lines omitted" in raw_output
                or "items omitted" in raw_output
                or "<!-- cf:compacted -->" in raw_output
            ) and len(raw_output) <= self.max_char_threshold * 4:
                return raw_output, 0
            # If oversized content is disguised with compaction headers, strip the fake prefix and re-compact
            raw_output = raw_output.split("\n", 1)[-1]

        original_len = len(raw_output)

        # 1. Check if output is pytest
        if self._is_pytest_output(raw_output, tool_name):
            compacted_pytest = self._compact_pytest_output(raw_output, tool_args=tool_args)
            chars_saved = max(0, original_len - len(compacted_pytest))
            return compacted_pytest, chars_saved

        # 2. Check if output is JSON
        compacted_json = self._try_compact_json(raw_output, tool_name)
        if compacted_json is not None:
            chars_saved = max(0, original_len - len(compacted_json))
            return compacted_json, chars_saved

        # 3. Check if output is stack trace / build log
        compacted_log = self._compact_text_log(raw_output, tool_name)
        chars_saved = max(0, original_len - len(compacted_log))
        return compacted_log, chars_saved

    def _is_pytest_output(self, text: str, tool_name: str = "") -> bool:
        t_clean = (tool_name or "").lower().strip()
        return t_clean in ("run_pytest", "pytest")

    def _extract_target_from_args(self, tool_args: str) -> str | None:
        if not tool_args:
            return None
        try:
            data = json.loads(tool_args)
            if isinstance(data, dict):
                t = data.get("target")
                if t:
                    return str(t).replace("\\", "/").strip()
        except (json.JSONDecodeError, TypeError):
            pass
        m = re.search(r'"target"\s*:\s*"([^"]+)"', tool_args)
        if m:
            return m.group(1).replace("\\", "/").strip()
        return None

    def _extract_per_file_counts(self, text: str) -> dict[str, dict[str, int]]:
        file_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "failed": 0, "skipped": 0}
        )
        for line in text.splitlines():
            norm = line.replace("\\", "/").strip()
            m = re.match(
                r"^(?:tests/[^\s:]+\.py|\S+\.py)::(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)", norm
            )
            if m:
                fpath = norm.split("::")[0]
                status = m.group(2)
                if status == "PASSED":
                    file_counts[fpath]["passed"] += 1
                elif status in ("FAILED", "ERROR"):
                    file_counts[fpath]["failed"] += 1
                elif status == "SKIPPED":
                    file_counts[fpath]["skipped"] += 1
        for line in text.splitlines():
            norm = line.replace("\\", "/").strip()
            m = re.match(r"^FAILED\s+([^\s:]+\.py)::", norm)
            if m:
                fpath = m.group(1)
                if fpath not in file_counts:
                    file_counts[fpath]["failed"] += 1
        return dict(file_counts)

    def _extract_final_summary(self, text: str) -> str:
        for line in reversed(text.splitlines()):
            s = line.strip()
            if (
                s.startswith("=")
                and s.endswith("=")
                and any(k in s for k in ("passed", "failed", "error", "skipped"))
            ):
                return s
        return ""

    def _determine_target_status(
        self, target: str | None, file_counts: dict[str, dict[str, int]], text: str
    ) -> str | None:
        if not target:
            return None
        norm_t = target.replace("\\", "/").strip()
        matched_key = None
        for k in file_counts:
            if k == norm_t or k.endswith("/" + norm_t) or norm_t.endswith("/" + k) or (norm_t in k):
                matched_key = k
                break
        if matched_key:
            stats = file_counts[matched_key]
            p, f = stats["passed"], stats["failed"]
            if f == 0 and p > 0:
                return f"Requested target '{target}': PASSED ({p} passed, 0 failed)"
            elif f > 0:
                return f"Requested target '{target}': FAILED ({f} failed, {p} passed)"
            else:
                return f"Requested target '{target}': 0 tests run"
        if f"FAILED {norm_t}" in text.replace("\\", "/"):
            return f"Requested target '{target}': FAILED"
        return f"Requested target '{target}': status not identified in test output"

    def _extract_failure_excerpts(self, text: str) -> list[str]:
        lines = text.splitlines()
        failure_lines: list[str] = []
        in_failures = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("=== FAILURES ===", "=== ERRORS ===", "====")):
                if "FAILURES" in stripped or "ERRORS" in stripped:
                    in_failures = True
                    continue
                if (
                    "short test summary info" in stripped
                    or "failed" in stripped
                    or "error" in stripped
                    or "passed" in stripped
                    or "slowest" in stripped
                ):
                    in_failures = False
                    continue
            if in_failures:
                norm_stripped = stripped.replace("\\", "/")
                if (
                    stripped.startswith((">", "___", "E ", "FAILED ", "ERROR "))
                    or "tests/test_" in norm_stripped
                    or "src/tensorroute/" in norm_stripped
                ):
                    failure_lines.append(stripped[:300])
            elif stripped.startswith(("FAILED ", "ERROR ")):
                failure_lines.append(stripped[:300])
        return failure_lines

    def _compact_pytest_output(self, text: str, tool_args: str = "") -> str:
        target = self._extract_target_from_args(tool_args)
        file_counts = self._extract_per_file_counts(text)
        final_summary = self._extract_final_summary(text)
        target_status = self._determine_target_status(target, file_counts, text)
        failure_excerpts = self._extract_failure_excerpts(text)

        parts = ["[COMPACTED PYTEST OUTPUT]"]
        if target_status:
            parts.append(target_status)
        if file_counts:
            parts.append("Test Files Summary:")
            for fpath in sorted(file_counts.keys()):
                p = file_counts[fpath]["passed"]
                f = file_counts[fpath]["failed"]
                parts.append(f"- {fpath}: {p} passed, {f} failed")
        if failure_excerpts:
            parts.append("\n--- [Failures] ---")
            parts.extend(failure_excerpts)
        if final_summary:
            parts.append(f"\n{final_summary}")
        parts.append("<!-- cf:compacted -->")
        return "\n".join(parts)

    def _try_compact_json(self, text: str, tool_name: str) -> str | None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

        # Compact large lists of items (for non-exempt tools like database queries)
        if isinstance(data, list):
            total_items = len(data)
            if total_items > 6 and len(text) > self.max_char_threshold:
                sample_count = min(3, total_items)
                sample = data[:sample_count]
                omitted = total_items - sample_count
                summary = {
                    "_runsift_summary": f"Array of {total_items} items (compacted)",
                    "claim": f"Tool '{tool_name}' returned {total_items} structured records",
                    "measurement": f"Retained {sample_count} representative records; pruned {omitted} entries ({len(text)} bytes raw)",
                    "reopen_condition": "Query specific item ID or narrow filter parameters if omitted records are required",
                    "total_count": total_items,
                    "sample": sample,
                }
                serialized = json.dumps(summary, separators=(",", ":"), ensure_ascii=False)
                return (
                    f"[COMPACTED TOOL OUTPUT :: {tool_name}]\n"
                    f"{serialized}\n"
                    f"<!-- {omitted} items omitted to preserve KV-cache -->\n"
                    f"<!-- cf:compacted -->"
                )

        # Compact large dictionary with oversized list values
        if isinstance(data, dict) and len(text) > self.max_char_threshold:
            modified = False
            compacted_dict: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 15:
                    compacted_dict[k] = {
                        "_total_count": len(v),
                        "_sample": v[:10],
                        "_omitted": len(v) - 10,
                    }
                    modified = True
                else:
                    compacted_dict[k] = v

            if modified:
                serialized = json.dumps(compacted_dict, separators=(",", ":"), ensure_ascii=False)
                if len(serialized) < len(text):
                    return (
                        f"[COMPACTED TOOL OUTPUT :: {tool_name}]\n"
                        f"{serialized}\n"
                        f"<!-- Nested collections pruned for KV-cache efficiency -->\n"
                        f"<!-- cf:compacted -->"
                    )

        return None

    def _compact_text_log(self, text: str, tool_name: str) -> str:
        lines = text.strip().split("\n")
        total_lines = len(lines)

        # Detect genuine failure and error lines (skip passing test lines like 'test_error_case_0 PASSED')
        failure_patterns = [
            "failed",
            "failure",
            "error:",
            "exception:",
            "traceback",
            "fatal:",
            "panic:",
            "e   assert",
            "assertionerror",
            "command failed",
        ]
        critical_lines: list[str] = []
        for line in lines:
            line_lower = line.lower()
            if " passed" in line_lower or line_lower.endswith("passed") or " skipped" in line_lower:
                continue
            if any(pat in line_lower for pat in failure_patterns):
                critical_lines.append(line)

        head_lines = [line[:300] for line in lines[: self.head_lines_budget]]
        tail_lines = (
            [line[:300] for line in lines[-self.tail_lines_budget :]]
            if total_lines > self.head_lines_budget
            else []
        )

        omitted_lines_count = max(0, total_lines - len(head_lines) - len(tail_lines))

        result_parts = [
            f"[TOOL EXECUTION SUMMARY :: {tool_name}]",
            f"- Claim: Tool '{tool_name}' emitted {total_lines} lines ({len(text)} bytes)",
        ]

        if critical_lines:
            primary_diagnostic = critical_lines[0].strip()[:300]
            result_parts.append(
                f"- Measurement (Critical Diagnostic, {len(critical_lines)} matches): {primary_diagnostic}"
            )
            result_parts.append("\n--- [Critical Failure Excerpt] ---")
            for cl in critical_lines[:6]:
                result_parts.append(cl[:300])
        else:
            result_parts.append(
                f"- Measurement: Execution completed with 0 error indicators across {total_lines} lines"
            )

        result_parts.append(
            f"- Reopen Condition: Re-invoke '{tool_name}' with targeted line filter if middle {omitted_lines_count} lines are needed"
        )

        result_parts.append("\n--- [Output Head] ---")
        result_parts.extend(head_lines)

        if omitted_lines_count > 0:
            result_parts.append(
                f"\n<!-- ... {omitted_lines_count} lines omitted by RunSift ... -->\n"
            )
            result_parts.append("--- [Output Tail] ---")
            result_parts.extend(tail_lines)

        result_parts.append("\n<!-- cf:compacted -->")
        return "\n".join(result_parts)

    def compact_messages(
        self, messages: list[ChatMessage | dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Compacts verbose tool execution returns in chat history while preserving user and assistant messages intact."""
        total_saved = 0
        compacted: list[dict[str, Any]] = []

        # Index tool call IDs to function names and arguments from assistant messages
        tool_call_names: dict[str, str] = {}
        tool_call_args: dict[str, str] = {}
        tool_args_by_name: dict[str, str] = {}
        for m in messages:
            tc_list = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
            if tc_list and isinstance(tc_list, list):
                for tc in tc_list:
                    tc_dict = (
                        tc
                        if isinstance(tc, dict)
                        else (tc.model_dump() if hasattr(tc, "model_dump") else {})
                    )
                    t_id = tc_dict.get("id")
                    fn_dict = (
                        tc_dict.get("function", {})
                        if isinstance(tc_dict.get("function"), dict)
                        else {}
                    )
                    fn_name = fn_dict.get("name")
                    fn_args = fn_dict.get("arguments", "")
                    if fn_name:
                        tool_args_by_name[fn_name] = str(fn_args)
                    if t_id and fn_name:
                        tool_call_names[t_id] = fn_name
                        tool_call_args[t_id] = str(fn_args)

        for msg in messages:
            msg_dict = (
                msg.model_dump(exclude_none=True) if isinstance(msg, ChatMessage) else dict(msg)
            )
            role = msg_dict.get("role")
            content = msg_dict.get("content", "")

            if role == "tool" and isinstance(content, str):
                tool_id = msg_dict.get("tool_call_id")
                tool_name = msg_dict.get("name") or tool_call_names.get(tool_id or "", "tool")
                tool_arg_str = tool_call_args.get(
                    tool_id or "", tool_args_by_name.get(tool_name, "")
                )
                compact_content, saved = self.compact_tool_output(
                    content, tool_name=tool_name, tool_call_id=tool_id, tool_args=tool_arg_str
                )
                msg_dict["content"] = compact_content
                total_saved += saved

            compacted.append(msg_dict)

        return compacted, total_saved
