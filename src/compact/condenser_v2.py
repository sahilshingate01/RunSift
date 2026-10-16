"""RunSift Condenser v2 (spec revision 2 + shell-test addendum).

Extracts verbatim diagnostic digests from verbose tool outputs, keeping
referenced setup lines via Python's stdlib tokenize module, clipping long
lines while preserving both head and tail, and collapsing superseded pytest
runs into a compact stale form.
Allowed dependencies per wiring.yaml: ["core", "compact"].
"""

from __future__ import annotations

import builtins
import io
import json
import keyword
import os
import re
import shlex
import tokenize
from collections import defaultdict
from typing import Any

from src.compact.synthesizer import EXEMPT_TOOL_NAMES, ToolOutputCompactor
from src.core.models import ChatMessage

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins)) | {"self", "cls"}
_V2_MARKER = "<!-- cf:v2 -->"
_V1_MARKER = "<!-- cf:compacted -->"

# Section 3 (D2): Shell-like tool name tokens
_SHELL_TOOL_TOKENS: frozenset[str] = frozenset(
    {
        "bash",
        "sh",
        "zsh",
        "shell",
        "terminal",
        "console",
        "command",
        "cmd",
        "exec",
        "execute",
        "run",
        "powershell",
        "pwsh",
        "process",
        "test",
        "tests",
        "pytest",
    }
)

# Section 3 (D3) & Section 4 (S1, S2) & Section 5 (G1, G2): Pytest format regexes
_SESSION_BANNER_RE = re.compile(r"^=+ test session starts =+$")
_DECORATED_SUMMARY_RE = re.compile(r"^=+ (.+) in \d+(\.\d+)?s( \([^)]*\))? =+$")
_SUMMARY_OUTCOME_WORD_RE = re.compile(
    r"\b(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b|no tests ran"
)
_PLAIN_SUMMARY_RE = re.compile(
    r"^(?:(?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)(?:, )?)+|no tests ran) in \d+(?:\.\d+)?s(?:\s*\([^)]*\))?$"
)
_PROGRESS_MARKER_RE = re.compile(r"\[\s*\d+%\]$")
_FAILURES_ERRORS_HEADER_RE = re.compile(r"^=+ (?:FAILURES|ERRORS) =+$")
_SHORT_SUMMARY_HEADER_RE = re.compile(r"^=+ short test summary info =+$")
_FAILED_ERROR_TEST_LINE_RE = re.compile(r"^(?:FAILED|ERROR) (?:\S+::\S+|\S+\.py\b)")
_VERBOSE_RESULT_LINE_RE = re.compile(
    r"^\S+\.py::\S+\s+(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
)
_PROGRESS_FILE_LINE_RE = re.compile(r"^(\S+\.py)\s+([.FEsxX]+)\s+\[\s*\d+%\]$")
_PROGRESS_CONT_LINE_RE = re.compile(r"^([.FEsxX]+)\s+\[\s*\d+%\]$")

# Section 5 (G6): Pytest command options that take a separate value
_PYTEST_VALUE_OPTIONS: frozenset[str] = frozenset(
    {
        "-k",
        "-m",
        "-o",
        "-p",
        "-c",
        "--ignore",
        "--deselect",
        "--rootdir",
        "--junitxml",
        "--maxfail",
        "--durations",
    }
)
_SHELL_CHAIN_OPERATORS: frozenset[str] = frozenset({"&&", "||", ";", "|", "&"})


def _is_dedicated_pytest_tool(tool_name: str) -> bool:
    """D1: Tool named `run_pytest` or `pytest`."""
    return (tool_name or "").lower().strip() in ("run_pytest", "pytest")


def is_shell_like_tool_name(tool_name: str) -> bool:
    """D2: Split tool name into lowercase tokens on non-alphanumerics and camelCase boundaries."""
    if not tool_name:
        return False
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tool_name)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", spaced)
    tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if tok]
    return any(tok in _SHELL_TOOL_TOKENS for tok in tokens)


def _is_decorated_summary_line(stripped: str) -> bool:
    m = _DECORATED_SUMMARY_RE.match(stripped)
    if not m:
        return False
    inner = m.group(1)
    return bool(_SUMMARY_OUTCOME_WORD_RE.search(inner))


def _is_plain_summary_line(stripped: str) -> bool:
    return bool(_PLAIN_SUMMARY_RE.match(stripped))


def _is_any_summary_line(stripped: str) -> bool:
    return _is_decorated_summary_line(stripped) or _is_plain_summary_line(stripped)


def _is_second_pytest_marker(stripped: str) -> bool:
    """D3 second marker required when only a plain summary line is present."""
    norm = stripped.replace("\\", "/")
    if _PROGRESS_MARKER_RE.search(stripped):
        return True
    if _FAILURES_ERRORS_HEADER_RE.match(stripped):
        return True
    if _SHORT_SUMMARY_HEADER_RE.match(stripped):
        return True
    return bool(_FAILED_ERROR_TEST_LINE_RE.match(norm))


def has_pytest_format_evidence(text: str) -> bool:
    """D3: Format evidence required in addition to D2."""
    if not text:
        return False
    has_plain_summary = False
    has_second_marker = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SESSION_BANNER_RE.match(stripped):
            return True
        if _is_decorated_summary_line(stripped):
            return True
        if _is_plain_summary_line(stripped):
            has_plain_summary = True
        elif _is_second_pytest_marker(stripped):
            has_second_marker = True
        if has_plain_summary and has_second_marker:
            return True
    return False


def _is_shell_pytest_enabled() -> bool:
    """D5: Kill switch `RUNSIFT_SHELL_PYTEST=off` (or legacy `CACHEFORGE_SHELL_PYTEST=off`) disables D2 and D3."""
    val = os.environ.get("RUNSIFT_SHELL_PYTEST", os.environ.get("CACHEFORGE_SHELL_PYTEST", "")).strip().lower()
    return val != "off"


def _is_pytest_marker_line_s1(stripped: str) -> bool:
    """S1: First pytest marker line if no banner line is present."""
    norm = stripped.replace("\\", "/")
    if _SESSION_BANNER_RE.match(stripped):
        return True
    if _PROGRESS_MARKER_RE.search(stripped):
        return True
    if _VERBOSE_RESULT_LINE_RE.match(norm):
        return True
    if _FAILURES_ERRORS_HEADER_RE.match(stripped):
        return True
    if _SHORT_SUMMARY_HEADER_RE.match(stripped):
        return True
    if _FAILED_ERROR_TEST_LINE_RE.match(norm):
        return True
    return bool(_is_any_summary_line(stripped))


def split_shell_pytest_region(text: str) -> tuple[list[str], str, list[str]]:
    """Implements Section 4 (S1-S3): Splits shell output into (prefix_lines, region_text, suffix_lines)."""
    lines = text.splitlines()
    if not lines:
        return [], "", []

    # S1: First banner line; if none, first pytest marker line
    start_idx = -1
    for i, line in enumerate(lines):
        if _SESSION_BANNER_RE.match(line.strip()):
            start_idx = i
            break
    if start_idx == -1:
        for i, line in enumerate(lines):
            if _is_pytest_marker_line_s1(line.strip()):
                start_idx = i
                break
    if start_idx == -1:
        start_idx = 0

    # S2: Last summary line (decorated or plain) at or after start_idx; if none, end of output
    end_idx = -1
    for i in range(len(lines) - 1, start_idx - 1, -1):
        if _is_any_summary_line(lines[i].strip()):
            end_idx = i
            break

    prefix_lines = lines[:start_idx]
    if end_idx != -1:
        region_lines = lines[start_idx : end_idx + 1]
        suffix_lines = [ln for ln in lines[end_idx + 1 :]]
        # Strip trailing empty lines from suffix if they are just trailing newline artifacts
        while suffix_lines and not suffix_lines[-1].strip():
            suffix_lines.pop()
    else:
        region_lines = lines[start_idx:]
        suffix_lines = []

    return prefix_lines, "\n".join(region_lines), suffix_lines


def _region_has_verbose_lines(region_text: str) -> bool:
    """G1: Checks whether the region has at least one verbose result line."""
    for line in region_text.splitlines():
        norm = line.strip().replace("\\", "/")
        if _VERBOSE_RESULT_LINE_RE.match(norm):
            return True
    return False


def _extract_progress_per_file_counts(region_text: str) -> dict[str, dict[str, int]]:
    """G2 switch 2: Per-file counts from default-mode progress lines."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0}
    )
    current_file: str | None = None
    for line in region_text.splitlines():
        norm = line.strip().replace("\\", "/")
        m_file = _PROGRESS_FILE_LINE_RE.match(norm)
        if m_file:
            current_file = m_file.group(1)
            outcomes = m_file.group(2)
        else:
            m_cont = _PROGRESS_CONT_LINE_RE.match(norm)
            if m_cont and current_file is not None:
                outcomes = m_cont.group(1)
            else:
                continue
        for ch in outcomes:
            if ch == ".":
                counts[current_file]["passed"] += 1
            elif ch in ("F", "E"):
                counts[current_file]["failed"] += 1
            elif ch == "s":
                counts[current_file]["skipped"] += 1
    return dict(counts)


def _format_capped_wrapper_lines(lines: list[str], label: str, cap_each_side: int) -> list[str]:
    """G4: Formats prefix or suffix lines under `label` with head/tail caps and clip_long_line."""
    if not lines:
        return []
    out: list[str] = [label]
    total = len(lines)
    if total <= cap_each_side * 2:
        for ln in lines:
            out.append(clip_long_line(ln))
    else:
        for ln in lines[:cap_each_side]:
            out.append(clip_long_line(ln))
        omitted = total - (cap_each_side * 2)
        out.append(f"[cf: {omitted} lines omitted]")
        for ln in lines[-cap_each_side:]:
            out.append(clip_long_line(ln))
    return out


def extract_target_from_shell_args(tool_args: str, fallback_extractor: Any = None) -> str | None:
    """Implements Section 5 (G6): Target from a shell command or `target` key."""
    if not tool_args:
        return None
    if fallback_extractor is not None:
        direct = fallback_extractor(tool_args)
        if direct:
            return direct

    # Parse JSON strings or inspect raw tool_args if not JSON
    candidate_strings: list[str] = []
    try:
        data = json.loads(tool_args)
        if isinstance(data, dict):
            t_val = data.get("target")
            if t_val:
                return str(t_val).replace("\\", "/").strip()
            for v in data.values():
                if isinstance(v, str) and "pytest" in v:
                    candidate_strings.append(v)
        elif isinstance(data, str) and "pytest" in data:
            candidate_strings.append(data)
    except (json.JSONDecodeError, TypeError):
        if "pytest" in tool_args:
            candidate_strings.append(tool_args)

    for cmd_str in candidate_strings:
        try:
            tokens = shlex.split(cmd_str, posix=True)
        except ValueError:
            tokens = cmd_str.split()

        # Locate the pytest token (`pytest`, `py.test`, or `-m pytest`)
        pytest_idx = -1
        for i, tok in enumerate(tokens):
            base = tok.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if base in ("pytest", "py.test", "pytest.exe"):
                pytest_idx = i
                break
            if tok == "-m" and i + 1 < len(tokens) and tokens[i + 1].lower() == "pytest":
                pytest_idx = i + 1
                break

        if pytest_idx == -1:
            continue

        i = pytest_idx + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in _SHELL_CHAIN_OPERATORS:
                break
            if tok in _PYTEST_VALUE_OPTIONS:
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            return tok.replace("\\", "/").strip()

    return None


def clip_long_line(line: str, limit: int = 600, keep: int = 290) -> str:
    """Clips any line longer than `limit` chars to its first `keep` and last `keep` chars."""
    if len(line) <= limit:
        return line
    clipped_count = len(line) - (keep * 2)
    return f"{line[:keep]}…[cf: {clipped_count} chars clipped]…{line[-keep:]}"


def _is_wanted_identifier(name: str) -> bool:
    return bool(name) and not keyword.iskeyword(name) and name not in _BUILTIN_NAMES


def _tokenize_statement(stmt_text: str) -> list[tokenize.TokenInfo]:
    """Tokenizes a statement snippet without raising exceptions."""
    dedented = "\n".join(line.lstrip() for line in stmt_text.splitlines())
    tokens: list[tokenize.TokenInfo] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(dedented).readline):
            if tok.type in (
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            ):
                continue
            tokens.append(tok)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return tokens


def _extract_names_from_text(text: str) -> set[str]:
    tokens = _tokenize_statement(text)
    if tokens:
        return {
            tok.string
            for tok in tokens
            if tok.type == tokenize.NAME and _is_wanted_identifier(tok.string)
        }
    return {
        m.group(0)
        for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
        if _is_wanted_identifier(m.group(0))
    }


def _analyze_statement(stmt_lines: list[str]) -> tuple[set[str], set[str], str | None, bool]:
    """Analyzes a statement (one or more physical lines) using `tokenize`.

    Returns:
        bound_names: names bound by assignment, for, with, or import
        rhs_names: names referenced on the RHS / iterable / context manager
        first_name: first NAME token if this is an expression statement
        is_block_header: True if statement ends with ':' (e.g. for/with/if/while)
    """
    stmt_text = "\n".join(stmt_lines)
    tokens = _tokenize_statement(stmt_text)
    if not tokens:
        return set(), set(), None, False

    is_block_header = tokens[-1].type == tokenize.OP and tokens[-1].string == ":"
    first_tok = tokens[0]

    # 1. import / from ... import
    if first_tok.type == tokenize.NAME and first_tok.string == "import":
        bound: set[str] = set()
        i = 1
        while i < len(tokens):
            if tokens[i].type == tokenize.NAME and tokens[i].string == "as" and i + 1 < len(tokens):
                if tokens[i + 1].type == tokenize.NAME:
                    bound.add(tokens[i + 1].string)
                i += 2
                continue
            if tokens[i].type == tokenize.NAME and (i == 1 or tokens[i - 1].string == ","):
                bound.add(tokens[i].string)
            i += 1
        return {b for b in bound if _is_wanted_identifier(b)}, set(), None, is_block_header

    if first_tok.type == tokenize.NAME and first_tok.string == "from":
        bound = set()
        rhs: set[str] = set()
        in_import = False
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == tokenize.NAME and tok.string == "import":
                in_import = True
                i += 1
                continue
            if not in_import:
                if tok.type == tokenize.NAME and _is_wanted_identifier(tok.string):
                    rhs.add(tok.string)
            else:
                if tok.type == tokenize.NAME and tok.string == "as" and i + 1 < len(tokens):
                    if tokens[i + 1].type == tokenize.NAME:
                        bound.add(tokens[i + 1].string)
                    i += 2
                    continue
                if tok.type == tokenize.NAME and tokens[i - 1].string in ("import", ",", "("):
                    if i + 1 < len(tokens) and tokens[i + 1].string == "as":
                        pass
                    else:
                        bound.add(tok.string)
            i += 1
        return (
            {b for b in bound if _is_wanted_identifier(b)},
            rhs,
            None,
            is_block_header,
        )

    # 2. for NAME in EXPR: (or async for)
    idx = 0
    if (
        len(tokens) >= 2
        and tokens[0].type == tokenize.NAME
        and tokens[0].string == "async"
        and tokens[1].type == tokenize.NAME
        and tokens[1].string == "for"
    ):
        idx = 1
    if idx < len(tokens) and tokens[idx].type == tokenize.NAME and tokens[idx].string == "for":
        in_pos = -1
        depth = 0
        for i in range(idx + 1, len(tokens)):
            t = tokens[i]
            if t.type == tokenize.OP and t.string in ("(", "[", "{"):
                depth += 1
            elif t.type == tokenize.OP and t.string in (")", "]", "}"):
                depth = max(0, depth - 1)
            elif depth == 0 and t.type == tokenize.NAME and t.string == "in":
                in_pos = i
                break
        if in_pos != -1:
            bound = {
                t.string
                for t in tokens[idx + 1 : in_pos]
                if t.type == tokenize.NAME and _is_wanted_identifier(t.string)
            }
            rhs = {
                t.string
                for t in tokens[in_pos + 1 :]
                if t.type == tokenize.NAME and _is_wanted_identifier(t.string)
            }
            return bound, rhs, None, True

    # 3. with EXPR as NAME: (or async with)
    if (
        len(tokens) >= 2
        and tokens[0].type == tokenize.NAME
        and tokens[0].string == "async"
        and tokens[1].type == tokenize.NAME
        and tokens[1].string == "with"
    ):
        idx = 1
    if idx < len(tokens) and tokens[idx].type == tokenize.NAME and tokens[idx].string == "with":
        bound = set()
        rhs = set()
        depth = 0
        in_as_target = False
        for i in range(idx + 1, len(tokens)):
            t = tokens[i]
            if t.type == tokenize.OP and t.string in ("(", "[", "{"):
                depth += 1
            elif t.type == tokenize.OP and t.string in (")", "]", "}"):
                depth = max(0, depth - 1)
            if depth == 0 and t.type == tokenize.NAME and t.string == "as":
                in_as_target = True
                continue
            if depth == 0 and t.type == tokenize.OP and t.string in (",", ":"):
                in_as_target = False
                continue
            if t.type == tokenize.NAME and _is_wanted_identifier(t.string):
                if in_as_target:
                    bound.add(t.string)
                else:
                    rhs.add(t.string)
        return bound, rhs, None, True

    # 4. Assignment or augmented assignment at top-level bracket depth 0
    assign_ops = {
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "//=",
        "%=",
        "**=",
        "&=",
        "|=",
        "^=",
        ">>=",
        "<<=",
        "@=",
    }
    depth = 0
    assign_idx = -1
    colon_idx = -1
    for i, t in enumerate(tokens):
        if t.type == tokenize.OP and t.string in ("(", "[", "{"):
            depth += 1
        elif t.type == tokenize.OP and t.string in (")", "]", "}"):
            depth = max(0, depth - 1)
        elif depth == 0 and t.type == tokenize.OP:
            if t.string == ":" and colon_idx == -1 and assign_idx == -1:
                colon_idx = i
            elif t.string in assign_ops:
                assign_idx = i
                break

    if assign_idx != -1:
        lhs_end = colon_idx if (colon_idx != -1 and colon_idx < assign_idx) else assign_idx
        lhs_tokens = tokens[:lhs_end]
        bound = set()
        # Extract root target identifiers on LHS (before '.' or '[' at depth 0, or inside tuple unpacking)
        i = 0
        while i < len(lhs_tokens):
            t = lhs_tokens[i]
            if (
                t.type == tokenize.NAME
                and _is_wanted_identifier(t.string)
                and (i == 0 or lhs_tokens[i - 1].string in (",", "(", "[", "*"))
            ):
                bound.add(t.string)
            i += 1
        rhs_tokens = tokens[assign_idx + 1 :]
        if colon_idx != -1 and colon_idx < assign_idx:
            rhs_tokens = tokens[colon_idx + 1 : assign_idx] + rhs_tokens
        rhs = {
            t.string
            for t in rhs_tokens
            if t.type == tokenize.NAME and _is_wanted_identifier(t.string)
        }
        # If augmented assignment or attribute/subscript assignment (e.g. a[k] = v), also include LHS index/attr expressions in rhs
        for i in range(1, len(lhs_tokens)):
            t = lhs_tokens[i]
            if (
                t.type == tokenize.NAME
                and _is_wanted_identifier(t.string)
                and t.string not in bound
            ):
                rhs.add(t.string)
        return bound, rhs, None, is_block_header

    # 5. Expression statement or other block header (e.g., if/while)
    first_name = (
        first_tok.string
        if (first_tok.type == tokenize.NAME and _is_wanted_identifier(first_tok.string))
        else None
    )
    all_names = {
        t.string for t in tokens if t.type == tokenize.NAME and _is_wanted_identifier(t.string)
    }
    return set(), all_names, first_name, is_block_header


def _group_body_statements(body_lines: list[str]) -> list[tuple[list[int], list[str], int]]:
    """Groups physical lines in the test body into logical statements `(line_indices, lines, indent)`."""
    if not body_lines:
        return []
    groups: list[tuple[list[int], list[str], int]] = []
    cur_indices: list[int] = []
    cur_lines: list[str] = []
    cur_indent: int = 0
    paren_depth = 0

    for idx, line in enumerate(body_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if not cur_lines:
            cur_indices = [idx]
            cur_lines = [line]
            cur_indent = indent
        else:
            cur_indices.append(idx)
            cur_lines.append(line)

        # Update bracket balance using tokenize on current line (ignoring strings/comments)
        try:
            for tok in tokenize.generate_tokens(io.StringIO(line).readline):
                if tok.type == tokenize.OP:
                    if tok.string in ("(", "[", "{"):
                        paren_depth += 1
                    elif tok.string in (")", "]", "}"):
                        paren_depth = max(0, paren_depth - 1)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass

        if paren_depth == 0 and not stripped.endswith("\\"):
            groups.append((cur_indices, cur_lines, cur_indent))
            cur_indices = []
            cur_lines = []
            paren_depth = 0

    if cur_lines:
        groups.append((cur_indices, cur_lines, cur_indent))

    return groups


def extract_referenced_setup_lines(
    frame_source_lines: list[str],
    gt_lines: list[str],
    max_lines: int = 8,
) -> list[str]:
    """Implements Section 4.4: Referenced setup lines for a failing test frame.

    Uses only Python's `tokenize` and never executes code.
    """
    try:
        # Locate `def test_...` header and skip any multi-line signature lines
        def_idx = -1
        for i, line in enumerate(frame_source_lines):
            stripped = line.lstrip()
            if stripped.startswith(("def test_", "async def test_")):
                def_idx = i
                break

        if def_idx == -1:
            # 4.4.7 Fallback: no `def test_` found -> keep up to 3 source lines directly above `>`
            fallback = [
                line
                for line in frame_source_lines
                if line.strip() and not line.lstrip().startswith(("def test_", "async def test_"))
            ]
            return fallback[-3:]

        # Advance past multi-line `def test_...(...):` signature
        sig_end_idx = def_idx
        paren_depth = 0
        for i in range(def_idx, len(frame_source_lines)):
            line = frame_source_lines[i]
            for ch in line:
                if ch in "([{":
                    paren_depth += 1
                elif ch in ")]}":
                    paren_depth = max(0, paren_depth - 1)
            sig_end_idx = i
            if paren_depth == 0 and line.rstrip().endswith(":"):
                break

        body_lines = frame_source_lines[sig_end_idx + 1 :]
        if not body_lines:
            return []

        # Confirm body_lines can be tokenized; if completely malformed syntax, trigger 4.4.7 fallback
        non_empty_body = [ln for ln in body_lines if ln.strip() and not ln.strip().startswith("#")]
        if not non_empty_body:
            return []

        first_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty_body)
        dedented_body = "\n".join(
            ln[first_indent:] if len(ln) >= first_indent else ln.lstrip() for ln in body_lines
        )
        try:
            list(tokenize.generate_tokens(io.StringIO(dedented_body).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            fallback = [
                line
                for line in body_lines
                if line.strip() and not line.lstrip().startswith(("def test_", "async def test_"))
            ]
            return fallback[-3:]

        # 1. Wanted names: NAME tokens on the `>` line(s), excluding keywords and builtins
        gt_code = "\n".join(re.sub(r"^\s*>\s?", "", ln) for ln in gt_lines)
        wanted: set[str] = _extract_names_from_text(gt_code)

        statements = _group_body_statements(body_lines)
        if not statements:
            return []

        base_indent = statements[0][2]
        kept_stmt_indices: set[int] = set()
        required_enclosing_indent: int | None = None

        # 2. Walk the body upward from the `>` line
        for s_idx in range(len(statements) - 1, -1, -1):
            _line_idxs, stmt_lines, indent = statements[s_idx]
            bound, rhs, first_name, is_block_header = _analyze_statement(stmt_lines)

            keep = False
            # (a) binds a wanted name
            if bound & wanted:
                keep = True
                # 3. Add names on its right-hand side to the wanted set
                wanted.update(rhs)
            # (b) expression statement on a wanted object, or (4) enclosing block header
            elif (first_name and first_name in wanted and not is_block_header) or (
                is_block_header
                and required_enclosing_indent is not None
                and indent < required_enclosing_indent
            ):
                keep = True
                wanted.update(rhs)

            if keep:
                kept_stmt_indices.add(s_idx)
                if indent > base_indent:
                    if required_enclosing_indent is None or indent > required_enclosing_indent:
                        required_enclosing_indent = indent
                elif is_block_header and indent <= base_indent:
                    required_enclosing_indent = None

        # Collect kept physical lines in original order
        kept_line_indices: list[int] = []
        for s_idx in sorted(kept_stmt_indices):
            kept_line_indices.extend(statements[s_idx][0])

        # 5. Keep at most `max_lines` (8) lines per failure, preferring the lines closest to `>`
        if len(kept_line_indices) > max_lines:
            kept_line_indices = kept_line_indices[-max_lines:]

        return [body_lines[i] for i in kept_line_indices]
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError, TypeError, IndexError):
        # 4.4.7 Never raise an exception; return 3-line fallback
        fallback = [
            line
            for line in frame_source_lines
            if line.strip() and not line.lstrip().startswith(("def test_", "async def test_"))
        ]
        return fallback[-3:]


_SEP_HEADER_RE = re.compile(r"^_{3,}\s+.+\s+_{3,}$")
_FRAME_DIVIDER_RE = re.compile(r"^_ (?:\s*_)+$")
_LOCATION_LINE_RE = re.compile(r"^\S+:\d+:\s*\S*.*$")


def _is_project_frame(location_line: str) -> bool:
    norm = location_line.strip().replace("\\", "/")
    return norm.startswith("src/") or "/src/" in norm


class CondenserV2:
    """Deterministic, stateless tool-output condenser implementing Condenser v2."""

    def __init__(
        self,
        max_char_threshold: int = 4000,
        head_lines_budget: int = 12,
        tail_lines_budget: int = 16,
        enable_stale_rewrite: bool = True,
    ):
        self.max_char_threshold = max_char_threshold
        self.head_lines_budget = head_lines_budget
        self.tail_lines_budget = tail_lines_budget
        self.enable_stale_rewrite = enable_stale_rewrite
        self._v1 = ToolOutputCompactor(
            max_char_threshold=max_char_threshold,
            head_lines_budget=head_lines_budget,
            tail_lines_budget=tail_lines_budget,
        )

    def _is_shell_pytest_output(self, text: str, tool_name: str = "") -> bool:
        """Checks D2 + D3 + D5 (shell-like tool name and pytest format evidence)."""
        if not _is_shell_pytest_enabled():
            return False
        if _is_dedicated_pytest_tool(tool_name):
            return False
        return is_shell_like_tool_name(tool_name) and has_pytest_format_evidence(text)

    def _is_any_pytest_output(self, text: str, tool_name: str = "") -> bool:
        """Returns True if D1 (dedicated pytest tool) or D2+D3+D5 (shell pytest tool)."""
        if self._v1._is_pytest_output(text, tool_name):
            return True
        t_clean = (tool_name or "").lower().strip()
        if t_clean in EXEMPT_TOOL_NAMES or any(
            t_clean.startswith(p) for p in ("read_", "view_", "cat_", "search_", "find_", "web_")
        ):
            return False
        return self._is_shell_pytest_output(text, tool_name)

    def is_compaction_needed(self, text: str, tool_name: str = "tool", tool_args: str = "") -> bool:
        if not isinstance(text, str):
            return False
        if _V2_MARKER in text:
            return False
        t_clean = (tool_name or "").lower().strip()
        if t_clean in EXEMPT_TOOL_NAMES or any(
            t_clean.startswith(p) for p in ("read_", "view_", "cat_", "search_", "find_", "web_")
        ):
            return False

        args_clean = (tool_args or "").lower()
        if any(
            reader_cmd in args_clean
            for reader_cmd in ("cat ", "head ", "tail ", "less ", "more ", "type ", "grep ")
        ):
            return False

        if self._v1._is_pytest_output(text, tool_name):
            return len(text.strip().splitlines()) > 1 or len(text) > 200

        if self._is_shell_pytest_output(text, tool_name):
            return len(text.strip().splitlines()) > 1 or len(text) > 200

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

    def _parse_and_format_failure_block(self, block_lines: list[str]) -> list[str]:
        """Formats a single pytest failure block according to Section 4.3 step 4."""
        if not block_lines:
            return []

        out: list[str] = []
        # 1. Separator line verbatim
        sep_line = block_lines[0]
        out.append(clip_long_line(sep_line))

        # Split remaining lines into stack frames separated by `_ _ _ _ _`
        frames: list[list[str]] = []
        cur_frame: list[str] = []
        for line in block_lines[1:]:
            if _FRAME_DIVIDER_RE.match(line.strip()):
                if cur_frame:
                    frames.append(cur_frame)
                    cur_frame = []
            else:
                cur_frame.append(line)
        if cur_frame:
            frames.append(cur_frame)

        if not frames:
            return out

        # First frame is the test frame
        test_frame = frames[0]
        test_source_before_gt: list[str] = []
        test_gt_lines: list[str] = []
        test_e_lines: list[str] = []
        test_loc_line: str | None = None

        seen_gt = False
        for line in test_frame:
            stripped = line.strip()
            if not stripped:
                continue
            if line.lstrip().startswith(">"):
                seen_gt = True
                test_gt_lines.append(line)
            elif line.lstrip().startswith("E ") or stripped == "E":
                test_e_lines.append(line)
            elif seen_gt and _LOCATION_LINE_RE.match(stripped):
                test_loc_line = line
            elif not seen_gt:
                test_source_before_gt.append(line)
            elif seen_gt and not test_e_lines and set(stripped) == {"^"}:
                # Caret pointer line under `>` is dropped unless it's `>` or `E`
                continue

        # Referenced setup lines from 4.4
        setup_lines = extract_referenced_setup_lines(test_source_before_gt, test_gt_lines)
        for s_line in setup_lines:
            out.append(clip_long_line(s_line))

        # `>` line(s) and every `E` line of the test frame
        for gt_line in test_gt_lines:
            out.append(clip_long_line(gt_line))
        for e_line in test_e_lines:
            out.append(clip_long_line(e_line))

        # Deeper frames: keep only project frames (`src/`)
        final_loc_line: str | None = test_loc_line
        for deeper in frames[1:]:
            d_gt_lines: list[str] = []
            d_e_lines: list[str] = []
            d_loc_line: str | None = None
            d_seen_gt = False
            for line in deeper:
                stripped = line.strip()
                if not stripped:
                    continue
                if line.lstrip().startswith(">"):
                    d_seen_gt = True
                    d_gt_lines.append(line)
                elif line.lstrip().startswith("E ") or stripped == "E":
                    d_e_lines.append(line)
                elif d_seen_gt and _LOCATION_LINE_RE.match(stripped):
                    d_loc_line = line

            if d_loc_line and _is_project_frame(d_loc_line):
                for gt_line in d_gt_lines:
                    out.append(clip_long_line(gt_line))
                for e_line in d_e_lines:
                    out.append(clip_long_line(e_line))
                final_loc_line = d_loc_line
            else:
                # Non-project frame (e.g. stdlib json): keep its E lines if the test frame had none
                if d_e_lines and not test_e_lines:
                    for e_line in d_e_lines:
                        out.append(clip_long_line(e_line))

        if final_loc_line:
            out.append(clip_long_line(final_loc_line))

        return out

    def condense_pytest_fresh(
        self,
        text: str,
        tool_args: str = "",
        *,
        accept_non_verbose_summary: bool = False,
        use_progress_file_counts: bool = False,
        omit_unbacked_file_counts: bool = False,
        keep_incomplete_tail_lines: bool = False,
        extract_shell_target: bool = False,
    ) -> str:
        """Implements Section 4.3: Fresh pytest digest (with Section 5 G2/G3/G6 switches)."""
        if _V2_MARKER in text:
            return text

        if extract_shell_target:
            target = extract_target_from_shell_args(
                tool_args, fallback_extractor=self._v1._extract_target_from_args
            )
        else:
            target = self._v1._extract_target_from_args(tool_args)

        if use_progress_file_counts or omit_unbacked_file_counts:
            if _region_has_verbose_lines(text):
                file_counts = self._v1._extract_per_file_counts(text)
            elif use_progress_file_counts:
                file_counts = _extract_progress_per_file_counts(text)
            else:
                file_counts = {}
        else:
            file_counts = self._v1._extract_per_file_counts(text)

        target_status = self._v1._determine_target_status(target, file_counts, text)

        lines = text.splitlines()
        in_failures = False
        in_summary = False
        failure_blocks: list[list[str]] = []
        tb_line_failures: list[str] = []
        cur_block: list[str] = []
        summary_failed_lines: list[str] = []
        final_summary_line: str = ""
        timeout_lines: list[str] = []
        last_started_test_line: str = ""

        for line in lines:
            stripped = line.strip()
            if "[TIMEOUT]" in line:
                timeout_lines.append(clip_long_line(line))
                continue
            if "::test_" in line and not stripped.startswith(("FAILED ", "ERROR ", "_", ">", "E ")):
                last_started_test_line = line
            if stripped.startswith("=") and stripped.endswith("="):
                if "FAILURES" in stripped or "ERRORS" in stripped:
                    in_failures = True
                    in_summary = False
                    continue
                if "short test summary info" in stripped:
                    if cur_block:
                        failure_blocks.append(cur_block)
                        cur_block = []
                    in_failures = False
                    in_summary = True
                    continue
                if any(k in stripped for k in ("passed", "failed", "error", "skipped")) or (
                    accept_non_verbose_summary and _is_decorated_summary_line(stripped)
                ):
                    if cur_block:
                        failure_blocks.append(cur_block)
                        cur_block = []
                    in_failures = False
                    in_summary = False
                    final_summary_line = line
                    continue
                if "slowest" in stripped or "warnings summary" in stripped:
                    if cur_block:
                        failure_blocks.append(cur_block)
                        cur_block = []
                    in_failures = False
                    in_summary = False
                    continue

            if accept_non_verbose_summary and _is_plain_summary_line(stripped):
                if cur_block:
                    failure_blocks.append(cur_block)
                    cur_block = []
                in_failures = False
                in_summary = False
                final_summary_line = line
                continue

            if in_failures:
                if _SEP_HEADER_RE.match(stripped):
                    if cur_block:
                        failure_blocks.append(cur_block)
                    cur_block = [line]
                elif cur_block:
                    cur_block.append(line)
                elif accept_non_verbose_summary and stripped and not stripped.startswith("!"):
                    tb_line_failures.append(clip_long_line(line))
            elif in_summary and stripped.startswith(("FAILED ", "ERROR ")):
                summary_failed_lines.append(clip_long_line(line))

        if cur_block:
            failure_blocks.append(cur_block)

        incomplete_run = bool(timeout_lines) or not final_summary_line

        out: list[str] = ["[cf: condensed pytest output, v2]"]
        if incomplete_run:
            out.append(
                "[cf: test run did not finish (timed out or interrupted); per-file counts below are partial]"
            )
        if target_status:
            if incomplete_run:
                out.append(f"[cf: {target_status} (partial)]")
            else:
                out.append(f"[cf: {target_status}]")

        for fpath in sorted(file_counts.keys()):
            p = file_counts[fpath]["passed"]
            f = file_counts[fpath]["failed"]
            if incomplete_run:
                out.append(f"[cf: {fpath}: {p} passed, {f} failed (partial)]")
            else:
                out.append(f"[cf: {fpath}: {p} passed, {f} failed]")

        for block in failure_blocks:
            out.extend(self._parse_and_format_failure_block(block))
        if not failure_blocks and tb_line_failures:
            out.extend(tb_line_failures)

        out.extend(summary_failed_lines)
        if incomplete_run:
            if last_started_test_line:
                out.append(clip_long_line(last_started_test_line))
            out.extend(timeout_lines)
            if keep_incomplete_tail_lines and not timeout_lines and not final_summary_line:
                existing_set = set(out)
                non_empty_region = [clip_long_line(ln) for ln in lines if ln.strip()]
                for tail_ln in non_empty_region[-3:]:
                    if tail_ln not in existing_set:
                        out.append(tail_ln)
                        existing_set.add(tail_ln)
        if final_summary_line:
            out.append(clip_long_line(final_summary_line))
        out.append(_V2_MARKER)
        return "\n".join(out)

    def condense_pytest_stale(
        self,
        text: str,
        *,
        accept_non_verbose_summary: bool = False,
        use_progress_file_counts: bool = False,
        omit_unbacked_file_counts: bool = False,
        keep_incomplete_tail_lines: bool = False,
    ) -> str:
        """Implements Section 4.6: Stale pytest digest (with Section 5 G2/G3 switches)."""
        if _V2_MARKER in text:
            return text

        lines = text.splitlines()
        final_summary_line = ""
        failed_lines: list[str] = []
        timeout_lines: list[str] = []
        last_started_test_line = ""
        in_summary = False

        for line in lines:
            stripped = line.strip()
            if "[TIMEOUT]" in line:
                timeout_lines.append(clip_long_line(line))
                continue
            if "::test_" in line and not stripped.startswith(("FAILED ", "ERROR ", "_", ">", "E ")):
                last_started_test_line = line
            if stripped.startswith("=") and stripped.endswith("="):
                if "short test summary info" in stripped:
                    in_summary = True
                    continue
                if any(k in stripped for k in ("passed", "failed", "error", "skipped")) or (
                    accept_non_verbose_summary and _is_decorated_summary_line(stripped)
                ):
                    in_summary = False
                    final_summary_line = line
                    continue
                in_summary = False
                continue
            if accept_non_verbose_summary and _is_plain_summary_line(stripped):
                in_summary = False
                final_summary_line = line
                continue
            if in_summary and stripped.startswith(("FAILED ", "ERROR ")):
                failed_lines.append(clip_long_line(line))

        if not failed_lines:
            for line in lines:
                if line.strip().startswith(("FAILED ", "ERROR ")):
                    failed_lines.append(clip_long_line(line))

        incomplete_run = bool(timeout_lines) or not final_summary_line
        out: list[str] = ["[cf: earlier test run, superseded by a later run of the same tests]"]
        if incomplete_run:
            out.append(
                "[cf: test run did not finish (timed out or interrupted); per-file counts below are partial]"
            )
            if use_progress_file_counts or omit_unbacked_file_counts:
                if _region_has_verbose_lines(text):
                    file_counts = self._v1._extract_per_file_counts(text)
                elif use_progress_file_counts:
                    file_counts = _extract_progress_per_file_counts(text)
                else:
                    file_counts = {}
            else:
                file_counts = self._v1._extract_per_file_counts(text)

            for fpath in sorted(file_counts.keys()):
                p = file_counts[fpath]["passed"]
                f = file_counts[fpath]["failed"]
                out.append(f"[cf: {fpath}: {p} passed, {f} failed (partial)]")
            if last_started_test_line:
                out.append(clip_long_line(last_started_test_line))
            out.extend(timeout_lines)
            if keep_incomplete_tail_lines and not timeout_lines and not final_summary_line:
                existing_set = set(out)
                non_empty_region = [clip_long_line(ln) for ln in lines if ln.strip()]
                for tail_ln in non_empty_region[-3:]:
                    if tail_ln not in existing_set:
                        out.append(tail_ln)
                        existing_set.add(tail_ln)
        if final_summary_line:
            out.append(clip_long_line(final_summary_line))
        kept_failed = failed_lines[:20]
        out.extend(kept_failed)
        if len(failed_lines) > 20:
            out.append(f"[cf: {len(failed_lines) - 20} more failures]")
        out.append(_V2_MARKER)
        return "\n".join(out)

    def _condense_shell_pytest(
        self, raw_output: str, tool_args: str = "", is_stale: bool = False
    ) -> str:
        """Implements Sections 4 and 5 (S1-S3, G1-G6) for pytest output from a shell tool."""
        if _V2_MARKER in raw_output:
            return raw_output

        prefix_lines, region_text, suffix_lines = split_shell_pytest_region(raw_output)
        is_verbose = _region_has_verbose_lines(region_text)

        # G6: if arguments do not have a `target` key, extract target from shell command
        has_direct_target = bool(self._v1._extract_target_from_args(tool_args))
        effective_args = tool_args
        if not has_direct_target:
            shell_target = extract_target_from_shell_args(tool_args)
            if shell_target:
                effective_args = json.dumps({"target": shell_target})

        if is_verbose:
            # G1: run today's condense_pytest_fresh or condense_pytest_stale on the region text, unchanged
            if is_stale and self.enable_stale_rewrite:
                core_digest = self.condense_pytest_stale(region_text)
            else:
                core_digest = self.condense_pytest_fresh(region_text, tool_args=effective_args)
        else:
            # G2 & G3: non-verbose format switches enabled
            if is_stale and self.enable_stale_rewrite:
                core_digest = self.condense_pytest_stale(
                    region_text,
                    accept_non_verbose_summary=True,
                    use_progress_file_counts=True,
                    omit_unbacked_file_counts=True,
                    keep_incomplete_tail_lines=True,
                )
            else:
                core_digest = self.condense_pytest_fresh(
                    region_text,
                    tool_args=effective_args,
                    accept_non_verbose_summary=True,
                    use_progress_file_counts=True,
                    omit_unbacked_file_counts=True,
                    keep_incomplete_tail_lines=True,
                    extract_shell_target=True,
                )

        # G4: Prefix and suffix kept verbatim, each line through clip_long_line
        if not prefix_lines and not suffix_lines:
            condensed = core_digest
        else:
            core_lines = core_digest.splitlines()
            # Strip trailing _V2_MARKER so suffix goes just before it
            has_marker = bool(core_lines and core_lines[-1] == _V2_MARKER)
            body_lines = core_lines[:-1] if has_marker else core_lines

            assembled: list[str] = []
            if prefix_lines:
                assembled.extend(
                    _format_capped_wrapper_lines(
                        prefix_lines, "[cf: output before the test run]", 4
                    )
                )
            assembled.extend(body_lines)
            if suffix_lines:
                assembled.extend(
                    _format_capped_wrapper_lines(
                        suffix_lines, "[cf: output after the test run]", 10
                    )
                )
            assembled.append(_V2_MARKER)
            condensed = "\n".join(assembled)

        # G5: If condensed text isn't shorter than the input, send the input unchanged (shell path only)
        if len(condensed) >= len(raw_output):
            return raw_output
        return condensed

    def _extract_shell_pytest_run_key(
        self, raw_output: str, tool_name: str, tool_args: str
    ) -> tuple[str, tuple[str, ...]]:
        """Implements Section 5 (G7): Run key is tool name plus file set from G1 or G2, else target."""
        _, region_text, _ = split_shell_pytest_region(raw_output)
        if _region_has_verbose_lines(region_text):
            file_counts = self._v1._extract_per_file_counts(region_text)
        else:
            file_counts = _extract_progress_per_file_counts(region_text)
        files_tuple = tuple(sorted(file_counts.keys()))
        if not files_tuple:
            norm_target = (
                extract_target_from_shell_args(
                    tool_args, fallback_extractor=self._v1._extract_target_from_args
                )
                or ""
            )
            files_tuple = (norm_target,) if norm_target else ()
        return (tool_name.lower().strip(), files_tuple)

    def _condense_non_pytest(self, raw_output: str, tool_name: str) -> str:
        """Implements Section 4.7: Non-pytest output (JSON and text logs with 4.5 clipping and v2 marker)."""
        try:
            data = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            data = None

        if isinstance(data, list) and len(data) > 6 and len(raw_output) > self.max_char_threshold:
            sample_count = min(3, len(data))
            omitted = len(data) - sample_count
            out_lines: list[str] = [
                f"[cf: COMPACTED TOOL OUTPUT :: {tool_name}, {len(data)} items total, {omitted} items omitted]"
            ]
            raw_lines = [ln for ln in raw_output.splitlines() if ln.strip()]
            if len(raw_lines) > 1:
                for ln in raw_lines[: min(sample_count + 1, len(raw_lines))]:
                    out_lines.append(clip_long_line(ln))
            else:
                out_lines.append(clip_long_line(raw_output))
            out_lines.append(_V2_MARKER)
            return "\n".join(out_lines)

        if isinstance(data, dict) and len(raw_output) > self.max_char_threshold:
            out_lines = [f"[cf: COMPACTED TOOL OUTPUT :: {tool_name}]"]
            raw_lines = [ln for ln in raw_output.splitlines() if ln.strip()]
            for ln in raw_lines[: self.head_lines_budget]:
                out_lines.append(clip_long_line(ln))
            out_lines.append(_V2_MARKER)
            return "\n".join(out_lines)

        # Text log handling: verbatim kept lines from raw_output with 4.5 long-line clipping + [cf: ...] metadata
        lines = [ln for ln in raw_output.strip().split("\n") if ln.strip()]
        total_lines = len(lines)
        failure_patterns = (
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
        )
        critical_lines: list[str] = []
        for line in lines:
            line_lower = line.lower()
            if " passed" in line_lower or line_lower.endswith("passed") or " skipped" in line_lower:
                continue
            if any(pat in line_lower for pat in failure_patterns):
                critical_lines.append(line)

        head_lines = [clip_long_line(line) for line in lines[: self.head_lines_budget]]
        tail_lines = (
            [clip_long_line(line) for line in lines[-self.tail_lines_budget :]]
            if total_lines > self.head_lines_budget
            else []
        )
        omitted_lines_count = max(0, total_lines - len(head_lines) - len(tail_lines))

        out_parts: list[str] = [
            f"[cf: TOOL EXECUTION SUMMARY :: {tool_name}, {total_lines} lines ({len(raw_output)} bytes)]"
        ]
        if critical_lines:
            out_parts.append(f"[cf: Critical failure excerpt ({len(critical_lines)} matches)]")
            for cl in critical_lines[:6]:
                out_parts.append(clip_long_line(cl))
        else:
            out_parts.append(
                f"[cf: Execution completed with 0 error indicators across {total_lines} lines]"
            )

        out_parts.append("[cf: Output head]")
        out_parts.extend(head_lines)
        if omitted_lines_count > 0:
            out_parts.append(f"[cf: {omitted_lines_count} lines omitted by RunSift]")
            out_parts.append("[cf: Output tail]")
            out_parts.extend(tail_lines)
        out_parts.append(_V2_MARKER)
        return "\n".join(out_parts)

    def compact_tool_output(
        self,
        raw_output: str,
        tool_name: str = "tool",
        tool_call_id: str | None = None,
        tool_args: str = "",
        is_stale: bool = False,
    ) -> tuple[str, int]:
        """Condenses a single tool output using v2 rules. Returns (condensed_text, chars_saved)."""
        if not self.is_compaction_needed(raw_output, tool_name=tool_name, tool_args=tool_args):
            return raw_output, 0

        original_len = len(raw_output)
        if self._v1._is_pytest_output(raw_output, tool_name):
            if is_stale and self.enable_stale_rewrite:
                condensed = self.condense_pytest_stale(raw_output)
            else:
                condensed = self.condense_pytest_fresh(raw_output, tool_args=tool_args)
            return condensed, max(0, original_len - len(condensed))

        if self._is_shell_pytest_output(raw_output, tool_name):
            condensed = self._condense_shell_pytest(
                raw_output, tool_args=tool_args, is_stale=is_stale
            )
            return condensed, max(0, original_len - len(condensed))

        condensed = self._condense_non_pytest(raw_output, tool_name)
        return condensed, max(0, original_len - len(condensed))

    def compact_messages(
        self, messages: list[ChatMessage | dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """Implements Section 4.1 and 4.2 message pass (and v2.1 I8 when enable_stale_rewrite=False).

        Returns `(compacted_messages, total_saved_chars, chars_before, chars_after)` where
        `chars_before` and `chars_after` sum over condensable tool messages in the request.
        """
        # 1. Index assistant tool_calls by id
        tool_call_names: dict[str, str] = {}
        tool_call_args: dict[str, str] = {}
        tool_args_by_name: dict[str, str] = {}

        normalized_msgs: list[dict[str, Any]] = []
        for m in messages:
            msg_dict = m.model_dump(exclude_none=True) if isinstance(m, ChatMessage) else dict(m)
            normalized_msgs.append(msg_dict)
            tc_list = msg_dict.get("tool_calls")
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
                    if fn_name and self.enable_stale_rewrite:
                        tool_args_by_name[fn_name] = str(fn_args)
                    if t_id and fn_name:
                        tool_call_names[t_id] = fn_name
                        tool_call_args[t_id] = str(fn_args)

        # 2. Identify pytest run keys (only in v2 mode when enable_stale_rewrite=True;
        #    in v2.1 mode, run-key and freshness logic does not run per I8)
        last_pytest_idx_by_key: dict[tuple[str, tuple[str, ...]], int] = {}
        resolved_meta: dict[int, tuple[str, str, tuple[str, tuple[str, ...]] | None]] = {}

        for idx, msg_dict in enumerate(normalized_msgs):
            if msg_dict.get("role") != "tool":
                continue
            content = msg_dict.get("content")
            if not isinstance(content, str):
                continue
            tool_id = msg_dict.get("tool_call_id")
            tool_name = msg_dict.get("name") or tool_call_names.get(tool_id or "", "tool")
            tool_arg_str = tool_call_args.get(
                tool_id or "",
                tool_args_by_name.get(tool_name, "") if self.enable_stale_rewrite else "",
            )
            run_key: tuple[str, tuple[str, ...]] | None = None
            if self.enable_stale_rewrite:
                if self._v1._is_pytest_output(content, tool_name):
                    file_counts = self._v1._extract_per_file_counts(content)
                    files_tuple = tuple(sorted(file_counts.keys()))
                    if not files_tuple:
                        norm_target = self._v1._extract_target_from_args(tool_arg_str) or ""
                        files_tuple = (norm_target,) if norm_target else ()
                    run_key = (tool_name.lower().strip(), files_tuple)
                    last_pytest_idx_by_key[run_key] = idx
                elif self._is_any_pytest_output(content, tool_name):
                    run_key = self._extract_shell_pytest_run_key(content, tool_name, tool_arg_str)
                    last_pytest_idx_by_key[run_key] = idx
            resolved_meta[idx] = (tool_name, tool_arg_str, run_key)

        # 3. Condense each condensable tool message
        total_saved = 0
        chars_before = 0
        chars_after = 0
        compacted: list[dict[str, Any]] = []

        for idx, msg_dict in enumerate(normalized_msgs):
            if idx in resolved_meta:
                content = msg_dict["content"]
                tool_name, tool_arg_str, run_key = resolved_meta[idx]
                already_v2 = _V2_MARKER in content
                needs_compaction = already_v2 or self.is_compaction_needed(
                    content, tool_name=tool_name, tool_args=tool_arg_str
                )
                if needs_compaction:
                    chars_before += len(content)
                    if already_v2:
                        new_content = content
                        saved = 0
                    else:
                        is_stale = (
                            self.enable_stale_rewrite
                            and run_key is not None
                            and last_pytest_idx_by_key.get(run_key) != idx
                        )
                        new_content, saved = self.compact_tool_output(
                            content,
                            tool_name=tool_name,
                            tool_call_id=msg_dict.get("tool_call_id"),
                            tool_args=tool_arg_str,
                            is_stale=is_stale,
                        )
                    msg_dict["content"] = new_content
                    chars_after += len(new_content)
                    total_saved += saved

            compacted.append(msg_dict)

        return compacted, total_saved, chars_before, chars_after


def get_active_condenser_version(default: str = "v2") -> str:
    """Returns 'v1', 'v2', or 'v2.1' from RUNSIFT_CONDENSER or CACHEFORGE_CONDENSER environment variable or default."""
    val = os.getenv("RUNSIFT_CONDENSER", os.getenv("CACHEFORGE_CONDENSER", default)).lower().strip()
    if val in ("v1", "v2", "v2.1"):
        return val
    return "v2"


def condense_tool_output(
    raw_output: str,
    tool_name: str = "tool",
    tool_args: str = "",
    version: str | None = None,
    max_char_threshold: int = 4000,
) -> str:
    """Unified helper to condense a single tool output under v1, v2, or v2.1."""
    ver = (version or get_active_condenser_version()).lower().strip()
    if ver == "v1":
        c1 = ToolOutputCompactor(max_char_threshold=max_char_threshold)
        out, _ = c1.compact_tool_output(raw_output, tool_name=tool_name, tool_args=tool_args)
        return out
    c2 = CondenserV2(
        max_char_threshold=max_char_threshold,
        enable_stale_rewrite=(ver != "v2.1"),
    )
    out, _ = c2.compact_tool_output(raw_output, tool_name=tool_name, tool_args=tool_args)
    return out
