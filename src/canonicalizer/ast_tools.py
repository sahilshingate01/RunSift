"""AST and semantic prompt segmenter.

Detects dynamic metadata (timestamps, UUIDs, volatile session headers)
and relocates them to the tail of system prompts, ensuring the static invariant
prefix remains stable in the provider's prefix cache.
Allowed dependencies per wiring.yaml: ["core"].
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SegmentedPrompt:
    static_prefix: str
    dynamic_suffix: str
    is_reordered: bool

    @property
    def unified_prompt(self) -> str:
        if not self.dynamic_suffix:
            return self.static_prefix
        if not self.static_prefix:
            return self.dynamic_suffix
        return f"{self.static_prefix}\n\n[Runtime Context]\n{self.dynamic_suffix}"


# Strictly anchored patterns indicating leading dynamic/volatile session metadata
DYNAMIC_HEADER_PATTERNS = [
    # Timestamps & dates (e.g. 'Current Time: 2026-10-14 23:00:00 UTC', 'Date: 2026-10-14')
    r"^\s*(?:current\s+)?(?:date|time|datetime|timestamp)\s*[:=]\s*\S+",
    r"^\s*today\s+is\s+\S+",
    r"^\s*\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}",
    # Ephemeral session / request / trace IDs (strictly requiring '-id', '_id', or ' id' to avoid matching 'Return:' or 'Run:')
    r"^\s*(?:session|request|conversation|interaction|trace|sess)[\s_-]*id\s*[:=]\s*[a-zA-Z0-9_-]+",
    r"^\s*sess[-_][a-zA-Z0-9_-]+\b",
    r"^\s*[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\s*$",
    # Ephemeral working directories or temporary paths
    r"^\s*(?:current\s+)?temp(?:orary)?\s+(?:dir|path)\s*[:=]\s*\S+",
]

COMPILED_DYNAMIC_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in DYNAMIC_HEADER_PATTERNS]


def is_volatile_header_line(line: str) -> bool:
    """Checks whether a single line is a pure volatile metadata header."""
    clean = line.strip()
    if not clean or len(clean) > 120:
        return False
    # If the line contains sentence punctuation, markdown fences, or dividers, it's an instruction
    if any(sep in clean for sep in (". ", "!", "?", "```", "---", "###")):
        return False
    return any(pat.match(clean) for pat in COMPILED_DYNAMIC_PATTERNS)


class PromptASTSegmenter:
    """Extracts and hoists leading static invariant sections ahead of volatile runtime variables."""

    def __init__(self, max_header_scan_lines: int = 8):
        self.max_header_scan_lines = max_header_scan_lines

    def segment_prompt(self, prompt: str) -> SegmentedPrompt:
        if not prompt or not prompt.strip():
            return SegmentedPrompt(static_prefix="", dynamic_suffix="", is_reordered=False)

        lines = prompt.strip().split("\n")
        leading_headers: list[str] = []
        body_start_idx = 0

        # Scan only leading contiguous header lines up to max_header_scan_lines
        scan_limit = min(len(lines), self.max_header_scan_lines)
        for idx in range(scan_limit):
            line = lines[idx]
            if not line.strip():
                # Allow empty lines separating initial header lines
                continue
            if is_volatile_header_line(line):
                leading_headers.append(line.strip())
                body_start_idx = idx + 1
            else:
                # Stop immediately as soon as real prompt instructions begin
                break

        # If leading dynamic headers were found, relocating them to the tail protects the KV-cache
        if leading_headers and body_start_idx < len(lines):
            static_text = "\n".join(lines[body_start_idx:]).strip()
            dynamic_text = "\n".join(leading_headers).strip()
            return SegmentedPrompt(
                static_prefix=static_text,
                dynamic_suffix=dynamic_text,
                is_reordered=True,
            )

        return SegmentedPrompt(
            static_prefix=prompt.strip(),
            dynamic_suffix="",
            is_reordered=False,
        )
