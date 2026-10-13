"""RunSift Compact Subsystem.

Invariant-based message compactor and negative memory synthesizer.
Allowed dependencies per wiring.yaml: ["core"].
"""

from src.compact.condenser_v2 import (
    CondenserV2,
    clip_long_line,
    condense_tool_output,
    extract_referenced_setup_lines,
    get_active_condenser_version,
)
from src.compact.synthesizer import EXEMPT_TOOL_NAMES, ToolOutputCompactor

__all__ = [
    "EXEMPT_TOOL_NAMES",
    "CondenserV2",
    "ToolOutputCompactor",
    "clip_long_line",
    "condense_tool_output",
    "extract_referenced_setup_lines",
    "get_active_condenser_version",
]
