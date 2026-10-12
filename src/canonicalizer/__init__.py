"""RunSift Canonicalizer Subsystem.

AST and schema prefix normalizer; guarantees prefix stability across turns.
Allowed dependencies per wiring.yaml: ["core"].
"""

from src.canonicalizer.ast_tools import (
    PromptASTSegmenter,
    SegmentedPrompt,
)
from src.canonicalizer.normalizer import (
    PrefixCanonicalizer,
    canonical_json_dump,
    canonicalize_message,
    canonicalize_tool_schema,
    canonicalize_tools,
    compute_prefix_hash,
    normalize_whitespace_and_newlines,
)

__all__ = [
    "PrefixCanonicalizer",
    "PromptASTSegmenter",
    "SegmentedPrompt",
    "canonical_json_dump",
    "canonicalize_message",
    "canonicalize_tool_schema",
    "canonicalize_tools",
    "compute_prefix_hash",
    "normalize_whitespace_and_newlines",
]
