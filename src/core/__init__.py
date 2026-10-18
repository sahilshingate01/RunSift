"""RunSift Core Subsystem.

Shared schemas, telemetry metrics, models, configuration, and invariant definitions.
Conforms to SDCS v1.3.0 wiring.yaml as a zero-dependency leaf.
"""

from src.core.config import (
    SUPPORTED_NVIDIA_MODELS,
    CacheForgeConfig,
    RunSiftConfig,
    _load_dotenv_if_present,
    candidate_dotenv_paths,
    get_config,
)
from src.core.invariants import (
    DEFAULT_CACHE_ALIGNMENT_TOLERANCE,
    GatewayTelemetry,
    telemetry,
)
from src.core.models import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    FunctionCall,
    PrefixMetrics,
    StreamChoice,
    ToolCall,
    ToolDefinition,
    ToolFunctionSchema,
    UsageInfo,
)

__all__ = [
    "DEFAULT_CACHE_ALIGNMENT_TOLERANCE",
    "SUPPORTED_NVIDIA_MODELS",
    "RunSiftConfig",
    "CacheForgeConfig",
    "ChatChoice",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "DeltaMessage",
    "FunctionCall",
    "GatewayTelemetry",
    "PrefixMetrics",
    "StreamChoice",
    "ToolCall",
    "ToolDefinition",
    "ToolFunctionSchema",
    "UsageInfo",
    "_load_dotenv_if_present",
    "candidate_dotenv_paths",
    "get_config",
    "telemetry",
]
