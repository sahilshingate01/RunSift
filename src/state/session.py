"""Agent session state definitions and lifecycle models.

Tracks agent loops across tool execution suspension boundaries.
Allowed dependencies per wiring.yaml: ["core", "compact"].
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED_TOOL_WAIT = "suspended_tool_wait"
    READY = "ready"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"


class SuspendedToolCall(BaseModel):
    id: str
    name: str
    arguments: str
    suspended_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    output: str | None = None
    characters_saved_by_compaction: int = 0


class AgentSession(BaseModel):
    session_id: str
    model: str
    status: SessionStatus = SessionStatus.ACTIVE
    messages: list[dict[str, Any]] = Field(default_factory=list)
    pending_tool_calls: dict[str, SuspendedToolCall] = Field(default_factory=dict)
    prefix_hash: str = ""
    created_at: float = Field(default_factory=time.time)
    last_active_at: float = Field(default_factory=time.time)
    total_suspended_seconds: float = 0.0

    def is_all_tools_resolved(self) -> bool:
        return all(tc.completed_at is not None for tc in self.pending_tool_calls.values())
