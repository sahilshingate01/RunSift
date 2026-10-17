"""Async tool execution suspension engine and session state manager.

Tracks client tool wait intervals and session continuation state across gateway turns.
Allowed dependencies per wiring.yaml: ["core", "compact"].
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.compact import ToolOutputCompactor
from src.state.session import AgentSession, SessionStatus, SuspendedToolCall


class StateManager:
    """Manages active agent sessions, tool suspension intervals, and continuation resumes."""

    def __init__(
        self,
        session_ttl_seconds: float = 3600.0,
        compactor: ToolOutputCompactor | None = None,
        max_sessions: int = 200,
    ):
        self.session_ttl_seconds = session_ttl_seconds
        self.compactor = compactor or ToolOutputCompactor()
        self.max_sessions = max_sessions
        self._sessions: dict[str, AgentSession] = {}

    def get_or_create_session(
        self,
        session_id: str | None,
        model: str,
        messages: list[dict[str, Any]],
        prefix_hash: str = "",
    ) -> AgentSession:
        now = time.time()
        self._cleanup_stale_sessions(now)

        # Cap retained message history to last 5 entries to prevent unbounded RAM exhaustion
        capped_messages = list(messages[-5:]) if messages else []

        if not session_id or session_id not in self._sessions:
            sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
            session = AgentSession(
                session_id=sid,
                model=model,
                status=SessionStatus.ACTIVE,
                messages=capped_messages,
                prefix_hash=prefix_hash,
                created_at=now,
                last_active_at=now,
            )
            self._sessions[sid] = session
            return session

        session = self._sessions[session_id]
        session.last_active_at = now
        session.messages = capped_messages
        if prefix_hash:
            session.prefix_hash = prefix_hash
        return session

    def suspend_for_tools(
        self,
        session_id: str,
        tool_calls: list[dict[str, Any]],
    ) -> AgentSession:
        """Suspends active session state while awaiting external tool execution returns."""
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found")

        now = time.time()
        session.status = SessionStatus.SUSPENDED_TOOL_WAIT
        session.last_active_at = now

        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")

            session.pending_tool_calls[tc_id] = SuspendedToolCall(
                id=tc_id,
                name=name,
                arguments=args,
                suspended_at=now,
            )

        return session

    def record_tool_return(
        self,
        session_id: str,
        tool_call_id: str,
        raw_output: str,
    ) -> AgentSession:
        """Records a completed tool execution return, compacting the payload if needed."""
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found")

        now = time.time()
        session.last_active_at = now

        pending = session.pending_tool_calls.get(tool_call_id)
        tool_name = pending.name if pending else "tool"

        # Apply negative memory compaction to raw output
        compacted_output, chars_saved = self.compactor.compact_tool_output(
            raw_output, tool_name=tool_name, tool_call_id=tool_call_id
        )

        suspended_duration = 0.0
        if pending and pending.completed_at is None:
            pending.completed_at = now
            pending.output = compacted_output
            pending.characters_saved_by_compaction = chars_saved
            suspended_duration = max(0.0, now - pending.suspended_at)

        session.total_suspended_seconds += suspended_duration

        # Append tool message to session history if not already present at tail
        already_present = any(
            m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id
            for m in session.messages
        )
        if not already_present:
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": compacted_output,
            }
            session.messages.append(tool_message)

        if session.is_all_tools_resolved():
            session.status = SessionStatus.READY

        return session

    def resume_continuation(
        self,
        session_id: str,
    ) -> tuple[list[dict[str, Any]], float]:
        """Resumes session, clearing pending tools and returning cache-aligned messages."""
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found")

        session.status = SessionStatus.ACTIVE
        session.pending_tool_calls.clear()
        session.last_active_at = time.time()

        return session.messages, session.total_suspended_seconds

    def _cleanup_stale_sessions(self, now: float) -> None:
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_active_at) > self.session_ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

        # Enforce LRU hard capacity bound against heap exhaustion
        if len(self._sessions) >= self.max_sessions:
            excess = len(self._sessions) - self.max_sessions + 1
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].last_active_at)[:excess]
            for sid, _ in oldest:
                self._sessions.pop(sid, None)
