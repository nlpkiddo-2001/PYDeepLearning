"""
Streaming Module
================
WebSocket-friendly event emitter for real-time execution logs.
Every step of the ReAct loop emits structured events that the UI
renders in a terminal-style log panel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("agentforge.streaming")


class EventType(str, Enum):
    """Types of streaming log events."""
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    DONE = "done"
    ERROR = "error"
    INFO = "info"
    RETRY = "retry"
    # v3: Multi-agent events
    AGENT_SPAWN = "agent_spawn"
    AGENT_MESSAGE = "agent_message"
    AGENT_DELEGATE = "agent_delegate"
    AGENT_DELEGATE_RESULT = "agent_delegate_result"
    AGENT_TERMINATE = "agent_terminate"
    SHARED_MEMORY_WRITE = "shared_memory_write"
    SHARED_MEMORY_READ = "shared_memory_read"


@dataclass
class StreamEvent:
    """A single log event emitted during agent execution."""
    type: EventType
    data: Dict[str, Any]
    agent_id: str = ""
    task_id: str = ""
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d)

    @classmethod
    def plan(cls, thought: str, **kw: Any) -> "StreamEvent":
        return cls(type=EventType.PLAN, data={"thought": thought}, **kw)

    @classmethod
    def tool_call(cls, tool_name: str, args: Dict[str, Any], **kw: Any) -> "StreamEvent":
        return cls(type=EventType.TOOL_CALL, data={"tool": tool_name, "args": args}, **kw)

    @classmethod
    def tool_result(cls, tool_name: str, result: str, **kw: Any) -> "StreamEvent":
        return cls(
            type=EventType.TOOL_RESULT,
            data={"tool": tool_name, "result": result[:2000]},
            **kw,
        )

    @classmethod
    def done(cls, result: str, **kw: Any) -> "StreamEvent":
        return cls(type=EventType.DONE, data={"result": result}, **kw)

    @classmethod
    def error(cls, message: str, retry: int = 0, max_retries: int = 0, **kw: Any) -> "StreamEvent":
        return cls(
            type=EventType.ERROR,
            data={"message": message, "retry": retry, "max_retries": max_retries},
            **kw,
        )

    # v3: Multi-agent event constructors
    @classmethod
    def agent_spawn(
        cls, parent_id: str, child_id: str, child_name: str, goal: str = "", **kw: Any
    ) -> "StreamEvent":
        return cls(
            type=EventType.AGENT_SPAWN,
            data={"parent_id": parent_id, "child_id": child_id, "child_name": child_name, "goal": goal},
            **kw,
        )

    @classmethod
    def agent_message(
        cls, sender_id: str, receiver_id: str, content: str, msg_type: str = "direct", **kw: Any
    ) -> "StreamEvent":
        return cls(
            type=EventType.AGENT_MESSAGE,
            data={"sender_id": sender_id, "receiver_id": receiver_id, "content": content[:500], "msg_type": msg_type},
            **kw,
        )

    @classmethod
    def agent_delegate(
        cls, parent_id: str, child_id: str, goal: str, delegation_id: str = "", **kw: Any
    ) -> "StreamEvent":
        return cls(
            type=EventType.AGENT_DELEGATE,
            data={"parent_id": parent_id, "child_id": child_id, "goal": goal, "delegation_id": delegation_id},
            **kw,
        )

    @classmethod
    def agent_delegate_result(
        cls, child_id: str, parent_id: str, result: str, success: bool = True, **kw: Any
    ) -> "StreamEvent":
        return cls(
            type=EventType.AGENT_DELEGATE_RESULT,
            data={"child_id": child_id, "parent_id": parent_id, "result": result[:500], "success": success},
            **kw,
        )


# ─── Event Bus ───────────────────────────────────────────────────────

class EventBus:
    """
    Pub-sub event bus for streaming execution events.

    WebSocket endpoints subscribe per agent_id / task_id.
    The planner publishes events as it executes.

    v3.1: Added per-agent ring buffer so reconnecting WebSocket clients
    can replay events that were published while they were disconnected.
    """

    DEFAULT_BUFFER_SIZE = 200  # max events buffered per agent

    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE):
        # agent_id -> list of subscriber queues
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._global_subscribers: List[asyncio.Queue] = []
        # v3.1: ring buffer of recent events per agent_id
        self._buffer_size = buffer_size
        self._event_buffer: Dict[str, List[StreamEvent]] = {}

    # ── Subscribe / Unsubscribe ──────────────────────────────────

    def subscribe(self, agent_id: Optional[str] = None) -> asyncio.Queue:
        """Subscribe to events. If agent_id is None, gets ALL events."""
        q: asyncio.Queue = asyncio.Queue()
        if agent_id:
            self._subscribers.setdefault(agent_id, []).append(q)
        else:
            self._global_subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue, agent_id: Optional[str] = None) -> None:
        if agent_id and agent_id in self._subscribers:
            self._subscribers[agent_id] = [
                q for q in self._subscribers[agent_id] if q is not queue
            ]
        else:
            self._global_subscribers = [
                q for q in self._global_subscribers if q is not queue
            ]

    # ── Buffered history ─────────────────────────────────────────

    def get_event_history(self, agent_id: str, since_event_id: Optional[str] = None) -> List[StreamEvent]:
        """
        Return buffered events for an agent.

        If *since_event_id* is given, return only events that came **after**
        that event (useful for catching up after a reconnect).
        """
        buf = self._event_buffer.get(agent_id, [])
        if not since_event_id:
            return list(buf)
        # Find the event and return everything after it
        for idx, ev in enumerate(buf):
            if ev.event_id == since_event_id:
                return list(buf[idx + 1:])
        # event_id not found — return full buffer (client is too far behind)
        return list(buf)

    def clear_event_history(self, agent_id: str) -> None:
        """Clear the event buffer for an agent (e.g. on new run)."""
        self._event_buffer.pop(agent_id, None)

    # ── Publish ──────────────────────────────────────────────────

    async def publish(self, event: StreamEvent) -> None:
        """Publish an event to all relevant subscribers and buffer it."""
        logger.debug("Event [%s/%s]: %s", event.agent_id, event.type.value, event.data)

        # Buffer the event
        if event.agent_id:
            buf = self._event_buffer.setdefault(event.agent_id, [])
            buf.append(event)
            # Trim ring buffer
            if len(buf) > self._buffer_size:
                self._event_buffer[event.agent_id] = buf[-self._buffer_size:]

        # Agent-specific subscribers
        for q in self._subscribers.get(event.agent_id, []):
            await q.put(event)
        # Global subscribers
        for q in self._global_subscribers:
            await q.put(event)


# Singleton event bus
event_bus = EventBus()
