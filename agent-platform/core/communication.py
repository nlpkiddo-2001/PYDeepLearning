"""
Agent Communication Protocol
=============================
Message-passing system for agent-to-agent communication (v3).

Supports:
  - Direct messages: one agent to another
  - Broadcasts: one agent to all agents
  - Request/Response: ask another agent and await reply
  - Delegation: assign a sub-goal to another agent
  - Status updates: report progress on delegated tasks

All messages flow through a central MessageRouter which handles
delivery, buffering, and persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("agentforge.communication")


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    DIRECT = "direct"              # Point-to-point message
    BROADCAST = "broadcast"        # One-to-all
    REQUEST = "request"            # Expects a response
    RESPONSE = "response"          # Reply to a request
    DELEGATE = "delegate"          # Assign a sub-task
    DELEGATE_RESULT = "delegate_result"  # Result of a delegated task
    STATUS = "status"              # Progress update
    ERROR = "error"                # Error notification


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AgentMessage:
    """A single inter-agent message."""
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    type: MessageType = MessageType.DIRECT
    sender_id: str = ""
    receiver_id: str = ""         # empty for broadcasts
    content: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    reply_to: Optional[str] = None  # message_id this is replying to
    correlation_id: Optional[str] = None  # for request/response tracking
    timestamp: float = field(default_factory=time.time)
    delivered: bool = False
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "payload": self.payload,
            "priority": self.priority.value,
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "delivered": self.delivered,
            "read": self.read,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(
            id=data.get("id", uuid4().hex[:16]),
            type=MessageType(data.get("type", "direct")),
            sender_id=data.get("sender_id", ""),
            receiver_id=data.get("receiver_id", ""),
            content=data.get("content", ""),
            payload=data.get("payload", {}),
            priority=MessagePriority(data.get("priority", "normal")),
            reply_to=data.get("reply_to"),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", time.time()),
            delivered=data.get("delivered", False),
            read=data.get("read", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DelegationTask:
    """Tracks a delegated task from parent to child agent."""
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    parent_id: str = ""
    child_id: str = ""
    goal: str = ""
    status: str = "pending"  # pending | running | completed | failed
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    message_id: str = ""  # the delegation message ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "goal": self.goal,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "message_id": self.message_id,
        }


class MessageRouter:
    """
    Central message router for agent-to-agent communication.

    Each agent has an inbox (asyncio.Queue). The router delivers
    messages to the correct inbox. Messages are also persisted
    to SQLite for history / debugging.
    """

    def __init__(self, db_path: str = "./data/messages.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Agent inboxes: agent_id -> asyncio.Queue
        self._inboxes: Dict[str, asyncio.Queue] = {}

        # Pending request futures: correlation_id -> asyncio.Future
        self._pending_requests: Dict[str, asyncio.Future] = {}

        # Delegation tracking: delegation_id -> DelegationTask
        self._delegations: Dict[str, DelegationTask] = {}

        # Listeners for all messages (for UI streaming)
        self._global_listeners: List[asyncio.Queue] = []

        # Registered agent IDs (for broadcasts)
        self._registered_agents: set = set()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                receiver_id TEXT DEFAULT '',
                content TEXT NOT NULL,
                payload TEXT DEFAULT '{}',
                priority TEXT DEFAULT 'normal',
                reply_to TEXT,
                correlation_id TEXT,
                timestamp REAL NOT NULL,
                delivered INTEGER DEFAULT 0,
                read_flag INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS delegations (
                id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                result TEXT,
                created_at REAL NOT NULL,
                completed_at REAL,
                message_id TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_receiver
            ON messages(receiver_id, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_sender
            ON messages(sender_id, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_delegation_parent
            ON delegations(parent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_delegation_child
            ON delegations(child_id)
        """)
        conn.commit()
        conn.close()

    # ── Agent Registration ───────────────────────────────────────

    def register_agent(self, agent_id: str) -> asyncio.Queue:
        """Register an agent and return its inbox queue."""
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = asyncio.Queue()
        self._registered_agents.add(agent_id)
        logger.info("Agent registered with message router: %s", agent_id)
        return self._inboxes[agent_id]

    def unregister_agent(self, agent_id: str):
        """Remove an agent from the router."""
        self._inboxes.pop(agent_id, None)
        self._registered_agents.discard(agent_id)

    def get_inbox(self, agent_id: str) -> Optional[asyncio.Queue]:
        """Get an agent's inbox queue."""
        return self._inboxes.get(agent_id)

    # ── Send Operations ──────────────────────────────────────────

    async def send(self, message: AgentMessage) -> AgentMessage:
        """Send a message to its intended recipient(s)."""
        # Persist
        self._persist_message(message)

        if message.type == MessageType.BROADCAST:
            # Deliver to all registered agents except sender
            for agent_id in self._registered_agents:
                if agent_id != message.sender_id:
                    await self._deliver(agent_id, message)
        else:
            # Direct delivery
            await self._deliver(message.receiver_id, message)

        # Notify global listeners
        for q in self._global_listeners:
            try:
                await q.put(message.to_dict())
            except Exception:
                pass

        logger.debug(
            "Message sent: %s -> %s (type=%s)",
            message.sender_id, message.receiver_id or "ALL", message.type.value,
        )
        return message

    async def send_direct(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Convenience: send a direct message."""
        msg = AgentMessage(
            type=MessageType.DIRECT,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            payload=payload or {},
        )
        return await self.send(msg)

    async def broadcast(
        self,
        sender_id: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Convenience: broadcast a message to all agents."""
        msg = AgentMessage(
            type=MessageType.BROADCAST,
            sender_id=sender_id,
            content=content,
            payload=payload or {},
        )
        return await self.send(msg)

    async def request(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for a response.
        Returns the response message, or None on timeout.
        """
        correlation_id = uuid4().hex[:16]
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[correlation_id] = future

        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            payload=payload or {},
            correlation_id=correlation_id,
        )
        await self.send(msg)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(
                "Request %s from %s to %s timed out after %ss",
                correlation_id, sender_id, receiver_id, timeout,
            )
            return None
        finally:
            self._pending_requests.pop(correlation_id, None)

    async def respond(
        self,
        original_message: AgentMessage,
        sender_id: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Send a response to a request message."""
        msg = AgentMessage(
            type=MessageType.RESPONSE,
            sender_id=sender_id,
            receiver_id=original_message.sender_id,
            content=content,
            payload=payload or {},
            reply_to=original_message.id,
            correlation_id=original_message.correlation_id,
        )
        await self.send(msg)

        # Resolve pending future if exists
        if original_message.correlation_id in self._pending_requests:
            future = self._pending_requests.pop(original_message.correlation_id)
            if not future.done():
                future.set_result(msg)

        return msg

    # ── Delegation ───────────────────────────────────────────────

    async def delegate(
        self,
        parent_id: str,
        child_id: str,
        goal: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> DelegationTask:
        """
        Delegate a sub-task from parent agent to child agent.
        Returns a DelegationTask for tracking.
        """
        delegation = DelegationTask(
            parent_id=parent_id,
            child_id=child_id,
            goal=goal,
            status="pending",
        )

        # Send delegation message
        msg = AgentMessage(
            type=MessageType.DELEGATE,
            sender_id=parent_id,
            receiver_id=child_id,
            content=goal,
            payload={
                "delegation_id": delegation.id,
                **(payload or {}),
            },
        )
        await self.send(msg)

        delegation.message_id = msg.id
        self._delegations[delegation.id] = delegation
        self._persist_delegation(delegation)

        logger.info(
            "Task delegated: %s -> %s (goal: %s)",
            parent_id, child_id, goal[:80],
        )
        return delegation

    async def report_delegation_result(
        self,
        delegation_id: str,
        child_id: str,
        result: str,
        success: bool = True,
    ) -> Optional[DelegationTask]:
        """Report the result of a delegated task back to the parent."""
        delegation = self._delegations.get(delegation_id)
        if not delegation:
            logger.warning("Delegation %s not found", delegation_id)
            return None

        delegation.status = "completed" if success else "failed"
        delegation.result = result
        delegation.completed_at = time.time()

        # Send result message to parent
        msg = AgentMessage(
            type=MessageType.DELEGATE_RESULT,
            sender_id=child_id,
            receiver_id=delegation.parent_id,
            content=result,
            payload={
                "delegation_id": delegation_id,
                "success": success,
                "goal": delegation.goal,
            },
        )
        await self.send(msg)

        # Update persistence
        self._persist_delegation(delegation)

        return delegation

    def get_delegation(self, delegation_id: str) -> Optional[DelegationTask]:
        return self._delegations.get(delegation_id)

    def get_delegations_by_parent(self, parent_id: str) -> List[DelegationTask]:
        return [d for d in self._delegations.values() if d.parent_id == parent_id]

    def get_delegations_by_child(self, child_id: str) -> List[DelegationTask]:
        return [d for d in self._delegations.values() if d.child_id == child_id]

    # ── Message History ──────────────────────────────────────────

    def get_messages(
        self,
        agent_id: str,
        direction: str = "all",  # "sent" | "received" | "all"
        limit: int = 50,
    ) -> List[AgentMessage]:
        """Get message history for an agent."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        if direction == "sent":
            rows = conn.execute(
                "SELECT * FROM messages WHERE sender_id = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        elif direction == "received":
            rows = conn.execute(
                "SELECT * FROM messages WHERE receiver_id = ? OR (type = 'broadcast' AND sender_id != ?) ORDER BY timestamp DESC LIMIT ?",
                (agent_id, agent_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE sender_id = ? OR receiver_id = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_id, agent_id, limit),
            ).fetchall()

        conn.close()

        return [self._row_to_message(row) for row in rows]

    def get_conversation(
        self,
        agent_a: str,
        agent_b: str,
        limit: int = 50,
    ) -> List[AgentMessage]:
        """Get the message thread between two agents."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM messages
               WHERE (sender_id = ? AND receiver_id = ?)
                  OR (sender_id = ? AND receiver_id = ?)
               ORDER BY timestamp ASC LIMIT ?""",
            (agent_a, agent_b, agent_b, agent_a, limit),
        ).fetchall()
        conn.close()
        return [self._row_to_message(row) for row in rows]

    # ── Subscriptions ────────────────────────────────────────────

    def subscribe_global(self) -> asyncio.Queue:
        """Subscribe to ALL inter-agent messages (for UI)."""
        q: asyncio.Queue = asyncio.Queue()
        self._global_listeners.append(q)
        return q

    def unsubscribe_global(self, queue: asyncio.Queue):
        self._global_listeners = [q for q in self._global_listeners if q is not queue]

    # ── Internal ─────────────────────────────────────────────────

    async def _deliver(self, agent_id: str, message: AgentMessage):
        """Deliver a message to an agent's inbox."""
        inbox = self._inboxes.get(agent_id)
        if inbox:
            message.delivered = True
            await inbox.put(message)
        else:
            logger.warning("No inbox for agent %s — message buffered in DB", agent_id)

    def _persist_message(self, msg: AgentMessage):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO messages
               (id, type, sender_id, receiver_id, content, payload,
                priority, reply_to, correlation_id, timestamp, delivered, read_flag, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg.id,
                msg.type.value,
                msg.sender_id,
                msg.receiver_id,
                msg.content,
                json.dumps(msg.payload),
                msg.priority.value,
                msg.reply_to,
                msg.correlation_id,
                msg.timestamp,
                int(msg.delivered),
                int(msg.read),
                json.dumps(msg.metadata),
            ),
        )
        conn.commit()
        conn.close()

    def _persist_delegation(self, d: DelegationTask):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO delegations
               (id, parent_id, child_id, goal, status, result, created_at, completed_at, message_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d.id, d.parent_id, d.child_id, d.goal,
                d.status, d.result, d.created_at, d.completed_at, d.message_id,
            ),
        )
        conn.commit()
        conn.close()

    def _row_to_message(self, row: sqlite3.Row) -> AgentMessage:
        return AgentMessage(
            id=row["id"],
            type=MessageType(row["type"]),
            sender_id=row["sender_id"],
            receiver_id=row["receiver_id"],
            content=row["content"],
            payload=json.loads(row["payload"]),
            priority=MessagePriority(row["priority"]),
            reply_to=row["reply_to"],
            correlation_id=row["correlation_id"],
            timestamp=row["timestamp"],
            delivered=bool(row["delivered"]),
            read=bool(row["read_flag"]),
            metadata=json.loads(row["metadata"]),
        )

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        total_delegations = conn.execute("SELECT COUNT(*) FROM delegations").fetchone()[0]
        active_delegations = conn.execute(
            "SELECT COUNT(*) FROM delegations WHERE status IN ('pending', 'running')"
        ).fetchone()[0]
        conn.close()

        return {
            "total_messages": total_msgs,
            "total_delegations": total_delegations,
            "active_delegations": active_delegations,
            "registered_agents": len(self._registered_agents),
            "global_listeners": len(self._global_listeners),
        }
