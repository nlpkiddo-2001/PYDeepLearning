"""
Multi-Agent Orchestrator
========================
Central coordinator for v3 multi-agent capabilities.

Manages:
  - Agent lifecycle in multi-agent scenarios
  - Parent-child relationships
  - Sub-agent spawning and teardown
  - Routing between MessageRouter, SharedMemoryBus, and EventBus
  - Global view of all agent interactions
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from core.communication import (
    AgentMessage,
    DelegationTask,
    MessageRouter,
    MessageType,
)
from core.shared_memory import SharedMemoryBus
from core.streaming import EventBus, EventType, StreamEvent, event_bus

logger = logging.getLogger("agentforge.orchestrator")


# ═══════════════════════════════════════════════════════════════════════
# Agent Relationship Tracking
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentNode:
    """Tracks an agent's position in the multi-agent hierarchy."""
    agent_id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    spawned_at: float = field(default_factory=time.time)
    is_sub_agent: bool = False
    delegation_id: Optional[str] = None  # if spawned for a delegation
    status: str = "active"  # active | completed | failed | terminated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "children": self.children,
            "spawned_at": self.spawned_at,
            "is_sub_agent": self.is_sub_agent,
            "delegation_id": self.delegation_id,
            "status": self.status,
        }


# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class MultiAgentOrchestrator:
    """
    Coordinates multi-agent operations.

    Ties together:
      - AgentManager (from core.agent) for agent creation
      - MessageRouter for inter-agent communication
      - SharedMemoryBus for shared data
      - EventBus for real-time streaming to the UI
    """

    def __init__(
        self,
        message_router: Optional[MessageRouter] = None,
        shared_memory: Optional[SharedMemoryBus] = None,
        bus: Optional[EventBus] = None,
    ):
        self.router = message_router or MessageRouter()
        self.shared_memory = shared_memory or SharedMemoryBus()
        self.bus = bus or event_bus

        # Agent hierarchy: agent_id -> AgentNode
        self._hierarchy: Dict[str, AgentNode] = {}

        # Reference to AgentManager (set during server startup)
        self._agent_manager = None

        # Background message listeners
        self._listener_tasks: Dict[str, asyncio.Task] = {}

    def set_agent_manager(self, manager):
        """Inject the AgentManager reference (avoids circular import)."""
        self._agent_manager = manager

    # ── Agent Registration ───────────────────────────────────────

    def register_agent(
        self,
        agent_id: str,
        parent_id: Optional[str] = None,
        delegation_id: Optional[str] = None,
    ):
        """Register an agent in the orchestrator hierarchy."""
        node = AgentNode(
            agent_id=agent_id,
            parent_id=parent_id,
            is_sub_agent=parent_id is not None,
            delegation_id=delegation_id,
        )
        self._hierarchy[agent_id] = node

        # Register with message router
        self.router.register_agent(agent_id)

        # Update parent's children list
        if parent_id and parent_id in self._hierarchy:
            self._hierarchy[parent_id].children.append(agent_id)

        logger.info(
            "Orchestrator: registered agent %s (parent=%s, sub_agent=%s)",
            agent_id, parent_id, node.is_sub_agent,
        )

    def unregister_agent(self, agent_id: str):
        """Remove an agent from the orchestrator."""
        node = self._hierarchy.pop(agent_id, None)
        if node:
            # Clean up parent's children list
            if node.parent_id and node.parent_id in self._hierarchy:
                parent = self._hierarchy[node.parent_id]
                parent.children = [c for c in parent.children if c != agent_id]

            # Cancel listener task
            task = self._listener_tasks.pop(agent_id, None)
            if task:
                task.cancel()

            self.router.unregister_agent(agent_id)

    # ── Sub-Agent Spawning ───────────────────────────────────────

    async def spawn_sub_agent(
        self,
        parent_id: str,
        config: Dict[str, Any],
        goal: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Spawn a new sub-agent as a child of the given parent.

        If `goal` is provided, the sub-agent immediately starts executing it.
        Returns the sub-agent's info dict, or None if spawning failed.
        """
        if self._agent_manager is None:
            logger.error("AgentManager not set — cannot spawn sub-agents")
            return None

        # Ensure parent exists
        if parent_id not in self._hierarchy:
            logger.warning("Parent agent %s not in hierarchy", parent_id)
            return None

        # Add parent context to config
        config.setdefault("name", f"sub-{parent_id[:6]}-{uuid4().hex[:4]}")
        config.setdefault("description", f"Sub-agent spawned by {parent_id}")
        config["metadata"] = {
            **config.get("metadata", {}),
            "parent_id": parent_id,
            "is_sub_agent": True,
        }

        # Create the agent via AgentManager
        agent = self._agent_manager.register(config)

        # Register in hierarchy
        self.register_agent(agent.id, parent_id=parent_id)

        # Emit spawn event
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={
                "multi_agent_event": "agent_spawn",
                "parent_id": parent_id,
                "child_id": agent.id,
                "child_name": agent.name,
                "goal": goal,
            },
            agent_id=parent_id,
        ))

        # Start message listener for the sub-agent
        self._start_message_listener(agent.id)

        # If a goal is provided, start execution
        if goal:
            delegation = await self.router.delegate(parent_id, agent.id, goal)
            self._hierarchy[agent.id].delegation_id = delegation.id

            # Run the agent in background
            asyncio.create_task(self._run_sub_agent(agent, goal, delegation.id))

        return agent.get_info()

    async def _run_sub_agent(self, agent, goal: str, delegation_id: str):
        """Run a sub-agent and report its result."""
        try:
            result = await agent.run(goal)
            final_result = result.final_result or "Task completed with no output."

            # Report result back via delegation
            await self.router.report_delegation_result(
                delegation_id=delegation_id,
                child_id=agent.id,
                result=final_result,
                success=result.status == "completed",
            )

            # Update hierarchy status
            if agent.id in self._hierarchy:
                self._hierarchy[agent.id].status = "completed"

            # Emit completion event
            await self.bus.publish(StreamEvent(
                type=EventType.INFO,
                data={
                    "multi_agent_event": "sub_agent_complete",
                    "child_id": agent.id,
                    "delegation_id": delegation_id,
                    "result": final_result[:500],
                    "success": True,
                },
                agent_id=agent.id,
            ))

        except Exception as exc:
            logger.exception("Sub-agent %s failed: %s", agent.id, exc)

            await self.router.report_delegation_result(
                delegation_id=delegation_id,
                child_id=agent.id,
                result=str(exc),
                success=False,
            )

            if agent.id in self._hierarchy:
                self._hierarchy[agent.id].status = "failed"

            await self.bus.publish(StreamEvent.error(
                message=f"Sub-agent {agent.id} failed: {exc}",
                agent_id=agent.id,
            ))

    # ── Message Listening ────────────────────────────────────────

    def _start_message_listener(self, agent_id: str):
        """Start a background task that processes incoming messages for an agent."""
        if agent_id in self._listener_tasks:
            return

        async def _listener():
            inbox = self.router.get_inbox(agent_id)
            if not inbox:
                return
            while True:
                try:
                    message: AgentMessage = await inbox.get()
                    await self._handle_incoming_message(agent_id, message)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Message listener error for %s: %s", agent_id, exc)

        task = asyncio.create_task(_listener())
        self._listener_tasks[agent_id] = task

    async def _handle_incoming_message(self, agent_id: str, message: AgentMessage):
        """Process an incoming message for an agent."""
        # Emit to UI
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={
                "multi_agent_event": "agent_message",
                "direction": "incoming",
                "message": message.to_dict(),
            },
            agent_id=agent_id,
        ))

        # Handle delegation results (parent receives child's output)
        if message.type == MessageType.DELEGATE_RESULT:
            delegation_id = message.payload.get("delegation_id")
            if delegation_id:
                # Write result to shared memory so parent agent can access it
                await self.shared_memory.write(
                    channel=f"delegation_{delegation_id}",
                    key="result",
                    value={
                        "success": message.payload.get("success", False),
                        "result": message.content,
                        "goal": message.payload.get("goal", ""),
                        "child_id": message.sender_id,
                    },
                    agent_id=message.sender_id,
                )

    # ── Inter-Agent Communication (Convenience) ──────────────────

    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Send a direct message from one agent to another."""
        msg = await self.router.send_direct(sender_id, receiver_id, content, payload)

        # Emit to UI
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={
                "multi_agent_event": "agent_message",
                "direction": "outgoing",
                "message": msg.to_dict(),
            },
            agent_id=sender_id,
        ))
        return msg

    async def broadcast_message(
        self,
        sender_id: str,
        content: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Broadcast a message from one agent to all others."""
        msg = await self.router.broadcast(sender_id, content, payload)

        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={
                "multi_agent_event": "agent_broadcast",
                "message": msg.to_dict(),
            },
            agent_id=sender_id,
        ))
        return msg

    # ── Shared Memory Operations (Agent-Facing) ──────────────────

    async def write_shared(
        self,
        agent_id: str,
        channel: str,
        key: str,
        value: Any,
    ):
        """Write to shared memory on behalf of an agent."""
        entry = await self.shared_memory.write(channel, key, value, agent_id)

        await self.bus.publish(StreamEvent(
            type=EventType.MEMORY_WRITE,
            data={
                "multi_agent_event": "shared_memory_write",
                "channel": channel,
                "key": key,
                "agent_id": agent_id,
            },
            agent_id=agent_id,
        ))
        return entry

    def read_shared(
        self,
        channel: str,
        key: Optional[str] = None,
    ):
        """Read from shared memory."""
        if key:
            entry = self.shared_memory.read(channel, key)
            return entry.to_dict() if entry else None
        else:
            entries = self.shared_memory.read_channel(channel)
            return [e.to_dict() for e in entries]

    # ── Hierarchy Queries ────────────────────────────────────────

    def get_hierarchy(self) -> Dict[str, Any]:
        """Get the full agent hierarchy tree."""
        roots = [
            n for n in self._hierarchy.values()
            if not n.is_sub_agent
        ]

        def _build_tree(node: AgentNode) -> Dict[str, Any]:
            return {
                **node.to_dict(),
                "children_detail": [
                    _build_tree(self._hierarchy[cid])
                    for cid in node.children
                    if cid in self._hierarchy
                ],
            }

        return {
            "roots": [_build_tree(r) for r in roots],
            "total_agents": len(self._hierarchy),
            "sub_agents": sum(1 for n in self._hierarchy.values() if n.is_sub_agent),
        }

    def get_agent_node(self, agent_id: str) -> Optional[Dict[str, Any]]:
        node = self._hierarchy.get(agent_id)
        return node.to_dict() if node else None

    def get_children(self, agent_id: str) -> List[str]:
        node = self._hierarchy.get(agent_id)
        return node.children if node else []

    def get_parent(self, agent_id: str) -> Optional[str]:
        node = self._hierarchy.get(agent_id)
        return node.parent_id if node else None

    # ── Cleanup ──────────────────────────────────────────────────

    async def terminate_sub_agent(self, agent_id: str):
        """Terminate a sub-agent and clean up."""
        node = self._hierarchy.get(agent_id)
        if not node:
            return

        # Recursively terminate children
        for child_id in list(node.children):
            await self.terminate_sub_agent(child_id)

        # Update status
        node.status = "terminated"

        # Emit event
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={
                "multi_agent_event": "agent_terminate",
                "agent_id": agent_id,
                "parent_id": node.parent_id,
            },
            agent_id=agent_id,
        ))

        # Remove from manager
        if self._agent_manager:
            self._agent_manager.remove(agent_id)

        self.unregister_agent(agent_id)

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self._hierarchy),
            "root_agents": sum(
                1 for n in self._hierarchy.values() if not n.is_sub_agent
            ),
            "sub_agents": sum(
                1 for n in self._hierarchy.values() if n.is_sub_agent
            ),
            "active_agents": sum(
                1 for n in self._hierarchy.values() if n.status == "active"
            ),
            "message_router": self.router.get_stats(),
            "shared_memory": self.shared_memory.get_stats(),
        }
