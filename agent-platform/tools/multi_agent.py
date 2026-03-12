"""
Multi-Agent Tools
=================
Built-in tools that enable agents to interact with other agents (v3).

These tools are automatically registered via the @tool decorator and
allow any agent to:
  - Spawn sub-agents to handle sub-tasks
  - Send messages to other agents
  - Read/write shared memory channels
  - Check on delegated task status
  - List available agents
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from tools.registry import tool

logger = logging.getLogger("agentforge.tools.multi_agent")

# These references are injected at server startup
_orchestrator = None
_agent_manager = None


def set_orchestrator(orchestrator):
    """Inject the orchestrator reference (called during server startup)."""
    global _orchestrator
    _orchestrator = orchestrator


def set_agent_manager(manager):
    """Inject the agent manager reference (called during server startup)."""
    global _agent_manager
    _agent_manager = manager


def _get_orchestrator():
    if _orchestrator is None:
        raise RuntimeError("Multi-agent orchestrator not initialized")
    return _orchestrator


def _get_manager():
    if _agent_manager is None:
        raise RuntimeError("Agent manager not initialized")
    return _agent_manager


# ═══════════════════════════════════════════════════════════════════════
# Sub-Agent Spawning
# ═══════════════════════════════════════════════════════════════════════

@tool(
    name="spawn_agent",
    description=(
        "Spawn a new sub-agent to handle a specific sub-task. "
        "The sub-agent will execute the goal autonomously and return results. "
        "Use this when a task can be broken down into independent sub-tasks."
    ),
    parameters={
        "goal": {
            "type": "string",
            "description": "The goal/task for the sub-agent to accomplish",
            "required": True,
        },
        "agent_name": {
            "type": "string",
            "description": "Optional name for the sub-agent",
            "required": False,
        },
        "parent_agent_id": {
            "type": "string",
            "description": "ID of the parent agent spawning this sub-agent",
            "required": True,
        },
    },
    tags=["multi-agent", "delegation"],
)
async def spawn_agent(
    goal: str,
    parent_agent_id: str,
    agent_name: str = "",
) -> str:
    """Spawn a sub-agent to handle a sub-task."""
    orchestrator = _get_orchestrator()

    config = {}
    if agent_name:
        config["name"] = agent_name

    result = await orchestrator.spawn_sub_agent(
        parent_id=parent_agent_id,
        config=config,
        goal=goal,
    )

    if result:
        return (
            f"Sub-agent '{result['name']}' (ID: {result['id']}) spawned successfully "
            f"and is working on: {goal}\n"
            f"The sub-agent will execute autonomously. "
            f"Use 'check_delegation_status' with the agent ID to check progress."
        )
    else:
        return f"Failed to spawn sub-agent for goal: {goal}"


# ═══════════════════════════════════════════════════════════════════════
# Inter-Agent Messaging
# ═══════════════════════════════════════════════════════════════════════

@tool(
    name="send_agent_message",
    description=(
        "Send a message to another agent. "
        "Use for coordination, information sharing, or requesting help from a specific agent."
    ),
    parameters={
        "sender_id": {
            "type": "string",
            "description": "ID of the agent sending the message",
            "required": True,
        },
        "receiver_id": {
            "type": "string",
            "description": "ID of the agent to send the message to",
            "required": True,
        },
        "message": {
            "type": "string",
            "description": "The message content to send",
            "required": True,
        },
    },
    tags=["multi-agent", "communication"],
)
async def send_agent_message(
    sender_id: str,
    receiver_id: str,
    message: str,
) -> str:
    """Send a direct message to another agent."""
    orchestrator = _get_orchestrator()

    msg = await orchestrator.send_message(
        sender_id=sender_id,
        receiver_id=receiver_id,
        content=message,
    )

    return f"Message sent to agent {receiver_id} (message_id: {msg.id})"


@tool(
    name="broadcast_message",
    description=(
        "Broadcast a message to ALL registered agents. "
        "Use for announcements or when you need input from multiple agents."
    ),
    parameters={
        "sender_id": {
            "type": "string",
            "description": "ID of the agent broadcasting the message",
            "required": True,
        },
        "message": {
            "type": "string",
            "description": "The message to broadcast to all agents",
            "required": True,
        },
    },
    tags=["multi-agent", "communication"],
)
async def broadcast_message(
    sender_id: str,
    message: str,
) -> str:
    """Broadcast a message to all registered agents."""
    orchestrator = _get_orchestrator()

    msg = await orchestrator.broadcast_message(
        sender_id=sender_id,
        content=message,
    )

    return f"Message broadcast to all agents (message_id: {msg.id})"


# ═══════════════════════════════════════════════════════════════════════
# Shared Memory
# ═══════════════════════════════════════════════════════════════════════

@tool(
    name="write_shared_memory",
    description=(
        "Write data to a shared memory channel accessible by all agents. "
        "Use for sharing results, intermediate data, or coordinating state between agents."
    ),
    parameters={
        "channel": {
            "type": "string",
            "description": "The shared memory channel name (e.g., 'research_results', 'task_data')",
            "required": True,
        },
        "key": {
            "type": "string",
            "description": "The key to store the data under within the channel",
            "required": True,
        },
        "value": {
            "type": "string",
            "description": "The data to store (will be stored as-is)",
            "required": True,
        },
        "agent_id": {
            "type": "string",
            "description": "ID of the agent writing the data",
            "required": True,
        },
    },
    tags=["multi-agent", "shared-memory"],
)
async def write_shared_memory(
    channel: str,
    key: str,
    value: str,
    agent_id: str,
) -> str:
    """Write data to a shared memory channel."""
    orchestrator = _get_orchestrator()

    await orchestrator.write_shared(
        agent_id=agent_id,
        channel=channel,
        key=key,
        value=value,
    )

    return f"Data written to shared memory: channel='{channel}', key='{key}'"


@tool(
    name="read_shared_memory",
    description=(
        "Read data from a shared memory channel. "
        "Use to retrieve data written by other agents or check shared state."
    ),
    parameters={
        "channel": {
            "type": "string",
            "description": "The shared memory channel to read from",
            "required": True,
        },
        "key": {
            "type": "string",
            "description": "Optional specific key to read. If not provided, returns all entries in the channel.",
            "required": False,
        },
    },
    tags=["multi-agent", "shared-memory"],
)
async def read_shared_memory(
    channel: str,
    key: str = "",
) -> str:
    """Read data from a shared memory channel."""
    orchestrator = _get_orchestrator()

    result = orchestrator.read_shared(channel, key if key else None)

    if result is None:
        return f"No data found in channel='{channel}'" + (f", key='{key}'" if key else "")

    if isinstance(result, list):
        if not result:
            return f"Channel '{channel}' is empty"
        entries = []
        for entry in result[:20]:  # Limit to 20 entries
            entries.append(f"  [{entry.get('key', '?')}] by {entry.get('written_by', '?')}: {str(entry.get('value', ''))[:200]}")
        return f"Shared memory channel '{channel}' ({len(result)} entries):\n" + "\n".join(entries)

    return f"Shared memory [{channel}/{key}]: {str(result.get('value', ''))[:500]}"


# ═══════════════════════════════════════════════════════════════════════
# Agent Discovery & Status
# ═══════════════════════════════════════════════════════════════════════

@tool(
    name="list_agents",
    description=(
        "List all registered agents in the platform. "
        "Returns agent IDs, names, status, and capabilities. "
        "Use to discover available agents for collaboration."
    ),
    parameters={},
    tags=["multi-agent", "discovery"],
)
async def list_agents() -> str:
    """List all registered agents."""
    manager = _get_manager()

    agents = manager.list_agents()
    if not agents:
        return "No agents currently registered."

    lines = [f"Registered agents ({len(agents)}):"]
    for a in agents:
        sub = " [sub-agent]" if a.get("is_sub_agent") else ""
        lines.append(
            f"  - {a['name']} (ID: {a['id']}) status={a['status']}{sub} "
            f"tools={a.get('tools', [])}"
        )
    return "\n".join(lines)


@tool(
    name="check_delegation_status",
    description=(
        "Check the status of a delegated task or sub-agent. "
        "Use after spawning a sub-agent to see if it has completed its task."
    ),
    parameters={
        "agent_id": {
            "type": "string",
            "description": "ID of the parent agent to check delegations for",
            "required": True,
        },
    },
    tags=["multi-agent", "delegation"],
)
async def check_delegation_status(agent_id: str) -> str:
    """Check delegation status for an agent's sub-tasks."""
    orchestrator = _get_orchestrator()

    delegations = orchestrator.router.get_delegations_by_parent(agent_id)
    if not delegations:
        return f"No delegated tasks found for agent {agent_id}"

    lines = [f"Delegated tasks for agent {agent_id} ({len(delegations)}):"]
    for d in delegations:
        result_preview = ""
        if d.result:
            result_preview = f" result: {d.result[:100]}..."
        lines.append(
            f"  - [{d.status.upper()}] → {d.child_id}: {d.goal[:80]}{result_preview}"
        )
    return "\n".join(lines)


@tool(
    name="get_agent_hierarchy",
    description=(
        "Get the full hierarchy tree of agents showing parent-child relationships. "
        "Useful for understanding the multi-agent structure."
    ),
    parameters={},
    tags=["multi-agent", "discovery"],
)
async def get_agent_hierarchy() -> str:
    """Get the agent hierarchy tree."""
    orchestrator = _get_orchestrator()

    hierarchy = orchestrator.get_hierarchy()
    total = hierarchy.get("total_agents", 0)
    sub_count = hierarchy.get("sub_agents", 0)

    lines = [f"Agent Hierarchy ({total} total, {sub_count} sub-agents):"]

    def _format_tree(node: Dict, indent: int = 0):
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        status = node.get("status", "unknown")
        lines.append(f"{prefix}{node['agent_id']} [{status}]")
        for child in node.get("children_detail", []):
            _format_tree(child, indent + 1)

    for root in hierarchy.get("roots", []):
        _format_tree(root)

    return "\n".join(lines) if len(lines) > 1 else "No agents in hierarchy."
