"""
Multi-Agent Routes
==================
REST API endpoints for v3 multi-agent features.

POST   /multi-agent/spawn                  → spawn a sub-agent
POST   /multi-agent/message                → send inter-agent message
POST   /multi-agent/broadcast              → broadcast to all agents
GET    /multi-agent/hierarchy              → get agent hierarchy tree
GET    /multi-agent/agents/{id}/children   → list sub-agents
GET    /multi-agent/agents/{id}/messages   → get message history
GET    /multi-agent/agents/{id}/delegations → get delegated tasks
POST   /multi-agent/agents/{id}/terminate  → terminate a sub-agent

Shared Memory:
GET    /shared-memory/channels             → list all channels
GET    /shared-memory/{channel}            → read from channel
POST   /shared-memory/{channel}            → write to channel
DELETE /shared-memory/{channel}            → clear channel
GET    /shared-memory/stats                → shared memory stats

Communication:
GET    /multi-agent/stats                  → overall multi-agent stats
GET    /multi-agent/conversation/{a}/{b}   → message thread between two agents
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.orchestrator import MultiAgentOrchestrator

router = APIRouter(tags=["multi-agent"])

# Injected at startup (see server.py)
_orchestrator: Optional[MultiAgentOrchestrator] = None


def set_orchestrator(orchestrator: MultiAgentOrchestrator):
    global _orchestrator
    _orchestrator = orchestrator


def _get_orchestrator() -> MultiAgentOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=500, detail="Multi-agent orchestrator not initialized")
    return _orchestrator


# ─── Request Models ───────────────────────────────────────────────

class SpawnAgentRequest(BaseModel):
    parent_agent_id: str
    goal: str
    agent_name: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    sender_id: str
    receiver_id: str
    content: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class BroadcastRequest(BaseModel):
    sender_id: str
    content: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class SharedMemoryWriteRequest(BaseModel):
    key: str
    value: Any
    agent_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Endpoints
# ═══════════════════════════════════════════════════════════════════════

@router.post("/multi-agent/spawn", status_code=201)
async def spawn_sub_agent(req: SpawnAgentRequest):
    """Spawn a new sub-agent under a parent agent."""
    orch = _get_orchestrator()

    config = dict(req.config)
    if req.agent_name:
        config["name"] = req.agent_name

    result = await orch.spawn_sub_agent(
        parent_id=req.parent_agent_id,
        config=config,
        goal=req.goal if req.goal else None,
    )

    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to spawn sub-agent (parent: {req.parent_agent_id})",
        )

    return {
        "status": "spawned",
        "agent": result,
        "parent_id": req.parent_agent_id,
        "goal": req.goal,
    }


@router.post("/multi-agent/message")
async def send_message(req: SendMessageRequest):
    """Send a direct message between agents."""
    orch = _get_orchestrator()

    msg = await orch.send_message(
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        content=req.content,
        payload=req.payload,
    )

    return {"status": "sent", "message": msg.to_dict()}


@router.post("/multi-agent/broadcast")
async def broadcast(req: BroadcastRequest):
    """Broadcast a message to all agents."""
    orch = _get_orchestrator()

    msg = await orch.broadcast_message(
        sender_id=req.sender_id,
        content=req.content,
        payload=req.payload,
    )

    return {"status": "broadcast", "message": msg.to_dict()}


@router.get("/multi-agent/hierarchy")
async def get_hierarchy():
    """Get the full agent hierarchy tree."""
    orch = _get_orchestrator()
    return orch.get_hierarchy()


@router.get("/multi-agent/agents/{agent_id}/children")
async def get_children(agent_id: str):
    """List all sub-agents of a given agent."""
    orch = _get_orchestrator()
    children = orch.get_children(agent_id)
    return {"agent_id": agent_id, "children": children}


@router.get("/multi-agent/agents/{agent_id}/messages")
async def get_agent_messages(
    agent_id: str,
    direction: str = "all",
    limit: int = 50,
):
    """Get message history for an agent."""
    orch = _get_orchestrator()
    messages = orch.router.get_messages(agent_id, direction=direction, limit=limit)
    return {
        "agent_id": agent_id,
        "direction": direction,
        "messages": [m.to_dict() for m in messages],
    }


@router.get("/multi-agent/agents/{agent_id}/delegations")
async def get_delegations(agent_id: str):
    """Get all delegated tasks for an agent."""
    orch = _get_orchestrator()
    as_parent = orch.router.get_delegations_by_parent(agent_id)
    as_child = orch.router.get_delegations_by_child(agent_id)
    return {
        "agent_id": agent_id,
        "delegated_to_others": [d.to_dict() for d in as_parent],
        "received_from_parent": [d.to_dict() for d in as_child],
    }


@router.post("/multi-agent/agents/{agent_id}/terminate")
async def terminate_agent(agent_id: str):
    """Terminate a sub-agent and clean up its resources."""
    orch = _get_orchestrator()
    await orch.terminate_sub_agent(agent_id)
    return {"status": "terminated", "agent_id": agent_id}


@router.get("/multi-agent/conversation/{agent_a}/{agent_b}")
async def get_conversation(agent_a: str, agent_b: str, limit: int = 50):
    """Get the message thread between two agents."""
    orch = _get_orchestrator()
    messages = orch.router.get_conversation(agent_a, agent_b, limit=limit)
    return {
        "agents": [agent_a, agent_b],
        "messages": [m.to_dict() for m in messages],
    }


@router.get("/multi-agent/stats")
async def get_multi_agent_stats():
    """Get overall multi-agent statistics."""
    orch = _get_orchestrator()
    return orch.get_stats()


# ═══════════════════════════════════════════════════════════════════════
# Shared Memory Endpoints
# ═══════════════════════════════════════════════════════════════════════

@router.get("/shared-memory/stats")
async def shared_memory_stats():
    """Get shared memory statistics."""
    orch = _get_orchestrator()
    return orch.shared_memory.get_stats()


@router.get("/shared-memory/channels")
async def list_channels():
    """List all shared memory channels."""
    orch = _get_orchestrator()
    channels = orch.shared_memory.list_channels()
    return {"channels": [c.to_dict() for c in channels]}


@router.get("/shared-memory/{channel}")
async def read_channel(channel: str, key: Optional[str] = None, limit: int = 100):
    """Read from a shared memory channel."""
    orch = _get_orchestrator()

    if key:
        entry = orch.shared_memory.read(channel, key)
        if entry is None:
            raise HTTPException(404, detail=f"Key '{key}' not found in channel '{channel}'")
        return entry.to_dict()
    else:
        entries = orch.shared_memory.read_channel(channel, limit=limit)
        return {
            "channel": channel,
            "entries": [e.to_dict() for e in entries],
        }


@router.post("/shared-memory/{channel}")
async def write_to_channel(channel: str, req: SharedMemoryWriteRequest):
    """Write data to a shared memory channel."""
    orch = _get_orchestrator()

    entry = await orch.shared_memory.write(
        channel=channel,
        key=req.key,
        value=req.value,
        agent_id=req.agent_id,
        metadata=req.metadata,
    )

    return {"status": "written", "entry": entry.to_dict()}


@router.delete("/shared-memory/{channel}")
async def clear_channel(channel: str):
    """Clear all entries in a shared memory channel."""
    orch = _get_orchestrator()
    orch.shared_memory.clear_channel(channel)
    return {"status": "cleared", "channel": channel}
