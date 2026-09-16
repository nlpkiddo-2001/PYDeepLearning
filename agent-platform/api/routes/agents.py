"""
Agent Routes
============
REST API endpoints for agent CRUD and operations.

POST   /agents              → register a new agent
GET    /agents              → list all agents + status
GET    /agents/{id}         → get agent details
DELETE /agents/{id}         → remove an agent
POST   /agents/{id}/run     → run with a goal (returns task_id)
GET    /agents/{id}/status  → current run state
POST   /agents/{id}/chat    → send a chat message
POST   /agents/{id}/chat/stream → stream a chat response (SSE)
GET    /agents/{id}/memory  → inspect memory state
DELETE /agents/{id}/memory  → clear memory
PATCH  /agents/{id}/config  → update agent LLM config
POST   /agents/{id}/reset   → reset run stats and memory
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agent import AgentManager

logger = logging.getLogger("agentforge.routes.agents")

router = APIRouter(prefix="/agents", tags=["agents"])

# The manager is injected at startup (see server.py)
_manager: Optional[AgentManager] = None

# Strong references to background tasks to prevent garbage collection.
# Tasks remove themselves from this set when they complete.
_background_tasks: Set[asyncio.Task] = set()


def set_manager(manager: AgentManager):
    global _manager
    _manager = manager


def _get_manager() -> AgentManager:
    if _manager is None:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    return _manager


# ─── Request / Response Models ────────────────────────────────────

class RegisterAgentRequest(BaseModel):
    id: Optional[str] = None
    name: str = "my-agent"
    description: str = ""
    goal_prompt: str = ""
    llm: Dict[str, Any] = Field(default_factory=dict)
    tools: list = Field(default_factory=list)
    max_steps: int = 100
    max_retries: int = 3
    template: str = ""


class RunRequest(BaseModel):
    goal: str


class ChatRequest(BaseModel):
    message: str
    clear_history: bool = False  # set True to wipe memory before this message


class ConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: Optional[str] = None
    jwt_expiry_minutes: Optional[int] = None
    validate: Optional[bool] = True  # validate token on provider switch (default: yes)


# ─── Endpoints ────────────────────────────────────────────────────

@router.post("", status_code=201)
async def register_agent(req: RegisterAgentRequest):
    """Register a new agent."""
    manager = _get_manager()
    agent = manager.register(req.model_dump(exclude_none=True))
    return {"status": "created", "agent": agent.get_info()}


@router.get("")
async def list_agents():
    """List all registered agents."""
    manager = _get_manager()
    return {"agents": manager.list_agents()}


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    """Get details for a specific agent."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent.get_info()


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Remove an agent."""
    manager = _get_manager()
    if not manager.remove(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {"status": "deleted", "agent_id": agent_id}


@router.post("/{agent_id}/run")
async def run_agent(agent_id: str, req: RunRequest):
    """Start an autonomous run with a goal. Returns immediately with task_id."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    # Run in background so the endpoint returns immediately
    async def _background_run():
        logger.info("Background run starting for agent %s — goal: %s", agent_id, req.goal[:120])
        try:
            result = await agent.run(req.goal)
            logger.info(
                "Background run finished for agent %s — status=%s, steps=%d",
                agent_id, result.status, len(result.steps),
            )
        except asyncio.CancelledError:
            logger.warning("Agent run cancelled: %s", agent_id)
        except Exception as exc:
            logger.exception("Agent run failed for %s: %s", agent_id, exc)

    # Store strong reference to prevent garbage collection (critical!)
    task = asyncio.create_task(_background_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "status": "started",
        "agent_id": agent_id,
        "goal": req.goal,
        "message": "Connect to WebSocket /agents/{id}/stream for live updates",
    }


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get the current status and run history."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {
        "agent_id": agent_id,
        "status": agent.status,
        "runs": agent.get_runs(),
    }


@router.get("/{agent_id}/events")
async def get_agent_events(agent_id: str, since: str | None = None):
    """Return buffered stream events (polling fallback for missed WS events)."""
    from core.streaming import event_bus
    events = event_bus.get_event_history(agent_id, since_event_id=since)
    return {
        "agent_id": agent_id,
        "events": [e.to_json() for e in events],
    }


@router.post("/{agent_id}/chat")
async def chat_with_agent(agent_id: str, req: ChatRequest):
    """Send a chat message to the agent and get a response."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    try:
        response = await agent.chat(req.message, clear_history=req.clear_history)
        return {"agent_id": agent_id, "response": response}
    except (ConnectionError, TimeoutError) as exc:
        logger.error("LLM unreachable for agent %s: %s", agent_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider is unreachable: {exc}",
        )
    except Exception as exc:
        logger.exception("Chat failed for agent %s: %s", agent_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {exc}",
        )


@router.post("/{agent_id}/chat/stream")
async def chat_stream(agent_id: str, req: ChatRequest):
    """Stream a chat response using Server-Sent Events."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    async def event_generator():
        try:
            async for chunk in agent.chat_stream(req.message, clear_history=req.clear_history):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.exception("Chat stream failed for agent %s: %s", agent_id, exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{agent_id}/reset")
async def reset_agent(agent_id: str):
    """Reset agent stats, run history, and memory."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    agent.reset_stats()
    return {"status": "reset", "agent_id": agent_id}


@router.get("/{agent_id}/memory")
async def inspect_memory(agent_id: str):
    """Inspect the agent's memory state."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {
        "agent_id": agent_id,
        "stats": agent.memory.get_stats(agent_id),
        "recent_history": agent.memory.get_conversation_history(agent_id, limit=10),
    }


@router.delete("/{agent_id}/memory")
async def clear_memory(agent_id: str):
    """Clear all memory for an agent."""
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    agent.memory.clear_all(agent_id)
    return {"status": "cleared", "agent_id": agent_id}


@router.patch("/{agent_id}/config")
async def update_config(agent_id: str, req: ConfigUpdateRequest):
    """Update agent LLM configuration at runtime.

    When switching to the vLLM provider, the endpoint validates the token
    (JWT or static api_key) by hitting the /v1/models endpoint before
    committing the change. Pass `"validate": false` to skip.
    """
    manager = _get_manager()
    agent = manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    update = req.model_dump(exclude_none=True)
    should_validate = update.pop("validate", True)

    if "provider" in update:
        # ── vLLM token validation ────────────────────────────────
        if update["provider"] == "vllm" and should_validate:
            from llm.base import create_provider

            try:
                temp_provider = create_provider(update)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create vLLM provider: {exc}",
                )

            result = await temp_provider.validate_token()
            if not result.get("valid"):
                reason = result.get("error", "Unknown error")
                raise HTTPException(
                    status_code=401,
                    detail=(
                        f"vLLM token validation failed: {reason}. "
                        f"Ensure base_url is reachable and jwt_secret "
                        f"(or api_key) is correct."
                    ),
                )

            logger.info(
                "vLLM token validated for agent %s — available models: %s",
                agent_id, result.get("models", []),
            )

        # Full provider switch
        agent.switch_provider(update)
    else:
        # Partial update
        agent.update_llm_config(**update)

    return {
        "status": "updated",
        "agent_id": agent_id,
        "llm": {
            "provider": agent.llm.provider_name,
            "model": agent.llm.model,
            "temperature": agent.llm.temperature,
        },
    }
