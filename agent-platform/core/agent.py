"""
Agent Module
============
The Agent class ties together the planner, tools, memory, and LLM provider.
It represents a single named agent instance with its own config and context.

v3: Added multi-agent support — agents can spawn sub-agents, communicate
with each other via MessageRouter, and share data through SharedMemoryBus.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from core.memory import MemoryManager
from core.planner import ReActPlanner, PlannerResult
from core.streaming import EventBus, StreamEvent, EventType, event_bus
from core.executor import ToolExecutor
from llm.base import LLMProvider, create_provider
from tools.registry import ToolRegistry

logger = logging.getLogger("agentforge.agent")


class AgentConfig:
    """Configuration for a single agent instance."""

    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get("id", uuid4().hex[:8])
        self.name: str = data.get("name", f"agent-{self.id}")
        self.description: str = data.get("description", "")
        self.goal_prompt: str = data.get("goal_prompt", "")
        self.llm: Dict[str, Any] = data.get("llm", {})
        self.tools: List[str] = data.get("tools", [])
        self.memory: Dict[str, Any] = data.get("memory", {})
        self.max_steps: int = data.get("max_steps", 15)
        self.max_retries: int = data.get("max_retries", 3)
        self.template: str = data.get("template", "")
        self.metadata: Dict[str, Any] = data.get("metadata", {})
        # v3: multi-agent config
        self.can_spawn: bool = data.get("can_spawn", True)
        self.max_sub_agents: int = data.get("max_sub_agents", 5)
        self.parent_id: Optional[str] = data.get("metadata", {}).get("parent_id")
        self.is_sub_agent: bool = data.get("metadata", {}).get("is_sub_agent", False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal_prompt": self.goal_prompt,
            "llm": self.llm,
            "tools": self.tools,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "template": self.template,
            "can_spawn": self.can_spawn,
            "max_sub_agents": self.max_sub_agents,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)


class Agent:
    """
    A single autonomous agent instance.

    Brings together:
    - An LLM provider for reasoning
    - A tool registry for action execution
    - Memory (short + long term)
    - A ReAct planner for the goal→reason→act loop
    - Streaming for real-time UI updates
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        bus: Optional[EventBus] = None,
        orchestrator: Optional[Any] = None,  # MultiAgentOrchestrator
    ):
        self.config = config
        self.id = config.id
        self.name = config.name
        self.llm = llm_provider
        self.tools = tool_registry
        self.memory = memory_manager
        self.bus = bus or event_bus
        self.executor = ToolExecutor(tool_registry)
        self.orchestrator = orchestrator  # v3: multi-agent orchestrator ref

        self.planner = ReActPlanner(
            llm=self.llm,
            tool_registry=self.tools,
            memory=self.memory,
            event_bus_instance=self.bus,
            max_steps=config.max_steps,
            max_retries=config.max_retries,
        )

        # Runtime stats
        self._runs: List[PlannerResult] = []
        self._status: str = "idle"  # idle | running | error
        self._created_at: float = time.time()

        # v3: Multi-agent tracking
        self._sub_agents: List[str] = []  # IDs of spawned sub-agents
        self._parent_id: Optional[str] = config.parent_id
        self._is_sub_agent: bool = config.is_sub_agent

    @property
    def status(self) -> str:
        return self._status

    # ── Run (goal-based) ─────────────────────────────────────────

    async def run(self, goal: str) -> PlannerResult:
        """Execute an autonomous run with the given goal."""
        self._status = "running"
        try:
            result = await self.planner.run(agent_id=self.id, goal=goal)
            self._runs.append(result)
            return result
        except Exception as exc:
            self._status = "error"
            raise
        finally:
            if self._status == "running":
                self._status = "idle"

    # ── Chat (conversational) ────────────────────────────────────

    async def chat(self, message: str) -> str:
        """Send a conversational message and get a response."""
        logger.info("Chat message for agent %s: %s", self.id, message[:120])

        # Add user message to memory
        self.memory.add_message(self.id, "user", message)

        # Build messages from history
        history = self.memory.get_conversation_history(self.id, limit=30)

        # Add system prompt
        system_prompt = self.config.goal_prompt or (
            f"You are {self.name}, an AI assistant. "
            f"{self.config.description}"
        )
        messages = [{"role": "system", "content": system_prompt}] + history

        # Emit a streaming event so the WebSocket shows progress
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={"message": f"Processing chat: {message[:100]}"},
            agent_id=self.id,
        ))

        try:
            # Check if this looks like a task (contains action verbs)
            task_keywords = ["find", "search", "get", "create", "send", "write", "analyze", "scrape"]
            is_task = any(kw in message.lower() for kw in task_keywords)

            if is_task and self.tools.names():
                # Use the planner for task-like messages
                result = await self.run(message)
                response = result.final_result or "I couldn't complete that task."
            else:
                # Simple conversation
                await self.bus.publish(StreamEvent(
                    type=EventType.LLM_REQUEST,
                    data={"message": "Sending to LLM..."},
                    agent_id=self.id,
                ))
                llm_resp = await self.llm.complete(messages)
                response = llm_resp.content
                await self.bus.publish(StreamEvent(
                    type=EventType.LLM_RESPONSE,
                    data={"message": "LLM responded", "tokens": llm_resp.usage},
                    agent_id=self.id,
                ))

            self.memory.add_message(self.id, "assistant", response)
            logger.info("Chat response for agent %s: %s", self.id, response[:120])
            return response

        except Exception as exc:
            logger.exception("Chat failed for agent %s: %s", self.id, exc)
            await self.bus.publish(StreamEvent.error(
                message=f"Chat error: {exc}",
                agent_id=self.id,
            ))
            raise

    # ── Config management ────────────────────────────────────────

    def update_llm_config(self, **kwargs: Any):
        """Update the LLM provider settings at runtime."""
        self.llm.update_config(**kwargs)
        # Update stored config too
        self.config.llm.update(kwargs)

    def switch_provider(self, provider_config: Dict[str, Any]):
        """Switch to a different LLM provider."""
        new_provider = create_provider(provider_config)
        self.llm = new_provider
        self.planner.llm = new_provider
        self.config.llm = provider_config

    # ── Introspection ────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        last_run = self._runs[-1].to_dict() if self._runs else None
        return {
            "id": self.id,
            "name": self.name,
            "description": self.config.description,
            "status": self._status,
            "tools": self.tools.names(),
            "llm": {
                "provider": self.llm.provider_name,
                "model": self.llm.model,
            },
            "runs_count": len(self._runs),
            "total_tokens": sum(r.total_tokens for r in self._runs),
            "total_tool_calls": sum(r.tool_calls for r in self._runs),
            "memory": self.memory.get_stats(self.id),
            "created_at": self._created_at,
            # v3: Multi-agent info
            "is_sub_agent": self._is_sub_agent,
            "parent_id": self._parent_id,
            "sub_agents": self._sub_agents,
            "can_spawn": self.config.can_spawn,
            # v3.1: Include last run for polling fallback
            "last_run": last_run,
        }

    def get_runs(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._runs]


# ═══════════════════════════════════════════════════════════════════════
# Agent Manager — manages all agent instances
# ═══════════════════════════════════════════════════════════════════════

class AgentManager:
    """
    Central manager for all agent instances.
    Handles creation, lookup, and lifecycle management.
    v3: Integrates with MultiAgentOrchestrator for sub-agent spawning.
    """

    def __init__(
        self,
        default_config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        orchestrator: Optional[Any] = None,  # MultiAgentOrchestrator
    ):
        self._agents: Dict[str, Agent] = {}
        self._default_config = default_config or {}
        self._tool_registry = tool_registry or ToolRegistry()
        self._memory = memory_manager or MemoryManager()
        self._orchestrator = orchestrator  # v3

    def set_orchestrator(self, orchestrator):
        """Inject the orchestrator reference (set after init to break circular deps)."""
        self._orchestrator = orchestrator
        orchestrator.set_agent_manager(self)

    def register(self, config: Dict[str, Any]) -> Agent:
        """Create and register a new agent."""
        agent_config = AgentConfig(config)

        # Merge LLM config with defaults
        llm_config = {**self._default_config.get("llm", {}), **agent_config.llm}
        llm_provider = create_provider(llm_config)

        agent = Agent(
            config=agent_config,
            llm_provider=llm_provider,
            tool_registry=self._tool_registry,
            memory_manager=self._memory,
            orchestrator=self._orchestrator,
        )

        self._agents[agent.id] = agent

        # v3: Register with orchestrator hierarchy
        if self._orchestrator:
            parent_id = config.get("metadata", {}).get("parent_id")
            self._orchestrator.register_agent(agent.id, parent_id=parent_id)

        logger.info("Registered agent: %s (%s)", agent.name, agent.id)
        return agent

    def get(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.get_info() for a in self._agents.values()]

    def remove(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            if self._orchestrator:
                self._orchestrator.unregister_agent(agent_id)
            return True
        return False

    def load_from_directory(self, directory: str):
        """Load agent configs from YAML files in a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                config = AgentConfig.from_yaml(str(yaml_file))
                self.register(config.to_dict())
            except Exception as exc:
                logger.warning("Failed to load agent config %s: %s", yaml_file, exc)
