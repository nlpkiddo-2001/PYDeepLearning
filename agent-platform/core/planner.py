"""
ReAct Planner
=============
The agent brain: Goal → LLM Reasoning → Tool Selection → Execute → Repeat.
Each step is streamed to the UI in real time via the EventBus.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from core.streaming import EventBus, StreamEvent, EventType, event_bus
from core.memory import MemoryManager
from llm.base import LLMProvider, LLMResponse
from tools.registry import ToolRegistry

logger = logging.getLogger("agentforge.planner")

# Default system prompt for the ReAct planner
DEFAULT_PLANNING_PROMPT = """You are an autonomous AI agent. Given a goal, reason step by step,
select the appropriate tool, execute it, observe the result, and repeat
until the goal is achieved or you determine it cannot be completed.

Available tools:
{tool_descriptions}

Always respond in this EXACT JSON format (no extra text):
{{
  "thought": "your reasoning about what to do next",
  "action": "tool_name",
  "action_input": {{ "param": "value" }}
}}

When the goal is complete, respond with:
{{
  "thought": "I have completed the task",
  "action": "finish",
  "action_input": {{ "result": "final summary of what was accomplished" }}
}}

If a tool fails, analyze the error and try an alternative approach. Never give up without trying at least one alternative."""


class PlannerResult:
    """Result of a complete planning run."""

    def __init__(self):
        self.task_id: str = uuid4().hex[:16]
        self.steps: List[Dict[str, Any]] = []
        self.final_result: Optional[str] = None
        self.status: str = "pending"  # pending | running | completed | failed
        self.total_tokens: int = 0
        self.tool_calls: int = 0
        self.started_at: float = time.time()
        self.ended_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "steps": self.steps,
            "final_result": self.final_result,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "duration_seconds": (self.ended_at or time.time()) - self.started_at,
        }


class ReActPlanner:
    """
    Implements the ReAct (Reasoning + Acting) loop.

    For each iteration:
    1. Build context from conversation history + memory
    2. Ask the LLM what to do next
    3. Parse the action (tool call or finish)
    4. Execute the tool
    5. Feed result back and repeat
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        memory: MemoryManager,
        event_bus_instance: Optional[EventBus] = None,
        max_steps: int = 100,
        max_retries: int = 3,
        planning_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.memory = memory
        self.bus = event_bus_instance or event_bus
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.planning_prompt = planning_prompt or DEFAULT_PLANNING_PROMPT

    async def run(self, agent_id: str, goal: str) -> PlannerResult:
        """Execute the full ReAct loop for a given goal."""
        result = PlannerResult()
        result.status = "running"

        # Build tool descriptions for the system prompt
        tool_descs = self._format_tool_descriptions()
        system_prompt = self.planning_prompt.format(tool_descriptions=tool_descs)

        # Initialize messages with system + goal
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Goal: {goal}"},
        ]

        # Add conversation history from memory
        history = self.memory.get_conversation_history(agent_id, limit=20)
        if history:
            messages[1:1] = history  # Insert after system prompt

        # Store the goal in memory
        self.memory.add_message(agent_id, "user", f"Goal: {goal}")

        # Emit info event
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,
            data={"message": f"Starting ReAct loop for goal: {goal}"},
            agent_id=agent_id,
            task_id=result.task_id,
            step=0,
        ))

        # ── ReAct Loop ─────────────────────────────────────────
        for step in range(1, self.max_steps + 1):
            try:
                step_result = await self._execute_step(
                    agent_id=agent_id,
                    task_id=result.task_id,
                    step=step,
                    messages=messages,
                    planner_result=result,
                )

                if step_result.get("finished"):
                    result.final_result = step_result.get("result", "")
                    result.status = "completed"
                    result.ended_at = time.time()

                    # Emit done event
                    await self.bus.publish(StreamEvent.done(
                        result=result.final_result,
                        agent_id=agent_id,
                        task_id=result.task_id,
                        step=step,
                    ))

                    # Store result in long-term memory
                    self.memory.store_long_term(
                        agent_id,
                        f"Goal: {goal}\nResult: {result.final_result}",
                        {"task_id": result.task_id, "type": "task_result"},
                    )
                    self.memory.add_message(agent_id, "assistant", result.final_result)
                    break

            except Exception as exc:
                logger.exception("Step %d failed: %s", step, exc)
                await self.bus.publish(StreamEvent.error(
                    message=str(exc),
                    agent_id=agent_id,
                    task_id=result.task_id,
                    step=step,
                ))
                result.steps.append({"step": step, "error": str(exc)})

        else:
            # Max steps reached without finishing
            result.status = "failed"
            result.final_result = "Max steps reached without completing the goal."
            result.ended_at = time.time()
            await self.bus.publish(StreamEvent.error(
                message="Max steps reached",
                agent_id=agent_id,
                task_id=result.task_id,
                step=self.max_steps,
            ))

        return result

    async def _execute_step(
        self,
        agent_id: str,
        task_id: str,
        step: int,
        messages: List[Dict[str, str]],
        planner_result: PlannerResult,
    ) -> Dict[str, Any]:
        """Execute a single ReAct step: reason → act → observe."""

        # 1. Ask the LLM what to do
        llm_response = await self.llm.complete(messages)
        planner_result.total_tokens += (
            llm_response.usage.get("prompt_tokens", 0)
            + llm_response.usage.get("completion_tokens", 0)
        )

        # 2. Parse the response
        raw_content = llm_response.content.strip()
        parsed = self._parse_action(raw_content)

        thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})

        # Emit plan event
        await self.bus.publish(StreamEvent.plan(
            thought=thought,
            agent_id=agent_id,
            task_id=task_id,
            step=step,
        ))

        planner_result.steps.append({
            "step": step,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        })

        # 3. Check if done
        if action == "finish":
            finish_result = action_input.get("result", str(action_input))
            return {"finished": True, "result": finish_result}

        # 4. Execute the tool
        await self.bus.publish(StreamEvent.tool_call(
            tool_name=action,
            args=action_input,
            agent_id=agent_id,
            task_id=task_id,
            step=step,
        ))

        tool_output = await self._execute_tool_with_retry(
            agent_id=agent_id,
            task_id=task_id,
            step=step,
            tool_name=action,
            tool_args=action_input,
        )

        planner_result.tool_calls += 1

        # Emit tool result
        await self.bus.publish(StreamEvent.tool_result(
            tool_name=action,
            result=tool_output,
            agent_id=agent_id,
            task_id=task_id,
            step=step,
        ))

        # 5. Feed the observation back
        messages.append({"role": "assistant", "content": raw_content})
        messages.append({
            "role": "user",
            "content": f"Observation from {action}:\n{tool_output}\n\nContinue with the next step.",
        })

        return {"finished": False}

    async def _execute_tool_with_retry(
        self,
        agent_id: str,
        task_id: str,
        step: int,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> str:
        """Execute a tool with retry logic."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = await self.tools.execute(tool_name, **tool_args)
                return str(result)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await self.bus.publish(StreamEvent(
                        type=EventType.RETRY,
                        data={
                            "tool": tool_name,
                            "attempt": attempt,
                            "max_retries": self.max_retries,
                            "error": str(exc),
                        },
                        agent_id=agent_id,
                        task_id=task_id,
                        step=step,
                    ))
        return f"ERROR: Tool '{tool_name}' failed after {self.max_retries} attempts: {last_error}"

    def _parse_action(self, content: str) -> Dict[str, Any]:
        """Parse the LLM output into thought/action/action_input.

        Handles multiple response formats:
        - Plain JSON
        - JSON wrapped in ```json ... ``` or ``` ... ```
        - <think>...</think> reasoning blocks followed by JSON (GLM, DeepSeek, etc.)
        - JSON embedded in free-form text
        """
        import re

        cleaned = content

        # 1. Strip <think>...</think> blocks (models like GLM-5, DeepSeek-R1)
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL,
        ).strip()

        # Also handle unclosed <think> blocks (model started thinking but
        # the closing tag is the boundary before JSON).
        if "<think>" in cleaned:
            cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

        # 2. Extract from markdown code blocks
        try:
            if "```json" in cleaned:
                json_str = cleaned.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                return self._normalise_parsed(parsed)
            elif "```" in cleaned:
                json_str = cleaned.split("```")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                return self._normalise_parsed(parsed)
        except (json.JSONDecodeError, IndexError):
            pass  # fall through to next strategy

        # 3. Try parsing the cleaned text directly as JSON
        try:
            parsed = json.loads(cleaned)
            return self._normalise_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

        # 4. Extract the first top-level JSON object from the text
        #    (handles text before/after the JSON block)
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            candidate = match.group()
            try:
                parsed = json.loads(candidate)
                return self._normalise_parsed(parsed)
            except json.JSONDecodeError:
                # Try to find balanced braces more carefully
                depth, start = 0, None
                for i, ch in enumerate(cleaned):
                    if ch == "{":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            try:
                                parsed = json.loads(cleaned[start : i + 1])
                                return self._normalise_parsed(parsed)
                            except json.JSONDecodeError:
                                start = None

        logger.warning("Failed to parse LLM output as JSON: %s", content[:300])
        # Fallback: treat as a conversational reply
        return {
            "thought": content,
            "action": "finish",
            "action_input": {"result": content},
        }

    @staticmethod
    def _normalise_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a parsed JSON dict into the expected schema."""
        return {
            "thought": parsed.get("thought", ""),
            "action": parsed.get("action", ""),
            "action_input": parsed.get("action_input", {}),
        }

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for the system prompt."""
        tools = self.tools.list_tools()
        if not tools:
            return "No tools available."

        lines = []
        for t in tools:
            params = ", ".join(
                f"{k}: {v.get('type', 'string')}" for k, v in t.parameters.items()
            )
            lines.append(f"- **{t.name}**({params}): {t.description}")
        return "\n".join(lines)
