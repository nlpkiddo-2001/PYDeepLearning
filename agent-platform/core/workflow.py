"""
Workflow Engine
===============
Multi-step pipelines where each step is a named tool invocation.

Workflows are defined as a DAG of steps. Each step specifies:
  - A tool to invoke
  - Input mapping (from previous step outputs or static values)
  - Conditions (optional: only run if a condition is met)

Workflows can be:
  - Defined in YAML and loaded at startup
  - Created via the API / drag-and-drop UI
  - Exported to YAML for sharing / version control
  - Debugged step-by-step with breakpoints and live streaming
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from core.streaming import EventBus, EventType, StreamEvent, event_bus
from tools.registry import ToolRegistry

logger = logging.getLogger("agentforge.workflow")


# ═══════════════════════════════════════════════════════════════════════
# Workflow Schema
# ═══════════════════════════════════════════════════════════════════════

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"       # step hit a breakpoint


@dataclass
class WorkflowStep:
    """A single step in a workflow pipeline."""
    id: str
    name: str
    tool: str                                  # tool name from the registry
    inputs: Dict[str, Any] = field(default_factory=dict)
    # input_mapping: map step input keys to outputs of previous steps
    # e.g. {"query": "$steps.step1.output"} or {"query": "static value"}
    input_mapping: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None            # Python expression evaluated at runtime
    timeout: float = 60.0
    retry_count: int = 0
    max_retries: int = 2
    depends_on: List[str] = field(default_factory=list)  # step IDs this step depends on

    # UI positioning for drag-and-drop editor
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    output: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_ms: float = 0
    attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tool": self.tool,
            "inputs": self.inputs,
            "input_mapping": self.input_mapping,
            "condition": self.condition,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "depends_on": self.depends_on,
            "position": self.position,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 2),
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(
            id=data.get("id", uuid4().hex[:8]),
            name=data.get("name", "Unnamed Step"),
            tool=data.get("tool", ""),
            inputs=data.get("inputs", {}),
            input_mapping=data.get("input_mapping", {}),
            condition=data.get("condition"),
            timeout=data.get("timeout", 60.0),
            max_retries=data.get("max_retries", 2),
            depends_on=data.get("depends_on", []),
            position=data.get("position", {"x": 0, "y": 0}),
        )


@dataclass
class WorkflowEdge:
    """A connection between two steps in the workflow DAG."""
    id: str
    source: str       # source step ID
    target: str       # target step ID
    label: str = ""   # optional label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowEdge":
        return cls(
            id=data.get("id", uuid4().hex[:8]),
            source=data["source"],
            target=data["target"],
            label=data.get("label", ""),
        )


@dataclass
class Workflow:
    """A complete workflow definition (DAG of steps)."""
    id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        """Export workflow as YAML for sharing/versioning."""
        export = {
            "name": self.name,
            "description": self.description,
            "steps": [],
            "edges": [],
        }
        for step in self.steps:
            step_data: Dict[str, Any] = {
                "id": step.id,
                "name": step.name,
                "tool": step.tool,
            }
            if step.inputs:
                step_data["inputs"] = step.inputs
            if step.input_mapping:
                step_data["input_mapping"] = step.input_mapping
            if step.condition:
                step_data["condition"] = step.condition
            if step.timeout != 60.0:
                step_data["timeout"] = step.timeout
            if step.max_retries != 2:
                step_data["max_retries"] = step.max_retries
            if step.depends_on:
                step_data["depends_on"] = step.depends_on
            if step.position:
                step_data["position"] = step.position
            export["steps"].append(step_data)

        for edge in self.edges:
            export["edges"].append({
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            })

        return yaml.dump(export, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        return cls(
            id=data.get("id", uuid4().hex[:8]),
            name=data.get("name", "Untitled Workflow"),
            description=data.get("description", ""),
            steps=[WorkflowStep.from_dict(s) for s in data.get("steps", [])],
            edges=[WorkflowEdge.from_dict(e) for e in data.get("edges", [])],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Workflow":
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "Workflow":
        with open(path) as f:
            return cls.from_yaml(f.read())

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def get_execution_order(self) -> List[List[str]]:
        """
        Topological sort → returns layers of step IDs that can run in parallel.
        E.g., [[step1], [step2, step3], [step4]] means step1 runs first, then
        step2 and step3 can run in parallel, then step4.
        """
        # Build adjacency
        in_degree: Dict[str, int] = {s.id: 0 for s in self.steps}
        children: Dict[str, List[str]] = {s.id: [] for s in self.steps}

        for edge in self.edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
            children.setdefault(edge.source, []).append(edge.target)

        # Also add depends_on as edges
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in children:
                    continue
                in_degree[step.id] = in_degree.get(step.id, 0) + 1
                children[dep].append(step.id)

        # BFS layer-by-layer
        layers: List[List[str]] = []
        queue = [sid for sid, deg in in_degree.items() if deg == 0]

        while queue:
            layers.append(list(queue))
            next_queue: List[str] = []
            for sid in queue:
                for child in children.get(sid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        return layers


# ═══════════════════════════════════════════════════════════════════════
# Workflow Run (execution state)
# ═══════════════════════════════════════════════════════════════════════

class WorkflowRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"       # hit a breakpoint


@dataclass
class WorkflowRun:
    """Runtime state for a single workflow execution."""
    run_id: str = field(default_factory=lambda: uuid4().hex[:16])
    workflow_id: str = ""
    workflow_name: str = ""
    status: WorkflowRunStatus = WorkflowRunStatus.PENDING
    steps: List[WorkflowStep] = field(default_factory=list)  # cloned from workflow
    step_outputs: Dict[str, str] = field(default_factory=dict)  # step_id -> output
    current_step_id: Optional[str] = None
    breakpoints: List[str] = field(default_factory=list)  # step IDs to pause at
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "step_outputs": self.step_outputs,
            "current_step_id": self.current_step_id,
            "breakpoints": self.breakpoints,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "duration_seconds": (
                ((self.ended_at or time.time()) - self.started_at)
                if self.started_at else 0
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# Workflow Executor
# ═══════════════════════════════════════════════════════════════════════

# New event types for workflow streaming
WORKFLOW_EVENT_TYPES = {
    "workflow_start": "workflow_start",
    "workflow_step_start": "workflow_step_start",
    "workflow_step_complete": "workflow_step_complete",
    "workflow_step_error": "workflow_step_error",
    "workflow_step_skip": "workflow_step_skip",
    "workflow_step_paused": "workflow_step_paused",
    "workflow_complete": "workflow_complete",
    "workflow_error": "workflow_error",
}


class WorkflowExecutor:
    """
    Executes a workflow DAG, streaming events for each step.
    Supports:
    - Parallel execution of independent steps
    - Input mapping between steps
    - Conditional execution
    - Breakpoints for debugging
    - Retry logic per step
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        bus: Optional[EventBus] = None,
    ):
        self.tools = tool_registry
        self.bus = bus or event_bus

    async def execute(
        self,
        workflow: Workflow,
        agent_id: str = "workflow",
        breakpoints: Optional[List[str]] = None,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRun:
        """Execute a workflow and return the run result."""
        run = WorkflowRun(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            steps=[deepcopy(s) for s in workflow.steps],
            breakpoints=breakpoints or [],
        )
        run.status = WorkflowRunStatus.RUNNING
        run.started_at = time.time()

        # Inject initial inputs into step_outputs
        if initial_inputs:
            run.step_outputs["__initial__"] = json.dumps(initial_inputs)

        # Emit workflow start
        await self._emit(agent_id, run.run_id, "workflow_start", {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "total_steps": len(workflow.steps),
        })

        try:
            # Get execution order (layered topological sort)
            layers = workflow.get_execution_order()

            for layer_idx, layer in enumerate(layers):
                # Execute steps in this layer (can run in parallel)
                tasks = []
                for step_id in layer:
                    step = self._get_run_step(run, step_id)
                    if step is None:
                        continue

                    # Check breakpoints
                    if step_id in run.breakpoints:
                        step.status = StepStatus.PAUSED
                        run.current_step_id = step_id
                        run.status = WorkflowRunStatus.PAUSED
                        await self._emit(agent_id, run.run_id, "workflow_step_paused", {
                            "step_id": step_id,
                            "step_name": step.name,
                            "message": f"Breakpoint hit at step '{step.name}'",
                        })
                        # In a real implementation, we'd wait for a resume signal.
                        # For now, we continue after emitting the pause event.
                        run.status = WorkflowRunStatus.RUNNING
                        step.status = StepStatus.PENDING

                    tasks.append(
                        self._execute_step(run, step, agent_id)
                    )

                # Run the layer in parallel
                await asyncio.gather(*tasks)

                # Check if any step failed
                for step_id in layer:
                    step = self._get_run_step(run, step_id)
                    if step and step.status == StepStatus.FAILED:
                        run.status = WorkflowRunStatus.FAILED
                        run.error = f"Step '{step.name}' failed: {step.error}"
                        run.ended_at = time.time()
                        await self._emit(agent_id, run.run_id, "workflow_error", {
                            "error": run.error,
                            "failed_step": step_id,
                        })
                        return run

            # All done
            run.status = WorkflowRunStatus.COMPLETED
            run.ended_at = time.time()
            await self._emit(agent_id, run.run_id, "workflow_complete", {
                "total_steps": len(run.steps),
                "completed_steps": sum(1 for s in run.steps if s.status == StepStatus.COMPLETED),
                "duration_seconds": run.ended_at - run.started_at,
                "step_outputs": run.step_outputs,
            })

        except Exception as exc:
            run.status = WorkflowRunStatus.FAILED
            run.error = str(exc)
            run.ended_at = time.time()
            logger.exception("Workflow execution failed: %s", exc)
            await self._emit(agent_id, run.run_id, "workflow_error", {
                "error": str(exc),
            })

        return run

    async def _execute_step(
        self,
        run: WorkflowRun,
        step: WorkflowStep,
        agent_id: str,
    ) -> None:
        """Execute a single workflow step with retry logic."""
        run.current_step_id = step.id

        # Evaluate condition
        if step.condition:
            try:
                should_run = self._evaluate_condition(step.condition, run.step_outputs)
                if not should_run:
                    step.status = StepStatus.SKIPPED
                    await self._emit(agent_id, run.run_id, "workflow_step_skip", {
                        "step_id": step.id,
                        "step_name": step.name,
                        "condition": step.condition,
                    })
                    return
            except Exception as exc:
                logger.warning("Condition evaluation failed for step %s: %s", step.id, exc)

        # Resolve inputs
        resolved_inputs = self._resolve_inputs(step, run.step_outputs)

        # Execute with retries
        for attempt in range(1, step.max_retries + 2):  # +2 because range is exclusive
            step.attempt = attempt
            step.status = StepStatus.RUNNING
            step.started_at = time.time()

            await self._emit(agent_id, run.run_id, "workflow_step_start", {
                "step_id": step.id,
                "step_name": step.name,
                "tool": step.tool,
                "inputs": resolved_inputs,
                "attempt": attempt,
            })

            try:
                result = await asyncio.wait_for(
                    self.tools.execute(step.tool, **resolved_inputs),
                    timeout=step.timeout,
                )
                step.output = str(result)
                step.status = StepStatus.COMPLETED
                step.ended_at = time.time()
                step.duration_ms = (step.ended_at - step.started_at) * 1000
                run.step_outputs[step.id] = step.output

                await self._emit(agent_id, run.run_id, "workflow_step_complete", {
                    "step_id": step.id,
                    "step_name": step.name,
                    "tool": step.tool,
                    "output": step.output[:2000],
                    "duration_ms": round(step.duration_ms, 2),
                    "attempt": attempt,
                })
                return  # success

            except asyncio.TimeoutError:
                step.error = f"Timeout after {step.timeout}s"
            except Exception as exc:
                step.error = str(exc)

            step.ended_at = time.time()
            step.duration_ms = (step.ended_at - step.started_at) * 1000

            if attempt <= step.max_retries:
                logger.warning(
                    "Step %s failed (attempt %d/%d): %s",
                    step.id, attempt, step.max_retries + 1, step.error,
                )
                await self._emit(agent_id, run.run_id, "workflow_step_error", {
                    "step_id": step.id,
                    "step_name": step.name,
                    "error": step.error,
                    "attempt": attempt,
                    "max_retries": step.max_retries,
                    "will_retry": True,
                })
                await asyncio.sleep(1)  # brief delay before retry
            else:
                step.status = StepStatus.FAILED
                await self._emit(agent_id, run.run_id, "workflow_step_error", {
                    "step_id": step.id,
                    "step_name": step.name,
                    "error": step.error,
                    "attempt": attempt,
                    "max_retries": step.max_retries,
                    "will_retry": False,
                })

    def _resolve_inputs(
        self,
        step: WorkflowStep,
        step_outputs: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Resolve input mappings.

        Mapping syntax:
        - "$steps.<step_id>.output" → output of a previous step
        - "$steps.__initial__" → initial workflow inputs
        - Any other string → used as-is (static value)
        """
        resolved: Dict[str, Any] = dict(step.inputs)  # start with static inputs

        for key, mapping in step.input_mapping.items():
            if isinstance(mapping, str) and mapping.startswith("$steps."):
                parts = mapping.split(".")
                if len(parts) >= 3:
                    ref_step_id = parts[1]
                    output = step_outputs.get(ref_step_id, "")
                    resolved[key] = output
                else:
                    resolved[key] = mapping
            else:
                resolved[key] = mapping

        return resolved

    def _evaluate_condition(
        self,
        condition: str,
        step_outputs: Dict[str, str],
    ) -> bool:
        """Evaluate a simple condition expression."""
        # Safe evaluation context
        context = {"steps": step_outputs, "len": len, "bool": bool}
        try:
            return bool(eval(condition, {"__builtins__": {}}, context))
        except Exception:
            return True  # default: run the step if condition can't be evaluated

    def _get_run_step(self, run: WorkflowRun, step_id: str) -> Optional[WorkflowStep]:
        for s in run.steps:
            if s.id == step_id:
                return s
        return None

    async def _emit(
        self,
        agent_id: str,
        run_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a workflow event to the event bus."""
        await self.bus.publish(StreamEvent(
            type=EventType.INFO,  # use INFO as base type, real type in data
            data={"workflow_event": event_type, **data},
            agent_id=agent_id,
            task_id=run_id,
        ))


# ═══════════════════════════════════════════════════════════════════════
# Workflow Manager
# ═══════════════════════════════════════════════════════════════════════

class WorkflowManager:
    """
    Central registry for workflow definitions and executions.
    Handles CRUD, execution, and run history.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        bus: Optional[EventBus] = None,
    ):
        self._workflows: Dict[str, Workflow] = {}
        self._runs: Dict[str, WorkflowRun] = {}
        self.executor = WorkflowExecutor(tool_registry, bus)

    # ── CRUD ─────────────────────────────────────────────────────

    def create(self, data: Dict[str, Any]) -> Workflow:
        workflow = Workflow.from_dict(data)
        self._workflows[workflow.id] = workflow
        logger.info("Created workflow: %s (%s)", workflow.name, workflow.id)
        return workflow

    def get(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        return [w.to_dict() for w in self._workflows.values()]

    def update(self, workflow_id: str, data: Dict[str, Any]) -> Optional[Workflow]:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return None
        # Update fields
        wf.name = data.get("name", wf.name)
        wf.description = data.get("description", wf.description)
        if "steps" in data:
            wf.steps = [WorkflowStep.from_dict(s) for s in data["steps"]]
        if "edges" in data:
            wf.edges = [WorkflowEdge.from_dict(e) for e in data["edges"]]
        wf.updated_at = time.time()
        return wf

    def delete(self, workflow_id: str) -> bool:
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False

    # ── Execution ────────────────────────────────────────────────

    async def run(
        self,
        workflow_id: str,
        agent_id: str = "workflow",
        breakpoints: Optional[List[str]] = None,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowRun]:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return None
        result = await self.executor.execute(
            wf,
            agent_id=agent_id,
            breakpoints=breakpoints,
            initial_inputs=initial_inputs,
        )
        self._runs[result.run_id] = result
        return result

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        return self._runs.get(run_id)

    def list_runs(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        runs = self._runs.values()
        if workflow_id:
            runs = [r for r in runs if r.workflow_id == workflow_id]
        return [r.to_dict() for r in runs]

    # ── YAML Import/Export ───────────────────────────────────────

    def export_yaml(self, workflow_id: str) -> Optional[str]:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return None
        return wf.to_yaml()

    def import_yaml(self, yaml_str: str) -> Workflow:
        wf = Workflow.from_yaml(yaml_str)
        self._workflows[wf.id] = wf
        return wf

    def load_from_directory(self, directory: str) -> int:
        """Load workflow definitions from YAML files in a directory."""
        dir_path = Path(directory)
        count = 0
        if not dir_path.is_dir():
            return 0
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                wf = Workflow.from_yaml_file(str(yaml_file))
                self._workflows[wf.id] = wf
                count += 1
                logger.info("Loaded workflow: %s from %s", wf.name, yaml_file)
            except Exception as exc:
                logger.warning("Failed to load workflow %s: %s", yaml_file, exc)
        return count
