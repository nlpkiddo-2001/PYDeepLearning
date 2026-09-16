"""
Workflow Routes
===============
REST API endpoints for workflow CRUD, execution, YAML export/import,
and step-level debugging.

POST   /workflows                → create a new workflow
GET    /workflows                → list all workflows
GET    /workflows/{id}           → get workflow details
PUT    /workflows/{id}           → update a workflow
DELETE /workflows/{id}           → delete a workflow
POST   /workflows/{id}/run       → execute a workflow
GET    /workflows/{id}/runs      → list runs for a workflow
GET    /workflows/runs/{run_id}  → get run details
POST   /workflows/{id}/export    → export workflow as YAML
POST   /workflows/import         → import workflow from YAML
GET    /workflows/templates      → list built-in workflow templates
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from core.workflow import WorkflowManager

router = APIRouter(prefix="/workflows", tags=["workflows"])

# Injected at startup (see server.py)
_workflow_manager: Optional[WorkflowManager] = None


def set_workflow_manager(manager: WorkflowManager):
    global _workflow_manager
    _workflow_manager = manager


def _get_manager() -> WorkflowManager:
    if _workflow_manager is None:
        raise HTTPException(status_code=500, detail="Workflow manager not initialized")
    return _workflow_manager


# ─── Request / Response Models ────────────────────────────────────

class StepModel(BaseModel):
    id: Optional[str] = None
    name: str = "Step"
    tool: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    condition: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 2
    depends_on: List[str] = Field(default_factory=list)
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})


class EdgeModel(BaseModel):
    id: Optional[str] = None
    source: str
    target: str
    label: str = ""


class CreateWorkflowRequest(BaseModel):
    id: Optional[str] = None
    name: str = "Untitled Workflow"
    description: str = ""
    steps: List[StepModel] = Field(default_factory=list)
    edges: List[EdgeModel] = Field(default_factory=list)


class UpdateWorkflowRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[StepModel]] = None
    edges: Optional[List[EdgeModel]] = None


class RunWorkflowRequest(BaseModel):
    agent_id: str = "workflow"
    breakpoints: List[str] = Field(default_factory=list)
    initial_inputs: Dict[str, Any] = Field(default_factory=dict)


class ImportYAMLRequest(BaseModel):
    yaml_content: str


# ─── Endpoints ────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_workflow(req: CreateWorkflowRequest):
    """Create a new workflow."""
    manager = _get_manager()
    data = req.model_dump(exclude_none=True)
    # Convert step/edge models to dicts
    if "steps" in data:
        data["steps"] = [s if isinstance(s, dict) else s for s in data["steps"]]
    if "edges" in data:
        data["edges"] = [e if isinstance(e, dict) else e for e in data["edges"]]

    workflow = manager.create(data)
    return {"status": "created", "workflow": workflow.to_dict()}


@router.get("")
async def list_workflows():
    """List all workflows."""
    manager = _get_manager()
    return {"workflows": manager.list_workflows()}


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get a specific workflow."""
    manager = _get_manager()
    wf = manager.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return wf.to_dict()


@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, req: UpdateWorkflowRequest):
    """Update an existing workflow (used by the drag-and-drop editor)."""
    manager = _get_manager()
    data = req.model_dump(exclude_none=True)
    wf = manager.update(workflow_id, data)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return {"status": "updated", "workflow": wf.to_dict()}


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow."""
    manager = _get_manager()
    if not manager.delete(workflow_id):
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return {"status": "deleted", "workflow_id": workflow_id}


@router.post("/{workflow_id}/run")
async def run_workflow(workflow_id: str, req: RunWorkflowRequest):
    """Execute a workflow. Streams events via the existing WebSocket."""
    manager = _get_manager()
    wf = manager.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    # Run in background so endpoint returns immediately
    async def _bg_run():
        try:
            await manager.run(
                workflow_id,
                agent_id=req.agent_id,
                breakpoints=req.breakpoints,
                initial_inputs=req.initial_inputs,
            )
        except Exception:
            pass  # errors streamed via WebSocket

    task = asyncio.create_task(_bg_run())

    return {
        "status": "started",
        "workflow_id": workflow_id,
        "message": "Connect to WebSocket /agents/{agent_id}/stream for live updates",
    }


@router.get("/{workflow_id}/runs")
async def list_workflow_runs(workflow_id: str):
    """List all runs for a workflow."""
    manager = _get_manager()
    return {"runs": manager.list_runs(workflow_id)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get details of a specific run."""
    manager = _get_manager()
    run = manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run.to_dict()


@router.post("/{workflow_id}/export", response_class=PlainTextResponse)
async def export_workflow_yaml(workflow_id: str):
    """Export a workflow as YAML."""
    manager = _get_manager()
    yaml_str = manager.export_yaml(workflow_id)
    if yaml_str is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return yaml_str


@router.post("/import")
async def import_workflow_yaml(req: ImportYAMLRequest):
    """Import a workflow from YAML."""
    manager = _get_manager()
    try:
        wf = manager.import_yaml(req.yaml_content)
        return {"status": "imported", "workflow": wf.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")


@router.get("/templates/list")
async def list_templates():
    """List available built-in workflow templates."""
    templates = [
        {
            "id": "job_hunter",
            "name": "Job Hunter",
            "description": "Scrape jobs → Match resume → Summarize → Notify",
            "steps": [
                {"id": "scrape", "name": "Scrape Jobs", "tool": "web_search",
                 "inputs": {"query": "ML engineer jobs remote"}, "position": {"x": 100, "y": 100}},
                {"id": "filter", "name": "Filter Results", "tool": "scrape_url",
                 "input_mapping": {"url": "$steps.scrape.output"}, "position": {"x": 350, "y": 100}},
                {"id": "summarize", "name": "Summarize", "tool": "write_file",
                 "input_mapping": {"content": "$steps.filter.output"},
                 "inputs": {"path": "results.md"}, "position": {"x": 600, "y": 100}},
            ],
            "edges": [
                {"source": "scrape", "target": "filter"},
                {"source": "filter", "target": "summarize"},
            ],
        },
        {
            "id": "deep_research",
            "name": "Deep Research",
            "description": "Search → Scrape → Summarize → Save",
            "steps": [
                {"id": "search", "name": "Web Search", "tool": "web_search",
                 "inputs": {"query": ""}, "position": {"x": 100, "y": 100}},
                {"id": "scrape", "name": "Scrape Pages", "tool": "scrape_url",
                 "input_mapping": {"url": "$steps.search.output"}, "position": {"x": 350, "y": 100}},
                {"id": "save", "name": "Save Report", "tool": "write_file",
                 "input_mapping": {"content": "$steps.scrape.output"},
                 "inputs": {"path": "research.md"}, "position": {"x": 600, "y": 100}},
            ],
            "edges": [
                {"source": "search", "target": "scrape"},
                {"source": "scrape", "target": "save"},
            ],
        },
        {
            "id": "email_automator",
            "name": "Email Automator",
            "description": "Read emails → Categorize → Draft reply → Queue",
            "steps": [
                {"id": "read", "name": "Read Emails", "tool": "http_request",
                 "inputs": {"url": "", "method": "GET"}, "position": {"x": 100, "y": 100}},
                {"id": "categorize", "name": "Categorize", "tool": "web_search",
                 "input_mapping": {"query": "$steps.read.output"}, "position": {"x": 350, "y": 100}},
                {"id": "draft", "name": "Draft Reply", "tool": "write_file",
                 "input_mapping": {"content": "$steps.categorize.output"},
                 "inputs": {"path": "drafts.md"}, "position": {"x": 600, "y": 100}},
            ],
            "edges": [
                {"source": "read", "target": "categorize"},
                {"source": "categorize", "target": "draft"},
            ],
        },
    ]
    return {"templates": templates}
