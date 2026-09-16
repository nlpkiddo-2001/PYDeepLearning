"""
AgentForge — FastAPI Server
===========================
Main application entry point. Wires up routes, WebSocket endpoints,
tool discovery, and agent management.

v3: Added Multi-Agent orchestration — shared memory bus, message router,
    sub-agent spawning, and inter-agent communication endpoints.

Run with:
    uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Ensure project root is importable ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes.agents import router as agent_router, set_manager
from api.routes.workflows import router as workflow_router, set_workflow_manager
from api.routes.multi_agent import router as multi_agent_router, set_orchestrator as set_route_orchestrator
from api.ws.stream import router as ws_router, set_ws_orchestrator
from core.agent import AgentManager
from core.memory import MemoryManager
from core.workflow import WorkflowManager
from core.orchestrator import MultiAgentOrchestrator
from core.communication import MessageRouter
from core.shared_memory import SharedMemoryBus
from tools.registry import ToolRegistry

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agentforge.server")


def _load_config() -> Dict[str, Any]:
    """Load the main agent.yaml config."""
    config_path = PROJECT_ROOT / "configs" / "agent.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = _load_config()

    # Initialize components
    logger.info("Starting AgentForge platform v0.3.0 (Multi-Agent)...")

    # Tool registry
    tool_dirs = config.get("tools", {}).get("tool_dirs", ["tools/"])
    abs_tool_dirs = [str(PROJECT_ROOT / d) for d in tool_dirs]
    tool_registry = ToolRegistry(tool_dirs=abs_tool_dirs)
    tool_registry.discover()
    tool_registry.start_watcher()
    logger.info("Discovered %d tools: %s", len(tool_registry.names()), tool_registry.names())

    # Memory
    memory = MemoryManager(config.get("memory", {}))

    # v3: Multi-Agent components
    data_dir = str(PROJECT_ROOT / "data")
    message_router = MessageRouter(db_path=f"{data_dir}/messages.db")
    shared_memory = SharedMemoryBus(db_path=f"{data_dir}/shared_memory.db")
    orchestrator = MultiAgentOrchestrator(
        message_router=message_router,
        shared_memory=shared_memory,
    )

    # Agent manager (with orchestrator)
    manager = AgentManager(
        default_config=config,
        tool_registry=tool_registry,
        memory_manager=memory,
        orchestrator=orchestrator,
    )

    # Wire orchestrator <-> manager bidirectional ref
    orchestrator.set_agent_manager(manager)

    # Inject orchestrator into multi-agent tools
    try:
        from tools.multi_agent import set_orchestrator as set_tool_orchestrator
        from tools.multi_agent import set_agent_manager as set_tool_manager
        set_tool_orchestrator(orchestrator)
        set_tool_manager(manager)
        logger.info("Multi-agent tools initialized")
    except ImportError:
        logger.warning("Multi-agent tools module not found — skipping")

    # Workflow manager
    wf_manager = WorkflowManager(tool_registry=tool_registry)

    # Load workflow templates from configs/workflows/ if it exists
    wf_dir = PROJECT_ROOT / "configs" / "workflows"
    if wf_dir.is_dir():
        loaded = wf_manager.load_from_directory(str(wf_dir))
        logger.info("Loaded %d workflow templates", loaded)

    # Load any pre-configured agent templates
    agents_dir = PROJECT_ROOT / "agents"
    if agents_dir.is_dir():
        for yaml_file in agents_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    agent_data = yaml.safe_load(f)
                if agent_data:
                    manager.register(agent_data)
                    logger.info("Loaded agent template: %s", yaml_file.stem)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", yaml_file, exc)

    # Inject manager into routes
    set_manager(manager)
    set_workflow_manager(wf_manager)
    set_route_orchestrator(orchestrator)
    set_ws_orchestrator(orchestrator)

    # Store refs on app state
    app.state.manager = manager
    app.state.tool_registry = tool_registry
    app.state.config = config
    app.state.workflow_manager = wf_manager
    app.state.orchestrator = orchestrator
    app.state.shared_memory = shared_memory
    app.state.message_router = message_router

    logger.info("AgentForge v0.3.0 ready — http://0.0.0.0:%s", config.get("server", {}).get("port", 8000))

    yield

    # Shutdown
    tool_registry.stop_watcher()
    logger.info("AgentForge shut down.")


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AgentForge",
    description="Lightweight, self-hostable agentic AI platform with Multi-Agent Support",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS
config = _load_config()
cors_origins = config.get("server", {}).get("cors_origins", ["http://localhost:3000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins + ["*"],  # permissive for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(agent_router)
app.include_router(workflow_router)
app.include_router(multi_agent_router)
app.include_router(ws_router)


# ── Health check ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "platform": "AgentForge", "version": "0.3.0", "multi_agent": True}


@app.get("/config/providers")
async def list_providers():
    """Return available provider profiles for the UI model switcher."""
    providers = app.state.config.get("providers", {})
    current = app.state.config.get("llm", {})
    return {
        "current": {
            "provider": current.get("provider", "gemini"),
            "model": current.get("model", "gemini-3-flash-preview"),
        },
        "profiles": {
            name: {
                "provider": p.get("provider", name),
                "model": p.get("model", ""),
                "label": f"{p.get('provider', name).upper()} — {p.get('model', '')}",
            }
            for name, p in providers.items()
        },
    }


@app.get("/tools")
async def list_tools():
    """List all discovered tools."""
    registry: ToolRegistry = app.state.tool_registry
    return {"tools": [t.name for t in registry.list_tools()]}


# ── Global error handler ────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ── CLI entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    server_config = _load_config().get("server", {})
    uvicorn.run(
        "api.server:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)],
    )
