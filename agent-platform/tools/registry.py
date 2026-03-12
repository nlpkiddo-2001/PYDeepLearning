"""
AgentForge Tool System
======================
Provides the @tool decorator and ToolRegistry for auto-discovering,
registering, and invoking tool functions.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger("agentforge.tools")

# ─── Global tool catalogue ───────────────────────────────────────────
_TOOL_CATALOGUE: Dict[str, "ToolDefinition"] = {}


@dataclass
class ToolDefinition:
    """Metadata wrapper around a registered tool function."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON-Schema-style param descriptions
    func: Callable
    is_async: bool = False
    tags: List[str] = field(default_factory=list)

    def to_schema(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [
                        k for k, v in self.parameters.items()
                        if v.get("required", False)
                    ],
                },
            },
        }


# ─── @tool decorator ─────────────────────────────────────────────────

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
):
    """
    Decorator that registers a function as an AgentForge tool.

    Usage:
        @tool(name="web_search", description="Search the web", parameters={...})
        async def web_search(query: str) -> str:
            ...

    If *name* is omitted the function name is used.
    If *description* is omitted the docstring is used.
    If *parameters* is omitted they are inferred from type hints.
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip() or f"Tool: {tool_name}"
        tool_params = parameters or _infer_parameters(func)
        tool_tags = tags or []

        defn = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            func=func,
            is_async=asyncio.iscoroutinefunction(func),
            tags=tool_tags,
        )
        _TOOL_CATALOGUE[tool_name] = defn
        logger.info("Registered tool: %s", tool_name)
        # Attach metadata so the function is self-describing
        func._tool_definition = defn
        return func

    return decorator


def _infer_parameters(func: Callable) -> Dict[str, Any]:
    """Build a JSON-Schema-style parameters dict from type hints."""
    sig = inspect.signature(func)
    hints = func.__annotations__
    params: Dict[str, Any] = {}
    _type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        hint = hints.get(pname, str)
        # Unwrap Optional
        origin = getattr(hint, "__origin__", None)
        if origin is not None:
            hint = hint.__args__[0] if hasattr(hint, "__args__") else hint
        json_type = _type_map.get(hint, "string")
        params[pname] = {
            "type": json_type,
            "description": f"Parameter: {pname}",
            "required": param.default is inspect.Parameter.empty,
        }
    return params


def _coerce_params(defn: ToolDefinition, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce parameter values to their declared types.

    LLMs frequently pass all values as strings (e.g., ``"10"`` for an int,
    ``"{}"`` for a dict).  This helper inspects the tool's parameter schema
    and casts values so the underlying Python function gets real types.
    """
    import json as _json

    coerced = dict(kwargs)
    for key, schema in defn.parameters.items():
        if key not in coerced:
            continue
        value = coerced[key]
        expected = schema.get("type", "string")

        try:
            if expected == "integer" and not isinstance(value, int):
                coerced[key] = int(value)
            elif expected == "number" and not isinstance(value, (int, float)):
                coerced[key] = float(value)
            elif expected == "boolean" and not isinstance(value, bool):
                if isinstance(value, str):
                    coerced[key] = value.lower() in ("true", "1", "yes")
                else:
                    coerced[key] = bool(value)
            elif expected == "object" and isinstance(value, str):
                coerced[key] = _json.loads(value) if value.strip() else {}
            elif expected == "array" and isinstance(value, str):
                coerced[key] = _json.loads(value) if value.strip() else []
        except (ValueError, TypeError, _json.JSONDecodeError):
            pass  # Leave the original value; the tool will handle the error

    return coerced


# ─── Tool Registry ───────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry that auto-discovers @tool-decorated functions
    from configured directories and supports hot-reloading.
    """

    def __init__(self, tool_dirs: Optional[List[str]] = None):
        self._tool_dirs = tool_dirs or []
        self._observer: Optional[Observer] = None

    # ── Discovery ────────────────────────────────────────────────────

    def discover(self) -> None:
        """Walk tool_dirs, import every .py file, and collect @tool functions."""
        for dir_path in self._tool_dirs:
            abs_dir = Path(dir_path).resolve()
            if not abs_dir.is_dir():
                logger.warning("Tool directory not found: %s", abs_dir)
                continue
            # Ensure the directory is importable
            parent = str(abs_dir.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

            for py_file in abs_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                module_name = f"{abs_dir.name}.{py_file.stem}"
                try:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    else:
                        importlib.import_module(module_name)
                    logger.debug("Imported tool module: %s", module_name)
                except Exception:
                    logger.exception("Failed to import tool module: %s", module_name)

    # ── Access ───────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[ToolDefinition]:
        return _TOOL_CATALOGUE.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(_TOOL_CATALOGUE.values())

    def list_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in _TOOL_CATALOGUE.values()]

    def names(self) -> List[str]:
        return list(_TOOL_CATALOGUE.keys())

    # ── Execution ────────────────────────────────────────────────────

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name.  Works for sync & async tools."""
        defn = _TOOL_CATALOGUE.get(name)
        if defn is None:
            raise ValueError(f"Unknown tool: {name}")
        # Coerce parameters to declared types (LLMs often send everything as strings)
        kwargs = _coerce_params(defn, kwargs)
        try:
            if defn.is_async:
                return await defn.func(**kwargs)
            else:
                return await asyncio.to_thread(defn.func, **kwargs)
        except Exception as exc:
            logger.exception("Tool %s failed: %s", name, exc)
            raise

    # ── Hot-reload watcher ───────────────────────────────────────────

    def start_watcher(self) -> None:
        """Start a filesystem watcher that re-discovers tools on changes."""
        if self._observer is not None:
            return
        handler = _ToolReloadHandler(self)
        self._observer = Observer()
        for dir_path in self._tool_dirs:
            abs_dir = str(Path(dir_path).resolve())
            if os.path.isdir(abs_dir):
                self._observer.schedule(handler, abs_dir, recursive=False)
        self._observer.start()
        logger.info("Tool hot-reload watcher started")

    def stop_watcher(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None


class _ToolReloadHandler(FileSystemEventHandler):
    """Re-discovers tools when .py files in tool dirs change."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def on_modified(self, event):  # type: ignore[override]
        if event.src_path.endswith(".py"):
            logger.info("Tool file changed: %s — re-discovering", event.src_path)
            self._registry.discover()

    on_created = on_modified
