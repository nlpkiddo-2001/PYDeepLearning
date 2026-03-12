"""
Executor Module
===============
Manages tool execution with sandboxing, timeout, and resource tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from tools.registry import ToolRegistry

logger = logging.getLogger("agentforge.executor")


class ExecutionResult:
    """Result of a single tool execution."""

    def __init__(
        self,
        tool_name: str,
        success: bool,
        output: str,
        duration_ms: float,
        error: Optional[str] = None,
    ):
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.duration_ms = duration_ms
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": self.output[:2000],
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


class ToolExecutor:
    """
    Wraps the ToolRegistry with execution tracking, timeouts, and stats.
    """

    def __init__(self, registry: ToolRegistry, default_timeout: float = 60.0):
        self.registry = registry
        self.default_timeout = default_timeout
        self._stats: Dict[str, Dict[str, Any]] = {}  # tool_name -> {calls, errors, total_ms}

    async def execute(
        self,
        tool_name: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a tool with timeout and tracking."""
        timeout = timeout or self.default_timeout
        start = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self.registry.execute(tool_name, **kwargs),
                timeout=timeout,
            )
            duration = (time.monotonic() - start) * 1000
            self._track(tool_name, success=True, duration_ms=duration)
            return ExecutionResult(
                tool_name=tool_name,
                success=True,
                output=str(result),
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            duration = (time.monotonic() - start) * 1000
            error_msg = f"Tool '{tool_name}' timed out after {timeout}s"
            self._track(tool_name, success=False, duration_ms=duration)
            logger.warning(error_msg)
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                output="",
                duration_ms=duration,
                error=error_msg,
            )

        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            self._track(tool_name, success=False, duration_ms=duration)
            logger.exception("Tool %s failed: %s", tool_name, exc)
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                output="",
                duration_ms=duration,
                error=str(exc),
            )

    def _track(self, tool_name: str, success: bool, duration_ms: float):
        if tool_name not in self._stats:
            self._stats[tool_name] = {"calls": 0, "errors": 0, "total_ms": 0.0}
        self._stats[tool_name]["calls"] += 1
        if not success:
            self._stats[tool_name]["errors"] += 1
        self._stats[tool_name]["total_ms"] += duration_ms

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._stats)

    def reset_stats(self):
        self._stats.clear()
