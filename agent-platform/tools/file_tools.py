"""
File I/O Tools
==============
Read and write files within the agent's sandboxed workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

from tools.registry import tool

# Sandbox root — agents can only access files under this directory
_SANDBOX_ROOT = Path(os.getenv("AGENTFORGE_SANDBOX", "./workspace")).resolve()


def _safe_path(filepath: str) -> Path:
    """Resolve the path and ensure it stays within the sandbox."""
    resolved = (_SANDBOX_ROOT / filepath).resolve()
    if not str(resolved).startswith(str(_SANDBOX_ROOT)):
        raise PermissionError(f"Access denied: path escapes sandbox — {filepath}")
    return resolved


@tool(
    name="read_file",
    description="Read the contents of a file. Path is relative to the agent workspace.",
    tags=["file", "io"],
)
def read_file(filepath: str, max_chars: int = 50000) -> str:
    """Read a file from the sandbox."""
    path = _safe_path(filepath)
    if not path.exists():
        return f"ERROR: File not found — {filepath}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[truncated]"
    return text


@tool(
    name="write_file",
    description="Write content to a file. Path is relative to the agent workspace. Creates directories as needed.",
    tags=["file", "io"],
)
def write_file(filepath: str, content: str) -> str:
    """Write a file to the sandbox."""
    path = _safe_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"File written: {filepath} ({len(content)} chars)"
