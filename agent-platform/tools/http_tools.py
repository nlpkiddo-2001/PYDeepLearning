"""
HTTP Tools
==========
Make arbitrary HTTP requests (GET, POST, PUT, DELETE).
"""

from __future__ import annotations

from typing import Optional, Dict

import httpx

from tools.registry import tool


@tool(
    name="http_request",
    description="Make an HTTP request to any URL. Supports GET, POST, PUT, DELETE.",
    tags=["http", "api"],
)
async def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """Make an HTTP request and return the response."""
    import json as _json

    # ── Coerce LLM-provided string params ───────────────────────
    timeout = int(timeout) if isinstance(timeout, str) else timeout

    if isinstance(headers, str):
        try:
            headers = _json.loads(headers)
        except (ValueError, TypeError):
            headers = None
    if not isinstance(headers, dict):
        headers = None

    method = method.upper()
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        kwargs: dict = {"headers": headers or {}}
        if body and method in ("POST", "PUT", "PATCH"):
            kwargs["content"] = body
            if "Content-Type" not in (headers or {}):
                kwargs["headers"]["Content-Type"] = "application/json"

        resp = await client.request(method, url, **kwargs)

    status = resp.status_code
    body_text = resp.text[:8000]
    return f"HTTP {status}\n\n{body_text}"


@tool(
    name="query_database",
    description="Execute a read-only SQL query against the agent's SQLite database.",
    tags=["database", "sql"],
)
def query_database(query: str, db_path: str = "./data/memory.db") -> str:
    """Execute a read-only SQL query."""
    import sqlite3

    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed for safety."

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No results."

        # Format as table
        columns = rows[0].keys()
        lines = [" | ".join(columns)]
        lines.append(" | ".join("---" for _ in columns))
        for row in rows[:50]:  # cap at 50 rows
            lines.append(" | ".join(str(row[c]) for c in columns))
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: {exc}"
