"""
Web Search Tool
===============
Performs web searches using a configurable backend.
Falls back to DuckDuckGo HTML scraping when no API key is set.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from tools.registry import tool


@tool(
    name="web_search",
    description="Search the web for a given query and return top results with titles, URLs, and snippets.",
    tags=["search", "web"],
)
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return formatted results."""
    # Coerce num_results to int (LLM may send as string)
    num_results = int(num_results)

    # ── Try SerpAPI if key is available ──────────────────────────
    serp_key = os.getenv("SERPAPI_KEY")
    if serp_key:
        return await _serpapi_search(query, num_results, serp_key)

    # ── Fallback: DuckDuckGo HTML lite ───────────────────────────
    return await _ddg_search(query, num_results)


async def _serpapi_search(query: str, num: int, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={"q": query, "num": num, "api_key": api_key, "engine": "google"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("organic_results", [])[:num]
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.get('title', '')}**")
        lines.append(f"   URL: {r.get('link', '')}")
        lines.append(f"   {r.get('snippet', '')}")
        lines.append("")
    return "\n".join(lines) if lines else "No results found."


async def _ddg_search(query: str, num: int) -> str:
    """Lightweight DuckDuckGo HTML scrape (no API key needed)."""
    from bs4 import BeautifulSoup

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (AgentForge Bot)"},
        )
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    result_blocks = soup.select(".result__body")[:num]

    lines = []
    for i, block in enumerate(result_blocks, 1):
        title_el = block.select_one(".result__a")
        snippet_el = block.select_one(".result__snippet")
        title = title_el.get_text(strip=True) if title_el else "No title"
        url = title_el.get("href", "") if title_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        lines.append(f"{i}. **{title}**")
        lines.append(f"   URL: {url}")
        lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines) if lines else "No results found."
