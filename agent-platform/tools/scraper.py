"""
URL Scraper Tool
================
Scrapes a URL and returns cleaned text content.
"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from tools.registry import tool


@tool(
    name="scrape_url",
    description="Fetch a URL and return the cleaned text content of the page.",
    tags=["web", "scraper"],
)
async def scrape_url(url: str, max_chars: int = 8000) -> str:
    """Scrape a URL and return cleaned text."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AgentForge/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n\n...[truncated]"

    return cleaned
