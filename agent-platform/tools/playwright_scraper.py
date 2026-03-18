"""
Playwright-based Real-Time Job Scraper
=======================================
Architecture (from design diagram):

    Playwright browser
        ├── Direct career pages  (Navigate + scrape HTML)
        ├── Intercept XHR/fetch  (Tap internal API calls)
        └── Portal login flow    (LinkedIn / Naukri)
                │
        Anti-bot bypass layer
        (stealth plugin + human delays + realistic UA + context reuse)
                │
        Parse + extract structured data
        (title, company, exp, skills, salary, URL)
                │
        ┌───────┴───────┐
   filter_jobs()     SQLite / JSON store
   (agent pipeline)    (deduplication)

Dependencies:
    pip install playwright playwright-stealth
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from tools.registry import tool

logger = logging.getLogger("agentforge.tools.playwright_scraper")

# ── Paths ────────────────────────────────────────────────────────────
_DATA_DIR = Path(os.getenv("AGENTFORGE_DATA_DIR", "./data")).resolve()
_SCRAPER_DB = _DATA_DIR / "scraped_jobs.db"
_CONTEXT_DIR = _DATA_DIR / "browser_contexts"


# ═════════════════════════════════════════════════════════════════════
# Anti-Bot Bypass Layer
# ═════════════════════════════════════════════════════════════════════

# Realistic User-Agent rotation pool
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# Viewport sizes for realistic browsing
_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1366, "height": 768},
    {"width": 1280, "height": 720},
]


async def _human_delay(min_s: float = 0.5, max_s: float = 2.5) -> None:
    """Simulate human-like delays between actions."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def _human_scroll(page: Any, scrolls: int = 3) -> None:
    """Simulate human-like scrolling to trigger lazy-loaded content."""
    for _ in range(scrolls):
        scroll_amount = random.randint(300, 800)
        await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        await _human_delay(0.3, 1.0)


async def _create_stealth_browser(
    headless: bool = True,
    reuse_context: str = "",
) -> Tuple[Any, Any, Any]:
    """
    Create a Playwright browser with anti-bot stealth measures:
      - playwright-stealth plugin (patches navigator, webdriver, etc.)
      - Realistic user-agent rotation
      - Random viewport sizes
      - Browser context reuse (persistent cookies/sessions)

    Returns (playwright, browser, context) tuple.
    """
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()

    ua = random.choice(_USER_AGENTS)
    viewport = random.choice(_VIEWPORTS)

    launch_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
    ]

    browser = await pw.chromium.launch(
        headless=headless,
        args=launch_args,
    )

    # Reuse a persistent context for portal sessions (cookies survive)
    if reuse_context:
        _CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
        context_path = _CONTEXT_DIR / reuse_context
        context = await pw.chromium.launch_persistent_context(
            str(context_path),
            headless=headless,
            user_agent=ua,
            viewport=viewport,
            args=launch_args,
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
            java_script_enabled=True,
        )
        # For persistent context, browser is embedded in context
        browser = None
    else:
        context = await browser.new_context(
            user_agent=ua,
            viewport=viewport,
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
            java_script_enabled=True,
        )

    # Apply stealth patches
    try:
        from playwright_stealth import stealth_async
        page = await context.new_page()
        await stealth_async(page)
    except ImportError:
        logger.warning("playwright-stealth not installed — running without stealth patches")
        page = await context.new_page()

    # Additional JS-level anti-detection
    await page.add_init_script("""
        // Override navigator.webdriver
        Object.defineProperty(navigator, 'webdriver', { get: () => false });
        // Fake plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });
        // Fake languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });
        // Remove Chrome automation indicators
        window.chrome = { runtime: {} };
    """)

    return pw, browser, context


async def _close_browser(pw: Any, browser: Any, context: Any) -> None:
    """Safely close browser resources."""
    try:
        if context:
            await context.close()
        if browser:
            await browser.close()
        if pw:
            await pw.stop()
    except Exception as e:
        logger.debug("Browser cleanup error: %s", e)


# ═════════════════════════════════════════════════════════════════════
# SQLite Dedup Store
# ═════════════════════════════════════════════════════════════════════

def _get_scraper_db() -> sqlite3.Connection:
    """Get a connection to the scraper dedup database."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_SCRAPER_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scraped_jobs (
            job_hash    TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            company     TEXT NOT NULL,
            url         TEXT DEFAULT '',
            location    TEXT DEFAULT '',
            salary      TEXT DEFAULT '',
            experience  TEXT DEFAULT '',
            skills      TEXT DEFAULT '',
            source      TEXT DEFAULT '',
            source_url  TEXT DEFAULT '',
            description TEXT DEFAULT '',
            scraped_at  REAL DEFAULT (strftime('%s','now')),
            raw_json    TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_scraped_source
        ON scraped_jobs(source)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_scraped_at
        ON scraped_jobs(scraped_at DESC)
    """)
    conn.commit()
    return conn


def _job_hash(title: str, company: str, url: str = "") -> str:
    """Generate a dedup hash for a job listing."""
    key = f"{title.lower().strip()}|{company.lower().strip()}|{url.strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _dedup_and_store(jobs: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Store new jobs in SQLite, skip duplicates.
    Returns only genuinely new jobs.
    """
    db = _get_scraper_db()
    new_jobs: List[Dict[str, Any]] = []

    try:
        for job in jobs:
            h = _job_hash(
                job.get("title", ""),
                job.get("company", ""),
                job.get("url", ""),
            )
            existing = db.execute(
                "SELECT 1 FROM scraped_jobs WHERE job_hash = ?", (h,)
            ).fetchone()

            if not existing:
                db.execute(
                    """INSERT OR IGNORE INTO scraped_jobs
                       (job_hash, title, company, url, location, salary,
                        experience, skills, source, source_url, description, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        h,
                        job.get("title", ""),
                        job.get("company", ""),
                        job.get("url", ""),
                        job.get("location", ""),
                        job.get("salary", ""),
                        job.get("experience", ""),
                        json.dumps(job.get("skills", [])) if isinstance(job.get("skills"), list)
                        else job.get("skills", ""),
                        source,
                        job.get("source_url", ""),
                        job.get("description", "")[:2000],
                        json.dumps(job),
                    ),
                )
                job["is_new"] = True
                job["source"] = source
                new_jobs.append(job)
            else:
                job["is_new"] = False

        db.commit()
    finally:
        db.close()

    return new_jobs


# ═════════════════════════════════════════════════════════════════════
# Parse + Extract Structured Data
# ═════════════════════════════════════════════════════════════════════

def _extract_experience_from_text(text: str) -> str:
    """Extract experience requirements from job text."""
    patterns = [
        r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
        r"(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
        r"experience\s*(?:of\s*)?(\d+)\s*\+?\s*(?:years?|yrs?)",
        r"minimum\s*(\d+)\s*(?:years?|yrs?)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return f"{groups[0]}-{groups[1]} years"
            return f"{groups[0]}+ years"

    # Seniority heuristics
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["fresher", "entry level", "entry-level", "0-1 year", "intern"]):
        return "0-1 years"
    if any(kw in text_lower for kw in ["junior", "associate"]):
        return "1-3 years"
    if any(kw in text_lower for kw in ["mid-level", "mid level", "intermediate"]):
        return "3-5 years"
    if any(kw in text_lower for kw in ["senior", "sr.", "lead"]):
        return "5+ years"
    if any(kw in text_lower for kw in ["principal", "staff", "architect"]):
        return "8+ years"

    return ""


def _extract_salary_from_text(text: str) -> str:
    """Extract salary information from job text."""
    patterns = [
        # INR: ₹10L-20L, 10-20 LPA, Rs. 10,00,000
        r"(?:₹|Rs\.?|INR)\s*(\d[\d,.]*)\s*(?:L|lakh|lac|LPA)",
        r"(\d[\d,.]*)\s*-\s*(\d[\d,.]*)\s*(?:L|lakh|lac|LPA)",
        # USD: $100k-$150k, $100,000 - $150,000
        r"\$\s*(\d[\d,]*)\s*[kK]?\s*[-–]\s*\$?\s*(\d[\d,]*)\s*[kK]?",
        r"\$\s*(\d[\d,]+)",
        # Generic: 10-20 LPA, CTC 15L
        r"CTC\s*(?:of\s*)?(?:₹|Rs\.?)?\s*(\d[\d,.]*)\s*(?:L|lakh|LPA)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def _extract_skills_from_text(text: str) -> List[str]:
    """Extract technical skills from job text using keyword matching."""
    # Common tech skills to look for
    skill_patterns = [
        "python", "java", "javascript", "typescript", "go", "golang", "rust",
        "c\\+\\+", "c#", "ruby", "php", "swift", "kotlin", "scala", "r\\b",
        "react", "angular", "vue", "next\\.?js", "node\\.?js", "express",
        "django", "flask", "fastapi", "spring", "spring boot",
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "ci/cd", "github actions",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "rabbitmq", "graphql", "rest api", "grpc",
        "machine learning", "deep learning", "nlp", "computer vision",
        "pytorch", "tensorflow", "pandas", "numpy", "scikit-learn",
        "sql", "nosql", "data engineering", "etl", "airflow",
        "microservices", "system design", "distributed systems",
        "linux", "git", "agile", "scrum",
    ]

    found_skills: List[str] = []
    text_lower = text.lower()

    for skill in skill_patterns:
        if re.search(r"\b" + skill + r"\b", text_lower):
            # Normalize the skill name
            clean = re.sub(r"\\[+.?]", lambda m: m.group(0)[-1], skill)
            clean = clean.replace("\\b", "").strip()
            if clean and clean not in found_skills:
                found_skills.append(clean)

    return found_skills[:15]  # Cap at 15 skills


def _parse_job_card(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Normalize a raw scraped job into the standard structured format:
    title, company, experience, skills, salary, URL, location, source
    """
    title = raw.get("title", "").strip()
    company = raw.get("company", "").strip()
    description = raw.get("description", "")
    full_text = f"{title} {description}"

    return {
        "id": _job_hash(title, company, raw.get("url", "")),
        "title": title,
        "company": company,
        "url": raw.get("url", ""),
        "location": raw.get("location", ""),
        "salary": raw.get("salary", "") or _extract_salary_from_text(full_text),
        "experience": raw.get("experience", "") or _extract_experience_from_text(full_text),
        "skills": raw.get("skills", []) or _extract_skills_from_text(full_text),
        "date": raw.get("date", ""),
        "source": source,
        "source_url": raw.get("source_url", ""),
        "description": description[:500] if description else "",
    }


# ═════════════════════════════════════════════════════════════════════
# Strategy 1: Direct Career Page Scraper
# ═════════════════════════════════════════════════════════════════════

# Pre-configured selectors for popular career page formats
_CAREER_PAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "greenhouse": {
        "job_card": "div.opening",
        "title": "a",
        "location": "span.location",
        "url_attr": "a[href]",
        "base_url": "",  # derive from page
    },
    "lever": {
        "job_card": "div.posting",
        "title": "h5[data-qa='posting-name']",
        "location": "span.sort-by-location",
        "url_attr": "a.posting-title[href]",
        "base_url": "",
    },
    "workday": {
        "job_card": "li.css-1q2dra3",
        "title": "a[data-automation-id='jobTitle']",
        "location": "dd.css-129m7dg",
        "url_attr": "a[data-automation-id='jobTitle'][href]",
        "base_url": "",
    },
    "ashby": {
        "job_card": "div[class*='ashby-job-posting-brief']",
        "title": "a[class*='posting-title']",
        "location": "span[class*='posting-location']",
        "url_attr": "a[class*='posting-title'][href]",
        "base_url": "",
    },
    "generic": {
        "job_card": "[class*='job'], [class*='posting'], [class*='position'], [class*='vacancy'], [class*='opening']",
        "title": "a, h2, h3, h4, [class*='title']",
        "location": "[class*='location'], [class*='place'], [class*='city']",
        "url_attr": "a[href]",
        "base_url": "",
    },
}


async def _scrape_career_page(
    page: Any,
    url: str,
    company_name: str,
    config_key: str = "generic",
) -> List[Dict[str, Any]]:
    """
    Navigate to a career page and scrape job listings from the HTML.

    Strategy 1 from the diagram: Direct career pages.
    """
    config = _CAREER_PAGE_CONFIGS.get(config_key, _CAREER_PAGE_CONFIGS["generic"])

    await page.goto(url, wait_until="networkidle", timeout=30000)
    await _human_delay(1.0, 2.0)
    await _human_scroll(page, scrolls=5)  # Trigger lazy loads
    await _human_delay(0.5, 1.0)

    # Try to click "Load More" / "Show All" buttons if present
    for btn_text in ["Load More", "Show More", "View All", "See All", "Load all"]:
        try:
            btn = page.get_by_text(btn_text, exact=False).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await _human_delay(1.5, 3.0)
                await _human_scroll(page, scrolls=2)
        except Exception:
            pass

    # Extract jobs using configured selectors
    jobs: List[Dict[str, Any]] = []

    cards = await page.query_selector_all(config["job_card"])
    logger.info("Found %d job cards on %s", len(cards), url)

    for card in cards:
        try:
            # Title
            title_el = await card.query_selector(config["title"])
            title = await title_el.inner_text() if title_el else ""
            title = title.strip()

            if not title:
                continue

            # Location
            loc_el = await card.query_selector(config["location"])
            location = await loc_el.inner_text() if loc_el else ""
            location = location.strip()

            # URL
            link_el = await card.query_selector(config["url_attr"])
            href = await link_el.get_attribute("href") if link_el else ""
            job_url = urljoin(url, href) if href else url

            # Description snippet (card-level text)
            card_text = await card.inner_text()

            jobs.append({
                "title": title,
                "company": company_name,
                "url": job_url,
                "location": location,
                "description": card_text.strip()[:500],
                "source_url": url,
                "date": "",
            })
        except Exception as e:
            logger.debug("Error parsing job card: %s", e)
            continue

    return jobs


# ═════════════════════════════════════════════════════════════════════
# Strategy 2: Intercept XHR/Fetch (Internal API Tap)
# ═════════════════════════════════════════════════════════════════════

async def _intercept_job_apis(
    page: Any,
    url: str,
    company_name: str,
    api_patterns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Navigate to a career page and intercept XHR/fetch calls to internal
    job APIs. Many modern career pages load jobs via JSON APIs — this
    captures that data directly.

    Strategy 2 from the diagram: Intercept XHR/fetch.
    """
    if api_patterns is None:
        # Common patterns for internal job APIs
        api_patterns = [
            "*/api/jobs*", "*/api/positions*", "*/api/careers*",
            "*/api/v*/jobs*", "*/api/v*/positions*",
            "*/jobs.json*", "*/positions.json*",
            "*boards.greenhouse.io/*/jobs*",
            "*api.lever.co/v0/postings*",
            "*jobs.ashbyhq.com/api*",
            "*api.smartrecruiters.com*",
            "*/search/results*",
        ]

    captured_responses: List[Dict[str, Any]] = []

    async def handle_response(response: Any) -> None:
        """Capture JSON responses from job API endpoints."""
        req_url = response.url
        if response.status == 200:
            for pattern in api_patterns:
                # Simple glob matching
                pattern_regex = pattern.replace("*", ".*")
                if re.search(pattern_regex, req_url, re.IGNORECASE):
                    try:
                        body = await response.json()
                        captured_responses.append({
                            "url": req_url,
                            "data": body,
                        })
                        logger.info("Captured API response from: %s", req_url)
                    except Exception:
                        pass
                    break

    page.on("response", handle_response)

    await page.goto(url, wait_until="networkidle", timeout=30000)
    await _human_delay(1.0, 2.0)
    await _human_scroll(page, scrolls=5)
    await _human_delay(1.0, 2.0)

    # Try pagination / load-more to trigger more API calls
    for btn_text in ["Load More", "Show More", "Next", "View All"]:
        try:
            btn = page.get_by_text(btn_text, exact=False).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await _human_delay(2.0, 4.0)
        except Exception:
            pass

    # Parse captured API responses into structured jobs
    jobs: List[Dict[str, Any]] = []

    for resp in captured_responses:
        data = resp["data"]
        extracted = _extract_jobs_from_api_response(data, company_name, url)
        jobs.extend(extracted)

    return jobs


def _extract_jobs_from_api_response(
    data: Any,
    company_name: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """
    Heuristically extract job listings from an API JSON response.
    Handles various formats: arrays, nested objects, paginated responses.
    """
    jobs: List[Dict[str, Any]] = []

    # Flatten — find the array of job objects
    job_arrays: List[List[Any]] = []

    if isinstance(data, list):
        job_arrays.append(data)
    elif isinstance(data, dict):
        # Common keys that contain job arrays
        for key in ["jobs", "results", "data", "positions", "postings",
                     "items", "records", "content", "hits", "openings"]:
            if key in data and isinstance(data[key], list):
                job_arrays.append(data[key])
                break
        else:
            # Check nested structures
            for val in data.values():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    job_arrays.append(val)
                    break

    for arr in job_arrays:
        for item in arr:
            if not isinstance(item, dict):
                continue

            # Try common field name patterns
            title = (
                item.get("title") or item.get("name") or
                item.get("job_title") or item.get("jobTitle") or
                item.get("position") or item.get("text") or ""
            )
            if not title:
                continue

            company = (
                item.get("company") or item.get("company_name") or
                item.get("companyName") or item.get("employer") or
                company_name
            )

            location = (
                item.get("location") or item.get("city") or
                item.get("locationName") or
                _nested_get(item, "location", "name") or
                _nested_get(item, "location", "display_name") or ""
            )
            if isinstance(location, dict):
                location = location.get("name", "") or location.get("display_name", "")

            url = (
                item.get("url") or item.get("apply_url") or
                item.get("applyUrl") or item.get("hostedUrl") or
                item.get("absolute_url") or item.get("redirect_url") or ""
            )

            salary = (
                item.get("salary") or item.get("compensation") or
                item.get("salary_range") or ""
            )
            if isinstance(salary, dict):
                salary = f"{salary.get('min', '')} - {salary.get('max', '')}"

            description = (
                item.get("description") or item.get("content") or
                item.get("summary") or item.get("snippet") or ""
            )
            if isinstance(description, str) and len(description) > 500:
                description = description[:500]

            jobs.append({
                "title": str(title).strip(),
                "company": str(company).strip(),
                "url": str(url).strip(),
                "location": str(location).strip() if isinstance(location, str) else "",
                "salary": str(salary).strip() if isinstance(salary, str) else "",
                "description": str(description).strip() if isinstance(description, str) else "",
                "source_url": source_url,
            })

    return jobs


def _nested_get(d: Dict, *keys: str) -> Any:
    """Safely get a nested dictionary value."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return None
    return d


# ═════════════════════════════════════════════════════════════════════
# Strategy 3: Portal Login Flow (LinkedIn / Naukri)
# ═════════════════════════════════════════════════════════════════════

async def _scrape_linkedin_jobs(
    context: Any,
    query: str,
    location: str = "India",
    pages: int = 2,
) -> List[Dict[str, Any]]:
    """
    Scrape LinkedIn job search results.
    Uses persistent context for session reuse.

    NOTE: This scrapes the public job search page (no login required for
    basic results). For logged-in results, ensure the persistent context
    has a valid LinkedIn session.

    Strategy 3 from the diagram: Portal login flow — LinkedIn.
    """
    page = await context.new_page()
    jobs: List[Dict[str, Any]] = []

    try:
        for page_num in range(pages):
            start = page_num * 25
            search_url = (
                f"https://www.linkedin.com/jobs/search/"
                f"?keywords={query.replace(' ', '%20')}"
                f"&location={location.replace(' ', '%20')}"
                f"&start={start}"
                f"&sortBy=DD"  # Sort by date
            )

            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await _human_delay(2.0, 4.0)
            await _human_scroll(page, scrolls=5)
            await _human_delay(1.0, 2.0)

            # Extract job cards
            cards = await page.query_selector_all(
                "div.base-card, li.jobs-search-results__list-item, "
                "div.job-search-card, ul.jobs-search__results-list > li"
            )

            for card in cards:
                try:
                    title_el = await card.query_selector(
                        "h3.base-search-card__title, "
                        "a.job-card-list__title, "
                        "h3.job-search-card__title"
                    )
                    title = await title_el.inner_text() if title_el else ""

                    company_el = await card.query_selector(
                        "h4.base-search-card__subtitle, "
                        "a.job-card-container__company-name, "
                        "h4.job-search-card__subtitle"
                    )
                    company = await company_el.inner_text() if company_el else ""

                    loc_el = await card.query_selector(
                        "span.job-search-card__location, "
                        "li.job-card-container__metadata-item"
                    )
                    loc = await loc_el.inner_text() if loc_el else ""

                    link_el = await card.query_selector(
                        "a.base-card__full-link, "
                        "a.job-card-list__title, "
                        "a.job-search-card__title-link"
                    )
                    href = await link_el.get_attribute("href") if link_el else ""

                    date_el = await card.query_selector(
                        "time, span.job-search-card__listdate"
                    )
                    date_str = ""
                    if date_el:
                        date_str = await date_el.get_attribute("datetime") or await date_el.inner_text()

                    if title.strip():
                        jobs.append({
                            "title": title.strip(),
                            "company": company.strip(),
                            "url": href.strip() if href else "",
                            "location": loc.strip(),
                            "date": date_str.strip()[:10] if date_str else "",
                            "source_url": search_url,
                        })
                except Exception as e:
                    logger.debug("Error parsing LinkedIn card: %s", e)

            await _human_delay(2.0, 5.0)  # Rate limit between pages

    except Exception as e:
        logger.warning("LinkedIn scraping error: %s", e)
    finally:
        await page.close()

    return jobs


async def _scrape_naukri_jobs(
    context: Any,
    query: str,
    location: str = "",
    experience: str = "",
    pages: int = 2,
) -> List[Dict[str, Any]]:
    """
    Scrape Naukri.com job search results.

    Strategy 3 from the diagram: Portal login flow — Naukri.
    """
    page = await context.new_page()
    jobs: List[Dict[str, Any]] = []

    try:
        keywords = query.replace(" ", "-")
        base_url = f"https://www.naukri.com/{keywords}-jobs"

        params = []
        if location:
            base_url += f"-in-{location.lower().replace(' ', '-')}"
        if experience:
            match = re.search(r"(\d+)", experience)
            if match:
                params.append(f"experience={match.group(1)}")

        search_url = base_url
        if params:
            search_url += "?" + "&".join(params)

        for page_num in range(1, pages + 1):
            page_url = f"{search_url}&page={page_num}" if page_num > 1 else search_url

            await page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
            await _human_delay(2.0, 4.0)
            await _human_scroll(page, scrolls=5)
            await _human_delay(1.0, 2.0)

            # Naukri job card selectors
            cards = await page.query_selector_all(
                "article.jobTuple, div.srp-jobtuple-wrapper, "
                "div.cust-job-tuple, div.jobTupleHeader"
            )

            for card in cards:
                try:
                    title_el = await card.query_selector(
                        "a.title, a.job-title, h2 a"
                    )
                    title = await title_el.inner_text() if title_el else ""
                    href = await title_el.get_attribute("href") if title_el else ""

                    company_el = await card.query_selector(
                        "a.subTitle, a.comp-name, span.comp-name"
                    )
                    company = await company_el.inner_text() if company_el else ""

                    loc_el = await card.query_selector(
                        "span.locWdth, span.loc-wrap, li.location"
                    )
                    loc = await loc_el.inner_text() if loc_el else ""

                    exp_el = await card.query_selector(
                        "span.expwdth, li.experience, span.exp"
                    )
                    exp = await exp_el.inner_text() if exp_el else ""

                    sal_el = await card.query_selector(
                        "span.sal, li.salary, span.salary"
                    )
                    sal = await sal_el.inner_text() if sal_el else ""

                    desc_el = await card.query_selector(
                        "span.job-desc, div.job-description"
                    )
                    desc = await desc_el.inner_text() if desc_el else ""

                    if title.strip():
                        jobs.append({
                            "title": title.strip(),
                            "company": company.strip(),
                            "url": href.strip() if href else "",
                            "location": loc.strip(),
                            "salary": sal.strip(),
                            "experience": exp.strip(),
                            "description": desc.strip()[:500],
                            "source_url": page_url,
                        })
                except Exception as e:
                    logger.debug("Error parsing Naukri card: %s", e)

            await _human_delay(3.0, 6.0)  # Naukri is aggressive with rate-limiting

    except Exception as e:
        logger.warning("Naukri scraping error: %s", e)
    finally:
        await page.close()

    return jobs


# ═════════════════════════════════════════════════════════════════════
# Tool: scrape_live_jobs — Main Orchestrator (Registered Tool)
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="scrape_live_jobs",
    description=(
        "Scrape real-time job listings using a headless Playwright browser. "
        "Supports three strategies: (1) Direct career page scraping, "
        "(2) Intercepting internal job APIs (XHR/fetch), "
        "(3) Portal scraping from LinkedIn and Naukri. "
        "Includes anti-bot bypass (stealth plugin, human delays, UA rotation, "
        "persistent sessions). Returns structured JSON with deduplication. "
        "Output plugs directly into filter_jobs() for the agent pipeline."
    ),
    tags=["job", "scraper", "playwright", "realtime"],
)
async def scrape_live_jobs(
    query: str,
    sources: str = "linkedin,naukri",
    career_pages: str = "",
    location: str = "India",
    experience: str = "",
    max_pages: int = 2,
    headless: bool = True,
) -> str:
    """Scrape real-time job listings from career pages and job portals.

    Args:
        query: Job search keywords (e.g. "python backend developer")
        sources: Comma-separated scraping sources — 'linkedin', 'naukri',
                 'career_pages'. Default: 'linkedin,naukri'
        career_pages: JSON array of career page configs, each with:
                      {"url": "...", "company": "...", "type": "greenhouse|lever|generic"}
                      Example: [{"url": "https://boards.greenhouse.io/stripe", "company": "Stripe", "type": "greenhouse"}]
        location: Location filter (default: 'India')
        experience: Experience filter (e.g. '3-5 years', '5+')
        max_pages: Max pages to scrape per source (default: 2)
        headless: Run browser in headless mode (default: true)
    """
    max_pages = int(max_pages)
    if isinstance(headless, str):
        headless = headless.lower() in ("true", "1", "yes")

    source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
    all_raw_jobs: List[Dict[str, Any]] = []
    errors: List[str] = []

    # ── Check Playwright availability ────────────────────────────
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return json.dumps({
            "error": "Playwright not installed. Run: pip install playwright && playwright install chromium",
            "jobs": [],
        })

    # ── Launch browser with anti-bot measures ────────────────────
    pw = None
    browser = None
    context = None

    try:
        pw, browser, context = await _create_stealth_browser(
            headless=headless,
            reuse_context="job_scraper" if "linkedin" in source_list or "naukri" in source_list else "",
        )

        tasks: List[asyncio.Task] = []

        # Strategy 3: LinkedIn
        if "linkedin" in source_list:
            try:
                linkedin_jobs = await _scrape_linkedin_jobs(
                    context, query, location, pages=max_pages,
                )
                for job in linkedin_jobs:
                    parsed = _parse_job_card(job, source="LinkedIn (Scraped)")
                    all_raw_jobs.append(parsed)
                logger.info("LinkedIn: scraped %d jobs", len(linkedin_jobs))
            except Exception as e:
                errors.append(f"LinkedIn: {e}")
                logger.warning("LinkedIn scraping failed: %s", e)

        # Strategy 3: Naukri
        if "naukri" in source_list:
            try:
                naukri_jobs = await _scrape_naukri_jobs(
                    context, query, location, experience, pages=max_pages,
                )
                for job in naukri_jobs:
                    parsed = _parse_job_card(job, source="Naukri (Scraped)")
                    all_raw_jobs.append(parsed)
                logger.info("Naukri: scraped %d jobs", len(naukri_jobs))
            except Exception as e:
                errors.append(f"Naukri: {e}")
                logger.warning("Naukri scraping failed: %s", e)

        # Strategy 1 & 2: Career Pages
        if "career_pages" in source_list and career_pages:
            try:
                pages_config = json.loads(career_pages) if isinstance(career_pages, str) else career_pages
            except json.JSONDecodeError:
                pages_config = []
                errors.append("Invalid career_pages JSON")

            for cp in pages_config:
                cp_url = cp.get("url", "")
                cp_company = cp.get("company", "Unknown")
                cp_type = cp.get("type", "generic")

                if not cp_url:
                    continue

                try:
                    page = await context.new_page()

                    # Strategy 2: Try API interception first
                    api_jobs = await _intercept_job_apis(page, cp_url, cp_company)
                    await page.close()

                    if api_jobs:
                        for job in api_jobs:
                            parsed = _parse_job_card(job, source=f"{cp_company} (API)")
                            all_raw_jobs.append(parsed)
                        logger.info("%s API: captured %d jobs", cp_company, len(api_jobs))
                    else:
                        # Strategy 1: Fallback to HTML scraping
                        page = await context.new_page()
                        html_jobs = await _scrape_career_page(
                            page, cp_url, cp_company, config_key=cp_type,
                        )
                        await page.close()

                        for job in html_jobs:
                            parsed = _parse_job_card(job, source=f"{cp_company} (Career Page)")
                            all_raw_jobs.append(parsed)
                        logger.info("%s HTML: scraped %d jobs", cp_company, len(html_jobs))

                except Exception as e:
                    errors.append(f"{cp_company}: {e}")
                    logger.warning("Career page scraping failed for %s: %s", cp_company, e)

    finally:
        await _close_browser(pw, browser, context)

    # ── Dedup & Store ────────────────────────────────────────────
    new_jobs = _dedup_and_store(all_raw_jobs, source="playwright_scraper")

    # ── Filter by experience if specified ────────────────────────
    if experience:
        exp_match = re.search(r"(\d+)", experience)
        if exp_match:
            target_exp = int(exp_match.group(1))
            filtered = []
            for job in all_raw_jobs:
                job_exp = job.get("experience", "")
                job_exp_num = re.search(r"(\d+)", job_exp) if job_exp else None
                if job_exp_num:
                    if int(job_exp_num.group(1)) <= target_exp + 2:
                        filtered.append(job)
                else:
                    filtered.append(job)  # Keep if experience unknown
            all_raw_jobs = filtered

    # ── Build output ─────────────────────────────────────────────
    output = {
        "query": query,
        "location": location,
        "sources": source_list,
        "total": len(all_raw_jobs),
        "new_jobs": len(new_jobs),
        "jobs": all_raw_jobs[:30],  # Cap output
    }

    if errors:
        output["warnings"] = errors

    return json.dumps(output, indent=2)


# ═════════════════════════════════════════════════════════════════════
# Tool: scrape_career_page — Individual Career Page Scraper
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="scrape_career_page",
    description=(
        "Scrape job listings from a specific company's career page. "
        "Automatically detects the page type (Greenhouse, Lever, Workday, etc.) "
        "and tries API interception before falling back to HTML scraping. "
        "Returns structured JSON compatible with filter_jobs()."
    ),
    tags=["job", "scraper", "career"],
)
async def scrape_career_page_tool(
    url: str,
    company_name: str = "",
    page_type: str = "generic",
    headless: bool = True,
) -> str:
    """Scrape a single company career page for job listings.

    Args:
        url: Career page URL (e.g. "https://boards.greenhouse.io/stripe")
        company_name: Company name (auto-detected from URL if empty)
        page_type: ATS type — 'greenhouse', 'lever', 'workday', 'ashby', 'generic'
        headless: Run browser in headless mode (default: true)
    """
    if isinstance(headless, str):
        headless = headless.lower() in ("true", "1", "yes")

    # Auto-detect company name from URL
    if not company_name:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        company_name = parts[0] if parts else parsed.hostname or "Unknown"
        company_name = company_name.replace("-", " ").title()

    # Auto-detect page type from URL
    url_lower = url.lower()
    if "greenhouse.io" in url_lower:
        page_type = "greenhouse"
    elif "lever.co" in url_lower:
        page_type = "lever"
    elif "myworkday" in url_lower or "workday" in url_lower:
        page_type = "workday"
    elif "ashbyhq.com" in url_lower:
        page_type = "ashby"

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return json.dumps({
            "error": "Playwright not installed. Run: pip install playwright && playwright install chromium",
            "jobs": [],
        })

    pw = None
    browser = None
    context = None
    all_jobs: List[Dict[str, Any]] = []

    try:
        pw, browser, context = await _create_stealth_browser(headless=headless)
        page = await context.new_page()

        # Try API interception first
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
        except ImportError:
            pass

        api_jobs = await _intercept_job_apis(page, url, company_name)
        await page.close()

        if api_jobs:
            for job in api_jobs:
                all_jobs.append(_parse_job_card(job, source=f"{company_name} (API)"))
        else:
            # Fallback to HTML scraping
            page = await context.new_page()
            try:
                from playwright_stealth import stealth_async
                await stealth_async(page)
            except ImportError:
                pass

            html_jobs = await _scrape_career_page(page, url, company_name, page_type)
            await page.close()

            for job in html_jobs:
                all_jobs.append(_parse_job_card(job, source=f"{company_name} (Career Page)"))

    finally:
        await _close_browser(pw, browser, context)

    # Dedup & store
    new_jobs = _dedup_and_store(all_jobs, source=f"career_page:{company_name}")

    output = {
        "company": company_name,
        "url": url,
        "page_type": page_type,
        "total": len(all_jobs),
        "new_jobs": len(new_jobs),
        "jobs": all_jobs,
    }
    return json.dumps(output, indent=2)


# ═════════════════════════════════════════════════════════════════════
# Tool: get_scraped_jobs_history — Query the Dedup Store
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="get_scraped_jobs_history",
    description=(
        "Query the SQLite store of previously scraped jobs. "
        "Filter by source, keyword, or date range. Useful for tracking "
        "what jobs have been seen and detecting new listings."
    ),
    tags=["job", "history", "dedup"],
)
def get_scraped_jobs_history(
    keyword: str = "",
    source: str = "",
    days_back: int = 7,
    limit: int = 50,
) -> str:
    """Query scraped jobs history from the SQLite dedup store.

    Args:
        keyword: Filter by keyword in title or company (optional)
        source: Filter by source name (optional)
        days_back: How many days back to search (default: 7)
        limit: Max results to return (default: 50)
    """
    days_back = int(days_back)
    limit = int(limit)

    db = _get_scraper_db()
    try:
        query = "SELECT * FROM scraped_jobs WHERE scraped_at > strftime('%s','now') - ?"
        params: list = [days_back * 86400]

        if keyword:
            query += " AND (title LIKE ? OR company LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")

        query += " ORDER BY scraped_at DESC LIMIT ?"
        params.append(limit)

        rows = db.execute(query, params).fetchall()

        if not rows:
            return json.dumps({"jobs": [], "total": 0, "message": "No scraped jobs found."})

        jobs = []
        for row in rows:
            jobs.append({
                "title": row["title"],
                "company": row["company"],
                "url": row["url"],
                "location": row["location"],
                "salary": row["salary"],
                "experience": row["experience"],
                "skills": row["skills"],
                "source": row["source"],
                "scraped_at": row["scraped_at"],
            })

        return json.dumps({"jobs": jobs, "total": len(jobs)}, indent=2)

    finally:
        db.close()
