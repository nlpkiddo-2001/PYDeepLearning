"""
Job Hunting Tools
=================
Real-time tools for the Job Hunter agent:
  search_jobs       — Live job listings from Remotive & Adzuna
  get_job_details   — Full JD scrape from a job URL
  match_resume      — Skill gap analysis (LLM-powered)
  check_company     — Company health signal check
  track_application — Application state machine (SQLite)
  monitor_new_jobs  — Detect genuinely new listings (dedup)
  send_alert        — Push notification on high-match jobs

Stack: Remotive (free) + Adzuna (free tier) + httpx + SQLite + LLM call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import smtplib
import sqlite3
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import tool

logger = logging.getLogger("agentforge.tools.job")

# ── Paths ────────────────────────────────────────────────────────────
_DATA_DIR = Path(os.getenv("AGENTFORGE_DATA_DIR", "./data")).resolve()
_JOB_DB_PATH = _DATA_DIR / "job_tracker.db"


# ── SQLite helpers ───────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Return a connection to the job tracker database, creating tables if needed."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_JOB_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            job_id      TEXT PRIMARY KEY,
            company     TEXT NOT NULL,
            role        TEXT NOT NULL,
            url         TEXT DEFAULT '',
            source      TEXT DEFAULT '',
            status      TEXT DEFAULT 'found',
            match_score INTEGER DEFAULT 0,
            salary      TEXT DEFAULT '',
            location    TEXT DEFAULT '',
            notes       TEXT DEFAULT '',
            created_at  REAL DEFAULT (strftime('%s','now')),
            updated_at  REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            job_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            first_seen REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.commit()
    return conn


# ═════════════════════════════════════════════════════════════════════
# 1. search_jobs — Live Job Listings
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="search_jobs",
    description=(
        "Search live job listings across Remotive (remote-only, free), "
        "Adzuna (free tier), and JSearch via RapidAPI. Returns structured "
        "JSON array of jobs with title, company, URL, location, salary, "
        "posted date, and source. Use location='remote' for remote roles."
    ),
    tags=["job", "search"],
)
async def search_jobs(
    query: str,
    location: str = "remote",
    days_ago: int = 7,
    experience: str = "",
    employment_type: str = "",
) -> str:
    """Search live job listings across multiple free sources.

    Args:
        query: Job search keywords (e.g. "python backend developer")
        location: Location filter — use 'remote' for remote roles
        days_ago: Only return jobs posted within this many days
        experience: Experience filter for JSearch — one of:
                    'under_3_years_experience', 'more_than_3_years_experience',
                    'no_experience', 'no_degree'
        employment_type: Employment type for JSearch — one of:
                         'FULLTIME', 'PARTTIME', 'CONTRACTOR', 'INTERN'
    """
    days_ago = int(days_ago)
    results: List[Dict[str, Any]] = []

    # Run all three sources concurrently
    remotive_task = asyncio.create_task(_search_remotive(query))
    adzuna_task = asyncio.create_task(_search_adzuna(query, location, days_ago))
    jsearch_task = asyncio.create_task(
        _search_jsearch(query, location, days_ago, experience, employment_type)
    )

    remotive_results, adzuna_results, jsearch_results = await asyncio.gather(
        remotive_task, adzuna_task, jsearch_task, return_exceptions=True
    )

    if isinstance(remotive_results, list):
        results.extend(remotive_results)
    else:
        logger.warning("Remotive search failed: %s", remotive_results)

    if isinstance(adzuna_results, list):
        results.extend(adzuna_results)
    else:
        logger.warning("Adzuna search failed: %s", adzuna_results)

    if isinstance(jsearch_results, list):
        results.extend(jsearch_results)
    else:
        logger.warning("JSearch search failed: %s", jsearch_results)

    if not results:
        return json.dumps({"query": query, "location": location, "jobs": []})

    # De-duplicate by title+company
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("title", "").lower(), r.get("company", "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Return structured JSON for downstream pipeline tools
    output = {
        "query": query,
        "location": location,
        "total": len(unique),
        "jobs": unique[:20],
    }
    return json.dumps(output, indent=2)


async def _search_remotive(query: str) -> List[Dict[str, Any]]:
    """Search Remotive API — completely free, no key needed."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": query, "limit": 10},
        )
        resp.raise_for_status()
        data = resp.json()

    jobs = data.get("jobs", [])
    results = []
    for job in jobs[:10]:
        results.append({
            "id": str(job.get("id", "")),
            "title": job.get("title", ""),
            "company": job.get("company_name", ""),
            "url": job.get("url", ""),
            "location": job.get("candidate_required_location", "Worldwide"),
            "salary": job.get("salary", ""),
            "date": job.get("publication_date", "")[:10],
            "source": "Remotive",
        })
    return results


async def _search_adzuna(query: str, location: str, days_ago: int) -> List[Dict[str, Any]]:
    """Search Adzuna API — free tier (250 req/day)."""
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        logger.debug("Adzuna credentials not set — skipping")
        return []

    # Default to GB; override via ADZUNA_COUNTRY env var
    country = os.getenv("ADZUNA_COUNTRY", "us")
    params: Dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,
        "results_per_page": 10,
        "max_days_old": days_ago,
        "content-type": "application/json",
    }
    if location and location.lower() != "remote":
        params["where"] = location

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for job in data.get("results", [])[:10]:
        salary_min = job.get("salary_min")
        salary_max = job.get("salary_max")
        salary = ""
        if salary_min and salary_max:
            salary = f"${int(salary_min):,} – ${int(salary_max):,}"
        elif salary_min:
            salary = f"From ${int(salary_min):,}"

        results.append({
            "id": str(job.get("id", "")),
            "title": job.get("title", ""),
            "company": job.get("company", {}).get("display_name", ""),
            "url": job.get("redirect_url", ""),
            "location": job.get("location", {}).get("display_name", ""),
            "salary": salary,
            "date": job.get("created", "")[:10],
            "source": "Adzuna",
        })
    return results


async def _search_jsearch(
    query: str,
    location: str,
    days_ago: int,
    experience: str = "",
    employment_type: str = "",
) -> List[Dict[str, Any]]:
    """Search JSearch API via RapidAPI — free tier (100 req/month)."""
    rapid_key = os.getenv("RAPIDAPI_KEY", "")
    if not rapid_key:
        logger.debug("RAPIDAPI_KEY not set — skipping JSearch")
        return []

    params: Dict[str, Any] = {
        "query": f"{query} in {location}" if location and location.lower() != "remote"
                 else f"{query} remote",
        "page": "1",
        "num_pages": "1",
        "date_posted": "week" if days_ago <= 7 else "month",
    }
    if experience:
        params["job_requirements"] = experience
    if employment_type:
        params["employment_types"] = employment_type
    if location.lower() == "remote":
        params["remote_jobs_only"] = "true"

    headers = {
        "X-RapidAPI-Key": rapid_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://jsearch.p.rapidapi.com/search",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for job in data.get("data", [])[:10]:
        salary_min = job.get("job_min_salary")
        salary_max = job.get("job_max_salary")
        salary_period = job.get("job_salary_period", "")
        salary = ""
        if salary_min and salary_max:
            salary = f"${int(salary_min):,} – ${int(salary_max):,}"
            if salary_period:
                salary += f" ({salary_period})"
        elif salary_min:
            salary = f"From ${int(salary_min):,}"

        results.append({
            "id": job.get("job_id", ""),
            "title": job.get("job_title", ""),
            "company": job.get("employer_name", ""),
            "url": job.get("job_apply_link", "") or job.get("job_google_link", ""),
            "location": job.get("job_city", "") or job.get("job_country", ""),
            "salary": salary,
            "date": (job.get("job_posted_at_datetime_utc", "") or "")[:10],
            "source": "JSearch",
            "description_snippet": (job.get("job_description", "") or "")[:300],
            "employment_type": job.get("job_employment_type", ""),
            "is_remote": job.get("job_is_remote", False),
            "employer_logo": job.get("employer_logo", ""),
        })
    return results


# ═════════════════════════════════════════════════════════════════════
# 2. get_job_details — Full Job Description Scrape
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="get_job_details",
    description=(
        "Fetch the full job description from a job posting URL. "
        "Returns title, company, description, requirements, salary if listed, "
        "and other structured details scraped from the page."
    ),
    tags=["job", "scraper"],
)
async def get_job_details(url: str) -> str:
    """Scrape full job description from a given URL."""
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
        tag.decompose()

    # Try to extract structured info
    title = ""
    title_el = soup.find("h1") or soup.find("title")
    if title_el:
        title = title_el.get_text(strip=True)

    # Look for common JD containers
    jd_text = ""
    for selector in [
        ".job-description", ".description", ".job-details",
        "#job-description", "#description", "[data-testid='jobDescription']",
        "article", ".posting-page", ".content-wrapper",
    ]:
        container = soup.select_one(selector)
        if container:
            jd_text = container.get_text(separator="\n", strip=True)
            break

    if not jd_text:
        # Fallback to body text
        body = soup.find("body")
        jd_text = body.get_text(separator="\n", strip=True) if body else ""

    # Clean up
    lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) > 6000:
        cleaned = cleaned[:6000] + "\n\n...[truncated]"

    output = f"**{title}**\n\nURL: {url}\n\n{cleaned}"
    return output


# ═════════════════════════════════════════════════════════════════════
# 3. match_resume — Skill Gap Analysis
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="match_resume",
    description=(
        "Compare a resume against a job description and return a match score "
        "(0–100), matched skills, missing skills, and seniority fit. "
        "Pass the job description text and either resume_text directly or "
        "resume_path to a PDF/TXT file."
    ),
    tags=["job", "analysis"],
)
async def match_resume(
    job_description: str,
    resume_text: str = "",
    resume_path: str = "",
) -> str:
    """Analyze skill match between resume and job description."""
    # Resolve resume text
    if not resume_text and resume_path:
        resume_text = _extract_resume_text(resume_path)
    if not resume_text:
        return "ERROR: Provide either resume_text or resume_path."

    # Use LLM for analysis (import here to avoid circular deps at module level)
    from llm.base import create_provider
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "configs" / "agent.yaml"
    llm_config = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
            llm_config = cfg.get("llm", {})

    provider = create_provider(llm_config)

    prompt = f"""You are a recruitment matching expert. Analyze the skill match between
the candidate's resume and the job description below.

Return a JSON object with exactly these fields:
{{
  "match_score": <0-100>,
  "matched_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "seniority_fit": "under-qualified | good fit | over-qualified",
  "summary": "2-3 sentence assessment"
}}

===== JOB DESCRIPTION =====
{job_description[:3000]}

===== RESUME =====
{resume_text[:3000]}

Return ONLY the JSON object, no other text."""

    messages = [{"role": "user", "content": prompt}]
    response = await provider.complete(messages)

    # Try to parse as JSON; return raw if parsing fails
    raw = response.content.strip()
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        parsed = json.loads(raw)
        # Format nicely
        lines = [
            f"**Match Score:** {parsed.get('match_score', 'N/A')}/100",
            f"**Seniority Fit:** {parsed.get('seniority_fit', 'N/A')}",
            "",
            f"**Matched Skills:** {', '.join(parsed.get('matched_skills', []))}",
            f"**Missing Skills:** {', '.join(parsed.get('missing_skills', []))}",
            "",
            f"**Summary:** {parsed.get('summary', '')}",
        ]
        return "\n".join(lines)
    except (json.JSONDecodeError, IndexError):
        return raw


def _extract_resume_text(path: str) -> str:
    """Extract text from a resume file (PDF or TXT)."""
    p = Path(path)
    if not p.exists():
        return ""
    if p.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(p) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except ImportError:
            return f"ERROR: pdfplumber not installed — run: pip install pdfplumber"
    else:
        return p.read_text(encoding="utf-8", errors="replace")


# ═════════════════════════════════════════════════════════════════════
# 4. check_company — Company Health Signal
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="check_company",
    description=(
        "Quick background check on a company before applying. "
        "Searches Crunchbase (free tier) for funding info and web for "
        "Glassdoor ratings. Flags red flags like low ratings, tiny headcount, "
        "or stale funding."
    ),
    tags=["job", "research"],
)
async def check_company(company_name: str) -> str:
    """Check company health signals — funding, rating, headcount."""
    signals: Dict[str, Any] = {"company": company_name, "flags": []}

    # Run checks concurrently
    crunchbase_task = asyncio.create_task(_check_crunchbase(company_name))
    web_task = asyncio.create_task(_check_company_web(company_name))

    crunchbase_info, web_info = await asyncio.gather(
        crunchbase_task, web_task, return_exceptions=True
    )

    if isinstance(crunchbase_info, dict):
        signals.update(crunchbase_info)
    if isinstance(web_info, dict):
        signals.update(web_info)

    # Analyze flags
    flags = []
    if signals.get("employee_count") and signals["employee_count"] < 10:
        flags.append("Very small company (<10 employees)")
    if signals.get("glassdoor_rating") and signals["glassdoor_rating"] < 3.0:
        flags.append(f"Low Glassdoor rating ({signals['glassdoor_rating']})")
    if signals.get("last_funding_year"):
        years_since = 2026 - signals["last_funding_year"]
        if years_since > 3:
            flags.append(f"Last funded {years_since} years ago")
    signals["flags"] = flags

    # Format output
    lines = [f"**Company Report: {company_name}**\n"]
    if signals.get("description"):
        lines.append(f"Description: {signals['description']}")
    if signals.get("funding_stage"):
        lines.append(f"Funding Stage: {signals['funding_stage']}")
    if signals.get("total_funding"):
        lines.append(f"Total Funding: {signals['total_funding']}")
    if signals.get("employee_count"):
        lines.append(f"Employees: ~{signals['employee_count']}")
    if signals.get("glassdoor_rating"):
        lines.append(f"Glassdoor Rating: {signals['glassdoor_rating']}/5.0")
    if signals.get("headquarters"):
        lines.append(f"HQ: {signals['headquarters']}")
    if signals.get("website"):
        lines.append(f"Website: {signals['website']}")

    if flags:
        lines.append(f"\n⚠ Red Flags:")
        for f in flags:
            lines.append(f"  - {f}")
    else:
        lines.append(f"\n✓ No red flags detected")

    return "\n".join(lines)


async def _check_crunchbase(company: str) -> Dict[str, Any]:
    """Query Crunchbase basic (free) API if key is available."""
    api_key = os.getenv("CRUNCHBASE_API_KEY", "")
    if not api_key:
        return {}

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                "https://api.crunchbase.com/api/v4/autocompletes",
                params={"query": company, "user_key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            entities = data.get("entities", [])
            if entities:
                entity = entities[0]
                props = entity.get("properties", {})
                return {
                    "description": props.get("short_description", ""),
                    "funding_stage": props.get("funding_stage", ""),
                    "employee_count": props.get("num_employees_enum", ""),
                    "headquarters": props.get("location_identifiers", [{}])[0].get("value", "")
                    if props.get("location_identifiers") else "",
                }
        except Exception as exc:
            logger.debug("Crunchbase lookup failed: %s", exc)
    return {}


async def _check_company_web(company: str) -> Dict[str, Any]:
    """Search the web for company info (Glassdoor rating, etc.)."""
    from bs4 import BeautifulSoup

    info: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": f"{company} glassdoor rating employees"},
                headers={"User-Agent": "Mozilla/5.0 (AgentForge Bot)"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text()

            # Try to extract Glassdoor rating
            rating_match = re.search(r"(\d\.\d)\s*/?\s*5(?:\.\d)?\s*(?:stars?|rating)", text, re.IGNORECASE)
            if rating_match:
                info["glassdoor_rating"] = float(rating_match.group(1))

            # Try to find employee count
            emp_match = re.search(r"(\d[\d,]*)\s*(?:employees|staff|people)", text, re.IGNORECASE)
            if emp_match:
                info["employee_count"] = int(emp_match.group(1).replace(",", ""))

        except Exception as exc:
            logger.debug("Company web check failed: %s", exc)

    return info


# ═════════════════════════════════════════════════════════════════════
# 5. track_application — Application State Machine
# ═════════════════════════════════════════════════════════════════════

VALID_STATUSES = {"found", "applied", "interviewing", "offer", "rejected", "withdrawn"}

@tool(
    name="track_application",
    description=(
        "Track and update job application status in a local SQLite database. "
        "Actions: 'add' (new application), 'update' (change status), "
        "'list' (show all applications), 'get' (details for one job_id). "
        "Valid statuses: found, applied, interviewing, offer, rejected, withdrawn."
    ),
    tags=["job", "tracking"],
)
def track_application(
    action: str = "list",
    job_id: str = "",
    company: str = "",
    role: str = "",
    url: str = "",
    source: str = "",
    status: str = "found",
    match_score: int = 0,
    salary: str = "",
    location: str = "",
    notes: str = "",
) -> str:
    """Log and update application status in local DB."""
    db = _get_db()

    if action == "add":
        if not job_id or not company or not role:
            return "ERROR: 'add' requires job_id, company, and role."
        try:
            db.execute(
                """INSERT OR REPLACE INTO applications
                   (job_id, company, role, url, source, status, match_score, salary, location, notes, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))""",
                (job_id, company, role, url, source, status, int(match_score), salary, location, notes),
            )
            db.commit()
            return f"Application tracked: {role} at {company} [{status}]"
        finally:
            db.close()

    elif action == "update":
        if not job_id:
            return "ERROR: 'update' requires job_id."
        if status and status not in VALID_STATUSES:
            return f"ERROR: Invalid status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}"
        try:
            updates = []
            params: list = []
            if status:
                updates.append("status = ?")
                params.append(status)
            if notes:
                updates.append("notes = ?")
                params.append(notes)
            if match_score:
                updates.append("match_score = ?")
                params.append(int(match_score))
            updates.append("updated_at = strftime('%s','now')")
            params.append(job_id)
            db.execute(f"UPDATE applications SET {', '.join(updates)} WHERE job_id = ?", params)
            db.commit()
            return f"Updated application {job_id}: status={status}"
        finally:
            db.close()

    elif action == "get":
        if not job_id:
            return "ERROR: 'get' requires job_id."
        try:
            row = db.execute("SELECT * FROM applications WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return f"No application found for job_id: {job_id}"
            return json.dumps(dict(row), indent=2)
        finally:
            db.close()

    elif action == "list":
        try:
            rows = db.execute(
                "SELECT job_id, company, role, status, match_score, url FROM applications ORDER BY updated_at DESC"
            ).fetchall()
            if not rows:
                return "No applications tracked yet."
            lines = [f"**Tracked Applications ({len(rows)}):**\n"]
            for r in rows:
                score_str = f" [{r['match_score']}%]" if r["match_score"] else ""
                lines.append(
                    f"- [{r['status'].upper():^13}] **{r['role']}** @ {r['company']}{score_str}"
                )
            return "\n".join(lines)
        finally:
            db.close()

    else:
        return f"ERROR: Unknown action '{action}'. Use: add, update, get, list."


# ═════════════════════════════════════════════════════════════════════
# 6. monitor_new_jobs — Background Dedup Watcher
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="monitor_new_jobs",
    description=(
        "Check for new job listings that haven't been seen before. "
        "Runs a search and compares results against a local seen-jobs table. "
        "Returns only genuinely new listings. Use this for periodic monitoring."
    ),
    tags=["job", "monitoring"],
)
async def monitor_new_jobs(query: str, location: str = "remote") -> str:
    """Search for jobs and return only unseen ones."""
    # Run the search
    all_results: List[Dict[str, Any]] = []

    try:
        remotive = await _search_remotive(query)
        all_results.extend(remotive)
    except Exception as exc:
        logger.warning("Remotive failed during monitor: %s", exc)

    try:
        adzuna = await _search_adzuna(query, location, 3)
        all_results.extend(adzuna)
    except Exception as exc:
        logger.warning("Adzuna failed during monitor: %s", exc)

    if not all_results:
        return "No listings found to check."

    db = _get_db()
    try:
        new_jobs = []
        for job in all_results:
            jid = job.get("id") or f"{job.get('title', '')}-{job.get('company', '')}"
            source = job.get("source", "unknown")
            row = db.execute("SELECT 1 FROM seen_jobs WHERE job_id = ?", (jid,)).fetchone()
            if not row:
                # New job — record it
                db.execute(
                    "INSERT OR IGNORE INTO seen_jobs (job_id, source) VALUES (?, ?)",
                    (jid, source),
                )
                new_jobs.append(job)
        db.commit()
    finally:
        db.close()

    if not new_jobs:
        return f"No new jobs found for '{query}'. All {len(all_results)} listings were already seen."

    lines = [f"🆕 {len(new_jobs)} new job(s) found for '{query}':\n"]
    for i, job in enumerate(new_jobs[:10], 1):
        lines.append(f"{i}. **{job.get('title', 'N/A')}** — {job.get('company', 'N/A')}")
        lines.append(f"   URL: {job.get('url', 'N/A')}")
        lines.append(f"   Source: {job.get('source', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# 7. send_alert — Notification on High-Match Job
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="send_alert",
    description=(
        "Send a notification alert for a high-match job. "
        "Supports channels: 'email' (Gmail SMTP), 'telegram', 'slack'. "
        "Requires corresponding env vars to be set. "
        "Falls back to logging the alert if no channel is configured."
    ),
    tags=["job", "notification"],
)
async def send_alert(
    role: str,
    company: str,
    match_score: int = 0,
    url: str = "",
    salary: str = "",
    channel: str = "email",
) -> str:
    """Send a job match notification via the configured channel."""
    match_score = int(match_score)

    subject = f"Job Match Alert: {role} @ {company} ({match_score}%)"
    body_lines = [
        f"Role: {role}",
        f"Company: {company}",
        f"Match Score: {match_score}%",
    ]
    if salary:
        body_lines.append(f"Salary: {salary}")
    if url:
        body_lines.append(f"Apply: {url}")
    body = "\n".join(body_lines)

    sent_via = []

    # ── Email (Gmail SMTP) ───────────────────────────────────────
    if channel in ("email", "all"):
        email_user = os.getenv("ALERT_EMAIL_USER", "")
        email_pass = os.getenv("ALERT_EMAIL_PASSWORD", "")  # Gmail app password
        email_to = os.getenv("ALERT_EMAIL_TO", email_user)
        if email_user and email_pass:
            try:
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = email_user
                msg["To"] = email_to
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(email_user, email_pass)
                    server.send_message(msg)
                sent_via.append("email")
            except Exception as exc:
                logger.warning("Email alert failed: %s", exc)

    # ── Telegram Bot ─────────────────────────────────────────────
    if channel in ("telegram", "all"):
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if bot_token and chat_id:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    await client.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        json={"chat_id": chat_id, "text": f"{subject}\n\n{body}"},
                    )
                sent_via.append("telegram")
            except Exception as exc:
                logger.warning("Telegram alert failed: %s", exc)

    # ── Slack Webhook ────────────────────────────────────────────
    if channel in ("slack", "all"):
        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        if webhook_url:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    await client.post(
                        webhook_url,
                        json={"text": f"*{subject}*\n{body}"},
                    )
                sent_via.append("slack")
            except Exception as exc:
                logger.warning("Slack alert failed: %s", exc)

    if sent_via:
        return f"Alert sent via: {', '.join(sent_via)}\n{subject}"
    else:
        # Fallback: just log it
        logger.info("JOB ALERT (no channel configured): %s\n%s", subject, body)
        return (
            f"Alert logged (no notification channel configured):\n{subject}\n\n"
            f"Set env vars to enable: ALERT_EMAIL_USER/PASSWORD, "
            f"TELEGRAM_BOT_TOKEN/CHAT_ID, or SLACK_WEBHOOK_URL"
        )


# ═════════════════════════════════════════════════════════════════════
# 8. classify_companies — Product-based vs Service-based Classifier
# ═════════════════════════════════════════════════════════════════════

# Built-in knowledge base of well-known companies by type.
# This avoids API calls for common names and provides a fast fallback.
_COMPANY_CLASSIFICATIONS: Dict[str, str] = {
    # Product-based
    "google": "product", "meta": "product", "facebook": "product",
    "apple": "product", "amazon": "product", "microsoft": "product",
    "netflix": "product", "spotify": "product", "uber": "product",
    "lyft": "product", "airbnb": "product", "stripe": "product",
    "shopify": "product", "slack": "product", "zoom": "product",
    "dropbox": "product", "twilio": "product", "datadog": "product",
    "snowflake": "product", "palantir": "product", "coinbase": "product",
    "robinhood": "product", "swiggy": "product", "zomato": "product",
    "cred": "product", "razorpay": "product", "phonepe": "product",
    "paytm": "product", "flipkart": "product", "meesho": "product",
    "groww": "product", "zerodha": "product", "ola": "product",
    "myntra": "product", "dream11": "product", "nykaa": "product",
    "freshworks": "product", "zoho": "product", "postman": "product",
    "notion": "product", "figma": "product", "canva": "product",
    "atlassian": "product", "jira": "product", "github": "product",
    "gitlab": "product", "hashicorp": "product", "elastic": "product",
    "mongodb": "product", "redis": "product", "confluent": "product",
    "databricks": "product", "openai": "product", "anthropic": "product",
    "hugging face": "product", "tesla": "product", "nvidia": "product",
    "amd": "product", "intel": "product", "adobe": "product",
    "salesforce": "product", "oracle": "product", "sap": "product",
    "vmware": "product", "cloudflare": "product", "crowdstrike": "product",
    "palo alto networks": "product", "servicenow": "product",
    "twitter": "product", "x": "product", "linkedin": "product",
    "pinterest": "product", "snap": "product", "snapchat": "product",
    "disney": "product", "hotstar": "product",
    # Service / Consulting / IT outsourcing
    "tcs": "service", "tata consultancy services": "service",
    "infosys": "service", "wipro": "service", "hcl": "service",
    "hcltech": "service", "cognizant": "service", "tech mahindra": "service",
    "capgemini": "service", "accenture": "service", "deloitte": "consulting",
    "kpmg": "consulting", "pwc": "consulting", "ey": "consulting",
    "ernst & young": "consulting", "mckinsey": "consulting",
    "bcg": "consulting", "boston consulting group": "consulting",
    "bain": "consulting", "bain & company": "consulting",
    "lti": "service", "ltimindtree": "service", "mindtree": "service",
    "mphasis": "service", "hexaware": "service", "cyient": "service",
    "l&t infotech": "service", "persistent": "service",
    "zensar": "service", "birlasoft": "service", "niit": "service",
    "coforge": "service", "sonata software": "service",
    "ibm": "hybrid", "sap": "hybrid",
    # Startups (notable)
    "ycombinator": "startup", "sequoia": "startup",
}

# Heuristic keywords that hint at company type from a job description
_PRODUCT_SIGNALS = [
    "our product", "our platform", "ship features", "product team",
    "b2c", "b2b saas", "consumer app", "our users", "user growth",
    "product-market fit", "product roadmap", "feature development",
]
_SERVICE_SIGNALS = [
    "client projects", "client-facing", "consulting engagement",
    "delivery team", "onsite", "offshore", "billable hours",
    "multiple clients", "client requirements", "staffing",
    "managed services", "it outsourcing",
]


@tool(
    name="classify_companies",
    description=(
        "Classify companies as 'product' (product-based), 'service' (IT services/outsourcing), "
        "'consulting' (strategy/management consulting), 'startup', or 'hybrid'. "
        "Pass a JSON array of company names OR a JSON string from search_jobs output. "
        "Returns JSON with each company mapped to its classification and confidence."
    ),
    tags=["job", "classification"],
)
async def classify_companies(companies_json: str) -> str:
    """Classify companies by type — product-based, service-based, consulting, etc.

    Args:
        companies_json: JSON array of company names, e.g. '["Swiggy", "TCS", "Stripe"]'
                        OR the full JSON output from search_jobs (will extract company names).
    """
    # Parse input — accept either a plain list or search_jobs output
    try:
        data = json.loads(companies_json)
    except json.JSONDecodeError:
        # Treat as a single company name
        data = [companies_json.strip()]

    company_names: List[str] = []
    if isinstance(data, list):
        # Could be list of strings or list of job dicts
        for item in data:
            if isinstance(item, str):
                company_names.append(item)
            elif isinstance(item, dict) and "company" in item:
                company_names.append(item["company"])
    elif isinstance(data, dict) and "jobs" in data:
        # search_jobs output format
        for job in data["jobs"]:
            if isinstance(job, dict) and job.get("company"):
                company_names.append(job["company"])

    # Deduplicate while preserving order
    seen_names: set = set()
    unique_names: List[str] = []
    for name in company_names:
        key = name.strip().lower()
        if key and key not in seen_names:
            seen_names.add(key)
            unique_names.append(name.strip())

    if not unique_names:
        return json.dumps({"error": "No company names found in input."})

    classifications: List[Dict[str, Any]] = []

    for name in unique_names:
        normalized = name.lower().strip()
        # 1. Check built-in knowledge base
        if normalized in _COMPANY_CLASSIFICATIONS:
            classifications.append({
                "company": name,
                "type": _COMPANY_CLASSIFICATIONS[normalized],
                "confidence": "high",
                "source": "knowledge_base",
            })
            continue

        # 2. Check partial matches (e.g. "Tata Consultancy" matching "tcs")
        matched = False
        for known, ctype in _COMPANY_CLASSIFICATIONS.items():
            if known in normalized or normalized in known:
                classifications.append({
                    "company": name,
                    "type": ctype,
                    "confidence": "medium",
                    "source": "knowledge_base_partial",
                })
                matched = True
                break

        if matched:
            continue

        # 3. Web-based heuristic — search for the company
        ctype = await _classify_company_web(name)
        classifications.append({
            "company": name,
            "type": ctype,
            "confidence": "low" if ctype == "unknown" else "medium",
            "source": "web_heuristic",
        })

    return json.dumps({"classifications": classifications}, indent=2)


async def _classify_company_web(company: str) -> str:
    """Use a DuckDuckGo search to heuristically classify a company."""
    from bs4 import BeautifulSoup

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": f"{company} company about product service"},
                headers={"User-Agent": "Mozilla/5.0 (AgentForge Bot)"},
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text().lower()

        product_score = sum(1 for signal in _PRODUCT_SIGNALS if signal in text)
        service_score = sum(1 for signal in _SERVICE_SIGNALS if signal in text)

        if product_score > service_score and product_score >= 2:
            return "product"
        elif service_score > product_score and service_score >= 2:
            return "service"
        elif "consulting" in text and "partner" in text:
            return "consulting"
        elif "startup" in text or "seed" in text or "series a" in text:
            return "startup"
        else:
            return "unknown"
    except Exception as exc:
        logger.debug("Web classification failed for %s: %s", company, exc)
        return "unknown"


# ═════════════════════════════════════════════════════════════════════
# 9. filter_jobs — Smart Job Filtering Pipeline
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="filter_jobs",
    description=(
        "Filter job listings by experience level, company type, and role relevance. "
        "Takes JSON input from search_jobs and/or classify_companies. "
        "Returns a filtered JSON array of jobs that match the criteria."
    ),
    tags=["job", "filter"],
)
async def filter_jobs(
    jobs_json: str,
    min_experience_years: int = 0,
    max_experience_years: int = 99,
    company_types: str = "",
    role_keywords: str = "",
    exclude_companies: str = "",
    remote_only: bool = False,
    classifications_json: str = "",
) -> str:
    """Filter jobs by experience, company type, role relevance, and more.

    Args:
        jobs_json: JSON output from search_jobs (contains 'jobs' array).
        min_experience_years: Minimum years of experience required (default: 0).
        max_experience_years: Maximum years of experience to target (default: 99).
        company_types: Comma-separated list of desired types — 'product', 'service',
                       'consulting', 'startup' (empty = no filter).
        role_keywords: Comma-separated keywords the role title must contain
                       (e.g. 'backend,python,engineer'). Empty = no filter.
        exclude_companies: Comma-separated company names to exclude.
        remote_only: If true, only keep remote positions.
        classifications_json: JSON output from classify_companies (optional).
                              Used to filter by company type.
    """
    min_experience_years = int(min_experience_years)
    max_experience_years = int(max_experience_years)

    # Parse inputs
    try:
        jobs_data = json.loads(jobs_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid jobs_json — expected JSON from search_jobs."})

    jobs: List[Dict[str, Any]] = []
    if isinstance(jobs_data, dict) and "jobs" in jobs_data:
        jobs = jobs_data["jobs"]
    elif isinstance(jobs_data, list):
        jobs = jobs_data

    if not jobs:
        return json.dumps({"filtered_jobs": [], "total": 0, "reason": "No jobs in input."})

    # Parse classifications
    company_type_map: Dict[str, str] = {}
    if classifications_json:
        try:
            cls_data = json.loads(classifications_json)
            for item in cls_data.get("classifications", []):
                company_type_map[item["company"].lower()] = item["type"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Parse filter criteria
    desired_types = {t.strip().lower() for t in company_types.split(",") if t.strip()}
    keywords = [k.strip().lower() for k in role_keywords.split(",") if k.strip()]
    excluded = {c.strip().lower() for c in exclude_companies.split(",") if c.strip()}

    filtered: List[Dict[str, Any]] = []
    reasons_excluded: Dict[str, int] = {
        "company_type": 0, "role_keywords": 0, "excluded_company": 0,
        "experience": 0, "remote": 0,
    }

    for job in jobs:
        company = job.get("company", "").strip()
        title = job.get("title", "").strip()
        title_lower = title.lower()
        company_lower = company.lower()

        # 1. Exclude specific companies
        if company_lower in excluded:
            reasons_excluded["excluded_company"] += 1
            continue

        # 2. Company type filter
        if desired_types:
            ctype = company_type_map.get(company_lower, "unknown")
            if ctype not in desired_types and ctype != "unknown":
                reasons_excluded["company_type"] += 1
                continue

        # 3. Role keyword filter
        if keywords:
            if not any(kw in title_lower for kw in keywords):
                reasons_excluded["role_keywords"] += 1
                continue

        # 4. Experience filter — extract years from title/description
        exp_years = _extract_experience(title, job.get("description_snippet", ""))
        if exp_years is not None:
            if exp_years < min_experience_years or exp_years > max_experience_years:
                reasons_excluded["experience"] += 1
                continue

        # 5. Remote filter
        if remote_only:
            loc = job.get("location", "").lower()
            is_remote = job.get("is_remote", False)
            if not is_remote and "remote" not in loc and "worldwide" not in loc:
                reasons_excluded["remote"] += 1
                continue

        # Annotate with company type if known
        job["company_type"] = company_type_map.get(company_lower, "unknown")
        filtered.append(job)

    output = {
        "filtered_jobs": filtered,
        "total": len(filtered),
        "removed": len(jobs) - len(filtered),
        "exclusion_reasons": {k: v for k, v in reasons_excluded.items() if v > 0},
    }
    return json.dumps(output, indent=2)


def _extract_experience(title: str, description: str = "") -> Optional[int]:
    """Try to extract years-of-experience requirement from title or description."""
    text = f"{title} {description}".lower()
    # Patterns: "3+ years", "5-7 years", "senior (8+)", "junior", "entry level"
    patterns = [
        r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"(\d+)\s*-\s*\d+\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:years?|yrs?)\s*(?:experience|exp)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return int(match.group(1))

    # Seniority heuristics
    if any(kw in text for kw in ["entry level", "entry-level", "junior", "fresher", "intern"]):
        return 0
    if any(kw in text for kw in ["mid-level", "mid level", "intermediate"]):
        return 3
    if any(kw in text for kw in ["senior", "sr.", "lead", "principal", "staff"]):
        return 5

    return None  # Unknown


# ═════════════════════════════════════════════════════════════════════
# 10. enrich_job — Company Enrichment (Clearbit + Wikipedia)
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="enrich_job",
    description=(
        "Enrich job listings with company metadata — employee count, industry, "
        "founding year, domain, and description. Uses Clearbit (if API key set) "
        "and Wikipedia as fallback. Takes JSON input from filter_jobs or search_jobs."
    ),
    tags=["job", "enrichment"],
)
async def enrich_job(jobs_json: str, max_enrichments: int = 10) -> str:
    """Enrich job listings with company info from Clearbit and Wikipedia.

    Args:
        jobs_json: JSON from filter_jobs or search_jobs (contains 'jobs' or 'filtered_jobs').
        max_enrichments: Max number of unique companies to enrich (default: 10).
    """
    max_enrichments = int(max_enrichments)

    try:
        data = json.loads(jobs_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})

    jobs: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        jobs = data.get("filtered_jobs", data.get("jobs", []))
    elif isinstance(data, list):
        jobs = data

    if not jobs:
        return json.dumps({"enriched_jobs": [], "total": 0})

    # Collect unique companies to enrich
    seen_companies: set = set()
    companies_to_enrich: List[str] = []
    for job in jobs:
        company = job.get("company", "").strip()
        if company.lower() not in seen_companies:
            seen_companies.add(company.lower())
            companies_to_enrich.append(company)

    # Enrich concurrently (capped)
    enrichment_cache: Dict[str, Dict[str, Any]] = {}
    tasks = []
    for company in companies_to_enrich[:max_enrichments]:
        tasks.append(_enrich_company(company))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for company, result in zip(companies_to_enrich[:max_enrichments], results):
        if isinstance(result, dict):
            enrichment_cache[company.lower()] = result
        else:
            logger.debug("Enrichment failed for %s: %s", company, result)
            enrichment_cache[company.lower()] = {}

    # Merge enrichment data into jobs
    for job in jobs:
        company = job.get("company", "").strip().lower()
        enrichment = enrichment_cache.get(company, {})
        if enrichment:
            job["company_info"] = enrichment

    output = {
        "enriched_jobs": jobs,
        "total": len(jobs),
        "companies_enriched": len([v for v in enrichment_cache.values() if v]),
    }
    return json.dumps(output, indent=2)


async def _enrich_company(company: str) -> Dict[str, Any]:
    """Enrich a single company using Clearbit and/or Wikipedia."""
    info: Dict[str, Any] = {}

    # Try Clearbit first
    clearbit_key = os.getenv("CLEARBIT_API_KEY", "")
    if clearbit_key:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://company.clearbit.com/v2/companies/find",
                    params={"name": company},
                    headers={"Authorization": f"Bearer {clearbit_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    info = {
                        "name": data.get("name", company),
                        "domain": data.get("domain", ""),
                        "description": data.get("description", ""),
                        "industry": data.get("category", {}).get("industry", ""),
                        "employee_count": data.get("metrics", {}).get("employees", ""),
                        "employee_range": data.get("metrics", {}).get("employeesRange", ""),
                        "founded_year": data.get("foundedYear", ""),
                        "location": data.get("geo", {}).get("city", ""),
                        "country": data.get("geo", {}).get("country", ""),
                        "tech": data.get("tech", [])[:5],
                        "source": "clearbit",
                    }
                    return info
        except Exception as exc:
            logger.debug("Clearbit failed for %s: %s", company, exc)

    # Fallback — Wikipedia summary
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + company.replace(" ", "_"),
            )
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                info = {
                    "name": data.get("title", company),
                    "description": extract[:400],
                    "source": "wikipedia",
                }

                # Try to extract employee count from extract
                emp_match = re.search(
                    r"(\d[\d,]+)\s*(?:employees|staff|people|workers)",
                    extract, re.IGNORECASE,
                )
                if emp_match:
                    info["employee_count"] = int(emp_match.group(1).replace(",", ""))

                # Try to extract founding year
                year_match = re.search(
                    r"(?:founded|established|incorporated)\s*(?:in\s*)?(\d{4})",
                    extract, re.IGNORECASE,
                )
                if year_match:
                    info["founded_year"] = int(year_match.group(1))

                return info
    except Exception as exc:
        logger.debug("Wikipedia failed for %s: %s", company, exc)

    return {"name": company, "source": "none", "description": "No enrichment data found."}


# ═════════════════════════════════════════════════════════════════════
# 11. rank_and_format — Score, Rank & Present Results
# ═════════════════════════════════════════════════════════════════════

@tool(
    name="rank_and_format",
    description=(
        "Score and rank job listings based on configurable weights: salary, "
        "company size/reputation, role relevance, recency, and company type. "
        "Returns a clean formatted report. Takes JSON from enrich_job or filter_jobs."
    ),
    tags=["job", "ranking"],
)
async def rank_and_format(
    jobs_json: str,
    preferred_role: str = "",
    preferred_company_type: str = "product",
    salary_weight: float = 0.25,
    relevance_weight: float = 0.30,
    company_weight: float = 0.25,
    recency_weight: float = 0.20,
) -> str:
    """Score, rank, and format job listings into a clean report.

    Args:
        jobs_json: JSON from enrich_job, filter_jobs, or search_jobs.
        preferred_role: Preferred role title keywords (e.g. 'backend engineer').
        preferred_company_type: Preferred company type — 'product', 'startup', etc.
        salary_weight: Weight for salary score (0.0 - 1.0).
        relevance_weight: Weight for role relevance score (0.0 - 1.0).
        company_weight: Weight for company quality score (0.0 - 1.0).
        recency_weight: Weight for recency score (0.0 - 1.0).
    """
    salary_weight = float(salary_weight)
    relevance_weight = float(relevance_weight)
    company_weight = float(company_weight)
    recency_weight = float(recency_weight)

    # Normalize weights
    total_weight = salary_weight + relevance_weight + company_weight + recency_weight
    if total_weight > 0:
        salary_weight /= total_weight
        relevance_weight /= total_weight
        company_weight /= total_weight
        recency_weight /= total_weight

    try:
        data = json.loads(jobs_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})

    jobs: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        jobs = data.get("enriched_jobs", data.get("filtered_jobs", data.get("jobs", [])))
    elif isinstance(data, list):
        jobs = data

    if not jobs:
        return "No jobs to rank."

    preferred_keywords = [k.strip().lower() for k in preferred_role.split() if k.strip()]
    preferred_type = preferred_company_type.strip().lower()

    scored_jobs: List[Dict[str, Any]] = []

    for job in jobs:
        scores: Dict[str, float] = {}

        # ── Salary score (0-100) ─────────────────────────────────
        salary_str = job.get("salary", "")
        salary_score = 0.0
        if salary_str:
            # Extract numeric salary values
            nums = re.findall(r"[\d,]+", salary_str.replace(",", ""))
            if nums:
                max_salary = max(int(n) for n in nums if n.isdigit())
                # Normalize: assume $250k+ is top tier
                salary_score = min(100.0, (max_salary / 250000) * 100)
        scores["salary"] = salary_score

        # ── Relevance score (0-100) ──────────────────────────────
        title = job.get("title", "").lower()
        if preferred_keywords:
            matches = sum(1 for kw in preferred_keywords if kw in title)
            relevance_score = (matches / len(preferred_keywords)) * 100
        else:
            relevance_score = 50.0  # Neutral if no preference
        scores["relevance"] = relevance_score

        # ── Company score (0-100) ────────────────────────────────
        company_score = 50.0  # Default neutral
        company_info = job.get("company_info", {})
        company_type = job.get("company_type", "unknown")

        # Type match bonus
        if company_type == preferred_type:
            company_score += 25.0
        elif company_type == "unknown":
            company_score += 0  # Neutral
        else:
            company_score -= 10.0

        # Employee count bonus (larger = more stable, generally)
        emp = company_info.get("employee_count")
        if isinstance(emp, (int, float)) and emp > 0:
            if emp >= 1000:
                company_score += 15.0
            elif emp >= 100:
                company_score += 10.0
            elif emp >= 10:
                company_score += 5.0

        # Has enrichment data = more reputable (findable company)
        if company_info.get("source") in ("clearbit", "wikipedia"):
            company_score += 10.0

        company_score = min(100.0, max(0.0, company_score))
        scores["company"] = company_score

        # ── Recency score (0-100) ────────────────────────────────
        date_str = job.get("date", "")
        recency_score = 50.0
        if date_str:
            try:
                from datetime import datetime, timezone
                posted = datetime.strptime(date_str[:10], "%Y-%m-%d")
                days_old = (datetime.now() - posted).days
                if days_old <= 1:
                    recency_score = 100.0
                elif days_old <= 3:
                    recency_score = 85.0
                elif days_old <= 7:
                    recency_score = 70.0
                elif days_old <= 14:
                    recency_score = 50.0
                elif days_old <= 30:
                    recency_score = 30.0
                else:
                    recency_score = 10.0
            except ValueError:
                pass
        scores["recency"] = recency_score

        # ── Weighted final score ─────────────────────────────────
        final_score = (
            scores["salary"] * salary_weight
            + scores["relevance"] * relevance_weight
            + scores["company"] * company_weight
            + scores["recency"] * recency_weight
        )

        job["_scores"] = scores
        job["_final_score"] = round(final_score, 1)
        scored_jobs.append(job)

    # Sort by final score descending
    scored_jobs.sort(key=lambda j: j["_final_score"], reverse=True)

    # ── Format clean report ──────────────────────────────────────
    lines = [
        f"# Job Search Results — Ranked ({len(scored_jobs)} jobs)\n",
        f"Preferred role: {preferred_role or 'any'} | "
        f"Preferred company type: {preferred_company_type or 'any'}\n",
        "---\n",
    ]

    for rank, job in enumerate(scored_jobs, 1):
        scores = job["_scores"]
        company_info = job.get("company_info", {})
        ctype = job.get("company_type", "")
        ctype_tag = f" [{ctype.upper()}]" if ctype and ctype != "unknown" else ""

        lines.append(f"### #{rank}  {job.get('title', 'N/A')}  —  {job.get('company', 'N/A')}{ctype_tag}")
        lines.append(f"**Score: {job['_final_score']}/100**  "
                      f"(Relevance: {scores['relevance']:.0f} | "
                      f"Company: {scores['company']:.0f} | "
                      f"Salary: {scores['salary']:.0f} | "
                      f"Recency: {scores['recency']:.0f})")
        lines.append("")

        if job.get("salary"):
            lines.append(f"- **Salary:** {job['salary']}")
        lines.append(f"- **Location:** {job.get('location', 'N/A')}")
        lines.append(f"- **Posted:** {job.get('date', 'N/A')}")
        lines.append(f"- **Source:** {job.get('source', 'N/A')}")
        if company_info.get("employee_count"):
            lines.append(f"- **Company Size:** ~{company_info['employee_count']:,} employees")
        if company_info.get("industry"):
            lines.append(f"- **Industry:** {company_info['industry']}")
        if company_info.get("founded_year"):
            lines.append(f"- **Founded:** {company_info['founded_year']}")
        lines.append(f"- **Apply:** {job.get('url', 'N/A')}")
        lines.append("")
        lines.append("---\n")

    # Summary
    avg_score = sum(j["_final_score"] for j in scored_jobs) / len(scored_jobs) if scored_jobs else 0
    top_companies = list(dict.fromkeys(j.get("company", "") for j in scored_jobs[:5]))
    lines.append(f"\n**Summary:** {len(scored_jobs)} jobs ranked. "
                  f"Average score: {avg_score:.1f}/100. "
                  f"Top companies: {', '.join(top_companies)}.")

    return "\n".join(lines)
