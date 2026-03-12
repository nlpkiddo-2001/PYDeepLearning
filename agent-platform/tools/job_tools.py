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
        "Search live job listings across Remotive (remote-only, free) and "
        "Adzuna (free tier). Returns title, company, URL, location, salary, "
        "and posted date. Use location='remote' for remote roles."
    ),
    tags=["job", "search"],
)
async def search_jobs(query: str, location: str = "remote", days_ago: int = 7) -> str:
    """Search live job listings across multiple free sources."""
    days_ago = int(days_ago)
    results: List[Dict[str, Any]] = []

    # Run sources concurrently
    remotive_task = asyncio.create_task(_search_remotive(query))
    adzuna_task = asyncio.create_task(_search_adzuna(query, location, days_ago))

    remotive_results, adzuna_results = await asyncio.gather(
        remotive_task, adzuna_task, return_exceptions=True
    )

    if isinstance(remotive_results, list):
        results.extend(remotive_results)
    else:
        logger.warning("Remotive search failed: %s", remotive_results)

    if isinstance(adzuna_results, list):
        results.extend(adzuna_results)
    else:
        logger.warning("Adzuna search failed: %s", adzuna_results)

    if not results:
        return f"No job listings found for '{query}' in '{location}'."

    # De-duplicate by title+company
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = (r.get("title", "").lower(), r.get("company", "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Format output
    lines = [f"Found {len(unique)} jobs for '{query}' ({location}):\n"]
    for i, job in enumerate(unique[:15], 1):
        lines.append(f"{i}. **{job.get('title', 'N/A')}** — {job.get('company', 'N/A')}")
        lines.append(f"   Location: {job.get('location', 'N/A')}")
        if job.get("salary"):
            lines.append(f"   Salary: {job['salary']}")
        lines.append(f"   Posted: {job.get('date', 'N/A')}")
        lines.append(f"   URL: {job.get('url', 'N/A')}")
        lines.append(f"   Source: {job.get('source', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


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
