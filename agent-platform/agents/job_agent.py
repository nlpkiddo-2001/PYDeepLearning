"""
Job Hunter Agent Template
=========================
Workflow: scrape_jobs → resume_match → summarize → send_notification

Pre-wired to search job boards, filter by criteria, match against a resume,
and summarize the top opportunities.
"""

from agents.base_agent import BaseAgentTemplate


class JobHunterAgent(BaseAgentTemplate):
    default_name = "job-hunter"
    description = (
        "An autonomous job hunting agent that scrapes real-time job listings "
        "from LinkedIn, Naukri, and company career pages using Playwright, "
        "filters results by criteria, matches against your resume, "
        "and sends you a summary of the best opportunities."
    )
    recommended_provider = "openai"
    recommended_model = "gpt-4o"
    required_tools = [
        # Real-time Playwright scraping
        "scrape_live_jobs",
        "scrape_career_page",
        "get_scraped_jobs_history",
        # Core pipeline
        "search_jobs",
        "filter_jobs",
        "classify_companies",
        "enrich_job",
        "rank_and_format",
        # Supporting
        "web_search",
        "scrape_url",
        "write_file",
        "send_email",
    ]
    default_goal_prompt = """You are a Job Hunter AI agent with real-time scraping capabilities.

When given a job search goal:
1. Use scrape_live_jobs to scrape real-time listings from LinkedIn, Naukri, and career pages
   (Playwright browser with anti-bot bypass: stealth plugin, human delays, UA rotation)
2. Supplement with search_jobs for broader API-based coverage (Adzuna, Remotive, JSearch)
3. Use classify_companies to tag companies as product/service/consulting/startup
4. Use filter_jobs to filter by experience, company type, and role keywords
5. Use enrich_job for company metadata (size, industry, funding)
6. Use rank_and_format to score and present the top results
7. For promising matches, use match_resume for skill gap analysis
8. Track applications and send alerts for high-match jobs (>75%)

For specific companies, use scrape_career_page with their careers URL.
Use get_scraped_jobs_history to check previously seen jobs and avoid duplicates.

Always provide structured output with:
- Job Title
- Company
- Location (remote/hybrid/onsite)
- Key Requirements / Skills
- Experience Level
- Salary (if available)
- Match Score (1-10)
- Application URL
- Source (LinkedIn Scraped / Naukri Scraped / Career Page / API)

Focus on quality over quantity — present the top 5-10 most relevant matches."""
