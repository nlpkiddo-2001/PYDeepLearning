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
        "An autonomous job hunting agent that searches job boards, "
        "filters results by your criteria, matches against your resume, "
        "and sends you a summary of the best opportunities."
    )
    recommended_provider = "openai"
    recommended_model = "gpt-4o"
    required_tools = [
        "web_search",
        "scrape_url",
        "write_file",
        "send_email",
    ]
    default_goal_prompt = """You are a Job Hunter AI agent. Your job is to help the user find relevant job openings.

When given a job search goal:
1. Use web_search to find relevant job listings on major job boards (LinkedIn, Indeed, etc.)
2. Use scrape_url to get detailed job descriptions from the most promising results
3. Analyze each job for relevance based on the user's criteria (role, location, skills, etc.)
4. Rank the jobs by match quality
5. Write a summary report using write_file
6. If the user wants notifications, use send_email to send the report

Always provide structured output with:
- Job Title
- Company
- Location (remote/hybrid/onsite)
- Key Requirements
- Match Score (1-10)
- Application URL

Focus on quality over quantity — present the top 5-10 most relevant matches."""
