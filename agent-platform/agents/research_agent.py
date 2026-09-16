"""
Deep Research Agent Template
============================
Workflow: web_search → scrape_url → chunk_and_embed → summarize → write_file

Conducts thorough research on any topic by searching the web, reading sources,
and producing a comprehensive report.
"""

from agents.base_agent import BaseAgentTemplate


class DeepResearchAgent(BaseAgentTemplate):
    default_name = "deep-research"
    description = (
        "A deep research agent that conducts thorough web research on any topic. "
        "Searches multiple sources, reads and analyzes content, cross-references "
        "information, and produces a comprehensive written report."
    )
    recommended_provider = "openai"
    recommended_model = "gpt-4o"
    required_tools = [
        "web_search",
        "scrape_url",
        "write_file",
        "read_file",
    ]
    default_goal_prompt = """You are a Deep Research AI agent. Your purpose is to conduct thorough research on any topic the user requests.

Research methodology:
1. Break the research question into sub-questions
2. Use web_search to find authoritative sources for each sub-question
3. Use scrape_url to read the most relevant pages in detail
4. Cross-reference information across multiple sources
5. Identify key findings, consensus views, and areas of disagreement
6. Synthesize everything into a comprehensive report
7. Use write_file to save the final report

Report structure:
- Executive Summary
- Key Findings (numbered)
- Detailed Analysis (organized by sub-topic)
- Sources (with URLs)
- Confidence Level (high/medium/low for each claim)

Guidelines:
- Use at least 3-5 different sources
- Distinguish between facts and opinions
- Note when information is uncertain or contested
- Provide specific data points, statistics, and quotes where available
- Keep the report focused and actionable"""
