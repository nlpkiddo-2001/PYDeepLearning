"""
Example: Running the Job Hunter Agent programmatically.

Usage:
    cd agent-platform
    python examples/job_hunting_agent/run.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.agent import AgentManager
from core.memory import MemoryManager
from core.streaming import event_bus, StreamEvent
from tools.registry import ToolRegistry


async def main():
    # 1. Initialize components
    tool_registry = ToolRegistry(tool_dirs=["tools/"])
    tool_registry.discover()
    print(f"Discovered tools: {tool_registry.names()}")

    memory = MemoryManager()

    # 2. Create agent manager
    manager = AgentManager(
        default_config={
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "${OPENAI_API_KEY}",
                "temperature": 0.7,
            }
        },
        tool_registry=tool_registry,
        memory_manager=memory,
    )

    # 3. Register the Job Hunter agent
    agent = manager.register({
        "id": "job_hunter_demo",
        "name": "Job Hunter Demo",
        "description": "Find ML engineering jobs",
        "tools": ["web_search", "scrape_url", "write_file"],
        "max_steps": 100,
    })

    # 4. Subscribe to events and print them
    queue = event_bus.subscribe("job_hunter_demo")

    async def print_events():
        while True:
            event: StreamEvent = await queue.get()
            print(f"  [{event.type.value.upper():12s}] Step {event.step}: {event.data}")
            if event.type.value in ("done", "error"):
                break

    # 5. Run the agent
    print("\n⚡ Starting Job Hunter Agent...\n")
    event_task = asyncio.create_task(print_events())
    result = await agent.run("Find remote ML engineer jobs posted this week")

    await event_task
    print(f"\n✓ Result: {result.final_result}")
    print(f"  Steps: {len(result.steps)}, Tokens: {result.total_tokens}, Tool calls: {result.tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
