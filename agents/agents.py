import json

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import requests

def generate( text: str, max_tokens: int = 256) -> str:
    url = "http://localhost:9012/llm/text/api/qwen/text/v1/chat/completions"

    payload = json.dumps({
        "model": "crm-di-qwen_text_14b-it",
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ],
        "max_tokens": max_tokens
    })
    headers = {
        'e': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

    return response.json()['choices'][0]['message']['content']


@dataclass
class AgentState:
    global_context: Dict[str, Any] = field(default_factory=dict)
    agent_memory: Dict[str, List[Any]] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)

    def update_context(self, key: str, value: Any):
        if key in self.global_context:
            if isinstance(self.global_context[key], dict):
                self.global_context[key].update(value)
            elif isinstance(self.global_context[key], list):
                self.global_context[key].extend(value)
            else:
                self.global_context[key] = value
        else:
            self.global_context[key] = value

    def add_memory(self, agent_id: str, memory: Any):
        if agent_id not in self.agent_memory:
            self.agent_memory[agent_id] = []
        self.agent_memory[agent_id].append(memory)

    def log_error(self, error: Dict[str, Any]):
        error['timestamp'] = str(uuid.uuid4())
        self.error_log.append(error)

class StateGraph:
    def __init__(self):
        self.nodes: Dict[str, 'BaseAgent'] = {}
        self.edges: Dict[str, List[str]] = {}
        self.state = AgentState()

    def add_node(self, agent: 'BaseAgent'):
        """Add a node to the graph"""
        self.nodes[agent.id] = agent
        agent.state_graph = self

    def add_edge(self, from_agent: str, to_agents: List[str]):
        """Define communication paths between agents"""
        self.edges[from_agent] = to_agents

    def propagate_state(self, source_agent_id: str, state_update: Dict[str, Any]):
        """
        Propagate state updates to connected agents
        Implements graph-based state communication
        """
        for target_agent_id in self.edges.get(source_agent_id, []):
            target_agent = self.nodes.get(target_agent_id)
            if target_agent:
                target_agent.receive_state_update(state_update)

class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        tools: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.tools = tools or {}
        self.state_graph: Optional[StateGraph] = None

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method"""
        pass

    def receive_state_update(self, update: Dict[str, Any]):
        if self.state_graph:
            self.state_graph.state.update_context(f"{self.name}_received", update)

    def communicate(self, message: Dict[str, Any], target_agents: Optional[List[str]] = None):
        if self.state_graph:
            target_agents = target_agents or self.state_graph.edges.get(self.id, [])
            for agent_id in target_agents:
                target_agent = self.state_graph.nodes.get(agent_id)
                if target_agent:
                    target_agent.receive_state_update(message)


class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        """Format the prompt template with given variables"""
        return self.template.format(**kwargs)


class EnhancedResearchGenerationTool:
    def __init__(self):
        pass

    def execute(self, params):
        query = params.get("query", "")
        num_results = params.get("num_results", 3)

        if not query:
            return {"error": "Query is required"}

        outputs = self.generate_from_assistant(query, num_results)

        return {
            "query": query,
            "results": outputs,
            "total_results": len(outputs),
        }

    def generate_from_assistant(self, query, num_results):
        generation_prompt = f"""You are a highly knowledgeable research assistant. I will provide you with a specific topic, and your task is to generate a concise, research-oriented response of approximately 200 words. Your response should be well-structured, factual, and written in a formal academic tone. Start with a brief introduction to the topic, provide key insights, and conclude with the implications or importance of the subject. Cite key concepts, historical context, or notable findings where applicable. Do not include personal opinions or assumptions. Aim for clarity, depth, and precision. \n{query}"""
        output = generate(generation_prompt, max_tokens=125)

        results = [{
            "title": query,
            "snippet": output
        }]

        return results


class GoalDecompositionTool:
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        goal = params.get('goal', '')

        search_queries = [
            f"Comprehensive overview of {goal}",
            f"Detailed analysis of {goal}",
            f"Current trends in {goal}",
            f"Expert perspectives on {goal}"
        ]

        return {
            "original_goal": goal,
            "search_queries": search_queries
        }

class MainAgent(BaseAgent):
    def __init__(self, name="Main Goal-Setting Agent"):
        super().__init__(name, tools={
            "goal_decomposition": GoalDecompositionTool()
        })

    def _call_external_api(self, text: str) -> str:
        url = "http://localhost:9012/llm/text/api/qwen/text/v1/chat/completions"

        payload = json.dumps({
            "model": "crm-di-qwen_text_14b-it",
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ],
            "max_tokens": 4096
        })
        headers = {
            'e': 'application/json',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

        return response.json()['choices'][0]['message']['content']

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        goal_decomposition_prompt = PromptTemplate(
            template="""You are a strategic goal decomposition AI. 
Your objective is to break down the complex goal: "{goal}" 
into a maximum of 5 concise, sequential (How to Start and How to End) and targeted search queries. 

Goal Breakdown Rules:
1. Queries must be specific and actionable
2. Cover different aspects of the goal
3. Maximize information gathering potential
4. Use clear, precise language

Return a JSON array of strings representing search queries.

Example:
Goal: "Understand the impact of AI on modern healthcare"
Queries: [
    "AI applications in medical diagnosis",
    "Machine learning in healthcare predictive analytics",
    "Ethical considerations of AI in medical treatment",
    "Cost-effectiveness of AI healthcare solutions",
    "Future trends in AI-driven medical research"
    
]""",
            input_variables=["goal"]
        )

        goal = task.get('goal', '')

        decomposition_result = goal_decomposition_prompt.format(goal=goal)

        try:
            search_queries = self._call_external_api(decomposition_result)
            if "json" in search_queries:
                search_queries = search_queries.replace("```json","").replace("```","").strip()

            return {
                "original_goal": goal,
                "search_queries": json.loads(search_queries)
            }
        except Exception as e:
            return {
                "error": f"Failed to generate search queries: {str(e)}"
            }


class ResearchAgent(BaseAgent):
    def __init__(self, name="Advanced Research Agent"):
        super().__init__(name, tools={
            "llm_generate_content": EnhancedResearchGenerationTool()
        })

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        search_queries = task.get('search_queries', [])
        comprehensive_results = []

        for query in search_queries:
            search_result = self.tools['llm_generate_content'].execute({
                "query": query,
                "num_results": 3
            })
            comprehensive_results.append(search_result)

        return {
            "research_findings": comprehensive_results
        }

class CodeGenerationTool:
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get('prompt','')

        return {
            "code" : generate(code, max_tokens=4096)
        }

class CodeGenerationAgent(BaseAgent):
    def __init__(self, name="Intelligent Code Generation Agent"):
        super().__init__(name, tools={
            "code_generation": CodeGenerationTool()
        })

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        research_context = task.get('research_context', {})

        code_generation_prompt = PromptTemplate(
            template="""Based on the research context:
{research_context}

Generate a {language} code solution that addresses the following requirements:
- Purpose: {purpose}
- Complexity Level: {complexity}
- Key Constraints: {constraints}

Provide a clean, efficient, and well-documented implementation.""",
            input_variables=["research_context", "language", "purpose", "complexity", "constraints"]
        )

        code_prompt = code_generation_prompt.format(
            research_context=json.dumps(research_context),
            language=task.get('language', 'python'),
            purpose=task.get('purpose', 'Generic solution'),
            complexity=task.get('complexity', 'moderate'),
            constraints=task.get('constraints', 'None specified')
        )

        code_result = self.tools['code_generation'].execute({
            "prompt": code_prompt,
            "language": task.get('language', 'python')
        })

        return code_result


def main():
    goal = "I need a model for `Custom Named Entity Recognition` using Transformers."

    main_agent = MainAgent()
    research_agent = ResearchAgent()
    code_agent = CodeGenerationAgent()

    goal_decomposition = main_agent.execute({"goal": goal})

    research_results = research_agent.execute(goal_decomposition)

    code_generation_task = {
        "research_context": research_results,
        "language": "python",
        "purpose": goal,
        "complexity": "advanced",
        "constraints": "Use any python library to achieve the given purpose [torch, transformers, datasets, sklearn, numpy]"
    }
    code_result = code_agent.execute(code_generation_task)

    print("Goal Decomposition:", goal_decomposition)
    print("Research Findings:", research_results)
    print("Generated Code:", code_result)


if __name__ == "__main__":
    main()
