"""
Base Agent Template
===================
Shared base class that all agent templates extend.
Provides common configuration and lifecycle hooks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class BaseAgentTemplate:
    """
    Base class for agent templates.
    Subclasses define:
      - default_name / description
      - recommended_model / provider
      - required_tools
      - default_goal_prompt
    """

    default_name: str = "base-agent"
    description: str = "A general-purpose AI agent."
    recommended_provider: str = "openai"
    recommended_model: str = "gpt-4o"
    required_tools: List[str] = []
    default_goal_prompt: str = ""

    @classmethod
    def to_config(cls, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a registration config dict for this template."""
        config: Dict[str, Any] = {
            "name": cls.default_name,
            "description": cls.description,
            "goal_prompt": cls.default_goal_prompt,
            "tools": cls.required_tools,
            "template": cls.default_name,
            "llm": {
                "provider": cls.recommended_provider,
                "model": cls.recommended_model,
            },
        }
        if overrides:
            config.update(overrides)
        return config

    @classmethod
    def to_yaml_dict(cls) -> Dict[str, Any]:
        """Return a dict suitable for writing to a YAML template file."""
        return {
            "name": cls.default_name,
            "description": cls.description,
            "goal_prompt": cls.default_goal_prompt,
            "tools": cls.required_tools,
            "template": cls.default_name,
            "llm": {
                "provider": cls.recommended_provider,
                "model": cls.recommended_model,
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "max_steps": 15,
            "max_retries": 3,
        }
