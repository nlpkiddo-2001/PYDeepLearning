"""
LLM Base Provider
=================
Abstract interface all LLM providers implement.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    raw: Any = None  # original provider response


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str = "base"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and return a normalized response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens one chunk at a time."""
        ...

    def update_config(self, **kwargs: Any) -> None:
        """Dynamically update provider settings."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)


# ─── Factory ─────────────────────────────────────────────────────────

_PROVIDER_REGISTRY: Dict[str, type] = {}


def register_provider(name: str):
    """Class decorator to register an LLM provider."""
    def wrapper(cls):
        _PROVIDER_REGISTRY[name] = cls
        cls.provider_name = name
        return cls
    return wrapper


def create_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Create an LLM provider from a config dict.

    Expected keys: provider, model, api_key (or env var), base_url, temperature, max_tokens
    """
    provider_name = config.get("provider", "openai")
    cls = _PROVIDER_REGISTRY.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available: {list(_PROVIDER_REGISTRY.keys())}"
        )

    api_key = config.get("api_key", "")
    # Resolve env var references like ${OPENAI_API_KEY}
    if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.getenv(env_var, "")

    # Base kwargs that all providers accept
    base_kwargs: Dict[str, Any] = {
        "model": config.get("model", "gpt-4o"),
        "api_key": api_key,
        "base_url": config.get("base_url"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 4096),
    }

    # Pass provider-specific kwargs (vLLM JWT, chat_template_kwargs, etc.)
    extra_keys = {
        "jwt_secret", "jwt_algorithm", "jwt_expiry_minutes",
        "chat_template_kwargs",
    }
    for key in extra_keys:
        if key in config:
            base_kwargs[key] = config[key]

    return cls(**base_kwargs)
