"""
Ollama LLM Provider
===================
Talk to locally-running Ollama models via its OpenAI-compatible API.
Default endpoint: http://localhost:11434/v1
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from llm.base import LLMProvider, LLMResponse, register_provider


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """Ollama provider using OpenAI-compatible API."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.base_url:
            self.base_url = "http://localhost:11434/v1"
        if not self.api_key:
            self.api_key = "ollama"  # Ollama doesn't need a real key

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._client()
        req: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        # Ollama may not support max_tokens on all models
        if self.max_tokens:
            req["max_tokens"] = self.max_tokens
        if tools:
            req["tools"] = tools

        resp = await client.chat.completions.create(**req)
        choice = resp.choices[0]
        content = choice.message.content or ""

        return LLMResponse(
            content=content,
            model=resp.model or self.model,
            provider="ollama",
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            raw=resp,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = self._client()
        req: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens:
            req["max_tokens"] = self.max_tokens

        stream = await client.chat.completions.create(**req)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
