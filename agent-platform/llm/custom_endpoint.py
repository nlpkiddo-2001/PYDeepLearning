"""
Custom Endpoint LLM Provider
=============================
Connect to any OpenAI-compatible endpoint (vLLM, TGI, LiteLLM, etc.).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from llm.base import LLMProvider, LLMResponse, register_provider


@register_provider("custom")
class CustomEndpointProvider(LLMProvider):
    """Generic OpenAI-compatible endpoint provider."""

    def _client(self) -> AsyncOpenAI:
        if not self.base_url:
            raise ValueError("CustomEndpointProvider requires base_url to be set.")
        return AsyncOpenAI(
            api_key=self.api_key or "no-key",
            base_url=self.base_url,
        )

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
            "max_tokens": self.max_tokens,
        }
        if tools:
            req["tools"] = tools

        resp = await client.chat.completions.create(**req)
        choice = resp.choices[0]
        content = choice.message.content or ""

        return LLMResponse(
            content=content,
            model=resp.model or self.model,
            provider="custom",
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
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            req["tools"] = tools

        stream = await client.chat.completions.create(**req)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
