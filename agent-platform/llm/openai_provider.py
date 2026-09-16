"""
OpenAI LLM Provider
===================
Supports GPT-4o, GPT-4-turbo, GPT-3.5-turbo, o1, etc.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from llm.base import LLMProvider, LLMResponse, register_provider


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """OpenAI chat completions provider."""

    def _client(self) -> AsyncOpenAI:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

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
            req["tool_choice"] = "auto"

        resp = await client.chat.completions.create(**req)
        choice = resp.choices[0]

        # Handle tool calls
        content = choice.message.content or ""
        if choice.message.tool_calls:
            # Return tool call info as structured content
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
            content = str(tool_calls) if not content else content

        return LLMResponse(
            content=content,
            model=resp.model,
            provider="openai",
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
