"""
Anthropic LLM Provider
======================
Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, etc.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic

from llm.base import LLMProvider, LLMResponse, register_provider


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def _client(self) -> AsyncAnthropic:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncAnthropic(**kwargs)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._client()

        # Anthropic requires system message to be separate
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                chat_messages.append(m)

        req: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_msg.strip():
            req["system"] = system_msg.strip()

        if tools:
            # Convert OpenAI-style tool schemas to Anthropic format
            anthropic_tools = []
            for t in tools:
                func = t.get("function", t)
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            req["tools"] = anthropic_tools

        resp = await client.messages.create(**req)

        content_parts = []
        for block in resp.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                content_parts.append(
                    f'{{"tool": "{block.name}", "input": {block.input}}}'
                )

        return LLMResponse(
            content="\n".join(content_parts),
            model=resp.model,
            provider="anthropic",
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
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

        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                chat_messages.append(m)

        req: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_msg.strip():
            req["system"] = system_msg.strip()

        async with client.messages.stream(**req) as stream:
            async for text in stream.text_stream:
                yield text
