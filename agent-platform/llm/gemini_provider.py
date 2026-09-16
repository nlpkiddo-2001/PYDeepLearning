"""
Google Gemini LLM Provider
===========================
Supports Gemini 3 Flash Preview, Gemini 2.5 Pro, etc.
Uses the official google-genai SDK.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from google import genai
from google.genai import types

from llm.base import LLMProvider, LLMResponse, register_provider


@register_provider("gemini")
class GeminiProvider(LLMProvider):
    """Google Gemini provider via google-genai SDK."""

    def _client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _messages_to_contents(
        messages: List[Dict[str, str]],
    ) -> tuple[Optional[str], list[types.Content]]:
        """
        Convert OpenAI-style messages into a Gemini system instruction
        and a list of Content objects.
        """
        system_instruction: Optional[str] = None
        contents: list[types.Content] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_instruction = (system_instruction or "") + text + "\n"
            else:
                # Gemini uses "user" and "model" roles
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=text)],
                    )
                )

        if system_instruction:
            system_instruction = system_instruction.strip()

        return system_instruction, contents

    @staticmethod
    def _convert_tools(
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[list[types.Tool]]:
        """Convert OpenAI-style tool schemas to Gemini function declarations."""
        if not tools:
            return None

        declarations = []
        for t in tools:
            func = t.get("function", t)
            declarations.append(
                types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                )
            )
        return [types.Tool(function_declarations=declarations)]

    # ── core interface ────────────────────────────────────────────

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._client()
        system_instruction, contents = self._messages_to_contents(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        gemini_tools = self._convert_tools(tools)
        if gemini_tools:
            config.tools = gemini_tools

        resp = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Extract text content
        content = resp.text or ""

        # Handle function calls returned by the model
        if resp.candidates and resp.candidates[0].content.parts:
            fn_calls = []
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    fn_calls.append({
                        "name": part.function_call.name,
                        "arguments": json.dumps(
                            dict(part.function_call.args) if part.function_call.args else {}
                        ),
                    })
            if fn_calls and not content:
                content = str(fn_calls)

        usage = {}
        if resp.usage_metadata:
            usage = {
                "prompt_tokens": resp.usage_metadata.prompt_token_count or 0,
                "completion_tokens": resp.usage_metadata.candidates_token_count or 0,
            }

        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            usage=usage,
            raw=resp,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = self._client()
        system_instruction, contents = self._messages_to_contents(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        gemini_tools = self._convert_tools(tools)
        if gemini_tools:
            config.tools = gemini_tools

        for chunk in client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text
