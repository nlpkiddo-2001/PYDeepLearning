"""
vLLM Provider
=============
Connect to vLLM-hosted models with JWT authentication.
Supports custom parameters like chat_template_kwargs.

Example config in agent.yaml:
    llm:
      provider: "vllm"
      model: "glm-5"
      base_url: "http://103.42.51.234:443/llm/text/api/glm/v1"
      jwt_secret: "eyJhbGciOiJI"
      jwt_algorithm: "HS256"
      jwt_expiry_minutes: 15
      temperature: 0.7
      max_tokens: 4096
      chat_template_kwargs:
        enable_thinking: false
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from llm.base import LLMProvider, LLMResponse, register_provider

logger = logging.getLogger("agentforge.llm.vllm")


def _generate_jwt(secret: str, algorithm: str = "HS256", expiry_minutes: int = 15) -> str:
    """Generate a JWT token for vLLM authentication."""
    try:
        import jwt as pyjwt
    except ImportError:
        raise ImportError(
            "PyJWT is required for vLLM JWT auth. Install it: pip install PyJWT"
        )

    now = datetime.now(tz=timezone.utc)
    payload = {
        "iat": now,
        "exp": now + timedelta(minutes=expiry_minutes),
    }
    return pyjwt.encode(payload, secret, algorithm=algorithm)


@register_provider("vllm")
class VLLMProvider(LLMProvider):
    """
    vLLM-hosted model provider with JWT authentication.

    Uses raw httpx instead of the openai SDK to support:
    - JWT token generation and auto-refresh
    - Custom request body fields (chat_template_kwargs, etc.)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        jwt_expiry_minutes: int = 15,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiry_minutes = jwt_expiry_minutes
        self.chat_template_kwargs = chat_template_kwargs or {}

        # Token caching
        self._cached_token: Optional[str] = None
        self._token_expiry: float = 0

    def _get_auth_token(self) -> str:
        """Get a valid JWT token, refreshing if needed."""
        now = time.time()
        # Refresh 60s before expiry
        if self._cached_token and now < (self._token_expiry - 60):
            return self._cached_token

        if self.jwt_secret:
            self._cached_token = _generate_jwt(
                self.jwt_secret,
                self.jwt_algorithm,
                self.jwt_expiry_minutes,
            )
            self._token_expiry = now + (self.jwt_expiry_minutes * 60)
            logger.debug("Generated new JWT token for vLLM")
            return self._cached_token

        # Fallback: use api_key as static bearer token
        return self.api_key or "no-key"

    def _build_headers(self) -> Dict[str, str]:
        token = self._get_auth_token()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    async def validate_token(self) -> Dict[str, Any]:
        """Validate connectivity and authentication against the vLLM server.

        Strategy:
        1. Try GET {base_url}/models (standard OpenAI-compatible listing).
        2. If /models returns 404 (common behind API gateways / proxies that
           only expose /chat/completions), fall back to a minimal POST to
           {base_url}/chat/completions with max_tokens=1 so we can still
           verify the server is reachable and auth is accepted.

        Returns:
            {"valid": True, "models": [...]} on success
            {"valid": False, "error": "..."} on failure
        """
        base = (self.base_url or "").rstrip("/")
        if not base:
            return {"valid": False, "error": "base_url is not set."}

        # Strip /chat/completions or /completions suffix to get the root /v1
        for suffix in ("/chat/completions", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        models_url = f"{base}/models"
        headers = self._build_headers()

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(models_url, headers=headers)

                if resp.status_code == 401:
                    return {
                        "valid": False,
                        "error": "Authentication failed (HTTP 401). "
                                 "Check jwt_secret or api_key.",
                    }
                if resp.status_code == 403:
                    return {
                        "valid": False,
                        "error": "Authorization denied (HTTP 403). "
                                 "Token is valid but lacks permission.",
                    }

                # /models not exposed (gateway/proxy) — fall back to a
                # lightweight chat completions probe.
                if resp.status_code == 404:
                    return await self._validate_via_chat_probe(client, base, headers)

                resp.raise_for_status()
                data = resp.json()
                model_ids = [
                    m.get("id", "") for m in data.get("data", [])
                ]
                return {"valid": True, "models": model_ids}

        except httpx.ConnectError as exc:
            return {
                "valid": False,
                "error": f"Cannot reach vLLM server at {models_url}: {exc}",
            }
        except httpx.TimeoutException:
            return {
                "valid": False,
                "error": f"Connection to {models_url} timed out (15s).",
            }
        except httpx.HTTPStatusError as exc:
            return {
                "valid": False,
                "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
            }
        except Exception as exc:
            return {
                "valid": False,
                "error": f"Unexpected error: {exc}",
            }

    async def _validate_via_chat_probe(
        self,
        client: httpx.AsyncClient,
        base: str,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Fall-back validation when /models is not available (404).

        Sends a minimal chat completion request (max_tokens=1) to confirm
        the server is reachable and the auth token is accepted.  We don't
        care about the reply content — any 2xx (or a 400 for bad schema)
        means both connectivity and authentication are fine.
        """
        chat_url = f"{base}/chat/completions"
        probe_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        logger.debug(
            "/models returned 404 — probing %s instead", chat_url,
        )
        try:
            probe = await client.post(chat_url, headers=headers, json=probe_body)

            if probe.status_code == 401:
                return {
                    "valid": False,
                    "error": "Authentication failed (HTTP 401). "
                             "Check jwt_secret or api_key.",
                }
            if probe.status_code == 403:
                return {
                    "valid": False,
                    "error": "Authorization denied (HTTP 403). "
                             "Token is valid but lacks permission.",
                }
            if probe.status_code == 404:
                return {
                    "valid": False,
                    "error": (
                        f"Both {base}/models and {chat_url} returned 404. "
                        f"Check that base_url is correct."
                    ),
                }

            # Any 2xx or even a 400 (bad request body) confirms the server
            # is up and auth succeeded.
            if 200 <= probe.status_code < 500:
                return {
                    "valid": True,
                    "models": [self.model],
                    "note": "/models not available; validated via chat completions probe.",
                }

            # Unexpected server error
            return {
                "valid": False,
                "error": f"Chat probe returned HTTP {probe.status_code}: "
                         f"{probe.text[:300]}",
            }

        except httpx.ConnectError as exc:
            return {
                "valid": False,
                "error": f"Cannot reach vLLM server at {chat_url}: {exc}",
            }
        except httpx.TimeoutException:
            return {
                "valid": False,
                "error": f"Connection to {chat_url} timed out (15s).",
            }
        except Exception as exc:
            return {
                "valid": False,
                "error": f"Chat probe unexpected error: {exc}",
            }

    def _build_request_body(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if tools:
            body["tools"] = tools
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        if stream:
            body["stream"] = True
        return body

    def _get_endpoint(self) -> str:
        base = (self.base_url or "").rstrip("/")
        if not base:
            raise ValueError("VLLMProvider requires base_url to be set.")
        # If base_url already ends with /chat/completions, use as-is
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        endpoint = self._get_endpoint()
        headers = self._build_headers()
        body = self._build_request_body(messages, tools, stream=False, **kwargs)

        logger.info("vLLM request → %s  model=%s", endpoint, self.model)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(endpoint, json=body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            logger.error("vLLM connection failed (%s): %s", endpoint, exc)
            raise ConnectionError(
                f"Cannot reach vLLM server at {endpoint}. "
                f"Is the server running? Error: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("vLLM request timed out (%s): %s", endpoint, exc)
            raise TimeoutError(
                f"vLLM request timed out after 60s ({endpoint}). "
                f"The model may be overloaded or unreachable."
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("vLLM HTTP error %s: %s", exc.response.status_code, exc.response.text[:500])
            raise ValueError(
                f"vLLM returned HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"vLLM returned no choices: {data}")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Handle tool calls if present
        tool_calls = message.get("tool_calls")
        if tool_calls and not content:
            content = str(tool_calls)

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            provider="vllm",
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            raw=data,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        endpoint = self._get_endpoint()
        headers = self._build_headers()
        body = self._build_request_body(messages, tools, stream=True, **kwargs)

        logger.debug("vLLM stream request → %s  model=%s", endpoint, self.model)

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", endpoint, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    # SSE format: data: {...}
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except Exception:
                        continue
