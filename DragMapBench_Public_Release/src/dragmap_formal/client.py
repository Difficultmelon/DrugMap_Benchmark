from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ChatResult:
    content: str
    raw_response: dict[str, Any]
    usage: dict[str, Any]
    latency_seconds: float
    request_id: str | None = None
    message: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs, vLLM, TGI gateways, and local servers."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 5,
        send_temperature: bool = True,
        send_response_format: bool = False,
        token_parameter: str = "max_tokens",
        api_mode: str = "chat_completions",
        search_options: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not base_url:
            raise ValueError(f"Missing base URL for model {model}")
        if not api_key:
            raise ValueError(f"Missing API key for model {model}")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.send_temperature = send_temperature
        self.send_response_format = send_response_format
        self.token_parameter = token_parameter
        self.api_mode = api_mode
        self.search_options = dict(search_options or {})
        self.extra_body = dict(extra_body or {})

    @property
    def endpoint(self) -> str:
        if self.api_mode == "responses":
            if self.base_url.endswith("/responses"):
                return self.base_url
            return f"{self.base_url}/responses"
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _content_from_message(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("text") is not None:
                    pieces.append(str(part["text"]))
                elif part is not None:
                    pieces.append(str(part))
            return "".join(pieces)
        return str(content)

    @staticmethod
    def _tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            return [call for call in tool_calls if isinstance(call, dict)]

        # Some older OpenAI-compatible gateways still emit the legacy
        # function_call field. Normalize it to the tool_calls shape used by
        # the agent loop.
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            return [
                {
                    "id": "legacy_tool_call_0",
                    "type": "function",
                    "function": function_call,
                }
            ]
        return []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult:
        if self.api_mode == "responses":
            return self._responses_chat(
                messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=tools,
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.token_parameter: max_output_tokens,
        }
        if self.send_temperature:
            payload["temperature"] = temperature
        # DeepSeek JSON Output constrains final answer requests. Tool-call
        # requests must remain unconstrained so the model can emit function
        # calls for the external search loop.
        if self.send_response_format and not tools:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        payload.update(self.extra_body)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(self.endpoint, json=payload, headers=headers)
                    if response.status_code in {429, 500, 502, 503, 504}:
                        retry_after = response.headers.get("retry-after")
                        delay = float(retry_after) if retry_after else min(60.0, 2.0**attempt)
                        if attempt < self.max_retries:
                            time.sleep(delay)
                            continue
                    response.raise_for_status()
                    raw = response.json()
                    if not isinstance(raw, dict):
                        raise RuntimeError(f"Unexpected non-object response: {raw!r}")
                    message = raw["choices"][0]["message"]
                    if not isinstance(message, dict):
                        raise RuntimeError(f"Unexpected assistant message: {message!r}")
                    content = self._content_from_message(message)
                    tool_calls = self._tool_calls_from_message(message)
                    if not content.strip() and not tool_calls:
                        raise RuntimeError(
                            "Model returned empty message content; "
                            "the response may have been truncated before producing an answer"
                        )
                    return ChatResult(
                        content=content,
                        raw_response=raw,
                        usage=raw.get("usage") or {},
                        latency_seconds=time.perf_counter() - started,
                        request_id=response.headers.get("x-request-id") or raw.get("id"),
                        message=message,
                        tool_calls=tool_calls,
                    )
                except httpx.TimeoutException as exc:
                    # Timeouts are item-level failures. Move on immediately so
                    # a slow provider cannot stall the entire benchmark.
                    last_error = exc
                    break
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                    time.sleep(min(60.0, 2.0**attempt))
        raise RuntimeError(f"Model request failed after retries: {last_error}") from last_error

    def _responses_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        del temperature
        input_messages = [
            message
            for message in messages
            if message.get("role") in {"system", "user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "max_output_tokens": max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload.update(self.search_options)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(self.endpoint, json=payload, headers=headers)
                    if response.status_code in {429, 500, 502, 503, 504}:
                        retry_after = response.headers.get("retry-after")
                        delay = float(retry_after) if retry_after else min(60.0, 2.0**attempt)
                        if attempt < self.max_retries:
                            time.sleep(delay)
                            continue
                    response.raise_for_status()
                    raw = response.json()
                    if not isinstance(raw, dict):
                        raise RuntimeError(f"Unexpected non-object response: {raw!r}")
                    content, tool_calls = self._responses_content_and_tool_calls(raw)
                    if not content.strip() and not tool_calls:
                        raise RuntimeError(
                            "Model returned empty Responses output; "
                            "the response may have been truncated before producing an answer"
                        )
                    return ChatResult(
                        content=content,
                        raw_response=raw,
                        usage=raw.get("usage") or {},
                        latency_seconds=time.perf_counter() - started,
                        request_id=response.headers.get("x-request-id") or raw.get("id"),
                        message={"role": "assistant", "content": content},
                        tool_calls=tool_calls,
                    )
                except httpx.TimeoutException as exc:
                    last_error = exc
                    break
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break
                    time.sleep(min(60.0, 2.0**attempt))
        raise RuntimeError(f"Model request failed after retries: {last_error}") from last_error

    @classmethod
    def _responses_content_and_tool_calls(
        cls,
        raw: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        pieces: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in raw.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("text") is not None:
                        pieces.append(str(part["text"]))
            elif item_type == "function_call":
                tool_calls.append(
                    {
                        "id": item.get("id") or f"responses_search_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "",
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )
            # Native Responses web-search calls have already been executed by
            # the provider. Keep them in raw_response for audit, but do not
            # expose them as client-side function calls.
        return "".join(pieces), tool_calls


class MockClient:
    def __init__(self, answer: str = "A") -> None:
        self.answer = answer

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult:
        del messages, temperature, max_output_tokens, tools, tool_choice
        content = f'{{"answer": "{self.answer}", "explanation": "mock"}}'
        return ChatResult(
            content=content,
            raw_response={"mock": True},
            usage={},
            latency_seconds=0.0,
            request_id="mock",
            message={"role": "assistant", "content": content},
            tool_calls=[],
        )
