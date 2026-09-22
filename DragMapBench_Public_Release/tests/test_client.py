from __future__ import annotations

import pytest
import httpx

from dragmap_formal.client import OpenAICompatibleClient


class _FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "id": "test-response",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "The model used its full response budget.",
                    },
                    "finish_reason": "length",
                }
            ],
        }


class _FakeHttpClient:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __enter__(self) -> "_FakeHttpClient":
        return self

    def __exit__(self, *args) -> None:
        del args

    def post(self, *args, **kwargs) -> _FakeResponse:
        del args, kwargs
        return _FakeResponse()


class _TimeoutHttpClient:
    calls = 0

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __enter__(self) -> "_TimeoutHttpClient":
        return self

    def __exit__(self, *args) -> None:
        del args

    def post(self, *args, **kwargs) -> _FakeResponse:
        del args, kwargs
        type(self).calls += 1
        raise httpx.ReadTimeout("simulated timeout")


class _ToolCallResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "id": "tool-response",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"zalcitabine molecular weight","max_results":3}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }


class _ToolCallHttpClient:
    last_payload = None

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __enter__(self) -> "_ToolCallHttpClient":
        return self

    def __exit__(self, *args) -> None:
        del args

    def post(self, *args, **kwargs) -> _ToolCallResponse:
        del args
        type(self).last_payload = kwargs["json"]
        return _ToolCallResponse()


def test_empty_message_content_is_not_success(monkeypatch) -> None:
    monkeypatch.setattr("dragmap_formal.client.httpx.Client", _FakeHttpClient)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_retries=0,
    )

    with pytest.raises(RuntimeError, match="empty message content"):
        client.chat(
            [{"role": "user", "content": "Answer the question."}],
            temperature=0.0,
            max_output_tokens=512,
        )


def test_timeout_fails_one_item_without_retries(monkeypatch) -> None:
    _TimeoutHttpClient.calls = 0
    monkeypatch.setattr("dragmap_formal.client.httpx.Client", _TimeoutHttpClient)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_retries=5,
    )

    with pytest.raises(RuntimeError, match="simulated timeout"):
        client.chat(
            [{"role": "user", "content": "Answer the question."}],
            temperature=0.0,
            max_output_tokens=512,
        )

    assert _TimeoutHttpClient.calls == 1


def test_client_parses_tool_calls_and_sends_tool_schema(monkeypatch) -> None:
    monkeypatch.setattr("dragmap_formal.client.httpx.Client", _ToolCallHttpClient)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_retries=0,
    )

    result = client.chat(
        [{"role": "user", "content": "Search if needed."}],
        temperature=0.0,
        max_output_tokens=512,
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        tool_choice="auto",
    )

    assert result.content == ""
    assert result.tool_calls[0]["function"]["name"] == "web_search"
    assert _ToolCallHttpClient.last_payload["tools"][0]["function"]["name"] == "web_search"
    assert _ToolCallHttpClient.last_payload["tool_choice"] == "auto"


def test_response_format_is_sent_only_without_tools(monkeypatch) -> None:
    monkeypatch.setattr("dragmap_formal.client.httpx.Client", _ToolCallHttpClient)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_retries=0,
        send_response_format=True,
    )

    client.chat(
        [{"role": "user", "content": "Search if needed."}],
        temperature=0.0,
        max_output_tokens=512,
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        tool_choice="auto",
    )

    assert "response_format" not in _ToolCallHttpClient.last_payload

    client.chat(
        [{"role": "user", "content": "Answer as JSON."}],
        temperature=0.0,
        max_output_tokens=512,
    )

    assert _ToolCallHttpClient.last_payload["response_format"] == {"type": "json_object"}
