from __future__ import annotations

import json

from dragmap_formal.client import ChatResult, OpenAICompatibleClient
from dragmap_formal.core import _run_native_web_search_chat
from dragmap_formal.registry import ModelSpec


class _ResponsesResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "id": "resp-1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "search-1",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "zalcitabine molecular weight",
                        "sources": [{"type": "url", "url": "https://pubchem.ncbi.nlm.nih.gov/"}],
                    },
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "answer": "211.22 g/mol",
                                    "explanation": "",
                                    "citations": ["https://pubchem.ncbi.nlm.nih.gov/"],
                                }
                            ),
                        }
                    ],
                },
            ],
            "usage": {"total_tokens": 12},
        }


class _ResponsesHttpClient:
    last_payload = None

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __enter__(self) -> "_ResponsesHttpClient":
        return self

    def __exit__(self, *args) -> None:
        del args

    def post(self, *args, **kwargs) -> _ResponsesResponse:
        del args
        type(self).last_payload = kwargs["json"]
        return _ResponsesResponse()


def test_responses_client_parses_native_search_output(monkeypatch) -> None:
    monkeypatch.setattr("dragmap_formal.client.httpx.Client", _ResponsesHttpClient)
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="qwen3.8-max",
        api_mode="responses",
        search_options={"search_options": {"forced_search": True}},
        max_retries=0,
    )

    result = client.chat(
        [{"role": "system", "content": "Answer as JSON."}, {"role": "user", "content": "Search."}],
        temperature=0.0,
        max_output_tokens=2048,
        tools=[{"type": "web_search"}],
    )

    assert result.content.startswith('{"answer"')
    assert result.tool_calls == []
    assert _ResponsesHttpClient.last_payload["tools"] == [{"type": "web_search"}]
    assert _ResponsesHttpClient.last_payload["search_options"]["forced_search"] is True


def test_native_search_trace_requires_provider_search_call() -> None:
    model = ModelSpec(
        display_name="Qwen-3.8-max",
        provider="openai_compatible",
        model="qwen3.8-max",
        base_url_env="BASE",
        api_key_env="KEY",
        enabled=True,
        web_search_mode="native_responses",
    )
    raw = _ResponsesResponse().json()
    response = ChatResult(
        content=raw["output"][1]["content"][0]["text"],
        raw_response=raw,
        usage=raw["usage"],
        latency_seconds=0.1,
        request_id="resp-1",
    )

    class _NativeClient:
        def chat(self, messages, *, temperature, max_output_tokens, tools=None, tool_choice=None):
            del messages, temperature, max_output_tokens, tools, tool_choice
            return response

    _, trace = _run_native_web_search_chat(
        client=_NativeClient(),
        messages=[{"role": "user", "content": "Search."}],
        model=model,
    )

    assert trace["verified"] is True
    assert trace["search_call_count"] == 1
    assert trace["queries"] == ["zalcitabine molecular weight"]
