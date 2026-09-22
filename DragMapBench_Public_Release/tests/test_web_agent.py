from __future__ import annotations

import json

from dragmap_formal.client import ChatResult
from dragmap_formal.core import _run_open_web_chat
from dragmap_formal.prompts import WEB_SEARCH_TOOL
from dragmap_formal.registry import ModelSpec
from dragmap_formal.search import SearchProvider, SearchResult


class _FakeSearchProvider(SearchProvider):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        return [
            SearchResult(
                title="Zalcitabine",
                url="https://example.test/zalcitabine",
                snippet="Molecular weight 211.22 g/mol.",
            )
        ][:max_results]


class _AgentClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature, max_output_tokens, tools=None, tool_choice=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        if len(self.calls) == 1:
            raw_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps(
                                {"query": "zalcitabine molecular weight", "max_results": 3}
                            ),
                        },
                    }
                ],
            }
            return ChatResult(
                content="",
                raw_response={"choices": [{"message": raw_message}]},
                usage={},
                latency_seconds=0.01,
                request_id="model-1",
                message=raw_message,
                tool_calls=raw_message["tool_calls"],
            )

        content = json.dumps(
            {
                "answer": "211.22 g/mol",
                "explanation": "",
                "citations": ["https://example.test/zalcitabine"],
            }
        )
        raw_message = {"role": "assistant", "content": content}
        return ChatResult(
            content=content,
            raw_response={"choices": [{"message": raw_message}]},
            usage={},
            latency_seconds=0.01,
            request_id="model-2",
            message=raw_message,
            tool_calls=[],
        )


def test_model_controls_search_tool_loop() -> None:
    client = _AgentClient()
    provider = _FakeSearchProvider()
    model = ModelSpec(
        display_name="test",
        provider="openai_compatible",
        model="test-model",
        base_url_env="BASE",
        api_key_env="KEY",
        enabled=True,
    )

    response, trace = _run_open_web_chat(
        client=client,
        messages=[{"role": "system", "content": "test"}],
        model=model,
        provider=provider,
        web_config={"max_tool_turns": 3, "max_queries": 5, "max_pages": 10},
    )

    assert response.content.startswith('{"answer"')
    assert provider.queries == ["zalcitabine molecular weight"]
    assert trace["stopped_reason"] == "model_answered"
    assert trace["queries"][0]["results"][0]["url"] == "https://example.test/zalcitabine"
    assert client.calls[0]["tools"] == [WEB_SEARCH_TOOL]
    assert client.calls[1]["messages"][-1]["role"] == "tool"
