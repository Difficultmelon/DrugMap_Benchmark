from __future__ import annotations

import httpx

from dragmap_formal.search import BochaSearchProvider


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "Example result",
                            "url": "https://example.test/page",
                            "snippet": "Short snippet",
                            "summary": "Longer summary",
                        }
                    ]
                }
            }
        }


class _HttpClient:
    payload = None
    headers = None

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __enter__(self) -> "_HttpClient":
        return self

    def __exit__(self, *args) -> None:
        del args

    def post(self, url, *, json, headers):
        type(self).payload = (url, json)
        type(self).headers = headers
        return _Response()


class _ForbiddenResponse:
    status_code = 403
    text = "You do not have enough money or package quota"

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            "forbidden",
            request=httpx.Request("POST", "https://api.bocha.cn/v1/web-search"),
            response=httpx.Response(403),
        )


class _RotatingHttpClient:
    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __enter__(self) -> "_RotatingHttpClient":
        return self

    def __exit__(self, *args) -> None:
        del args

    def post(self, url, *, json, headers):
        del url, json
        key = headers["Authorization"].removeprefix("Bearer ")
        type(self).calls.append(key)
        if key == "empty-key":
            return _ForbiddenResponse()
        return _Response()


def test_bocha_search_adapter(monkeypatch) -> None:
    monkeypatch.setattr("dragmap_formal.search.httpx.Client", _HttpClient)
    provider = BochaSearchProvider(api_key="test-key")

    results = provider.search("aspirin mechanism", max_results=3)

    assert results[0].title == "Example result"
    assert results[0].url == "https://example.test/page"
    assert results[0].snippet == "Longer summary"
    assert _HttpClient.payload == (
        "https://api.bocha.cn/v1/web-search",
        {
            "query": "aspirin mechanism",
            "freshness": "noLimit",
            "summary": True,
            "count": 3,
        },
    )
    assert _HttpClient.headers["Authorization"] == "Bearer test-key"


def test_bocha_search_rotates_quota_exhausted_keys(monkeypatch) -> None:
    _RotatingHttpClient.calls = []
    monkeypatch.setattr("dragmap_formal.search.httpx.Client", _RotatingHttpClient)
    provider = BochaSearchProvider(
        api_key="empty-key",
        api_keys=["empty-key", "working-key"],
    )

    results = provider.search("zalcitabine molecular weight", max_results=1)

    assert results[0].title == "Example result"
    assert _RotatingHttpClient.calls == ["empty-key", "working-key"]
