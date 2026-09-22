from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .normalize import as_text


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider:
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class DisabledSearchProvider(SearchProvider):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        del query, max_results
        return []


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": max_results},
            )
            response.raise_for_status()
            raw = response.json()
        return [
            SearchResult(
                title=as_text(item.get("title")),
                url=as_text(item.get("url")),
                snippet=as_text(item.get("content") or item.get("snippet")),
            )
            for item in raw.get("results", [])
            if isinstance(item, dict)
        ][:max_results]


class SerpApiSearchProvider(SearchProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                "https://serpapi.com/search.json",
                params={"engine": "google", "q": query, "api_key": self.api_key, "num": max_results},
            )
            response.raise_for_status()
            raw = response.json()
        rows = raw.get("organic_results", [])
        return [
            SearchResult(
                title=as_text(item.get("title")),
                url=as_text(item.get("link")),
                snippet=as_text(item.get("snippet")),
            )
            for item in rows
            if isinstance(item, dict)
        ][:max_results]


class BochaSearchProvider(SearchProvider):
    """Bocha Web Search API adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        api_keys: list[str] | None = None,
        base_url: str = "https://api.bocha.cn/v1/web-search",
    ) -> None:
        candidates = [str(key).strip() for key in (api_keys or [api_key]) if str(key).strip()]
        if not candidates:
            raise ValueError("At least one Bocha API key is required")
        self.api_keys = list(dict.fromkeys(candidates))
        self._active_key_index = 0
        self._lock = threading.Lock()
        self.base_url = base_url

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        payload = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": max(1, min(50, max_results)),
        }
        with self._lock:
            start_index = self._active_key_index
        last_error: Exception | None = None
        raw: dict[str, Any] | None = None
        for offset in range(len(self.api_keys)):
            key_index = (start_index + offset) % len(self.api_keys)
            headers = {
                "Authorization": f"Bearer {self.api_keys[key_index]}",
                "Content-Type": "application/json",
            }
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(self.base_url, json=payload, headers=headers)
                    status_code = getattr(response, "status_code", 200)
                    if status_code in {401, 402, 403, 429}:
                        body = response.text.casefold()
                        quota_error = any(
                            marker in body
                            for marker in (
                                "quota",
                                "balance",
                                "enough money",
                                "package quota",
                                "insufficient",
                                "too many",
                            )
                        )
                        if status_code != 403 or quota_error or offset < len(self.api_keys) - 1:
                            raise _BochaKeyRotationError(
                                f"Bocha key rejected ({status_code}): {response.text[:240]}"
                            )
                    response.raise_for_status()
                    candidate = response.json()
                    if not isinstance(candidate, dict):
                        raise ValueError("Bocha response is not a JSON object")
                    raw = candidate
            except _BochaKeyRotationError as exc:
                last_error = exc
                continue
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                break
            with self._lock:
                self._active_key_index = key_index
            break
        else:
            raise RuntimeError(f"All configured Bocha API keys failed: {last_error}") from last_error
        if raw is None:
            raise RuntimeError(f"Bocha search failed: {last_error}") from last_error

        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        web_pages = data.get("webPages", {}) if isinstance(data, dict) else {}
        items = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        if not isinstance(items, list):
            return []
        return [
            SearchResult(
                title=as_text(item.get("name") or item.get("title")),
                url=as_text(item.get("url") or item.get("link")),
                snippet=as_text(
                    item.get("summary")
                    or item.get("snippet")
                    or item.get("description")
                ),
            )
            for item in items[:max_results]
            if isinstance(item, dict)
        ]


class _BochaKeyRotationError(RuntimeError):
    """Internal marker for quota or authorization failures that should rotate keys."""


class GenericGetSearchProvider(SearchProvider):
    """Simple adapter for a private search gateway returning JSON results."""

    def __init__(self, *, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=30.0) as client:
            response = client.get(self.base_url, params={"q": query, "num": max_results}, headers=headers)
            response.raise_for_status()
            raw = response.json()
        items = raw.get("results") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return []
        out: list[SearchResult] = []
        for item in items[:max_results]:
            if not isinstance(item, dict):
                continue
            out.append(
                SearchResult(
                    title=as_text(item.get("title")),
                    url=as_text(item.get("url") or item.get("link")),
                    snippet=as_text(item.get("snippet") or item.get("content")),
                )
            )
        return out


def build_search_provider() -> SearchProvider:
    provider = os.environ.get("DRUGMAP_SEARCH_PROVIDER", "").strip().casefold()
    api_key = os.environ.get("DRUGMAP_SEARCH_API_KEY", "")
    configured_keys = [
        api_key,
        os.environ.get("DRUGMAP_SEARCH_API_KEY_2", ""),
        os.environ.get("DRUGMAP_SEARCH_API_KEY_3", ""),
        *os.environ.get("DRUGMAP_SEARCH_API_KEYS", "").replace("\n", ",").split(","),
    ]
    configured_keys = list(dict.fromkeys(key.strip() for key in configured_keys if key.strip()))
    base_url = os.environ.get("DRUGMAP_SEARCH_BASE_URL", "")
    if provider == "tavily" and api_key:
        return TavilySearchProvider(api_key)
    if provider == "serpapi" and api_key:
        return SerpApiSearchProvider(api_key)
    if provider == "bocha" and configured_keys:
        return BochaSearchProvider(
            api_key=configured_keys[0],
            api_keys=configured_keys,
            base_url=base_url or "https://api.bocha.cn/v1/web-search",
        )
    if provider == "generic" and base_url:
        return GenericGetSearchProvider(base_url=base_url, api_key=api_key or None)
    return DisabledSearchProvider()


def default_query(record: dict[str, Any]) -> str:
    question = as_text(record.get("question"))
    options = " ".join(as_text(x.get("text")) for x in record.get("options", []) if isinstance(x, dict))
    return " ".join([question, options, "pharmacology drug database"]).strip()


def collect_search_context(
    record: dict[str, Any],
    *,
    provider: SearchProvider,
    max_queries: int = 1,
    max_pages: int = 5,
) -> dict[str, Any]:
    del max_queries
    query = default_query(record)
    started = time.perf_counter()
    try:
        results = provider.search(query, max_results=max_pages)
        error = None
    except Exception as exc:  # noqa: BLE001 - formal logs should retain provider failure.
        results = []
        error = repr(exc)
    return {
        "provider": provider.__class__.__name__,
        "query": query,
        "results": [result.__dict__ for result in results],
        "error": error,
        "latency_seconds": time.perf_counter() - started,
    }


def collect_queries_context(
    queries: list[str],
    *,
    provider: SearchProvider,
    max_pages: int = 10,
) -> dict[str, Any]:
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    total_results = 0
    for query in queries:
        if total_results >= max_pages:
            break
        remaining = max_pages - total_results
        try:
            results = provider.search(query, max_results=max(1, remaining))
            error = None
        except Exception as exc:  # noqa: BLE001 - search failures are part of the trace.
            results = []
            error = repr(exc)
        total_results += len(results)
        traces.append(
            {
                "query": query,
                "results": [result.__dict__ for result in results],
                "error": error,
            }
        )
    return {
        "provider": provider.__class__.__name__,
        "queries": traces,
        "results": [item for trace in traces for item in trace["results"]],
        "latency_seconds": time.perf_counter() - started,
    }
