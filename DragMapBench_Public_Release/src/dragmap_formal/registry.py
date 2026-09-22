from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from .client import MockClient, OpenAICompatibleClient
from .io import read_json


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    provider: str
    model: str
    base_url_env: str
    api_key_env: str
    enabled: bool
    temperature: float = 0.0
    max_output_tokens: int = 512
    token_parameter: str = "max_tokens"
    notes: str = ""
    api_mode: str = "chat_completions"
    web_search_mode: str = "external"
    search_options: dict[str, Any] = field(default_factory=dict)
    web_base_url_env: str = ""
    web_api_key_env: str = ""
    web_api_mode: str = ""
    web_search_options: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    send_response_format: bool = False


def _spec(row: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        display_name=str(row["display_name"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        base_url_env=str(row.get("base_url_env", "")),
        api_key_env=str(row.get("api_key_env", "")),
        enabled=bool(row.get("enabled", False)),
        temperature=float(row.get("temperature", 0.0)),
        max_output_tokens=int(row.get("max_output_tokens", 512)),
        token_parameter=str(row.get("token_parameter", "max_tokens")),
        notes=str(row.get("notes", "")),
        api_mode=str(row.get("api_mode", "chat_completions")),
        web_search_mode=str(row.get("web_search_mode", "external")),
        search_options=dict(row.get("search_options", {})),
        web_base_url_env=str(row.get("web_base_url_env", "")),
        web_api_key_env=str(row.get("web_api_key_env", "")),
        web_api_mode=str(row.get("web_api_mode", "")),
        web_search_options=dict(row.get("web_search_options", {})),
        extra_body=dict(row.get("extra_body", {})),
        send_response_format=bool(row.get("send_response_format", False)),
    )


def load_specs(path: str) -> tuple[ModelSpec, list[ModelSpec]]:
    data = read_json(path)
    return _spec(data["judge"]), [_spec(row) for row in data["test_models"]]


def find_test_model(path: str, display_name: str) -> ModelSpec:
    _, specs = load_specs(path)
    for spec in specs:
        if spec.display_name.casefold() == display_name.casefold():
            return spec
    raise KeyError(f"Model not found in registry: {display_name}")


def build_client(
    spec: ModelSpec,
    *,
    timeout_seconds: float,
    max_retries: int,
    mock: bool = False,
    for_web: bool = False,
):
    if mock:
        return MockClient()
    if spec.provider != "openai_compatible":
        raise ValueError(
            f"Provider {spec.provider!r} is not executable yet. "
            "Confirm an endpoint or local serving backend and update the registry."
        )
    base_url_env = spec.web_base_url_env if for_web and spec.web_base_url_env else spec.base_url_env
    api_key_env = spec.web_api_key_env if for_web and spec.web_api_key_env else spec.api_key_env
    base_url = os.environ.get(base_url_env, "")
    api_key = os.environ.get(api_key_env, "")
    api_mode = spec.web_api_mode if for_web and spec.web_api_mode else spec.api_mode
    search_options = spec.web_search_options if for_web and spec.web_search_options else spec.search_options
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=spec.model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        send_temperature=True,
        send_response_format=spec.send_response_format,
        token_parameter=spec.token_parameter,
        api_mode=api_mode,
        search_options=search_options,
        extra_body=spec.extra_body,
    )
