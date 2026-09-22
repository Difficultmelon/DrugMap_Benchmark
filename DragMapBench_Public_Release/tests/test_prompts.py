from __future__ import annotations

import json

from dragmap_formal.client import ChatResult
from dragmap_formal.core import _record_from_success
from dragmap_formal.prompts import WEB_SEARCH_TOOL, build_core_messages
from dragmap_formal.registry import ModelSpec


def _payload(record: dict) -> dict:
    return json.loads(build_core_messages(record, condition="closed_book")[1]["content"])


def test_reasoning_heavy_types_request_detailed_explanations() -> None:
    for item_type in ("drug_repositioning", "metabolism_clinical_reasoning"):
        payload = _payload(
            {
                "id": "Q1",
                "type": item_type,
                "subtype": "example",
                "task_format": "multiple_choice",
                "question": "Which option is correct?",
                "options": [],
            }
        )
        instruction = payload["explanation_instruction"]
        assert "3-5 concise sentences" in instruction
        assert "checkable" in instruction
        assert "hidden chain-of-thought" in instruction


def test_other_types_require_empty_explanation() -> None:
    payload = _payload(
        {
            "id": "Q2",
            "type": "physicochemical_property",
            "subtype": "example",
            "task_format": "short_answer",
            "question": "What is the answer?",
        }
    )
    assert "empty string" in payload["explanation_instruction"]
    assert '""' in payload["explanation_instruction"]
    assert "3-5 concise sentences" not in payload["explanation_instruction"]


def test_open_web_prompt_exposes_autonomous_search_instruction() -> None:
    payload = json.loads(
        build_core_messages(
            {
                "id": "Q3",
                "type": "unit",
                "task_format": "short_answer",
                "question": "What is the answer?",
            },
            condition="open_web",
        )[1]["content"]
    )
    assert "web_search tool" in payload["open_web_instruction"]
    assert "decide autonomously" in payload["open_web_instruction"].casefold()
    assert WEB_SEARCH_TOOL["function"]["name"] == "web_search"


def test_result_storage_suppresses_unsolicited_short_explanations() -> None:
    model = ModelSpec(
        display_name="test",
        provider="openai_compatible",
        model="test-model",
        base_url_env="BASE",
        api_key_env="KEY",
        enabled=True,
    )
    response = ChatResult(
        content=json.dumps({"answer": "A", "explanation": "unrequested rationale"}),
        raw_response={},
        usage={},
        latency_seconds=0.0,
        request_id="request-1",
    )
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]

    ordinary = _record_from_success(
        run_id="run",
        model=model,
        condition="closed_book",
        item_id="ordinary",
        item_type="physicochemical_property",
        messages=messages,
        response=response,
    )
    detailed = _record_from_success(
        run_id="run",
        model=model,
        condition="closed_book",
        item_id="detailed",
        item_type="drug_repositioning",
        messages=messages,
        response=response,
    )

    assert ordinary["parsed_explanation"] == ""
    assert detailed["parsed_explanation"] == "unrequested rationale"
