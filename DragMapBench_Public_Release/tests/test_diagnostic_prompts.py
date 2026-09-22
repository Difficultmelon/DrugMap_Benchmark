from __future__ import annotations

import json

from dragmap_formal.prompts import build_diagnostic_messages


def test_diagnostic_prompt_requires_minimal_json_answer() -> None:
    payload = json.loads(
        build_diagnostic_messages(
            item_id="CB_MCR_001_A1",
            question="Which enzyme is identified as a primary metabolizing enzyme?",
            answer_instruction="Return only the standardized enzyme name.",
        )[1]["content"]
    )

    contract = payload["diagnostic_output_contract"]
    assert "exactly two keys" in contract
    assert '"explanation":""' in contract
    assert "Markdown" in contract
    assert "only the requested answer" in contract
