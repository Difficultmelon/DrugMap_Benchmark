from __future__ import annotations

import json
import math
import re
from typing import Any


def _escape_invalid_json_backslashes(text: str) -> str:
    """Repair common model JSON with LaTeX-style backslashes in strings."""
    # Escape only invalid JSON escapes while preserving already-valid sequences.
    # LaTeX commands such as ``\beta`` begin with ``\b``, which is technically
    # a JSON escape for backspace. Track math spans so those commands are not
    # accidentally decoded as control characters.
    repaired: list[str] = []
    in_string = False
    in_math = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            repaired.append(char)
            in_string = not in_string
            index += 1
            continue
        if char != "\\" or not in_string:
            if char == "$" and in_string:
                in_math = not in_math
            repaired.append(char)
            index += 1
            continue

        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_math:
            if next_char == "\\":
                repaired.extend((char, next_char))
                index += 2
                continue
            repaired.extend((char, char))
            index += 1
            continue
        if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            repaired.extend((char, next_char))
            index += 2
            continue
        if (
            next_char == "u"
            and index + 5 < len(text)
            and all(candidate in "0123456789abcdefABCDEF" for candidate in text[index + 2 : index + 6])
        ):
            repaired.extend(text[index : index + 6])
            index += 6
            continue

        repaired.extend((char, char))
        index += 1
    return "".join(repaired)


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    repaired = _escape_invalid_json_backslashes(stripped)
    if repaired != stripped:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for candidate in (stripped, repaired):
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            return value
    return None


def parse_answer(text: str) -> dict[str, Any]:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict):
        return {
            "answer": parsed.get("answer", ""),
            "explanation": parsed.get("explanation", ""),
            "citations": parsed.get("citations", []),
            "raw_json": parsed,
            "parse_status": "json",
        }
    return {
        "answer": text.strip(),
        "explanation": "",
        "citations": [],
        "raw_json": None,
        "parse_status": "fallback_text",
    }


def parse_queries(text: str) -> list[str]:
    parsed = extract_json_object(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
        return [str(x).strip() for x in parsed["queries"] if str(x).strip()]
    return [line.strip(" -\t") for line in text.splitlines() if line.strip()][:5]


def parse_judge(text: str) -> dict[str, Any]:
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return {"parse_status": "error", "raw_text": text}

    answer_value = parsed.get("answer_correct", parsed.get("semantic_correct"))
    if isinstance(answer_value, str):
        normalized_answer = answer_value.strip().lower()
        if normalized_answer == "true":
            answer_value = True
        elif normalized_answer == "false":
            answer_value = False
        elif normalized_answer in {"null", "none", ""}:
            answer_value = None
        else:
            answer_value = None
    elif not isinstance(answer_value, bool):
        answer_value = None

    raw_scores = parsed.get("explanation_scores")
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    dimension_names = (
        "logical_coherence",
        "factual_support",
        "clinical_relevance",
        "conciseness_faithfulness",
    )
    explanation_scores: dict[str, int | None] = {}
    for name in dimension_names:
        value = raw_scores.get(name, parsed.get(name))
        if isinstance(value, bool):
            explanation_scores[name] = None
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            explanation_scores[name] = None
            continue
        explanation_scores[name] = (
            int(numeric) if math.isfinite(numeric) and numeric.is_integer() and 1 <= numeric <= 5 else None
        )

    applicable = parsed.get("explanation_applicable")
    if isinstance(applicable, str):
        applicable = applicable.strip().lower() == "true"
    if not isinstance(applicable, bool):
        applicable = None
    complete_scores = all(explanation_scores[name] is not None for name in dimension_names)
    explanation_score = (
        sum(int(explanation_scores[name]) for name in dimension_names) / len(dimension_names)
        if complete_scores
        else None
    )

    result = {
        "answer_correct": answer_value,
        # Keep the old field as a compatibility alias for existing reports.
        "semantic_correct": answer_value,
        "explanation_applicable": applicable,
        "explanation_scores": explanation_scores,
        "explanation_score": explanation_score,
        "confidence": parsed.get("confidence"),
        "error_type": parsed.get("error_type", ""),
        "rationale": parsed.get("rationale", ""),
        "parse_status": "json",
        "raw_json": parsed,
    }
    if isinstance(result["semantic_correct"], str):
        result["semantic_correct"] = result["semantic_correct"].strip().lower() == "true"
    if result["semantic_correct"] not in {True, False}:
        result["semantic_correct"] = None
    try:
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
    except (TypeError, ValueError):
        result["confidence"] = None
    return result
