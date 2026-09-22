from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .normalize import as_text, decimal_from, normalized, parse_boolean, parse_choice_label


SCORING_POLICY_VERSION = "v5.2-precision-tolerance-3-interval-subset"
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_INTERVAL_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL_RANGE_RE = re.compile(
    rf"^\s*(?P<lower_cmp>>=|<=|>|<|≥|≤)?\s*(?P<lower>{_INTERVAL_NUMBER})"
    rf"\s*(?:to|-)\s*"
    rf"(?P<upper_cmp>>=|<=|>|<|≥|≤)?\s*(?P<upper>{_INTERVAL_NUMBER})\s*$",
    flags=re.IGNORECASE,
)
_INTERVAL_BETWEEN_RE = re.compile(
    rf"^\s*between\s+(?P<lower>{_INTERVAL_NUMBER})\s+and\s+"
    rf"(?P<upper_cmp><=|<|≤)?\s*(?P<upper>{_INTERVAL_NUMBER})\s*$",
    flags=re.IGNORECASE,
)
_INTERVAL_ONE_SIDED_RE = re.compile(
    rf"^\s*(?P<cmp>>=|<=|>|<|≥|≤)\s*(?P<value>{_INTERVAL_NUMBER})\s*$",
    flags=re.IGNORECASE,
)
_INTERVAL_UNIT_RE = re.compile(
    r"(?P<unit>kg\s*/\s*mol|mg\s*/\s*mol|g\s*/\s*mol|"
    r"kg\s*mol(?:\s*[-−]?\s*1)?|g\s*mol(?:\s*[-−]?\s*1)?|"
    r"mg\s*mol(?:\s*[-−]?\s*1)?|daltons?|da|amu|u)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class _NumericInterval:
    lower: Decimal | None
    upper: Decimal | None
    lower_inclusive: bool
    upper_inclusive: bool
    # A bare `300-350`/`300 to 350` is treated as a category boundary range.
    # This lets it fit under a gold upper boundary written as `<350`.
    upper_is_boundary: bool
    unit_family: str | None
    unit_scale: Decimal


def _normalize_interval_text(value: Any) -> str:
    text = normalized(value)
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .strip()
    )


def _interval_unit(text: str) -> tuple[str, Decimal, str]:
    match = _INTERVAL_UNIT_RE.search(text)
    if not match:
        return text, Decimal("1"), ""
    raw = re.sub(r"\s+", "", match.group("unit").casefold())
    if raw in {"da", "dalton", "daltons", "amu", "u"}:
        family, scale = "molar_mass", Decimal("1")
    elif raw in {"kg/mol", "kgmol-1", "kgmol1"}:
        family, scale = "molar_mass", Decimal("1000")
    elif raw in {"mg/mol", "mgmol-1", "mgmol1"}:
        family, scale = "molar_mass", Decimal("0.001")
    else:
        family, scale = "molar_mass", Decimal("1")
    return text[: match.start()].strip(), scale, family


def _interval_decimal(value: str) -> Decimal | None:
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _parse_numeric_interval(value: Any) -> _NumericInterval | None:
    """Parse a bounded or one-sided numeric answer interval.

    This parser is intentionally limited to numeric ranges. It does not infer
    intervals from arbitrary prose or from entity names containing digits.
    """
    text = _normalize_interval_text(value).rstrip(".。;；")
    body, unit_scale, unit_family = _interval_unit(text)

    range_match = _INTERVAL_RANGE_RE.fullmatch(body) or _INTERVAL_BETWEEN_RE.fullmatch(body)
    if range_match:
        lower_raw = range_match.group("lower")
        upper_raw = range_match.group("upper")
        lower = _interval_decimal(lower_raw)
        upper = _interval_decimal(upper_raw)
        if lower is None or upper is None:
            return None
        lower_cmp = range_match.groupdict().get("lower_cmp")
        upper_cmp = range_match.groupdict().get("upper_cmp")
        # A bare hyphen/to endpoint is a category boundary rather than a
        # request to include the exact upper endpoint.
        upper_is_boundary = not upper_cmp
        lower_inclusive = lower_cmp not in {">", ">="} or lower_cmp == ">="
        if lower_cmp == ">":
            lower_inclusive = False
        elif lower_cmp in {">=", None}:
            lower_inclusive = True
        upper_inclusive = upper_cmp in {"<=", ">="}
        if upper_cmp is None:
            upper_inclusive = False
        if upper_cmp == ">":
            return None
        if upper_cmp == ">=":
            upper_inclusive = True
        lower *= unit_scale
        upper *= unit_scale
        if lower > upper:
            return None
        return _NumericInterval(
            lower=lower,
            upper=upper,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
            upper_is_boundary=upper_is_boundary,
            unit_family=unit_family or None,
            unit_scale=unit_scale,
        )

    one_sided = _INTERVAL_ONE_SIDED_RE.fullmatch(body)
    if one_sided:
        number = _interval_decimal(one_sided.group("value"))
        if number is None:
            return None
        comparator = one_sided.group("cmp")
        number *= unit_scale
        if comparator in {"<", "<="}:
            return _NumericInterval(
                lower=None,
                upper=number,
                lower_inclusive=True,
                upper_inclusive=comparator == "<=",
                upper_is_boundary=False,
                unit_family=unit_family or None,
                unit_scale=unit_scale,
            )
        return _NumericInterval(
            lower=number,
            upper=None,
            lower_inclusive=comparator == ">=",
            upper_inclusive=True,
            upper_is_boundary=False,
            unit_family=unit_family or None,
            unit_scale=unit_scale,
        )
    return None


def _units_compatible(candidate: _NumericInterval, gold: _NumericInterval) -> bool:
    return (
        candidate.unit_family is None
        or gold.unit_family is None
        or candidate.unit_family == gold.unit_family
    )


def _candidate_interval_is_subset(
    candidate: _NumericInterval,
    gold: _NumericInterval,
) -> bool:
    if not _units_compatible(candidate, gold):
        return False

    if gold.lower is not None:
        if candidate.lower is None or candidate.lower < gold.lower:
            return False
        if candidate.lower == gold.lower and not gold.lower_inclusive and candidate.lower_inclusive:
            return False

    if gold.upper is not None:
        if candidate.upper is None or candidate.upper > gold.upper:
            return False
        if candidate.upper == gold.upper and not gold.upper_inclusive:
            if candidate.upper_inclusive and not candidate.upper_is_boundary:
                return False
    return True


def interval_subset_match(
    prediction: Any,
    accepted_answers: Iterable[Any],
) -> bool | None:
    """Return whether a numeric candidate interval fits a gold interval.

    `None` means that the answer is not a pair of numeric intervals and should
    be handled by the ordinary semantic/exact scoring path.
    """
    candidate = _parse_numeric_interval(prediction)
    if candidate is None:
        return None
    gold_intervals = [
        parsed
        for answer in accepted_answers
        if (parsed := _parse_numeric_interval(answer)) is not None
    ]
    if not gold_intervals:
        return None
    return any(_candidate_interval_is_subset(candidate, gold) for gold in gold_intervals)


def answer_from_result(result: dict[str, Any] | None) -> Any:
    if not result:
        return None
    if "parsed_answer" in result:
        return result.get("parsed_answer")
    return result.get("answer")


def explanation_from_result(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    return as_text(result.get("parsed_explanation", result.get("explanation", "")))


def _accepted(gold: dict[str, Any]) -> list[Any]:
    answer = gold.get("answer") or {}
    values = answer.get("accepted_answers") or []
    return list(values)


def _numeric_tolerance(gold: dict[str, Any]) -> Decimal | None:
    normalization = (gold.get("answer") or {}).get("normalization") or {}
    raw = normalization.get("numeric_tolerance")
    if raw is None:
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() and value >= 0 else None


def _numeric_match(prediction: Any, accepted: Iterable[Any], tolerance: Decimal) -> bool:
    predicted = _single_numeric_with_precision(prediction)
    if predicted is None:
        return False
    predicted_number, _ = predicted
    for value in accepted:
        gold = _single_numeric_with_precision(value)
        if gold is not None and abs(predicted_number - gold[0]) <= tolerance:
            return True
    return False


def _single_numeric_with_precision(value: Any) -> tuple[Decimal, int] | None:
    """Parse only scalar numeric answers, retaining the gold decimal precision."""
    text = as_text(value).strip()
    if not re.fullmatch(_NUMERIC_TOKEN_RE.pattern, text):
        return None
    if (
        re.search(r"[<>≥≤]", text)
        or re.search(r"\bto\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:or\s+more|or\s+less|at\s+least|at\s+most)\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:greater|less)\s+than\b", text, flags=re.IGNORECASE)
        or re.search(r"\d\s*\+", text)
    ):
        return None
    token = text
    try:
        number = Decimal(token)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    mantissa = re.split(r"[eE]", token, maxsplit=1)[0]
    decimal_places = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    return number, decimal_places


def _numeric_precision_match(prediction: Any, accepted: Iterable[Any]) -> str | None:
    """Apply half-unit-in-the-last-place tolerance for scalar decimal gold."""
    predicted = _single_numeric_with_precision(prediction)
    if predicted is None:
        return None
    predicted_number, _ = predicted
    for value in accepted:
        parsed_gold = _single_numeric_with_precision(value)
        if parsed_gold is None:
            continue
        gold_number, decimal_places = parsed_gold
        if decimal_places == 0:
            if predicted_number == gold_number:
                return "numeric_exact"
            continue
        half_unit = Decimal("0.5") * (Decimal(10) ** (-decimal_places))
        lower = gold_number - half_unit
        upper = gold_number + half_unit
        if lower <= predicted_number < upper:
            return f"numeric_precision_interval_{half_unit}"
    return None


def _choice_text_map(question: dict[str, Any]) -> dict[str, str]:
    return {
        str(option.get("label", "")).upper(): str(option.get("text", ""))
        for option in question.get("options", [])
        if isinstance(option, dict) and option.get("label") is not None
    }


def _choice_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    return [option for option in question.get("options", []) if isinstance(option, dict)]


def _option_semantic_key(option: dict[str, Any] | None) -> str:
    if not option:
        return ""
    text_norm = normalized(option.get("text"))
    if text_norm:
        return f"semantic:{text_norm}"
    option_id = as_text(option.get("option_id")).strip()
    if option_id:
        return f"option_id:{option_id}"
    label = as_text(option.get("label")).strip().upper()
    return f"option_label:{label}" if label else ""


def choice_prediction_details(question: dict[str, Any], prediction: Any) -> dict[str, Any]:
    """Map a multiple-choice prediction onto the current option layout.

    The explicit label, if present, is preserved separately from inferred labels
    so permutation scoring can trust the current visible A/B/C/D before falling
    back to semantic text matching.
    """
    options = _choice_options(question)
    explicit_label = parse_choice_label(prediction)
    selected_option: dict[str, Any] | None = None
    if explicit_label:
        selected_option = next(
            (option for option in options if as_text(option.get("label")).strip().upper() == explicit_label),
            None,
        )
    else:
        pred_norm = normalized(prediction)
        pred_compact = normalized(prediction, strip_punctuation=True)
        for option in options:
            text = option.get("text")
            if pred_norm and pred_norm == normalized(text):
                selected_option = option
                break
            if pred_compact and pred_compact == normalized(text, strip_punctuation=True):
                selected_option = option
                break

    selected_label = explicit_label
    if selected_option is not None:
        selected_label = as_text(selected_option.get("label")).strip().upper() or explicit_label

    semantic_key = _option_semantic_key(selected_option)
    if not semantic_key:
        pred_norm = normalized(prediction)
        semantic_key = f"semantic:{pred_norm}" if pred_norm else "prediction:"

    return {
        "explicit_label": explicit_label,
        "selected_label": selected_label,
        "selected_option_id": as_text(selected_option.get("option_id")).strip() if selected_option else "",
        "selected_option_text": as_text(selected_option.get("text")).strip() if selected_option else "",
        "semantic_key": semantic_key,
    }


def _choice_match(question: dict[str, Any], gold: dict[str, Any], prediction: Any) -> tuple[bool, str]:
    accepted = _accepted(gold)
    accepted_norm = {normalized(value, strip_punctuation=True) for value in accepted}
    pred_label = parse_choice_label(prediction)
    accepted_labels = {
        label
        for value in accepted
        for label in [parse_choice_label(value)]
        if label is not None
    }
    if pred_label and pred_label in accepted_labels:
        return True, "choice_label"
    pred_norm = normalized(prediction, strip_punctuation=True)
    if pred_norm in accepted_norm:
        return True, "choice_text_or_label"
    if pred_label:
        text = _choice_text_map(question).get(pred_label, "")
        if normalized(text, strip_punctuation=True) in accepted_norm:
            return True, "choice_label_to_text"
    return False, "multiple_choice_no_match"


def score_answer(
    question: dict[str, Any],
    gold: dict[str, Any],
    prediction: Any,
) -> tuple[bool, str]:
    """Score one V5.2 answer using only the release's declared gold rules."""
    method = str((gold.get("scoring") or {}).get("method", "normalized_exact"))
    if method == "multiple_choice":
        return _choice_match(question, gold, prediction)
    if method == "boolean_exact":
        predicted_bool = parse_boolean(prediction)
        gold_bool = parse_boolean((gold.get("answer") or {}).get("canonical"))
        if predicted_bool is not None and gold_bool is not None and predicted_bool == gold_bool:
            return True, "boolean_exact"
        return False, "boolean_no_match"
    tolerance = _numeric_tolerance(gold)
    accepted = _accepted(gold)
    pred_norm = normalized(prediction)
    accepted_norm = {normalized(value) for value in accepted}
    # Preserve the historical method for an exactly matching declared label;
    # interval_subset is reserved for a semantically narrower alternative.
    if pred_norm and pred_norm in accepted_norm:
        return True, "normalized_exact"
    interval_match = interval_subset_match(prediction, accepted)
    if interval_match is True:
        return True, "interval_subset"
    if interval_match is False:
        return False, "interval_not_subset"
    if tolerance is not None and _numeric_match(prediction, accepted, tolerance):
        return True, "numeric_tolerance"
    numeric_method = _numeric_precision_match(prediction, accepted)
    if numeric_method is not None:
        return True, numeric_method
    return False, "normalized_no_match"


def score_core_results(
    questions: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        item_id = str(question["id"])
        gold = gold_by_id[item_id]
        result = results_by_id.get(item_id)
        status = str(result.get("status")) if result else "missing"
        prediction = answer_from_result(result)
        if status == "ok":
            correct, method = score_answer(question, gold, prediction)
            scoring_status = "scored"
        else:
            correct, method = False, "not_submitted"
            scoring_status = "missing_or_error"
        rows.append(
            {
                "id": item_id,
                "type": question.get("type"),
                "subtype": question.get("subtype"),
                "task_format": question.get("task_format"),
                "knowledge_unit_id": (question.get("group_ids") or {}).get("knowledge_unit_id"),
                "model_status": status,
                "prediction": as_text(prediction),
                "prediction_normalized": normalized(prediction),
                "correct": bool(correct),
                "score": 1.0 if correct else 0.0,
                "match_method": method,
                "scoring_status": scoring_status,
                "has_explanation": bool(explanation_from_result(result)),
            }
        )
    return rows


def _group_accuracy(rows: Iterable[dict[str, Any]], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(float(row.get("score", 0.0)))
    return {name: sum(values) / len(values) for name, values in sorted(groups.items()) if values}


def aggregate_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    submitted = sum(row.get("model_status") == "ok" for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    by_subtype = _group_accuracy(rows, "subtype")
    by_type = _group_accuracy(rows, "type")
    by_format = _group_accuracy(rows, "task_format")
    macro_subtype = sum(by_subtype.values()) / len(by_subtype) if by_subtype else None
    macro_type = sum(by_type.values()) / len(by_type) if by_type else None
    return {
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "numeric_precision_policy": (
            "Scalar decimal gold answers use [gold - 0.5*10^-p, gold + 0.5*10^-p), "
            "where p is the number of gold decimal places; scalar integers require "
            "numeric equality. For numeric interval answers, a candidate interval "
            "is correct when it is a subset of the accepted gold interval; units "
            "such as Da and g/mol are treated as equivalent molar-mass units. "
            "Non-numeric category labels remain exact-label scored."
        ),
        "total_items": total,
        "submitted_items": submitted,
        "missing_or_error_items": total - submitted,
        "correct_items": correct,
        "accuracy": correct / total if total else None,
        "submitted_accuracy": (
            sum(bool(row.get("correct")) for row in rows if row.get("model_status") == "ok") / submitted
            if submitted
            else None
        ),
        "macro_type_accuracy": macro_type,
        "macro_subtype_accuracy": macro_subtype,
        "by_type": by_type,
        "by_subtype": by_subtype,
        "by_task_format": by_format,
        "match_method_counts": _counts(rows, "match_method"),
        "status_counts": _counts(rows, "model_status"),
    }


def merge_judge_scores(
    deterministic_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build final rows without discarding deterministic or Judge evidence.

    Judge answer decisions replace deterministic answers only for rows explicitly
    evaluated by Judge. The targeted semantic subtype is therefore Judge-scored,
    while other long-explanation answer correctness remains deterministic.
    """
    judge_by_id = {str(row.get("item_id")): row for row in judge_rows}
    merged: list[dict[str, Any]] = []
    for original in deterministic_rows:
        row = dict(original)
        item_id = str(row.get("id", ""))
        judge = judge_by_id.get(item_id)
        deterministic_correct = bool(original.get("correct"))
        final_correct = deterministic_correct
        final_source = "deterministic_binary_accuracy"
        judge_answer_correct = None
        judge_status = None
        explanation_score = None
        explanation_scores: dict[str, Any] = {}

        if judge is not None:
            judge_status = judge.get("status")
            judge_answer_evaluated = judge.get("answer_evaluated")
            if judge_answer_evaluated is None:
                item_type = str(row.get("type", ""))
                task_format = str(row.get("task_format", ""))
                judge_answer_evaluated = (
                    task_format in {"short_answer", "classification"}
                    and item_type not in {"drug_repositioning", "metabolism_clinical_reasoning"}
                )
            raw_judge_answer_correct = judge.get(
                "judge_answer_correct_raw",
                judge.get("answer_correct", judge.get("semantic_correct")),
            )
            judge_answer_correct = (
                judge.get("answer_correct", judge.get("semantic_correct"))
                if judge_answer_evaluated
                else None
            )
            deterministic_interval_method = original.get("match_method")
            if (
                judge_answer_evaluated is True
                and deterministic_interval_method in {"interval_subset", "interval_not_subset"}
            ):
                # Apply the deterministic interval rule even to legacy Judge
                # rows that predate the explicit interval audit fields.
                judge_answer_correct = deterministic_interval_method == "interval_subset"
            if (
                judge.get("status") == "ok"
                and judge_answer_evaluated is True
                and isinstance(judge_answer_correct, bool)
            ):
                final_correct = judge_answer_correct
                final_source = (
                    "judge_interval_subset_rule"
                    if judge.get("interval_subset_applied") is True
                    else "judge_semantic_answer"
                )
            if judge.get("status") == "ok" and judge_answer_evaluated is True:
                # If an older Judge row predates the explicit interval audit
                # fields, the newly generated deterministic row still carries
                # the same rule in its match method.
                if deterministic_interval_method in {"interval_subset", "interval_not_subset"}:
                    final_correct = deterministic_interval_method == "interval_subset"
                    final_source = "judge_interval_subset_rule"
            if judge.get("explanation_evaluated") is True:
                explanation_score = judge.get("explanation_score")
                explanation_scores = judge.get("explanation_scores") or {}

        row["deterministic_correct"] = deterministic_correct
        row["judge_answer_correct"] = judge_answer_correct
        row["judge_answer_correct_raw"] = (
            raw_judge_answer_correct if judge is not None else None
        )
        row["judge_status"] = judge_status
        row["final_correct"] = final_correct
        row["final_score_source"] = final_source
        row["explanation_score"] = explanation_score
        row["explanation_scores"] = explanation_scores
        row["correct"] = final_correct
        row["score"] = 1.0 if final_correct else 0.0
        merged.append(row)
    return merged


def aggregate_final_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the final score after applying the configured Judge policy."""
    summary = aggregate_score_rows(rows)
    deterministic_correct = sum(bool(row.get("deterministic_correct")) for row in rows)
    total = len(rows)
    judge_answer_sources = {
        "judge_semantic_answer",
        "judge_interval_subset_rule",
    }
    judge_answer_rows = [
        row
        for row in rows
        if row.get("final_score_source") in judge_answer_sources
    ]
    judge_overrides = [
        row
        for row in judge_answer_rows
        if bool(row.get("deterministic_correct")) != bool(row.get("correct"))
    ]
    explanation_rows = [
        row
        for row in rows
        if isinstance(row.get("explanation_score"), (int, float))
    ]
    summary.update(
        {
            "score_policy": (
                "Judge answer correctness is used for non-long-explanation "
                "short_answer and classification items plus the targeted "
                "reduced_enzyme_activity_exposure semantic subtype; all other "
                "multiple-choice and long-explanation final answers use deterministic "
                "Binary Accuracy."
            ),
            "deterministic_correct_items": deterministic_correct,
            "deterministic_accuracy": deterministic_correct / total if total else None,
            "judge_answer_evaluated_items": len(judge_answer_rows),
            "judge_answer_overrides": len(judge_overrides),
            "explanation_scored_items": len(explanation_rows),
            "explanation_score_mean": (
                sum(float(row["explanation_score"]) for row in explanation_rows) / len(explanation_rows)
                if explanation_rows
                else None
            ),
            "final_correct_items": summary["correct_items"],
            "final_accuracy": summary["accuracy"],
        }
    )
    return summary


def _counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(sorted(counts.items()))


def diagnostic_answer_score(
    *,
    prediction: Any,
    accepted_answers: list[Any],
    question: dict[str, Any] | None = None,
    correct_labels: set[str] | None = None,
) -> tuple[bool, str, str]:
    """Score a diagnostic answer and return correctness, method, and semantic key."""
    question = question or {}
    details = choice_prediction_details(question, prediction)
    pred_norm = normalized(prediction)
    accepted_norm = {normalized(value) for value in accepted_answers}
    if correct_labels:
        label = details["explicit_label"]
        if label:
            # In a permutation experiment, a letter is meaningful only in the
            # current option layout. Never score the original canonical letter
            # before applying the permutation-specific gold label.
            if label in correct_labels:
                return True, "diagnostic_choice_label", str(details["semantic_key"])
            return False, "diagnostic_choice_label_wrong_position", str(details["semantic_key"])
        if pred_norm and pred_norm in accepted_norm:
            return True, "diagnostic_normalized_exact", str(details["semantic_key"])
        if details["selected_option_text"] and normalized(details["selected_option_text"]) in accepted_norm:
            return True, "diagnostic_choice_text", str(details["semantic_key"])
        return False, "diagnostic_choice_no_match", str(details["semantic_key"])
    if pred_norm and pred_norm in accepted_norm:
        return True, "diagnostic_normalized_exact", f"semantic:{pred_norm}"
    return False, "diagnostic_no_match", f"prediction:{pred_norm}"


def score_generic_diagnostic_row(
    *,
    variant_id: str,
    group_id: str,
    question: str,
    result: dict[str, Any] | None,
    accepted_answers: list[Any],
    task_format: str = "short_answer",
    question_record: dict[str, Any] | None = None,
    correct_labels: set[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(result.get("status")) if result else "missing"
    prediction = answer_from_result(result)
    if status == "ok":
        correct, method, semantic_key = diagnostic_answer_score(
            prediction=prediction,
            accepted_answers=accepted_answers,
            question=question_record,
            correct_labels=correct_labels,
        )
    else:
        correct, method, semantic_key = False, "not_submitted", f"prediction:{normalized(prediction)}"
    row: dict[str, Any] = {
        "variant_id": variant_id,
        "group_id": group_id,
        "question": question,
        "task_format": task_format,
        "model_status": status,
        "prediction": as_text(prediction),
        "prediction_normalized": normalized(prediction),
        "semantic_key": semantic_key,
        "correct": bool(correct),
        "score": 1.0 if correct else 0.0,
        "match_method": method,
    }
    if extra:
        row.update(extra)
    return row


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return as_text(value)
