from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .client import ChatResult
from .data import Dataset
from .hashing import sha256_json
from .io import read_jsonl, write_jsonl
from .parse import parse_judge
from .paths import load_model_registry, write_json
from .prompts import (
    JUDGE_SEMANTIC_ANSWER_SUBTYPES,
    LONG_EXPLANATION_TYPES,
    build_judge_messages,
)
from .reporting import persist_score_outputs
from .registry import ModelSpec, build_client, load_specs
from .runs import JUDGE_RESULTS_FILE, JUDGE_SUMMARY_FILE, append_result, load_metadata
from .scoring import (
    aggregate_final_score_rows,
    aggregate_score_rows,
    interval_subset_match,
    merge_judge_scores,
    score_core_results,
)


JUDGE_ANSWER_TASK_FORMATS = frozenset({"short_answer", "classification"})
JUDGE_EXPLANATION_TYPES = LONG_EXPLANATION_TYPES
JUDGE_POLICY_VERSION = "judge-v3-targeted-semantic-answer-interval-subset"
# Backward-compatible names for callers that imported the old constants.
JUDGE_DEFAULT_TASK_FORMATS = JUDGE_ANSWER_TASK_FORMATS
JUDGE_DEFAULT_TYPES = JUDGE_EXPLANATION_TYPES


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _dry_judge_response(
    *,
    evaluate_answer: bool,
    evaluate_explanation: bool,
) -> ChatResult:
    answer_value = "true" if evaluate_answer else "null"
    if evaluate_explanation:
        explanation_scores = (
            '{"logical_coherence": 3, "factual_support": 3, '
            '"clinical_relevance": 3, "conciseness_faithfulness": 3}'
        )
        explanation_applicable = "true"
    else:
        explanation_scores = (
            '{"logical_coherence": null, "factual_support": null, '
            '"clinical_relevance": null, "conciseness_faithfulness": null}'
        )
        explanation_applicable = "false"
    return ChatResult(
        content=(
            f'{{"answer_correct": {answer_value}, '
            f'"explanation_applicable": {explanation_applicable}, '
            f'"explanation_scores": {explanation_scores}, '
            '"confidence": 1.0, '
            '"error_type": "", "rationale": "dry run"}'
        ),
        raw_response={"mock": True},
        usage={},
        latency_seconds=0.0,
        request_id="mock-judge",
    )


def _judge_one(
    *,
    client: Any,
    dry_run: bool,
    judge: ModelSpec,
    run_id: str,
    item_id: str,
    question: str,
    candidate_answer: Any,
    candidate_explanation: str,
    reference_answers: list[Any],
    item_type: str,
    subtype: str,
    task_format: str,
    source: str,
    evaluate_answer: bool,
    evaluate_explanation: bool,
) -> dict[str, Any]:
    messages = build_judge_messages(
        item_id=item_id,
        question=question,
        candidate_answer=candidate_answer,
        candidate_explanation=candidate_explanation,
        reference_answers=reference_answers,
        task_format=task_format,
        source=source,
        item_type=item_type,
        subtype=subtype,
        evaluate_answer=evaluate_answer,
        evaluate_explanation=evaluate_explanation,
    )
    try:
        response = (
            _dry_judge_response(
                evaluate_answer=evaluate_answer,
                evaluate_explanation=evaluate_explanation,
            )
            if dry_run
            else client.chat(
                messages,
                temperature=judge.temperature,
                max_output_tokens=judge.max_output_tokens,
            )
        )
        parsed = parse_judge(response.content)
        raw_answer_correct = parsed.get("answer_correct")
        interval_result = (
            interval_subset_match(candidate_answer, reference_answers)
            if evaluate_answer
            else None
        )
        if interval_result is not None:
            # The interval rule is deterministic and takes precedence over a
            # Judge model's free-form interpretation of the boundary.
            parsed["answer_correct"] = interval_result
            parsed["semantic_correct"] = interval_result
        usage = response.usage or {}
        return {
            "run_id": run_id,
            "request_id": item_id,
            "item_id": item_id,
            "judge_model": judge.display_name,
            "judge_snapshot": judge.model,
            "item_type": item_type,
            "subtype": subtype,
            "task_format": task_format,
            "source": source,
            "answer_evaluated": evaluate_answer,
            "explanation_evaluated": evaluate_explanation,
            "judge_policy_version": JUDGE_POLICY_VERSION,
            "judge_answer_correct_raw": raw_answer_correct,
            "answer_correct_source": (
                "interval_subset_rule" if interval_result is not None else "judge_model"
            ),
            "interval_subset_applied": interval_result is not None,
            "interval_subset_correct": interval_result,
            "prompt_hash": sha256_json(messages),
            "status": "ok" if parsed.get("parse_status") == "json" else "error",
            "raw_response": response.content,
            "raw_provider_response": response.raw_response,
            **parsed,
            "latency_ms": round(float(response.latency_seconds) * 1000, 3),
            "input_tokens": _usage_value(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": _usage_value(usage, "completion_tokens", "output_tokens"),
            "usage": usage,
            "provider_request_id": response.request_id,
        }
    except Exception as exc:  # noqa: BLE001 - preserve item-level audit failures.
        return {
            "run_id": run_id,
            "request_id": item_id,
            "item_id": item_id,
            "judge_model": judge.display_name,
            "judge_snapshot": judge.model,
            "item_type": item_type,
            "subtype": subtype,
            "task_format": task_format,
            "source": source,
            "answer_evaluated": evaluate_answer,
            "explanation_evaluated": evaluate_explanation,
            "judge_policy_version": JUDGE_POLICY_VERSION,
            "judge_answer_correct_raw": None,
            "answer_correct_source": "unavailable",
            "interval_subset_applied": False,
            "interval_subset_correct": None,
            "prompt_hash": sha256_json(messages),
            "status": "error",
            "answer_correct": None,
            "semantic_correct": None,
            "explanation_applicable": evaluate_explanation,
            "explanation_scores": {
                "logical_coherence": None,
                "factual_support": None,
                "clinical_relevance": None,
                "conciseness_faithfulness": None,
            },
            "explanation_score": None,
            "confidence": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=3),
        }


def _evaluation_scope(question: dict[str, Any]) -> tuple[bool, bool]:
    task_format = str(question.get("task_format", "")).strip()
    item_type = str(question.get("type", "")).strip()
    subtype = str(question.get("subtype", "")).strip()
    return (
        (
            task_format in JUDGE_ANSWER_TASK_FORMATS
            and item_type not in JUDGE_EXPLANATION_TYPES
        )
        or subtype in JUDGE_SEMANTIC_ANSWER_SUBTYPES,
        item_type in JUDGE_EXPLANATION_TYPES,
    )


def _normalize_existing_row(
    row: dict[str, Any],
    *,
    question: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(row)
    evaluate_answer, evaluate_explanation = _evaluation_scope(question)
    normalized.setdefault("item_type", str(question.get("type", "")))
    normalized.setdefault("subtype", str(question.get("subtype", "")))
    normalized.setdefault("task_format", str(question.get("task_format", "")))
    normalized.setdefault("answer_evaluated", evaluate_answer)
    normalized.setdefault("explanation_evaluated", evaluate_explanation)
    normalized.setdefault("judge_policy_version", "legacy")
    if "answer_correct" not in normalized:
        normalized["answer_correct"] = normalized.get("semantic_correct")
    if "semantic_correct" not in normalized:
        normalized["semantic_correct"] = normalized.get("answer_correct")
    normalized.setdefault("judge_answer_correct_raw", normalized.get("answer_correct"))
    normalized.setdefault("answer_correct_source", "legacy_judge_model")
    normalized.setdefault("interval_subset_applied", False)
    normalized.setdefault("interval_subset_correct", None)
    if "explanation_applicable" not in normalized:
        normalized["explanation_applicable"] = evaluate_explanation
    if "explanation_scores" not in normalized:
        normalized["explanation_scores"] = {
            "logical_coherence": None,
            "factual_support": None,
            "clinical_relevance": None,
            "conciseness_faithfulness": None,
        }
    normalized.setdefault("explanation_score", None)
    if (
        normalized.get("answer_evaluated") is evaluate_answer
        and normalized.get("explanation_evaluated") is evaluate_explanation
    ):
        # Preserve existing judgments whose evaluation scope did not change.
        # The targeted subtype changes scope and is therefore refreshed.
        normalized["judge_policy_version"] = JUDGE_POLICY_VERSION
    return normalized


def _needs_refresh(row: dict[str, Any], question: dict[str, Any]) -> bool:
    evaluate_answer, evaluate_explanation = _evaluation_scope(question)
    if row.get("status") != "ok":
        return True
    if row.get("judge_policy_version") != JUDGE_POLICY_VERSION:
        return True
    if row.get("answer_evaluated") is not evaluate_answer:
        return True
    if row.get("explanation_evaluated") is not evaluate_explanation:
        return True
    if evaluate_answer and row.get("answer_correct") is None:
        return True
    if evaluate_explanation:
        scores = row.get("explanation_scores")
        required = (
            "logical_coherence",
            "factual_support",
            "clinical_relevance",
            "conciseness_faithfulness",
        )
        if not isinstance(scores, dict) or any(scores.get(name) is None for name in required):
            return True
    return False


def summarize_judge_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    answer_rows = [
        row
        for row in valid
        if row.get(
            "answer_evaluated",
            "answer_correct" in row or "semantic_correct" in row,
        )
    ]
    answer_valid = [row for row in answer_rows if row.get("answer_correct") is not None]
    explanation_rows = [
        row
        for row in valid
        if row.get("explanation_evaluated", False)
    ]
    explanation_valid = [
        row
        for row in explanation_rows
        if isinstance(row.get("explanation_score"), (int, float))
    ]
    dimensions = (
        "logical_coherence",
        "factual_support",
        "clinical_relevance",
        "conciseness_faithfulness",
    )
    dimension_means: dict[str, float | None] = {}
    for name in dimensions:
        values = [
            row.get("explanation_scores", {}).get(name)
            for row in explanation_valid
            if isinstance(row.get("explanation_scores"), dict)
        ]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        dimension_means[name] = sum(numeric) / len(numeric) if numeric else None

    answer_correct_count = sum(bool(row.get("answer_correct")) for row in answer_valid)
    explanation_mean = (
        sum(float(row["explanation_score"]) for row in explanation_valid) / len(explanation_valid)
        if explanation_valid
        else None
    )
    interval_rows = [
        row
        for row in answer_valid
        if row.get("interval_subset_applied") is True
    ]
    return {
        "judge_policy_version": JUDGE_POLICY_VERSION,
        "judge_model": str(valid[0].get("judge_model", "")) if valid else None,
        "judge_snapshot": str(valid[0].get("judge_snapshot", "")) if valid else None,
        "audited_items": len(rows),
        "valid_judgments": len(valid),
        "answer_audited_items": len(answer_rows),
        "answer_valid_judgments": len(answer_valid),
        "answer_correct_count": answer_correct_count,
        "answer_semantic_accuracy": (
            answer_correct_count / len(answer_valid) if answer_valid else None
        ),
        # Retain this name for older consumers, but it now means answer audit accuracy.
        "semantic_accuracy": (
            answer_correct_count / len(answer_valid) if answer_valid else None
        ),
        "interval_subset_audited_items": len(interval_rows),
        "interval_subset_correct_items": sum(
            bool(row.get("interval_subset_correct")) for row in interval_rows
        ),
        "explanation_evaluated_items": len(explanation_rows),
        "explanation_scored_items": len(explanation_valid),
        "explanation_score_scale": "1-5 for each of four dimensions; mean is reported",
        "explanation_score_mean": explanation_mean,
        "explanation_dimension_means": dimension_means,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "official_score_policy": (
            "Judge answer correctness is used for ordinary short_answer and "
            "classification items, plus reduced_enzyme_activity_exposure using "
            "directional semantic evaluation. Multiple-choice final-answer "
            "correctness and other long-explanation final answers use Binary "
            "Accuracy. Long explanations are scored separately on four quality "
            "dimensions. For numeric intervals, a candidate interval that is a "
            "subset of the gold interval is correct, including a bare upper "
            "category boundary."
        ),
    }


def refresh_final_score_outputs(
    *,
    dataset: Dataset,
    run_dir: str | Path,
    judge_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    run_path = Path(run_dir)
    score_path = run_path / "core_score_rows.jsonl"
    if not score_path.exists():
        return None
    deterministic_rows = read_jsonl(score_path)
    # Recompute deterministic rows so the new interval-subset rule is reflected
    # even when this run was scored before the policy change.
    raw_path = run_path / "results.jsonl"
    if raw_path.exists():
        raw_rows = read_jsonl(raw_path)
        raw_by_id = {
            str(row.get("item_id") or row.get("request_id")): row
            for row in raw_rows
            if row.get("item_id") or row.get("request_id")
        }
        questions_by_id = {str(question["id"]): question for question in dataset.questions}
        ordered_questions = [
            questions_by_id[str(row["id"])]
            for row in deterministic_rows
            if str(row.get("id")) in questions_by_id
        ]
        recomputed = score_core_results(ordered_questions, dataset.gold_by_id, raw_by_id)
        recomputed_by_id = {str(row["id"]): row for row in recomputed}
        deterministic_rows = [
            recomputed_by_id.get(str(row.get("id")), row)
            for row in deterministic_rows
        ]
        deterministic_summary = aggregate_score_rows(deterministic_rows)
        deterministic_summary["score_scope"] = (
            "full_core"
            if len(deterministic_rows) == len(dataset.questions)
            else "available_rows"
        )
        deterministic_summary["expected_items"] = len(deterministic_rows)
        persist_score_outputs(
            run_dir=run_path,
            rows=deterministic_rows,
            summary=deterministic_summary,
            prefix="core_score",
        )
        write_jsonl(score_path, deterministic_rows)
    final_rows = merge_judge_scores(deterministic_rows, judge_rows)
    final_summary = aggregate_final_score_rows(final_rows)
    final_summary["score_scope"] = "full_core" if len(final_rows) == len(dataset.questions) else "available_rows"
    final_summary["expected_items"] = len(final_rows)
    final_summary["judge_results_file"] = str(run_path / JUDGE_RESULTS_FILE)
    persist_score_outputs(
        run_dir=run_path,
        rows=final_rows,
        summary=final_summary,
        prefix="final_score",
    )
    write_jsonl(run_path / "final_score_rows.jsonl", final_rows)
    return final_summary


def judge_core(
    *,
    dataset: Dataset,
    run_dir: str | Path,
    models_config_path: str | Path,
    dry_run: bool = False,
    include_all_formats: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_path = Path(run_dir)
    metadata = load_metadata(run_path)
    judge, _ = load_specs(str(models_config_path))
    if not judge.enabled:
        raise ValueError("The configured judge is disabled")
    models_config = load_model_registry(models_config_path)
    del models_config
    client = build_client(
        judge,
        timeout_seconds=float(metadata.get("config", {}).get("request_timeout_seconds", 120)),
        max_retries=int(metadata.get("config", {}).get("max_retries", 5)),
        mock=dry_run,
    )
    source_rows = read_jsonl(run_path / "results.jsonl")
    existing = {str(row.get("request_id")): row for row in read_jsonl(run_path / JUDGE_RESULTS_FILE)}
    # Public questions use `id`; `item_id` is the run-log field.
    rows_by_id = {str(row.get("id")): row for row in dataset.questions}
    candidates: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for result in source_rows:
        if result.get("status") != "ok":
            continue
        item_id = str(result.get("item_id", ""))
        question = rows_by_id.get(item_id)
        if not question:
            continue
        evaluate_answer, evaluate_explanation = _evaluation_scope(question)
        # Retain the CLI flag for compatibility with older launch scripts, but
        # never create no-op Judge rows for items outside the official scope.
        if not (evaluate_answer or evaluate_explanation):
            continue
        candidates.append((item_id, question, result, "core"))
    output_rows: list[dict[str, Any]] = []
    pending: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for item_id, question, result, source in candidates:
        existing_row = existing.get(item_id)
        if existing_row:
            existing_row = _normalize_existing_row(existing_row, question=question)
        if existing_row and not _needs_refresh(existing_row, question):
            output_rows.append(existing_row)
            continue
        pending.append((item_id, question, result, source))

    def judge_candidate(
        candidate: tuple[str, dict[str, Any], dict[str, Any], str],
    ) -> dict[str, Any]:
        item_id, question, result, source = candidate
        gold = dataset.gold_by_id[item_id]
        evaluate_answer, evaluate_explanation = _evaluation_scope(question)
        return _judge_one(
            client=client,
            dry_run=dry_run,
            judge=judge,
            run_id=str(metadata["run_id"]),
            item_id=item_id,
            question=str(question.get("question", "")),
            candidate_answer=result.get("parsed_answer"),
            candidate_explanation=str(result.get("parsed_explanation", "")),
            reference_answers=list((gold.get("answer") or {}).get("accepted_answers", [])),
            item_type=str(question.get("type", "")),
            subtype=str(question.get("subtype", "")),
            task_format=str(question.get("task_format", "")),
            source=source,
            evaluate_answer=evaluate_answer,
            evaluate_explanation=evaluate_explanation,
        )

    concurrency = max(1, int(os.environ.get("DRUGMAP_JUDGE_CONCURRENCY", "1")))
    write_lock = Lock()
    if concurrency == 1:
        for candidate in pending:
            row = judge_candidate(candidate)
            with write_lock:
                append_result(run_path, row, filename=JUDGE_RESULTS_FILE)
            output_rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(judge_candidate, candidate) for candidate in pending]
            for future in as_completed(futures):
                row = future.result()
                with write_lock:
                    append_result(run_path, row, filename=JUDGE_RESULTS_FILE)
                output_rows.append(row)

    summary = summarize_judge_rows(output_rows)
    summary["judge_model"] = judge.display_name
    summary["judge_snapshot"] = judge.model
    final_summary = refresh_final_score_outputs(
        dataset=dataset,
        run_dir=run_path,
        judge_rows=output_rows,
    )
    if final_summary is not None:
        summary["final_score_file"] = str(run_path / "final_score_summary.json")
        summary["final_accuracy"] = final_summary["accuracy"]
        summary["deterministic_accuracy"] = final_summary["deterministic_accuracy"]
        summary["final_correct_items"] = final_summary["correct_items"]
    write_json(run_path / JUDGE_SUMMARY_FILE, summary)
    return output_rows, summary
