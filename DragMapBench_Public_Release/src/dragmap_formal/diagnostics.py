from __future__ import annotations

import random
import traceback
from pathlib import Path
from typing import Any, Callable

from .data import Dataset
from .hashing import sha256_json
from .parse import parse_answer
from .paths import load_model_registry
from .prompts import build_core_messages, build_diagnostic_messages
from .registry import ModelSpec, build_client
from .reporting import persist_score_outputs
from .runs import append_result, completed_keys, create_run_dir, expected_count_from_metadata, load_metadata, results_by_key
from .scoring import choice_prediction_details, diagnostic_answer_score
from .statistics import mean, proportion


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _ensure_json_response(
    *,
    client: Any,
    messages: list[dict[str, str]],
    model: ModelSpec,
    response: Any,
) -> tuple[Any, dict[str, Any] | None]:
    if parse_answer(response.content).get("parse_status") == "json":
        return response, None
    repair_messages = [
        *messages,
        {"role": "assistant", "content": response.content},
        {
            "role": "user",
            "content": (
                "Format repair only. Preserve the answer from your previous response "
                "and return exactly one valid JSON object with these keys: "
                '{"answer":"...","explanation":"","citations":[]}. '
                "Do not add analysis, Markdown fences, or any other text. "
                "Do not change the answer merely because you are repairing its format."
            ),
        },
    ]
    try:
        repaired = client.chat(
            repair_messages,
            temperature=model.temperature,
            max_output_tokens=model.max_output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - preserve original response for audit.
        return response, {
            "status": "failed",
            "error": repr(exc),
            "original_parse_status": parse_answer(response.content).get("parse_status"),
        }
    if parse_answer(repaired.content).get("parse_status") != "json":
        return response, {
            "status": "failed",
            "error": "repair_response_was_not_json",
            "original_parse_status": parse_answer(response.content).get("parse_status"),
            "repair_response": repaired.content,
        }
    return repaired, {
        "status": "applied",
        "original_parse_status": parse_answer(response.content).get("parse_status"),
        "original_response": response.content,
        "original_provider_response": response.raw_response,
    }


def _call(
    *,
    client: Any,
    messages: list[dict[str, str]],
    model: ModelSpec,
    run_id: str,
    request_id: str,
    condition: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = client.chat(
            messages,
            temperature=model.temperature,
            max_output_tokens=model.max_output_tokens,
        )
        response, format_repair = _ensure_json_response(
            client=client,
            messages=messages,
            model=model,
            response=response,
        )
        parsed = parse_answer(response.content)
        usage = response.usage or {}
        row = {
            "run_id": run_id,
            "request_id": request_id,
            "model": model.display_name,
            "model_snapshot": model.model,
            "provider": model.provider,
            "condition": condition,
            "prompt_hash": sha256_json(messages),
            "status": "ok",
            "raw_response": response.content,
            "raw_provider_response": response.raw_response,
            "parsed_answer": parsed.get("answer"),
            # Diagnostic runs do not request long explanations. Keep this field
            # empty even if a provider adds an unsolicited rationale.
            "parsed_explanation": "",
            "parsed_citations": parsed.get("citations"),
            "parse_status": parsed.get("parse_status"),
            "latency_ms": round(float(response.latency_seconds) * 1000, 3),
            "input_tokens": _usage_value(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": _usage_value(usage, "completion_tokens", "output_tokens"),
            "usage": usage,
            "provider_request_id": response.request_id,
        }
        if format_repair is not None:
            row["format_repair"] = format_repair
            if format_repair.get("status") == "applied":
                row["raw_response_before_format_repair"] = format_repair["original_response"]
                row["raw_provider_response_before_format_repair"] = format_repair[
                    "original_provider_response"
                ]
            elif format_repair.get("status") == "failed":
                row["status"] = "error"
                row["error_type"] = "FormatRepairError"
                row["error"] = str(format_repair.get("error") or "format repair failed")
    except Exception as exc:  # noqa: BLE001 - retain item-level failure for resumption.
        row = {
            "run_id": run_id,
            "request_id": request_id,
            "model": model.display_name,
            "model_snapshot": model.model,
            "provider": model.provider,
            "condition": condition,
            "prompt_hash": sha256_json(messages),
            "status": "error",
            "raw_response": "",
            "parsed_answer": None,
            "parsed_explanation": "",
            "parse_status": "error",
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "usage": {},
            "provider_request_id": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=3),
        }
    if extra:
        row.update(extra)
    return row


def _run_path(
    *,
    experiment: str,
    model: ModelSpec,
    config: dict[str, Any],
    models_config: dict[str, Any],
    limit: int | None,
    dry_run: bool,
    run_dir: str | Path | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    return Path(run_dir) if run_dir else create_run_dir(
        experiment=experiment,
        model=model,
        condition=None,
        config=config,
        models_config=models_config,
        limit=limit,
        dry_run=dry_run,
        extra=extra,
    )


def _client(model: ModelSpec, config: dict[str, Any], dry_run: bool) -> Any:
    return build_client(
        model,
        timeout_seconds=float(config.get("request_timeout_seconds", 120)),
        max_retries=int(config.get("max_retries", 5)),
        mock=dry_run,
    )


def _diagnostic_score(
    *,
    result: dict[str, Any] | None,
    accepted: list[Any],
    question: dict[str, Any] | None = None,
    correct_labels: set[str] | None = None,
) -> dict[str, Any]:
    status = str(result.get("status")) if result else "missing"
    prediction = result.get("parsed_answer") if result else None
    if status == "ok":
        correct, method, semantic_key = diagnostic_answer_score(
            prediction=prediction,
            accepted_answers=accepted,
            question=question,
            correct_labels=correct_labels,
        )
    else:
        correct, method, semantic_key = False, "not_submitted", "prediction:"
    return {
        "status": status,
        "prediction": prediction,
        "correct": bool(correct),
        "score": 1.0 if correct else 0.0,
        "match_method": method,
        "semantic_key": semantic_key,
    }


def _write_diagnostic_outputs(run_path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], name: str) -> None:
    persist_score_outputs(run_dir=run_path, rows=rows, summary=summary, prefix=name)


def _completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("model_status") == "ok"]


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = sorted({str(row.get("model_status")) for row in rows})
    return {status: sum(str(row.get("model_status")) == status for row in rows) for status in statuses}


def _gold_atom_context(bundle: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    public_ids = [str(item["atomic_id"]) for item in bundle.get("atomic_questions", [])]
    gold_by_id = {str(item["atomic_id"]): item.get("canonical", "") for item in gold.get("atomic_answers", [])}
    missing = [atomic_id for atomic_id in public_ids if atomic_id not in gold_by_id]
    if missing:
        raise ValueError(f"Missing atomic gold for {bundle.get('bundle_id')}: {missing}")
    return {atomic_id: gold_by_id[atomic_id] for atomic_id in public_ids}


def _read_item_ids(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"Item ID file is empty: {path}")
    return list(dict.fromkeys(values))


def _selected_ids(run_path: Path, default: list[str]) -> list[str]:
    metadata = load_metadata(run_path)
    item_ids = (metadata.get("extra") or {}).get("item_ids")
    if isinstance(item_ids, list) and item_ids:
        return [str(value) for value in item_ids]
    return default


def _compositional_atomic_instruction(prerequisite_type: str) -> str:
    if prerequisite_type == "drug_enzyme_fact":
        return (
            "Return only the standardized enzyme name. Prefer the full enzyme name "
            "followed by its gene symbol in parentheses when known, for example "
            "'Cytochrome P450 3A4 (CYP3A4)'. Do not add an explanation."
        )
    if prerequisite_type == "local_pharmacokinetic_rule":
        return (
            "Return only one short phrase describing the direction of metabolism or "
            "metabolic clearance, such as 'Metabolism decreases' or "
            "'Metabolic clearance decreases'. Do not include plasma concentration, "
            "half-life, toxicity, compensation, or reasoning."
        )
    return "Return only the shortest standardized semantic answer phrase. Do not explain."


def _compositional_target_instruction(target_instruction: str | None = None) -> str:
    if target_instruction:
        return (
            f"{target_instruction} Return one answer phrase only. Do not include "
            "the enzyme name, reasoning, or a generic pronoun."
        )
    return (
        "Return one concise causal effect phrase containing both links in this order: "
        "the direction of metabolic clearance, followed by the direction of parent-drug "
        "systemic exposure. Use the form 'Metabolic clearance [direction] and systemic "
        "exposure [direction]'. Do not include the enzyme name, a complete sentence, "
        "half-life, toxicity, compensation, or reasoning."
    )


def run_compositional(
    *,
    dataset: Dataset,
    config: dict[str, Any],
    models_config_path: str | Path,
    model: ModelSpec,
    limit: int | None = None,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    item_ids_file: str | Path | None = None,
) -> Path:
    models_config = load_model_registry(models_config_path)
    run_path = _run_path(
        experiment="compositional",
        model=model,
        config=config,
        models_config=models_config,
        limit=limit,
        dry_run=dry_run,
        run_dir=run_dir,
        extra={"item_ids": _read_item_ids(item_ids_file)} if item_ids_file else None,
    )
    run_id = str(load_metadata(run_path)["run_id"])
    done = completed_keys(run_path)
    client = _client(model, config, dry_run)
    selected_ids = _read_item_ids(item_ids_file)
    public = dataset.load_diagnostic("compositional", private=False)
    if selected_ids is not None:
        selected = set(selected_ids)
        public = [row for row in public if str(row["bundle_id"]) in selected]
    private = {str(row["id"]): row for row in dataset.load_diagnostic("compositional", private=True)}
    for bundle in public[: limit or None]:
        bundle_id = str(bundle["bundle_id"])
        if bundle_id in done:
            continue
        gold = private[bundle_id]
        target = dataset.question_by_id[str(bundle["target_item_id"])]
        atomic_results: dict[str, dict[str, Any]] = {}
        for atomic in bundle.get("atomic_questions", []):
            atomic_id = str(atomic["atomic_id"])
            messages = build_diagnostic_messages(
                item_id=atomic_id,
                question=str(atomic["question"]),
                answer_instruction=_compositional_atomic_instruction(
                    str(atomic.get("prerequisite_type", ""))
                ),
            )
            atomic_results[atomic_id] = _call(
                client=client,
                messages=messages,
                model=model,
                run_id=run_id,
                request_id=f"{bundle_id}::{atomic_id}",
                condition="compositional_atomic",
                extra={"bundle_id": bundle_id, "atomic_id": atomic_id},
            )
        atomic_context = {
            str(item["atomic_id"]): atomic_results[str(item["atomic_id"])].get("parsed_answer")
            for item in bundle.get("atomic_questions", [])
        }
        target_question = str(target.get("question", ""))
        target_id = str(target["id"])
        conditions: list[tuple[str, dict[str, Any] | None]] = [
            ("direct", None),
            ("elicited_atoms", atomic_context),
            ("gold_atoms", _gold_atom_context(bundle, gold)),
        ]
        target_results: dict[str, dict[str, Any]] = {}
        for condition, context in conditions:
            messages = build_diagnostic_messages(
                item_id=f"{bundle_id}::{condition}",
                question=target_question,
                answer_instruction=_compositional_target_instruction(
                    str(bundle.get("target_answer_instruction", ""))
                ),
                context={"atomic_answers": context} if context is not None else None,
            )
            target_results[condition] = _call(
                client=client,
                messages=messages,
                model=model,
                run_id=run_id,
                request_id=f"{bundle_id}::{condition}",
                condition=f"compositional_{condition}",
                extra={"bundle_id": bundle_id, "target_item_id": target_id, "atomic_context": context},
            )
        append_result(
            run_path,
            {
                "run_id": run_id,
                "request_id": bundle_id,
                "bundle_id": bundle_id,
                "target_item_id": target_id,
                "model": model.display_name,
                "model_snapshot": model.model,
                "status": "ok"
                if all(x.get("status") == "ok" for x in target_results.values())
                and all(x.get("status") == "ok" for x in atomic_results.values())
                else "partial",
                "atomic_results": atomic_results,
                "target_results": target_results,
            },
        )
        if all(x.get("status") == "ok" for x in target_results.values()) and all(
            x.get("status") == "ok" for x in atomic_results.values()
        ):
            done.add(bundle_id)
    return run_path


def score_compositional(*, dataset: Dataset, run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_path = Path(run_dir)
    public = {str(row["bundle_id"]): row for row in dataset.load_diagnostic("compositional", private=False)}
    private = {str(row["id"]): row for row in dataset.load_diagnostic("compositional", private=True)}
    records = results_by_key(run_path)
    rows: list[dict[str, Any]] = []
    expected_count = expected_count_from_metadata(run_path, len(public))
    expected_bundle_ids = _selected_ids(run_path, list(public))[:expected_count]
    for bundle_id in expected_bundle_ids:
        bundle = public[bundle_id]
        record = records.get(bundle_id)
        gold = private[bundle_id]
        atomic_gold = {str(x["atomic_id"]): list(x.get("accepted_answers", [])) for x in gold.get("atomic_answers", [])}
        atomic_scores: dict[str, Any] = {}
        if record:
            for atomic_id, atomic_result in (record.get("atomic_results") or {}).items():
                atomic_scores[atomic_id] = _diagnostic_score(result=atomic_result, accepted=atomic_gold.get(atomic_id, []))
        target_scores: dict[str, Any] = {}
        for condition in ("direct", "elicited_atoms", "gold_atoms"):
            result = (record or {}).get("target_results", {}).get(condition)
            target_scores[condition] = _diagnostic_score(
                result=result,
                accepted=list(gold.get("target_answer", {}).get("accepted_answers", [])),
            )
        atomic_all_correct = bool(atomic_scores) and all(x["correct"] for x in atomic_scores.values())
        rows.append(
            {
                "bundle_id": bundle_id,
                "target_item_id": bundle.get("target_item_id"),
                "atomic_scores": atomic_scores,
                "atomic_all_correct": atomic_all_correct,
                "direct_correct": target_scores["direct"]["correct"],
                "elicited_atoms_correct": target_scores["elicited_atoms"]["correct"],
                "gold_atoms_correct": target_scores["gold_atoms"]["correct"],
                "target_scores": target_scores,
                "model_status": (record or {}).get("status", "missing"),
            }
        )
    completed = _completed_rows(rows)
    n = len(completed)
    all_atoms_n = sum(x["atomic_all_correct"] for x in completed)
    summary = {
        "score_scope": "metadata_limit" if expected_count != len(public) else "full_diagnostic",
        "expected_bundles": expected_count,
        "completed_bundles": n,
        "available_bundles": len(public),
        "status_counts": _status_counts(rows),
        "atomic_all_prerequisites_accuracy": proportion(sum(x["atomic_all_correct"] for x in completed), n),
        "direct_target_accuracy": proportion(sum(x["direct_correct"] for x in completed), n),
        "elicited_atoms_target_accuracy": proportion(sum(x["elicited_atoms_correct"] for x in completed), n),
        "gold_atoms_target_accuracy": proportion(sum(x["gold_atoms_correct"] for x in completed), n),
        "conditional_elicited_target_accuracy": proportion(
            sum(x["elicited_atoms_correct"] for x in completed if x["atomic_all_correct"]),
            all_atoms_n,
        ),
        "conditional_reasoning_accuracy": proportion(
            sum(x["direct_correct"] for x in completed if x["atomic_all_correct"]),
            all_atoms_n,
        ),
        "compositionality_gap": (
            proportion(all_atoms_n, n) - proportion(
                sum(x["direct_correct"] for x in completed if x["atomic_all_correct"]),
                all_atoms_n,
            )
            if n and all_atoms_n
            else None
        ),
        "conditional_gold_target_accuracy": proportion(
            sum(x["gold_atoms_correct"] for x in completed if x["atomic_all_correct"]),
            all_atoms_n,
        ),
        "lucky_composition_rate": proportion(
            sum(x["direct_correct"] for x in completed if not x["atomic_all_correct"]),
            sum(not x["atomic_all_correct"] for x in completed),
        ),
        "gold_atom_utilization": mean(
            float(x["gold_atoms_correct"]) - float(x["direct_correct"]) for x in completed
        ),
        "compositionality_gap_direct_to_gold_atoms": mean(
            float(x["gold_atoms_correct"]) - float(x["direct_correct"]) for x in completed
        ),
    }
    _write_diagnostic_outputs(run_path, rows, summary, "compositional_score")
    return rows, summary


def run_counterfactual(
    *,
    dataset: Dataset,
    config: dict[str, Any],
    models_config_path: str | Path,
    model: ModelSpec,
    limit: int | None = None,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    item_ids_file: str | Path | None = None,
) -> Path:
    models_config = load_model_registry(models_config_path)
    run_path = _run_path(
        experiment="counterfactual",
        model=model,
        config=config,
        models_config=models_config,
        limit=limit,
        dry_run=dry_run,
        run_dir=run_dir,
        extra={
            "execution_order_seed": int(config.get("statistics", {}).get("random_seed", 0)),
            **({"item_ids": _read_item_ids(item_ids_file)} if item_ids_file else {}),
        },
    )
    run_id = str(load_metadata(run_path)["run_id"])
    done = completed_keys(run_path)
    client = _client(model, config, dry_run)
    selected_ids = _read_item_ids(item_ids_file)
    public = dataset.load_diagnostic("counterfactual", private=False)
    if selected_ids is not None:
        selected = set(selected_ids)
        public = [row for row in public if str(row["triplet_id"]) in selected]
    private = {str(row["id"]): row for row in dataset.load_diagnostic("counterfactual", private=True)}
    seed = int(config.get("statistics", {}).get("random_seed", 0))
    for triplet in public[: limit or None]:
        triplet_id = str(triplet["triplet_id"])
        if triplet_id in done:
            continue
        rng = random.Random(f"{seed}:{triplet_id}")
        variants = ["original", "lexical_control", "causal_intervention"]
        rng.shuffle(variants)
        variant_results: dict[str, dict[str, Any]] = {}
        for variant in variants:
            question = str(triplet[variant]["question"])
            messages = build_diagnostic_messages(
                item_id=f"{triplet_id}::{variant}",
                question=question,
                answer_instruction=(
                    "Fill in the blank with only the shortest standardized "
                    "pharmacokinetic effect phrase. State the requested direction "
                    "without explanation, reasoning, or a complete sentence."
                ),
            )
            variant_results[variant] = _call(
                client=client,
                messages=messages,
                model=model,
                run_id=run_id,
                request_id=f"{triplet_id}::{variant}",
                condition=f"counterfactual_{variant}",
                extra={"triplet_id": triplet_id, "variant": variant, "execution_order": variants.index(variant)},
            )
        append_result(
            run_path,
            {
                "run_id": run_id,
                "request_id": triplet_id,
                "triplet_id": triplet_id,
                "model": model.display_name,
                "model_snapshot": model.model,
                "status": "ok" if all(x.get("status") == "ok" for x in variant_results.values()) else "partial",
                "execution_order": variants,
                "variant_results": variant_results,
            },
        )
        if all(x.get("status") == "ok" for x in variant_results.values()):
            done.add(triplet_id)
    return run_path


def score_counterfactual(*, dataset: Dataset, run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = {str(row["triplet_id"]): row for row in dataset.load_diagnostic("counterfactual", private=False)}
    private = {str(row["id"]): row for row in dataset.load_diagnostic("counterfactual", private=True)}
    records = results_by_key(run_dir)
    rows: list[dict[str, Any]] = []
    run_path = Path(run_dir)
    expected_count = expected_count_from_metadata(run_path, len(public))
    expected_triplet_ids = _selected_ids(run_path, list(public))[:expected_count]
    for triplet_id in expected_triplet_ids:
        triplet = public[triplet_id]
        record = records.get(triplet_id)
        gold = private[triplet_id]
        variant_scores: dict[str, Any] = {}
        for variant in ("original", "lexical_control", "causal_intervention"):
            result = (record or {}).get("variant_results", {}).get(variant)
            accepted = list((gold.get(variant) or {}).get("accepted_answers", []))
            variant_scores[variant] = _diagnostic_score(result=result, accepted=accepted)
        original_pred = variant_scores["original"].get("prediction")
        lexical_pred = variant_scores["lexical_control"].get("prediction")
        causal_pred = variant_scores["causal_intervention"].get("prediction")
        original_key = variant_scores["original"].get("semantic_key")
        lexical_key = variant_scores["lexical_control"].get("semantic_key")
        causal_key = variant_scores["causal_intervention"].get("semantic_key")
        rows.append(
            {
                "triplet_id": triplet_id,
                "variant_scores": variant_scores,
                "original_correct": variant_scores["original"]["correct"],
                "lexical_correct": variant_scores["lexical_control"]["correct"],
                "causal_correct": variant_scores["causal_intervention"]["correct"],
                "lexical_invariant": bool(
                    original_key and lexical_key and original_key == lexical_key
                ),
                "causal_flip": bool(
                    original_key and causal_key and original_key != causal_key
                ),
                "model_status": (record or {}).get("status", "missing"),
            }
        )
    completed = _completed_rows(rows)
    n = len(completed)
    pair_correct = sum(x["original_correct"] and x["causal_correct"] for x in completed)
    invalid_invariance = sum(not x["causal_flip"] for x in completed)
    lexical_instability = sum(not x["lexical_invariant"] for x in completed)
    flip_consistency = proportion(sum(x["causal_flip"] for x in completed), n)
    lexical_instability_rate = proportion(lexical_instability, n)
    summary = {
        "score_scope": "metadata_limit" if expected_count != len(public) else "full_diagnostic",
        "expected_triplets": expected_count,
        "completed_triplets": n,
        "available_triplets": len(public),
        "status_counts": _status_counts(rows),
        "original_accuracy": proportion(sum(x["original_correct"] for x in completed), n),
        "lexical_control_accuracy": proportion(sum(x["lexical_correct"] for x in completed), n),
        "causal_intervention_accuracy": proportion(sum(x["causal_correct"] for x in completed), n),
        "pair_accuracy_original_and_causal": proportion(pair_correct, n),
        "lexical_invariance_rate": proportion(sum(x["lexical_invariant"] for x in completed), n),
        "lexical_instability_rate": lexical_instability_rate,
        "causal_flip_rate": flip_consistency,
        "invalid_invariance_rate": proportion(invalid_invariance, n),
        "causal_sensitivity_score": (
            flip_consistency - lexical_instability_rate
            if flip_consistency is not None and lexical_instability_rate is not None
            else None
        ),
        "causal_flip_consistency": proportion(
            sum(x["causal_flip"] and x["original_correct"] and x["causal_correct"] for x in completed),
            sum(x["original_correct"] for x in completed),
        ),
    }
    _write_diagnostic_outputs(run_path, rows, summary, "counterfactual_score")
    return rows, summary


def run_answer_format(
    *,
    dataset: Dataset,
    config: dict[str, Any],
    models_config_path: str | Path,
    model: ModelSpec,
    limit: int | None = None,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    group_ids_file: str | Path | None = None,
) -> Path:
    models_config = load_model_registry(models_config_path)
    run_path = _run_path(
        experiment="answer_format",
        model=model,
        config=config,
        models_config=models_config,
        limit=limit,
        dry_run=dry_run,
        run_dir=run_dir,
    )
    run_id = str(load_metadata(run_path)["run_id"])
    done = completed_keys(run_path)
    client = _client(model, config, dry_run)
    public = dataset.load_diagnostic("answer_format", private=False)
    group_ids: set[str] | None = None
    if group_ids_file:
        group_ids = {
            line.strip()
            for line in Path(group_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    for group in public[: limit or None]:
        group_id = str(group["format_group_id"])
        if group_ids is not None and group_id not in group_ids:
            continue
        if group_id in done:
            continue
        variant_results: dict[str, dict[str, Any]] = {}
        open_question = str(group["open_ended"]["question"])
        open_messages = build_diagnostic_messages(
            item_id=f"{group_id}::open_ended",
            question=open_question,
            answer_instruction="Return one concise semantic answer, not an option letter.",
        )
        variant_results["open_ended"] = _call(
            client=client,
            messages=open_messages,
            model=model,
            run_id=run_id,
            request_id=f"{group_id}::open_ended",
            condition="answer_format_open_ended",
            extra={"format_group_id": group_id, "variant": "open_ended"},
        )
        for permutation in group.get("mcq_permutations", []):
            permutation_id = str(permutation["permutation_id"])
            payload = {
                "id": permutation_id,
                "task_format": "multiple_choice",
                "question": open_question,
                "options": permutation.get("options", []),
            }
            messages = build_core_messages(payload, condition="closed_book")
            variant_results[permutation_id] = _call(
                client=client,
                messages=messages,
                model=model,
                run_id=run_id,
                request_id=f"{group_id}::{permutation_id}",
                condition=f"answer_format_{permutation_id}",
                extra={"format_group_id": group_id, "variant": permutation_id, "options": permutation.get("options", [])},
            )
        append_result(
            run_path,
            {
                "run_id": run_id,
                "request_id": group_id,
                "format_group_id": group_id,
                "model": model.display_name,
                "model_snapshot": model.model,
                "status": "ok" if all(x.get("status") == "ok" for x in variant_results.values()) else "partial",
                "variant_results": variant_results,
            },
        )
        if all(x.get("status") == "ok" for x in variant_results.values()):
            done.add(group_id)
    return run_path


def score_answer_format(*, dataset: Dataset, run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = {str(row["format_group_id"]): row for row in dataset.load_diagnostic("answer_format", private=False)}
    private = {str(row["id"]): row for row in dataset.load_diagnostic("answer_format", private=True)}
    records = results_by_key(run_dir)
    rows: list[dict[str, Any]] = []
    run_path = Path(run_dir)
    expected_count = expected_count_from_metadata(run_path, len(public))
    expected_group_ids = list(public)[:expected_count]
    for group_id in expected_group_ids:
        group = public[group_id]
        record = records.get(group_id)
        gold = private[group_id]
        semantic = gold.get("semantic_answer") or {}
        open_result = (record or {}).get("variant_results", {}).get("open_ended")
        open_score = _diagnostic_score(result=open_result, accepted=list(semantic.get("accepted_answers", [])))
        permutation_scores: dict[str, Any] = {}
        correct_positions: dict[str, str] = {}
        for permutation in group.get("mcq_permutations", []):
            permutation_id = str(permutation["permutation_id"])
            result = (record or {}).get("variant_results", {}).get(permutation_id)
            correct_label = str((gold.get("correct_labels") or {}).get(permutation_id, ""))
            correct_positions[permutation_id] = correct_label
            permutation_scores[permutation_id] = _diagnostic_score(
                result=result,
                accepted=list(semantic.get("accepted_answers", [])),
                question={"options": permutation.get("options", [])},
                correct_labels={correct_label} if correct_label else None,
            )
            prediction = permutation_scores[permutation_id].get("prediction")
            permutation_scores[permutation_id]["prediction_details"] = choice_prediction_details(
                {"options": permutation.get("options", [])},
                prediction,
            )
            permutation_scores[permutation_id]["correct_position"] = correct_label
        permutation_keys = [
            str(score.get("semantic_key", ""))
            for score in permutation_scores.values()
            if str(score.get("semantic_key", ""))
        ]
        selected_positions = [
            str(score.get("prediction_details", {}).get("selected_label", ""))
            for score in permutation_scores.values()
            if str(score.get("prediction_details", {}).get("selected_label", ""))
        ]
        distinct_semantic_answers = len(set(permutation_keys))
        rows.append(
            {
                "format_group_id": group_id,
                "open_correct": open_score["correct"],
                "open_score": open_score,
                "permutation_scores": permutation_scores,
                "correct_positions": correct_positions,
                "all_permutations_correct": bool(permutation_scores)
                and all(x["correct"] for x in permutation_scores.values()),
                "permutation_accuracy": mean(float(x["correct"]) for x in permutation_scores.values()),
                "permutation_accuracy_range": (
                    max(float(x["correct"]) for x in permutation_scores.values())
                    - min(float(x["correct"]) for x in permutation_scores.values())
                    if permutation_scores
                    else None
                ),
                "permutation_semantic_consistent": bool(permutation_keys)
                and distinct_semantic_answers == 1,
                "permutation_selected_positions": selected_positions,
                "model_status": (record or {}).get("status", "missing"),
            }
        )
    completed = _completed_rows(rows)
    n = len(completed)
    mean_permutation_accuracy = mean(float(x["permutation_accuracy"]) for x in completed)
    open_accuracy = proportion(sum(x["open_correct"] for x in completed), n)
    selection_counts: dict[str, int] = {}
    position_total = 0
    for row in completed:
        for position in row.get("permutation_selected_positions", []):
            selection_counts[position] = selection_counts.get(position, 0) + 1
            position_total += 1
    correct_position_accuracy: dict[str, float | None] = {}
    for position in ("A", "B", "C", "D"):
        position_scores: list[bool] = []
        for row in completed:
            for score in row.get("permutation_scores", {}).values():
                if score.get("correct_position") == position:
                    position_scores.append(bool(score.get("correct")))
        correct_position_accuracy[position] = proportion(sum(position_scores), len(position_scores))
    summary = {
        "score_scope": "metadata_limit" if expected_count != len(public) else "full_diagnostic",
        "expected_semantic_groups": expected_count,
        "completed_semantic_groups": n,
        "available_semantic_groups": len(public),
        "status_counts": _status_counts(rows),
        "open_ended_accuracy": open_accuracy,
        "all_permutations_consistency": proportion(sum(x["all_permutations_correct"] for x in completed), n),
        "permutation_semantic_consistency": proportion(
            sum(x["permutation_semantic_consistent"] for x in completed),
            n,
        ),
        "mean_permutation_accuracy_range": mean(
            float(x["permutation_accuracy_range"])
            for x in completed
            if x["permutation_accuracy_range"] is not None
        ),
        "mean_permutation_accuracy": mean_permutation_accuracy,
        "mcq_rescue_rate": proportion(
            sum((not x["open_correct"]) and x["permutation_accuracy"] > 0 for x in completed),
            sum(not x["open_correct"] for x in completed),
        ),
        "open_advantage_rate": proportion(
            sum(x["open_correct"] and x["permutation_accuracy"] < 1 for x in completed),
            sum(x["permutation_accuracy"] < 1 for x in completed),
        ),
        "predicted_option_position_distribution": {
            position: proportion(selection_counts.get(position, 0), position_total)
            for position in ("A", "B", "C", "D")
        },
        "accuracy_by_correct_option_position": correct_position_accuracy,
        "recognition_recall_gap": (
            mean_permutation_accuracy - open_accuracy
            if mean_permutation_accuracy is not None and open_accuracy is not None
            else None
        ),
    }
    _write_diagnostic_outputs(run_path, rows, summary, "answer_format_score")
    return rows, summary
