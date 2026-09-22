from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .client import ChatResult
from .data import Dataset
from .hashing import sha256_json
from .io import read_jsonl, write_jsonl
from .parse import parse_answer
from .paths import load_model_registry
from .prompts import (
    LONG_EXPLANATION_TYPES,
    NATIVE_WEB_SEARCH_TOOL,
    WEB_SEARCH_TOOL,
    build_core_messages,
)
from .registry import ModelSpec, build_client
from .reporting import persist_score_outputs
from .runs import append_result, completed_keys, create_run_dir, expected_count_from_metadata, load_metadata, results_by_key
from .scoring import aggregate_score_rows, score_core_results
from .search import SearchProvider, build_search_provider, default_query
from .statistics import paired_state_metrics


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _assistant_tool_message(response: ChatResult) -> dict[str, Any]:
    """Return the minimal assistant message needed for the next tool turn."""
    raw_message = response.message if isinstance(response.message, dict) else {}
    message: dict[str, Any] = {
        "role": "assistant",
        "content": raw_message.get("content"),
        "tool_calls": response.tool_calls or [],
    }
    if raw_message.get("reasoning_content") is not None:
        message["reasoning_content"] = raw_message["reasoning_content"]
    return message


def _tool_call_parts(call: dict[str, Any], fallback_id: str) -> tuple[str, str, dict[str, Any]]:
    call_id = str(call.get("id") or fallback_id)
    function = call.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"Tool call {call_id} has no function payload")
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool call {call_id} has invalid JSON arguments") from exc
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ValueError(f"Tool call {call_id} has unsupported arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool call {call_id} arguments must be a JSON object")
    return call_id, name, arguments


def _tool_result_message(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _last_json_user_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _ensure_json_response(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: ModelSpec,
    response: ChatResult,
) -> tuple[ChatResult, dict[str, Any] | None]:
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
                "Do not add analysis, tool tags, Markdown fences, or any other text. "
                "For non-long-answer items, keep explanation as an empty string. "
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


def _run_open_web_chat(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: ModelSpec,
    provider: SearchProvider,
    web_config: dict[str, Any],
) -> tuple[ChatResult, dict[str, Any]]:
    """Run a model-controlled web-search/tool loop and return its final answer."""
    max_tool_turns = max(0, int(web_config.get("max_tool_turns", 6)))
    max_queries = max(0, int(web_config.get("max_queries", 5)))
    max_pages = max(0, int(web_config.get("max_pages", 10)))
    fallback_search_if_no_tool = bool(web_config.get("fallback_search_if_no_tool", True))
    fallback_max_results = max(
        1,
        min(max_pages or 1, int(web_config.get("fallback_max_results", 5))),
    )
    trace: dict[str, Any] = {
        "provider": provider.__class__.__name__,
        "tool": "web_search",
        "max_tool_turns": max_tool_turns,
        "max_queries": max_queries,
        "max_pages": max_pages,
        "model_turns": [],
        "queries": [],
        "results": [],
        "errors": [],
        "model_latency_seconds": 0.0,
        "search_latency_seconds": 0.0,
        "stopped_reason": None,
    }
    search_count = 0
    result_count = 0

    def record_model_turn(turn: int, response: ChatResult, *, forced_final: bool = False) -> None:
        trace["model_latency_seconds"] += float(response.latency_seconds)
        trace["model_turns"].append(
            {
                "turn": turn,
                "forced_final": forced_final,
                "request_id": response.request_id,
                "latency_ms": round(float(response.latency_seconds) * 1000, 3),
                "tool_call_count": len(response.tool_calls or []),
                "raw_response": response.raw_response,
            }
        )

    def run_fallback_search() -> dict[str, Any]:
        nonlocal search_count, result_count
        payload = _last_json_user_payload(messages)
        query = default_query(payload)
        query_trace: dict[str, Any] = {
            "turn": "fallback",
            "tool_call_id": None,
            "query": query,
            "requested_max_results": fallback_max_results,
            "results": [],
            "error": None,
        }
        started = time.perf_counter()
        try:
            results = provider.search(query, max_results=fallback_max_results)
            query_trace["results"] = [result.__dict__ for result in results]
            search_count += 1
            result_count += len(results)
        except Exception as exc:  # noqa: BLE001 - retain provider failures in the trace.
            search_count += 1
            query_trace["error"] = repr(exc)
        trace["search_latency_seconds"] += time.perf_counter() - started
        trace["queries"].append(query_trace)
        trace["results"].extend(query_trace["results"])
        if query_trace["error"]:
            trace["errors"].append(
                {
                    "turn": "fallback",
                    "query": query,
                    "error": query_trace["error"],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "open_web_search_evidence": {
                            "query": query,
                            "results": query_trace["results"],
                            "error": query_trace["error"],
                        },
                        "instruction": (
                            "Use the external search evidence above when relevant. "
                            "Return the final benchmark answer as one JSON object. "
                            "If sources support the answer, include their URLs under "
                            "the citations key."
                        ),
                    },
                    ensure_ascii=False,
                ),
            }
        )
        return query_trace

    def fallback_final_response(first_error: Exception | None = None) -> tuple[ChatResult, dict[str, Any]]:
        if first_error is not None:
            trace["errors"].append({"turn": 0, "error": repr(first_error), "phase": "tool_request"})
        run_fallback_search()
        final_response = client.chat(
            messages,
            temperature=model.temperature,
            max_output_tokens=model.max_output_tokens,
        )
        record_model_turn(max_tool_turns, final_response, forced_final=True)
        if final_response.tool_calls:
            error = "Model returned a tool call during the fallback final request."
            trace["errors"].append({"error": error})
            trace["stopped_reason"] = "fallback_tool_call"
            trace["latency_seconds"] = trace["model_latency_seconds"] + trace["search_latency_seconds"]
            raise RuntimeError(error)
        trace["stopped_reason"] = "fallback_prefetch"
        trace["latency_seconds"] = trace["model_latency_seconds"] + trace["search_latency_seconds"]
        return final_response, trace

    for turn in range(max_tool_turns):
        try:
            response = client.chat(
                messages,
                temperature=model.temperature,
                max_output_tokens=model.max_output_tokens,
                tools=[WEB_SEARCH_TOOL],
                tool_choice="auto",
            )
        except Exception as exc:
            if fallback_search_if_no_tool:
                return fallback_final_response(exc)
            raise
        record_model_turn(turn, response)
        tool_calls = response.tool_calls or []
        if not tool_calls:
            if fallback_search_if_no_tool and search_count == 0:
                return fallback_final_response()
            trace["stopped_reason"] = "model_answered"
            trace["latency_seconds"] = trace["model_latency_seconds"] + trace["search_latency_seconds"]
            return response, trace

        messages.append(_assistant_tool_message(response))
        for call_index, call in enumerate(tool_calls):
            fallback_id = f"web_search_{turn}_{call_index}"
            try:
                call_id, name, arguments = _tool_call_parts(call, fallback_id)
            except ValueError as exc:
                call_id = fallback_id
                error = str(exc)
                trace["errors"].append({"turn": turn, "tool_call_id": call_id, "error": error})
                messages.append(_tool_result_message(call_id, {"error": error, "results": []}))
                continue

            if name != "web_search":
                error = f"Unsupported tool: {name or '<missing name>'}"
                trace["errors"].append({"turn": turn, "tool_call_id": call_id, "error": error})
                messages.append(_tool_result_message(call_id, {"error": error, "results": []}))
                continue

            query = str(arguments.get("query") or "").strip()
            requested_results = arguments.get("max_results", 5)
            try:
                requested_results = int(requested_results)
            except (TypeError, ValueError):
                requested_results = 5
            requested_results = max(1, min(10, requested_results))
            query_trace: dict[str, Any] = {
                "turn": turn,
                "tool_call_id": call_id,
                "query": query,
                "requested_max_results": requested_results,
                "results": [],
                "error": None,
            }

            if not query:
                query_trace["error"] = "The web_search query must not be empty."
            elif search_count >= max_queries:
                query_trace["error"] = "The maximum number of search queries has been reached."
            elif result_count >= max_pages:
                query_trace["error"] = "The maximum number of search results has been reached."
            elif provider.__class__.__name__ == "DisabledSearchProvider":
                query_trace["error"] = "No web search provider is configured."
            else:
                remaining = max_pages - result_count
                started = time.perf_counter()
                try:
                    results = provider.search(
                        query,
                        max_results=min(requested_results, remaining),
                    )
                    search_count += 1
                    query_trace["results"] = [result.__dict__ for result in results]
                    result_count += len(results)
                except Exception as exc:  # noqa: BLE001 - retain provider failures in the trace.
                    search_count += 1
                    query_trace["error"] = repr(exc)
                finally:
                    elapsed = time.perf_counter() - started
                    trace["search_latency_seconds"] += elapsed

            trace["queries"].append(query_trace)
            trace["results"].extend(query_trace["results"])
            if query_trace["error"]:
                trace["errors"].append(
                    {
                        "turn": turn,
                        "tool_call_id": call_id,
                        "query": query,
                        "error": query_trace["error"],
                    }
                )
            messages.append(
                _tool_result_message(
                    call_id,
                    {
                        "query": query,
                        "results": query_trace["results"],
                        "error": query_trace["error"],
                    },
                )
            )

    # The final call is deliberately made without tools so the model must
    # answer with the evidence it has already collected.
    response = client.chat(
        messages,
        temperature=model.temperature,
        max_output_tokens=model.max_output_tokens,
    )
    record_model_turn(max_tool_turns, response, forced_final=True)
    if response.tool_calls:
        error = "Model requested another web search after the tool-turn limit."
        trace["errors"].append({"error": error})
        trace["stopped_reason"] = "tool_turn_limit"
        trace["latency_seconds"] = trace["model_latency_seconds"] + trace["search_latency_seconds"]
        raise RuntimeError(error)
    trace["stopped_reason"] = "tool_turn_limit"
    trace["latency_seconds"] = trace["model_latency_seconds"] + trace["search_latency_seconds"]
    return response, trace


def _run_native_web_search_chat(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: ModelSpec,
) -> tuple[ChatResult, dict[str, Any]]:
    """Run a provider-executed Responses API web search."""
    response = client.chat(
        messages,
        temperature=model.temperature,
        max_output_tokens=model.max_output_tokens,
        tools=[NATIVE_WEB_SEARCH_TOOL],
    )
    raw_output = response.raw_response.get("output", [])
    search_calls = [
        item
        for item in raw_output
        if isinstance(item, dict) and item.get("type") == "web_search_call"
    ]
    queries: list[str] = []
    for item in search_calls:
        action = item.get("action")
        if not isinstance(action, dict):
            continue
        query = action.get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return response, {
        "provider": "native_responses_api",
        "tool": "web_search",
        "mode": "provider_executed",
        "search_calls": search_calls,
        "queries": queries,
        "search_call_count": len(search_calls),
        "verified": bool(search_calls),
        "usage": response.usage,
        "latency_seconds": response.latency_seconds,
    }


def _record_from_success(
    *,
    run_id: str,
    model: ModelSpec,
    condition: str,
    item_id: str,
    item_type: str,
    messages: list[dict[str, Any]],
    response: Any,
    search_trace: dict[str, Any] | None = None,
    open_web_provider_status: str | None = None,
) -> dict[str, Any]:
    parsed = parse_answer(response.content)
    usage = response.usage or {}
    explanation = (
        parsed.get("explanation", "")
        if str(item_type).strip() in LONG_EXPLANATION_TYPES
        else ""
    )
    return {
        "run_id": run_id,
        "request_id": item_id,
        "item_id": item_id,
        "model": model.display_name,
        "model_snapshot": model.model,
        "provider": model.provider,
        "condition": condition,
        "prompt_hash": sha256_json(messages),
        "status": "ok",
        "raw_response": response.content,
        "raw_provider_response": response.raw_response,
        "parsed_answer": parsed.get("answer"),
        "parsed_explanation": explanation,
        "parsed_citations": parsed.get("citations"),
        "parse_status": parsed.get("parse_status"),
        "latency_ms": round(float(response.latency_seconds) * 1000, 3),
        "input_tokens": _usage_value(usage, "prompt_tokens", "input_tokens"),
        "output_tokens": _usage_value(usage, "completion_tokens", "output_tokens"),
        "usage": usage,
        "provider_request_id": response.request_id,
        "search_trace": search_trace,
        "open_web_provider_status": open_web_provider_status,
    }


def _record_from_error(
    *,
    run_id: str,
    model: ModelSpec,
    condition: str,
    item_id: str,
    messages: list[dict[str, Any]],
    exc: BaseException,
    search_trace: dict[str, Any] | None = None,
    open_web_provider_status: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "request_id": item_id,
        "item_id": item_id,
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
        "search_trace": search_trace,
        "open_web_provider_status": open_web_provider_status,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc(limit=3),
    }


def run_core(
    *,
    dataset: Dataset,
    config: dict[str, Any],
    models_config_path: str | Path,
    model: ModelSpec,
    condition: str = "closed_book",
    limit: int | None = None,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    question_types: set[str] | None = None,
    item_ids: set[str] | None = None,
) -> Path:
    if condition not in {"closed_book", "open_web"}:
        raise ValueError("condition must be closed_book or open_web")
    models_config = load_model_registry(models_config_path)
    run_path = Path(run_dir) if run_dir else create_run_dir(
        experiment="core",
        model=model,
        condition=condition,
        config=config,
        models_config=models_config,
        limit=limit,
        dry_run=dry_run,
    )
    metadata = load_metadata(run_path)
    run_id = str(metadata["run_id"])
    existing = results_by_key(run_path)
    done = {
        key
        for key, row in existing.items()
        if row.get("status") == "ok" and row.get("parse_status") == "json"
    }
    request_timeout_seconds = float(
        os.environ.get(
            "DRUGMAP_REQUEST_TIMEOUT_SECONDS",
            config.get("request_timeout_seconds", 120),
        )
    )
    max_retries = int(
        os.environ.get(
            "DRUGMAP_MAX_RETRIES",
            config.get("max_retries", 5),
        )
    )
    client = build_client(
        model,
        timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        mock=dry_run,
    )
    native_web_client = None
    if condition == "open_web" and model.web_search_mode == "native_responses":
        native_web_client = build_client(
            model,
            timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            mock=dry_run,
            for_web=True,
        )
    search_provider = build_search_provider() if condition == "open_web" else None
    open_web_provider_status = None
    if condition == "open_web":
        if model.web_search_mode == "native_responses":
            open_web_provider_status = "native_model_api"
        else:
            open_web_provider_status = (
                "disabled_no_provider"
                if search_provider.__class__.__name__ == "DisabledSearchProvider"
                else "configured"
            )
        metadata["open_web_provider_status"] = open_web_provider_status
        from .paths import write_json

        write_json(run_path / "metadata.json", metadata)
    records = [
        record for record in dataset.questions
        if (
            (not question_types or str(record.get("type", "")).strip() in question_types)
            and (not item_ids or str(record.get("id", "")) in item_ids)
        )
    ]
    records = records[: limit or None]
    if question_types:
        metadata["question_types"] = sorted(question_types)
        metadata["limit"] = len(records)
        from .paths import write_json

        write_json(run_path / "metadata.json", metadata)
    concurrency = max(1, int(os.environ.get("DRUGMAP_CONCURRENCY", "1")))
    write_lock = Lock()

    def process_record(record: dict[str, Any]) -> dict[str, Any]:
        item_id = str(record["id"])
        if item_id in done:
            return {}
        search_trace = None
        record_for_prompt = dict(record)
        messages = build_core_messages(record_for_prompt, condition=condition)
        try:
            if condition == "open_web" and model.web_search_mode == "native_responses":
                response, search_trace = _run_native_web_search_chat(
                    client=native_web_client,
                    messages=messages,
                    model=model,
                )
            elif condition == "open_web" and search_provider is not None:
                response, search_trace = _run_open_web_chat(
                    client=client,
                    messages=messages,
                    model=model,
                    provider=search_provider,
                    web_config=config.get("open_web", {}),
                )
            else:
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
            row = _record_from_success(
                run_id=run_id,
                model=model,
                condition=condition,
                item_id=item_id,
                item_type=str(record_for_prompt.get("type", "")),
                messages=messages,
                response=response,
                search_trace=search_trace,
                open_web_provider_status=open_web_provider_status,
            )
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
        except Exception as exc:  # noqa: BLE001 - every item should be logged for resumption.
            row = _record_from_error(
                run_id=run_id,
                model=model,
                condition=condition,
                item_id=item_id,
                messages=messages,
                exc=exc,
                search_trace=search_trace,
                open_web_provider_status=open_web_provider_status,
            )
        with write_lock:
            append_result(run_path, row)
            if row.get("status") == "ok":
                done.add(item_id)
        return row

    pending = [record for record in records if str(record["id"]) not in done]
    if concurrency == 1:
        for record in pending:
            process_record(record)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_record, record) for record in pending]
            for future in as_completed(futures):
                future.result()
    return run_path


def score_core(*, dataset: Dataset, run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_path = Path(run_dir)
    result_map = results_by_key(run_path)
    expected_count = expected_count_from_metadata(run_path, len(dataset.questions))
    metadata = load_metadata(run_path)
    question_types = {
        str(item).strip()
        for item in metadata.get("question_types", [])
        if str(item).strip()
    }
    source_questions = [
        question for question in dataset.questions
        if not question_types or str(question.get("type", "")).strip() in question_types
    ]
    expected_questions = source_questions[:expected_count]
    rows = score_core_results(expected_questions, dataset.gold_by_id, result_map)
    summary = aggregate_score_rows(rows)
    summary["score_scope"] = "metadata_limit" if expected_count != len(dataset.questions) else "full_core"
    summary["expected_items"] = expected_count
    summary["available_core_items"] = len(source_questions)
    if question_types:
        summary["question_types"] = sorted(question_types)
    persist_score_outputs(run_dir=run_path, rows=rows, summary=summary, prefix="core_score")
    write_jsonl(run_path / "core_score_rows.jsonl", rows)
    return rows, summary


def compare_core_runs(
    *,
    closed_score_rows: str | Path,
    open_score_rows: str | Path,
    output_path: str | Path,
    bootstrap_replicates: int = 10000,
    random_seed: int = 20260911,
) -> dict[str, Any]:
    closed_rows = read_jsonl(closed_score_rows)
    open_rows = read_jsonl(open_score_rows)
    summary = paired_state_metrics(
        closed_rows,
        open_rows,
        id_key="id",
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
    )
    from .paths import write_json

    write_json(output_path, summary)
    return summary
