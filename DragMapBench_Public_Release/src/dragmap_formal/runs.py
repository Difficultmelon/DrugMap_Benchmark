from __future__ import annotations

import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .io import append_jsonl, read_jsonl
from .paths import PROJECT_ROOT, write_json
from .registry import ModelSpec


RESULTS_FILE = "results.jsonl"
SCORES_FILE = "scores.jsonl"
SUMMARY_FILE = "summary.json"
JUDGE_RESULTS_FILE = "judge_results.jsonl"
JUDGE_SUMMARY_FILE = "judge_summary.json"
METADATA_FILE = "metadata.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    allowed = []
    for char in value.strip().lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "-", "_", ".", "/"}:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def runs_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root else PROJECT_ROOT / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_run_id(*, experiment: str, model_name: str, condition: str | None = None) -> str:
    pieces = [timestamp_id(), slugify(experiment), slugify(model_name)]
    if condition:
        pieces.append(slugify(condition))
    return "__".join(pieces)


def create_run_dir(
    *,
    experiment: str,
    model: ModelSpec,
    condition: str | None,
    config: dict[str, Any],
    models_config: dict[str, Any],
    limit: int | None = None,
    dry_run: bool = False,
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_id = make_run_id(experiment=experiment, model_name=model.display_name, condition=condition)
    path = runs_root(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "experiment": experiment,
        "condition": condition,
        "created_at_utc": utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dry_run": dry_run,
        "limit": limit,
        "model": {
            "display_name": model.display_name,
            "provider": model.provider,
            "model": model.model,
            "enabled": model.enabled,
            "temperature": model.temperature,
            "max_output_tokens": model.max_output_tokens,
            "base_url_env": model.base_url_env,
            "api_key_env": model.api_key_env,
            "api_mode": model.api_mode,
            "web_search_mode": model.web_search_mode,
            "search_options": model.search_options,
            "web_base_url_env": model.web_base_url_env,
            "web_api_key_env": model.web_api_key_env,
            "web_api_mode": model.web_api_mode,
            "web_search_options": model.web_search_options,
        },
        "config": {k: v for k, v in config.items() if not str(k).startswith("_")},
        "config_hash": sha256_json({k: v for k, v in config.items() if not str(k).startswith("_")}),
        "models_config_hash": sha256_json(models_config),
    }
    if extra:
        metadata["extra"] = extra
    write_json(path / METADATA_FILE, metadata)
    return path


def load_metadata(run_dir: str | Path) -> dict[str, Any]:
    from .paths import load_json

    metadata = load_json(Path(run_dir) / METADATA_FILE)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid run metadata in {run_dir}")
    return metadata


def result_key(row: dict[str, Any]) -> str:
    for key in ("request_id", "variant_id", "item_id", "bundle_id", "triplet_id", "format_group_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Cannot derive result key from row: {row}")


def completed_keys(run_dir: str | Path, filename: str = RESULTS_FILE) -> set[str]:
    out: set[str] = set()
    for row in read_jsonl(Path(run_dir) / filename):
        status = row.get("status")
        # Errors remain retryable on resume. A diagnostic outer record is only
        # complete when every requested variant was written successfully.
        if status == "ok":
            out.add(result_key(row))
    return out


def results_by_key(run_dir: str | Path, filename: str = RESULTS_FILE) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(Path(run_dir) / filename)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[result_key(row)] = row
    return out


def expected_count_from_metadata(run_dir: str | Path, total: int) -> int:
    metadata = load_metadata(run_dir)
    limit = metadata.get("limit")
    if isinstance(limit, int) and limit > 0:
        return min(limit, total)
    return total


def append_result(run_dir: str | Path, row: dict[str, Any], filename: str = RESULTS_FILE) -> None:
    append_jsonl(Path(run_dir) / filename, row)


def write_run_json(run_dir: str | Path, filename: str, value: Any) -> None:
    write_json(Path(run_dir) / filename, value)
