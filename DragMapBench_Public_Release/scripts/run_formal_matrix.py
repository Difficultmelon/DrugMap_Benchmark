from __future__ import annotations

"""Single entry point for the formal DragMapBench experiment matrix.

This script is the submission-facing runner for the main experiment and experiments 1-4:

main. 13-model, 6,200-question closed-book baseline
1. the same core questions with open-web access
2. compositional diagnostics
3. counterfactual diagnostics
4. answer-format diagnostics

The scientific implementations live in ``src/dragmap_formal``.  This file
consolidates the orchestration that had accumulated across multiple ad-hoc
scripts: selecting models, optional sharding, merging shard outputs, scoring,
and writing a single status log.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
RUNS = PROJECT / "runs"
REPORTS = PROJECT / "reports"
PYTHON = sys.executable

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dragmap_formal.data import Dataset  # noqa: E402
from dragmap_formal.paths import (  # noqa: E402
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    load_dotenv,
    load_experiment_config,
    load_model_registry,
    write_json,
)
from dragmap_formal.registry import ModelSpec, find_test_model  # noqa: E402
from dragmap_formal.runs import create_run_dir, load_metadata  # noqa: E402


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    run_command: str
    score_command: str
    merge_key: str
    id_file_arg: str
    condition: str | None = None


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "main": ExperimentSpec(
        name="main",
        run_command="run-core",
        score_command="score-core",
        merge_key="item_id",
        id_file_arg="--item-ids-file",
        condition="closed_book",
    ),
    "experiment1": ExperimentSpec(
        name="experiment1",
        run_command="run-core",
        score_command="score-core",
        merge_key="item_id",
        id_file_arg="--item-ids-file",
        condition="open_web",
    ),
    "core": ExperimentSpec(
        name="core",
        run_command="run-core",
        score_command="score-core",
        merge_key="item_id",
        id_file_arg="--item-ids-file",
        condition="closed_book",
    ),
    "compositional": ExperimentSpec(
        name="compositional",
        run_command="run-compositional",
        score_command="score-compositional",
        merge_key="bundle_id",
        id_file_arg="--item-ids-file",
    ),
    "counterfactual": ExperimentSpec(
        name="counterfactual",
        run_command="run-counterfactual",
        score_command="score-counterfactual",
        merge_key="triplet_id",
        id_file_arg="--item-ids-file",
    ),
    "answer_format": ExperimentSpec(
        name="answer_format",
        run_command="run-answer-format",
        score_command="score-answer-format",
        merge_key="format_group_id",
        id_file_arg="--group-ids-file",
    ),
}

EXPERIMENT_ALIASES = {
    "main": "main",
    "baseline": "main",
    "closed-book": "main",
    "closedbook": "main",
    "0": "main",
    "1": "experiment1",
    "experiment1": "experiment1",
    "exp1": "experiment1",
    "open-web": "experiment1",
    "openweb": "experiment1",
    "core": "core",
    "2": "compositional",
    "experiment2": "compositional",
    "exp2": "compositional",
    "compositional": "compositional",
    "3": "counterfactual",
    "experiment3": "counterfactual",
    "exp3": "counterfactual",
    "counterfactual": "counterfactual",
    "4": "answer_format",
    "experiment4": "answer_format",
    "exp4": "answer_format",
    "answer_format": "answer_format",
    "answer-format": "answer_format",
}


def slug(value: str) -> str:
    out = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "run"


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_experiments(value: str) -> list[str]:
    selected: list[str] = []
    for raw in split_csv(value):
        key = raw.strip().casefold().replace("_", "-")
        canonical = EXPERIMENT_ALIASES.get(key)
        if canonical is None:
            raise ValueError(f"Unknown experiment: {raw}")
        if canonical not in selected:
            selected.append(canonical)
    return selected


def shard(values: list[str], count: int) -> list[list[str]]:
    return [values[index::count] for index in range(count)]


def latest_by_key(path: Path, key_name: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = row.get(key_name) or row.get("request_id")
        if key:
            latest[str(key)] = row
    return latest


def run_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(SRC), env.get("PYTHONPATH", "")) if item
    )
    env.setdefault("DRUGMAP_CONCURRENCY", "1")
    if extra:
        env.update(extra)
    return env


class Controller:
    def __init__(self, status_path: Path, log_path: Path) -> None:
        self.status_path = status_path
        self.log_path = log_path
        self._lock = Lock()
        self.status: dict[str, Any] = {
            "state": "starting",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "project": str(PROJECT),
            "jobs": [],
            "final_runs": {},
        }

    def log(self, message: str) -> None:
        REPORTS.mkdir(parents=True, exist_ok=True)
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self._lock:
            print(line, flush=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def save(self) -> None:
        with self._lock:
            self.status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json(self.status_path, self.status)

    def update_job(self, label: str, **updates: Any) -> None:
        """Update one controller job atomically, including terminal state."""
        with self._lock:
            for job in self.status.get("jobs", []):
                if job.get("label") == label:
                    job.update(updates)
                    break
            self.status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json(self.status_path, self.status)


def cli_command(
    *,
    config_path: Path,
    models_config_path: Path,
    command: str,
    args: list[str],
) -> list[str]:
    return [
        PYTHON,
        "-u",
        "-m",
        "dragmap_formal.cli",
        "--config",
        str(config_path),
        "--models-config",
        str(models_config_path),
        command,
        *args,
    ]


def dataset_ids(dataset: Dataset, experiment: str) -> list[str]:
    if experiment in {"main", "experiment1", "core"}:
        return [str(row["id"]) for row in dataset.questions]
    if experiment == "compositional":
        return [
            str(row["bundle_id"])
            for row in dataset.load_diagnostic("compositional", private=False)
        ]
    if experiment == "counterfactual":
        return [
            str(row["triplet_id"])
            for row in dataset.load_diagnostic("counterfactual", private=False)
        ]
    if experiment == "answer_format":
        return [
            str(row["format_group_id"])
            for row in dataset.load_diagnostic("answer_format", private=False)
        ]
    raise ValueError(f"Unknown experiment: {experiment}")


def make_run_dir(
    *,
    root: Path,
    experiment: str,
    model: ModelSpec,
    condition: str | None,
    config: dict[str, Any],
    models_config: dict[str, Any],
    ids: list[str],
    shard_index: int | None,
    shard_count: int | None,
    dry_run: bool,
) -> Path:
    extra: dict[str, Any] = {
        "formal_matrix_runner": True,
        "item_ids": ids,
    }
    if shard_index is not None and shard_count is not None:
        extra["shard_index"] = shard_index
        extra["shard_count"] = shard_count
    return create_run_dir(
        experiment=experiment,
        model=model,
        condition=condition,
        config=config,
        models_config=models_config,
        limit=len(ids) if dry_run else None,
        dry_run=dry_run,
        root=root,
        extra=extra,
    )


def progress(run_dir: Path, experiment: str, expected_ids: list[str]) -> dict[str, Any]:
    spec = EXPERIMENTS[experiment]
    latest = latest_by_key(run_dir / "results.jsonl", spec.merge_key)
    expected = set(expected_ids)
    found = {key: latest[key] for key in expected if key in latest}
    ok = sum(row.get("status") == "ok" for row in found.values())
    return {
        "expected": len(expected_ids),
        "unique": len(found),
        "ok": ok,
        "errors": len(found) - ok,
        "missing": len(expected - set(found)),
    }


def run_process(
    *,
    controller: Controller,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
    poll_run_dir: Path | None = None,
    poll_experiment: str | None = None,
    poll_ids: list[str] | None = None,
    poll_seconds: int = 20,
) -> int:
    controller.log(f"start {label}")
    with log_path.open("a", encoding="utf-8") as output:
        output.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {label} =====\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            if poll_run_dir and poll_experiment and poll_ids is not None:
                item = {
                    "label": label,
                    "pid": process.pid,
                    "run_dir": str(poll_run_dir),
                    "progress": progress(poll_run_dir, poll_experiment, poll_ids),
                    "state": "running",
                }
                with controller._lock:
                    controller.status["jobs"] = [
                        existing
                        for existing in controller.status.get("jobs", [])
                        if existing.get("label") != label
                    ] + [item]
                controller.save()
            time.sleep(max(5, poll_seconds))
        return_code = process.returncode or 0
    controller.log(f"finish {label} return_code={return_code}")
    controller.update_job(
        label,
        state="finished" if return_code == 0 else "failed",
        return_code=return_code,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    return return_code


def run_shard(
    *,
    controller: Controller,
    config_path: Path,
    models_config_path: Path,
    model_name: str,
    experiment: str,
    condition: str | None,
    run_dir: Path,
    ids_file: Path,
    ids: list[str],
    env: dict[str, str],
    dry_run: bool,
    poll_seconds: int,
) -> dict[str, Any]:
    spec = EXPERIMENTS[experiment]
    args = ["--model", model_name, "--run-dir", str(run_dir), spec.id_file_arg, str(ids_file)]
    if experiment in {"main", "experiment1", "core"}:
        args += ["--condition", condition or "closed_book"]
    if dry_run:
        args += ["--dry-run"]
    log_path = run_dir / "process.log"
    label = f"{model_name}::{experiment}::{run_dir.name}"
    command = cli_command(
        config_path=config_path,
        models_config_path=models_config_path,
        command=spec.run_command,
        args=args,
    )
    return_code = run_process(
        controller=controller,
        command=command,
        cwd=PROJECT,
        env=env,
        log_path=log_path,
        label=label,
        poll_run_dir=run_dir,
        poll_experiment=experiment,
        poll_ids=ids,
        poll_seconds=poll_seconds,
    )
    return {
        "run_dir": str(run_dir),
        "ids": ids,
        "return_code": return_code,
        "progress": progress(run_dir, experiment, ids),
    }


def merge_shards(
    *,
    model: ModelSpec,
    experiment: str,
    condition: str | None,
    shard_results: list[dict[str, Any]],
    expected_ids: list[str],
    config: dict[str, Any],
    models_config: dict[str, Any],
    output_root: Path,
) -> Path:
    spec = EXPERIMENTS[experiment]
    merged_dir = create_run_dir(
        experiment=experiment,
        model=model,
        condition=condition,
        config=config,
        models_config=models_config,
        limit=None,
        dry_run=False,
        root=output_root,
        extra={
            "formal_matrix_runner": True,
            "merged_from": [item["run_dir"] for item in shard_results],
            "item_ids": expected_ids,
            "shard_count": len(shard_results),
        },
    )
    rows_by_id: dict[str, dict[str, Any]] = {}
    for item in shard_results:
        for key, row in latest_by_key(Path(item["run_dir"]) / "results.jsonl", spec.merge_key).items():
            rows_by_id[key] = row
    ordered = [rows_by_id[item_id] for item_id in expected_ids if item_id in rows_by_id]
    write_jsonl(merged_dir / "results.jsonl", ordered)
    audit = {
        "expected": len(expected_ids),
        "merged": len(ordered),
        "missing": sorted(set(expected_ids) - set(rows_by_id)),
        "unknown": sorted(set(rows_by_id) - set(expected_ids)),
        "status_counts": {
            status: sum(row.get("status") == status for row in ordered)
            for status in sorted({str(row.get("status")) for row in ordered})
        },
    }
    write_json(merged_dir / "formal_matrix_merge_audit.json", audit)
    return merged_dir


def score_run(
    *,
    controller: Controller,
    config_path: Path,
    models_config_path: Path,
    model_name: str,
    experiment: str,
    run_dir: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    spec = EXPERIMENTS[experiment]
    command = cli_command(
        config_path=config_path,
        models_config_path=models_config_path,
        command=spec.score_command,
        args=["--run-dir", str(run_dir)],
    )
    code = run_process(
        controller=controller,
        command=command,
        cwd=PROJECT,
        env=env,
        log_path=log_path,
        label=f"{model_name}::{experiment}::score",
    )
    if code:
        raise RuntimeError(f"Scoring failed for {model_name} {experiment}: {code}")


def run_experiment(
    *,
    controller: Controller,
    config: dict[str, Any],
    config_path: Path,
    models_config: dict[str, Any],
    models_config_path: Path,
    dataset: Dataset,
    model_name: str,
    experiment: str,
    shards: int,
    core_condition: str,
    env: dict[str, str],
    run_root: Path,
    dry_run: bool,
    limit: int | None,
    skip_score: bool,
    poll_seconds: int,
) -> Path:
    model = find_test_model(str(models_config_path), model_name)
    spec = EXPERIMENTS[experiment]
    # The submission protocol fixes the conditions: main is closed-book and
    # experiment1 is open-web. Keep the legacy argument for CLI compatibility.
    condition = spec.condition
    all_ids = dataset_ids(dataset, experiment)
    if limit is not None:
        all_ids = all_ids[:limit]
    if dry_run and limit is None:
        all_ids = all_ids[:1]
    shards = max(1, min(shards, len(all_ids) or 1))
    model_slug = slug(model_name)
    exp_root = run_root / model_slug / experiment
    exp_root.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS / f"formal_matrix_{model_slug}.log"
    controller.log(
        f"{model_name} {experiment}: ids={len(all_ids)} shards={shards} condition={condition}"
    )

    shard_results: list[dict[str, Any]] = []
    pieces = shard(all_ids, shards)
    with ThreadPoolExecutor(max_workers=shards) as executor:
        futures = []
        for index, ids in enumerate(pieces, start=1):
            shard_root = exp_root / f"shard_{index:02d}_of_{shards:02d}"
            run_dir = make_run_dir(
                root=shard_root,
                experiment=experiment,
                model=model,
                condition=condition,
                config=config,
                models_config=models_config,
                ids=ids,
                shard_index=index,
                shard_count=shards,
                dry_run=dry_run,
            )
            ids_file = run_dir / (
                "group_ids.txt" if experiment == "answer_format" else "item_ids.txt"
            )
            ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
            futures.append(
                executor.submit(
                    run_shard,
                    controller=controller,
                    config_path=config_path,
                    models_config_path=models_config_path,
                    model_name=model_name,
                    experiment=experiment,
                    condition=condition,
                    run_dir=run_dir,
                    ids_file=ids_file,
                    ids=ids,
                    env=env,
                    dry_run=dry_run,
                    poll_seconds=poll_seconds,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            shard_results.append(result)
            if result["return_code"]:
                raise RuntimeError(
                    f"{model_name} {experiment} shard failed: {result['run_dir']}"
                )

    final_dir = (
        Path(shard_results[0]["run_dir"])
        if shards == 1
        else merge_shards(
            model=model,
            experiment=experiment,
            condition=condition,
            shard_results=shard_results,
            expected_ids=all_ids,
            config=config,
            models_config=models_config,
            output_root=exp_root / "merged",
        )
    )
    if not skip_score:
        score_run(
            controller=controller,
            config_path=config_path,
            models_config_path=models_config_path,
            model_name=model_name,
            experiment=experiment,
            run_dir=final_dir,
            env=env,
            log_path=log_path,
        )
    controller.status["final_runs"][f"{model_name}::{experiment}"] = str(final_dir)
    controller.save()
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DragMapBench main experiment and experiments 1-4 from one submission-facing file."
    )
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS_CONFIG))
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model display names, or all for every enabled registry model.",
    )
    parser.add_argument(
        "--experiments",
        default="main,experiment1,compositional,counterfactual,answer_format",
        help="Comma-separated layers. Accepts main, experiment1, 2, 3, 4, or legacy aliases.",
    )
    parser.add_argument(
        "--core-condition",
        choices=["closed_book", "open_web"],
        default="closed_book",
        help="Legacy compatibility option; formal main/experiment1 conditions are fixed.",
    )
    parser.add_argument("--core-shards", type=int, default=1)
    parser.add_argument("--compositional-shards", type=int, default=1)
    parser.add_argument("--counterfactual-shards", type=int, default=1)
    parser.add_argument("--answer-format-shards", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-experiment item/group limit, mainly for smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument(
        "--run-root",
        default="",
        help="Optional output root. Defaults to runs/<timestamp>__formal_matrix.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT / ".env")
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    models_config_path = Path(args.models_config)
    config = load_experiment_config(config_path)
    models_config = load_model_registry(models_config_path)
    dataset = Dataset(config)
    models = split_csv(args.models)
    if len(models) == 1 and models[0].casefold() == "all":
        models = [
            str(row["display_name"])
            for row in models_config.get("test_models", [])
            if bool(row.get("enabled", False))
        ]
    if not models:
        raise ValueError("No enabled test models selected")
    experiments = parse_experiments(args.experiments)
    limit = args.limit if args.limit > 0 else None
    run_root = Path(args.run_root) if args.run_root else RUNS / f"{now_stamp()}__formal_matrix"
    run_root.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    controller = Controller(
        status_path=REPORTS / "formal_matrix_status.json",
        log_path=REPORTS / "formal_matrix.log",
    )
    controller.status.update(
        {
            "state": "running",
            "config": str(config_path),
            "models_config": str(models_config_path),
            "models": models,
            "experiments": experiments,
            "run_root": str(run_root),
            "dry_run": bool(args.dry_run),
            "limit": limit,
            "core_condition": args.core_condition,
        }
    )
    controller.save()

    shard_counts = {
        "main": args.core_shards,
        "experiment1": args.core_shards,
        "core": args.core_shards,
        "compositional": args.compositional_shards,
        "counterfactual": args.counterfactual_shards,
        "answer_format": args.answer_format_shards,
    }
    env = run_environment()
    try:
        for model_name in models:
            for experiment in experiments:
                run_experiment(
                    controller=controller,
                    config=config,
                    config_path=config_path,
                    models_config=models_config,
                    models_config_path=models_config_path,
                    dataset=dataset,
                    model_name=model_name,
                    experiment=experiment,
                    shards=shard_counts[experiment],
                    core_condition=args.core_condition,
                    env=env,
                    run_root=run_root,
                    dry_run=bool(args.dry_run),
                    limit=limit,
                    skip_score=bool(args.skip_score),
                    poll_seconds=args.poll_seconds,
                )
        controller.status["state"] = "completed"
        controller.status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        controller.save()
        controller.log("formal matrix completed")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level controller should record failures.
        controller.status["state"] = "failed"
        controller.status["error"] = repr(exc)
        controller.status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        controller.save()
        controller.log(f"formal matrix failed: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
