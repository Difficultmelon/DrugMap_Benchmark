from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Callable

from .data import Dataset
from .experiments import (
    compare_core_runs,
    run_answer_format,
    run_compositional,
    run_counterfactual,
    run_core,
    score_answer_format,
    score_compositional,
    score_counterfactual,
    score_core,
)
from .hashing import sha256_json
from .judge import judge_core
from .paths import DEFAULT_EXPERIMENT_CONFIG, DEFAULT_MODELS_CONFIG, load_dotenv, load_experiment_config, load_model_registry
from .registry import find_test_model, load_specs
from .reporting import write_markdown_summary


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dragmap_formal", description="DragMapBench V5.2 formal experiments")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS_CONFIG))
    return parser


def _limit(value: int) -> int | None:
    return value if value and value > 0 else None


def _dataset(args: argparse.Namespace) -> tuple[dict[str, Any], Dataset]:
    config = load_experiment_config(args.config)
    return config, Dataset(config)


def _model(args: argparse.Namespace, models_path: str):
    model = find_test_model(models_path, args.model)
    if not model.enabled and not args.dry_run:
        raise ValueError(f"Model is disabled in registry: {model.display_name}. Configure endpoint/checkpoint first.")
    return model


def cmd_audit_data(args: argparse.Namespace) -> int:
    config, dataset = _dataset(args)
    profile = dataset.profile()
    profile["dataset_root"] = str(dataset.paths["root"])
    profile["questions_sha256"] = sha256_json(dataset.questions)
    profile["gold_sha256"] = sha256_json(dataset.gold)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.models_config)
    judge, models = load_specs(args.models_config)
    print(f"Judge: {judge.display_name} ({judge.model}) enabled={judge.enabled}")
    for model in models:
        print(f"{model.display_name}\t{model.model}\tprovider={model.provider}\tenabled={model.enabled}")
    return 0


def _env_status(name: str) -> str:
    return "set" if os.environ.get(name, "").strip() else "missing"


def cmd_preflight(args: argparse.Namespace) -> int:
    registry = load_model_registry(args.models_config)
    judge, models = load_specs(args.models_config)
    rows = []
    for spec in [judge, *models]:
        base_status = _env_status(spec.base_url_env) if spec.base_url_env else "not_required"
        key_status = _env_status(spec.api_key_env) if spec.api_key_env else "not_required"
        executable = spec.provider == "openai_compatible" and base_status == "set" and key_status == "set"
        web_base_status = _env_status(spec.web_base_url_env) if spec.web_base_url_env else base_status
        web_key_status = _env_status(spec.web_api_key_env) if spec.web_api_key_env else key_status
        web_executable = (
            spec.provider == "openai_compatible"
            and web_base_status == "set"
            and web_key_status == "set"
        )
        rows.append({
            "display_name": spec.display_name,
            "role": "judge" if spec is judge else "test",
            "provider": spec.provider,
            "model": spec.model,
            "enabled": spec.enabled,
            "base_url": base_status,
            "api_key": key_status,
            "executable": executable,
            "web_search_mode": spec.web_search_mode,
            "web_base_url": web_base_status,
            "web_api_key": web_key_status,
            "web_executable": web_executable,
            "notes": spec.notes,
        })
    search_provider = os.environ.get("DRUGMAP_SEARCH_PROVIDER", "").strip() or "disabled"
    output = {"judge": judge.display_name, "models": rows, "search_provider": search_provider}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_run_core(args: argparse.Namespace) -> int:
    config, dataset = _dataset(args)
    model = _model(args, args.models_config)
    item_ids = None
    if args.item_ids_file:
        item_ids = {
            line.strip()
            for line in Path(args.item_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    path = run_core(
        dataset=dataset,
        config=config,
        models_config_path=args.models_config,
        model=model,
        condition=args.condition,
        limit=_limit(args.limit),
        dry_run=args.dry_run,
        run_dir=args.run_dir,
        question_types={
            item.strip()
            for item in args.types.split(",")
            if item.strip()
        } if args.types else None,
        item_ids=item_ids,
    )
    print(path)
    return 0


def cmd_score_core(args: argparse.Namespace) -> int:
    _, dataset = _dataset(args)
    _, summary = score_core(dataset=dataset, run_dir=args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_compare_core(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    stats = config.get("statistics", {})
    summary = compare_core_runs(
        closed_score_rows=Path(args.closed_run) / "core_score_rows.jsonl",
        open_score_rows=Path(args.open_run) / "core_score_rows.jsonl",
        output_path=Path(args.output),
        bootstrap_replicates=int(stats.get("bootstrap_replicates", 10000)),
        random_seed=int(stats.get("random_seed", 20260911)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_judge_core(args: argparse.Namespace) -> int:
    _, dataset = _dataset(args)
    _, summary = judge_core(
        dataset=dataset,
        run_dir=args.run_dir,
        models_config_path=args.models_config,
        dry_run=args.dry_run,
        include_all_formats=args.include_all_formats,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _diagnostic_command(args: argparse.Namespace, name: str) -> int:
    config, dataset = _dataset(args)
    model = _model(args, args.models_config)
    runner: Callable[..., Path] = {
        "compositional": run_compositional,
        "counterfactual": run_counterfactual,
        "answer_format": run_answer_format,
    }[name]
    runner_args = dict(
        dataset=dataset,
        config=config,
        models_config_path=args.models_config,
        model=model,
        limit=_limit(args.limit),
        dry_run=args.dry_run,
        run_dir=args.run_dir,
    )
    if name == "answer_format":
        runner_args["group_ids_file"] = getattr(args, "group_ids_file", None)
    else:
        runner_args["item_ids_file"] = getattr(args, "item_ids_file", None)
    path = runner(**runner_args)
    print(path)
    return 0


def _diagnostic_score_command(args: argparse.Namespace, name: str) -> int:
    _, dataset = _dataset(args)
    scorer = {
        "compositional": score_compositional,
        "counterfactual": score_counterfactual,
        "answer_format": score_answer_format,
    }[name]
    _, summary = scorer(dataset=dataset, run_dir=args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    config, dataset = _dataset(args)
    model = find_test_model(args.models_config, "GPT-5.6 Terra")
    run_path = run_core(
        dataset=dataset,
        config=config,
        models_config_path=args.models_config,
        model=model,
        condition="closed_book",
        limit=3,
        dry_run=True,
    )
    _, core_summary = score_core(dataset=dataset, run_dir=run_path)
    comp_path = run_compositional(
        dataset=dataset, config=config, models_config_path=args.models_config, model=model, limit=1, dry_run=True
    )
    _, comp_summary = score_compositional(dataset=dataset, run_dir=comp_path)
    cf_path = run_counterfactual(
        dataset=dataset, config=config, models_config_path=args.models_config, model=model, limit=1, dry_run=True
    )
    _, cf_summary = score_counterfactual(dataset=dataset, run_dir=cf_path)
    af_path = run_answer_format(
        dataset=dataset, config=config, models_config_path=args.models_config, model=model, limit=1, dry_run=True
    )
    _, af_summary = score_answer_format(dataset=dataset, run_dir=af_path)
    _, judge_summary = judge_core(
        dataset=dataset,
        run_dir=run_path,
        models_config_path=args.models_config,
        dry_run=True,
    )
    summary = {
        "core": core_summary,
        "compositional": comp_summary,
        "counterfactual": cf_summary,
        "answer_format": af_summary,
        "judge": judge_summary,
        "run_dirs": [str(run_path), str(comp_path), str(cf_path), str(af_path)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _add_run_args(parser: argparse.ArgumentParser, *, model_required: bool = True) -> None:
    parser.add_argument("--model", required=model_required)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = _common_parser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-data")
    sub.add_parser("list-models")
    sub.add_parser("preflight")
    sub.add_parser("smoke-test")
    run_core_parser = sub.add_parser("run-core")
    _add_run_args(run_core_parser)
    run_core_parser.add_argument("--condition", choices=["closed_book", "open_web"], default="closed_book")
    run_core_parser.add_argument(
        "--types",
        help="Comma-separated question types to run, for example drug_repositioning,metabolism_clinical_reasoning",
    )
    run_core_parser.add_argument(
        "--item-ids-file",
        help="Optional newline-delimited item_id allowlist for sharded runs.",
    )
    score_core_parser = sub.add_parser("score-core")
    score_core_parser.add_argument("--run-dir", required=True)
    compare_parser = sub.add_parser("compare-core")
    compare_parser.add_argument("--closed-run", required=True)
    compare_parser.add_argument("--open-run", required=True)
    compare_parser.add_argument("--output", required=True)
    judge_parser = sub.add_parser("judge-core")
    judge_parser.add_argument("--run-dir", required=True)
    judge_parser.add_argument("--dry-run", action="store_true")
    judge_parser.add_argument("--include-all-formats", action="store_true")
    for name in ("compositional", "counterfactual", "answer-format"):
        run_parser = sub.add_parser(f"run-{name}")
        _add_run_args(run_parser)
        if name == "answer-format":
            run_parser.add_argument(
                "--group-ids-file",
                help="Optional newline-delimited format_group_id allowlist for sharded runs.",
            )
        else:
            run_parser.add_argument(
                "--item-ids-file",
                help="Optional newline-delimited diagnostic bundle/triplet ID allowlist for sharded runs.",
            )
        score_parser = sub.add_parser(f"score-{name}")
        score_parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "audit-data":
        return cmd_audit_data(args)
    if args.command == "list-models":
        return cmd_list_models(args)
    if args.command == "preflight":
        return cmd_preflight(args)
    if args.command == "smoke-test":
        return cmd_smoke_test(args)
    if args.command == "run-core":
        return cmd_run_core(args)
    if args.command == "score-core":
        return cmd_score_core(args)
    if args.command == "compare-core":
        return cmd_compare_core(args)
    if args.command == "judge-core":
        return cmd_judge_core(args)
    if args.command.startswith("run-"):
        name = args.command[4:].replace("-", "_")
        return _diagnostic_command(args, name)
    if args.command.startswith("score-"):
        name = args.command[6:].replace("-", "_")
        return _diagnostic_score_command(args, name)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
