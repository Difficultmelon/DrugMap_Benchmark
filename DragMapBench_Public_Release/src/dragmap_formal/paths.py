from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "experiment.json"
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.json"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_dotenv(path: str | Path = ".env") -> None:
    """Load a minimal dotenv file without overwriting process variables."""
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_from_root(root: str | Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def load_experiment_config(path: str | Path = DEFAULT_EXPERIMENT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be an object: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def load_model_registry(path: str | Path = DEFAULT_MODELS_CONFIG) -> dict[str, Any]:
    registry = load_json(path)
    if not isinstance(registry, dict) or not isinstance(registry.get("test_models"), list):
        raise ValueError(f"Invalid model registry: {path}")
    return registry


def dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    config_path = Path(str(config.get("_config_path", "")))
    base = config_path.parent.parent if config_path else PROJECT_ROOT
    root = resolve_from_root(base, config["dataset_root"])
    return {
        "root": root,
        "questions": resolve_from_root(root, config["core_questions"]),
        "gold": resolve_from_root(root, config["core_gold"]),
        "diagnostic_public": resolve_from_root(root, config["diagnostic_public"]),
        "diagnostic_private": resolve_from_root(root, config["diagnostic_private"]),
        "expected_core_size": config.get("expected_core_size", 6200),
    }
