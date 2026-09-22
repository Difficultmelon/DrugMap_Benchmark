from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {".env", "src.zip", "experiment_base_original.json"}
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(password|secret|private_key)\s*[:=]"),
    re.compile(r"(?i)ssh\s+-p\s+\d+"),
]
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
    re.compile(r"/root/"),
]


def text_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "audit_release.py"
        and ".git" not in path.parts
        and path.suffix.lower() in {".py", ".json", ".md", ".toml", ".txt", ".cff", ".yml", ".yaml"}
    ]


def main() -> int:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden file: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in {".jsonl", ".log", ".csv"}:
            errors.append(f"raw/generated artifact: {path.relative_to(ROOT)}")

    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret-like content: {path.relative_to(ROOT)}")
                break
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"absolute/private path: {path.relative_to(ROOT)}")
                break

    config = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    if Path(config["dataset_root"]).is_absolute():
        errors.append("experiment.json uses an absolute dataset_root")
    if not (ROOT / "data" / "public" / "dragmapbench_V5.2_questions.json").exists():
        errors.append("public core question file is missing")

    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"status": "ok", "root": str(ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
