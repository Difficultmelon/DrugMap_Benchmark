from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .paths import write_json


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    fieldnames: list[str] = []
    for row in rows_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_summary(path: str | Path, title: str, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            lines.append("")
            for sub_key, sub_value in value.items():
                shown = pct(sub_value) if isinstance(sub_value, float) and 0 <= sub_value <= 1 else sub_value
                lines.append(f"- **{sub_key}**: {shown}")
            lines.append("")
        else:
            shown = pct(value) if isinstance(value, float) and 0 <= value <= 1 else value
            lines.append(f"- **{key}**: {shown}")
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def persist_score_outputs(
    *,
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    prefix: str = "score",
) -> None:
    run_path = Path(run_dir)
    write_csv(run_path / f"{prefix}_rows.csv", rows)
    write_json(run_path / f"{prefix}_summary.json", summary)
    write_markdown_summary(run_path / f"{prefix}_summary.md", f"{prefix.replace('_', ' ').title()} Summary", summary)

