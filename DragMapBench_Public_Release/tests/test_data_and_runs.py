from __future__ import annotations

import pytest

from dragmap_formal.data import Dataset
from dragmap_formal.io import append_jsonl
from dragmap_formal.paths import write_json
from dragmap_formal.runs import completed_keys


def _minimal_rows(leak: bool = False) -> tuple[list[dict], list[dict]]:
    questions = []
    gold = []
    for i in range(6200):
        item_id = f"T_{i:04d}"
        question = {
            "id": item_id,
            "type": "unit",
            "subtype": "unit",
            "question": "Question?",
            "task_format": "short_answer",
            "answer_semantics": "entity_attribute",
            "group_ids": {"knowledge_unit_id": f"KU_{i:04d}"},
            "benchmark_version": "5.2",
        }
        if leak and i == 0:
            question["answer"] = "leaked"
        questions.append(question)
        gold.append(
            {
                "id": item_id,
                "answer": {"canonical": "A", "accepted_answers": ["A"]},
                "scoring": {"method": "normalized_exact"},
                "knowledge_unit_id": f"KU_{i:04d}",
            }
        )
    return questions, gold


def test_dataset_rejects_public_answer_leakage(tmp_path) -> None:
    questions, gold = _minimal_rows(leak=True)
    root = tmp_path / "data"
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir(parents=True)
    (root / "diagnostic_sets" / "public").mkdir(parents=True)
    (root / "diagnostic_sets" / "private").mkdir(parents=True)
    write_json(root / "public" / "q.json", questions)
    write_json(root / "private" / "g.json", gold)
    config = {
        "dataset_root": str(root),
        "core_questions": "public/q.json",
        "core_gold": "private/g.json",
        "diagnostic_public": "diagnostic_sets/public",
        "diagnostic_private": "diagnostic_sets/private",
    }
    with pytest.raises(ValueError, match="forbidden keys"):
        Dataset(config)


def test_completed_keys_ignores_error_rows(tmp_path) -> None:
    append_jsonl(tmp_path / "results.jsonl", {"request_id": "ok", "status": "ok"})
    append_jsonl(tmp_path / "results.jsonl", {"request_id": "retry", "status": "error"})
    assert completed_keys(tmp_path) == {"ok"}

