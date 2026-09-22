from __future__ import annotations

from dragmap_formal.io import append_jsonl
from dragmap_formal.judge import _evaluation_scope, judge_core
from dragmap_formal.paths import write_json


def _write_dataset(root) -> None:
    public = [
        {
            "id": "Q1",
            "type": "unit",
            "subtype": "unit",
            "question": "What is the answer?",
            "task_format": "short_answer",
            "answer_semantics": "unit",
            "group_ids": {
                "knowledge_unit_id": "KU1",
                "entity_cluster_id": "EC1",
                "template_group_id": "TG1",
            },
            "benchmark_version": "5.2",
        }
    ]
    gold = [
        {
            "id": "Q1",
            "answer_semantics": "unit",
            "answer": {
                "canonical": "correct",
                "accepted_answers": ["correct"],
                "canonical_id": None,
                "normalization": {},
            },
            "scoring": {"method": "normalized_exact"},
            "knowledge_unit_id": "KU1",
        }
    ]
    write_json(root / "public" / "questions.json", public)
    write_json(root / "private" / "gold.json", gold)
    for path in (
        root / "diagnostic_sets" / "public" / "compositional_bundles.json",
        root / "diagnostic_sets" / "public" / "counterfactual_triplets.json",
        root / "diagnostic_sets" / "public" / "answer_format_variants.json",
        root / "diagnostic_sets" / "private" / "compositional_bundles_gold.json",
        root / "diagnostic_sets" / "private" / "counterfactual_triplets_gold.json",
        root / "diagnostic_sets" / "private" / "answer_format_variants_gold.json",
    ):
        write_json(path, [])


def test_judge_finds_public_questions_by_id(tmp_path) -> None:
    root = tmp_path / "data"
    for folder in (
        root / "public",
        root / "private",
        root / "diagnostic_sets" / "public",
        root / "diagnostic_sets" / "private",
    ):
        folder.mkdir(parents=True)
    _write_dataset(root)
    config = {
        "dataset_root": str(root),
        "core_questions": "public/questions.json",
        "core_gold": "private/gold.json",
        "diagnostic_public": "diagnostic_sets/public",
        "diagnostic_private": "diagnostic_sets/private",
        "expected_core_size": 1,
        "request_timeout_seconds": 1,
        "max_retries": 0,
    }
    dataset_config = tmp_path / "experiment.json"
    write_json(dataset_config, config)
    models_config = tmp_path / "models.json"
    write_json(
        models_config,
        {
            "judge": {
                "display_name": "GPT-4o-mini",
                "provider": "openai_compatible",
                "model": "gpt-4o-mini",
                "base_url_env": "MISSING_BASE",
                "api_key_env": "MISSING_KEY",
                "enabled": True,
            },
            "test_models": [],
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": "run",
            "config": config,
        },
    )
    append_jsonl(
        run_dir / "results.jsonl",
        {
            "item_id": "Q1",
            "request_id": "Q1",
            "status": "ok",
            "parsed_answer": "correct",
            "parsed_explanation": "short rationale",
        },
    )

    from dragmap_formal.data import Dataset

    dataset = Dataset(config)
    rows, summary = judge_core(
        dataset=dataset,
        run_dir=run_dir,
        models_config_path=models_config,
        dry_run=True,
    )
    assert len(rows) == 1
    assert summary["audited_items"] == 1
    assert rows[0]["item_id"] == "Q1"


def test_judge_includes_reasoning_heavy_types_but_not_unrelated_mcq(tmp_path) -> None:
    root = tmp_path / "data"
    for folder in (
        root / "public",
        root / "private",
        root / "diagnostic_sets" / "public",
        root / "diagnostic_sets" / "private",
    ):
        folder.mkdir(parents=True)

    public = [
        {
            "id": "Q1",
            "type": "unit",
            "subtype": "unit",
            "question": "What is the answer?",
            "task_format": "short_answer",
            "answer_semantics": "unit",
            "group_ids": {"knowledge_unit_id": "KU1"},
            "benchmark_version": "5.2",
        },
        {
            "id": "Q2",
            "type": "drug_repositioning",
            "subtype": "prioritization",
            "question": "Which program should be prioritized?",
            "task_format": "multiple_choice",
            "options": [{"label": "A", "text": "Program A"}],
            "answer_semantics": "choice",
            "group_ids": {"knowledge_unit_id": "KU2"},
            "benchmark_version": "5.2",
        },
        {
            "id": "Q3",
            "type": "physicochemical_property",
            "subtype": "formula",
            "question": "What is the formula?",
            "task_format": "multiple_choice",
            "options": [{"label": "A", "text": "Formula A"}],
            "answer_semantics": "choice",
            "group_ids": {"knowledge_unit_id": "KU3"},
            "benchmark_version": "5.2",
        },
    ]
    gold = [
        {
            "id": row["id"],
            "answer_semantics": row["answer_semantics"],
            "answer": {"canonical": "correct", "accepted_answers": ["correct"], "normalization": {}},
            "scoring": {"method": "normalized_exact"},
            "knowledge_unit_id": row["group_ids"]["knowledge_unit_id"],
        }
        for row in public
    ]
    write_json(root / "public" / "questions.json", public)
    write_json(root / "private" / "gold.json", gold)
    for path in (
        root / "diagnostic_sets" / "public" / "compositional_bundles.json",
        root / "diagnostic_sets" / "public" / "counterfactual_triplets.json",
        root / "diagnostic_sets" / "public" / "answer_format_variants.json",
        root / "diagnostic_sets" / "private" / "compositional_bundles_gold.json",
        root / "diagnostic_sets" / "private" / "counterfactual_triplets_gold.json",
        root / "diagnostic_sets" / "private" / "answer_format_variants_gold.json",
    ):
        write_json(path, [])

    config = {
        "dataset_root": str(root),
        "core_questions": "public/questions.json",
        "core_gold": "private/gold.json",
        "diagnostic_public": "diagnostic_sets/public",
        "diagnostic_private": "diagnostic_sets/private",
        "expected_core_size": 3,
        "request_timeout_seconds": 1,
        "max_retries": 0,
    }
    models_config = tmp_path / "models.json"
    write_json(
        models_config,
        {
            "judge": {
                "display_name": "GPT-4o-mini",
                "provider": "openai_compatible",
                "model": "gpt-4o-mini",
                "base_url_env": "MISSING_BASE",
                "api_key_env": "MISSING_KEY",
                "enabled": True,
            },
            "test_models": [],
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "metadata.json", {"run_id": "run", "config": config})
    for item_id in ("Q1", "Q2", "Q3"):
        append_jsonl(
            run_dir / "results.jsonl",
            {
                "item_id": item_id,
                "request_id": item_id,
                "status": "ok",
                "parsed_answer": "correct",
                "parsed_explanation": "checkable rationale",
            },
        )

    from dragmap_formal.data import Dataset

    rows, summary = judge_core(
        dataset=Dataset(config),
        run_dir=run_dir,
        models_config_path=models_config,
        dry_run=True,
    )
    assert {row["item_id"] for row in rows} == {"Q1", "Q2"}
    assert summary["audited_items"] == 2


def test_judge_scope_separates_answer_and_explanation_evaluation() -> None:
    assert _evaluation_scope(
        {"type": "ontology_entity_normalization", "task_format": "short_answer"}
    ) == (True, False)
    assert _evaluation_scope(
        {"type": "unit_conversion", "task_format": "classification"}
    ) == (True, False)
    assert _evaluation_scope(
        {"type": "drug_repositioning", "task_format": "short_answer"}
    ) == (False, True)
    assert _evaluation_scope(
        {"type": "metabolism_clinical_reasoning", "task_format": "multiple_choice"}
    ) == (False, True)
    assert _evaluation_scope(
        {
            "type": "metabolism_clinical_reasoning",
            "subtype": "reduced_enzyme_activity_exposure",
            "task_format": "short_answer",
        }
    ) == (True, True)
    assert _evaluation_scope(
        {"type": "physicochemical_property", "task_format": "multiple_choice"}
    ) == (False, False)


def test_dry_run_fields_follow_evaluation_scope(tmp_path) -> None:
    root = tmp_path / "data"
    for folder in (
        root / "public",
        root / "private",
        root / "diagnostic_sets" / "public",
        root / "diagnostic_sets" / "private",
    ):
        folder.mkdir(parents=True)

    public = [
        {
            "id": "SA",
            "type": "ontology_entity_normalization",
            "subtype": "entity",
            "question": "What is it?",
            "task_format": "short_answer",
            "answer_semantics": "entity",
            "group_ids": {"knowledge_unit_id": "KU1"},
            "benchmark_version": "5.2",
        },
        {
            "id": "DR",
            "type": "drug_repositioning",
            "subtype": "prioritization",
            "question": "Which program?",
            "task_format": "short_answer",
            "answer_semantics": "choice",
            "group_ids": {"knowledge_unit_id": "KU2"},
            "benchmark_version": "5.2",
        },
        {
            "id": "MC",
            "type": "physicochemical_property",
            "subtype": "formula",
            "question": "Which formula?",
            "task_format": "multiple_choice",
            "options": [{"label": "A", "text": "Formula"}],
            "answer_semantics": "choice",
            "group_ids": {"knowledge_unit_id": "KU3"},
            "benchmark_version": "5.2",
        },
    ]
    gold = [
        {
            "id": row["id"],
            "answer_semantics": row["answer_semantics"],
            "answer": {
                "canonical": "correct",
                "accepted_answers": ["correct"],
                "normalization": {},
            },
            "scoring": {"method": "normalized_exact"},
            "knowledge_unit_id": row["group_ids"]["knowledge_unit_id"],
        }
        for row in public
    ]
    write_json(root / "public" / "questions.json", public)
    write_json(root / "private" / "gold.json", gold)
    for path in (
        root / "diagnostic_sets" / "public" / "compositional_bundles.json",
        root / "diagnostic_sets" / "public" / "counterfactual_triplets.json",
        root / "diagnostic_sets" / "public" / "answer_format_variants.json",
        root / "diagnostic_sets" / "private" / "compositional_bundles_gold.json",
        root / "diagnostic_sets" / "private" / "counterfactual_triplets_gold.json",
        root / "diagnostic_sets" / "private" / "answer_format_variants_gold.json",
    ):
        write_json(path, [])
    config = {
        "dataset_root": str(root),
        "core_questions": "public/questions.json",
        "core_gold": "private/gold.json",
        "diagnostic_public": "diagnostic_sets/public",
        "diagnostic_private": "diagnostic_sets/private",
        "expected_core_size": 3,
        "request_timeout_seconds": 1,
        "max_retries": 0,
    }
    models_config = tmp_path / "models.json"
    write_json(
        models_config,
        {
            "judge": {
                "display_name": "mock-judge",
                "provider": "openai_compatible",
                "model": "mock-judge",
                "base_url_env": "MISSING_BASE",
                "api_key_env": "MISSING_KEY",
                "enabled": True,
            },
            "test_models": [],
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "metadata.json", {"run_id": "run", "config": config})
    for item_id in ("SA", "DR", "MC"):
        append_jsonl(
            run_dir / "results.jsonl",
            {
                "item_id": item_id,
                "request_id": item_id,
                "status": "ok",
                "parsed_answer": "correct",
                "parsed_explanation": "checkable rationale",
            },
        )

    from dragmap_formal.data import Dataset

    rows, summary = judge_core(
        dataset=Dataset(config),
        run_dir=run_dir,
        models_config_path=models_config,
        dry_run=True,
        include_all_formats=True,
    )
    by_id = {row["item_id"]: row for row in rows}
    assert set(by_id) == {"SA", "DR"}
    assert by_id["SA"]["answer_evaluated"] is True
    assert by_id["SA"]["answer_correct"] is True
    assert by_id["SA"]["explanation_evaluated"] is False
    assert by_id["SA"]["explanation_score"] is None
    assert by_id["DR"]["answer_evaluated"] is False
    assert by_id["DR"]["answer_correct"] is None
    assert by_id["DR"]["explanation_evaluated"] is True
    assert by_id["DR"]["explanation_score"] == 3.0
    assert summary["answer_audited_items"] == 1
    assert summary["explanation_scored_items"] == 1
