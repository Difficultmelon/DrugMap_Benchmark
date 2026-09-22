from __future__ import annotations

from dragmap_formal.scoring import (
    aggregate_final_score_rows,
    diagnostic_answer_score,
    merge_judge_scores,
    score_answer,
)
from dragmap_formal.statistics import paired_state_metrics


def test_multiple_choice_accepts_label_and_option_text() -> None:
    question = {"options": [{"label": "A", "text": "wrong"}, {"label": "B", "text": "CYP3A4"}]}
    gold = {"answer": {"accepted_answers": ["B", "CYP3A4"]}, "scoring": {"method": "multiple_choice"}}
    assert score_answer(question, gold, "B")[0]
    assert score_answer(question, gold, "CYP3A4")[0]


def test_boolean_and_normalized_exact() -> None:
    bool_gold = {"answer": {"canonical": "true", "accepted_answers": ["true"]}, "scoring": {"method": "boolean_exact"}}
    assert score_answer({}, bool_gold, "Yes")[0]
    text_gold = {"answer": {"accepted_answers": ["200 to <350 g/mol"]}, "scoring": {"method": "normalized_exact"}}
    assert score_answer({}, text_gold, "  200 to <350 g/mol. ")[0]


def test_decimal_answers_use_gold_precision_interval() -> None:
    gold_47 = {
        "answer": {"accepted_answers": ["4.7"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, gold_47, "4.65")[0]
    assert score_answer({}, gold_47, "4.73")[0]
    assert not score_answer({}, gold_47, "4.75")[0]

    gold_025 = {
        "answer": {"accepted_answers": ["0.25"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, gold_025, "0.245")[0]
    assert score_answer({}, gold_025, "0.254")[0]
    assert not score_answer({}, gold_025, "0.255")[0]


def test_integer_answers_remain_exact_numeric_values() -> None:
    gold = {
        "answer": {"accepted_answers": ["2"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, gold, "2.0")[0]
    assert not score_answer({}, gold, "2.4")[0]


def test_numeric_tolerance_does_not_extract_digits_from_entity_strings() -> None:
    gold = {
        "answer": {"accepted_answers": ["COPS5"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, gold, "COPS5")[0]
    assert not score_answer({}, gold, "COPS5_gene_symbol")[0]


def test_multi_bound_text_range_accepts_narrower_and_boundary_ranges() -> None:
    gold = {
        "answer": {"accepted_answers": ["200 to <350 g/mol"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, gold, "200 to <350 g/mol") == (True, "normalized_exact")
    assert score_answer({}, gold, "200-300 g/mol") == (True, "interval_subset")
    assert score_answer({}, gold, "200–300 g/mol") == (True, "interval_subset")
    assert not score_answer({}, gold, "211.2 g/mol")[0]
    assert not score_answer({}, gold, "150-300 g/mol")[0]
    assert score_answer({}, gold, "200-350 g/mol") == (True, "interval_subset")
    assert not score_answer({}, gold, "200-351 g/mol")[0]


def test_numeric_interval_candidate_can_be_a_gold_subset_and_convert_units() -> None:
    gold = {
        "answer": {"accepted_answers": ["200 to <350 g/mol"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, gold, "300-350 Da") == (True, "interval_subset")
    assert score_answer({}, gold, "300 to <350 Da") == (True, "interval_subset")
    assert score_answer({}, gold, "199-300 Da") == (False, "interval_not_subset")
    assert score_answer({}, gold, "300-351 Da") == (False, "interval_not_subset")


def test_open_ended_numeric_ranges_remain_exact_label_scored() -> None:
    high_gold = {
        "answer": {"accepted_answers": [">=500 g/mol"]},
        "scoring": {"method": "normalized_exact"},
    }
    low_gold = {
        "answer": {"accepted_answers": ["<200 g/mol"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, high_gold, ">=500 g/mol")[0]
    assert score_answer({}, low_gold, "<200 g/mol")[0]
    assert not score_answer({}, high_gold, "600")[0]
    assert not score_answer({}, low_gold, "199.9")[0]
    assert not score_answer({}, high_gold, "499.9")[0]
    assert not score_answer({}, low_gold, "200")[0]

    unicode_high_gold = {
        "answer": {"accepted_answers": ["≥500 g/mol"]},
        "scoring": {"method": "normalized_exact"},
    }
    assert score_answer({}, unicode_high_gold, "≥500 g/mol") == (True, "normalized_exact")
    assert not score_answer({}, unicode_high_gold, "500")[0]


def test_answer_format_scores_semantic_text_despite_permuted_label() -> None:
    question = {
        "options": [
            {"label": "A", "option_id": "OPT_WRONG", "text": "CYP2D6"},
            {"label": "B", "option_id": "OPT_RIGHT", "text": "CYP3A4"},
        ]
    }
    correct, method, semantic_key = diagnostic_answer_score(
        prediction="CYP3A4",
        accepted_answers=["C", "CYP3A4"],
        question=question,
        correct_labels={"B"},
    )
    assert correct
    assert method == "diagnostic_normalized_exact"
    assert semantic_key == "semantic:cyp3a4"


def test_answer_format_rejects_original_letter_in_permuted_layout() -> None:
    question = {
        "options": [
            {"label": "A", "option_id": "OPT_WRONG", "text": "CYP2D6"},
            {"label": "B", "option_id": "OPT_RIGHT", "text": "CYP3A4"},
        ]
    }
    correct, method, _ = diagnostic_answer_score(
        prediction="A",
        accepted_answers=["C", "CYP3A4"],
        question=question,
        correct_labels={"B"},
    )
    assert not correct
    assert method == "diagnostic_choice_label_wrong_position"


def test_answer_format_uses_current_layout_for_label_and_semantic_text_for_text() -> None:
    question = {
        "options": [
            {"label": "A", "option_id": "OPT_RIGHT", "text": "CYP3A4"},
            {"label": "B", "option_id": "OPT_WRONG", "text": "CYP2D6"},
        ]
    }
    label_result = diagnostic_answer_score(
        prediction="A",
        accepted_answers=["C", "CYP3A4"],
        question=question,
        correct_labels={"A"},
    )
    text_result = diagnostic_answer_score(
        prediction="CYP3A4",
        accepted_answers=["C", "CYP3A4"],
        question=question,
        correct_labels={"A"},
    )
    assert label_result[0] and label_result[1] == "diagnostic_choice_label"
    assert text_result[0] and text_result[1] == "diagnostic_normalized_exact"
    assert text_result[2] == "semantic:cyp3a4"


def test_paired_state_metrics_rescue_and_harm() -> None:
    closed = [{"id": "a", "correct": False}, {"id": "b", "correct": True}, {"id": "c", "correct": False}]
    open_web = [{"id": "a", "correct": True}, {"id": "b", "correct": False}, {"id": "c", "correct": False}]
    metrics = paired_state_metrics(closed, open_web)
    assert metrics["rescued"] == 1
    assert metrics["harmed"] == 1
    assert metrics["both_wrong"] == 1
    assert metrics["net_gain"] == 0
    assert metrics["web_rescue_rate"] == 0.5
    assert metrics["web_harm_rate"] == 1.0


def test_judge_answer_overrides_only_eligible_final_answer() -> None:
    deterministic = [
        {
            "id": "SA",
            "type": "ontology_entity_normalization",
            "task_format": "short_answer",
            "correct": False,
            "score": 0.0,
            "model_status": "ok",
        },
        {
            "id": "DR",
            "type": "drug_repositioning",
            "task_format": "short_answer",
            "correct": False,
            "score": 0.0,
            "model_status": "ok",
        },
    ]
    judge_rows = [
        {
            "item_id": "SA",
            "status": "ok",
            "answer_evaluated": True,
            "answer_correct": True,
            "explanation_evaluated": False,
        },
        {
            "item_id": "DR",
            "status": "ok",
            "answer_evaluated": False,
            "answer_correct": None,
            "explanation_evaluated": True,
            "explanation_score": 4.0,
            "explanation_scores": {
                "logical_coherence": 4,
                "factual_support": 4,
                "clinical_relevance": 4,
                "conciseness_faithfulness": 4,
            },
        },
    ]
    merged = merge_judge_scores(deterministic, judge_rows)
    by_id = {row["id"]: row for row in merged}
    assert by_id["SA"]["correct"] is True
    assert by_id["SA"]["deterministic_correct"] is False
    assert by_id["SA"]["final_score_source"] == "judge_semantic_answer"
    assert by_id["DR"]["correct"] is False
    assert by_id["DR"]["final_score_source"] == "deterministic_binary_accuracy"
    assert by_id["DR"]["explanation_score"] == 4.0

    summary = aggregate_final_score_rows(merged)
    assert summary["accuracy"] == 0.5
    assert summary["deterministic_accuracy"] == 0.0
    assert summary["judge_answer_evaluated_items"] == 1
    assert summary["explanation_scored_items"] == 1
