from dragmap_formal.parse import extract_json_object, parse_answer


def test_parse_answer_repairs_latex_backslashes_without_double_processing() -> None:
    response = r'{"answer": "$\alpha_4\beta_1$ integrin", "explanation": "VLA-4"}'

    parsed = parse_answer(response)

    assert parsed["parse_status"] == "json"
    assert parsed["answer"] == r"$\alpha_4\beta_1$ integrin"


def test_extract_json_object_preserves_valid_json_escapes() -> None:
    response = r'{"answer": "line\nbreak", "explanation": "$\text{HbA1c} \ge 6.5\%$"}'

    parsed = extract_json_object(response)

    assert parsed == {
        "answer": "line\nbreak",
        "explanation": r"$\text{HbA1c} \ge 6.5\%$",
    }
