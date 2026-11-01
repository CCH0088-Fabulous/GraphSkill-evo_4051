"""Known-answer tests for benchmark scoring rather than model quality."""

from __future__ import annotations

import pytest

from repro.evaluation.metrics import evaluate, exact_match, extract_answer, f1_score, normalize_answer


def test_answer_extraction_uses_last_complete_answer_tag() -> None:
    assert extract_answer("<answer>first</answer>\nReasoning\n<ANSWER> final </ANSWER>") == "final"
    assert extract_answer("Reasoning\n\nlast line\n") == "last line"
    assert extract_answer(" \n ") == ""


def test_squad_normalization_and_multiple_answers() -> None:
    assert normalize_answer(" The, Eiffel   Tower! ") == "eiffel tower"
    assert exact_match("Eiffel Tower", ("the Eiffel tower", "Paris")) == 1
    assert f1_score("red red blue", ("red blue",)) == pytest.approx(0.8)
    assert f1_score("blue red", ("other", "red blue")) == 1
    assert f1_score("", ("answer",)) == 0


def test_hotpot_categorical_mismatch_does_not_receive_partial_credit() -> None:
    assert f1_score("yes indeed", ("yes",), dataset="searchqa") > 0
    assert f1_score("yes indeed", ("yes",), dataset="hotpotqa") == 0
    assert f1_score("no", ("no answer",), dataset="hotpotqa") == 0
    assert f1_score("noanswer", ("noanswer",), dataset="hotpotqa") == 1
    assert f1_score("yes", ("yes",), dataset="hotpotqa") == 1


def test_empty_answers_follow_each_benchmark_rule() -> None:
    assert exact_match("", ("",)) == 1
    assert f1_score("", ("",), dataset="searchqa") == 1
    assert f1_score("", ("",), dataset="hotpotqa") == 0
    with pytest.raises(ValueError):
        f1_score("answer", [])
    with pytest.raises(ValueError):
        exact_match("answer", "answer")


def test_evaluation_reports_extracted_answer_and_fraction_scores() -> None:
    score = evaluate("Explanation.\n<answer>New York</answer>", ("New York City",))
    assert score == {"answer": "New York", "em": 0.0, "f1": 0.8}


def test_searchqa_metrics_match_pinned_upstream() -> None:
    upstream = pytest.importorskip("graphskillevo.envs.searchqa.evaluator")
    cases = [
        ("<answer>The Eiffel Tower!</answer>", ["Eiffel Tower"]),
        ("reasoning\nNew York", ["New York City", "NYC"]),
        ("", [""]),
        ("red red blue", ["red blue"]),
    ]
    for raw, answers in cases:
        ours = evaluate(raw, answers)
        theirs = upstream.evaluate(raw, answers)
        assert ours["answer"] == theirs["predicted_answer"]
        assert ours["em"] == theirs["em"]
        assert ours["f1"] == theirs["f1"]
