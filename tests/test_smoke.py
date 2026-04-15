"""Smoke tests: verify the harness modules import and basic functions work."""

from src.conditioning.prompts import Condition, FEEDBACK_TEMPLATES, INTENSITY_LEVELS
from src.conditioning.tasks import get_arithmetic_problems, get_transfer_tasks
from src.scoring.accuracy import extract_numeric_answer, score_arithmetic
from src.scoring.hedging import detect_hedges, hedge_summary, HedgeCategory
from src.scoring.confidence import extract_confidence
from src.scoring.diversity import lexical_diversity


def test_conditions_defined():
    assert len(Condition) == 6
    assert Condition.STRONG_POSITIVE in Condition
    assert Condition.ACCURATE_NEGATIVE in Condition


def test_feedback_templates_exist():
    for cond in [Condition.STRONG_POSITIVE, Condition.MILD_NEGATIVE, 
                 Condition.STRONG_NEGATIVE, Condition.NEUTRAL, Condition.ACCURATE_NEGATIVE]:
        assert cond in FEEDBACK_TEMPLATES


def test_intensity_levels():
    assert len(INTENSITY_LEVELS) == 7
    assert INTENSITY_LEVELS[0].level == 1
    assert INTENSITY_LEVELS[6].level == 7


def test_arithmetic_problems():
    problems = get_arithmetic_problems(5, seed=42)
    assert len(problems) == 5
    # Same seed = same problems
    problems2 = get_arithmetic_problems(5, seed=42)
    assert [p.question for p in problems] == [p.question for p in problems2]


def test_transfer_tasks():
    tasks = get_transfer_tasks("factual_qa", 3, seed=42)
    assert len(tasks) == 3
    assert all(t.task_type == "factual_qa" for t in tasks)


def test_extract_numeric_answer():
    assert extract_numeric_answer("The answer is 42.") == 42.0
    assert extract_numeric_answer("= 82") == 82.0
    assert extract_numeric_answer("No numbers here") is None
    assert extract_numeric_answer("First 10, then 20, final 30") == 30.0


def test_score_arithmetic():
    assert score_arithmetic("The answer is 42.", 42.0) is True
    assert score_arithmetic("The answer is 43.", 42.0) is False
    assert score_arithmetic("I don't know", 42.0) is False


def test_hedging_detection():
    matches = detect_hedges("I think the answer is Paris, but I'm not sure.")
    categories = {m.category for m in matches}
    assert HedgeCategory.EPISTEMIC in categories
    assert HedgeCategory.UNCERTAINTY in categories


def test_hedging_rlhf_separation():
    summary = hedge_summary("As an AI, I think the answer might be 42.")
    assert summary["counts"]["rlhf_safety"] > 0
    assert summary["counts"]["epistemic"] > 0
    # RLHF excluded from primary total
    assert summary["total_rlhf"] > 0
    assert summary["total_primary"] > 0


def test_confidence_extraction():
    assert extract_confidence("My confidence is 7 out of 10") == 7
    assert extract_confidence("confidence: 8") == 8
    assert extract_confidence("No confidence mentioned") is None
    assert extract_confidence("confidence: 15") is None  # out of range


def test_lexical_diversity():
    texts = ["The cat sat on the mat", "A dog ran in the park"]
    result = lexical_diversity(texts)
    assert "unique_2gram_ratio" in result
    assert result["unique_2gram_ratio"] > 0
