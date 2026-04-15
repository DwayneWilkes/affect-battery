"""Tests for numeric answer extraction and scoring."""

from src.scoring.accuracy import extract_numeric_answer, score_arithmetic, score_factual_qa


class TestExtractNumericAnswer:
    def test_explicit_answer_marker(self):
        assert extract_numeric_answer("The answer is 42.") == 42.0

    def test_equals_sign(self):
        assert extract_numeric_answer("45 + 37 = 82") == 82.0

    def test_boxed_latex(self):
        assert extract_numeric_answer("Therefore \\boxed{42}") == 42.0

    def test_last_number_fallback(self):
        assert extract_numeric_answer("First 10, then 20, final 30") == 30.0

    def test_no_number(self):
        assert extract_numeric_answer("I don't know the answer") is None

    def test_negative_number_explicit(self):
        assert extract_numeric_answer("The answer is -15.") == -15.0

    def test_comma_separated(self):
        assert extract_numeric_answer("The answer is 1,234.") == 1234.0

    def test_decimal(self):
        assert extract_numeric_answer("The answer is 3.14.") == 3.14

    def test_range_not_negative(self):
        """Hyphens in ranges like '3-5' should NOT be parsed as negative numbers."""
        result = extract_numeric_answer("between 3-5 possible answers")
        assert result is not None
        assert result >= 0, f"Got {result}, hyphenated range parsed as negative"

    def test_subtraction_in_chain_of_thought(self):
        assert extract_numeric_answer("45 minus 17, which gives 28") == 28.0

    def test_multiple_numbers_with_marker(self):
        """Explicit marker should take priority over last number."""
        assert extract_numeric_answer("I tried 10 and 20 but the answer is 15") == 15.0

    def test_equals_priority_over_fallback(self):
        assert extract_numeric_answer("3 + 4 = 7, and then 9") == 7.0


class TestScoreArithmetic:
    def test_correct(self):
        assert score_arithmetic("The answer is 42.", 42.0) is True

    def test_incorrect(self):
        assert score_arithmetic("The answer is 43.", 42.0) is False

    def test_no_answer(self):
        assert score_arithmetic("I don't know", 42.0) is False

    def test_close_float(self):
        assert score_arithmetic("The answer is 42.001", 42.0) is True


class TestScoreFactualQA:
    def test_exact_match(self):
        assert score_factual_qa("The capital is Canberra.", "Canberra") == 1.0

    def test_case_insensitive(self):
        assert score_factual_qa("CANBERRA is the capital", "Canberra") == 1.0

    def test_no_match(self):
        assert score_factual_qa("The capital is Sydney.", "Canberra") == 0.0

    def test_numeric_match(self):
        assert score_factual_qa("The year was 1989.", "1989") == 1.0

    def test_empty_expected(self):
        """Empty expected answer (creative tasks) should return 0.0, not crash."""
        assert score_factual_qa("Any response here", "") == 0.0
