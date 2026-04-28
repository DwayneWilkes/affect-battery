"""Tests for numeric answer extraction and scoring.

Spec priority (Requirement: Numeric answer extraction):
    1. Explicit answer markers ("the answer is X", "answer: X")
    2. Boxed answers (\\boxed{X})
    3. Equals sign ("= X")
    4. Last number in response (fallback)
"""

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

    def test_negative_in_fallback(self):
        """Last-number fallback should handle a lone negative number."""
        assert extract_numeric_answer("I subtracted and got -7") == -7.0

    def test_decimal_with_comma_thousands(self):
        """'1,234.56' should parse as 1234.56 (European-style decimals not supported)."""
        assert extract_numeric_answer("The answer is 1,234.56") == 1234.56

    def test_zero_answer(self):
        assert extract_numeric_answer("The answer is 0.") == 0.0

    def test_empty_string(self):
        assert extract_numeric_answer("") is None

    def test_boxed_with_preceding_chain_of_thought(self):
        """Boxed answer should be picked over intermediate equals in chain of thought."""
        assert extract_numeric_answer(
            "First 10 + 5 = 15, then we multiply by 3: \\boxed{45}"
        ) == 45.0

    def test_chain_of_thought_picks_final_total(self):
        """Multi-step calculation with intermediate '=' signs picks the final total.

        Real-world failure: chain-of-thought responses like the calorie
        sum below produce multiple '=' lines. The original implementation
        picked the FIRST '=' match (450) instead of the final total (770).
        """
        response = (
            "Let's add up the calories from each ingredient:\n"
            "- Eggs: 6 eggs × 75 calories = 450 calories\n"
            "- Cheese: 2 oz × 120 calories/oz = 240 calories\n"
            "- Ham: equal amount of cheese = 2 oz × 40 calories/oz = 80 calories\n"
            "Total calories = 450 + 240 + 80 = 770 calories."
        )
        assert extract_numeric_answer(response) == 770.0

    def test_multiple_equals_picks_last(self):
        """When several '=' signs appear, the rightmost one is the final answer."""
        assert extract_numeric_answer("3 + 4 = 7, then 7 * 2 = 14") == 14.0

    def test_running_total_with_arrows(self):
        """Stepwise calculations using '=' as running totals: pick the last."""
        assert extract_numeric_answer(
            "Step 1: 5 + 3 = 8. Step 2: 8 * 2 = 16. Step 3: 16 - 1 = 15."
        ) == 15.0

    def test_total_marker_in_bold(self):
        """Models often emit '**Total = N**' or '**Answer: N**' for the final line."""
        assert extract_numeric_answer(
            "First 6 × 75 = 450, then 2 × 120 = 240. **Total = 690**"
        ) == 690.0


class TestExtractionPriority:
    """Spec: explicit markers > boxed > equals > last number."""

    def test_explicit_marker_beats_boxed(self):
        """When both explicit marker and boxed are present, explicit wins."""
        assert extract_numeric_answer(
            "the answer is 15, which in LaTeX is \\boxed{42}"
        ) == 15.0

    def test_boxed_beats_equals(self):
        """When both boxed and an intermediate equals are present, boxed wins."""
        assert extract_numeric_answer(
            "3 + 4 = 7, and finally \\boxed{42}"
        ) == 42.0

    def test_equals_beats_last_number_fallback(self):
        """When an equals marker is present, it wins over last-number fallback."""
        assert extract_numeric_answer(
            "computing 3 + 4 = 7, and mentioning 9 separately"
        ) == 7.0

    def test_answer_colon_marker(self):
        """'answer: X' marker is explicit and highest priority."""
        assert extract_numeric_answer(
            "Working: 3 + 4 = 7. \\boxed{9}. answer: 42"
        ) == 42.0


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

    def test_substring_match(self):
        """Expected answer appearing as a substring scores 1.0."""
        assert score_factual_qa(
            "After research, the capital I found is Canberra, apparently.",
            "Canberra",
        ) == 1.0

    def test_partial_name_not_substring(self):
        """A fragment that is not a substring of response scores 0.0."""
        assert score_factual_qa(
            "The author was Austen.",
            "Jane Austen",
        ) == 0.0

    def test_numeric_answer_with_extra_text(self):
        """Numeric expected answer matches when extracted number matches."""
        assert score_factual_qa("It happened around 1989, I think.", "1989") == 1.0

    def test_numeric_mismatch(self):
        assert score_factual_qa("It happened around 1985.", "1989") == 0.0

    def test_whitespace_only_response(self):
        assert score_factual_qa("   ", "Canberra") == 0.0

    def test_response_with_punctuation(self):
        """Punctuation shouldn't defeat substring match."""
        assert score_factual_qa(
            "Canberra!", "Canberra",
        ) == 1.0
