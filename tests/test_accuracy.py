"""Tests for numeric answer extraction and scoring.

Spec priority (Requirement: Numeric answer extraction):
    1. Explicit answer markers ("the answer is X", "answer: X")
    2. Boxed answers (\\boxed{X})
    3. Equals sign ("= X")
    4. Last number in response (fallback)
"""

from src.scoring.accuracy import (
    ARITHMETIC_TOLERANCE,
    extract_numeric_answer,
    score_arithmetic,
    score_arithmetic_binary,
    score_factual_qa,
)


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


class TestLatexDigitGroupSeparators:
    """LaTeX digit-group separators must not truncate the answer.

    gpt-5.4-nano writes GSM-Hard answers in display math and groups
    digits with the TeX idioms `{,}` and the thin-space macros
    (`\\,` `\\!` `\\;` `\\:`). Every response quoted here is taken from a
    stored result in results/h3b_2026-05-07 or
    results/h3a_2026-05-10_n122_20reps except where noted.
    """

    def test_brace_separators_inside_boxed(self):
        """h3b_2026-05-07, expected_answer 39652742.0."""
        response = r"**There are \(\boxed{39{,}652{,}742}\) pink gumballs.**"
        assert extract_numeric_answer(response) == 39652742.0

    def test_brace_separators_outside_boxed(self):
        """h3b_2026-05-07, expected_answer 22463288.0."""
        response = r"**Answer: \(22{,}463{,}288\) crabs.**"
        assert extract_numeric_answer(response) == 22463288.0

    def test_brace_separators_bare_token(self):
        """A bare grouped number reaching the last-number fallback."""
        assert extract_numeric_answer(r"22{,}463{,}288") == 22463288.0

    def test_brace_separators_after_equals_in_display_math(self):
        """h3b_2026-05-07, expected_answer 22463288.0."""
        response = (
            "Total for all three:\n"
            r"\[" "\n"
            r"7{,}487{,}762 + 7{,}487{,}758 + 7{,}487{,}768 = 22{,}463{,}288" "\n"
            r"\]"
        )
        assert extract_numeric_answer(response) == 22463288.0

    def test_thin_space_comma_separator(self):
        """h3b_2026-05-07, expected_answer 40556910.0."""
        response = (
            r"\[" "\n"
            r"\text{Father} = 5 \times 8\,111\,382 = 40\,556\,910" "\n"
            r"\]"
        )
        assert extract_numeric_answer(response) == 40556910.0

    def test_thin_space_negative_kern_separator(self):
        """h3a_2026-05-10_n122_20reps: a `\\!` kern splits the digits."""
        response = r"\(870373 \cdot 75 = 65{,}278{,}0\!75\)"
        assert extract_numeric_answer(response) == 65278075.0

    def test_thin_space_medium_separator(self):
        """`\\;` is the same TeX spacing family; not seen in these corpora."""
        assert extract_numeric_answer(r"The answer is 8\;111\;382") == 8111382.0

    def test_thin_space_thick_separator(self):
        """`\\:` is the same TeX spacing family; not seen in these corpora."""
        assert extract_numeric_answer(r"The answer is 8\:111\:382") == 8111382.0


class TestCurrencyMarkerAfterEquals:
    """A currency marker between `=` and the number must not hide it."""

    def test_escaped_dollar_after_equals_in_display_math(self):
        assert extract_numeric_answer(r"\[6 + 5 + 3 = \$14\]") == 14.0

    def test_escaped_dollar_after_equals_real_response(self):
        """h3b_2026-05-07, expected_answer 9731083.0."""
        response = (
            "Total:\n"
            r"\[" "\n"
            r"\$24 + \$9{,}731{,}053 + \$6 = \$9{,}731{,}083" "\n"
            r"\]"
        )
        assert extract_numeric_answer(response) == 9731083.0

    def test_plain_dollar_after_equals_in_bold(self):
        assert extract_numeric_answer("= **$300**") == 300.0

    def test_plain_dollar_after_equals_outside_math(self):
        """h3b_2026-05-07, expected_answer 5631305.0.

        No display math anywhere in this response; every `=` is followed
        by a bare `$`, so the equals tier finds nothing and the fallback
        lands on an intermediate operand.
        """
        response = (
            "He received from the first bank: **$4000**.  \n"
            "From the second bank: **twice as much as the first**, "
            "so **$2 × 4000 = $8000**.\n\n"
            "Total added to his capital: **$4000 + $8000 = $12000**.  \n\n"
            "Initial capital: **$5,619,305**  \n"
            "New capital: **$5,619,305 + $12,000 = $5,631,305**.\n\n"
            "✅ **He has $5,631,305 in capital now.**"
        )
        assert extract_numeric_answer(response) == 5631305.0


    def test_spaced_dollar_is_a_math_delimiter_not_currency(self):
        """h3b_qwen3_2026-07-22, expected_answer -22868213.0.

        Qwen wraps inline math in `$ ... $`. A `$` separated from its
        digits by a space is a delimiter, never a currency marker, so
        stripping it would expose an operand ("Total = $ 22 + ...") to
        the explicit-marker tier and hide the final answer.
        """
        response = (
            "### Step 3: Total cost of cheese  \n"
            "- Total = $ 22 + 22,868,241 = 22,868,263 $ dollars\n\n"
            "### Step 4: Subtract total cost from initial amount  \n"
            "- Amor starts with $50  \n"
            "- Money left = $ 50 - 22,868,263 = -22,868,213 $ dollars\n\n"
            "### Final Answer:\n"
            "Amor will have **–22,868,213** left."
        )
        assert extract_numeric_answer(response) == -22868213.0


class TestNoNumberGuard:
    def test_marker_with_no_number_returns_none(self):
        """A marker followed by a stray comma must return None, not raise.

        The number token has to start with a digit; otherwise a bare
        comma reaches float() and raises ValueError.
        """
        assert extract_numeric_answer("Total = , see the table above.") is None

    def test_prose_with_no_digits_returns_none(self):
        assert extract_numeric_answer("I could not work this one out.") is None


class TestSeparatorRegressionPins:
    """Pins for the forms that already worked before the LaTeX fix."""

    def test_ascii_comma_grouping(self):
        assert extract_numeric_answer("The answer is 22,463,288") == 22463288.0

    def test_plain_boxed(self):
        assert extract_numeric_answer(r"The answer is \boxed{42}") == 42.0


class TestScoreArithmetic:
    def test_correct(self):
        assert score_arithmetic("The answer is 42.", 42.0) is True

    def test_incorrect(self):
        assert score_arithmetic("The answer is 43.", 42.0) is False

    def test_no_answer(self):
        assert score_arithmetic("I don't know", 42.0) is False

    def test_close_float(self):
        assert score_arithmetic("The answer is 42.001", 42.0) is True


class TestScoreArithmeticBinary:
    """Shared binary scorer: the runner and the analysis scripts all need
    a str-valued `expected` scored at the same tolerance."""

    def test_tolerance_is_one_hundredth(self):
        assert ARITHMETIC_TOLERANCE == 0.01

    def test_correct_returns_one(self):
        assert score_arithmetic_binary("The answer is 42.", "42") == 1

    def test_incorrect_returns_zero(self):
        assert score_arithmetic_binary("The answer is 43.", "42") == 0

    def test_no_number_returns_zero(self):
        assert score_arithmetic_binary("I don't know", "42") == 0

    def test_float_expected_string(self):
        assert score_arithmetic_binary("The answer is 2796088", "2796088.0") == 1

    def test_unparseable_expected_returns_zero(self):
        assert score_arithmetic_binary("The answer is 42.", "not a number") == 0

    def test_none_expected_returns_zero(self):
        assert score_arithmetic_binary("The answer is 42.", None) == 0

    def test_inside_tolerance(self):
        assert score_arithmetic_binary("The answer is 42.005", "42") == 1

    def test_outside_tolerance(self):
        assert score_arithmetic_binary("The answer is 42.5", "42") == 0

    def test_accepts_numeric_expected(self):
        assert score_arithmetic_binary("The answer is 42.", 42.0) == 1


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
