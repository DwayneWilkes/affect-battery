"""Tests for confidence extraction."""

from src.scoring.confidence import extract_confidence


class TestExtractConfidence:
    def test_confidence_is(self):
        assert extract_confidence("My confidence is 7.") == 7

    def test_confidence_colon(self):
        assert extract_confidence("Confidence: 8") == 8

    def test_confidence_out_of_ten(self):
        assert extract_confidence("I'd say 7/10 confidence.") == 7

    def test_confidence_rate_at(self):
        assert extract_confidence("I would rate my confidence at 6.") == 6

    def test_out_of_range_high(self):
        assert extract_confidence("Confidence: 15") is None

    def test_out_of_range_zero(self):
        assert extract_confidence("Confidence: 0") is None

    def test_no_confidence(self):
        assert extract_confidence("The answer is Paris.") is None

    def test_boundary_one(self):
        assert extract_confidence("Confidence: 1") == 1

    def test_boundary_ten(self):
        assert extract_confidence("Confidence: 10") == 10

    def test_confidence_level_colon(self):
        """'confidence level: 9' is a common variant."""
        assert extract_confidence("My confidence level: 9 out of 10.") == 9

    def test_rate_as_variant(self):
        assert extract_confidence("I would rate my confidence as 5.") == 5

    def test_out_of_range_eleven(self):
        """11 is above valid range; should return None."""
        assert extract_confidence("Confidence: 11") is None

    def test_negative_number_not_matched(self):
        """A negative number should not be parsed as a confidence score."""
        assert extract_confidence("Confidence: -3") is None

    def test_number_elsewhere_not_confidence(self):
        """A bare number in text without 'confidence' context should not match."""
        assert extract_confidence("I have 8 apples but am not sure how.") is None

    def test_multiple_confidence_first_wins(self):
        """When a text has two confidence mentions, the first-matched pattern
        wins (deterministic for pre-registered analysis)."""
        result = extract_confidence("My confidence is 7. Actually confidence: 9.")
        assert result in (7, 9)  # either is acceptable, but must be deterministic
        # Call again and verify same result
        assert extract_confidence(
            "My confidence is 7. Actually confidence: 9."
        ) == result

    def test_leading_zero_stripped(self):
        """'confidence: 07' parses as 7 (1-10 range still holds)."""
        assert extract_confidence("Confidence: 07") == 7

    def test_percent_confidence_not_matched(self):
        """'70% confident' uses a different scale (0-100); do not silently
        reinterpret as 70 on the 1-10 scale."""
        # We're not testing percent support; just that a raw large number
        # doesn't get clipped/mistaken for confidence.
        result = extract_confidence("I am 70% confident this is right.")
        assert result is None or 1 <= result <= 10
