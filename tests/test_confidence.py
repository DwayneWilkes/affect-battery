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
