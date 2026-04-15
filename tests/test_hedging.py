"""Tests for hedging language detection."""

from src.scoring.hedging import detect_hedges, hedge_summary, HedgeCategory


class TestDetectHedges:
    def test_epistemic_i_think(self):
        matches = detect_hedges("I think the answer is Paris.")
        assert any(m.category == HedgeCategory.EPISTEMIC for m in matches)

    def test_i_think_about_not_hedge(self):
        """'I think about' is not an epistemic hedge."""
        matches = detect_hedges("I think about this topic often.")
        epistemic = [m for m in matches if m.pattern_name == "i_think_claim"]
        assert len(epistemic) == 0

    def test_perhaps(self):
        matches = detect_hedges("Perhaps the answer is 42.")
        assert any(m.category == HedgeCategory.EPISTEMIC for m in matches)

    def test_uncertainty_not_sure(self):
        matches = detect_hedges("I'm not sure about this.")
        assert any(m.category == HedgeCategory.UNCERTAINTY for m in matches)

    def test_uncertainty_am_not_sure(self):
        matches = detect_hedges("I am not sure about this.")
        assert any(m.category == HedgeCategory.UNCERTAINTY for m in matches)

    def test_qualification(self):
        matches = detect_hedges("To some extent this is true.")
        assert any(m.category == HedgeCategory.QUALIFICATION for m in matches)

    def test_rlhf_safety(self):
        matches = detect_hedges("As an AI, I cannot help with that.")
        assert any(m.category == HedgeCategory.RLHF_SAFETY for m in matches)

    def test_multiple_categories(self):
        text = "As an AI, I think the answer might be 42, but I'm not sure."
        matches = detect_hedges(text)
        categories = {m.category for m in matches}
        assert HedgeCategory.RLHF_SAFETY in categories
        assert HedgeCategory.EPISTEMIC in categories
        assert HedgeCategory.UNCERTAINTY in categories

    def test_no_hedges(self):
        matches = detect_hedges("The answer is 42.")
        assert len(matches) == 0


class TestHedgeSummary:
    def test_rlhf_excluded_from_primary(self):
        summary = hedge_summary("As an AI, I think the answer is 42.")
        assert summary["total_rlhf"] > 0
        assert summary["total_primary"] > 0
        # Primary should not include RLHF count
        assert summary["total_primary"] < summary["total_rlhf"] + summary["total_primary"]

    def test_normalization(self):
        # 200 words, 4 hedges = 2.0 per 100 words
        text = "I think perhaps maybe " + "word " * 196 + "I'm not sure could be"
        summary = hedge_summary(text)
        assert summary["word_count"] > 0
        assert summary["normalized_per_100_words"] > 0

    def test_empty_text(self):
        summary = hedge_summary("")
        assert summary["total_primary"] == 0
        assert summary["word_count"] == 0 or summary["normalized_per_100_words"] == 0
