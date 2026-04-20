"""Tests for hedging codebook YAML schema and loader.

Spec (scoring-pipeline, Requirement: Hedging language codebook with five categories):
- 5 categories: EPISTEMIC, UNCERTAINTY, QUALIFICATION, CONFIDENCE_DISCLAIMER, RLHF_SAFETY.
- RLHF_SAFETY is tracked but excluded from the primary hedging metric.
- Patterns loaded from YAML data file (design.md Decision 3: not hardcoded).
- Each pattern has category, pattern_name, regex, edge_case_notes.

Spec task 5.2: ~20 patterns across 5 categories.
"""

from pathlib import Path

import pytest
import yaml


CODEBOOK_PATH = Path(__file__).parent.parent / "configs" / "hedging_codebook.yaml"


@pytest.fixture(scope="module")
def codebook():
    assert CODEBOOK_PATH.exists(), f"Missing codebook: {CODEBOOK_PATH}"
    return yaml.safe_load(CODEBOOK_PATH.read_text())


class TestCodebookSchema:
    def test_codebook_loads(self, codebook):
        assert codebook is not None

    def test_codebook_has_categories_key(self, codebook):
        assert "categories" in codebook

    def test_five_categories_present(self, codebook):
        expected = {
            "EPISTEMIC",
            "UNCERTAINTY",
            "QUALIFICATION",
            "CONFIDENCE_DISCLAIMER",
            "RLHF_SAFETY",
        }
        assert set(codebook["categories"].keys()) == expected

    def test_each_category_has_patterns(self, codebook):
        for category, entries in codebook["categories"].items():
            assert isinstance(entries, list), (
                f"{category}: expected list of patterns, got {type(entries)}"
            )
            assert len(entries) >= 1, f"{category}: empty pattern list"

    def test_each_pattern_has_required_fields(self, codebook):
        required = ("pattern_name", "regex", "edge_case_notes")
        for category, entries in codebook["categories"].items():
            for entry in entries:
                for field in required:
                    assert field in entry, (
                        f"{category}/{entry.get('pattern_name', '?')}: "
                        f"missing {field}"
                    )

    def test_pattern_names_unique_per_category(self, codebook):
        for category, entries in codebook["categories"].items():
            names = [e["pattern_name"] for e in entries]
            assert len(names) == len(set(names)), (
                f"{category}: duplicate pattern_name values"
            )

    def test_total_pattern_count_matches_spec(self, codebook):
        """Spec task 5.2: populate initial codebook with ~20 patterns."""
        total = sum(len(v) for v in codebook["categories"].values())
        assert 15 <= total <= 30, f"Expected ~20 patterns, got {total}"

    def test_primary_exclusion_field_present(self, codebook):
        """Codebook must declare which categories are excluded from primary."""
        assert "primary_exclusions" in codebook
        assert "RLHF_SAFETY" in codebook["primary_exclusions"]


class TestPatternsCompile:
    def test_every_regex_compiles(self, codebook):
        import re
        for category, entries in codebook["categories"].items():
            for entry in entries:
                try:
                    re.compile(entry["regex"])
                except re.error as e:
                    pytest.fail(
                        f"{category}/{entry['pattern_name']}: "
                        f"regex does not compile: {entry['regex']!r}: {e}"
                    )


class TestHedgingLoaderUsesCodebook:
    """After task 5.4 refactor, the hedging module loads patterns from YAML,
    not from hardcoded lists. Verify by checking that patterns known to be
    in the YAML are detected."""

    def test_epistemic_detection_from_loaded_codebook(self):
        from src.scoring.hedging import detect_hedges, HedgeCategory
        matches = detect_hedges("I think the answer is 42.")
        assert any(m.category == HedgeCategory.EPISTEMIC for m in matches)

    def test_rlhf_detection_from_loaded_codebook(self):
        from src.scoring.hedging import detect_hedges, HedgeCategory
        matches = detect_hedges("As an AI, I cannot help.")
        assert any(m.category == HedgeCategory.RLHF_SAFETY for m in matches)


class TestIThinkContextSensitivity:
    """Spec task 5.5: 'I think' followed by a claim is an epistemic hedge;
    'I think about' is not."""

    @pytest.mark.parametrize("text", [
        "I think the capital is Paris.",
        "I think the answer is 42.",
        "I think this approach will work.",
    ])
    def test_i_think_claim_detected(self, text):
        from src.scoring.hedging import detect_hedges
        matches = detect_hedges(text)
        i_think = [m for m in matches if m.pattern_name == "i_think_claim"]
        assert len(i_think) == 1, f"Expected 1 match for {text!r}, got {len(i_think)}"

    @pytest.mark.parametrize("text", [
        "I think about this topic often.",
        "I think about the implications every day.",
        "When I think about physics, I wonder.",
    ])
    def test_i_think_about_not_detected(self, text):
        from src.scoring.hedging import detect_hedges
        matches = detect_hedges(text)
        i_think = [m for m in matches if m.pattern_name == "i_think_claim"]
        assert len(i_think) == 0, f"False positive for {text!r}"


class TestRLHFSeparation:
    """Spec task 5.6: RLHF_SAFETY is tracked but excluded from primary."""

    def test_as_an_ai_is_rlhf_not_primary(self):
        from src.scoring.hedging import hedge_summary
        summary = hedge_summary("As an AI, I cannot perform that task.")
        assert summary["total_rlhf"] >= 1
        # The text has no epistemic/uncertainty/qualification hedges.
        assert summary["total_primary"] == 0

    def test_as_an_ai_and_epistemic_together(self):
        from src.scoring.hedging import hedge_summary
        summary = hedge_summary("As an AI, I think the answer is 42.")
        assert summary["total_rlhf"] >= 1, "Should detect 'As an AI'"
        assert summary["total_primary"] >= 1, "Should detect 'I think' as epistemic"

    def test_i_should_note_is_rlhf(self):
        from src.scoring.hedging import detect_hedges, HedgeCategory
        matches = detect_hedges("I should note that this is approximate.")
        rlhf = [m for m in matches if m.category == HedgeCategory.RLHF_SAFETY]
        assert len(rlhf) >= 1

    def test_primary_total_does_not_include_rlhf(self):
        from src.scoring.hedging import hedge_summary
        # Craft text with only RLHF hedges.
        summary = hedge_summary("As an AI language model, I should note that I cannot provide that.")
        assert summary["total_rlhf"] >= 2
        assert summary["total_primary"] == 0, (
            f"Pure-RLHF text leaked into primary: {summary}"
        )
