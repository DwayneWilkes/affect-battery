"""Task 7.4 Red — hedging codebook with 4 paper §3.4.3 patterns flagged.

Per scoring-pipeline spec MODIFIED "Hedging language codebook with five
categories": the codebook MUST contain 4 patterns flagged
`paper_pattern: true` corresponding to the paper §3.4.3 hedging
categories. Loading the codebook with any of the flagged patterns
missing raises ValueError.
"""

from __future__ import annotations

import pytest

# 4 paper-required pattern names (§3.4.3). These names are stable and
# checked at codebook-load time.
REQUIRED_PAPER_PATTERN_NAMES = {
    "i_think_claim",       # epistemic
    "not_sure",            # uncertainty
    "could_be",            # uncertainty / qualification
    "cant_be_certain",     # confidence_disclaimer
}


class TestHedgingCodebook:
    def test_codebook_has_paper_patterns(self):
        """All 4 paper §3.4.3 patterns are present and flagged
        paper_pattern: true."""
        from src.scoring.hedging import paper_required_patterns

        required = paper_required_patterns()
        for name in REQUIRED_PAPER_PATTERN_NAMES:
            assert name in required, (
                f"missing paper §3.4.3 pattern {name!r} (paper_pattern: true)"
            )

    def test_loader_rejects_missing_paper_pattern(self, tmp_path):
        """If a paper-flagged pattern is removed from the YAML, loader
        raises rather than silently dropping it."""
        from src.scoring.hedging import _load_codebook
        import yaml

        # Build a minimal codebook missing one paper-flagged pattern.
        yaml_path = tmp_path / "broken_codebook.yaml"
        broken = {
            "primary_exclusions": ["RLHF_SAFETY"],
            "categories": {
                "EPISTEMIC": [
                    {"pattern_name": "i_think_claim", "regex": "I think",
                     "edge_case_notes": "x", "paper_pattern": True},
                ],
                # Missing: not_sure, could_be, cant_be_certain
                "UNCERTAINTY": [],
                "QUALIFICATION": [],
                "CONFIDENCE_DISCLAIMER": [],
                "RLHF_SAFETY": [],
            },
        }
        yaml_path.write_text(yaml.safe_dump(broken))
        with pytest.raises(ValueError, match="paper"):
            _load_codebook(yaml_path)
