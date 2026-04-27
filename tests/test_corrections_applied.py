"""Family-wise correction across all experiments.

Per power-analysis spec "Family-wise error correction" + :
apply Holm-Bonferroni to the primary family (H1, H2, H3a, H4, H1b-TOST)
and BH-FDR to the exploratory family. Both raw and corrected p-values
appear in reports.
"""

from __future__ import annotations


PRIMARY_HYPOTHESES = ("H1", "H2", "H3a", "H4", "H1b_TOST")


class TestApplyFamilyCorrections:
    def test_primary_family_holm_bonferroni(self):
        from src.analysis.stats.corrections import apply_family_corrections

        p_values_by_hypothesis = {
            "H1": 0.001,
            "H2": 0.04,
            "H3a": 0.02,
            "H4": 0.10,
            "H1b_TOST": 0.005,
            "H_exploratory_1": 0.03,
            "H_exploratory_2": 0.045,
        }
        family_membership = {h: "primary" for h in PRIMARY_HYPOTHESES}
        family_membership["H_exploratory_1"] = "exploratory"
        family_membership["H_exploratory_2"] = "exploratory"

        result = apply_family_corrections(
            p_values_by_hypothesis, family_membership,
        )

        # Each hypothesis has both raw + corrected p
        for h in p_values_by_hypothesis:
            assert h in result
            assert "raw" in result[h]
            assert "corrected" in result[h]
            assert "family" in result[h]
            assert "method" in result[h]
        # Primary uses Holm
        assert result["H1"]["method"] == "holm"
        # Exploratory uses BH
        assert result["H_exploratory_1"]["method"] == "bh"

    def test_corrected_p_values_are_at_least_raw(self):
        """Corrected p-values are always >= raw p-values (correction never
        deflates)."""
        from src.analysis.stats.corrections import apply_family_corrections

        p = {h: 0.01 * (i + 1) for i, h in enumerate(PRIMARY_HYPOTHESES)}
        membership = {h: "primary" for h in PRIMARY_HYPOTHESES}
        result = apply_family_corrections(p, membership)
        for h in p:
            assert result[h]["corrected"] >= result[h]["raw"] - 1e-12
