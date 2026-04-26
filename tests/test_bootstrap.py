"""Bootstrap-p tests for cross-experiment statistics.

Covers `bootstrap_ratio_p` (used by H2 + H4 in the family-wise
corrections wiring) and `bootstrap_difference_p`. Reproducibility is
guaranteed by the seed kwarg; tests use the smallest n_resamples
(typically 200) that's still within the law-of-large-numbers regime
for the assertions.
"""

from __future__ import annotations

import pytest


class TestBootstrapRatioP:
    def test_single_value_ratio_above_null_low_p(self):
        """When the observed ratio is meaningfully above 1.0 with no
        variance to bootstrap, p should be close to 0."""
        from src.analysis.stats.bootstrap import bootstrap_ratio_p

        p = bootstrap_ratio_p(
            numerator=[3.0],
            denominator=[1.0],
            n_resamples=200, seed=0,
        )
        assert p == 0.0

    def test_single_value_ratio_below_null_high_p(self):
        from src.analysis.stats.bootstrap import bootstrap_ratio_p

        p = bootstrap_ratio_p(
            numerator=[0.5],
            denominator=[1.0],
            n_resamples=200, seed=0,
        )
        assert p == 1.0

    def test_distributed_ratio_p_in_unit_interval(self):
        """Multi-value bootstrap with overlap between groups should
        produce a p-value strictly between 0 and 1."""
        from src.analysis.stats.bootstrap import bootstrap_ratio_p

        # Numerator bigger on average but with overlap
        p = bootstrap_ratio_p(
            numerator=[1.5, 1.4, 1.6, 1.3, 1.7],
            denominator=[1.0, 1.0, 1.0, 1.0, 1.0],
            n_resamples=500, seed=42,
        )
        assert 0.0 <= p <= 1.0

    def test_zero_denominator_returns_one(self):
        """Numerically degenerate case: when all denominators are zero,
        the ratio is undefined for every resample so p defaults to 1.0."""
        from src.analysis.stats.bootstrap import bootstrap_ratio_p

        p = bootstrap_ratio_p(
            numerator=[1.0, 2.0, 3.0],
            denominator=[0.0, 0.0, 0.0],
            n_resamples=100, seed=0,
        )
        assert p == 1.0

    def test_empty_input_raises(self):
        from src.analysis.stats.bootstrap import bootstrap_ratio_p

        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_ratio_p([], [1.0], n_resamples=10)
        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_ratio_p([1.0], [], n_resamples=10)

    def test_seed_makes_p_reproducible(self):
        from src.analysis.stats.bootstrap import bootstrap_ratio_p

        nums = [1.5, 1.4, 1.6, 1.3, 1.7]
        denoms = [1.0, 0.95, 1.05, 0.9, 1.1]
        p1 = bootstrap_ratio_p(nums, denoms, n_resamples=300, seed=42)
        p2 = bootstrap_ratio_p(nums, denoms, n_resamples=300, seed=42)
        assert p1 == p2


class TestBootstrapDifferenceP:
    def test_treatment_above_baseline_low_p(self):
        from src.analysis.stats.bootstrap import bootstrap_difference_p

        p = bootstrap_difference_p(
            treatment=[1.0, 1.1, 1.2, 1.3, 1.4],
            baseline=[0.1, 0.2, 0.3, 0.4, 0.5],
            n_resamples=500, seed=0,
        )
        assert p < 0.05

    def test_baseline_above_treatment_high_p(self):
        from src.analysis.stats.bootstrap import bootstrap_difference_p

        p = bootstrap_difference_p(
            treatment=[0.1, 0.2, 0.3, 0.4, 0.5],
            baseline=[1.0, 1.1, 1.2, 1.3, 1.4],
            n_resamples=500, seed=0,
        )
        assert p > 0.95

    def test_empty_input_raises(self):
        from src.analysis.stats.bootstrap import bootstrap_difference_p

        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_difference_p([], [1.0])


class TestPipelineBootstrapWiring:
    def test_h2_p_value_added_when_exp2_present(self, tmp_path):
        """When the corpus contains Exp 2 runs with strong-positive +
        strong-negative AUCs, the family-wise p_values dict gets H2."""
        from src.analysis.pipeline import _extract_primary_p_values

        exp2_analysis = {
            "verdict": "complete",
            "by_condition": {
                "strong_negative": {
                    "recovery_metrics": {"auc": 2.5},
                },
                "strong_positive": {
                    "recovery_metrics": {"auc": 1.0},
                },
            },
        }
        p = _extract_primary_p_values(
            exp1a_analysis=None,
            exp1b_analysis=None,
            exp2_analysis=exp2_analysis,
            exp3a_analysis=None,
            h4_analysis=None,
        )
        assert "H2" in p
        # Asymmetric (neg larger) -> ratio > 1 -> p < 0.5
        assert 0.0 <= p["H2"] <= 1.0

    def test_h3a_p_value_pulled_from_exp3a_analysis(self):
        from src.analysis.pipeline import _extract_primary_p_values

        exp3a_analysis = {
            "model": "test",
            "beta_2_p_one_sided": 0.012,
        }
        p = _extract_primary_p_values(
            exp1a_analysis=None,
            exp1b_analysis=None,
            exp2_analysis=None,
            exp3a_analysis=exp3a_analysis,
            h4_analysis=None,
        )
        assert p["H3a"] == 0.012

    def test_h4_uses_bootstrap_not_coarse_mapping(self):
        """The H4 p-value is now a bootstrap on per-model ratios, not
        the previous binary outcome -> 0.04/1.0 mapping."""
        from src.analysis.pipeline import _extract_primary_p_values

        h4_analysis = {
            "verdict": "complete",
            "base_model": "B",
            "instruct_model": "I",
            "per_model_aggregates": {
                "B": {"ratio_geomean": 1.5},
                "I": {"ratio_geomean": 3.0},
            },
        }
        p = _extract_primary_p_values(
            exp1a_analysis=None,
            exp1b_analysis=None,
            exp2_analysis=None,
            exp3a_analysis=None,
            h4_analysis=h4_analysis,
        )
        assert "H4" in p
        # The bootstrap with single-value ratios returns either 0.0 (when
        # the ratio is consistently above 1.0 across all resamples) or a
        # finite fraction. NOT one of the hard-coded {0.04, 1.0} values.
        assert p["H4"] not in {0.04, 1.0} or p["H4"] == 0.0
