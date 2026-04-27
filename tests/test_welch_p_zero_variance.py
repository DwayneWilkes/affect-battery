"""Welch's t with degenerate-variance inputs must compute a clean p-value
without scipy precision-loss warnings.

The call site (src/analysis/_effect_size.py::welch_p) is hot during pilot
analysis: at n=5 per condition it's common for one cell to score
all-identical (e.g. baseline=[0.8, 0.8] when 4/5 transfer items are
correct in two runs at the same baseline accuracy). scipy's t-test
implementation flags precision loss in that case via a RuntimeWarning,
which (a) clutters test output, and (b) hints at unreliable p-values
that propagate into family-wise corrections.

The fix: extend the existing zero-variance short-circuit so it covers
the case where one group has zero variance and the other doesn't.
The underlying math is well-defined (Welch reduces to Student's t with
df = nx - 1 when sy^2 = 0), so we can compute it directly without
calling scipy at all in this branch.

Spec: affect-battery-proposal-realignment :: scoring-pipeline.
"""

from __future__ import annotations

import warnings

import pytest

from src.analysis._effect_size import cohens_d, welch_p


class TestWelchPNoSpuriousWarnings:
    def test_zero_variance_baseline_emits_no_runtime_warning(self):
        """The case from the actual exp1a pilot test fixtures:
        treatment has variance, baseline does not. Must NOT emit
        scipy's catastrophic-cancellation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            p = welch_p(treatment=[0.4, 0.2], baseline=[0.8, 0.8])
        # And the p-value is finite + in [0, 1].
        assert 0.0 <= p <= 1.0

    def test_zero_variance_treatment_emits_no_runtime_warning(self):
        """Symmetric inverse: treatment has no variance, baseline does."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            p = welch_p(treatment=[1.0, 1.0], baseline=[0.4, 0.6, 0.5])
        assert 0.0 <= p <= 1.0


class TestWelchPDegenerateCases:
    def test_both_zero_variance_means_differ(self):
        """Perfect separation: both groups constant, means differ → p=0."""
        assert welch_p(treatment=[1.0, 1.0, 1.0], baseline=[0.0, 0.0, 0.0]) == 0.0

    def test_both_zero_variance_means_match(self):
        """Both constant + equal → no signal → p=1."""
        assert welch_p(treatment=[0.5, 0.5], baseline=[0.5, 0.5]) == 1.0

    def test_zero_variance_one_side_recovers_meaningful_p(self):
        """When one group has zero variance, the other has variance,
        and means differ substantially, the p-value should be small
        (not 1.0) because the difference is detectable."""
        # treatment ~ 0.3, baseline = 0.8 (constant). Big mean gap,
        # one group has variance: should yield a small p.
        p = welch_p(treatment=[0.4, 0.2], baseline=[0.8, 0.8])
        assert p < 0.5, f"expected p < 0.5 for clear mean separation; got {p}"

    def test_n_too_small_returns_one(self):
        """One-sample groups can't support a t-test."""
        assert welch_p(treatment=[0.5], baseline=[0.5, 0.6, 0.7]) == 1.0
        assert welch_p(treatment=[0.5, 0.6, 0.7], baseline=[0.5]) == 1.0


class TestWelchPMatchesScipyOnNonDegenerateInputs:
    """Sanity: when both groups have variance, our function still goes
    through scipy and produces the same result it always did.
    Regression guard against accidental drift in the non-degenerate path."""

    def test_normal_inputs_match_scipy(self):
        from scipy.stats import ttest_ind

        treatment = [0.5, 0.6, 0.7, 0.4, 0.55]
        baseline = [0.8, 0.85, 0.75, 0.9, 0.82]
        ours = welch_p(treatment, baseline)
        theirs = float(ttest_ind(treatment, baseline, equal_var=False).pvalue)
        assert ours == pytest.approx(theirs, rel=1e-9)


class TestCohensDDegenerateCases:
    """cohens_d also depends on pooled SD; make sure our existing
    perfect-separation handling still works after the welch_p fix."""

    def test_both_zero_variance_means_differ_returns_signed_inf(self):
        import math
        d = cohens_d(treatment=[1.0, 1.0], baseline=[0.0, 0.0])
        assert math.isinf(d) and d > 0

    def test_both_zero_variance_means_match_returns_zero(self):
        assert cohens_d(treatment=[0.5, 0.5], baseline=[0.5, 0.5]) == 0.0
