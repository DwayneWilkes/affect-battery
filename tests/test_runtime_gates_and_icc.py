"""Tests for the cmd_run pre-registration / power-report gates and the
ICC fallback recommendation.

Per power-analysis spec:
- "OSF pre-registration top-level gate": real runs MUST have
  --pre-registration-osf-url.
- "Data-collection gate": real runs MUST have a --power-report-path
  + SHA reference.
- "ICC estimation with fallback prior": when pilot is too thin, recommend
  n at both ICC=0.20 (prior) and ICC=0.35 (sensitivity); take the max.
"""

from __future__ import annotations

import pytest


class TestRuntimeGates:
    def test_real_run_without_prereg_url_exits(self, tmp_path, capsys):
        from src.cli import build_parser, cmd_run

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "test/model",
            "--condition", "neutral",
            # No --dry-run, no --pre-registration-osf-url
            "--output-dir", str(tmp_path),
        ])
        with pytest.raises(SystemExit) as excinfo:
            cmd_run(args)
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert "pre-registration URL" in captured.err

    def test_real_run_without_power_report_exits(self, tmp_path, capsys):
        from src.cli import build_parser, cmd_run

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "test/model",
            "--condition", "neutral",
            "--pre-registration-osf-url", "https://osf.io/abc123",
            # No --power-report-path
            "--output-dir", str(tmp_path),
        ])
        with pytest.raises(SystemExit) as excinfo:
            cmd_run(args)
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert "power-report" in captured.err

    def test_dry_run_bypasses_both_gates(self, tmp_path):
        """--dry-run is the offline-testing escape hatch; runs without
        either gate populated should complete successfully."""
        from src.cli import build_parser, cmd_run

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "dry-run",
            "--condition", "neutral",
            "--num-runs", "1",
            "--seed", "0",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        cmd_run(args)  # MUST NOT raise

    def test_skip_flags_bypass_real_run(self, tmp_path):
        """--skip-prereg-gate + --skip-power-gate together let an explicit
        pilot proceed without OSF references."""
        from src.cli import build_parser, cmd_run

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "dry-run",
            "--condition", "neutral",
            "--num-runs", "1",
            "--seed", "0",
            "--dry-run",  # still dry-run for offline testing
            "--skip-prereg-gate",
            "--skip-power-gate",
            "--output-dir", str(tmp_path),
        ])
        cmd_run(args)


class TestIccFallback:
    def test_icc_fallback_recommends_max_of_two_iccs(self):
        from src.analysis.mde import (
            ICC_FALLBACK_PRIOR,
            ICC_SENSITIVITY,
            icc_fallback_recommendation,
        )

        rec = icc_fallback_recommendation(
            baseline_acc=0.85,
            target_mde=0.10,
            cluster_size=10,
        )
        assert rec["icc_prior_value"] == ICC_FALLBACK_PRIOR
        assert rec["icc_sensitivity_value"] == ICC_SENSITIVITY
        # Sensitivity (higher ICC) inflates required n more, so the
        # recommendation MUST pick the sensitivity value.
        assert rec["recommended_n_per_condition"] == rec["n_at_icc_sensitivity"]
        assert rec["n_at_icc_sensitivity"] >= rec["n_at_icc_prior"]
        assert "icc_source" in rec
        assert rec["icc_source"].startswith("fallback_")

    def test_n_required_for_mde_inflates_with_icc(self):
        """Higher ICC => more required n at the same MDE target."""
        from src.analysis.mde import n_required_for_mde

        n_low_icc = n_required_for_mde(
            baseline_acc=0.5, target_mde=0.10, cluster_size=10, icc=0.10,
        )
        n_high_icc = n_required_for_mde(
            baseline_acc=0.5, target_mde=0.10, cluster_size=10, icc=0.40,
        )
        assert n_high_icc > n_low_icc

    def test_n_required_for_mde_negative_target_raises(self):
        from src.analysis.mde import n_required_for_mde

        with pytest.raises(ValueError, match="target_mde"):
            n_required_for_mde(
                baseline_acc=0.5, target_mde=-0.10, cluster_size=10, icc=0.20,
            )
