"""Tests for the calibration go/no-go gate.

Task group 10 of affect-battery-task-difficulty-calibration.

Covers:
- gate.load_config(path) parses valid YAML into a GateConfig dataclass.
- Missing or empty null_acceptance is an error, not a default.
- Missing/empty/malformed pre_registration_tag or pre_registration_sha is an error.
- The "REPLACE_BEFORE_CALIBRATION" sentinel is a warning, not an error.
- gate.evaluate() is a pure function returning a Verdict with PASS / RECALIBRATE /
  PIPELINE_REGRESSION states, annotating null_accepted when appropriate.
- Verdict carries a config_hash (SHA-256 of raw YAML bytes) for tamper detection.
"""
from __future__ import annotations

import hashlib
import textwrap
import warnings
from pathlib import Path

import pytest

from src.calibration import gate
from src.calibration.gate import (
    GateConfig,
    GateConfigError,
    Verdict,
    VerdictStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_YAML = textwrap.dedent(
    """
    baseline_window:
      min: 0.60
      max: 0.85
      applies_to: at_least_one_of [base, instruct]
    pipeline_sanity:
      easy_regression_delta_floor_pp: 8
    null_acceptance:
      baseline_window: [0.60, 0.85]
      delta_ceiling_pp: 5
      min_n_per_condition: 50
    pre_registration_tag: "gate-prereg-arithmetic_hard_v1-2026-04-20"
    pre_registration_sha: "0123456789abcdef0123456789abcdef01234567"
    """
).strip()


def _write(tmp_path: Path, body: str, name: str = "calibration-gate.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_parses_valid_yaml_into_gateconfig(self, tmp_path: Path) -> None:
        p = _write(tmp_path, VALID_YAML)
        cfg = gate.load_config(p)
        assert isinstance(cfg, GateConfig)
        assert cfg.baseline_window.min == 0.60
        assert cfg.baseline_window.max == 0.85
        assert cfg.pipeline_sanity.easy_regression_delta_floor_pp == 8
        assert cfg.null_acceptance.baseline_window == (0.60, 0.85)
        assert cfg.null_acceptance.delta_ceiling_pp == 5
        assert cfg.null_acceptance.min_n_per_condition == 50
        assert cfg.pre_registration_tag.startswith("gate-prereg-")
        assert len(cfg.pre_registration_sha) == 40

    def test_config_hash_is_sha256_of_file_bytes(self, tmp_path: Path) -> None:
        p = _write(tmp_path, VALID_YAML)
        cfg = gate.load_config(p)
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        assert cfg.config_hash == expected

    def test_missing_null_acceptance_is_error(self, tmp_path: Path) -> None:
        body = textwrap.dedent(
            """
            baseline_window: {min: 0.60, max: 0.85}
            pipeline_sanity: {easy_regression_delta_floor_pp: 8}
            pre_registration_tag: "gate-prereg-x-2026-04-20"
            pre_registration_sha: "0123456789abcdef0123456789abcdef01234567"
            """
        ).strip()
        p = _write(tmp_path, body)
        with pytest.raises(GateConfigError, match="null_acceptance"):
            gate.load_config(p)

    def test_empty_null_acceptance_is_error(self, tmp_path: Path) -> None:
        body = textwrap.dedent(
            """
            baseline_window: {min: 0.60, max: 0.85}
            pipeline_sanity: {easy_regression_delta_floor_pp: 8}
            null_acceptance: {}
            pre_registration_tag: "gate-prereg-x-2026-04-20"
            pre_registration_sha: "0123456789abcdef0123456789abcdef01234567"
            """
        ).strip()
        p = _write(tmp_path, body)
        with pytest.raises(GateConfigError, match="null_acceptance"):
            gate.load_config(p)

    def test_missing_pre_registration_tag_is_error(self, tmp_path: Path) -> None:
        body = "\n".join(
            line for line in VALID_YAML.splitlines()
            if not line.startswith("pre_registration_tag")
        )
        p = _write(tmp_path, body)
        with pytest.raises(GateConfigError, match="pre_registration_tag"):
            gate.load_config(p)

    def test_empty_pre_registration_tag_is_error(self, tmp_path: Path) -> None:
        body = VALID_YAML.replace(
            'pre_registration_tag: "gate-prereg-arithmetic_hard_v1-2026-04-20"',
            'pre_registration_tag: ""',
        )
        p = _write(tmp_path, body)
        with pytest.raises(GateConfigError, match="pre_registration_tag"):
            gate.load_config(p)

    def test_missing_pre_registration_sha_is_error(self, tmp_path: Path) -> None:
        body = "\n".join(
            line for line in VALID_YAML.splitlines()
            if not line.startswith("pre_registration_sha")
        )
        p = _write(tmp_path, body)
        with pytest.raises(GateConfigError, match="pre_registration_sha"):
            gate.load_config(p)

    def test_malformed_pre_registration_sha_is_error(self, tmp_path: Path) -> None:
        body = VALID_YAML.replace(
            '"0123456789abcdef0123456789abcdef01234567"',
            '"not-a-real-sha"',
        )
        p = _write(tmp_path, body)
        with pytest.raises(GateConfigError, match="pre_registration_sha"):
            gate.load_config(p)

    def test_sentinel_placeholder_emits_warning_not_error(self, tmp_path: Path) -> None:
        body = textwrap.dedent(
            """
            baseline_window: {min: 0.60, max: 0.85}
            pipeline_sanity: {easy_regression_delta_floor_pp: 8}
            null_acceptance:
              baseline_window: [0.60, 0.85]
              delta_ceiling_pp: 5
              min_n_per_condition: 50
            pre_registration_tag: "REPLACE_BEFORE_CALIBRATION"
            pre_registration_sha: "REPLACE_BEFORE_CALIBRATION"
            """
        ).strip()
        p = _write(tmp_path, body)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = gate.load_config(p)
        assert cfg.pre_registration_tag == "REPLACE_BEFORE_CALIBRATION"
        assert cfg.pre_registration_sha == "REPLACE_BEFORE_CALIBRATION"
        messages = " ".join(str(w.message) for w in caught)
        assert "REPLACE_BEFORE_CALIBRATION" in messages

    def test_shipped_config_loads(self) -> None:
        """The in-repo configs/calibration-gate.yaml is valid (with sentinels)."""
        repo_root = Path(__file__).resolve().parents[1]
        p = repo_root / "configs" / "calibration-gate.yaml"
        assert p.exists(), f"expected {p} to exist"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = gate.load_config(p)
        assert cfg.null_acceptance.min_n_per_condition >= 1


# ---------------------------------------------------------------------------
# evaluate() — pure function
# ---------------------------------------------------------------------------


def _base_config(tmp_path: Path) -> GateConfig:
    p = _write(tmp_path, VALID_YAML)
    return gate.load_config(p)


def _results(
    *,
    base_baseline: float = 0.72,
    instruct_baseline: float = 0.70,
    easy_regression_delta_pp: float = 14.0,
    manipulation_delta_pp: float = 10.0,
    n_per_condition: int = 50,
) -> dict:
    """Build a calibration_results dict the evaluator consumes."""
    return {
        "no_conditioning_baseline": {
            "base": base_baseline,
            "instruct": instruct_baseline,
        },
        "easy_regression_delta_pp": easy_regression_delta_pp,
        "manipulation_delta_pp": manipulation_delta_pp,
        "n_per_condition": n_per_condition,
    }


class TestEvaluate:
    def test_pass_when_baseline_in_window_and_pipeline_sane(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(_results(), cfg)
        assert verdict.status is VerdictStatus.PASS
        assert not verdict.null_accepted
        assert cfg.config_hash == verdict.config_hash
        assert "baseline" in verdict.justification.lower()

    def test_recalibrate_when_baseline_out_of_window_on_both(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(
            _results(base_baseline=0.91, instruct_baseline=0.95),
            cfg,
        )
        assert verdict.status is VerdictStatus.RECALIBRATE

    def test_pass_when_one_variant_in_window(self, tmp_path: Path) -> None:
        cfg = _base_config(tmp_path)
        # base outside, instruct inside -> PASS (at_least_one_of)
        verdict = gate.evaluate(
            _results(base_baseline=0.95, instruct_baseline=0.72),
            cfg,
        )
        assert verdict.status is VerdictStatus.PASS

    def test_pipeline_regression_when_delta_below_floor(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(
            _results(easy_regression_delta_pp=2.0),
            cfg,
        )
        assert verdict.status is VerdictStatus.PIPELINE_REGRESSION

    def test_pipeline_regression_takes_priority_over_recalibrate(
        self, tmp_path: Path
    ) -> None:
        """If both baseline AND pipeline fail, report pipeline first: bank
        validity cannot be judged with a broken pipeline."""
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(
            _results(
                base_baseline=0.95,
                instruct_baseline=0.95,
                easy_regression_delta_pp=2.0,
            ),
            cfg,
        )
        assert verdict.status is VerdictStatus.PIPELINE_REGRESSION

    def test_null_accepted_when_criteria_met(self, tmp_path: Path) -> None:
        cfg = _base_config(tmp_path)
        # baseline in window, delta tiny (|2| <= 5 pp), n >= 50
        verdict = gate.evaluate(
            _results(manipulation_delta_pp=2.0, n_per_condition=50),
            cfg,
        )
        assert verdict.status is VerdictStatus.PASS
        assert verdict.null_accepted is True
        assert "null" in verdict.justification.lower()

    def test_null_not_accepted_when_n_below_minimum(self, tmp_path: Path) -> None:
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(
            _results(manipulation_delta_pp=2.0, n_per_condition=10),
            cfg,
        )
        assert verdict.status is VerdictStatus.PASS
        assert verdict.null_accepted is False

    def test_null_not_accepted_when_delta_above_ceiling(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(
            _results(manipulation_delta_pp=10.0, n_per_condition=50),
            cfg,
        )
        assert verdict.status is VerdictStatus.PASS
        assert verdict.null_accepted is False

    def test_null_delta_uses_absolute_value(self, tmp_path: Path) -> None:
        """A delta of -3 pp should still count as 'below ceiling'."""
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(
            _results(manipulation_delta_pp=-3.0, n_per_condition=50),
            cfg,
        )
        assert verdict.null_accepted is True

    def test_is_pure_function_identical_inputs_identical_output(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_config(tmp_path)
        results = _results()
        v1 = gate.evaluate(results, cfg)
        v2 = gate.evaluate(results, cfg)
        assert v1 == v2

    def test_evaluate_does_not_mutate_inputs(self, tmp_path: Path) -> None:
        cfg = _base_config(tmp_path)
        results = _results()
        snapshot = {
            "no_conditioning_baseline": dict(results["no_conditioning_baseline"]),
            "easy_regression_delta_pp": results["easy_regression_delta_pp"],
            "manipulation_delta_pp": results["manipulation_delta_pp"],
            "n_per_condition": results["n_per_condition"],
        }
        gate.evaluate(results, cfg)
        assert results["no_conditioning_baseline"] == snapshot["no_conditioning_baseline"]
        assert results["easy_regression_delta_pp"] == snapshot["easy_regression_delta_pp"]
        assert results["manipulation_delta_pp"] == snapshot["manipulation_delta_pp"]
        assert results["n_per_condition"] == snapshot["n_per_condition"]

    def test_verdict_carries_config_hash(self, tmp_path: Path) -> None:
        cfg = _base_config(tmp_path)
        verdict = gate.evaluate(_results(), cfg)
        assert isinstance(verdict, Verdict)
        assert verdict.config_hash == cfg.config_hash
        # Hash reflects the file contents, including pre-reg fields.
        assert len(verdict.config_hash) == 64

    def test_config_hash_changes_when_pre_registration_changes(
        self, tmp_path: Path
    ) -> None:
        """Design recommendation: config_hash INCLUDES pre_registration fields
        so post-hoc pre-reg updates invalidate prior calibration verdicts."""
        p1 = _write(tmp_path, VALID_YAML, name="a.yaml")
        cfg1 = gate.load_config(p1)
        altered = VALID_YAML.replace(
            '"0123456789abcdef0123456789abcdef01234567"',
            '"fedcba9876543210fedcba9876543210fedcba98"',
        )
        p2 = _write(tmp_path, altered, name="b.yaml")
        cfg2 = gate.load_config(p2)
        assert cfg1.config_hash != cfg2.config_hash
