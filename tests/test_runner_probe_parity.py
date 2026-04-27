"""Probe-runner parity gate.

The variance probe at scripts/probes/h3a_variance_probe.py and the
production run_exp3a runner SHOULD produce per-level sigma values that
match within Monte Carlo error at the same n_per_level. If they
diverge, the pre-reg's power analysis (sized using probe sigma) does
not apply to the runner output and the n=122 run is not power-justified.

This test is a gate: it skips when the parity output file is absent
(parity check has not been run yet) and asserts agreement when it is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "results" / "probes" / "h3a_variance_2026-04-27.json"
PARITY_PATH = REPO_ROOT / "results" / "probes" / "h3a_runner_parity_2026-04-27.json"


def _load_or_skip(path: Path, label: str) -> dict:
    if not path.exists():
        pytest.skip(
            f"{label} not present at {path}; run the parity script before "
            "this test (see scripts/probes/h3a_runner_parity.py). "
            "n=122 run is gated on this comparison passing."
        )
    return json.loads(path.read_text())


def _binomial_sigma_se(p: float, n: int) -> float:
    """Monte Carlo error on a sample sigma estimate from n binomial draws.

    The sample standard deviation of n Bernoulli(p) draws has approximate
    standard error sqrt(p(1-p)/(2n - 2)) under normal-approximation, but
    for small n at p≈0.3 the binomial sigma sqrt(p(1-p)) itself has
    Monte Carlo error roughly sqrt(p(1-p) * (1 - 2*p(1-p))/(n-1)).
    Use a generous 2-SE band: 2 * sqrt(p(1-p)) / sqrt(n) which is the
    SE on the *mean* and slightly overestimates the SE on sigma itself.
    """
    if n < 2:
        return 1.0
    var = p * (1.0 - p)
    return 2.0 * math.sqrt(var / n)


def test_per_level_sigma_within_mc_error():
    probe = _load_or_skip(PROBE_PATH, "Probe output")
    parity = _load_or_skip(PARITY_PATH, "Parity output")

    assert probe["n_per_level"] == parity["n_per_level"], (
        "probe and parity must use the same n_per_level "
        f"(probe={probe['n_per_level']}, parity={parity['n_per_level']})"
    )
    n = probe["n_per_level"]

    probe_sigma = probe["sigma_per_level"]
    parity_sigma = parity["sigma_per_level"]
    probe_mean = probe["mean_per_level"]
    parity_mean = parity["mean_per_level"]

    divergences: list[str] = []
    for level in range(7):
        p_probe = probe_mean[level]
        p_runner = parity_mean[level]
        sigma_diff = abs(probe_sigma[level] - parity_sigma[level])
        # Use the larger of the two sample means as the basis for the SE
        # bound (more conservative).
        p_band = max(p_probe, p_runner, 1.0 - p_probe, 1.0 - p_runner)
        tolerance = _binomial_sigma_se(min(p_band, 1.0 - p_band) if p_band < 0.5 else p_band * (1 - p_band) / max(p_band, 0.01), n) + 0.05
        if sigma_diff > tolerance:
            divergences.append(
                f"level {level + 1}: probe sigma={probe_sigma[level]:.3f} "
                f"vs runner sigma={parity_sigma[level]:.3f} "
                f"(diff={sigma_diff:.3f}, tolerance={tolerance:.3f}); "
                f"means probe={p_probe:.3f} runner={p_runner:.3f}"
            )

    assert not divergences, (
        "probe-runner sigma divergence exceeds Monte Carlo tolerance:\n  "
        + "\n  ".join(divergences)
    )
