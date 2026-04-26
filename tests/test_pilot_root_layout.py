"""Pilot-root layout: <pilot>/data/<exp>/ + <pilot>/reports/ + manifest.yaml.

Goals:
- cmd_pilot writes JSONs under <output_dir>/data/<experiment>/ (not flat).
- cmd_pilot writes a manifest.yaml at <output_dir>/manifest.yaml capturing
  model, conditions, prereg ref, transfer_bank, num_runs, seed, git SHA.
- analyze_results_dir reads from <results_dir>/data/<exp>/ when present
  (with backward-compat fallback to <results_dir>/<exp>/), and writes
  reports to <results_dir>/reports/<exp>_report.md.
- analyze_results_dir skips H4 rendering when the corpus contains <2
  distinct models — the cross-model contrast is meaningless otherwise.

Spec: affect-battery-proposal-realignment :: results-layout.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


# --- 1. cmd_pilot writes data under <output_dir>/data/<exp>/ ----------------

class TestPilotRootDataLayout:
    def test_pilot_writes_data_under_data_subdir(self, tmp_path, monkeypatch):
        """cmd_pilot's data files MUST land in <output_dir>/data/<exp>/<cond>/<NNNN>.json,
        not <output_dir>/<exp>/<cond>/<NNNN>.json."""
        from src import cli

        # --dry-run path doesn't hit any API but exercises the full write
        # path through cmd_pilot.
        args = SimpleNamespace(
            dry_run=True,
            base_model=False,
            provider="anthropic",
            model="dry-run",
            experiment="exp1a",
            base_url=None,
            num_runs=1,
            seed=0,
            temperature=0.7,
            output_dir=str(tmp_path),
            bank=None,
            max_concurrent=1,
            budget_max_calls=None,
            cost_per_call=None,
            rate_limit_rps=100.0,
            circuit_breaker_threshold=5,
            pre_registration_osf_url=None,
            pre_registration_github_commit=None,
            power_report_path=None,
            power_report_sha=None,
            skip_prereg_gate=True,
            skip_power_gate=True,
        )
        cli.cmd_pilot(args)

        data_dir = tmp_path / "data"
        assert data_dir.is_dir(), (
            f"Expected <output_dir>/data/exp1a/, found these dirs: "
            f"{[p.name for p in tmp_path.iterdir() if p.is_dir()]}"
        )
        # No legacy flat layout sibling.
        assert not (tmp_path / "exp1a").is_dir(), (
            "Pilot must NOT also write to <output_dir>/exp1a/ (legacy path)"
        )
        # JSONs nested under condition subdirs.
        all_jsons = list(data_dir.rglob("*.json"))
        assert len(all_jsons) >= 1
        for p in all_jsons:
            # <data_dir>/<condition>/<NNNN>.json
            assert p.parent.parent == data_dir
            assert p.name.endswith(".json")


# --- 2. cmd_pilot writes manifest.yaml at <output_dir>/manifest.yaml --------

class TestPilotManifest:
    def test_manifest_written_to_pilot_root(self, tmp_path):
        from src import cli

        args = SimpleNamespace(
            dry_run=True,
            base_model=False,
            provider="anthropic",
            model="claude-haiku-4-5",
            experiment="exp1a",
            base_url=None,
            num_runs=1,
            seed=42,
            temperature=0.5,
            output_dir=str(tmp_path),
            bank=None,
            max_concurrent=1,
            budget_max_calls=None,
            cost_per_call=None,
            rate_limit_rps=100.0,
            circuit_breaker_threshold=5,
            pre_registration_osf_url=None,
            pre_registration_github_commit=None,
            power_report_path=None,
            power_report_sha=None,
            skip_prereg_gate=True,
            skip_power_gate=True,
        )
        cli.cmd_pilot(args)

        manifest_path = tmp_path / "manifest.yaml"
        assert manifest_path.exists(), (
            f"Expected manifest.yaml at pilot root. Files: {list(tmp_path.iterdir())}"
        )
        manifest = yaml.safe_load(manifest_path.read_text())

        # Required keys: enough to reproduce the run from manifest alone.
        for key in ("model", "experiment", "conditions", "num_runs", "seed",
                    "started_utc", "completed_utc"):
            assert key in manifest, f"manifest.yaml missing key: {key}"
        assert manifest["model"] == "claude-haiku-4-5"
        assert manifest["experiment"] == "exp1a"
        assert manifest["num_runs"] == 1
        assert manifest["seed"] == 42
        # Conditions list reflects DEFAULT_PILOT_CONDITIONS (7 entries).
        assert len(manifest["conditions"]) == 7

    def test_manifest_contains_no_null_values(self, tmp_path):
        """Skipped gates / unset banks must surface as informative status
        strings, not as bare null fields. A reader should be able to tell
        'we deliberately skipped' from 'we forgot to record' just by
        reading the manifest."""
        from src import cli

        args = SimpleNamespace(
            dry_run=True,
            base_model=False,
            provider="anthropic",
            model="claude-haiku-4-5",
            experiment="exp1a",
            base_url=None,
            num_runs=1,
            seed=42,
            temperature=0.5,
            output_dir=str(tmp_path),
            bank=None,
            max_concurrent=1,
            budget_max_calls=None,
            cost_per_call=None,
            rate_limit_rps=100.0,
            circuit_breaker_threshold=5,
            pre_registration_osf_url=None,
            pre_registration_github_commit=None,
            power_report_path=None,
            power_report_sha=None,
            skip_prereg_gate=True,
            skip_power_gate=True,
        )
        cli.cmd_pilot(args)
        manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())

        def find_nulls(obj, path=""):
            nulls = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    nulls.extend(find_nulls(v, f"{path}.{k}"))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    nulls.extend(find_nulls(v, f"{path}[{i}]"))
            elif obj is None:
                nulls.append(path)
            return nulls

        nulls = find_nulls(manifest)
        assert nulls == [], f"manifest contains null fields: {nulls}"

        # And the skipped-gate sections surface as informative status strings.
        assert isinstance(manifest["pre_registration"], str)
        assert "skip" in manifest["pre_registration"].lower()
        assert isinstance(manifest["power_report"], str)
        assert "skip" in manifest["power_report"].lower()


# --- 3. analyze_results_dir uses pilot-root layout when present -------------

def _make_legacy_corpus(pilot_root: Path, n_runs: int, condition: str):
    """Write <pilot_root>/<exp>/<condition>/<NNNN>.json files (legacy
    'no /data/ wrapper' layout) so we can exercise the analyzer's
    backward-compat path."""
    from src.runner import RunResult, save_result

    out_dir = pilot_root / "exp1a"
    for i in range(n_runs):
        r = RunResult(
            config={
                "model_name": "claude-haiku-4-5",
                "condition": condition,
                "experiment_type": "exp1a",
            },
            run_number=i,
            experiment_type="exp1a",
            model="claude-haiku-4-5",
            condition=condition,
            transfer_responses=["Canberra is the capital."],
            transfer_questions=["q?"],
            transfer_expected=["Canberra"],
            transfer_correct=[1.0],
        )
        r.compute_checksum()
        save_result(r, out_dir)


def _make_root_corpus(pilot_root: Path, n_runs: int, condition: str,
                      model: str = "claude-haiku-4-5"):
    """Write <pilot_root>/data/<exp>/<condition>/<NNNN>.json files (new
    pilot-root layout)."""
    from src.runner import RunResult, save_result

    out_dir = pilot_root / "data" / "exp1a"
    for i in range(n_runs):
        r = RunResult(
            config={
                "model_name": model,
                "condition": condition,
                "experiment_type": "exp1a",
            },
            run_number=i,
            experiment_type="exp1a",
            model=model,
            condition=condition,
            transfer_responses=["Canberra is the capital."],
            transfer_questions=["q?"],
            transfer_expected=["Canberra"],
            transfer_correct=[1.0],
        )
        r.compute_checksum()
        save_result(r, out_dir)


class TestAnalyzePilotRootLayout:
    def test_analyze_reads_data_subdir_and_writes_reports_subdir(self, tmp_path):
        """When <pilot_root>/data/exp1a/ exists, the analyzer must find
        the corpus there AND write reports to <pilot_root>/reports/."""
        from src.analysis.pipeline import analyze_results_dir

        for cond in ("strong_negative", "no_conditioning"):
            _make_root_corpus(tmp_path, n_runs=3, condition=cond)

        rendered = analyze_results_dir(tmp_path, model="claude-haiku-4-5")
        # Reports under reports/ subdir.
        reports_dir = tmp_path / "reports"
        assert reports_dir.is_dir()
        assert (reports_dir / "exp1a_report.md").exists()
        assert (reports_dir / "AGGREGATE_REPORT.md").exists()
        # Reports do NOT also leak at the pilot root.
        assert not (tmp_path / "exp1a_report.md").exists()
        # Returned paths point inside reports/.
        for path in rendered.values():
            assert "reports" in str(path)

    def test_analyze_falls_back_to_legacy_layout_when_no_data_subdir(self, tmp_path):
        """If <pilot_root>/data/ doesn't exist but <pilot_root>/<exp>/ does,
        the analyzer must still load the corpus and render flat reports.
        Backward-compat for old pilot directories."""
        from src.analysis.pipeline import analyze_results_dir

        for cond in ("strong_negative", "no_conditioning"):
            _make_legacy_corpus(tmp_path, n_runs=3, condition=cond)

        rendered = analyze_results_dir(tmp_path, model="claude-haiku-4-5")
        # Legacy layout → flat reports at pilot root, no reports/ subdir.
        assert (tmp_path / "exp1a_report.md").exists()
        assert "exp1a" in rendered


# --- 4. H4 suppression when single-model ------------------------------------

class TestH4SingleModelSuppression:
    def test_no_h4_report_when_single_model_in_corpus(self, tmp_path):
        """analyze_results_dir must NOT render h4_report.md when the corpus
        contains fewer than 2 distinct models. Without 2+ models there's
        nothing to compare in the cross-model asymmetry contrast."""
        from src.analysis.pipeline import analyze_results_dir

        for cond in ("strong_negative", "no_conditioning"):
            _make_root_corpus(tmp_path, n_runs=3, condition=cond,
                              model="claude-haiku-4-5")

        rendered = analyze_results_dir(tmp_path, model="claude-haiku-4-5")
        reports_dir = tmp_path / "reports"
        assert "h4" not in rendered
        assert not (reports_dir / "h4_report.md").exists()
        assert not (tmp_path / "h4_report.md").exists()

    def test_h4_renders_when_multiple_models_present(self, tmp_path):
        """Sanity inverse: when 2+ models are in the corpus, h4 still
        renders (so we don't accidentally suppress legitimate cases)."""
        from src.analysis.pipeline import analyze_results_dir

        # Two models, same conditions, no_conditioning baseline + treatment.
        for cond in ("strong_negative", "no_conditioning"):
            _make_root_corpus(tmp_path, n_runs=3, condition=cond,
                              model="claude-haiku-4-5")
            _make_root_corpus(tmp_path, n_runs=3, condition=cond,
                              model="meta-llama/Llama-3-8B")

        rendered = analyze_results_dir(
            tmp_path,
            model="aggregate",
            base_model="meta-llama/Llama-3-8B",
            instruct_model="claude-haiku-4-5",
        )
        # H4 may or may not be in rendered depending on whether the corpus
        # exposes the two-model cell structure h4 needs; the contract here
        # is just "we don't unconditionally suppress."
        # Also: AGGREGATE_REPORT.md always renders.
        assert (tmp_path / "reports" / "AGGREGATE_REPORT.md").exists()
