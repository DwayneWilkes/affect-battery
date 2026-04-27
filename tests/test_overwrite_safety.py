"""--overwrite flag for pilot/run: resume-by-default, explicit consent
required to move prior pilot data aside.

Design (per safety conversation 2026-04-26):

- No flag: proceed. The cache layer (`is_valid_cached_result` +
  `transfer_bank_hash`) handles per-cell correctness. Identical-config
  re-runs hit cache; differing-config re-runs invalidate stale cells
  per-write.
- `--overwrite`: before starting, move the entire pilot dir to a
  sibling `<dir>.bak.<UTC-timestamp>/` so the prior run's state is
  preserved as an audit trail (recoverable, not deleted). Then create
  a fresh empty pilot dir and run.

Spec: affect-battery-proposal-realignment :: results-layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _base_args(tmp_path: Path, experiment: str = "exp1a", overwrite: bool = False,
               **overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        dry_run=True,
        base_model=False,
        provider="anthropic",
        model="dry-run",
        experiment=experiment,
        base_url=None,
        num_runs=1,
        seed=0,
        temperature=0.7,
        output_dir=str(tmp_path),
        bank=None,
        transfer_bank=None,
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
        runner_config=None,
        neutral_turns=0,
        condition="neutral",
        overwrite=overwrite,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _seed_existing_pilot(pilot_root: Path):
    """Create a non-empty pilot dir resembling a prior run's output."""
    pilot_root.mkdir(parents=True, exist_ok=True)
    (pilot_root / "manifest.yaml").write_text(yaml.safe_dump({"prior": "run"}))
    data_dir = pilot_root / "data" / "exp1a" / "neutral"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "0000.json").write_text(json.dumps({"prior_run_marker": True}))


# --- 1. Resume-by-default: no flag, prior data is preserved on disk ---------

class TestResumeIsDefault:
    def test_no_overwrite_flag_does_not_create_backup(self, tmp_path):
        """When --overwrite is unset, the pilot dir is not moved aside.
        The cache layer handles per-cell correctness."""
        from src import cli

        _seed_existing_pilot(tmp_path)
        args = _base_args(tmp_path, overwrite=False)
        cli.cmd_pilot(args)

        # No .bak.* sibling created.
        siblings = list(tmp_path.parent.iterdir())
        bak_dirs = [
            p for p in siblings
            if p.name.startswith(tmp_path.name + ".bak.")
        ]
        assert bak_dirs == [], (
            f"--overwrite=false should not create backup; got {bak_dirs}"
        )

    def test_no_overwrite_preserves_prior_data_files_that_dont_collide(self, tmp_path):
        """A prior file at a path the new run doesn't write to MUST
        survive the no-flag re-run."""
        from src import cli

        # Prior file in a condition the dry-run pilot won't touch
        # (we'll put it in a fake exp that the new run won't write to).
        unrelated_file = tmp_path / "data" / "exp_legacy" / "0000.json"
        unrelated_file.parent.mkdir(parents=True, exist_ok=True)
        unrelated_file.write_text('{"legacy": true}')

        args = _base_args(tmp_path, experiment="exp1a", overwrite=False)
        cli.cmd_pilot(args)

        # Unrelated prior file untouched.
        assert unrelated_file.exists()
        assert json.loads(unrelated_file.read_text()) == {"legacy": True}


# --- 2. --overwrite moves existing dir to a timestamped backup --------------

class TestOverwriteMovesToBackup:
    def test_overwrite_creates_timestamped_backup(self, tmp_path):
        """With --overwrite, an existing pilot dir is renamed to
        <dir>.bak.<UTC-timestamp>/ before the new run starts. The
        backup contains the prior run's content."""
        from src import cli

        _seed_existing_pilot(tmp_path)
        args = _base_args(tmp_path, overwrite=True)
        cli.cmd_pilot(args)

        siblings = list(tmp_path.parent.iterdir())
        bak_dirs = [
            p for p in siblings
            if p.name.startswith(tmp_path.name + ".bak.") and p.is_dir()
        ]
        assert len(bak_dirs) == 1, (
            f"expected exactly one backup dir; got {bak_dirs}"
        )

        # Backup contains the prior manifest + data files.
        bak = bak_dirs[0]
        assert (bak / "manifest.yaml").exists()
        prior_data = bak / "data" / "exp1a" / "neutral" / "0000.json"
        assert prior_data.exists()
        assert json.loads(prior_data.read_text()) == {"prior_run_marker": True}

    def test_overwrite_starts_fresh_after_backup(self, tmp_path):
        """After the backup move, the new run writes a fresh manifest
        and fresh result files — no prior_run_marker survives in the
        live pilot dir."""
        from src import cli

        _seed_existing_pilot(tmp_path)
        args = _base_args(tmp_path, overwrite=True)
        cli.cmd_pilot(args)

        # Live manifest is the new run's, not the prior stub.
        manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
        assert manifest.get("model") == "dry-run"
        assert manifest.get("experiment") == "exp1a"

        # Prior_run_marker file does NOT exist in the live pilot dir
        # (the dir contents from the prior run were moved to backup).
        live_files = list((tmp_path / "data").rglob("*.json"))
        for p in live_files:
            payload = json.loads(p.read_text())
            assert "prior_run_marker" not in payload, (
                f"live pilot dir contains prior-run residue at {p}"
            )

    def test_overwrite_with_no_existing_dir_is_noop(self, tmp_path):
        """--overwrite when the pilot dir doesn't exist yet must NOT
        create a spurious backup. The flag is idempotent for
        first-time runs."""
        from src import cli

        # Note: tmp_path EXISTS (pytest creates it) but is empty.
        # Use a child path that does NOT exist.
        fresh_root = tmp_path / "fresh_pilot"
        args = _base_args(fresh_root, overwrite=True)
        cli.cmd_pilot(args)

        # No .bak.* sibling.
        bak_dirs = [
            p for p in tmp_path.iterdir()
            if p.name.startswith("fresh_pilot.bak.")
        ]
        assert bak_dirs == []

    def test_overwrite_skips_backup_when_dir_is_empty(self, tmp_path):
        """An empty pilot dir doesn't need to be backed up — there's
        nothing to preserve. --overwrite should still be a no-op for
        empty dirs."""
        from src import cli

        # tmp_path exists but has no files.
        args = _base_args(tmp_path, overwrite=True)
        cli.cmd_pilot(args)

        bak_dirs = [
            p for p in tmp_path.parent.iterdir()
            if p.name.startswith(tmp_path.name + ".bak.")
        ]
        assert bak_dirs == []


# --- 3. Same contract for cmd_run ------------------------------------------

class TestCmdRunOverwriteParity:
    def test_run_overwrite_creates_backup(self, tmp_path):
        from src.cli import build_parser

        # Seed prior data, then run with --overwrite.
        _seed_existing_pilot(tmp_path)
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
            "--overwrite",
        ])
        args.func(args)

        bak_dirs = [
            p for p in tmp_path.parent.iterdir()
            if p.name.startswith(tmp_path.name + ".bak.") and p.is_dir()
        ]
        assert len(bak_dirs) == 1

    def test_run_no_overwrite_preserves_dir(self, tmp_path):
        from src.cli import build_parser

        _seed_existing_pilot(tmp_path)
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
        args.func(args)

        bak_dirs = [
            p for p in tmp_path.parent.iterdir()
            if p.name.startswith(tmp_path.name + ".bak.")
        ]
        assert bak_dirs == []
