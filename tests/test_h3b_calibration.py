"""Tests for scripts/calibration/h3b_calibration.py.

The script's API-call paths can't be unit-tested without burning budget,
but the supporting machinery (repo-root discovery, bank fingerprinting,
ExperimentTracker integration) is testable. Subprocess-level behavior
against the real API is exercised separately via the runbook's smoke
step.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "calibration" / "h3b_calibration.py"
AFFECT_BATTERY_ROOT = REPO_ROOT


def _import_script(monkeypatch):
    """Load h3b_calibration as a module. The script's import-time code
    calls _find_affect_battery_root(), so the env or cwd has to point at
    a real affect-battery checkout for the import to succeed."""
    monkeypatch.setenv("AFFECT_BATTERY_ROOT", str(AFFECT_BATTERY_ROOT))
    spec = importlib.util.spec_from_file_location("h3b_calib", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Repo root discovery
# ----------------------------------------------------------------------

def test_find_root_uses_env_var_when_set(tmp_path: Path, monkeypatch):
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    monkeypatch.setenv("AFFECT_BATTERY_ROOT", str(fake_repo))
    monkeypatch.chdir(tmp_path)
    mod = _import_script(monkeypatch)
    monkeypatch.setenv("AFFECT_BATTERY_ROOT", str(fake_repo))
    assert mod._find_affect_battery_root() == fake_repo.resolve()


def test_find_root_falls_back_to_cwd_when_env_unset(tmp_path: Path, monkeypatch):
    repo_like = tmp_path / "affect-battery-like"
    (repo_like / "src" / "banks").mkdir(parents=True)
    (repo_like / "src" / "banks" / "loader.py").write_text("# stub\n")
    monkeypatch.delenv("AFFECT_BATTERY_ROOT", raising=False)
    monkeypatch.chdir(repo_like)
    mod = _import_script(monkeypatch)
    monkeypatch.delenv("AFFECT_BATTERY_ROOT", raising=False)
    assert mod._find_affect_battery_root() == repo_like.resolve()


def test_find_root_errors_when_neither_env_nor_cwd_works(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AFFECT_BATTERY_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    mod = _import_script(monkeypatch)
    monkeypatch.delenv("AFFECT_BATTERY_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        mod._find_affect_battery_root()


# ----------------------------------------------------------------------
# Bank fingerprint (drives per-bank tracker isolation)
# ----------------------------------------------------------------------

def test_bank_fingerprint_changes_when_bank_content_changes(tmp_path: Path, monkeypatch):
    mod = _import_script(monkeypatch)
    bank_a = tmp_path / "a.yaml"
    bank_a.write_text("items:\n- id: x\n  question: Q1\n")
    bank_b = tmp_path / "b.yaml"
    bank_b.write_text("items:\n- id: x\n  question: Q2\n")
    fp_a = mod._bank_fingerprint(bank_a)
    fp_b = mod._bank_fingerprint(bank_b)
    assert fp_a != fp_b
    bank_a2 = tmp_path / "a2.yaml"
    bank_a2.write_text(bank_a.read_text())
    assert mod._bank_fingerprint(bank_a2) == fp_a


# ----------------------------------------------------------------------
# ExperimentTracker integration: end-to-end via --dry-run subprocess
# ----------------------------------------------------------------------

def _build_minimal_bank(tmp_path: Path, n_items: int = 3) -> Path:
    """Synthesize a tiny GSM-Hard-shaped bank for subprocess tests."""
    bank = tmp_path / "tiny_bank.yaml"
    items = "\n".join(
        f"- id: gsm8k_test_{i:04d}\n"
        f"  question: 'What is {i}+{i}?'\n"
        f"  expected: '{i*2}.0'\n"
        f"  difficulty: hard\n"
        for i in range(n_items)
    )
    bank.write_text(
        "bank_id: tiny\nbank_version: 1\nbank_type: task\nitems:\n" + items
    )
    return bank


def test_subprocess_dry_run_creates_tracker_with_metadata_and_cache(
    tmp_path: Path, monkeypatch
):
    """End-to-end: invoke the script with --dry-run, verify it creates
    an ExperimentTracker dir with the expected layout (metadata JSON,
    cache subdirectory, per-item cache files)."""
    import subprocess
    bank = _build_minimal_bank(tmp_path, n_items=3)
    output = tmp_path / "calib.json"
    env = os.environ.copy()
    env["AFFECT_BATTERY_ROOT"] = str(REPO_ROOT)
    env["UV_CACHE_DIR"] = os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--bank", str(bank),
         "--provider", "openai",
         "--model", "gpt-5.4-nano",
         "--n-candidates", "3", "--n-reps", "5",
         "--max-concurrent", "5",
         "--target-lo", "0.40", "--target-hi", "0.60",
         "--output", str(output),
         "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert output.is_file(), "calibration JSON not written"
    # ExperimentTracker layout: <output>.tracker/bank_<sha>/
    tracker_root = output.with_suffix(output.suffix + ".tracker")
    assert tracker_root.is_dir()
    bank_subdirs = [p for p in tracker_root.iterdir()
                    if p.is_dir() and p.name.startswith("bank_")]
    assert len(bank_subdirs) == 1, (
        f"expected exactly one bank subdir, got {[p.name for p in bank_subdirs]}"
    )
    bank_dir = bank_subdirs[0]
    # ExperimentTracker writes run_metadata.json + cache/
    metadata_path = bank_dir / "run_metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["params"]["n_reps"] == 5
    assert metadata["params"]["n_candidates"] == 3
    assert metadata["params"]["model"] == "gpt-5.4-nano"
    assert "pre_screen" in metadata["stages"]  # stage timing recorded
    assert "n_per_item" in metadata["metrics"]  # final metrics recorded
    cache_dir = bank_dir / "cache"
    assert cache_dir.is_dir()
    cached_files = list(cache_dir.glob("*.json"))
    assert len(cached_files) == 3, f"expected 3 cached items, got {len(cached_files)}"


def test_subprocess_resume_skips_cached_items(tmp_path: Path, monkeypatch):
    """Run twice with the same args; the second run should report
    `resuming: 3 of 3 candidates cached` and not redispatch."""
    import subprocess
    bank = _build_minimal_bank(tmp_path, n_items=3)
    output = tmp_path / "calib.json"
    env = os.environ.copy()
    env["AFFECT_BATTERY_ROOT"] = str(REPO_ROOT)
    env["UV_CACHE_DIR"] = os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")
    common_args = [
        sys.executable, str(SCRIPT),
        "--bank", str(bank),
        "--provider", "openai", "--model", "gpt-5.4-nano",
        "--n-candidates", "3", "--n-reps", "5",
        "--max-concurrent", "5",
        "--output", str(output),
        "--dry-run",
    ]
    r1 = subprocess.run(common_args, cwd=REPO_ROOT, env=env,
                        capture_output=True, text=True, timeout=60)
    assert r1.returncode == 0, r1.stderr[:300]
    r2 = subprocess.run(common_args, cwd=REPO_ROOT, env=env,
                        capture_output=True, text=True, timeout=60)
    assert r2.returncode == 0, r2.stderr[:300]
    assert "resuming: 3 of 3 candidates cached" in r2.stderr, (
        f"second run did not detect cached items: {r2.stderr[:500]}"
    )


def test_subprocess_different_bank_creates_separate_tracker_subdir(
    tmp_path: Path, monkeypatch
):
    """Two banks with different content but same item IDs must not share
    a tracker subdir — switching banks preserves the prior bank's data
    untouched."""
    import subprocess
    bank_a_dir = tmp_path / "a"
    bank_a_dir.mkdir()
    bank_a = _build_minimal_bank(bank_a_dir, n_items=2)
    bank_b_dir = tmp_path / "b"
    bank_b_dir.mkdir()
    bank_b = bank_b_dir / "tiny_bank.yaml"
    bank_b.write_text(
        "bank_id: tiny\nbank_version: 1\nbank_type: task\nitems:\n"
        "- id: gsm8k_test_0000\n  question: 'Different Q?'\n"
        "  expected: '99.0'\n  difficulty: hard\n"
        "- id: gsm8k_test_0001\n  question: 'Different Q2?'\n"
        "  expected: '98.0'\n  difficulty: hard\n"
    )
    output = tmp_path / "calib.json"
    env = os.environ.copy()
    env["AFFECT_BATTERY_ROOT"] = str(REPO_ROOT)
    env["UV_CACHE_DIR"] = os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache")
    base = [sys.executable, str(SCRIPT), "--provider", "openai",
            "--model", "gpt-5.4-nano", "--n-candidates", "2", "--n-reps", "3",
            "--max-concurrent", "5", "--output", str(output), "--dry-run"]
    subprocess.run(base + ["--bank", str(bank_a)], cwd=REPO_ROOT, env=env,
                   capture_output=True, text=True, timeout=60)
    subprocess.run(base + ["--bank", str(bank_b)], cwd=REPO_ROOT, env=env,
                   capture_output=True, text=True, timeout=60)
    tracker_root = output.with_suffix(output.suffix + ".tracker")
    bank_subdirs = sorted(p.name for p in tracker_root.iterdir()
                          if p.is_dir() and p.name.startswith("bank_"))
    assert len(bank_subdirs) == 2, (
        f"expected two bank subdirs, got {bank_subdirs}"
    )
