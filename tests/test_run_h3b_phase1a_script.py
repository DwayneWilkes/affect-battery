"""Tests for `scripts/pilots/run_h3b_phase1a.sh`.

The wrapper orchestrates 20 within-subjects passes of run_exp3a, each
writing to its own output subdirectory. Tests exercise every branch:
argument validation, cell-count derivation, pass-skipping discipline
(fresh / partial / complete), parallel-failure cleanup, and manifest
output.

Tests use a stub `affect-battery` placed on PATH that mimics the
runner's output structure (N_items × N_levels JSONs under
`level_<N>/neutral/`) so the wrapper's downstream checks pass. Stub
behavior is keyed off env vars so each test can dial in success,
failure-on-specific-pass, and slow-execution scenarios independently.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "pilots" / "run_h3b_phase1a.sh"


# Stub `affect-battery` that pretends to be the runner. Behavior knobs:
#   STUB_FAIL_ON_PASS    fail with exit 17 when --output-dir matches pass_<N>
#   STUB_SLEEP_SECONDS   sleep before producing output (simulate slow run)
#   STUB_PRODUCE_OUTPUT  if "0", skip writing cell files (used to verify
#                        that pass-skipping really skips invocation)
STUB_AFFECT_BATTERY = textwrap.dedent("""\
    #!/usr/bin/env bash
    set -e
    OUTPUT_DIR=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir) OUTPUT_DIR="$2"; shift 2;;
            *) shift;;
        esac
    done
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "stub: missing --output-dir" >&2
        exit 1
    fi
    pass_num=$(echo "$OUTPUT_DIR" | grep -oE 'pass_[0-9]+' | grep -oE '[0-9]+' | sed 's/^0*//')
    if [[ -z "$pass_num" ]]; then pass_num=0; fi
    if [[ -n "${STUB_FAIL_ON_PASS:-}" && "$pass_num" == "${STUB_FAIL_ON_PASS}" ]]; then
        echo "stub: failing on pass $pass_num" >&2
        exit 17
    fi
    if [[ -n "${STUB_SLEEP_SECONDS:-}" ]]; then
        sleep "${STUB_SLEEP_SECONDS}"
    fi
    if [[ "${STUB_PRODUCE_OUTPUT:-1}" == "1" ]]; then
        for level in 1 2 3 4 5 6 7; do
            mkdir -p "$OUTPUT_DIR/level_$level/neutral"
            for i in $(seq 0 17); do
                printf '{"item": %d, "level": %d}' "$i" "$level" \\
                    > "$OUTPUT_DIR/level_$level/neutral/$(printf '%04d' $i).json"
            done
        done
    fi
    exit 0
""")

STUB_BANK_HEADER = textwrap.dedent("""\
    bank_id: h3b_calibrated_v1
    bank_version: 1
    bank_type: task
    items:
""")

STUB_RUNNER_CONFIG = textwrap.dedent("""\
    intensity_levels: [1, 2, 3, 4, 5, 6, 7]
    pilot_seed_path: configs/intensity_pilot_seed.json
    sampling_mode: within_subjects
""")


def _build_bank(item_count: int = 18) -> str:
    items = "\n".join(
        f"- id: gsm8k_{i:04d}\n  question: Test Q{i}\n  expected: '{i}.0'"
        for i in range(item_count)
    )
    return STUB_BANK_HEADER + items + "\n"


@pytest.fixture
def env_setup(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build a tmp repo skeleton with stub bank, runner-config, and a
    stub `affect-battery` on PATH. Returns (cwd, env)."""
    (tmp_path / "configs" / "banks").mkdir(parents=True)
    (tmp_path / "configs" / "banks" / "h3b_calibrated_v1.yaml").write_text(_build_bank())
    (tmp_path / "configs" / "exp3a_runner_h3b_2026-05-07.yaml").write_text(STUB_RUNNER_CONFIG)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "affect-battery"
    stub.write_text(STUB_AFFECT_BATTERY)
    stub.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPENAI_API_KEY"] = "test-key"
    for var in ("STUB_FAIL_ON_PASS", "STUB_SLEEP_SECONDS", "STUB_PRODUCE_OUTPUT"):
        env.pop(var, None)
    return tmp_path, env


def _run(cwd: Path, env: dict[str, str], args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------

def test_help_prints_usage_and_exits_zero(env_setup):
    cwd, env = env_setup
    result = _run(cwd, env, ["--help"])
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "--prereg-commit" in result.stdout


def test_missing_prereg_commit_exits_one(env_setup):
    cwd, env = env_setup
    result = _run(cwd, env, [])
    assert result.returncode == 1
    assert "--prereg-commit is required" in result.stderr


def test_missing_openai_api_key_exits_one(env_setup):
    cwd, env = env_setup
    del env["OPENAI_API_KEY"]
    result = _run(cwd, env, ["--prereg-commit", "owner/repo@abc1234"])
    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr


def test_missing_bank_yaml_exits_one(env_setup):
    cwd, env = env_setup
    (cwd / "configs" / "banks" / "h3b_calibrated_v1.yaml").unlink()
    result = _run(cwd, env, ["--prereg-commit", "owner/repo@abc1234"])
    assert result.returncode == 1
    assert "bank YAML not found" in result.stderr


def test_missing_runner_config_exits_one(env_setup):
    cwd, env = env_setup
    (cwd / "configs" / "exp3a_runner_h3b_2026-05-07.yaml").unlink()
    result = _run(cwd, env, ["--prereg-commit", "owner/repo@abc1234"])
    assert result.returncode == 1
    assert "runner config not found" in result.stderr


def test_unknown_flag_exits_one_with_usage(env_setup):
    cwd, env = env_setup
    result = _run(cwd, env, ["--prereg-commit", "x", "--bogus", "y"])
    assert result.returncode == 1
    assert "unknown flag" in result.stderr
    assert "Usage:" in result.stderr


# ----------------------------------------------------------------------
# Cell-count derivation
# ----------------------------------------------------------------------

def test_expected_cells_per_pass_derived_correctly(env_setup):
    cwd, env = env_setup
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "1", "--max-parallel", "1"],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    manifest = (cwd / "results" / "h3b_2026-05-07" / "run_manifest.txt").read_text()
    assert "n_bank_items:          18" in manifest
    assert "n_levels:              7" in manifest
    assert "expected_cells/pass:   126" in manifest


def test_zero_items_in_bank_fails_derivation(env_setup):
    cwd, env = env_setup
    (cwd / "configs" / "banks" / "h3b_calibrated_v1.yaml").write_text(
        "bank_id: empty\nitems: []\n"
    )
    result = _run(cwd, env, ["--prereg-commit", "owner/repo@x"])
    assert result.returncode == 1
    assert "could not derive" in result.stderr


def test_runner_config_without_intensity_levels_fails_derivation(env_setup):
    cwd, env = env_setup
    (cwd / "configs" / "exp3a_runner_h3b_2026-05-07.yaml").write_text(
        "sampling_mode: within_subjects\n"
    )
    result = _run(cwd, env, ["--prereg-commit", "owner/repo@x"])
    assert result.returncode == 1
    assert "could not derive" in result.stderr


# ----------------------------------------------------------------------
# Pass dispatch
# ----------------------------------------------------------------------

def test_fresh_run_dispatches_all_passes(env_setup):
    cwd, env = env_setup
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "3", "--max-parallel", "2"],
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output_base = cwd / "results" / "h3b_2026-05-07"
    for pass_n in (1, 2, 3):
        pass_dir = output_base / f"pass_{pass_n:02d}"
        assert pass_dir.exists(), f"pass {pass_n} dir missing"
        n_cells = len(list(pass_dir.glob("level_*/neutral/*.json")))
        assert n_cells == 126, f"pass {pass_n}: {n_cells} cells, expected 126"
    manifest = (output_base / "run_manifest.txt").read_text()
    assert "failed_passes:         0" in manifest


def test_fully_complete_pass_skipped_no_runner_invocation(env_setup):
    cwd, env = env_setup
    output_base = cwd / "results" / "h3b_2026-05-07"
    for level in range(1, 8):
        d = output_base / "pass_01" / f"level_{level}" / "neutral"
        d.mkdir(parents=True)
        for i in range(18):
            (d / f"{i:04d}.json").write_text("{}")
    # If the wrapper redispatched, the stub (with PRODUCE_OUTPUT=0) would
    # leave the dir as-is — but the post-run cell-count check still
    # passes because the pre-existing files are already there. Use
    # log-text assertion instead.
    env["STUB_PRODUCE_OUTPUT"] = "0"
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "1", "--max-parallel", "1"],
    )
    assert result.returncode == 0
    assert "complete, skipping" in result.stdout
    assert "dispatching" not in result.stdout
    assert "re-dispatching" not in result.stdout


def test_partial_pass_redispatched(env_setup):
    cwd, env = env_setup
    output_base = cwd / "results" / "h3b_2026-05-07"
    d = output_base / "pass_01" / "level_1" / "neutral"
    d.mkdir(parents=True)
    for i in range(50):
        (d / f"{i:04d}.json").write_text("{}")
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "1", "--max-parallel", "1"],
    )
    assert result.returncode == 0
    assert "partial, re-dispatching" in result.stdout


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------

def test_failing_pass_exits_nonzero_with_failure_message(env_setup):
    cwd, env = env_setup
    env["STUB_FAIL_ON_PASS"] = "2"
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "3", "--max-parallel", "1"],
    )
    assert result.returncode == 1
    assert "FAILURE" in result.stderr
    assert "pass(es) failed" in result.stderr


def test_manifest_records_failure_count(env_setup):
    cwd, env = env_setup
    env["STUB_FAIL_ON_PASS"] = "1"
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "2", "--max-parallel", "1"],
    )
    assert result.returncode == 1
    manifest = (cwd / "results" / "h3b_2026-05-07" / "run_manifest.txt").read_text()
    failed_line = next(L for L in manifest.splitlines() if "failed_passes:" in L)
    assert int(failed_line.split(":")[1].strip()) >= 1


def test_failure_kills_inflight_passes_quickly(env_setup):
    """If a pass fails while siblings are sleeping, cleanup_on_failure
    must kill the siblings rather than letting them run to completion.
    Asserted via wall-clock: with sleep=4 and pass 1 failing, the
    wrapper should exit well under 4s if cleanup works."""
    cwd, env = env_setup
    env["STUB_FAIL_ON_PASS"] = "1"
    env["STUB_SLEEP_SECONDS"] = "4"
    start = time.time()
    result = _run(
        cwd, env,
        ["--prereg-commit", "owner/repo@x", "--n-passes", "5", "--max-parallel", "5"],
    )
    elapsed = time.time() - start
    assert result.returncode == 1
    # Cleanup should bring the wrapper down quickly. Allow generous slack
    # for parent-shell teardown but well under the sleep duration.
    assert elapsed < 4.0, (
        f"wrapper took {elapsed:.1f}s; expected fast cleanup. "
        f"stderr: {result.stderr[:400]}"
    )
