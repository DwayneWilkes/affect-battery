"""Atomicity and key-sanitization contracts for src.lib.tracking.

Atomicity: the H3b calibrator runs for hours and may receive SIGTERM
mid-write (operator cancel, sandbox timeout, run_h3b_phase1a.py's
terminate_inflight fan-out). Per-item cache writes and run_metadata
writes must be atomic so a torn write cannot leave a truncated JSON
file that fails on resume.

Key sanitization: cache keys are item IDs read from operator-curated
bank YAMLs. Defense-in-depth: even though callers are trusted today,
the sanitizer is the natural enforcement boundary for any future bank
source (uploads, scraped datasets) and must use an allowlist, not a
denylist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.lib.tracking import ExperimentTracker


def test_log_item_is_atomic_under_failure(tmp_path: Path, monkeypatch):
    """If a write fails partway through, the cache file on disk must
    still hold the previous value (not a half-written truncation)."""
    tracker = ExperimentTracker(tmp_path / "out", experiment_name="atomicity_test")
    tracker.log_item("k1", {"v": 1})
    cache_path = tracker.cache_dir / "k1.json"
    assert json.loads(cache_path.read_text()) == {"v": 1}

    # Simulate a SIGTERM mid-write: json.dump raises after writing some bytes.
    def failing_dump(obj, fp, *a, **kw):
        fp.write('{"v": 2,')  # partial — invalid JSON
        raise RuntimeError("simulated SIGTERM mid-write")

    monkeypatch.setattr("src.lib.tracking.json.dump", failing_dump)
    with pytest.raises(RuntimeError):
        tracker.log_item("k1", {"v": 2})

    # is_cached/load_cached use json.load (read), not json.dump, so the
    # patched dump doesn't affect the verification reads below.
    assert tracker.is_cached("k1"), "is_cached lies if the original was lost"
    assert tracker.load_cached("k1") == {"v": 1}, (
        "torn write corrupted the cache: load_cached returned wrong data "
        "or raised JSONDecodeError"
    )


def test_save_metadata_is_atomic_under_failure(tmp_path: Path, monkeypatch):
    """run_metadata.json drives the dashboard cost projection. A torn
    write would leave the dashboard reading a truncated file."""
    tracker = ExperimentTracker(tmp_path / "out", experiment_name="atomicity_test")
    tracker.log_params(model="x", n_reps=10)
    metadata_path = tracker._metadata_path
    pre_failure = json.loads(metadata_path.read_text())
    assert pre_failure["params"]["n_reps"] == 10

    def failing_dump(obj, fp, *a, **kw):
        fp.write('{"params":')
        raise RuntimeError("simulated SIGTERM mid-write")

    monkeypatch.setattr("src.lib.tracking.json.dump", failing_dump)
    with pytest.raises(RuntimeError):
        tracker.log_params(n_reps=999)

    # metadata_path.read_text() doesn't go through json.dump, so the
    # patched dump doesn't affect the verification read below.
    after_failure = json.loads(metadata_path.read_text())
    assert after_failure["params"]["n_reps"] == 10, (
        "metadata corrupted by torn write — dashboard projection would "
        "be reading garbage"
    )


def test_sanitize_key_admits_realistic_item_ids():
    """The sanitizer must pass through real item IDs unchanged so cache
    lookup remains stable across runs."""
    cases = [
        "gsm8k_0001",
        "gsm8k_test_0042",
        "gsm_hard_block_007",
        "item-with-dashes",
        "item.with.dots",
    ]
    for k in cases:
        assert ExperimentTracker._sanitize_key(k) == k, (
            f"legitimate item ID was mangled: {k!r} -> "
            f"{ExperimentTracker._sanitize_key(k)!r}"
        )


def test_sanitize_key_rejects_path_traversal_and_separators():
    """Allowlist contract: the sanitizer must replace any character
    outside [A-Za-z0-9._\\-] so an attacker-controlled item ID cannot
    write outside the cache directory or to a hidden file."""
    attack_inputs = [
        "../../etc/passwd",
        "..\\windows\\system32",
        "/absolute/path",
        "a/b/c",
        "name with spaces",
        "weird;semicolon",
        "shell$injection",
        "tab\there",
        "newline\nhere",
        "null\x00byte",
    ]
    for k in attack_inputs:
        sanitized = ExperimentTracker._sanitize_key(k)
        for ch in sanitized:
            assert ch.isalnum() or ch in "._-", (
                f"sanitized output of {k!r} contains disallowed char "
                f"{ch!r}: {sanitized!r}"
            )
        assert ".." not in sanitized, (
            f"path-traversal sequence survived: {k!r} -> {sanitized!r}"
        )
        assert "/" not in sanitized and "\\" not in sanitized, (
            f"path separator survived: {k!r} -> {sanitized!r}"
        )


def test_sanitize_key_rejects_dot_only_keys():
    """An item ID that's only dots would yield a hidden file (`.json`,
    `..json`) or collide with the cache dir entries themselves."""
    for k in [".", "..", "...", ""]:
        sanitized = ExperimentTracker._sanitize_key(k)
        assert sanitized and not sanitized.startswith("."), (
            f"dot-only key {k!r} produced unsafe sanitized form: "
            f"{sanitized!r}"
        )
