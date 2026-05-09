"""Tests for the cache carry-over helper.

When the bank source-of-truth is replaced (e.g. expanding from a 286-item
GSM-Hard subset to the full 1,319-item pool), the calibration script's
per-bank cache fingerprint changes. Items that overlap by content are
re-screened from scratch unless we explicitly carry over.

`carry_over_cache` matches old cache cells to new-bank items by
(question, expected), rewrites the cell's `item_id` to match the new
bank's ID, and writes into the new cache dir. Tests focus on the
contract: matching semantics, item_id rewriting, no-overwrite safety,
and dry-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.dev.carry_over_cache import CarryOverResult, carry_over_cache


def _write_cell(cache_dir: Path, item_id: str, cell: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{item_id}.json").write_text(json.dumps(cell))


def _scored(item_id: str, question: str, expected: str, p_hat: float = 0.5) -> dict:
    return {
        "kind": "scored",
        "item_id": item_id,
        "question": question,
        "expected": expected,
        "n_reps": 100,
        "n_correct": int(p_hat * 100),
        "p_hat": p_hat,
        "n_blocked_reps": 0,
        "n_error_reps": 0,
    }


def test_carry_over_copies_matching_cells_with_rewritten_id(tmp_path: Path):
    old_bank = [
        {"id": "gsm8k_0568", "question": "Q1?", "expected": "10.0",
         "source_dataset": "gsm-hard"},
        {"id": "gsm8k_0569", "question": "Q2?", "expected": "20.0",
         "source_dataset": "gsm-hard"},
    ]
    new_bank = [
        {"id": "gsm_hard_0042", "question": "Q1?", "expected": "10.0",
         "source_dataset": "gsm-hard"},
        {"id": "gsm_hard_0099", "question": "Q2?", "expected": "20.0",
         "source_dataset": "gsm-hard"},
        {"id": "gsm_hard_1000", "question": "Q3?", "expected": "30.0",
         "source_dataset": "gsm-hard"},  # not in old bank
    ]
    old_cache = tmp_path / "old_cache"
    new_cache = tmp_path / "new_cache"
    _write_cell(old_cache, "gsm8k_0568", _scored("gsm8k_0568", "Q1?", "10.0", 0.45))
    _write_cell(old_cache, "gsm8k_0569", _scored("gsm8k_0569", "Q2?", "20.0", 0.55))

    result = carry_over_cache(old_bank, old_cache, new_bank, new_cache)

    assert result.matched == 2
    assert result.copied == 2
    assert result.already_cached == 0
    assert result.unmatched == 0

    # New cache has the new IDs; old IDs absent.
    assert (new_cache / "gsm_hard_0042.json").exists()
    assert (new_cache / "gsm_hard_0099.json").exists()
    assert not (new_cache / "gsm8k_0568.json").exists()

    # The cell's item_id was rewritten in-place; question/expected/p_hat preserved.
    cell_42 = json.loads((new_cache / "gsm_hard_0042.json").read_text())
    assert cell_42["item_id"] == "gsm_hard_0042"
    assert cell_42["question"] == "Q1?"
    assert cell_42["expected"] == "10.0"
    assert cell_42["p_hat"] == 0.45


def test_unmatched_old_cells_skipped(tmp_path: Path):
    """Cells whose (question, expected) doesn't appear in the new bank are
    counted as unmatched and not copied."""
    old_bank = [{"id": "gsm8k_0001", "question": "Orphan?", "expected": "1",
                 "source_dataset": "gsm-hard"}]
    new_bank = [{"id": "gsm_hard_0000", "question": "Different?", "expected": "1",
                 "source_dataset": "gsm-hard"}]
    old_cache = tmp_path / "old"
    new_cache = tmp_path / "new"
    _write_cell(old_cache, "gsm8k_0001", _scored("gsm8k_0001", "Orphan?", "1"))

    result = carry_over_cache(old_bank, old_cache, new_bank, new_cache)

    assert result.matched == 0
    assert result.copied == 0
    assert result.unmatched == 1
    assert not new_cache.exists() or not list(new_cache.glob("*.json"))


def test_does_not_overwrite_existing_new_cache_cells(tmp_path: Path):
    """If a new-cache cell already exists for the matched ID, leave it
    alone — don't clobber. Cache-isolation invariant: do not overwrite
    existing cells when bridging from a prior bank's cache."""
    old_bank = [{"id": "old_0", "question": "Q?", "expected": "5",
                 "source_dataset": "gsm-hard"}]
    new_bank = [{"id": "new_0", "question": "Q?", "expected": "5",
                 "source_dataset": "gsm-hard"}]
    old_cache = tmp_path / "old"
    new_cache = tmp_path / "new"
    _write_cell(old_cache, "old_0", _scored("old_0", "Q?", "5", p_hat=0.30))
    # Pre-existing cell in the new cache from a partial run with a
    # different p_hat — must NOT be replaced.
    pre_existing = _scored("new_0", "Q?", "5", p_hat=0.71)
    _write_cell(new_cache, "new_0", pre_existing)

    result = carry_over_cache(old_bank, old_cache, new_bank, new_cache)

    assert result.matched == 1
    assert result.copied == 0
    assert result.already_cached == 1
    cell = json.loads((new_cache / "new_0.json").read_text())
    assert cell["p_hat"] == 0.71  # untouched


def test_dry_run_reports_but_does_not_write(tmp_path: Path):
    old_bank = [{"id": "old_0", "question": "Q?", "expected": "5",
                 "source_dataset": "gsm-hard"}]
    new_bank = [{"id": "new_0", "question": "Q?", "expected": "5",
                 "source_dataset": "gsm-hard"}]
    old_cache = tmp_path / "old"
    new_cache = tmp_path / "new"
    _write_cell(old_cache, "old_0", _scored("old_0", "Q?", "5"))

    result = carry_over_cache(old_bank, old_cache, new_bank, new_cache, dry_run=True)

    assert result.matched == 1
    assert result.copied == 0
    assert result.would_copy == 1
    assert not (new_cache / "new_0.json").exists()


def test_blocked_cells_carried_over_via_bank_lookup(tmp_path: Path):
    """Bank-driven matching: cells whose schema doesn't include question
    or expected (blocked cells) still match via the OLD BANK's entry for
    their item_id. The bank is the source of truth for question/expected;
    the cell only needs to know its item_id to find its bank record."""
    old_bank = [{"id": "old_0", "question": "Q?", "expected": "5",
                 "source_dataset": "gsm-hard"}]
    new_bank = [{"id": "new_0", "question": "Q?", "expected": "5",
                 "source_dataset": "gsm-hard"}]
    old_cache = tmp_path / "old"
    new_cache = tmp_path / "new"
    # Blocked cells in the schema only carry item_id + reason, no question/expected.
    _write_cell(old_cache, "old_0", {
        "kind": "blocked", "item_id": "old_0", "reason": "rate_limit",
    })

    result = carry_over_cache(old_bank, old_cache, new_bank, new_cache)

    assert result.matched == 1
    assert result.copied == 1
    cell = json.loads((new_cache / "new_0.json").read_text())
    assert cell["item_id"] == "new_0"
    assert cell["kind"] == "blocked"


def test_extra_new_bank_items_unaffected(tmp_path: Path):
    """New-bank items not present in the old bank are not pre-populated —
    they'll be screened fresh by the calibration run."""
    old_bank = [{"id": "old_0", "question": "A?", "expected": "1",
                 "source_dataset": "gsm-hard"}]
    new_bank = [
        {"id": "new_0", "question": "A?", "expected": "1",
         "source_dataset": "gsm-hard"},
        {"id": "new_1", "question": "B?", "expected": "2",
         "source_dataset": "gsm-hard"},
    ]
    old_cache = tmp_path / "old"
    new_cache = tmp_path / "new"
    _write_cell(old_cache, "old_0", _scored("old_0", "A?", "1"))

    result = carry_over_cache(old_bank, old_cache, new_bank, new_cache)

    assert result.copied == 1
    assert (new_cache / "new_0.json").exists()
    assert not (new_cache / "new_1.json").exists()
