"""Tests for the gsm-hard-all mode in scripts/banks/source_gsm8k.py.

The default `balanced` mode samples 1/3 each tier from GSM8K + GSM-Hard.
The `gsm-hard-all` mode emits every available GSM-Hard item as a single
hard-tier bank, used when we want to pre-screen the entire pool for
calibration to maximize the in-band yield.

These tests exercise the helpers (`to_bank_item`, `emit_bank_yaml` /
`emit_gsm_hard_full_bank_yaml`) directly with synthetic raw data so they
don't require network access to HuggingFace.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.banks.source_gsm8k import (
    emit_gsm_hard_full_bank_yaml,
    to_bank_item,
)


def _hard_raw(target: int) -> dict:
    """Build a synthetic GSM-Hard row in the schema returned by load_gsm_hard."""
    return {
        "question": f"Janet's ducks lay {target * 2} eggs per day...",
        "answer_text": f"some PAL-style reasoning\n#### {target}",
        "source": "gsm-hard",
    }


def test_to_bank_item_with_gsm_hard_prefix_assigns_hard_tier():
    """All GSM-Hard items go to the hard tier regardless of step count."""
    raw = _hard_raw(4933828)
    item = to_bank_item(raw, 0, id_prefix="gsm_hard")
    assert item["difficulty"] == "hard"
    assert item["source_dataset"] == "gsm-hard"
    assert item["id"] == "gsm_hard_0000"
    assert item["expected"] == "4933828"


def test_to_bank_item_default_prefix_unchanged():
    """Back-compat: default prefix is gsm8k so existing callers don't break."""
    raw = _hard_raw(42)
    item = to_bank_item(raw, 7)
    assert item["id"] == "gsm8k_0007"


def test_emit_gsm_hard_full_bank_yaml_metadata():
    """The all-GSM-Hard bank gets a distinct bank_id and single-source profile."""
    raws = [_hard_raw(1000 + i) for i in range(5)]
    items = [to_bank_item(r, i, id_prefix="gsm_hard") for i, r in enumerate(raws)]
    yaml_text = emit_gsm_hard_full_bank_yaml(items)
    bank = yaml.safe_load(yaml_text)
    assert bank["bank_id"] == "gsm_hard_full_v1"
    assert bank["bank_version"] == 1
    assert bank["bank_type"] == "task"
    assert bank["status"] == "active"
    assert len(bank["items"]) == 5
    profile = bank["difficulty_profile"]
    assert profile["n_total"] == 5
    assert profile["n_per_tier"] == {"hard": 5}
    sources = [s["name"] for s in profile["sources"]]
    assert sources == ["GSM-Hard"], f"expected single source, got {sources}"
    assert profile["sources"][0]["n_sampled"] == 5


def test_emit_gsm_hard_full_bank_yaml_alignment_review_cites_gao_2023():
    """The single-source bank's rationale must cite the GSM-Hard paper."""
    raws = [_hard_raw(1000 + i) for i in range(2)]
    items = [to_bank_item(r, i, id_prefix="gsm_hard") for i, r in enumerate(raws)]
    bank = yaml.safe_load(emit_gsm_hard_full_bank_yaml(items))
    rationale = bank["alignment_review"]["rationale"]
    assert "Gao" in rationale or "gsm-hard" in rationale.lower()
    assert "PAL" in rationale or "2211.10435" in rationale or "Gao" in rationale


def test_emit_gsm_hard_full_bank_ids_are_unique():
    """No duplicate IDs across the full pool."""
    raws = [_hard_raw(i) for i in range(100)]
    items = [to_bank_item(r, i, id_prefix="gsm_hard") for i, r in enumerate(raws)]
    ids = [it["id"] for it in items]
    assert len(set(ids)) == len(ids)
