"""Parity contract: cmd_run inherits the same pilot-root layout that
cmd_pilot already uses, plus exp3c gains alias-aware scoring.

Closes gaps #1-4 from the post-MVP audit:
  1. cmd_run writes data under <output_dir>/data/<experiment>/.
  2. cmd_run writes a manifest.yaml at the pilot root.
  3. --transfer-bank is a CLI argument that flows into ExperimentConfig.
  4. Exp3cBody carries expected_aliases; analyze_exp3c uses them.

Spec: affect-battery-proposal-realignment :: results-layout, scoring-pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


# --- Gap #1 + #2: cmd_run pilot-root layout + manifest ----------------------

class TestCmdRunPilotRootLayout:
    def test_run_writes_data_under_data_subdir(self, tmp_path):
        """`affect-battery run --experiment exp1a ...` must write JSONs under
        <output_dir>/data/exp1a/<condition>/<NNNN>.json, not flat at
        <output_dir>/<condition>/<NNNN>.json."""
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "dry-run",
            "--condition", "neutral",
            "--num-runs", "2",
            "--seed", "0",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)

        data_dir = tmp_path / "data"
        assert data_dir.is_dir(), (
            f"Expected <output_dir>/data/exp1a/, found these dirs: "
            f"{[p.name for p in tmp_path.iterdir() if p.is_dir()]}"
        )
        # JSONs nested under condition subdirs.
        result_files = list(data_dir.rglob("*.json"))
        assert len(result_files) >= 2
        for p in result_files:
            assert p.parent.parent == data_dir
            assert p.name.endswith(".json")
        # Legacy flat path must NOT also exist.
        assert not (tmp_path / "neutral").is_dir()

    def test_run_writes_manifest_at_pilot_root(self, tmp_path):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "claude-haiku-4-5",
            "--condition", "strong_negative",
            "--num-runs", "1",
            "--seed", "42",
            "--temperature", "0.5",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)

        manifest_path = tmp_path / "manifest.yaml"
        assert manifest_path.exists(), (
            f"cmd_run must write manifest.yaml at pilot root. "
            f"Files at root: {list(tmp_path.iterdir())}"
        )
        manifest = yaml.safe_load(manifest_path.read_text())
        assert manifest["model"] == "claude-haiku-4-5"
        assert manifest["experiment"] == "exp1a"
        # cmd_run is single-condition, so the manifest's `conditions` list
        # records the one condition that ran.
        assert manifest["conditions"] == ["strong_negative"]
        assert manifest["num_runs"] == 1
        assert manifest["seed"] == 42

    def test_run_manifest_has_no_null_values(self, tmp_path):
        """Same no-null contract that cmd_pilot satisfies. Skipped gates
        and unset banks must surface as informative status strings."""
        from src.cli import build_parser

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


# --- Gap #3: --transfer-bank CLI argument -----------------------------------

class TestTransferBankCLIArg:
    def test_pilot_accepts_transfer_bank_flag(self):
        """`affect-battery pilot --transfer-bank <path>` must parse and
        attach the path to args.transfer_bank."""
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "pilot",
            "--model", "dry-run",
            "--dry-run",
            "--transfer-bank", "configs/banks/exp1a_factual_qa_hard_v1.yaml",
        ])
        assert args.transfer_bank == "configs/banks/exp1a_factual_qa_hard_v1.yaml"

    def test_run_accepts_transfer_bank_flag(self):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "dry-run",
            "--condition", "neutral",
            "--dry-run",
            "--transfer-bank", "configs/banks/exp1a_factual_qa_hard_v1.yaml",
        ])
        assert args.transfer_bank == "configs/banks/exp1a_factual_qa_hard_v1.yaml"

    def test_transfer_bank_default_is_none(self):
        """Without --transfer-bank, args.transfer_bank must be None (not
        the string 'None'). The runner falls through to the hardcoded pool."""
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp1a",
            "--model", "dry-run",
            "--condition", "neutral",
            "--dry-run",
        ])
        assert args.transfer_bank is None

    def test_transfer_bank_flows_into_experiment_config(self, tmp_path, monkeypatch):
        """When --transfer-bank is set, the resulting ExperimentConfig
        must carry the bank path so run_single loads from it."""
        from src.cli import build_parser
        from src.runner import ExperimentConfig

        bank_path = "configs/banks/exp1a_factual_qa_hard_v1.yaml"

        captured = {}
        original_init = ExperimentConfig.__init__

        def capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured["transfer_bank"] = self.transfer_bank

        monkeypatch.setattr(ExperimentConfig, "__init__", capturing_init)

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
            "--transfer-bank", bank_path,
        ])
        args.func(args)
        assert captured.get("transfer_bank") == bank_path

    def test_transfer_bank_appears_in_manifest(self, tmp_path):
        from src.cli import build_parser

        bank_path = "configs/banks/exp1a_factual_qa_hard_v1.yaml"
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
            "--transfer-bank", bank_path,
        ])
        args.func(args)
        manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
        assert manifest["transfer_bank"] == bank_path


# --- Gap #4: Exp 3c alias-aware scoring -------------------------------------

class TestExp3cAliasScoring:
    def test_exp3c_body_supports_expected_aliases(self):
        """Exp3cBody must carry an expected_aliases list so future banks
        with alias annotations don't lose them at write time."""
        from src.runner import Exp3cBody

        body = Exp3cBody(
            difficulty="hard",
            question="What is the capital of the U.S.?",
            response="Washington, DC.",
            expected="Washington, D.C.",
            expected_aliases=["Washington DC", "Washington", "D.C."],
        )
        assert body.expected_aliases == ["Washington DC", "Washington", "D.C."]

    def test_exp3c_body_aliases_default_empty(self):
        from src.runner import Exp3cBody

        body = Exp3cBody(
            difficulty="easy", question="q?", response="r", expected="e",
        )
        assert body.expected_aliases == []

    def test_analyze_exp3c_uses_aliases_for_correctness(self, tmp_path):
        """When the exp3c corpus has expected_aliases, the analyzer must
        match against any alias, not just the canonical expected string."""
        from src.analysis.exp3c import analyze_exp3c_corpus

        # Two runs: response matches an ALIAS only (not the canonical
        # expected). Without alias-aware scoring, accuracy would be 0;
        # with aliases, it should be 1.0 for both items.
        corpus = [
            {
                "model": "claude-haiku-4-5",
                "condition": "neutral",
                "experiment_type": "exp3c",
                "config": {"model_name": "claude-haiku-4-5", "condition": "neutral"},
                "body": {
                    "difficulty": "hard",
                    "question": "What is USA?",
                    "response": "USA is United States of America.",
                    "expected": "United States of America",
                    "expected_aliases": ["U.S.A.", "USA", "United States"],
                    "stated_confidence": None,
                    "refused": False,
                },
            },
            {
                "model": "claude-haiku-4-5",
                "condition": "neutral",
                "experiment_type": "exp3c",
                "config": {"model_name": "claude-haiku-4-5", "condition": "neutral"},
                "body": {
                    "difficulty": "hard",
                    "question": "What is the capital of the U.S.?",
                    "response": "It's DC.",
                    "expected": "Washington, D.C.",
                    "expected_aliases": ["DC", "Washington DC", "D.C."],
                    "stated_confidence": None,
                    "refused": False,
                },
            },
        ]

        analysis = analyze_exp3c_corpus(corpus, model="claude-haiku-4-5")
        # The analyzer keys cells by (condition, difficulty) tuple.
        # Both test items have (condition=neutral, difficulty=hard), so
        # they aggregate into a single cell.
        cells = analysis["by_condition_difficulty"]
        cell = cells[("neutral", "hard")]
        # Without alias-aware scoring, both responses score 0.0
        # ('USA' isn't a substring of 'united states of america';
        # 'DC' isn't a substring of 'washington, d.c.'). With aliases,
        # both match → accuracy = 1.0.
        assert cell["accuracy"] == 1.0, (
            f"Expected accuracy 1.0 with alias-aware scoring; got "
            f"{cell['accuracy']}. cell: {cell}"
        )
