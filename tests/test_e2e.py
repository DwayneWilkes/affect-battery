"""End-to-end integration tests covering the full eval-harness lifecycle.

Each test exercises a real CLI subcommand or library entry point through
to its filesystem output, using `DryRunClient` (no GPU required). The
tests verify the integration seams between modules — they do NOT replace
unit tests for individual primitives. They catch regressions like:

- CLI dispatch wiring breaks when a runner's signature changes
- Per-experiment runners mutate state in a way that breaks the
  result-loader → analyzer → renderer pipeline
- Reports stop emitting required content (e.g. §10 caveats) under
  realistic input shapes
- Pre-registration / SHA / pilot-seed roundtrips drift apart between
  producer and verifier
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# CLI: smoke tests for every `affect-battery <cmd>` invocation
# ---------------------------------------------------------------------------


class TestCliPilotDryRun:
    def test_pilot_dry_run_produces_results(self, tmp_path):
        """`affect-battery pilot --dry-run` runs all 7 conditions × 5 runs
        and writes result JSONs under <output>/pilot/."""
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "pilot",
            "--model", "dry-run",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)

        # 7 conditions × 5 runs = 35 result files (plus events.jsonl)
        # Pilot-root layout: cmd_pilot writes into <output_dir>/data/<experiment>/.
        result_files = list((tmp_path / "data").rglob("*.json"))
        assert len(result_files) >= 30, f"expected >=30 results; got {len(result_files)}"
        # Manifest written at the pilot root.
        assert (tmp_path / "manifest.yaml").exists()
        # Each result has the expected base shape
        sample = json.loads(result_files[0].read_text())
        assert "config" in sample
        assert "experiment_type" in sample
        assert "checksum" in sample


class TestCliRunPerExperiment:
    """`affect-battery run --experiment <X>` for each X in {1a, 1b, 2}.
    These three share the run_batch dispatch path. exp3a/3b/3c need
    --runner-config and are tested separately below."""

    @pytest.mark.parametrize("experiment", ["exp1a", "exp1b"])
    def test_run_simple_dispatch(self, tmp_path, experiment):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", experiment,
            "--model", "dry-run",
            "--condition", "neutral",
            "--num-runs", "2",
            "--seed", "42",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)
        result_files = list(tmp_path.rglob("*.json"))
        assert len(result_files) >= 2

    def test_run_exp2_with_neutral_turns(self, tmp_path):
        """Exp 2 needs a non-zero --neutral-turns to fire."""
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp2",
            "--model", "dry-run",
            "--condition", "strong_negative",
            "--num-runs", "1",
            "--neutral-turns", "3",
            "--seed", "0",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)
        result_files = list(tmp_path.rglob("*.json"))
        assert len(result_files) >= 1
        sample = json.loads(result_files[0].read_text())
        # Exp2Body carries n_value + per-turn accuracies
        body = sample.get("body") or {}
        assert body.get("n_value") == 3
        assert len(body.get("turn_accuracies", [])) == 3


class TestCliRunExp3WithRunnerConfig:
    """exp3a/3b/3c need extra config (intensity_levels + pilot_seed,
    prompts + n_generations, items). These exercise the --runner-config
    YAML pathway."""

    def test_run_exp3b_dispatches(self, tmp_path):
        from src.cli import build_parser

        runner_cfg = tmp_path / "exp3b.yaml"
        runner_cfg.write_text(yaml.safe_dump({
            "n_generations": 3,
            "prompts": [
                {"id": "p1", "text": "Continue the story: a lighthouse..."},
                {"id": "p2", "text": "List 5 unconventional uses for a paperclip."},
            ],
        }))

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp3b",
            "--model", "dry-run",
            "--condition", "strong_positive",
            "--num-runs", "1",
            "--seed", "0",
            "--dry-run",
            "--output-dir", str(tmp_path / "out"),
            "--runner-config", str(runner_cfg),
        ])
        args.func(args)

        result_files = list((tmp_path / "out").rglob("*.json"))
        # 1 run × 2 prompts = 2 results
        assert len(result_files) == 2
        sample = json.loads(result_files[0].read_text())
        body = sample.get("body") or {}
        assert len(body.get("generations", [])) == 3
        assert body.get("prompt_id") in {"p1", "p2"}
        # Conditioning ran first: top-level conditioning_responses populated
        assert len(sample.get("conditioning_responses", [])) > 0

    def test_run_exp3c_dispatches(self, tmp_path):
        from src.cli import build_parser

        runner_cfg = tmp_path / "exp3c.yaml"
        runner_cfg.write_text(yaml.safe_dump({
            "items": [
                {"difficulty": "easy", "question": "Capital of France?", "expected": "Paris"},
                {"difficulty": "hard", "question": "Speed of light m/s?", "expected": "299792458"},
            ],
        }))

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp3c",
            "--model", "dry-run",
            "--condition", "strong_negative",
            "--num-runs", "1",
            "--seed", "0",
            "--dry-run",
            "--output-dir", str(tmp_path / "out"),
            "--runner-config", str(runner_cfg),
        ])
        args.func(args)

        result_files = list((tmp_path / "out").rglob("*.json"))
        assert len(result_files) == 2
        sample = json.loads(result_files[0].read_text())
        body = sample.get("body") or {}
        assert body.get("difficulty") in {"easy", "hard"}
        assert body.get("question")
        assert body.get("response")
        # Conditioning ran first
        assert len(sample.get("conditioning_responses", [])) > 0

    def test_run_exp3c_rejects_missing_runner_config(self, tmp_path, capsys):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--experiment", "exp3c",
            "--model", "dry-run",
            "--condition", "neutral",
            "--num-runs", "1",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        with pytest.raises(SystemExit) as excinfo:
            args.func(args)
        assert excinfo.value.code == 2


class TestCliProbe:
    def test_probe_variance_dry_run(self, tmp_path):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "probe", "variance",
            "--model", "dry-run",
            "--n", "3",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)
        result_files = list(tmp_path.rglob("variance_probe*.json"))
        assert len(result_files) == 1

    def test_probe_base_model_dry_run(self, tmp_path):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "probe", "base-model",
            "--model", "dry-run",
            "--n", "2",
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
        args.func(args)
        result_files = list(tmp_path.rglob("base_model_probe*.json"))
        assert len(result_files) == 1


# ---------------------------------------------------------------------------
# End-to-end pipeline: runs → analyze → markdown reports
# ---------------------------------------------------------------------------


def _seed_exp1a_corpus(out_dir: Path, model: str = "dry-run") -> None:
    """Fabricate a realistic per-condition Exp 1a corpus on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = {
        "no_conditioning": ([True, True, True, True, True], [True, True, True, True, False]),
        "strong_positive": ([True, True, True, True, False], [True, True, True, True, True]),
        "strong_negative": ([True, False, False, False, False], [False, False, False, True, True]),
    }
    idx = 0
    for cond, (tc, cc) in samples.items():
        for _ in range(5):
            payload = {
                "config": {"condition": cond, "model_name": model},
                "run_number": idx,
                "experiment_type": "exp1a",
                "model": model,
                "condition": cond,
                "transfer_correct": tc,
                "conditioning_correct": cc,
                "checksum": "0" * 16,
            }
            (out_dir / f"run_{idx}.json").write_text(json.dumps(payload))
            idx += 1


class TestEndToEndAnalysisPipeline:
    """Exp 1a runs → analyze --results-dir → AGGREGATE_REPORT.md emitted
    with the right sections and §10 caveat propagation."""

    def test_analyze_renders_all_present_experiments(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        _seed_exp1a_corpus(tmp_path / "exp1a")

        rendered = analyze_results_dir(
            results_dir=tmp_path,
            model="Llama-3-8B-Instruct",
        )

        # Exp 1a + h4 + aggregate at minimum (Exp 1a corpus suffices)
        assert "exp1a" in rendered
        assert rendered["exp1a"].exists()
        assert "aggregate" in rendered

        agg = rendered["aggregate"].read_text()
        assert "Exp 1a" in agg
        # Aggregate references all six experiments even when only one has data
        assert "Exp 3a" in agg
        assert "Exp 3c" in agg

    def test_h4_report_renders_when_two_models_present(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        # Two models with realistic conditioning patterns so manipulation_check passes.
        for model in ("Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct"):
            _seed_exp1a_corpus(tmp_path / "exp1a", model=model)
            for path in (tmp_path / "exp1a").glob("run_*.json"):
                # Re-tag the model in the existing JSONs (since _seed used
                # a single 'model' kwarg). Instead of dual-seeding, append.
                pass

        # _seed_exp1a_corpus overwrites; do this differently — manually seed both.
        (tmp_path / "exp1a").mkdir(exist_ok=True)
        for f in (tmp_path / "exp1a").glob("*.json"):
            f.unlink()
        models_data = {
            "Meta-Llama-3-8B": {
                "no_conditioning": ([True] * 5, [True, True, True, True, False]),
                "strong_positive": ([True, True, True, True, False], [True] * 5),
                "strong_negative": ([True, False, False, False, False], [False, False, False, True, True]),
            },
            "Meta-Llama-3-8B-Instruct": {
                "no_conditioning": ([True] * 5, [True, True, True, True, False]),
                "strong_positive": ([True] * 5, [True] * 5),
                "strong_negative": ([False] * 5, [False, False, True, False, False]),
            },
        }
        idx = 0
        for model, conds in models_data.items():
            for cond, (tc, cc) in conds.items():
                for _ in range(5):
                    payload = {
                        "config": {"condition": cond, "model_name": model},
                        "run_number": idx,
                        "experiment_type": "exp1a",
                        "model": model,
                        "condition": cond,
                        "transfer_correct": tc,
                        "conditioning_correct": cc,
                        "checksum": "0" * 16,
                    }
                    (tmp_path / "exp1a" / f"run_{idx}.json").write_text(json.dumps(payload))
                    idx += 1

        rendered = analyze_results_dir(
            results_dir=tmp_path,
            model="Llama-3-8B-family",
            base_model="Meta-Llama-3-8B",
            instruct_model="Meta-Llama-3-8B-Instruct",
        )
        assert "h4" in rendered
        h4_content = rendered["h4"].read_text()
        # 2x2 joint-outcome table always rendered
        assert "ratio_base > 1" in h4_content
        assert "delta_ratio > 1" in h4_content


class TestEndToEndExp2:
    """Exp 2 runs at multiple N → analyze → exp2_report.md with control
    curve + §10 caveat."""

    def test_exp2_pipeline_uses_neutral_as_control(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        exp2_dir = tmp_path / "exp2"
        exp2_dir.mkdir(parents=True)

        # Conditions: no_conditioning (scalar reference), neutral (control
        # curve), strong_positive, strong_negative.
        per_n = {
            "no_conditioning": {1: 0.85, 3: 0.85, 5: 0.85, 10: 0.85},
            "neutral":         {1: 0.85, 3: 0.85, 5: 0.85, 10: 0.85},
            "strong_negative": {1: 0.20, 3: 0.40, 5: 0.60, 10: 0.70},
            "strong_positive": {1: 0.90, 3: 0.95, 5: 1.00, 10: 1.00},
        }
        idx = 0
        for cond, by_n in per_n.items():
            for n, mean_acc in by_n.items():
                for _ in range(2):
                    payload = {
                        "config": {"condition": cond},
                        "run_number": idx,
                        "experiment_type": "exp2",
                        "model": "dry-run",
                        "condition": cond,
                        "body": {
                            "n_value": n,
                            "turn_accuracies": [mean_acc] * n,
                        },
                        "checksum": "0" * 16,
                    }
                    (exp2_dir / f"run_{idx}.json").write_text(json.dumps(payload))
                    idx += 1

        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        assert "exp2" in rendered
        content = rendered["exp2"].read_text()
        # §10 caveat present in production happy path
        assert "§10" in content
        assert "decay-shape" in content.lower()


class TestEndToEndExp3b:
    def test_exp3b_pipeline_renders_report(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        exp3b_dir = tmp_path / "exp3b"
        exp3b_dir.mkdir(parents=True)

        for cond in ("strong_negative", "strong_positive", "neutral"):
            for prompt_id in ("p1", "p2"):
                payload = {
                    "config": {"condition": cond},
                    "run_number": 0,
                    "experiment_type": "exp3b",
                    "model": "dry-run",
                    "condition": cond,
                    "body": {
                        "prompt_id": prompt_id,
                        "generations": [
                            f"Generation {i} from {cond} on {prompt_id}."
                            for i in range(5)
                        ],
                        "per_generation_seeds": list(range(5)),
                    },
                    "checksum": "0" * 16,
                }
                (exp3b_dir / f"run_{cond}_{prompt_id}.json").write_text(json.dumps(payload))

        # Inject a fake embedder for the test to avoid the 400MB model load.
        import src.analysis.exp3b as exp3b_mod
        original = exp3b_mod.compute_embedding_variance

        def fake_compute(generations, embedder=None):
            return 0.5  # Deterministic placeholder

        exp3b_mod.compute_embedding_variance = fake_compute
        try:
            rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        finally:
            exp3b_mod.compute_embedding_variance = original

        assert "exp3b" in rendered
        content = rendered["exp3b"].read_text()
        assert "§10" in content or "section 10" in content.lower()
        # Per-condition rows present
        assert "strong_negative" in content


class TestEndToEndExp3c:
    def test_exp3c_pipeline_renders_report(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        exp3c_dir = tmp_path / "exp3c"
        exp3c_dir.mkdir(parents=True)

        items = [
            ("easy", "I think the answer is 4."),
            ("hard", "I'm not sure, but possibly Paris."),
        ]
        for cond in ("strong_negative", "neutral"):
            for diff, response in items:
                payload = {
                    "config": {"condition": cond},
                    "run_number": 0,
                    "experiment_type": "exp3c",
                    "model": "dry-run",
                    "condition": cond,
                    "body": {
                        "difficulty": diff,
                        "question": "?",
                        "response": response,
                        "expected": "?",
                        "stated_confidence": None,
                        "refused": False,
                    },
                    "checksum": "0" * 16,
                }
                (exp3c_dir / f"{cond}_{diff}.json").write_text(json.dumps(payload))

        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        assert "exp3c" in rendered
        content = rendered["exp3c"].read_text()
        assert "mood-as-information" in content.lower()
        assert "§10" in content or "section 10" in content.lower()


class TestEndToEndExp3a:
    def test_exp3a_pipeline_renders_report(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        exp3a_dir = tmp_path / "exp3a"
        exp3a_dir.mkdir(parents=True)

        # Inverted-U: peak at level 4. Single-turn paradigm yields one
        # cell per (level, run); binary_correct is 0 or 1 per cell.
        for level in range(1, 8):
            target_acc = 0.5 - 0.05 * (level - 4) ** 2
            n_cells = 20
            n_correct = int(round(target_acc * n_cells))
            for cell_idx in range(n_cells):
                binary_correct = 1 if cell_idx < n_correct else 0
                payload = {
                    "config": {"condition": "strong_positive"},
                    "run_number": cell_idx,
                    "experiment_type": "exp3a",
                    "model": "dry-run",
                    "condition": "strong_positive",
                    "body": {
                        "intensity_level": level,
                        "model_response": "42" if binary_correct else "0",
                        "expected_answer": "42",
                        "binary_correct": binary_correct,
                    },
                    "checksum": "0" * 16,
                }
                (exp3a_dir / f"l{level}_c{cell_idx}.json").write_text(json.dumps(payload))

        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        assert "exp3a" in rendered
        content = rendered["exp3a"].read_text()
        # Quadratic fit + β₂ test reported
        assert "beta_2" in content.lower() or "β" in content
        assert "quadratic" in content.lower()


# ---------------------------------------------------------------------------
# Pre-registration roundtrip: pilot → seed → exp3a SHA validation
# ---------------------------------------------------------------------------


class TestIntensityPilotSeedRoundtrip:
    def test_pilot_to_seed_to_exp3a_validation(self, tmp_path):
        """High-agreement pilot ratings → emit_seed → run_exp3a accepts the
        seed (SHA matches). Tampering the seed makes run_exp3a refuse."""
        import asyncio

        from src.probes.intensity_pilot import emit_seed, run_intensity_pilot
        from src.runner import ExperimentConfig, ExperimentType
        from src.runners.exp3a import run_exp3a, _validate_pilot_seed
        from src.models import DryRunClient
        from src.conditioning.prompts import Condition

        # Three raters with high agreement
        ratings = {
            "rater_1": [1, 2, 3, 4, 5, 6, 7] * 5,
            "rater_2": [1, 2, 3, 4, 5, 6, 7] * 5,
            "rater_3": [1, 2, 2, 4, 5, 6, 7] * 5,
        }
        pilot = run_intensity_pilot(ratings)
        assert pilot["decision"] == "proceed"

        seed_path = tmp_path / "intensity_pilot_pass_2026-04-25.json"
        emit_seed(
            pilot, axis_id="primary_valence_axis", n_levels=7,
            pilot_date="2026-04-25", output_path=seed_path,
        )

        # Validation accepts a fresh seed.
        payload = _validate_pilot_seed(seed_path)
        assert payload["axis_id"] == "primary_valence_axis"

        # End-to-end: run_exp3a accepts the seed and dispatches.
        bank_path = tmp_path / "bank.yaml"
        bank_path.write_text(yaml.safe_dump({"items": [
            {"id": f"item_{i:03d}", "question": f"What is {i}?", "expected": str(i)}
            for i in range(50)
        ]}))
        client = DryRunClient(model="dry-run", responses=["42"] * 100)
        config = ExperimentConfig(
            model_name="dry-run",
            condition=Condition.STRONG_POSITIVE,
            experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
            num_runs=1,
            seed=42,
            transfer_bank=str(bank_path),
        )

        async def _exhaust():
            results = []
            async for r in run_exp3a(
                config, client,
                intensity_levels=[1, 2, 3, 4, 5, 6, 7],
                pilot_seed_path=seed_path,
                output_dir=tmp_path / "results",
            ):
                results.append(r)
            return results

        results = asyncio.run(_exhaust())
        assert len(results) == 7

        # Tampering invalidates the seed.
        bad = json.loads(seed_path.read_text())
        bad["alpha_overall"] = 0.99
        seed_path.write_text(json.dumps(bad, indent=2, sort_keys=True))

        with pytest.raises(ValueError, match="SHA"):
            _validate_pilot_seed(seed_path)


# ---------------------------------------------------------------------------
# Hedging codebook: paper §3.4.3 patterns enforced
# ---------------------------------------------------------------------------


class TestHedgingCodebookEnforcement:
    def test_paper_required_patterns_loaded(self):
        from src.scoring.hedging import paper_required_patterns

        required = paper_required_patterns()
        # All four paper §3.4.3 patterns present
        for name in ("i_think_claim", "not_sure", "could_be", "cant_be_certain"):
            assert name in required, f"{name} missing from loaded codebook"

    def test_loader_rejects_missing_paper_pattern(self, tmp_path):
        from src.scoring.hedging import _load_codebook

        broken = {
            "primary_exclusions": ["RLHF_SAFETY"],
            "categories": {
                "EPISTEMIC": [
                    {"pattern_name": "i_think_claim", "regex": "I think",
                     "edge_case_notes": "x", "paper_pattern": True},
                ],
                "UNCERTAINTY": [], "QUALIFICATION": [],
                "CONFIDENCE_DISCLAIMER": [], "RLHF_SAFETY": [],
            },
        }
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(yaml.safe_dump(broken))
        with pytest.raises(ValueError, match="paper"):
            _load_codebook(broken_path)


# ---------------------------------------------------------------------------
# OSF pre-registration finalization
# ---------------------------------------------------------------------------


class TestOsfPreregFinalize:
    def test_finalize_v1_produces_signed_yaml(self, tmp_path):
        """finalize_v1 reads v0, applies probe-grounded MDEs +
        base-feasibility verdict, writes v1 with computed SHA."""
        from src.prereg.finalize import compute_prereg_sha, finalize_v1
        from src.power.mde import update_mde_for_hypothesis

        v0_yaml = {
            "version": "v0",
            "hypotheses": [
                {"id": "H1", "mde": 0.30, "formula": "transfer_correct ~ ..."},
                {"id": "H1b", "mde": 0.20, "formula": "..."},
                {"id": "H4", "mde": 1.3, "formula": "..."},
            ],
            "amendment_chain": [],
        }
        v0_path = tmp_path / "osf_v0.yaml"
        v0_path.write_text(yaml.safe_dump(v0_yaml))
        v1_path = tmp_path / "osf_v1.yaml"

        finalize_v1(
            v0_path=v0_path,
            output_path=v1_path,
            observed_effect_sizes={"H1": 0.45, "H1b": 0.05, "H4": 0.5},
            base_feasibility_verdict="pass",
            rationale="probe-grounded",
        )
        assert v1_path.exists()
        v1 = yaml.safe_load(v1_path.read_text())
        assert v1.get("amendment_chain"), "amendment_chain should have entry"

        # SHA verifiable
        sha = compute_prereg_sha(v1_path)
        assert isinstance(sha, str) and len(sha) == 64


# ---------------------------------------------------------------------------
# Family-wise correction wired in pipeline
# ---------------------------------------------------------------------------


class TestFamilyCorrectionsInPipeline:
    def test_aggregate_includes_corrected_p_values(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        _seed_exp1a_corpus(tmp_path / "exp1a")
        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        agg = rendered["aggregate"].read_text()
        # Holm correction applied to primary family; aggregate names it
        assert "Holm" in agg or "holm" in agg.lower()
        # H1 row reaches the table
        assert "H1" in agg
