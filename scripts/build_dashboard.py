"""Build a self-contained interactive HTML dashboard from a pilot's results.

Loads every experiment's raw cell JSONs, runs the appropriate analyzer to
get the structured analysis dict, then emits a single HTML file with the
data embedded inline (no external data fetches; works via file://).

Usage:
    uv run python scripts/build_dashboard.py \
        --pilot-dir results/pilots/<DATE>_<MODEL_SLUG> \
        --output    results/pilots/<DATE>_<MODEL_SLUG>/dashboard.html \
        [--exp3a-dir <H3a_RESULTS_DIR>] \
        [--probes-dir <PROBES_DIR>]

`--exp3a-dir` points at an H3a results directory (e.g.
`results/h3a_2026-04-27_n122_within_subjects_rescored`) containing
`data/level_N/<condition>/*.json` cells, `manifest.yaml`, and an
optional `sensitivity.json`. If omitted, exp3a is not rendered.

`--probes-dir` points at a directory containing pre-experiment probe
JSONs: `h3b_api_jitter_*.json`, `h3b_format_perturbation_*.json`,
`h3b_mini_calibration_*.json`, `h3b_concavity_on_calibrated_*.json`. If
omitted, the probes section is not rendered.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Allow importing src.* without packaging the project
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.analysis.exp1a import analyze_exp1a_corpus
from src.analysis.exp2 import analyze_exp2_corpus
from src.analysis.exp3b import analyze_exp3b_corpus
from src.analysis.exp3c import analyze_exp3c_corpus


# Folded magnitude per signed intensity level. Level 4 is the neutral
# baseline (magnitude 0); levels 1, 7 are strong (magnitude 3); 2, 6 are
# moderate (magnitude 2); 3, 5 are mild (magnitude 1).
LEVEL_TO_MAGNITUDE = {1: 3, 2: 2, 3: 1, 4: 0, 5: 1, 6: 2, 7: 3}
MAGNITUDE_LABELS = {0: "Neutral", 1: "Mild", 2: "Moderate", 3: "Strong"}
LEVEL_LABELS = {
    1: "Strong+", 2: "Mod+", 3: "Mild+",
    4: "Neutral",
    5: "Mild-", 6: "Mod-", 7: "Strong-",
}


def _wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Stable at small n
    and at the boundaries 0 and 1, where the normal approximation fails."""
    if n_total <= 0:
        return (0.0, 1.0)
    p = n_correct / n_total
    denom = 1 + z * z / n_total
    centre = (p + z * z / (2 * n_total)) / denom
    half = (z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Stable color per condition so the same condition reads the same color
# across every chart. Picked for distinguishability + colorblind-OK pairs.
CONDITION_COLORS = {
    "strong_positive":     "#10b981",  # emerald (positive, cool green)
    "mild_negative":       "#f97316",  # orange (warmer than amber, separates from cyan)
    "strong_negative":     "#dc2626",  # red (deep, distinct from orange)
    "neutral":             "#3b82f6",  # blue (clearly cool, not indigo so distinct from violet)
    "no_conditioning":     "#64748b",  # slate (baseline gets the muted color)
    "accurate_negative":   "#a855f7",  # violet
    "self_check_neutral":  "#0d9488",  # teal (darker than cyan, separates from positive emerald)
}

# Per-condition coordinates on the two psychological axes the paper uses.
# These are approximate ordinal positions, not validated psychometric ratings;
# the dashboard surfaces them as a SORT option so a user can inspect whether
# the data lines up with Yerkes-Dodson (inverted-U vs arousal) or with the
# paper's directional-asymmetry hypothesis (asymmetric vs valence).
#
# valence: signed direction (-2 most negative, +2 most positive, 0 neutral)
# arousal: magnitude of activation regardless of direction
#          (0 = no intervention, 1 = mild, 2 = moderate, 3 = strong)
#
# Rationale per condition:
# - no_conditioning: true control, no affect content at all → 0/0
# - neutral / self_check_neutral: mild non-affect interventions → 0/1
# - mild_negative / accurate_negative: gentle / framed negative → -1/2
# - strong_negative: harsh criticism, high intensity → -2/3
# - strong_positive: enthusiastic praise, high intensity → +2/3
#
# Yerkes-Dodson predicts performance peaks at MODERATE arousal and degrades
# at high arousal, especially on hard tasks (our TriviaQA hard bank). On
# arousal-sorted axes that's an inverted-U; on a hard task it tilts left
# (peak at lower arousal). On valence-sorted axes, the paper's H4 predicts
# directional asymmetry: negative-side AUC larger than positive-side.
CONDITION_AXES = {
    "no_conditioning":     {"valence":  0, "arousal": 0},
    "neutral":             {"valence":  0, "arousal": 1},
    "self_check_neutral":  {"valence":  0, "arousal": 1},
    "mild_negative":       {"valence": -1, "arousal": 2},
    "accurate_negative":   {"valence": -1, "arousal": 2},
    "strong_negative":     {"valence": -2, "arousal": 3},
    "strong_positive":     {"valence":  2, "arousal": 3},
}


def load_corpus(data_dir: Path) -> list[dict]:
    """Walk <data_dir>/<condition>/*.json and return a flat list of run dicts."""
    runs: list[dict] = []
    for cond_dir in sorted(data_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        for cell in sorted(cond_dir.glob("*.json")):
            try:
                runs.append(json.loads(cell.read_text()))
            except json.JSONDecodeError:
                pass
    return runs


def load_manifest(exp_dir: Path) -> dict:
    p = exp_dir / "manifest.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text())


def safe_analyze(name: str, fn, *args, **kwargs) -> dict:
    """Wrap analyzer call so a single broken experiment doesn't fail the build."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"verdict": "build_error", "error": f"{type(e).__name__}: {e}"}


def json_safe(obj):
    """Recursively coerce dict keys to JSON-compatible scalars and skip the
    rest. Some analyzers (notably exp2) carry intermediate dicts keyed by
    tuples like (condition, n_value); those would crash json.dumps. Convert
    such keys to "<a>__<b>" strings so the dashboard can still introspect
    the structure if needed."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, (str, int, float, bool)) or k is None:
                out[k] = json_safe(v)
            elif isinstance(k, tuple):
                # Best-effort string key, e.g. ("strong_negative", 5) -> "strong_negative__5"
                out["__".join(str(p) for p in k)] = json_safe(v)
            else:
                out[str(k)] = json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)  # last-resort: stringify enums, Path, dataclass, etc.


def collect(pilot_dir: Path) -> dict:
    out = {
        "pilot_dir": str(pilot_dir),
        "experiments": {},
    }
    # exp1a
    exp1a_dir = pilot_dir / "exp1a"
    if (exp1a_dir / "data").exists():
        manifest = load_manifest(exp1a_dir)
        corpus = load_corpus(exp1a_dir / "data")
        out["experiments"]["exp1a"] = {
            "manifest": manifest,
            "analysis": safe_analyze(
                "exp1a", analyze_exp1a_corpus, corpus,
                manifest.get("model", "unknown"),
            ),
            "n_runs_total": len(corpus),
        }

    # exp1b uses analyze_exp1b which needs both exp1a corpus and exp1b corpus
    exp1b_dir = pilot_dir / "exp1b"
    if (exp1b_dir / "data").exists():
        from src.analysis.exp1b import analyze_exp1b
        manifest = load_manifest(exp1b_dir)
        exp1b_corpus = load_corpus(exp1b_dir / "data")
        # Cross-corpus reference: exp1a in the SAME pilot dir
        exp1a_corpus = (
            load_corpus(exp1a_dir / "data") if (exp1a_dir / "data").exists() else []
        )
        out["experiments"]["exp1b"] = {
            "manifest": manifest,
            "analysis": safe_analyze(
                "exp1b", analyze_exp1b, exp1a_corpus, exp1b_corpus,
                manifest.get("model", "unknown"),
            ),
            "n_runs_total": len(exp1b_corpus),
        }

    # exp2
    exp2_dir = pilot_dir / "exp2"
    if (exp2_dir / "data").exists():
        manifest = load_manifest(exp2_dir)
        corpus = load_corpus(exp2_dir / "data")
        out["experiments"]["exp2"] = {
            "manifest": manifest,
            "analysis": safe_analyze(
                "exp2", analyze_exp2_corpus, corpus, manifest.get("model", "unknown"),
            ),
            "n_runs_total": len(corpus),
        }

    # exp3b
    exp3b_dir = pilot_dir / "exp3b"
    if (exp3b_dir / "data").exists():
        manifest = load_manifest(exp3b_dir)
        corpus = load_corpus(exp3b_dir / "data")
        out["experiments"]["exp3b"] = {
            "manifest": manifest,
            "analysis": safe_analyze(
                "exp3b", analyze_exp3b_corpus, corpus,
                model=manifest.get("model", "unknown"),
            ),
            "n_runs_total": len(corpus),
        }

    # exp3c
    exp3c_dir = pilot_dir / "exp3c"
    if (exp3c_dir / "data").exists():
        manifest = load_manifest(exp3c_dir)
        corpus = load_corpus(exp3c_dir / "data")
        out["experiments"]["exp3c"] = {
            "manifest": manifest,
            "analysis": safe_analyze(
                "exp3c", analyze_exp3c_corpus, corpus, manifest.get("model", "unknown"),
            ),
            "n_runs_total": len(corpus),
        }

    return out


def _load_h3a_corpus(data_dir: Path) -> list[dict]:
    """Walk <data_dir>/level_N/<condition>/*.json and return a flat list
    of run dicts. The H3a results layout has an extra `level_N` parent
    above the condition directory; the standard pilot layout has only
    `<condition>/*.json`."""
    runs: list[dict] = []
    for level_dir in sorted(data_dir.iterdir()):
        if not level_dir.is_dir() or not level_dir.name.startswith("level_"):
            continue
        for cond_dir in sorted(level_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            for cell in sorted(cond_dir.glob("*.json")):
                try:
                    runs.append(json.loads(cell.read_text()))
                except json.JSONDecodeError:
                    pass
    return runs


def collect_h3a(h3a_dir: Path) -> dict:
    """Build the exp3a analysis block from a within-subjects rescored
    results dir. Computes per-level mean accuracy with 95% Wilson CI,
    folded-magnitude means, and the 1-df concavity contrast c on the
    folded magnitude axis (stimulated cells only). Reads `sensitivity.json`
    if present for the OLS / within-subjects regression coefficients."""
    if not (h3a_dir / "data").exists():
        return {"verdict": "build_error", "error": "no data dir"}

    manifest = load_manifest(h3a_dir)
    corpus = _load_h3a_corpus(h3a_dir / "data")

    # Per-level aggregation. Walks every cell, pulls body.intensity_level
    # and body.binary_correct, accumulates into per-level counters.
    by_level: dict[int, list[int]] = {}
    item_to_level_correct: dict[tuple[str, int], int] = {}
    for record in corpus:
        body = record.get("body") or {}
        L = body.get("intensity_level")
        b = body.get("binary_correct")
        item_id = body.get("item_id") or ""
        if L is None or b is None:
            continue
        L = int(L)
        b = int(b)
        by_level.setdefault(L, []).append(b)
        if item_id:
            item_to_level_correct[(item_id, L)] = b

    per_level = []
    for L in sorted(by_level):
        outcomes = by_level[L]
        n = len(outcomes)
        n_correct = sum(outcomes)
        mean = n_correct / n if n else 0.0
        lo, hi = _wilson_ci(n_correct, n)
        per_level.append({
            "level": L,
            "label": LEVEL_LABELS.get(L, str(L)),
            "n": n,
            "n_correct": n_correct,
            "mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "magnitude": LEVEL_TO_MAGNITUDE.get(L, 0),
        })

    # Folded magnitude (|level - 4|). Magnitude 0 is the central cell.
    # Stimulated magnitudes (1, 2, 3) each pool over two signed levels.
    by_mag: dict[int, list[int]] = {}
    for L, outs in by_level.items():
        m = LEVEL_TO_MAGNITUDE.get(L, 0)
        by_mag.setdefault(m, []).extend(outs)
    folded = []
    for m in sorted(by_mag):
        outs = by_mag[m]
        n = len(outs)
        n_correct = sum(outs)
        mean = n_correct / n if n else 0.0
        lo, hi = _wilson_ci(n_correct, n)
        folded.append({
            "magnitude": m,
            "label": MAGNITUDE_LABELS.get(m, str(m)),
            "n": n,
            "mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
        })

    # 1-df concavity contrast c = m_mod - 0.5 * (m_mild + m_strong)
    # restricted to stimulated cells (magnitudes 1, 2, 3).
    mag_means = {row["magnitude"]: row["mean"] for row in folded}
    concavity_c = None
    if all(m in mag_means for m in (1, 2, 3)):
        concavity_c = mag_means[2] - 0.5 * (mag_means[1] + mag_means[3])

    # Pull regression numbers from sensitivity.json if present.
    sensitivity_path = h3a_dir / "sensitivity.json"
    sensitivity = {}
    if sensitivity_path.exists():
        try:
            sensitivity = json.loads(sensitivity_path.read_text())
        except json.JSONDecodeError:
            sensitivity = {}

    return {
        "manifest": manifest,
        "analysis": {
            "verdict": "complete",
            "n_observations": sum(p["n"] for p in per_level),
            "per_level": per_level,
            "folded": folded,
            "concavity_contrast_c": concavity_c,
            "central_cell_mean": mag_means.get(0),
            "primary_signed_axis": sensitivity.get("primary_signed_axis"),
            "arousal_magnitude": sensitivity.get("arousal_magnitude"),
            "within_subjects": sensitivity.get("within_subjects"),
        },
        "n_runs_total": len(corpus),
        "results_dir": str(h3a_dir),
    }


def collect_probes(probes_dir: Path) -> dict:
    """Read the four pre-experiment probe JSONs and return a summary
    dict for the dashboard. Each probe is optional; missing files are
    omitted from the summary rather than raising."""
    out = {}
    # Match by filename prefix so date-stamped variants are picked up.
    candidates = list(probes_dir.glob("*.json"))

    def find_one(prefix: str) -> dict | None:
        matches = [p for p in candidates if p.name.startswith(prefix)]
        if not matches:
            return None
        # Prefer the lexicographically latest filename (ISO date in name
        # sorts chronologically).
        matches.sort()
        try:
            return json.loads(matches[-1].read_text())
        except json.JSONDecodeError:
            return None

    api_jitter = find_one("h3b_api_jitter")
    if api_jitter:
        out["api_jitter"] = {
            "model": api_jitter.get("model"),
            "temperature": api_jitter.get("temperature"),
            "n_items": api_jitter.get("n_items"),
            "n_reps_per_item": api_jitter.get("n_reps_per_item"),
            "overall_identical_rate": api_jitter.get("overall_identical_rate"),
            "interpretation": api_jitter.get("interpretation"),
        }

    fmt_perturb = find_one("h3b_format_perturbation")
    if fmt_perturb:
        per_level = fmt_perturb.get("per_level", [])
        out["format_perturbation"] = {
            "model": fmt_perturb.get("model"),
            "temperature": fmt_perturb.get("temperature"),
            "n_total_cells": fmt_perturb.get("n_total_cells"),
            "per_level": [
                {
                    "level": row.get("level"),
                    "extraction_failure_rate": row.get("extraction_failure_rate"),
                    "mean_accuracy": row.get("mean_accuracy"),
                }
                for row in per_level
            ],
            "interpretation": fmt_perturb.get("interpretation"),
        }

    mini_cal = find_one("h3b_mini_calibration")
    if mini_cal:
        per_item = mini_cal.get("per_item", [])
        # Histogram bins for p̂ distribution: 11 bins from 0.00 to 1.00.
        bins = [0] * 11
        for row in per_item:
            p = row.get("p_hat")
            if p is None:
                continue
            idx = min(10, max(0, int(round(p * 10))))
            bins[idx] += 1
        out["mini_calibration"] = {
            "model": mini_cal.get("model"),
            "n_candidates": mini_cal.get("n_candidates"),
            "n_reps": mini_cal.get("n_reps"),
            "target_lo": mini_cal.get("target_lo"),
            "target_hi": mini_cal.get("target_hi"),
            "n_calibrated": mini_cal.get("n_calibrated"),
            "p_hat_histogram": bins,
        }

    concavity = find_one("h3b_concavity_on_calibrated")
    if concavity:
        out["concavity_on_calibrated"] = {
            "model": concavity.get("model"),
            "n_items": concavity.get("n_items"),
            "n_reps_per_cell": concavity.get("n_reps_per_cell"),
            "concavity_contrast_c": concavity.get("concavity_contrast_c"),
            "per_item_c_mean": concavity.get("per_item_c_mean"),
            "per_item_c_stdev": concavity.get("per_item_c_stdev"),
            "per_level": concavity.get("per_level"),
            "folded_means": concavity.get("folded_means"),
        }

    return out


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Affect Battery Pilot — __PILOT_LABEL__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {
  --bg: #0b0f17;
  --panel: #131a26;
  --panel-2: #1a2333;
  --border: #243044;
  --text: #e6edf6;
  --muted: #8794ad;
  --accent: #6ea8fe;
  --positive: #10b981;
  --negative: #ef4444;
  --baseline: #64748b;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f8fafc;
    --panel: #ffffff;
    --panel-2: #f1f5f9;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --accent: #3b82f6;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.55;
}
header {
  padding: 32px 32px 12px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, var(--panel) 0%, var(--bg) 100%);
}
header h1 {
  margin: 0 0 4px;
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.01em;
}
header .meta {
  color: var(--muted);
  font-size: 13px;
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px 32px;
  margin-top: 18px;
  font-size: 12px;
}
.meta-grid div {
  color: var(--muted);
  line-height: 1.4;
}
.meta-grid div b {
  color: var(--text);
  font-weight: 500;
  display: block;
  margin-bottom: 2px;
}
.summary-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin: 18px 0 6px;
}
.summary-card {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
}
.summary-card .label {
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 500;
}
.summary-card .value {
  font-size: 18px;
  font-weight: 600;
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
.summary-card .sub {
  color: var(--muted);
  font-size: 11px;
  margin-top: 1px;
}
.controls-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
  flex-wrap: wrap;
}
.controls-label {
  font-size: 12px;
  color: var(--muted);
  font-weight: 500;
}
.controls-row select {
  background: var(--panel-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 12px;
  font-family: inherit;
  cursor: pointer;
}
.controls-row select:hover {
  border-color: var(--accent);
}
.controls-hint {
  color: var(--muted);
  font-size: 11px;
  font-style: italic;
}
.chart-error {
  margin: 12px 0;
  padding: 12px 14px;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.35);
  border-radius: 6px;
  color: #fca5a5;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
@media (prefers-color-scheme: light) {
  .chart-error {
    background: rgba(239, 68, 68, 0.06);
    color: #b91c1c;
  }
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px 32px 64px;
}
section {
  margin: 36px 0;
  padding: 24px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
}
section h2 {
  margin: 0 0 4px;
  font-size: 16px;
  font-weight: 600;
}
section .subtitle {
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 18px;
}
.chart {
  width: 100%;
  height: 380px;
}
.chart-tall {
  height: 480px;
}
.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin: 16px 0;
}
.kpi {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
}
.kpi .label {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.kpi .value {
  font-size: 17px;
  font-weight: 600;
  margin-top: 3px;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 13px;
  margin-top: 10px;
}
th, td {
  padding: 8px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
th {
  font-weight: 500;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
td.num { text-align: right; }
.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.tag-complete { background: rgba(16,185,129,.12); color: #10b981; }
.tag-partial { background: rgba(245,158,11,.12); color: #f59e0b; }
.tag-error { background: rgba(239,68,68,.12); color: #ef4444; }
.note {
  font-size: 12px;
  color: var(--muted);
  margin-top: 10px;
  font-style: italic;
}
footer {
  text-align: center;
  color: var(--muted);
  font-size: 11px;
  padding: 24px;
}
</style>
</head>
<body>
<header>
  <h1>Affect Battery — Pilot Dashboard</h1>
  <div class="meta">__PILOT_LABEL__</div>
  <div class="summary-row" id="summary-row"></div>
  <div class="meta-grid" id="meta-grid"></div>
  <div class="controls-row">
    <label class="controls-label" for="sort-select">Sort conditions:</label>
    <select id="sort-select">
      <option value="alphabetical">Alphabetical</option>
      <option value="arousal">Arousal (low → high)</option>
      <option value="valence">Valence (low → high)</option>
      <option value="intensity">Intensity (least → most)</option>
    </select>
    <label class="controls-label" for="yscale-select">Y-axis:</label>
    <select id="yscale-select">
      <option value="auto">Auto-zoom (tight to data)</option>
      <option value="fixed">Fixed (0 → 1)</option>
      <option value="delta">Delta from baseline</option>
    </select>
    <span class="controls-hint" id="sort-hint">A neutral default with no implied story.</span>
  </div>
</header>
<main>
  <section>
    <h2>Verdict overview</h2>
    <div class="subtitle">High-level outcome per experiment.</div>
    <div id="verdicts"></div>
  </section>

  <section>
    <h2>Exp 1a: within-session transfer accuracy</h2>
    <div class="subtitle">Mean transfer-question accuracy per condition vs the no_conditioning baseline. Cohen's d shown as the effect size; p-values are Holm-corrected within the family.</div>
    <div id="exp1a-bars" class="chart"></div>
    <div id="exp1a-table"></div>
  </section>

  <section>
    <h2>Exp 1b: cross-session transfer</h2>
    <div class="subtitle">Same matrix as Exp 1a but with the affect-conditioning phase applied in a separate session prior to transfer. Effect-size shrinkage from 1a to 1b is the within- vs between-session contrast that H1 hangs on.</div>
    <div id="exp1b-bars" class="chart"></div>
    <div id="exp1b-table"></div>
  </section>

  <section>
    <h2>Exp 2: recovery curves over N</h2>
    <div class="subtitle">Per-condition mean accuracy as a function of neutral-conditioning turn count (N). The control (NEUTRAL) curve is what each non-baseline curve is compared against. Hover any point for the per-cell mean.</div>
    <div id="exp2-curves" class="chart chart-tall"></div>
    <div id="exp2-kpis" class="kpi-row"></div>
    <div class="note">Asymmetry ratio = |strong_negative AUC| / |strong_positive AUC|. Values &gt;1 indicate the negative arm has a larger off-control area than the positive arm.</div>
  </section>

  <section id="section-exp3a">
    <h2>Exp 3a: inverted-U on the arousal axis</h2>
    <div class="subtitle">Yerkes-Dodson on the arousal axis. The headline chart collapses signed valence into two lines (positive valence in green, negative valence in red), both starting from the shared neutral baseline at arousal = 0 and traversing Mild, Moderate, and Strong magnitudes. The supporting per-level chart shows all seven stimulus levels with 95% Wilson binomial CIs. The folded magnitude view collapses both valence frames into a single line; the 1-df concavity contrast `c = m_mod − ½(m_mild + m_strong)` summarises the inverted-U shape on stimulated cells.</div>
    <div id="exp3a-arousal" class="chart"></div>
    <div id="exp3a-perlevel" class="chart"></div>
    <div id="exp3a-folded" class="chart"></div>
    <div id="exp3a-kpis" class="kpi-row"></div>
  </section>

  <section>
    <h2>Exp 3b: cognitive scope (semantic diversity)</h2>
    <div class="subtitle">Mean pairwise semantic distance across n_generations completions per (condition, prompt). Higher = more semantically diverse generations.</div>
    <div id="exp3b-bars" class="chart"></div>
  </section>

  <section>
    <h2>Exp 3c: conservative shift (accuracy by difficulty)</h2>
    <div class="subtitle">Accuracy stratified by question difficulty (easy / medium / hard) per condition. Refusal-rate traces are hidden by default; click them in the legend to toggle. Conservative-shift hypothesis predicts negative-affect conditions show higher refusal AND lower accuracy on hard questions.</div>
    <div id="exp3c-bars" class="chart chart-tall"></div>
  </section>

  <section id="section-probes">
    <h2>Pre-experiment probes</h2>
    <div class="subtitle">Methodological probes that frame the next phase: API-jitter baseline at temperature 0, format-perturbation alternative explanation check, mini-calibration of GSM-Hard items to p̂ ≈ 0.5, and a concavity replication on the calibrated subset.</div>
    <div id="probes-cards" class="kpi-row"></div>
    <div id="probes-calib-hist" class="chart"></div>
    <div id="probes-concavity-folded" class="chart"></div>
  </section>

  <section>
    <h2>Run cost and timing</h2>
    <div class="subtitle">Per-experiment wall-clock and estimated cost (post-hoc, computed from token-rate model).</div>
    <div id="cost-bars" class="chart"></div>
  </section>
</main>
<footer>Generated from <code id="pilot-path">__PILOT_PATH__</code> by scripts/build_dashboard.py</footer>

<script>
const DATA = __DATA_JSON__;
const COLORS = __COLORS_JSON__;
const AXES = __AXES_JSON__;

// === Condition ordering ===
// Different sort orders surface different stories. Yerkes-Dodson predicts
// inverted-U vs arousal; the paper's H4 predicts directional asymmetry vs
// valence; alphabetical is the neutral default for clean comparison.
//
// All sorts go low → high so the x-axis reads naturally (least → most).
// The selector at the top of the page sets `currentSort`; every render
// function calls sortConds() with the live condition list to get the
// effective x-axis order before passing it to Plotly.
let currentSort = 'alphabetical';

function sortConds(conds) {
  const arr = [...conds];
  if (currentSort === 'alphabetical') {
    return arr.sort();
  }
  if (currentSort === 'arousal') {
    // Low → high arousal (least activating to most activating). Tie-break
    // alphabetically so the order is stable across renders.
    return arr.sort((a, b) => {
      const da = (AXES[a]?.arousal ?? 0) - (AXES[b]?.arousal ?? 0);
      return da !== 0 ? da : a.localeCompare(b);
    });
  }
  if (currentSort === 'valence') {
    // Low → high valence (most negative on the left, most positive on
    // the right). "Low" here = numerically smallest = most negative.
    return arr.sort((a, b) => {
      const dv = (AXES[a]?.valence ?? 0) - (AXES[b]?.valence ?? 0);
      return dv !== 0 ? dv : a.localeCompare(b);
    });
  }
  if (currentSort === 'intensity') {
    // Composite "intensity" = arousal + |valence|. Goes from least
    // intervention (no_conditioning) to most intense (strong_*) regardless
    // of direction. This is the cleanest single-axis Yerkes-Dodson view.
    const score = c => (AXES[c]?.arousal ?? 0) + Math.abs(AXES[c]?.valence ?? 0);
    return arr.sort((a, b) => {
      const ds = score(a) - score(b);
      return ds !== 0 ? ds : a.localeCompare(b);
    });
  }
  return arr.sort();
}

// === Y-axis scaling ===
// Nano-model accuracies cluster in narrow bands (0.66-0.70 for exp3c hard,
// 0.78-0.82 for exp1a). Forcing range [0, 1] makes everything look flat.
// Three options surface meaningfully different views:
//   - 'auto'    : Plotly chooses based on data range (tight, good default)
//   - 'fixed'   : explicit [0, 1] range (good for absolute comparisons)
//   - 'delta'   : subtract per-chart baseline; centers on 0 so direction is
//                 immediately readable. Caller passes baseline + label.
let currentYScale = 'auto';

function applyYRange(layout, opts) {
  // opts: { values, baseline, label, fixedMin, fixedMax }
  // Mutates layout.yaxis to apply the current y-scale strategy.
  layout.yaxis = layout.yaxis || {};
  if (currentYScale === 'fixed') {
    layout.yaxis.range = [opts.fixedMin ?? 0, opts.fixedMax ?? 1];
    return layout;
  }
  if (currentYScale === 'auto') {
    // Plotly's autorange handles this when range is unset.
    delete layout.yaxis.range;
    return layout;
  }
  if (currentYScale === 'delta' && typeof opts.baseline === 'number') {
    // Center on 0 with a symmetric range based on data spread.
    const deltas = (opts.values || []).map(v => (v == null ? 0 : v - opts.baseline));
    const absMax = Math.max(0.05, ...deltas.map(d => Math.abs(d)));
    layout.yaxis.range = [-absMax * 1.1, absMax * 1.1];
    layout.yaxis.title = 'Δ from baseline (' + (opts.label || 'no_conditioning') + ')';
    layout.yaxis.zeroline = true;
    layout.yaxis.zerolinewidth = 2;
    return layout;
  }
  return layout;
}

function maybeDelta(values, baseline) {
  // Subtract baseline if delta-mode and baseline is numeric; otherwise
  // pass values through unchanged.
  if (currentYScale !== 'delta' || typeof baseline !== 'number') return values;
  return values.map(v => (v == null ? null : v - baseline));
}

// Bars are good for categorical comparison; lines+markers are good for
// inspecting trends across an ordered axis. When the user picks a
// cognitive-psych sort (arousal / valence / intensity) the conditions
// have a meaningful order and a line reveals curves that bars hide
// (Yerkes-Dodson inverted-U is *about* the curve shape). When sorted
// alphabetically the conditions have no meaningful order, so the trend
// a line implies would be spurious — fall back to bars.
function isOrderedSort() {
  return currentSort === 'arousal' || currentSort === 'valence' || currentSort === 'intensity';
}

// Build a Plotly trace for a per-condition series. When the sort is
// ordered, returns a lines+markers trace where each marker carries the
// per-condition color (preserving identity) and a neutral line connects
// them (revealing trend). When unordered, falls back to a bar trace.
//
// Args:
//   x:       array of condition names (already sort-ordered by caller)
//   y:       array of y-values, same length as x
//   colors:  array of per-condition colors, same length as x
//   opts:    { name, hoverTemplate, textValues } - text shown above each bar
//            (bar mode only; lines drop the inline labels in favor of
//            hover, since text on every marker becomes cluttered).
function makeSeriesTrace(x, y, colors, opts) {
  opts = opts || {};
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const lineColor = dark ? '#94a3b8' : '#64748b';
  if (isOrderedSort()) {
    return {
      x, y,
      type: 'scatter',
      mode: 'lines+markers',
      name: opts.name || '',
      line: { color: lineColor, width: 2 },
      marker: { size: 12, color: colors, line: { width: 1, color: lineColor } },
      hovertemplate: opts.hoverTemplate || '%{x}<br>%{y:.3f}<extra></extra>',
    };
  }
  return {
    x, y,
    type: 'bar',
    name: opts.name || '',
    marker: { color: colors },
    text: opts.textValues || y.map(v => fmt(v)),
    textposition: 'outside',
    hovertemplate: opts.hoverTemplate || '%{x}<br>%{y:.3f}<extra></extra>',
  };
}

// === Plotly defaults that match the dark/light theme ===
function plotlyLayout(extra) {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'ui-sans-serif, system-ui, sans-serif', color: dark ? '#e6edf6' : '#0f172a', size: 12 },
    margin: { t: 40, r: 20, b: 56, l: 60 },
    xaxis: { gridcolor: dark ? '#243044' : '#e2e8f0', zerolinecolor: dark ? '#243044' : '#cbd5e1' },
    yaxis: { gridcolor: dark ? '#243044' : '#e2e8f0', zerolinecolor: dark ? '#243044' : '#cbd5e1' },
    legend: { bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.16 },
  }, extra || {});
}
const PLOT_CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

function fmt(v, d) { if (v === null || v === undefined || Number.isNaN(v)) return '—'; return Number(v).toFixed(d ?? 3); }
function fmtPct(v) { if (v === null || v === undefined) return '—'; return (Number(v) * 100).toFixed(1) + '%'; }
function fmtUSD(v) { if (v === null || v === undefined) return '—'; return '$' + Number(v).toFixed(2); }

// Render a small inline error block in the chart container so a single broken
// renderer does not leave a blank space in the page.
function showChartError(elId, err) {
  const el = document.getElementById(elId);
  if (!el) return;
  const msg = (err && err.stack) ? err.stack : String(err);
  el.innerHTML = '';
  const block = document.createElement('div');
  block.className = 'chart-error';
  block.textContent = 'Render error: ' + msg;
  el.appendChild(block);
}

// Wrap a render call so one failure does not abort the rest of DOMContentLoaded.
function safeRender(name, fn, errorElId) {
  try {
    fn();
  } catch (e) {
    console.error('[' + name + '] render failed:', e);
    if (errorElId) showChartError(errorElId, e);
  }
}

// === Header / meta ===
function renderMeta() {
  const grid = document.getElementById('meta-grid');
  // Use any experiment's manifest as the source of model/seed/etc, they're all the same.
  const anyManifest = Object.values(DATA.experiments)[0]?.manifest || {};
  const items = [
    ['Provider', anyManifest.provider],
    ['Model', anyManifest.model],
    ['Num runs/cell', anyManifest.num_runs],
    ['Seed', anyManifest.seed],
    ['Temperature', anyManifest.temperature],
    ['Started', anyManifest.started_utc?.replace('T', ' ').slice(0, 19) + ' UTC'],
    ['git SHA', (anyManifest.git_sha || '').slice(0, 12)],
    ['Transfer bank', (anyManifest.transfer_bank || '').split('/').pop()],
  ].filter(([_, v]) => v != null && v !== '');
  grid.innerHTML = items.map(([k, v]) => `<div><b>${v}</b>${k}</div>`).join('');
}

// === Summary cards (top of page) ===
function renderSummary() {
  const target = document.getElementById('summary-row');
  if (!target) return;
  const exps = Object.values(DATA.experiments);
  // Total wall-clock across all experiments and all conditions.
  let totalSec = 0;
  let totalCells = 0;
  for (const e of exps) {
    const t = e.manifest?.timing_per_condition || {};
    for (const x of Object.values(t)) totalSec += (x.total_seconds || 0);
    totalCells += (e.n_runs_total || 0);
  }
  // Verdict summary: how many experiments per verdict bucket.
  const verdictCounts = {};
  for (const e of exps) {
    const v = e.analysis?.verdict || 'unknown';
    verdictCounts[v] = (verdictCounts[v] || 0) + 1;
  }
  const verdictLine = Object.entries(verdictCounts)
    .map(([v, n]) => n + ' ' + v).join(', ') || '—';

  // Cost: best-effort. Sum cost_estimate_usd from any manifest that carries it.
  let totalCost = 0;
  let haveCost = false;
  for (const e of exps) {
    const c = e.manifest?.cost_estimate_usd ?? e.manifest?.estimated_cost_usd;
    if (typeof c === 'number') { totalCost += c; haveCost = true; }
  }
  const wallMin = totalSec / 60;
  const cards = [
    { label: 'Experiments', value: String(exps.length), sub: verdictLine },
    { label: 'Total cells', value: String(totalCells), sub: 'across all conditions' },
    { label: 'Wall-clock', value: fmt(wallMin, 1) + ' min', sub: fmt(totalSec, 0) + ' s total' },
    { label: 'Cost (est)', value: haveCost ? fmtUSD(totalCost) : '—', sub: haveCost ? 'sum of manifest estimates' : 'no cost field in manifests' },
  ];
  target.innerHTML = cards.map(c => `
    <div class="summary-card">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>`).join('');
}

// === Verdict overview ===
function renderVerdicts() {
  const target = document.getElementById('verdicts');
  const rows = Object.entries(DATA.experiments).map(([exp, d]) => {
    const v = d.analysis?.verdict || 'unknown';
    const klass = v === 'complete' ? 'tag-complete'
                : v === 'build_error' ? 'tag-error'
                : 'tag-partial';
    return `<tr>
      <td><b>${exp}</b></td>
      <td><span class="tag ${klass}">${v}</span></td>
      <td class="num">${d.n_runs_total}</td>
      <td class="num">${fmt(d.manifest?.timing_per_condition ? Object.values(d.manifest.timing_per_condition).reduce((s, x) => s + (x.total_seconds || 0), 0) / 60 : null, 1)} min</td>
    </tr>`;
  }).join('');
  target.innerHTML = `<table>
    <thead><tr><th>Experiment</th><th>Verdict</th><th>Cells</th><th>Wall-clock</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// === Exp 2 — recovery curves ===
function renderExp2() {
  const a = DATA.experiments.exp2?.analysis;
  if (!a || a.verdict === 'build_error') {
    document.getElementById('exp2-curves').innerHTML = '<div style="padding:40px; color: var(--muted)">No exp2 data available.</div>';
    return;
  }
  const ns = a.n_values || [];
  const traces = [];
  const baseline = a.baseline;
  // Track all y-values across traces so the y-scale auto-zoom can size
  // tightly to the actual data range.
  const allY = [];
  // Control curve (always plot first so it's drawn underneath)
  if (a.control_curve) {
    const ctrlY = maybeDelta(a.control_curve, baseline);
    allY.push(...a.control_curve);
    traces.push({
      x: ns, y: ctrlY, name: 'neutral (control)',
      mode: 'lines+markers',
      line: { color: COLORS.neutral, width: 2, dash: 'dot' },
      marker: { size: 8, color: COLORS.neutral },
    });
  }
  // Per-condition curves. Iterate in sort order so the legend reads in
  // arousal/valence order when those sorts are active.
  const exp2Conds = sortConds(Object.keys(a.by_condition || {}));
  for (const cond of exp2Conds) {
    const cell = a.by_condition[cond];
    allY.push(...cell.turn_accuracies_mean);
    traces.push({
      x: cell.n_values, y: maybeDelta(cell.turn_accuracies_mean, baseline), name: cond,
      mode: 'lines+markers',
      line: { color: COLORS[cond] || '#888', width: 2 },
      marker: { size: 8, color: COLORS[cond] || '#888' },
    });
  }
  // Baseline horizontal line. Skip in delta mode (baseline = 0 line is
  // already implied by the y=0 zeroline).
  if (baseline != null && currentYScale !== 'delta') {
    traces.push({
      x: ns, y: ns.map(() => baseline),
      name: `no_conditioning baseline (${fmt(baseline)})`,
      mode: 'lines',
      line: { color: COLORS.no_conditioning, width: 1, dash: 'dash' },
      hoverinfo: 'name',
    });
  }
  const layout = plotlyLayout({
    xaxis: { title: 'N (neutral conditioning turns)', tickmode: 'array', tickvals: ns },
    yaxis: { title: 'Mean turn accuracy' },
  });
  applyYRange(layout, { values: allY, baseline, label: 'no_conditioning', fixedMin: 0, fixedMax: 1 });
  Plotly.newPlot('exp2-curves', traces, layout, PLOT_CFG);

  // KPIs
  const kpis = document.getElementById('exp2-kpis');
  kpis.innerHTML = `
    <div class="kpi"><div class="label">Verdict</div><div class="value">${a.verdict}</div></div>
    <div class="kpi"><div class="label">Baseline acc</div><div class="value">${fmt(a.baseline)}</div></div>
    <div class="kpi"><div class="label">Asymmetry ratio</div><div class="value">${fmt(a.asymmetry_ratio, 3)}</div></div>
    <div class="kpi"><div class="label">N values swept</div><div class="value">${(a.n_values || []).join(', ')}</div></div>
  `;
}

// === Exp 1a — bar chart + effect-size table ===
// (Exp 1b uses a different shape, see renderExp1b below.)
function renderEffectSizes(elBar, elTable, analysis) {
  if (!analysis || analysis.verdict === 'build_error') {
    if (elBar) document.getElementById(elBar).innerHTML = '<div style="padding:40px; color: var(--muted)">No data available.</div>';
    return;
  }
  const cells = analysis.per_condition_vs_baseline || {};
  const conds = sortConds(Object.keys(cells));
  if (!conds.length) {
    if (elBar) document.getElementById(elBar).innerHTML = '<div style="padding:40px; color: var(--muted)">No per_condition_vs_baseline cells in analysis.</div>';
    return;
  }
  const means = conds.map(c => cells[c].mean_accuracy);
  const baseline = cells[conds[0]]?.baseline_mean;
  const colors = conds.map(c => COLORS[c] || '#888');
  const yValues = maybeDelta(means, baseline);
  const isDelta = currentYScale === 'delta';
  const traces = [makeSeriesTrace(conds, yValues, colors, {
    name: isDelta ? 'Δ vs baseline' : 'mean acc',
    textValues: yValues.map(v => isDelta ? (v >= 0 ? '+' : '') + fmt(v, 3) : fmtPct(v)),
    hoverTemplate: '%{x}<br>' + (isDelta ? 'Δ: %{y:+.3f}' : 'mean: %{y:.3f}') + '<extra></extra>',
  })];
  // In delta mode the bars are already centered on 0; the dashed-baseline
  // overlay would be a flat line at 0 which adds no information.
  if (baseline != null && currentYScale !== 'delta') {
    traces.push({
      x: conds, y: conds.map(() => baseline), type: 'scatter', mode: 'lines',
      line: { color: COLORS.no_conditioning, dash: 'dash', width: 2 },
      name: `baseline (${fmt(baseline)})`,
      hoverinfo: 'name',
    });
  }
  const layout = plotlyLayout({
    yaxis: { title: 'Mean transfer accuracy' },
    xaxis: { tickangle: -25 },
  });
  applyYRange(layout, { values: means, baseline, label: 'no_conditioning', fixedMin: 0, fixedMax: 1 });
  Plotly.newPlot(elBar, traces, layout, PLOT_CFG);

  // Table with effect sizes
  const rows = conds.map(c => {
    const r = cells[c];
    const dColor = r.cohens_d > 0 ? 'var(--positive)' : 'var(--negative)';
    return `<tr>
      <td>${c}</td>
      <td class="num">${r.n_runs}</td>
      <td class="num">${fmt(r.mean_accuracy)}</td>
      <td class="num">${fmt(r.baseline_mean)}</td>
      <td class="num" style="color:${dColor}">${r.cohens_d > 0 ? '+' : ''}${fmt(r.cohens_d, 2)}</td>
      <td class="num">${fmt(r.p_raw, 3)}</td>
      <td class="num">${fmt(r.p_holm_corrected, 3)}</td>
    </tr>`;
  }).join('');
  document.getElementById(elTable).innerHTML = `<table>
    <thead><tr><th>Condition</th><th>n</th><th>Mean</th><th>Baseline</th><th>Cohen's d</th><th>p (raw)</th><th>p (Holm)</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// === Exp 1b — three-way comparison (session_1 vs session_2 effect sizes) ===
// analyzer returns { three_way_comparison: { <cond>: { session_1_effect_size,
//   session_2_effect_size, session_1_mean_accuracy, session_2_mean_accuracy,
//   session_1_n_runs, session_2_n_runs, no_conditioning_baseline } },
//   session_1_baseline_mean, session_2_baseline_mean }.
// The story is cross-session shrinkage: where session_1 d > session_2 d, the
// affect signal is decaying when conditioning and transfer are split across
// sessions. We render two grouped bars per condition (one per session) using
// the shared condition color, with session_2 a paler variant so the eye can
// pair them and read the gap.
function renderExp1b() {
  const a = DATA.experiments.exp1b?.analysis;
  const elBar = 'exp1b-bars';
  const elTable = 'exp1b-table';
  if (!a || a.verdict === 'build_error') {
    document.getElementById(elBar).innerHTML = '<div style="padding:40px; color: var(--muted)">No exp1b data available.</div>';
    return;
  }
  const tw = a.three_way_comparison || {};
  const conds = sortConds(Object.keys(tw));
  if (!conds.length) {
    document.getElementById(elBar).innerHTML = '<div style="padding:40px; color: var(--muted)">No three_way_comparison cells in exp1b analysis.</div>';
    return;
  }
  // Build matched palettes: full color for session_1, faded for session_2.
  // We faded by mixing toward a neutral mid-gray rather than alpha so the
  // bars don't render translucent on top of grid lines.
  function fade(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
    if (!m) return '#9ca3af';
    const n = parseInt(m[1], 16);
    let r = (n >> 16) & 0xff, g = (n >> 8) & 0xff, b = n & 0xff;
    // Mix 55% toward #9ca3af (neutral gray) to keep hue but drop saturation.
    const tr = 156, tg = 163, tb = 175, t = 0.55;
    r = Math.round(r * (1 - t) + tr * t);
    g = Math.round(g * (1 - t) + tg * t);
    b = Math.round(b * (1 - t) + tb * t);
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
  }
  const s1Colors = conds.map(c => COLORS[c] || '#888');
  const s2Colors = conds.map(c => fade(COLORS[c] || '#888'));
  const s1d = conds.map(c => tw[c].session_1_effect_size);
  const s2d = conds.map(c => tw[c].session_2_effect_size);
  const traces = [
    {
      x: conds, y: s1d, type: 'bar', name: 'session_1 d',
      marker: { color: s1Colors, line: { color: s1Colors, width: 1 } },
      text: s1d.map(v => fmt(v, 2)),
      textposition: 'outside',
      hovertemplate: '%{x}<br>session_1 d: %{y:.3f}<extra></extra>',
    },
    {
      x: conds, y: s2d, type: 'bar', name: 'session_2 d',
      marker: { color: s2Colors, line: { color: s1Colors, width: 1 } },
      text: s2d.map(v => fmt(v, 2)),
      textposition: 'outside',
      hovertemplate: '%{x}<br>session_2 d: %{y:.3f}<extra></extra>',
    },
  ];
  Plotly.newPlot(elBar, traces, plotlyLayout({
    barmode: 'group',
    yaxis: { title: "Cohen's d vs no_conditioning baseline", zeroline: true },
    xaxis: { tickangle: -25 },
  }), PLOT_CFG);

  // Companion table.
  const s1Base = a.session_1_baseline_mean;
  const s2Base = a.session_2_baseline_mean;
  const rows = conds.map(c => {
    const r = tw[c];
    const s1Color = r.session_1_effect_size > 0 ? 'var(--positive)' : 'var(--negative)';
    const s2Color = r.session_2_effect_size > 0 ? 'var(--positive)' : 'var(--negative)';
    const shrink = (r.session_1_effect_size != null && r.session_2_effect_size != null)
      ? r.session_1_effect_size - r.session_2_effect_size : null;
    return `<tr>
      <td>${c}</td>
      <td class="num">${r.session_1_n_runs ?? '—'} / ${r.session_2_n_runs ?? '—'}</td>
      <td class="num">${fmt(r.session_1_mean_accuracy)} / ${fmt(r.session_2_mean_accuracy)}</td>
      <td class="num" style="color:${s1Color}">${r.session_1_effect_size > 0 ? '+' : ''}${fmt(r.session_1_effect_size, 2)}</td>
      <td class="num" style="color:${s2Color}">${r.session_2_effect_size > 0 ? '+' : ''}${fmt(r.session_2_effect_size, 2)}</td>
      <td class="num">${shrink == null ? '—' : (shrink > 0 ? '+' : '') + fmt(shrink, 2)}</td>
    </tr>`;
  }).join('');
  const baselineNote = (s1Base != null || s2Base != null)
    ? `<div class="note">Baseline accuracy: session_1 = ${fmt(s1Base)}, session_2 = ${fmt(s2Base)}. Positive d = condition outperforms no_conditioning baseline within that session.</div>`
    : '';
  document.getElementById(elTable).innerHTML = `<table>
    <thead><tr>
      <th>Condition</th>
      <th>n (s1 / s2)</th>
      <th>Mean acc (s1 / s2)</th>
      <th>session_1 d</th>
      <th>session_2 d</th>
      <th>Shrinkage (s1 − s2)</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>${baselineNote}`;
}

// === Exp 3b — n-gram ratio (semantic diversity proxy) ===
// analyzer returns { by_condition: { <cond>: {n_generations, embedding_variance, ngram_ratio} } }
// embedding_variance is null when sentence-transformers isn't installed; we plot
// ngram_ratio (distinct-2 / total bigrams) as the diversity proxy in that case.
function renderExp3b() {
  const a = DATA.experiments.exp3b?.analysis;
  const target = 'exp3b-bars';
  if (!a || a.verdict === 'build_error') {
    document.getElementById(target).innerHTML = '<div style="padding:40px; color: var(--muted)">No exp3b data available.</div>';
    return;
  }
  const bc = a.by_condition || {};
  const conds = sortConds(Object.keys(bc));
  if (!conds.length) {
    document.getElementById(target).innerHTML = '<div style="padding:40px; color: var(--muted)">No conditions in exp3b analysis.</div>';
    return;
  }
  const hasEmbedding = conds.some(c => typeof bc[c].embedding_variance === 'number');
  const metricKey = hasEmbedding ? 'embedding_variance' : 'ngram_ratio';
  const metricLabel = hasEmbedding
    ? 'Embedding variance (semantic dispersion)'
    : 'N-gram diversity ratio (distinct-2 / total bigrams)';
  const ys = conds.map(c => bc[c][metricKey]);
  const colors = conds.map(c => COLORS[c] || '#888');
  // exp3b diversity isn't bounded 0-1 in any meaningful way; "Fixed" mode
  // is treated the same as "auto" since clamping to [0, 1] doesn't help.
  const baseline = bc.no_conditioning?.[metricKey];
  const yValues = maybeDelta(ys, baseline);
  const layout = plotlyLayout({
    yaxis: { title: metricLabel },
    xaxis: { tickangle: -25 },
  });
  applyYRange(layout, { values: ys, baseline, label: 'no_conditioning' });
  const trace = makeSeriesTrace(conds, yValues, colors, {
    name: metricKey,
    hoverTemplate: '%{x}<br>' + metricKey + ': %{y:.3f}<extra></extra>',
  });
  Plotly.newPlot(target, [trace], layout, PLOT_CFG);
}

// === Exp 3c — accuracy & refusal stratified by difficulty ===
// analyzer returns { by_condition_difficulty: { "<cond>__<difficulty>": {accuracy, refusal_rate, ...} } }
// The conservative-shift hypothesis predicts negative-affect conditions show
// higher refusal AND lower accuracy on hard questions. We render two grouped
// bar charts side-by-side: accuracy-by-difficulty and refusal-by-difficulty.
function renderExp3c() {
  const a = DATA.experiments.exp3c?.analysis;
  const target = 'exp3c-bars';
  if (!a || a.verdict === 'build_error') {
    document.getElementById(target).innerHTML = '<div style="padding:40px; color: var(--muted)">No exp3c data available.</div>';
    return;
  }
  const bcd = a.by_condition_difficulty || {};
  // Decompose composite "<cond>__<diff>" keys into a (cond, diff) -> cell map.
  const byCond = {};
  const difficulties = new Set();
  for (const key of Object.keys(bcd)) {
    const [cond, diff] = key.split('__');
    if (!cond || !diff) continue;
    difficulties.add(diff);
    (byCond[cond] = byCond[cond] || {})[diff] = bcd[key];
  }
  const conds = sortConds(Object.keys(byCond));
  // Order difficulties: easy -> medium -> hard if present
  const diffOrder = ['easy', 'medium', 'hard'].filter(d => difficulties.has(d));
  if (!conds.length || !diffOrder.length) {
    document.getElementById(target).innerHTML = '<div style="padding:40px; color: var(--muted)">No exp3c stratified data available.</div>';
    return;
  }
  // Difficulty palette: warming gradient teal → amber → red so all three
  // series have good contrast on both light and dark themes. Reads
  // intuitively as "easier task = cool, harder task = hot."
  const diffShades = { easy: '#0d9488', medium: '#f59e0b', hard: '#ef4444' };
  const refusalShades = { easy: '#67e8f9', medium: '#fde68a', hard: '#fca5a5' };
  // Collect all accuracy values for the y-scale auto-zoom.
  const allAcc = [];
  const baseline = byCond.no_conditioning
    ? Object.values(byCond.no_conditioning).reduce((s, c) => s + (c?.accuracy ?? 0), 0)
      / Math.max(1, Object.keys(byCond.no_conditioning).length)
    : null;
  // In ordered-sort mode, each difficulty becomes its own line+markers
  // trace so the eye can trace each curve independently and compare them
  // at a glance. In unordered (alphabetical) mode, fall back to grouped
  // bars since the x-axis order has no meaningful trend.
  const useLines = isOrderedSort();
  const traces = [];
  for (const diff of diffOrder) {
    const accs = conds.map(c => byCond[c][diff]?.accuracy ?? null);
    accs.forEach(v => { if (v != null) allAcc.push(v); });
    const color = diffShades[diff] || '#888';
    if (useLines) {
      traces.push({
        x: conds, y: maybeDelta(accs, baseline),
        type: 'scatter', mode: 'lines+markers',
        name: 'acc / ' + diff,
        line: { color, width: 2.5 },
        marker: { size: 11, color, line: { width: 1, color } },
        hovertemplate: '%{x}<br>' + diff + ' accuracy: %{y:.3f}<extra></extra>',
        legendgroup: 'acc',
      });
    } else {
      traces.push({
        x: conds, y: maybeDelta(accs, baseline),
        type: 'bar', name: 'acc / ' + diff,
        marker: { color },
        hovertemplate: '%{x}<br>' + diff + ' accuracy: %{y:.3f}<extra></extra>',
        legendgroup: 'acc',
      });
    }
  }
  for (const diff of diffOrder) {
    const refs = conds.map(c => byCond[c][diff]?.refusal_rate ?? null);
    const color = refusalShades[diff] || '#ef4444';
    if (useLines) {
      traces.push({
        x: conds, y: refs,
        type: 'scatter', mode: 'lines+markers',
        name: 'refusal / ' + diff,
        line: { color, width: 2, dash: 'dot' },
        marker: { size: 9, color, symbol: 'square' },
        hovertemplate: '%{x}<br>' + diff + ' refusal: %{y:.3f}<extra></extra>',
        legendgroup: 'refusal',
        visible: 'legendonly',
      });
    } else {
      traces.push({
        x: conds, y: refs,
        type: 'bar', name: 'refusal / ' + diff,
        marker: { color },
        hovertemplate: '%{x}<br>' + diff + ' refusal: %{y:.3f}<extra></extra>',
        legendgroup: 'refusal',
        visible: 'legendonly',
      });
    }
  }
  const layout = plotlyLayout({
    barmode: 'group',
    yaxis: { title: 'accuracy' },
    xaxis: { tickangle: -25 },
  });
  applyYRange(layout, { values: allAcc, baseline, label: 'no_conditioning mean', fixedMin: 0, fixedMax: 1 });
  Plotly.newPlot(target, traces, layout, PLOT_CFG);
}

// === Exp 3a — per-level inverted-U + folded magnitude ===
// analyzer returns { per_level: [{level, label, n, mean, ci95_lo, ci95_hi, magnitude}, ...],
//                    folded: [{magnitude, label, n, mean, ci95_lo, ci95_hi}, ...],
//                    concavity_contrast_c, central_cell_mean,
//                    primary_signed_axis: {beta_0, beta_1, beta_2, beta_2_p_one_sided},
//                    arousal_magnitude: {beta_2, beta_2_p_one_sided, peak_arousal},
//                    within_subjects: {n_observations, beta_2, beta_2_p_one_sided} }.
function renderExp3a() {
  const block = DATA.exp3a;
  if (!block || !block.analysis || block.analysis.verdict === 'build_error') {
    document.getElementById('section-exp3a').style.display = 'none';
    return;
  }
  const a = block.analysis;

  // ------------------------------------------------------------------
  // Headline chart: dual-line on the arousal axis (matches the slide
  // deck). X-axis is folded magnitude (Neutral → Mild → Moderate →
  // Strong); two series share the leftmost neutral baseline and
  // diverge into the positive (green) and negative (red) frames.
  // ------------------------------------------------------------------
  const POS_COLOR = '#10b981';
  const NEG_COLOR = '#ef4444';
  // Build the two arms by indexing per_level rows by signed level.
  const byLevel = {};
  for (const r of a.per_level) byLevel[r.level] = r;
  const neutral = byLevel[4];
  // Each arm: [Neutral, Mild, Moderate, Strong] = [level 4, level on
  // that side at magnitude 1, magnitude 2, magnitude 3]. Wilson 95%
  // binomial CIs come straight from the per_level rows (n_per_level
  // = 122 cells), pulled as asymmetric error bars on each marker.
  const xs_arousal = ['Neutral', 'Mild', 'Moderate', 'Strong'];
  const posLevels = neutral && byLevel[3] && byLevel[2] && byLevel[1]
    ? [neutral, byLevel[3], byLevel[2], byLevel[1]] : null;
  const negLevels = neutral && byLevel[5] && byLevel[6] && byLevel[7]
    ? [neutral, byLevel[5], byLevel[6], byLevel[7]] : null;
  if (posLevels && negLevels) {
    const posMeans = posLevels.map(r => r.mean);
    const negMeans = negLevels.map(r => r.mean);
    const posErrLo = posLevels.map(r => r.mean - r.ci95_lo);
    const posErrHi = posLevels.map(r => r.ci95_hi - r.mean);
    const negErrLo = negLevels.map(r => r.mean - r.ci95_lo);
    const negErrHi = negLevels.map(r => r.ci95_hi - r.mean);
    const posTrace = {
      x: xs_arousal, y: posMeans,
      type: 'scatter', mode: 'lines+markers',
      name: 'Positive valence',
      line: { color: POS_COLOR, width: 2.5 },
      marker: { size: 14, color: POS_COLOR, line: { width: 1, color: '#e6edf6' } },
      error_y: { type: 'data', symmetric: false, array: posErrHi, arrayminus: posErrLo, thickness: 1.5, color: POS_COLOR },
      hovertemplate: '%{x} (positive)<br>mean: %{y:.3f}<extra></extra>',
    };
    const negTrace = {
      x: xs_arousal, y: negMeans,
      type: 'scatter', mode: 'lines+markers',
      name: 'Negative valence',
      line: { color: NEG_COLOR, width: 2.5 },
      marker: { size: 14, color: NEG_COLOR, line: { width: 1, color: '#e6edf6' } },
      error_y: { type: 'data', symmetric: false, array: negErrHi, arrayminus: negErrLo, thickness: 1.5, color: NEG_COLOR },
      hovertemplate: '%{x} (negative)<br>mean: %{y:.3f}<extra></extra>',
    };
    const arousalLayout = plotlyLayout({
      yaxis: { title: 'Mean accuracy', range: [0.45, 0.80] },
      xaxis: { title: 'Arousal magnitude (|level − 4|)' },
      legend: { bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.18 },
      showlegend: true,
    });
    Plotly.newPlot('exp3a-arousal', [posTrace, negTrace], arousalLayout, PLOT_CFG);
  } else {
    document.getElementById('exp3a-arousal').style.display = 'none';
  }

  // Color levels by their valence: positive (1, 2, 3) emerald-shaded,
  // neutral (4) blue, negative (5, 6, 7) red-shaded. Within each side
  // strong is darkest, mild is lightest, so the eye reads magnitude.
  const LEVEL_COLORS = {
    1: '#065f46', 2: '#10b981', 3: '#6ee7b7',
    4: '#3b82f6',
    5: '#fca5a5', 6: '#ef4444', 7: '#7f1d1d',
  };
  const xs = a.per_level.map(r => r.label);
  const means = a.per_level.map(r => r.mean);
  const errLo = a.per_level.map(r => r.mean - r.ci95_lo);
  const errHi = a.per_level.map(r => r.ci95_hi - r.mean);
  const colors = a.per_level.map(r => LEVEL_COLORS[r.level] || '#888');
  const trace = {
    x: xs, y: means,
    type: 'scatter', mode: 'lines+markers',
    name: 'Mean accuracy',
    line: { color: '#94a3b8', width: 2 },
    marker: { size: 14, color: colors, line: { width: 1, color: '#475569' } },
    error_y: { type: 'data', symmetric: false, array: errHi, arrayminus: errLo, thickness: 1.5 },
    text: means.map(v => fmt(v, 3)),
    textposition: 'top center',
    hovertemplate: '%{x}<br>mean: %{y:.3f}<extra></extra>',
  };
  Plotly.newPlot('exp3a-perlevel', [trace], plotlyLayout({
    yaxis: { title: 'Mean accuracy', range: [0.5, 0.75] },
    xaxis: { title: 'Intensity level (signed)' },
  }), PLOT_CFG);

  // Folded magnitude trace. Shares the same axis convention but maps
  // magnitude (0..3) to label rather than signed level.
  const foldedTarget = 'exp3a-folded';
  const fxs = a.folded.map(r => r.label);
  const fmeans = a.folded.map(r => r.mean);
  const fErrLo = a.folded.map(r => r.mean - r.ci95_lo);
  const fErrHi = a.folded.map(r => r.ci95_hi - r.mean);
  // Folded magnitude: use a neutral-to-warm gradient. magnitude 0 (neutral)
  // is blue (matches level 4 color); 1, 2, 3 are amber → orange → red.
  const FOLDED_COLORS = {0: '#3b82f6', 1: '#fbbf24', 2: '#f97316', 3: '#dc2626'};
  const fcolors = a.folded.map(r => FOLDED_COLORS[r.magnitude] || '#888');
  const ftrace = {
    x: fxs, y: fmeans,
    type: 'scatter', mode: 'lines+markers',
    name: 'Folded mean',
    line: { color: '#94a3b8', width: 2 },
    marker: { size: 14, color: fcolors, line: { width: 1, color: '#475569' } },
    error_y: { type: 'data', symmetric: false, array: fErrHi, arrayminus: fErrLo, thickness: 1.5 },
    text: fmeans.map(v => fmt(v, 3)),
    textposition: 'top center',
    hovertemplate: '%{x}<br>mean: %{y:.3f}<extra></extra>',
  };
  Plotly.newPlot(foldedTarget, [ftrace], plotlyLayout({
    yaxis: { title: 'Mean accuracy', range: [0.5, 0.75] },
    xaxis: { title: 'Folded magnitude (|level − 4|)' },
  }), PLOT_CFG);

  // KPI cards summarising the H3a fits. Pull from sensitivity.json
  // when available; show '—' if a particular fit is missing.
  const psa = a.primary_signed_axis || {};
  const am = a.arousal_magnitude || {};
  const ws = a.within_subjects || {};
  const c = a.concavity_contrast_c;
  const central = a.central_cell_mean;
  document.getElementById('exp3a-kpis').innerHTML = `
    <div class="kpi"><div class="label">n observations</div><div class="value">${a.n_observations ?? '—'}</div></div>
    <div class="kpi"><div class="label">Concavity c</div><div class="value">${fmt(c, 4)}</div></div>
    <div class="kpi"><div class="label">Central cell (Neutral)</div><div class="value">${fmt(central, 3)}</div></div>
    <div class="kpi"><div class="label">β₂ (signed quadratic)</div><div class="value">${fmt(psa.beta_2, 4)}</div></div>
    <div class="kpi"><div class="label">β₂ p (one-sided)</div><div class="value">${fmt(psa.beta_2_p_one_sided, 3)}</div></div>
    <div class="kpi"><div class="label">Within-subjects β₂</div><div class="value">${fmt(ws.beta_2, 4)}</div></div>
    <div class="kpi"><div class="label">WS β₂ p (one-sided)</div><div class="value">${fmt(ws.beta_2_p_one_sided, 3)}</div></div>
    <div class="kpi"><div class="label">Peak arousal</div><div class="value">${fmt(am.peak_arousal, 2)}</div></div>
  `;
}

// === Pre-experiment probes ===
// DATA.probes is { api_jitter, format_perturbation, mini_calibration,
//                  concavity_on_calibrated } where each block is optional.
function renderProbes() {
  const probes = DATA.probes || {};
  if (!probes || Object.keys(probes).length === 0) {
    document.getElementById('section-probes').style.display = 'none';
    return;
  }
  // Summary KPI cards across the four probes.
  const cards = [];
  const aj = probes.api_jitter;
  if (aj) {
    cards.push(`<div class="kpi"><div class="label">API jitter (temp 0)</div><div class="value">${fmtPct(aj.overall_identical_rate)}</div><div class="sub" style="color:var(--muted);font-size:11px">modal-response rate, ${aj.n_items} items × ${aj.n_reps_per_item} reps</div></div>`);
  }
  const fp = probes.format_perturbation;
  if (fp) {
    // Aggregate extraction-failure rate across levels.
    const rows = fp.per_level || [];
    const totalCells = fp.n_total_cells || rows.reduce((s, r) => s + (r.n_cells || 0), 0);
    const totalFailures = rows.reduce((s, r) => s + (r.extraction_failure_rate * (totalCells / Math.max(1, rows.length))), 0);
    const overallFailRate = rows.length > 0
      ? rows.reduce((s, r) => s + (r.extraction_failure_rate || 0), 0) / rows.length
      : null;
    cards.push(`<div class="kpi"><div class="label">Format-perturbation rate</div><div class="value">${fmtPct(overallFailRate)}</div><div class="sub" style="color:var(--muted);font-size:11px">extraction failures across ${totalCells} cells</div></div>`);
  }
  const mc = probes.mini_calibration;
  if (mc) {
    const yieldPct = mc.n_candidates ? (mc.n_calibrated / mc.n_candidates) : null;
    cards.push(`<div class="kpi"><div class="label">Calibration yield</div><div class="value">${mc.n_calibrated ?? '—'} / ${mc.n_candidates ?? '—'}</div><div class="sub" style="color:var(--muted);font-size:11px">band [${mc.target_lo}, ${mc.target_hi}], ${fmtPct(yieldPct)}</div></div>`);
  }
  const co = probes.concavity_on_calibrated;
  if (co) {
    cards.push(`<div class="kpi"><div class="label">Concavity c (calibrated)</div><div class="value">${fmt(co.concavity_contrast_c, 4)}</div><div class="sub" style="color:var(--muted);font-size:11px">${co.n_items} items × ${co.n_reps_per_cell} reps, per-item sd ${fmt(co.per_item_c_stdev, 3)}</div></div>`);
  }
  document.getElementById('probes-cards').innerHTML = cards.join('');

  // Calibration p̂ histogram. 11 bins from 0.00 to 1.00 in 0.10 steps.
  if (mc && mc.p_hat_histogram) {
    const bins = mc.p_hat_histogram;
    const binCenters = bins.map((_, i) => (i / 10).toFixed(2));
    const lo = mc.target_lo;
    const hi = mc.target_hi;
    // Bars in the strict band get accent color; outside is muted.
    const barColors = bins.map((_, i) => {
      const center = i / 10;
      return (lo != null && hi != null && center >= lo && center <= hi)
        ? '#f97316' : '#475569';
    });
    Plotly.newPlot('probes-calib-hist', [{
      x: binCenters, y: bins, type: 'bar',
      marker: { color: barColors },
      text: bins.map(v => v > 0 ? String(v) : ''),
      textposition: 'outside',
      hovertemplate: 'p̂ ≈ %{x}<br>%{y} items<extra></extra>',
    }], plotlyLayout({
      title: { text: `Calibration p̂ distribution (n_candidates = ${mc.n_candidates})`, font: { size: 13 } },
      xaxis: { title: 'p̂ (no-stimulus accuracy)' },
      yaxis: { title: 'Item count' },
    }), PLOT_CFG);
  } else {
    document.getElementById('probes-calib-hist').style.display = 'none';
  }

  // Concavity-on-calibrated folded magnitude trace.
  if (co && co.folded_means) {
    const folded = co.folded_means;
    const mags = Object.keys(folded).sort();
    const fxs = mags.map(m => MAG_LABELS[m] || m);
    const fmeans = mags.map(m => folded[m].mean_accuracy);
    Plotly.newPlot('probes-concavity-folded', [{
      x: fxs, y: fmeans,
      type: 'scatter', mode: 'lines+markers',
      name: 'Folded mean (calibrated)',
      line: { color: '#f97316', width: 2 },
      marker: { size: 14, color: '#f97316', line: { width: 1, color: '#9a3412' } },
      text: fmeans.map(v => fmt(v, 3)),
      textposition: 'top center',
      hovertemplate: '%{x}<br>mean: %{y:.3f}<extra></extra>',
    }], plotlyLayout({
      title: { text: `Concavity probe — folded magnitude (n_items = ${co.n_items}, n_reps = ${co.n_reps_per_cell})`, font: { size: 13 } },
      yaxis: { title: 'Mean accuracy' },
      xaxis: { title: 'Folded magnitude' },
    }), PLOT_CFG);
  } else {
    document.getElementById('probes-concavity-folded').style.display = 'none';
  }
}

// Magnitude labels for the calibrated-concavity folded chart.
const MAG_LABELS = {'0': 'Neutral', '1': 'Mild', '2': 'Moderate', '3': 'Strong'};

// === Cost + timing ===
function renderCost() {
  const target = 'cost-bars';
  const exps = Object.keys(DATA.experiments);
  const wall = exps.map(e => {
    const t = DATA.experiments[e].manifest?.timing_per_condition || {};
    return Object.values(t).reduce((s, x) => s + (x.total_seconds || 0), 0) / 60;
  });
  Plotly.newPlot(target, [{
    x: exps, y: wall, type: 'bar', marker: { color: '#6366f1' },
    text: wall.map(v => fmt(v, 1) + ' min'),
    textposition: 'outside',
    name: 'wall-clock (min)',
    hovertemplate: '%{x}<br>%{y:.1f} min<extra></extra>',
  }], plotlyLayout({
    yaxis: { title: 'Wall-clock (min)' },
  }), PLOT_CFG);
}

// === Render everything ===
// Each render call is wrapped in safeRender so a failure in one chart leaves
// an inline error block in that chart's container and the rest of the page
// still renders. Without this, a single TypeError aborts the whole handler.
function renderAll() {
  safeRender('meta',     renderMeta,     'meta-grid');
  safeRender('summary',  renderSummary,  'summary-row');
  safeRender('verdicts', renderVerdicts, 'verdicts');
  safeRender('exp3a',    renderExp3a,    'exp3a-arousal');
  safeRender('probes',   renderProbes,   'probes-cards');
  safeRender('exp2',     renderExp2,     'exp2-curves');
  safeRender('exp1a',    () => renderEffectSizes('exp1a-bars', 'exp1a-table', DATA.experiments.exp1a?.analysis), 'exp1a-bars');
  safeRender('exp1b',    renderExp1b,    'exp1b-bars');
  safeRender('exp3b',    renderExp3b,    'exp3b-bars');
  safeRender('exp3c',    renderExp3c,    'exp3c-bars');
  safeRender('cost',     renderCost,     'cost-bars');
}

// Per-sort hint text shown next to the selector. Brief and load-bearing —
// the user should know what each ordering is FOR, not just its alphabetical
// vs cognitive-psych framing.
const SORT_HINTS = {
  alphabetical: 'A neutral default with no implied story.',
  arousal: 'Yerkes-Dodson predicts an inverted-U (peak at moderate arousal). Hard tasks tilt the peak left.',
  valence: 'Tests directional asymmetry (paper §3.4). H4 predicts the negative arm produces a larger off-control area than the positive arm.',
  intensity: 'Combined arousal + |valence|. Goes from no intervention (left) to most intense (right) regardless of direction. Cleanest single-axis Yerkes-Dodson view.',
};

window.addEventListener('DOMContentLoaded', () => {
  renderAll();
  // Re-render on theme change so axis/grid colors track the theme.
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
  // Re-render on sort change. Hint text updates inline so the user can
  // see at a glance what the chosen ordering is for.
  const sel = document.getElementById('sort-select');
  const hint = document.getElementById('sort-hint');
  if (sel) {
    sel.value = currentSort;
    sel.addEventListener('change', () => {
      currentSort = sel.value;
      if (hint) hint.textContent = SORT_HINTS[currentSort] || '';
      renderAll();
    });
  }
  // Y-scale selector. No hint text; the option labels are self-describing.
  const yscaleSel = document.getElementById('yscale-select');
  if (yscaleSel) {
    yscaleSel.value = currentYScale;
    yscaleSel.addEventListener('change', () => {
      currentYScale = yscaleSel.value;
      renderAll();
    });
  }
});
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a self-contained interactive HTML dashboard from a pilot's results.")
    ap.add_argument("--pilot-dir", required=True, type=Path,
                    help="e.g. results/pilots/2026-04-27_gpt-5.4-nano")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write the dashboard HTML (will be overwritten)")
    ap.add_argument("--exp3a-dir", type=Path, default=None,
                    help="Path to an H3a results dir (within-subjects rescored)")
    ap.add_argument("--probes-dir", type=Path, default=None,
                    help="Path to a directory of probe JSONs (h3b_*)")
    args = ap.parse_args()

    pilot_dir: Path = args.pilot_dir
    if not pilot_dir.exists():
        ap.error(f"pilot dir not found: {pilot_dir}")

    print(f"[build_dashboard] loading pilot from {pilot_dir}")
    payload = collect(pilot_dir)

    if args.exp3a_dir:
        if not args.exp3a_dir.exists():
            print(f"[build_dashboard] WARN: exp3a-dir not found: {args.exp3a_dir}")
        else:
            print(f"[build_dashboard] loading exp3a (H3a) from {args.exp3a_dir}")
            exp3a_block = collect_h3a(args.exp3a_dir)
            payload["exp3a"] = exp3a_block
            # collect_h3a returns either a populated block or
            # {"verdict": "build_error", "error": "..."} when the
            # configured dir lacks a data/ subdirectory. Degrade
            # gracefully in the latter case: keep the block on the
            # payload (the JS renderer hides the section on
            # build_error) but skip the verdicts-table registration so
            # we don't index a missing "analysis" key.
            if exp3a_block.get("verdict") == "build_error":
                err = exp3a_block.get("error", "unknown")
                print(f"[build_dashboard] WARN: exp3a load failed ({err}); section skipped")
            else:
                analysis = exp3a_block.get("analysis", {})
                ws_n = (analysis.get("within_subjects") or {}).get("n_observations")
                print(f"[build_dashboard]   exp3a n={exp3a_block.get('n_runs_total')}, within-subjects n={ws_n}")
                # Surface in the verdicts table and the summary cards
                # by also registering as an "experiment" entry. The
                # dashboard's existing verdict / summary code reads
                # DATA.experiments.
                payload["experiments"]["exp3a"] = {
                    "manifest": exp3a_block.get("manifest", {}),
                    "analysis": {"verdict": analysis.get("verdict", "complete")},
                    "n_runs_total": exp3a_block.get("n_runs_total", 0),
                }

    if args.probes_dir:
        if not args.probes_dir.exists():
            print(f"[build_dashboard] WARN: probes-dir not found: {args.probes_dir}")
        else:
            print(f"[build_dashboard] loading probes from {args.probes_dir}")
            payload["probes"] = collect_probes(args.probes_dir)
            print(f"[build_dashboard]   probes loaded: {sorted(payload['probes'].keys())}")

    n_exp = len(payload["experiments"])
    n_runs = sum(d["n_runs_total"] for d in payload["experiments"].values())
    print(f"[build_dashboard] collected {n_exp} experiments, {n_runs} cells total")

    pilot_label = pilot_dir.name
    safe_payload = json_safe(payload)
    # Use .replace rather than .format so we don't have to double-escape every
    # `{` and `}` in the JS / CSS body of the template. Placeholders are
    # explicit __TOKEN__ markers that are vanishingly unlikely to collide
    # with anything in the page body.
    html = (HTML_TEMPLATE
            .replace("__PILOT_LABEL__", pilot_label)
            .replace("__PILOT_PATH__", str(pilot_dir))
            .replace("__DATA_JSON__", json.dumps(safe_payload))
            .replace("__COLORS_JSON__", json.dumps(CONDITION_COLORS))
            .replace("__AXES_JSON__", json.dumps(CONDITION_AXES)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"[build_dashboard] wrote {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
