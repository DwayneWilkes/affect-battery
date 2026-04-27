"""Build a self-contained interactive HTML dashboard from a pilot's results.

Loads every experiment's raw cell JSONs, runs the appropriate analyzer to
get the structured analysis dict, then emits a single HTML file with the
data embedded inline (no external data fetches; works via file://).

Usage:
    uv run python scripts/build_dashboard.py \
        --pilot-dir results/pilots/2026-04-27_gpt-5.4-nano \
        --output     results/pilots/2026-04-27_gpt-5.4-nano/dashboard.html
"""
from __future__ import annotations

import argparse
import json
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
    <h2>Exp 2 — Recovery curves over N</h2>
    <div class="subtitle">Per-condition mean accuracy as a function of neutral-conditioning turn count (N). The control (NEUTRAL) curve is what each non-baseline curve is compared against. Hover any point for the per-cell mean.</div>
    <div id="exp2-curves" class="chart chart-tall"></div>
    <div id="exp2-kpis" class="kpi-row"></div>
    <div class="note">Asymmetry ratio = |strong_negative AUC| / |strong_positive AUC|. Values &gt;1 indicate the negative arm has a larger off-control area than the positive arm.</div>
  </section>

  <section>
    <h2>Exp 1a — Within-session transfer accuracy</h2>
    <div class="subtitle">Mean transfer-question accuracy per condition vs the no_conditioning baseline. Cohen's d shown as the effect size; p-values are Holm-corrected within the family.</div>
    <div id="exp1a-bars" class="chart"></div>
    <div id="exp1a-table"></div>
  </section>

  <section>
    <h2>Exp 1b — Cross-session transfer</h2>
    <div class="subtitle">Same matrix as Exp 1a but with the affect-conditioning phase applied in a separate session prior to transfer. Effect-size shrinkage from 1a to 1b is the within- vs between-session contrast that H1 hangs on.</div>
    <div id="exp1b-bars" class="chart"></div>
    <div id="exp1b-table"></div>
  </section>

  <section>
    <h2>Exp 3b — Cognitive scope (semantic diversity)</h2>
    <div class="subtitle">Mean pairwise semantic distance across n_generations completions per (condition, prompt). Higher = more semantically diverse generations.</div>
    <div id="exp3b-bars" class="chart"></div>
  </section>

  <section>
    <h2>Exp 3c — Conservative shift (accuracy by difficulty)</h2>
    <div class="subtitle">Accuracy stratified by question difficulty (easy / medium / hard) per condition. Refusal-rate traces are hidden by default; click them in the legend to toggle. Conservative-shift hypothesis predicts negative-affect conditions show higher refusal AND lower accuracy on hard questions.</div>
    <div id="exp3c-bars" class="chart chart-tall"></div>
  </section>

  <section>
    <h2>Run cost &amp; timing</h2>
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
    args = ap.parse_args()

    pilot_dir: Path = args.pilot_dir
    if not pilot_dir.exists():
        ap.error(f"pilot dir not found: {pilot_dir}")

    print(f"[build_dashboard] loading pilot from {pilot_dir}")
    payload = collect(pilot_dir)
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
