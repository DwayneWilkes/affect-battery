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
    "strong_positive":     "#10b981",  # emerald
    "mild_negative":       "#f59e0b",  # amber
    "strong_negative":     "#ef4444",  # red
    "neutral":             "#6366f1",  # indigo
    "no_conditioning":     "#64748b",  # slate (baseline gets the muted color)
    "accurate_negative":   "#a855f7",  # violet
    "self_check_neutral":  "#06b6d4",  # cyan
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
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px 24px;
  margin-top: 14px;
  font-size: 12px;
}
.meta-grid div {
  color: var(--muted);
}
.meta-grid div b {
  color: var(--text);
  font-weight: 500;
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
  font-size: 20px;
  font-weight: 600;
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
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
  <div class="meta-grid" id="meta-grid"></div>
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

// === Header / meta ===
function renderMeta() {
  const grid = document.getElementById('meta-grid');
  // Use any experiment's manifest as the source of model/seed/etc — they're all the same.
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
  grid.innerHTML = items.map(([k, v]) => `<div><b>${v}</b><br>${k}</div>`).join('');
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
  // Control curve (always plot first so it's drawn underneath)
  if (a.control_curve) {
    traces.push({
      x: ns, y: a.control_curve, name: 'neutral (control)',
      mode: 'lines+markers',
      line: { color: COLORS.neutral, width: 2, dash: 'dot' },
      marker: { size: 8, color: COLORS.neutral },
    });
  }
  // Per-condition curves
  for (const [cond, cell] of Object.entries(a.by_condition || {})) {
    traces.push({
      x: cell.n_values, y: cell.turn_accuracies_mean, name: cond,
      mode: 'lines+markers',
      line: { color: COLORS[cond] || '#888', width: 2 },
      marker: { size: 8, color: COLORS[cond] || '#888' },
    });
  }
  // Baseline horizontal line
  if (a.baseline != null) {
    traces.push({
      x: ns, y: ns.map(() => a.baseline),
      name: `no_conditioning baseline (${fmt(a.baseline)})`,
      mode: 'lines',
      line: { color: COLORS.no_conditioning, width: 1, dash: 'dash' },
      hoverinfo: 'name',
    });
  }
  Plotly.newPlot('exp2-curves', traces, plotlyLayout({
    xaxis: { title: 'N (neutral conditioning turns)', tickmode: 'array', tickvals: ns },
    yaxis: { title: 'Mean turn accuracy', range: [0, 1] },
  }), PLOT_CFG);

  // KPIs
  const kpis = document.getElementById('exp2-kpis');
  kpis.innerHTML = `
    <div class="kpi"><div class="label">Verdict</div><div class="value">${a.verdict}</div></div>
    <div class="kpi"><div class="label">Baseline acc</div><div class="value">${fmt(a.baseline)}</div></div>
    <div class="kpi"><div class="label">Asymmetry ratio</div><div class="value">${fmt(a.asymmetry_ratio, 3)}</div></div>
    <div class="kpi"><div class="label">N values swept</div><div class="value">${(a.n_values || []).join(', ')}</div></div>
  `;
}

// === Exp 1a / 1b — bar chart + table ===
function renderEffectSizes(elBar, elTable, analysis) {
  if (!analysis || analysis.verdict === 'build_error') {
    if (elBar) document.getElementById(elBar).innerHTML = '<div style="padding:40px; color: var(--muted)">No data available.</div>';
    return;
  }
  const cells = analysis.per_condition_vs_baseline || {};
  const conds = Object.keys(cells).sort();
  const means = conds.map(c => cells[c].mean_accuracy);
  const baseline = analysis.per_condition_vs_baseline[conds[0]]?.baseline_mean;
  const colors = conds.map(c => COLORS[c] || '#888');
  const traces = [{
    x: conds, y: means, type: 'bar', marker: { color: colors },
    text: means.map(v => fmtPct(v)),
    textposition: 'outside',
    name: 'mean acc',
    hovertemplate: '%{x}<br>mean acc: %{y:.3f}<extra></extra>',
  }];
  if (baseline != null) {
    traces.push({
      x: conds, y: conds.map(() => baseline), type: 'scatter', mode: 'lines',
      line: { color: COLORS.no_conditioning, dash: 'dash', width: 2 },
      name: `baseline (${fmt(baseline)})`,
      hoverinfo: 'name',
    });
  }
  Plotly.newPlot(elBar, traces, plotlyLayout({
    yaxis: { title: 'Mean transfer accuracy', range: [0, 1] },
    xaxis: { tickangle: -25 },
  }), PLOT_CFG);

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
  const conds = Object.keys(bc).sort();
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
  Plotly.newPlot(target, [{
    x: conds, y: ys, type: 'bar', marker: { color: colors },
    text: ys.map(v => fmt(v)),
    textposition: 'outside',
    hovertemplate: '%{x}<br>' + metricKey + ': %{y:.3f}<extra></extra>',
  }], plotlyLayout({
    yaxis: { title: metricLabel },
    xaxis: { tickangle: -25 },
  }), PLOT_CFG);
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
  const conds = Object.keys(byCond).sort();
  // Order difficulties: easy -> medium -> hard if present
  const diffOrder = ['easy', 'medium', 'hard'].filter(d => difficulties.has(d));
  if (!conds.length || !diffOrder.length) {
    document.getElementById(target).innerHTML = '<div style="padding:40px; color: var(--muted)">No exp3c stratified data available.</div>';
    return;
  }
  // Difficulty palette: easy (light) -> hard (saturated). Refusal-on-hard is
  // the headline of the conservative-shift hypothesis.
  const diffShades = { easy: '#94a3b8', medium: '#64748b', hard: '#0f172a' };
  const refusalShades = { easy: '#fca5a5', medium: '#f87171', hard: '#dc2626' };
  const traces = [];
  for (const diff of diffOrder) {
    traces.push({
      x: conds, y: conds.map(c => byCond[c][diff]?.accuracy ?? null),
      type: 'bar', name: 'acc / ' + diff,
      marker: { color: diffShades[diff] || '#888' },
      hovertemplate: '%{x}<br>' + diff + ' accuracy: %{y:.3f}<extra></extra>',
      legendgroup: 'acc',
    });
  }
  for (const diff of diffOrder) {
    traces.push({
      x: conds, y: conds.map(c => byCond[c][diff]?.refusal_rate ?? null),
      type: 'bar', name: 'refusal / ' + diff,
      marker: { color: refusalShades[diff] || '#ef4444' },
      hovertemplate: '%{x}<br>' + diff + ' refusal: %{y:.3f}<extra></extra>',
      legendgroup: 'refusal',
      visible: 'legendonly',  // hidden by default; click legend to toggle
    });
  }
  Plotly.newPlot(target, traces, plotlyLayout({
    barmode: 'group',
    yaxis: { title: 'rate', range: [0, 1] },
    xaxis: { tickangle: -25 },
  }), PLOT_CFG);
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
window.addEventListener('DOMContentLoaded', () => {
  renderMeta();
  renderVerdicts();
  renderExp2();
  renderEffectSizes('exp1a-bars', 'exp1a-table', DATA.experiments.exp1a?.analysis);
  renderEffectSizes('exp1b-bars', 'exp1b-table', DATA.experiments.exp1b?.analysis);
  renderExp3b();
  renderExp3c();
  renderCost();
  // Re-render on theme change so colors stay readable.
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    renderVerdicts(); renderExp2();
    renderEffectSizes('exp1a-bars', 'exp1a-table', DATA.experiments.exp1a?.analysis);
    renderEffectSizes('exp1b-bars', 'exp1b-table', DATA.experiments.exp1b?.analysis);
    renderExp3b(); renderExp3c(); renderCost();
  });
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
            .replace("__COLORS_JSON__", json.dumps(CONDITION_COLORS)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"[build_dashboard] wrote {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
