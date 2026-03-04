#!/usr/bin/env python3
"""
04_plot.py — Generate all figures for the KV cache compression analysis.

Usage:
    python3 04_plot.py

Output (in results/figures/):
    fig1_accuracy_by_budget.png   — overall accuracy vs budget for each method
    fig2_category_heatmap.png     — method × category accuracy heatmap (at 20% budget)
    fig3_speedup.png              — wall-clock speedup vs cache budget
    fig4_accuracy_vs_speedup.png  — Pareto frontier (accuracy vs speedup)
    fig5_dataset_radar.png        — radar chart across task categories
"""

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SCORES_DIR  = os.path.join(RESULTS_DIR, "scores")
TIMING_DIR  = os.path.join(RESULTS_DIR, "timing")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
summary_path = os.path.join(SCORES_DIR, "summary.json")
latency_path = os.path.join(TIMING_DIR, "latency_report.json")

if not os.path.exists(summary_path):
    print(f"ERROR: {summary_path} not found. Run 02_score.sh first.")
    sys.exit(1)

with open(summary_path) as f:
    summary = json.load(f)

latency = None
if os.path.exists(latency_path):
    with open(latency_path) as f:
        latency = json.load(f)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})

METHOD_COLORS = {
    "FullKV":      "#3A3A3A",
    "StreamingLLM":"#E15759",
    "H2O":         "#F28E2B",
    "SnapKV":      "#4E79A7",
    "PyramidKV":   "#59A14F",
}
METHOD_MARKERS = {
    "FullKV":      "D",
    "StreamingLLM":"s",
    "H2O":         "^",
    "SnapKV":      "o",
    "PyramidKV":   "P",
}

BUDGET_ORDER   = ["10pct", "20pct", "50pct", "full"]
BUDGET_LABELS  = {"10pct":"10%", "20pct":"20%", "50pct":"50%", "full":"Full"}
METHODS_COMP   = ["StreamingLLM", "H2O", "SnapKV", "PyramidKV"]  # excluding FullKV for plots
CATEGORIES     = ["Single-Doc QA","Multi-Doc QA","Summarization","Few-Shot","Synthetic","Code"]

def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

# ════════════════════════════════════════════════════════════════════════════════
# Fig 1 — Overall accuracy vs cache budget (line plot)
# ════════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for method in METHODS_COMP + ["FullKV"]:
    if method not in summary:
        continue
    xs, ys = [], []
    for b in BUDGET_ORDER:
        if b == "full" and method != "FullKV":
            continue
        if b != "full" and method == "FullKV":
            continue
        val = safe_get(summary, method, b, "overall")
        if val is None:
            continue
        xs.append(BUDGET_LABELS[b])
        ys.append(val)

    if method == "FullKV":
        # Draw as dashed horizontal reference line
        if ys:
            full_acc = ys[0]
            ax.axhline(full_acc, color=METHOD_COLORS[method], lw=1.5,
                       ls="--", label=f"FullKV ({full_acc:.1f})", alpha=0.8)
    else:
        ax.plot(xs, ys,
                marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                lw=2, ms=8, label=method)

ax.set_xlabel("KV Cache Budget")
ax.set_ylabel("LongBench Score (avg)")
ax.set_title("Fig 1 — Overall LongBench Accuracy vs Cache Budget")
ax.legend(framealpha=0.9, loc="lower right")
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="y", alpha=0.3, ls="--")
ax.grid(axis="y", which="minor", alpha=0.1, ls=":")
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig1_accuracy_by_budget.png"))
plt.close(fig)
print("  Saved: fig1_accuracy_by_budget.png")

# ════════════════════════════════════════════════════════════════════════════════
# Fig 2 — Category-level heatmap at 20% budget
# ════════════════════════════════════════════════════════════════════════════════
HEATMAP_BUDGET = "20pct"

heat_data = np.zeros((len(METHODS_COMP), len(CATEGORIES)))
for i, m in enumerate(METHODS_COMP):
    for j, c in enumerate(CATEGORIES):
        heat_data[i, j] = safe_get(summary, m, HEATMAP_BUDGET, "categories", c, default=0.0)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(CATEGORIES)))
ax.set_xticklabels(CATEGORIES, rotation=20, ha="right")
ax.set_yticks(range(len(METHODS_COMP)))
ax.set_yticklabels(METHODS_COMP)
ax.set_title(f"Fig 2 — Category Accuracy Heatmap ({BUDGET_LABELS.get(HEATMAP_BUDGET,'?')} Budget)")
for i in range(len(METHODS_COMP)):
    for j in range(len(CATEGORIES)):
        val = heat_data[i, j]
        text_color = "white" if val < 35 or val > 75 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold")
plt.colorbar(im, ax=ax, label="Score (%)", fraction=0.03, pad=0.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig2_category_heatmap.png"))
plt.close(fig)
print("  Saved: fig2_category_heatmap.png")

# ════════════════════════════════════════════════════════════════════════════════
# Fig 3 — Speedup bar chart (if latency data available)
# ════════════════════════════════════════════════════════════════════════════════
if latency:
    rows = latency.get("rows", [])
    # Group by method × budget
    speedup_by = {}  # method → {budget: speedup}
    for r in rows:
        m = r["method"]
        b = r["budget"]
        s = r.get("speedup_total", 1.0)
        speedup_by.setdefault(m, {})[b] = s

    budgets_plot = [b for b in BUDGET_ORDER if b != "full"]
    x = np.arange(len(budgets_plot))
    n = len(METHODS_COMP)
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, method in enumerate(METHODS_COMP):
        offsets = (idx - n/2 + 0.5) * width
        vals = [speedup_by.get(method, {}).get(b, 1.0) for b in budgets_plot]
        bars = ax.bar(x + offsets, vals, width,
                      label=method, color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}×", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="FullKV baseline")
    ax.set_xlabel("KV Cache Budget")
    ax.set_ylabel("Speedup (×) over FullKV")
    ax.set_title("Fig 3 — Inference Speedup vs Cache Budget")
    ax.set_xticks(x)
    ax.set_xticklabels([BUDGET_LABELS[b] for b in budgets_plot])
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig3_speedup.png"))
    plt.close(fig)
    print("  Saved: fig3_speedup.png")

    # ── Fig 4 — Accuracy–Speedup Pareto scatter ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in METHODS_COMP:
        xs, ys, ls_vals = [], [], []
        for b in BUDGET_ORDER:
            if b == "full":
                continue
            acc  = safe_get(summary, method, b, "overall")
            spd  = speedup_by.get(method, {}).get(b, None)
            if acc is None or spd is None:
                continue
            xs.append(spd)
            ys.append(acc)
            ls_vals.append(b)

        if xs:
            ax.scatter(xs, ys, s=100, color=METHOD_COLORS[method],
                       marker=METHOD_MARKERS[method], zorder=5, label=method)
            ax.plot(xs, ys, lw=1.2, color=METHOD_COLORS[method], alpha=0.5)
            for x_val, y_val, lbl in zip(xs, ys, ls_vals):
                ax.annotate(BUDGET_LABELS[lbl], (x_val, y_val),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)

    # FullKV reference
    full_acc = safe_get(summary, "FullKV", "full", "overall")
    if full_acc:
        ax.axhline(full_acc, ls="--", color=METHOD_COLORS["FullKV"],
                   lw=1.5, label=f"FullKV ({full_acc:.1f})", alpha=0.8)

    ax.set_xlabel("Speedup (×) over FullKV")
    ax.set_ylabel("LongBench Score (avg)")
    ax.set_title("Fig 4 — Accuracy–Efficiency Trade-off")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig4_accuracy_vs_speedup.png"))
    plt.close(fig)
    print("  Saved: fig4_accuracy_vs_speedup.png")

# ════════════════════════════════════════════════════════════════════════════════
# Fig 5 — Radar chart (category scores at 20% budget)
# ════════════════════════════════════════════════════════════════════════════════
from matplotlib.patches import FancyArrowPatch

N_cats  = len(CATEGORIES)
angles  = np.linspace(0, 2*np.pi, N_cats, endpoint=False).tolist()
angles += angles[:1]   # close polygon

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for method in METHODS_COMP:
    vals = [safe_get(summary, method, HEATMAP_BUDGET, "categories", c, default=0.0)
            for c in CATEGORIES]
    vals += vals[:1]
    ax.plot(angles, vals, lw=2, color=METHOD_COLORS[method], label=method,
            marker=METHOD_MARKERS[method], ms=6)
    ax.fill(angles, vals, alpha=0.08, color=METHOD_COLORS[method])

# FullKV reference
full_vals = [safe_get(summary, "FullKV", "full", "categories", c, default=0.0)
             for c in CATEGORIES]
if any(full_vals):
    full_vals += full_vals[:1]
    ax.plot(angles, full_vals, lw=1.5, ls="--",
            color=METHOD_COLORS["FullKV"], label="FullKV", alpha=0.7)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([c.replace(" ", "\n") for c in CATEGORIES], fontsize=9)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
ax.set_title(f"Fig 5 — Task-Category Profile ({BUDGET_LABELS.get(HEATMAP_BUDGET,'?')} Budget)",
             pad=20, fontsize=12)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig5_dataset_radar.png"))
plt.close(fig)
print("  Saved: fig5_dataset_radar.png")

print("\nAll figures written to:", FIGS_DIR)
