#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
config_path = os.path.join(SCRIPT_DIR, "config.env")
if os.path.exists(config_path):
    with open(config_path) as f:
        for line in f:
            if line.startswith("RESULTS_DIR="):
                RESULTS_DIR = line.strip().split("=", 1)[1].strip('"\'')
                break
SCORES_DIR = os.path.join(RESULTS_DIR, "scores")
FIGS_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

summary_path = os.path.join(SCORES_DIR, "summary_run8.json")
if not os.path.exists(summary_path):
    print(f"ERROR: {summary_path} not found.")
    sys.exit(1)

with open(summary_path) as f:
    summary = json.load(f)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

METHODS = ["FullKV", "SnapKV", "PyramidKV", "StreamingLLM"]
BUDGETS = ["full", "10pct", "20pct", "50pct"]
DATASETS = [
    "narrativeqa",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "qmsum",
    "multi_news",
    "triviaqa",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
CATEGORY_GROUPS = [
    ("Single-Doc QA", ["narrativeqa"]),
    ("Multi-Doc QA", ["hotpotqa", "2wikimqa"]),
    ("Summarization", ["gov_report", "qmsum", "multi_news"]),
    ("Few-Shot", ["triviaqa"]),
    ("Synthetic", ["passage_retrieval_en"]),
    ("Code", ["lcc", "repobench-p"]),
]
METRIC_NOTE = (
    "Metrics: narrativeqa/hotpotqa/2wikimqa/triviaqa = F1; "
    "gov_report/qmsum/multi_news = ROUGE-L; "
    "passage_retrieval_en = ExactMatch; "
    "lcc/repobench-p = Edit Similarity"
)


def safe_get(d, *keys, default=None):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def build_matrix(budget):
    raw = np.full((len(METHODS), len(DATASETS)), np.nan, dtype=float)
    for i, method in enumerate(METHODS):
        source_budget = "full" if method == "FullKV" else budget
        for j, dataset in enumerate(DATASETS):
            score = safe_get(summary, method, source_budget, "datasets", dataset, "score")
            if score is not None:
                raw[i, j] = float(score)
    return raw


def robust_normalize(raw):
    norm = np.zeros_like(raw)
    finite = np.isfinite(raw)
    if not finite.any():
        return norm

    vals = raw[finite]
    lo = float(np.nanpercentile(vals, 5))
    hi = float(np.nanpercentile(vals, 85))
    if hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if hi <= lo:
        norm[finite] = 0.5
        return norm

    clipped = np.clip(vals, lo, hi)
    norm[finite] = (clipped - lo) / (hi - lo)
    return norm


def plot_budget(budget):
    raw = build_matrix(budget)
    norm = robust_normalize(raw)

    fig, ax = plt.subplots(figsize=(16, 4.8))
    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)

    yticklabels = ["FullKV(full)", "SnapKV", "PyramidKV", "StreamingLLM"]
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=25, ha="right")

    for i in range(len(METHODS)):
        for j in range(len(DATASETS)):
            if np.isfinite(raw[i, j]):
                text_color = "white" if norm[i, j] < 0.18 or norm[i, j] > 0.82 else "black"
                ax.text(
                    j,
                    i,
                    f"{raw[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="dimgray")

    offset = 0
    for label, group in CATEGORY_GROUPS:
        group_start = offset
        group_end = offset + len(group) - 1
        center = (group_start + group_end) / 2
        ax.text(center, -1.05, label, ha="center", va="bottom", fontsize=10, fontweight="bold", clip_on=False)
        offset += len(group)
        if offset < len(DATASETS):
            ax.axvline(offset - 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.9)

    label = "No Compression" if budget == "full" else budget
    ax.set_title(f"Fig 7 - Per-Dataset Accuracy Heatmap ({label})", pad=42)
    ax.set_xlabel(METRIC_NOTE)
    plt.colorbar(
        im,
        ax=ax,
        label="Relative performance across plotted scores (clipped 5th-85th pct)",
        fraction=0.025,
        pad=0.02,
    )
    fig.tight_layout()

    out_name = f"fig7_dataset_heatmap_{budget}.png"
    out_path = os.path.join(FIGS_DIR, out_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


for budget in BUDGETS:
    plot_budget(budget)
