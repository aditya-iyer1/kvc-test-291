#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
CONFIG_PATH = SCRIPT_DIR / "config.env"
if CONFIG_PATH.exists():
    for line in CONFIG_PATH.read_text().splitlines():
        if line.startswith("RESULTS_DIR="):
            RESULTS_DIR = Path(line.split("=", 1)[1].strip().strip('"\''))
            break

SCORES_DIR = RESULTS_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

KVC_DIR = PROJECT_DIR / "KVCache-Factory"
METRICS_PATH = KVC_DIR / "metrics.py"

spec = importlib.util.spec_from_file_location("metrics_mod", METRICS_PATH)
metrics_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_mod)

METHODS = ["FullKV", "SnapKV", "PyramidKV", "StreamingLLM"]
COMPRESSED_METHODS = ["SnapKV", "PyramidKV", "StreamingLLM"]
BUDGETS = ["10pct", "20pct", "50pct"]
BUDGET_TO_DIR = {
    "10pct": ("budget_10pct", 3150),
    "20pct": ("budget_20pct", 6300),
    "50pct": ("budget_50pct", 15750),
}
FULL_DIR = ("budget_full", 7950)
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
CATEGORY_MAP = {
    "Single-Doc QA": ["narrativeqa"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-Shot": ["triviaqa"],
    "Synthetic": ["passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}
DATASET_TO_CATEGORY = {
    dataset: category
    for category, datasets in CATEGORY_MAP.items()
    for dataset in datasets
}
DATASET_TO_METRIC = {
    "narrativeqa": metrics_mod.qa_f1_score,
    "hotpotqa": metrics_mod.qa_f1_score,
    "2wikimqa": metrics_mod.qa_f1_score,
    "triviaqa": metrics_mod.qa_f1_score,
    "gov_report": metrics_mod.rouge_score,
    "qmsum": metrics_mod.rouge_score,
    "multi_news": metrics_mod.rouge_score,
    "passage_retrieval_en": metrics_mod.retrieval_score,
    "lcc": metrics_mod.code_sim_score,
    "repobench-p": metrics_mod.code_sim_score,
}


def scorer(dataset, records):
    if not records:
        return None
    metric = DATASET_TO_METRIC[dataset]
    total_score = 0.0
    for data in records:
        prediction = data["pred"]
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        score = 0.0
        for ground_truth in data["answers"]:
            score = max(
                score,
                metric(
                    prediction,
                    ground_truth,
                    all_classes=data.get("all_classes"),
                ),
            )
        total_score += score
    return round(100.0 * total_score / len(records), 2)


def read_filtered_records(path, dataset, threshold):
    if not path.exists():
        return None
    kept = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if int(data.get("length", -1)) > threshold:
                kept.append(data)
    return kept


def nested_path(method, budget, dataset):
    if method == "FullKV":
        budget_dir, cap = FULL_DIR
    else:
        budget_dir, cap = BUDGET_TO_DIR[budget]
    return RESULTS_DIR / budget_dir / f"mistral-7b-instruct-v0.2_{cap}" / dataset / f"{method}.json"


def dataset_score_table_for_budget(budget):
    threshold = BUDGET_TO_DIR[budget][1]
    method_scores = {}
    counts = {}
    for method in METHODS:
        method_scores[method] = {}
        for dataset in DATASETS:
            path = nested_path(method, budget, dataset)
            records = read_filtered_records(path, dataset, threshold)
            if records is None:
                method_scores[method][dataset] = None
                continue
            if dataset not in counts:
                counts[dataset] = len(records)
            method_scores[method][dataset] = scorer(dataset, records)
    return method_scores, counts


def mean_of(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def build_summary():
    out = {
        "counts": {},
        "comparison": {},
        "category_tables": {},
    }

    for method in METHODS:
        out[method] = {}

    for budget in BUDGETS:
        method_scores, counts = dataset_score_table_for_budget(budget)
        out["counts"][budget] = counts
        out["category_tables"][budget] = {}

        for method in METHODS:
            dataset_entries = {}
            for dataset in DATASETS:
                score = method_scores[method].get(dataset)
                dataset_entries[dataset] = {
                    "score": score,
                    "category": DATASET_TO_CATEGORY[dataset],
                    "n_active": counts.get(dataset, 0),
                }

            category_scores = {}
            for category, datasets in CATEGORY_MAP.items():
                category_scores[category] = mean_of(
                    [method_scores[method].get(dataset) for dataset in datasets]
                )

            overall = mean_of([method_scores[method].get(dataset) for dataset in DATASETS])
            out[method][budget] = {
                "threshold": BUDGET_TO_DIR[budget][1],
                "overall": overall,
                "categories": category_scores,
                "datasets": dataset_entries,
            }
            out["category_tables"][budget][method] = category_scores

        out["comparison"][budget] = {
            method: out[method][budget]["overall"]
            for method in METHODS
        }

    return out


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def print_overall_table(summary):
    header = "| Method | 10% | 20% | 50% |"
    divider = "|--------|---:|---:|---:|"
    print("Table 4-Style: Compression-Active Subset Only")
    print(header)
    print(divider)
    for method in METHODS:
        vals = [fmt(summary[method][budget]["overall"]) for budget in BUDGETS]
        print(f"| {method} | " + " | ".join(vals) + " |")
    print()


def print_category_tables(summary):
    cats = list(CATEGORY_MAP.keys())
    header = "| Method | " + " | ".join(cats) + " |"
    divider = "|--------|" + "|".join(["---:"] * len(cats)) + "|"
    for budget in BUDGETS:
        print(f"Table 5-Style: Compression-Active Subset Only ({budget})")
        print(header)
        print(divider)
        for method in METHODS:
            vals = [fmt(summary[method][budget]["categories"][cat]) for cat in cats]
            print(f"| {method} | " + " | ".join(vals) + " |")
        print()


def print_counts(summary):
    header = "| Dataset | 10% n | 20% n | 50% n |"
    divider = "|---------|---:|---:|---:|"
    print("Active Example Counts by Dataset/Budget")
    print(header)
    print(divider)
    for dataset in DATASETS:
        vals = [str(summary["counts"][budget].get(dataset, 0)) for budget in BUDGETS]
        print(f"| {dataset} | " + " | ".join(vals) + " |")
    print()


def main():
    summary = build_summary()
    out_path = SCORES_DIR / "summary_compression_active.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_overall_table(summary)
    print_category_tables(summary)
    print_counts(summary)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
