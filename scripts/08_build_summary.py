#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path
from statistics import mean

METHODS = ["FullKV", "SnapKV", "PyramidKV", "StreamingLLM"]
COMP_METHODS = ["SnapKV", "PyramidKV", "StreamingLLM"]
BUDGETS_COMP = ["10pct", "20pct", "50pct"]
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


def parse_score(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v == -1.0:
        return None
    return round(v, 2)


def safe_mean(vals):
    vals = list(vals)
    return round(mean(vals), 2) if vals else None


def read_results_csv(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}

    datasets = rows[0][1:]
    method_scores = {}
    for row in rows[1:]:
        if not row:
            continue
        method = row[0].strip()
        if method not in METHODS:
            continue
        scores = {}
        for ds, raw in zip(datasets, row[1:]):
            val = parse_score(raw)
            if val is not None:
                scores[ds] = val
        method_scores[method] = scores
    return method_scores


def build_intersections(comp_scores_by_budget):
    intersections = {}
    for budget in BUDGETS_COMP:
        by_method = comp_scores_by_budget.get(budget, {})
        sets = [set(by_method.get(method, {}).keys()) for method in COMP_METHODS]
        if all(sets):
            inter = set.intersection(*sets)
        else:
            inter = set()
        intersections[budget] = sorted(inter)
    return intersections


def category_scores_for(dataset_scores):
    categories = {}
    for category, datasets in CATEGORY_MAP.items():
        vals = [dataset_scores[d] for d in datasets if d in dataset_scores]
        categories[category] = safe_mean(vals)
    return categories


def datasets_block_for(dataset_scores):
    return {
        dataset: {
            "score": score,
            "category": DATASET_TO_CATEGORY[dataset],
        }
        for dataset, score in sorted(dataset_scores.items())
        if dataset in DATASET_TO_CATEGORY
    }


def build_budget_entry(method_scores, dataset_subset):
    effective_scores = {
        dataset: method_scores[dataset]
        for dataset in dataset_subset
        if dataset in method_scores and dataset in DATASET_TO_CATEGORY
    }
    return {
        "overall": safe_mean(effective_scores.values()),
        "categories": category_scores_for(effective_scores),
        "datasets": datasets_block_for(effective_scores),
    }


def compute_comparison(comp_scores_by_budget, full_scores, intersections):
    table = {
        "FullKV(full)": {},
        "SnapKV": {},
        "PyramidKV": {},
        "StreamingLLM": {},
    }

    for budget in BUDGETS_COMP:
        inter = intersections[budget]
        for method in COMP_METHODS:
            ds_scores = comp_scores_by_budget.get(budget, {}).get(method, {})
            vals = [ds_scores[d] for d in inter if d in ds_scores]
            table[method][budget] = safe_mean(vals)

        overlap = [d for d in inter if d in full_scores]
        table["FullKV(full)"][budget] = safe_mean(full_scores[d] for d in overlap)

    return table


def print_table_1(comparison):
    print("Table 1: Overall scores on 3-method intersection datasets (per budget)")
    header = ["Method", "10pct overall", "20pct overall", "50pct overall"]
    print(" | ".join(f"{h:>14}" for h in header))
    print("-" * (len(" | ".join(f"{h:>14}" for h in header))))

    order = ["FullKV(full)", "SnapKV", "PyramidKV", "StreamingLLM"]
    for method in order:
        row = [
            method,
            "-" if comparison[method]["10pct"] is None else f"{comparison[method]['10pct']:.2f}",
            "-" if comparison[method]["20pct"] is None else f"{comparison[method]['20pct']:.2f}",
            "-" if comparison[method]["50pct"] is None else f"{comparison[method]['50pct']:.2f}",
        ]
        print(" | ".join(f"{c:>14}" for c in row))


def print_table_2(intersections, full_scores):
    print("\nTable 2: Intersection datasets by compressed budget")
    full_set = set(full_scores.keys())
    for budget in BUDGETS_COMP:
        inter = intersections[budget]
        overlap = [d for d in inter if d in full_set]
        missing_in_full = [d for d in inter if d not in full_set]

        print(f"\n[{budget}] n={len(inter)}")
        print("intersection:", ", ".join(inter) if inter else "(none)")
        print("FullKV overlap:", ", ".join(overlap) if overlap else "(none)")
        print("FullKV missing from intersection:", ", ".join(missing_in_full) if missing_in_full else "(none)")


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", "~/kvc-test-291/results_mistral_run7")).expanduser()

    budget_csv = {
        "10pct": results_dir / "budget_10pct" / "mistral-7b-instruct-v0.2_3150" / "results.csv",
        "20pct": results_dir / "budget_20pct" / "mistral-7b-instruct-v0.2_6300" / "results.csv",
        "50pct": results_dir / "budget_50pct" / "mistral-7b-instruct-v0.2_15750" / "results.csv",
        "full": results_dir / "budget_full" / "mistral-7b-instruct-v0.2_7950" / "results.csv",
    }

    comp_scores_by_budget = {}
    for budget in BUDGETS_COMP:
        path = budget_csv[budget]
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            comp_scores_by_budget[budget] = {}
            continue
        comp_scores_by_budget[budget] = read_results_csv(path)

    full_scores = {}
    full_path = budget_csv["full"]
    if full_path.exists():
        full_scores = read_results_csv(full_path).get("FullKV", {})
    else:
        print(f"[SKIP] missing: {full_path}")

    intersections = build_intersections(comp_scores_by_budget)
    comparison = compute_comparison(comp_scores_by_budget, full_scores, intersections)

    out = {method: {} for method in METHODS}

    for budget in BUDGETS_COMP:
        inter = intersections[budget]
        for method in COMP_METHODS:
            method_scores = comp_scores_by_budget.get(budget, {}).get(method, {})
            out[method][budget] = build_budget_entry(method_scores, inter)

    # 04_plot.py expects a single FullKV/full entry. Use overlap with the
    # common compressed-task set so the reference row is comparable.
    full_overlap = sorted(set(full_scores.keys()) & set(DATASET_TO_CATEGORY))
    out["FullKV"]["full"] = build_budget_entry(full_scores, full_overlap)

    out["comparison"] = comparison
    out["intersections"] = intersections
    out["fullkv_full_available"] = sorted(full_scores.keys())

    out_dir = results_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_run8.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {out_path}\n")
    print_table_1(comparison)
    print_table_2(intersections, full_scores)


if __name__ == "__main__":
    main()
