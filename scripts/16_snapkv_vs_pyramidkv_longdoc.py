#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
CONFIG_PATH = SCRIPT_DIR / "config.env"
if CONFIG_PATH.exists():
    for line in CONFIG_PATH.read_text().splitlines():
        if line.startswith("RESULTS_DIR="):
            RESULTS_DIR = Path(line.split("=", 1)[1].strip().strip("\"'"))
            break

KVC_DIR = PROJECT_DIR / "KVCache-Factory"
METRICS_PATH = KVC_DIR / "metrics.py"

spec = importlib.util.spec_from_file_location("metrics_mod", METRICS_PATH)
metrics_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_mod)

DATASETS = [
    "narrativeqa", "hotpotqa", "2wikimqa", "gov_report", "qmsum",
    "multi_news", "triviaqa", "passage_retrieval_en", "lcc", "repobench-p",
]
BUDGETS = {
    "10pct": ("budget_10pct", 3150),
    "20pct": ("budget_20pct", 6300),
    "50pct": ("budget_50pct", 15750),
}
LONG_THRESHOLD = 15750
METHODS = ["SnapKV", "PyramidKV"]
DATASET2METRIC = {
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


def score_file(path, dataset):
    metric = DATASET2METRIC[dataset]
    total = 0.0
    kept = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if int(data["length"]) <= LONG_THRESHOLD:
                continue
            pred = data["pred"]
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip("\n").split("\n")[0]
            best = 0.0
            for gt in data["answers"]:
                best = max(best, metric(pred, gt, all_classes=data.get("all_classes")))
            total += best
            kept += 1
    return (round(100.0 * total / kept, 2) if kept else None), kept


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


def main():
    print(f"SnapKV vs PyramidKV on examples with length > {LONG_THRESHOLD}\n")
    overall = {method: [] for method in METHODS}
    for budget_name, (budget_dir, cap) in BUDGETS.items():
        print(f"[{budget_name}]")
        print(f"{'Dataset':22s} {'n':>5s} {'SnapKV':>10s} {'PyramidKV':>12s} {'Delta':>10s}")
        print("-" * 64)
        for dataset in DATASETS:
            row = {}
            n_ref = None
            for method in METHODS:
                path = RESULTS_DIR / budget_dir / f"mistral-7b-instruct-v0.2_{cap}" / dataset / f"{method}.json"
                score, n = score_file(path, dataset)
                row[method] = score
                if score is not None:
                    overall[method].append(score)
                if n_ref is None:
                    n_ref = n
            delta = None if row["SnapKV"] is None or row["PyramidKV"] is None else row["PyramidKV"] - row["SnapKV"]
            delta_str = "—" if delta is None else f"{delta:+.2f}"
            print(f"{dataset:22s} {n_ref:5d} {fmt(row['SnapKV']):>10s} {fmt(row['PyramidKV']):>12s} {delta_str:>10s}")
        print("-" * 64)
        snap_scores = []
        pyr_scores = []
        for dataset in DATASETS:
            path_snap = RESULTS_DIR / budget_dir / f"mistral-7b-instruct-v0.2_{cap}" / dataset / "SnapKV.json"
            path_pyr = RESULTS_DIR / budget_dir / f"mistral-7b-instruct-v0.2_{cap}" / dataset / "PyramidKV.json"
            snap_score, n_snap = score_file(path_snap, dataset)
            pyr_score, n_pyr = score_file(path_pyr, dataset)
            if snap_score is not None:
                snap_scores.append(snap_score)
            if pyr_score is not None:
                pyr_scores.append(pyr_score)
        snap_overall = round(sum(snap_scores) / len(snap_scores), 2) if snap_scores else None
        pyr_overall = round(sum(pyr_scores) / len(pyr_scores), 2) if pyr_scores else None
        overall_delta = None if snap_overall is None or pyr_overall is None else pyr_overall - snap_overall
        print(f"{'Overall':22s} {'':5s} {fmt(snap_overall):>10s} {fmt(pyr_overall):>12s} {('—' if overall_delta is None else f'{overall_delta:+.2f}'):>10s}")
        print()


if __name__ == "__main__":
    main()
