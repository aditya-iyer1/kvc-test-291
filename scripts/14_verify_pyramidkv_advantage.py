#!/usr/bin/env python3
import importlib.util
import json
import math
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

DATASET = "narrativeqa"
BUDGETS = {
    "10pct": ("budget_10pct", 3150),
    "20pct": ("budget_20pct", 6300),
    "50pct": ("budget_50pct", 15750),
}
LONGEST_THRESHOLD = 15750
METHODS = ["SnapKV", "PyramidKV"]


def score_example(data):
    pred = data["pred"]
    best = 0.0
    for gt in data["answers"]:
        best = max(best, metrics_mod.qa_f1_score(pred, gt, all_classes=data.get("all_classes")))
    return 100.0 * best


def load_scores(method, budget, cap):
    path = RESULTS_DIR / budget / f"mistral-7b-instruct-v0.2_{cap}" / DATASET / f"{method}.json"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            rows.append({
                "idx": idx,
                "length": int(data["length"]),
                "score": score_example(data),
            })
    return rows


def stats(rows):
    vals = [r["score"] for r in rows]
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    variance = sum((x - mean) ** 2 for x in vals_sorted) / n
    median = vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "median": median,
    }


def fmt_stats(label, st):
    return f"{label:10s} n={st['n']:3d} mean={st['mean']:.2f} std={st['std']:.2f} median={st['median']:.2f}"


def main():
    overall_verdicts = []
    print("NarrativeQA only: SnapKV vs PyramidKV\n")
    for budget_name, (budget_dir, cap) in BUDGETS.items():
        snap = load_scores("SnapKV", budget_dir, cap)
        pyr = load_scores("PyramidKV", budget_dir, cap)
        assert len(snap) == len(pyr)
        snap_st = stats(snap)
        pyr_st = stats(pyr)

        snap_long = [r for r in snap if r["length"] > LONGEST_THRESHOLD]
        pyr_long = [r for r in pyr if r["length"] > LONGEST_THRESHOLD]
        snap_long_st = stats(snap_long)
        pyr_long_st = stats(pyr_long)

        delta_all = pyr_st["mean"] - snap_st["mean"]
        delta_long = pyr_long_st["mean"] - snap_long_st["mean"]
        overall_verdicts.append((budget_name, delta_long))

        print(f"[{budget_name}] full narrativeqa distribution")
        print(fmt_stats("SnapKV", snap_st))
        print(fmt_stats("PyramidKV", pyr_st))
        print(f"delta mean (PyramidKV - SnapKV): {delta_all:+.2f}")
        print()
        print(f"[{budget_name}] longest narrativeqa subset (length > {LONGEST_THRESHOLD})")
        print(fmt_stats("SnapKV", snap_long_st))
        print(fmt_stats("PyramidKV", pyr_long_st))
        print(f"delta mean (PyramidKV - SnapKV): {delta_long:+.2f}")
        print()

    improved = [budget for budget, delta in overall_verdicts if delta > 0]
    if improved:
        verdict = f"PyramidKV outperforms SnapKV on the longest narrativeqa examples at: {', '.join(improved)}."
    else:
        verdict = "PyramidKV does not outperform SnapKV on the longest narrativeqa examples at any tested budget."
    print("Verdict:", verdict)


if __name__ == "__main__":
    main()
