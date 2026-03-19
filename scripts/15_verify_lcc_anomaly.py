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

DATASET = "lcc"
FULL_PATH = RESULTS_DIR / "budget_full" / "mistral-7b-instruct-v0.2_7950" / DATASET / "FullKV.json"
COMPRESSED = {
    "SnapKV_10pct": ("budget_10pct", 3150, "SnapKV", 3150),
    "SnapKV_20pct": ("budget_20pct", 6300, "SnapKV", 6300),
    "SnapKV_50pct": ("budget_50pct", 15750, "SnapKV", 15750),
    "PyramidKV_10pct": ("budget_10pct", 3150, "PyramidKV", 3150),
    "PyramidKV_20pct": ("budget_20pct", 6300, "PyramidKV", 6300),
    "PyramidKV_50pct": ("budget_50pct", 15750, "PyramidKV", 15750),
    "StreamingLLM_10pct": ("budget_10pct", 3150, "StreamingLLM", 3150),
    "StreamingLLM_20pct": ("budget_20pct", 6300, "StreamingLLM", 6300),
    "StreamingLLM_50pct": ("budget_50pct", 15750, "StreamingLLM", 15750),
}


def score_example(data):
    pred = data["pred"]
    best = 0.0
    for gt in data["answers"]:
        best = max(best, metrics_mod.code_sim_score(pred, gt, all_classes=data.get("all_classes")))
    return 100.0 * best


def load_rows(path):
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


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats(vals):
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": sorted_vals[0],
        "p10": percentile(sorted_vals, 0.10),
        "median": percentile(sorted_vals, 0.50),
        "p90": percentile(sorted_vals, 0.90),
        "max": sorted_vals[-1],
    }


def fmt_stats(name, st):
    return (
        f"{name:18s} mean={st['mean']:.2f} std={st['std']:.2f} min={st['min']:.2f} "
        f"p10={st['p10']:.2f} median={st['median']:.2f} p90={st['p90']:.2f} max={st['max']:.2f}"
    )


def main():
    full_rows = load_rows(FULL_PATH)
    full_by_idx = {r["idx"]: r for r in full_rows}
    print("LCC anomaly diagnostic\n")
    print(fmt_stats("FullKV", stats([r["score"] for r in full_rows])))
    print()

    verdict_lines = []
    for label, (budget_dir, cap, method, threshold) in COMPRESSED.items():
        path = RESULTS_DIR / budget_dir / f"mistral-7b-instruct-v0.2_{cap}" / DATASET / f"{method}.json"
        rows = load_rows(path)
        deltas = []
        win = lose = tie = 0
        short_deltas = []
        long_deltas = []
        for row in rows:
            full = full_by_idx[row["idx"]]["score"]
            delta = row["score"] - full
            deltas.append(delta)
            if delta > 1e-9:
                win += 1
            elif delta < -1e-9:
                lose += 1
            else:
                tie += 1
            if row["length"] <= threshold:
                short_deltas.append(delta)
            else:
                long_deltas.append(delta)

        row_stats = stats([r["score"] for r in rows])
        delta_stats = stats(deltas)
        print(f"[{label}]")
        print(fmt_stats(label, row_stats))
        print(
            f"delta vs FullKV: mean={delta_stats['mean']:.2f} std={delta_stats['std']:.2f} "
            f"min={delta_stats['min']:.2f} median={delta_stats['median']:.2f} max={delta_stats['max']:.2f}"
        )
        print(f"wins={win} loses={lose} ties={tie}")
        print(
            f"short examples (length <= {threshold}): n={len(short_deltas)} mean_delta="
            f"{(sum(short_deltas)/len(short_deltas) if short_deltas else 0.0):+.2f}"
        )
        print(
            f"long examples  (length >  {threshold}): n={len(long_deltas)} mean_delta="
            f"{(sum(long_deltas)/len(long_deltas) if long_deltas else 0.0):+.2f}"
        )
        print()

        if abs(delta_stats["mean"]) < 0.5 and max(abs(x) for x in deltas) < 5:
            verdict_lines.append(f"{label}: mostly noise-level variation.")
        elif len(long_deltas) and abs(sum(long_deltas)/len(long_deltas)) > abs(sum(short_deltas)/len(short_deltas) if short_deltas else 0.0):
            verdict_lines.append(f"{label}: advantage/disadvantage is concentrated in the few long examples, not the many short ones.")
        else:
            verdict_lines.append(f"{label}: pattern is spread more broadly than just the longest examples.")

    print("Verdict:")
    for line in verdict_lines:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
