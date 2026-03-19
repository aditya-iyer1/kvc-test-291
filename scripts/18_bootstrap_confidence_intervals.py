#!/usr/bin/env python3
import difflib
import json
import os
import re
import string
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np


ITERATIONS = 10_000
SEED = 0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "config.env"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
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
ALL_METHODS = ["FullKV", "SnapKV", "PyramidKV", "StreamingLLM"]
COMPRESSED_METHODS = ["SnapKV", "PyramidKV", "StreamingLLM"]
EXTRA_METHODS = ["StreamingLLMSliding"]
BUDGETS = {
    "10pct": {"dir": "budget_10pct", "cap": 3150, "threshold": 3150},
    "20pct": {"dir": "budget_20pct", "cap": 6300, "threshold": 6300},
    "50pct": {"dir": "budget_50pct", "cap": 15750, "threshold": 15750},
}
FULLKV_INFO = {"dir": "budget_full", "cap": 7950, "label": "full"}
TRIM_FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}

RNG = np.random.default_rng(SEED)
BOOT_WEIGHTS = {}
WARNINGS = []


def load_results_dir():
    results_dir = DEFAULT_RESULTS_DIR
    if CONFIG_PATH.exists():
        for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("RESULTS_DIR="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                results_dir = Path(value)
                break
    return results_dir


RESULTS_DIR = load_results_dir()
OUT_PATH = RESULTS_DIR / "scores" / "bootstrap_ci_results.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        return "".join(ch for ch in value if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth, **_kwargs):
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def retrieval_score(prediction, ground_truth, **_kwargs):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    correct = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(correct / len(numbers))


def lcs_length(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0
    if len(tokens_a) < len(tokens_b):
        shorter, longer = tokens_a, tokens_b
    else:
        shorter, longer = tokens_b, tokens_a
    previous = [0] * (len(shorter) + 1)
    for token_long in longer:
        current = [0]
        for idx, token_short in enumerate(shorter, start=1):
            if token_long == token_short:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def code_sim_score(prediction, ground_truth, **_kwargs):
    lines = prediction.lstrip("\n").split("\n")
    candidate = ""
    for line in lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            candidate = line
            break
    return float(difflib.SequenceMatcher(None, candidate, ground_truth).ratio())


def rouge_score(prediction, ground_truth, **_kwargs):
    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return float((2 * precision * recall) / (precision + recall))


DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "triviaqa": qa_f1_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def nested_path(method, budget, dataset):
    if method == "FullKV":
        info = FULLKV_INFO
    else:
        info = BUDGETS[budget]
    return RESULTS_DIR / info["dir"] / f"mistral-7b-instruct-v0.2_{info['cap']}" / dataset / f"{method}.json"


def normalize_prediction(dataset, prediction):
    if dataset in TRIM_FIRST_LINE_DATASETS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_record(dataset, record):
    prediction = normalize_prediction(dataset, record.get("pred", ""))
    answers = record.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    metric = DATASET2METRIC[dataset]
    score = 0.0
    for ground_truth in answers:
        score = max(
            score,
            metric(
                prediction,
                ground_truth,
                all_classes=record.get("all_classes"),
            ),
        )
    return 100.0 * score


def load_score_table(method, budget, dataset):
    path = nested_path(method, budget, dataset)
    if not path.exists():
        WARNINGS.append(f"Missing input: {path}")
        return None

    ids = []
    scores = []
    lengths = []

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("_id")
            if record_id is None:
                record_id = f"idx:{idx}"
            ids.append(str(record_id))
            scores.append(score_record(dataset, record))
            lengths.append(int(record.get("length", -1)))

    if not scores:
        WARNINGS.append(f"Empty input: {path}")
        return None

    return {
        "ids": ids,
        "scores": np.asarray(scores, dtype=np.float64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "index_by_id": {record_id: idx for idx, record_id in enumerate(ids)},
        "path": str(path),
    }


def load_score_table_task(task):
    method, budget, dataset = task
    return method, budget, dataset, load_score_table(method, budget, dataset)


def load_all_raw():
    raw = {method: {budget: {} for budget in BUDGETS} for method in ALL_METHODS + EXTRA_METHODS}
    tasks = []

    for dataset in DATASETS:
        tasks.append(("FullKV", "10pct", dataset))
    for method in COMPRESSED_METHODS + EXTRA_METHODS:
        for budget in BUDGETS:
            for dataset in DATASETS:
                tasks.append((method, budget, dataset))

    max_workers = min(16, os.cpu_count() or 1, len(tasks))
    loader = map(load_score_table_task, tasks)
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            loader = pool.map(load_score_table_task, tasks, chunksize=1)

            for method, budget, dataset, table in loader:
                if method == "FullKV":
                    for budget_name in BUDGETS:
                        raw["FullKV"][budget_name][dataset] = table
                else:
                    raw[method][budget][dataset] = table
        return raw

    for method, budget, dataset, table in loader:
        if method == "FullKV":
            for budget_name in BUDGETS:
                raw["FullKV"][budget_name][dataset] = table
        else:
            raw[method][budget][dataset] = table
    return raw


RAW = None


def bootstrap_mean(samples):
    values = np.asarray(samples, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty array")
    if n not in BOOT_WEIGHTS:
        probs = np.full(n, 1.0 / n, dtype=np.float64)
        weights = RNG.multinomial(n, probs, size=ITERATIONS).astype(np.float32) / float(n)
        BOOT_WEIGHTS[n] = weights
    return BOOT_WEIGHTS[n] @ values


def ci_from_samples(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def significant(ci):
    lo, hi = ci
    return not (lo <= 0.0 <= hi)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    lo, hi = ci_from_samples(bootstrap_mean(values))
    return mean, lo, hi


def method_scores(item, budget=None, compression_active=False):
    if item is None:
        return None
    scores = item["scores"]
    lengths = item["lengths"]
    if compression_active:
        assert budget is not None
        mask = lengths > BUDGETS[budget]["threshold"]
        scores = scores[mask]
    if len(scores) == 0:
        return None
    return scores


def paired_scores(item_a, item_b, budget=None, compression_active=False):
    if item_a is None or item_b is None:
        return None, None, None

    if item_a["ids"] == item_b["ids"]:
        scores_a = item_a["scores"]
        scores_b = item_b["scores"]
        lengths_a = item_a["lengths"]
        lengths_b = item_b["lengths"]
    else:
        common_ids = [record_id for record_id in item_a["ids"] if record_id in item_b["index_by_id"]]
        if not common_ids:
            WARNINGS.append(
                "No overlapping IDs for "
                f"{item_a['path']} vs {item_b['path']}"
            )
            return None, None, None
        idx_a = np.asarray([item_a["index_by_id"][record_id] for record_id in common_ids], dtype=np.int64)
        idx_b = np.asarray([item_b["index_by_id"][record_id] for record_id in common_ids], dtype=np.int64)
        scores_a = item_a["scores"][idx_a]
        scores_b = item_b["scores"][idx_b]
        lengths_a = item_a["lengths"][idx_a]
        lengths_b = item_b["lengths"][idx_b]

    if not np.array_equal(lengths_a, lengths_b):
        WARNINGS.append(
            "Length mismatch for paired comparison "
            f"{item_a['path']} vs {item_b['path']}; using method A lengths"
        )

    if compression_active:
        assert budget is not None
        mask = lengths_a > BUDGETS[budget]["threshold"]
        scores_a = scores_a[mask]
        scores_b = scores_b[mask]
        lengths_a = lengths_a[mask]

    if len(scores_a) == 0:
        return None, None, None
    return scores_a, scores_b, lengths_a


def overall_method_ci(method, budget, compression_active=False):
    point_means = []
    bootstrap_means = []
    used_datasets = []
    n_examples = {}

    for dataset in DATASETS:
        scores = method_scores(RAW[method][budget][dataset], budget=budget, compression_active=compression_active)
        if scores is None:
            continue
        point_means.append(float(scores.mean()))
        bootstrap_means.append(bootstrap_mean(scores))
        used_datasets.append(dataset)
        n_examples[dataset] = int(len(scores))

    if not bootstrap_means:
        return None

    stacked = np.stack(bootstrap_means, axis=0)
    mean = float(np.mean(point_means))
    ci = ci_from_samples(stacked.mean(axis=0))
    return {
        "mean": mean,
        "ci95": [ci[0], ci[1]],
        "datasets": used_datasets,
        "n_examples_by_dataset": n_examples,
        "n_datasets": len(used_datasets),
        "compression_active": compression_active,
    }


def overall_diff_ci(method_a, method_b, budget, compression_active=False):
    point_means = []
    bootstrap_means = []
    used_datasets = []
    n_examples = {}

    for dataset in DATASETS:
        scores_a, scores_b, _ = paired_scores(
            RAW[method_a][budget][dataset],
            RAW[method_b][budget][dataset],
            budget=budget,
            compression_active=compression_active,
        )
        if scores_a is None:
            continue
        diff = scores_a - scores_b
        point_means.append(float(diff.mean()))
        bootstrap_means.append(bootstrap_mean(diff))
        used_datasets.append(dataset)
        n_examples[dataset] = int(len(diff))

    if not bootstrap_means:
        return None

    stacked = np.stack(bootstrap_means, axis=0)
    ci = ci_from_samples(stacked.mean(axis=0))
    return {
        "mean_diff": float(np.mean(point_means)),
        "ci95": [ci[0], ci[1]],
        "significant": significant(ci),
        "datasets": used_datasets,
        "n_examples_by_dataset": n_examples,
        "n_datasets": len(used_datasets),
        "compression_active": compression_active,
    }


def dataset_method_ci(method, budget, dataset):
    scores = method_scores(RAW[method][budget][dataset], budget=budget, compression_active=False)
    if scores is None:
        return None
    mean, lo, hi = mean_ci(scores)
    return {
        "mean": mean,
        "ci95": [lo, hi],
        "n_examples": int(len(scores)),
    }


def dataset_diff_ci(method_a, method_b, budget, dataset, compression_active=False):
    scores_a, scores_b, _ = paired_scores(
        RAW[method_a][budget][dataset],
        RAW[method_b][budget][dataset],
        budget=budget,
        compression_active=compression_active,
    )
    if scores_a is None:
        return None
    diff = scores_a - scores_b
    mean, lo, hi = mean_ci(diff)
    return {
        "mean_diff": mean,
        "ci95": [lo, hi],
        "significant": significant((lo, hi)),
        "n_examples": int(len(diff)),
        "compression_active": compression_active,
    }


def add_comparison(results, metric, method_a, method_b, stats, section):
    row = {
        "section": section,
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "mean_diff": stats["mean_diff"],
        "ci95": stats["ci95"],
        "significant": stats["significant"],
    }
    for key in ("datasets", "n_examples_by_dataset", "n_datasets", "n_examples", "compression_active"):
        if key in stats:
            row[key] = stats[key]
    results["comparisons"].append(row)
    return row


def build_results():
    results = {
        "meta": {
            "results_dir": str(RESULTS_DIR),
            "output_path": str(OUT_PATH),
            "iterations": ITERATIONS,
            "seed": SEED,
            "datasets": DATASETS,
            "budgets": {budget: info["threshold"] for budget, info in BUDGETS.items()},
            "fullkv_source_budget": FULLKV_INFO["label"],
            "notes": [
                "Per-example scores are computed with the same dataset2metric mapping and prediction normalization as KVCache-Factory/eval.py.",
                "Overall means are equal-weight averages across datasets, matching the paper's aggregate style.",
                "FullKV uses budget_full predictions as a constant baseline for all budget comparisons.",
                "Compression-active subsets keep examples with length > budget threshold, matching scripts/11_compression_active_analysis.py.",
            ],
        },
        "overall_means": [],
        "per_dataset_10pct_means": [],
        "comparisons": [],
        "flags": {},
        "warnings": WARNINGS,
    }

    overall_index = {}
    per_dataset_index = {}
    comparison_index = {}

    for budget in BUDGETS:
        overall_index[budget] = {}
        for method in ALL_METHODS:
            stats = overall_method_ci(method, budget, compression_active=False)
            if stats is None:
                continue
            row = {
                "budget": budget,
                "method": method,
                "source_budget": "full" if method == "FullKV" else budget,
                **stats,
            }
            overall_index[budget][method] = row
            results["overall_means"].append(row)

    for dataset in DATASETS:
        per_dataset_index[dataset] = {}
        for method in ALL_METHODS:
            stats = dataset_method_ci(method, "10pct", dataset)
            if stats is None:
                continue
            row = {
                "budget": "10pct",
                "dataset": dataset,
                "method": method,
                "source_budget": "full" if method == "FullKV" else "10pct",
                **stats,
            }
            per_dataset_index[dataset][method] = row
            results["per_dataset_10pct_means"].append(row)

    for budget in BUDGETS:
        for method in COMPRESSED_METHODS:
            stats = overall_diff_ci(method, "FullKV", budget, compression_active=False)
            if stats is None:
                continue
            row = add_comparison(
                results,
                metric=f"overall_{budget}",
                method_a=method,
                method_b="FullKV",
                stats=stats,
                section="overall_fullkv_gap",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

    for dataset in DATASETS:
        for method in COMPRESSED_METHODS:
            stats = dataset_diff_ci(method, "FullKV", "10pct", dataset, compression_active=False)
            if stats is None:
                continue
            row = add_comparison(
                results,
                metric=f"{dataset}_10pct",
                method_a=method,
                method_b="FullKV",
                stats=stats,
                section="per_dataset_10pct_vs_fullkv",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

    for budget in BUDGETS:
        stats = overall_diff_ci("SnapKV", "PyramidKV", budget, compression_active=False)
        if stats is not None:
            row = add_comparison(
                results,
                metric=f"overall_{budget}",
                method_a="SnapKV",
                method_b="PyramidKV",
                stats=stats,
                section="snapkv_vs_pyramidkv",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

        stats = overall_diff_ci("SnapKV", "PyramidKV", budget, compression_active=True)
        if stats is not None:
            row = add_comparison(
                results,
                metric=f"compression_active_{budget}",
                method_a="SnapKV",
                method_b="PyramidKV",
                stats=stats,
                section="snapkv_vs_pyramidkv",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

    for budget in BUDGETS:
        for method in COMPRESSED_METHODS:
            stats = dataset_diff_ci(method, "FullKV", budget, "lcc", compression_active=False)
            if stats is None:
                continue
            row = add_comparison(
                results,
                metric=f"lcc_{budget}",
                method_a=method,
                method_b="FullKV",
                stats=stats,
                section="lcc_anomaly",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

    for budget in BUDGETS:
        stats = overall_diff_ci("StreamingLLMSliding", "StreamingLLM", budget, compression_active=False)
        if stats is not None:
            row = add_comparison(
                results,
                metric=f"overall_{budget}",
                method_a="StreamingLLMSliding",
                method_b="StreamingLLM",
                stats=stats,
                section="streaming_vs_sliding",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

        stats = dataset_diff_ci(
            "StreamingLLMSliding",
            "StreamingLLM",
            budget,
            "passage_retrieval_en",
            compression_active=False,
        )
        if stats is not None:
            row = add_comparison(
                results,
                metric=f"passage_retrieval_en_{budget}",
                method_a="StreamingLLMSliding",
                method_b="StreamingLLM",
                stats=stats,
                section="streaming_vs_sliding",
            )
            comparison_index[(row["section"], row["metric"], row["method_a"], row["method_b"])] = row

    results["flags"] = {
        "per_dataset_10pct_vs_fullkv_ci_includes_zero": [
            row
            for row in results["comparisons"]
            if row["section"] == "per_dataset_10pct_vs_fullkv" and not row["significant"]
        ],
        "snapkv_minus_pyramidkv_10pct_overall": comparison_index.get(
            ("snapkv_vs_pyramidkv", "overall_10pct", "SnapKV", "PyramidKV")
        ),
        "snapkv_minus_pyramidkv_10pct_compression_active": comparison_index.get(
            ("snapkv_vs_pyramidkv", "compression_active_10pct", "SnapKV", "PyramidKV")
        ),
        "lcc_anomaly_rows": [
            row for row in results["comparisons"] if row["section"] == "lcc_anomaly"
        ],
        "streaming_vs_sliding_rows": [
            row for row in results["comparisons"] if row["section"] == "streaming_vs_sliding"
        ],
    }
    return results, overall_index, per_dataset_index


def format_ci(ci):
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def format_mean_ci(entry):
    return f"{entry['mean']:.2f} [{entry['ci95'][0]:.2f}, {entry['ci95'][1]:.2f}]"


def print_overall_means(overall_index):
    print("Overall 10-dataset mean ± 95% CI\n")
    header = f"{'Budget':<8} {'FullKV':>24} {'SnapKV':>24} {'PyramidKV':>24} {'StreamingLLM':>24}"
    print(header)
    print("-" * len(header))
    for budget in BUDGETS:
        row = [f"{budget:<8}"]
        for method in ALL_METHODS:
            entry = overall_index[budget].get(method)
            cell = "NA" if entry is None else format_mean_ci(entry)
            row.append(f"{cell:>24}")
        print(" ".join(row))
    print()


def print_per_dataset_means(per_dataset_index):
    print("Per-dataset means at 10% budget ± 95% CI\n")
    header = f"{'Dataset':<22} {'FullKV':>24} {'SnapKV':>24} {'PyramidKV':>24} {'StreamingLLM':>24}"
    print(header)
    print("-" * len(header))
    for dataset in DATASETS:
        row = [f"{dataset:<22}"]
        for method in ALL_METHODS:
            entry = per_dataset_index[dataset].get(method)
            cell = "NA" if entry is None else format_mean_ci(entry)
            row.append(f"{cell:>24}")
        print(" ".join(row))
    print()


def print_comparisons(results):
    print("Pairwise comparisons\n")
    header = f"{'Metric':<38} {'Method A':<22} {'Method B':<18} {'Mean diff':>10} {'95% CI':>24} {'Significant?':>13}"
    print(header)
    print("-" * len(header))
    for row in results["comparisons"]:
        print(
            f"{row['metric']:<38} {row['method_a']:<22} {row['method_b']:<18} "
            f"{row['mean_diff']:+10.2f} {format_ci(row['ci95']):>24} {str(row['significant']):>13}"
        )
    print()


def print_flags(results):
    print("Key flags\n")

    noisy_rows = results["flags"]["per_dataset_10pct_vs_fullkv_ci_includes_zero"]
    if noisy_rows:
        print("10% budget compressed vs FullKV differences whose CI includes zero:")
        for row in noisy_rows:
            print(
                f"  {row['metric']}: {row['method_a']} - {row['method_b']} = "
                f"{row['mean_diff']:+.2f} {format_ci(row['ci95'])}"
            )
    else:
        print("All 10% budget compressed vs FullKV dataset-level gaps exclude zero.")
    print()

    snap_overall = results["flags"]["snapkv_minus_pyramidkv_10pct_overall"]
    snap_active = results["flags"]["snapkv_minus_pyramidkv_10pct_compression_active"]
    if snap_overall is not None:
        print(
            "SnapKV - PyramidKV at 10% overall: "
            f"{snap_overall['mean_diff']:+.2f} {format_ci(snap_overall['ci95'])} "
            f"significant={snap_overall['significant']}"
        )
    if snap_active is not None:
        print(
            "SnapKV - PyramidKV at 10% compression-active: "
            f"{snap_active['mean_diff']:+.2f} {format_ci(snap_active['ci95'])} "
            f"significant={snap_active['significant']}"
        )
    print()

    print("lcc anomaly checks (compressed - FullKV):")
    for row in results["flags"]["lcc_anomaly_rows"]:
        print(
            f"  {row['metric']} {row['method_a']} - FullKV = "
            f"{row['mean_diff']:+.2f} {format_ci(row['ci95'])} significant={row['significant']}"
        )
    print()

    print("StreamingLLMSliding - StreamingLLM checks:")
    for row in results["flags"]["streaming_vs_sliding_rows"]:
        print(
            f"  {row['metric']}: {row['mean_diff']:+.2f} {format_ci(row['ci95'])} "
            f"significant={row['significant']}"
        )
    print()

    if WARNINGS:
        print("Warnings")
        for warning in WARNINGS:
            print(f"  {warning}")
        print()


def main():
    global RAW
    RAW = load_all_raw()
    results, overall_index, per_dataset_index = build_results()
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_overall_means(overall_index)
    print_per_dataset_means(per_dataset_index)
    print_comparisons(results)
    print_flags(results)
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
