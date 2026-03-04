#!/usr/bin/env bash
# =============================================================================
# 02_score.sh — Score raw LongBench predictions using the official evaluation
#
# Usage:
#   bash 02_score.sh
#
# Output:
#   results/scores/scores_<method>_<budget>.json   — per-dataset scores
#   results/scores/summary.json                    — aggregated results table
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/config.env"

SCORES_DIR="$RESULTS_DIR/scores"
mkdir -p "$SCORES_DIR"

echo "======================================================"
echo "  Scoring LongBench predictions"
echo "======================================================"

# ── Run the KVCache-Factory scorer ─────────────────────────────────────────────
# The repo ships eval.py (or it uses the THUDM eval); we call it per-budget dir.

for BUDGET_DIR in "$RESULTS_DIR"/budget_*/; do
    BUDGET_LABEL=$(basename "$BUDGET_DIR" | sed 's/budget_//')
    SCORE_OUT="$SCORES_DIR/scores_${BUDGET_LABEL}.json"

    if [[ -f "$SCORE_OUT" ]]; then
        echo "  [SKIP] budget=$BUDGET_LABEL already scored"
        continue
    fi

    echo "  Scoring budget=$BUDGET_LABEL ..."

    python3 - <<PYEOF
import os, json, re, string
from collections import Counter

# ────────────────── Scorer functions (from THUDM/LongBench) ──────────────────
def normalize_answer(s):
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t): return ' '.join(t.split())
    def remove_punc(t): return ''.join(ch for ch in t if ch not in string.punctuation)
    def lower(t): return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens  = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(truth_tokens)
    return 2 * p * r / (p + r)

def qa_f1(pred, golds):
    return max(f1_score(pred, g) for g in golds)

def exact_match(pred, golds):
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))

def rouge_l(pred, golds):
    try:
        from rouge import Rouge
        r = Rouge()
        scores = [r.get_scores(pred, g)[0]['rouge-l']['f'] if pred.strip() and g.strip() else 0.0
                  for g in golds]
        return max(scores)
    except Exception:
        return 0.0

# Dataset → metric mapping
DATASET_METRIC = {
    "narrativeqa":      qa_f1,
    "qasper":           qa_f1,
    "multifieldqa_en":  qa_f1,
    "hotpotqa":         qa_f1,
    "2wikimqa":         qa_f1,
    "musique":          qa_f1,
    "gov_report":       rouge_l,
    "qmsum":            rouge_l,
    "multi_news":       rouge_l,
    "trec":             exact_match,
    "triviaqa":         qa_f1,
    "samsum":           rouge_l,
    "passage_count":    exact_match,
    "passage_retrieval_en": exact_match,
    "lcc":              rouge_l,
    "repobench-p":      rouge_l,
}

DATASET_CATEGORY = {
    "narrativeqa":            "Single-Doc QA",
    "qasper":                 "Single-Doc QA",
    "multifieldqa_en":        "Single-Doc QA",
    "hotpotqa":               "Multi-Doc QA",
    "2wikimqa":               "Multi-Doc QA",
    "musique":                "Multi-Doc QA",
    "gov_report":             "Summarization",
    "qmsum":                  "Summarization",
    "multi_news":             "Summarization",
    "trec":                   "Few-Shot",
    "triviaqa":               "Few-Shot",
    "samsum":                 "Few-Shot",
    "passage_count":          "Synthetic",
    "passage_retrieval_en":   "Synthetic",
    "lcc":                    "Code",
    "repobench-p":            "Code",
}

budget_dir   = "${BUDGET_DIR}"
budget_label = "${BUDGET_LABEL}"
scores_dir   = "${SCORES_DIR}"

all_scores = {}

for fname in os.listdir(budget_dir):
    if not fname.endswith(".json"):
        continue

    # filename: <METHOD>_<dataset>.json
    parts = fname.replace(".json", "").split("_", 1)
    if len(parts) != 2:
        continue
    method, dataset = parts

    if dataset not in DATASET_METRIC:
        continue

    metric_fn = DATASET_METRIC[dataset]
    results_path = os.path.join(budget_dir, fname)

    scores_list = []
    with open(results_path) as f:
        for line in f:
            ex = json.loads(line)
            pred  = ex.get("pred", "")
            golds = ex.get("answers", [])
            if isinstance(golds, str):
                golds = [golds]
            scores_list.append(metric_fn(pred, golds))

    avg = sum(scores_list) / len(scores_list) * 100 if scores_list else 0.0

    if method not in all_scores:
        all_scores[method] = {}
    all_scores[method][dataset] = {
        "score":    round(avg, 2),
        "n":        len(scores_list),
        "category": DATASET_CATEGORY.get(dataset, "Other"),
    }
    print(f"  {method}/{dataset}: {avg:.2f} (n={len(scores_list)})")

# Compute category-level averages
for method in all_scores:
    cats = {}
    for ds, info in all_scores[method].items():
        cat = info["category"]
        cats.setdefault(cat, []).append(info["score"])
    all_scores[method]["_category_avg"] = {c: round(sum(vs)/len(vs), 2) for c, vs in cats.items()}
    avg_all = [info["score"] for ds, info in all_scores[method].items() if not ds.startswith("_")]
    all_scores[method]["_overall_avg"] = round(sum(avg_all)/len(avg_all), 2) if avg_all else 0.0

out_path = os.path.join(scores_dir, f"scores_{budget_label}.json")
with open(out_path, "w") as f:
    json.dump({"budget": budget_label, "results": all_scores}, f, indent=2)
print(f"\n  Written: {out_path}")
PYEOF

done

# ── Merge all budgets into a single summary.json ───────────────────────────────
echo ""
echo "  Merging into summary.json ..."

python3 - <<PYEOF
import os, json

scores_dir = "${SCORES_DIR}"
summary = {}

for fname in sorted(os.listdir(scores_dir)):
    if not fname.startswith("scores_") or not fname.endswith(".json"):
        continue
    with open(os.path.join(scores_dir, fname)) as f:
        data = json.load(f)
    budget = data["budget"]
    for method, results in data["results"].items():
        summary.setdefault(method, {})[budget] = {
            "overall": results.get("_overall_avg", 0.0),
            "categories": results.get("_category_avg", {}),
            "datasets":   {k: v for k, v in results.items() if not k.startswith("_")},
        }

with open(os.path.join(scores_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Written: summary.json")

# Print a quick ASCII table
BUDGETS  = sorted({b for m in summary.values() for b in m})
METHODS  = sorted(summary)
CATS     = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-Shot", "Synthetic", "Code"]

header = f"{'Method':<15}  {'Budget':<8}  {'Overall':>7}  " + "  ".join(f"{c[:12]:>12}" for c in CATS)
print("\n" + "="*len(header))
print(header)
print("="*len(header))
for method in METHODS:
    for budget in BUDGETS:
        info = summary[method].get(budget, {})
        overall = info.get("overall", "-")
        cat_avgs = [str(info.get("categories", {}).get(c, "-")) for c in CATS]
        row = f"{method:<15}  {budget:<8}  {overall:>7}  " + "  ".join(f"{v:>12}" for v in cat_avgs)
        print(row)
print("="*len(header))
PYEOF

echo ""
echo "======================================================"
echo "  Scoring complete!"
echo "  Next step: run scripts/03_speedup.sh"
echo "======================================================"
