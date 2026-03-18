#!/usr/bin/env python3
"""
diagnose_truncation.py
======================
Checks every possible source of truncation in the run7 results.

Run on the cluster:
    python3 diagnose_truncation.py

Checks performed:
  1. Shell script bug: line 116 of 01_run_eval.sh — FullKV cap hardcoded at 7950
  2. run_longbench.py: how max_capacity_prompts is used, what MODEL_MAX_LEN is set to
  3. Per-example token lengths in each raw prediction JSON vs the 7950 threshold
  4. Whether any FullKV examples show truncation evidence (prompt length field)
  5. Summary of how many examples per dataset exceed 7950 tokens
"""

import os, json, sys
from pathlib import Path
from collections import defaultdict

# ── Config — adjust if needed ─────────────────────────────────────────────────
RESULTS_DIR   = Path(os.path.expanduser("~/kvc-test-291/results_mistral_run7"))
REPO_DIR      = Path(os.path.expanduser("~/kvc-test-291/KVCache-Factory"))
LONGBENCH_DIR = REPO_DIR / "data" / "LongBench"
BUDGET_FULL   = RESULTS_DIR / "budget_full"

THRESHOLD = 7950   # the hardcoded cap that may have caused truncation

DATASETS = [
    "narrativeqa",
    "hotpotqa",
    "multi_news",
    "triviaqa",
    "passage_retrieval_en",
    "lcc",
]

SEP = "=" * 70

# ── Helper ────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

# ── Check 1: Shell script line 116 ───────────────────────────────────────────
print(SEP)
print("CHECK 1: Shell script 01_run_eval.sh — FullKV cap argument")
print(SEP)

script_path = Path(os.path.expanduser("~/kvc-test-291/scripts/01_run_eval.sh"))
if script_path.exists():
    lines = script_path.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if "max_capacity_prompts" in line and "FullKV" not in line:
            print(f"  Line {i:3d}: {line.rstrip()}")
        if "FullKV" in line and ("CAP_ARG" in line or "max_capacity" in line):
            flag = "  *** BUG PRESENT ***" if "7950" in line else "  (looks ok)"
            print(f"  Line {i:3d}: {line.rstrip()} {flag}")
    # Also find MODEL_MAX_LEN
    for i, line in enumerate(lines, 1):
        if "MODEL_MAX_LEN" in line:
            print(f"  Line {i:3d}: {line.rstrip()}")
else:
    print(f"  WARN: script not found at {script_path}")

# ── Check 2: run_longbench.py — how cap is used ───────────────────────────────
print()
print(SEP)
print("CHECK 2: run_longbench.py — max_capacity_prompts and MODEL_MAX_LEN handling")
print(SEP)

rlb_path = REPO_DIR / "run_longbench.py"
if rlb_path.exists():
    lines = rlb_path.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if any(kw in line for kw in [
            "max_capacity_prompts", "model_max_len", "MODEL_MAX_LEN",
            "max_length", "truncat", "7950", "31500", "31000"
        ]):
            print(f"  Line {i:3d}: {line.rstrip()}")
else:
    print(f"  WARN: run_longbench.py not found at {rlb_path}")

# ── Check 3: LongBench source data — token length distribution ───────────────
print()
print(SEP)
print("CHECK 3: LongBench source data — examples exceeding 7950 tokens (by 'length' field)")
print(SEP)

grand_total   = 0
grand_over    = 0

for ds in DATASETS:
    src = LONGBENCH_DIR / f"{ds}.jsonl"
    if not src.exists():
        print(f"  [{ds}] SOURCE NOT FOUND: {src}")
        continue

    items    = load_jsonl(src)
    lengths  = [ex.get("length", 0) for ex in items]
    over     = [l for l in lengths if l > THRESHOLD]
    pct      = 100 * len(over) / len(lengths) if lengths else 0
    max_len  = max(lengths) if lengths else 0
    mean_len = sum(lengths) / len(lengths) if lengths else 0

    grand_total += len(lengths)
    grand_over  += len(over)

    flag = "  *** AFFECTED ***" if over else ""
    print(f"  [{ds}]  n={len(lengths)}  mean={mean_len:.0f}  max={max_len}  "
          f"over_7950={len(over)} ({pct:.1f}%){flag}")

    if over:
        buckets = defaultdict(int)
        for l in over:
            bucket = (l // 5000) * 5
            buckets[bucket] += 1
        for b in sorted(buckets):
            print(f"        {b}K–{b+5}K tokens: {buckets[b]} examples")

print(f"\n  TOTAL: {grand_over}/{grand_total} examples exceed {THRESHOLD} tokens "
      f"({100*grand_over/grand_total:.1f}%)")

# ── Check 4: FullKV prediction files — inspect 'length' or context field ──────
print()
print(SEP)
print("CHECK 4: FullKV prediction JSONs — checking for truncation evidence")
print(SEP)

if not BUDGET_FULL.exists():
    print(f"  WARN: {BUDGET_FULL} does not exist")
else:
    for ds in DATASETS:
        pred_path = BUDGET_FULL / f"FullKV_{ds}.json"
        if not pred_path.exists():
            print(f"  [{ds}] prediction file NOT FOUND: {pred_path}")
            continue

        preds = load_jsonl(pred_path)

        # Check if predictions have a 'length' or 'input_len' field
        lengths_in_pred = [p.get("length", p.get("input_len", None)) for p in preds]
        lengths_in_pred = [l for l in lengths_in_pred if l is not None]

        # Also load source to compare
        src = LONGBENCH_DIR / f"{ds}.jsonl"
        src_items = load_jsonl(src) if src.exists() else []
        src_lengths = {i: ex.get("length", 0) for i, ex in enumerate(src_items)}

        n_preds = len(preds)
        n_src   = len(src_items)

        print(f"\n  [{ds}]  predictions={n_preds}  source_examples={n_src}")

        if lengths_in_pred:
            over_in_pred = [l for l in lengths_in_pred if l > THRESHOLD]
            print(f"    'length' field present in predictions: yes")
            print(f"    Predictions with length > {THRESHOLD}: {len(over_in_pred)}/{len(lengths_in_pred)}")
        else:
            print(f"    'length' field NOT present in predictions — cannot directly verify")

        # Check if pred count matches capped source
        src_over = [i for i, l in src_lengths.items() if l > THRESHOLD]
        if src_over:
            print(f"    Source examples over threshold: {len(src_over)} — "
                  f"these may have been seen with truncated prompts by FullKV")

        # Check if predictions look suspiciously short (possible sign of bad truncation)
        pred_lengths = [len(p.get("pred", "")) for p in preds]
        empty_preds  = sum(1 for l in pred_lengths if l == 0)
        very_short   = sum(1 for l in pred_lengths if 0 < l < 5)
        if empty_preds or very_short:
            print(f"    *** WARNING: {empty_preds} empty predictions, "
                  f"{very_short} predictions under 5 chars — possible truncation artifact ***")
        else:
            print(f"    Prediction lengths look ok (no empty/trivially short outputs)")

# ── Check 5: config.env — what RESULTS_DIR and MODEL_PATH are set to ──────────
print()
print(SEP)
print("CHECK 5: config.env — active configuration")
print(SEP)

config_path = Path(os.path.expanduser("~/kvc-test-291/scripts/config.env"))
if config_path.exists():
    print(config_path.read_text())
else:
    print(f"  WARN: config.env not found at {config_path}")

# ── Check 6: run7 timing logs — grep for actual cap used ──────────────────────
print()
print(SEP)
print("CHECK 6: Timing logs — actual max_capacity_prompts passed to run_longbench.py")
print(SEP)

timing_dir = RESULTS_DIR / "timing"
if timing_dir.exists():
    fullkv_logs = sorted(timing_dir.glob("FullKV_full_*.txt"))
    if not fullkv_logs:
        print("  No FullKV timing logs found.")
    for log in fullkv_logs:
        lines = log.read_text().splitlines()
        cap_lines = [l for l in lines if "max_capacity_prompts" in l or "capacity" in l.lower()]
        print(f"\n  [{log.name}]")
        if cap_lines:
            for l in cap_lines[:5]:
                print(f"    {l.strip()}")
        else:
            # Print first few lines for context
            print(f"    (no capacity mention found; first 3 lines:)")
            for l in lines[:3]:
                print(f"    {l.strip()}")
else:
    print(f"  WARN: timing dir not found at {timing_dir}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY")
print(SEP)
print("""
Things to verify manually from the output above:

  1. CHECK 1: Does line 116 of 01_run_eval.sh still say '7950'?
     If YES → the FullKV run used cap=7950, not full context.

  2. CHECK 2: Does run_longbench.py truncate input to MODEL_MAX_LEN (7950)?
     If it truncates BEFORE applying max_capacity_prompts, then even FullKV
     with cap=31500 would be truncated at the tokenization step.

  3. CHECK 3: Which datasets have examples > 7950 tokens?
     These are the ones where FullKV truncation materially affects scores.

  4. CHECK 4: Any empty or suspiciously short FullKV predictions?
     These are a strong signal of truncation-induced generation failure.

  5. CHECK 6: What cap value appears in the actual timing logs?
     This is ground truth — what was actually passed to the script.
""")