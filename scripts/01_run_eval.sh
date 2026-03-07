#!/usr/bin/env bash
# =============================================================================
# 01_run_eval.sh — Run all KV cache compression experiments on LongBench
#
# Usage:
#   bash 01_run_eval.sh [--gpu 0] [--methods all] [--budgets all]
#
# Examples:
#   bash 01_run_eval.sh                          # All methods, all budgets
#   bash 01_run_eval.sh --methods PyramidKV,SnapKV --budgets 0.1,0.5
#   bash 01_run_eval.sh --gpu 0,1                # Multi-GPU
#
# Output:
#   results/<model>_<budget_int>/  per method JSON files
#   results/timing/                per-method per-budget timing logs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load config ────────────────────────────────────────────────────────────────
CONFIG_FILE="$SCRIPT_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.env not found. Run scripts/00_setup.sh first."
    exit 1
fi
source "$CONFIG_FILE"

# ── Defaults ───────────────────────────────────────────────────────────────────
GPUS="0"
METHODS_STR="FullKV,StreamingLLM,H2O,SnapKV,PyramidKV"   # methods to run
BUDGETS_STR="0.1,0.2,0.5"                                  # cache budget ratios

# ── Parse CLI overrides ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPUS="$2";       shift 2 ;;
        --methods)  METHODS_STR="$2"; shift 2 ;;
        --budgets)  BUDGETS_STR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra METHODS <<< "$METHODS_STR"
IFS=',' read -ra BUDGETS <<< "$BUDGETS_STR"

# Llama-3-8B has a model_max_len of 7950 in KVCache-Factory (see run_longbench.py)
MODEL_MAX_LEN=7950

# ── LongBench datasets ─────────────────────────────────────────────────────────
DATASETS=(
    triviaqa hotpotqa gov_report
)

mkdir -p "$RESULTS_DIR/timing"

echo "======================================================"
echo "  KV Cache Compression — LongBench Evaluation"
echo "======================================================"
echo "Model      : $MODEL_PATH"
echo "GPU(s)     : $GPUS"
echo "Methods    : ${METHODS[*]}"
echo "Budgets    : ${BUDGETS[*]}"
echo "AttnImpl   : $ATTN_IMPL"
echo "Results    : $RESULTS_DIR"
echo ""

# ── Helper: compute absolute capacity from ratio ───────────────────────────────
ratio_to_abs() {
    local ratio="$1"
    python3 -c "print(int(round($ratio * $MODEL_MAX_LEN)))"
}

# ── Helper: budget ratio → readable label (e.g. 0.1 → 10pct) ─────────────────
ratio_to_label() {
    python3 -c "v=$1; print(f'{int(round(v*100))}pct')"
}

# ── Main loop ──────────────────────────────────────────────────────────────────
for METHOD in "${METHODS[@]}"; do
    echo "==============================="
    echo "  Method: $METHOD"
    echo "==============================="

    # FullKV uses max capacity = full context (no compression)
    if [[ "$METHOD" == "FullKV" ]]; then
        BUDGET_LIST=("1.0")   # full context
    else
        BUDGET_LIST=("${BUDGETS[@]}")
    fi

    for BUDGET in "${BUDGET_LIST[@]}"; do

        ABS_CAP=$(ratio_to_abs "$BUDGET")
        LABEL=$(ratio_to_label "$BUDGET")

        if [[ "$METHOD" == "FullKV" ]]; then
            LABEL="full"
            ABS_CAP=-1   # signal to run_longbench.py to use full context
        fi

        echo ""
        echo ""
        echo "  Budget: $LABEL  (max_capacity_prompts=$ABS_CAP)"

        OUT_DIR="$RESULTS_DIR/budget_${LABEL}"
        mkdir -p "$OUT_DIR"
        TIMING_LOG="$RESULTS_DIR/timing/${METHOD}_${LABEL}.txt"

        # Check if all datasets already done
        ALL_DONE=true
        for DATASET in "${DATASETS[@]}"; do
            OUT_FILE="$OUT_DIR/${METHOD}_${DATASET}.json"
            if [[ ! -f "$OUT_FILE" ]]; then
                ALL_DONE=false
                break
            fi
        done
        if [[ "$ALL_DONE" == "true" ]]; then
            echo "    [SKIP] all datasets for $METHOD/$LABEL already exist"
            continue
        fi

        if [[ "$METHOD" == "FullKV" ]]; then
            CAP_ARG="--max_capacity_prompts 7950"
        else
            CAP_ARG="--max_capacity_prompts_ratio $BUDGET"
        fi

        echo -n "    Running all datasets ... "
        START_TS=$(date +%s)

        CMD=(
            python3 "$REPO_DIR/run_longbench.py"
                --model_path  "$MODEL_PATH"
                --method      "$METHOD"
                --attn_implementation "$ATTN_IMPL"
                $CAP_ARG
                --save_dir    "$OUT_DIR"
                --use_cache   True
                --eval_batch_size 4
                --max_num_examples 200
                --seed        42
        )

        CUDA_VISIBLE_DEVICES="$GPUS" \
            "${CMD[@]}" \
            > "$TIMING_LOG" 2>&1

        END_TS=$(date +%s)
        ELAPSED=$(( END_TS - START_TS ))

        # Copy results to flat structure
        MODEL_NAME=$(basename "$MODEL_PATH")
        for DATASET in "${DATASETS[@]}"; do
            GENERATED=$(find "$OUT_DIR" -name "${METHOD}.json" -path "*${DATASET}*" | head -1 || true)
            if [[ -n "$GENERATED" ]]; then
                cp "$GENERATED" "$OUT_DIR/${METHOD}_${DATASET}.json"
            fi
        done

        echo "done (${ELAPSED}s)"
        echo "${ELAPSED}" >> "$TIMING_LOG"
    done       # BUDGET loop
done           # METHOD loop

echo ""
echo "======================================================"
echo "  All evaluations complete."
echo "  Next step: run scripts/02_score.sh"
echo "======================================================"
