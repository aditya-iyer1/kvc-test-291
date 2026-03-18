#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.env not found. Run scripts/00_setup.sh first."
    exit 1
fi
source "$CONFIG_FILE"

DATASETS=(2wikimqa qmsum repobench-p gov_report)
BUDGETS=(10pct 20pct 50pct)
METHODS=(SnapKV PyramidKV StreamingLLM)
NEEDLE_RESULTS_DIR="$REPO_DIR/results_needle/results"

echo "=== Status Snapshot ($(date '+%F %T %Z')) ==="
echo

echo "[GPU] nvidia-smi (memory used/free MiB per GPU)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo

echo "[LongBench Flat Files] PRESENT/MISSING matrix"
for ds in "${DATASETS[@]}"; do
    echo "Dataset: $ds"
    printf '%-14s %-8s %-8s %-8s\n' "Method" "10pct" "20pct" "50pct"
    for m in "${METHODS[@]}"; do
        statuses=()
        for b in "${BUDGETS[@]}"; do
            p="$RESULTS_DIR/budget_${b}/${m}_${ds}.json"
            if [[ -f "$p" ]]; then
                statuses+=("PRESENT")
            else
                statuses+=("MISSING")
            fi
        done
        printf '%-14s %-8s %-8s %-8s\n' "$m" "${statuses[0]}" "${statuses[1]}" "${statuses[2]}"
    done
    echo

done

echo "[Needle] Files newer than 1 hour in $NEEDLE_RESULTS_DIR"
if [[ -d "$NEEDLE_RESULTS_DIR" ]]; then
    if ! find "$NEEDLE_RESULTS_DIR" -type f -mmin -60 -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort; then
        true
    fi
else
    echo "Directory not found"
fi
echo

echo "[RULER] Output file counts by method and context length"
RULER_DIR="$RESULTS_DIR/ruler"
RULER_METHODS=(SnapKV PyramidKV StreamingLLM FullKV)
RULER_CONTEXTS=(4096 8192 16384)
if [[ -d "$RULER_DIR" ]]; then
    printf '%-14s %-8s %-8s %-8s\n' "Method" "4096" "8192" "16384"
    for m in "${RULER_METHODS[@]}"; do
        counts=()
        for ctx in "${RULER_CONTEXTS[@]}"; do
            c=$(find "$RULER_DIR" -type f -path "*/${ctx}/*/${m}.json" | wc -l)
            counts+=("$c")
        done
        printf '%-14s %-8s %-8s %-8s\n' "$m" "${counts[0]}" "${counts[1]}" "${counts[2]}"
    done
else
    echo "Directory not found: $RULER_DIR"
fi
echo

echo "[Needle run8 logs] tail -n 3 of each needle_*_run8.log"
needle_logs=("$RESULTS_DIR"/timing/needle_*_run8.log)
if [[ ${#needle_logs[@]} -gt 0 && -e "${needle_logs[0]}" ]]; then
    for f in "${needle_logs[@]}"; do
        echo "--- $f ---"
        tail -n 3 "$f"
        echo
    done
else
    echo "No matching logs found in $RESULTS_DIR/timing"
fi
echo

echo "[Active Launch Logs] tail -n 5"
active_methods=()
while IFS= read -r line; do
    method=$(sed -n 's/.*--methods \([^ ]*\).*/\1/p' <<< "$line")
    if [[ -n "$method" ]]; then
        active_methods+=("$method")
    fi
done < <(pgrep -af "$SCRIPT_DIR/01_run_eval.sh" || true)

if [[ ${#active_methods[@]} -eq 0 ]]; then
    echo "No active launch jobs detected."
else
    readarray -t uniq_methods < <(printf '%s\n' "${active_methods[@]}" | sort -u)
    for method in "${uniq_methods[@]}"; do
        log_file=$(ls -1t "$RESULTS_DIR"/timing/launch_${method}_*.log 2>/dev/null | head -n 1 || true)
        if [[ -n "$log_file" ]]; then
            echo "--- $method :: $log_file ---"
            tail -n 5 "$log_file"
        else
            echo "--- $method :: no launch log found ---"
        fi
        echo
    done
fi
