#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.env not found. Run scripts/00_setup.sh first."
    exit 1
fi
source "$CONFIG_FILE"

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR/timing"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATASETS="2wikimqa,qmsum,repobench-p,gov_report"
LONG_BUDGETS="0.1,0.2,0.5"

print_cmd_with_redirect() {
    local log_file="$1"
    shift
    printf '%q ' "$@"
    printf '> %q 2>&1\n' "$log_file"
}

launch_longbench_job() {
    local gpu="$1"
    local method="$2"
    local budgets="$3"
    local log_file="$RESULTS_DIR/timing/launch_${method}_${TIMESTAMP}.log"

    local cmd=(
        bash "$SCRIPT_DIR/01_run_eval.sh"
        --gpu "$gpu"
        --methods "$method"
        --datasets "$DATASETS"
        --budgets "$budgets"
    )

    if [[ "$DRY_RUN" == true ]]; then
        print_cmd_with_redirect "$log_file" "${cmd[@]}"
        return
    fi

    (
        local start_ts end_ts elapsed
        start_ts=$(date +%s)
        "${cmd[@]}" > "$log_file" 2>&1
        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        echo "[$(date '+%F %T')] Completed ${method} LongBench on GPU ${gpu} in ${elapsed}s (log: ${log_file})"
    ) &
}

# GPUs 0-3: longbench (parallel)
launch_longbench_job 0 SnapKV "$LONG_BUDGETS"
launch_longbench_job 1 PyramidKV "$LONG_BUDGETS"
launch_longbench_job 2 StreamingLLM "$LONG_BUDGETS"
launch_longbench_job 3 FullKV "1.0"

if [[ "$DRY_RUN" == false ]]; then
    wait
fi
