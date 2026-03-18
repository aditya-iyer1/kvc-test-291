#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.env not found."
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

mkdir -p "$RESULTS_DIR/timing" "$RESULTS_DIR/ruler"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONTEXTS=(4096 8192 16384)

cap_for_method_context() {
    local method="$1"
    local ctx="$2"
    if [[ "$method" == "FullKV" ]]; then
        echo 31500
        return
    fi
    case "$ctx" in
        4096) echo 410 ;;
        8192) echo 819 ;;
        16384) echo 1638 ;;
        *) echo "ERROR: unsupported context length $ctx" >&2; return 1 ;;
    esac
}

print_ctx_cmd() {
    local gpu="$1"
    local method="$2"
    local ctx="$3"
    local cap="$4"
    printf 'CUDA_VISIBLE_DEVICES=%q python3 run_ruler.py [patched context_length_list=%q] --method %q --model_path %q --max_capacity_prompts %q --attn_implementation %q --save_dir %q --use_cache True\n' \
        "$gpu" "$ctx" "$method" "$MODEL_PATH" "$cap" "$ATTN_IMPL" "$RESULTS_DIR/ruler/"
}

run_one_context() {
    local gpu="$1"
    local method="$2"
    local ctx="$3"
    local cap="$4"

    local tmp_py
    tmp_py="$(mktemp "$REPO_DIR/.tmp_run_ruler_${method}_${ctx}_XXXX.py")"

    sed "s/^context_length_list = .*/context_length_list = [${ctx}]/" "$REPO_DIR/run_ruler.py" > "$tmp_py"

    (
        cd "$REPO_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" python3 "$tmp_py" \
            --method "$method" \
            --model_path "$MODEL_PATH" \
            --max_capacity_prompts "$cap" \
            --attn_implementation "$ATTN_IMPL" \
            --save_dir "$RESULTS_DIR/ruler/" \
            --use_cache True
    )

    rm -f "$tmp_py"
}

launch_method_job() {
    local gpu="$1"
    local method="$2"
    local log_file="$RESULTS_DIR/timing/ruler_${method}_${TIMESTAMP}.log"

    if [[ "$DRY_RUN" == true ]]; then
        echo "# $method (GPU $gpu) -> $log_file"
        for ctx in "${CONTEXTS[@]}"; do
            local cap
            cap="$(cap_for_method_context "$method" "$ctx")"
            print_ctx_cmd "$gpu" "$method" "$ctx" "$cap"
        done
        return
    fi

    (
        local start_ts end_ts elapsed
        start_ts="$(date +%s)"
        echo "[$(date '+%F %T')] Start ${method} on GPU ${gpu}"
        for ctx in "${CONTEXTS[@]}"; do
            cap="$(cap_for_method_context "$method" "$ctx")"
            echo "[$(date '+%F %T')] Running ctx=${ctx}, cap=${cap}, method=${method}"
            run_one_context "$gpu" "$method" "$ctx" "$cap"
        done
        end_ts="$(date +%s)"
        elapsed=$(( end_ts - start_ts ))
        echo "[$(date '+%F %T')] Completed ${method} on GPU ${gpu} in ${elapsed}s"
    ) > "$log_file" 2>&1 &
}

launch_method_job 0 SnapKV
launch_method_job 1 PyramidKV
launch_method_job 2 StreamingLLM
launch_method_job 3 FullKV

if [[ "$DRY_RUN" == false ]]; then
    wait
fi
