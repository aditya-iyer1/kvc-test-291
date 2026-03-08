#!/usr/bin/env bash
# =============================================================================
# 03_speedup.sh — Measure prefill & decoding latency for each method/budget
#
# Usage:
#   bash 03_speedup.sh [--gpu 0] [--seq_len 4096]
#
# Output:
#   results/timing/latency_report.json   — structured timing data
#   results/timing/speedup_table.txt     — formatted speedup table
#
# This script generates a synthetic prompt of fixed length, runs a fixed
# number of decode steps under each method, and measures wall-clock time.
# We compare prefill + first-token latency relative to FullKV.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/config.env"

GPU="0"
# SEQ_LEN=4096      # synthetic prompt length (tokens)
SEQ_LEN=16384
# N_DECODE=50       # number of decode steps per run
N_DECODE=200
N_WARMUP=3        # warmup iterations before timing

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2";     shift 2 ;;
        --seq_len) SEQ_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMING_DIR="$RESULTS_DIR/timing"
mkdir -p "$TIMING_DIR"

echo "======================================================"
echo "  Latency Benchmark"
echo "======================================================"
echo "  GPU        : $GPU"
echo "  Seq len    : $SEQ_LEN tokens"
echo "  Decode steps: $N_DECODE"
echo "  AttnImpl   : $ATTN_IMPL"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" python3 - <<PYEOF
import os, json, time, torch, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_dir   = "${REPO_DIR}"
model_path = "${MODEL_PATH}"
attn_impl  = "${ATTN_IMPL}"
seq_len    = ${SEQ_LEN}
n_decode   = ${N_DECODE}
n_warmup   = ${N_WARMUP}
timing_dir = "${TIMING_DIR}"

import sys
sys.path.insert(0, repo_dir)

# ── Load tokenizer once ───────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.pad_token_id  = tokenizer.eos_token_id

# ── Build a fixed-length prompt ───────────────────────────────────────────────
# Repeat a token to fill the context window

# filler_token_id = tokenizer.encode("the", add_special_tokens=False)[0]
# input_ids = torch.tensor([[filler_token_id] * seq_len], dtype=torch.long).cuda()

# Random tokens for realistic attention sparsity (ADDED TO REPLACE ABOVE 2 LINES)
import random
random.seed(42)
vocab_size = tokenizer.vocab_size
input_ids = torch.tensor(
    [[random.randint(100, vocab_size - 100) for _ in range(seq_len)]],
    dtype=torch.long
).cuda()

METHODS  = ["FullKV", "StreamingLLM", "H2O", "SnapKV", "PyramidKV"]
BUDGETS  = [1.0, 0.5, 0.2, 0.1]
MODEL_MAX_LEN = 7950

results = {}

for method in METHODS:
    print(f"\n── Method: {method} ──")
    method_results = {}

    budget_list = [1.0] if method == "FullKV" else BUDGETS

    for budget in budget_list:
        max_cap = int(round(budget * MODEL_MAX_LEN))
        label   = f"{int(round(budget*100))}pct" if method != "FullKV" else "full"

        # ── Load fresh model + apply monkeypatch ──────────────────────────────
        from pyramidkv.monkeypatch import replace_llama, replace_mistral
        # Reset by reimporting (each method needs fresh patch)
        import importlib, pyramidkv.monkeypatch as mp_mod
        importlib.reload(mp_mod)
        mp_mod.replace_llama(method.lower())
        mp_mod.replace_mistral(method.lower())

        print(f"  Loading model for {method}/{label}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            attn_implementation=attn_impl,
        )
        model.eval()

        # Configure per-layer KV budgets
        if method != "FullKV":
            layers = len(model.model.layers)
            window = max_cap - 4 if method.lower() == "streamingllm" else 8
            for i in range(layers):
                cfg_attn = model.model.layers[i].self_attn.config
                cfg_attn.max_capacity_prompt = max_cap
                cfg_attn.window_size         = window
                cfg_attn.kernel_size         = 7
                cfg_attn.pooling             = "maxpool"

        # ── Warmup ────────────────────────────────────────────────────────────
        print(f"  Warmup ({n_warmup} iters)...", flush=True)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model.generate(
                    input_ids,
                    max_new_tokens=n_decode,
                    num_beams=1, do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
        torch.cuda.synchronize()

        # ── Timed runs (3 trials) ──────────────────────────────────────────────
        N_TRIALS = 3
        elapsed_list = []
        tprefill_list = []

        with torch.no_grad():
            for trial in range(N_TRIALS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Prefill only (pass through without generating)
                out_prefill = model(input_ids, use_cache=True)
                torch.cuda.synchronize()
                t_prefill = time.perf_counter() - t0

                # Full generate (prefill + decode)
                t_start = time.perf_counter()
                _ = model.generate(
                    input_ids,
                    max_new_tokens=n_decode,
                    num_beams=1, do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
                torch.cuda.synchronize()
                t_total = time.perf_counter() - t_start

                elapsed_list.append(t_total)
                tprefill_list.append(t_prefill)

        avg_total   = statistics.mean(elapsed_list)
        avg_prefill = statistics.mean(tprefill_list)
        avg_decode  = avg_total - avg_prefill

        method_results[label] = {
            "method":             method,
            "budget":             label,
            "max_capacity":       max_cap,
            "seq_len":            seq_len,
            "n_decode_steps":     n_decode,
            "prefill_ms":         round(avg_prefill * 1000, 1),
            "decode_ms":          round(avg_decode  * 1000, 1),
            "total_ms":           round(avg_total   * 1000, 1),
            "decode_ms_per_tok":  round(avg_decode / n_decode * 1000, 2),
        }

        print(f"  budget={label}: prefill={avg_prefill*1000:.1f}ms  decode={avg_decode*1000:.1f}ms  total={avg_total*1000:.1f}ms")

        del model
        torch.cuda.empty_cache()

    results[method] = method_results

# ── Compute speedup relative to FullKV ────────────────────────────────────────
fullkv_total = results["FullKV"]["full"]["total_ms"]

COLS    = ["method", "budget", "max_capacity", "prefill_ms", "decode_ms", "total_ms", "decode_ms_per_tok", "speedup_total"]
rows    = []
for method, bdict in results.items():
    for label, info in bdict.items():
        info["speedup_total"] = round(fullkv_total / info["total_ms"], 3)
        rows.append(info)

# Save JSON
out_path = os.path.join(timing_dir, "latency_report.json")
with open(out_path, "w") as f:
    json.dump({"fullkv_baseline_ms": fullkv_total, "rows": rows}, f, indent=2)
print(f"\nWritten: {out_path}")

# Print formatted table
header = f"{'Method':<14} {'Budget':<6} {'Cache':>6} {'Prefill':>9} {'Decode':>9} {'Total':>8} {'ms/tok':>7} {'Speedup':>8}"
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
for r in rows:
    print(f"{r['method']:<14} {r['budget']:<6} {r['max_capacity']:>6} "
          f"{r['prefill_ms']:>9} {r['decode_ms']:>9} {r['total_ms']:>8} "
          f"{r['decode_ms_per_tok']:>7} {r['speedup_total']:>8}x")
print("=" * len(header))

# Save table too
table_path = os.path.join(timing_dir, "speedup_table.txt")
with open(table_path, "w") as f:
    f.write("=" * len(header) + "\n" + header + "\n" + "=" * len(header) + "\n")
    for r in rows:
        f.write(f"{r['method']:<14} {r['budget']:<6} {r['max_capacity']:>6} "
                f"{r['prefill_ms']:>9} {r['decode_ms']:>9} {r['total_ms']:>8} "
                f"{r['decode_ms_per_tok']:>7} {r['speedup_total']:>8}x\n")
    f.write("=" * len(header) + "\n")
print(f"Written: {table_path}")
PYEOF

echo ""
echo "======================================================"
echo "  Latency benchmark complete!"
echo "  Next step: run scripts/04_plot.py"
echo "======================================================"
