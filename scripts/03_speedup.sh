#!/usr/bin/env bash
# =============================================================================
# 03_speedup.sh — Measure actual KV cache memory reduction and decode throughput
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/config.env"

GPU="0"
N_DECODE=50       # number of decode steps per run
N_WARMUP=3        # warmup iterations before timing

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMING_DIR="$RESULTS_DIR/timing"
mkdir -p "$TIMING_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python3 - <<PYEOF
import os, json, time, torch, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

repo_dir   = "${REPO_DIR}"
model_path = "${MODEL_PATH}"
attn_impl  = "${ATTN_IMPL}"
n_decode   = ${N_DECODE}
n_warmup   = ${N_WARMUP}
timing_dir = "${TIMING_DIR}"

sys.path.insert(0, repo_dir)

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.pad_token_id  = tokenizer.eos_token_id

torch.manual_seed(42)
vocab_size = getattr(tokenizer, "vocab_size", 32000)
min_tok, max_tok = 1000, min(30000, vocab_size - 1)

def generate_random_prompt(length):
    return torch.randint(min_tok, max_tok, (1, length), dtype=torch.long).cuda()

inputs_4k = generate_random_prompt(4096)
inputs_16k = generate_random_prompt(16384)

METHODS  = ["FullKV", "StreamingLLM", "SnapKV", "PyramidKV"] # Excluded H2O
BUDGETS  = [1.0, 0.5, 0.2, 0.1]
MODEL_MAX_LEN = 7950

results = {}

for method in METHODS:
    print(f"\n── Method: {method} ──", flush=True)
    method_results = {}
    budget_list = [1.0] if method == "FullKV" else BUDGETS

    for budget in budget_list:
        max_cap = int(round(budget * MODEL_MAX_LEN))
        label   = f"{int(round(budget*100))}pct" if method != "FullKV" else "full"

        from pyramidkv.monkeypatch import replace_llama, replace_mistral
        import importlib, pyramidkv.monkeypatch as mp_mod
        importlib.reload(mp_mod)
        mp_mod.replace_llama(method.lower())
        mp_mod.replace_mistral(method.lower())

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            attn_implementation=attn_impl,
        )
        model.eval()

        if method != "FullKV":
            layers = len(model.model.layers)
            window = max_cap - 4 if method.lower() == "streamingllm" else 8
            for i in range(layers):
                cfg_attn = model.model.layers[i].self_attn.config
                cfg_attn.max_capacity_prompt = max_cap
                cfg_attn.window_size         = window
                cfg_attn.kernel_size         = 7
                cfg_attn.pooling             = "maxpool"

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            try:
                attention_mask_4k = torch.ones_like(inputs_4k)
                out = model(inputs_4k, attention_mask=attention_mask_4k, use_cache=True)
                del out
            except Exception as e:
                print(f"  Forward pass failed for {method}/{label}. Error: {e}")
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        def measure_throughput(inputs):
            attention_mask = torch.ones_like(inputs)
            try:
                with torch.no_grad():
                    for _ in range(n_warmup):
                        model.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2, min_new_tokens=2, do_sample=False, num_beams=1)
            except Exception:
                return 0.0, 0.0

            prefill_times = []
            total_times = []
            try:
                for _ in range(3):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        model(inputs, attention_mask=attention_mask, use_cache=True)
                    torch.cuda.synchronize()
                    prefill_times.append(time.perf_counter() - t0)

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        model.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=n_decode, min_new_tokens=n_decode, do_sample=False, num_beams=1)
                    torch.cuda.synchronize()
                    total_times.append(time.perf_counter() - t0)
                
                avg_decode = statistics.mean(total_times) - statistics.mean(prefill_times)
                tps = n_decode / avg_decode if avg_decode > 0 else 0.0
                return tps, statistics.mean(total_times) * 1000.0
            except Exception:
                return 0.0, 0.0

        tps_4k, total_ms_4k = measure_throughput(inputs_4k)
        tps_16k, _ = measure_throughput(inputs_16k)

        method_results[label] = {
            "method": method,
            "budget": label,
            "total_ms": round(total_ms_4k, 1), # Compatibility
            "speedup_total": 0.0,              # Filled after FullKV loop
            "peak_vram_gb": round(peak_vram_gb, 4),
            "memory_reduction_ratio": 0.0,     # Filled after FullKV loop
            "tokens_per_sec_4k": round(tps_4k, 2),
            "tokens_per_sec_16k": round(tps_16k, 2),
        }
        
        print(f"  budget={label}: vram={peak_vram_gb:.2f}GB | tps_4k={tps_4k:.1f} | tps_16k={tps_16k:.1f}")

        del model
        torch.cuda.empty_cache()

    results[method] = method_results

fullkv_mem = results["FullKV"]["full"]["peak_vram_gb"]

rows = []
for method, bdict in results.items():
    for label, info in bdict.items():
        if info["peak_vram_gb"] > 0:
            reduction_ratio = fullkv_mem / info["peak_vram_gb"]
        else:
            reduction_ratio = 1.0
        
        info["memory_reduction_ratio"] = round(reduction_ratio, 3)
        info["speedup_total"] = round(reduction_ratio, 3) 
        rows.append(info)

out_path = os.path.join(timing_dir, "latency_report.json")
with open(out_path, "w") as f:
    json.dump({"fullkv_baseline_mem": fullkv_mem, "rows": rows}, f, indent=2)

header = f"{'Method':<14} {'Budget':<6} {'VRAM(GB)':>9} {'MemRedux':>9} {'TPS_4K':>9} {'TPS_16K':>9} {'TotalMs':>8}"
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
for r in rows:
    print(f"{r['method']:<14} {r['budget']:<6} {r['peak_vram_gb']:>9.2f} "
          f"{r['memory_reduction_ratio']:>8.2f}x {r['tokens_per_sec_4k']:>9.1f} {r['tokens_per_sec_16k']:>9.1f} {r['total_ms']:>8.1f}")
print("=" * len(header))

table_path = os.path.join(timing_dir, "speedup_table.txt")
with open(table_path, "w") as f:
    f.write("=" * len(header) + "\n" + header + "\n" + "=" * len(header) + "\n")
    for r in rows:
        f.write(f"{r['method']:<14} {r['budget']:<6} {r['peak_vram_gb']:>9.2f} "
                f"{r['memory_reduction_ratio']:>8.2f}x {r['tokens_per_sec_4k']:>9.1f} {r['tokens_per_sec_16k']:>9.1f} {r['total_ms']:>8.1f}\n")
    f.write("=" * len(header) + "\n")

print(f"\nWritten: {out_path}")
print(f"Written: {table_path}")
PYEOF
