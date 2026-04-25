# benchmark_llm_speed.py
# Tests tok/sec on Llama 3.3-70B Q4 with a realistic extraction prompt
# Run on Research PC (i7-14700K, RTX 4080 Super 16GB, 64GB RAM)

from llama_cpp import Llama
import time
import json

MODEL_PATH = r"C:\Users\T2530913\.lmstudio\models\lmstudio-community\Qwen3-14B-GGUF\Qwen3-14B-Q6_K.gguf" 

# --- Realistic extraction prompt (mirrors actual workload) ---
PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a power systems safety auditor. You verify extracted rules against source text.
Output ONLY a JSON array. No preamble.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Source text:
"According to IEEE Std 1547-2018 Section 7.4, the voltage at any point of common coupling
shall not exceed 1.05 pu or fall below 0.95 pu during normal operating conditions.
Any distributed energy resource that causes a voltage deviation beyond these limits
must be immediately disconnected from the grid. Additionally, all protection systems
must respond within 2 seconds of detecting a violation to prevent cascading failures.
Line loading shall not exceed 100% of the thermal limit under any N-1 contingency."

Extracted rules:
[
  {"rule_id": "R_001", "entity": "Bus", "condition": "voltage_pu > 1.05 OR voltage_pu < 0.95", "action": "DISCONNECT", "severity": "critical", "source": "IEEE Std 1547-2018, Section 7.4"},
  {"rule_id": "R_002", "entity": "ProtectionDevice", "condition": "violation_detected == True", "action": "RESPOND_WITHIN_2S", "severity": "critical", "source": "IEEE Std 1547-2018, Section 7.4"},
  {"rule_id": "R_003", "entity": "Line", "condition": "loading_pct > 100 AND contingency == N-1", "action": "BLOCK", "severity": "high", "source": "IEEE Std 1547-2018, Section 7.4"}
]

For each rule: confirm whether it is supported by the source text.
Output verdict as CONFIRM, REJECT, or CORRECT (with corrected_fields if CORRECT).
Output ONLY a JSON array.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

def run_benchmark(n_gpu_layers: int, max_tokens: int = 300):
    print(f"\n{'='*60}")
    print(f"Config: n_gpu_layers={n_gpu_layers}, max_tokens={max_tokens}")
    print(f"{'='*60}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=n_gpu_layers,   # -1 = offload as many as VRAM allows
        n_ctx=2048,
        verbose=True                  # shows layer offload split — keep True for diagnosis
    )

    # Warm-up (1 token) — excludes model load time from timing
    _ = llm(PROMPT, max_tokens=1)

    # Timed run
    t0 = time.perf_counter()
    output = llm(PROMPT, max_tokens=max_tokens, temperature=0.0)
    elapsed = time.perf_counter() - t0

    text = output["choices"][0]["text"]
    n_tokens = output["usage"]["completion_tokens"]
    tok_per_sec = n_tokens / elapsed

    print(f"\nTokens generated : {n_tokens}")
    print(f"Elapsed          : {elapsed:.1f}s")
    print(f"Speed            : {tok_per_sec:.2f} tok/sec")
    print(f"\nOutput:\n{text}")

    # Decision guidance
    print(f"\n--- Workability Assessment ---")
    if tok_per_sec >= 5:
        print("✓ WORKABLE — full extraction run is feasible overnight")
    elif tok_per_sec >= 2:
        print("⚠ SLOW but VIABLE — schedule as overnight batch job")
    else:
        print("✗ TOO SLOW — fall back to Qwen2.5-32B Q4 for validation pass")

    del llm
    return tok_per_sec

if __name__ == "__main__":
    # Test 1: Max GPU offload (most layers on RTX 4080 Super 16GB)
    speed = run_benchmark(n_gpu_layers=25, max_tokens=300)

    # Optionally test a reduced offload if -1 causes OOM
    # run_benchmark(n_gpu_layers=60, max_tokens=300)