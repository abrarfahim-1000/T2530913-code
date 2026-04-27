# benchmark_llm_speed.py
# Tests tok/sec via Ollama with a realistic extraction prompt
# Run on Research PC (i7-14700K, RTX 4080 Super 16GB, 64GB RAM)
#
# Models tested:
#   Extractor : qwen3:14b
#   Validator : nemotron-mini (30b-a3b) — add when ready
#
# Usage:
#   ollama pull qwen3:14b
#   python bench.py

import time
import json
import ollama

EXTRACTOR_MODEL = "qwen3:14b"
VALIDATOR_MODEL = "nemotron-3-nano"   # placeholder — swap in when pulled

# Realistic extraction prompt (mirrors actual workload)
# /no_think disables Qwen3's chain-of-thought mode — essential for clean JSON output
EXTRACT_PROMPT = """/no_think
You are a power systems engineer extracting safety rules from grid documentation.
From the text below, extract ALL operational constraints as a JSON array.
Each rule MUST have exactly these keys: rule_id, source, entity, condition, action, severity, explanation.
severity must be one of: critical, high, medium, low.
If no rule is present, return an empty array [].
Output ONLY a valid JSON array. No preamble, no explanation, no markdown.

Text:
"According to IEEE Std 1547-2018 Section 7.4, the voltage at any point of common coupling
shall not exceed 1.05 pu or fall below 0.95 pu during normal operating conditions.
Any distributed energy resource that causes a voltage deviation beyond these limits
must be immediately disconnected from the grid. Additionally, all protection systems
must respond within 2 seconds of detecting a violation to prevent cascading failures.
Line loading shall not exceed 100% of the thermal limit under any N-1 contingency."
"""

VALIDATE_PROMPT = """/no_think
You are a power systems safety auditor. You verify extracted rules against source text.
Output ONLY a JSON array. No preamble.

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
Output ONLY a JSON array.
"""


def strip_think_tags(text: str) -> str:
    """
    Qwen3 sometimes emits <think>...</think> even with /no_think.
    Strip it so JSON parsing doesn't break.
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_json_array(text: str) -> list:
    """Robustly extract a JSON array from LLM output."""
    text = strip_think_tags(text)
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found. Raw output:\n{text[:300]}")
    return json.loads(text[start:end+1])


def run_benchmark(model: str, prompt: str, label: str):
    print(f"\n{'='*60}")
    print(f"Model  : {model}")
    print(f"Task   : {label}")
    print(f"{'='*60}")

    # Check model is available locally
    try:
        models = [m.model for m in ollama.list().models]
        if not any(model in m for m in models):
            print(f"✗ Model '{model}' not found. Run: ollama pull {model}")
            return None
    except Exception as e:
        print(f"✗ Could not connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return None

    options = ollama.Options(
        temperature=0.0,    # deterministic — important for benchmarking
        num_predict=2048,   # enough for a full rule set to complete
        num_ctx=4096,
    )

    # Warm-up pass — excludes model load time from timing
    print("Running warm-up (1 token)...")
    ollama.generate(model=model, prompt=prompt, options=ollama.Options(num_predict=1), stream=False)

    # Timed run
    print("Running timed benchmark...")
    t0 = time.perf_counter()
    resp = ollama.generate(model=model, prompt=prompt, options=options, stream=False)
    elapsed = time.perf_counter() - t0

    raw_text  = resp["response"]
    n_tokens  = resp.get("eval_count", 0)       # output tokens generated
    tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0

    print(f"\nTokens generated : {n_tokens}")
    print(f"Elapsed          : {elapsed:.1f}s")
    print(f"Speed            : {tok_per_sec:.2f} tok/sec")

    # Show raw output
    print(f"\n--- Raw Output ---")
    print(raw_text[:2000])

    # Attempt JSON parse
    print(f"\n--- JSON Parse ---")
    try:
        parsed = parse_json_array(raw_text)
        print(f"✓ Valid JSON array — {len(parsed)} item(s)")
        print(json.dumps(parsed, indent=2)[:2000])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"✗ JSON parse failed: {e}")

    # Workability assessment for thesis extraction workload
    print(f"\n--- Workability for Thesis Extraction ---")
    if tok_per_sec >= 20:
        print(f"✓ FAST ({tok_per_sec:.1f} tok/s) — extraction run completes in hours")
    elif tok_per_sec >= 5:
        print(f"⚠ MODERATE ({tok_per_sec:.1f} tok/s) — schedule as overnight batch")
    elif tok_per_sec >= 2:
        print(f"⚠ SLOW ({tok_per_sec:.1f} tok/s) — viable but will take 12–24h for full doc set")
    else:
        print(f"✗ TOO SLOW ({tok_per_sec:.1f} tok/s) — fall back to smaller model")

    return tok_per_sec


if __name__ == "__main__":
    # Test extractor only for now
    # Add validator test once nemotron-mini is pulled
    # speed = run_benchmark(
    #     model=EXTRACTOR_MODEL,
    #     prompt=EXTRACT_PROMPT,
    #     label="Extraction (IEEE text → JSON rules)",
    # )

    # Uncomment when nemotron-mini is pulled:
    run_benchmark(
        model=VALIDATOR_MODEL,
        prompt=VALIDATE_PROMPT,
        label="Validation (candidates → CONFIRM/REJECT)",
    )