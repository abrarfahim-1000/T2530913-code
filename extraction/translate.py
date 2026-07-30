"""
translate.py — Stage 2: Condition Translation
==============================================
Reads *_candidates.jsonl (raw rules in engineering language), calls the
extractor model to translate each condition into the Grid2Op vocabulary,
and writes *_translated.jsonl + *_untranslatable.jsonl.

The translator runs on the same model as the extractor (Qwen3-14B) —
it's loaded once after extraction completes, then unloaded before the
validator runs.

Usage:
    python extraction/translate.py --candidates rules/
    python extraction/translate.py --candidates rules/ --out rules/

Output (in --out folder):
    <pdf_stem>_translated.jsonl        — rules with Grid2Op-compatible conditions
    <pdf_stem>_untranslatable.jsonl    — rules that couldn't be translated (audit trail)
    translation_run_summary.json       — stats + metadata

Next step: python extraction/validate.py --candidates rules/
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from pydantic import ValidationError

from common import (
    TRANSLATOR_MODEL,
    TRANSLATE_PROMPT,
    Rule,
    TranslationResult,
    get_logger,
    extract_json_array,
)

log = get_logger("translate")


def unload_model(model_name: str):
    """Unload a model from Ollama to free VRAM."""
    try:
        ollama.generate(
            model=model_name,
            prompt="",
            options=ollama.Options(num_predict=0),
            keep_alive=0,
        )
        log.info(f"Unloaded model {model_name}")
    except Exception as e:
        log.warning(f"Failed to unload model {model_name}: {e}")


# ── PER-CHUNK RESULT ──────────────────────────────────────────────────────────
@dataclass
class TranslationBatchResult:
    chunk_label:     str
    n_rules:         int
    n_translatable:  int
    n_untranslatable: int
    n_schema_valid:  int
    translate_ms:    float


# ── OLLAMA CALL ───────────────────────────────────────────────────────────────
def run_translator(chunk: str, rules: list[dict]) -> list[dict]:
    resp = ollama.generate(
        model=TRANSLATOR_MODEL,
        prompt=TRANSLATE_PROMPT.format(
            chunk=chunk,
            rules=json.dumps(rules, indent=2),
        ),
        options=ollama.Options(temperature=0.0, num_predict=2048, num_ctx=4096),
        keep_alive=-1,  # Keep model loaded indefinitely
        stream=False,
    )
    return extract_json_array(resp["response"])


# ── BATCH TRANSLATOR ──────────────────────────────────────────────────────────
def translate_batch(
    chunk: str,
    rules: list[dict],
    chunk_label: str,
) -> tuple[list[dict], list[dict], TranslationBatchResult]:
    """Sends all rules from one chunk to the translator in a single call.
    Returns (translated_records, untranslatable_records, stats).
    """
    n_translatable = n_untranslatable = n_schema_valid = 0
    translated: list[dict] = []
    untranslatable: list[dict] = []

    try:
        t0 = time.perf_counter()
        results_raw = run_translator(chunk, rules)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        log.warning(f"  [{chunk_label}] Translator call failed: {exc} — all flagged untranslatable")
        for rule in rules:
            untranslatable.append({
                "rule": rule,
                "reason": f"TRANSLATOR_FAIL: {exc}",
            })
        return translated, untranslatable, TranslationBatchResult(
            chunk_label, len(rules), 0, len(rules), 0, 0.0,
        )

    # Parse translation results
    result_map = {}
    for r in results_raw:
        try:
            tr = TranslationResult(**r)
            result_map[tr.rule_id] = tr
        except ValidationError:
            pass  # malformed result → treated as untranslatable below

    per_rule_ms = elapsed_ms / max(len(rules), 1)

    for rule in rules:
        rid = rule["rule_id"]
        tr = result_map.get(rid)

        if tr is not None and tr.translatable and tr.condition:
            # Validate the translated condition with the strict Rule schema
            merged = {**rule, "condition": tr.condition}
            try:
                validated = Rule(**merged).model_dump()
                translated.append({"rule": validated, "chunk": chunk})
                n_translatable += 1
                n_schema_valid += 1
            except ValidationError as exc:
                # Translation passed format but failed lint — treat as untranslatable
                untranslatable.append({
                    "rule": rule,
                    "reason": f"SCHEMA_REJECTED: {tr.condition!r} — {exc}",
                })
                n_untranslatable += 1
        else:
            reason = tr.reason if tr is not None else "NO_VERDICT"
            untranslatable.append({
                "rule": rule,
                "reason": reason,
            })
            n_untranslatable += 1

    return translated, untranslatable, TranslationBatchResult(
        chunk_label, len(rules), n_translatable, n_untranslatable,
        n_schema_valid, elapsed_ms,
    )


# ── FILE PROCESSOR ────────────────────────────────────────────────────────────
def process_candidates_file(candidates_file: Path, out_dir: Path) -> dict:
    """Reads one *_candidates.jsonl, groups records by chunk, calls translator
    per chunk, writes translated + untranslatable output files."""
    stem = candidates_file.stem.replace("_candidates", "")
    log.info(f"{'=' * 60}")
    log.info(f"Translating: {candidates_file.name}")

    # Load all candidate records
    records = []
    with candidates_file.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        log.warning("  No candidate records — skipping")
        return {"file": candidates_file.name, "n_candidates": 0}

    log.info(f"  Candidates: {len(records)}")

    # Group by chunk text so we send each chunk's rules in one translator call
    chunk_groups: dict[str, list[dict]] = {}
    for rec in records:
        chunk = rec["chunk"]
        rule  = rec["rule"]
        chunk_groups.setdefault(chunk, []).append(rule)

    translated_file = out_dir / f"{stem}_translated.jsonl"
    untranslatable_file = out_dir / f"{stem}_untranslatable.jsonl"

    stats = {
        "file":                candidates_file.name,
        "n_candidates":        len(records),
        "n_translatable":      0,
        "n_untranslatable":    0,
        "n_schema_valid":      0,
        "total_translate_ms":  0.0,
    }

    with translated_file.open("w") as tf, untranslatable_file.open("w") as uf:
        for chunk_idx, (chunk, rules) in enumerate(chunk_groups.items()):
            label = f"chunk {chunk_idx + 1}/{len(chunk_groups)}"
            log.info(f"  {label} — {len(rules)} rule(s)")

            translated, untranslatable, result = translate_batch(chunk, rules, label)

            for rec in translated:
                tf.write(json.dumps(rec) + "\n")
            for rec in untranslatable:
                uf.write(json.dumps(rec) + "\n")

            stats["n_translatable"]    += result.n_translatable
            stats["n_untranslatable"]  += result.n_untranslatable
            stats["n_schema_valid"]    += result.n_schema_valid
            stats["total_translate_ms"] += result.translate_ms

            log.info(
                f"    translatable={result.n_schema_valid}  "
                f"untranslatable={result.n_untranslatable}  "
                f"{result.translate_ms:.0f}ms"
            )

    n_written = stats["n_translatable"]
    log.info(f"  → {translated_file.name}  ({n_written} rules)")
    log.info(f"  → {untranslatable_file.name}  ({stats['n_untranslatable']} untranslatable)")
    return stats


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2: LLM condition translation")
    parser.add_argument("--candidates", required=True, help="Folder with *_candidates.jsonl files")
    parser.add_argument("--out",        default=None,  help="Output folder (default: same as --candidates)")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates)
    out_dir        = Path(args.out) if args.out else candidates_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(candidates_dir.glob("*_candidates.jsonl"))
    if not candidate_files:
        log.error(f"No *_candidates.jsonl files in {candidates_dir}")
        log.error("Run extraction/extract.py first.")
        sys.exit(1)

    log.info(f"Found {len(candidate_files)} candidate file(s)")
    log.info(f"Translator model : {TRANSLATOR_MODEL}")
    log.info(f"Output           : {out_dir}/")

    try:
        available = [m.model for m in ollama.list().models]
        if not any(TRANSLATOR_MODEL in m for m in available):
            log.error(f"Model '{TRANSLATOR_MODEL}' not found. Run: ollama pull {TRANSLATOR_MODEL}")
            sys.exit(1)
    except Exception as exc:
        log.error(f"Cannot connect to Ollama: {exc}")
        sys.exit(1)

    run_stats = {
        "run_start":        datetime.now().isoformat(),
        "translator_model": TRANSLATOR_MODEL,
        "files":            [],
    }

    try:
        t0 = time.perf_counter()
        for cf in candidate_files:
            file_stats = process_candidates_file(cf, out_dir)
            run_stats["files"].append(file_stats)
    finally:
        unload_model(TRANSLATOR_MODEL)

    run_stats["run_end"]        = datetime.now().isoformat()
    run_stats["total_time_sec"] = round(time.perf_counter() - t0, 1)

    run_stats["total_translatable"] = sum(
        s.get("n_translatable", 0) for s in run_stats["files"]
    )
    run_stats["total_untranslatable"] = sum(
        s.get("n_untranslatable", 0) for s in run_stats["files"]
    )
    run_stats["total_schema_valid"] = sum(
        s.get("n_schema_valid", 0) for s in run_stats["files"]
    )

    summary_path = out_dir / "translation_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(run_stats, f, indent=2)

    log.info(f"\nTranslation complete.")
    log.info(f"  Translatable   : {run_stats['total_translatable']}")
    log.info(f"  Untranslatable : {run_stats['total_untranslatable']}")
    log.info(f"  Schema-valid   : {run_stats['total_schema_valid']}")
    log.info(f"  Time           : {run_stats['total_time_sec']}s")
    log.info(f"  Summary        : {summary_path}")
    log.info(f"\nNext step: python extraction/validate.py --candidates {out_dir}/")


if __name__ == "__main__":
    main()