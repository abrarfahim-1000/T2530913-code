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
    generate_no_think,
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
    n_unparseable:   int = 0   # translator entries discarded before mapping


# ── OLLAMA CALL ───────────────────────────────────────────────────────────────
DEBUG_RAW_DIR: Optional[Path] = None   # set by main() when --debug-raw is passed


def _dump_raw(chunk_label: str, raw: str) -> None:
    """Persist a response that could not be parsed, so a parsing failure can be
    diagnosed from the response itself rather than inferred from the counts."""
    if DEBUG_RAW_DIR is None:
        return
    DEBUG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in chunk_label)
    path = DEBUG_RAW_DIR / f"{safe}.txt"
    path.write_text(raw, encoding="utf-8")
    log.info(f"  [{chunk_label}] raw response -> {path}")


def run_translator(chunk: str, rules: list[dict], chunk_label: str = "") -> list[dict]:
    raw = generate_no_think(
        TRANSLATOR_MODEL,
        TRANSLATE_PROMPT.format(chunk=chunk, rules=json.dumps(rules, indent=2)),
    )
    try:
        return extract_json_array(raw)
    except (ValueError, json.JSONDecodeError):
        _dump_raw(chunk_label, raw)
        raise


# ── RESULT MAPPING ────────────────────────────────────────────────────────────
def build_result_map(
    results_raw: list[dict],
    rules: list[dict],
    chunk_label: str = "",
) -> tuple[dict[str, TranslationResult], int]:
    """Map the translator's array back onto the rules it was asked about.

    Primary key is the echoed `rule_id`. A response that omits it is NOT silently
    dropped: when the array length matches the rules sent, entries are zipped
    positionally instead, because the prompt requires same-order output.

    Why the fallback exists (2026-08-19): every worked example in TRANSLATE_PROMPT
    omitted `rule_id` while `TranslationResult` required it, so every entry failed
    validation and was discarded by a bare `except ValidationError: pass`. The
    result_map came back empty and 2,190 of 2,463 rules (88.9%) were written out
    as NO_VERDICT — a total plumbing failure that read as a property of the
    corpus. The prompt is fixed; this is the belt to that braces, and the counts
    below are what make a recurrence visible instead of silent.
    """
    result_map: dict[str, TranslationResult] = {}
    n_rejected = 0
    unkeyed: list[dict] = []

    for r in results_raw:
        if not isinstance(r, dict):
            n_rejected += 1
            log.warning(f"  [{chunk_label}] translator entry is {type(r).__name__}, not an object")
            continue
        try:
            tr = TranslationResult(**r)
        except ValidationError as exc:
            missing = {e["loc"][0] for e in exc.errors() if e["type"] == "missing"}
            if missing == {"rule_id"}:
                unkeyed.append(r)          # recoverable by position — see below
            else:
                n_rejected += 1
                log.warning(f"  [{chunk_label}] unparseable translator entry "
                            f"({', '.join(sorted(str(m) for m in missing)) or 'invalid'}): "
                            f"{json.dumps(r)[:160]}")
            continue
        result_map[tr.rule_id] = tr

    if unkeyed:
        # Positional recovery is only sound when the response accounts for every
        # rule exactly once; a partial response gives no way to know which rule an
        # unkeyed entry answers, and guessing would mislabel rules.
        if len(results_raw) == len(rules) and not result_map:
            log.warning(f"  [{chunk_label}] {len(unkeyed)} entr(y/ies) omitted `rule_id`; "
                        f"recovering by position (array length matches rules sent)")
            for rule, raw in zip(rules, results_raw):
                try:
                    tr = TranslationResult(**{**raw, "rule_id": rule["rule_id"]})
                except ValidationError:
                    n_rejected += 1
                    continue
                result_map[tr.rule_id] = tr
        else:
            n_rejected += len(unkeyed)
            log.warning(f"  [{chunk_label}] {len(unkeyed)} entr(y/ies) omitted `rule_id` and "
                        f"positional recovery is unsafe ({len(results_raw)} results for "
                        f"{len(rules)} rules) — those rules get NO_VERDICT")

    if not result_map and rules:
        log.error(f"  [{chunk_label}] NO rule received a verdict from {len(results_raw)} "
                  f"translator result(s) — all {len(rules)} rules will be NO_VERDICT. "
                  f"This is a parsing failure, not a corpus property; re-run with --debug-raw.")

    return result_map, n_rejected


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
        results_raw = run_translator(chunk, rules, chunk_label)
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

    result_map, n_rejected = build_result_map(results_raw, rules, chunk_label)

    per_rule_ms = elapsed_ms / max(len(rules), 1)

    for rule in rules:
        rid = rule["rule_id"]
        tr = result_map.get(rid)

        if tr is not None and tr.translatable and tr.condition:
            # Validate the translated condition with the strict Rule schema.
            # Role/affirms come from the translator; Rule defaults an unset role to
            # CONSTRAINT and rejects an AFFIRMATION that names no class.
            merged = {**rule, "condition": tr.condition}
            if tr.role:
                merged["role"] = tr.role
            if tr.affirms:
                merged["affirms"] = tr.affirms
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
        n_schema_valid, elapsed_ms, n_rejected,
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
        "n_unparseable":       0,
        "n_no_verdict":        0,
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
            stats["n_unparseable"]     += result.n_unparseable
            stats["n_no_verdict"]      += sum(
                1 for rec in untranslatable if rec["reason"] == "NO_VERDICT"
            )
            stats["total_translate_ms"] += result.translate_ms

            log.info(
                f"    translatable={result.n_schema_valid}  "
                f"untranslatable={result.n_untranslatable}  "
                f"{result.translate_ms:.0f}ms"
            )

    n_written = stats["n_translatable"]
    log.info(f"  → {translated_file.name}  ({n_written} rules)")
    log.info(f"  → {untranslatable_file.name}  ({stats['n_untranslatable']} untranslatable)")

    # A high NO_VERDICT rate means rules were never assessed, which is a very
    # different claim from "the corpus is not expressible" - and looks identical
    # in the totals. Say so at the point it happens.
    no_verdict = stats["n_no_verdict"]
    if no_verdict:
        share = no_verdict / max(len(records), 1)
        msg = (f"  {no_verdict}/{len(records)} rule(s) ({share:.1%}) got NO_VERDICT - "
               f"the translator's answer never reached them")
        if share > 0.20:
            log.error(msg + ". This is a PARSING failure, not a corpus property. "
                            "Re-run with --debug-raw before reporting these counts.")
        else:
            log.warning(msg)
    return stats


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2: LLM condition translation")
    parser.add_argument("--candidates", required=True, help="Folder with *_candidates.jsonl files")
    parser.add_argument("--out",        default=None,  help="Output folder (default: same as --candidates)")
    parser.add_argument("--debug-raw", action="store_true",
                        help="Dump translator responses that fail to parse to <out>/_raw/")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates)
    out_dir        = Path(args.out) if args.out else candidates_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.debug_raw:
        global DEBUG_RAW_DIR
        DEBUG_RAW_DIR = out_dir / "_raw"

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
    run_stats["total_no_verdict"] = sum(
        s.get("n_no_verdict", 0) for s in run_stats["files"]
    )
    run_stats["total_unparseable"] = sum(
        s.get("n_unparseable", 0) for s in run_stats["files"]
    )

    summary_path = out_dir / "translation_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(run_stats, f, indent=2)

    log.info(f"\nTranslation complete.")
    log.info(f"  Translatable   : {run_stats['total_translatable']}")
    log.info(f"  Untranslatable : {run_stats['total_untranslatable']}")
    log.info(f"  Schema-valid   : {run_stats['total_schema_valid']}")
    log.info(f"  NO_VERDICT     : {run_stats['total_no_verdict']}  "
             f"(rules the translator's answer never reached)")
    log.info(f"  Unparseable    : {run_stats['total_unparseable']}  (discarded results)")
    log.info(f"  Time           : {run_stats['total_time_sec']}s")
    log.info(f"  Summary        : {summary_path}")
    log.info(f"\nNext step: python extraction/validate.py --candidates {out_dir}/")


if __name__ == "__main__":
    main()