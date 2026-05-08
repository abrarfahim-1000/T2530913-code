"""
extract.py — Stage 1: LLM Extraction (qwen3:14b)
=================================================
Reads PDFs, chunks them, calls the extractor model, validates schema
with Pydantic, and writes candidate rules to disk.

Each output record contains the extracted rule PLUS the source chunk
that produced it — so validate.py can run independently later.

Usage:
    python extract.py --docs docs/
    python extract.py --docs docs/ --out rules/ --dry-run

Output (in --out folder):
    <pdf_stem>_candidates.jsonl     — one candidate record per line:
                                      { "rule": {...}, "chunk": "..." }
    extraction_run_summary.json     — stats + metadata for this run

Run validate.py on the candidates folder next.
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
    EXTRACTOR_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EXTRACT_PROMPT,
    Rule,
    get_logger,
    extract_text_from_pdf,
    chunk_text,
    extract_json_array,
    rewrite_rule_ids,
)

log = get_logger("extract")


# ── PER-CHUNK RESULT ──────────────────────────────────────────────────────────
@dataclass
class ChunkResult:
    chunk_idx:   int
    candidates:  list   # list of { "rule": dict, "chunk": str }
    n_raw:       int    # rules the model returned
    n_valid:     int    # rules that passed Pydantic schema
    ids_consumed: int
    extract_ms:  float
    parse_error: Optional[str] = None


# ── OLLAMA CALL ───────────────────────────────────────────────────────────────
def run_extractor(chunk: str) -> list[dict]:
    resp = ollama.generate(
        model=EXTRACTOR_MODEL,
        prompt=EXTRACT_PROMPT.format(chunk=chunk),
        options=ollama.Options(temperature=0.0, num_predict=2048, num_ctx=4096),
        keep_alive=0,
        stream=False,
    )
    return extract_json_array(resp["response"])


# ── CHUNK PROCESSOR ───────────────────────────────────────────────────────────
def process_chunk(chunk: str, chunk_idx: int, rule_id_base: int) -> ChunkResult:
    n_raw = n_valid = ids_consumed = 0
    extract_ms = 0.0
    parse_error = None
    candidates = []

    try:
        t0 = time.perf_counter()
        raw_rules = run_extractor(chunk)
        extract_ms = (time.perf_counter() - t0) * 1000
        n_raw = len(raw_rules)
    except Exception as exc:
        log.warning(f"  [chunk {chunk_idx}] Extractor failed: {exc}")
        return ChunkResult(chunk_idx, [], 0, 0, 0, 0.0, str(exc))

    if not raw_rules:
        return ChunkResult(chunk_idx, [], 0, 0, 0, extract_ms)

    # Pydantic validation — drop malformed rules, keep valid ones
    valid_rules = []
    for r in raw_rules:
        try:
            valid_rules.append(Rule(**r).model_dump())
        except ValidationError as exc:
            log.debug(f"  [chunk {chunk_idx}] Schema drop: {exc}")
    n_valid = len(valid_rules)

    if not valid_rules:
        return ChunkResult(chunk_idx, [], n_raw, 0, 0, extract_ms)

    # Assign globally unique rule IDs
    valid_rules = rewrite_rule_ids(valid_rules, rule_id_base)
    ids_consumed = len(valid_rules)

    # Bundle each rule with its source chunk so the validator is self-contained
    for rule in valid_rules:
        candidates.append({"rule": rule, "chunk": chunk})

    return ChunkResult(chunk_idx, candidates, n_raw, n_valid, ids_consumed, extract_ms)


# ── PDF PROCESSOR ─────────────────────────────────────────────────────────────
def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    rule_id_counter: list,   # [int] — mutable single-element counter
    dry_run: bool = False,
) -> dict:
    log.info(f"{'=' * 60}")
    log.info(f"Extracting: {pdf_path.name}")

    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as exc:
        log.error(f"  PDF read failed: {exc}")
        return {"file": pdf_path.name, "error": str(exc)}

    if not text.strip():
        log.warning("  No extractable text — skipping")
        return {"file": pdf_path.name, "error": "empty_text"}

    log.info(f"  Text: {len(text):,} chars")
    chunks = chunk_text(text)
    log.info(f"  Chunks: {len(chunks)}")

    if dry_run:
        log.info("  [dry-run] Skipping LLM calls")
        return {"file": pdf_path.name, "dry_run": True, "n_chunks": len(chunks)}

    out_file = out_dir / f"{pdf_path.stem}_candidates.jsonl"
    stats = {
        "file": pdf_path.name,
        "n_chunks": len(chunks),
        "n_raw_extracted": 0,
        "n_schema_valid": 0,
        "n_parse_errors": 0,
        "total_extract_ms": 0.0,
    }

    with out_file.open("w") as f:
        for i, chunk in enumerate(chunks):
            log.info(f"  Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
            result = process_chunk(chunk, i, rule_id_counter[0])
            rule_id_counter[0] += result.ids_consumed

            for record in result.candidates:
                f.write(json.dumps(record) + "\n")

            stats["n_raw_extracted"]  += result.n_raw
            stats["n_schema_valid"]   += result.n_valid
            stats["total_extract_ms"] += result.extract_ms
            if result.parse_error:
                stats["n_parse_errors"] += 1

            log.info(
                f"    raw={result.n_raw}  valid={result.n_valid}  "
                f"ids_assigned={result.ids_consumed}  {result.extract_ms:.0f}ms"
            )

    log.info(f"  → {out_file.name}  ({stats['n_schema_valid']} candidates)")
    return stats


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 1: LLM rule extraction")
    parser.add_argument("--docs",          required=True,       help="Folder with PDFs")
    parser.add_argument("--out",           default="rules",     help="Output folder")
    parser.add_argument("--dry-run",       action="store_true", help="No LLM calls")
    parser.add_argument("--chunk-size",    type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDFs in {docs_dir}")
        sys.exit(1)

    log.info(f"Found {len(pdf_files)} PDF(s)")
    log.info(f"Extractor model : {EXTRACTOR_MODEL}")
    log.info(f"Output          : {out_dir}/")

    if not args.dry_run:
        try:
            available = [m.model for m in ollama.list().models]
            if not any(EXTRACTOR_MODEL in m for m in available):
                log.error(f"Model '{EXTRACTOR_MODEL}' not found. Run: ollama pull {EXTRACTOR_MODEL}")
                sys.exit(1)
        except Exception as exc:
            log.error(f"Cannot connect to Ollama: {exc}")
            sys.exit(1)

    rule_id_counter = [1]
    run_stats = {
        "run_start":       datetime.now().isoformat(),
        "extractor_model": EXTRACTOR_MODEL,
        "chunk_size":      args.chunk_size,
        "chunk_overlap":   args.chunk_overlap,
        "dry_run":         args.dry_run,
        "files":           [],
    }

    t0 = time.perf_counter()
    for pdf_path in pdf_files:
        file_stats = process_pdf(pdf_path, out_dir, rule_id_counter, dry_run=args.dry_run)
        run_stats["files"].append(file_stats)

    run_stats["run_end"]        = datetime.now().isoformat()
    run_stats["total_time_sec"] = round(time.perf_counter() - t0, 1)
    run_stats["total_candidates"] = sum(
        s.get("n_schema_valid", 0) for s in run_stats["files"]
    )

    summary_path = out_dir / "extraction_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(run_stats, f, indent=2)

    log.info(f"\nExtraction complete.")
    log.info(f"  Total candidates : {run_stats['total_candidates']}")
    log.info(f"  Time             : {run_stats['total_time_sec']}s")
    log.info(f"  Summary          : {summary_path}")
    log.info(f"\nNext step: python validate.py --candidates {out_dir}/")


if __name__ == "__main__":
    main()
