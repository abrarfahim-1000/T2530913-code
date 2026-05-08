"""
extract_rules.py — LLM Rule Extraction Pipeline for nesygrid
=============================================================
Reads all PDFs from a source folder, chunks each document, runs
Extractor (qwen3:14b) then Validator (nemotron-3-nano) per chunk,
and writes structured output for Knowledge Graph ingestion.

Usage:
    python extract_rules.py --docs docs/          # all PDFs in folder
    python extract_rules.py --docs docs/ --dry-run  # test chunking only
    python extract_rules.py --docs docs/ --skip-validation  # extractor only

Output (per run):
    rules/<pdf_stem>_rules.jsonl       — one JSON rule per line, confirmed
    rules/<pdf_stem>_flagged.jsonl     — ambiguous rules for manual review
    rules/pipeline_run_summary.json    — per-file stats + run metadata

Requires (Ollama running on localhost:11434):
    ollama pull qwen3:14b
    ollama pull nemotron-3-nano   # or whatever your nemotron tag is
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
import pdfplumber
from pydantic import BaseModel, ValidationError, field_validator

# ── CONFIG ────────────────────────────────────────────────────────────────────
# EXTRACTOR_MODEL = "qwen3:14b"
# VALIDATOR_MODEL = "nemotron-3-nano"   # update tag if yours differs
EXTRACTOR_MODEL = "qwen3.5:4b"  # Your small test model
VALIDATOR_MODEL = "qwen3.5:4b"  # Use the same model for both steps in the smoke run

CHUNK_SIZE = 1200  # chars per chunk — ~300–400 tokens for dense IEEE prose
CHUNK_OVERLAP = 200  # overlap to avoid cutting mid-rule

# Updated config sets based on prompt instructions
ENTITY_VALUES = {
    "bus",
    "line",
    "transformer",
    "generator",
    "load",
    "protectiondevice",
    "grid",
}
SEVERITY_VALUES = {"critical", "high", "medium", "low"}
ACTION_VALUES = {
    "BLOCK",
    "DISCONNECT",
    "ALERT",
    "REDISPATCH",
    "RECONNECT",
    "RESPOND_WITHIN_2S",
    "SHED_LOAD",
    "OTHER",
}

EXTRACT_PROMPT = """/no_think
You are a power systems engineer extracting operational safety rules from grid documentation.
From the text below, extract ALL constraints, thresholds, and operational limits as a JSON array.

Each rule MUST have exactly these keys:
- rule_id   : string, format "R_001" (sequential, unique within this response)
- source    : string, e.g. "IEEE Std 1547-2018, Section 7.4"
- entity    : string, one of: Bus, Line, Transformer, Generator, Load, ProtectionDevice, Grid
- condition : string, logical expression using grid variable names, e.g. "voltage_pu > 1.05 OR voltage_pu < 0.95"
- action    : string, one of: BLOCK, DISCONNECT, ALERT, REDISPATCH, RECONNECT, RESPOND_WITHIN_2S, SHED_LOAD, OTHER
- severity  : string, one of: critical, high, medium, low
- explanation : string, plain-English reason for the rule (one sentence)

Rules:
- condition fields MUST use variable names (voltage_pu, loading_pct, rho, p_mw, etc.), NOT prose
- If no rule is present in the text, return an empty array: []
- Output ONLY a valid JSON array. No preamble, no markdown, no explanation.

Text:
{chunk}
"""

VALIDATE_PROMPT = """You are a power systems safety auditor verifying extracted rules against source text.

For each rule below, verify:
1. Is this constraint actually stated in the source text?
2. Is the condition boundary (threshold value) correctly parsed?
3. Is the entity type correct?

Output ONLY a JSON array. Each item must have:
- rule_id  : matching the input rule
- verdict  : one of "CONFIRM", "REJECT", "CORRECT"
- corrected_fields : object with corrected key-value pairs (only if verdict is "CORRECT", else omit)
- reason   : one sentence explaining the verdict

No preamble, no markdown.

Source text:
{chunk}

Extracted rules:
{rules}
"""

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_rules")


# ── PYDANTIC SCHEMA ───────────────────────────────────────────────────────────
class Rule(BaseModel):
    rule_id: str
    source: str
    entity: str
    condition: str
    action: str
    severity: str
    explanation: str

    @field_validator("entity")
    @classmethod
    def check_entity(cls, v):
        if v.lower() not in ENTITY_VALUES:
            raise ValueError(f"Invalid entity: {v}")
        # Return properly cased version if needed, or just keep original
        return v

    @field_validator("severity")
    @classmethod
    def check_severity(cls, v):
        if v.lower() not in SEVERITY_VALUES:
            raise ValueError(f"Invalid severity: {v}")
        return v.lower()

    @field_validator("action")
    @classmethod
    def check_action(cls, v):
        if v.upper() not in ACTION_VALUES:
            # Don't reject — normalize to OTHER
            return "OTHER"
        return v.upper()


class Verdict(BaseModel):
    rule_id: str
    verdict: str  # CONFIRM | REJECT | CORRECT
    corrected_fields: Optional[dict] = None
    reason: Optional[str] = None

    @field_validator("verdict")
    @classmethod
    def check_verdict(cls, v):
        v = v.upper()
        if v not in {"CONFIRM", "REJECT", "CORRECT"}:
            raise ValueError(f"Invalid verdict: {v}")
        return v


# ── CHUNKING ──────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF using pdfplumber (layout-aware)."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Sliding window chunker. Tries to break at paragraph boundaries."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Try to end at a paragraph boundary within the last 20% of the window
        if end < n:
            search_from = start + int(size * 0.8)
            para_break = text.rfind("\n\n", search_from, end)
            if para_break > search_from:
                end = para_break

        chunks.append(text[start:end].strip())

        # CRITICAL FIX: If we've reached the end of the text, stop chunking!
        if end >= n:
            break

        start = end - overlap

    return [c for c in chunks if c]  # drop empty


# ── JSON PARSING ──────────────────────────────────────────────────────────────
def _strip_think(text: str) -> str:
    """Strip <think>...</think> blocks that leak through despite /no_think."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_array(text: str) -> list:
    """Robustly pull a JSON array out of model output."""
    text = _strip_think(text)
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"No JSON array found. Raw output (first 300 chars):\n{text[:300]}"
        )
    return json.loads(text[start : end + 1])


# ── OLLAMA CALLS ──────────────────────────────────────────────────────────────
def run_extractor(chunk: str) -> list[dict]:
    """
    Call qwen3:14b via ollama.generate().
    keep_alive=0 instantly clears VRAM when inference finishes.
    """
    resp = ollama.generate(
        model=EXTRACTOR_MODEL,
        prompt=EXTRACT_PROMPT.format(chunk=chunk),
        options=ollama.Options(
            temperature=0.0,
            num_predict=2048,
            num_ctx=4096,
        ),
        keep_alive=0,  # CRITICAL FIX: prevents VRAM overlap with Validator
        stream=False,
    )
    raw = resp["response"]
    return _extract_json_array(raw)


def run_validator(chunk: str, candidates: list[dict]) -> list[dict]:
    """
    Call nemotron-mini via ollama.chat() with think=False.
    keep_alive=0 instantly clears VRAM when inference finishes.
    """
    resp = ollama.chat(
        model=VALIDATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a power systems safety auditor. Output ONLY valid JSON arrays. No preamble.",
            },
            {
                "role": "user",
                "content": VALIDATE_PROMPT.format(
                    chunk=chunk,
                    rules=json.dumps(candidates, indent=2),
                ),
            },
        ],
        options=ollama.Options(
            temperature=0.0,
            num_predict=2048,
            num_ctx=8192,  # needs chunk + candidates in context
        ),
        keep_alive=0,  # CRITICAL FIX: prevents VRAM overlap with Extractor
        think=False,
        stream=False,
    )
    raw = resp["message"]["content"]
    return _extract_json_array(raw)


# ── RULE ID REWRITING ─────────────────────────────────────────────────────────
def rewrite_rule_ids(rules: list[dict], base: int) -> list[dict]:
    """Assign globally unique rule IDs across all chunks and PDFs."""
    for i, r in enumerate(rules):
        r["rule_id"] = f"R_{base + i:03d}"
    return rules


# ── CHUNK PROCESSOR ───────────────────────────────────────────────────────────
@dataclass
class ChunkResult:
    chunk_idx: int
    confirmed: list
    flagged: list
    n_extracted: int
    n_confirmed: int
    n_rejected: int
    n_flagged: int
    ids_consumed: int  # Added to track how many rule_ids we actually handed out
    extract_ms: float
    validate_ms: float
    parse_error: Optional[str] = None


def process_chunk(
    chunk: str,
    chunk_idx: int,
    rule_id_base: int,
    skip_validation: bool = False,
) -> ChunkResult:
    confirmed = []
    flagged = []
    n_ext = n_conf = n_rej = n_flag = ids_consumed = 0
    e_ms = v_ms = 0.0
    parse_error = None

    # ── EXTRACTION ────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        raw_rules = run_extractor(chunk)
        e_ms = (time.perf_counter() - t0) * 1000
        n_ext = len(raw_rules)
    except Exception as exc:
        log.warning(f"  [chunk {chunk_idx}] Extractor failed: {exc}")
        parse_error = str(exc)
        return ChunkResult(chunk_idx, [], [], 0, 0, 0, 0, 0, 0.0, 0.0, parse_error)

    if not raw_rules:
        return ChunkResult(chunk_idx, [], [], 0, 0, 0, 0, 0, e_ms, 0.0)

    # Validate + enrich with Pydantic
    valid_candidates = []
    for r in raw_rules:
        try:
            valid_candidates.append(Rule(**r).model_dump())
        except ValidationError as exc:
            log.debug(f"  [chunk {chunk_idx}] Schema drop: {exc}")

    if not valid_candidates:
        return ChunkResult(chunk_idx, [], [], n_ext, 0, n_ext, 0, 0, e_ms, 0.0)

    # Assign provisional IDs
    valid_candidates = rewrite_rule_ids(valid_candidates, rule_id_base)
    ids_consumed = len(
        valid_candidates
    )  # CRITICAL FIX: Note exactly how many IDs we assigned to candidates

    # ── VALIDATION (skip if --skip-validation) ────────────────────────────────
    if skip_validation:
        confirmed = valid_candidates
        n_conf = len(confirmed)
        return ChunkResult(
            chunk_idx, confirmed, [], n_ext, n_conf, 0, 0, ids_consumed, e_ms, 0.0
        )

    try:
        t0 = time.perf_counter()
        verdicts = run_validator(chunk, valid_candidates)
        v_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        log.warning(
            f"  [chunk {chunk_idx}] Validator failed: {exc} — keeping candidates as unvalidated"
        )
        flagged = [
            {"rule": r, "verdict": {"verdict": "VALIDATOR_FAIL", "reason": str(exc)}}
            for r in valid_candidates
        ]
        n_flag = len(flagged)
        return ChunkResult(
            chunk_idx, [], flagged, n_ext, 0, 0, n_flag, ids_consumed, e_ms, v_ms
        )

    # Parse verdicts
    verdict_map = {}
    for v in verdicts:
        try:
            vv = Verdict(**v)
            verdict_map[vv.rule_id] = vv
        except ValidationError:
            pass  # malformed verdict — treat as CONFLICT below

    for rule in valid_candidates:
        rid = rule["rule_id"]
        vd = verdict_map.get(rid)

        if vd is None:
            # No verdict returned — flag for manual review
            flagged.append(
                {
                    "rule": rule,
                    "verdict": {
                        "verdict": "NO_VERDICT",
                        "reason": "Validator returned no verdict for this rule_id",
                    },
                }
            )
            n_flag += 1

        elif vd.verdict == "CONFIRM":
            confirmed.append(rule)
            n_conf += 1

        elif vd.verdict == "CORRECT":
            merged = {**rule, **(vd.corrected_fields or {})}
            try:
                confirmed.append(Rule(**merged).model_dump())
                n_conf += 1
            except ValidationError:
                # Corrected fields broke schema — flag it
                flagged.append({"rule": rule, "verdict": vd.model_dump()})
                n_flag += 1

        elif vd.verdict == "REJECT":
            n_rej += 1

        else:
            flagged.append({"rule": rule, "verdict": vd.model_dump()})
            n_flag += 1

    return ChunkResult(
        chunk_idx,
        confirmed,
        flagged,
        n_ext,
        n_conf,
        n_rej,
        n_flag,
        ids_consumed,
        e_ms,
        v_ms,
        parse_error,
    )


# ── PDF PROCESSOR ─────────────────────────────────────────────────────────────
def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    rule_id_counter: list,  # mutable single-element list for shared state
    skip_validation: bool = False,
    dry_run: bool = False,
) -> dict:
    log.info(f"{'=' * 60}")
    log.info(f"Processing: {pdf_path.name}")

    # Extract text
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as exc:
        log.error(f"  PDF read failed: {exc}")
        return {"file": pdf_path.name, "error": str(exc)}

    if not text.strip():
        log.warning(f"  No extractable text — skipping")
        return {"file": pdf_path.name, "error": "empty_text"}

    log.info(f"  Text length: {len(text):,} chars")

    # Chunk
    chunks = chunk_text(text)
    log.info(f"  Chunks: {len(chunks)}")

    if dry_run:
        log.info("  [dry-run] Skipping LLM calls")
        return {"file": pdf_path.name, "dry_run": True, "n_chunks": len(chunks)}

    # Output files
    stem = pdf_path.stem
    rules_file = out_dir / f"{stem}_rules.jsonl"
    flagged_file = out_dir / f"{stem}_flagged.jsonl"

    file_stats = {
        "file": pdf_path.name,
        "n_chunks": len(chunks),
        "n_extracted": 0,
        "n_confirmed": 0,
        "n_rejected": 0,
        "n_flagged": 0,
        "n_parse_errors": 0,
        "total_extract_ms": 0.0,
        "total_validate_ms": 0.0,
    }

    with rules_file.open("w") as rf, flagged_file.open("w") as ff:
        for i, chunk in enumerate(chunks):
            log.info(f"  Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars) ...")

            result = process_chunk(
                chunk,
                chunk_idx=i,
                rule_id_base=rule_id_counter[0],
                skip_validation=skip_validation,
            )

            # CRITICAL FIX: Advance global rule ID counter by ALL ids generated (prevent collisions)
            rule_id_counter[0] += result.ids_consumed

            # Write confirmed rules
            for rule in result.confirmed:
                rf.write(json.dumps(rule) + "\n")

            # Write flagged rules
            for item in result.flagged:
                ff.write(json.dumps(item) + "\n")

            # Accumulate stats
            file_stats["n_extracted"] += result.n_extracted
            file_stats["n_confirmed"] += result.n_confirmed
            file_stats["n_rejected"] += result.n_rejected
            file_stats["n_flagged"] += result.n_flagged
            file_stats["total_extract_ms"] += result.extract_ms
            file_stats["total_validate_ms"] += result.validate_ms
            if result.parse_error:
                file_stats["n_parse_errors"] += 1

            log.info(
                f"    extracted={result.n_extracted}  "
                f"confirmed={result.n_confirmed}  "
                f"rejected={result.n_rejected}  "
                f"flagged={result.n_flagged}  "
                f"ext={result.extract_ms:.0f}ms  "
                f"val={result.validate_ms:.0f}ms"
            )

    log.info(
        f"  DONE — {file_stats['n_confirmed']} rules confirmed, "
        f"{file_stats['n_flagged']} flagged, "
        f"{file_stats['n_rejected']} rejected"
    )
    log.info(f"  → {rules_file.name}")
    log.info(f"  → {flagged_file.name}")

    return file_stats


# ── DEDUPLICATION ─────────────────────────────────────────────────────────────
def deduplicate_rules(rules_dir: Path) -> int:
    """
    After all PDFs are processed, merge all *_rules.jsonl into
    all_rules_deduped.jsonl, dropping rules with identical (entity, condition) pairs.
    Source references are merged.
    """
    seen = {}  # (entity, condition) → rule dict
    all_jsonl = list(rules_dir.glob("*_rules.jsonl"))
    dropped = 0

    for jf in all_jsonl:
        with jf.open() as f:
            for line in f:
                try:
                    rule = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rule.get("entity", ""), rule.get("condition", ""))
                if key not in seen:
                    seen[key] = rule
                else:
                    # Merge sources and log drop
                    existing_src = seen[key]["source"]
                    new_src = rule["source"]
                    if new_src not in existing_src:
                        seen[key]["source"] = f"{existing_src}; {new_src}"
                    dropped += 1

    # Note: Because of rejected IDs and deduplication drops, final IDs won't be perfectly sequential.
    if dropped > 0:
        log.info(
            f"  [Dedup] Dropped {dropped} redundant rules (IDs merged and discarded)."
        )

    out_path = rules_dir / "all_rules_deduped.jsonl"
    with out_path.open("w") as f:
        for rule in seen.values():
            f.write(json.dumps(rule) + "\n")

    return len(seen)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM rule extraction pipeline for nesygrid"
    )
    parser.add_argument("--docs", required=True, help="Folder containing PDFs")
    parser.add_argument(
        "--out", default="rules", help="Output folder (default: rules/)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Run extractor only, skip validator",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse PDFs and chunk, no LLM calls"
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDFs found in {docs_dir}")
        sys.exit(1)

    log.info(f"Found {len(pdf_files)} PDF(s) in {docs_dir}")
    log.info(f"Output → {out_dir}/")
    if args.skip_validation:
        log.info("Mode: EXTRACTION ONLY (no validator)")
    if args.dry_run:
        log.info("Mode: DRY RUN (no LLM calls)")

    # Check Ollama connectivity (unless dry run)
    if not args.dry_run:
        try:
            available = [m.model for m in ollama.list().models]
            log.info(f"Ollama models available: {available}")
            if not any(EXTRACTOR_MODEL in m for m in available):
                log.error(
                    f"Extractor model '{EXTRACTOR_MODEL}' not found. Run: ollama pull {EXTRACTOR_MODEL}"
                )
                sys.exit(1)
            if not args.skip_validation and not any(
                VALIDATOR_MODEL in m for m in available
            ):
                log.error(
                    f"Validator model '{VALIDATOR_MODEL}' not found. Run: ollama pull {VALIDATOR_MODEL}"
                )
                sys.exit(1)
        except Exception as exc:
            log.error(f"Cannot connect to Ollama: {exc}. Is 'ollama serve' running?")
            sys.exit(1)

    rule_id_counter = [1]  # mutable shared counter across PDFs
    run_stats = {
        "run_start": datetime.now().isoformat(),
        "extractor_model": EXTRACTOR_MODEL,
        "validator_model": VALIDATOR_MODEL if not args.skip_validation else "skipped",
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "skip_validation": args.skip_validation,
        "dry_run": args.dry_run,
        "files": [],
    }

    t_total = time.perf_counter()

    for pdf_path in pdf_files:
        file_stats = process_pdf(
            pdf_path,
            out_dir,
            rule_id_counter,
            skip_validation=args.skip_validation,
            dry_run=args.dry_run,
        )
        run_stats["files"].append(file_stats)

    # Deduplication pass (skip in dry-run)
    if not args.dry_run:
        n_deduped = deduplicate_rules(out_dir)
        log.info(
            f"\nDeduplication complete: {n_deduped} unique rules → {out_dir}/all_rules_deduped.jsonl"
        )
        run_stats["n_unique_rules_after_dedup"] = n_deduped

    run_stats["run_end"] = datetime.now().isoformat()
    run_stats["total_time_sec"] = round(time.perf_counter() - t_total, 1)

    summary_path = out_dir / "pipeline_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(run_stats, f, indent=2)

    log.info(f"\nSummary → {summary_path}")
    log.info(f"Total time: {run_stats['total_time_sec']}s")

    # Final tally
    total_conf = sum(s.get("n_confirmed", 0) for s in run_stats["files"])
    total_flag = sum(s.get("n_flagged", 0) for s in run_stats["files"])
    total_rej = sum(s.get("n_rejected", 0) for s in run_stats["files"])
    log.info(
        f"\nAll files done:\n"
        f"  Confirmed : {total_conf}\n"
        f"  Flagged   : {total_flag}\n"
        f"  Rejected  : {total_rej}\n"
    )


if __name__ == "__main__":
    main()
