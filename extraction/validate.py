"""
validate.py — Stage 2: LLM Validation (nemotron-mini)
======================================================
Reads *_candidates.jsonl files produced by extract.py, calls the
validator model per candidate, and writes confirmed/flagged output.

Each candidate record already contains the source chunk, so this
script is fully independent of the original PDFs.

Usage:
    python validate.py --candidates rules/
    python validate.py --candidates rules/ --out rules/

Output (in --out folder):
    <pdf_stem>_confirmed.jsonl      — rules the validator confirmed
    <pdf_stem>_flagged.jsonl        — REJECT / CORRECT / NO_VERDICT
    all_rules_deduped.jsonl         — merged, deduplicated confirmed rules
    validation_run_summary.json     — stats + metadata
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
    VALIDATOR_MODEL,
    VALIDATE_PROMPT,
    Rule,
    Verdict,
    get_logger,
    extract_json_array,
    deduplicate_rules,
)

log = get_logger("validate")


# ── PER-RECORD RESULT ─────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    rule_id:     str
    outcome:     str          # "confirmed" | "corrected" | "rejected" | "flagged"
    rule:        Optional[dict] = None
    verdict:     Optional[dict] = None
    validate_ms: float = 0.0


# ── OLLAMA CALL ───────────────────────────────────────────────────────────────
def run_validator(chunk: str, candidates: list[dict]) -> list[dict]:
    """
    Uses ollama.chat() with think=False — required for nemotron-mini
    to suppress chain-of-thought before JSON output.
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
        options=ollama.Options(temperature=0.0, num_predict=2048, num_ctx=8192),
        keep_alive=0,
        think=False,
        stream=False,
    )
    return extract_json_array(resp["message"]["content"])


# ── SINGLE-RULE VALIDATOR ─────────────────────────────────────────────────────
def validate_batch(
    chunk: str,
    rules: list[dict],
    chunk_label: str,
) -> list[ValidationResult]:
    """
    Sends all rules from one chunk to the validator in a single call.
    Returns one ValidationResult per rule.
    """
    results = []

    try:
        t0 = time.perf_counter()
        verdicts_raw = run_validator(chunk, rules)
        elapsed_ms   = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        log.warning(f"  [{chunk_label}] Validator call failed: {exc} — flagging all")
        for rule in rules:
            results.append(ValidationResult(
                rule_id=rule["rule_id"],
                outcome="flagged",
                rule=rule,
                verdict={"verdict": "VALIDATOR_FAIL", "reason": str(exc)},
            ))
        return results

    # Parse verdict objects
    verdict_map = {}
    for v in verdicts_raw:
        try:
            vv = Verdict(**v)
            verdict_map[vv.rule_id] = vv
        except ValidationError:
            pass  # malformed verdict treated as NO_VERDICT below

    per_rule_ms = elapsed_ms / max(len(rules), 1)

    for rule in rules:
        rid = rule["rule_id"]
        vd  = verdict_map.get(rid)

        if vd is None:
            results.append(ValidationResult(
                rule_id=rid, outcome="flagged", rule=rule,
                verdict={"verdict": "NO_VERDICT", "reason": "Validator returned no verdict"},
                validate_ms=per_rule_ms,
            ))
            continue

        if vd.verdict == "CONFIRM":
            results.append(ValidationResult(rid, "confirmed", rule, vd.model_dump(), per_rule_ms))

        elif vd.verdict == "CORRECT":
            merged = {**rule, **(vd.corrected_fields or {})}
            try:
                corrected_rule = Rule(**merged).model_dump()
                results.append(ValidationResult(rid, "corrected", corrected_rule, vd.model_dump(), per_rule_ms))
            except ValidationError:
                # Corrected fields broke schema — flag it
                results.append(ValidationResult(rid, "flagged", rule, vd.model_dump(), per_rule_ms))

        elif vd.verdict == "REJECT":
            results.append(ValidationResult(rid, "rejected", rule, vd.model_dump(), per_rule_ms))

        else:
            results.append(ValidationResult(rid, "flagged", rule, vd.model_dump(), per_rule_ms))

    return results


# ── FILE PROCESSOR ────────────────────────────────────────────────────────────
def process_candidates_file(candidates_file: Path, out_dir: Path) -> dict:
    """
    Reads one *_candidates.jsonl, groups records by chunk, calls validator
    per chunk, writes confirmed + flagged output files.
    """
    stem = candidates_file.stem.replace("_candidates", "")
    log.info(f"{'=' * 60}")
    log.info(f"Validating: {candidates_file.name}")

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

    # Group by chunk text so we send each chunk's rules in one validator call
    chunk_groups: dict[str, list[dict]] = {}
    for rec in records:
        chunk = rec["chunk"]
        rule  = rec["rule"]
        chunk_groups.setdefault(chunk, []).append(rule)

    confirmed_file = out_dir / f"{stem}_confirmed.jsonl"
    flagged_file   = out_dir / f"{stem}_flagged.jsonl"

    stats = {
        "file":          candidates_file.name,
        "n_candidates":  len(records),
        "n_confirmed":   0,
        "n_corrected":   0,
        "n_rejected":    0,
        "n_flagged":     0,
        "total_validate_ms": 0.0,
    }

    with confirmed_file.open("w") as cf, flagged_file.open("w") as ff:
        for chunk_idx, (chunk, rules) in enumerate(chunk_groups.items()):
            label = f"chunk {chunk_idx + 1}/{len(chunk_groups)}"
            log.info(f"  {label} — {len(rules)} rule(s)")

            results = validate_batch(chunk, rules, label)

            for res in results:
                stats["total_validate_ms"] += res.validate_ms

                if res.outcome in ("confirmed", "corrected"):
                    cf.write(json.dumps(res.rule) + "\n")
                    if res.outcome == "confirmed":
                        stats["n_confirmed"] += 1
                    else:
                        stats["n_corrected"] += 1
                elif res.outcome == "rejected":
                    stats["n_rejected"] += 1
                else:  # flagged
                    ff.write(json.dumps({"rule": res.rule, "verdict": res.verdict}) + "\n")
                    stats["n_flagged"] += 1

            log.info(
                f"    confirmed={stats['n_confirmed']}  "
                f"corrected={stats['n_corrected']}  "
                f"rejected={stats['n_rejected']}  "
                f"flagged={stats['n_flagged']}"
            )

    n_written = stats["n_confirmed"] + stats["n_corrected"]
    log.info(f"  → {confirmed_file.name}  ({n_written} rules)")
    log.info(f"  → {flagged_file.name}  ({stats['n_flagged']} flagged)")
    return stats


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2: LLM rule validation")
    parser.add_argument("--candidates", required=True, help="Folder with *_candidates.jsonl files")
    parser.add_argument("--out",        default=None,  help="Output folder (default: same as --candidates)")
    args = parser.parse_args()

    candidates_dir = Path(args.candidates)
    out_dir        = Path(args.out) if args.out else candidates_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(candidates_dir.glob("*_candidates.jsonl"))
    if not candidate_files:
        log.error(f"No *_candidates.jsonl files in {candidates_dir}")
        log.error("Run extract.py first.")
        sys.exit(1)

    log.info(f"Found {len(candidate_files)} candidate file(s)")
    log.info(f"Validator model : {VALIDATOR_MODEL}")
    log.info(f"Output          : {out_dir}/")

    try:
        available = [m.model for m in ollama.list().models]
        if not any(VALIDATOR_MODEL in m for m in available):
            log.error(f"Model '{VALIDATOR_MODEL}' not found. Run: ollama pull {VALIDATOR_MODEL}")
            sys.exit(1)
    except Exception as exc:
        log.error(f"Cannot connect to Ollama: {exc}")
        sys.exit(1)

    run_stats = {
        "run_start":       datetime.now().isoformat(),
        "validator_model": VALIDATOR_MODEL,
        "files":           [],
    }

    t0 = time.perf_counter()
    for cf in candidate_files:
        file_stats = process_candidates_file(cf, out_dir)
        run_stats["files"].append(file_stats)

    # Deduplication across all confirmed files
    n_unique = deduplicate_rules(out_dir, log)
    log.info(f"\nDeduplication: {n_unique} unique rules → all_rules_deduped.jsonl")
    run_stats["n_unique_rules_after_dedup"] = n_unique

    run_stats["run_end"]        = datetime.now().isoformat()
    run_stats["total_time_sec"] = round(time.perf_counter() - t0, 1)

    # Aggregate totals
    run_stats["total_confirmed"] = sum(s.get("n_confirmed", 0) + s.get("n_corrected", 0) for s in run_stats["files"])
    run_stats["total_rejected"]  = sum(s.get("n_rejected", 0) for s in run_stats["files"])
    run_stats["total_flagged"]   = sum(s.get("n_flagged", 0)  for s in run_stats["files"])

    summary_path = out_dir / "validation_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(run_stats, f, indent=2)

    log.info(f"\nValidation complete.")
    log.info(f"  Confirmed : {run_stats['total_confirmed']}")
    log.info(f"  Rejected  : {run_stats['total_rejected']}")
    log.info(f"  Flagged   : {run_stats['total_flagged']}")
    log.info(f"  Unique KG-ready rules : {n_unique}")
    log.info(f"  Time      : {run_stats['total_time_sec']}s")
    log.info(f"  Summary   : {summary_path}")


if __name__ == "__main__":
    main()
