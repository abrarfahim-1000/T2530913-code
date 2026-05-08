"""
common.py — Shared constants, schemas, and utilities for the two-stage pipeline.
Both extract.py and validate.py import from here.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pdfplumber
from pydantic import BaseModel, ValidationError, field_validator

# ── MODEL CONFIG ──────────────────────────────────────────────────────────────
EXTRACTOR_MODEL = "qwen3:14b"
VALIDATOR_MODEL = "nemotron-mini"   # update tag if yours differs

# ── CHUNKING CONFIG ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 1200   # chars — ~300–400 tokens for dense IEEE prose
CHUNK_OVERLAP = 200

# ── SCHEMA CONSTANTS ──────────────────────────────────────────────────────────
ENTITY_VALUES = {
    "bus", "line", "transformer", "generator",
    "load", "protectiondevice", "grid",
}
SEVERITY_VALUES = {"critical", "high", "medium", "low"}
ACTION_VALUES = {
    "BLOCK", "DISCONNECT", "ALERT", "REDISPATCH",
    "RECONNECT", "RESPOND_WITHIN_2S", "SHED_LOAD", "OTHER",
}

# ── PROMPTS ───────────────────────────────────────────────────────────────────
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
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────
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
            return "OTHER"
        return v.upper()


class Verdict(BaseModel):
    rule_id: str
    verdict: str   # CONFIRM | REJECT | CORRECT
    corrected_fields: Optional[dict] = None
    reason: Optional[str] = None

    @field_validator("verdict")
    @classmethod
    def check_verdict(cls, v):
        v = v.upper()
        if v not in {"CONFIRM", "REJECT", "CORRECT"}:
            raise ValueError(f"Invalid verdict: {v}")
        return v


# ── PDF + CHUNKING ────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            search_from = start + int(size * 0.8)
            para_break = text.rfind("\n\n", search_from, end)
            if para_break > search_from:
                end = para_break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]


# ── JSON PARSING ──────────────────────────────────────────────────────────────
def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_array(text: str) -> list:
    text = _strip_think(text)
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found. Raw (first 300 chars):\n{text[:300]}")
    return json.loads(text[start:end + 1])


# ── RULE ID REWRITING ─────────────────────────────────────────────────────────
def rewrite_rule_ids(rules: list[dict], base: int) -> list[dict]:
    for i, r in enumerate(rules):
        r["rule_id"] = f"R_{base + i:03d}"
    return rules


# ── DEDUPLICATION (called by validate.py at the end) ──────────────────────────
def deduplicate_rules(rules_dir: Path, log) -> int:
    """
    Merges all *_confirmed.jsonl files → all_rules_deduped.jsonl.
    Drops rules with identical (entity, condition); merges sources.
    """
    seen = {}
    dropped = 0

    for jf in sorted(rules_dir.glob("*_confirmed.jsonl")):
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
                    existing_src = seen[key]["source"]
                    new_src = rule["source"]
                    if new_src not in existing_src:
                        seen[key]["source"] = f"{existing_src}; {new_src}"
                    dropped += 1

    if dropped:
        log.info(f"  [Dedup] Dropped {dropped} duplicate rules.")

    out_path = rules_dir / "all_rules_deduped.jsonl"
    with out_path.open("w") as f:
        for rule in seen.values():
            f.write(json.dumps(rule) + "\n")

    return len(seen)
