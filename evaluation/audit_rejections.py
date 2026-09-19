"""
audit_rejections.py — make the validator's 21 shared rejections auditable.

WHY THIS EXISTS
---------------
`thesis_findings.md` §12.4 states that of the 21 rejections shared by both
prompt arms, "at least 3 are false rejections and 4 more should have been
CORRECT verdicts". That claim was a recollection. Every other claim in that
chapter resolves to a file under `results/`; this one did not. This script
closes that gap.

It needs no LLM and no GPU. The raw material is already on disk because of the
2026-08-20 fix: run 1 rejected 31 of 32 rules and persisted nothing but a
count, so `validate.py` now writes `<stem>_rejected.jsonl` carrying the full
`Verdict`, `reason` included.

MECHANICAL vs ADJUDICATED — the distinction is the point
-------------------------------------------------------
Every record this script emits is split in two, and the split is load-bearing:

  "mechanical"   — copied verbatim off disk, or computed deterministically.
                   The rule, the validator's verdict, its `reason` word for
                   word, the source clause, and a vocabulary check that is a
                   set membership test against CONDITION_VOCABULARY. Nobody
                   has to trust the author to re-derive any of it.

  "adjudication" — a human reading. A category, a one-line justification, and
                   the `basis` on which the call was made. It is a JUDGEMENT,
                   not a measurement, and the artifact says so in the payload
                   itself (`"method": "reading, not measurement"`).

A reader who disagrees with one call can discard that call without discarding
the artifact, because each carries its own basis and the mechanical half is
untouched by it.

CATEGORIES
----------
  SOUND                    the stated ground is accurate about the source text
                           and the conclusion follows from it.
  WRONG                    the stated ground contains a load-bearing factual
                           error — about the vocabulary, the source text, or
                           the logic of CONSTRAINT polarity.
  SHOULD_HAVE_BEEN_CORRECT the stated defect is a field-level repair (role,
                           threshold, scope). CORRECT existed as a verdict and
                           was not used.
  UNVERIFIABLE             cannot be adjudicated from the artifacts alone.

RECONCILIATION
--------------
The script compares its own counts against the documented "≥3 WRONG and 4 more
SHOULD_HAVE_BEEN_CORRECT" and against §12.4's table membership, and prints a
DISCREPANCY block when they diverge. The counts were NOT tuned to match. They
do not match, and the mismatch is reported rather than absorbed.

KEYING
------
`rule_id` is NOT globally unique. Five documents in `rules_35b/` restarted
numbering, so `R_001`..`R_172` are reused across five of them (2,463 records,
2,291 distinct ids — `thesis_findings.md` §19.2). Every join here is on
**(document, rule_id)**. `load_rejections()` refuses to key on `rule_id`
alone, and `tests/test_audit_rejections.py` pins that.

Usage:
    python evaluation/audit_rejections.py
    python evaluation/audit_rejections.py --json results/audit/rejection_adjudication.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from extraction.common import CONDITION_VOCABULARY, get_logger  # noqa: E402

log = get_logger("audit_rejections")

REPO = Path(__file__).resolve().parents[1]

DEFAULT_TRANSLATED = REPO / "validated_translated"
DEFAULT_STRICT = REPO / "validated_strict"
DEFAULT_GUARDED = REPO / "translated_rules" / "guarded"

# The figures thesis_findings.md §12.4 commits to. Reconciled against, never
# used to steer the adjudication.
DOCUMENTED_MIN_WRONG = 3
DOCUMENTED_SHOULD_HAVE_BEEN_CORRECT = 4
DOCUMENTED_TOTAL_REJECTIONS = 21

# §12.4's "verdict-selection defect" row names these four and says their "reason
# text says 'requiring correction'". That is a mechanically checkable claim, and
# `correction_language` below checks it.
FINDINGS_12_4_VERDICT_DEFECT = ("R_797", "R_798", "R_434", "R_741")

# The rules §12.4's table actually names, and the row each sits in. Four of the
# 21 appear in no row at all; this table is what makes that visible.
FINDINGS_12_4_TABLE: Dict[str, str] = {
    "R_1833": "PRC-024 ride-through (assessed: correct)",
    "R_1835": "PRC-024 ride-through (assessed: correct)",
    "R_2282": "PRC-024 ride-through (assessed: correct)",
    "R_116": "ENTSO-E time-bound envelope (assessed: correct)",
    "R_120": "ENTSO-E time-bound envelope (assessed: correct)",
    "R_121": "ENTSO-E time-bound envelope (assessed: correct)",
    "R_123": "ENTSO-E time-bound envelope (assessed: correct)",
    "R_1108": "role catch, 0.9-1.1 affirmation (assessed: correct)",
    "R_069": "role catch, 0.9-1.1 affirmation (assessed: correct)",
    "R_797": "threshold catch (assessed: correct) + verdict-selection defect",
    "R_798": "threshold catch (assessed: correct) + verdict-selection defect",
    "R_1319": "STATCOM ride-through range (assessed: correct)",
    "R_1926": "role inversion (assessed: WRONG)",
    "R_1927": "role inversion (assessed: WRONG)",
    "R_1098": "role inversion (assessed: WRONG)",
    "R_434": "verdict-selection defect",
    "R_741": "verdict-selection defect",
}

CATEGORIES = ("SOUND", "WRONG", "SHOULD_HAVE_BEEN_CORRECT", "UNVERIFIABLE")

# Basis codes, so a reader can see what kind of evidence a call rests on.
#   SOURCE_TEXT     the `chunk` carried beside the rule in translated_rules/guarded/
#   VOCABULARY      set membership in CONDITION_VOCABULARY — deterministic
#   CONSTRAINT_LOGIC the semantics of role=CONSTRAINT (condition == violation predicate)
#   VERDICT_SCHEMA  the validator had CORRECT available and did not use it
BASES = ("SOURCE_TEXT", "VOCABULARY", "CONSTRAINT_LOGIC", "VERDICT_SCHEMA")


@dataclass(frozen=True)
class Adjudication:
    category: str
    basis: Tuple[str, ...]
    justification: str
    # True when the stated defect was a field-level repair the CORRECT verdict
    # was designed for. Independent of `category`: a rejection can rest on a
    # sound reading AND still have been the wrong verdict to return.
    repair_was_available: bool = False


# ---------------------------------------------------------------------------
# THE ADJUDICATION TABLE — this is the judgement, and nothing below it is.
#
# Keyed on (document_stem, rule_id). Each justification is one line and cites
# what it rests on, so an individual call can be argued with in isolation.
# ---------------------------------------------------------------------------
NERC = "NERC Reliability Standards (Complete Set - PRC, TOP, TPL, VAR, BAL, CIP, EOP, FAC, MOD)"
ENTSOE = "130308_Final_Version_NC_RfG"
GBCODE = "8589935310-Complete Grid Code"
BDESH = "Bangladesh_Electricity-Grid-Code-regulations"
IEEE = "guideline-ieee_1547-2018_bps_perspectives_clean-1"
IBR = "inverter-based_resource_performance_guideline"

ADJUDICATIONS: Dict[Tuple[str, str], Adjudication] = {
    # -- ENTSO-E NC RfG Table 6.1: a time-bound operating envelope -----------
    (ENTSOE, "R_116"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "Chunk confirms Table 6.1 is an envelope of bands x durations (0.85-0.90 pu / 60 min); "
        "a stateless predicate cannot carry the duration, and the rule holds on 100% of frames "
        "on all three grids, so the affirmation is vacuous as well as incomplete.",
    ),
    (ENTSOE, "R_120"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "Same envelope argument as R_116; Great Britain row is 0.90-1.10 pu unlimited, but the "
        "clause as a whole is duration-indexed. Fires at 1.00 on all three grids.",
    ),
    (ENTSOE, "R_121"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "Same envelope argument. Minor wobble: the reason attributes the 30-minute 0.85-0.90 row "
        "to Ireland where the chunk's garbled table layout puts it under Baltic - not load-bearing.",
    ),
    (ENTSOE, "R_123"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "Same envelope argument; the cited 20-minute 1.12-1.15 pu Baltic row is verbatim in the chunk.",
    ),

    # -- GB Grid Code CC.6.1.4 ----------------------------------------------
    (GBCODE, "R_260"): Adjudication(
        "WRONG", ("SOURCE_TEXT",),
        "The reason asserts the source describes only a +/-5% band. The chunk's very next sentence "
        "reads 'The minimum voltage is -10% and the maximum voltage is +10%', which is exactly the "
        "0.9/1.1 the condition encodes. The validator stopped reading one sentence early.",
    ),
    (GBCODE, "R_434"): Adjudication(
        "SHOULD_HAVE_BEEN_CORRECT", ("SOURCE_TEXT", "VERDICT_SCHEMA"),
        "The chunk states 'above 120% (115% for 275kV)' verbatim, so the 1.2 threshold IS stated. "
        "The objection is a scope gap (the 275 kV exception), repairable by narrowing entity/source - "
        "precisely what CORRECT exists for.",
        repair_was_available=True,
    ),
    (GBCODE, "R_741"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "The chunk is a Generator DATA-SUBMISSION FORM ('120% rated terminal volts A []', '50% rated "
        "terminal volts'), a list of data points to supply, not an operating limit. There is no "
        "requirement here to operationalize. Good catch.",
    ),

    # -- Bangladesh Grid Code 5.4.4 -----------------------------------------
    (BDESH, "R_797"): Adjudication(
        "SHOULD_HAVE_BEEN_CORRECT", ("SOURCE_TEXT", "VERDICT_SCHEMA"),
        "The reason concedes the threshold 'is correct' and closes 'thus requiring correction', then "
        "returns REJECT. The chunk backs the concession: '+/-10 % at 400 kV ... during emergencies' "
        "is 0.90/1.10 exactly.",
        repair_was_available=True,
    ),
    (BDESH, "R_798"): Adjudication(
        "SHOULD_HAVE_BEEN_CORRECT", ("SOURCE_TEXT", "VERDICT_SCHEMA"),
        "Identical structure to R_797. Chunk: '+ 10%-15% for 230 kV and 132 kV buses during "
        "emergencies' is 0.85/1.10 exactly. Reason says 'which is correct ... requiring correction', "
        "verdict says REJECT.",
        repair_was_available=True,
    ),

    # -- NERC -----------------------------------------------------------------
    (NERC, "R_1444"): Adjudication(
        "WRONG", ("SOURCE_TEXT",),
        "Self-contradictory: it grants the source says '+/- 10%' and then calls 0.9/1.1 a deviation "
        "from that range. 0.9 and 1.1 ARE +/-10%. The appended rule-6 ride-through claim is also "
        "false - the chunk is a planning-study cascading screen, not a ride-through curve.",
    ),
    (NERC, "R_1575"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "PRC-006-5 R3 makes 25% a DESIGN ENVELOPE the UFLS scheme must survive, not a limit whose "
        "breach is a violation. Rendering it as a CONSTRAINT that BLOCKs is a genuine semantic error, "
        "and the reason names it precisely.",
    ),
    (NERC, "R_1833"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "PRC-024 Attachment tables are ride-through envelopes - they state how long a unit must NOT "
        "trip, not a continuous ceiling. Reading 1.10 pu as a standing limit inverts the standard's "
        "purpose.",
    ),
    (NERC, "R_1835"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "Low-voltage mirror of R_1833; same correct reading of the attachment as a duration curve.",
    ),
    (NERC, "R_1926"): Adjudication(
        "WRONG", ("CONSTRAINT_LOGIC", "VOCABULARY"),
        "Two errors, both load-bearing. (a) For role=CONSTRAINT the condition IS the violation "
        "predicate, so 'must be >= 0.95' correctly negates to 'voltage_pu_min < 0.95'; calling that "
        "'the opposite logical requirement' applies affirmation semantics to a constraint. "
        "(b) 'invalid variable name' is false - voltage_pu_min is in CONDITION_VOCABULARY.",
    ),
    (NERC, "R_1927"): Adjudication(
        "WRONG", ("CONSTRAINT_LOGIC",),
        "Same inversion error as R_1926, without the vocabulary fabrication. The chunk's "
        "'calls for a 0.95 per unit ... for the generator bus voltage' negates to exactly this condition.",
    ),

    # -- IEEE 1547 ------------------------------------------------------------
    (IEEE, "R_1098"): Adjudication(
        "WRONG", ("CONSTRAINT_LOGIC",),
        "Table 2.7 reads 'Minimum Value >= 0.917 pu'; for a CONSTRAINT the violation predicate is its "
        "negation, i.e. < 0.917. Calling that 'the opposite of the stated minimum requirement' is the "
        "same polarity error as R_1926/R_1927. (A sound rejection was available on other grounds - "
        "enter-service criteria are not an N-1 security limit - but that is not the ground given.)",
    ),

    # -- IBR performance guideline -------------------------------------------
    (IBR, "R_1108"): Adjudication(
        "SHOULD_HAVE_BEEN_CORRECT", ("SOURCE_TEXT", "VERDICT_SCHEMA"),
        "The role objection is right - the chunk frames 0.9-1.1 pu as the continuous operating range, "
        "and separately records inverters tripping at 1.2 pu, so CONSTRAINT at 0.9/1.1 misreads it. "
        "But 'CONSTRAINT should be AFFIRMATION' is a one-field repair, which is what CORRECT is for.",
        repair_was_available=True,
    ),
    (IBR, "R_1152"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "The chunk uses 0.9-1.1 pu parenthetically ('e.g.') while DEFINING the term 'large "
        "disturbance'. Promoting a definitional aside to a normative affirmation of `normal` is not a "
        "field-level repair, and the rule holds on 100% of frames on all three grids.",
    ),
    (IBR, "R_1319"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "'STATCOMs block at very low voltages around 0.2 to 0.3 pu' is device self-protection inside a "
        "ride-through range, not a grid limit. Matches the deterministic criterion-6 flag.",
    ),

    # -- PRC-024-4 / PRC-025-2 ------------------------------------------------
    ("prc-024-4", "R_2282"): Adjudication(
        "SOUND", ("SOURCE_TEXT",),
        "Attachment 2A Table 7 is a list of (voltage, duration) boundary points. Reading one endpoint "
        "as a static affirmation of `normal` drops the axis that makes it meaningful; the rule then "
        "holds on 100% of frames everywhere.",
    ),
    ("prc-025-2", "R_069"): Adjudication(
        "WRONG", ("CONSTRAINT_LOGIC", "VOCABULARY"),
        "Same pair of errors as R_1926: the inversion claim applies affirmation semantics to a "
        "CONSTRAINT, and 'uses an invalid variable' is false for voltage_pu_min. NOTE: findings "
        "12.4 files R_069 under 'correct role catch' - but that wording is the STRICT arm's reason, "
        "not this one.",
    ),
}


# ---------------------------------------------------------------------------
# Mechanical half — loading and deterministic checks
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_PY_KEYWORDS = {"and", "or", "not", "True", "False", "None", "abs", "min", "max"}


def document_stem(path: str, suffix: str) -> str:
    """Document stem from a per-document jsonl filename.

    The stem is the join key's first component and must survive the different
    suffixes each stage writes (`_rejected`, `_translated`, `_confirmed`).
    """
    return os.path.basename(path)[: -len(suffix)] if suffix else os.path.basename(path)


def load_rejections(directory: str | os.PathLike) -> Dict[Tuple[str, str], dict]:
    """Load every `*_rejected.jsonl` under `directory`, keyed on (document, rule_id).

    Keying on `rule_id` alone is a silent-mismatch bug: five source documents
    restarted numbering, so R_001..R_172 are reused across them
    (`thesis_findings.md` §19.2). A collision under the composite key is a real
    duplicate and raises.
    """
    out: Dict[Tuple[str, str], dict] = {}
    for path in sorted(glob.glob(os.path.join(str(directory), "*_rejected.jsonl"))):
        doc = document_stem(path, "_rejected.jsonl")
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec["rule"]["rule_id"]
                key = (doc, rid)
                if key in out:
                    raise ValueError(
                        f"duplicate (document, rule_id) key {key} at {path}:{lineno} — "
                        "the composite key is supposed to be unique"
                    )
                out[key] = rec
    return out


def load_guarded(directory: str | os.PathLike) -> Dict[Tuple[str, str], dict]:
    """Load the stage-2.5 guarded corpus, keyed the same way, for the source chunk."""
    out: Dict[Tuple[str, str], dict] = {}
    for path in sorted(glob.glob(os.path.join(str(directory), "*_translated.jsonl"))):
        doc = document_stem(path, "_translated.jsonl")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[(doc, rec["rule"]["rule_id"])] = rec
    return out


def condition_variables(condition: str) -> List[str]:
    """Identifiers a condition reads, minus Python keywords and builtins."""
    return [t for t in _IDENT.findall(condition or "") if t not in _PY_KEYWORDS]


def vocabulary_check(condition: str) -> dict:
    """Deterministic: does every identifier in the condition exist in the vocabulary?

    This is the one thing about a rejection that is decidable without reading
    anything. §12.3 showed 11 of the 31 `strict` rejections cited a
    vocabulary breach that does not exist; running the same test over the
    `translated` arm is how we find out whether that behaviour was confined to
    one arm.
    """
    used = condition_variables(condition)
    unknown = [v for v in used if v not in CONDITION_VOCABULARY]
    return {
        "variables_used": used,
        "unknown_variables": unknown,
        "all_in_vocabulary": not unknown,
    }


_VOCAB_CLAIM = re.compile(
    r"(invalid variable|undefined variable|not a valid variable|not defined in the allowed"
    r"|not listed in the allowed|variable not (?:listed|present|defined))",
    re.IGNORECASE,
)


def claims_vocabulary_breach(reason: str) -> Optional[str]:
    """The phrase in a reason that asserts a vocabulary violation, if any.

    Paired with `vocabulary_check`, this turns "the validator fabricated its
    reasoning" from an anecdote into a decidable test: a reason that asserts a
    breach against a condition whose identifiers are all in the vocabulary is
    fabricated, full stop.
    """
    m = _VOCAB_CLAIM.search(reason or "")
    return m.group(0) if m else None


_CORRECTION_CLAIM = re.compile(
    r"(requiring correction|requires correction|should be corrected|which is correct)",
    re.IGNORECASE,
)


def correction_language(reason: str) -> Optional[str]:
    """The phrase in a reason that concedes the rule was repairable, if any.

    §12.4 says four rejections' "reason text says 'requiring correction'". That
    is decidable. It is true of two of the four.
    """
    m = _CORRECTION_CLAIM.search(reason or "")
    return m.group(0) if m else None


@dataclass
class RejectionRecord:
    document: str
    rule_id: str
    mechanical: dict
    adjudication: dict


def build_records(
    translated: Dict[Tuple[str, str], dict],
    strict: Dict[Tuple[str, str], dict],
    guarded: Dict[Tuple[str, str], dict],
) -> List[RejectionRecord]:
    records: List[RejectionRecord] = []
    for (doc, rid), rec in sorted(translated.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rule = rec["rule"]
        verdict = rec["verdict"]
        reason = verdict.get("reason") or ""
        cond = rule.get("condition", "")
        vocab = vocabulary_check(cond)
        claim = claims_vocabulary_breach(reason)
        g = guarded.get((doc, rid), {})
        s = strict.get((doc, rid))

        mechanical = {
            "source_clause": rule.get("source"),
            "standard_document": doc,
            "condition": cond,
            "role": rule.get("role"),
            "affirms": rule.get("affirms"),
            "action": rule.get("action"),
            "severity": rule.get("severity"),
            "explanation": rule.get("explanation"),
            "fire_rate": {
                "neurips2020": rule.get("fire_rate_neurips2020"),
                "case14": rule.get("fire_rate_case14"),
                "wcci2022": rule.get("fire_rate_wcci2022"),
            },
            "verdict": verdict.get("verdict"),
            "reason_verbatim": reason,
            "corrected_fields": verdict.get("corrected_fields"),
            "also_rejected_by_strict_arm": s is not None,
            "strict_reason_verbatim": (s or {}).get("verdict", {}).get("reason"),
            "source_chunk": g.get("chunk"),
            "vocabulary_check": vocab,
            # The decidable fabrication test. True == the reason asserts a
            # vocabulary breach that provably does not exist.
            "reason_claims_vocabulary_breach": claim,
            "vocabulary_claim_is_fabricated": bool(claim) and vocab["all_in_vocabulary"],
            "reason_concedes_correction": correction_language(reason),
            "named_in_findings_12_4": FINDINGS_12_4_TABLE.get(rid),
            "named_in_findings_12_4_verdict_defect_row": rid in FINDINGS_12_4_VERDICT_DEFECT,
        }

        adj = ADJUDICATIONS.get((doc, rid))
        if adj is None:
            adjudication = {
                "category": "UNVERIFIABLE",
                "basis": [],
                "justification": "no adjudication recorded for this (document, rule_id)",
                "repair_was_available": False,
                "method": "reading, not measurement",
            }
        else:
            adjudication = {
                "category": adj.category,
                "basis": list(adj.basis),
                "justification": adj.justification,
                "repair_was_available": adj.repair_was_available,
                "method": "reading, not measurement",
            }

        records.append(RejectionRecord(doc, rid, mechanical, adjudication))
    return records


def reconcile(records: List[RejectionRecord]) -> dict:
    """Compare the adjudication against what §12.4 committed to."""
    counts = {c: 0 for c in CATEGORIES}
    for r in records:
        counts[r.adjudication["category"]] += 1

    wrong = [r.rule_id for r in records if r.adjudication["category"] == "WRONG"]
    shbc = [r.rule_id for r in records
            if r.adjudication["category"] == "SHOULD_HAVE_BEEN_CORRECT"]
    repairable = [r.rule_id for r in records if r.adjudication["repair_was_available"]]
    unnamed = [r.rule_id for r in records
               if r.mechanical["named_in_findings_12_4"] is None]
    fabricated = [r.rule_id for r in records
                  if r.mechanical["vocabulary_claim_is_fabricated"]]

    discrepancies: List[str] = []
    if len(records) != DOCUMENTED_TOTAL_REJECTIONS:
        discrepancies.append(
            f"rejection count is {len(records)}, documented {DOCUMENTED_TOTAL_REJECTIONS}"
        )
    if counts["WRONG"] != DOCUMENTED_MIN_WRONG:
        discrepancies.append(
            f"WRONG adjudicated {counts['WRONG']}, documented 'at least {DOCUMENTED_MIN_WRONG}' "
            f"({', '.join(wrong)}). Literally compatible with 'at least', but it breaks "
            f"§12.4's companion sentence that 'roughly 17 of 21 rest on a defensible reading' — "
            f"this reading gives {counts['SOUND']}."
        )
    if counts["SHOULD_HAVE_BEEN_CORRECT"] != DOCUMENTED_SHOULD_HAVE_BEEN_CORRECT:
        discrepancies.append(
            f"SHOULD_HAVE_BEEN_CORRECT adjudicated {counts['SHOULD_HAVE_BEEN_CORRECT']}, "
            f"documented {DOCUMENTED_SHOULD_HAVE_BEEN_CORRECT}"
        )
    elif set(shbc) != set(FINDINGS_12_4_VERDICT_DEFECT):
        discrepancies.append(
            f"SHOULD_HAVE_BEEN_CORRECT count matches ({len(shbc)}) but MEMBERSHIP does not. "
            f"Adjudicated {sorted(shbc)}; §12.4's verdict-selection-defect row names "
            f"{sorted(FINDINGS_12_4_VERDICT_DEFECT)}. Differences: "
            f"only-here={sorted(set(shbc) - set(FINDINGS_12_4_VERDICT_DEFECT))}, "
            f"only-§12.4={sorted(set(FINDINGS_12_4_VERDICT_DEFECT) - set(shbc))}."
        )

    # §12.4 asserts the four verdict-defect rules' "reason text says 'requiring
    # correction'". Decidable, and false for two of them.
    concedes = [r.rule_id for r in records if r.mechanical["reason_concedes_correction"]]
    missing_concession = [rid for rid in FINDINGS_12_4_VERDICT_DEFECT if rid not in concedes]
    if missing_concession:
        discrepancies.append(
            f"§12.4 says the verdict-selection-defect rules' 'reason text says \"requiring "
            f"correction\"'. Only {sorted(concedes)} contain correction language; "
            f"{sorted(missing_concession)} do not — their translated-arm reasons make no such "
            f"concession. The claim holds for {len(concedes)} of "
            f"{len(FINDINGS_12_4_VERDICT_DEFECT)} named rules."
        )

    # §12.4 quotes reasons that are verbatim from the STRICT arm for rules whose
    # translated-arm reason argues something else. Flag any rule §12.4 assessed
    # as "correct" whose translated reason concedes the rule was right.
    misattributed = [r.rule_id for r in records
                     if r.mechanical["reason_concedes_correction"]
                     and "assessed: correct" in (r.mechanical["named_in_findings_12_4"] or "")]
    if misattributed:
        discrepancies.append(
            f"§12.4 files {sorted(misattributed)} under an 'assessed: correct' row, but their "
            f"TRANSLATED-arm reasons concede the condition 'is correct'. The wording §12.4 quotes "
            f"for those rows is the STRICT arm's reason, not the served arm's."
        )
    if unnamed:
        discrepancies.append(
            f"§12.4's table names {len(FINDINGS_12_4_TABLE)} of {len(records)} rejections; "
            f"{len(unnamed)} appear in no row: {', '.join(unnamed)}"
        )
    if fabricated:
        discrepancies.append(
            f"{len(fabricated)} of {len(records)} TRANSLATED-arm rejections assert a vocabulary "
            f"breach that the deterministic check disproves ({', '.join(fabricated)}). §12.3 "
            f"presents fabrication as a `strict`-arm phenomenon; it is not confined to that arm."
        )

    return {
        "counts": counts,
        "wrong": wrong,
        "should_have_been_correct": shbc,
        "repair_was_available": repairable,
        "not_named_in_findings_12_4": unnamed,
        "fabricated_vocabulary_claims": fabricated,
        "reasons_conceding_correction": concedes,
        "documented": {
            "min_wrong": DOCUMENTED_MIN_WRONG,
            "should_have_been_correct": DOCUMENTED_SHOULD_HAVE_BEEN_CORRECT,
            "total_rejections": DOCUMENTED_TOTAL_REJECTIONS,
        },
        "reconciles": not discrepancies,
        "discrepancies": discrepancies,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(records: List[RejectionRecord], rec: dict) -> None:
    print()
    print("=" * 78)
    print("  VALIDATOR REJECTION ADJUDICATION — translated arm, 21 shared rejections")
    print("=" * 78)
    print()
    print("  MECHANICAL fields are verbatim from disk or deterministic.")
    print("  ADJUDICATION is a reading — a judgement, not a measurement.")
    print()

    header = f"  {'rule':8} {'document':26} {'category':24} {'repair?':8} fabricated?"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(records, key=lambda x: (x.adjudication["category"], x.rule_id)):
        doc = r.document[:24] + ".." if len(r.document) > 26 else r.document
        print(
            f"  {r.rule_id:8} {doc:26} {r.adjudication['category']:24} "
            f"{'yes' if r.adjudication['repair_was_available'] else '-':8} "
            f"{'YES' if r.mechanical['vocabulary_claim_is_fabricated'] else '-'}"
        )
    print()
    print("  counts: " + "  ".join(f"{k}={v}" for k, v in rec["counts"].items()))
    print(f"  repair was available in {len(rec['repair_was_available'])} of {len(records)} "
          f"({', '.join(rec['repair_was_available'])})")
    print()

    if rec["reconciles"]:
        print("  RECONCILES with thesis_findings.md §12.4.")
    else:
        print("  " + "!" * 74)
        print("  !! DISCREPANCY against thesis_findings.md §12.4 — DO NOT ABSORB, REPORT")
        print("  " + "!" * 74)
        for d in rec["discrepancies"]:
            print(f"   - {d}")
    print()

    for cat in CATEGORIES:
        members = [r for r in records if r.adjudication["category"] == cat]
        if not members:
            continue
        print(f"  -- {cat} ({len(members)}) " + "-" * max(0, 56 - len(cat)))
        for r in members:
            print(f"     {r.rule_id:8} {r.mechanical['condition'][:58]}")
            print(f"              reason  : \"{r.mechanical['reason_verbatim'][:150]}\"")
            print(f"              call    : {r.adjudication['justification'][:150]}")
            print(f"              basis   : {', '.join(r.adjudication['basis'])}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--translated", default=str(DEFAULT_TRANSLATED),
                    help="stage-3 translated-arm output dir (the served arm)")
    ap.add_argument("--strict", default=str(DEFAULT_STRICT),
                    help="stage-3 strict-arm output dir (the counterfactual)")
    ap.add_argument("--guarded", default=str(DEFAULT_GUARDED),
                    help="stage-2.5 guarded corpus, for the source chunk")
    ap.add_argument("--json", default=str(REPO / "results" / "audit" / "rejection_adjudication.json"))
    args = ap.parse_args()

    translated = load_rejections(args.translated)
    strict = load_rejections(args.strict)
    guarded = load_guarded(args.guarded)
    log.info(f"translated rejections: {len(translated)}  "
             f"strict rejections: {len(strict)}  guarded rules: {len(guarded)}")

    records = build_records(translated, strict, guarded)
    rec = reconcile(records)
    print_summary(records, rec)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "what": "per-rejection adjudication of the 21 shared stage-3 rejections",
            "arm": "translated (--prompt-variant translated) — the arm that produced the served corpus",
            "method": {
                "mechanical": "verbatim from validated_translated/*_rejected.jsonl and "
                              "translated_rules/guarded/*_translated.jsonl; the vocabulary check is "
                              "set membership in extraction.common.CONDITION_VOCABULARY",
                "adjudication": "READING, NOT MEASUREMENT. A human classification with a stated "
                                "basis. Disagree with a call by its rule key; the mechanical half "
                                "is unaffected.",
                "join_key": "(document, rule_id) — rule_id is NOT globally unique (§19.2)",
            },
            "categories": {
                "SOUND": "stated ground is accurate about the source text and the conclusion follows",
                "WRONG": "stated ground contains a load-bearing factual error",
                "SHOULD_HAVE_BEEN_CORRECT": "stated defect is a field-level repair; CORRECT existed "
                                            "as a verdict and was not used",
                "UNVERIFIABLE": "cannot be adjudicated from the artifacts alone",
            },
            "reconciliation": rec,
            "rejections": [asdict(r) for r in records],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        log.info(f"Wrote {args.json}")

        md = Path(args.json).with_suffix(".md")
        md.write_text(render_markdown(records, rec), encoding="utf-8")
        log.info(f"Wrote {md}")


def render_markdown(records: List[RejectionRecord], rec: dict) -> str:
    lines = [
        "# Validator rejection adjudication — the 21 shared rejections",
        "",
        "Generated by `evaluation/audit_rejections.py`. Arm: `--prompt-variant translated`,",
        "the arm that produced the served four-rule corpus.",
        "",
        "**MECHANICAL columns are verbatim from disk. The `call` column is a READING —",
        "a judgement, not a measurement.** Each call carries its own basis so an individual",
        "call can be disputed without discarding the artifact.",
        "",
        "| rule | document | condition | call | repair available | fabricated vocab claim |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda x: (x.adjudication["category"], x.rule_id)):
        lines.append(
            f"| `{r.rule_id}` | {r.document[:34]} | `{r.mechanical['condition']}` | "
            f"**{r.adjudication['category']}** | "
            f"{'yes' if r.adjudication['repair_was_available'] else '—'} | "
            f"{'**YES**' if r.mechanical['vocabulary_claim_is_fabricated'] else '—'} |"
        )
    lines += ["", "## Counts", ""]
    for k, v in rec["counts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Reconciliation against `thesis_findings.md` §12.4", ""]
    if rec["reconciles"]:
        lines.append("Reconciles.")
    else:
        lines.append("**DOES NOT RECONCILE.**")
        lines.append("")
        for d in rec["discrepancies"]:
            lines.append(f"- {d}")
    lines += ["", "## Per-rejection detail", ""]
    for cat in CATEGORIES:
        members = [r for r in records if r.adjudication["category"] == cat]
        if not members:
            continue
        lines += [f"### {cat} ({len(members)})", ""]
        for r in members:
            lines += [
                f"#### `{r.rule_id}` — {r.document}",
                "",
                f"- **source clause** (mechanical): {r.mechanical['source_clause']}",
                f"- **condition** (mechanical): `{r.mechanical['condition']}` "
                f"[role `{r.mechanical['role']}`]",
                f"- **validator reason, verbatim** (mechanical): "
                f"> {r.mechanical['reason_verbatim']}",
                f"- **call** (adjudication, a reading): **{r.adjudication['category']}** — "
                f"{r.adjudication['justification']}",
                f"- **basis**: {', '.join(r.adjudication['basis']) or '—'}",
                "",
            ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
