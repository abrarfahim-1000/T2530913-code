"""
Assign every expressible rule to an output channel.

The shield currently has one channel: a rule either BLOCKS or it is discarded.
That is why 54 of 58 expressible rules are on the floor -- not because they are
wrong, but because "blocks a prediction" was the only thing a rule was allowed
to do. A rule that is a faithful reading of a standard and evaluable on the
telemetry still has something to say when it cannot justify a veto.

This pass re-partitions all 58 into four channels, deterministically, from
measurements already on disk. No model is called and no stage is re-run.

    BLOCK           validated CONSTRAINT. Vetoes an over-permissive prediction.
                    DELIBERATELY UNCHANGED -- this set is what the 92-94%
                    intervention-precision result is measured on, and admitting
                    anything here moves that number.

    WARN            fires, discriminates, but was not validated as authoritative.
                    Annotates without changing the prediction. Must carry its
                    measured rate: a warning without a rate is an alarm.

    NORMAL          AFFIRMATION with support on its own class. "Telemetry is
                    consistent with normal operation per <clause>."

    NOT_APPLICABLE  correct rule, evaluable, whose CALIBRATION does not transfer
                    to these grids. Almost entirely the +/-5% voltage bands: the
                    reference grids operate ~6% above nominal (findings 11.2),
                    so a band written for utility practice is degenerate here.
                    Recorded and cited rather than silently dropped.

    INERT           cannot fire on any class on any grid. Excluded, and counted.

Channels never mix: a rule outside BLOCK is structurally incapable of vetoing.

Run:  .venv\\Scripts\\python.exe evaluation\\readmit_rules.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.audit_rules import audit_rule  # noqa: E402
from extraction.polarity_guard import sample_contexts  # noqa: E402

log = logging.getLogger("readmit")

DEFAULT_TAGS = ("neurips2020", "case14", "wcci2022")
LABELS = ("normal", "overload", "line_trip", "cascade")
SAMPLE = 5000

# A constraint firing on this share of HEALTHY frames on every grid is not
# discriminating between states -- it is reporting that the grid's operating
# point sits outside the band the rule was written for.
DEGENERATE_HIGH = 0.98
DEGENERATE_LOW = 0.02


def load_jsonl(pattern: str) -> list[dict]:
    out = []
    for path in sorted(Path(".").glob(pattern)):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def rule_of(rec: dict) -> dict:
    return rec.get("rule", rec)


MIN_DISCRIMINATION = 0.05   # matches evaluation/audit_rules.py


def grid_verdicts(audit) -> dict[str, str]:
    """Per-grid behaviour of a CONSTRAINT: 'discriminates' | 'degenerate' | 'silent'.

    Decided per grid rather than over the corpus, because the interesting case
    is a rule that works on some topologies and misfires on others. Collapsing
    to one verdict loses exactly that rule -- R_128 fires on 0.1% of healthy
    neurips frames and 98.6% of healthy wcci frames, which is one statement
    about the rule and a different statement about each grid.
    """
    out = {}
    for tag, by in audit.rates.items():
        if "normal" not in by:
            continue
        healthy = by["normal"]
        abnormal = [r for lab, r in by.items() if lab != "normal"]
        lift = (max(abnormal) - healthy) if abnormal else 0.0
        if healthy >= DEGENERATE_HIGH:
            out[tag] = "degenerate"          # fires on healthy grids: cannot mean "violation"
        elif lift >= MIN_DISCRIMINATION:
            out[tag] = "discriminates"
        elif healthy <= DEGENERATE_LOW and (not abnormal or max(abnormal) <= DEGENERATE_LOW):
            out[tag] = "silent"              # never fires on anything
        else:
            out[tag] = "degenerate"
    return out


def assign_channel(rule: dict, audit, guard_kept: bool, validated: str) -> tuple[str, str]:
    """Return (channel, one-line justification). Deterministic."""
    role = (rule.get("role") or "CONSTRAINT").upper()
    verdict = audit.verdict
    healthy = [by.get("normal") for by in audit.rates.values() if "normal" in by]

    if validated == "CONFIRM" and role == "CONSTRAINT":
        return "BLOCK", "validated constraint; authoritative veto"

    if role == "AFFIRMATION":
        if verdict in ("USEFUL", "TOPOLOGY_DEPENDENT"):
            return "NORMAL", f"affirmation with support on its own class ({verdict})"
        if healthy and all(r <= DEGENERATE_LOW for r in healthy):
            return ("NOT_APPLICABLE",
                    "affirms `normal` but never holds on a healthy frame; band sits "
                    "outside this grid's operating envelope (findings 11.2)")
        if verdict == "INERT":
            return "INERT", "never fires on any class on any grid"
        return "NORMAL", f"affirmation, weak support ({verdict})"

    # ── CONSTRAINT, not validated as authoritative ────────────────────────────
    if verdict == "INERT":
        return "INERT", "never fires on any class on any grid"

    gv = grid_verdicts(audit)
    good = sorted(t for t, v in gv.items() if v == "discriminates")
    bad = sorted(t for t, v in gv.items() if v == "degenerate")

    if good:
        scope = "on every grid" if not bad else f"on {', '.join(good)} (degenerate on {', '.join(bad)})"
        return "WARN", f"discriminates {scope}; not validated as authoritative"

    if bad:
        return ("NOT_APPLICABLE",
                f"degenerate on every grid ({', '.join(bad)}): fires regardless of state, "
                "so its calibration does not transfer (findings 11.2)")

    if not guard_kept:
        return ("NOT_APPLICABLE",
                "rejected by the polarity guard and discriminates nowhere")
    return "WARN", f"expressible and clean, low information ({verdict})"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    translated = load_jsonl("translated_rules/*_translated.jsonl")
    guarded = load_jsonl("translated_rules/guarded/*_translated.jsonl")
    confirmed = load_jsonl("validated_translated/*_confirmed.jsonl")
    rejected = load_jsonl("validated_translated/*_rejected.jsonl")

    guard_ids = {rule_of(r).get("rule_id") for r in guarded}
    conf_ids = {rule_of(r).get("rule_id") for r in confirmed}
    rej_ids = {rule_of(r).get("rule_id") for r in rejected}

    # The guard is what attaches `role`; stage-2 output has none. Prefer the
    # guarded/polarity-rejected copy of a rule, which carries it.
    enriched: dict[str, dict] = {}
    for rec in load_jsonl("translated_rules/guarded/*_polarity_rejected.jsonl") + guarded:
        r = rule_of(rec)
        enriched[r.get("rule_id")] = r
    rules = []
    for rec in translated:
        r = rule_of(rec)
        rules.append(enriched.get(r.get("rule_id"), r))

    log.info(f"expressible rules to re-partition: {len(rules)}")

    contexts = {}
    for tag in DEFAULT_TAGS:
        try:
            contexts[tag] = sample_contexts(tag, labels=LABELS, n=SAMPLE, data_dir="data")
        except (FileNotFoundError, ValueError) as exc:
            log.warning(f"[{tag}] unavailable - skipping: {exc}")
    if not contexts:
        log.error("no topology produced contexts")
        return 1

    rows = []
    for rule in rules:
        rid = rule.get("rule_id")
        validated = "CONFIRM" if rid in conf_ids else ("REJECT" if rid in rej_ids else "n/a")
        audit = audit_rule(rule, contexts)
        channel, why = assign_channel(rule, audit, rid in guard_ids, validated)
        rows.append({
            "rule_id": rid,
            "channel": channel,
            "justification": why,
            "role": rule.get("role"),
            "condition": rule.get("condition"),
            "source": rule.get("source"),
            "severity": rule.get("severity"),
            "audit_verdict": audit.verdict,
            "guard_kept": rid in guard_ids,
            "validation": validated,
            "rates": audit.rates,
            "per_grid": grid_verdicts(audit) if (rule.get("role") or "CONSTRAINT").upper() == "CONSTRAINT" else {},
        })

    counts = Counter(r["channel"] for r in rows)
    order = ("BLOCK", "WARN", "NORMAL", "NOT_APPLICABLE", "INERT")
    # Rule records are not distinct predicates: 10 BLOCK records are 3 distinct
    # conditions restated across standards. Report both or the count inflates.
    distinct = {ch: len({(r["condition"] or "").strip()
                         for r in rows if r["channel"] == ch}) for ch in order}

    print("\n" + "=" * 76)
    print("RE-ADMISSION — all 58 expressible rules, partitioned by channel")
    print("=" * 76)
    print()
    print(f"  {'channel':<18}{'rules':>7}{'distinct':>10}{'today':>7}   what it does")
    print("  " + "-" * 76)
    today = {"BLOCK": 4, "WARN": 0, "NORMAL": 1, "NOT_APPLICABLE": 0, "INERT": 0}
    blurb = {
        "BLOCK": "vetoes an over-permissive prediction",
        "WARN": "annotates, never vetoes; carries its measured rate",
        "NORMAL": "affirms telemetry is consistent with normal operation",
        "NOT_APPLICABLE": "correct rule, calibration does not transfer (11.2)",
        "INERT": "cannot fire on any class on any grid; excluded",
    }
    for ch in order:
        print(f"  {ch:<18}{counts.get(ch,0):>7}{distinct[ch]:>10}{today.get(ch,0):>7}   {blurb[ch]}")
    print("  " + "-" * 76)
    speaking = sum(counts.get(c, 0) for c in ("BLOCK", "WARN", "NORMAL"))
    speak_distinct = len({(r["condition"] or "").strip() for r in rows
                          if r["channel"] in ("BLOCK", "WARN", "NORMAL")})
    print(f"  {'rules that SPEAK':<18}{speaking:>7}{speak_distinct:>10}{5:>7}   (BLOCK + WARN + NORMAL)")
    doc_n = speaking + counts.get("NOT_APPLICABLE", 0)
    doc_d = len({(r["condition"] or "").strip() for r in rows
                 if r["channel"] != "INERT"})
    print(f"  {'documented':<18}{doc_n:>7}{doc_d:>10}{5:>7}   (+ NOT_APPLICABLE, cited not dropped)")

    for ch in order:
        sel = [r for r in rows if r["channel"] == ch]
        if not sel:
            continue
        print(f"\n--- {ch} ({len(sel)}) ---")
        for r in sel:
            print(f"  {r['rule_id']:<9}{(r['condition'] or '')[:54]:<56}{r['audit_verdict']}")

    out = {
        "n_expressible": len(rows),
        "channel_counts": {c: counts.get(c, 0) for c in order},
        "channel_distinct_conditions": distinct,
        "counts_today": today,
        "n_speaking": speaking,
        "degenerate_high": DEGENERATE_HIGH,
        "degenerate_low": DEGENERATE_LOW,
        "sample_size": SAMPLE,
        "tags": list(contexts),
        "note": ("BLOCK is deliberately unchanged: it is the set the 92-94% "
                 "intervention-precision result is measured on. WARN rules must "
                 "carry a measured rate before use and are structurally barred "
                 "from the veto path."),
        "rules": rows,
    }
    dest = Path("results/audit/readmission.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
