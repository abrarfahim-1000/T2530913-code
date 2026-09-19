"""
What the 2,463 extracted candidates are ABOUT, and whether Grid2Op can see it.

`corpus_accounting.py` partitions the corpus by where each candidate DIED —
the translator's own stated reason. That answers "why did the pipeline drop
it?". It does not answer "what kind of rule was it?", because the rejection
reason is a property of the pipeline, not of the standard.

This script partitions the same 2,463 candidates by SUBJECT: the physical
quantity the rule constrains. The two views disagree productively. A voltage
ride-through curve is filed under `time / dynamics` by the rejection reason
and under `Voltage magnitude & bands` here, and the second is the one that
tells you the family is 22.7% of the corpus and mostly unreachable anyway.

The output feeds `supplimentary_docs/rule_corpus_vs_grid2op.pdf`.

Run:  .venv\\Scripts\\python.exe evaluation\\subject_families.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.corpus_accounting import key_of, load_jsonl, rule_of  # noqa: E402

OUT_JSON = Path("results/audit/subject_families.json")

# Ordered, FIRST MATCH WINS, and THE ORDER IS LOAD-BEARING.
#
# Matched against the condition's variable tokens first, then against the
# explanation prose. The order encodes "what is this rule fundamentally about"
# when a condition mentions several quantities at once:
#
#   frequency before everything    -- a frequency rule with a time limit is a
#                                     FREQUENCY rule; Grid2Op models no frequency
#                                     at any tier, so nothing downstream matters
#   protection before voltage      -- "undervoltage relay setting" is an equipment
#                                     setting, not a voltage band the grid must hold
#   voltage before time            -- a ride-through curve is a VOLTAGE requirement
#                                     that happens to be time-bound; filing it under
#                                     time would hide the largest expressible family
#   thermal before active power    -- "flow exceeds rating" is a thermal limit even
#                                     when the flow is written in MW
#   timing-only near the end       -- it must catch only what has NO physical
#                                     quantity left, or it swallows half the corpus
#   administrative last            -- its keywords (test_, study, category) appear
#                                     incidentally in physical rules
#
# `time_bound` is reported as a SEPARATE column rather than a family, because a
# time limit is an attribute of a rule, not its subject. 547 of 2,463 candidates
# carry one.
FAMILIES: list[tuple[str, re.Pattern]] = [
    ("Frequency / RoCoF / governor droop",
     re.compile(r"frequenc|freq_|\bhz\b|rocof|droop|deadband|lfsm|fsm|"
                r"v_per_hz|primary_frequency|synthetic_inertia|inertia")),
    ("Protection & equipment settings",
     re.compile(r"impedance|overcurrent|relay|breaker|fault_clear|fault_current|"
                r"winding|hot_spot|temp_c|stator|rotor|excitation|tap_chang|"
                r"insulat|withstand|short_circuit|earth_fault|switchgear|"
                r"protection|trip_setting|auto_reclos")),
    ("Voltage magnitude & bands",
     re.compile(r"voltage|_pu\b|^pu_|retained_volt|terminal_volt|v_min|v_max|"
                r"undervolt|overvolt|u_min|u_max")),
    ("Reactive power & power factor",
     re.compile(r"power_factor|\bpf\b|reactive|mvar|q_max|q_min|cos_phi")),
    ("Thermal loading & current",
     re.compile(r"loading|rho|current_a|current_pu|\bamp|thermal|rating|"
                r"\bmva\b|apparent_power|capacity_mw|loading_duration")),
    ("Active power, dispatch & capacity",
     re.compile(r"active_power|power_output|\bmw\b|\bp_\w|generation|dispatch|"
                r"ramp|output_pct|registered_capacity|rated")),
    ("Load shedding & system balance",
     re.compile(r"load_shed|shed|imbalance|reserve|demand|curtail|restoration|"
                r"blackstart|black_start|island")),
    ("Power quality (flicker, harmonics, unbalance)",
     re.compile(r"flicker|harmonic|thd|unbalance|distortion|overshoot|"
                r"voltage_variation|step_change")),
    ("Topology, connectivity & switching",
     re.compile(r"topolog|busbar|\bbus\b|switching|connect|circuit_count|"
                r"outage|contingenc|n_minus|redundan")),
    ("Timing / duration only (no physical quantity)",
     re.compile(r"time_|duration|_ms\b|_seconds|_hours|delay|cycle|response_time|"
                r"reaction_time|settling")),
    ("Administrative / process / reporting",
     re.compile(r"notif|report|document|complian|agreement|approv|register|"
                r"submit|record|audit|test_|study|procedure|annual|impacted_entit|"
                r"is_offshore|is_large|type_[a-d]|category")),
]

UNCLASSIFIED = "Unclassified"

TIME_BOUND = re.compile(
    r"time_|duration|_ms\b|_seconds|_hours|delay|cycle|response_time")

# Whether a Grid2Op observation can evaluate the family at all. Hand-labelled
# against shield/context.py's CONTEXT_VARIABLES and CompleteObservation's
# attr_list_vect, and reported as a figure — so it is stated here, not inferred.
OBSERVABLE: dict[str, str] = {
    "Frequency / RoCoF / governor droop": "no",
    "Protection & equipment settings": "no",
    "Voltage magnitude & bands": "yes",
    "Reactive power & power factor": "yes",
    "Thermal loading & current": "yes",
    "Active power, dispatch & capacity": "yes",
    "Load shedding & system balance": "partial",
    "Power quality (flicker, harmonics, unbalance)": "no",
    "Topology, connectivity & switching": "yes",
    "Timing / duration only (no physical quantity)": "partial",
    "Administrative / process / reporting": "no",
    UNCLASSIFIED: "no",
}

# The stage totals this partition must reproduce (corpus_accounting.py).
EXPECTED = {"candidates": 2463, "translated": 58, "guarded": 32, "validated": 11}


def subject_of(rule: dict) -> str:
    """The family a candidate belongs to. Formal content wins over prose."""
    cond = str(rule.get("condition", ""))
    toks = " ".join(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", cond)).lower()
    for name, pat in FAMILIES:
        if pat.search(toks):
            return name
    prose = f"{rule.get('explanation', '')} {rule.get('source', '')}".lower()
    for name, pat in FAMILIES:
        if pat.search(prose):
            return name
    return UNCLASSIFIED


def is_time_bound(rule: dict) -> bool:
    return bool(TIME_BOUND.search(str(rule.get("condition", "")).lower()))


def main() -> int:
    root = Path(".")

    candidates = load_jsonl("rules_35b/*_candidates.jsonl", root)
    ids_tr = {key_of(r) for r in load_jsonl("translated_rules/*_translated.jsonl", root)}
    ids_gu = {key_of(r) for r in
              load_jsonl("translated_rules/guarded/*_translated.jsonl", root)}
    ids_cf = {key_of(r) for r in
              load_jsonl("validated_translated/*_confirmed.jsonl", root)}

    total = Counter()
    timed = Counter()
    trans = Counter()
    guard = Counter()
    valid = Counter()

    for rec in candidates:
        rule = rule_of(rec)
        fam = subject_of(rule)
        key = key_of(rec)
        total[fam] += 1
        if is_time_bound(rule):
            timed[fam] += 1
        if key in ids_tr:
            trans[fam] += 1
        if key in ids_gu:
            guard[fam] += 1
        if key in ids_cf:
            valid[fam] += 1

    order = [n for n, _ in FAMILIES] + [UNCLASSIFIED]
    order.sort(key=lambda f: -total[f])
    n_cand = len(candidates)

    print("=" * 96)
    print("SUBJECT FAMILIES — what the 2,463 candidates are about")
    print("=" * 96)
    print(f"\n{'subject family':<46}{'n':>6}{'%':>7}{'time':>7}"
          f"{'obsrv':>8}{'transl':>8}{'guard':>7}{'valid':>7}")
    print("-" * 96)
    for fam in order:
        if not total[fam]:
            continue
        print(f"{fam:<46}{total[fam]:>6}{100 * total[fam] / n_cand:>6.1f}%"
              f"{timed[fam]:>7}{OBSERVABLE[fam]:>8}{trans[fam]:>8}"
              f"{guard[fam]:>7}{valid[fam]:>7}")
    print("-" * 96)
    print(f"{'TOTAL':<46}{n_cand:>6}{100.0:>6.1f}%{sum(timed.values()):>7}"
          f"{'':>8}{sum(trans.values()):>8}{sum(guard.values()):>7}"
          f"{sum(valid.values()):>7}")

    # Tripwire: the families must re-partition the corpus and reproduce the
    # stage totals exactly. A regex edit that quietly drops or double-counts a
    # candidate is otherwise invisible in a table of plausible-looking numbers.
    got = {
        "candidates": n_cand,
        "translated": sum(trans.values()),
        "guarded": sum(guard.values()),
        "validated": sum(valid.values()),
    }
    problems = [f"{k}: got {v}, expected {EXPECTED[k]}"
                for k, v in got.items() if v != EXPECTED[k]]
    if sum(total.values()) != n_cand:
        problems.append(f"families sum to {sum(total.values())}, expected {n_cand}")
    if problems:
        print("\nPARTITION BROKEN:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n  partition verified: every candidate in exactly one family, "
          "stage totals reconcile")

    payload = {
        "n_candidates": n_cand,
        "expected_stage_totals": EXPECTED,
        "families": [
            {
                "family": fam,
                "n": total[fam],
                "share": round(total[fam] / n_cand, 4),
                "time_bound": timed[fam],
                "grid2op_observable": OBSERVABLE[fam],
                "translated": trans[fam],
                "guarded": guard[fam],
                "validated": valid[fam],
            }
            for fam in order if total[fam]
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
