"""
Build the v2 shield corpus: every expressible rule, stamped with its channel and,
for warnings, the N-1 rate it earned.

Three measurements already exist and were made independently. This joins them
into the single artifact the gate loads:

    results/audit/readmission.json          which channel each rule belongs in
                                            (thesis_findings 19.3)
    results/audit/warn_n1_calibration.json  whether a WARN predicate actually
                                            predicts N-1 risk (21.2)
    validated_translated/                   the validated rule bodies
    translated_rules/guarded/               the rest of the rule bodies, with the
                                            `role`/`affirms` the guard attaches

The join is where the calibration is ENFORCED rather than merely reported. A WARN
predicate that measured INVERTED or INSUFFICIENT is demoted to NOT_APPLICABLE
here, carrying the measured reason, so it can never be rendered as a warning at
inference. 21.2 found two INVERTED predicates -- the whole power-factor family,
which fires more often when N-1 risk is *lower* -- so this is not defensive
programming against a hypothetical.

Output: shield_corpus/all_rules_channels.jsonl

It is a SUPERSET of validated_translated/all_rules_deduped.jsonl, not a
replacement. The BLOCK channel is copied through unchanged and the eval harness
must produce identical numbers from either file; `--verify` asserts the BLOCK set
matches the served corpus condition-for-condition before writing.

Run:  .venv\\Scripts\\python.exe evaluation\\build_channel_corpus.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shield.channels import (  # noqa: E402
    BLOCK,
    INERT,
    NORMAL,
    NOT_APPLICABLE,
    WARN,
    assert_warnings_calibrated,
)

SERVED = Path("validated_translated/all_rules_deduped.jsonl")


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


def rule_bodies() -> dict[str, dict]:
    """rule_id -> the fullest rule dict available.

    Later sources win, so a rule that reached validation is served with the body
    validation saw. `role` and `affirms` come from the guard, which is the only
    stage that attaches them.
    """
    bodies: dict[str, dict] = {}
    for pattern in ("translated_rules/*_translated.jsonl",
                    "translated_rules/guarded/*_polarity_rejected.jsonl",
                    "translated_rules/guarded/*_translated.jsonl",
                    "validated_translated/*_confirmed.jsonl"):
        for rec in load_jsonl(pattern):
            rule = rec.get("rule", rec)
            rid = rule.get("rule_id")
            if rid:
                bodies[rid] = {**bodies.get(rid, {}), **rule}
    return bodies


def calibration_index(path: Path) -> dict[str, dict]:
    """condition -> the pooled calibration row, keyed exactly as 21.2 reports it."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["condition"].strip(): c for c in data.get("conditions", [])}


def contributing_rates(cal: dict, pooled: dict) -> dict:
    """The rates a warning may quote — taken from the grids that scored it.

    NOT from `pooled.p_violation_given_fires`. The pooled rate is the naive one,
    computed over the summed counts; only `pooled.discrimination` is stratified.
    Shipping them together produced a warning that read "21.0% when this fired
    against 24.5% when it did not" on a predicate labelled ELEVATED — the
    confounded pair, and backwards. The contributing grid's own rates are
    +0.474 / 0.245, which is what the rule actually earned.

    With several contributing grids the two rates are averaged over frames, the
    same weighting `warn_rule_calibration.stratified` uses, so the quoted pair
    and the quoted discrimination come from one calculation.
    """
    per_grid = cal.get("per_grid") or {}
    contributing = pooled.get("grids_contributing") or []
    num_f = num_q = den = 0.0
    for tag in contributing:
        row = per_grid.get(tag) or {}
        w = row.get("frames_fired") or 0
        if not w or row.get("p_violation_given_fires") is None:
            continue
        num_f += row["p_violation_given_fires"] * w
        num_q += (row.get("p_violation_given_silent") or 0.0) * w
        den += w
    return {
        "verdict": pooled.get("verdict"),
        "p_violation_given_fires": (num_f / den) if den else None,
        "p_violation_given_silent": (num_q / den) if den else None,
        "discrimination": pooled.get("discrimination"),
        "grids": list(contributing),
        "frames_fired": int(den),
        "rates_are_from_contributing_grids_only": True,
        "source": "thesis_findings 21.2",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the v2 channel-stamped shield corpus")
    ap.add_argument("--readmission", default="results/audit/readmission.json")
    ap.add_argument("--calibration", default="results/audit/warn_n1_calibration.json")
    ap.add_argument("--out", default="shield_corpus/all_rules_channels.jsonl")
    args = ap.parse_args()

    rows = json.loads(Path(args.readmission).read_text(encoding="utf-8"))["rules"]
    cal_by_cond = calibration_index(Path(args.calibration))
    bodies = rule_bodies()

    # 19.2: rule_id is NOT unique across rules_35b/. It happens to be unique
    # across these 58, and the gate's dedup depends on that -- so assert it here,
    # where a collision can still be fixed, rather than discovering it at
    # inference as a silently dropped rule.
    dupes = [rid for rid, n in Counter(r["rule_id"] for r in rows).items() if n > 1]
    if dupes:
        raise SystemExit(f"rule_id collision in the channel corpus: {dupes}. "
                         f"Key on (document, rule_id) - see thesis_findings 19.2.")

    corpus: list[dict] = []
    demotions: list[tuple[str, str, str]] = []

    for row in rows:
        rid = row["rule_id"]
        channel = row["channel"]
        rule = {**bodies.get(rid, {}), "rule_id": rid}
        rule.setdefault("condition", row.get("condition"))
        rule.setdefault("source", row.get("source"))
        rule.setdefault("severity", row.get("severity"))
        rule.setdefault("role", row.get("role") or "CONSTRAINT")
        rule["channel_justification"] = row.get("justification", "")

        if channel == WARN:
            cal = cal_by_cond.get((row.get("condition") or "").strip())
            pooled = (cal or {}).get("pooled") or {}
            verdict = pooled.get("verdict")
            if verdict == "ELEVATED":
                rule["calibration"] = contributing_rates(cal, pooled)
            else:
                # Measured and did not earn a voice. It stays in the corpus, with
                # the number that disqualified it, because "we looked and it does
                # not hold" is a finding; silently dropping it is not.
                channel = NOT_APPLICABLE
                why = verdict or "UNCALIBRATED"
                disc = pooled.get("discrimination")
                detail = f"discrimination {disc:+.3f}" if disc is not None else "no rate"
                rule["channel_justification"] = (
                    f"demoted from WARN: N-1 calibration came back {why} ({detail}); "
                    f"firing does not predict higher N-1 risk (thesis_findings 21.2)"
                )
                rule["calibration"] = {"verdict": why,
                                       "discrimination": disc,
                                       "source": "thesis_findings 21.2"}
                demotions.append((rid, why, detail))

        rule["channel"] = channel
        corpus.append(rule)

    # The BLOCK channel must be the served corpus, predicate for predicate. If it
    # is not, 13's intervention precision is being measured on a different set
    # than the one being shipped.
    served = [json.loads(line) for line in SERVED.read_text(encoding="utf-8").splitlines() if line.strip()]
    served_block = {r["condition"].strip() for r in served
                    if str(r.get("role", "CONSTRAINT")).upper() == "CONSTRAINT"}
    corpus_block = {r["condition"].strip() for r in corpus if r["channel"] == BLOCK}
    if served_block != corpus_block:
        raise SystemExit(
            f"BLOCK channel does not match the served corpus.\n"
            f"  served only: {served_block - corpus_block}\n"
            f"  corpus only: {corpus_block - served_block}"
        )

    assert_warnings_calibrated(corpus)

    counts = Counter(r["channel"] for r in corpus)
    distinct = {c: len({r["condition"].strip() for r in corpus if r["channel"] == c})
                for c in (BLOCK, WARN, NORMAL, NOT_APPLICABLE, INERT)}

    print("\n" + "=" * 72)
    print("v2 SHIELD CORPUS")
    print("=" * 72)
    print(f"\n  {'channel':<18}{'rules':>7}{'distinct':>10}{'v1':>6}")
    print("  " + "-" * 44)
    v1 = {BLOCK: 3, WARN: 0, NORMAL: 1, NOT_APPLICABLE: 0, INERT: 0}
    for c in (BLOCK, WARN, NORMAL, NOT_APPLICABLE, INERT):
        print(f"  {c:<18}{counts.get(c, 0):>7}{distinct[c]:>10}{v1[c]:>6}")
    print("  " + "-" * 44)
    speaking = sum(counts.get(c, 0) for c in (BLOCK, WARN, NORMAL))
    sd = len({r["condition"].strip() for r in corpus
              if r["channel"] in (BLOCK, WARN, NORMAL)})
    print(f"  {'SPEAKING':<18}{speaking:>7}{sd:>10}{4:>6}")

    if demotions:
        print(f"\n  demoted out of WARN by the N-1 calibration ({len(demotions)}):")
        for rid, why, detail in demotions:
            print(f"    {rid:<9} {why:<13} {detail}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rule in corpus:
            fh.write(json.dumps(rule) + "\n")
    print(f"\nwrote {dest}  ({len(corpus)} rules)")
    print("BLOCK channel verified identical to " + str(SERVED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
