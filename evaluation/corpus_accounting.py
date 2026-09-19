"""
Exhaustive accounting of the extraction corpus: where every candidate rule went.

The end-to-end yield (2,463 -> 4) is reported elsewhere as a number with a
narrative explanation. A number invites the reading "the pipeline underperformed
and the result is noise". This script replaces that with a partition: every one
of the 2,463 candidates is assigned to exactly one terminal bucket, and the
assignment is asserted rather than assumed.

The partition is built from artifacts already on disk. Nothing is re-derived
with a model, and no stage is re-run:

    rules_35b/*_candidates.jsonl                      stage 1   2,463
    translated_rules/*_translated.jsonl               stage 2      58
    translated_rules/*_untranslatable.jsonl           stage 2   2,405  (+ reason)
    translated_rules/guarded/*_translated.jsonl       stage 2.5    32
    translated_rules/guarded/*_polarity_rejected.jsonl stage 2.5   26  (+ fire rates)
    validated_translated/*_confirmed.jsonl            stage 3      11
    validated_translated/*_rejected.jsonl             stage 3      21  (+ reason)

Untranslatable rules are bucketed by the translator's own stated reason. The
buckets are matched in the order declared in UNTRANSLATABLE_BUCKETS and the
first match wins, so they are disjoint by construction; the ordering is
load-bearing and is documented there.

Run:  .venv\\Scripts\\python.exe evaluation\\corpus_accounting.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Ordered, first-match-wins, and the ORDER IS LOAD-BEARING.
#
#   frequency before time        -- Grid2Op models no frequency at all, whereas a
#                                   duration could in principle be mapped onto steps
#   equipment-internal before scope -- "stator current, not grid-wide observable"
#                                   mentions scope but the real blocker is that the
#                                   simulator has no stator
#   scope before the catch-all   -- this is the ONLY bucket that is recoverable
#                                   without a new simulator, so it must not be
#                                   absorbed by the generic "unobservable" text
#
# The distinction that carries the thesis argument is `scope / aggregation`
# (the quantity IS in every record, the 14-variable vocabulary flattens it to a
# grid-wide min/max) versus `quantity not modelled` (the simulator does not have
# it at any granularity).
UNTRANSLATABLE_BUCKETS: list[tuple[str, re.Pattern]] = [
    ("frequency",
     re.compile(r"frequen|\bhz\b|rocof|droop|deadband")),
    ("time / dynamics",
     re.compile(r"\btime\b|duration|second|delay|transient|dynamic|ride.?through|"
                r"\bms\b|cycle|rate of change|ramp")),
    ("administrative / process",
     re.compile(r"administrat|regulator|procedur|document|report|complian|"
                r"notif|agreement|annual|study|record.?keeping")),
    ("equipment-internal",
     re.compile(r"relay|protect|winding|temperat|nameplate|setting|excitation|"
                r"governor|impedance|breaker|insulat|withstand|damping|oscillat|"
                r"stator|p-?q capab|capability diagram|lfsm|switchgear|earthing|"
                r"short.?circuit|synchroni|communicat|interface|signal|tap.?chang")),
    ("scope / aggregation  [RECOVERABLE]",
     re.compile(r"grid.?wide|specific (module|generat|node|bus|site|circuit|line|"
                r"connection)|individual|scope:|terminal voltage|not a global|"
                r"module-specific")),
    ("free variable / no number",
     re.compile(r"undefined|unspecified|tso.?defined|not specified|no numeric|"
                r"variable threshold|placeholder")),
    ("quantity not modelled by the simulator",
     re.compile(r"unobserv|unmeasurable|not (a |an )?grid2op|not available|"
                r"not in grid2op|observable")),
]


# Suffixes stripped from a filename to recover the document it belongs to.
_STEM_SUFFIXES = ("_candidates", "_translated", "_untranslatable",
                  "_polarity_rejected", "_confirmed", "_rejected", "_flagged")


def doc_of(path: Path) -> str:
    stem = path.stem
    for suf in _STEM_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def load_jsonl(pattern: str, root: Path) -> list[dict]:
    """Load records, tagging each with the document its file belongs to.

    `rule_id` is NOT unique across the corpus -- one document restarted its
    numbering at R_001 and collides with another over R_001..R_172 -- so every
    downstream key here is the pair (document, rule_id).
    """
    out: list[dict] = []
    for path in sorted(root.glob(pattern)):
        doc = doc_of(path)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_doc"] = doc
                out.append(rec)
    return out


def key_of(rec: dict) -> tuple[str, str | None]:
    """The composite identity of a rule: (document, rule_id)."""
    return (rec.get("_doc", "?"), rid_of(rec))


def rule_of(rec: dict) -> dict:
    """Records are either a bare rule or {'rule': ..., ...}."""
    return rec.get("rule", rec)


def rid_of(rec: dict) -> str | None:
    return rule_of(rec).get("rule_id") or rec.get("rule_id")


def bucket_untranslatable(reason: str) -> str:
    low = (reason or "").lower()
    for name, pat in UNTRANSLATABLE_BUCKETS:
        if pat.search(low):
            return name
    return "unclassified"


def main() -> int:
    root = Path(".")

    candidates = load_jsonl("rules_35b/*_candidates.jsonl", root)
    translated = load_jsonl("translated_rules/*_translated.jsonl", root)
    untrans = load_jsonl("translated_rules/*_untranslatable.jsonl", root)
    guarded = load_jsonl("translated_rules/guarded/*_translated.jsonl", root)
    pol_rej = load_jsonl("translated_rules/guarded/*_polarity_rejected.jsonl", root)
    confirmed = load_jsonl("validated_translated/*_confirmed.jsonl", root)
    val_rej = load_jsonl("validated_translated/*_rejected.jsonl", root)
    deduped = load_jsonl("validated_translated/all_rules_deduped.jsonl", root)

    ids_cand = {key_of(r) for r in candidates}
    n_cand = len(ids_cand)
    ids_tr = {key_of(r) for r in translated}
    ids_un = {key_of(r) for r in untrans}
    ids_gu = {key_of(r) for r in guarded}
    ids_pr = {key_of(r) for r in pol_rej}
    ids_cf = {key_of(r) for r in confirmed}
    ids_vr = {key_of(r) for r in val_rej}

    # rule_id collisions across documents -- a broken stage-1 invariant
    by_rid = {}
    for d, r in ids_cand:
        by_rid.setdefault(r, set()).add(d)
    collisions = {r: d for r, d in by_rid.items() if len(d) > 1}
    served_ids = {rid_of(r) for r in deduped}
    served_collide = served_ids & set(collisions)

    print("=" * 74)
    print("CORPUS ACCOUNTING — where all 2,463 extracted candidates went")
    print("=" * 74)

    print("\nSTAGE-1 IDENTITY")
    print(f"  records in rules_35b/                     {len(candidates):>6}")
    print(f"  distinct (document, rule_id)              {n_cand:>6}")
    print(f"  distinct rule_id alone                    {len(by_rid):>6}")
    print(f"  rule_ids REUSED across documents          {len(collisions):>6}"
          + ("   <-- stage-1 invariant broken" if collisions else ""))
    if collisions:
        rng = sorted(collisions)
        docs = sorted({d for ds in collisions.values() for d in ds})
        print(f"    range {rng[0]}..{rng[-1]}, across {len(docs)} documents:")
        for d in docs:
            print(f"      - {d[:64]}")
        print(f"    served rules carrying a colliding id: {len(served_collide)}"
              + ("   (the headline 4-rule corpus is CLEAN)" if not served_collide
                 else f"  {sorted(served_collide)}"))

    # ── stage integrity ───────────────────────────────────────────────────────
    problems: list[str] = []
    if len(ids_tr | ids_un) != n_cand:
        problems.append(
            f"stage 2 does not partition stage 1: "
            f"{len(ids_tr)} translated + {len(ids_un)} untranslatable "
            f"= {len(ids_tr | ids_un)}, expected {n_cand}")
    if ids_tr & ids_un:
        problems.append(f"{len(ids_tr & ids_un)} rules are BOTH translated and untranslatable")
    if len(ids_gu | ids_pr) != len(ids_tr):
        problems.append(
            f"stage 2.5 does not partition stage 2: "
            f"{len(ids_gu)} + {len(ids_pr)} = {len(ids_gu | ids_pr)}, expected {len(ids_tr)}")
    if len(ids_cf | ids_vr) != len(ids_gu):
        problems.append(
            f"stage 3 does not partition stage 2.5: "
            f"{len(ids_cf)} + {len(ids_vr)} = {len(ids_cf | ids_vr)}, expected {len(ids_gu)}")

    print("\nSTAGE PARTITION (each stage must exactly split the one before it)")
    print(f"  stage 1  candidates                       {n_cand:>6}")
    print(f"  stage 2  translated                       {len(ids_tr):>6}")
    print(f"           untranslatable                   {len(ids_un):>6}")
    print(f"  stage 2.5 kept by polarity guard          {len(ids_gu):>6}")
    print(f"           polarity-rejected                {len(ids_pr):>6}")
    print(f"  stage 3  confirmed                        {len(ids_cf):>6}")
    print(f"           rejected                         {len(ids_vr):>6}")
    print(f"  dedup    distinct served rules            {len(deduped):>6}")

    if problems:
        print("\n  !! PARTITION IS NOT EXHAUSTIVE:")
        for p in problems:
            print(f"     - {p}")
    else:
        print("\n  partition verified: every candidate accounted for exactly once "
              "at every stage")

    # ── why the untranslatable were untranslatable ────────────────────────────
    counts = Counter(bucket_untranslatable(r.get("reason", "")) for r in untrans)
    print(f"\nWHY {len(untrans)} RULES WERE NOT EXPRESSIBLE (translator's own reason)")
    print(f"  {'bucket':<32}{'n':>7}{'of untrans':>12}{'of corpus':>11}")
    print("  " + "-" * 60)
    for name, _ in UNTRANSLATABLE_BUCKETS:
        n = counts.get(name, 0)
        if n:
            print(f"  {name:<32}{n:>7}{n/len(untrans):>11.1%}{n/n_cand:>11.1%}")
    n_unc = counts.get("unclassified", 0)
    if n_unc:
        print(f"  {'unclassified':<32}{n_unc:>7}{n_unc/len(untrans):>11.1%}{n_unc/n_cand:>11.1%}")
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<32}{len(untrans):>7}")

    # ── terminal partition of the whole corpus ───────────────────────────────
    print(f"\nTERMINAL FATE OF ALL {n_cand} CANDIDATES")
    terminal: Counter = Counter()
    for r in untrans:
        terminal[f"not expressible — {bucket_untranslatable(r.get('reason',''))}"] += 1
    for r in pol_rej:
        terminal["expressible, rejected by polarity guard"] += 1
    for r in val_rej:
        terminal["expressible and clean, rejected by validator"] += 1
    for r in confirmed:
        terminal["VALIDATED (served to the shield)"] += 1
    print(f"  {'fate':<50}{'n':>7}{'share':>10}")
    print("  " + "-" * 67)
    for k, n in terminal.most_common():
        print(f"  {k:<50}{n:>7}{n/n_cand:>10.1%}")
    print("  " + "-" * 67)
    total = sum(terminal.values())
    print(f"  {'TOTAL':<50}{total:>7}{total/n_cand:>10.1%}")
    if total != n_cand:
        print(f"  !! terminal buckets sum to {total}, expected {n_cand}")

    out = {
        "n_records_stage1": len(candidates),
        "n_candidates_distinct": n_cand,
        "rule_id_collisions": len(collisions),
        "served_rules_with_colliding_id": sorted(served_collide),
        "stages": {
            "translated": len(ids_tr),
            "untranslatable": len(ids_un),
            "guard_kept": len(ids_gu),
            "guard_rejected": len(ids_pr),
            "validated_confirmed": len(ids_cf),
            "validated_rejected": len(ids_vr),
            "distinct_served": len(deduped),
        },
        "partition_verified": not problems,
        "partition_problems": problems,
        "untranslatable_buckets": dict(counts),
        "terminal_fate": dict(terminal),
        "bucket_order": [n for n, _ in UNTRANSLATABLE_BUCKETS],
        "note": ("Buckets are first-match-wins in the declared order; frequency "
                 "precedes time because Grid2Op models no frequency at all."),
    }
    dest = root / "results/audit/corpus_accounting.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
