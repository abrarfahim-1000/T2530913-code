"""
How many of the 2,463 candidates are blocked because Grid2Op has no frequency
and no sub-second time -- counted two ways, and reconciled.

`corpus_accounting.py` buckets each untranslatable candidate by its FIRST
matching reason, in a fixed order (frequency before time/dynamics before
everything else). That is an EXCLUSIVE partition: a candidate whose reason
mentions both frequency and duration lands in "frequency" only, because
frequency is checked first.

The ANDES investigation (`andes_investigation.md`, `thesis_findings.md` S20.2)
asks a different, INCLUSIVE question: "how many candidates are blocked on
frequency-or-time AT ALL", regardless of which bucket absorbed them. That
number was previously reported by hand as 726 evaluable / 782 time / 683
frequency / 1,332 union, with no script producing any of the four. This
script produces the real one, from the same regexes `corpus_accounting.py`
already uses, and shows that it reconciles exactly with the exclusive
partition -- so there is only one number, not two disagreeing ones.

Run:  .venv\\Scripts\\python.exe evaluation\\capability_gap.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.corpus_accounting import load_jsonl  # noqa: E402

FREQUENCY = re.compile(r"frequen|\bhz\b|rocof|droop|deadband")
TIME = re.compile(r"\btime\b|duration|second|delay|transient|dynamic|ride.?through|"
                   r"\bms\b|cycle|rate of change|ramp")

OUT_JSON = Path("results/audit/capability_gap.json")


def main() -> int:
    root = Path(".")
    untrans = load_jsonl("translated_rules/*_untranslatable.jsonl", root)

    n_freq = n_time = n_both = n_either = 0
    for r in untrans:
        reason = (r.get("reason") or "").lower()
        f = bool(FREQUENCY.search(reason))
        t = bool(TIME.search(reason))
        if f:
            n_freq += 1
        if t:
            n_time += 1
        if f and t:
            n_both += 1
        if f or t:
            n_either += 1

    n_cand = 2463
    print("=" * 78)
    print("CAPABILITY GAP — candidates blocked on frequency and/or sub-second time")
    print("=" * 78)
    print(f"\nuntranslatable candidates scanned              {len(untrans):>6}")
    print(f"mention a frequency-related reason              {n_freq:>6}")
    print(f"mention a time/duration-related reason          {n_time:>6}")
    print(f"mention BOTH                                    {n_both:>6}")
    print(f"mention EITHER (union)                          {n_either:>6}"
          f"   {100 * n_either / n_cand:>5.1f}% of the 2,463-candidate corpus")

    # Reconciliation with corpus_accounting.py's exclusive, first-match-wins
    # buckets: frequency-first means its "frequency" bucket should equal
    # n_freq exactly, and its "time / dynamics" bucket should equal the
    # candidates that mention time but NOT frequency.
    acc_path = Path("results/audit/corpus_accounting.json")
    reconciled = None
    if acc_path.exists():
        acc = json.loads(acc_path.read_text(encoding="utf-8"))
        b = acc["untranslatable_buckets"]
        excl_freq = b.get("frequency", 0)
        excl_time = b.get("time / dynamics", 0)
        excl_union = excl_freq + excl_time
        reconciled = (excl_freq == n_freq) and (excl_union == n_either)
        print(f"\nreconciliation against corpus_accounting.json:")
        print(f"  exclusive frequency bucket    {excl_freq:>6}  vs inclusive freq-mentions {n_freq:>6}")
        print(f"  exclusive union (freq+time)   {excl_union:>6}  vs inclusive union         {n_either:>6}")
        print(f"  RECONCILED: {reconciled}")

    payload = {
        "n_corpus": n_cand,
        "n_untranslatable_scanned": len(untrans),
        "mentions_frequency": n_freq,
        "mentions_time": n_time,
        "mentions_both": n_both,
        "union_either": n_either,
        "share_of_corpus": round(n_either / n_cand, 4),
        "reconciled_with_corpus_accounting": reconciled,
        "note": ("Replaces the unscripted '782 need time, 683 need frequency, "
                 "union 1,332' figure and the unscripted '726 evaluable under a "
                 "dynamic simulator' figure. Both were produced by hand and are "
                 "not reproducible; this script is."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
