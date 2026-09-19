"""Stamp per-grid COVERAGE onto the channel corpus, so a warning can be silenced
on the grids where it was measured to say nothing.

The defect
----------
A rule earns the WARN channel by discriminating on *any* grid. The channel is
then fixed for all of them. `voltage_pu_min < 0.95 or voltage_pu_max > 1.05`
discriminates on wcci2022 (coverage 1.12%) and is degenerate on neurips2020 and
case14 (coverage 1.0000 — it fires on every frame), and it ships as a warning on
all three. The corpus already knew: `channel_justification` reads "discriminates
on wcci2022 (degenerate on case14, neurips2020)". It said so in prose and then
ignored itself.

That predicate is stated in 10 clauses across 4 documents — tied with the thermal
check as the most corroborated predicate in the corpus. It is not a
mis-extraction. These grids simply rest at 1.05–1.08 pu, so a band written about
the post-disturbance state is already breached before anything has happened.

What this does
--------------
Copies the per-grid `coverage` out of `results/audit/warn_n1_calibration.json`
and onto each WARN record, keyed by tag. Nothing else is touched: no threshold,
no condition, no severity, and no rule's `channel` field. The demotion itself
happens at serving time in `shield.channels.resolve_channels_for_grid`, so the
shipped corpus still reads WARN 20 and `tests/test_shield_channels.py` still
pins the documented counts.

It cannot move a reported number. WARN never reaches the veto path, so no rule
that could block is affected — `tests/test_warn_degeneracy.py` pins that too.

    python evaluation/stamp_warn_coverage.py
    python evaluation/stamp_warn_coverage.py --report     # measure, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shield.channels import (  # noqa: E402
    DEGENERATE_COVERAGE,
    WARN,
    channel_of,
)

CORPUS = "shield_corpus/all_rules_channels.jsonl"
CALIBRATION = "results/audit/warn_n1_calibration.json"
REPORT = "results/audit/warn_coverage_stamp.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def coverage_index(calibration_path: Path) -> dict[str, dict[str, float]]:
    """condition -> {tag: coverage}, straight out of the calibration run.

    Keyed on the condition rather than on rule_id because that is how the
    calibration was computed: one measurement per distinct predicate, shared by
    every record stating it. Keying on rule_id would silently miss the records
    that share a condition with the one that was measured.
    """
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, float]] = {}
    for entry in data.get("conditions", []):
        per_grid = entry.get("per_grid") or {}
        index[entry["condition"]] = {
            tag: float(v["coverage"])
            for tag, v in per_grid.items()
            if isinstance(v.get("coverage"), (int, float))
        }
    return index


def stamp(corpus: list[dict], index: dict[str, dict[str, float]]) -> dict:
    """Add `coverage` to every WARN record the calibration measured."""
    stamped = unmeasured = 0
    degenerate: dict[str, list[str]] = {}
    for rule in corpus:
        if channel_of(rule) != WARN:
            continue
        coverage = index.get(rule.get("condition", ""))
        if not coverage:
            # Reported, never assumed. An unmeasured rule keeps its voice; the
            # opposite default would silence rules for the sin of being new.
            unmeasured += 1
            continue
        rule["coverage"] = dict(sorted(coverage.items()))
        stamped += 1
        for tag, value in coverage.items():
            if value >= DEGENERATE_COVERAGE:
                degenerate.setdefault(tag, []).append(str(rule.get("rule_id")))

    return {
        "n_warn_records": sum(1 for r in corpus if channel_of(r) == WARN),
        "stamped": stamped,
        "unmeasured": unmeasured,
        "degenerate_cutoff": DEGENERATE_COVERAGE,
        "degenerate_by_grid": {t: sorted(v) for t, v in sorted(degenerate.items())},
        "n_degenerate_by_grid": {t: len(v) for t, v in sorted(degenerate.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--calibration", default=CALIBRATION)
    ap.add_argument("--out", default=None, help="default: in place")
    ap.add_argument("--report-path", default=REPORT)
    ap.add_argument("--report", action="store_true",
                    help="measure and print, write no corpus")
    ap.add_argument("--kg", default="kg/knowledge_graph.json",
                    help="graph to keep in step with the stamped corpus")
    ap.add_argument("--no-sync-kg", action="store_true",
                    help="leave the graph alone. It then carries pre-stamp "
                         "payloads until rebuilt; KgRuleProvider repairs that "
                         "in memory, but the file on disk stays stale.")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    calibration_path = Path(args.calibration)
    for path in (corpus_path, calibration_path):
        if not path.exists():
            sys.exit(f"Missing {path}")

    corpus = read_jsonl(corpus_path)
    summary = stamp(corpus, coverage_index(calibration_path))

    print(f"WARN records            : {summary['n_warn_records']}")
    print(f"  stamped with coverage : {summary['stamped']}")
    print(f"  unmeasured (kept)     : {summary['unmeasured']}")
    print(f"\ndegenerate at coverage >= {DEGENERATE_COVERAGE} "
          f"(silenced on that grid only):")
    for tag, n in summary["n_degenerate_by_grid"].items():
        speaking = summary["n_warn_records"] - n
        print(f"  {tag:<12} {n:>2} silenced, {speaking:>2} still warn")
    for tag in sorted(set(summary["degenerate_by_grid"])):
        print(f"  {tag}: {', '.join(summary['degenerate_by_grid'][tag])}")

    if args.report:
        print("\n--report: nothing written.")
        return 0

    out = Path(args.out) if args.out else corpus_path
    with out.open("w", encoding="utf-8") as f:
        for rule in corpus:
            f.write(json.dumps(rule, ensure_ascii=False) + "\n")
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ncorpus -> {out}\nreport -> {report_path}")

    # Keep the graph in step. It carries each rule dict verbatim, so stamping
    # the corpus and leaving the graph alone is what made this fix a silent
    # no-op the first time it ran. `KgRuleProvider` now repairs that drift in
    # memory, but the artifact on disk should not be the one thing still
    # holding stale payloads - somebody will read it.
    if not args.no_sync_kg:
        _sync_graph(Path(args.kg))
    return 0


def _sync_graph(kg_path: Path) -> None:
    if not kg_path.exists():
        return
    from kg.build import (
        explanatory_corpus_status,
        load_kg,
        refresh_explanatory_payloads,
        save_kg,
    )

    kg = load_kg(kg_path)
    if explanatory_corpus_status(kg)["state"] == "absent":
        return
    result = refresh_explanatory_payloads(kg)
    if result["state"] == "refreshed":
        save_kg(kg, kg_path)
        print(f"graph  -> {kg_path} ({result['refreshed']} payloads refreshed)")
    elif result["state"] == "rebuild_required":
        print(f"\nWARNING: {kg_path} needs a full REBUILD, not a refresh "
              f"({result.get('reason')})."
              f"\n         Run: python scripts/build_kg.py --channels")
    else:
        print(f"graph  -> {kg_path} (already current)")


if __name__ == "__main__":
    sys.exit(main())
