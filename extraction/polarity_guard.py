"""
polarity_guard.py — Stage 2.5: the healthy-frame polarity guard.

A violation predicate must be FALSE on a healthy grid. Any condition that is TRUE
there is wrong-polarity or wrong-threshold *by definition*, whatever the source
text said. This stage measures that deterministically — no LLM, no GPU — by
evaluating every translated condition against thousands of simulator-labelled
`normal` frames and rejecting the ones that fire.

Why a fire rate over real frames, and not a hand-written "healthy band":

  - `voltage_pu_min < 0 or voltage_pu_max > 1` (a real extracted rule, severity
    critical) is FALSE at a synthetic 1.0 pu point but fires on ~72% of real
    healthy case14 frames. A band-based check ships it.
  - `loading_pct > 80` is a legitimate near-limit rule, yet fires on ~14% of
    healthy case14 frames because that grid genuinely runs hot (normal-frame
    rho_max median 0.72). A band-based check culls it.

The fire rate separates those two without being told how. It is also, by
definition, the rule's per-frame false block rate — so persisting it predicts the
shield's overall false block rate before the shield is built, and names the rules
that drive it.

Usage:
    # report the distribution, choose the cutoff from its shape, write nothing
    python extraction/polarity_guard.py --translated rules/ --tag case14 --report

    # partition into *_translated_clean.jsonl + *_polarity_rejected.jsonl
    python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shield.context import build_context, load_base_kv  # noqa: E402
from shield.evaluator import Verdict, evaluate_condition  # noqa: E402

try:
    from common import get_logger
except ImportError:  # invoked as a module rather than a script
    from extraction.common import get_logger

log = get_logger("polarity_guard")

DEFAULT_MAX_FIRE_RATE = 0.5   # CONSTRAINT: max tolerated firing on healthy frames
DEFAULT_MIN_SUPPORT = 0.5     # AFFIRMATION: min firing on frames of its own class
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_MAX_SCAN = 400_000
SEED = 42
LABELS = ("normal", "overload", "line_trip", "cascade")


# ── THE CRITERION ─────────────────────────────────────────────────────────────

def fire_rate(condition: str, contexts: Sequence[dict]) -> float:
    """Fraction of healthy contexts on which `condition` evaluates TRUE.

    0.0 means the rule never fires on a healthy grid (good). ~1.0 means it fires
    on essentially all of them, which is only possible if the polarity or the
    threshold is wrong. NOT_EVALUABLE / ERROR verdicts are not counted as fires —
    they never block, so they cannot cause a false block either.
    """
    if not contexts:
        raise ValueError("No healthy contexts — cannot measure a fire rate.")
    fires = sum(
        1 for ctx in contexts if evaluate_condition(condition, dict(ctx)) is Verdict.VIOLATED
    )
    return fires / len(contexts)


def first_firing_context(condition: str, contexts: Sequence[dict]) -> Optional[dict]:
    """The first healthy context that trips `condition` — evidence for the audit file."""
    for ctx in contexts:
        if evaluate_condition(condition, dict(ctx)) is Verdict.VIOLATED:
            return ctx
    return None


def is_affirmation(rule: dict) -> bool:
    return str(rule.get("role", "CONSTRAINT")).upper() == "AFFIRMATION"


def probe_label(rule: dict) -> str:
    """Which ground-truth class a rule should be measured against.

    A CONSTRAINT is tested on `normal` frames: it must never fire there. An
    AFFIRMATION is tested on frames of the class it claims to affirm: it must fire
    there, or it affirms nothing.
    """
    return (rule.get("affirms") or "normal").lower() if is_affirmation(rule) else "normal"


def assert_no_rule_fires(rules: Iterable[dict], contexts: Sequence[dict]) -> None:
    """Backstop for the shield's rule loader (handoff §3c item 4).

    Cheap insurance that a contaminated ruleset cannot reach inference even if the
    guard was skipped when the corpus was built. Only CONSTRAINT rules are checked
    — an affirmation of `normal` firing on every healthy frame is correct behaviour,
    not contamination.
    """
    offenders = [
        (r.get("rule_id", "?"), r["condition"], rate)
        for r in rules
        if not is_affirmation(r) and (rate := fire_rate(r["condition"], contexts)) > 0.99
    ]
    if offenders:
        listed = "\n  ".join(f"{rid}: {cond!r} fires at {rate:.1%}" for rid, cond, rate in offenders)
        raise ValueError(
            f"{len(offenders)} loaded rule(s) fire on a healthy grid — the ruleset is "
            f"polarity-contaminated and would block healthy frames:\n  {listed}"
        )


# ── HEALTHY-FRAME CORPUS ──────────────────────────────────────────────────────

def sample_contexts(
    tag: str,
    labels: Sequence[str] = LABELS,
    n: int = DEFAULT_SAMPLE_SIZE,
    max_scan: int = DEFAULT_MAX_SCAN,
    data_dir: str = "data",
    cache: bool = True,
) -> dict[str, list[dict]]:
    """Reservoir-sample ~n frames of EACH ground-truth class as evaluation contexts.

    Constraints are scored against `normal` frames; affirmations against frames of
    the class they claim to affirm. One pass fills every reservoir.

    Reservoir rather than the first n: records are written in chronic order, so a
    prefix would sample only the earliest chronics and miss the seasonal spread of
    loading conditions the fire rate is supposed to average over.

    Results are cached — grid_dataset_neurips2020.jsonl is 3.4 GB and a scan costs
    minutes, while the derived contexts are a few hundred kilobytes.
    """
    cache_path = Path(data_dir) / f"label_contexts_{tag}.json"
    if cache and cache_path.exists():
        with cache_path.open() as f:
            cached = json.load(f)
        if (cached.get("n_requested") == n and cached.get("max_scan") == max_scan
                and set(cached.get("contexts", {})) >= set(labels)):
            counts = {k: len(v) for k, v in cached["contexts"].items()}
            log.info(f"[{tag}] contexts from cache: {counts}")
            return cached["contexts"]

    jsonl = Path(data_dir) / f"grid_dataset_{tag}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{jsonl} not found. Generate it with scripts/generate_dataset.py first."
        )
    base_kv = load_base_kv(tag, data_dir=data_dir)

    rng = random.Random(SEED)
    reservoirs: dict[str, list[dict]] = {lab: [] for lab in labels}
    seen_counts: dict[str, int] = {lab: 0 for lab in labels}
    n_scanned = 0

    log.info(f"[{tag}] scanning up to {max_scan:,} records for {list(labels)} frames...")
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            if n_scanned >= max_scan:
                break
            n_scanned += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = record.get("label")
            if label not in reservoirs:
                continue

            try:
                ctx = build_context(record, base_kv)
            except (KeyError, ValueError) as exc:
                log.warning(f"  skipping malformed record: {exc}")
                continue

            seen_counts[label] += 1
            bucket = reservoirs[label]
            if len(bucket) < n:
                bucket.append(ctx)
            else:
                j = rng.randint(0, seen_counts[label] - 1)
                if j < n:
                    bucket[j] = ctx

    if not any(reservoirs.values()):
        raise ValueError(f"[{tag}] no labelled frames found in the first {n_scanned:,} records.")

    log.info(f"[{tag}] sampled { {k: len(v) for k, v in reservoirs.items()} } "
             f"from {n_scanned:,} records")
    for lab, bucket in reservoirs.items():
        if not bucket:
            log.warning(f"[{tag}] no `{lab}` frames sampled - affirmations of that class "
                        f"cannot be scored on this topology")

    if cache:
        with cache_path.open("w") as f:
            json.dump(
                {"tag": tag, "n_requested": n, "max_scan": max_scan,
                 "n_seen": seen_counts, "contexts": reservoirs},
                f,
            )
        log.info(f"[{tag}] cached -> {cache_path}")

    return reservoirs


def sample_healthy_contexts(tag: str, **kwargs) -> list[dict]:
    """Just the `normal` frames — used by the shield's load-time backstop."""
    return sample_contexts(tag, labels=("normal",), **kwargs)["normal"]


# ── PARTITIONING ──────────────────────────────────────────────────────────────

@dataclass
class RuleAssessment:
    rule: dict
    rates: dict = field(default_factory=dict)     # tag -> fire rate on the probe class
    evidence: dict = field(default_factory=dict)  # tag -> a firing context

    @property
    def role(self) -> str:
        return "AFFIRMATION" if is_affirmation(self.rule) else "CONSTRAINT"

    @property
    def probe(self) -> str:
        return probe_label(self.rule)

    @property
    def worst_rate(self) -> float:
        """The rate that most incriminates the rule, given its role.

        A CONSTRAINT is damned by its HIGHEST rate on `normal` frames (it should
        never fire); an AFFIRMATION is damned by its LOWEST rate on frames of the
        class it affirms (it should nearly always fire).
        """
        if not self.rates:
            return 0.0
        return min(self.rates.values()) if self.role == "AFFIRMATION" else max(self.rates.values())

    @property
    def worst_tag(self) -> str:
        if not self.rates:
            return ""
        pick = min if self.role == "AFFIRMATION" else max
        return pick(self.rates, key=self.rates.get)

    def spread(self) -> float:
        """Max-min rate across topologies. A large spread is a cross-topology
        finding in its own right, not merely a filtering detail (handoff §3c)."""
        return (max(self.rates.values()) - min(self.rates.values())) if len(self.rates) > 1 else 0.0


def assess_rules(
    rules: Sequence[dict],
    contexts_by_tag: dict[str, dict[str, list[dict]]],
) -> list[RuleAssessment]:
    """Measure each rule against the ground-truth class its role implicates.

    `contexts_by_tag` maps tag -> label -> contexts. A rule with no frames of its
    probe class on a given topology is simply not scored there.
    """
    assessments = []
    for rule in rules:
        a = RuleAssessment(rule=rule)
        for tag, by_label in contexts_by_tag.items():
            contexts = by_label.get(a.probe) or []
            if not contexts:
                continue
            a.rates[tag] = fire_rate(rule["condition"], contexts)
            firing = first_firing_context(rule["condition"], contexts)
            if firing is not None:
                a.evidence[tag] = firing
        assessments.append(a)
    return assessments


def partition(
    assessments: Sequence[RuleAssessment],
    max_fire_rate: float = DEFAULT_MAX_FIRE_RATE,
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> tuple[list[RuleAssessment], list[RuleAssessment]]:
    """Split into (kept, rejected), applying the test each role deserves.

    - CONSTRAINT  rejected when it fires on more than `max_fire_rate` of healthy
      frames — a violation predicate must be False on a healthy grid.
    - AFFIRMATION rejected when it fires on fewer than `min_support` of the frames
      of the class it claims to affirm — an affirmation that rarely holds on its
      own class affirms nothing.

    Worst-case across topologies rather than mean: a rule silent on the training
    grid but firing constantly on case14 is still unusable, and averaging hides it.
    Rules with no measurable rate anywhere are kept — absence of evidence is not
    evidence of a defect, and the shield's load-time backstop still guards them.
    """
    kept, rejected = [], []
    for a in assessments:
        if not a.rates:
            kept.append(a)
        elif a.role == "AFFIRMATION":
            (rejected if a.worst_rate < min_support else kept).append(a)
        else:
            (rejected if a.worst_rate > max_fire_rate else kept).append(a)
    return kept, rejected


# ── REPORTING ─────────────────────────────────────────────────────────────────

_BINS = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.1), (0.1, 0.25), (0.25, 0.5),
         (0.5, 0.75), (0.75, 0.99), (0.99, 1.01)]


def print_distribution(assessments: Sequence[RuleAssessment], max_fire_rate: float,
                       min_support: float = DEFAULT_MIN_SUPPORT) -> None:
    """Print the fire-rate histogram the cutoff should be chosen from.

    Expect it to be bimodal — a cluster at 0.0 (correct polarity) and one near 1.0
    (inverted) with an empty middle to cut in. If it is NOT bimodal, say so rather
    than pretending the gap exists, and choose the cutoff conservatively.
    """
    constraints = [a for a in assessments if a.role == "CONSTRAINT"]
    affirmations = [a for a in assessments if a.role == "AFFIRMATION"]
    log.info("")
    log.info(f"Roles: {len(constraints)} CONSTRAINT, {len(affirmations)} AFFIRMATION")
    if affirmations:
        by_class = collections.Counter(a.probe for a in affirmations)
        log.info(f"  affirmation coverage by class: {dict(by_class)}")
        missing = [c for c in LABELS if c not in by_class]
        if missing:
            log.info(f"  NOTE: no affirmations for {missing} - a gating shield (Option B) "
                     f"could not support predictions of those classes")
    log.info("")
    log.info("Fire-rate distribution (CONSTRAINTs on `normal`; AFFIRMATIONs on own class):")
    total = len(assessments)
    for lo, hi in _BINS:
        if lo == hi == 0.0:
            members = [a for a in assessments if a.worst_rate == 0.0]
            label = "0.00 (never fires)"
        else:
            members = [a for a in assessments if lo < a.worst_rate <= hi]
            label = f"{lo:.2f} < r <= {hi:.2f}"
        if not members:
            continue
        bar = "#" * min(60, max(1, round(60 * len(members) / max(total, 1))))
        log.info(f"  {label:22s} {len(members):5d}  {bar}")

    middle = [a for a in assessments if 0.1 < a.worst_rate < 0.9]
    log.info("")
    if middle:
        log.info(f"  NOTE: {len(middle)} rule(s) fall in the ambiguous 0.1-0.9 band - the "
                 f"distribution is not cleanly bimodal. Inspect these before trusting "
                 f"--max-fire-rate {max_fire_rate}:")
        for a in sorted(middle, key=lambda x: -x.worst_rate)[:10]:
            log.info(f"    {a.worst_rate:.3f}  {a.rule.get('rule_id','?')}  {a.rule['condition']}")
    else:
        log.info("  Distribution is cleanly bimodal (nothing between 0.1 and 0.9).")

    spread = [a for a in assessments if a.spread() > 0.25]
    if spread:
        log.info("")
        log.info(f"  CROSS-TOPOLOGY FINDING: {len(spread)} rule(s) differ by >0.25 between grids:")
        for a in sorted(spread, key=lambda x: -x.spread())[:10]:
            rates = "  ".join(f"{t}={r:.2f}" for t, r in a.rates.items())
            log.info(f"    {a.rule.get('rule_id','?')}  {a.rule['condition']}   {rates}")


# ── FILE I/O ──────────────────────────────────────────────────────────────────

def _load_translated(path: Path) -> list[dict]:
    """Read a *_translated.jsonl written by translate.py ({"rule":..., "chunk":...})."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def process_file(
    translated_file: Path,
    out_dir: Path,
    contexts_by_tag: dict[str, list[dict]],
    max_fire_rate: float,
    report_only: bool,
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> dict:
    stem = translated_file.stem.replace("_translated", "")
    records = _load_translated(translated_file)
    if not records:
        log.warning(f"  {translated_file.name}: no records - skipping")
        return {"file": translated_file.name, "n_rules": 0}

    rules = [rec["rule"] for rec in records]
    assessments = assess_rules(rules, contexts_by_tag)
    # assess_rules preserves order, so each assessment keeps its originating record
    # (and therefore its source chunk, which validate.py needs downstream).
    chunk_of = {
        id(a): rec.get("chunk", "") for a, rec in zip(assessments, records)
    }
    kept, rejected = partition(assessments, max_fire_rate, min_support)

    log.info(f"  {translated_file.name}: {len(rules)} rules -> "
             f"kept={len(kept)}  rejected={len(rejected)}")

    if not report_only:
        # The kept file keeps the *_translated.jsonl name so validate.py's glob
        # picks it up unchanged — it lives in its own directory, so the unfiltered
        # originals survive alongside it for the filtered-vs-unfiltered counterfactual.
        # NOTE: never name an output *_confirmed.jsonl — deduplicate_rules() globs
        # that pattern and would merge rejects back into all_rules_deduped.jsonl.
        clean_file = out_dir / f"{stem}_translated.jsonl"
        reject_file = out_dir / f"{stem}_polarity_rejected.jsonl"

        with clean_file.open("w") as cf:
            for a in kept:
                rule = dict(a.rule)
                for tag, rate in a.rates.items():
                    rule[f"fire_rate_{tag}"] = round(rate, 5)
                cf.write(json.dumps({"rule": rule, "chunk": chunk_of[id(a)]}) + "\n")

        with reject_file.open("w") as rf:
            for a in rejected:
                rates = ", ".join(f"{t}={r:.3f}" for t, r in a.rates.items())
                evidence = a.evidence.get(a.worst_tag, {})
                probe = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in evidence.items()
                )
                rf.write(json.dumps({
                    "rule": a.rule,
                    "reason": (
                        f"POLARITY_REJECTED: fires on healthy frames ({rates}); "
                        f"a violation predicate must be False on a healthy grid. "
                        f"Example firing context: {probe}"
                    ),
                    "fire_rates": a.rates,
                }) + "\n")

        log.info(f"    -> {clean_file.name} ({len(kept)})  {reject_file.name} ({len(rejected)})")

    return {
        "file": translated_file.name,
        "n_rules": len(rules),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "assessments": assessments,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2.5: healthy-frame polarity guard")
    ap.add_argument("--translated", required=True, help="Folder with *_translated.jsonl files")
    ap.add_argument("--out", default=None,
                    help="Output folder (default: <translated>/guarded). Kept separate from the "
                         "input so the unfiltered ruleset survives for the counterfactual.")
    ap.add_argument("--tag", action="append", dest="tags", default=None,
                    help="Dataset tag to measure against; repeat for several "
                         "(default: neurips2020 case14)")
    ap.add_argument("--max-fire-rate", type=float, default=DEFAULT_MAX_FIRE_RATE,
                    help=f"Reject above this rate (default {DEFAULT_MAX_FIRE_RATE}). "
                         f"Choose it from --report, not by guessing.")
    ap.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT,
                    help=f"Reject an AFFIRMATION firing on less than this fraction of its "
                         f"own class (default {DEFAULT_MIN_SUPPORT})")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--max-scan", type=int, default=DEFAULT_MAX_SCAN)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--report", action="store_true",
                    help="Print the fire-rate distribution and write nothing")
    ap.add_argument("--no-cache", action="store_true", help="Ignore cached healthy contexts")
    args = ap.parse_args()

    translated_dir = Path(args.translated)
    out_dir = Path(args.out) if args.out else translated_dir / "guarded"
    if out_dir.resolve() == translated_dir.resolve():
        log.error("--out must differ from --translated: the guarded files reuse the "
                  "*_translated.jsonl name and would overwrite the unfiltered inputs.")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(translated_dir.glob("*_translated.jsonl"))
    if not files:
        log.error(f"No *_translated.jsonl files in {translated_dir}")
        log.error("Run extraction/translate.py first.")
        sys.exit(1)

    tags = args.tags or ["neurips2020", "case14"]
    contexts_by_tag: dict[str, list[dict]] = {}
    for tag in tags:
        try:
            contexts_by_tag[tag] = sample_contexts(
                tag, labels=LABELS, n=args.sample_size, max_scan=args.max_scan,
                data_dir=args.data_dir, cache=not args.no_cache,
            )
        except (FileNotFoundError, ValueError) as exc:
            log.warning(f"[{tag}] unavailable - skipping: {exc}")

    if not contexts_by_tag:
        log.error("No topology produced healthy contexts; cannot measure fire rates.")
        sys.exit(1)

    log.info(f"Guarding {len(files)} file(s) against {', '.join(contexts_by_tag)} "
             f"@ cutoff {args.max_fire_rate}")

    all_assessments: list[RuleAssessment] = []
    file_stats = []
    for tf in files:
        stats = process_file(tf, out_dir, contexts_by_tag, args.max_fire_rate,
                             args.report, args.min_support)
        all_assessments.extend(stats.pop("assessments", []))
        file_stats.append(stats)

    print_distribution(all_assessments, args.max_fire_rate, args.min_support)

    kept, rejected = partition(all_assessments, args.max_fire_rate, args.min_support)
    n_total = len(all_assessments)
    log.info("")
    log.info(f"TOTAL  rules={n_total}  kept={len(kept)}  rejected={len(rejected)}"
             + (f"  ({len(rejected) / n_total:.1%} contaminated)" if n_total else ""))

    if not args.report:
        summary = {
            "tags": list(contexts_by_tag),
            "n_contexts": {t: {l: len(v) for l, v in by.items()} for t, by in contexts_by_tag.items()},
            "max_fire_rate": args.max_fire_rate,
            "min_support": args.min_support,
            "n_rules": n_total,
            "n_kept": len(kept),
            "n_rejected": len(rejected),
            "files": file_stats,
        }
        summary_path = out_dir / "polarity_guard_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary -> {summary_path}")
        log.info(f"\nNext step: python extraction/validate.py --candidates {out_dir}/")


if __name__ == "__main__":
    main()
