"""
audit_rules.py — deterministic post-guard audit: can each rule actually do work?

The polarity guard (stage 2.5) answers "is this rule WRONG?" — does a constraint
fire on healthy frames, does an affirmation hold on its own class. It does not
answer "is this rule USEFUL?", and the two come apart:

  - A CONSTRAINT that fires on 0% of healthy frames passes the guard. If it also
    fires on 0% of overload, line_trip and cascade frames it can never block
    anything, so it contributes nothing to the shield. It is INERT.
  - An AFFIRMATION of `normal` holding on 100% of normal frames passes the guard
    at full support. If it also holds on 100% of overload frames it does not
    distinguish the classes, so it supports every prediction equally and carries
    no evidence. It is UNINFORMATIVE.

Neither is a defect in the rule as written — a plus/minus 10% voltage band is a
true statement about the grid. It is a statement about what the rule can
*contribute*, which is what the shield's rule-count ablation needs to know.

Runs entirely offline: no LLM, no GPU. Same context sampler as the guard, so the
frames audited here are the frames guarded there.

Usage:
    python evaluation/audit_rules.py --rules translated_rules/guarded/
    python evaluation/audit_rules.py --rules translated_rules/guarded/ --json audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from extraction.common import get_logger  # noqa: E402
from extraction.polarity_guard import (  # noqa: E402
    DEFAULT_TAGS,
    LABELS,
    fire_rate,
    sample_contexts,
)

log = get_logger("audit_rules")

ABNORMAL = tuple(c for c in LABELS if c != "normal")

# A rule must separate the classes by at least this much to count as carrying
# evidence. Deliberately permissive: the question is whether the rule
# discriminates AT ALL, not whether it discriminates well.
MIN_DISCRIMINATION = 0.05

VERDICT_ORDER = ("INERT", "UNINFORMATIVE", "NO_EVIDENCE", "TOPOLOGY_DEPENDENT", "USEFUL")


@dataclass
class RuleAudit:
    rule_id: str
    entity: str
    condition: str
    role: str
    affirms: Optional[str]
    rates: dict = field(default_factory=dict)   # tag -> label -> rate
    verdict: str = ""
    detail: str = ""


def rate_table(condition: str, contexts_by_tag: dict) -> dict:
    return {
        tag: {label: fire_rate(condition, frames)
              for label, frames in by_label.items() if frames}
        for tag, by_label in contexts_by_tag.items()
    }


def audit_rule(rule: dict, contexts_by_tag: dict) -> RuleAudit:
    """Classify one rule as USEFUL / INERT / UNINFORMATIVE / TOPOLOGY_DEPENDENT."""
    role = rule.get("role") or "CONSTRAINT"
    affirms = rule.get("affirms")
    a = RuleAudit(
        rule_id=rule.get("rule_id", "?"),
        entity=rule.get("entity", "?"),
        condition=rule["condition"],
        role=role,
        affirms=affirms,
        rates=rate_table(rule["condition"], contexts_by_tag),
    )

    if role == "AFFIRMATION":
        own = affirms or "normal"
        # Evidence = how much more often it holds on its own class than on others.
        margins = []
        for by_label in a.rates.values():
            if own not in by_label:
                continue
            others = [r for lab, r in by_label.items() if lab != own]
            if others:
                margins.append(by_label[own] - max(others))
        if not margins:
            a.verdict = "NO_EVIDENCE"
            a.detail = "no frames of the affirmed class were sampled"
        elif max(margins) < MIN_DISCRIMINATION:
            a.verdict = "UNINFORMATIVE"
            a.detail = (f"holds on `{own}` and on every other class alike "
                        f"(best margin {max(margins):+.3f})")
        elif min(margins) < MIN_DISCRIMINATION:
            a.verdict = "TOPOLOGY_DEPENDENT"
            a.detail = (f"discriminates on some grids only "
                        f"(margins {min(margins):+.3f}..{max(margins):+.3f})")
        else:
            a.verdict = "USEFUL"
            a.detail = f"separates `{own}` by {min(margins):+.3f}..{max(margins):+.3f}"
        return a

    # CONSTRAINT: earns its place by firing on abnormal frames, not on normal ones.
    fires_abnormal = {
        tag: max((by_label.get(c, 0.0) for c in ABNORMAL), default=0.0)
        for tag, by_label in a.rates.items()
    }
    best = max(fires_abnormal.values(), default=0.0)
    worst = min(fires_abnormal.values(), default=0.0)
    if best < MIN_DISCRIMINATION:
        a.verdict = "INERT"
        a.detail = (f"never fires on any class on any grid (max abnormal rate {best:.3f}) — "
                    f"cannot block, contributes nothing")
    elif worst < MIN_DISCRIMINATION:
        a.verdict = "TOPOLOGY_DEPENDENT"
        a.detail = f"fires on abnormal frames on some grids only ({worst:.3f}..{best:.3f})"
    else:
        a.verdict = "USEFUL"
        a.detail = f"fires on abnormal frames at {worst:.3f}..{best:.3f}"
    return a


def load_rules(rules_dir: Path) -> list[dict]:
    rules: list[dict] = []
    for jf in sorted(rules_dir.glob("*_translated.jsonl")):
        with jf.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rules.append(rec.get("rule", rec))
    return rules


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic usefulness audit of a guarded ruleset")
    ap.add_argument("--rules", required=True, help="Folder with guarded *_translated.jsonl")
    ap.add_argument("--tag", action="append", dest="tags", default=None,
                    help=f"Topologies to audit against (default: {' '.join(DEFAULT_TAGS)})")
    ap.add_argument("--sample-size", type=int, default=5000)
    ap.add_argument("--json", default=None, help="Write the full per-rule table here")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    rules = load_rules(Path(args.rules))
    if not rules:
        log.error(f"No *_translated.jsonl rules in {args.rules}")
        sys.exit(1)
    log.info(f"Auditing {len(rules)} rule(s) from {args.rules}")

    contexts_by_tag = {}
    for tag in (args.tags or list(DEFAULT_TAGS)):
        try:
            contexts_by_tag[tag] = sample_contexts(
                tag, labels=LABELS, n=args.sample_size, data_dir=args.data_dir)
        except (FileNotFoundError, ValueError) as exc:
            log.warning(f"[{tag}] unavailable - skipping: {exc}")
    if not contexts_by_tag:
        log.error("No topology produced contexts.")
        sys.exit(1)

    audits = [audit_rule(r, contexts_by_tag) for r in rules]
    counts = {v: sum(1 for a in audits if a.verdict == v) for v in VERDICT_ORDER}
    distinct = {(a.entity, a.condition) for a in audits}

    print()
    print(f"  {len(rules)} rules  ({len(distinct)} distinct entity/condition pairs)")
    print(f"  audited against: {', '.join(contexts_by_tag)}")
    print()
    for v in VERDICT_ORDER:
        if not counts[v]:
            continue
        print(f"  -- {v}  ({counts[v]}) " + "-" * max(0, 54 - len(v)))
        seen = set()
        for a in audits:
            key = (a.entity, a.condition)
            if a.verdict != v or key in seen:
                continue
            seen.add(key)
            label = f"{a.role[:5]}/{a.affirms or '-'}"
            print(f"     {a.rule_id:8} [{label:14}] {a.condition[:56]}")
            print(f"              {a.detail}")
        print()

    working = {(a.entity, a.condition) for a in audits
               if a.verdict in ("USEFUL", "TOPOLOGY_DEPENDENT")}
    n_working = counts["USEFUL"] + counts["TOPOLOGY_DEPENDENT"]
    print(f"  CAN DO WORK: {n_working}/{len(rules)} rules ({len(working)} distinct)")
    print()

    if args.json:
        Path(args.json).write_text(json.dumps({
            "n_rules": len(rules),
            "n_distinct": len(distinct),
            "counts": counts,
            "tags": list(contexts_by_tag),
            "min_discrimination": MIN_DISCRIMINATION,
            "rules": [a.__dict__ for a in audits],
        }, indent=2), encoding="utf-8")
        log.info(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
