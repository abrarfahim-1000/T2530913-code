"""
shield.py — the gate itself.

`validate()` takes a context and a plain list of rules. It deliberately does NOT
reach into the knowledge graph: rule retrieval sits behind the `RuleProvider`
protocol, so the KG can be redesigned from scratch against the surviving ruleset
without touching anything here. `JsonlRuleProvider` needs no graph at all, which
is what lets the shield be built and tested before the KG exists.

Interaction with the GNN is strictly post-hoc: the model never sees the rules, the
shield only sees the prediction plus raw telemetry, and there is no bypass path.
Only a VIOLATED verdict blocks — see evaluator.py for why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

from shield.evaluator import Verdict, evaluate_condition

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ── RULE RETRIEVAL ────────────────────────────────────────────────────────────

class RuleProvider(Protocol):
    """Supplies the rules applicable to one prediction.

    Implemented now by JsonlRuleProvider; a KgRuleProvider slots in after the
    knowledge graph is rebuilt, without changing validate() or the eval harness.
    """

    def rules_for(self, fault_type: str) -> list[dict]:
        ...


class JsonlRuleProvider:
    """Serves rules straight from all_rules_deduped.jsonl — no graph required.

    Every rule applies to every prediction: the corpus is dominated by
    system-level entities (grid codes speak about generating modules and the grid,
    not about line 17), and the shield gates a graph-level classification anyway.
    Discrimination comes from the conditions, not from entity routing.
    """

    def __init__(self, rules: Sequence[dict]):
        self._rules = list(rules)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "JsonlRuleProvider":
        rules = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rules.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rules:
            raise ValueError(f"No rules loaded from {path}")
        return cls(rules)

    def rules_for(self, fault_type: str) -> list[dict]:  # noqa: ARG002 - uniform by design
        return list(self._rules)

    def assert_clean(self, healthy_contexts: Sequence[dict]) -> None:
        """Refuse to serve a polarity-contaminated ruleset (handoff §3c item 4).

        Delegates to the guard so 'fires on a healthy grid' has exactly one
        definition across the pipeline.
        """
        from extraction.polarity_guard import assert_no_rule_fires

        assert_no_rule_fires(self._rules, healthy_contexts)

    def __len__(self) -> int:
        return len(self._rules)


# ── RESULT ────────────────────────────────────────────────────────────────────

@dataclass
class ShieldResult:
    status: str                              # "PASS" | "BLOCK"
    fault_type: Optional[str] = None
    confidence: Optional[float] = None
    violated_rules: list[dict] = field(default_factory=list)
    highest_severity: Optional[str] = None
    explanation: str = ""
    n_evaluated: int = 0
    n_not_evaluable: int = 0
    n_error: int = 0
    # ── affirmation channel (Option A: evidence, never a block on its own) ─────
    supporting_rules: list[dict] = field(default_factory=list)   # affirm the prediction
    contradicting_rules: list[dict] = field(default_factory=list)  # affirm another class
    n_affirmations_for_class: int = 0   # how many were available to fire at all

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCK"

    @property
    def unsupported(self) -> bool:
        """True when affirmations for the predicted class existed but none fired.

        This is the signal Option B would block on. Under Option A it is recorded
        only, so the eval harness can report what a gating shield would have done
        without paying its false block rate.
        """
        return self.n_affirmations_for_class > 0 and not self.supporting_rules

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "fault_type": self.fault_type,
            "confidence": self.confidence,
            "violated_rule_ids": [r.get("rule_id") for r in self.violated_rules],
            "highest_severity": self.highest_severity,
            "explanation": self.explanation,
            "n_evaluated": self.n_evaluated,
            "n_not_evaluable": self.n_not_evaluable,
            "n_error": self.n_error,
            "supporting_rule_ids": [r.get("rule_id") for r in self.supporting_rules],
            "contradicting_rule_ids": [r.get("rule_id") for r in self.contradicting_rules],
            "n_affirmations_for_class": self.n_affirmations_for_class,
            "unsupported": self.unsupported,
        }


def build_explanation(violated: Sequence[dict]) -> str:
    """Human-readable justification, most severe first, citing the source standard."""
    if not violated:
        return "No applicable rule was violated."
    lines = [f"Prediction blocked by {len(violated)} rule violation(s):"]
    for rule in violated:
        lines.append(
            f"  [{rule.get('severity', 'unknown').upper()}] "
            f"{rule.get('rule_id', '?')} ({rule.get('source', 'unknown source')}): "
            f"{rule.get('explanation', '').strip()} "
            f"— condition: {rule.get('condition', '')}"
        )
    return "\n".join(lines)


# ── THE GATE ──────────────────────────────────────────────────────────────────

def validate(context: dict, rules: Iterable[dict]) -> ShieldResult:
    """Gate one prediction against the rule corpus.

    `context` comes from shield.context.build_context (raw telemetry, never
    z-scored features). Rules are deduplicated by rule_id and all are evaluated.

    Two channels, because the corpus carries two kinds of knowledge:

    - **CONSTRAINT** rules: condition TRUE means the standard is violated. These
      block, ordered critical > high > medium > low.
    - **AFFIRMATION** rules: condition TRUE means the telemetry is consistent with
      the class in `affirms`. Firing for the predicted class is *support*; firing
      for a different class is a *contradiction*.

    **Option A semantics: affirmations never block.** Absence of support is
    recorded (`unsupported`) so the harness can report what a gating shield would
    have done, but only a constraint violation produces BLOCK. Blocking on missing
    support would make the false block rate a function of how evenly the corpus
    happens to cover the four classes, which is not a property of the grid.
    """
    predicted = context.get("fault_type")
    seen: set[str] = set()
    violated: list[dict] = []
    supporting: list[dict] = []
    contradicting: list[dict] = []
    n_evaluated = n_not_evaluable = n_error = n_affirmations_for_class = 0

    for rule in rules:
        rule_id = rule.get("rule_id")
        if rule_id is not None:
            if rule_id in seen:
                continue
            seen.add(rule_id)

        condition = rule.get("condition")
        if not condition:
            n_error += 1
            continue

        is_affirmation = str(rule.get("role", "CONSTRAINT")).upper() == "AFFIRMATION"
        affirms = str(rule.get("affirms", "")).lower() or None
        if is_affirmation and affirms == predicted:
            n_affirmations_for_class += 1

        verdict = evaluate_condition(condition, dict(context))
        n_evaluated += 1

        if verdict is Verdict.NOT_EVALUABLE:
            n_not_evaluable += 1
        elif verdict is Verdict.ERROR:
            n_error += 1
        elif verdict is Verdict.VIOLATED:      # condition evaluated TRUE
            if not is_affirmation:
                violated.append(rule)
            elif affirms == predicted:
                supporting.append(rule)
            elif predicted is not None:
                contradicting.append(rule)

    violated.sort(key=lambda r: SEVERITY_ORDER.get(str(r.get("severity", "low")).lower(), 3))

    return ShieldResult(
        status="BLOCK" if violated else "PASS",
        fault_type=predicted,
        confidence=context.get("confidence"),
        violated_rules=violated,
        highest_severity=violated[0].get("severity") if violated else None,
        explanation=build_explanation(violated),
        n_evaluated=n_evaluated,
        n_not_evaluable=n_not_evaluable,
        n_error=n_error,
        supporting_rules=supporting,
        contradicting_rules=contradicting,
        n_affirmations_for_class=n_affirmations_for_class,
    )


# ── N-1 GATE (asymmetric) ─────────────────────────────────────────────────────

SECURE, VIOLATION = "secure", "violation"


def validate_n1(context: dict, rules: Iterable[dict], predicted: str) -> ShieldResult:
    """Gate one N-1 *contingency* verdict.

    What this can and cannot do, stated plainly because the distinction matters
    for what the thesis is allowed to claim:

    The shield **cannot** verify the contingency itself. Establishing whether
    losing line k actually violates a limit requires a post-contingency power
    flow, which is precisely the computation the GNN is standing in for. If the
    shield could run it, the GNN would be unnecessary.

    What it verifies instead is that the verdict is **consistent with the rules
    governing the present state**. N-1 security presupposes a secure base case:
    a grid already violating a thermal or voltage constraint cannot be declared
    safe against the loss of any further element. So a `secure` verdict issued
    over a base case that breaks a CONSTRAINT rule is unsupportable, and blocks.

    **The gate is asymmetric, and only the permissive direction blocks:**

      predicted `secure`    + base case violates a constraint  -> BLOCK
      predicted `violation` + anything                         -> PASS

    Predicting `violation` on a calm grid is a false alarm: wasteful, but it
    fails toward caution and a safety gate must never suppress it. Predicting
    `secure` on a grid that is already outside its limits is the failure that
    actually endangers a network, and it is the only one worth blocking.

    Affirmations keep Option A semantics from `validate()`: they supply
    supporting evidence for a `secure` verdict and never block on their own.
    """
    result = validate({**context, "fault_type": predicted}, rules)

    if predicted == SECURE:
        return result

    # Conservative prediction: record the evidence, withhold the block.
    return ShieldResult(
        status="PASS",
        fault_type=predicted,
        confidence=result.confidence,
        violated_rules=[],
        highest_severity=None,
        explanation=(
            f"Prediction '{predicted}' is the conservative verdict; the shield gates "
            f"only over-permissive predictions. "
            f"{len(result.violated_rules)} constraint(s) fired on the base case and "
            f"are recorded as corroboration, not as grounds to block."
        ),
        n_evaluated=result.n_evaluated,
        n_not_evaluable=result.n_not_evaluable,
        n_error=result.n_error,
        supporting_rules=result.violated_rules + result.supporting_rules,
        contradicting_rules=result.contradicting_rules,
        n_affirmations_for_class=result.n_affirmations_for_class,
    )
