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

from shield.channels import (
    BLOCK,
    NORMAL,
    NOT_APPLICABLE,
    WARN,
    calibration_of,
    is_cross_vocabulary,
    partition_by_channel,
)
from shield.evaluator import Verdict, evaluate_condition

#: v1 had one channel: a rule vetoed or it was discarded. v2 adds WARN / NORMAL /
#: NOT_APPLICABLE and renders the PASS path, so 41 rules speak where 5 did
#: (thesis_findings 19.3, 21.3). The BLOCK channel is byte-for-byte v1 behaviour,
#: which is why 13's intervention precision is unmoved.
SHIELD_VERSION = "2.0"

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
    # ── v2 explanation channels ───────────────────────────────────────────────
    # WARN rules that fired. These NEVER appear in `violated_rules` and never
    # change `status`; they annotate a prediction that was let through.
    warning_rules: list[dict] = field(default_factory=list)
    # Rules carried for citation that were evaluated but cannot speak: their
    # calibration does not transfer, or it was measured and did not earn a voice.
    not_applicable_rules: list[dict] = field(default_factory=list)
    n_by_channel: dict = field(default_factory=dict)
    version: str = SHIELD_VERSION

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCK"

    @property
    def warned(self) -> bool:
        """Let through, with at least one calibrated warning attached."""
        return self.status == "PASS" and bool(self.warning_rules)

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
            "shield_version": self.version,
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
            "warning_rule_ids": [r.get("rule_id") for r in self.warning_rules],
            "not_applicable_rule_ids": [r.get("rule_id") for r in self.not_applicable_rules],
            "n_by_channel": dict(self.n_by_channel),
        }


def build_explanation(violated: Sequence[dict]) -> str:
    """The BLOCK register: why this prediction was refused, most severe first.

    Unchanged from v1, including the empty-case string, because it is what every
    recorded result was rendered with. The v2 additions live in
    `render_explanation`, which calls this for the block paragraph.
    """
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


def _cite(rule: dict) -> str:
    return f"{rule.get('rule_id', '?')} ({rule.get('source', 'unknown source')})"


def _rate_clause(rule: dict) -> str:
    """The measured N-1 rate a warning is allowed to quote, or an honest blank.

    A warning that cannot say how often it is right is an alarm. When the
    calibration is missing the text says so, rather than implying a rate the
    rule does not have.
    """
    cal = calibration_of(rule)
    if not cal or cal.get("p_violation_given_fires") is None:
        return "no measured rate on this task"
    p = cal["p_violation_given_fires"]
    quiet = cal.get("p_violation_given_silent")
    grids = cal.get("grids") or []
    where = f"; measured on {', '.join(grids)}" if grids else ""
    if quiet is None:
        return f"observed in {p:.1%} of such states{where}"
    return (f"{p:.1%} of contingencies violated a limit when this fired, against "
            f"{quiet:.1%} when it did not{where}")


def render_explanation(result: "ShieldResult") -> str:
    """The full four-channel explanation — the reason v2 exists.

    v1 rendered one path. On a PASS it returned "No applicable rule was
    violated.", which is true and says nothing: an operator cannot tell a grid
    with nine satisfied standards from a grid no rule could evaluate. Three
    registers, in the order an operator reads them:

      BLOCK    what was refused, and under which clause.
      WARN     what was let through, what could still happen, and how often that
               has actually happened on this task.
      NORMAL   which standards affirmatively hold right now.

    Each channel renders from the list the gate filled separately, so a warning
    cannot be typeset as a block by a formatting mistake.
    """
    paras: list[str] = []

    if result.violated_rules:
        paras.append(build_explanation(result.violated_rules))
    else:
        paras.append(
            f"Prediction allowed: no applicable rule was violated "
            f"({result.n_by_channel.get(BLOCK, 0)} constraint(s) checked)."
        )

    if result.warning_rules:
        head = (f"Warning - let through, but {len(result.warning_rules)} rule(s) "
                f"report a condition that raises N-1 risk:")
        body = [
            f"  [WARN] {_cite(rule)}: {rule.get('explanation', '').strip()} "
            f"- condition: {rule.get('condition', '')} - {_rate_clause(rule)}"
            for rule in result.warning_rules
        ]
        paras.append("\n".join([head, *body]))

    if result.supporting_rules:
        head = (f"Environment normal according to {len(result.supporting_rules)} "
                f"rule(s):")
        body = [
            f"  [OK] {_cite(rule)}: {rule.get('explanation', '').strip()} "
            f"- condition holds: {rule.get('condition', '')}"
            for rule in result.supporting_rules
        ]
        paras.append("\n".join([head, *body]))

    if result.contradicting_rules:
        # An affirmation written against the retired 4-class vocabulary is not a
        # contradiction of an N-1 verdict — `normal` and `secure` describe the
        # same physical state under two task definitions. The gate still files it
        # as contradicting, because changing that would move the Option B
        # counterfactual; the text is what stops it being read as disagreement.
        crossed = [r for r in result.contradicting_rules
                   if is_cross_vocabulary(r.get("affirms"), result.fault_type)]
        genuine = [r for r in result.contradicting_rules if r not in crossed]
        if genuine:
            head = (f"{len(genuine)} rule(s) affirm a state other than the one "
                    f"predicted:")
            body = [f"  [?] {_cite(rule)}: affirms `{rule.get('affirms')}`"
                    for rule in genuine]
            paras.append("\n".join([head, *body]))
        if crossed:
            head = (f"{len(crossed)} rule(s) hold, but affirm a state in the "
                    f"retired 4-class vocabulary, which does not compare with an "
                    f"N-1 verdict — reported, not counted either way:")
            body = [f"  [--] {_cite(rule)}: affirms `{rule.get('affirms')}`"
                    for rule in crossed]
            paras.append("\n".join([head, *body]))

    if result.not_applicable_rules:
        paras.append(
            f"{len(result.not_applicable_rules)} further rule(s) fired but are "
            f"carried for citation only: their calibration does not transfer to "
            f"this grid (thesis_findings 11.2, 21.2)."
        )

    return "\n".join(paras)


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
    buckets = partition_by_channel(rules)

    violated: list[dict] = []
    warnings: list[dict] = []
    supporting: list[dict] = []
    contradicting: list[dict] = []
    not_applicable: list[dict] = []
    n_evaluated = n_not_evaluable = n_error = n_affirmations_for_class = 0

    def fired(rule: dict) -> bool:
        """Evaluate one rule, maintaining the shared verdict counters."""
        nonlocal n_evaluated, n_not_evaluable, n_error
        condition = rule.get("condition")
        if not condition:
            n_error += 1
            return False
        verdict = evaluate_condition(condition, dict(context))
        n_evaluated += 1
        if verdict is Verdict.NOT_EVALUABLE:
            n_not_evaluable += 1
        elif verdict is Verdict.ERROR:
            n_error += 1
        return verdict is Verdict.VIOLATED

    # ── the veto path ─────────────────────────────────────────────────────────
    # This loop iterates the BLOCK bucket and nothing else. A WARN rule cannot
    # reach `violated` by being mislabelled or mis-sorted, because it is not in
    # the list. Pinned by tests/test_shield_channels.py.
    for rule in buckets[BLOCK]:
        is_affirmation = str(rule.get("role", "CONSTRAINT")).upper() == "AFFIRMATION"
        affirms = str(rule.get("affirms", "")).lower() or None
        if is_affirmation and affirms == predicted:
            n_affirmations_for_class += 1
        if not fired(rule):
            continue
        if not is_affirmation:
            violated.append(rule)
        elif affirms == predicted:
            supporting.append(rule)
        elif predicted is not None:
            contradicting.append(rule)

    # ── the annotation paths — none of these can change `status` ──────────────
    for rule in buckets[WARN]:
        if fired(rule):
            warnings.append(rule)

    for rule in buckets[NORMAL]:
        affirms = str(rule.get("affirms", "")).lower() or None
        if affirms == predicted:
            n_affirmations_for_class += 1
        if not fired(rule):
            continue
        # Exactly v1's branch, including the `affirms is None` case falling to
        # `contradicting`: an affirmation that does not say what it affirms is
        # not support for anything.
        if affirms == predicted:
            supporting.append(rule)
        elif predicted is not None:
            contradicting.append(rule)

    for rule in buckets[NOT_APPLICABLE]:
        if fired(rule):
            not_applicable.append(rule)

    # INERT rules are not evaluated: by measurement they cannot fire on any class
    # on any grid (19.3), so evaluating them buys nothing but a counter.

    violated.sort(key=lambda r: SEVERITY_ORDER.get(str(r.get("severity", "low")).lower(), 3))

    result = ShieldResult(
        status="BLOCK" if violated else "PASS",
        fault_type=predicted,
        confidence=context.get("confidence"),
        violated_rules=violated,
        highest_severity=violated[0].get("severity") if violated else None,
        explanation="",
        n_evaluated=n_evaluated,
        n_not_evaluable=n_not_evaluable,
        n_error=n_error,
        supporting_rules=supporting,
        contradicting_rules=contradicting,
        n_affirmations_for_class=n_affirmations_for_class,
        warning_rules=warnings,
        not_applicable_rules=not_applicable,
        n_by_channel={c: len(v) for c, v in buckets.items() if v},
    )
    result.explanation = render_explanation(result)
    return result


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
    corroboration = (
        f"Prediction '{predicted}' is the conservative verdict; the shield gates "
        f"only over-permissive predictions. "
        f"{len(result.violated_rules)} constraint(s) fired on the base case and "
        f"are recorded as corroboration, not as grounds to block."
    )
    downgraded = ShieldResult(
        status="PASS",
        fault_type=predicted,
        confidence=result.confidence,
        violated_rules=[],
        highest_severity=None,
        explanation=corroboration,
        n_evaluated=result.n_evaluated,
        n_not_evaluable=result.n_not_evaluable,
        n_error=result.n_error,
        supporting_rules=result.violated_rules + result.supporting_rules,
        contradicting_rules=result.contradicting_rules,
        n_affirmations_for_class=result.n_affirmations_for_class,
        # Warnings survive the downgrade. A `violation` verdict is not a reason to
        # withhold the reason it was right -- the operator still wants to read
        # which standard says the grid is near an edge.
        warning_rules=result.warning_rules,
        not_applicable_rules=result.not_applicable_rules,
        n_by_channel=dict(result.n_by_channel),
    )
    if downgraded.warning_rules:
        downgraded.explanation = corroboration + "\n" + render_explanation(downgraded)
    return downgraded
