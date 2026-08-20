"""
cite.py — turn a ShieldResult into a citation chain.

This lives in `kg/`, not in `shield/`, on purpose. The shield must keep working
with no knowledge graph present — that is stated in its module docstring and it is
what let Component D be built, tested and measured before Component C existed.
Nothing under `shield/` imports anything from here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from kg.schema import Citation, format_citation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kg.provider import KgRuleProvider
    from shield.shield import ShieldResult


def cite_result(result: "ShieldResult", provider: "KgRuleProvider") -> list[Citation]:
    """Citation chain for every rule that produced this verdict.

    Covers `violated_rules` (what blocked) and, on a PASS, `supporting_rules` —
    the asymmetric N-1 gate folds fired constraints into the supporting channel
    when the prediction was already the conservative one, and those are still
    worth citing as corroboration.
    """
    rules = result.violated_rules or result.supporting_rules
    out: list[Citation] = []
    for rule in rules:
        rid = rule.get("rule_id")
        if rid is None:
            continue
        try:
            out.append(provider.cite(rid))
        except KeyError:
            continue
    return out


def explain(result: "ShieldResult", provider: "KgRuleProvider") -> str:
    """`ShieldResult.explanation`, but with the full provenance chain attached.

    The shield's own `build_explanation` already names the rule and its `source`
    string. This adds what a flat rule list cannot: every other clause, in every
    other standard, that states the same check.
    """
    citations = cite_result(result, provider)
    if not citations:
        return result.explanation

    verb = "blocked by" if result.blocked else "corroborated by"
    lines = [f"Prediction {verb} {len(citations)} rule(s):", ""]
    for cit in citations:
        lines.append(format_citation(cit))
        lines.append("")
    return "\n".join(lines).rstrip()
