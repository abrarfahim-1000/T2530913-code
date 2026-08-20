"""
provider.py — serve rules to the shield out of the knowledge graph.

`KgRuleProvider` satisfies `shield.shield.RuleProvider` structurally, so it drops
into the gate and the eval harness with no change to either. It is **opt-in**:
`JsonlRuleProvider` stays the default.

The contract that makes the swap safe, and which `tests/test_kg.py` pins:

    KgRuleProvider(kg).rules_for(ft) == JsonlRuleProvider.from_jsonl(...).rules_for(ft)

as sets of full rule dicts, for every fault type. The graph carries each served
record verbatim, so retrieval cannot alter a threshold, a severity or a role. The
Predicate layer normalizes across noisy `entity` labels for *citation* only — it
never collapses what is served. If it did, `highest_severity` and the
blocks-by-severity histogram would move, and every measured number with them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx

from kg.build import iter_served_rules, load_kg, nodes_of
from kg.schema import (
    CLAUSE,
    PREDICATE,
    RULE,
    SERVED_RULE,
    Citation,
    ClauseRef,
    doc_id,
    rule_id as rule_node_id,
    served_id,
)


class KgRuleProvider:
    """Serves rules from the provenance graph, and can cite any of them.

    Like `JsonlRuleProvider`, every rule applies to every prediction: the corpus
    speaks about generating modules and facilities, not about line 17, and the
    shield gates a graph-level verdict. Discrimination comes from the conditions.
    Entity-based routing would drop `R_1443` from a line-level query purely
    because the extractor labelled it `Facility` while an identical rule from
    another standard was labelled `Line` — which is exactly the mistake the first
    knowledge graph made.
    """

    def __init__(self, kg: nx.DiGraph):
        self._kg = kg
        self._rules = list(iter_served_rules(kg))
        if not self._rules:
            raise ValueError("Knowledge graph contains no ServedRule nodes.")

    @classmethod
    def from_json(cls, path: str | Path) -> "KgRuleProvider":
        return cls(load_kg(path))

    # ── RuleProvider protocol ────────────────────────────────────────────────
    def rules_for(self, fault_type: str) -> list[dict]:
        return [dict(r) for r in self._rules]

    def __len__(self) -> int:
        return len(self._rules)

    def assert_clean(self, healthy_contexts: Sequence[dict]) -> None:
        """Load-time polarity backstop, identical to JsonlRuleProvider's.

        Delegates to the guard so 'fires on a healthy grid' has exactly one
        definition across the pipeline.
        """
        from extraction.polarity_guard import assert_no_rule_fires

        assert_no_rule_fires(self._rules, healthy_contexts)

    # ── citation ─────────────────────────────────────────────────────────────
    @property
    def graph(self) -> nx.DiGraph:
        return self._kg

    def cite(self, rule_id: str) -> Citation:
        """Full provenance chain for a rule the shield acted on.

        Accepts the id of a served rule (what `ShieldResult.violated_rules`
        carries) or of any individual validated rule that deduped into one.
        """
        node = self._resolve(rule_id)
        rec = self._kg.nodes[node]["rule"]

        pred_node = next(
            (n for n in self._kg.successors(node)
             if self._kg.nodes[n].get("type") == PREDICATE),
            None,
        )
        predicate = (
            self._kg.nodes[pred_node]["plain_english"] if pred_node
            else rec.get("condition", "")
        )

        sources: list[ClauseRef] = []
        for rnode in sorted(self._kg.predecessors(node)):
            attrs = self._kg.nodes[rnode]
            if attrs.get("type") != RULE:
                continue
            for cnode in self._kg.predecessors(rnode):
                cattrs = self._kg.nodes[cnode]
                if cattrs.get("type") != CLAUSE:
                    continue
                stem = cattrs["document"]
                sources.append(ClauseRef(
                    rule_id=attrs["rule_id"],
                    clause=cattrs["text"],
                    document=stem,
                    issuing_body=self._kg.nodes[doc_id(stem)].get("issuing_body"),
                ))

        sources.sort(key=lambda s: (s.document, s.clause))
        return Citation(
            served_rule_id=rec.get("rule_id", rule_id),
            condition=rec.get("condition", ""),
            predicate=predicate,
            severity=str(rec.get("severity", "")),
            role=str(rec.get("role", "CONSTRAINT")).upper(),
            sources=sources,
        )

    def _resolve(self, rule_id: str) -> str:
        """rule id -> ServedRule node, following deduped_into if needed."""
        node = served_id(rule_id)
        if node in self._kg:
            return node

        member = rule_node_id(rule_id)
        if member in self._kg:
            for succ in self._kg.successors(member):
                if self._kg.nodes[succ].get("type") == SERVED_RULE:
                    return succ

        known = sorted(self._kg.nodes[n]["rule_id"] for n in nodes_of(self._kg, SERVED_RULE))
        raise KeyError(f"{rule_id!r} is not in the graph. Served rules: {known}")
