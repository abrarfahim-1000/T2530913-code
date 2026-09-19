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

from kg.build import (
    explanatory_corpus_status,
    iter_explanatory_rules,
    iter_served_rules,
    load_kg,
    nodes_of,
    refresh_explanatory_payloads,
)
from kg.schema import (
    CLAUSE,
    EXPLANATORY_RULE,
    PREDICATE,
    RULE,
    SERVED_RULE,
    Citation,
    ClauseRef,
    doc_id,
    explanatory_id,
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

    def __init__(self, kg: nx.DiGraph, include_explanatory: bool = False):
        self._kg = kg
        self._rules = list(iter_served_rules(kg))
        if not self._rules:
            raise ValueError("Knowledge graph contains no ServedRule nodes.")
        # OPT-IN, and off by default on purpose. Every shield figure in the
        # thesis was measured on the served four; widening what the provider
        # hands the gate without being asked would move them silently, and
        # `test_kg_provider_serves_exactly_what_the_jsonl_provider_serves`
        # exists to make that impossible.
        # Heal before reading, not after: the payloads are what gets served.
        self.refreshed = 0
        if include_explanatory:
            self.refreshed = self._sync_with_corpus()
        self._explanatory = list(iter_explanatory_rules(kg)) if include_explanatory else []
        if include_explanatory and not self._explanatory:
            raise ValueError(
                "include_explanatory=True but the graph carries no "
                "ExplanatoryRule nodes. Build it with "
                "scripts/build_kg.py --channels."
            )


    @classmethod
    def from_json(cls, path: str | Path,
                  include_explanatory: bool = False) -> "KgRuleProvider":
        return cls(load_kg(path), include_explanatory=include_explanatory)

    # ── RuleProvider protocol ────────────────────────────────────────────────
    def rules_for(self, fault_type: str) -> list[dict]:
        # The explanatory records carry an explicit `channel`, so the gate's
        # own partition keeps them off the veto path - a WARN rule cannot
        # block by arriving here (shield/channels.py::VETO_CHANNELS).
        return [dict(r) for r in (*self._rules, *self._explanatory)]

    def __len__(self) -> int:
        return len(self._rules) + len(self._explanatory)

    def assert_clean(self, healthy_contexts: Sequence[dict]) -> None:
        """Load-time polarity backstop, identical to JsonlRuleProvider's.

        Delegates to the guard so 'fires on a healthy grid' has exactly one
        definition across the pipeline.
        """
        from extraction.polarity_guard import assert_no_rule_fires

        assert_no_rule_fires(self._rules, healthy_contexts)

    def _sync_with_corpus(self) -> int:
        """Bring the layer up to date with the corpus, or say why it cannot.

        The graph stores rule dicts verbatim, so it is a snapshot of a file
        that keeps being regenerated. Rather than make every caller remember
        to rebuild after stamping - the caller that forgot was the only one
        that knew to - a payload-only drift is repaired here and the caller
        gets current data. A drift that would move EDGES is refused, because
        repairing that silently would be inventing provenance.
        """
        result = refresh_explanatory_payloads(self._kg)
        if result["state"] in ("current", "absent", "missing"):
            return 0
        if result["state"] == "refreshed":
            return int(result["refreshed"])
        raise ValueError(
            f"The explanatory layer no longer matches {result['path']} and "
            f"cannot be repaired in place: {result.get('reason')}. The clause "
            f"and predicate edges would have to move, which is a rebuild, not "
            f"a refresh. Run: python scripts/build_kg.py --channels"
        )

    def _assert_corpus_current(self) -> None:
        """Refuse a layer built from a corpus that has since changed.

        The graph stores each rule dict verbatim, so it is a snapshot. Restamp
        the corpus without rebuilding and the gate is handed the old records
        while every count still reconciles - which is how a three-grid run
        applied no per-grid demotion at all and reported success. Checked here
        rather than left to the caller, because the caller that got it wrong
        was the one place that knew to check.
        """
        status = explanatory_corpus_status(self._kg)
        if status["state"] != "stale":
            return
        raise ValueError(
            f"The explanatory layer is STALE: {status['path']} has changed "
            f"since the graph was built (built from {status['sha256'][:12]}, "
            f"on disk now {status['actual_sha256'][:12]}). The graph carries "
            f"rule dicts verbatim, so serving it would hand the shield the old "
            f"records. Rebuild: python scripts/build_kg.py --channels"
        )

    # ── citation ─────────────────────────────────────────────────────────────
    @property
    def graph(self) -> nx.DiGraph:
        return self._kg

    def cite(self, rule_id: str, layer: str | None = None) -> Citation:
        """Full provenance chain for a rule the shield acted on.

        Accepts the id of a served rule (what `ShieldResult.violated_rules`
        carries) or of any individual validated rule that deduped into one.

        `layer` disambiguates an id that exists in BOTH layers, which happens
        for every thermal rule: R_1492 is a validated Rule that deduped into
        served R_1443, AND a record of its own in the channel corpus. Left
        `None` the served group wins, so the veto path cites the corroborated
        group exactly as it always has. Pass `EXPLANATORY_RULE` when reporting
        what SPOKE on a channel: there the honest attribution is the record
        that actually fired, not the group it would have deduped into.
        """
        node = self._resolve(rule_id, layer=layer)
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

        sources = self._sources_for(node)
        sources.sort(key=lambda s: (s.document, s.clause))
        return Citation(
            served_rule_id=rec.get("rule_id", rule_id),
            condition=rec.get("condition", ""),
            predicate=predicate,
            severity=str(rec.get("severity", "")),
            role=str(rec.get("role", "CONSTRAINT")).upper(),
            sources=sources,
            channel=self._kg.nodes[node].get("channel"),
        )

    def _sources_for(self, node: str) -> list[ClauseRef]:
        """Every clause stating this rule, for either node shape.

        A ServedRule is reached from its clauses through the Rule layer
        (Clause -> Rule -> ServedRule); an ExplanatoryRule hangs off its
        clause directly. Both are walked here so a WARN rule cites the same
        way a BLOCK rule does.
        """
        sources: list[ClauseRef] = []
        for parent in sorted(self._kg.predecessors(node)):
            attrs = self._kg.nodes[parent]
            ptype = attrs.get("type")
            if ptype == CLAUSE:
                stem = attrs["document"]
                sources.append(ClauseRef(
                    rule_id=self._kg.nodes[node]["rule_id"],
                    clause=attrs["text"],
                    document=stem,
                    issuing_body=self._kg.nodes[doc_id(stem)].get("issuing_body"),
                ))
            elif ptype == RULE:
                for cnode in self._kg.predecessors(parent):
                    cattrs = self._kg.nodes[cnode]
                    if cattrs.get("type") != CLAUSE:
                        continue
                    stem = cattrs["document"]
                    sources.append(ClauseRef(
                        rule_id=attrs["rule_id"],
                        clause=cattrs["text"],
                        document=stem,
                        issuing_body=self._kg.nodes[doc_id(stem)].get(
                            "issuing_body"),
                    ))
        return sources

    def _resolve(self, rule_id: str, layer: str | None = None) -> str:
        """rule id -> the node that carries it.

        Served first, so a rule that can veto always cites through the chain
        its measured numbers were produced on, even when the explanatory layer
        also carries a record with that id. `layer` overrides that, and is
        strict: asking for a layer that has no such node raises rather than
        silently handing back the other layer's answer.
        """
        if layer == EXPLANATORY_RULE:
            node = explanatory_id(rule_id)
            if node in self._kg:
                return node
            raise KeyError(
                f"{rule_id!r} has no ExplanatoryRule node. Build the graph "
                f"with scripts/build_kg.py --channels.")

        node = served_id(rule_id)
        if node in self._kg:
            return node

        member = rule_node_id(rule_id)
        if member in self._kg:
            for succ in self._kg.successors(member):
                if self._kg.nodes[succ].get("type") == SERVED_RULE:
                    return succ

        explanatory = explanatory_id(rule_id)
        if explanatory in self._kg:
            return explanatory

        known = sorted(
            self._kg.nodes[n]["rule_id"]
            for n in (*nodes_of(self._kg, SERVED_RULE),
                      *nodes_of(self._kg, EXPLANATORY_RULE))
        )
        raise KeyError(
            f"{rule_id!r} is not in the graph. Citable rules: {known}")
