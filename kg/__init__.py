"""
Component C — the provenance knowledge graph.

Answers one question: **when the shield blocks, where did that rule come from,
and who else says the same thing?**

    Document --contains--> Clause --states--> Rule --deduped_into--> ServedRule
        --instantiates--> Predicate --reads--> Variable

Topology-agnostic by construction, so one graph serves all three grids. Retrieval
through `KgRuleProvider` is opt-in and returns exactly what `JsonlRuleProvider`
returns — pinned by `tests/test_kg.py` — so the measured shield results cannot
move when the graph is used.

Layout:
    schema.py    node/edge vocabulary, Citation dataclasses
    build.py     build_knowledge_graph(), save_kg/load_kg, predicate normalization
    provider.py  KgRuleProvider — RuleProvider + cite()
    cite.py      ShieldResult -> citation chain (keeps shield/ free of kg imports)
    figures.py   static thesis figures
"""

from kg.build import (
    build_knowledge_graph,
    kg_stats,
    load_kg,
    normalize_predicate,
    plain_english,
    save_kg,
    used_variables,
)
from kg.cite import cite_result, explain
from kg.provider import KgRuleProvider
from kg.schema import Citation, ClauseRef, format_citation

__all__ = [
    "build_knowledge_graph",
    "save_kg",
    "load_kg",
    "kg_stats",
    "normalize_predicate",
    "plain_english",
    "used_variables",
    "KgRuleProvider",
    "Citation",
    "ClauseRef",
    "format_citation",
    "cite_result",
    "explain",
]
