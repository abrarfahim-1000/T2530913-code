"""
schema.py — node and edge vocabulary for the provenance knowledge graph.

The graph answers one question: **when the shield blocks, where did that rule
come from, and who else says the same thing?**

It is deliberately NOT built around grid topology. The first iteration hung rules
off Bus/Line/Generator nodes taken from one grid's metadata; `Line` and `Bus`
accounted for 3.5% of the corpus, so 6,375 of its 6,632 edges were `has_rule`
noise, and the whole graph was welded to the 36-bus topology while the shield runs
on three. The rules are topology-agnostic by construction, so the graph is too.

Layers, left to right:

    Document --contains--> Clause --states--> Rule --deduped_into--> ServedRule
        --instantiates--> Predicate --reads--> Variable

`ServedRule` is the layer the shield actually consumes: one node per record in
`all_rules_deduped.jsonl`, carrying that record verbatim. `Predicate` sits behind
it and is the graph's own contribution — dedup keys on `(entity, condition)`, so
one physical check fragments across several ServedRules whenever the extractor
labelled it `Line` in one standard and `Facility` in another, or wrote `100` here
and `100.0` there. The Predicate layer puts those back together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── NODE TYPES ────────────────────────────────────────────────────────────────
DOCUMENT = "Document"
CLAUSE = "Clause"
RULE = "Rule"
SERVED_RULE = "ServedRule"
PREDICATE = "Predicate"
VARIABLE = "Variable"

NODE_TYPES = (DOCUMENT, CLAUSE, RULE, SERVED_RULE, PREDICATE, VARIABLE)

# ── EDGE TYPES ────────────────────────────────────────────────────────────────
CONTAINS = "contains"            # Document  -> Clause
STATES = "states"                # Clause    -> Rule
DEDUPED_INTO = "deduped_into"    # Rule      -> ServedRule
INSTANTIATES = "instantiates"    # ServedRule-> Predicate
READS = "reads"                  # Predicate -> Variable

EDGE_TYPES = (CONTAINS, STATES, DEDUPED_INTO, INSTANTIATES, READS)

#: The layer order, used by the figures and by the orphan checks.
LAYERS = (DOCUMENT, CLAUSE, RULE, SERVED_RULE, PREDICATE, VARIABLE)


# ── NODE IDS ──────────────────────────────────────────────────────────────────
# Prefixed so a rule and the ServedRule representing its group can share an id
# without colliding, and so a JSON dump stays readable.

def doc_id(stem: str) -> str:
    return f"{DOCUMENT}:{stem}"


def clause_id(n: int) -> str:
    return f"{CLAUSE}:C_{n:03d}"


def rule_id(rid: str) -> str:
    return f"{RULE}:{rid}"


def served_id(rid: str) -> str:
    return f"{SERVED_RULE}:{rid}"


def predicate_id(n: int) -> str:
    return f"{PREDICATE}:P_{n:03d}"


def variable_id(name: str) -> str:
    return f"{VARIABLE}:{name}"


# ── ISSUING BODIES ────────────────────────────────────────────────────────────
# Hand-labelled from the document titles, NOT inferred by the pipeline. `None`
# means the title does not identify the body unambiguously — leave it None rather
# than guess, because the number of distinct bodies backing a predicate is a
# reported figure and must not be inflated by a guess.
ISSUING_BODY: dict[str, str | None] = {
    "NERC Reliability Standards (Complete Set - PRC, TOP, TPL, VAR, BAL, CIP, EOP, FAC, MOD)": "NERC",
    "TPL-001-5.1 — Transmission System Planning Performance Requirements": "NERC",
    "inverter-based_resource_performance_guideline": "NERC",
    "guideline-ieee_1547-2018_bps_perspectives_clean-1": "IEEE",
    "Bangladesh_Electricity-Grid-Code-regulations": "Bangladesh Energy Regulatory Commission",
    "130308_Final_Version_NC_RfG": "ENTSO-E",
    "8589935310-Complete Grid Code": "National Grid ESO (GB)",
    "prc-024-4": "NERC",
    "prc-025-2": "NERC",
    "prc-029-1": "NERC",
    "prc-006-5": "NERC",
    "prc-023-6": "NERC",
    "NERC FAC-008-5 — Facility Ratings": "NERC",
    "order-842": "FERC",
    "RM16-1-000": "FERC",
    "power-system-requirements": None,   # "EMO Dispatch Computer Constraints" — operator not identified
}


def issuing_body(stem: str) -> str | None:
    return ISSUING_BODY.get(stem)


# ── CITATIONS ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClauseRef:
    """One place in one standard that states a rule."""

    rule_id: str
    clause: str
    document: str
    issuing_body: str | None = None

    def __str__(self) -> str:
        body = f"{self.issuing_body}, " if self.issuing_body else ""
        return f"{body}{self.clause} [{self.rule_id}]"


@dataclass
class Citation:
    """Why the shield was entitled to act on one served rule.

    `sources` is every clause that states this check, across every document —
    which is the thing a flat rule list cannot express. `n_documents` and
    `n_bodies` are the corroboration counts.
    """

    served_rule_id: str
    condition: str
    predicate: str
    severity: str
    role: str
    sources: list[ClauseRef] = field(default_factory=list)

    @property
    def n_documents(self) -> int:
        return len({s.document for s in self.sources})

    @property
    def n_bodies(self) -> int:
        return len({s.issuing_body for s in self.sources if s.issuing_body})

    def to_dict(self) -> dict:
        return {
            "served_rule_id": self.served_rule_id,
            "condition": self.condition,
            "predicate": self.predicate,
            "severity": self.severity,
            "role": self.role,
            "n_clauses": len(self.sources),
            "n_documents": self.n_documents,
            "n_bodies": self.n_bodies,
            "sources": [
                {
                    "rule_id": s.rule_id,
                    "clause": s.clause,
                    "document": s.document,
                    "issuing_body": s.issuing_body,
                }
                for s in self.sources
            ],
        }


def format_citation(cit: Citation, indent: str = "  ") -> str:
    """Human-readable citation chain, for a block explanation or the console."""
    head = (
        f"{cit.served_rule_id} [{cit.severity.upper()}] {cit.role}: {cit.predicate}\n"
        f"{indent}condition: {cit.condition}\n"
        f"{indent}stated in {len(cit.sources)} clause(s) across "
        f"{cit.n_documents} document(s)"
    )
    if cit.n_bodies:
        head += f", {cit.n_bodies} standards body/bodies"
    lines = [head]
    for src in cit.sources:
        lines.append(f"{indent}  - {src.clause} — {src.document} [{src.rule_id}]")
    return "\n".join(lines)
