"""
Tests for Component C, the provenance knowledge graph.

The first test is the load-bearing one. Every shield number in the thesis was
measured with `JsonlRuleProvider`; the knowledge graph is opt-in precisely so that
"did the KG change your results?" can be answered with a test rather than an
argument. If `KgRuleProvider` ever returns a different set, that answer becomes
"yes" and the measured results are void.

The rest encode the failure that killed the first knowledge graph: it was built
around grid topology, hanging rules off Bus/Line nodes taken from one grid's
metadata. `Line` and `Bus` were 3.5% of the corpus, so 6,375 of its 6,632 edges
carried no information, and the graph was welded to the 36-bus grid while the
shield runs on three. `test_graph_is_topology_agnostic` is the regression guard.

Run: pytest tests/test_kg.py -v
"""
import json
import os
import re
import sys

import networkx as nx
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kg.build import (  # noqa: E402
    build_knowledge_graph,
    kg_stats,
    load_kg,
    normalize_predicate,
    nodes_of,
    plain_english,
    save_kg,
    used_variables,
)
from kg.provider import KgRuleProvider  # noqa: E402
from kg.schema import (  # noqa: E402
    CLAUSE,
    DOCUMENT,
    PREDICATE,
    RULE,
    SERVED_RULE,
    VARIABLE,
    format_citation,
)
from shield.shield import JsonlRuleProvider  # noqa: E402

CONFIRMED_DIR = "validated_translated"
DEDUPED = "validated_translated/all_rules_deduped.jsonl"


@pytest.fixture(scope="module")
def kg():
    if not os.path.exists(DEDUPED):
        pytest.skip(f"{DEDUPED} not on disk — run extraction/validate.py")
    return build_knowledge_graph(CONFIRMED_DIR, DEDUPED, "rules_35b")


# ── THE GUARANTEE ─────────────────────────────────────────────────────────────

def test_kg_provider_serves_exactly_what_the_jsonl_provider_serves(kg):
    """The reason the KG is opt-in. Compares full dicts, not just ids: a changed
    severity or threshold would move `highest_severity`, the blocks-by-severity
    histogram, and every number measured against them."""
    jsonl = JsonlRuleProvider.from_jsonl(DEDUPED)
    graph = KgRuleProvider(kg)

    assert len(graph) == len(jsonl)
    for fault_type in ("secure", "violation", "normal", "overload"):
        a = sorted(json.dumps(r, sort_keys=True) for r in jsonl.rules_for(fault_type))
        b = sorted(json.dumps(r, sort_keys=True) for r in graph.rules_for(fault_type))
        assert a == b, f"rule sets diverge for {fault_type!r}"


def test_served_rules_are_carried_verbatim(kg):
    """The graph stores the record, it does not rebuild it from node attributes."""
    on_disk = {}
    with open(DEDUPED, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                on_disk[rec["rule_id"]] = rec

    for node in nodes_of(kg, SERVED_RULE):
        rec = kg.nodes[node]["rule"]
        assert rec == on_disk[rec["rule_id"]]


def test_rules_for_returns_copies_so_a_caller_cannot_corrupt_the_graph(kg):
    provider = KgRuleProvider(kg)
    first = provider.rules_for("secure")
    first[0]["condition"] = "MUTATED"
    assert provider.rules_for("secure")[0]["condition"] != "MUTATED"


# ── PREDICATE NORMALIZATION ───────────────────────────────────────────────────

def test_numeric_formatting_does_not_fragment_a_predicate():
    """`loading_pct > 100` and `loading_pct > 100.0` are the same physical check.
    Dedup keys on the raw string, so it splits them; the graph must not."""
    assert normalize_predicate("loading_pct > 100") == normalize_predicate("loading_pct > 100.0")
    assert normalize_predicate("loading_pct>100") == normalize_predicate("loading_pct > 100")


def test_genuinely_different_thresholds_stay_different():
    assert normalize_predicate("loading_pct > 100") != normalize_predicate("loading_pct > 90")


def test_fractional_literals_survive_normalization():
    """0.9 must not be rounded away by the integer canonicalization."""
    canon = normalize_predicate("voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1")
    assert "0.9" in canon and "1.1" in canon


def test_the_entity_split_does_not_fragment_the_thermal_check(kg):
    """Three served rules carry `loading_pct > 100`-equivalent conditions under
    two different `entity` labels (`Line`, `Facility`) — noise in LLM output, not
    physics. They must resolve to ONE predicate."""
    thermal = [
        p for p in kg_stats(kg)["predicates"] if "loading_pct" in p["canonical"]
    ]
    assert len(thermal) == 1, "the thermal check fragmented across predicates"
    assert thermal[0]["n_served_rules"] > 1


def test_malformed_conditions_do_not_crash_normalization():
    assert normalize_predicate("loading_pct >>> 100") == "loading_pct >>> 100"
    assert used_variables("loading_pct >>> 100") == []


def test_plain_english_reads_from_the_vocabulary():
    text = plain_english("loading_pct > 100")
    assert "exceeds" in text and "100" in text
    assert "loading_pct" not in text, "should use the description, not the identifier"


# ── PROVENANCE AND CITATION ───────────────────────────────────────────────────

def test_citation_recovers_every_clause_that_states_a_rule(kg):
    """The thing a flat rule list cannot do. R_1443's deduped group is stated by
    several clauses across more than one document."""
    cit = KgRuleProvider(kg).cite("R_1443")
    assert cit.served_rule_id == "R_1443"
    assert len(cit.sources) > 1
    assert cit.n_documents > 1
    assert all(s.clause and s.document for s in cit.sources)


def test_citation_is_reachable_from_a_member_rule_id_too(kg):
    """R_167 deduped into R_1443; citing either must reach the same group."""
    provider = KgRuleProvider(kg)
    assert provider.cite("R_167").served_rule_id == provider.cite("R_1443").served_rule_id
    assert "R_167" in {s.rule_id for s in provider.cite("R_1443").sources}


def test_citation_reports_corroboration_counts(kg):
    stats = {p["canonical"]: p for p in kg_stats(kg)["predicates"]}
    thermal = stats["loading_pct > 100"]
    assert thermal["n_clauses"] > thermal["n_documents"] > 1
    assert thermal["n_bodies"] >= 1


def test_format_citation_names_the_rule_the_clause_and_the_document(kg):
    text = format_citation(KgRuleProvider(kg).cite("R_1443"))
    assert "R_1443" in text and "condition:" in text
    assert "document(s)" in text


def test_unknown_rule_id_raises_rather_than_returning_an_empty_citation(kg):
    with pytest.raises(KeyError):
        KgRuleProvider(kg).cite("R_NOT_A_RULE")


# ── STRUCTURE ─────────────────────────────────────────────────────────────────

def test_every_served_rule_traces_back_to_a_document(kg):
    """No orphans: a rule the shield can act on must be citable."""
    docs = set(nodes_of(kg, DOCUMENT))
    undirected = kg.to_undirected()
    for node in nodes_of(kg, SERVED_RULE):
        reachable = nx.node_connected_component(undirected, node)
        assert reachable & docs, f"{node} has no document provenance"


def test_each_validated_rule_has_exactly_one_clause_and_one_served_rule(kg):
    for node in nodes_of(kg, RULE):
        parents = [n for n in kg.predecessors(node)
                   if kg.nodes[n].get("type") == CLAUSE]
        children = [n for n in kg.successors(node)
                    if kg.nodes[n].get("type") == SERVED_RULE]
        assert len(parents) == 1, f"{node} has {len(parents)} clauses"
        assert len(children) == 1, f"{node} has {len(children)} served rules"


def test_every_predicate_reads_at_least_one_known_variable(kg):
    for node in nodes_of(kg, PREDICATE):
        variables = [n for n in kg.successors(node)
                     if kg.nodes[n].get("type") == VARIABLE]
        assert variables, f"{node} reads no variable"
        for v in variables:
            assert kg.nodes[v]["description"], f"{v} is not in CONDITION_VOCABULARY"


def test_a_rule_with_no_deduped_representative_raises(tmp_path):
    """Silently dropping such a rule would understate corroboration, which is the
    one number this graph exists to report."""
    confirmed = tmp_path / "doc_confirmed.jsonl"
    confirmed.write_text(json.dumps({
        "rule_id": "R_001", "source": "S 1", "entity": "Line",
        "condition": "loading_pct > 100", "action": "BLOCK", "severity": "high",
        "explanation": "x", "role": "CONSTRAINT",
    }) + "\n", encoding="utf-8")
    deduped = tmp_path / "all_rules_deduped.jsonl"
    deduped.write_text(json.dumps({
        "rule_id": "R_002", "source": "S 2", "entity": "Facility",
        "condition": "loading_pct > 100", "action": "BLOCK", "severity": "high",
        "explanation": "x", "role": "CONSTRAINT",
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no deduped representative"):
        build_knowledge_graph(tmp_path, deduped, None)


# ── THE V1 REGRESSION ─────────────────────────────────────────────────────────

def test_graph_is_topology_agnostic(kg):
    """The failure that made the first knowledge graph useless.

    It hung rules off Bus_17 / Line_42 nodes from one grid's metadata, which made
    it both uninformative (Line+Bus were 3.5% of the corpus) and unusable on the
    two grids the shield is actually evaluated on. No node id and no structural
    attribute may reference a bus index, a line index, or a grid tag.

    One deliberate exemption: the verbatim `rule` payload carries the polarity
    guard's `fire_rate_<tag>` fields, which do name all three grids. Those are
    *measurements about a rule*, not structure — the same rule, and the same
    graph, still serves every grid. They cannot be stripped either, because the
    payload has to stay byte-identical to `all_rules_deduped.jsonl` for the
    set-equality guarantee above. Indexed entities are still banned inside it.
    """
    banned = ("bus_", "line_", "generator_", "neurips2020", "case14", "wcci2022")
    indexed_entity = re.compile(r"\b(bus|line|generator)_\d", re.IGNORECASE)

    for node, attrs in kg.nodes(data=True):
        structural = {k: v for k, v in attrs.items() if k != "rule"}
        haystack = (str(node) + " " + json.dumps(structural, default=str)).lower()
        for token in banned:
            assert token not in haystack, f"{node} references topology: {token!r}"

        payload = json.dumps(attrs.get("rule", {}), default=str)
        assert not indexed_entity.search(payload), (
            f"{node} binds a rule to a specific grid element: {payload[:120]}"
        )


def test_no_node_type_dominates_the_edge_count(kg):
    """v1 had 6,375 of 6,632 edges (96%) on one type, all of it noise. A healthy
    provenance graph has edges spread across its layers."""
    from collections import Counter

    by_type = Counter(d.get("type") for _, _, d in kg.edges(data=True))
    assert by_type, "graph has no typed edges"
    assert max(by_type.values()) / kg.number_of_edges() < 0.60


# ── PERSISTENCE ───────────────────────────────────────────────────────────────

def test_save_load_round_trip_preserves_the_graph(tmp_path, kg):
    path = save_kg(kg, tmp_path / "kg.json")
    reloaded = load_kg(path)

    assert set(reloaded.nodes) == set(kg.nodes)
    assert set(reloaded.edges) == set(kg.edges)
    for node in kg.nodes:
        assert reloaded.nodes[node] == kg.nodes[node]


def test_persisted_graph_is_readable_json_not_a_pickle(tmp_path, kg):
    """v1 persisted a .pkl, it vanished from disk, and nothing noticed. JSON is
    diffable, survives a networkx upgrade, and fails loudly when truncated."""
    path = save_kg(kg, tmp_path / "kg.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["directed"] is True
    assert len(data["nodes"]) == kg.number_of_nodes()


def test_provider_loads_straight_from_the_saved_file(tmp_path, kg):
    path = save_kg(kg, tmp_path / "kg.json")
    provider = KgRuleProvider.from_json(path)
    assert len(provider) == len(KgRuleProvider(kg))


def test_provider_refuses_an_empty_graph():
    with pytest.raises(ValueError, match="no ServedRule"):
        KgRuleProvider(nx.DiGraph())


# ── POLARITY BACKSTOP ─────────────────────────────────────────────────────────

def test_polarity_backstop_survives_the_swap(kg):
    """`JsonlRuleProvider.assert_clean` is a load-time guard against a rule that
    fires on a healthy grid (tests/test_shield.py:247). Swapping in the graph must
    not quietly drop it."""
    healthy = {
        "fault_type": "secure",
        "voltage_pu_min": 1.0, "voltage_pu_max": 1.0,
        "loading_pct": 40.0, "rho_max": 0.4,
        "n_tripped_lines": 0, "any_line_tripped": False,
    }
    KgRuleProvider(kg).assert_clean([healthy])   # the real corpus is clean

    contaminated = nx.DiGraph()
    contaminated.add_node(
        "ServedRule:R_BAD", type=SERVED_RULE, rule_id="R_BAD",
        rule={"rule_id": "R_BAD", "condition": "loading_pct < 500",
              "role": "CONSTRAINT", "severity": "high", "entity": "Line",
              "source": "fabricated", "action": "BLOCK", "explanation": "x"},
    )
    with pytest.raises(ValueError, match="polarity-contaminated"):
        KgRuleProvider(contaminated).assert_clean([healthy])
