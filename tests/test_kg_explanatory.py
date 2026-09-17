"""
Tests for the KG's explanatory layer — Component C carrying shield v2's channels.

The defect this layer fixes: the graph knew only the four served rules, so 47 of
the 58 rules in the channel corpus raised KeyError when cited. The shield could
render a warning to a reader and be unable to say which standard it came from,
which is precisely what the WARN channel was built to avoid (thesis_findings 21).

The load-bearing test is the first one. Every shield figure in the thesis was
measured on the served four; the layer is only safe if adding it cannot change
what the gate is handed unless a caller explicitly asks for it.

Run: pytest tests/test_kg_explanatory.py -v
"""
import json
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from kg.build import (  # noqa: E402
    CHANNELS_CORPUS,
    TRANSLATED_DIR,
    add_explanatory_layer,
    build_knowledge_graph,
    kg_stats,
    nodes_of,
)
from kg.provider import KgRuleProvider  # noqa: E402
from kg.schema import (  # noqa: E402
    CLAUSE,
    EXPLANATORY_RULE,
    PREDICATE,
    SERVED_RULE,
)
from shield.channels import BLOCK, VETO_CHANNELS, partition_by_channel  # noqa: E402

CONFIRMED_DIR = "validated_translated"
DEDUPED = "validated_translated/all_rules_deduped.jsonl"


@pytest.fixture(scope="module")
def plain_kg():
    if not os.path.exists(DEDUPED):
        pytest.skip(f"{DEDUPED} not on disk — run extraction/validate.py")
    return build_knowledge_graph(CONFIRMED_DIR, DEDUPED, "rules_35b")


@pytest.fixture(scope="module")
def layered_kg():
    if not os.path.exists(CHANNELS_CORPUS):
        pytest.skip(f"{CHANNELS_CORPUS} not on disk — build the channel corpus")
    kg = build_knowledge_graph(CONFIRMED_DIR, DEDUPED, "rules_35b")
    return add_explanatory_layer(kg, CHANNELS_CORPUS, TRANSLATED_DIR)


@pytest.fixture(scope="module")
def channel_records():
    with open(CHANNELS_CORPUS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── THE GUARANTEE ─────────────────────────────────────────────────────────────

def test_layering_does_not_change_what_the_provider_serves(layered_kg, plain_kg):
    """The whole safety argument. Adding the layer must not widen the corpus the
    gate receives, or every measured number moves without anyone asking."""
    assert (KgRuleProvider(layered_kg).rules_for("secure")
            == KgRuleProvider(plain_kg).rules_for("secure"))
    assert len(KgRuleProvider(layered_kg)) == len(KgRuleProvider(plain_kg))


def test_explanatory_rules_are_served_only_when_asked_for(layered_kg):
    opted_in = KgRuleProvider(layered_kg, include_explanatory=True)
    assert len(opted_in) > len(KgRuleProvider(layered_kg))


def test_asking_for_the_layer_on_a_graph_without_one_raises(plain_kg):
    """Silently serving four rules when the caller asked for the corpus would
    look like a clean run that measured the wrong thing."""
    with pytest.raises(ValueError, match="no ExplanatoryRule"):
        KgRuleProvider(plain_kg, include_explanatory=True)


def test_serving_the_layer_does_not_widen_the_veto_path(layered_kg):
    """Only BLOCK may veto. A WARN rule reaching the veto set would move the
    92-94% intervention precision of thesis_findings 13 silently."""
    rules = KgRuleProvider(layered_kg, include_explanatory=True).rules_for("secure")
    buckets = partition_by_channel(rules)
    assert VETO_CHANNELS == (BLOCK,)
    for rule in buckets[BLOCK]:
        channel = rule.get("channel")
        assert channel in (None, BLOCK), (
            f"{rule.get('rule_id')} reached the veto path on channel {channel!r}")


# ── CITATION ──────────────────────────────────────────────────────────────────

def test_every_channel_rule_is_citable(layered_kg, channel_records):
    """Before the layer, 47 of 58 raised KeyError."""
    provider = KgRuleProvider(layered_kg, include_explanatory=True)
    uncitable = []
    for rec in channel_records:
        try:
            provider.cite(rec["rule_id"])
        except KeyError:
            uncitable.append(rec["rule_id"])
    assert not uncitable, f"{len(uncitable)} rule(s) cannot be cited: {uncitable[:8]}"


def test_a_channel_citation_names_its_clause_and_document(layered_kg,
                                                          channel_records):
    provider = KgRuleProvider(layered_kg, include_explanatory=True)
    warn = next(r for r in channel_records if r.get("channel") == "WARN")
    cit = provider.cite(warn["rule_id"], layer=EXPLANATORY_RULE)
    assert cit.sources, "a citation with no source is not a citation"
    assert all(s.clause and s.document for s in cit.sources)
    assert cit.channel == "WARN"


def test_layer_selects_the_record_that_spoke_not_the_group(layered_kg):
    """R_1492 deduped into served R_1443 AND is its own channel record. Reporting
    what warned must name R_1492; the veto path must keep citing the group."""
    provider = KgRuleProvider(layered_kg, include_explanatory=True)
    assert provider.cite("R_1492").served_rule_id == "R_1443"
    assert provider.cite("R_1492", layer=EXPLANATORY_RULE).served_rule_id == "R_1492"


def test_asking_for_a_layer_that_has_no_such_node_raises(layered_kg):
    """Strict rather than falling back, or a caller asking about the channel
    corpus quietly receives the served group's answer instead."""
    provider = KgRuleProvider(layered_kg, include_explanatory=True)
    with pytest.raises(KeyError):
        provider.cite("R_NOT_A_RULE", layer=EXPLANATORY_RULE)


def test_served_rules_still_cite_their_whole_deduped_group(layered_kg):
    """The corroboration figure is the graph's headline; the layer must not
    fragment it."""
    cit = KgRuleProvider(layered_kg).cite("R_1443")
    assert len(cit.sources) > 1
    assert cit.n_documents > 1


# ── STRUCTURE ─────────────────────────────────────────────────────────────────

def test_the_layer_is_strictly_additive(layered_kg, plain_kg):
    for node in plain_kg.nodes:
        assert node in layered_kg
    for edge in plain_kg.edges:
        assert layered_kg.has_edge(*edge)
    for node in nodes_of(plain_kg, SERVED_RULE):
        assert layered_kg.nodes[node] == plain_kg.nodes[node]


def test_every_explanatory_rule_hangs_off_a_clause(layered_kg):
    for node in nodes_of(layered_kg, EXPLANATORY_RULE):
        clauses = [n for n in layered_kg.predecessors(node)
                   if layered_kg.nodes[n].get("type") == CLAUSE]
        assert len(clauses) == 1, f"{node} has {len(clauses)} clauses"


def test_every_explanatory_rule_instantiates_a_predicate(layered_kg):
    for node in nodes_of(layered_kg, EXPLANATORY_RULE):
        predicates = [n for n in layered_kg.successors(node)
                      if layered_kg.nodes[n].get("type") == PREDICATE]
        assert len(predicates) == 1, f"{node} has {len(predicates)} predicates"


def test_predicate_stats_keep_served_and_explanatory_apart(layered_kg):
    """Pooling them would inflate the corroboration count the graph reports."""
    thermal = next(p for p in kg_stats(layered_kg)["predicates"]
                   if p["canonical"] == "loading_pct > 100")
    assert thermal["n_served_rules"] == 3
    assert thermal["n_explanatory_rules"] > 0


def test_the_layer_stays_topology_agnostic(layered_kg):
    """The channel corpus carries fire_rate_<tag> and a justification naming all
    three grids. Those are measurements about a rule and live in the verbatim
    payload; no structural attribute may name a grid."""
    banned = ("neurips2020", "case14", "wcci2022", "bus_", "line_")
    for node in nodes_of(layered_kg, EXPLANATORY_RULE):
        attrs = layered_kg.nodes[node]
        structural = {k: v for k, v in attrs.items() if k != "rule"}
        haystack = (str(node) + " " + json.dumps(structural, default=str)).lower()
        for token in banned:
            assert token not in haystack, f"{node} references topology: {token!r}"


def test_a_channel_rule_with_no_stage_two_record_raises(plain_kg, tmp_path):
    """Without a stage-2 record the document is unknown, so the rule cannot be
    attributed — which is the one thing this layer exists to do."""
    orphan = tmp_path / "orphan.jsonl"
    orphan.write_text(json.dumps({
        "rule_id": "R_NOT_IN_STAGE_TWO",
        "source": "Invented Clause 1",
        "condition": "loading_pct > 100",
        "role": "CONSTRAINT",
        "channel": "WARN",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no stage-2 record"):
        add_explanatory_layer(plain_kg.copy(), orphan, TRANSLATED_DIR)
