"""
A warning that never falls silent is not a warning.

`voltage_pu_min < 0.95 or voltage_pu_max > 1.05` fires on 100% of neurips2020's
113,205 contingencies and 100% of case14's. It ships as WARN because a rule earns
that channel by discriminating on *any* grid, and it does discriminate on
wcci2022 (coverage 1.12%). So it warned on two grids where it was already
measured to carry no information — the corpus recorded the degeneracy in prose
and then ignored it.

The rule is not the problem. That predicate is stated in 10 clauses across 4
documents, tied with the thermal check as the most corroborated in the corpus.
These grids rest at 1.05–1.08 pu, so a band written about the post-disturbance
state is breached before anything has happened.

The fix silences such a rule ON THAT GRID ONLY. The load-bearing property, which
the first two tests pin, is that this cannot reach the veto path and therefore
cannot move a single reported number.

Run: pytest tests/test_warn_degeneracy.py -v
"""
import json
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shield.channels import (  # noqa: E402
    BLOCK,
    DEGENERATE_COVERAGE,
    NOT_APPLICABLE,
    WARN,
    channel_of,
    coverage_on,
    is_degenerate_on,
    partition_by_channel,
    resolve_channels_for_grid,
)

CORPUS = "shield_corpus/all_rules_channels.jsonl"
TAGS = ("neurips2020", "case14", "wcci2022")


@pytest.fixture(scope="module")
def corpus():
    if not os.path.exists(CORPUS):
        pytest.skip(f"{CORPUS} not on disk")
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── THE GUARANTEE ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tag", TAGS)
def test_resolution_never_touches_the_veto_path(corpus, tag):
    """The whole safety argument: BLOCK is identical before and after, so the
    92-94% intervention precision of thesis_findings 13 cannot move."""
    before = partition_by_channel(corpus)[BLOCK]
    after = partition_by_channel(resolve_channels_for_grid(corpus, tag))[BLOCK]
    assert ([r.get("rule_id") for r in before]
            == [r.get("rule_id") for r in after])


@pytest.mark.parametrize("tag", TAGS)
def test_resolution_only_ever_demotes_warnings(corpus, tag):
    """Nothing is promoted, and no channel other than WARN is disturbed."""
    before = {r["rule_id"]: channel_of(r) for r in corpus}
    for rule in resolve_channels_for_grid(corpus, tag):
        was, now = before[rule["rule_id"]], channel_of(rule)
        if was == now:
            continue
        assert (was, now) == (WARN, NOT_APPLICABLE), (
            f"{rule['rule_id']} moved {was} -> {now}")


def test_the_corpus_is_not_mutated(corpus, tag="neurips2020"):
    """Returns copies. A caller that resolves for one grid then another must not
    find the first grid's demotions still applied."""
    snapshot = json.dumps(corpus, sort_keys=True)
    resolve_channels_for_grid(corpus, tag)
    assert json.dumps(corpus, sort_keys=True) == snapshot


def test_resolving_for_two_grids_in_a_row_is_independent(corpus):
    once = partition_by_channel(resolve_channels_for_grid(corpus, "wcci2022"))
    resolve_channels_for_grid(corpus, "neurips2020")
    twice = partition_by_channel(resolve_channels_for_grid(corpus, "wcci2022"))
    assert ([r["rule_id"] for r in once[WARN]]
            == [r["rule_id"] for r in twice[WARN]])


# ── THE MEASURED EFFECT ───────────────────────────────────────────────────────

def test_every_warn_rule_carries_per_grid_coverage(corpus):
    """Without it nothing can be silenced, and the fix degrades to a no-op that
    looks like it worked."""
    missing = [r["rule_id"] for r in corpus
               if channel_of(r) == WARN and not isinstance(r.get("coverage"), dict)]
    assert not missing, (
        f"{len(missing)} WARN rule(s) carry no coverage: {missing[:8]}. "
        f"Run evaluation/stamp_warn_coverage.py.")


@pytest.mark.parametrize("tag,expected_silenced", [
    ("neurips2020", 9),
    ("case14", 10),
    ("wcci2022", 0),
])
def test_the_documented_number_of_rules_is_silenced(corpus, tag, expected_silenced):
    before = partition_by_channel(corpus)[WARN]
    after = partition_by_channel(resolve_channels_for_grid(corpus, tag))[WARN]
    assert len(before) - len(after) == expected_silenced


def test_wcci_loses_nothing(corpus):
    """The same records that go silent on the other two grids still warn here.
    If this ever fails the fix has become a corpus-wide cull, which is not what
    it is for."""
    after = partition_by_channel(resolve_channels_for_grid(corpus, "wcci2022"))
    assert len(after[WARN]) == len(partition_by_channel(corpus)[WARN])


def test_the_named_predicate_is_silenced_where_it_is_vacuous(corpus):
    """The specific rule this fix exists for.

    Scoped to the WARN records. Ten records state this condition and one of them
    (R_790) sits on NORMAL, where coverage is neither measured nor consulted —
    the fix touches the warning channel and nothing else.
    """
    condition = "voltage_pu_min < 0.95 or voltage_pu_max > 1.05"
    rules = [r for r in corpus
             if r["condition"] == condition and channel_of(r) == WARN]
    assert rules, "the corpus no longer carries the predicate this test is about"
    for rule in rules:
        assert is_degenerate_on(rule, "neurips2020")
        assert is_degenerate_on(rule, "case14")
        assert not is_degenerate_on(rule, "wcci2022")


def test_a_demoted_rule_records_what_it_shipped_as(corpus):
    """Demotion must be visible in the artifact, or it reads as a corpus that
    always said NOT_APPLICABLE and the finding disappears."""
    demoted = [r for r in resolve_channels_for_grid(corpus, "neurips2020")
               if r.get("channel_was") == WARN]
    assert demoted
    for rule in demoted:
        assert rule["channel"] == NOT_APPLICABLE
        assert "neurips2020" in rule["demoted_because"]


# ── THE CUTOFF ────────────────────────────────────────────────────────────────

def test_unmeasured_is_not_degenerate():
    """Silencing a rule nobody measured is the same overreach in the other
    direction."""
    assert not is_degenerate_on({"channel": WARN, "condition": "x > 1"}, "case14")
    assert coverage_on({"channel": WARN}, "case14") is None


def test_a_rule_just_under_the_cutoff_keeps_its_voice():
    rule = {"rule_id": "R_X", "channel": WARN, "condition": "x > 1",
            "coverage": {"case14": DEGENERATE_COVERAGE - 0.01}}
    assert not is_degenerate_on(rule, "case14")
    assert channel_of(resolve_channels_for_grid([rule], "case14")[0]) == WARN


def test_a_rule_at_the_cutoff_is_silenced():
    rule = {"rule_id": "R_X", "channel": WARN, "condition": "x > 1",
            "coverage": {"case14": DEGENERATE_COVERAGE}}
    assert is_degenerate_on(rule, "case14")
    assert channel_of(
        resolve_channels_for_grid([rule], "case14")[0]) == NOT_APPLICABLE


def test_coverage_for_another_grid_does_not_silence_this_one():
    rule = {"rule_id": "R_X", "channel": WARN, "condition": "x > 1",
            "coverage": {"case14": 1.0, "wcci2022": 0.01}}
    assert channel_of(
        resolve_channels_for_grid([rule], "wcci2022")[0]) == WARN


# ── STALENESS ─────────────────────────────────────────────────────────────────
# The failure that made this fix silently do nothing the first time it ran: the
# graph carries each rule dict VERBATIM, so it is a snapshot. Restamping the
# corpus without rebuilding served the old records while every count still
# reconciled, and the run reported success having demoted nothing.

def test_a_stale_explanatory_layer_is_refused(tmp_path):
    import json as _json

    from kg.build import (
        add_explanatory_layer,
        build_knowledge_graph,
        explanatory_corpus_status,
        save_kg,
    )
    from kg.provider import KgRuleProvider

    if not os.path.exists(CORPUS):
        pytest.skip(f"{CORPUS} not on disk")

    with open(CORPUS, encoding="utf-8") as f:
        records = [_json.loads(line) for line in f if line.strip()]

    corpus_copy = tmp_path / "corpus.jsonl"
    corpus_copy.write_text(
        "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")

    kg = add_explanatory_layer(
        build_knowledge_graph("validated_translated",
                              "validated_translated/all_rules_deduped.jsonl",
                              "rules_35b"),
        corpus_copy, "translated_rules")
    assert explanatory_corpus_status(kg)["state"] == "current"
    KgRuleProvider(kg, include_explanatory=True)          # fine while current

    # restamp the corpus, do not rebuild the graph — the exact mistake
    records[0]["coverage"] = {"case14": 1.0}
    corpus_copy.write_text(
        "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")

    assert explanatory_corpus_status(kg)["state"] == "stale"

    # THE FIX: a payload-only drift is repaired, not reported. Before this, the
    # provider served the pre-stamp records and the run demoted nothing.
    provider = KgRuleProvider(kg, include_explanatory=True)
    assert provider.refreshed == len(records)
    served = {r["rule_id"]: r for r in provider.rules_for("secure")}
    assert served[records[0]["rule_id"]].get("coverage") == {"case14": 1.0}, (
        "the provider is still serving the snapshot it was built from")
    assert explanatory_corpus_status(kg)["state"] == "current"

    # ...and a stale layer must not break the served path, which does not use it
    assert len(KgRuleProvider(kg)) == 4

    saved = save_kg(kg, tmp_path / "kg.json")
    from kg.build import load_kg
    assert explanatory_corpus_status(load_kg(saved))["state"] == "current", (
        "the fingerprint must survive save/load, and the repair must be saved "
        "with it or the graph on disk still holds stale payloads")


def test_a_missing_corpus_file_is_not_reported_as_stale(tmp_path):
    """Nothing can be checked once the file is gone — that is not an error, the
    repo may simply have moved."""
    import json as _json

    from kg.build import (
        add_explanatory_layer,
        build_knowledge_graph,
        explanatory_corpus_status,
    )

    if not os.path.exists(CORPUS):
        pytest.skip(f"{CORPUS} not on disk")
    with open(CORPUS, encoding="utf-8") as f:
        records = [_json.loads(line) for line in f if line.strip()]
    corpus_copy = tmp_path / "corpus.jsonl"
    corpus_copy.write_text(
        "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    kg = add_explanatory_layer(
        build_knowledge_graph("validated_translated",
                              "validated_translated/all_rules_deduped.jsonl",
                              "rules_35b"),
        corpus_copy, "translated_rules")
    corpus_copy.unlink()
    assert explanatory_corpus_status(kg)["state"] == "missing"


def test_the_shipped_graph_is_current():
    """Guards the working tree itself: if this fails, rebuild before trusting any
    run that serves the explanatory layer."""
    from kg.build import explanatory_corpus_status, load_kg

    if not os.path.exists("kg/knowledge_graph.json"):
        pytest.skip("no graph on disk")
    status = explanatory_corpus_status(load_kg("kg/knowledge_graph.json"))
    assert status["state"] in ("current", "absent"), (
        f"kg/knowledge_graph.json is {status['state']} — "
        f"run: python scripts/build_kg.py --channels")


def test_a_structural_change_is_refused_rather_than_faked(tmp_path):
    """A refresh may move payloads. It may not move edges — repairing that
    silently would be inventing provenance."""
    import json as _json

    from kg.build import add_explanatory_layer, build_knowledge_graph
    from kg.provider import KgRuleProvider

    if not os.path.exists(CORPUS):
        pytest.skip(f"{CORPUS} not on disk")
    with open(CORPUS, encoding="utf-8") as f:
        records = [_json.loads(line) for line in f if line.strip()]
    corpus_copy = tmp_path / "corpus.jsonl"

    def write(rows):
        corpus_copy.write_text(
            "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")

    write(records)
    kg = add_explanatory_layer(
        build_knowledge_graph("validated_translated",
                              "validated_translated/all_rules_deduped.jsonl",
                              "rules_35b"),
        corpus_copy, "translated_rules")

    changed = [dict(r) for r in records]
    changed[0]["condition"] = "loading_pct > 999"   # moves a Predicate edge
    write(changed)

    with pytest.raises(ValueError, match="rebuild"):
        KgRuleProvider(kg, include_explanatory=True)


def test_dropping_a_rule_from_the_corpus_is_refused(tmp_path):
    import json as _json

    from kg.build import add_explanatory_layer, build_knowledge_graph
    from kg.provider import KgRuleProvider

    if not os.path.exists(CORPUS):
        pytest.skip(f"{CORPUS} not on disk")
    with open(CORPUS, encoding="utf-8") as f:
        records = [_json.loads(line) for line in f if line.strip()]
    corpus_copy = tmp_path / "corpus.jsonl"

    def write(rows):
        corpus_copy.write_text(
            "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")

    write(records)
    kg = add_explanatory_layer(
        build_knowledge_graph("validated_translated",
                              "validated_translated/all_rules_deduped.jsonl",
                              "rules_35b"),
        corpus_copy, "translated_rules")

    write(records[:-1])
    with pytest.raises(ValueError, match="rebuild"):
        KgRuleProvider(kg, include_explanatory=True)
