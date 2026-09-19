"""
Tests for evaluation/audit_rejections.py.

The thing these exist to stop is the silent join failure. `rule_id` is NOT
globally unique — five documents in `rules_35b/` restarted numbering, so
`R_001`..`R_172` are reused across them (2,463 records, 2,291 distinct ids;
`thesis_findings.md` §19.2). A loader keyed on `rule_id` alone would look like
it worked, drop or cross-wire records, and produce an adjudication artifact
that reads as evidence while being wrong. So the composite key is pinned, and
so is the deterministic vocabulary check that turns "the validator fabricated
its reasoning" into something decidable.

Run: pytest tests/test_audit_rejections.py -v
"""
import json
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluation.audit_rejections import (  # noqa: E402
    ADJUDICATIONS,
    BASES,
    CATEGORIES,
    DEFAULT_GUARDED,
    DEFAULT_STRICT,
    DEFAULT_TRANSLATED,
    DOCUMENTED_TOTAL_REJECTIONS,
    build_records,
    claims_vocabulary_breach,
    condition_variables,
    correction_language,
    document_stem,
    load_guarded,
    load_rejections,
    reconcile,
    vocabulary_check,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _rejection(rule_id, condition="voltage_pu_min < 0.95", reason="because"):
    return {
        "rule": {
            "rule_id": rule_id,
            "source": "Some Standard, Section 1",
            "entity": "Bus",
            "condition": condition,
            "action": "ALERT",
            "severity": "high",
            "explanation": "e",
            "role": "CONSTRAINT",
            "affirms": None,
            "fire_rate_neurips2020": 0.0,
            "fire_rate_case14": 0.0,
            "fire_rate_wcci2022": 0.0,
        },
        "verdict": {
            "rule_id": rule_id,
            "verdict": "REJECT",
            "corrected_fields": None,
            "reason": reason,
        },
    }


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------
# THE KEYING RULE — the whole reason this test file exists
# --------------------------------------------------------------------------

def test_same_rule_id_in_two_documents_stays_two_records(tmp_path):
    """Two documents both numbering a rule R_042 must yield TWO records.

    This is the real collision class: five source documents restarted
    numbering. Keyed on rule_id alone one of these silently overwrites the
    other and the artifact undercounts without any error.
    """
    _write(tmp_path / "doc_alpha_rejected.jsonl",
           [_rejection("R_042", condition="loading_pct > 100", reason="alpha reason")])
    _write(tmp_path / "doc_beta_rejected.jsonl",
           [_rejection("R_042", condition="voltage_pu_max > 1.1", reason="beta reason")])

    loaded = load_rejections(tmp_path)

    assert len(loaded) == 2, "rule_id collision across documents collapsed into one record"
    assert ("doc_alpha", "R_042") in loaded
    assert ("doc_beta", "R_042") in loaded
    assert loaded[("doc_alpha", "R_042")]["verdict"]["reason"] == "alpha reason"
    assert loaded[("doc_beta", "R_042")]["verdict"]["reason"] == "beta reason"


def test_every_key_is_a_document_rule_id_pair(tmp_path):
    """Keys are 2-tuples, not bare ids — a caller cannot accidentally key on one."""
    _write(tmp_path / "doc_alpha_rejected.jsonl", [_rejection("R_1")])
    for key in load_rejections(tmp_path):
        assert isinstance(key, tuple) and len(key) == 2


def test_a_genuine_duplicate_within_one_document_raises(tmp_path):
    """The composite key is supposed to be unique; a real collision is a bug, not a merge."""
    _write(tmp_path / "doc_alpha_rejected.jsonl", [_rejection("R_9"), _rejection("R_9")])
    with pytest.raises(ValueError, match="duplicate"):
        load_rejections(tmp_path)


def test_document_stem_survives_spaces_and_punctuation():
    """The NERC filename carries spaces, commas, parentheses and a hyphen."""
    name = ("NERC Reliability Standards (Complete Set - PRC, TOP, TPL, VAR, BAL, "
            "CIP, EOP, FAC, MOD)_rejected.jsonl")
    assert document_stem(name, "_rejected.jsonl") == (
        "NERC Reliability Standards (Complete Set - PRC, TOP, TPL, VAR, BAL, CIP, EOP, FAC, MOD)"
    )


def test_guarded_and_rejected_stems_agree_so_the_join_lands(tmp_path):
    """The two stages write different suffixes; the stem must match across them."""
    _write(tmp_path / "some doc (v2)_rejected.jsonl", [_rejection("R_7")])
    g = tmp_path / "guarded"
    g.mkdir()
    _write(g / "some doc (v2)_translated.jsonl",
           [{"rule": _rejection("R_7")["rule"], "chunk": "the source text"}])

    rej = load_rejections(tmp_path)
    guarded = load_guarded(g)
    assert set(rej) == set(guarded), "stems diverged; the chunk join would silently miss"

    recs = build_records(rej, {}, guarded)
    assert recs[0].mechanical["source_chunk"] == "the source text"


def test_empty_lines_are_skipped(tmp_path):
    path = tmp_path / "doc_rejected.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_rejection("R_1")) + "\n\n")
        fh.write(json.dumps(_rejection("R_2")) + "\n")
    assert len(load_rejections(tmp_path)) == 2


# --------------------------------------------------------------------------
# The deterministic half — the vocabulary check
# --------------------------------------------------------------------------

def test_condition_variables_ignores_operators_and_literals():
    assert condition_variables("voltage_pu_min < 0.9 or voltage_pu_max > 1.1") == [
        "voltage_pu_min", "voltage_pu_max"
    ]
    assert condition_variables("loading_pct > 100.0") == ["loading_pct"]


def test_real_vocabulary_variables_pass():
    for cond in ("voltage_pu_min < 0.95",
                 "loading_pct > 100",
                 "generation_load_imbalance_pct > 25.0",
                 "voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1"):
        assert vocabulary_check(cond)["all_in_vocabulary"], cond


def test_an_invented_variable_is_caught():
    v = vocabulary_check("frequency_hz < 49.5")
    assert not v["all_in_vocabulary"]
    assert v["unknown_variables"] == ["frequency_hz"]


def test_fabrication_test_is_decidable(tmp_path):
    """A reason asserting a vocabulary breach against a valid condition is fabricated.

    This is the mechanical form of §12.3's finding. It must fire on the claim
    and stay silent when the claim is absent, otherwise the artifact's headline
    is an artefact of the regex.
    """
    _write(tmp_path / "doc_rejected.jsonl", [
        _rejection("R_1", condition="voltage_pu_min < 0.95",
                   reason="the condition uses an invalid variable and misrepresents it"),
        _rejection("R_2", condition="voltage_pu_min < 0.95",
                   reason="the source text does not state this threshold"),
        _rejection("R_3", condition="frequency_hz < 49.5",
                   reason="uses an undefined variable frequency_hz"),
    ])
    by_id = {r.rule_id: r for r in build_records(load_rejections(tmp_path), {}, {})}

    assert by_id["R_1"].mechanical["vocabulary_claim_is_fabricated"] is True
    assert by_id["R_2"].mechanical["vocabulary_claim_is_fabricated"] is False
    # Claim made AND true — not fabrication.
    assert by_id["R_3"].mechanical["reason_claims_vocabulary_breach"] is not None
    assert by_id["R_3"].mechanical["vocabulary_claim_is_fabricated"] is False


def test_vocabulary_claim_phrasings_are_all_caught():
    for phrase in ("uses an undefined variable loading_pct",
                   "'100' which is not a valid variable",
                   "not defined in the allowed variables",
                   "a variable not listed in the allowed set",
                   "the condition uses an invalid variable"):
        assert claims_vocabulary_breach(phrase), phrase
    assert claims_vocabulary_breach("the source text says something else") is None


def test_correction_language_is_detected():
    assert correction_language("which is correct; however, the role is wrong")
    assert correction_language("making the role mismatch and thus requiring correction")
    assert correction_language("the source does not state this") is None


# --------------------------------------------------------------------------
# Reconciliation must report, not absorb
# --------------------------------------------------------------------------

def test_reconcile_flags_a_count_mismatch(tmp_path):
    _write(tmp_path / "doc_rejected.jsonl", [_rejection("R_1")])
    rec = reconcile(build_records(load_rejections(tmp_path), {}, {}))
    assert rec["reconciles"] is False
    assert any("rejection count is 1" in d for d in rec["discrepancies"])


def test_unadjudicated_rules_fall_to_unverifiable_not_sound(tmp_path):
    """A missing adjudication must never read as an endorsement."""
    _write(tmp_path / "unknown_doc_rejected.jsonl", [_rejection("R_999")])
    recs = build_records(load_rejections(tmp_path), {}, {})
    assert recs[0].adjudication["category"] == "UNVERIFIABLE"


def test_every_adjudication_declares_its_method_and_a_valid_basis():
    """The mechanical/adjudicated split is the artifact's whole claim to honesty."""
    for key, adj in ADJUDICATIONS.items():
        assert adj.category in CATEGORIES, key
        assert adj.justification.strip(), key
        assert adj.basis, f"{key} has no basis; a call without one cannot be argued with"
        for b in adj.basis:
            assert b in BASES, (key, b)


# --------------------------------------------------------------------------
# Against the real corpus on disk
# --------------------------------------------------------------------------

@pytest.mark.skipif(not DEFAULT_TRANSLATED.exists(), reason="validated_translated/ absent")
def test_the_real_corpus_has_exactly_21_shared_rejections():
    assert len(load_rejections(DEFAULT_TRANSLATED)) == DOCUMENTED_TOTAL_REJECTIONS


@pytest.mark.skipif(not DEFAULT_TRANSLATED.exists(), reason="validated_translated/ absent")
def test_every_real_rejection_is_adjudicated_and_joins_to_its_chunk():
    translated = load_rejections(DEFAULT_TRANSLATED)
    strict = load_rejections(DEFAULT_STRICT)
    guarded = load_guarded(DEFAULT_GUARDED)
    records = build_records(translated, strict, guarded)

    unadjudicated = [r.rule_id for r in records
                     if r.adjudication["category"] == "UNVERIFIABLE"]
    assert not unadjudicated, f"no adjudication for {unadjudicated}"

    missing_chunk = [r.rule_id for r in records if not r.mechanical["source_chunk"]]
    assert not missing_chunk, f"(document, rule_id) join to the guarded corpus missed {missing_chunk}"


@pytest.mark.skipif(not DEFAULT_TRANSLATED.exists(), reason="validated_translated/ absent")
def test_the_translated_arm_is_a_subset_of_the_strict_arm():
    """§12.2's nesting: every translated rejection was also rejected by strict."""
    translated = set(load_rejections(DEFAULT_TRANSLATED))
    strict = set(load_rejections(DEFAULT_STRICT))
    assert translated <= strict
    assert len(strict) == 31


@pytest.mark.skipif(not DEFAULT_TRANSLATED.exists(), reason="validated_translated/ absent")
def test_the_recorded_discrepancy_against_findings_12_4_is_still_there():
    """Pins the finding: this adjudication does NOT reproduce §12.4's accounting.

    If someone revises §12.4 to match, update the constants at the top of
    audit_rejections.py in the same change — do not delete this test to make
    it green.
    """
    records = build_records(load_rejections(DEFAULT_TRANSLATED),
                            load_rejections(DEFAULT_STRICT),
                            load_guarded(DEFAULT_GUARDED))
    rec = reconcile(records)
    assert rec["reconciles"] is False
    assert rec["counts"]["WRONG"] == 6
    assert rec["counts"]["SHOULD_HAVE_BEEN_CORRECT"] == 4
    assert rec["counts"]["SOUND"] == 11
    assert rec["counts"]["UNVERIFIABLE"] == 0
    # Fabrication reaches the served arm, not just the strict counterfactual.
    assert set(rec["fabricated_vocabulary_claims"]) == {"R_1926", "R_069"}
    # Four rejections §12.4's table never examined.
    assert set(rec["not_named_in_findings_12_4"]) == {"R_260", "R_1444", "R_1575", "R_1152"}
