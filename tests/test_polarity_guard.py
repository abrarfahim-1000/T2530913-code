"""
Tests for the healthy-frame polarity guard (extraction/polarity_guard.py).

The load-bearing test here is `test_rule_clean_at_nominal_but_firing_on_real_frames`:
it encodes *why* the guard measures a fire rate over real frames instead of
checking a hand-written healthy band. Everything else is plumbing around that.

Run: pytest tests/test_polarity_guard.py -v
"""
import json
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from extraction.polarity_guard import (
    RuleAssessment,
    is_affirmation,
    probe_label,
    assert_no_rule_fires,
    assess_rules,
    fire_rate,
    first_firing_context,
    partition,
    process_file,
)

# Stand-ins for simulator-labelled `normal` frames. Voltages sit a hair either
# side of 1.0 pu and loading is high but sub-limit — which is what the real
# case14 normal frames actually look like (rho_max median 0.72, max 0.99).
HEALTHY_FRAMES = [
    {"voltage_pu_min": 0.995, "voltage_pu_max": 1.004, "loading_pct": 72.0,
     "rho_max": 0.72, "n_tripped_lines": 0, "any_line_tripped": False},
    {"voltage_pu_min": 0.988, "voltage_pu_max": 1.001, "loading_pct": 61.0,
     "rho_max": 0.61, "n_tripped_lines": 0, "any_line_tripped": False},
    {"voltage_pu_min": 0.999, "voltage_pu_max": 1.008, "loading_pct": 88.0,
     "rho_max": 0.88, "n_tripped_lines": 0, "any_line_tripped": False},
    {"voltage_pu_min": 0.991, "voltage_pu_max": 1.000, "loading_pct": 65.0,
     "rho_max": 0.65, "n_tripped_lines": 0, "any_line_tripped": False},
]


def _rule(rule_id, condition, severity="high", role="CONSTRAINT", affirms=None):
    return {"rule_id": rule_id, "source": "test", "entity": "Grid",
            "condition": condition, "action": "BLOCK", "severity": severity,
            "explanation": "test rule", "role": role, "affirms": affirms}


# ── THE CRITERION ─────────────────────────────────────────────────────────────

def test_correct_polarity_rule_never_fires():
    assert fire_rate("loading_pct > 100", HEALTHY_FRAMES) == 0.0
    assert fire_rate("voltage_pu_min < 0.90 or voltage_pu_max > 1.10", HEALTHY_FRAMES) == 0.0


def test_inverted_ride_through_band_fires_on_everything():
    """"remain connected between 0.7 and 0.9 pu" inverted into a violation
    predicate fires at nominal — the exact defect this guard exists to catch."""
    assert fire_rate("voltage_pu_min < 0.7 or voltage_pu_max > 0.9", HEALTHY_FRAMES) == 1.0


def test_rule_clean_at_nominal_but_firing_on_real_frames():
    """THE regression test for the guard's design.

    `voltage_pu_min < 0 or voltage_pu_max > 1` is a real extracted rule, severity
    critical. At a synthetic 1.0 pu point it evaluates FALSE and a band-based
    filter ships it. Against real healthy frames — where voltage_pu_max sits just
    above 1.0 on ~72% of case14 normals — it fires. Only the fire rate catches it.
    """
    condition = "voltage_pu_min < 0 or voltage_pu_max > 1"

    synthetic_nominal = {"voltage_pu_min": 1.0, "voltage_pu_max": 1.0, "loading_pct": 40.0,
                         "rho_max": 0.4, "n_tripped_lines": 0, "any_line_tripped": False}
    assert fire_rate(condition, [synthetic_nominal]) == 0.0, "band-based check would pass it"

    assert fire_rate(condition, HEALTHY_FRAMES) == 0.75, "fire rate exposes it"


def test_legitimate_near_limit_rule_survives_a_partial_fire_rate():
    """`loading_pct > 80` fires on some healthy frames because the grid runs hot.
    It must be KEPT and its rate recorded, not culled — that rate is the rule's
    false block rate, which is a finding rather than a defect."""
    rate = fire_rate("loading_pct > 80", HEALTHY_FRAMES)
    assert 0.0 < rate < 0.5
    kept, rejected = partition(assess_rules([_rule("R_1", "loading_pct > 80")],
                                            {"case14": {"normal": HEALTHY_FRAMES}}))
    assert len(kept) == 1 and not rejected
    assert kept[0].rates["case14"] == rate


def test_unevaluable_condition_does_not_count_as_a_fire():
    """A rule referencing a variable outside the vocabulary never blocks, so it
    cannot cause a false block either."""
    assert fire_rate("frequency_hz < 49.5", HEALTHY_FRAMES) == 0.0


def test_fire_rate_needs_contexts():
    with pytest.raises(ValueError, match="No healthy contexts"):
        fire_rate("loading_pct > 100", [])


def test_first_firing_context_is_evidence_for_the_audit_trail():
    ctx = first_firing_context("loading_pct > 80", HEALTHY_FRAMES)
    assert ctx is not None and ctx["loading_pct"] == 88.0
    assert first_firing_context("loading_pct > 200", HEALTHY_FRAMES) is None


# ── PARTITIONING ──────────────────────────────────────────────────────────────

def test_partition_rejects_on_the_worst_topology_not_the_average():
    """A rule silent on one grid and constant on another is still unusable;
    averaging the two would hide it."""
    a = RuleAssessment(rule=_rule("R_X", "voltage_pu_max > 1.0"),
                       rates={"neurips2020": 0.0, "case14": 1.0})
    kept, rejected = partition([a], max_fire_rate=0.5)
    assert rejected and not kept
    assert a.worst_tag == "case14"
    assert a.spread() == 1.0


def test_cutoff_is_configurable():
    a = RuleAssessment(rule=_rule("R_Y", "x"), rates={"case14": 0.3})
    assert partition([a], max_fire_rate=0.5)[0]      # kept
    assert partition([a], max_fire_rate=0.2)[1]      # rejected


def test_assess_rules_measures_every_topology():
    assessments = assess_rules(
        [_rule("R_1", "loading_pct > 80")],
        {"case14": {"normal": HEALTHY_FRAMES}, "neurips2020": {"normal": HEALTHY_FRAMES[:2]}},
    )
    assert set(assessments[0].rates) == {"case14", "neurips2020"}


# ── LOAD-TIME BACKSTOP ────────────────────────────────────────────────────────

def test_assert_no_rule_fires_passes_a_clean_ruleset():
    assert_no_rule_fires([_rule("R_1", "loading_pct > 100")], HEALTHY_FRAMES)


def test_assert_no_rule_fires_rejects_a_contaminated_one():
    with pytest.raises(ValueError, match="polarity-contaminated"):
        assert_no_rule_fires(
            [_rule("R_BAD", "voltage_pu_min < 0.7 or voltage_pu_max > 0.9")],
            HEALTHY_FRAMES,
        )


# ── FILE PARTITIONING ─────────────────────────────────────────────────────────

def _write_translated(path, rules):
    with path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps({"rule": r, "chunk": f"source text for {r['rule_id']}"}) + "\n")


def test_process_file_splits_and_annotates(tmp_path):
    src = tmp_path / "doc_translated.jsonl"
    out = tmp_path / "guarded"
    out.mkdir()
    _write_translated(src, [
        _rule("R_GOOD", "loading_pct > 100"),
        _rule("R_BAD", "voltage_pu_min < 0.7 or voltage_pu_max > 0.9", severity="critical"),
    ])

    stats = process_file(src, out, {"case14": {"normal": HEALTHY_FRAMES}}, 0.5, report_only=False)
    assert stats["n_kept"] == 1 and stats["n_rejected"] == 1

    # The kept file reuses the *_translated.jsonl name so validate.py's glob finds it.
    kept = [json.loads(l) for l in (out / "doc_translated.jsonl").read_text().splitlines()]
    assert [k["rule"]["rule_id"] for k in kept] == ["R_GOOD"]
    assert kept[0]["rule"]["fire_rate_case14"] == 0.0
    assert kept[0]["chunk"] == "source text for R_GOOD", "source chunk must survive for validate.py"

    rejects = [json.loads(l) for l in (out / "doc_polarity_rejected.jsonl").read_text().splitlines()]
    assert rejects[0]["rule"]["rule_id"] == "R_BAD"
    assert rejects[0]["fire_rates"]["case14"] == 1.0
    assert "POLARITY_REJECTED" in rejects[0]["reason"]
    assert "voltage_pu_max" in rejects[0]["reason"], "firing context recorded as evidence"


def test_rejected_file_is_not_named_confirmed(tmp_path):
    """deduplicate_rules() globs *_confirmed.jsonl — a reject file named that way
    would be merged straight back into all_rules_deduped.jsonl."""
    src = tmp_path / "doc_translated.jsonl"
    out = tmp_path / "guarded"
    out.mkdir()
    _write_translated(src, [_rule("R_BAD", "voltage_pu_min < 1.5")])
    process_file(src, out, {"case14": {"normal": HEALTHY_FRAMES}}, 0.5, report_only=False)
    assert not list(out.glob("*_confirmed.jsonl"))


def test_report_only_writes_nothing(tmp_path):
    src = tmp_path / "doc_translated.jsonl"
    out = tmp_path / "guarded"
    out.mkdir()
    _write_translated(src, [_rule("R_BAD", "voltage_pu_min < 1.5")])
    process_file(src, out, {"case14": {"normal": HEALTHY_FRAMES}}, 0.5, report_only=True)
    assert not list(out.iterdir())


# ── AFFIRMATIONS (role-aware scoring) ─────────────────────────────────────────

OVERLOAD_FRAMES = [
    {"voltage_pu_min": 0.96, "voltage_pu_max": 1.01, "loading_pct": 118.0,
     "rho_max": 1.18, "n_tripped_lines": 0, "any_line_tripped": False},
    {"voltage_pu_min": 0.94, "voltage_pu_max": 1.02, "loading_pct": 104.0,
     "rho_max": 1.04, "n_tripped_lines": 0, "any_line_tripped": False},
]


def test_probe_label_routes_by_role():
    assert probe_label(_rule("R_C", "loading_pct > 100")) == "normal"
    assert probe_label(_rule("R_A", "rho_max >= 1.0", role="AFFIRMATION",
                             affirms="overload")) == "overload"
    assert is_affirmation(_rule("R_A", "x", role="AFFIRMATION", affirms="normal"))


def test_healthy_band_affirmation_is_kept_not_rejected():
    """The regression that Option A exists to fix.

    'voltage shall remain within 0.95-1.05' fires on EVERY healthy frame. As a
    CONSTRAINT that is contamination; as an AFFIRMATION of `normal` it is exactly
    correct, and the guard must keep it."""
    cond = "voltage_pu_min >= 0.95 and voltage_pu_max <= 1.05"
    assert fire_rate(cond, HEALTHY_FRAMES) == 1.0          # would be rejected as a constraint

    affirmation = _rule("R_AFF", cond, role="AFFIRMATION", affirms="normal")
    kept, rejected = partition(assess_rules([affirmation],
                                            {"case14": {"normal": HEALTHY_FRAMES}}))
    assert len(kept) == 1 and not rejected

    constraint = _rule("R_CON", cond)
    kept2, rejected2 = partition(assess_rules([constraint],
                                              {"case14": {"normal": HEALTHY_FRAMES}}))
    assert len(rejected2) == 1 and not kept2


def test_affirmation_that_never_holds_on_its_class_is_rejected():
    """An affirmation of `normal` that is false on healthy frames affirms nothing."""
    rule = _rule("R_AFF", "voltage_pu_min > 1.5", role="AFFIRMATION", affirms="normal")
    kept, rejected = partition(assess_rules([rule], {"case14": {"normal": HEALTHY_FRAMES}}))
    assert len(rejected) == 1 and not kept


def test_affirmation_is_scored_against_its_own_class_frames():
    """`rho_max >= 1.0` affirms `overload`: false on healthy frames, true on
    overloaded ones. Scoring it against `normal` would wrongly reject it."""
    rule = _rule("R_OVL", "rho_max >= 1.0", role="AFFIRMATION", affirms="overload")
    contexts = {"case14": {"normal": HEALTHY_FRAMES, "overload": OVERLOAD_FRAMES}}
    assessments = assess_rules([rule], contexts)
    assert assessments[0].rates["case14"] == 1.0
    assert partition(assessments)[0], "must be kept"


def test_affirmation_worst_case_is_the_lowest_rate_not_the_highest():
    a = RuleAssessment(rule=_rule("R_A", "x", role="AFFIRMATION", affirms="normal"),
                       rates={"neurips2020": 0.98, "case14": 0.10})
    assert a.worst_rate == 0.10 and a.worst_tag == "case14"
    assert partition([a], min_support=0.5)[1], "rejected on the topology where it fails"


def test_backstop_ignores_affirmations():
    """assert_no_rule_fires guards constraints only — an affirmation of `normal`
    firing on every healthy frame is correct, not contamination."""
    assert_no_rule_fires(
        [_rule("R_A", "voltage_pu_min >= 0.95 and voltage_pu_max <= 1.05",
               role="AFFIRMATION", affirms="normal")],
        HEALTHY_FRAMES,
    )


def test_rule_with_no_measurable_class_is_kept():
    """No frames of the probe class -> no evidence -> keep. Absence of evidence is
    not evidence of a defect."""
    rule = _rule("R_CAS", "n_tripped_lines >= 3", role="AFFIRMATION", affirms="cascade")
    kept, rejected = partition(assess_rules([rule], {"case14": {"normal": HEALTHY_FRAMES}}))
    assert len(kept) == 1 and not rejected
