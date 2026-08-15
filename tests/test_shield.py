"""
Tests for the symbolic shield — context construction, verdict taxonomy, and the gate.

Contexts here are hand-written fixtures rather than dataset frames: the suite must
run on the personal PC without loading a 300k-record, 3.4 GB JSONL. This is the
ONLY legitimate use of a synthetic "healthy band" — as a test fixture, never as a
filtering criterion (see extraction/polarity_guard.py for why).

Run: pytest tests/test_shield.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shield.context import build_context
from shield.evaluator import Verdict, evaluate_condition
from shield.shield import JsonlRuleProvider, build_explanation, validate

# Two voltage levels, mirroring the real grids: neurips2020 mixes ~150 kV and
# ~365 kV lines, case14 mixes ~138 kV and ~20 kV. A flat nominal breaks both.
BASE_KV = np.array([150.0, 150.0, 365.0, 20.0])


def _record(rho, v_or, line_status, label="normal"):
    return {"rho": rho, "v_or": v_or, "line_status": line_status, "label": label}


# ── CONTEXT: THE VOLTAGE CONTRACT ─────────────────────────────────────────────

def test_per_line_base_kv_gives_unit_voltage_across_levels():
    """A 365 kV line at nominal must read 1.0 pu, not 2.43 — the flat-150 bug."""
    ctx = build_context(
        _record([0.4, 0.4, 0.4, 0.4], [150.0, 150.0, 365.0, 20.0], [True] * 4),
        BASE_KV,
    )
    assert ctx["voltage_pu_min"] == pytest.approx(1.0)
    assert ctx["voltage_pu_max"] == pytest.approx(1.0)


def test_tripped_lines_are_masked_out_of_voltage():
    """A tripped line reports v_or = 0; unmasked it would read as 0 pu and fire
    every undervoltage rule on any frame with a disconnection."""
    ctx = build_context(
        _record([0.4, 0.0, 0.4, 0.4], [150.0, 0.0, 365.0, 20.0], [True, False, True, True]),
        BASE_KV,
    )
    assert ctx["voltage_pu_min"] == pytest.approx(1.0)
    assert ctx["n_tripped_lines"] == 1
    assert ctx["any_line_tripped"] is True


def test_blackout_omits_voltage_rather_than_defaulting_to_zero():
    """No energized line -> voltage undefined. It must be ABSENT (NOT_EVALUABLE),
    not 0.0, which would read as a catastrophic undervoltage and block."""
    ctx = build_context(
        _record([0.0] * 4, [0.0] * 4, [False] * 4),
        BASE_KV,
    )
    assert "voltage_pu_min" not in ctx
    assert "voltage_pu_max" not in ctx
    assert ctx["n_tripped_lines"] == 4
    assert evaluate_condition("voltage_pu_min < 0.9", ctx) is Verdict.NOT_EVALUABLE


def test_loading_and_rho_track_the_maximum():
    ctx = build_context(
        _record([0.4, 1.35, 0.2, 0.1], [150.0] * 2 + [365.0, 20.0], [True] * 4),
        BASE_KV,
    )
    assert ctx["rho_max"] == pytest.approx(1.35)
    assert ctx["loading_pct"] == pytest.approx(135.0)
    assert ctx["any_line_tripped"] is False


def test_wrong_sidecar_is_rejected_loudly():
    with pytest.raises(ValueError, match="wrong sidecar"):
        build_context(_record([0.4], [150.0], [True]), BASE_KV)


def test_context_accepts_an_observation_like_object():
    """Same key names for a Grid2Op obs as for a JSONL record."""
    class Obs:
        rho = [0.4, 0.4, 0.4, 0.4]
        v_or = [150.0, 150.0, 365.0, 20.0]
        line_status = [True] * 4

    ctx = build_context(Obs(), BASE_KV)
    assert ctx["voltage_pu_max"] == pytest.approx(1.0)


# ── EVALUATOR: VERDICT TAXONOMY ───────────────────────────────────────────────

HEALTHY = {"voltage_pu_min": 0.99, "voltage_pu_max": 1.01,
           "loading_pct": 45.0, "rho_max": 0.45,
           "n_tripped_lines": 0, "any_line_tripped": False}

OVERLOADED = {**HEALTHY, "loading_pct": 118.0, "rho_max": 1.18}

TRIPPED = {**HEALTHY, "n_tripped_lines": 2, "any_line_tripped": True,
           "voltage_pu_min": 0.88}


def test_violated_and_satisfied():
    assert evaluate_condition("loading_pct > 100", OVERLOADED) is Verdict.VIOLATED
    assert evaluate_condition("loading_pct > 100", HEALTHY) is Verdict.SATISFIED


def test_missing_variable_is_not_evaluable_not_error():
    assert evaluate_condition("frequency_hz < 49.5", HEALTHY) is Verdict.NOT_EVALUABLE


def test_malformed_condition_is_error():
    assert evaluate_condition("loading_pct >", HEALTHY) is Verdict.ERROR


def test_non_predicate_condition_is_error():
    """A bare quantity is not a violation predicate."""
    assert evaluate_condition("rho_max", HEALTHY) is Verdict.ERROR


def test_sandbox_strips_builtins():
    """Builtins are unreachable, so an injected call dies as an undefined NAME
    before it can execute — and, like any unevaluable condition, never blocks."""
    for hostile in ("__import__('os').system('echo hi') == 0",
                    "open('x') == 1",
                    "eval('1==1')"):
        verdict = evaluate_condition(hostile, HEALTHY)
        assert verdict is Verdict.NOT_EVALUABLE, hostile
        assert not verdict.blocks


@pytest.mark.parametrize("verdict,blocks", [
    (Verdict.VIOLATED, True),
    (Verdict.SATISFIED, False),
    (Verdict.NOT_EVALUABLE, False),
    (Verdict.ERROR, False),
])
def test_only_violated_blocks(verdict, blocks):
    assert verdict.blocks is blocks


# ── THE GATE ──────────────────────────────────────────────────────────────────

RULES = [
    {"rule_id": "R_001", "source": "IEEE 1547-2018 §7.4", "entity": "Grid",
     "condition": "loading_pct > 100", "action": "BLOCK", "severity": "high",
     "explanation": "Line loading beyond the thermal limit."},
    {"rule_id": "R_002", "source": "NERC TPL-001-5.1", "entity": "Grid",
     "condition": "voltage_pu_min < 0.90 or voltage_pu_max > 1.10", "action": "BLOCK",
     "severity": "critical", "explanation": "Voltage outside the operating envelope."},
    {"rule_id": "R_003", "source": "NERC PRC-023-6", "entity": "Grid",
     "condition": "n_tripped_lines >= 3", "action": "ALERT", "severity": "medium",
     "explanation": "Multiple simultaneous outages indicate a cascade."},
]


def test_clean_context_passes():
    result = validate({**HEALTHY, "fault_type": "normal", "confidence": 0.97}, RULES)
    assert result.status == "PASS"
    assert result.violated_rules == []
    assert result.n_evaluated == 3
    assert result.n_not_evaluable == 0 and result.n_error == 0


def test_single_overload_blocks():
    result = validate({**OVERLOADED, "fault_type": "overload"}, RULES)
    assert result.status == "BLOCK"
    assert [r["rule_id"] for r in result.violated_rules] == ["R_001"]
    assert result.highest_severity == "high"


def test_multi_violation_orders_by_severity():
    """critical must lead, whatever order the rules arrived in."""
    ctx = {**OVERLOADED, "voltage_pu_min": 0.85, "n_tripped_lines": 4,
           "any_line_tripped": True, "fault_type": "cascade"}
    result = validate(ctx, RULES)
    assert result.status == "BLOCK"
    assert [r["severity"] for r in result.violated_rules] == ["critical", "high", "medium"]
    assert result.highest_severity == "critical"


def test_duplicate_rule_ids_are_evaluated_once():
    result = validate(OVERLOADED, RULES + [dict(RULES[0])])
    assert result.n_evaluated == 3
    assert len(result.violated_rules) == 1


def test_unevaluable_rule_never_blocks_and_is_counted():
    rules = RULES + [{"rule_id": "R_099", "condition": "frequency_hz < 49.5",
                      "severity": "critical", "source": "x", "explanation": "y"}]
    result = validate(HEALTHY, rules)
    assert result.status == "PASS"
    assert result.n_not_evaluable == 1


def test_malformed_rule_never_blocks_and_is_counted():
    rules = RULES + [{"rule_id": "R_098", "condition": "loading_pct >>", "severity": "critical",
                      "source": "x", "explanation": "y"}]
    result = validate(HEALTHY, rules)
    assert result.status == "PASS"
    assert result.n_error == 1


def test_explanation_cites_rule_severity_and_source():
    result = validate(OVERLOADED, RULES)
    assert "R_001" in result.explanation
    assert "IEEE 1547-2018" in result.explanation
    assert "HIGH" in result.explanation


def test_explanation_of_nothing_is_not_a_block():
    assert "No applicable rule" in build_explanation([])


def test_result_serializes_for_the_eval_log():
    payload = validate({**OVERLOADED, "fault_type": "overload", "confidence": 0.8}, RULES).to_dict()
    assert payload["status"] == "BLOCK"
    assert payload["violated_rule_ids"] == ["R_001"]
    assert payload["fault_type"] == "overload"


# ── PROVIDER ──────────────────────────────────────────────────────────────────

def test_jsonl_provider_round_trip(tmp_path):
    import json
    path = tmp_path / "all_rules_deduped.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RULES), encoding="utf-8")

    provider = JsonlRuleProvider.from_jsonl(path)
    assert len(provider) == 3
    # Every rule applies to every prediction — discrimination is by condition.
    assert len(provider.rules_for("normal")) == 3
    assert validate(OVERLOADED, provider.rules_for("overload")).status == "BLOCK"


def test_empty_ruleset_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No rules loaded"):
        JsonlRuleProvider.from_jsonl(path)


def test_provider_refuses_a_contaminated_ruleset(tmp_path):
    """The load-time backstop: an inverted rule must not reach inference even if
    the polarity guard was skipped when the corpus was built."""
    import json
    inverted = {"rule_id": "R_BAD", "source": "s", "entity": "Grid",
                "condition": "voltage_pu_min < 1.05 or voltage_pu_max > 1.10",
                "action": "BLOCK", "severity": "critical", "explanation": "inverted"}
    path = tmp_path / "rules.jsonl"
    path.write_text(json.dumps(inverted), encoding="utf-8")

    provider = JsonlRuleProvider.from_jsonl(path)
    with pytest.raises(ValueError, match="polarity-contaminated"):
        provider.assert_clean([HEALTHY, {**HEALTHY, "voltage_pu_min": 0.98}])


# ── AFFIRMATIONS (Option A: evidence, never a block) ──────────────────────────

def _affirm(rule_id, condition, affirms, severity="medium"):
    return {"rule_id": rule_id, "source": "test", "entity": "Grid", "condition": condition,
            "action": "OTHER", "severity": severity, "explanation": "affirmation",
            "role": "AFFIRMATION", "affirms": affirms}


NORMAL_AFFIRMATION = _affirm(
    "A_NORM", "voltage_pu_min >= 0.95 and voltage_pu_max <= 1.05 and not any_line_tripped",
    "normal")
OVERLOAD_AFFIRMATION = _affirm("A_OVL", "rho_max >= 1.0", "overload")


def test_affirmation_supports_the_predicted_class():
    result = validate({**HEALTHY, "fault_type": "normal"}, RULES + [NORMAL_AFFIRMATION])
    assert result.status == "PASS"
    assert [r["rule_id"] for r in result.supporting_rules] == ["A_NORM"]
    assert result.n_affirmations_for_class == 1
    assert result.unsupported is False


def test_affirmation_of_another_class_is_a_contradiction_not_a_block():
    """Predicting `normal` while an `overload` affirmation fires is exactly the
    signal worth recording — but under Option A it does not block."""
    result = validate({**OVERLOADED, "fault_type": "normal"}, [OVERLOAD_AFFIRMATION])
    assert result.status == "PASS"
    assert [r["rule_id"] for r in result.contradicting_rules] == ["A_OVL"]
    assert result.supporting_rules == []


def test_affirmations_never_block_even_when_none_fire():
    """Option A: absence of support is recorded, not enforced. This is the case
    Option B would block on."""
    # A line is tripped and voltage has sagged, so normality cannot be affirmed —
    # yet the GNN predicted `normal`. Exactly the case Option B would gate on.
    result = validate({**TRIPPED, "fault_type": "normal"}, [NORMAL_AFFIRMATION])
    assert result.status == "PASS"
    assert result.supporting_rules == []
    assert result.n_affirmations_for_class == 1
    assert result.unsupported is True, "the signal Option B would gate on"


def test_unsupported_is_false_when_no_affirmations_exist_for_the_class():
    """A class with zero affirmations in the corpus is uncovered, not unsupported
    - otherwise coverage gaps would masquerade as evidence against the prediction."""
    result = validate({**HEALTHY, "fault_type": "cascade"}, RULES + [NORMAL_AFFIRMATION])
    assert result.n_affirmations_for_class == 0
    assert result.unsupported is False


def test_constraint_still_blocks_alongside_affirmations():
    result = validate({**OVERLOADED, "fault_type": "overload"},
                      RULES + [OVERLOAD_AFFIRMATION])
    assert result.status == "BLOCK"
    assert [r["rule_id"] for r in result.violated_rules] == ["R_001"]
    assert [r["rule_id"] for r in result.supporting_rules] == ["A_OVL"]


def test_affirmation_channel_serializes():
    payload = validate({**HEALTHY, "fault_type": "normal"}, [NORMAL_AFFIRMATION]).to_dict()
    assert payload["supporting_rule_ids"] == ["A_NORM"]
    assert payload["n_affirmations_for_class"] == 1
    assert payload["unsupported"] is False


# ── SCHEMA: role / affirms pairing ────────────────────────────────────────────

def test_rule_schema_requires_a_class_for_affirmations():
    from extraction.common import Rule
    with pytest.raises(Exception, match="must name the class"):
        Rule(rule_id="R_1", source="s", entity="Grid", condition="rho_max < 1.0",
             action="OTHER", severity="low", explanation="e", role="AFFIRMATION")


def test_rule_schema_rejects_a_constraint_that_names_a_class():
    from extraction.common import Rule
    with pytest.raises(Exception, match="must not set"):
        Rule(rule_id="R_1", source="s", entity="Grid", condition="rho_max > 1.0",
             action="BLOCK", severity="low", explanation="e",
             role="CONSTRAINT", affirms="normal")


def test_rule_schema_defaults_to_constraint():
    from extraction.common import Rule
    r = Rule(rule_id="R_1", source="s", entity="Grid", condition="rho_max > 1.0",
             action="BLOCK", severity="low", explanation="e")
    assert r.role == "CONSTRAINT" and r.affirms is None


def test_new_vocabulary_variables_are_accepted():
    from extraction.common import lint_condition
    for cond in ("power_factor_at_max_load < 0.95",
                 "apparent_power_mva_max > 250",
                 "current_a_max > 1200",
                 "generation_load_imbalance_pct > 5"):
        assert lint_condition(cond) == cond
