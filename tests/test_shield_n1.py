"""
Tests for the asymmetric N-1 gate (shield.validate_n1).

The whole point of the asymmetry: an over-permissive prediction ("losing this
line is fine") on a grid that is already outside its limits is the failure that
endangers a network. An over-cautious prediction is merely wasteful. A safety
gate that suppressed the cautious one would be worse than no gate at all.

Run: pytest tests/test_shield_n1.py -v
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shield.shield import SECURE, VIOLATION, validate_n1

OVERLOAD_CONSTRAINT = {
    "rule_id": "R_001",
    "role": "CONSTRAINT",
    "condition": "rho_max >= 1.0",
    "severity": "critical",
    "source": "IEEE Std 1234, §5.1",
    "explanation": "Conductors shall not be operated beyond their thermal rating.",
}
VOLTAGE_CONSTRAINT = {
    "rule_id": "R_002",
    "role": "CONSTRAINT",
    "condition": "voltage_pu_max > 1.05",
    "severity": "high",
    "source": "IEEE Std 1547-2018, §7.4",
    "explanation": "Voltage shall remain within +/-5% of nominal.",
}
SECURE_AFFIRMATION = {
    "rule_id": "R_003",
    "role": "AFFIRMATION",
    "affirms": "secure",
    "condition": "rho_max < 0.8",
    "severity": "low",
    "source": "NERC TPL-001",
    "explanation": "Loading well inside rating is consistent with a secure base case.",
}

CALM = {"rho_max": 0.42, "voltage_pu_max": 1.01, "loading_pct": 42.0}
STRESSED = {"rho_max": 1.14, "voltage_pu_max": 1.01, "loading_pct": 114.0}


def test_secure_verdict_blocks_when_the_base_case_is_already_violating():
    """N-1 security presupposes a secure base case — you cannot be safe against
    losing another element while already outside your limits."""
    r = validate_n1(STRESSED, [OVERLOAD_CONSTRAINT], predicted=SECURE)
    assert r.blocked
    assert r.highest_severity == "critical"
    assert [x["rule_id"] for x in r.violated_rules] == ["R_001"]


def test_secure_verdict_passes_on_a_calm_base_case():
    r = validate_n1(CALM, [OVERLOAD_CONSTRAINT, VOLTAGE_CONSTRAINT], predicted=SECURE)
    assert not r.blocked
    assert r.violated_rules == []


def test_violation_verdict_never_blocks_even_on_a_calm_grid():
    """The conservative direction. A false alarm is wasteful, not dangerous, and
    a safety gate must not suppress it."""
    r = validate_n1(CALM, [OVERLOAD_CONSTRAINT, VOLTAGE_CONSTRAINT], predicted=VIOLATION)
    assert not r.blocked
    assert r.status == "PASS"


def test_violation_verdict_never_blocks_on_a_stressed_grid_either():
    r = validate_n1(STRESSED, [OVERLOAD_CONSTRAINT], predicted=VIOLATION)
    assert not r.blocked
    # the fired constraint corroborates the cautious call rather than opposing it
    assert [x["rule_id"] for x in r.supporting_rules] == ["R_001"]


def test_the_gate_is_asymmetric_on_identical_evidence():
    """Same context, same rules — only the predicted verdict differs."""
    blocked = validate_n1(STRESSED, [OVERLOAD_CONSTRAINT], predicted=SECURE)
    passed = validate_n1(STRESSED, [OVERLOAD_CONSTRAINT], predicted=VIOLATION)
    assert blocked.blocked and not passed.blocked


def test_affirmations_alone_never_block_a_secure_verdict():
    """Option A carries over: missing support is recorded, not gated on."""
    r = validate_n1(STRESSED, [SECURE_AFFIRMATION], predicted=SECURE)
    assert not r.blocked
    assert r.n_affirmations_for_class == 1
    assert r.unsupported, "affirmation existed but did not fire — recorded only"


def test_supporting_affirmation_is_reported_for_a_secure_verdict():
    r = validate_n1(CALM, [SECURE_AFFIRMATION], predicted=SECURE)
    assert not r.blocked
    assert [x["rule_id"] for x in r.supporting_rules] == ["R_003"]
    assert not r.unsupported


def test_severity_ordering_survives_the_n1_wrapper():
    ctx = {"rho_max": 1.2, "voltage_pu_max": 1.09, "loading_pct": 120.0}
    r = validate_n1(ctx, [VOLTAGE_CONSTRAINT, OVERLOAD_CONSTRAINT], predicted=SECURE)
    assert [x["rule_id"] for x in r.violated_rules] == ["R_001", "R_002"]
    assert r.highest_severity == "critical"


def test_missing_context_variable_does_not_block():
    """A rule referencing a variable the frame cannot supply is NOT_EVALUABLE.
    Blocking on it would penalise the model for a gap in the corpus."""
    r = validate_n1({"rho_max": 0.3}, [VOLTAGE_CONSTRAINT], predicted=SECURE)
    assert not r.blocked
    assert r.n_not_evaluable == 1
