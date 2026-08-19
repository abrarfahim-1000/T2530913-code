"""
Tests for the post-guard usefulness audit (evaluation/audit_rules.py).

The distinction under test: the polarity guard asks "is this rule WRONG?", the
audit asks "can this rule DO ANYTHING?". A rule can pass the guard cleanly and
still be worthless — a constraint whose threshold no frame ever reaches, or an
affirmation that holds on every class equally.

Run: pytest tests/test_audit_rules.py -v
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluation.audit_rules import MIN_DISCRIMINATION, audit_rule  # noqa: E402


def _frame(v_min=1.0, v_max=1.0, loading=40.0, rho=0.4, tripped=0):
    return {"voltage_pu_min": v_min, "voltage_pu_max": v_max, "loading_pct": loading,
            "rho_max": rho, "n_tripped_lines": tripped, "any_line_tripped": tripped > 0,
            "power_factor_at_max_load": 0.93, "generation_load_imbalance_pct": 1.0,
            "active_power_mw_max": 100.0, "reactive_power_mvar_max": 10.0,
            "apparent_power_mva_max": 100.0, "current_a_max": 500.0,
            "total_generation_mw": 1000.0, "total_load_mw": 1000.0}


NORMAL = [_frame() for _ in range(10)]
OVERLOAD = [_frame(loading=120.0, rho=1.2) for _ in range(10)]
TRIPPED = [_frame(tripped=1) for _ in range(10)]
CTX = {"gridA": {"normal": NORMAL, "overload": OVERLOAD, "line_trip": TRIPPED}}


def _rule(cond, role="CONSTRAINT", affirms=None):
    return {"rule_id": "R_T", "entity": "Grid", "condition": cond,
            "role": role, "affirms": affirms}


# ── CONSTRAINTS ───────────────────────────────────────────────────────────────

def test_constraint_that_fires_on_abnormal_frames_is_useful():
    a = audit_rule(_rule("loading_pct > 100"), CTX)
    assert a.verdict == "USEFUL"


def test_constraint_no_frame_can_reach_is_inert():
    """Passes the guard (never fires on healthy) but can never block either."""
    a = audit_rule(_rule("voltage_pu_max > 1.5"), CTX)
    assert a.verdict == "INERT"
    assert "cannot block" in a.detail


def test_constraint_firing_on_only_one_grid_is_topology_dependent():
    ctx = {"gridA": {"normal": NORMAL, "overload": OVERLOAD},
           "gridB": {"normal": NORMAL, "overload": NORMAL}}   # gridB never overloads
    a = audit_rule(_rule("loading_pct > 100"), ctx)
    assert a.verdict == "TOPOLOGY_DEPENDENT"


def test_rule_without_an_explicit_role_is_treated_as_a_constraint():
    a = audit_rule({"rule_id": "R_X", "entity": "Grid", "condition": "loading_pct > 100"}, CTX)
    assert a.role == "CONSTRAINT" and a.verdict == "USEFUL"


# ── AFFIRMATIONS ──────────────────────────────────────────────────────────────

def test_affirmation_that_separates_its_class_is_useful():
    """Must separate its class from EVERY other class, not just one — `loading_pct
    <= 100` alone is also true on line_trip frames and so proves nothing."""
    a = audit_rule(
        _rule("loading_pct <= 100 and n_tripped_lines == 0",
              role="AFFIRMATION", affirms="normal"),
        CTX,
    )
    assert a.verdict == "USEFUL"


def test_affirmation_separating_only_one_of_several_classes_is_uninformative():
    a = audit_rule(_rule("loading_pct <= 100", role="AFFIRMATION", affirms="normal"), CTX)
    assert a.verdict == "UNINFORMATIVE", "true on line_trip frames too"


def test_affirmation_true_on_every_class_is_uninformative():
    """The measured case: a wide voltage band holds on normal AND on overload,
    so it supports every prediction equally and is evidence for nothing."""
    a = audit_rule(
        _rule("voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1",
              role="AFFIRMATION", affirms="normal"),
        CTX,
    )
    assert a.verdict == "UNINFORMATIVE"
    assert "alike" in a.detail


def test_affirmation_of_an_unsampled_class_yields_no_evidence():
    a = audit_rule(_rule("n_tripped_lines >= 3", role="AFFIRMATION", affirms="cascade"), CTX)
    assert a.verdict == "NO_EVIDENCE"


def test_discrimination_threshold_is_the_documented_constant():
    """A margin just under the cutoff must not be reported as useful."""
    tiny = [_frame()] * 99 + [_frame(loading=120.0)]
    ctx = {"g": {"normal": [_frame()] * 100, "overload": tiny}}
    a = audit_rule(_rule("loading_pct <= 100", role="AFFIRMATION", affirms="normal"), ctx)
    assert MIN_DISCRIMINATION == 0.05
    assert a.verdict == "UNINFORMATIVE"
