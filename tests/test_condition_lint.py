"""
Tests for extraction.common.lint_condition — the deterministic guarantee that
every rule reaching the KG is machine-evaluable against Grid2Op telemetry.
Run: pytest tests/test_condition_lint.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction"))

from extraction.common import CONDITION_VOCABULARY, EXTRACT_PROMPT, VALIDATE_PROMPT, Rule, RawRule, lint_condition, lint_condition_raw
from pydantic import ValidationError


# ── ACCEPT ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cond", [
    "voltage_pu_min < 0.95 or voltage_pu_max > 1.05",
    "loading_pct > 100",
    "rho_max >= 1.0",
    "n_tripped_lines >= 2",
    "any_line_tripped",
    "not any_line_tripped",
    "rho_max > 1.0 and n_tripped_lines >= 1",
    "(voltage_pu_min < 0.9) or (loading_pct > 120 and any_line_tripped)",
    "voltage_pu_max > 1.1 or voltage_pu_max < -0.5",   # negative literal allowed
    "  loading_pct >= 110  ",                          # stripped
])
def test_accepts_valid_conditions(cond):
    assert lint_condition(cond) == cond.strip()


# ── REJECT ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cond", [
    # non-Python syntax seen in the v1 ruleset
    "voltage_pu BETWEEN 0.2 AND 0.3",
    "voltage_pu < 0.90 within 80ms OR voltage_pu < 0.97 within 2s",
    # out-of-vocabulary variables (unobservable in Grid2Op)
    "frequency_hz < 49",
    "droop_pct > 5",
    "voltage_pu > 1.05",                 # v1 name — replaced by _min/_max forms
    "p_mw < 0.2 * rated_mw",
    "time_since_verification_months > 6",
    # disallowed constructs
    "max(rho_max, 1.0) > 1.0",           # function call
    "rho_max + 0.1 > 1.0",               # arithmetic (BinOp not whitelisted)
    "loading_pct > '100'",               # non-numeric literal
    "__import__('os')",                  # sandbox escape attempt
    # not boolean / not an expression
    "rho_max",                           # evaluates to float, not bool
    "manual verification required",      # prose
    "",                                  # empty
])
def test_rejects_invalid_conditions(cond):
    with pytest.raises(ValueError):
        lint_condition(cond)


# ── INTEGRATION WITH Rule SCHEMA ──────────────────────────────────────────────

def _rule(condition: str) -> dict:
    return {
        "rule_id": "R_001",
        "source": "IEEE Std 1547-2018, Section 7.4",
        "entity": "Grid",
        "condition": condition,
        "action": "BLOCK",
        "severity": "high",
        "explanation": "test rule",
    }


def test_rule_schema_accepts_linted_condition():
    r = Rule(**_rule("voltage_pu_min < 0.95 or voltage_pu_max > 1.05"))
    assert r.condition == "voltage_pu_min < 0.95 or voltage_pu_max > 1.05"


def test_rule_schema_drops_unlintable_condition():
    with pytest.raises(ValidationError):
        Rule(**_rule("frequency_hz < 49"))


# ── PROMPT RENDERING ──────────────────────────────────────────────────────────

def test_prompts_contain_rendered_vocabulary():
    # VALIDATE_PROMPT and TRANSLATE_PROMPT contain the vocabulary (EXTRACT_PROMPT is raw)
    for name in CONDITION_VOCABULARY:
        assert name in VALIDATE_PROMPT
    assert "{vocabulary}" not in VALIDATE_PROMPT
    # runtime placeholders must survive the pre-render
    assert "{chunk}" in EXTRACT_PROMPT
    assert "{chunk}" in VALIDATE_PROMPT and "{rules}" in VALIDATE_PROMPT


# ── RAW RULE / lint_condition_raw ─────────────────────────────────────────────

@pytest.mark.parametrize("cond", [
    # Variables outside CONDITION_VOCABULARY are accepted by lint_condition_raw
    "frequency_hz < 49",
    "frequency_hz > 51.5",
    "power_factor < 0.9",
    "droop_pct > 5",
    "time_seconds > 0.16",
    "voltage_pu > 1.05",                   # old name — no _min/_max suffix needed in raw
    "n_tripped_lines >= 1 and voltage_pu_min < 0.95",  # mixed vocab + raw
])
def test_lint_condition_raw_accepts_any_variable(cond):
    """lint_condition_raw must accept any variable name (syntax-only check)."""
    assert lint_condition_raw(cond) == cond.strip()


@pytest.mark.parametrize("cond", [
    "voltage_pu BETWEEN 0.2 AND 0.3",      # non-Python syntax
    "max(rho_max, 1.0) > 1.0",             # function call
    "manual verification required",         # prose
    "",                                     # empty
])
def test_lint_condition_raw_rejects_bad_syntax(cond):
    with pytest.raises(ValueError):
        lint_condition_raw(cond)


def test_raw_rule_schema_accepts_out_of_vocab():
    """RawRule must accept conditions with variables outside Grid2Op vocabulary."""
    r = RawRule(**_rule("frequency_hz < 49"))
    assert r.condition == "frequency_hz < 49"


def test_raw_rule_schema_rejects_bad_syntax():
    """RawRule must still reject non-Python syntax."""
    with pytest.raises(ValidationError):
        RawRule(**_rule("voltage_pu BETWEEN 0.2 AND 0.3"))


def test_raw_rule_and_rule_differ_on_condition():
    """The same condition should pass RawRule but fail Rule when it uses
    out-of-vocabulary variables."""
    condition = "frequency_hz < 49"
    RawRule(**_rule(condition))             # should pass
    with pytest.raises(ValidationError):
        Rule(**_rule(condition))            # should fail
