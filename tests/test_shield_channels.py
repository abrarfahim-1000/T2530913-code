"""
Tests for the v2 explanation channels.

The load-bearing one is `test_no_non_block_rule_can_ever_veto`. Everything else
in this file is about what the shield *says*; that one is about what it is
structurally incapable of saying. Section 13's 92-94% intervention precision is
measured on the BLOCK channel, so a WARN rule that found its way into the veto
path would move a headline result silently -- the shield would still return
BLOCK, the harness would still compute an F1, and nothing would look broken.

Run: pytest tests/test_shield_channels.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shield.channels import (
    BLOCK,
    INERT,
    NORMAL,
    NOT_APPLICABLE,
    WARN,
    assert_warnings_calibrated,
    channel_of,
    partition_by_channel,
)
from shield.shield import SECURE, SHIELD_VERSION, VIOLATION, validate, validate_n1

CORPUS = Path(__file__).resolve().parent.parent / "shield_corpus" / "all_rules_channels.jsonl"
SERVED = (Path(__file__).resolve().parent.parent / "validated_translated"
          / "all_rules_deduped.jsonl")

# A context that trips essentially anything, so "did not block" can never be
# confused with "did not fire".
EXTREME = {
    "loading_pct": 250.0,
    "rho_max": 2.5,
    "voltage_pu_min": 0.40,
    "voltage_pu_max": 1.90,
    "n_tripped_lines": 9,
    "any_line_tripped": True,
    "power_factor_at_max_load": 0.10,
    "active_power_mw_max": 900.0,
    "reactive_power_mvar_max": 900.0,
    "apparent_power_mva_max": 1200.0,
    "current_a_max": 9000.0,
    "total_generation_mw": 900.0,
    "total_load_mw": 500.0,
    "generation_load_imbalance_pct": 80.0,
}

BLOCKER = {
    "rule_id": "B_1", "channel": BLOCK, "role": "CONSTRAINT",
    "condition": "loading_pct > 100", "severity": "high",
    "source": "IEEE Std 1234, 5.1", "explanation": "Thermal rating shall not be exceeded.",
}
WARNER = {
    "rule_id": "W_1", "channel": WARN, "role": "CONSTRAINT",
    "condition": "voltage_pu_min < 0.95", "severity": "high",
    "source": "Grid Code 4.2", "explanation": "Voltage shall be held within band.",
    "calibration": {"verdict": "ELEVATED", "p_violation_given_fires": 0.592,
                    "p_violation_given_silent": 0.225, "discrimination": 0.537,
                    "grids": ["case14", "wcci2022"]},
}
AFFIRMER = {
    "rule_id": "N_1", "channel": NORMAL, "role": "AFFIRMATION", "affirms": "secure",
    "condition": "voltage_pu_min >= 0.4", "severity": "low",
    "source": "NERC TPL-001", "explanation": "Telemetry consistent with normal operation.",
}
DEMOTED = {
    "rule_id": "X_1", "channel": NOT_APPLICABLE, "role": "CONSTRAINT",
    "condition": "power_factor_at_max_load < 0.95", "severity": "high",
    "source": "Grid Code 9.9", "explanation": "Power factor requirement.",
}
DEAD = {
    "rule_id": "I_1", "channel": INERT, "role": "CONSTRAINT",
    "condition": "rho_max > 1.0", "severity": "high",
    "source": "nowhere", "explanation": "Inert by measurement.",
}


# -- THE STRUCTURAL BAR --------------------------------------------------------

@pytest.mark.parametrize("rule", [WARNER, AFFIRMER, DEMOTED, DEAD])
def test_no_non_block_rule_can_ever_veto(rule):
    """A non-BLOCK rule whose condition is TRUE must not produce a BLOCK.

    Each of these fires on EXTREME. If channel routing regressed, the status
    would flip and 13's precision would be measured on a wider set than the one
    it was validated on.
    """
    result = validate({**EXTREME, "fault_type": SECURE}, [rule])
    assert result.status == "PASS"
    assert result.violated_rules == []


def test_violated_rules_is_always_a_subset_of_the_block_channel():
    """The invariant, stated over a mixed ruleset rather than one rule at a time."""
    rules = [BLOCKER, WARNER, AFFIRMER, DEMOTED, DEAD]
    result = validate({**EXTREME, "fault_type": SECURE}, rules)
    block_ids = {r["rule_id"] for r in partition_by_channel(rules)[BLOCK]}
    assert {r["rule_id"] for r in result.violated_rules} <= block_ids
    assert result.status == "BLOCK"          # the BLOCK rule did fire
    assert result.violated_rules[0]["rule_id"] == "B_1"


def test_a_firing_warning_does_not_change_status_but_is_reported():
    result = validate({**EXTREME, "fault_type": SECURE}, [WARNER])
    assert result.status == "PASS"
    assert result.warned is True
    assert [r["rule_id"] for r in result.warning_rules] == ["W_1"]


def test_unknown_channel_fails_to_silence_not_to_veto():
    """A typo in `channel` must not promote a rule into the veto path."""
    rogue = {**WARNER, "channel": "BLOKC"}
    assert channel_of(rogue) == NOT_APPLICABLE
    assert validate({**EXTREME, "fault_type": SECURE}, [rogue]).status == "PASS"


def test_inert_rules_are_not_evaluated_at_all():
    """INERT is a measured verdict (19.3), not a runtime guess; evaluating them
    buys a counter and nothing else."""
    assert validate({**EXTREME, "fault_type": SECURE}, [DEAD]).n_evaluated == 0


# -- BACK-COMPATIBILITY --------------------------------------------------------

def test_a_rule_without_a_channel_keeps_v1_behaviour():
    """The served corpus carries no `channel`. Every recorded figure depends on
    CONSTRAINT still blocking and AFFIRMATION still not blocking."""
    constraint = {k: v for k, v in BLOCKER.items() if k != "channel"}
    affirmation = {k: v for k, v in AFFIRMER.items() if k != "channel"}
    assert channel_of(constraint) == BLOCK
    assert channel_of(affirmation) == NORMAL
    assert validate({**EXTREME, "fault_type": SECURE}, [constraint]).status == "BLOCK"
    assert validate({**EXTREME, "fault_type": SECURE}, [affirmation]).status == "PASS"


def test_served_corpus_still_blocks_exactly_as_v1():
    if not SERVED.exists():
        pytest.skip("served corpus not on disk")
    rules = [json.loads(line) for line in SERVED.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    result = validate({**EXTREME, "fault_type": SECURE}, rules)
    assert result.status == "BLOCK"
    assert all(str(r.get("role", "CONSTRAINT")).upper() == "CONSTRAINT"
               for r in result.violated_rules)


# -- THE THREE REGISTERS -------------------------------------------------------

def test_block_explanation_names_the_rule_and_its_source():
    text = validate({**EXTREME, "fault_type": SECURE}, [BLOCKER]).explanation
    assert "blocked" in text.lower()
    assert "B_1" in text and "IEEE Std 1234" in text


def test_warning_explanation_quotes_its_measured_rate():
    """An uncalibrated warning is an alarm. The rate has to reach the text."""
    text = validate({**EXTREME, "fault_type": SECURE}, [WARNER]).explanation
    assert "Warning" in text
    assert "59.2%" in text and "22.5%" in text    # P|fire against P|quiet, 21.2
    assert "case14" in text


def test_warning_without_calibration_says_so_rather_than_implying_a_rate():
    bare = {k: v for k, v in WARNER.items() if k != "calibration"}
    text = validate({**EXTREME, "fault_type": SECURE}, [bare]).explanation
    assert "no measured rate" in text


def test_normal_register_names_the_standards_that_hold():
    text = validate({**EXTREME, "fault_type": SECURE}, [AFFIRMER]).explanation
    assert "Environment normal" in text
    assert "N_1" in text and "NERC TPL-001" in text


def test_pass_with_no_rules_still_says_something_specific():
    """v1 returned 'No applicable rule was violated.' on every PASS, which cannot
    distinguish a clean grid from a grid nothing could evaluate."""
    text = validate({**EXTREME, "fault_type": SECURE}, [BLOCKER]).explanation
    clean = validate({"loading_pct": 10.0, "fault_type": SECURE}, [BLOCKER]).explanation
    assert clean != text
    assert "constraint(s) checked" in clean


def test_result_carries_its_version_and_channel_census():
    payload = validate({**EXTREME, "fault_type": SECURE},
                       [BLOCKER, WARNER, AFFIRMER, DEMOTED, DEAD]).to_dict()
    assert payload["shield_version"] == SHIELD_VERSION
    assert payload["n_by_channel"][WARN] == 1
    assert payload["warning_rule_ids"] == ["W_1"]


# -- THE N-1 GATE --------------------------------------------------------------

def test_warnings_survive_the_conservative_downgrade():
    """A `violation` verdict is never blocked, but the operator still wants the
    reason it was right."""
    result = validate_n1(EXTREME, [BLOCKER, WARNER], predicted=VIOLATION)
    assert result.status == "PASS"
    assert [r["rule_id"] for r in result.warning_rules] == ["W_1"]
    assert "Warning" in result.explanation


def test_a_warning_never_blocks_a_secure_verdict():
    result = validate_n1(EXTREME, [WARNER], predicted=SECURE)
    assert result.status == "PASS"
    assert result.warned is True


# -- THE REAL CORPUS -----------------------------------------------------------

def _corpus():
    if not CORPUS.exists():
        pytest.skip("run evaluation/build_channel_corpus.py first")
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_shipped_corpus_has_every_warning_calibrated():
    """21.2 measured two INVERTED predicates. They must not be shippable."""
    assert_warnings_calibrated(_corpus())


def test_shipped_block_channel_matches_the_served_corpus_predicate_for_predicate():
    """If it does not, 13's precision is measured on a different set than ships."""
    corpus = _corpus()
    served = [json.loads(line) for line in SERVED.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    assert ({r["condition"].strip() for r in corpus if r["channel"] == BLOCK}
            == {r["condition"].strip() for r in served
                if str(r.get("role", "CONSTRAINT")).upper() == "CONSTRAINT"})


def test_shipped_corpus_reproduces_the_documented_channel_counts():
    """thesis_findings 21.3. A silent drift in these is a silent drift in the
    headline rule count."""
    counts = {}
    for rule in _corpus():
        counts[rule["channel"]] = counts.get(rule["channel"], 0) + 1
    assert counts == {BLOCK: 10, WARN: 20, NORMAL: 11, NOT_APPLICABLE: 13, INERT: 4}


def test_shipped_corpus_never_vetoes_outside_the_block_channel():
    """The structural bar, run over the corpus that actually ships."""
    corpus = _corpus()
    block_ids = {r["rule_id"] for r in corpus if r["channel"] == BLOCK}
    for predicted in (SECURE, VIOLATION):
        result = validate({**EXTREME, "fault_type": predicted}, corpus)
        assert {r["rule_id"] for r in result.violated_rules} <= block_ids
