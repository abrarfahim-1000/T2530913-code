"""
Tests for mapping translator output back onto the rules it answers
(extraction/translate.py::build_result_map).

These encode the 2026-08-19 stage-2 failure: every worked example in
TRANSLATE_PROMPT omitted `rule_id` while `TranslationResult` required it, so
every entry failed validation, was swallowed by a bare `except ValidationError:
pass`, and 2,190 of 2,463 rules (88.9%) were written out as NO_VERDICT. A total
plumbing failure presented as a fact about the corpus.

Run: pytest tests/test_translate_mapping.py -v
"""
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "extraction"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from translate import build_result_map  # noqa: E402
from extraction.common import TRANSLATE_PROMPT  # noqa: E402


def _rules(*ids):
    return [{"rule_id": r, "condition": "loading > 100"} for r in ids]


# ── THE REGRESSION ────────────────────────────────────────────────────────────

def test_entries_carrying_rule_id_map_directly():
    raw = [{"rule_id": "R_001", "translatable": True, "condition": "loading_pct > 100"},
           {"rule_id": "R_002", "translatable": False, "reason": "frequency: not observable"}]
    result_map, rejected = build_result_map(raw, _rules("R_001", "R_002"))
    assert set(result_map) == {"R_001", "R_002"}
    assert rejected == 0


def test_entries_missing_rule_id_are_recovered_by_position():
    """The exact shape the old prompt asked for. Previously every one of these was
    discarded, costing the rule its verdict."""
    raw = [{"translatable": True, "condition": "loading_pct > 100", "role": "CONSTRAINT"},
           {"translatable": False, "condition": None, "reason": "ride-through: needs duration"}]
    result_map, rejected = build_result_map(raw, _rules("R_001", "R_002"))
    assert set(result_map) == {"R_001", "R_002"}
    assert result_map["R_001"].condition == "loading_pct > 100"
    assert result_map["R_002"].translatable is False
    assert rejected == 0


def test_positional_recovery_is_refused_on_a_length_mismatch():
    """A partial response gives no way to tell which rule an unkeyed entry answers.
    Guessing would attach a translated condition to the wrong standard."""
    raw = [{"translatable": True, "condition": "loading_pct > 100"}]
    result_map, rejected = build_result_map(raw, _rules("R_001", "R_002", "R_003"))
    assert result_map == {}, "must not guess"
    assert rejected == 1


def test_positional_recovery_is_refused_when_some_entries_were_keyed():
    """Mixed keying means the array is not reliably one-per-rule in order."""
    raw = [{"rule_id": "R_002", "translatable": True, "condition": "rho_max > 1.0"},
           {"translatable": True, "condition": "loading_pct > 100"}]
    result_map, rejected = build_result_map(raw, _rules("R_001", "R_002"))
    assert set(result_map) == {"R_002"}, "the keyed entry survives, the unkeyed one is not guessed"
    assert rejected == 1


def test_malformed_entries_are_counted_not_silently_dropped():
    raw = [{"rule_id": "R_001"},              # no `translatable`
           "not an object",
           {"rule_id": "R_002", "translatable": True, "condition": "rho_max > 1.0"}]
    result_map, rejected = build_result_map(raw, _rules("R_001", "R_002"))
    assert set(result_map) == {"R_002"}
    assert rejected == 2


def test_empty_response_yields_no_verdicts_and_is_counted():
    result_map, rejected = build_result_map([], _rules("R_001", "R_002"))
    assert result_map == {} and rejected == 0


# ── THE PROMPT SIDE OF THE SAME CONTRACT ──────────────────────────────────────

def test_prompt_asks_for_rule_id_in_every_worked_example():
    """The parser keys on `rule_id`; every example the model is shown must carry it,
    or the model is being instructed to produce output the parser cannot use."""
    openings = [ln.strip() for ln in TRANSLATE_PROMPT.splitlines() if '{"' in ln]
    assert len(openings) >= 6, f"expected the worked examples, found {len(openings)}"
    missing = [ln for ln in openings if '"rule_id"' not in ln]
    assert not missing, f"example(s) omit rule_id: {missing}"


def test_prompt_states_the_one_entry_per_rule_contract():
    assert "rule_id" in TRANSLATE_PROMPT
    assert "ONE entry per rule" in TRANSLATE_PROMPT
