"""
Tests for stage-3 rejection persistence (extraction/validate.py).

These encode the 2026-08-20 validation failure: the run rejected 31 of 32
guarded rules and wrote nothing but a count. `n_rejected` was incremented and
the Verdict — including the model's `reason` — was dropped on the floor, so
there was no way to tell a corpus property from a broken prompt. Among the
discarded 31 was the `loading_pct > 100` family the shield's measured wcci2022
gain rests on.

Run: pytest tests/test_validate_rejections.py -v
"""
import json
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "extraction"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import validate  # noqa: E402
from extraction.common import VALIDATE_PROMPT, VALIDATE_PROMPT_TRANSLATED  # noqa: E402


def _rule(rule_id, condition="loading_pct > 100"):
    return {
        "rule_id": rule_id,
        "source": "TPL-001-5.1, Section 5.1.f",
        "entity": "Facility",
        "condition": condition,
        "action": "BLOCK",
        "severity": "high",
        "explanation": "Applicable facility ratings must not be exceeded.",
        "role": "CONSTRAINT",
    }


def _write_candidates(tmp_path, rules, chunk="Applicable Facility Ratings shall not be exceeded."):
    f = tmp_path / "some-standard_translated.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for r in rules:
            fh.write(json.dumps({"rule": r, "chunk": chunk}) + "\n")
    return f


def _stub_validator(monkeypatch, verdicts):
    """Replace the Ollama call with a fixed verdict list."""
    monkeypatch.setattr(validate, "run_validator", lambda chunk, rules, prompt: verdicts)


# ── THE REGRESSION ────────────────────────────────────────────────────────────

def test_rejections_are_written_with_their_reason(tmp_path, monkeypatch):
    """A rejection without its reason is not a finding. It must reach disk."""
    cand = _write_candidates(tmp_path, [_rule("R_167")])
    _stub_validator(monkeypatch, [
        {"rule_id": "R_167", "verdict": "REJECT",
         "reason": "The text does not state a numeric 100% threshold."},
    ])

    stats = validate.process_candidates_file(cand, tmp_path, VALIDATE_PROMPT)

    assert stats["n_rejected"] == 1
    rejected = (tmp_path / "some-standard_rejected.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rejected) == 1
    rec = json.loads(rejected[0])
    assert rec["rule"]["rule_id"] == "R_167"
    assert rec["rule"]["condition"] == "loading_pct > 100"
    assert rec["verdict"]["verdict"] == "REJECT"
    assert "100%" in rec["verdict"]["reason"]


def test_a_wholesale_rejection_leaves_a_full_audit_trail(tmp_path, monkeypatch):
    """The shape of the 2026-08-20 run: near-total rejection. Every rejected rule
    must be recoverable from disk, not just counted."""
    rules = [_rule(f"R_{i:03d}") for i in range(10)]
    cand = _write_candidates(tmp_path, rules)
    _stub_validator(monkeypatch, [
        {"rule_id": r["rule_id"], "verdict": "REJECT", "reason": "not stated verbatim"}
        for r in rules
    ])

    stats = validate.process_candidates_file(cand, tmp_path, VALIDATE_PROMPT)

    assert stats["n_rejected"] == 10
    assert stats["n_confirmed"] == 0
    recovered = [json.loads(l)["rule"]["rule_id"]
                 for l in (tmp_path / "some-standard_rejected.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
    assert recovered == [r["rule_id"] for r in rules]


def test_confirmed_and_flagged_streams_still_separate(tmp_path, monkeypatch):
    """Rejections go to their own file — they must not contaminate the confirmed
    corpus (deduplicate_rules merges every *_confirmed.jsonl) or the flagged one."""
    cand = _write_candidates(tmp_path, [_rule("R_001"), _rule("R_002"), _rule("R_003")])
    _stub_validator(monkeypatch, [
        {"rule_id": "R_001", "verdict": "CONFIRM", "reason": "faithful"},
        {"rule_id": "R_002", "verdict": "REJECT", "reason": "no such requirement"},
        # R_003 gets no verdict at all → flagged
    ])

    stats = validate.process_candidates_file(cand, tmp_path, VALIDATE_PROMPT)

    assert (stats["n_confirmed"], stats["n_rejected"], stats["n_flagged"]) == (1, 1, 1)
    confirmed = (tmp_path / "some-standard_confirmed.jsonl").read_text(encoding="utf-8").splitlines()
    rejected = (tmp_path / "some-standard_rejected.jsonl").read_text(encoding="utf-8").splitlines()
    flagged = (tmp_path / "some-standard_flagged.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(confirmed[0])["rule_id"]] == ["R_001"]
    assert [json.loads(rejected[0])["rule"]["rule_id"]] == ["R_002"]
    assert json.loads(flagged[0])["verdict"]["verdict"] == "NO_VERDICT"


# ── PROMPT ARMS ───────────────────────────────────────────────────────────────

def test_both_prompt_variants_are_selectable_and_distinct():
    assert set(validate.PROMPT_VARIANTS) == {"strict", "translated"}
    # `==`, not `is`: validate.py imports common via sys.path, the test via the
    # package path, so the two module instances hold equal-but-distinct strings.
    assert validate.PROMPT_VARIANTS["strict"] == VALIDATE_PROMPT
    assert validate.PROMPT_VARIANTS["translated"] == VALIDATE_PROMPT_TRANSLATED
    assert VALIDATE_PROMPT != VALIDATE_PROMPT_TRANSLATED


@pytest.mark.parametrize("prompt", [VALIDATE_PROMPT, VALIDATE_PROMPT_TRANSLATED])
def test_both_arms_share_the_mechanical_criteria_and_format_cleanly(prompt):
    """Criteria 4-7 and the output contract are the shared body — a variant that
    lost them would silently change what is being measured."""
    for marker in ("HEALTHY-GRID SELF-CHECK", "RIDE-THROUGH RANGES ARE NEITHER ROLE",
                   "ROLE CORRECTNESS", "Output ONLY a JSON array"):
        assert marker in prompt
    rendered = prompt.format(chunk="some text", rules="[]")
    assert "some text" in rendered


def test_the_prompt_reaches_the_validator_call(tmp_path, monkeypatch):
    """--prompt-variant must actually change the prompt sent, not just the log line."""
    seen = []
    monkeypatch.setattr(validate, "run_validator",
                        lambda chunk, rules, prompt: seen.append(prompt) or [])
    cand = _write_candidates(tmp_path, [_rule("R_001")])
    validate.process_candidates_file(cand, tmp_path, VALIDATE_PROMPT_TRANSLATED)
    assert seen == [VALIDATE_PROMPT_TRANSLATED]
