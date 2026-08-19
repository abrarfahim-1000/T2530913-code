"""
Tests for the N-1 shield harness (evaluation/eval_shield_n1.py).

The harness makes three claims that are easy to get silently wrong, and each is
pinned here:

  1. BLOCK means escalate-to-unsafe, so a blocked `secure` prediction becomes a
     predicted violation. Get this backwards and the shield *lowers* recall.
  2. `validate_n1` is memoised per (frame, predicted label). That is only sound
     because the shield is a pure function of the frame context and the label —
     if that ever stops holding, this test fails rather than the numbers quietly
     drifting.
  3. The threshold is selected on a split that is not the split being reported.

Run: pytest tests/test_eval_shield_n1.py -v
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluation.eval_shield_n1 import (  # noqa: E402
    best_f1_and_thr,
    failure_mode,
    load_rules,
    prf,
    run_shield,
)
from shield.shield import SECURE, VIOLATION, validate_n1  # noqa: E402


OVERLOAD_RULE = {"rule_id": "R_1", "entity": "Line", "condition": "loading_pct > 100",
                 "role": "CONSTRAINT", "severity": "critical"}


# ── metric plumbing ───────────────────────────────────────────────────────────

def test_prf_counts_the_dangerous_error_separately():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    m = prf(y, pred)
    assert m["tp"] == 1 and m["missed_violations"] == 1 and m["false_alarms"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)


def test_prf_is_defined_when_nothing_is_predicted_positive():
    m = prf(np.array([1, 0]), np.array([0, 0]))
    assert m["f1"] == 0.0 and m["missed_violations"] == 1


def test_best_f1_matches_the_cross_topology_harness():
    """Same helper name, same semantics — the two harnesses must not disagree
    about what the model scores."""
    from evaluation.eval_n1_cross_topology import best_f1_and_thr as other
    rng = np.random.default_rng(0)
    score = rng.normal(size=500)
    y = (rng.random(500) < 0.3).astype(int)
    assert best_f1_and_thr(score, y) == other(score, y)


# ── failure taxonomy ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("truth,pred,blocked,expected", [
    (1, 0, True,  "missed_violation_caught"),
    (1, 0, False, "missed_violation_uncaught"),
    (0, 0, True,  "false_block"),
    (0, 0, False, "correct_secure"),
    (1, 1, False, "predicted_violation"),
    (0, 1, False, "predicted_violation"),
])
def test_failure_modes(truth, pred, blocked, expected):
    assert failure_mode(truth, pred, blocked) == expected


# ── the memoisation premise ───────────────────────────────────────────────────

def test_validate_n1_is_a_pure_function_of_context_and_label():
    """run_shield evaluates the shield once per (frame, label) and reuses it
    across that frame's contingencies. That is exact only if repeated calls agree."""
    ctx = {"loading_pct": 140.0, "rho_max": 1.4, "n_tripped_lines": 0,
           "any_line_tripped": False}
    a = validate_n1(ctx, [OVERLOAD_RULE], predicted=SECURE)
    b = validate_n1(ctx, [OVERLOAD_RULE], predicted=SECURE)
    assert a.status == b.status == "BLOCK"
    assert [r["rule_id"] for r in a.violated_rules] == [r["rule_id"] for r in b.violated_rules]
    assert validate_n1(ctx, [OVERLOAD_RULE], predicted=VIOLATION).status == "PASS"


# ── the arm semantics, end to end over a synthetic dataset ────────────────────

def _write_frames(tmp_path, frames):
    """Minimal N-1 records: two lines, one bus pair. Only what build_context reads."""
    p = tmp_path / "frames.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for rho_max in frames:
            f.write(json.dumps({
                "rho": [rho_max, 0.1],
                "v_or": [138.0, 138.0],
                "line_status": [True, True],
                "p_or": [50.0, 10.0], "q_or": [5.0, 1.0],
                "gen_p": [60.0], "load_p": [60.0],
            }) + "\n")
    return p


def _scored(jsonl, frame_of_contingency, y):
    n = len(y)
    return {
        "jsonl": str(jsonl),
        "frame": np.asarray(frame_of_contingency),
        "line": np.zeros(n, dtype=int),
        "y": np.asarray(y),
    }


def test_block_escalates_to_unsafe_and_recovers_a_missed_violation(tmp_path, monkeypatch):
    """Frame 0 is overloaded (rule fires); the model calls its contingency secure
    but it is truly a violation. The shield must flip it to 1."""
    import evaluation.eval_shield_n1 as mod
    monkeypatch.setattr(mod, "load_base_kv", lambda tag, **kw: np.array([138.0, 138.0]))

    jsonl = _write_frames(tmp_path, [1.4, 0.4])          # frame 0 hot, frame 1 calm
    scored = _scored(jsonl, [0, 1], [1, 1])
    pred = np.array([0, 0])                               # both called secure

    out = mod.run_shield(scored, pred, [OVERLOAD_RULE], "t", tmp_path / "f.jsonl")

    assert out["blocked"].tolist() == [True, False], "only the hot frame is blockable"
    assert out["shielded"].tolist() == [1, 0], "BLOCK escalates to violation"
    assert out["base_violates"].tolist() == [True, False]
    assert prf(scored["y"], out["shielded"])["missed_violations"] == 1
    assert prf(scored["y"], pred)["missed_violations"] == 2


def test_conservative_predictions_are_never_blocked(tmp_path, monkeypatch):
    """A `violation` prediction on an overloaded base case must pass — blocking the
    cautious answer is the one thing a safety gate must never do (§6.1)."""
    import evaluation.eval_shield_n1 as mod
    monkeypatch.setattr(mod, "load_base_kv", lambda tag, **kw: np.array([138.0, 138.0]))

    jsonl = _write_frames(tmp_path, [1.4])
    scored = _scored(jsonl, [0], [0])
    out = mod.run_shield(scored, np.array([1]), [OVERLOAD_RULE], "t", None)

    assert out["blocked"].tolist() == [False]
    assert out["shielded"].tolist() == [1]


def test_a_calm_grid_is_never_blocked_so_the_shield_adds_no_false_alarms(tmp_path, monkeypatch):
    import evaluation.eval_shield_n1 as mod
    monkeypatch.setattr(mod, "load_base_kv", lambda tag, **kw: np.array([138.0, 138.0]))

    jsonl = _write_frames(tmp_path, [0.4, 0.5, 0.6])
    scored = _scored(jsonl, [0, 1, 2], [0, 0, 0])
    out = mod.run_shield(scored, np.zeros(3, dtype=int), [OVERLOAD_RULE], "t", None)

    assert not out["blocked"].any()
    assert out["n_error"] == 0


def test_one_context_is_built_per_frame_not_per_contingency(tmp_path, monkeypatch):
    """Three contingencies on one frame must cost one context build — the whole
    reason the harness is tractable on wcci2022's 742k labels."""
    import evaluation.eval_shield_n1 as mod
    monkeypatch.setattr(mod, "load_base_kv", lambda tag, **kw: np.array([138.0, 138.0]))

    jsonl = _write_frames(tmp_path, [1.4])
    scored = _scored(jsonl, [0, 0, 0], [1, 0, 1])
    out = mod.run_shield(scored, np.zeros(3, dtype=int), [OVERLOAD_RULE], "t", None)

    assert out["n_frames"] == 1
    assert out["blocked"].tolist() == [True, True, True]


def test_failure_log_is_written_and_capped(tmp_path, monkeypatch):
    import evaluation.eval_shield_n1 as mod
    monkeypatch.setattr(mod, "load_base_kv", lambda tag, **kw: np.array([138.0, 138.0]))

    jsonl = _write_frames(tmp_path, [1.4])
    scored = _scored(jsonl, [0, 0, 0], [1, 1, 1])
    log = tmp_path / "failures.jsonl"
    out = mod.run_shield(scored, np.zeros(3, dtype=int), [OVERLOAD_RULE], "t", log,
                         max_failures=2)

    rows = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 2 and out["n_logged"] == 2
    assert rows[0]["mode"] == "missed_violation_caught"
    assert rows[0]["violated"] == ["R_1"]


# ── rule loading ──────────────────────────────────────────────────────────────

def test_load_rules_unwraps_guarded_records(tmp_path):
    """Guard output wraps each rule as {"rule": {...}, "fire_rates": {...}}; the
    deduped stage-3 file does not. Both must load."""
    d = tmp_path / "guarded"
    d.mkdir()
    (d / "doc_translated.jsonl").write_text(
        json.dumps({"rule": OVERLOAD_RULE, "fire_rates": {"case14": 0.0}}) + "\n",
        encoding="utf-8")
    assert load_rules(str(d)) == [OVERLOAD_RULE]

    flat = tmp_path / "all_rules_deduped.jsonl"
    flat.write_text(json.dumps(OVERLOAD_RULE) + "\n", encoding="utf-8")
    assert load_rules(str(flat)) == [OVERLOAD_RULE]


def test_load_rules_exits_with_guidance_when_stage_three_has_not_run(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_rules(str(tmp_path / "nope.jsonl"))
    assert "validate.py" in str(exc.value)
