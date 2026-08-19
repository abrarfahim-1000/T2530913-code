"""
Tests for the forecast (lookahead) labelling in scripts/generate_dataset.py.

This runs unattended during a multi-hour generation, so an off-by-one here is
expensive to discover. The properties that matter:

  - `at_risk` is TRUE iff a fault of the selected `risk_source` occurs in
    (t, t+horizon]
  - the last `horizon` frames of a chronic are DROPPED, not labelled safe
  - frames stay contiguous (a hole breaks someone's lookahead window)
  - every per-class distance is recorded on every frame, so both the horizon
    and the risk source can be re-tuned by relabelling rather than regenerating

Run: pytest tests/test_forecast_labels.py -v
"""
import io
import json
import os
import sys
from collections import Counter

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from generate_dataset import (DEFAULT_RISK_SOURCE, RISK_LABEL_MAP, RISK_SOURCES,
                              flush_forecast_chronic)


def _frames(labels):
    return [
        {"rho": [0.5], "v_or": [150.0], "line_status": [True],
         "label": lab, "label_int": 0, "fault_loc": -1,
         "timestep": i, "chronic_id": 0, "reward": 0.0}
        for i, lab in enumerate(labels)
    ]


def _run(labels, horizon, budget=10_000, risk_source="any"):
    buf = io.StringIO()
    n = flush_forecast_chronic(_frames(labels), horizon, buf, Counter(), budget,
                               risk_source=risk_source)
    rows = [json.loads(line) for line in buf.getvalue().splitlines()]
    return n, rows


def test_at_risk_marks_exactly_the_lookahead_window():
    # fault at index 5, horizon 2 -> only t=3 and t=4 can see it
    n, rows = _run(["normal"] * 5 + ["overload"] + ["normal"] * 4, horizon=2)
    assert n == 8, "last `horizon` frames must be dropped"
    assert [r["risk_label"] for r in rows] == (
        ["not_at_risk"] * 3 + ["at_risk"] * 2 + ["not_at_risk"] * 3
    )


def test_trailing_frames_are_dropped_not_labelled_safe():
    """Their lookahead window runs past the episode end, so the label is undefined.
    Labelling them `not_at_risk` would mark the end of every chronic as safe."""
    n, rows = _run(["normal"] * 10, horizon=3)
    assert n == 7
    assert max(r["timestep"] for r in rows) == 6


def test_chronic_shorter_than_horizon_yields_nothing():
    n, rows = _run(["normal"] * 4, horizon=6)
    assert n == 0 and rows == []


def test_fault_beyond_horizon_is_not_at_risk():
    """A fault 5 steps away is not visible at horizon 2 — but the distance is still
    recorded, so the horizon can be re-tuned without regenerating."""
    _, rows = _run(["normal"] * 8 + ["cascade"] + ["normal"] * 3, horizon=2)
    assert rows[0]["risk_label"] == "not_at_risk"
    assert rows[0]["steps_to_fault"] == 8


def test_every_non_normal_class_counts_as_a_fault_under_risk_source_any():
    for fault in ("overload", "line_trip", "cascade"):
        _, rows = _run(["normal", "normal", fault, "normal", "normal"], horizon=1)
        assert rows[1]["risk_label"] == "at_risk", fault


@pytest.mark.parametrize("risk_source,triggers", sorted(RISK_SOURCES.items()))
def test_risk_source_selects_which_faults_trigger(risk_source, triggers):
    """The default is `overload` because line-trip onset is injected by an
    unconditional Bernoulli draw and is therefore unpredictable by construction —
    an `any` target is dominated by that noise. See flush_forecast_chronic()."""
    for fault in ("overload", "line_trip", "cascade"):
        _, rows = _run(["normal", "normal", fault, "normal", "normal"],
                       horizon=1, risk_source=risk_source)
        expected = "at_risk" if fault in triggers else "not_at_risk"
        assert rows[1]["risk_label"] == expected, f"{risk_source} / {fault}"


def test_default_risk_source_ignores_injected_line_trips():
    assert DEFAULT_RISK_SOURCE == "overload"
    _, rows = _run(["normal"] * 3 + ["line_trip"] + ["normal"] * 3,
                   horizon=2, risk_source=DEFAULT_RISK_SOURCE)
    assert all(r["risk_label"] == "not_at_risk" for r in rows)
    # ...but the trip is still located, so an `any` target stays recoverable.
    assert rows[1]["steps_to_trip"] == 2
    assert rows[1]["steps_to_overload"] == -1


def test_per_class_distances_allow_relabelling_without_regenerating():
    labels = ["normal", "normal", "overload", "line_trip", "normal", "cascade",
              "normal", "normal"]
    _, rows = _run(labels, horizon=1, risk_source="overload")
    r0 = rows[0]
    assert r0["steps_to_overload"] == 2
    assert r0["steps_to_trip"] == 3
    assert r0["steps_to_cascade"] == 5
    assert r0["steps_to_fault"] == 2          # earliest of the three
    assert r0["risk_source"] == "overload"

    # The point of recording the distances: any (source, horizon) pair can be
    # reconstructed from a file generated under a DIFFERENT pair. Relabel the
    # `overload` rows above into every other combination and check they match
    # what the generator would have written natively.
    dist_key = {"overload": "steps_to_overload", "line_trip": "steps_to_trip",
                "cascade": "steps_to_cascade"}
    for source, triggers in RISK_SOURCES.items():
        for h in (1, 2, 3):
            _, native = _run(labels, horizon=h, risk_source=source)
            by_t = {r["timestep"]: r["risk_label"] for r in native}
            for row in rows:
                t = row["timestep"]
                if t not in by_t:
                    continue                  # dropped tail at this horizon
                ds = [row[dist_key[k]] for k in triggers if row[dist_key[k]] >= 0]
                relabelled = "at_risk" if (ds and min(ds) <= h) else "not_at_risk"
                assert relabelled == by_t[t], f"{source} h={h} t={t}"


def test_budget_caps_output():
    n, rows = _run(["normal"] * 20, horizon=2, budget=5)
    assert n == 5 and len(rows) == 5


def test_records_carry_horizon_and_keep_the_legacy_label():
    """The 4-class label is retained for reference/reporting but is NOT the target."""
    _, rows = _run(["normal"] * 6 + ["overload"] + ["normal"] * 3, horizon=2)
    assert all(r["horizon"] == 2 for r in rows)
    assert all("label" in r and "risk_int" in r for r in rows)
    assert {r["risk_int"] for r in rows} <= set(RISK_LABEL_MAP.values())


def test_timesteps_remain_contiguous():
    """A gap would silently corrupt the lookahead window of earlier frames."""
    _, rows = _run(["normal"] * 12, horizon=3)
    ts = [r["timestep"] for r in rows]
    assert ts == list(range(len(ts)))


@pytest.mark.parametrize("horizon", [1, 2, 4, 6])
@pytest.mark.parametrize("risk_source", sorted(RISK_SOURCES))
def test_risk_label_matches_a_brute_force_check(horizon, risk_source):
    labels = ["normal", "normal", "overload", "normal", "normal", "normal",
              "cascade", "normal", "normal", "normal", "line_trip", "normal"]
    triggers = RISK_SOURCES[risk_source]
    _, rows = _run(labels, horizon, risk_source=risk_source)
    for r in rows:
        t = r["timestep"]
        window = labels[t + 1: t + 1 + horizon]
        expected = "at_risk" if any(w in triggers for w in window) else "not_at_risk"
        assert r["risk_label"] == expected, f"t={t} h={horizon} src={risk_source}"
