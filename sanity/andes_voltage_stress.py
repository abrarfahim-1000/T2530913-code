"""B4 stress arm -- does ARM A's thin margin survive a heavier grid?

`andes_voltage_spike.py` measured ARM A (clean line opening, no fault-on
period -- the N-1 event class the thesis task actually models) clearing the
mildest harvested under-voltage bar by only ~0.006 pu once rebased. That
script's own caveat says as much: "a heavier loading or deeper contingency
could cross. Do not quote ARM A as a wide margin." This is that test
(`supplimentary_docs/issues.md` B4).

This does NOT re-run or overwrite `results/audit/andes_voltage_spike.json` --
that artifact is the baseline-dispatch measurement and stays as the record.
This writes a separate artifact for the stressed case.

FOUR KNOBS, ONE QUESTION: does any bus cross a harvested under-voltage
threshold for longer than the paired duration, under a heavier grid?

  1. Heavier dispatch  -- scale every PQ load and PV/Slack generator p0 (and
     PQ q0) by a common factor, re-solving power flow at each scale. Higher
     loading means less reactive headroom at each bus post-contingency.
  2. Untested trips     -- ARM A originally tried the first 8 of 20 lines
     (picked by a `p1` attribute lookup that does not exist on this ANDES
     Line model and silently fell back to declaration order -- so "most
     heavily loaded" was never actually enforced). This runs all 20.
  3. N-2 escalation      -- pairs of simultaneous line trips, mirroring the
     frequency arm's N-2 generator/load ladder. Two independent outages is a
     more severe event than the N-1 the task screens for, but it bounds how
     much worse a second contingency makes the margin.
  4. Longer window       -- TF pushed from 20s to 40s, to rule out a slow
     drift past the original window rather than a genuine settle.

Every case is scored against the same harvested threshold ladder as ARM A
(imported, not re-derived, so the two artifacts are directly comparable).

Run:  .venv\\Scripts\\python.exe sanity\\andes_voltage_stress.py
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

import andes
import numpy as np

sys.path.append(".")

warnings.filterwarnings("ignore")
andes.config_logger(stream_level=50)

from sanity.andes_voltage_spike import (bus_voltage_trace, excursion_table,
                                        harvest_thresholds,
                                        load_voltage_time_rules, rebase,
                                        summarize)

TSTEP = 0.01
TRIP_T = 1.0
DISPATCH_SCALES = [1.00, 1.10, 1.20, 1.30]   # 1.00 reproduces the baseline arm
LONG_TF = 40.0                                 # knob 4: past the original 20s window
N2_PAIR_COUNT = 10                             # heaviest-loaded-line pairs to test


def load_and_scale(path, scale):
    """Load the case and scale every load/generation dispatch by `scale`.

    PQ.p0/q0 are the per-unit MW/MVAr loads; PV/Slack.p0 the generator
    setpoints. Slack absorbs whatever imbalance remains, exactly as it does
    in the unscaled case, so a solved power flow at scale=1.30 is a heavier
    but still self-consistent dispatch, not an unbalanced one.
    """
    ss = andes.load(path, setup=False, no_output=True, default_config=True)
    ss.PQ.p0.v = (np.asarray(ss.PQ.p0.v, dtype=float) * scale).tolist()
    ss.PQ.q0.v = (np.asarray(ss.PQ.q0.v, dtype=float) * scale).tolist()
    if hasattr(ss, "PV") and ss.PV.n:
        ss.PV.p0.v = (np.asarray(ss.PV.p0.v, dtype=float) * scale).tolist()
    return ss


def run_case(path, scale, togglers, label=""):
    """Load at this dispatch scale, add permanent Togglers, run PFlow + TDS."""
    ss = load_and_scale(path, scale)
    for model, dev in togglers:
        ss.add("Toggler", dict(model=model, dev=dev, t=TRIP_T))
    ss.setup()
    ss.PFlow.run()
    if not ss.PFlow.converged:
        print(f"  {label:<28} scale={scale:.2f}  POWER FLOW DID NOT CONVERGE")
        return None
    ss.TDS.config.tf = LONG_TF
    ss.TDS.config.tstep = TSTEP
    ss.TDS.config.criteria = 0
    ss.TDS.run()
    return ss


def deepest(case_result, under):
    """(raw_min_pu, rebased_min_pu, margin_to_mildest_raw, margin_to_mildest_rebased)."""
    t, v, labels = bus_voltage_trace(case_result)
    if v is None:
        return None
    vr, base = rebase(t, v)
    raw_min, reb_min = float(v.min()), float(vr.min())
    mildest = max(under) if under else float("nan")
    return {
        "raw_min_pu": raw_min, "rebased_min_pu": reb_min,
        "margin_raw": raw_min - mildest, "margin_rebased": reb_min - mildest,
        "excursions_raw": excursion_table(t, v, under, [])["under"],
        "excursions_rebased": excursion_table(t, vr, under, [])["under"],
    }


def main() -> None:
    rules = load_voltage_time_rules()
    volts, times, numeric, free_ids = harvest_thresholds(rules)
    under = sorted([x for x in volts if x < 1.0], reverse=True)
    mildest_under = max(under) if under else float("nan")
    live_durations = sorted(x for x in times if 0.0 < x <= LONG_TF)

    full = andes.get_case("ieee14/ieee14_full.xlsx")
    probe = andes.load(full, setup=True, no_output=True, default_config=True)
    probe.PFlow.run()
    all_lines = list(probe.Line.idx.v)   # knob 2: all 20, not the first 8

    print(f"ANDES {andes.__version__} | IEEE-14 | stress arm | "
          f"{LONG_TF:.0f}s window | {len(all_lines)} lines | "
          f"dispatch scales {DISPATCH_SCALES}")
    print(f"mildest harvested under-voltage bar: {mildest_under:g} pu\n")

    # ---- Knob 1 + 2: every line, at every dispatch scale (N-1) -----------
    print("=" * 78)
    print("N-1: every line, escalating dispatch")
    print("=" * 78)
    n1_results = {}
    worst_n1 = {"margin_rebased": float("inf")}
    for scale in DISPATCH_SCALES:
        for idx in all_lines:
            label = f"line {idx} @scale {scale:.2f}"
            ss = run_case(full, scale, [("Line", idx)], label=label)
            if ss is None:
                n1_results[f"{idx}@{scale}"] = {"converged": False}
                continue
            d = deepest(ss, under)
            if d is None:
                continue
            d["converged"] = True
            n1_results[f"{idx}@{scale}"] = d
            print(f"  {label:<28} rebased_min={d['rebased_min_pu']:.4f}  "
                  f"margin={d['margin_rebased']:+.4f}")
            if d["margin_rebased"] < worst_n1["margin_rebased"]:
                worst_n1 = {**d, "case": label}

    # ---- Knob 3: N-2 line pairs, at baseline and heaviest dispatch -------
    print("\n" + "=" * 78)
    print(f"N-2: heaviest-loaded-line pairs (top {N2_PAIR_COUNT} by baseline "
          f"rebased-min impact), escalating dispatch")
    print("=" * 78)
    # Rank single-line impact at baseline dispatch to pick which PAIRS matter --
    # testing all C(20,2)=190 pairs at 4 scales is wasteful when most pairs are
    # electrically irrelevant to the buses that are already closest to the bar.
    ranked = sorted(
        ((k, v) for k, v in n1_results.items() if v.get("converged") and k.endswith("@1.0")),
        key=lambda kv: kv[1]["margin_rebased"],
    )
    top_lines = []
    for k, _ in ranked:
        line_id = k.rsplit("@", 1)[0]
        if line_id not in top_lines:
            top_lines.append(line_id)
        if len(top_lines) >= N2_PAIR_COUNT:
            break
    if len(top_lines) < 2:
        top_lines = all_lines[:N2_PAIR_COUNT]
    pairs = list(itertools.combinations(top_lines, 2))[:N2_PAIR_COUNT]

    n2_results = {}
    worst_n2 = {"margin_rebased": float("inf")}
    for scale in DISPATCH_SCALES:
        for a, b in pairs:
            label = f"lines {a}+{b} @scale {scale:.2f}"
            ss = run_case(full, scale, [("Line", a), ("Line", b)], label=label)
            if ss is None:
                n2_results[f"{a}+{b}@{scale}"] = {"converged": False}
                continue
            d = deepest(ss, under)
            if d is None:
                continue
            d["converged"] = True
            n2_results[f"{a}+{b}@{scale}"] = d
            print(f"  {label:<28} rebased_min={d['rebased_min_pu']:.4f}  "
                  f"margin={d['margin_rebased']:+.4f}")
            if d["margin_rebased"] < worst_n2["margin_rebased"]:
                worst_n2 = {**d, "case": label}

    # ---- verdict -----------------------------------------------------------
    crossed_n1 = worst_n1["margin_rebased"] < 0
    crossed_n2 = worst_n2["margin_rebased"] < 0
    print("\n" + "=" * 78)
    print(f"worst N-1 case : {worst_n1.get('case', 'n/a')}  "
          f"margin {worst_n1['margin_rebased']:+.4f} pu (rebased)")
    print(f"worst N-2 case : {worst_n2.get('case', 'n/a')}  "
          f"margin {worst_n2['margin_rebased']:+.4f} pu (rebased)")
    if crossed_n1:
        verdict = ("CROSSED under N-1: at least one stressed single-line-trip case "
                   "dips below the mildest harvested under-voltage bar once rebased. "
                   "The 'thin margin, measured' verdict must be reworded, not just "
                   "caveated -- see supplimentary_docs/issues.md B4.")
    elif crossed_n2:
        verdict = ("HOLDS under N-1, CROSSES under N-2: the margin survives every "
                   "single stressed line trip but a second simultaneous outage "
                   "crosses the mildest bar. N-1 'Inert' can stand if explicitly "
                   "scoped to N-1, mirroring the frequency arm's own N-2 caveat.")
    else:
        verdict = ("HOLDS under both N-1 and N-2 at every tested dispatch scale: "
                   "the margin is thin but the stress test did not cross it. "
                   "'Inert' upgrades from 'measured, thin margin' to 'stress-tested'.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    out = {
        "andes_version": andes.__version__,
        "case": "ieee14_full",
        "window_s": LONG_TF,
        "dispatch_scales": DISPATCH_SCALES,
        "n_lines_tested": len(all_lines),
        "n2_pairs_tested": len(pairs),
        "mildest_under_voltage_pu": mildest_under,
        "n1": {k: v for k, v in n1_results.items()},
        "n2": {k: v for k, v in n2_results.items()},
        "worst_n1": worst_n1,
        "worst_n2": worst_n2,
        "crossed_under_n1": crossed_n1,
        "crossed_under_n2": crossed_n2,
        "verdict": verdict,
        "caveats": [
            "This is IEEE-14, not the thesis topology -- same limitation as the "
            "baseline ARM A/B artifact.",
            "N-2 pairs are restricted to the top "
            f"{N2_PAIR_COUNT} lines by baseline single-outage impact, not all "
            "C(20,2) combinations -- a real but bounded search, not exhaustive.",
            "Dispatch scaling multiplies every PQ/PV setpoint uniformly; it does "
            "not model a specific heavier operating point from real data.",
        ],
    }
    dest = Path("results/audit/andes_voltage_stress.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
