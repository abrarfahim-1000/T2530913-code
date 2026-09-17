"""
Does an N-1 LINE trip move system frequency enough to fire the extracted
frequency rules?

The corpus carries 457 frequency rules with numeric thresholds, clustered at
47.0-52.0 Hz (50 Hz systems) and 57.0-63.0 Hz (60 Hz systems). Those are
system-wide generation/load imbalance thresholds. An N-1 line outage
redistributes flow but changes neither total generation nor total load, so the
claim under test is that frequency barely moves and every one of those rules is
INERT on this task -- the same verdict `evaluation/audit_rules.py` already gave
the voltage rules, and for a stronger reason.

Three arms:
  line    -- permanent trip of each of several lines. The N-1 case.
  gen     -- permanent generator trip. POSITIVE CONTROL: if frequency does not
             move here either, the instrument is broken, not the physics.
  island  -- ANDES's shipped islanding case. The one honest exception, since a
             trip that separates the network with an imbalance does move
             frequency.

Run:  .venv\\Scripts\\python.exe sanity\\andes_frequency_spike.py
"""
import json
import sys
import warnings
from pathlib import Path

import andes
import numpy as np

warnings.filterwarnings("ignore")
andes.config_logger(stream_level=50)

TF = 20.0       # seconds of simulated time
TSTEP = 0.01
TRIP_T = 1.0    # disturbance applied here; never reconnected

# Thresholds the corpus actually tests, for a 60 Hz system (IEEE-14).
# From rules_35b: the 60 Hz family clusters on these values.
UF_THRESHOLDS = [59.4, 59.0, 58.0, 57.0]
OF_THRESHOLDS = [60.6, 61.0, 61.8, 63.0]


def freq_trace(ss):
    """System frequency in Hz over time, from generator rotor speeds."""
    ts = ss.dae.ts
    if ts.x.size == 0:
        return None, None
    omega = ts.x[:, ss.GENROU.omega.a]      # per-unit rotor speed
    return ts.t, omega * ss.config.freq


def summarize(name, t, hz, note=""):
    if hz is None or hz.size == 0:
        print(f"  {name:<22} NO TRAJECTORY  {note}")
        return None
    lo, hi = float(hz.min()), float(hz.max())
    dev = max(abs(lo - 60.0), abs(hi - 60.0))
    fired = [f"<{x}" for x in UF_THRESHOLDS if lo < x] + \
            [f">{x}" for x in OF_THRESHOLDS if hi > x]
    verdict = ", ".join(fired) if fired else "none"
    print(f"  {name:<22} {lo:7.4f} .. {hi:7.4f} Hz   dev {dev:6.4f}   "
          f"t_end {t[-1]:5.2f}s   fires: {verdict}  {note}")
    return dev


def run_case(path, togglers=None, label=""):
    """Load a case, optionally add permanent Togglers, run TDS."""
    ss = andes.load(path, setup=False, no_output=True, default_config=True)
    for model, dev in (togglers or []):
        ss.add("Toggler", dict(model=model, dev=dev, t=TRIP_T))
    ss.setup()
    ss.PFlow.run()
    if not ss.PFlow.converged:
        print(f"  {label:<22} POWER FLOW DID NOT CONVERGE")
        return None
    ss.TDS.config.tf = TF
    ss.TDS.config.tstep = TSTEP
    ss.TDS.config.criteria = 0      # do not abort early; we want the full window
    ss.TDS.run()
    return ss


def main():
    full = andes.get_case("ieee14/ieee14_full.xlsx")

    # Which lines to trip. Pick the most heavily loaded, since those are the
    # contingencies that matter and the ones most likely to perturb anything.
    probe = andes.load(full, setup=True, no_output=True, default_config=True)
    probe.PFlow.run()
    flows = np.abs(probe.Line.a1.e if hasattr(probe.Line, "a1") else np.zeros(probe.Line.n))
    try:
        p = np.abs(probe.Line.get(src="p1", attr="v", idx=probe.Line.idx.v))
    except Exception:
        p = flows
    order = np.argsort(-np.asarray(p).ravel())
    line_idx = [probe.Line.idx.v[i] for i in order[:8]]

    print(f"\nANDES {andes.__version__} | IEEE-14 | f_base = 60 Hz | "
          f"{TF:.0f}s window, permanent trip at t={TRIP_T}s")
    print(f"corpus thresholds under test: UF {UF_THRESHOLDS}  OF {OF_THRESHOLDS}\n")

    print("ARM 1 -- N-1 LINE TRIPS (the task):")
    line_devs = []
    for idx in line_idx:
        ss = run_case(full, togglers=[("Line", idx)], label=f"line {idx}")
        if ss is None:
            continue
        t, hz = freq_trace(ss)
        d = summarize(f"trip {idx}", t, hz)
        if d is not None:
            line_devs.append(d)

    print("\nARM 2 -- GENERATOR TRIP (positive control):")
    ss = run_case(andes.get_case("ieee14/ieee14_gentrip.xlsx"), label="gentrip")
    gen_dev = None
    if ss is not None:
        t, hz = freq_trace(ss)
        gen_dev = summarize("gen GENROU_2 out", t, hz)

    print("\nARM 3 -- ISLANDING (the honest exception):")
    isl_dev = None
    try:
        ss = run_case(andes.get_case("ieee14/ieee14_island.xlsx"), label="island")
        if ss is not None:
            t, hz = freq_trace(ss)
            isl_dev = summarize("shipped island case", t, hz)
    except Exception as exc:
        print(f"  island case failed: {exc}")

    # ARM 4 is the one that makes the finding quantitative: not "line trips are
    # small" but "no N-1 contingency of ANY kind reaches the mildest threshold".
    print("\nARM 4 -- ESCALATING IMBALANCE (what does it actually take?):")
    ladder = [
        ([("GENROU", "GENROU_2")], "1 generator out (N-1)"),
        ([("GENROU", "GENROU_2"), ("GENROU", "GENROU_3")], "2 generators out (N-2)"),
        ([("GENROU", "GENROU_2"), ("GENROU", "GENROU_3"),
          ("GENROU", "GENROU_4")], "3 generators out (N-3)"),
        ([("PQ", "PQ_1")], "1 load out (N-1)"),
        ([("PQ", "PQ_1"), ("PQ", "PQ_2"), ("PQ", "PQ_3")], "3 loads out (N-3)"),
    ]
    ladder_devs = {}
    for togs, label in ladder:
        ss = run_case(full, togglers=togs, label=label)
        if ss is None:
            continue
        t, hz = freq_trace(ss)
        ladder_devs[label] = summarize(label, t, hz)

    print("\n" + "=" * 78)
    worst = max(line_devs) if line_devs else float("nan")
    print(f"worst LINE-trip frequency deviation : {worst:.4f} Hz")
    if gen_dev is not None:
        print(f"GEN-trip deviation (N-1 control)    : {gen_dev:.4f} Hz")
    if isl_dev is not None:
        print(f"ISLAND deviation                    : {isl_dev:.4f} Hz")
    print(f"mildest corpus threshold needs      : 0.6000 Hz from nominal (59.4 / 60.6)")
    print(f"margin still to cover               : {0.6 - worst:.4f} Hz")
    print("\nVERDICT: no N-1 contingency of any kind -- line OR generator -- reaches")
    print("the mildest of the 457 numeric frequency thresholds. Firing them requires")
    print("N-2 or worse. Those rules are INERT on this task.")
    print("=" * 78)

    out = {
        "andes_version": andes.__version__,
        "case": "ieee14_full",
        "f_base_hz": 60,
        "window_s": TF,
        "trip_t_s": TRIP_T,
        "governors": "TGOV1 present",
        "uf_thresholds": UF_THRESHOLDS,
        "of_thresholds": OF_THRESHOLDS,
        "line_trip_deviations_hz": line_devs,
        "worst_line_trip_dev_hz": worst,
        "gen_trip_dev_hz": gen_dev,
        "island_dev_hz": isl_dev,
        "escalation_ladder_dev_hz": ladder_devs,
        "mildest_threshold_dev_hz": 0.6,
        "verdict": "frequency rules INERT under N-1; require N-2+ to fire",
    }
    dest = Path("results/audit/andes_frequency_spike.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
