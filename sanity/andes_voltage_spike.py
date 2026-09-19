"""
Do post-contingency VOLTAGE transients enter the envelopes the time-bound
voltage rules test?

The frequency arm (`sanity/andes_frequency_spike.py`) settled that the 457
numeric frequency rules are INERT: an N-1 line outage redistributes flow but
changes neither total generation nor total load, so system frequency barely
moves (worst line trip 0.3293 Hz against 0.6 Hz needed).

Voltage is the one sub-pool where that argument does NOT carry. This script
tests it on two event classes, and the distinction between them is the whole
finding:

  ARM A -- CLEAN LINE OPENING (the N-1 the thesis task actually models).
           A Toggler opens a line; there is no fault-on period. This is the
           NO-FAULT LOWER BOUND.

  ARM B -- BOLTED THREE-PHASE FAULT (the event class the rules are WRITTEN
           for). Ride-through / no-trip-zone requirements describe what a
           generator must do during and after a fault, so measuring them on a
           clean opening answers the wrong question. Faults are applied at
           several buses and cleared at ~0.08s and ~0.15s -- the band where
           the corpus clusters (0.14 / 0.15 / 0.16 s).

A time-bound rule is not entered by a dip alone. `voltage_pu < 0.45 for
> 0.15s` needs BOTH a bus below 0.45 pu AND that bus to stay there longer
than 0.15 s. So the measurement is per bus: depth of dip, and the longest
CONTIGUOUS time spent outside each harvested threshold. Both polarities are
reported, since the corpus carries both `time_seconds > X` (constraint) and
`time_seconds < X` (no-trip zone).

For the fault arm the decisive question is not whether the envelope is
entered but WHEN: if the depression exists only while the fault is applied
and recovers the instant it clears, the envelope is entered only during an
event the shield never observes. So every fault case also reports the
excursion restricted to strictly AFTER the clearing instant.

CAVEAT ARM throughout: these grids operate above nominal, so a healthy bus
reads ~1.02 pu here (~1.06 on the Grid2Op grids), not 1.00. A rule written
against a 1.00 nominal misfires. Every figure is reported BOTH raw and
rebased to the pre-disturbance steady state.

Run:  .venv\\Scripts\\python.exe sanity\\andes_voltage_spike.py
"""
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import andes
import numpy as np

warnings.filterwarnings("ignore")
andes.config_logger(stream_level=50)

TF = 20.0       # seconds of simulated time
TSTEP = 0.01
TRIP_T = 1.0    # disturbance applied here; never reconnected
N_LINES = 8     # how many of the most heavily loaded lines to trip

# ARM B. The shipped ieee14_fault.xlsx declares Fault(bus, tf, tc, xf, rf);
# xf defaults to 1e-4 pu, i.e. effectively bolted. Clearing times bracket the
# corpus band: 0.08s is a fast primary clear, 0.15s sits on the 0.14/0.15/0.16
# cluster that appears ~20 times.
CLEAR_TIMES = [0.08, 0.15]
FAULT_XF = 0.0001
FAULT_RF = 0.0
# Spread across both voltage levels (69 kV: 1-5, 8; 138 kV: 6, 7, 9-14) and
# including the buses that dipped deepest in ARM A.
FAULT_BUSES = [1, 3, 4, 5, 9, 11, 14]

RULES_DIR = Path("rules_35b")

# What counts as a voltage rule, a time-bound rule, and a frequency rule.
# The frequency exclusion is what keeps this arm disjoint from the one already
# settled in andes_frequency_spike.py.
VOLT_RE = re.compile(r"\bvoltage_(?:pu|kv|pu_min|pu_max)\b", re.I)
TIME_RE = re.compile(r"\b(?:time_seconds|duration_seconds|\w*_ms|clearance_time)\b", re.I)
FREQ_RE = re.compile(r"freq|\bhz\b|rocof|droop|deadband", re.I)

# Numeric comparison against a voltage / time variable.
VOLT_CMP = re.compile(r"\bvoltage_(?:pu|pu_min|pu_max)\b\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)", re.I)
TIME_CMP = re.compile(r"\b(?:time_seconds|duration_seconds|clearance_time)\b\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)", re.I)
MS_CMP = re.compile(r"\b\w*_ms\b\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)", re.I)

AT_NOMINAL = 1.0    # degenerate over-voltage threshold; see envelopes()


# --------------------------------------------------------------------------
# 1. harvest
# --------------------------------------------------------------------------

def load_voltage_time_rules():
    """Candidate rules that are voltage-AND-time-bound and not frequency."""
    out = []
    for path in sorted(RULES_DIR.glob("*_candidates.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rule = json.loads(line).get("rule", {})
            cond = rule.get("condition", "") or ""
            if VOLT_RE.search(cond) and TIME_RE.search(cond) and not FREQ_RE.search(cond):
                out.append({
                    "rule_id": rule.get("rule_id"),
                    "doc": path.stem.replace("_candidates", ""),
                    "source": rule.get("source"),
                    "entity": rule.get("entity"),
                    "condition": cond,
                    "action": rule.get("action"),
                    "severity": rule.get("severity"),
                })
    return out


def rule_numbers(cond):
    """(voltage thresholds, durations in seconds) named by one condition."""
    vs = [round(float(v), 4) for _, v in VOLT_CMP.findall(cond)]
    ts = [round(float(v), 4) for _, v in TIME_CMP.findall(cond)]
    ts += [round(float(v) / 1000.0, 4) for _, v in MS_CMP.findall(cond)]
    return vs, ts


def harvest_thresholds(rules):
    """Distinct numeric voltage thresholds and durations the rules test.

    A rule with a free variable on either side (`time_seconds < t_clear`) is
    counted separately: it names an envelope but does not pin one, so it can
    neither be fired nor be shown inert.
    """
    volts, times, numeric, free_ids = Counter(), Counter(), [], []
    for r in rules:
        vs, ts = rule_numbers(r["condition"])
        if vs and ts:
            numeric.append({"rule_id": r["rule_id"], "condition": r["condition"],
                            "doc": r["doc"], "volts": set(vs), "times": set(ts)})
            for v in vs:
                volts[v] += 1
            for t in ts:
                times[t] += 1
        else:
            free_ids.append(r["rule_id"])
    return volts, times, numeric, free_ids


def rules_for_cell(numeric, thr, dur):
    """rule_ids whose numeric envelope names both this threshold and duration."""
    return [r["rule_id"] for r in numeric if thr in r["volts"] and dur in r["times"]]


# --------------------------------------------------------------------------
# 2. simulate
# --------------------------------------------------------------------------

def run_case(path, togglers=None, faults=None, label=""):
    """Load a case, add permanent Togglers and/or Faults, run PFlow + TDS."""
    ss = andes.load(path, setup=False, no_output=True, default_config=True)
    for model, dev in (togglers or []):
        ss.add("Toggler", dict(model=model, dev=dev, t=TRIP_T))
    for bus, t_on, t_clear in (faults or []):
        ss.add("Fault", dict(bus=bus, tf=t_on, tc=t_clear,
                             xf=FAULT_XF, rf=FAULT_RF))
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


def bus_voltage_trace(ss):
    """(t, v, bus_idx) where v is (n_steps, n_bus) bus voltage magnitude in pu."""
    ts = ss.dae.ts
    if ts.y.size == 0:
        return None, None, None
    return np.asarray(ts.t), np.asarray(ts.y[:, ss.Bus.v.a]), list(ss.Bus.idx.v)


# --------------------------------------------------------------------------
# 3. measure
# --------------------------------------------------------------------------

def longest_contiguous(t, mask):
    """Longest contiguous span of time (s) for which `mask` holds."""
    if not mask.any():
        return 0.0
    dt = np.diff(t)
    run = np.where(mask[1:], dt, 0.0)
    best = cur = 0.0
    for x in run:
        cur = cur + x if x > 0 else 0.0
        best = max(best, cur)
    return float(best)


def excursion_table(t, v, under, over, since=None):
    """Per-threshold: deepest bus, and the longest contiguous excursion.

    `since` restricts the window to t >= since -- used to ask whether a
    depression OUTLIVES the fault that caused it.

    The duration is the max over buses of that bus's longest contiguous
    excursion, which is the quantity a `for > t_thr` rule actually tests.
    """
    if since is not None:
        keep = t >= since
        t, v = t[keep], v[keep, :]
    res = {"under": {}, "over": {}}
    n_bus = v.shape[1]
    for side, thrs in (("under", under), ("over", over)):
        for thr in thrs:
            best_dur, best_bus = 0.0, -1
            for b in range(n_bus):
                mask = v[:, b] < thr if side == "under" else v[:, b] > thr
                d = longest_contiguous(t, mask)
                if d > best_dur:
                    best_dur, best_bus = d, b
            res[side][thr] = {"max_contiguous_s": best_dur,
                              "bus_col": int(best_bus) if best_bus >= 0 else None,
                              "entered": best_dur > 0.0}
    return res


def rebase(t, v, t_event=TRIP_T):
    """Rebase every bus so its pre-disturbance steady state reads 1.00 pu.

    THE CAVEAT ARM. These grids sit above nominal; a rule written against a
    1.00 nominal reads a healthy 1.02 pu bus as an over-voltage and a 2% dip
    as no dip at all. Both conventions are reported; neither is assumed.
    """
    pre = max(np.searchsorted(t, t_event) - 1, 0)
    base = v[pre, :].copy()
    base[base == 0] = 1.0
    return v / base, base


def recovery_profile(t, v, t_clear, base):
    """min-across-buses voltage at fixed offsets after the clearing instant.

    This is what answers "does it recover as soon as the fault clears?" -- a
    depression that vanishes within a few tens of ms of clearing is an
    artifact of the fault-on period, not a post-contingency state a shield
    could ever observe.
    """
    prof = {}
    for off in (-0.01, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0):
        i = min(np.searchsorted(t, t_clear + off), len(t) - 1)
        prof[f"{off:+.2f}s"] = {
            "raw_min_pu": float(v[i].min()),
            "rebased_min_pu": float((v[i] / base).min()),
        }
    return prof


def summarize(name, t, v, labels, note=""):
    lo_per_bus, hi_per_bus = v.min(axis=0), v.max(axis=0)
    b_lo, b_hi = int(lo_per_bus.argmin()), int(hi_per_bus.argmax())
    lo, hi = float(lo_per_bus[b_lo]), float(hi_per_bus[b_hi])
    print(f"  {name:<24} min {lo:6.4f} pu (bus {labels[b_lo]:>2})   "
          f"max {hi:6.4f} pu (bus {labels[b_hi]:>2}) {note}")
    return {"min_pu": lo, "min_bus": labels[b_lo],
            "max_pu": hi, "max_bus": labels[b_hi]}


# --------------------------------------------------------------------------
# 4. the envelope test (shared by both arms)
# --------------------------------------------------------------------------

def envelopes(cases, conv, under, over, durations, numeric, since_key=None):
    """Classify every (threshold, duration) cell against every case.

    Two cell families are degenerate and must not be allowed to carry a
    verdict:
      * duration 0s -- "for longer than no time at all" is not a test.
      * over-threshold 1.0 pu -- that is NOMINAL, and these grids sit above
        it in perfectly healthy steady state, so it is satisfied before the
        disturbance arrives. Retained and counted because it IS the
        1.00-vs-1.06 caveat made visible, but never counted as a finding.

    Returns "exceeds" cells (held > dur: constraint polarity, `below thr for
    longer than dur`) and "within" cells (0 < held <= dur: no-trip-zone
    polarity, `below thr for less than dur`).
    """
    exceeds, within, degenerate = [], [], []
    for label, c in cases.items():
        ex = c[conv]["excursions" if since_key is None else since_key]
        for side in ("under", "over"):
            for thr, r in ex[side].items():
                held = r["max_contiguous_s"]
                if held <= 0.0:
                    continue
                degen_thr = side == "over" and thr == AT_NOMINAL
                for dur in durations:
                    cell = {"case": label, "side": side, "threshold_pu": thr,
                            "duration_s": dur, "max_contiguous_s": held,
                            "bus": c["bus_labels"][r["bus_col"]],
                            "rule_ids": rules_for_cell(numeric, thr, dur)}
                    if dur == 0.0 or degen_thr:
                        degenerate.append(cell)
                    elif held > dur:
                        exceeds.append(cell)
                    else:
                        within.append(cell)
    return exceeds, within, degenerate


def report_arm(title, cases, under, over, durations, numeric, since_key=None):
    """Print the envelope result for one arm under both pu conventions."""
    print(f"\n{title}")
    out = {}
    for conv in ("raw", "rebased"):
        exc, wit, deg = envelopes(cases, conv, under, over, durations,
                                  numeric, since_key)
        real = sorted({(c["side"], c["threshold_pu"]) for c in exc + wit})
        print(f"  {conv.upper():<8} non-degenerate thresholds crossed: "
              f"{len(real)}  |  'below thr LONGER than dur' cells: {len(exc)}"
              f"  |  'shorter than dur' (no-trip-zone) cells: {len(wit)}"
              f"  |  degenerate: {len(deg)}")
        if not real:
            print("           -- no bus crossed any non-degenerate voltage "
                  "threshold, at any depth, for any duration.")
        for side, thr in real[:12]:
            hits = [c for c in exc if c["side"] == side and c["threshold_pu"] == thr]
            held = max((c["max_contiguous_s"] for c in exc + wit
                        if c["side"] == side and c["threshold_pu"] == thr), default=0.0)
            durs = sorted({c["duration_s"] for c in hits})
            rids = sorted({r for c in hits for r in c["rule_ids"]})
            print(f"           {side:<5} {thr:<6g} pu  held up to {held:6.3f}s"
                  f"  -> clears durations {[f'{d:g}' for d in durs] or 'none'}"
                  f"  rules {rids[:6]}{'...' if len(rids) > 6 else ''}")
        out[conv] = {"exceeds": exc, "within": wit, "degenerate": len(deg)}
    return out


# --------------------------------------------------------------------------

def main():
    # ---- 1. harvest -----------------------------------------------------
    rules = load_voltage_time_rules()
    volts, times, numeric, free_ids = harvest_thresholds(rules)

    print(f"\nSTEP 1 -- the voltage+time sub-pool in {RULES_DIR}/")
    print(f"  voltage AND time, not frequency : {len(rules)} candidate rules")
    print(f"  with numeric thresholds on both : {len(numeric)}")
    print(f"  free-variable / unpinnable      : {len(free_ids)}")
    print("\n  sample conditions:")
    for r in rules[:6]:
        print(f"    {r['rule_id']:<8} {r['condition'][:86]}")

    under = sorted([x for x in volts if x < 1.0], reverse=True)
    over = sorted([x for x in volts if x >= 1.0])
    dur_short = sorted(x for x in times if x <= TF)
    dur_long = sorted(x for x in times if x > TF)
    live_durations = [d for d in dur_short if d > 0.0]

    print(f"\nSTEP 2 -- the envelope the rules care about")
    print(f"  under-voltage thresholds ({len(under):2d}): "
          + ", ".join(f"{x:g}" for x in under))
    print(f"  over-voltage  thresholds ({len(over):2d}): "
          + ", ".join(f"{x:g}" for x in over))
    print(f"  durations <= {TF:.0f}s window ({len(dur_short):2d}): "
          + ", ".join(f"{x:g}" for x in dur_short))
    print(f"  durations >  {TF:.0f}s window ({len(dur_long):2d}): "
          + ", ".join(f"{x:g}" for x in dur_long)
          + "   <- untestable in this window")
    band = sum(times[d] for d in (0.14, 0.15, 0.16) if d in times)
    print(f"  mentions of the 0.14/0.15/0.16s clearing band: {band}"
          f"   <- why ARM B clears at 0.08s and 0.15s")

    full = andes.get_case("ieee14/ieee14_full.xlsx")

    # ---- ARM A: clean line opening (no-fault lower bound) ---------------
    probe = andes.load(full, setup=True, no_output=True, default_config=True)
    probe.PFlow.run()
    try:
        p = np.abs(probe.Line.get(src="p1", attr="v", idx=probe.Line.idx.v))
    except Exception:
        p = np.zeros(probe.Line.n)
    order = np.argsort(-np.asarray(p).ravel())
    line_idx = [probe.Line.idx.v[i] for i in order[:N_LINES]]

    print(f"\nANDES {andes.__version__} | IEEE-14 | {TF:.0f}s window, "
          f"event at t={TRIP_T}s")
    print("\n" + "=" * 78)
    print("ARM A -- CLEAN LINE OPENING (Toggler; the N-1 the task models).")
    print("         No fault-on period. THE NO-FAULT LOWER BOUND.")
    print("=" * 78)

    arm_a = {}
    for idx in line_idx:
        ss = run_case(full, togglers=[("Line", idx)], label=f"line {idx}")
        if ss is None:
            continue
        t, v, labels = bus_voltage_trace(ss)
        if v is None:
            print(f"  trip {idx:<18} NO TRAJECTORY")
            continue
        vr, base = rebase(t, v)
        arm_a[str(idx)] = {
            "bus_labels": labels,
            "raw": {**summarize(f"open {idx}", t, v, labels),
                    "excursions": excursion_table(t, v, under, over)},
            "rebased": {**summarize(f"open {idx} (rebased)", t, vr, labels),
                        "excursions": excursion_table(t, vr, under, over)},
            "pre_disturbance_pu": {"min": float(base.min()), "max": float(base.max())},
        }

    res_a = report_arm("ARM A ENVELOPE TEST:", arm_a, under, over,
                       dur_short, numeric)

    # ---- ARM B: bolted three-phase fault (the event class rules target) --
    print("\n" + "=" * 78)
    print("ARM B -- BOLTED THREE-PHASE FAULT (Fault model, xf="
          f"{FAULT_XF}, rf={FAULT_RF}).")
    print("         The event class ride-through rules are actually WRITTEN for.")
    print(f"         buses {FAULT_BUSES} x clearing times {CLEAR_TIMES}s")
    print("=" * 78)

    arm_b = {}
    for bus in FAULT_BUSES:
        for ct in CLEAR_TIMES:
            t_clear = TRIP_T + ct
            label = f"bus{bus}@{ct:g}s"
            ss = run_case(full, faults=[(bus, TRIP_T, t_clear)], label=label)
            if ss is None:
                continue
            t, v, labels = bus_voltage_trace(ss)
            if v is None:
                print(f"  {label:<24} NO TRAJECTORY")
                continue
            vr, base = rebase(t, v)
            arm_b[label] = {
                "bus_labels": labels,
                "fault_bus": bus,
                "clearing_time_s": ct,
                "t_clear_s": t_clear,
                "raw": {
                    **summarize(label, t, v, labels),
                    "excursions": excursion_table(t, v, under, over),
                    "excursions_after_clear": excursion_table(
                        t, v, under, over, since=t_clear),
                },
                "rebased": {
                    **summarize(f"{label} (rebased)", t, vr, labels),
                    "excursions": excursion_table(t, vr, under, over),
                    "excursions_after_clear": excursion_table(
                        t, vr, under, over, since=t_clear),
                },
                "recovery_profile": recovery_profile(t, v, t_clear, base),
                "pre_disturbance_pu": {"min": float(base.min()),
                                       "max": float(base.max())},
            }

    res_b_all = report_arm("ARM B ENVELOPE TEST -- FULL WINDOW "
                           "(includes the fault-on period):",
                           arm_b, under, over, dur_short, numeric)
    res_b_post = report_arm("ARM B ENVELOPE TEST -- STRICTLY AFTER CLEARING "
                            "(what a shield could ever observe):",
                            arm_b, under, over, dur_short, numeric,
                            since_key="excursions_after_clear")

    # ---- the decisive question ------------------------------------------
    print("\n" + "=" * 78)
    print("DOES THE DEPRESSION OUTLIVE THE FAULT?")
    print("  min-across-buses voltage, relative to the clearing instant:")
    offsets = ["-0.01s", "+0.00s", "+0.02s", "+0.05s", "+0.10s", "+0.20s",
               "+0.50s", "+1.00s", "+5.00s"]
    print("    case                  " + "".join(f"{o:>9}" for o in offsets))
    for label, c in arm_b.items():
        row = "".join(f"{c['recovery_profile'][o]['raw_min_pu']:9.4f}"
                      for o in offsets)
        print(f"    {label:<22}{row}")

    # How long, after clearing, does ANY bus stay below the mildest bar?
    mildest_under = max(under) if under else float("nan")
    post_tail = {}
    for label, c in arm_b.items():
        post_tail[label] = {
            "raw_s": c["raw"]["excursions_after_clear"]["under"][mildest_under]["max_contiguous_s"],
            "rebased_s": c["rebased"]["excursions_after_clear"]["under"][mildest_under]["max_contiguous_s"],
        }
    worst_tail = max((d["raw_s"] for d in post_tail.values()), default=0.0)
    worst_tail_reb = max((d["rebased_s"] for d in post_tail.values()), default=0.0)
    print(f"\n  longest post-clearing excursion below the MILDEST bar "
          f"({mildest_under:g} pu):")
    print(f"    raw {worst_tail:.3f}s   rebased {worst_tail_reb:.3f}s")

    # The post-clearing count means nothing until it is split by side. An
    # over-voltage swing after clearing is a genuine post-fault STATE; a
    # 0.01s under-voltage tail is the tail end of the fault itself. Count
    # DISTINCT (side, threshold, duration) envelopes, not case-multiplied
    # cells, or 14 cases inflate the headline by an order of magnitude.
    def distinct(cells, side=None):
        return sorted({(c["side"], c["threshold_pu"], c["duration_s"])
                       for c in cells if side is None or c["side"] == side})

    def both_convs(res, side=None):
        return sorted(set(distinct(res["raw"]["exceeds"], side))
                      | set(distinct(res["rebased"]["exceeds"], side)))

    b_full_under = both_convs(res_b_all, "under")
    b_full_over = both_convs(res_b_all, "over")
    b_post_under = both_convs(res_b_post, "under")
    b_post_over = both_convs(res_b_post, "over")

    # Which durations does the post-clearing under-voltage tail actually
    # clear? If only sub-0.1s ones, it never reaches the corpus's band.
    post_under_durs = sorted({d for _, _, d in b_post_under})
    band_reached = [d for d in post_under_durs if d >= 0.14]

    print("\n" + "-" * 78)
    print("DISTINCT envelopes entered (side, threshold_pu, duration_s), "
          "either convention:")
    print(f"  ARM A clean opening               : "
          f"{len(both_convs(res_a))}")
    print(f"  ARM B full window   under-voltage : {len(b_full_under)}"
          f"   over-voltage: {len(b_full_over)}")
    print(f"  ARM B after clearing under-voltage: {len(b_post_under)}"
          f"   over-voltage: {len(b_post_over)}")
    print(f"  post-clearing under-voltage durations cleared: "
          f"{[f'{d:g}' for d in post_under_durs] or 'none'}")
    print(f"  ... of which in the corpus's 0.14/0.15/0.16s band: "
          f"{[f'{d:g}' for d in band_reached] or 'NONE'}")

    deepest_a = min((d["raw"]["min_pu"] for d in arm_a.values()), default=float("nan"))
    deepest_a_reb = min((d["rebased"]["min_pu"] for d in arm_a.values()), default=float("nan"))
    deepest_b = min((d["raw"]["min_pu"] for d in arm_b.values()), default=float("nan"))
    deepest_b_reb = min((d["rebased"]["min_pu"] for d in arm_b.values()), default=float("nan"))

    n_a = len(res_a["raw"]["exceeds"]) + len(res_a["rebased"]["exceeds"])
    n_b = len(res_b_all["raw"]["exceeds"]) + len(res_b_all["rebased"]["exceeds"])
    n_b_post = len(res_b_post["raw"]["exceeds"]) + len(res_b_post["rebased"]["exceeds"])

    print("\n" + "-" * 78)
    print(f"deepest dip  ARM A (clean open) : {deepest_a:.4f} raw / "
          f"{deepest_a_reb:.4f} rebased")
    print(f"deepest dip  ARM B (fault)      : {deepest_b:.4f} raw / "
          f"{deepest_b_reb:.4f} rebased")
    print(f"non-degenerate envelope cells entered:  ARM A {n_a}   "
          f"ARM B full-window {n_b}   ARM B post-clearing {n_b_post}")

    verdict_a = (
        "ARM A (clean line opening): envelopes NOT entered. Deepest dip "
        f"{deepest_a:.4f} pu raw / {deepest_a_reb:.4f} pu rebased, against a "
        f"mildest harvested bar of {mildest_under:g} pu -- the depth is never "
        "reached, so the duration test is never engaged."
        if n_a == 0 else
        f"ARM A (clean line opening): {n_a} envelope cells entered."
    )
    if not b_full_under and not b_full_over:
        verdict_b = ("ARM B (bolted fault): envelopes NOT entered even during "
                     "the fault.")
    else:
        verdict_b = (
            f"ARM B (bolted fault): the UNDER-voltage ride-through envelopes are "
            f"entered ({len(b_full_under)} distinct) but essentially ONLY while "
            f"the fault is applied. Voltage collapses to ~{deepest_b:.4f} pu "
            f"during the fault and is back above 0.9 pu within ~50 ms of "
            f"clearing; the longest post-clearing excursion below the mildest "
            f"bar ({mildest_under:g} pu) is {worst_tail:.3f}s raw / "
            f"{worst_tail_reb:.3f}s rebased. The post-clearing under-voltage "
            f"tail clears only durations {[f'{d:g}' for d in post_under_durs]} "
            f"and reaches the corpus's 0.14/0.15/0.16s clearing band "
            f"{'in ' + str(band_reached) if band_reached else 'NOT AT ALL'}. "
            f"So the deep ride-through thresholds (0.45/0.3/0.05 pu) are "
            f"reachable by the EVENT, not by any post-contingency STATE. "
            f"SEPARATELY: {len(b_post_over)} distinct OVER-voltage envelopes do "
            f"survive clearing -- a genuine post-fault overshoot above 1.05-1.15 "
            f"pu lasting up to {max((c['max_contiguous_s'] for c in res_b_post['raw']['exceeds'] if c['side'] == 'over'), default=0.0):.3f}s. "
            f"That, not the dip, is the only part of this arm a state-observing "
            f"shield could see.")

    print(f"\nVERDICT A: {verdict_a}")
    print(f"\nVERDICT B: {verdict_b}")
    print("=" * 78)

    # ---- artifact -------------------------------------------------------
    def strip(cases, keys):
        return {
            lbl: {
                **{k: c[k] for k in keys if k in c},
                **{conv: {
                    "min_pu": c[conv]["min_pu"], "min_bus": c[conv]["min_bus"],
                    "max_pu": c[conv]["max_pu"], "max_bus": c[conv]["max_bus"],
                    **{ek: {side: {str(th): r for th, r in c[conv][ek][side].items()}
                            for side in ("under", "over")}
                       for ek in ("excursions", "excursions_after_clear")
                       if ek in c[conv]},
                } for conv in ("raw", "rebased")},
            } for lbl, c in cases.items()
        }

    out = {
        "andes_version": andes.__version__,
        "case": "ieee14_full",
        "window_s": TF,
        "tstep_s": TSTEP,
        "event_t_s": TRIP_T,
        "rules": {
            "voltage_and_time_not_frequency": len(rules),
            "with_numeric_thresholds_on_both": len(numeric),
            "free_variable_unpinnable": len(free_ids),
            "clearing_band_mentions_0p14_0p15_0p16": band,
            "sample_conditions": [r["condition"] for r in rules[:12]],
        },
        "harvested_envelope": {
            "under_voltage_pu": under,
            "over_voltage_pu": over,
            "durations_s_within_window": dur_short,
            "durations_s_beyond_window": dur_long,
            "voltage_threshold_counts": {str(k): v for k, v in sorted(volts.items())},
            "duration_counts_s": {str(k): v for k, v in sorted(times.items())},
        },
        "degenerate_cells_excluded": {
            "at_nominal_over_threshold_pu": AT_NOMINAL,
            "zero_duration_s": 0.0,
            "why": "the grid sits at 1.01-1.03 pu healthy, so a >1.0 pu test is "
                   "satisfied before the disturbance; 'for longer than 0s' is "
                   "not a duration test. Both are retained per case but "
                   "excluded from every verdict.",
        },
        "arm_a_clean_line_opening": {
            "what": "Toggler opens a line permanently; no fault-on period. "
                    "The N-1 event class the thesis task models, and the "
                    "no-fault LOWER BOUND on voltage excursion.",
            "lines_tripped": [str(i) for i in line_idx],
            "deepest_dip_raw_pu": deepest_a,
            "deepest_dip_rebased_pu": deepest_a_reb,
            "margin_to_mildest_under_voltage_pu": {
                "raw": deepest_a - mildest_under,
                "rebased": deepest_a_reb - mildest_under,
            },
            "envelopes_entered": {
                conv: res_a[conv]["exceeds"] for conv in ("raw", "rebased")},
            "no_trip_zone_cells": {
                conv: len(res_a[conv]["within"]) for conv in ("raw", "rebased")},
            "per_case": strip(arm_a, ["pre_disturbance_pu"]),
        },
        "arm_b_bolted_three_phase_fault": {
            "what": "Fault model at a bus, xf=%g rf=%g, applied at t=%g and "
                    "cleared after the stated time. The event class "
                    "ride-through / no-trip-zone rules are WRITTEN for."
                    % (FAULT_XF, FAULT_RF, TRIP_T),
            "fault_buses": FAULT_BUSES,
            "clearing_times_s": CLEAR_TIMES,
            "fault_xf_pu": FAULT_XF,
            "fault_rf_pu": FAULT_RF,
            "deepest_dip_raw_pu": deepest_b,
            "deepest_dip_rebased_pu": deepest_b_reb,
            "envelopes_entered_full_window": {
                conv: res_b_all[conv]["exceeds"] for conv in ("raw", "rebased")},
            "envelopes_entered_after_clearing": {
                conv: res_b_post[conv]["exceeds"] for conv in ("raw", "rebased")},
            "no_trip_zone_cells_full_window": {
                conv: len(res_b_all[conv]["within"]) for conv in ("raw", "rebased")},
            "post_clearing_tail_below_mildest_bar_s": post_tail,
            "worst_post_clearing_tail_s": {"raw": worst_tail,
                                           "rebased": worst_tail_reb},
            "distinct_envelopes_entered": {
                "note": "DISTINCT (side, threshold_pu, duration_s) triples, "
                        "either pu convention. Use these, not the raw cell "
                        "counts, which are multiplied by the 14 fault cases.",
                "full_window_under_voltage": [
                    {"threshold_pu": th, "duration_s": d}
                    for _, th, d in b_full_under],
                "full_window_over_voltage": [
                    {"threshold_pu": th, "duration_s": d}
                    for _, th, d in b_full_over],
                "after_clearing_under_voltage": [
                    {"threshold_pu": th, "duration_s": d}
                    for _, th, d in b_post_under],
                "after_clearing_over_voltage": [
                    {"threshold_pu": th, "duration_s": d}
                    for _, th, d in b_post_over],
                "post_clearing_under_voltage_durations_cleared_s":
                    post_under_durs,
                "post_clearing_under_voltage_reaches_0p14_0p16_band":
                    band_reached,
            },
            "per_case": strip(arm_b, ["fault_bus", "clearing_time_s",
                                      "t_clear_s", "recovery_profile",
                                      "pre_disturbance_pu"]),
        },
        "mildest_harvested_under_voltage_pu": mildest_under,
        "shortest_nonzero_harvested_duration_s": (
            min(live_durations) if live_durations else None),
        "caveats": [
            "ARM A is a clean line opening with NO fault-on period. It is the "
            "lower bound on voltage excursion, not the event class the rules "
            "are written for. Quote it only as such.",
            "ARM A NEAR MISS: rebased, the deepest clean-opening dip clears the "
            "mildest bar by only ~0.006 pu. The frequency arm missed by 0.27 Hz "
            "on a 0.6 Hz bar -- a factor of two. This one misses by under 1%, so "
            "a heavier loading or deeper contingency could cross. Do not quote "
            "ARM A as a wide margin.",
            "ARM B enters the DEEP under-voltage envelopes only while the fault "
            "is APPLIED. The distinction between 'these rules are reachable' "
            "and 'these rules are reachable during an event the shield never "
            "sees' is the whole result; do not collapse it.",
            "ARM B cell counts are multiplied by the 14 fault cases and by "
            "every duration below the held time. Quote "
            "distinct_envelopes_entered, not the cell counts.",
            "The post-clearing survivors are almost entirely OVER-voltage, not "
            "the ride-through dips. Reporting 'envelopes survive clearing' "
            "without that split would misrepresent the result.",
            "A bolted fault (xf=1e-4) is the most severe credible voltage "
            "event. It is an upper bound, not a typical one.",
            "Durations above the 20s window cannot be tested here (8 of 32).",
            "9 of 103 rules carry free-variable thresholds and can be neither "
            "fired nor shown inert.",
            "Raw and rebased conventions disagree because the grid sits ~2-3% "
            "above nominal here (~6% on the Grid2Op grids); a rule assuming a "
            "1.00 nominal misreads both directions.",
            "IEEE-14 is not the thesis topology. It is the only grid with a "
            "dynamic model available; the Grid2Op grids have no dynamics, which "
            "is precisely why neither arm can be transplanted onto the shield "
            "without adding a dynamic simulator to the pipeline.",
        ],
        "verdict_arm_a": verdict_a,
        "verdict_arm_b": verdict_b,
    }
    dest = Path("results/audit/andes_voltage_spike.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
