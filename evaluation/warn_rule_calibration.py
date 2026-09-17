"""
P(N-1 violation | this rule fires) -- the calibration every WARN rule must carry.

`evaluation/readmit_rules.py` re-admits 25 rules to a WARN channel on the
strength of their CLASSIFY-label fire rates: the rule separates `normal` frames
from `overload`/`line_trip`/`cascade` frames. That is a statement about the
retired 4-class target, not about the task the shield actually gates. A warning
phrased as "this could happen" on the N-1 task has to be measured against the
N-1 label, or it is an alarm wearing a citation.

This is `evaluation/loading_band_calibration.py` generalized from `rho_max`
bands to arbitrary rule conditions: stream every frame of all three datasets,
evaluate the condition on the base-case context, and split the per-line N-1
labels by whether the rule fired.

    P(violation | fires)    the number the warning text is allowed to quote
    P(violation | silent)   the counterfactual it has to beat
    discrimination          P|fire - P|silent. THIS is the whole claim.

`loading_band_calibration.py` reports lift against the BASE rate, which is right
for a partition into seven bands. It is wrong for a binary predicate, and the
first full run showed why: the power-factor rule fires on 97.5% of wcci2022
frames at P|fire 0.242 against P|silent 0.478, and lift-against-base still reads
-0.006, because a predicate that fires on nearly everything IS the base rate.
Measured against its own silent arm it is -0.236 -- firing means materially LOWER
risk. Both are reported; the verdict is taken on the discrimination.

The effective sample size is FRAMES, not contingencies. Lines within one frame
share a grid state, so 59 contingencies from one frame are not 59 independent
draws. Frame counts are printed beside every rate for that reason, and a rule
firing on too few frames is reported INSUFFICIENT rather than given a rate.

A rule whose lift is <= 0 is not a weak warning -- it is an inverted one. Firing
is then evidence of LOWER risk, and no phrasing fixes that. Those are reported,
not hidden, because shipping one would be the single thing in the four-channel
design a reviewer could fairly call noise.

Run:  .venv\\Scripts\\python.exe evaluation\\warn_rule_calibration.py
      .venv\\Scripts\\python.exe evaluation\\warn_rule_calibration.py --channel NORMAL
      .venv\\Scripts\\python.exe evaluation\\warn_rule_calibration.py --max-frames 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shield.context import build_context, load_base_kv  # noqa: E402
from shield.evaluator import Verdict, evaluate_condition  # noqa: E402

TAGS = ("neurips2020", "case14", "wcci2022")

# Below this many firing frames a rate is not reported as a rate. 30 is not a
# significance test; it is the point below which the trailing digits are theatre.
MIN_FRAMES = 30
# Matches evaluation/audit_rules.py MIN_DISCRIMINATION, deliberately: the bar for
# "carries information" should not move because the label changed.
MIN_LIFT = 0.05


def condition_groups(readmission: Path, channel: str) -> "OrderedDict[str, list[dict]]":
    """Group the channel's rules by condition -- the unit that actually differs.

    25 WARN records are 11 distinct predicates restated across standards, and
    calibrating the same predicate 25 times would report one measurement as
    though it were 25. Rules are carried alongside so every record can be cited
    with the rate its predicate earned.
    """
    rows = json.loads(readmission.read_text(encoding="utf-8"))["rules"]
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        if r.get("channel") != channel:
            continue
        cond = (r.get("condition") or "").strip()
        if not cond:
            continue
        groups.setdefault(cond, []).append(r)
    return groups


def blank(conditions) -> dict:
    return {c: {"frames_fired": 0, "frames_silent": 0, "frames_unevaluable": 0,
                "cont_fired": 0, "vio_fired": 0,
                "cont_silent": 0, "vio_silent": 0} for c in conditions}


def scan(tag: str, conditions: list[str], data_dir: Path, max_frames) -> dict:
    """One pass over a dataset, accumulating per-condition contingency counts."""
    path = data_dir / f"grid_dataset_{tag}_n1.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    base_kv = load_base_kv(tag, data_dir=str(data_dir))

    acc = blank(conditions)
    n_frames = 0
    n_skipped = 0

    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_frames is not None and i >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            y = np.asarray(rec.get("n1_violation", []), dtype=int)
            if y.size == 0:
                continue
            mask = y >= 0                     # -1 == not evaluated; never a negative
            if not mask.any():
                continue
            n_cont = int(mask.sum())
            n_vio = int((y[mask] == 1).sum())

            try:
                ctx = build_context(rec, base_kv)
            except (KeyError, ValueError):
                n_skipped += 1
                continue
            n_frames += 1

            for cond in conditions:
                a = acc[cond]
                verdict = evaluate_condition(cond, dict(ctx))
                if verdict is Verdict.VIOLATED:
                    a["frames_fired"] += 1
                    a["cont_fired"] += n_cont
                    a["vio_fired"] += n_vio
                elif verdict is Verdict.SATISFIED:
                    a["frames_silent"] += 1
                    a["cont_silent"] += n_cont
                    a["vio_silent"] += n_vio
                else:
                    # NOT_EVALUABLE / ERROR: excluded from BOTH arms. Counting an
                    # unevaluable frame as silent would credit the rule with a
                    # correct quiet it never actually made.
                    a["frames_unevaluable"] += 1

    return {"n_frames": n_frames, "n_skipped": n_skipped, "acc": acc}


def verdict_of(frames_fired: int, frames_silent: int, disc) -> str:
    """Verdict on the DISCRIMINATION, not on the lift against the base rate.

    Both arms need frames: a predicate that fires on every frame has no silent
    arm to be compared with, and "it is always true" is a statement about the
    grid's operating point, not about risk.
    """
    if disc is None or frames_fired < MIN_FRAMES or frames_silent < MIN_FRAMES:
        return "INSUFFICIENT"
    if disc >= MIN_LIFT:
        return "ELEVATED"
    if disc <= -MIN_LIFT:
        return "INVERTED"
    return "NEUTRAL"


def stratified(per_grid_rates: dict, key: str):
    """Pooled lift with the grid held fixed. NOT the lift of the pooled counts.

    The naive pooled lift is a Simpson trap here, and it fired on the first
    smoke run: `voltage_pu_min <= 0.90 or voltage_pu_max >= 1.10` fires on 100%
    of case14 frames and 0% of the other two. Its within-grid lift on case14 is
    exactly 0.000 -- it fires always, so it separates nothing -- yet pooling
    scores it +0.067, because case14's base violation rate (0.300) is higher
    than neurips2020's (0.177) and the predicate is really just selecting the
    grid. A warning may not be justified by which topology it is running on.

    Each grid's own lift is therefore weighted by the contingencies it
    contributed, and grids where the predicate never fires contribute nothing
    instead of contributing their base rate.
    """
    num = den = 0.0
    for r in per_grid_rates.values():
        # A grid that did not earn a verdict of its own does not get a vote. Three
        # firing frames on wcci2022 are three grid states, not 558 draws, and
        # weighting them by line count would let the smallest sample dominate the
        # largest -- the exact frames-vs-contingencies error this file warns about.
        if r[key] is None or r["verdict"] == "INSUFFICIENT":
            continue
        w = r["frames_fired"]           # frames, not contingencies
        if not w:
            continue
        num += r[key] * w
        den += w
    if not den:
        return None
    return num / den


def rates(a: dict) -> dict:
    """Turn raw counts into the three numbers the warning text may quote."""
    cf, cs = a["cont_fired"], a["cont_silent"]
    p_fire = (a["vio_fired"] / cf) if cf else None
    p_silent = (a["vio_silent"] / cs) if cs else None
    tot_c = cf + cs
    base = ((a["vio_fired"] + a["vio_silent"]) / tot_c) if tot_c else None
    lift = (p_fire - base) if (p_fire is not None and base is not None) else None
    disc = (p_fire - p_silent) if (p_fire is not None and p_silent is not None) else None
    ev = a["frames_fired"] + a["frames_silent"]
    return {
        "frames_fired": a["frames_fired"],
        "frames_silent": a["frames_silent"],
        "frames_evaluable": ev,
        "frames_unevaluable": a["frames_unevaluable"],
        "coverage": (a["frames_fired"] / ev) if ev else None,
        "contingencies_fired": cf,
        "p_violation_given_fires": p_fire,
        "p_violation_given_silent": p_silent,
        "p_violation_base": base,
        "lift_vs_base": lift,
        "discrimination": disc,
        "verdict": verdict_of(a["frames_fired"], a["frames_silent"], disc),
    }


def fmt(x, width=7, nd=3) -> str:
    return f"{x:>{width}.{nd}f}" if x is not None else f"{'-':>{width}}"


def main() -> int:
    ap = argparse.ArgumentParser(description="N-1 calibration of a shield channel")
    ap.add_argument("--readmission", default="results/audit/readmission.json")
    ap.add_argument("--channel", default="WARN",
                    help="Which channel to calibrate (WARN, NORMAL, BLOCK, ...)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per grid (default: every frame)")
    ap.add_argument("--json", default=None,
                    help="Default: results/audit/<channel>_n1_calibration.json")
    args = ap.parse_args()

    groups = condition_groups(Path(args.readmission), args.channel)
    if not groups:
        print(f"no {args.channel} rules in {args.readmission}")
        return 1
    conditions = list(groups)
    n_rules = sum(len(v) for v in groups.values())
    print(f"\ncalibrating {len(conditions)} distinct {args.channel} conditions "
          f"({n_rules} rule records) against the N-1 label")

    per_grid = {}
    for tag in TAGS:
        try:
            print(f"  [{tag}] scanning ...")
            per_grid[tag] = scan(tag, conditions, Path(args.data_dir), args.max_frames)
        except FileNotFoundError as exc:
            print(f"  [{tag}] dataset missing, skipping: {exc}")
    if not per_grid:
        print("no dataset found")
        return 1

    # Pooled counts, so a predicate that only speaks on one grid still gets a
    # number -- with its per-grid row visible beside it to say where it came from.
    pooled = blank(conditions)
    for res in per_grid.values():
        for cond, a in res["acc"].items():
            for k, v in a.items():
                pooled[cond][k] += v

    print("\n" + "=" * 100)
    print(f"P(N-1 violation | rule fires) -- {args.channel} channel, all frames")
    print("=" * 100)
    print(f"\n  {'condition':<50}{'grid':<13}{'fires':>7}{'cover':>8}"
          f"{'P|fire':>8}{'P|quiet':>9}{'P|f-P|q':>9}{'grids':>7}  verdict")
    print("  " + "-" * 98)

    out_rules = []
    for cond in conditions:
        ids = ", ".join(sorted({r["rule_id"] for r in groups[cond]}))
        rows = {tag: rates(per_grid[tag]["acc"][cond]) for tag in per_grid}
        pooled_row = rates(pooled[cond])
        # The naive pooled lift is replaced, not merely annotated: nothing should
        # be able to read it by accident. The unstratified value is kept under
        # its own key for the reader who wants to see the confound.
        pooled_row["discrimination_unstratified"] = pooled_row["discrimination"]
        pooled_row["lift_vs_base_unstratified"] = pooled_row["lift_vs_base"]
        pooled_row["discrimination"] = stratified(rows, "discrimination")
        pooled_row["lift_vs_base"] = stratified(rows, "lift_vs_base")
        pooled_row["stratified_by_grid"] = True
        n_grids = sum(1 for t in per_grid if rows[t]["verdict"] != "INSUFFICIENT")
        pooled_row["n_grids_contributing"] = n_grids
        pooled_row["grids_contributing"] = [t for t in per_grid
                                            if rows[t]["verdict"] != "INSUFFICIENT"]
        pooled_row["verdict"] = (
            verdict_of(pooled_row["frames_fired"], pooled_row["frames_silent"],
                       pooled_row["discrimination"])
            if n_grids else "INSUFFICIENT")
        rows["pooled"] = pooled_row
        first = True
        for tag, r in rows.items():
            label = cond[:48] if first else ""
            first = False
            mark = "*" if tag == "pooled" else " "
            print(f"  {label:<50}{mark}{tag:<12}{r['frames_fired']:>7}"
                  f"{fmt(r['coverage'], 8, 3)}{fmt(r['p_violation_given_fires'])}"
                  f"{fmt(r['p_violation_given_silent'], 9)}{fmt(r['discrimination'], 9, 3)}"
                  f"{r.get('n_grids_contributing', ''):>7}  {r['verdict']}")
        print(f"    -> {ids}")
        out_rules.append({
            "condition": cond,
            "rule_ids": sorted({r["rule_id"] for r in groups[cond]}),
            "n_records": len(groups[cond]),
            "role": groups[cond][0].get("role"),
            "per_grid": {t: rows[t] for t in per_grid},
            "pooled": rows["pooled"],
        })

    tally: dict = {}
    for r in out_rules:
        v = r["pooled"]["verdict"]
        tally[v] = tally.get(v, 0) + 1
    print("\n  pooled verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print()
    print("  ELEVATED     firing is associated with a higher N-1 violation rate;")
    print("               the warning may quote P|fire.")
    print("  NEUTRAL      firing carries no N-1 information. The rule is still a true")
    print("               statement about the telemetry, so it may be reported as an")
    print("               OBSERVATION -- but never as a risk warning.")
    print("  INVERTED     firing is associated with a LOWER violation rate. No phrasing")
    print("               makes this a warning.")
    print(f"  INSUFFICIENT fired on fewer than {MIN_FRAMES} frames; no rate is claimed.")
    print()
    print("  The pooled discrimination is STRATIFIED BY GRID -- a weighted mean of the three")
    print("  within-grid values, not the value of the pooled counts. A predicate that")
    print("  fires on one topology and not the others would otherwise score a lift")
    print("  for doing nothing but identifying that topology's base rate.")

    dest = Path(args.json or f"results/audit/{args.channel.lower()}_n1_calibration.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "channel": args.channel,
        "n_distinct_conditions": len(conditions),
        "n_rule_records": n_rules,
        "max_frames": args.max_frames,
        "min_frames_for_a_rate": MIN_FRAMES,
        "min_discrimination": MIN_LIFT,
        "frames_scanned": {t: per_grid[t]["n_frames"] for t in per_grid},
        "verdict_tally_pooled": tally,
        "note": ("Effective sample size is FRAMES, not contingencies -- lines in one "
                 "frame share a grid state. NOT_EVALUABLE frames are excluded from "
                 "both arms rather than counted as silent. -1 labels are missing, "
                 "never secure. The verdict is taken on `discrimination` "
                 "(P|fire - P|silent), not on lift against the base rate, which is "
                 "structurally ~0 for a predicate that fires on nearly every frame. "
                 "Pooled values are STRATIFIED BY GRID; the confounded ones are kept "
                 "as *_unstratified and must not be quoted. Generalizes "
                 "evaluation/loading_band_calibration.py from rho_max bands to "
                 "arbitrary conditions."),
        "conditions": out_rules,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
