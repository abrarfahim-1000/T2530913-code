"""
shield_threshold_analysis.py — the shield away from its single operating point.

Four questions the held-threshold table cannot answer, all from ONE forward pass
per grid:

  1. SWEEP        Does the shield help at every threshold, or only at 0.8849?
  2. RECALIBRATION  Does the shield beat per-grid threshold recalibration, and do
                    the two stack? (2x2: raw/gated x held/recalibrated)
  3. BURDEN       How many snapshots does the shield act on, and how many of its
                  flags are wrong, per snapshot?
  4. REGRESSIONS  Are the shield's wrong flags the contingencies that remove the
                  overloaded line itself?

WHY ONE PASS SUFFICES
---------------------
The veto path (`validate_n1` on a SECURE verdict) depends only on the frame's
base case, never on the model's score. So the set of contingencies the shield
would overturn is a fixed mask M, computed once by running the REAL shield with
every prediction set to secure. At any threshold t the shielded prediction is
exactly  (logit > t) OR M.  The held-threshold row is re-derived from this and
asserted against the recorded block counts, so the identity is checked, not
assumed.

RECALIBRATION PROTOCOL
----------------------
A fraction CAL_FRAC of each target set's CHRONICS (not frames — frames within a
chronic are near-duplicates) is used to pick the threshold; every arm is then
reported on the remaining chronics. Repeated over N_SPLITS random partitions.
The answer-key oracle on the full set is also reported, as an upper bound.
The recalibrated gated arm picks its threshold to maximise GATED F1.

Usage:
    python evaluation/shield_threshold_analysis.py            # score (cached) + analyse
    python evaluation/shield_threshold_analysis.py --rescore  # ignore the cache
"""
from __future__ import annotations

import argparse
import json
import linecache
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(".")

from evaluation.eval_shield_n1 import (  # noqa: E402
    CKPT, DEDUPED_RULES, EVAL_BATCH_SIZE, HOME, best_f1_and_thr, load_rules, prf,
    run_shield, score_topology, score_topology_indices,
)
from shield.channels import resolve_channels_for_grid  # noqa: E402
from training.config import DATA_DIR  # noqa: E402

TAGS = ["neurips2020", "case14", "wcci2022"]
CAL_FRAC = 0.30
N_SPLITS = 20
SWEEP_POINTS = 199
MIN_BLOCKS_FOR_PRECISION = 30     # below this an intervention precision is noise
OUT = Path("results") / "shield_sweep" / "shield_threshold_analysis.json"

# Held-threshold block counts recorded in results/shield/shield_<tag>_validated.json.
# The cached mask must reproduce them exactly, or the OR identity is wrong.
RECORDED_BLOCKS = {"neurips2020": 353, "case14": 103, "wcci2022": 22559}


# ── stage 1: score once, cache ────────────────────────────────────────────────

def cache_path(tag: str) -> str:
    return os.path.join(DATA_DIR, f"shield_sweep_scores_{tag}.npz")


def score_and_cache(tag: str, norm: tuple, rules: list[dict]) -> None:
    scored = score_topology(tag, CKPT, EVAL_BATCH_SIZE, norm)
    n = len(scored["y"])
    # Every prediction secure -> `blocked` is exactly the threshold-free veto mask.
    sh = run_shield(scored, np.zeros(n, dtype=int),
                    resolve_channels_for_grid(rules, tag), tag, None)

    frames = np.unique(scored["frame"])
    chronic, n_over = {}, {}
    for f in frames:
        raw = json.loads(linecache.getline(scored["jsonl"], int(f) + 1))
        rho = np.asarray(raw["rho"], dtype=float)
        live = np.asarray(raw["line_status"], dtype=bool)
        chronic[int(f)] = int(raw["chronic_id"])
        n_over[int(f)] = int(((rho >= 1.0) & live).sum())
    linecache.clearcache()

    np.savez_compressed(
        cache_path(tag),
        logits=scored["logits"], y=scored["y"], rho=scored["rho"],
        frame=scored["frame"], line=scored["line"], mask=sh["blocked"],
        chronic=np.array([chronic[int(f)] for f in scored["frame"]]),
        n_over=np.array([n_over[int(f)] for f in scored["frame"]]),
    )


def held_threshold(norm: tuple) -> float:
    val = score_topology_indices(HOME, CKPT, EVAL_BATCH_SIZE, norm,
                                 np.load(os.path.join(DATA_DIR,
                                                      "split_neurips2020_n1_val_idx.npy")))
    return best_f1_and_thr(val["logits"], val["y"])[1]


# ── stage 2: analyses ─────────────────────────────────────────────────────────

def f1_of(y: np.ndarray, pred: np.ndarray) -> float:
    return prf(y, pred)["f1"]


def best_gated_thr(logits, y, mask) -> tuple[float, float]:
    """Best F1 of (logit > t) | mask over all t, and the t achieving it.

    Mask-positive contingencies are predicted positive at every t, so only the
    ordering of the rest matters: sort them, sweep, and add the mask's fixed
    contribution to every cutoff.
    """
    tp0 = int((mask & (y == 1)).sum())
    fp0 = int((mask & (y == 0)).sum())
    P = int(y.sum())
    rest = ~mask
    s, ys = logits[rest], y[rest]
    order = np.argsort(-s)
    ys, s = ys[order], s[order]
    tp = tp0 + np.concatenate([[0], np.cumsum(ys)])
    fp = fp0 + np.concatenate([[0], np.cumsum(1 - ys)])
    f1 = 2 * tp / np.maximum(2 * tp + fp + (P - tp), 1)
    i = int(np.argmax(f1))
    thr = float(s[i - 1]) if i > 0 else float(s[0]) + 1.0   # predict (logit >= s[i-1])
    return float(f1[i]), thr - 1e-7


def sweep(d: dict, held: float) -> dict:
    logits, y, mask = d["logits"], d["y"], d["mask"]
    qs = np.quantile(logits, np.linspace(0.0005, 0.9995, SWEEP_POINTS))
    thrs = np.unique(np.concatenate([qs, [held]]))
    rows = []
    for t in thrs:
        raw = logits > t
        blocks = mask & ~raw
        n_b = int(blocks.sum())
        corr = int((blocks & (y == 1)).sum())
        f_raw, f_gat = f1_of(y, raw.astype(int)), f1_of(y, (raw | mask).astype(int))
        rows.append({
            "threshold": float(t), "f1_raw": f_raw, "f1_gated": f_gat,
            "delta": f_gat - f_raw, "blocks": n_b, "corrections": corr,
            "intervention_precision": corr / n_b if n_b else None,
            "precision_bar_f1_over_2": f_raw / 2,
        })
    trusted = [r for r in rows if r["blocks"] >= MIN_BLOCKS_FOR_PRECISION]
    worst = min(trusted, key=lambda r: r["intervention_precision"]) if trusted else None
    return {
        "n_thresholds": len(rows),
        "n_where_gated_below_raw": sum(r["delta"] < -1e-12 for r in rows),
        "min_delta": min(r["delta"] for r in rows),
        "max_delta": max(r["delta"] for r in rows),
        "min_intervention_precision": worst and worst["intervention_precision"],
        "min_intervention_precision_at": worst and worst["threshold"],
        "min_intervention_precision_blocks": worst and worst["blocks"],
        "n_where_precision_below_bar": sum(
            r["intervention_precision"] is not None
            and r["intervention_precision"] <= r["precision_bar_f1_over_2"] for r in trusted),
        "curve": rows,
    }


def two_by_two(d: dict, held: float) -> dict:
    logits, y, mask, chronic = d["logits"], d["y"], d["mask"], d["chronic"]
    raw_oracle, _ = best_f1_and_thr(logits, y)
    gated_oracle, _ = best_gated_thr(logits, y, mask)
    uniq = np.unique(chronic)
    n_cal = max(1, int(round(CAL_FRAC * len(uniq))))
    cells = {k: [] for k in ("raw_held", "raw_recal", "gated_held", "gated_recal")}
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        cal_ch = rng.choice(uniq, size=n_cal, replace=False)
        cal = np.isin(chronic, cal_ch)
        rep = ~cal
        _, t_raw = best_f1_and_thr(logits[cal], y[cal])
        _, t_gat = best_gated_thr(logits[cal], y[cal], mask[cal])
        lr, yr, mr = logits[rep], y[rep], mask[rep]
        cells["raw_held"].append(f1_of(yr, (lr > held).astype(int)))
        cells["raw_recal"].append(f1_of(yr, (lr > t_raw).astype(int)))
        cells["gated_held"].append(f1_of(yr, ((lr > held) | mr).astype(int)))
        cells["gated_recal"].append(f1_of(yr, ((lr > t_gat) | mr).astype(int)))

    def summ(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "sd": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}

    c = {k: np.asarray(v) for k, v in cells.items()}
    return {
        "n_chronics": int(len(uniq)), "n_cal_chronics": n_cal, "n_splits": N_SPLITS,
        "cells": {k: summ(v) for k, v in cells.items()},
        "paired_deltas": {
            "shield_given_held": summ(c["gated_held"] - c["raw_held"]),
            "shield_given_recal": summ(c["gated_recal"] - c["raw_recal"]),
            "recal_given_raw": summ(c["raw_recal"] - c["raw_held"]),
            "recal_given_shield": summ(c["gated_recal"] - c["gated_held"]),
            "gated_held_minus_raw_recal": summ(c["gated_held"] - c["raw_recal"]),
        },
        "oracle_full_set": {"raw": raw_oracle, "gated": gated_oracle,
                            "raw_held": f1_of(y, (logits > held).astype(int)),
                            "gated_held": f1_of(y, ((logits > held) | mask).astype(int))},
    }


def burden_and_regressions(d: dict, held: float) -> dict:
    logits, y, mask, frame = d["logits"], d["y"], d["mask"], d["frame"]
    rho, n_over = d["rho"], d["n_over"]
    raw = logits > held
    blocks = mask & ~raw
    false_b = blocks & (y == 0)
    n_frames = int(len(np.unique(frame)))

    removes_overloaded = rho >= 1.0
    removes_sole_overload = removes_overloaded & (n_over == 1)
    keep = blocks & ~removes_sole_overload
    return {
        "snapshots": n_frames,
        "snapshots_base_violating": int(len(np.unique(frame[mask]))),
        "snapshots_with_block": int(len(np.unique(frame[blocks]))),
        "blocks": int(blocks.sum()),
        "false_blocks": int(false_b.sum()),
        "false_blocks_in_snapshots": int(len(np.unique(frame[false_b]))),
        "raw_false_alarms": int((raw & (y == 0)).sum()),
        "added_false_alarm_pct": 100 * int(false_b.sum()) / max(int((raw & (y == 0)).sum()), 1),
        "blocks_per_blocked_snapshot": float(blocks.sum() / max(len(np.unique(frame[blocks])), 1)),
        "regressions": {
            "false_blocks": int(false_b.sum()),
            "removing_an_overloaded_line": int((false_b & removes_overloaded).sum()),
            "removing_the_sole_overloaded_line": int((false_b & removes_sole_overload).sum()),
            "true_blocks_removing_sole_overload": int((blocks & (y == 1) & removes_sole_overload).sum()),
            "precision_as_is": float((blocks & (y == 1)).sum() / max(blocks.sum(), 1)),
            "precision_exempting_sole_overload_removal":
                float((keep & (y == 1)).sum() / max(keep.sum(), 1)),
            "blocks_exempting_sole_overload_removal": int(keep.sum()),
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rescore", action="store_true", help="Ignore cached scores")
    args = ap.parse_args()

    need = [t for t in TAGS if args.rescore or not os.path.exists(cache_path(t))]
    thr_file = os.path.join(DATA_DIR, "shield_sweep_held_threshold.json")
    if need or not os.path.exists(thr_file):
        from scripts.pyg_data import PreloadedGridDataset
        from training.config import DEVICE
        from training.train_gnn import compute_normalization_stats

        print("Computing 36-bus train-split normalization stats...")
        home = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data_n1.pt"),
                                    device=DEVICE)
        norm = compute_normalization_stats(
            home, np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy")))
        del home
        print("Selecting the held threshold on the neurips2020 val split...")
        Path(thr_file).write_text(json.dumps({"held": held_threshold(norm)}))
        rules = load_rules(DEDUPED_RULES)
        for tag in need:
            print(f"Scoring {tag} (one forward pass + threshold-free shield mask)...")
            score_and_cache(tag, norm, rules)
    held = json.loads(Path(thr_file).read_text())["held"]
    print(f"held threshold: {held:.6f}")

    out = {"checkpoint": CKPT, "batch_size": EVAL_BATCH_SIZE, "rules": DEDUPED_RULES,
           "held_threshold": held, "cal_frac": CAL_FRAC, "n_splits": N_SPLITS,
           "grids": {}}
    for tag in TAGS:
        d = dict(np.load(cache_path(tag)))
        n_held_blocks = int((d["mask"] & ~(d["logits"] > held)).sum())
        assert n_held_blocks == RECORDED_BLOCKS[tag], (
            f"{tag}: mask reproduces {n_held_blocks} held blocks, recorded "
            f"{RECORDED_BLOCKS[tag]} — the (logit>t)|M identity does not hold")
        out["grids"][tag] = {
            "n_contingencies": int(len(d["y"])),
            "sweep": sweep(d, held),
            "two_by_two": two_by_two(d, held),
            "burden": burden_and_regressions(d, held),
        }
        g = out["grids"][tag]
        s, t, b = g["sweep"], g["two_by_two"], g["burden"]
        print(f"\n== {tag} ==")
        print(f"  sweep: gated<raw at {s['n_where_gated_below_raw']}/{s['n_thresholds']} "
              f"thresholds; delta range [{s['min_delta']:+.4f}, {s['max_delta']:+.4f}]; "
              f"min intervention precision {s['min_intervention_precision']:.3f} "
              f"({s['min_intervention_precision_blocks']} blocks)")
        for k, v in t["cells"].items():
            print(f"  {k:12s} {v['mean']:.4f} ± {v['sd']:.4f}")
        for k, v in t["paired_deltas"].items():
            print(f"  Δ {k:28s} {v['mean']:+.4f} [{v['min']:+.4f}, {v['max']:+.4f}]")
        print(f"  oracle full set: {t['oracle_full_set']}")
        print(f"  burden: {b['snapshots_with_block']}/{b['snapshots']} snapshots blocked, "
              f"{b['false_blocks']} false blocks in {b['false_blocks_in_snapshots']} snapshots, "
              f"+{b['added_false_alarm_pct']:.2f}% false alarms")
        print(f"  regressions: {b['regressions']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
