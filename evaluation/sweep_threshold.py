# -*- coding: utf-8 -*-
"""Threshold trade curve: F1-selected vs recall-weighted selection.

Protocol is unchanged in shape — the threshold is ALWAYS selected on the
neurips2020 VAL split and then held fixed across all three grids. The only thing
that varies is the OBJECTIVE used to select it on val:

    F-beta, beta = 1    -> the current protocol (symmetric: a false alarm costs
                           the same as a missed violation)
    F-beta, beta > 1    -> weights recall higher, i.e. a missed violation costs
                           beta^2 times a false alarm

Each grid is scored ONCE; thresholds are then applied to the cached logits, so
this is one forward pass per grid rather than one per operating point.
"""
import json
import os
import sys

import numpy as np

sys.path.append(".")

from evaluation.eval_shield_n1 import (EVAL_BATCH_SIZE, prf, score_topology,
                                       score_topology_indices)
from scripts.pyg_data import PreloadedGridDataset
from training.config import DATA_DIR, DEVICE
from training.train_gnn import compute_normalization_stats

CKPT = "gnn_checkpoint_n1.pt"
TAGS = ["neurips2020", "case14", "wcci2022"]
BETAS = [1.0, 1.5, 2.0, 3.0, 5.0]
OUT = os.path.join("results", "threshold", "threshold_sweep.json")


def fbeta_curve(score, y, beta):
    """Best F-beta over all cutoffs, and the cutoff achieving it."""
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P = ys.sum()
    if P == 0:
        return 0.0, 0.0
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    b2 = beta * beta
    den = b2 * prec + rec
    fb = np.where(den > 0, (1 + b2) * prec * rec / np.maximum(den, 1e-12), 0.0)
    i = int(np.argmax(fb))
    return float(fb[i]), float(score[order][i])


def main():
    print("Computing 36-bus train-split normalization stats...")
    home_pt = os.path.join(DATA_DIR, "processed_grid_data_n1.pt")
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    home = PreloadedGridDataset(home_pt, device=DEVICE)
    norm = compute_normalization_stats(home, train_idx)
    del home

    # ── select thresholds on the VAL split, one per beta ──────────────────────
    val_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy"))
    print(f"Scoring neurips2020 VAL split ({len(val_idx):,} frames) for selection...")
    val = score_topology_indices("neurips2020", CKPT, EVAL_BATCH_SIZE, norm, val_idx)

    thresholds = {}
    for b in BETAS:
        _, thr = fbeta_curve(val["logits"], val["y"], b)
        thresholds[b] = thr
        print(f"   beta={b:<4}  val-selected threshold = {thr:.4f}")

    # ── score each grid once, then apply every threshold ──────────────────────
    results = {}
    for tag in TAGS:
        print(f"\nScoring {tag} ...")
        s = score_topology(tag, CKPT, EVAL_BATCH_SIZE, norm)
        # score_topology already restricts neurips2020 to the held-out test split
        logits, y = s["logits"], s["y"]
        print(f"   scope: {s['scope']}")
        per_beta = {}
        for b in BETAS:
            r = prf(y, (logits >= thresholds[b]).astype(int))
            r["threshold"] = thresholds[b]
            per_beta[str(b)] = r
        results[tag] = {"n": int(len(y)), "positives": int(y.sum()), "betas": per_beta}
        print(f"   {len(y):,} contingencies, {int(y.sum()):,} positives")

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("  THRESHOLD TRADE CURVE — selection objective varied on the neurips2020 VAL split")
    print("=" * 92)
    for tag in TAGS:
        R = results[tag]
        print(f"\n  {tag}   ({R['n']:,} contingencies, {R['positives']:,} violations)")
        print(f"    {'objective':<12} {'thr':>7} {'F1':>8} {'recall':>8} {'prec':>8} "
              f"{'MISSED':>10} {'false alarms':>13}")
        base = R["betas"]["1.0"]
        for b in BETAS:
            r = R["betas"][str(b)]
            tagname = "F1 (current)" if b == 1.0 else f"F{b:g}"
            dm = r["missed_violations"] - base["missed_violations"]
            dfa = r["false_alarms"] - base["false_alarms"]
            extra = "" if b == 1.0 else f"   ({dm:+,} missed, {dfa:+,} FA)"
            print(f"    {tagname:<12} {r['threshold']:7.4f} {r['f1']:8.4f} "
                  f"{r['recall']:8.3f} {r['precision']:8.3f} "
                  f"{r['missed_violations']:10,} {r['false_alarms']:13,}{extra}")

    # exchange rate: false alarms bought per missed violation avoided
    print("\n" + "=" * 92)
    print("  EXCHANGE RATE — extra false alarms per missed violation avoided, vs F1")
    print("=" * 92)
    print(f"    {'grid':<14}" + "".join(f"{'F'+format(b,'g'):>12}" for b in BETAS[1:]))
    for tag in TAGS:
        R = results[tag]
        base = R["betas"]["1.0"]
        cells = []
        for b in BETAS[1:]:
            r = R["betas"][str(b)]
            dm = base["missed_violations"] - r["missed_violations"]
            dfa = r["false_alarms"] - base["false_alarms"]
            cells.append(f"{dfa/dm:12.1f}" if dm > 0 else f"{'n/a':>12}")
        print(f"    {tag:<14}" + "".join(cells))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"checkpoint": CKPT, "batch_size": EVAL_BATCH_SIZE,
                   "selection_split": "neurips2020 val", "betas": BETAS,
                   "thresholds": {str(k): v for k, v in thresholds.items()},
                   "results": results}, f, indent=2)
    print(f"\n  Wrote {OUT}")


if __name__ == "__main__":
    main()
