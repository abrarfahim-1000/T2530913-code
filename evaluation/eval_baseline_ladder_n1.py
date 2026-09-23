"""Non-graph baseline ladder for the N-1 screening task.

Companion to Arm A3 (`evaluation/eval_tabular_n1.py`), not a replacement for it.
Arm A3's logistic-regression and gradient-boosted-tree numbers are already
measured, cited (`thesis_findings.md` §26) and untouched by this script — a
non-graph tree ensemble tying/beating the GNN in-distribution and degrading
less off-distribution is a real, reported finding, not something to bury.

This script asks a narrower question with a capacity ladder, same features,
same protocol, same everything except the model:

    logistic regression (linear)  ->  MLP (nonlinear, gradient descent)
                                   ->  random forest (nonlinear, bagged trees)

Random forest stands in for "a different tree inductive bias than boosting"
without duplicating Arm A3's HistGradientBoostingClassifier run. Structure
(GIN/GINE vs GAT) is deliberately out of scope here — that's an architecture
ablation within the GNN family, not a non-graph baseline, and needs its own
training run against `training/train_gnn.py`.

Same 24-column feature set as Arm A3 — the GNN's own readout skip connection
with the three GATv2 layers removed: [ edge_attr_k || x_or || x_ex ]. Same
protocol: fit on neurips2020 train, threshold held from its val split, scored
on neurips2020 test and every frame of the two unseen grids.

Usage:
    python evaluation/eval_baseline_ladder_n1.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(".")

from evaluation.eval_n1_cross_topology import best_f1_and_thr
from evaluation.eval_tabular_n1 import (DATA_DIR, FEATURE_NAMES, HOME, SEED,
                                        TAGS, collect, frame_count, load_meta)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

OUT = os.path.join("results", "tabular", "baseline_ladder.json")


def build_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=SEED,
            max_iter=300,
            early_stopping=True,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    home_meta = load_meta(HOME)
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    val_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy"))
    test_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy"))

    print(f"Collecting neurips2020 train ({len(train_idx):,} frames)...")
    train = collect(HOME, train_idx, home_meta)
    print(f"  {train['X'].shape[0]:,} contingencies x {train['X'].shape[1]} features, "
          f"{train['y'].mean():.1%} positive")

    mean = train["X"].mean(axis=0)
    std = train["X"].std(axis=0)
    std[std < 1e-6] = 1.0

    models = build_models()
    for name, clf in models.items():
        print(f"Fitting {name}...")
        clf.fit((train["X"] - mean) / std, train["y"])

    def score(clf, data: dict) -> np.ndarray:
        return clf.predict_proba((data["X"] - mean) / std)[:, 1]

    print(f"Collecting neurips2020 val ({len(val_idx):,} frames) for the threshold...")
    val = collect(HOME, val_idx, home_meta)
    held = {name: best_f1_and_thr(score(clf, val), val["y"])[1]
            for name, clf in models.items()}

    report: dict = {
        "protocol": {
            "fit_on": f"neurips2020 train split ({len(train_idx)} frames, "
                      f"{train['X'].shape[0]} contingencies)",
            "threshold_selected_on": f"neurips2020 val split ({len(val_idx)} frames)",
            "features": FEATURE_NAMES,
            "seed": SEED,
            "note": "companion to Arm A3 (eval_tabular_n1.py); its logistic "
                    "regression and gradient-boosted-tree numbers are the ones "
                    "cited in thesis_findings.md §26 and are unchanged by "
                    "this script",
        },
        "held_thresholds": held,
        "grids": {},
    }

    for tag in TAGS:
        meta = home_meta if tag == HOME else load_meta(tag)
        if tag == HOME:
            idx, scope = test_idx, "held-out test split"
        else:
            idx, scope = np.arange(frame_count(tag)), "all frames (unseen topology)"
        print(f"Scoring {tag} ({len(idx):,} frames)...")
        data = collect(tag, idx, meta)
        y = data["y"]
        p = float(y.mean())

        entry = {
            "scope": scope,
            "contingencies": int(len(y)),
            "violation_rate": p,
            "baselines": {
                "all_positive": float(2 * p / (1 + p)) if p else 0.0,
                "best_rule_rho_of_removed_line": best_f1_and_thr(data["rho"], y)[0],
            },
            "arms": {},
        }
        for name, clf in models.items():
            s = score(clf, data)
            oracle_f1, oracle_thr = best_f1_and_thr(s, y)
            pred = s >= held[name]
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            entry["arms"][name] = {
                "held": {"threshold": float(held[name]),
                         "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
                         "precision": float(prec), "recall": float(rec)},
                "oracle": {"f1": oracle_f1, "threshold": oracle_thr},
            }
        report["grids"][tag] = entry
        del data

    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    import json
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)

    _print(report)
    print(f"\n  written: {args.json}")


def _print(rep: dict) -> None:
    names = list(rep["held_thresholds"].keys())
    print(f"\n{'=' * 88}")
    print("  baseline ladder — logistic regression / MLP / random forest, no graph")
    print(f"  fit on {rep['protocol']['fit_on']}")
    print(f"{'=' * 88}")
    head = f"  {'grid':<14}{'all-pos':>9}{'rule':>9}" + "".join(f"{n:>16}" for n in names)
    print(head + "   (held threshold)")
    for tag, g in rep["grids"].items():
        b = g["baselines"]
        vals = "".join(f"{g['arms'][n]['held']['f1']:>16.4f}" for n in names)
        print(f"  {tag:<14}{b['all_positive']:>9.4f}"
              f"{b['best_rule_rho_of_removed_line']:>9.4f}{vals}")
    print(f"\n  {'':14}{'':9}{'':9}" + "".join(f"{'oracle:':>16}" for _ in names))
    for tag, g in rep["grids"].items():
        vals = "".join(f"{g['arms'][n]['oracle']['f1']:>16.4f}" for n in names)
        print(f"  {tag:<14}{'':9}{'':9}{vals}")
    print("\n  ^ the rule column is the ALIGNMENT CHECK: it must read "
          "0.4639 / 0.5392 / 0.4915")


if __name__ == "__main__":
    main()
