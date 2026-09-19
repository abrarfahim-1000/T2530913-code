"""Arm A3 — plain tabular baselines, measured ACROSS topologies.

`supplimentary_docs/results_comparisons.md` §A3 specified this as in-distribution
insurance against "your GNN is an overparameterised tabular model". Re-scoped
2026-09-19 to the question the thesis actually turns on:

    when the GNN collapses on an unseen grid, does a NON-GRAPH learned model
    collapse with it?

The answer decides how broad the thesis's central claim may be. If a plain model
also fails to transfer, the claim is "learned screeners do not generalise across
topologies" — broad, and supported by two independent architectures. If the plain
model holds where the GNN does not, the claim narrows to "this GNN does not", and
the graph machinery becomes the suspect rather than the vindication.

The features are the RAW portion of the GNN's own readout — for line k, its edge
features plus the node features of both endpoints:

    [ edge_attr_k || x_or || x_ex ]        8 + 8 + 8 = 24 columns

That is the readout's skip connection with the three GATv2 layers removed. The
comparison is therefore "same information, no message passing, no graph" rather
than "some other feature set", which is what makes it interpretable.

PROTOCOL (§0). Fit on the neurips2020 TRAIN split only. Threshold selected on its
VAL split and held unchanged across all three grids. neurips2020 scored on its
test split, foreign grids on every frame. Scored by `best_f1_and_thr`, the same
function every other arm uses. Foreign features are standardized with the
neurips2020 train statistics, matching the GNN's protocol.

Usage:
    python evaluation/eval_tabular_n1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(".")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from evaluation.eval_n1_cross_topology import best_f1_and_thr
from scripts.pyg_data import (GridEnvMetadata, build_edges, build_line_targets,
                              build_node_features)

DATA_DIR = "data"
OUT = os.path.join("results", "tabular", "tabular_baselines.json")
HOME = "neurips2020"
TAGS = ("neurips2020", "case14", "wcci2022")
SEED = 42  # the project's seed everywhere else; see CLAUDE.md

FEATURE_NAMES = (
    ["edge_rho", "edge_p_or", "edge_q_or", "edge_near_limit",
     "edge_abs_p", "edge_abs_q", "edge_s", "edge_headroom"]
    + [f"or_{n}" for n in ("load_p", "mean_v", "max_rho", "conn_frac",
                           "trip_frac", "headroom", "abs_p", "degree")]
    + [f"ex_{n}" for n in ("load_p", "mean_v", "max_rho", "conn_frac",
                           "trip_frac", "headroom", "abs_p", "degree")]
)


def frame_rows(record: dict, meta: GridEnvMetadata) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame -> (X, y, rho) with one row per in-service line.

    `build_edges` and `build_line_targets` are the DataLoader's own functions, so
    the rows are aligned to the labels by construction — the same guarantee the
    LODF arm relies on (thesis_findings.md §25.2).
    """
    nodes = build_node_features(record, meta)
    _edge_index, edge_attr = build_edges(record, meta)
    line_y, line_mask, line_id = build_line_targets(record, meta)

    n = len(line_id)
    edge_fwd = np.asarray(edge_attr, dtype=np.float32)[:n]  # or->ex half only
    x = np.concatenate(
        [edge_fwd,
         nodes[meta.line_or_bus[line_id]],
         nodes[meta.line_ex_bus[line_id]]],
        axis=1,
    )
    rho = np.asarray(record["rho"], dtype=np.float32)[line_id]
    return x[line_mask], line_y[line_mask].astype(int), rho[line_mask]


def collect(tag: str, indices: np.ndarray, meta: GridEnvMetadata) -> dict[str, np.ndarray]:
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    wanted = set(indices.tolist())
    xs, ys, rhos = [], [], []
    with open(jsonl) as fh:
        for i, line in enumerate(fh):
            if i in wanted and line.strip():
                x, y, rho = frame_rows(json.loads(line), meta)
                xs.append(x)
                ys.append(y)
                rhos.append(rho)
    if len(xs) != len(indices):
        raise ValueError(f"{tag}: expected {len(indices)} frames, read {len(xs)}")
    return {"X": np.concatenate(xs).astype(np.float32),
            "y": np.concatenate(ys),
            "rho": np.concatenate(rhos)}


def load_meta(tag: str) -> GridEnvMetadata:
    with open(os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1_meta.json")) as fh:
        return GridEnvMetadata(json.load(fh))


def frame_count(tag: str) -> int:
    path = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    return sum(1 for line in open(path) if line.strip())


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

    # Standardization from the TRAIN split only, applied unchanged to every grid
    # — the same rule the GNN's normalization follows.
    mean = train["X"].mean(axis=0)
    std = train["X"].std(axis=0)
    std[std < 1e-6] = 1.0

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "gradient_boosted_trees": HistGradientBoostingClassifier(
            max_iter=300, random_state=SEED),
    }
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
                # the alignment check: must reproduce 0.4639 / 0.5392 / 0.4915
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
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)

    _print(report)
    print(f"\n  written: {args.json}")


def _print(rep: dict) -> None:
    print(f"\n{'=' * 78}")
    print("  A3 — tabular baselines, no graph, no message passing")
    print(f"  fit on {rep['protocol']['fit_on']}")
    print(f"{'=' * 78}")
    head = f"  {'grid':<14}{'all-pos':>9}{'rule':>9}{'logreg':>12}{'GBT':>12}"
    print(head + "   (held threshold)")
    for tag, g in rep["grids"].items():
        b = g["baselines"]
        lr = g["arms"]["logistic_regression"]["held"]["f1"]
        gb = g["arms"]["gradient_boosted_trees"]["held"]["f1"]
        print(f"  {tag:<14}{b['all_positive']:>9.4f}"
              f"{b['best_rule_rho_of_removed_line']:>9.4f}{lr:>12.4f}{gb:>12.4f}")
    print(f"\n  {'':14}{'':9}{'':9}{'oracle:':>12}")
    for tag, g in rep["grids"].items():
        lr = g["arms"]["logistic_regression"]["oracle"]["f1"]
        gb = g["arms"]["gradient_boosted_trees"]["oracle"]["f1"]
        print(f"  {tag:<14}{'':9}{'':9}{lr:>12.4f}{gb:>12.4f}")
    print("\n  ^ the rule column is the ALIGNMENT CHECK: it must read "
          "0.4639 / 0.5392 / 0.4915")


if __name__ == "__main__":
    main()
