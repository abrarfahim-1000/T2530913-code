"""
Cross-topology evaluation for the N-1 screening task.

The edge-level head has no dependence on `n_line` — it emits one logit per line
present — so the 36-bus checkpoint runs unchanged on the 20-line case14 and the
186-line WCCI graphs. This is a genuine improvement over the classify setup,
where the localization head emitted `n_nodes` logits and had to be DISABLED for
cross-topology evaluation.

Foreign features are normalized with the **36-bus train-split stats**, matching
the classify protocol: the quantities are physical and share scale across grids.

Every number is reported against the two baselines that decide whether the model
earned anything (component_d_plan.md §1.1):

  all-positive           is the target non-vacuous?
  rho of the removed line   does the model beat the best single rule?

A model that fails the second one has learned nothing a rule cannot state, and
that is the finding to report — not a metric in isolation.

Usage:
    python evaluation/eval_n1_cross_topology.py --tag neurips2020   # in-distribution (test split)
    python evaluation/eval_n1_cross_topology.py --tag case14
    python evaluation/eval_n1_cross_topology.py --tag wcci2022
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(".")

from sklearn.metrics import average_precision_score, confusion_matrix
from torch_geometric.loader import DataLoader

from scripts.pyg_data import GridDataset, GridEnvMetadata, PreloadedGridDataset
from training.config import (DATA_DIR, DEVICE, EDGE_FEATURES, NODE_FEATURES,
                             TRAIN_CONFIG)
from training.train_gnn import GridGNN, compute_normalization_stats

CKPT = "gnn_checkpoint_n1.pt"
HOME = "neurips2020"


def best_f1_and_thr(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score)
    ys = y[order]
    tp, fp, P = np.cumsum(ys), np.cumsum(1 - ys), ys.sum()
    if P == 0:
        return 0.0, 0.0
    prec, rec = tp / np.maximum(tp + fp, 1), tp / P
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
    i = int(np.argmax(f1))
    return float(f1[i]), float(score[order][i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="case14")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_n1.jsonl")
    meta_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_n1_meta.json")
    for p in (jsonl, meta_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}\nGenerate it:\n"
                     f"  python scripts/generate_dataset.py --env <env> --task n1")
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Missing {args.checkpoint} — train the n1 model first:\n"
                 f"  GRID_TASK=n1 python training/train_gnn.py")

    # ── 36-bus normalization stats (train split only) ────────────────────────
    print("Computing 36-bus n1 train-split normalization stats...")
    home_pt = os.path.join(DATA_DIR, "processed_grid_data_n1.pt")
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    home = PreloadedGridDataset(home_pt, device=DEVICE)
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(home, train_idx)
    del home

    # ── foreign dataset (lazy; its own topology meta) ────────────────────────
    with open(meta_path) as f:
        meta_dict = json.load(f)
    meta = GridEnvMetadata(meta_dict)
    n_records = sum(1 for line in open(jsonl) if line.strip())

    if args.tag == HOME:
        # in-distribution: score the held-out split, never the training frames
        test_path = os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy")
        indices = np.load(test_path) if os.path.exists(test_path) else np.arange(n_records)
        scope = f"held-out test split ({len(indices):,} of {n_records:,} frames)"
    else:
        indices = np.arange(n_records)
        scope = f"all {n_records:,} frames (topology never seen in training)"

    ds = GridDataset(jsonl, indices, meta)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GridGNN(
        node_features=NODE_FEATURES, edge_features=EDGE_FEATURES,
        hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    logits_all, y_all, rho_all, rho_glob = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch.x = (batch.x - node_mean) / node_std
            batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
            line_logits = model(batch.x, batch.edge_index, batch.edge_attr,
                                batch.edge_fwd)
            m = batch.line_mask
            fwd_attr = batch.edge_attr[batch.edge_fwd]
            logits_all.append(line_logits[m].float().cpu())
            y_all.append(batch.line_y[m].cpu())
            # rho is edge feature 0; de-normalize so the rule baseline is a rule
            # on a PHYSICAL quantity, not on a z-score
            rho = fwd_attr[m, 0].float().cpu() * edge_std[0].cpu() + edge_mean[0].cpu()
            rho_all.append(rho)

    logits = torch.cat(logits_all).numpy()
    y = torch.cat(y_all).numpy().astype(int)
    rho = torch.cat(rho_all).numpy()

    p = y.mean()
    allpos = 2 * p / (1 + p) if p else 0.0
    f1_model, thr = best_f1_and_thr(logits, y)
    f1_rule, _ = best_f1_and_thr(rho, y)

    print(f"\n{'=' * 72}")
    print(f"  N-1 screening — {args.tag}   ({meta.n_line} lines, {meta.n_sub} buses)")
    print(f"  {scope}")
    print(f"{'=' * 72}")
    print(f"  contingencies scored : {len(y):,}")
    print(f"  violation rate       : {p:.1%}")
    print(f"\n  all-positive baseline           : {allpos:.4f}")
    print(f"  best rule on rho of removed line: {f1_rule:.4f}   "
          f"({f1_rule / max(allpos, 1e-9):.2f}x baseline)")
    print(f"  MODEL                           : {f1_model:.4f}   "
          f"({f1_model / max(f1_rule, 1e-9):.2f}x the rule)   AP {average_precision_score(y, logits):.4f}")

    preds = (logits > thr).astype(int)
    tn, fp_, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    print(f"\n  at the best threshold:")
    print(f"    true violation caught   : {tp:,} / {tp + fn:,}  (recall {tp / max(tp + fn, 1):.3f})")
    print(f"    MISSED violations       : {fn:,}   <- the dangerous error")
    print(f"    false alarms            : {fp_:,}  (precision {tp / max(tp + fp_, 1):.3f})")

    if f1_model <= f1_rule:
        print(f"\n  ** The model does NOT beat the single-rule baseline here. **")
        print(f"     Report it as such — on this topology the neural component is "
              f"not adding\n     anything a rule cannot state.")


if __name__ == "__main__":
    main()
