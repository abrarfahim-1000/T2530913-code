"""
Cross-topology evaluation — does the model (and the Lever A logit margin) transfer to an UNSEEN
topology? The GNN classifier head is topology-agnostic (global pooling over nodes), so the 36-bus
checkpoint runs unchanged on 14-bus / 118-bus graphs. Foreign features are normalized with the
**36-bus train-split stats** (physical quantities share scale across topologies — see CLAUDE.md).

Reports the foreign-topology test metrics BOTH uncalibrated (pure argmax) and calibrated (with the
val-tuned Lever A margin from gnn_logit_margin.json), so we can see whether the margin still helps,
hurts, or is neutral off the topology it was tuned on.

Usage:
    python evaluation/eval_cross_topology.py --tag case14
    python evaluation/eval_cross_topology.py --tag wcci2022
"""
import os, sys, json, argparse
import numpy as np
import torch
sys.path.append(".")

from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from training.train_gnn import GridGNN, compute_normalization_stats
from training.config import DEVICE, DATA_DIR, NODE_FEATURES, EDGE_FEATURES, TRAIN_CONFIG
from scripts.pyg_data import PreloadedGridDataset, GridDataset, GridEnvMetadata, LABEL_MAP

CKPT       = "gnn_checkpoint_best.pt"
MARGIN_FILE = "gnn_logit_margin.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="case14", help="dataset tag → data/grid_dataset_<tag>.jsonl")
    ap.add_argument("--checkpoint", default=CKPT, help=f"model state_dict (default: {CKPT})")
    ap.add_argument("--margin", default=MARGIN_FILE, help=f"logit-margin json (default: {MARGIN_FILE})")
    ap.add_argument("--gsat", action="store_true", default=False,
                    help="checkpoint was GSAT-trained (build a gate-matched model so state_dict loads)")
    args = ap.parse_args()

    idx_to_label = {v: k for k, v in LABEL_MAP.items()}
    n_classes    = len(LABEL_MAP)
    target_names = [idx_to_label[i] for i in range(n_classes)]

    # ── 36-bus normalization stats (train split only) ────────────────────────
    print("Computing 36-bus train-split normalization stats...")
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_train_idx.npy"))
    home = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data.pt"), device=DEVICE)
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(home, train_idx)
    del home  # free memory; foreign graphs are built lazily below

    # ── Lever A margin ───────────────────────────────────────────────────────
    with open(args.margin) as f:
        m = json.load(f)
    offset = torch.tensor([m[idx_to_label[i]] for i in range(n_classes)],
                          dtype=torch.float32, device=DEVICE)
    print(f"Lever A margin: " + ", ".join(f"{idx_to_label[i]}={offset[i]:+.2f}" for i in range(n_classes)))

    # ── Foreign-topology dataset (lazy, with its own meta) ───────────────────
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}.jsonl")
    meta_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_meta.json")
    if not (os.path.exists(jsonl) and os.path.exists(meta_path)):
        sys.exit(f"Missing {jsonl} or {meta_path}. Generate it first:\n"
                 f"  python scripts/generate_dataset.py --env <env> --n_records 15000")
    with open(meta_path) as f:
        foreign_meta = GridEnvMetadata(json.load(f))
    n_lines = sum(1 for _ in open(jsonl, "r"))
    print(f"[{args.tag}] n_sub={foreign_meta.n_sub} n_line={foreign_meta.n_line} records={n_lines}")
    ds = GridDataset(jsonl, np.arange(n_lines), foreign_meta)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model = GridGNN(
        node_features=NODE_FEATURES, edge_features=EDGE_FEATURES, n_classes=n_classes,
        hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
        gsat_enabled=args.gsat, gsat_tau=TRAIN_CONFIG.get("gsat_tau", 1.0),
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    # ── Inference (normalize foreign features with 36-bus stats) ─────────────
    logits_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch.x         = (batch.x         - node_mean) / node_std
            batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits_all.append(logits.cpu())
            y_all.append(batch.y.cpu())
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all).numpy()

    LT = LABEL_MAP["line_trip"]
    for label, off in [("UNCALIBRATED (argmax)", torch.zeros(n_classes)),
                       ("CALIBRATED (Lever A margin)", offset.cpu())]:
        preds = (logits + off).argmax(dim=1).numpy()
        print(f"\n===== [{args.tag}] {label} =====")
        print(classification_report(y, preds, target_names=target_names, digits=4, zero_division=0))
        print("Confusion (rows=true, cols=pred):")
        print(confusion_matrix(y, preds, labels=list(range(n_classes))))
        print(f"line_trip predicted={(preds == LT).sum()}  true={(y == LT).sum()}")


if __name__ == "__main__":
    main()
