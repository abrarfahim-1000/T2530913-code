"""Does message passing actually help the N-1 readout, and does that survive seeds?

`CLAUDE.md` and the chapter quote a SINGLE-SEED figure for this ablation:
validation F1 0.8987 (full model) vs 0.8411 (head-only, message-passing
embeddings dropped from the line readout) -- a +0.058 delta. The head-only
checkpoint that produced 0.8411 is not on disk (confirmed 2026-09-19), so that
number cannot be re-measured, only re-run. This is the re-run
(`supplimentary_docs/issues.md` A4).

TWO ARMS, FOUR SEEDS, ONE DIFFERENCE
-------------------------------------
Both arms train on the identical graphs, split and epoch budget, differing
only in `line_head_use_mp` (whether the line_head reads the GATv2 embeddings
or only the raw skip features). Same shape as `reactance_transfer.py`: build
once, train both arms per seed, never touch the deployed checkpoint.

Protocol is the one CLAUDE.md documents as the intended command for the
deployed model: --epochs 30 --batch_size 128 --lr 3e-4. Evaluation uses the
pinned EVAL_BATCH_SIZE (64) from eval_n1_cross_topology.py, not the training
batch size, so these numbers are directly comparable to every other reported
F1 in the chapter.

Usage:
    python evaluation/headonly_ablation.py
    python evaluation/headonly_ablation.py --seeds 42 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from evaluation.eval_n1_cross_topology import EVAL_BATCH_SIZE, best_f1_and_thr
from scripts.pyg_data import GridEnvMetadata, PreloadedGridDataset
from training.config import DATA_DIR, DEVICE, PROCESSED_PT, TRAIN_CONFIG
from training.train_gnn import GridGNN, compute_normalization_stats

HOME = "neurips2020"
OUT = os.path.join("results", "seeds", "headonly_ablation.json")
CKPT_FMT = "gnn_checkpoint_n1_ablation_{arm}_seed{seed}.pt"

ARMS = ("full", "headonly")

# The documented intended command for the deployed model (CLAUDE.md, Model
# Training). Applied identically to both arms, so it cancels out of the delta.
EPOCHS, BATCH_SIZE, LR = 30, 128, 3e-4
DEFAULT_SEEDS = (42, 0, 1, 2)


def build_split(seed: int, node_mean, node_std, edge_mean, edge_std):
    """Normalized train/val/test PyG graphs for the home grid, this seed's split.

    The split itself is chronic-level and NOT reshuffled per seed -- only the
    model init and data order vary. Re-splitting per seed would confound
    "does the network transfer across seeds" with "does it transfer across
    train/val partitions", which is a different question already answered
    by the reactance experiment's four-seed table.
    """
    pt_path = os.path.join(DATA_DIR, PROCESSED_PT)
    full_dataset = PreloadedGridDataset(pt_path, device=DEVICE)
    full_dataset._data.x = (full_dataset._data.x - node_mean) / node_std
    full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std
    full_dataset._data_list = None  # see train_gnn.py's cache-invalidation note
    return full_dataset


def train_arm(arm: str, seed: int, train_ds, val_ds) -> tuple[str, float]:
    head_only = arm == "headonly"
    torch.manual_seed(seed)
    np.random.seed(seed)

    pos = sum(int(d.line_y[d.line_mask].sum()) for d in train_ds)
    total = sum(int(d.line_mask.sum()) for d in train_ds)
    pos_weight = torch.tensor(max(total - pos, 1) / max(pos, 1), device=DEVICE)

    model = GridGNN(
        node_features=8, edge_features=8,
        hidden_channels=TRAIN_CONFIG["hidden_channels"],
        heads=TRAIN_CONFIG["heads"], dropout=TRAIN_CONFIG["dropout"],
        line_head_use_mp=not head_only,
    ).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=TRAIN_CONFIG["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    ckpt = CKPT_FMT.format(arm=arm, seed=seed)
    best = 0.0
    print(f"\n[{arm} seed={seed}] {'headonly' if head_only else 'full'}, "
          f"{len(train_ds):,} train frames, pos_weight={pos_weight.item():.3f}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_fwd)
            m = batch.line_mask
            loss = F.binary_cross_entropy_with_logits(logits[m], batch.line_y[m], pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        scores, ys = infer(model, val_loader)
        val_f1 = best_f1_and_thr(scores, ys)[0]
        if val_f1 > best:
            best = val_f1
            torch.save(model.state_dict(), ckpt)
        print(f"  epoch {epoch + 1:3d}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}"
              f"  val_f1={val_f1:.4f}{'  *' if val_f1 == best else ''}")

    print(f"[{arm} seed={seed}] best val F1 {best:.4f} -> {ckpt}")
    return ckpt, best


@torch.no_grad()
def infer(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out, ys = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_fwd)
        m = batch.line_mask
        out.append(logits[m].float().cpu())
        ys.append(batch.line_y[m].cpu())
    return torch.cat(out).numpy(), torch.cat(ys).numpy().astype(int)


def load_meta() -> GridEnvMetadata:
    with open(os.path.join(DATA_DIR, f"grid_dataset_{HOME}_n1_meta.json")) as fh:
        return GridEnvMetadata(json.load(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = ap.parse_args()
    globals()["EPOCHS"] = args.epochs

    t0 = time.time()
    pt_path = os.path.join(DATA_DIR, PROCESSED_PT)
    train_idx_path = os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy")
    val_idx_path = os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy")
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    print(f"Loading {pt_path} ({len(train_idx):,} train / {len(val_idx):,} val frames)...")
    raw_dataset = PreloadedGridDataset(pt_path, device=DEVICE)
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(raw_dataset, train_idx)
    del raw_dataset

    results: dict[str, dict] = {}
    for seed in args.seeds:
        full_dataset = build_split(seed, node_mean, node_std, edge_mean, edge_std)
        train_ds = full_dataset[train_idx]
        val_ds = full_dataset[val_idx]

        seed_entry = {}
        for arm in ARMS:
            ckpt, best_val = train_arm(arm, seed, train_ds, val_ds)
            seed_entry[arm] = {"checkpoint": ckpt, "val_f1": best_val}
        seed_entry["delta"] = seed_entry["full"]["val_f1"] - seed_entry["headonly"]["val_f1"]
        results[str(seed)] = seed_entry
        del full_dataset, train_ds, val_ds
        print(f"[seed {seed}] full={seed_entry['full']['val_f1']:.4f}  "
              f"headonly={seed_entry['headonly']['val_f1']:.4f}  "
              f"delta={seed_entry['delta']:+.4f}")

    full_scores = [results[str(s)]["full"]["val_f1"] for s in args.seeds]
    head_scores = [results[str(s)]["headonly"]["val_f1"] for s in args.seeds]
    deltas = [results[str(s)]["delta"] for s in args.seeds]

    report = {
        "protocol": {
            "epochs": args.epochs, "batch_size": BATCH_SIZE, "lr": LR,
            "eval_batch_size": EVAL_BATCH_SIZE, "seeds": args.seeds,
            "device": str(DEVICE), "grid": HOME, "split": "chronic-level, fixed across seeds",
        },
        "per_seed": results,
        "summary": {
            "full_mean": float(np.mean(full_scores)), "full_std": float(np.std(full_scores)),
            "full_range": [float(min(full_scores)), float(max(full_scores))],
            "headonly_mean": float(np.mean(head_scores)), "headonly_std": float(np.std(head_scores)),
            "headonly_range": [float(min(head_scores)), float(max(head_scores))],
            "delta_mean": float(np.mean(deltas)), "delta_std": float(np.std(deltas)),
            "delta_range": [float(min(deltas)), float(max(deltas))],
        },
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)

    s = report["summary"]
    print(f"\n{'=' * 72}")
    print(f"  full     : {s['full_mean']:.4f} +/- {s['full_std']:.4f}  range {s['full_range']}")
    print(f"  headonly : {s['headonly_mean']:.4f} +/- {s['headonly_std']:.4f}  range {s['headonly_range']}")
    print(f"  delta    : {s['delta_mean']:+.4f} +/- {s['delta_std']:.4f}  range {s['delta_range']}")
    print(f"{'=' * 72}")
    print(f"  written: {args.json}")


if __name__ == "__main__":
    main()
