import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import numpy as np
import argparse
from tqdm import tqdm
try:
    from training.config import DEVICE, DATA_FILE, DATA_DIR, TRAIN_CONFIG, NODE_FEATURES, EDGE_FEATURES, SEED
except ImportError:
    from config import DEVICE, DATA_FILE, DATA_DIR, TRAIN_CONFIG, NODE_FEATURES, EDGE_FEATURES, SEED

from scripts.pyg_data import PreloadedGridDataset, GridEnvMetadata, LABEL_MAP
from scripts.split import compute_class_weights, load_labels, get_splits


class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, n_classes, hidden_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden_channels[0], heads=heads[0],
                             edge_dim=edge_features, dropout=dropout)
        self.conv2 = GATConv(hidden_channels[0] * heads[0], hidden_channels[1],
                             heads=heads[1], edge_dim=edge_features, dropout=dropout)
        self.conv3 = GATConv(hidden_channels[1] * heads[1], hidden_channels[2],
                             heads=heads[2], edge_dim=edge_features, dropout=dropout)

        last_dim = hidden_channels[2] * heads[2]

        # FIX: classifier takes last_dim only — no skip connection, no concatenation
        self.classifier = nn.Sequential(
            nn.Linear(last_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self.localizer = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x_emb = F.elu(self.conv1(x, edge_index, edge_attr))
        x_emb = F.dropout(x_emb, p=0.1, training=self.training)
        x_emb = F.elu(self.conv2(x_emb, edge_index, edge_attr))
        x_emb = F.dropout(x_emb, p=0.1, training=self.training)
        x_emb = self.conv3(x_emb, edge_index, edge_attr)

        loc_logits   = self.localizer(x_emb).squeeze(-1)

        # FIX: max pooling preserves the fault-node signal; mean pooling buries it
        graph_emb    = global_max_pool(x_emb, batch)
        class_logits = self.classifier(graph_emb)

        return class_logits, loc_logits


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter    = 0


def build_loc_targets_fast(batch):
    n_nodes  = batch.x.size(0)
    device   = batch.x.device
    targets  = torch.zeros(n_nodes, dtype=torch.float, device=device)

    fault_loc = batch.fault_loc
    ptr       = batch.ptr

    valid_mask   = fault_loc >= 0
    valid_graphs = valid_mask.nonzero(as_tuple=True)[0]

    if valid_graphs.numel() > 0:
        global_idx        = ptr[valid_graphs] + fault_loc[valid_graphs]
        n_nodes_per_graph = ptr[1:] - ptr[:-1]
        max_idx           = ptr[valid_graphs] + n_nodes_per_graph[valid_graphs] - 1
        global_idx        = torch.min(global_idx, max_idx)
        targets.scatter_(0, global_idx, 1.0)

    return targets


def compute_normalization_stats(dataset, train_indices):
    """Compute mean/std from train split only, in float64 to avoid precision loss."""
    device = dataset[0].x.device

    node_sum    = torch.zeros(dataset[0].x.size(1),        dtype=torch.float64, device=device)
    node_sq_sum = torch.zeros(dataset[0].x.size(1),        dtype=torch.float64, device=device)
    edge_sum    = torch.zeros(dataset[0].edge_attr.size(1), dtype=torch.float64, device=device)
    edge_sq_sum = torch.zeros(dataset[0].edge_attr.size(1), dtype=torch.float64, device=device)
    node_count  = 0
    edge_count  = 0

    for idx in train_indices:
        data = dataset[idx]
        x    = data.x.to(torch.float64)
        e    = data.edge_attr.to(torch.float64)

        node_sum    += x.sum(dim=0)
        node_sq_sum += x.pow(2).sum(dim=0)
        node_count  += x.size(0)

        edge_sum    += e.sum(dim=0)
        edge_sq_sum += e.pow(2).sum(dim=0)
        edge_count  += e.size(0)

    n_mean   = node_sum / node_count
    n_var    = torch.clamp(node_sq_sum / node_count - n_mean.pow(2), min=0.0)
    node_mean = n_mean.to(torch.float32)
    node_std  = torch.sqrt(n_var).to(torch.float32) + 1e-7

    e_mean   = edge_sum / edge_count
    e_var    = torch.clamp(edge_sq_sum / edge_count - e_mean.pow(2), min=0.0)
    edge_mean = e_mean.to(torch.float32)
    edge_std  = torch.sqrt(e_var).to(torch.float32) + 1e-7

    return node_mean, node_std, edge_mean, edge_std


def make_dataloader(dataset, batch_size, shuffle):
    is_win      = sys.platform == "win32"
    num_workers = 0 if is_win else 4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


@torch.no_grad()
def evaluate(model, loader, device, label_names=None):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device, non_blocking=True)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if label_names:
        print(classification_report(
            all_labels, all_preds,
            target_names=label_names,
            digits=4, zero_division=0
        ))

    return macro_f1


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    args = parser.parse_args()

    epochs     = args.epochs     if args.epochs     is not None else TRAIN_CONFIG["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else TRAIN_CONFIG["batch_size"]
    lr         = args.lr         if args.lr         is not None else TRAIN_CONFIG["lr"]

    torch.manual_seed(SEED)
    print(f"Using device : {DEVICE}")
    print(f"NODE_FEATURES: {NODE_FEATURES}  EDGE_FEATURES: {EDGE_FEATURES}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
    with open(meta_path) as f:
        meta_dict = json.load(f)
    meta             = GridEnvMetadata(meta_dict)
    LABEL_MAP_ACTIVE = meta_dict["label_map"]
    n_classes        = meta_dict["n_classes"]
    label_names      = [k for k, v in sorted(LABEL_MAP_ACTIVE.items(), key=lambda x: x[1])]

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        sys.exit(1)

    # ── Splits ────────────────────────────────────────────────────────────────
    split_prefix   = "split_neurips2020"
    train_idx_path = os.path.join(DATA_DIR, f"{split_prefix}_train_idx.npy")
    val_idx_path   = os.path.join(DATA_DIR, f"{split_prefix}_val_idx.npy")

    if not (os.path.exists(train_idx_path) and os.path.exists(val_idx_path)):
        print("Split files not found. Run: python scripts/split.py")
        sys.exit(1)

    train_idx = np.load(train_idx_path)
    val_idx   = np.load(val_idx_path)

    # ── Dataset ───────────────────────────────────────────────────────────────
    pt_data_path = os.path.join(DATA_DIR, "processed_grid_data.pt")
    full_dataset = PreloadedGridDataset(pt_data_path, device=DEVICE)

    # Sanity check — catch stale .pt immediately
    actual_node_feats = full_dataset[0].x.shape[1]
    actual_edge_feats = full_dataset[0].edge_attr.shape[1]
    assert actual_node_feats == NODE_FEATURES, \
        f"Stale .pt: node features={actual_node_feats}, expected {NODE_FEATURES}. Delete processed_grid_data.pt and rerun preprocess.py"
    assert actual_edge_feats == EDGE_FEATURES, \
        f"Stale .pt: edge features={actual_edge_feats}, expected {EDGE_FEATURES}. Delete processed_grid_data.pt and rerun preprocess.py"

    # ── Normalization (train split only) ──────────────────────────────────────
    print("Computing normalization statistics from training set...")
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)

    # FIX: normalize ALL features uniformly — do NOT exempt rho.
    # The "physics fix" was wrong: z-scoring rho is fine and consistent.
    # Exempting it created a scale mismatch inside GATConv attention.
    # connected_line_frac (index 4) has very low std when most lines are up;
    # the +1e-7 floor in compute_normalization_stats handles this safely.
    print("Node feature means  :", node_mean.tolist())
    print("Node feature stds   :", node_std.tolist())
    print("Edge feature means  :", edge_mean.tolist())
    print("Edge feature stds   :", edge_std.tolist())

    full_dataset._data.x         = (full_dataset._data.x         - node_mean) / node_std
    full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std

    train_ds = full_dataset[train_idx]
    val_ds   = full_dataset[val_idx]

    # ── Class weights ─────────────────────────────────────────────────────────
    all_labels   = load_labels(DATA_FILE)
    train_labels = [all_labels[i] for i in train_idx]
    weights      = compute_class_weights(train_labels, LABEL_MAP_ACTIVE)
    print("Label counts :", Counter(train_labels))
    print("Class weights:", dict(zip(LABEL_MAP_ACTIVE.keys(), weights)))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = make_dataloader(train_ds, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GridGNN(
        node_features=NODE_FEATURES,
        edge_features=EDGE_FEATURES,
        n_classes=n_classes,
        hidden_channels=TRAIN_CONFIG["hidden_channels"],
        heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)

    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs)
    loc_loss_fn = nn.BCEWithLogitsLoss()
    scaler      = GradScaler(device=DEVICE.type)

    best_val_f1   = 0.0
    early_stopping = EarlyStopping(patience=15)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type):
                class_logits, loc_logits = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                # FIX: cast weights to match autocast dtype explicitly
                w        = class_weights.to(class_logits.dtype)
                cls_loss = F.cross_entropy(class_logits, batch.y, weight=w)

                # FIX: cast loc_targets to match loc_logits dtype to avoid silent mismatch
                has_fault = (batch.fault_loc >= 0).any()
                if has_fault:
                    loc_targets = build_loc_targets_fast(batch).to(loc_logits.dtype)
                    loc_loss    = loc_loss_fn(loc_logits, loc_targets)
                else:
                    loc_loss = torch.tensor(0.0, dtype=class_logits.dtype, device=DEVICE)

                loss = cls_loss + TRAIN_CONFIG["loc_loss_weight"] * loc_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Print per-class breakdown every 5 epochs so you can see what's learning
        verbose = ((epoch + 1) % 5 == 0) or (epoch == 0)
        val_f1  = evaluate(model, val_loader, DEVICE,
                           label_names=label_names if verbose else None)
        print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "gnn_checkpoint_best.pt")
            print(f"  ✓ New best saved ({best_val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best val_macro_f1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()