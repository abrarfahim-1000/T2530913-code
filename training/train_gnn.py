import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
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
        self.classifier = nn.Sequential(
            nn.Linear(last_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )
        self.localizer = nn.Sequential(
            nn.Linear(last_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        loc_logits   = self.localizer(x).squeeze(-1)
        graph_emb    = global_mean_pool(x, batch)
        class_logits = self.classifier(graph_emb)
        return class_logits, loc_logits


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0

def build_loc_targets_fast(batch):
    """
    Vectorized, no Python loop, no .item() calls — stays on device.

    Strategy:
      Each graph i has n_nodes_i nodes. fault_loc[i] is the target node
      index *within* graph i (substation index). We need the global node
      index = cumulative node offset for graph i + fault_loc[i].

      batch.ptr gives cumulative node counts: ptr[i] is the start index
      of graph i in the flattened node list. So global_idx = ptr[i] + fl[i].
      We only set the target for graphs where fault_loc >= 0.
    """
    n_nodes  = batch.x.size(0)
    device   = batch.x.device
    targets  = torch.zeros(n_nodes, dtype=torch.float, device=device)

    fault_loc = batch.fault_loc          # (B,) — already on device after batch.to(device)
    ptr       = batch.ptr                # (B+1,) cumulative node offsets

    valid_mask   = fault_loc >= 0                          # (B,) bool
    valid_graphs = valid_mask.nonzero(as_tuple=True)[0]    # indices of graphs with a fault

    if valid_graphs.numel() > 0:
        # global node index = start of graph + local fault node
        global_idx = ptr[valid_graphs] + fault_loc[valid_graphs]
        # clamp to avoid rare edge case where fault_loc >= n_nodes in that graph
        n_nodes_per_graph = ptr[1:] - ptr[:-1]            # (B,)
        max_idx = ptr[valid_graphs] + n_nodes_per_graph[valid_graphs] - 1
        global_idx = torch.min(global_idx, max_idx)
        targets.scatter_(0, global_idx, 1.0)

    return targets


def compute_normalization_stats(dataset, train_indices):
    """Compute mean and std for node and edge features using only training data."""
    device = dataset[0].x.device  # Get device from first sample

    # Initialize accumulators
    node_sum = None
    node_sq_sum = None
    node_count = 0
    edge_sum = None
    edge_sq_sum = None
    edge_count = 0

    # Accumulate statistics from training set only
    for idx in train_indices:
        data = dataset[idx]
        x = data.x  # [num_nodes, num_node_features]
        e = data.edge_attr  # [num_edges, num_edge_features]

        # Node features
        if node_sum is None:
            node_sum = x.sum(dim=0)
            node_sq_sum = (x.pow(2)).sum(dim=0)
        else:
            node_sum += x.sum(dim=0)
            node_sq_sum += (x.pow(2)).sum(dim=0)
        node_count += x.size(0)

        # Edge features
        if edge_sum is None:
            edge_sum = e.sum(dim=0)
            edge_sq_sum = (e.pow(2)).sum(dim=0)
        else:
            edge_sum += e.sum(dim=0)
            edge_sq_sum += (e.pow(2)).sum(dim=0)
        edge_count += e.size(0)

    # Compute mean and std
    if node_count > 0:
        node_mean = node_sum / node_count
        node_var = torch.clamp(node_sq_sum / node_count - node_mean.pow(2), min=0.0)
        node_std = torch.sqrt(node_var) + 1e-7  # avoid division by zero
    else:
        # Fallback (should not happen with proper data)
        node_mean = torch.zeros(dataset[0].x.size(1), device=device)
        node_std = torch.ones_like(node_mean)

    if edge_count > 0:
        edge_mean = edge_sum / edge_count
        edge_var = torch.clamp(edge_sq_sum / edge_count - edge_mean.pow(2), min=0.0)
        edge_std = torch.sqrt(edge_var) + 1e-7
    else:
        edge_mean = torch.zeros(dataset[0].edge_attr.size(1), device=device)
        edge_std = torch.ones_like(edge_mean)

    return node_mean, node_std, edge_mean, edge_std


def make_dataloader(dataset, batch_size, shuffle):
    """
    num_workers=0 on Windows (PyG + multiprocessing is broken there).
    """
    is_win         = sys.platform == "win32"

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
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device, non_blocking=True)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    # Return Macro F1-Score
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)


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
    print(f"Using device: {DEVICE}")
    print("Environment: NeurIPS 2020 Track 1 Small")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta_dict = json.load(f)
        meta = GridEnvMetadata(meta_dict)
        # Use only present classes from the actual dataset
        LABEL_MAP_ACTIVE = meta_dict["label_map"]  # already stored in meta.json
        n_classes = meta_dict["n_classes"]         # = 4 for your current dataset
    else:
        print("Warning: meta JSON not found — falling back to Grid2Op init (slow).")
        meta = GridEnvMetadata()
        LABEL_MAP_ACTIVE = LABEL_MAP # imported from pyg_data
        n_classes = len(LABEL_MAP)

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found.")
        sys.exit(1)

    # ── Splits ────────────────────────────────────────────────────────────────
    split_prefix   = "split_neurips2020"
    train_idx_path = os.path.join(DATA_DIR, f"{split_prefix}_train_idx.npy")
    val_idx_path   = os.path.join(DATA_DIR, f"{split_prefix}_val_idx.npy")

    if not (os.path.exists(train_idx_path) and os.path.exists(val_idx_path)):
        print(f"Error: Split files not found. Run: python scripts/split.py")
        sys.exit(1)

    train_idx = np.load(train_idx_path)
    val_idx   = np.load(val_idx_path)

    pt_data_path = os.path.join(DATA_DIR, "processed_grid_data.pt")

    # 1. Load the ENTIRE dataset onto the GPU once
    full_dataset = PreloadedGridDataset(pt_data_path, device=DEVICE)

    # 2. Compute normalization statistics using ONLY training data
    print("Computing normalization statistics from training set...")
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)

    # THE PHYSICS FIX: Do not normalize rho (capacity). 
    # It is already a strict physical ratio [0.0, 2.0]. 
    # Force its mean to 0 and std to 1 so the math leaves it completely unscaled.
    node_mean[3] = 0.0
    node_std[3]  = 1.0
    edge_mean[0] = 0.0
    edge_std[0]  = 1.0

    # 3. Apply normalization to the entire dataset (in-place)
    print("Applying normalization to all datasets...")

    # 3. Apply normalization to the entire dataset (in-place)
    print("Applying normalization to all datasets...")
    # We apply the math directly to the hidden _data object, which contains 
    # every node and edge in the dataset concatenated together.
    full_dataset._data.x = (full_dataset._data.x - node_mean) / node_std
    full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std

    # 4. PyG natively slices the dataset using your numpy arrays!
    train_ds = full_dataset[train_idx]
    val_ds   = full_dataset[val_idx]

    # ── Class weights ─────────────────────────────────────────────────────────
    all_labels   = load_labels(DATA_FILE)
    train_labels = [all_labels[i] for i in train_idx]
    weights      = compute_class_weights(train_labels, LABEL_MAP_ACTIVE)  # use active map
    print("Label counts:", Counter(train_labels))
    print("Class weights:", dict(zip(LABEL_MAP_ACTIVE.keys(), weights)))
    class_weights = torch.tensor(weights, dtype=torch.float, device=DEVICE)

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
    cls_loss_fn = nn.CrossEntropyLoss()
    loc_loss_fn = nn.BCEWithLogitsLoss()

    scaler = GradScaler(device=DEVICE.type)
    best_val_f1 = 0.0
    early_stopping = EarlyStopping(patience=10) # Stops if F1 doesn't improve for 10 epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # 1. Cast forward pass to mixed precision
            with autocast(device_type=DEVICE.type):
                class_logits, loc_logits = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )

                cls_loss = cls_loss_fn(class_logits, batch.y)

                has_fault = (batch.fault_loc >= 0).any()
                if has_fault:
                    loc_targets = build_loc_targets_fast(batch)
                    loc_loss = loc_loss_fn(loc_logits, loc_targets)
                else:
                    loc_loss = torch.tensor(0.0, device=DEVICE)

                loss = cls_loss + TRAIN_CONFIG["loc_loss_weight"] * loc_loss

            # 2. Scale the loss and call backward
            scaler.scale(loss).backward()

            # 3. Step the optimizer and update the scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_f1 = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "gnn_checkpoint_best.pt")
        
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered! Model has converged.")
            break

    print(f"Training complete. Best val_f1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()