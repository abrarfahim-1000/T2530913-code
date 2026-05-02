import torch
import numpy as np
import json
import os, sys
sys.path.append(".")

from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
from training.train_gnn import GridGNN
from training.config import DEVICE, DATA_FILE, DATA_DIR, NODE_FEATURES, EDGE_FEATURES, TRAIN_CONFIG
from scripts.pyg_data import PreloadedGridDataset, LABEL_MAP
from training.train_gnn import compute_normalization_stats

# ── Load active label map from meta ─────────────────────────────────────────
meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
with open(meta_path) as f:
    meta_dict = json.load(f)
LABEL_MAP_ACTIVE = meta_dict["label_map"]
n_classes = meta_dict["n_classes"]
idx_to_label = {v: k for k, v in LABEL_MAP_ACTIVE.items()}

# ── Load test split ──────────────────────────────────────────────────────────
train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_train_idx.npy"))
test_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_test_idx.npy"))

full_dataset = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data.pt"), device=DEVICE)

print("Computing normalization from train split to apply to test split...")
node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)

# THE PHYSICS FIX: Override rho so it stays physically bounded
node_mean[3] = 0.0
node_std[3]  = 1.0
edge_mean[0] = 0.0
edge_std[0]  = 1.0

# Apply the fixed math to the dataset
full_dataset._data.x = (full_dataset._data.x - node_mean) / node_std
full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std

test_ds = full_dataset[test_idx]
loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# ── Load model ───────────────────────────────────────────────────────────────
model = GridGNN(
    node_features=NODE_FEATURES,
    edge_features=EDGE_FEATURES,
    n_classes=n_classes,
    hidden_channels=TRAIN_CONFIG["hidden_channels"],
    heads=TRAIN_CONFIG["heads"],
    dropout=TRAIN_CONFIG["dropout"],
).to(DEVICE)
model.load_state_dict(torch.load("gnn_checkpoint_best.pt", map_location=DEVICE))
model.eval()

# ── Inference ────────────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

# ── Report ────────────────────────────────────────────────────────────────────
target_names = [idx_to_label[i] for i in range(n_classes)]
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0))
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_labels, all_preds))