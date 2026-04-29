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

# ── Load active label map from meta ─────────────────────────────────────────
meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
with open(meta_path) as f:
    meta_dict = json.load(f)
LABEL_MAP_ACTIVE = meta_dict["label_map"]
n_classes = meta_dict["n_classes"]
idx_to_label = {v: k for k, v in LABEL_MAP_ACTIVE.items()}

# ── Load test split ──────────────────────────────────────────────────────────
test_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_test_idx.npy"))
ds = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data.pt"))
test_ds = ds[test_idx]
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
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_labels, all_preds))