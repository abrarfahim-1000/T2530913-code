# run this as a standalone script — diag.py
import torch
import numpy as np
import os, sys
sys.path.append(".")

from scripts.pyg_data import PreloadedGridDataset, LABEL_MAP
from torch_geometric.loader import DataLoader

pt_path = "data/processed_grid_data.pt"
train_idx = np.load("data/split_neurips2020_train_idx.npy")

ds = PreloadedGridDataset(pt_path)
train_ds = ds[train_idx]

loader = DataLoader(train_ds, batch_size=4, shuffle=False)
batch = next(iter(loader))

print("=== BATCH INSPECTION ===")
print(f"x shape      : {batch.x.shape}")
print(f"x sample     : {batch.x[0]}")
print(f"edge_attr    : {batch.edge_attr[0]}")
print(f"y (labels)   : {batch.y}")
print(f"fault_loc    : {batch.fault_loc}")
print(f"x has nan    : {torch.isnan(batch.x).any()}")
print(f"x has inf    : {torch.isinf(batch.x).any()}")
print(f"x all zeros  : {(batch.x == 0).all()}")
print(f"edge_attr zeros: {(batch.edge_attr == 0).all()}")

# Check label distribution in first 1000 samples
from collections import Counter
labels = [ds[i].y.item() for i in train_idx[:1000]]
print(f"\nLabel dist (first 1000): {Counter(labels)}")