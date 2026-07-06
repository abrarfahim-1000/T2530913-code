import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # add parent directory
# save_normalization_stats.py
import torch, os
import numpy as np
from config import DEVICE, DATA_DIR
from scripts.pyg_data import PreloadedGridDataset
from train_gnn import compute_normalization_stats

train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_train_idx.npy"))
full_dataset = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data.pt"), device=DEVICE)

node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)

torch.save({
    "node_mean": node_mean, "node_std": node_std,
    "edge_mean": edge_mean, "edge_std": edge_std,
}, "normalization_stats.pt")
print("Saved normalization_stats.pt")