import torch
import os

# ── DEVICE CONFIGURATION ─────────────────────────────────────────────────────
def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_best_device()

# ── DATA PATHS ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_FILE = os.path.join(DATA_DIR, "grid_dataset_neurips2020.jsonl")

# ── HYPERPARAMETERS ──────────────────────────────────────────────────────────
# NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]

# Auto-scale config based on device
if DEVICE.type == "cuda":
    # Research PC — full scale
    TRAIN_CONFIG = {
        "epochs": 50,
        "batch_size": 256,
        "lr": 5e-4,             # Increased for GraphNorm
        "weight_decay": 1e-4,
        "dropout": 0.0,         # Explicitly disabled for 100% determinism
        "hidden_channels": [64, 128, 128], # Reduced to prevent memorization
        "heads": [2, 2, 1],
        "loc_loss_weight": 0.3
    }
else:
    # Personal PC — smoke test only
    TRAIN_CONFIG = {
        "epochs": 10,
        "batch_size": 256,
        "lr": 5e-4,             # Increased for GraphNorm
        "weight_decay": 1e-4,
        "dropout": 0.0,         # Explicitly disabled for 100% determinism
        "hidden_channels": [64, 128, 128], # Reduced to prevent memorization
        "heads": [2, 2, 1],
        "loc_loss_weight": 0.3
    }

# ── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
# These dimensions are fixed by the GridDataset implementation in pyg_data.py
NODE_FEATURES = 5  # load_p, gen_p, mean_v_or, max_rho
EDGE_FEATURES = 4  # rho, p_or, q_or

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
