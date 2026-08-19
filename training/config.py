import torch
import os

# ── DEVICE CONFIGURATION ─────────────────────────────────────────────────────
def get_best_device():
    # GRID_DEVICE forces a backend. Added while diagnosing the n1 training
    # stall: the edge-level head gathers node embeddings (`x[src]`) on every
    # forward, whose backward is a scatter-add — a code path the graph-level
    # heads never exercised. Being able to re-run the identical job on CPU is
    # what distinguishes a backend bug from a modelling problem.
    forced = os.environ.get("GRID_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
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

# ── ARTIFACTS ────────────────────────────────────────────────────────────────
# N-1 contingency screening is the project's only task. The `_n1` suffix is kept
# on every filename deliberately: it names the domain concept, and it stops these
# artifacts colliding with the retired classify dataset/splits still on disk.
_SUFFIX = "_n1"
DATA_FILE = os.path.join(DATA_DIR, f"grid_dataset_neurips2020{_SUFFIX}.jsonl")

CHECKPOINT_FILE = f"gnn_checkpoint{_SUFFIX}.pt"
NORM_STATS_FILE = f"normalization_stats{_SUFFIX}.pt"
PROCESSED_PT    = f"processed_grid_data{_SUFFIX}.pt"
SPLIT_PREFIX    = f"split_neurips2020{_SUFFIX}"

# ── HYPERPARAMETERS ──────────────────────────────────────────────────────────
# NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]

# Auto-scale config based on device
if DEVICE.type == "cuda":
    # Research PC — full scale
    TRAIN_CONFIG = {
        "epochs": 50,
        "batch_size": 256,
        "lr": 5e-4,             # Increased for GraphNorm
        "weight_decay": 1e-5, # Reduced to prevent over-regularization
        "dropout": 0.0,         # Explicitly disabled for 100% determinism
        "hidden_channels": [64, 128, 128], # Reduced to prevent memorization
        "heads": [2, 2, 1],
        "loc_loss_weight": 0.5
    }
else:
    # Personal PC — best-performing config to date (Exp 1 = macro F1 0.795).
    # NOTE: scaling up to [64,128,128] was tried and OVERFITS/COLLAPSES under this schedule
    # (train loss drops while val F1 falls; normal+cascade go to 0.0). Kept at the proven
    # small config — its limited capacity acts as regularization.
    TRAIN_CONFIG = {
        "epochs": 10,
        "batch_size": 512,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.0,         # Explicitly disabled for 100% determinism
        "hidden_channels": [16, 32, 32], # Reduced to prevent memorization
        "heads": [4, 4, 1],
        "loc_loss_weight": 0.5
    }

# ── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
# Fixed by the feature builders in scripts/pyg_data.py — see the N1_*_FEATURES
# block there for why each feature is present.
#   node: load_p, mean_v, max_rho, connected_line_frac, global_trip_frac,
#         sum_headroom, sum_abs_p, degree
#   edge: rho, p_or, q_or, near_limit, |p_or|, |q_or|, apparent_s, headroom
NODE_FEATURES, EDGE_FEATURES = 8, 8

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
