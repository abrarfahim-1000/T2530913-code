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
#
# Two named configs. IMPORTANT (Round 3 finding): the DEPLOYED Lever A checkpoint
# (gnn_checkpoint_best.pt / gnn_checkpoint_leverA.pt, test macro F1 0.8277, normal
# recall 0.8806) was trained on SMALL_CONFIG [16,32,32]/[4,4,1] — verified from the
# checkpoint's tensor shapes (conv1.att (1,4,16), classifier.0 in-dim 96=32*3). ALL
# Round 1/2 fragility evidence was measured on this same config, so it transfers.
#
# CUDA_CONFIG [64,128,128] has NOT produced a validated checkpoint and is documented
# to overfit/collapse under this schedule. To reproduce or extend the baseline on a
# CUDA machine you MUST force the small config:  GRID_CONFIG=small python ...
# The auto-by-device default below is kept for backwards compatibility only.

SMALL_CONFIG = {
    # Personal PC — best-performing / DEPLOYED config. Its limited capacity acts as
    # regularization; scaling up to [64,128,128] was tried and OVERFITS/COLLAPSES
    # (train loss drops while val F1 falls; normal+cascade go to 0.0).
    "epochs": 10,
    "batch_size": 512,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "dropout": 0.0,            # Explicitly disabled for 100% determinism
    "hidden_channels": [16, 32, 32],
    "heads": [4, 4, 1],
    "loc_loss_weight": 0.5,
    # ── Round 3: soft-F1 hybrid loss (Phase 1). Default OFF = current behavior. ──
    # lambda_f1=0.0 makes soft_f1_loss a no-op (one-line revert). See train_gnn.py.
    "lambda_f1": 0.0,         # additive weight on the differentiable soft-macro-F1 term
    "f1_warmup_frac": 0.3,    # fraction of epochs of pure-CE warm-up before blending F1 in
    "lambda_ramp_epochs": 0,  # linear ramp of lambda_f1 over N epochs after warm-up (0 = hard step)
    # ── Round 3: GSAT (Graph Stochastic Attention, Miao et al. ICML 2022). Default OFF. ──
    # gsat_enabled=False (or beta=0.0) makes the edge gate absent / inert (one-line revert).
    # See StochasticEdgeGate / info_bottleneck_kl in train_gnn.py and gsat_lsgat_handoff.md §4.
    "gsat_enabled": False,    # build+apply the stochastic edge gate before the conv stack
    "beta": 0.0,              # weight on the information-bottleneck KL term
    "r": 0.6,                 # Bernoulli prior for the IB KL (target edge-keep rate)
    "gsat_tau": 1.0,          # Gumbel-softmax temperature (training-time gate sampling)
    "beta_warmup_frac": 0.3,  # fraction of epochs of pure-task warm-up before ramping beta in
}

CUDA_CONFIG = {
    # Research PC — full scale. UNVALIDATED for deployment (see note above).
    "epochs": 50,
    "batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "dropout": 0.0,
    "hidden_channels": [64, 128, 128],
    "heads": [2, 2, 1],
    "loc_loss_weight": 0.5,
    "lambda_f1": 0.0,
    "f1_warmup_frac": 0.3,
    "lambda_ramp_epochs": 0,
    "gsat_enabled": False,
    "beta": 0.0,
    "r": 0.6,
    "gsat_tau": 1.0,
    "beta_warmup_frac": 0.3,
}

# Selection: explicit GRID_CONFIG override wins; otherwise auto-by-device (legacy).
_override = os.environ.get("GRID_CONFIG", "").strip().lower()
if _override in ("small", "personal"):
    TRAIN_CONFIG = SMALL_CONFIG
    _config_name = "SMALL_CONFIG (forced via GRID_CONFIG)"
elif _override in ("cuda", "large", "big", "research"):
    TRAIN_CONFIG = CUDA_CONFIG
    _config_name = "CUDA_CONFIG (forced via GRID_CONFIG)"
elif DEVICE.type == "cuda":
    TRAIN_CONFIG = CUDA_CONFIG
    _config_name = "CUDA_CONFIG (auto-by-device)"
else:
    TRAIN_CONFIG = SMALL_CONFIG
    _config_name = "SMALL_CONFIG (auto-by-device)"

if _config_name.startswith("CUDA_CONFIG"):
    print(
        f"[config] Using {_config_name}. WARNING: the deployed Lever A baseline was "
        f"trained on SMALL_CONFIG. Set GRID_CONFIG=small to reproduce/extend it."
    )
else:
    print(f"[config] Using {_config_name}.")

# ── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
# These dimensions are fixed by the GridDataset implementation in pyg_data.py
NODE_FEATURES = 5  # load_p, mean_v, max_rho, connected_line_frac, global_trip_frac
EDGE_FEATURES = 4  # rho, p_or, q_or, near_limit

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
