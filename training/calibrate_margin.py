"""
Lever A — post-hoc per-class logit-margin calibration (no retraining).

Diagnosis (see supplimentary_docs/normal_recall_experiments.md, round 2): `line_trip` is an
over-predicted "magnet" class (precision 0.648, recall 0.915). Pure argmax over-commits to it and
absorbs 3,043 of 9,128 normals. This script searches a fixed additive offset vector `b` applied to
the logits at inference (`pred = argmax(logits + b)`), tuned on the VAL split, then reports on TEST.

b[line_trip] < 0 (and/or b[normal] > 0) de-biases the magnet. We pick the offset on val that
maximizes macro-F1 subject to the STRICT keep-what-works bar, then print the test report so it can be
compared against the baseline (macro 0.7830, normal recall 0.6466, others F1 0.8157/0.7588/0.8095).

Reuses gnn_checkpoint_best.pt — no model change, no reprocess.
"""
import os, sys, json, argparse
import numpy as np
import torch
sys.path.append(".")

from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from training.train_gnn import GridGNN, compute_normalization_stats
from training.config import DEVICE, DATA_FILE, DATA_DIR, NODE_FEATURES, EDGE_FEATURES, TRAIN_CONFIG
from scripts.pyg_data import PreloadedGridDataset

# ── CLI (Round 3): calibrate an arbitrary checkpoint to its own margin file, so an
# experiment never clobbers the deployed gnn_logit_margin.json until it wins the bar.
# Defaults reproduce the original hard-coded behavior exactly.
_parser = argparse.ArgumentParser(description="Post-hoc logit-margin calibration.")
_parser.add_argument("--checkpoint", default="gnn_checkpoint_best.pt",
                     help="model state_dict to calibrate (default: gnn_checkpoint_best.pt)")
_parser.add_argument("--out", default="gnn_logit_margin.json",
                     help="where to write the chosen margin (default: gnn_logit_margin.json)")
_parser.add_argument("--gsat", action="store_true", default=False,
                     help="checkpoint was GSAT-trained (build a gate-matched model so state_dict loads)")
args = _parser.parse_args()

# ── Label map ────────────────────────────────────────────────────────────────
meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
with open(meta_path) as f:
    meta_dict = json.load(f)
LABEL_MAP_ACTIVE = meta_dict["label_map"]
n_classes = meta_dict["n_classes"]
idx_to_label = {v: k for k, v in LABEL_MAP_ACTIVE.items()}
target_names = [idx_to_label[i] for i in range(n_classes)]
LT = LABEL_MAP_ACTIVE["line_trip"]
NM = LABEL_MAP_ACTIVE["normal"]

# ── Data + normalization (identical to evaluate.py) ──────────────────────────
train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_train_idx.npy"))
val_idx   = np.load(os.path.join(DATA_DIR, "split_neurips2020_val_idx.npy"))
test_idx  = np.load(os.path.join(DATA_DIR, "split_neurips2020_test_idx.npy"))

full_dataset = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data.pt"), device=DEVICE)
print("Computing normalization from train split...")
node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)
full_dataset._data.x         = (full_dataset._data.x         - node_mean) / node_std
full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std

# ── Model ────────────────────────────────────────────────────────────────────
model = GridGNN(
    node_features=NODE_FEATURES, edge_features=EDGE_FEATURES, n_classes=n_classes,
    hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
    dropout=TRAIN_CONFIG["dropout"],
    gsat_enabled=args.gsat, gsat_tau=TRAIN_CONFIG.get("gsat_tau", 1.0),
).to(DEVICE)
print(f"Loading checkpoint: {args.checkpoint}")
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()


@torch.no_grad()
def collect_logits(indices):
    loader = DataLoader(full_dataset[indices], batch_size=256, shuffle=False)
    logits_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        logits_all.append(logits.cpu().numpy())
        labels_all.append(batch.y.cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


print("Collecting logits for val and test splits...")
val_logits, val_y   = collect_logits(val_idx)
test_logits, test_y = collect_logits(test_idx)


def metrics(logits, y, offset):
    preds = (logits + offset).argmax(axis=1)
    per_f1 = f1_score(y, preds, average=None, labels=list(range(n_classes)), zero_division=0)
    macro  = f1_score(y, preds, average="macro", zero_division=0)
    rec    = recall_score(y, preds, average=None, labels=list(range(n_classes)), zero_division=0)
    return macro, per_f1, rec, preds


# ── Baseline (offset 0) on val for the strict-bar reference ──────────────────
zero = np.zeros(n_classes, dtype=np.float32)
val_macro0, val_f10, val_rec0, _ = metrics(val_logits, val_y, zero)
print("\n=== VAL baseline (offset 0) ===")
print(f"macro_f1={val_macro0:.4f}  normal_recall={val_rec0[NM]:.4f}")
print("per-class F1:", {target_names[i]: round(float(val_f10[i]), 4) for i in range(n_classes)})

# ── Sweep: negative margin on line_trip, optional positive on normal ─────────
# Strict bar (evaluated on VAL): normal recall up vs baseline AND the other three F1 each held
# within 0.02 of their baseline AND macro_f1 >= baseline macro.
lt_grid = np.round(np.arange(0.0, -2.01, -0.1), 2)
nm_grid = np.round(np.arange(0.0,  1.51,  0.1), 2)
TOL = 0.02
others = [i for i in range(n_classes) if i != NM]

best = None  # (val_macro, offset, val_rec_normal)
frontier = []
for b_lt in lt_grid:
    for b_nm in nm_grid:
        off = zero.copy()
        off[LT] = b_lt
        off[NM] = b_nm
        macro, f1s, rec, _ = metrics(val_logits, val_y, off)
        normal_up = rec[NM] > val_rec0[NM] + 1e-6
        others_hold = all(f1s[i] >= val_f10[i] - TOL for i in others)
        macro_ok = macro >= val_macro0 - 1e-6
        if normal_up and others_hold and macro_ok:
            frontier.append((macro, rec[NM], b_lt, b_nm))
            if best is None or macro > best[0]:
                best = (macro, off.copy(), rec[NM])

print(f"\n{len(frontier)} offset(s) pass the strict bar on VAL.")
if frontier:
    frontier.sort(reverse=True)
    print("Top val candidates (macro_f1, normal_recall, b_line_trip, b_normal):")
    for row in frontier[:8]:
        print(f"  macro={row[0]:.4f}  normal_rec={row[1]:.4f}  b_lt={row[2]}  b_nm={row[3]}")

if best is None:
    print("\nNo offset clears the strict bar on val. Reporting the best normal-recall-up "
          "offset for diagnostic purposes instead.")
    # diagnostic: max normal recall among macro>=baseline (relax 'others hold')
    cand = None
    for b_lt in lt_grid:
        for b_nm in nm_grid:
            off = zero.copy(); off[LT] = b_lt; off[NM] = b_nm
            macro, f1s, rec, _ = metrics(val_logits, val_y, off)
            if macro >= val_macro0 - 1e-6 and (cand is None or rec[NM] > cand[2]):
                cand = (macro, off.copy(), rec[NM])
    best = cand

chosen_off = best[1]
print(f"\nChosen offset: " + ", ".join(f"{target_names[i]}={chosen_off[i]:+.2f}" for i in range(n_classes)))

# ── Apply chosen offset to TEST and report ───────────────────────────────────
for tag, off in [("TEST baseline (offset 0)", zero), ("TEST with chosen offset", chosen_off)]:
    preds = (test_logits + off).argmax(axis=1)
    print(f"\n===== {tag} =====")
    print(classification_report(test_y, preds, target_names=target_names, digits=4, zero_division=0))
    print("Confusion (rows=true, cols=pred):")
    print(confusion_matrix(test_y, preds))
    lt_pred = int((preds == LT).sum()); lt_true = int((test_y == LT).sum())
    print(f"line_trip predicted={lt_pred}  true={lt_true}  (over-prediction = {lt_pred - lt_true})")

# ── Persist the chosen margin — this is the ONLY producer of gnn_logit_margin.json ──
# (evaluation/eval_cross_topology.py consumes it). NOTE: re-running this script overwrites
# the file with the margin tuned on the CURRENTLY-loaded data + checkpoint, so only run it
# on the model whose margin you intend to deploy.
test_preds = (test_logits + chosen_off).argmax(axis=1)
margin_out = {target_names[i]: round(float(chosen_off[i]), 3) for i in range(n_classes)}
margin_out["order"]              = list(target_names)
margin_out["tuned_on"]           = "val"
margin_out["test_macro_f1"]      = round(float(f1_score(test_y, test_preds, average="macro", zero_division=0)), 4)
margin_out["test_normal_recall"] = round(float(
    recall_score(test_y, test_preds, average=None, labels=list(range(n_classes)), zero_division=0)[NM]), 4)
# Per-class test F1 (all four classes) so multi-seed aggregation can apply the
# noise-aware strict bar to every class, not just macro + normal recall.
_per_f1 = f1_score(test_y, test_preds, average=None, labels=list(range(n_classes)), zero_division=0)
margin_out["test_per_class_f1"] = {target_names[i]: round(float(_per_f1[i]), 4) for i in range(n_classes)}
with open(args.out, "w") as f:
    json.dump(margin_out, f, indent=2)
print(f"\nSaved chosen margin -> {args.out}")
