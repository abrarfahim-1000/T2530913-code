"""
Trains the N-1 contingency-screening GNN â€” the project's only model.

For each energized line: if that line trips right now, does the grid violate a
thermal limit? The prediction is EDGE-level (one logit per line), not
graph-level, because 100% of frames are mixed â€” within a single frame some
contingencies violate and others do not, so a pooled graph embedding cannot
express the answer.

The 4-class classifier and the binary forecast model that preceded this were
built, measured to be degenerate, and retired; see
supplimentary_docs/revised_thesis_claim.md Â§2. Their code is in git history.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATv2Conv
from tqdm import tqdm

try:
    from training.config import (CHECKPOINT_FILE, DATA_DIR, DATA_FILE, DEVICE,
                                 EDGE_FEATURES, NODE_FEATURES, PROCESSED_PT,
                                 SEED, SPLIT_PREFIX, TRAIN_CONFIG)
except ImportError:
    from config import (CHECKPOINT_FILE, DATA_DIR, DATA_FILE, DEVICE,
                        EDGE_FEATURES, NODE_FEATURES, PROCESSED_PT,
                        SEED, SPLIT_PREFIX, TRAIN_CONFIG)

from scripts.pyg_data import GridEnvMetadata, PreloadedGridDataset


class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, heads, dropout,
                 line_head_use_mp=True, line_head_dropout=0.0):
        super().__init__()
        # line_head_use_mp=False drops the message-passing embeddings from the
        # N-1 readout, leaving only the skip features. This is the ablation that
        # answers "is message passing helping or hurting here", and it is
        # reported in the tightening doc rather than left as a private toggle.
        self.line_head_use_mp = line_head_use_mp
        
        # GATv2Conv: dynamic attention (attention computed after concat, not before)
        # Fixes rank collapse that GATConv suffers on small graphs like 36-bus grids.
        self.conv1 = GATv2Conv(node_features, hidden_channels[0], heads=heads[0], edge_dim=edge_features)
        self.bn1   = BatchNorm(hidden_channels[0] * heads[0], track_running_stats=False)

        self.conv2 = GATv2Conv(hidden_channels[0] * heads[0], hidden_channels[1], heads=heads[1], edge_dim=edge_features)
        self.bn2   = BatchNorm(hidden_channels[1] * heads[1], track_running_stats=False)

        self.conv3 = GATv2Conv(hidden_channels[1] * heads[1], hidden_channels[2], heads=heads[2], edge_dim=edge_features)
        self.bn3   = BatchNorm(hidden_channels[2] * heads[2], track_running_stats=False)

        last_dim = hidden_channels[2] * heads[2]

        # â”€â”€ Edge-level readout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # One logit per LINE, read off the or->ex edge as
        #   [h_or || h_ex || edge_attr]
        # Nothing is pooled: the answer to "is losing line k safe" is local to k
        # and its neighbourhood, and global pooling would erase exactly the
        # per-line distinction the task is about (100% of frames are mixed â€”
        # some contingencies violate, others do not, in the same frame).
        # It is also topology-agnostic: one logit per line present, so 20-line
        # case14 and 186-line WCCI run on this checkpoint unchanged.
        # The raw endpoint features are concatenated alongside the learned
        # embeddings (a skip connection). Three rounds of message passing, batch
        # norm and ELU are free to transform `sum_headroom` and `sum_abs_p` into
        # something unrecognisable, but those two quantities ARE the physics of
        # post-contingency redistribution: line k's flow has to be absorbed by
        # the spare capacity at its endpoints. Handing them to the readout
        # unmodified means the head can form the p/spare ratio directly instead
        # of hoping it survived the encoder.
        line_in = edge_features + node_features * 2
        if line_head_use_mp:
            line_in += last_dim * 2
        self.line_head = nn.Sequential(
            nn.Linear(line_in, 128),
            nn.ReLU(),
            nn.Dropout(line_head_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(line_head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr, edge_fwd):
        """Returns one logit per line, aligned to `edge_index[:, edge_fwd]`."""
        # Live batch statistics, not running stats â€” see the BatchNorm note in
        # CLAUDE.md.
        x_emb = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x_emb = F.elu(self.bn2(self.conv2(x_emb, edge_index, edge_attr)))
        x_emb = F.elu(self.bn3(self.conv3(x_emb, edge_index, edge_attr)))

        src, dst = edge_index[0][edge_fwd], edge_index[1][edge_fwd]
        parts = [edge_attr[edge_fwd], x[src], x[dst]]
        if self.line_head_use_mp:
            parts = [x_emb[src], x_emb[dst]] + parts
        return self.line_head(torch.cat(parts, dim=1)).squeeze(-1)


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
def evaluate(model, loader, device):
    """Per-CONTINGENCY metrics for the N-1 task.

    Scored over individual (frame, line) pairs, masking the entries the power
    flow could not evaluate. Reported against the two baselines that decide
    whether the model earned anything (component_d_plan.md Â§1.1):
      - all-positive, F1 = 2p/(1+p)   â€” is the target non-vacuous?
      - `rho` of the removed line     â€” does the model beat the best local rule?
    A model that ties the second one has learned nothing a rule cannot state.
    """
    model.eval()
    logits_all, y_all, rho_all = [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device, non_blocking=True)
        line_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_fwd)
        m = batch.line_mask
        logits_all.append(line_logits[m].float().cpu())
        y_all.append(batch.line_y[m].cpu())
        rho_all.append(batch.edge_attr[batch.edge_fwd][m, 0].float().cpu())

    logits = torch.cat(logits_all).numpy()
    y = torch.cat(y_all).numpy().astype(int)
    rho = torch.cat(rho_all).numpy()

    p = y.mean() if len(y) else 0.0

    def best_f1(score):
        order = np.argsort(-score)
        ys = y[order]
        tp, fp, P = np.cumsum(ys), np.cumsum(1 - ys), ys.sum()
        if P == 0:
            return 0.0
        prec, rec = tp / np.maximum(tp + fp, 1), tp / P
        return float(np.max(np.where(prec + rec > 0,
                                     2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)))

    # The rho baseline is scored at its BEST threshold, so the model is too -
    # comparing a fixed 0.5 cut against a swept one would understate the model
    # and make the baseline look closer than it is.
    f1_at_half = f1_score(y, (logits > 0).astype(int), average="binary", zero_division=0)
    f1_best = best_f1(logits)

    print(f"  contingencies {len(y):,}   violation rate {p:.1%}")
    print(f"  model F1 {f1_best:.4f} @best-thr ({f1_at_half:.4f} @0.5)  "
          f"AP {average_precision_score(y, logits):.4f}")
    print(f"  baselines -> all-positive {2 * p / (1 + p):.4f}   "
          f"best rho-of-removed-line rule {best_f1(rho):.4f}")
    return f1_best


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--head-only', action='store_true',
                        help="Drop the message-passing embeddings from the line "
                             "readout, leaving only the skip features. Ablation for "
                             "whether message passing helps or hurts.")
    parser.add_argument('--head-dropout', type=float, default=0.0,
                        help="Dropout inside the line readout MLP.")
    parser.add_argument('--report-train', action='store_true',
                        help="Also score a slice of the TRAINING set each epoch. "
                             "The train-val gap is what separates memorisation from "
                             "an optimiser that never fitted the signal.")
    args = parser.parse_args()

    epochs     = args.epochs     if args.epochs     is not None else TRAIN_CONFIG["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else TRAIN_CONFIG["batch_size"]
    lr         = args.lr         if args.lr         is not None else TRAIN_CONFIG["lr"]

    torch.manual_seed(SEED)
    print(f"Using device : {DEVICE}")
    print(f"NODE_FEATURES: {NODE_FEATURES}  EDGE_FEATURES: {EDGE_FEATURES}")

    # â”€â”€ Metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
    with open(meta_path) as f:
        meta_dict = json.load(f)
    meta = GridEnvMetadata(meta_dict)

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Generate it:\n"
              f"  python scripts/generate_dataset.py --env neurips --task n1 --n_records 12000")
        sys.exit(1)

    # â”€â”€ Splits â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_idx_path = os.path.join(DATA_DIR, f"{SPLIT_PREFIX}_train_idx.npy")
    val_idx_path   = os.path.join(DATA_DIR, f"{SPLIT_PREFIX}_val_idx.npy")

    n_records = sum(1 for line in open(DATA_FILE) if line.strip())
    have_splits = os.path.exists(train_idx_path) and os.path.exists(val_idx_path)
    stale = False
    if have_splits:
        # A split file left over from a SMALLER run of the same task indexes out
        # of range against a bigger regeneration - or worse, silently addresses
        # the wrong frames when the new file is larger. Both are caught here.
        covered = len(np.load(train_idx_path)) + len(np.load(val_idx_path))
        test_path = os.path.join(DATA_DIR, f"{SPLIT_PREFIX}_test_idx.npy")
        if os.path.exists(test_path):
            covered += len(np.load(test_path))
        stale = covered != n_records
        if stale:
            print(f"[split] STALE: {SPLIT_PREFIX}_*.npy covers {covered:,} records "
                  f"but {DATA_FILE} now has {n_records:,} â€” rebuilding.")

    if not have_splits or stale:
        # CHRONIC-level split, never frame-level: consecutive frames of one
        # chronic are near-duplicates, so splitting by frame leaks the val set
        # into training and flatters every metric.
        print(f"[split] building a chronic-level 70/15/15 split from {DATA_FILE}")
        chronics = []
        with open(DATA_FILE) as f:
            for line in f:
                if line.strip():
                    chronics.append(json.loads(
                        "{" + line[line.rindex('"label"'):])["chronic_id"])
        chronics = np.array(chronics)
        uniq = np.unique(chronics)
        rng = np.random.default_rng(SEED)
        rng.shuffle(uniq)
        n_tr, n_va = int(0.70 * len(uniq)), int(0.85 * len(uniq))
        sets = (set(uniq[:n_tr].tolist()), set(uniq[n_tr:n_va].tolist()),
                set(uniq[n_va:].tolist()))
        tr, va, te = (np.where([c in s for c in chronics])[0] for s in sets)
        for path, arr in ((train_idx_path, tr), (val_idx_path, va),
                          (os.path.join(DATA_DIR, f"{SPLIT_PREFIX}_test_idx.npy"), te)):
            np.save(path, arr)
        print(f"[split] {len(uniq)} chronics -> train {len(tr):,} / val {len(va):,} "
              f"/ test {len(te):,} frames")

    train_idx = np.load(train_idx_path)
    val_idx   = np.load(val_idx_path)

    # NOTE: the retired classify pipeline re-shuffled train+val together here to
    # "eliminate chronological domain shift". That destroys chronic-level
    # separation â€” frames from one chronic land on both sides, and consecutive
    # frames are near-duplicates, so the val score becomes partly memorised.
    # Deliberately not carried over.
    print("\n[split] chronic-level separation preserved (no train/val remix)\n")

    # â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pt_data_path = os.path.join(DATA_DIR, PROCESSED_PT)
    full_dataset = PreloadedGridDataset(pt_data_path, device=DEVICE)

    # Sanity check â€” catch stale .pt immediately
    actual_node_feats = full_dataset[0].x.shape[1]
    actual_edge_feats = full_dataset[0].edge_attr.shape[1]
    assert actual_node_feats == NODE_FEATURES, \
        f"Stale .pt: node features={actual_node_feats}, expected {NODE_FEATURES}. Delete processed_grid_data.pt and rerun preprocess.py"
    assert actual_edge_feats == EDGE_FEATURES, \
        f"Stale .pt: edge features={actual_edge_feats}, expected {EDGE_FEATURES}. Delete processed_grid_data.pt and rerun preprocess.py"

    # â”€â”€ Normalization (train split only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Computing normalization statistics from training set...")
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)

    # FIX: normalize ALL features uniformly â€” do NOT exempt rho.
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

    # ðŸš¨ CRITICAL: invalidate PyG's materialised-graph cache. ðŸš¨
    # InMemoryDataset memoises each graph in `_data_list` the first time it is
    # indexed. compute_normalization_stats() above indexes EVERY training graph,
    # so by this point the cache is fully populated with UNNORMALIZED copies â€”
    # and `get()` serves from that cache, not from `_data`. Mutating `_data`
    # therefore had no effect on what the DataLoaders yield, and the network was
    # being trained on raw physical magnitudes (load_p in MW, p_or in MW, v in
    # kV) while `normalization_stats.pt` was saved as if it had been applied.
    #
    # Measured cost on the n1 task: val contingency F1 0.44 with the stale cache
    # versus 0.86 with normalization actually reaching the model. Trees were
    # unaffected (scale-invariant), which is why the gradient-boosting ceiling
    # looked reachable while the network could not get near it.
    full_dataset._data_list = None
    _check = full_dataset[0].x[0, 0].item()
    _expect = ((full_dataset._data.x[0, 0])).item()
    assert abs(_check - _expect) < 1e-4, (
        "normalization still not visible through dataset[i] â€” the PyG cache was "
        "not invalidated; training would silently use raw features."
    )

    train_ds = full_dataset[train_idx]
    val_ds   = full_dataset[val_idx]

    # â”€â”€ DataLoaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_loader = make_dataloader(train_ds, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size, shuffle=False)
    # same size as val, so the two numbers are directly comparable
    train_eval_loader = make_dataloader(
        full_dataset[train_idx[:len(val_idx)]], batch_size, shuffle=False)

    # â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model = GridGNN(
        node_features=NODE_FEATURES,
        edge_features=EDGE_FEATURES,
        hidden_channels=TRAIN_CONFIG["hidden_channels"],
        heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
        line_head_use_mp=not args.head_only,
        line_head_dropout=args.head_dropout,
    ).to(DEVICE)
    print(f"[model] line_head: message-passing embeddings "
          f"{'OFF (skip features only)' if args.head_only else 'ON'}, "
          f"dropout={args.head_dropout}")

    wd = args.weight_decay if args.weight_decay is not None else TRAIN_CONFIG["weight_decay"]
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Roughly 20-26% positive, so a mild pos_weight suffices; computed from the
    # training split rather than hardcoded because the rate differs per topology.
    pos = neg = 0
    for i in train_idx:
        d = full_dataset[i]
        m = d.line_mask
        pos += int(d.line_y[m].sum())
        neg += int((d.line_y[m] == 0).sum())
    pos_weight = torch.tensor(max(neg, 1) / max(pos, 1), device=DEVICE)
    print(f"[data] train contingencies: {pos + neg:,} "
          f"({pos / max(pos + neg, 1):.1%} violation)  "
          f"pos_weight={pos_weight.item():.3f}")

    # Ablations must not overwrite the deliverable: the head-only variant has a
    # different line_head input width, so a checkpoint written under the main
    # name makes the eval script fail to load with a shape mismatch.
    checkpoint_file = CHECKPOINT_FILE
    if args.head_only:
        checkpoint_file = CHECKPOINT_FILE.replace(".pt", "_headonly.pt")
        print(f"[n1] ablation run -> checkpoint {checkpoint_file}")

    best_val_f1   = 0.0
    early_stopping = EarlyStopping(patience=15)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            line_logits = model(batch.x, batch.edge_index, batch.edge_attr,
                                batch.edge_fwd)

            # Masked BCE over individual contingencies. `line_mask` is False
            # where the power flow could not be evaluated â€” those are MISSING
            # labels, and folding them in as 0 would teach the model that losing
            # an already-dead line is safe, which is backwards.
            m = batch.line_mask
            loss = F.binary_cross_entropy_with_logits(
                line_logits[m], batch.line_y[m], pos_weight=pos_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_f1 = evaluate(model, val_loader, DEVICE)
        if args.report_train:
            if True:
                # Overfit vs underfit is the whole diagnosis here, and only the
                # TRAIN score separates them: a large train-val gap means
                # memorisation, while train ~= val ~= low means the optimiser
                # never fitted the signal in the first place.
                print("  [train-set]", end=" ")
                evaluate(model, train_eval_loader, DEVICE)
        print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | "
              f"val_contingency_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_file)
            print(f"  [best] New best saved ({best_val_f1:.4f})")

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best val_contingency_f1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
