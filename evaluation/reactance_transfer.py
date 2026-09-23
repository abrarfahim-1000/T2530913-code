"""Does giving the model the line reactances repair cross-topology transfer?

`thesis_findings.md` §25.5 diagnosed the GNN's cross-topology collapse: where power
goes when a line trips depends on the branch reactances, and the model was never
shown them, so it had nothing to recompute on an unseen grid. §26 then measured
that a logistic regression and a gradient-boosted tree — also denied those inputs —
collapse the same way. The diagnosis is consistent with everything measured and has
never been tested directly. This tests it.

    Retrain with the reactance as an edge feature. Score the same three grids.
    If transfer still fails, §26.3's claim survives its strongest attack.
    If it does not, the claim was an artifact of feature selection.

TWO ARMS, ONE DIFFERENCE
------------------------
The deployed checkpoint cannot serve as the control: `train_gnn.py` saves a bare
state_dict with no epoch field, so what actually produced it is not recoverable
from disk (CLAUDE.md, study.md §7). Comparing a new 9-feature model against it
would confound the feature with the training run. So BOTH arms are trained here,
from the same seed, for the same epochs, on the same graphs, differing only in
whether the model is handed the ninth edge column.

THE FEATURE, AND WHY IT IS NOT THE RAW REACTANCE
------------------------------------------------
Susceptance `b = 1/(x*tap)` is not comparable across these grids: the medians are
5.13, 985 and 1157, a ~200x spread driven by differing `baseMVA` conventions in the
source networks, not by physics. Feeding that raw would make this experiment a test
of unit conventions. The feature is therefore

    log1p( b / median(b) )

which is scale-free and lands in [0.19, 3.11] on all three grids. Nothing is lost:
LODF is invariant under a uniform scaling of every susceptance, so dividing by the
grid median discards exactly the quantity the analytical method also ignores.

Nothing in `scripts/pyg_data.py` is modified — the column is appended after
`build_data`, so the deployed 8-feature path is untouched and cannot regress.

Usage:
    python evaluation/reactance_transfer.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from evaluation.eval_n1_cross_topology import EVAL_BATCH_SIZE, best_f1_and_thr
from evaluation.branch_model import load_branch_model
from scripts.pyg_data import GridEnvMetadata, build_data
from training.config import DATA_DIR, DEVICE, TRAIN_CONFIG
from training.train_gnn import GridGNN, compute_normalization_stats

HOME = "neurips2020"
TAGS = ("neurips2020", "case14", "wcci2022")
OUT = os.path.join("results", "reactance", "reactance_transfer.json")
CKPT_FMT = "gnn_checkpoint_n1_reactance_{arm}_seed{seed}.pt"
SEED = 42

BASE_EDGE_FEATURES = 8  # the deployed feature count; the 9th is the reactance
ARMS = {"control": BASE_EDGE_FEATURES, "physics": BASE_EDGE_FEATURES + 1}

# The documented intended command for the deployed model (CLAUDE.md, Model
# Training). Applied identically to both arms, so it cancels out of the contrast.
EPOCHS, BATCH_SIZE, LR = 30, 128, 3e-4


def reactance_feature(tag: str) -> np.ndarray:
    """Scale-free per-line susceptance for one grid — see the module docstring."""
    model = load_branch_model(tag, os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1_meta.json"))
    return np.log1p(model.b / np.median(model.b)).astype(np.float32)


def build_graphs(tag: str, indices: np.ndarray, meta: GridEnvMetadata,
                 b_feat: np.ndarray) -> list:
    """Graphs carrying 9 edge features. The control arm slices off the last one.

    Building once and slicing guarantees the two arms differ in exactly one
    column and in nothing else — not in graph construction, not in ordering, not
    in the split.
    """
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    wanted = set(indices.tolist())
    graphs = []
    with open(jsonl) as fh:
        for i, line in enumerate(fh):
            if i not in wanted or not line.strip():
                continue
            record = json.loads(line)
            data = build_data(record, meta)
            live = np.asarray(record["line_status"], dtype=bool)
            # build_edges emits [forward_half, reverse_half]; both halves are the
            # same lines in the same order, so the column is simply duplicated.
            col = torch.tensor(np.concatenate([b_feat[live], b_feat[live]]),
                               dtype=torch.float).unsqueeze(1)
            data.edge_attr = torch.cat([data.edge_attr, col], dim=1)
            data.rho_removed = data.edge_attr[data.edge_fwd][:, 0].clone()
            graphs.append(data)
    if len(graphs) != len(indices):
        raise ValueError(f"{tag}: expected {len(indices)} frames, built {len(graphs)}")
    return graphs


def _slice(batch, n_edge: int) -> torch.Tensor:
    return batch.edge_attr[:, :n_edge]


def train_arm(arm: str, train_graphs: list, val_graphs: list) -> tuple[str, float]:
    n_edge = ARMS[arm]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(
        train_graphs, range(len(train_graphs)))
    node_mean, node_std = node_mean.to(DEVICE), node_std.to(DEVICE)
    edge_mean, edge_std = edge_mean[:n_edge].to(DEVICE), edge_std[:n_edge].to(DEVICE)

    pos = sum(int(d.line_y[d.line_mask].sum()) for d in train_graphs)
    total = sum(int(d.line_mask.sum()) for d in train_graphs)
    pos_weight = torch.tensor(max(total - pos, 1) / max(pos, 1), device=DEVICE)

    model = GridGNN(node_features=8, edge_features=n_edge,
                    hidden_channels=TRAIN_CONFIG["hidden_channels"],
                    heads=TRAIN_CONFIG["heads"],
                    dropout=TRAIN_CONFIG["dropout"]).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=TRAIN_CONFIG["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=0)

    ckpt = CKPT_FMT.format(arm=arm, seed=SEED)
    best = 0.0
    print(f"\n[{arm}] {n_edge} edge features, {len(train_graphs):,} frames, "
          f"pos_weight={pos_weight.item():.3f}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model((batch.x - node_mean) / node_std, batch.edge_index,
                           (_slice(batch, n_edge) - edge_mean) / edge_std,
                           batch.edge_fwd)
            m = batch.line_mask
            loss = F.binary_cross_entropy_with_logits(
                logits[m], batch.line_y[m], pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        scores, ys, _ = infer(model, val_loader, n_edge,
                              (node_mean, node_std, edge_mean, edge_std))
        val_f1 = best_f1_and_thr(scores, ys)[0]
        if val_f1 > best:
            best = val_f1
            torch.save(model.state_dict(), ckpt)
        print(f"  epoch {epoch + 1:3d}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}"
              f"  val_f1={val_f1:.4f}{'  *' if val_f1 == best else ''}")

    print(f"[{arm}] best val F1 {best:.4f} -> {ckpt}")
    return ckpt, best


@torch.no_grad()
def infer(model, loader, n_edge: int, stats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_mean, node_std, edge_mean, edge_std = stats
    model.eval()
    out, ys, rhos = [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model((batch.x - node_mean) / node_std, batch.edge_index,
                       (_slice(batch, n_edge) - edge_mean) / edge_std, batch.edge_fwd)
        m = batch.line_mask
        out.append(logits[m].float().cpu())
        ys.append(batch.line_y[m].cpu())
        rhos.append(batch.rho_removed[m].float().cpu())
    return (torch.cat(out).numpy(), torch.cat(ys).numpy().astype(int),
            torch.cat(rhos).numpy())


def frame_count(tag: str) -> int:
    return sum(1 for line in open(os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl"))
               if line.strip())


def load_meta(tag: str) -> GridEnvMetadata:
    with open(os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1_meta.json")) as fh:
        return GridEnvMetadata(json.load(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="CLAUDE.md records seed sensitivity on this architecture; "
                         "the contrast must be shown to survive it")
    args = ap.parse_args()
    globals()["EPOCHS"] = args.epochs
    globals()["SEED"] = args.seed
    if args.seed != 42 and args.json == OUT:
        args.json = OUT.replace(".json", f"_seed{args.seed}.json")

    t0 = time.time()
    home_meta = load_meta(HOME)
    home_b = reactance_feature(HOME)
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    val_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy"))

    print(f"Building neurips2020 train/val graphs "
          f"({len(train_idx):,} / {len(val_idx):,} frames)...")
    train_graphs = build_graphs(HOME, train_idx, home_meta, home_b)
    val_graphs = build_graphs(HOME, val_idx, home_meta, home_b)

    trained = {}
    for arm in ARMS:
        ckpt, best_val = train_arm(arm, train_graphs, val_graphs)
        trained[arm] = {"checkpoint": ckpt, "best_val_f1": best_val}
    del train_graphs

    # Normalization stats must come from the TRAIN split; rebuild them once per
    # arm rather than keeping the graphs alive.
    print("\nRecomputing train-split normalization stats for evaluation...")
    train_graphs = build_graphs(HOME, train_idx, home_meta, home_b)
    stats_full = compute_normalization_stats(train_graphs, range(len(train_graphs)))
    del train_graphs

    models, stats = {}, {}
    for arm, n_edge in ARMS.items():
        net = GridGNN(node_features=8, edge_features=n_edge,
                      hidden_channels=TRAIN_CONFIG["hidden_channels"],
                      heads=TRAIN_CONFIG["heads"],
                      dropout=TRAIN_CONFIG["dropout"]).to(DEVICE)
        net.load_state_dict(torch.load(trained[arm]["checkpoint"], map_location=DEVICE))
        models[arm] = net
        stats[arm] = (stats_full[0].to(DEVICE), stats_full[1].to(DEVICE),
                      stats_full[2][:n_edge].to(DEVICE), stats_full[3][:n_edge].to(DEVICE))

    # Held threshold: selected once on the home val split, per arm (§0 c2).
    held = {}
    val_loader = DataLoader(val_graphs, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    for arm in ARMS:
        s, y, _ = infer(models[arm], val_loader, ARMS[arm], stats[arm])
        held[arm] = best_f1_and_thr(s, y)[1]
    del val_graphs

    report = {
        "protocol": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "feature": "log1p(b / median(b)) appended as edge column 9",
            "device": str(DEVICE),
        },
        "training": trained,
        "held_thresholds": held,
        "grids": {},
    }

    for tag in TAGS:
        meta = home_meta if tag == HOME else load_meta(tag)
        if tag == HOME:
            idx, scope = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy")), \
                "held-out test split"
        else:
            idx, scope = np.arange(frame_count(tag)), "all frames (unseen topology)"
        print(f"Scoring {tag} ({len(idx):,} frames)...")
        graphs = build_graphs(tag, idx, meta, reactance_feature(tag))
        loader = DataLoader(graphs, batch_size=EVAL_BATCH_SIZE, shuffle=False)

        entry = {"scope": scope, "arms": {}}
        for arm in ARMS:
            s, y, rho = infer(models[arm], loader, ARMS[arm], stats[arm])
            oracle_f1, oracle_thr = best_f1_and_thr(s, y)
            pred = s >= held[arm]
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            entry["arms"][arm] = {
                "held": {"f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
                         "threshold": float(held[arm]),
                         "precision": float(prec), "recall": float(rec)},
                "oracle": {"f1": oracle_f1, "threshold": oracle_thr},
            }
            if "contingencies" not in entry:
                p = float(y.mean())
                entry["contingencies"] = int(len(y))
                entry["violation_rate"] = p
                entry["baselines"] = {
                    "all_positive": float(2 * p / (1 + p)) if p else 0.0,
                    # alignment check: 0.4639 / 0.5392 / 0.4915
                    "best_rule_rho_of_removed_line": best_f1_and_thr(rho, y)[0],
                }
        report["grids"][tag] = entry
        del graphs, loader

    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    _print(report)
    print(f"\n  written: {args.json}")


def _print(rep: dict) -> None:
    print(f"\n{'=' * 76}")
    print("  Does the reactance repair transfer?  control (8 feat) vs physics (9 feat)")
    print(f"  identical seed, epochs, graphs and split; one extra edge column")
    print(f"{'=' * 76}")
    print(f"  {'grid':<14}{'all-pos':>9}{'rule':>9}{'control':>11}{'physics':>11}"
          f"{'delta':>9}")
    for tag, g in rep["grids"].items():
        b = g["baselines"]
        c = g["arms"]["control"]["held"]["f1"]
        p = g["arms"]["physics"]["held"]["f1"]
        print(f"  {tag:<14}{b['all_positive']:>9.4f}"
              f"{b['best_rule_rho_of_removed_line']:>9.4f}{c:>11.4f}{p:>11.4f}"
              f"{p - c:>+9.4f}")
    print("\n  ^ held threshold, selected on the neurips2020 val split. The rule "
          "column is the\n    alignment check and must read 0.4639 / 0.5392 / 0.4915.")


if __name__ == "__main__":
    main()
