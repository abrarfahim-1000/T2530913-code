"""§24.2 Arm 1 — the ceiling analysis, run across all three grids.

`thesis_findings.md` §24 is the pickup-ready spec and this implements it verbatim.
The question it answers is NOT "is the GNN good" but:

    Is the thin rule corpus the extraction pipeline's fault, or the task's?

A small decision tree is fitted on the same 14 context variables the shield's rules
are written over, targeting the exact error the shield exists to catch. The tree is
the best a frame-level symbolic gate could possibly do on those variables, because
it is fitted directly on the answer. What the extracted rules achieve against that
ceiling is the measurement.

  tree cannot beat chance      -> (a) the information is not in present-state
                                  telemetry at all. The strongest result: it holds
                                  without reference to rule count.
  tree does well, shield does not -> (b) extraction was the bottleneck, and the
                                  tree's splits name the variables that were missed.
  tree ~= shield               -> the shield is at the ceiling; extraction was not
                                  the binding constraint.

PROTOCOL, from §24.2 — none of this is discretionary:
  * Target `model predicted secure AND the contingency was a violation`, per
    contingency. NOT symmetric error: `validate_n1` only blocks the over-permissive
    direction, so scoring symmetric error measures something the shield never tries.
  * Fit on the neurips2020 TRAIN split only. Evaluate on its test split and on both
    foreign grids.
  * Held threshold 0.8849 (re-derived on the val split here, and asserted against
    the recorded value), eval batch size 64.
  * Depth <= 4, so the tree stays a readable rule set.

⚠️ The honest constraint, which §24.2 requires be stated when reporting: the context
is PER FRAME and the error is PER CONTINGENCY, so the tree cannot separate two
contingencies within one frame. Neither can any extracted rule — the shield reads the
same per-frame namespace — so this is the fair upper bound for frame-level symbolic
gating, not a handicap imposed on the tree.

Usage:
    python evaluation/ceiling_tree_n1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(".")

from sklearn.tree import DecisionTreeClassifier, export_text
from torch_geometric.loader import DataLoader

from evaluation.eval_n1_cross_topology import EVAL_BATCH_SIZE, best_f1_and_thr
from scripts.pyg_data import (GridDataset, GridEnvMetadata,
                              PreloadedGridDataset, build_line_targets)
from shield.context import build_context, load_base_kv
from training.config import DATA_DIR, DEVICE, TRAIN_CONFIG
from training.train_gnn import GridGNN, compute_normalization_stats

HOME = "neurips2020"
TAGS = ("neurips2020", "case14", "wcci2022")
CKPT = "gnn_checkpoint_n1.pt"
OUT = os.path.join("results", "ceiling", "ceiling_tree.json")
SEED = 42
MAX_DEPTH = 4
RECORDED_THRESHOLD = 0.8849  # CLAUDE.md; re-derived below and checked against it

# The 14 variables of shield/context.py — the namespace every extracted rule is
# written over. Listed explicitly so the tree cannot quietly gain a feature the
# rules never had.
CONTEXT_VARS = (
    "loading_pct", "rho_max", "n_tripped_lines", "any_line_tripped",
    "active_power_mw_max", "reactive_power_mvar_max", "apparent_power_mva_max",
    "power_factor_at_max_load", "current_a_max",
    "total_generation_mw", "total_load_mw", "generation_load_imbalance_pct",
    "voltage_pu_min", "voltage_pu_max",
)


def score_model(tag: str, indices: np.ndarray, meta: GridEnvMetadata, norm
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-contingency logits and labels, at the pinned eval batch size."""
    node_mean, node_std, edge_mean, edge_std = norm
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    loader = DataLoader(GridDataset(jsonl, indices, meta), batch_size=EVAL_BATCH_SIZE,
                        shuffle=False, num_workers=0)
    model = GridGNN(node_features=8, edge_features=8,
                    hidden_channels=TRAIN_CONFIG["hidden_channels"],
                    heads=TRAIN_CONFIG["heads"],
                    dropout=TRAIN_CONFIG["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    logits, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model((batch.x - node_mean) / node_std, batch.edge_index,
                        (batch.edge_attr - edge_mean) / edge_std, batch.edge_fwd)
            m = batch.line_mask
            logits.append(out[m].float().cpu())
            ys.append(batch.line_y[m].cpu())
    return torch.cat(logits).numpy(), torch.cat(ys).numpy().astype(int)


def frame_contexts(tag: str, indices: np.ndarray, meta: GridEnvMetadata
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(X, counts) — the 14 context variables per frame, and its contingency count.

    Rows are later repeated by `counts` so every contingency in a frame carries its
    frame's context. That repetition IS the constraint §24.2 names.
    """
    base_kv = load_base_kv(tag)
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    wanted = set(indices.tolist())
    rows, counts = [], []
    with open(jsonl) as fh:
        for i, line in enumerate(fh):
            if i not in wanted or not line.strip():
                continue
            record = json.loads(line)
            ctx = build_context(record, base_kv)
            rows.append([float(ctx.get(v, np.nan)) for v in CONTEXT_VARS])
            _y, mask, _lid = build_line_targets(record, meta)
            counts.append(int(mask.sum()))
    if len(rows) != len(indices):
        raise ValueError(f"{tag}: expected {len(indices)} frames, built {len(rows)}")
    return np.asarray(rows, dtype=float), np.asarray(counts)


def load_meta(tag: str) -> GridEnvMetadata:
    with open(os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1_meta.json")) as fh:
        return GridEnvMetadata(json.load(fh))


def frame_count(tag: str) -> int:
    return sum(1 for line in open(os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl"))
               if line.strip())


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
            "precision": float(prec), "recall": float(rec),
            "tp": tp, "fp": fp, "fn": fn}


def shield_reference(tag: str) -> dict | None:
    """The extracted corpus's own numbers on this exact target, for the contrast."""
    path = os.path.join("results", "shield", f"shield_{tag}_validated.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    target = d["missed_violations"]
    tp, fp = d["corrections"], d["blocked"] - d["corrections"]
    fn = target - tp
    return {
        "target_contingencies": target,
        "blocked": d["blocked"],
        "caught": tp,
        "recall": tp / max(target, 1),
        "precision": tp / max(d["blocked"], 1),
        # comparable to the tree's F1 on the identical target and identical rows
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    ap.add_argument("--class-weight", default=None, choices=[None, "balanced"],
                    help="Robustness arm. §24.2 specifies an unweighted depth-4 tree; "
                         "this exists to close the objection that the tree was simply "
                         "fitted badly on a rare target.")
    args = ap.parse_args()
    if args.max_depth != MAX_DEPTH or args.class_weight:
        tag = f"_depth{args.max_depth}" + ("_balanced" if args.class_weight else "")
        if args.json == OUT:
            args.json = OUT.replace(".json", f"{tag}.json")
    t0 = time.time()

    home_meta = load_meta(HOME)
    print("Computing 36-bus train-split normalization stats...")
    home_pt = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data_n1.pt"),
                                   device=DEVICE)
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    val_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy"))
    test_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy"))
    norm = compute_normalization_stats(home_pt, train_idx)
    del home_pt

    # ── the held threshold, re-derived on the val split (§24.2) ──────────────
    print("Selecting the decision threshold on the neurips2020 VAL split...")
    val_logits, val_y = score_model(HOME, val_idx, home_meta, norm)
    _, thr = best_f1_and_thr(val_logits, val_y)
    if abs(thr - RECORDED_THRESHOLD) > 5e-4:
        raise SystemExit(
            f"threshold {thr:.6f} does not reproduce the recorded {RECORDED_THRESHOLD} "
            f"— refusing to run under a protocol that has silently changed")
    print(f"  threshold {thr:.6f}  (reproduces the recorded {RECORDED_THRESHOLD})")

    def build(tag: str, idx: np.ndarray, meta: GridEnvMetadata) -> dict:
        logits, y = score_model(tag, idx, meta, norm)
        ctx, counts = frame_contexts(tag, idx, meta)
        if counts.sum() != len(y):
            raise ValueError(f"{tag}: {counts.sum()} context rows vs {len(y)} contingencies")
        eligible = logits <= thr  # a gate only ever acts where the model said secure
        return {"X": np.repeat(ctx, counts, axis=0), "y": y, "logits": logits,
                # the target: the model called it secure and it was not
                "target": (eligible & (y == 1)).astype(int),
                "eligible": eligible,
                "frames": len(idx), "counts": counts}

    print(f"Building the TRAIN fit set ({len(train_idx):,} frames)...")
    train = build(HOME, train_idx, home_meta)
    print(f"  {len(train['y']):,} contingencies, target rate "
          f"{train['target'].mean():.4%}")

    tree = DecisionTreeClassifier(max_depth=args.max_depth, random_state=SEED,
                                  class_weight=args.class_weight)
    tree.fit(train["X"], train["target"])
    rules_text = export_text(tree, feature_names=list(CONTEXT_VARS), max_depth=args.max_depth)
    importances = {v: float(i) for v, i in zip(CONTEXT_VARS, tree.feature_importances_)
                   if i > 0}

    print("Selecting the tree's own threshold on the neurips2020 VAL split...")
    val = build(HOME, val_idx, home_meta)
    _, tree_thr = best_f1_and_thr(tree.predict_proba(val["X"])[:, 1], val["target"])
    del val

    report = {
        "protocol": {
            "target": "model predicted secure AND the contingency was a violation",
            "fit_on": f"neurips2020 train split ({len(train_idx)} frames, "
                      f"{len(train['y'])} contingencies)",
            "model_threshold": float(thr),
            "tree_threshold_held": float(tree_thr),
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_depth": args.max_depth,
            "class_weight": args.class_weight,
            "features": list(CONTEXT_VARS),
            "seed": SEED,
        },
        "train_target_rate": float(train["target"].mean()),
        "tree": {"feature_importances": importances, "rules": rules_text},
        "grids": {},
    }
    del train

    for tag in TAGS:
        meta = home_meta if tag == HOME else load_meta(tag)
        idx = test_idx if tag == HOME else np.arange(frame_count(tag))
        scope = "held-out test split" if tag == HOME else "all frames (unseen topology)"
        print(f"Scoring {tag} ({len(idx):,} frames)...")
        d = build(tag, idx, meta)
        t, X = d["target"], d["X"]
        proba = tree.predict_proba(X)[:, 1]
        base = float(t.mean())

        # Frames holding at least one target contingency — the ceiling on what a
        # per-frame gate could reach even with a perfect frame classifier.
        per_frame = np.split(t, np.cumsum(d["counts"])[:-1])
        frames_with_target = float(np.mean([bool(f.any()) for f in per_frame]))

        report["grids"][tag] = {
            "scope": scope,
            "contingencies": int(len(t)),
            "target_contingencies": int(t.sum()),
            "target_rate": base,
            "frames_containing_a_target": frames_with_target,
            "all_positive_f1_on_target": float(2 * base / (1 + base)) if base else 0.0,
            "tree_held": prf(t, proba >= tree_thr),
            "tree_oracle_f1": best_f1_and_thr(proba, t)[0],
            # GATE SEMANTICS — the like-for-like comparison with the shield, which
            # can only ever block a contingency the model called secure. Scoring a
            # predictor over rows no gate would touch charges it for false alarms
            # it would never raise.
            "tree_held_gate": prf(t, (proba >= tree_thr) & d["eligible"]),
            "tree_oracle_gate_f1": best_f1_and_thr(
                np.where(d["eligible"], proba, -np.inf), t)[0],
            "shield_on_same_target": shield_reference(tag),
        }
        del d

    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    _print(report)
    print(f"\n  written: {args.json}")


def _print(rep: dict) -> None:
    print(f"\n{'=' * 78}")
    print("  §24.2 ceiling tree — best a frame-level symbolic gate could do")
    print(f"  target: {rep['protocol']['target']}")
    print(f"  depth <= {rep['protocol']['max_depth']}, "
          f"fit on {rep['protocol']['fit_on']}")
    print(f"{'=' * 78}")
    print("\n  split variables (importance):")
    for v, i in sorted(rep["tree"]["feature_importances"].items(),
                       key=lambda kv: -kv[1]):
        print(f"    {v:<34}{i:.3f}")
    for tag, g in rep["grids"].items():
        s = g["shield_on_same_target"]
        print(f"\n  {tag}  —  {g['scope']}")
        print(f"    target                : {g['target_contingencies']:,} of "
              f"{g['contingencies']:,} contingencies ({g['target_rate']:.2%})")
        print(f"    frames with a target  : {g['frames_containing_a_target']:.1%}")
        print(f"    all-positive on target: F1 {g['all_positive_f1_on_target']:.4f}")
        t = g["tree_held"]
        print(f"    TREE (held, all rows) : F1 {t['f1']:.4f}   "
              f"prec {t['precision']:.3f}  rec {t['recall']:.3f}")
        tg = g["tree_held_gate"]
        print(f"    TREE (held, GATE)     : F1 {tg['f1']:.4f}   "
              f"prec {tg['precision']:.3f}  rec {tg['recall']:.3f}")
        print(f"    TREE (oracle, gate)   : F1 {g['tree_oracle_gate_f1']:.4f}  "
              f"(all rows {g['tree_oracle_f1']:.4f})")
        if s:
            print(f"    EXTRACTED SHIELD      : F1 {s['f1']:.4f}   "
                  f"prec {s['precision']:.3f}  rec {s['recall']:.3f}   "
                  f"({s['caught']:,} of {s['target_contingencies']:,})")
    print("\n  ⚠️ The context is PER FRAME and the error is PER CONTINGENCY, so the "
          "tree cannot\n     separate contingencies within a frame. Neither can any "
          "extracted rule — this is\n     the fair upper bound for frame-level "
          "symbolic gating (§24.2).")


if __name__ == "__main__":
    main()
