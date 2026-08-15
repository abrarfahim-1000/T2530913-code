"""
eval_shield.py — GNN-only vs GNN+shield, per topology.

The thesis headline (component_d_handoff.md §1) is the *delta* between the two
arms: what the raw GNN ships versus what survives the shield, per topology, with
false block rate and missed-fault catch rate broken out by failure mode.

Extends evaluation/eval_cross_topology.py's setup — same checkpoint load, same
36-bus train-split normalization, same foreign-meta GridDataset. Two arms:
  raw argmax    — every tag
  Lever A margin — neurips2020 only; the margin is val-tuned in-distribution and
                   does not transfer (CLAUDE.md Component A).

The shield sees the RAW record, never the z-scored features the model consumes.

Usage:
    python evaluation/eval_shield.py --tag case14
    python evaluation/eval_shield.py --tag neurips2020        # held-out test split
    python evaluation/eval_shield.py --tag wcci2022
"""
import argparse
import json
import linecache
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.append(".")

from sklearn.metrics import classification_report, f1_score
from torch_geometric.loader import DataLoader

from scripts.pyg_data import GridDataset, GridEnvMetadata, LABEL_MAP
from shield.context import build_context, load_base_kv
from shield.shield import JsonlRuleProvider, validate
from training.config import DATA_DIR, DEVICE, EDGE_FEATURES, NODE_FEATURES, TRAIN_CONFIG
from training.train_gnn import GridGNN, compute_normalization_stats

CKPT = "gnn_checkpoint_best.pt"
MARGIN_FILE = "gnn_logit_margin.json"
RULES_FILE = "rules/all_rules_deduped.jsonl"
OVERCONFIDENT_THRESHOLD = 0.85


def classify_failure_mode(true_label: str, pred_label: str, confidence: float,
                          context: dict) -> str:
    """CLAUDE.md's Component D failure-mode taxonomy, evaluated in priority order."""
    if pred_label == "normal" and true_label != "normal" and confidence > OVERCONFIDENT_THRESHOLD:
        return "overconfident_wrong"
    if pred_label != true_label and pred_label != "normal" and true_label != "normal":
        return "class_confusion"
    if pred_label == true_label:
        # Right call, yet the frame still trips a physical threshold.
        return "threshold_failure"
    return "novel_topology_state"


def _record_indices(tag: str, n_lines: int) -> np.ndarray:
    """Held-out test split for the training topology; everything for foreign ones."""
    if tag == "neurips2020":
        split = os.path.join(DATA_DIR, "split_neurips2020_test_idx.npy")
        if os.path.exists(split):
            return np.sort(np.load(split))
        print(f"WARNING: {split} missing - evaluating on ALL neurips2020 records, "
              f"which includes data the model trained on.")
    return np.arange(n_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="GNN-only vs GNN+shield evaluation")
    ap.add_argument("--tag", default="case14")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--margin", default=MARGIN_FILE)
    ap.add_argument("--rules", default=RULES_FILE)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N records")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    idx_to_label = {v: k for k, v in LABEL_MAP.items()}
    n_classes = len(LABEL_MAP)
    target_names = [idx_to_label[i] for i in range(n_classes)]

    # ── Rules ────────────────────────────────────────────────────────────────
    provider = JsonlRuleProvider.from_jsonl(args.rules)
    print(f"Loaded {len(provider)} rules from {args.rules}")

    # ── Data ─────────────────────────────────────────────────────────────────
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}.jsonl")
    meta_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_meta.json")
    if not (os.path.exists(jsonl) and os.path.exists(meta_path)):
        sys.exit(f"Missing {jsonl} or {meta_path}. Generate it first with "
                 f"scripts/generate_dataset.py.")
    with open(meta_path) as f:
        meta = GridEnvMetadata(json.load(f))
    base_kv = load_base_kv(args.tag, data_dir=DATA_DIR)

    n_lines = sum(1 for _ in open(jsonl, "r"))
    indices = _record_indices(args.tag, n_lines)
    if args.limit:
        indices = indices[: args.limit]
    print(f"[{args.tag}] n_sub={meta.n_sub} n_line={meta.n_line} evaluating={len(indices)}")

    ds = GridDataset(jsonl, indices, meta)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Normalization + model ────────────────────────────────────────────────
    from scripts.pyg_data import PreloadedGridDataset

    print("Computing 36-bus train-split normalization stats...")
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_train_idx.npy"))
    home = PreloadedGridDataset(os.path.join(DATA_DIR, "processed_grid_data.pt"), device=DEVICE)
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(home, train_idx)
    del home

    model = GridGNN(
        node_features=NODE_FEATURES, edge_features=EDGE_FEATURES, n_classes=n_classes,
        hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    # Lever A applies in-distribution only.
    use_margin = args.tag == "neurips2020"
    offset = torch.zeros(n_classes)
    if use_margin:
        with open(args.margin) as f:
            m = json.load(f)
        offset = torch.tensor([m[idx_to_label[i]] for i in range(n_classes)], dtype=torch.float32)
        print("Lever A margin: " + ", ".join(f"{idx_to_label[i]}={offset[i]:+.2f}"
                                             for i in range(n_classes)))
    else:
        print("Lever A margin NOT applied (does not transfer off the 36-bus topology).")

    # ── Inference + gating ───────────────────────────────────────────────────
    eval_path = os.path.join(args.out_dir, f"shield_eval_{args.tag}.jsonl")
    fail_path = os.path.join(args.out_dir, f"failures_{args.tag}.jsonl")

    y_true, pred_raw_all, pred_cal_all = [], [], []
    blocked_raw, blocked_cal, unsupported_raw = [], [], []
    n_not_evaluable = n_error = 0
    failure_modes: Counter = Counter()
    wrong_and_blocked: dict = defaultdict(int)
    wrong_total: dict = defaultdict(int)
    cursor = 0

    with torch.no_grad(), open(eval_path, "w") as ef, open(fail_path, "w") as ff:
        for batch in loader:
            batch = batch.to(DEVICE)
            batch.x = (batch.x - node_mean) / node_std
            batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = logits.cpu()

            probs = torch.softmax(logits, dim=1)
            preds_raw = logits.argmax(dim=1).numpy()
            preds_cal = (logits + offset).argmax(dim=1).numpy()
            ys = batch.y.cpu().numpy()

            for i in range(len(ys)):
                record_idx = int(indices[cursor])
                cursor += 1

                raw = json.loads(linecache.getline(jsonl, record_idx + 1))
                true_label = idx_to_label[int(ys[i])]
                pred_label = idx_to_label[int(preds_raw[i])]
                cal_label = idx_to_label[int(preds_cal[i])]
                confidence = float(probs[i].max())

                context = build_context(raw, base_kv, fault_type=pred_label,
                                        confidence=confidence)
                result = validate(context, provider.rules_for(pred_label))

                y_true.append(true_label)
                pred_raw_all.append(pred_label)
                pred_cal_all.append(cal_label)
                blocked_raw.append(result.blocked)
                # Option A ships this prediction; Option B would additionally block
                # when affirmations for the predicted class existed but none fired.
                unsupported_raw.append(result.unsupported)
                n_not_evaluable += result.n_not_evaluable
                n_error += result.n_error

                if use_margin:
                    cal_context = build_context(raw, base_kv, fault_type=cal_label,
                                                confidence=confidence)
                    cal_result = validate(cal_context, provider.rules_for(cal_label))
                    blocked_cal.append(cal_result.blocked)

                ef.write(json.dumps({
                    "record_idx": record_idx,
                    "true_label": true_label,
                    "pred_raw": pred_label,
                    "pred_calibrated": cal_label if use_margin else None,
                    "confidence": round(confidence, 5),
                    **result.to_dict(),
                }) + "\n")

                if result.blocked:
                    mode = classify_failure_mode(true_label, pred_label, confidence, context)
                    failure_modes[mode] += 1
                    ff.write(json.dumps({
                        "record_idx": record_idx,
                        "true_label": true_label,
                        "pred_raw": pred_label,
                        "confidence": round(confidence, 5),
                        "failure_mode": mode,
                        "highest_severity": result.highest_severity,
                        "violated_rule_ids": [r.get("rule_id") for r in result.violated_rules],
                        "explanation": result.explanation,
                        "context": {k: v for k, v in context.items()
                                    if k not in ("fault_type", "confidence")},
                    }) + "\n")

                if pred_label != true_label:
                    mode = classify_failure_mode(true_label, pred_label, confidence, context)
                    wrong_total[mode] += 1
                    if result.blocked:
                        wrong_and_blocked[mode] += 1

    # ── Metrics ──────────────────────────────────────────────────────────────
    y_true = np.array(y_true)
    pred_raw_all = np.array(pred_raw_all)
    blocked_raw = np.array(blocked_raw)
    normal_mask = y_true == "normal"
    wrong_mask = pred_raw_all != y_true

    print(f"\n===== [{args.tag}] GNN-only (raw argmax) =====")
    print(classification_report(y_true, pred_raw_all, labels=target_names,
                                digits=4, zero_division=0))

    unsupported_raw = np.array(unsupported_raw)
    option_b_blocked = blocked_raw | unsupported_raw

    summary = {
        "tag": args.tag,
        "n_records": int(len(y_true)),
        "n_rules": len(provider),
        "macro_f1_raw": float(f1_score(y_true, pred_raw_all, average="macro", zero_division=0)),
        "false_block_rate": float(blocked_raw[normal_mask].mean()) if normal_mask.any() else None,
        "block_rate_overall": float(blocked_raw.mean()),
        "missed_fault_catch_rate": (float(blocked_raw[wrong_mask].mean())
                                    if wrong_mask.any() else None),
        "catch_rate_by_failure_mode": {
            mode: round(wrong_and_blocked[mode] / wrong_total[mode], 4)
            for mode in wrong_total if wrong_total[mode]
        },
        "failure_mode_distribution": dict(failure_modes),
        "n_not_evaluable": int(n_not_evaluable),
        "n_error": int(n_error),
        "shield_applicability": (
            1.0 - n_not_evaluable / max(len(y_true) * len(provider), 1)
        ),
        # ── Option A vs Option B counterfactual ───────────────────────────────
        # A (deployed): only constraint violations block. B (recorded, not applied):
        # also block predictions no affirmation supports. Reported side by side so
        # the gating variant can be judged on measured coverage rather than guessed.
        "affirmation": {
            "n_unsupported": int(unsupported_raw.sum()),
            "option_a_block_rate": float(blocked_raw.mean()),
            "option_b_block_rate": float(option_b_blocked.mean()),
            "option_a_false_block_rate": (float(blocked_raw[normal_mask].mean())
                                          if normal_mask.any() else None),
            "option_b_false_block_rate": (float(option_b_blocked[normal_mask].mean())
                                          if normal_mask.any() else None),
            "option_a_catch_rate": (float(blocked_raw[wrong_mask].mean())
                                    if wrong_mask.any() else None),
            "option_b_catch_rate": (float(option_b_blocked[wrong_mask].mean())
                                    if wrong_mask.any() else None),
        },
    }
    if use_margin:
        pred_cal_all = np.array(pred_cal_all)
        summary["macro_f1_calibrated"] = float(
            f1_score(y_true, pred_cal_all, average="macro", zero_division=0))
        blocked_cal = np.array(blocked_cal)
        summary["false_block_rate_calibrated"] = (
            float(blocked_cal[normal_mask].mean()) if normal_mask.any() else None)
        print(f"\n===== [{args.tag}] Lever A calibrated =====")
        print(classification_report(y_true, pred_cal_all, labels=target_names,
                                    digits=4, zero_division=0))

    summary_path = os.path.join(args.out_dir, f"shield_summary_{args.tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== [{args.tag}] shield =====")
    print(f"  macro F1 (raw)          : {summary['macro_f1_raw']:.4f}")
    print(f"  false block rate        : {summary['false_block_rate']}")
    print(f"  missed-fault catch rate : {summary['missed_fault_catch_rate']}")
    print(f"  NOT_EVALUABLE / ERROR   : {n_not_evaluable} / {n_error}")
    aff = summary["affirmation"]
    print(f"  unsupported predictions : {aff['n_unsupported']} "
          f"(Option B would block these too)")
    print(f"  false block A vs B      : {aff['option_a_false_block_rate']} "
          f"vs {aff['option_b_false_block_rate']}")
    print(f"  catch rate  A vs B      : {aff['option_a_catch_rate']} "
          f"vs {aff['option_b_catch_rate']}")
    if n_error:
        print("  WARNING: nonzero ERROR verdicts - the ruleset should be pre-linted.")
    print(f"  per-record log -> {eval_path}")
    print(f"  blocks         -> {fail_path}")
    print(f"  summary        -> {summary_path}")


if __name__ == "__main__":
    main()
