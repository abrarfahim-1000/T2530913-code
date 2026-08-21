"""
eval_shield_n1.py — the N-1 shield arm: GNN alone vs GNN + symbolic gate.

Replaces the classify-era `eval_shield.py`, which loads `gnn_checkpoint_best.pt`,
uses the retired 4-class `LABEL_MAP`, applies Lever A and reports a
`classification_report` — none of which exists under N-1 screening. Model loading,
normalization and baselines follow `eval_n1_cross_topology.py` exactly so the two
harnesses cannot disagree about what the model scores.

WHAT THE SHIELD CAN DO HERE, AND WHAT IT CANNOT
-----------------------------------------------
`shield.validate_n1` gates asymmetrically: it blocks a `secure` verdict issued
over a base case that already violates a CONSTRAINT, and never blocks the
conservative `violation` verdict. It cannot verify the contingency itself — that
needs a post-contingency power flow, which is the computation the GNN stands in
for.

The consequence is arithmetic and should be read BEFORE the output table: the
shield can only speak about frames whose base case violates a rule. On the
generated sets that is 2.1% / 2.3% / 6.3% of frames (neurips2020 / case14 /
wcci2022), and only 0.9% / 0.8% / 2.7% also carry a contingency the model might
wrongly call secure. **A global F1 delta will therefore read ~0.000, and that is
the expected result, not a bug.** The number that carries information is the
CONDITIONAL one: within the set the shield is eligible to act on, how many of its
blocks are corrections and how many are regressions. Both are reported below.

The other reportable quantity is structural: of every missed violation the model
commits, what share sits on a base case a present-state rule could even see. That
bounds what ANY symbolic gate over this vocabulary could catch, independently of
how many rules the corpus happens to contain (plan §7.3).

BLOCK SEMANTICS
---------------
A BLOCK is not an abstention: the operational meaning of "you may not call this
secure" is escalate-to-unsafe. The shielded arm therefore predicts `violation`
wherever the shield blocked, which keeps both arms on one metric.

Usage:
    python evaluation/eval_shield_n1.py --tag neurips2020
    python evaluation/eval_shield_n1.py --tag case14  --rules translated_rules/guarded/
    python evaluation/eval_shield_n1.py --tag wcci2022 --json results/shield/shield_wcci2022.json

Every artifact this writes lands under results/, which is created on demand:
    results/shield/     the run summary   (--json)
    results/failures/   per-contingency failure log, ~12 MB on wcci2022
    results/citations/  provenance chains (--citations, requires --rules-kg)
"""
from __future__ import annotations

import argparse
import json
import linecache
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")

from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from scripts.pyg_data import GridDataset, GridEnvMetadata, PreloadedGridDataset
from shield.context import build_context, load_base_kv
from shield.shield import SECURE, VIOLATION, validate_n1
from training.config import DATA_DIR, DEVICE, EDGE_FEATURES, NODE_FEATURES, TRAIN_CONFIG
from training.train_gnn import GridGNN, compute_normalization_stats

CKPT = "gnn_checkpoint_n1.pt"
HOME = "neurips2020"

# ── EVAL BATCH SIZE — a RECORDED DECISION, not an incidental default ──────────
# Settled 2026-08-21: keep 64. Every number reported anywhere in this project was
# measured at this value.
#
# It is load-bearing. GridGNN uses BatchNorm(track_running_stats=False), so the
# network normalizes with LIVE BATCH STATISTICS at inference, not stored running
# averages. Batch composition therefore changes the logits, and the same
# checkpoint on the same split scores:
#
#     batch  64 -> F1 0.8972   (this default; the source of every reported figure)
#     batch 128 -> F1 0.9088
#     batch 256 -> F1 0.9241
#     batch 512 -> F1 0.9255   (TRAIN_CONFIG's value, what training evaluated at)
#
# Positives (20,801) and the rule baseline (0.4639) are constant across all four,
# so this is the forward pass, not the data.
#
# Shield DELTAS are stable across the same change (+0.0082 -> +0.0072) because
# both arms share one forward pass; absolute F1 is not. Report deltas, not levels.
#
# Changing this invalidates comparison with every recorded result. See
# supplimentary_docs/gnn_n1_tightening.md §8.
EVAL_BATCH_SIZE = 64

# The stage-3 corpus. This used to point at `rules/`, which was the v1 output
# directory and has held no live corpus since the v2 pipeline; the default
# silently resolved to a file that does not exist.
DEDUPED_RULES = "validated_translated/all_rules_deduped.jsonl"
RESULTS_DIR = Path("results")


def _mkparent(path: str) -> Path:
    """Resolve an output path, creating its directory. Outputs now nest under
    results/, so a plain open() would fail on a fresh clone."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ── rules ─────────────────────────────────────────────────────────────────────

def load_rules(spec: str) -> list[dict]:
    """Accept the stage-3 deduped file OR a stage-2.5 guarded folder.

    The folder form exists so the shield can be measured against the guarded
    corpus before validate.py has run — the guard is what removes polarity
    contamination, and that is the property the shield depends on. Records from
    the guard are wrapped as {"rule": {...}}; unwrap them.
    """
    path = Path(spec)
    if path.is_dir():
        files = sorted(path.glob("*_translated.jsonl"))
        if not files:
            sys.exit(f"No *_translated.jsonl rules under {path}/")
    elif path.exists():
        files = [path]
    else:
        sys.exit(
            f"Missing {path}.\n"
            f"Either run stage 3 (extraction/validate.py) to produce "
            f"{DEDUPED_RULES},\nor point --rules at a stage-2.5 guarded folder, e.g.\n"
            f"  --rules translated_rules/guarded/"
        )

    rules: list[dict] = []
    for jf in files:
        with jf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rules.append(rec.get("rule", rec))
    if not rules:
        sys.exit(f"No rules loaded from {spec}")
    return rules


# ── thresholding ──────────────────────────────────────────────────────────────

def best_f1_and_thr(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Identical to eval_n1_cross_topology.best_f1_and_thr — kept in step by test."""
    order = np.argsort(-score)
    ys = y[order]
    tp, fp, P = np.cumsum(ys), np.cumsum(1 - ys), ys.sum()
    if P == 0:
        return 0.0, 0.0
    prec, rec = tp / np.maximum(tp + fp, 1), tp / P
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
    i = int(np.argmax(f1))
    return float(f1[i]), float(score[order][i])


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "false_alarms": fp,
        "missed_violations": fn,
    }


# ── inference ─────────────────────────────────────────────────────────────────

def score_topology(tag: str, checkpoint: str, batch_size: int, norm: tuple):
    """Run the model over one topology. Returns per-contingency arrays.

    `frame` maps each contingency back to its row in the JSONL, which is what
    lets the shield see the raw record rather than the z-scored tensor.
    """
    node_mean, node_std, edge_mean, edge_std = norm
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    meta_path = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1_meta.json")
    for p in (jsonl, meta_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}\n  python scripts/generate_dataset.py --env <env>")

    with open(meta_path) as f:
        meta = GridEnvMetadata(json.load(f))
    n_records = sum(1 for line in open(jsonl, encoding="utf-8") if line.strip())

    if tag == HOME:
        test_path = os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy")
        indices = np.load(test_path) if os.path.exists(test_path) else np.arange(n_records)
        scope = f"held-out test split ({len(indices):,} of {n_records:,} frames)"
    else:
        indices = np.arange(n_records)
        scope = f"all {n_records:,} frames (topology never seen in training)"

    ds = GridDataset(jsonl, indices, meta)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GridGNN(
        node_features=NODE_FEATURES, edge_features=EDGE_FEATURES,
        hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    logits_l, y_l, rho_l, frame_l, line_l = [], [], [], [], []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            n_graphs = int(batch.num_graphs)
            batch = batch.to(DEVICE)
            batch.x = (batch.x - node_mean) / node_std
            batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
            line_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_fwd)

            m = batch.line_mask
            fwd_attr = batch.edge_attr[batch.edge_fwd]
            # graph id per FORWARD edge -> per scored contingency
            graph_of_edge = batch.batch[batch.edge_index[0]][batch.edge_fwd]

            logits_l.append(line_logits[m].float().cpu())
            y_l.append(batch.line_y[m].cpu())
            rho_l.append((fwd_attr[m, 0].float().cpu() * edge_std[0].cpu()
                          + edge_mean[0].cpu()))
            frame_l.append(graph_of_edge[m].cpu() + cursor)
            line_l.append(batch.line_id[m].cpu())
            cursor += n_graphs

    order_pos = torch.cat(frame_l).numpy()          # position within `indices`
    return {
        "logits": torch.cat(logits_l).numpy(),
        "y": torch.cat(y_l).numpy().astype(int),
        "rho": torch.cat(rho_l).numpy(),
        "frame": indices[order_pos],                # row number in the JSONL
        "line": torch.cat(line_l).numpy(),
        "jsonl": jsonl,
        "meta": meta,
        "scope": scope,
        "n_frames": len(indices),
    }


# ── shield ────────────────────────────────────────────────────────────────────

def run_shield(scored: dict, pred: np.ndarray, rules: list[dict], tag: str,
               failures_path: Path | None, max_failures: int = 50_000) -> dict:
    """Apply validate_n1 per contingency and return the shielded predictions.

    `validate_n1(context, rules, predicted)` is a pure function of the FRAME's
    context and the predicted label, so it is evaluated once per distinct
    (frame, label) pair and reused across that frame's contingencies. That is
    memoisation of the real code path, not a reimplementation of it — the shield
    module is the only thing deciding anything here.
    """
    base_kv = load_base_kv(tag)
    blocked = np.zeros(len(pred), dtype=bool)
    base_violates = np.zeros(len(pred), dtype=bool)
    n_not_evaluable = n_error = n_unsupported = 0
    n_frames_seen = n_logged = 0
    severities: Counter = Counter()
    fired: Counter = Counter()      # served rule_id -> blocks it produced
    fh = failures_path.open("w", encoding="utf-8") if failures_path else None

    cache: dict[tuple[int, str], object] = {}
    contexts: dict[int, dict] = {}

    try:
        for i, frame_idx in enumerate(scored["frame"]):
            frame_idx = int(frame_idx)
            ctx = contexts.get(frame_idx)
            if ctx is None:
                raw = json.loads(linecache.getline(scored["jsonl"], frame_idx + 1))
                ctx = build_context(raw, base_kv)
                contexts[frame_idx] = ctx
                n_frames_seen += 1

            label = SECURE if pred[i] == 0 else VIOLATION
            key = (frame_idx, label)
            res = cache.get(key)
            if res is None:
                res = validate_n1(ctx, rules, predicted=label)
                cache[key] = res
                n_not_evaluable += res.n_not_evaluable
                n_error += res.n_error
                n_unsupported += int(res.unsupported)

            if res.blocked:
                blocked[i] = True
                severities[str(res.highest_severity)] += 1
                for r in res.violated_rules:
                    fired[str(r.get("rule_id"))] += 1
            # Only meaningful on the SECURE branch, which is the only branch either
            # consumer reads: `eligible` and the structural ceiling both gate on
            # pred == 0. On the VIOLATION branch validate_n1 folds fired constraints
            # into `supporting_rules`, so that field cannot be read as "base case
            # violates a rule" without conflating constraints with affirmations.
            base_violates[i] = bool(label == SECURE and res.violated_rules)

            if fh is not None and n_logged < max_failures and (
                    blocked[i] or (scored["y"][i] == 1 and pred[i] == 0)):
                n_logged += 1
                fh.write(json.dumps({
                    "frame": frame_idx,
                    "line": int(scored["line"][i]),
                    "truth": int(scored["y"][i]),
                    "predicted": label,
                    "shield": res.status,
                    "mode": failure_mode(scored["y"][i], pred[i], blocked[i]),
                    "rho_max": ctx.get("rho_max"),
                    "voltage_pu_max": ctx.get("voltage_pu_max"),
                    "violated": [r.get("rule_id") for r in res.violated_rules],
                }) + "\n")
    finally:
        if fh is not None:
            fh.close()

    shielded = pred.copy()
    shielded[blocked] = 1          # BLOCK == escalate to unsafe

    return {
        "shielded": shielded,
        "blocked": blocked,
        "base_violates": base_violates,
        "n_frames": n_frames_seen,
        "fired": fired,
        "n_not_evaluable": n_not_evaluable,
        "n_error": n_error,
        "n_unsupported": n_unsupported,
        "n_logged": n_logged,
        "severities": dict(severities),
    }


def failure_mode(truth: int, pred: int, blocked: bool) -> str:
    if truth == 1 and pred == 0:
        return "missed_violation_caught" if blocked else "missed_violation_uncaught"
    if truth == 0 and pred == 0:
        return "false_block" if blocked else "correct_secure"
    return "predicted_violation"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="GNN vs GNN+shield on N-1 screening")
    ap.add_argument("--tag", default=HOME)
    ap.add_argument("--rules", default=DEDUPED_RULES,
                    help=f"{DEDUPED_RULES}, or a stage-2.5 guarded folder")
    ap.add_argument("--rules-kg", default=None,
                    help="Read rules from the Component C knowledge graph instead "
                         "(kg/knowledge_graph.json). Returns the same set as --rules "
                         "by construction — pinned by tests/test_kg.py — so the "
                         "reported numbers must not move. Overrides --rules.")
    ap.add_argument("--citations", default=None,
                    help="Write the provenance chain for every rule that fired "
                         "(requires --rules-kg). One record per distinct rule, not "
                         "per contingency.")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE,
                    help=f"Eval batch size (default {EVAL_BATCH_SIZE}). LOAD-BEARING — "
                         "BatchNorm uses live batch stats, so changing this changes the "
                         "absolute F1. Deltas stay comparable. See gnn_n1_tightening.md §8.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed decision threshold. Default: selected on the "
                         "neurips2020 VAL split and held across topologies "
                         "(thesis_findings.md §13)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--failures", default=None,
                    help="Write per-contingency failure records here (jsonl)")
    ap.add_argument("--max-failures", type=int, default=50_000,
                    help="Cap on logged failure records (wcci2022 misses ~86k)")
    args = ap.parse_args()

    if args.batch_size != EVAL_BATCH_SIZE:
        print(f"\n  !! WARNING: --batch-size {args.batch_size} != {EVAL_BATCH_SIZE}. "
              f"BatchNorm uses live batch statistics, so absolute F1 will NOT match "
              f"any recorded result. Deltas remain comparable. "
              f"(gnn_n1_tightening.md §8)\n")


    if not os.path.exists(args.checkpoint):
        sys.exit(f"Missing {args.checkpoint} — train the N-1 model first.")
    provider = None
    if args.rules_kg:
        from kg.provider import KgRuleProvider

        provider = KgRuleProvider.from_json(args.rules_kg)
        rules = provider.rules_for(SECURE)
        print(f"Rules from knowledge graph {args.rules_kg} ({len(rules)} served)")
    else:
        rules = load_rules(args.rules)
        if args.citations:
            sys.exit("--citations requires --rules-kg (provenance lives in the graph).")

    print("Computing 36-bus N-1 train-split normalization stats...")
    home_pt = os.path.join(DATA_DIR, "processed_grid_data_n1.pt")
    train_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    home = PreloadedGridDataset(home_pt, device=DEVICE)
    norm = compute_normalization_stats(home, train_idx)
    del home

    # ── threshold: chosen on val, held fixed. Never on the set being reported. ──
    thr = args.threshold
    thr_source = "supplied on the command line"
    if thr is None:
        val_path = os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy")
        if not os.path.exists(val_path):
            sys.exit("No val split on disk; pass --threshold explicitly.")
        print("Selecting the decision threshold on the neurips2020 VAL split...")
        val = score_topology_indices(HOME, args.checkpoint, args.batch_size, norm,
                                     np.load(val_path))
        _, thr = best_f1_and_thr(val["logits"], val["y"])
        thr_source = "selected on the neurips2020 val split, held fixed"
        del val

    scored = score_topology(args.tag, args.checkpoint, args.batch_size, norm)
    y, logits = scored["y"], scored["logits"]
    pred = (logits > thr).astype(int)

    failures_path = (Path(args.failures) if args.failures
                     else RESULTS_DIR / "failures" / f"failures_{args.tag}.jsonl")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    sh = run_shield(scored, pred, rules, args.tag, failures_path, args.max_failures)

    gnn = prf(y, pred)
    shd = prf(y, sh["shielded"])
    ref_f1, _ = best_f1_and_thr(logits, y)
    rule_f1, _ = best_f1_and_thr(scored["rho"], y)

    # ── the conditional view: only where the shield was eligible to act ───────
    eligible = (pred == 0) & sh["base_violates"]
    corrections = int((sh["blocked"] & (y == 1)).sum())
    regressions = int((sh["blocked"] & (y == 0)).sum())
    n_blocked = int(sh["blocked"].sum())

    # ── structural ceiling: are the dangerous errors even reachable? ──────────
    missed = (y == 1) & (pred == 0)
    reachable = int((missed & sh["base_violates"]).sum())
    n_missed = int(missed.sum())

    meta = scored["meta"]
    print(f"\n{'=' * 74}")
    print(f"  N-1 SHIELD — {args.tag}   ({meta.n_line} lines, {meta.n_sub} buses)")
    print(f"  {scored['scope']}")
    print(f"  rules: {len(rules)} from {args.rules}")
    print(f"  threshold: {thr:.4f}  ({thr_source})")
    print(f"{'=' * 74}")
    print(f"  contingencies scored : {len(y):,}   violation rate {y.mean():.1%}")
    print(f"  AP                   : {average_precision_score(y, logits):.4f}")
    print(f"  best rule on removed-line rho : F1 {rule_f1:.4f}")
    print(f"  model at best-on-this-topology threshold : F1 {ref_f1:.4f}  (reference only)")

    print(f"\n  ── ARMS, at the held threshold ─────────────────────────────────")
    print(f"  {'':22} {'F1':>8} {'precision':>10} {'recall':>8} {'missed':>10} {'false alarm':>12}")
    for name, m in (("GNN alone", gnn), ("GNN + shield", shd)):
        print(f"  {name:22} {m['f1']:8.4f} {m['precision']:10.3f} {m['recall']:8.3f} "
              f"{m['missed_violations']:10,} {m['false_alarms']:12,}")
    print(f"  {'delta':22} {shd['f1'] - gnn['f1']:+8.4f} "
          f"{shd['precision'] - gnn['precision']:+10.3f} "
          f"{shd['recall'] - gnn['recall']:+8.3f} "
          f"{shd['missed_violations'] - gnn['missed_violations']:+10,} "
          f"{shd['false_alarms'] - gnn['false_alarms']:+12,}")

    print(f"\n  ── WHERE THE SHIELD COULD SPEAK (the informative view) ─────────")
    print(f"  eligible contingencies (predicted secure, base case violates a rule)")
    print(f"                       : {int(eligible.sum()):,} "
          f"({eligible.mean():.3%} of all scored)")
    print(f"  blocked              : {n_blocked:,}")
    print(f"    corrections (truly a violation)  : {corrections:,}")
    print(f"    regressions (truly secure)       : {regressions:,}")
    if n_blocked:
        print(f"    intervention precision           : {corrections / n_blocked:.3f}")
    else:
        print(f"    intervention precision           : n/a — the shield never fired")

    print(f"\n  ── STRUCTURAL CEILING (independent of rule count) ──────────────")
    print(f"  missed violations               : {n_missed:,}")
    print(f"  on a base case a rule can see   : {reachable:,} "
          f"({reachable / max(n_missed, 1):.2%})")
    print(f"  -> no present-state rule over this vocabulary can reach the other "
          f"{n_missed - reachable:,}.")

    print(f"\n  ── SHIELD HEALTH ──────────────────────────────────────────────")
    print(f"  frames evaluated       : {sh['n_frames']:,}")
    print(f"  NOT_EVALUABLE verdicts : {sh['n_not_evaluable']:,}")
    print(f"  ERROR verdicts         : {sh['n_error']:,}"
          f"{'   <- MUST be 0 (plan §8 gate 2)' if sh['n_error'] else ''}")
    print(f"  Option B counterfactual (unsupported): {sh['n_unsupported']:,}")
    if sh["severities"]:
        print(f"  blocks by severity     : {sh['severities']}")
    print(f"\n  failure log -> {failures_path}")

    if provider is not None and sh["fired"]:
        from kg.schema import format_citation

        print("\n  ── WHICH RULES ACTED, AND ON WHOSE AUTHORITY ──────────────────")
        records = []
        for rid, n in sh["fired"].most_common():
            cit = provider.cite(rid)
            records.append({**cit.to_dict(), "n_blocks": n})
            print(f"\n  {n:,} block(s):")
            for line in format_citation(cit).splitlines():
                print(f"  {line}")
        if args.citations:
            _mkparent(args.citations).write_text(
                json.dumps({"tag": args.tag, "rules_kg": args.rules_kg,
                            "fired": records}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\n  citations -> {args.citations}")
    print()

    if n_blocked == 0:
        print("  ** The shield never fired on this topology. That is a measurement:")
        print("     with this corpus, no base case that the model called secure also")
        print("     violated a constraint. Report it, do not tune around it.\n")

    if args.json:
        _mkparent(args.json).write_text(json.dumps({
            "tag": args.tag,
            "rules": args.rules_kg or args.rules,
            "n_rules": len(rules),
            "threshold": thr,
            "threshold_source": thr_source,
            "n_contingencies": int(len(y)),
            "violation_rate": float(y.mean()),
            "ap": float(average_precision_score(y, logits)),
            "f1_rule_baseline": rule_f1,
            "f1_model_best_on_topology": ref_f1,
            "arms": {"gnn": gnn, "gnn_shield": shd},
            "eligible": int(eligible.sum()),
            "blocked": n_blocked,
            "corrections": corrections,
            "regressions": regressions,
            "missed_violations": n_missed,
            "missed_reachable_by_rule": reachable,
            "shield_health": {k: sh[k] for k in
                              ("n_frames", "n_not_evaluable", "n_error", "n_unsupported")},
            "severities": sh["severities"],
        }, indent=2), encoding="utf-8")
        print(f"  Wrote {args.json}")


def score_topology_indices(tag: str, checkpoint: str, batch_size: int, norm: tuple,
                           indices: np.ndarray) -> dict:
    """score_topology over an explicit index set — used for val-split thresholding."""
    node_mean, node_std, edge_mean, edge_std = norm
    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1.jsonl")
    with open(os.path.join(DATA_DIR, f"grid_dataset_{tag}_n1_meta.json")) as f:
        meta = GridEnvMetadata(json.load(f))
    loader = DataLoader(GridDataset(jsonl, indices, meta),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    model = GridGNN(
        node_features=NODE_FEATURES, edge_features=EDGE_FEATURES,
        hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    lg, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch.x = (batch.x - node_mean) / node_std
            batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_fwd)
            m = batch.line_mask
            lg.append(out[m].float().cpu())
            ys.append(batch.line_y[m].cpu())
    return {"logits": torch.cat(lg).numpy(), "y": torch.cat(ys).numpy().astype(int)}


if __name__ == "__main__":
    main()
