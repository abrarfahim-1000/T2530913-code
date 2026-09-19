"""§24.2, strengthened — the ceiling by ENUMERATION rather than by greedy search.

`thesis_findings.md` §28 answered "is the thin rule corpus the pipeline's fault or
the task's?" by fitting a depth-4 decision tree directly on the answer key and
comparing it to the four extracted rules. The tree lost. The weakness of that
argument is the search, not the framing:

    CART IS GREEDY. It takes the single best split, then the best split beneath
    it. A rule that only works as a PAIR — where neither half is useful alone but
    the conjunction is — is structurally invisible to it.

§28.3 already closed the "too shallow" objection from the other side (depth 8 and
class-balancing are both WORSE). It did not close "too greedy". So what §28 licenses
today is only *greedy trees of depth <= 8 do not beat the corpus*, which is a much
narrower claim than the one the thesis wants to make.

This script removes the search entirely and enumerates the language:

  arm 1  EVERY single-variable threshold rule. Not a sampled grid — every distinct
         cutpoint in the data, both directions, on all 14 variables. If a threshold
         exists that nobody wrote down, this finds it.
  arm 2  EVERY two-variable rule. All variable pairs, both directions per side,
         joined by AND and by OR. Not greedy: it sees pairs a tree cannot reach.
  arm 3  An UNCAPPED gradient-boosted ensemble on the same 14 variables. This is
         deliberately NOT a rule — it is unreadable, so no gate could ship it —
         and that is the point: it bounds the corpus from above by something
         strictly more expressive than any rule in the language.

If the best rule in the whole enumerated language is one the pipeline already
extracted, the ceiling claim stops depending on who searched for what, and the
question "could a human have written a better rule?" (§24.3, §29) is answered by
"nobody could have, and we checked every one."

PROTOCOL — inherited from §24.2 verbatim, none of it discretionary:
  * Target `model predicted secure AND the contingency was a violation`, per
    contingency. The asymmetric error `validate_n1` exists to catch.
  * GATE SEMANTICS. A gate can only act where the model predicted secure, so every
    arm is scored as `rule fires AND eligible`. §28 originally scored over all rows
    and §29.7 corrected it; this script does it right from the start, and reports
    `flag_all_eligible` (block everything the model called secure = NO RULE AT ALL)
    on every grid. Without that column a rule firing on ~100% of frames reads as a
    winner — which actually happened on case14 in §29.2.
  * Held threshold 0.8849, re-derived on the neurips2020 val split and asserted.
  * Eval batch size 64 (pinned project-wide).
  * neurips2020 scores its held-out test split; foreign grids score every frame.

⚠️ Arms 1 and 2 select the winning rule WITH THE ANSWER KEY on the grid being
scored. That is an oracle selection and it is intentional — the point is an upper
bound, not a deployable gate. It makes the enumeration strictly generous to the
challenger and the extracted corpus still has to survive it.

⚠️ The per-frame constraint of §24.2 still applies and still binds both sides: the
context is per frame, the error is per contingency, so no rule here can separate
two contingencies inside one frame. Neither can any extracted rule.

─────────────────────────────────────────────────────────────────────────────
ADDED 2026-09-20 — three arms that close what the enumeration above left open.

  arm A  THE F-BETA CROSSOVER ON THE RULE'S OWN THRESHOLD. The enumeration found
         the F1 optimum ~3 points below the standards' round 100, bought with a
         large loss of precision, and the thesis then DEFENDED 100 by argument
         ("a gate that overrides a model should not spend precision lightly").
         That argument is a claim about an exchange rate, and F-beta is exactly
         that exchange rate: one missed violation is worth beta^2 false blocks.
         So the full `loading_pct >` sweep is emitted and the critical beta — the
         price at which a sub-100 cutpoint overtakes 100 — is solved in closed
         form, per grid. The argument stops being an assertion.

  arm B  HELD-OUT RULE SELECTION. Arms 1 and 2 pick the winner with the answer
         key on the grid being scored, which no deployable gate can do. Arm B
         selects ONE rule on neurips2020 alone and then freezes it, mirroring the
         model's own protocol. It converts "an oracle rule beats the corpus" into
         a statement about what a rule-writer could actually have shipped.

  arm C  PAIR-GRID RESOLUTION. Arm 2 quantises to 32 cutpoints per variable
         because the space squares. Arm C re-runs it at 64 and 128 and asks the
         question that matters — not "does the winning pair change" (a different
         quantile grid moves every cutpoint, so of course it does) but "does any
         resolution find a pair that beats the best SINGLE rule".

Every new figure is verified against a slow `prf` pass over the full contingency
vectors before it is reported, and the run dies on any disagreement.

Usage:
    python evaluation/exhaustive_rules_n1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(".")

from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation.ceiling_tree_n1 import (CONTEXT_VARS, HOME, RECORDED_THRESHOLD,
                                        TAGS, frame_contexts, frame_count,
                                        load_meta, prf, score_model,
                                        shield_reference)
from evaluation.eval_n1_cross_topology import EVAL_BATCH_SIZE, best_f1_and_thr
from scripts.pyg_data import PreloadedGridDataset
from training.config import DATA_DIR, DEVICE
from training.train_gnn import compute_normalization_stats

OUT = os.path.join("results", "ceiling", "exhaustive_rules.json")
SEED = 42
# Cutpoints per variable for the PAIR search. Arm 1 uses every distinct cutpoint;
# arm 2 cannot (91 pairs x every cutpoint squared), so it quantises. 32 quantiles
# per variable x 2 directions x 14 variables = 896 frame masks, and the pair scan
# is then two matrix products against those masks.
PAIR_CUTS = 32
GBT_ITERS = 300
# ── added 2026-09-20, the three arms below ──────────────────────────────────
SWEEP_OUT = os.path.join("results", "ceiling", "loading_pct_sweep.json")
SERVED_THRESHOLD = 100.0  # the cutoff the validated corpus actually ships
BETA_MIN, BETA_MAX, BETA_STEPS = 0.25, 4.0, 61
# Four n_masks^2 float64 matrices live at once inside `best_pair_rule`; this caps
# them rather than discovering the ceiling as a MemoryError mid-run.
MAX_PAIR_MATRIX_GB = 1.5

# Alignment by construction — every arm in this project reproduces these before it
# reports anything. A mismatch means the rows are misaligned and nothing downstream
# is meaningful. Recorded in CLAUDE.md.
RECORDED_ALL_POSITIVE = {"neurips2020": 0.3104, "case14": 0.4345, "wcci2022": 0.3969}
# The best-single-rule baselines from the same table, carried for the reader — this
# arm does not recompute them (they are a rho threshold on the RAW label, not on the
# gate's target), but confusing the two columns is an easy and load-bearing mistake.
RECORDED_BEST_RULE = {"neurips2020": 0.4639, "case14": 0.5392, "wcci2022": 0.4915}
RECORDED_CONTINGENCIES = {"neurips2020": 113205, "case14": 118502, "wcci2022": 742472}


def frame_aggregates(target: np.ndarray, eligible: np.ndarray,
                     counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame target and eligible contingency counts.

    THIS is what makes enumeration cheap. Rules read per-frame variables, so a rule
    is a mask over ~4k-12k frames, not over ~113k-742k contingencies. Scoring one
    then costs two dot products instead of a pass over every contingency.
    """
    n_frames = len(counts)
    frame_id = np.repeat(np.arange(n_frames), counts)
    n_target = np.bincount(frame_id, weights=target, minlength=n_frames)
    n_eligible = np.bincount(frame_id, weights=eligible, minlength=n_frames)
    return n_target, n_eligible


def f1_from_counts(tp: np.ndarray, predicted: np.ndarray,
                   total: float) -> np.ndarray:
    """F1 from (true positives, predicted positives, total positives).

    fp = predicted - tp and fn = total - tp, so 2tp/(2tp+fp+fn) collapses to
    2tp/(predicted + total). Exact, and it needs no boolean arrays.
    """
    denom = np.asarray(predicted, dtype=float) + total
    return np.where(denom > 0, 2.0 * np.asarray(tp, dtype=float) / np.maximum(denom, 1e-12), 0.0)


def best_single_rule(values: np.ndarray, name: str, n_target: np.ndarray,
                     n_eligible: np.ndarray, total: float) -> dict | None:
    """The best `var > t` or `var < t` over EVERY distinct cutpoint in the data.

    One sort plus two cumulative sums scores all thresholds at once: `var < t` is a
    prefix of the sorted order and `var > t` is its suffix. Non-finite frames never
    fire, so they are dropped from the mask but still counted in `total` as misses.
    """
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return None
    order = np.argsort(values[finite], kind="stable")
    ordered = values[finite][order]
    cum_t = np.cumsum(n_target[finite][order])
    cum_e = np.cumsum(n_eligible[finite][order])
    bounds = np.flatnonzero(np.diff(ordered))
    if bounds.size == 0:
        return None  # constant on this grid — it can only fire on all or none

    cuts = 0.5 * (ordered[bounds] + ordered[bounds + 1])
    tp_lt, pred_lt = cum_t[bounds], cum_e[bounds]
    tp_gt, pred_gt = cum_t[-1] - tp_lt, cum_e[-1] - pred_lt
    f1_lt = f1_from_counts(tp_lt, pred_lt, total)
    f1_gt = f1_from_counts(tp_gt, pred_gt, total)

    if f1_lt.max() >= f1_gt.max():
        k, op, tp, pred = int(f1_lt.argmax()), "<", tp_lt, pred_lt
    else:
        k, op, tp, pred = int(f1_gt.argmax()), ">", tp_gt, pred_gt
    mask = values < cuts[k] if op == "<" else values > cuts[k]
    return {"condition": f"{name} {op} {cuts[k]:.6g}", "variable": name,
            "operator": op, "threshold": float(cuts[k]),
            "f1": float(f1_from_counts(tp[k], pred[k], total)),
            "tp": float(tp[k]), "predicted": float(pred[k]),
            "frames_firing": float(np.mean(mask))}


def build_pair_masks(X: np.ndarray, n_cuts: int) -> tuple:
    """Every `var > c` / `var < c` frame mask on a quantile grid, stacked.

    ⚠️ `labels` renders `c` at 6 significant figures, which is NOT always enough to
    reproduce the mask — quantile cutpoints land on tightly clustered values (the
    voltage bands especially), where rounding moves frames across the cut. The
    exact cut is therefore carried alongside in `terms`, and that is what the
    verification re-derives from. Added 2026-09-20; the labels and every score
    computed from these masks are unchanged.
    """
    masks, labels, terms = [], [], []
    for j, name in enumerate(CONTEXT_VARS):
        v = X[:, j]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            continue
        qs = np.unique(np.quantile(finite, np.linspace(0, 1, n_cuts + 2)[1:-1]))
        for c in qs:
            for op in (">", "<"):
                mask = v > c if op == ">" else v < c
                if 0 < mask.sum() < len(v):  # skip masks that fire on all or none
                    masks.append(mask)
                    labels.append(f"{name} {op} {c:.6g}")
                    terms.append((name, op, float(c)))
    return np.asarray(masks, dtype=np.float32), labels, terms


def best_pair_rule(masks: np.ndarray, labels: list[str], n_target: np.ndarray,
                   n_eligible: np.ndarray, total: float,
                   terms: list | None = None) -> dict | None:
    """The best two-term AND / OR over the stacked masks.

    Two matrix products give every pair at once:
        TP_and[i, j] = sum_f mask_i[f] * mask_j[f] * n_target[f]
    and OR follows by inclusion-exclusion, so nothing is scanned twice.
    """
    if len(labels) < 2:
        return None
    tp_and = (masks * n_target) @ masks.T
    pred_and = (masks * n_eligible) @ masks.T
    tp_one, pred_one = np.diag(tp_and).copy(), np.diag(pred_and).copy()
    tp_or = tp_one[:, None] + tp_one[None, :] - tp_and
    pred_or = pred_one[:, None] + pred_one[None, :] - pred_and

    upper = np.triu(np.ones_like(tp_and, dtype=bool), k=1)
    best = None
    for op, tp_m, pred_m in (("and", tp_and, pred_and), ("or", tp_or, pred_or)):
        f1 = np.where(upper, f1_from_counts(tp_m, pred_m, total), -1.0)
        i, j = np.unravel_index(int(f1.argmax()), f1.shape)
        cand = {"condition": f"({labels[i]}) {op} ({labels[j]})", "operator": op,
                "f1": float(f1[i, j]), "tp": float(tp_m[i, j]),
                "predicted": float(pred_m[i, j]),
                "frames_firing": float(np.mean(
                    (masks[i] > 0) & (masks[j] > 0) if op == "and"
                    else (masks[i] > 0) | (masks[j] > 0)))}
        if terms is not None:  # exact cutpoints — `condition` rounds them to 6 s.f.
            cand["terms"] = [list(terms[i]), list(terms[j])]
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    return best


def verify_fast_path(rule: dict, X: np.ndarray, counts: np.ndarray,
                     target: np.ndarray, eligible: np.ndarray, tag: str) -> None:
    """Re-score one winning rule the slow way and assert the two agree.

    The frame-level arithmetic is the whole reason this is tractable, so it is
    checked against `prf` over the full contingency vectors rather than trusted.
    """
    name, op, raw = rule["condition"].split()
    value = float(raw)
    v = X[:, CONTEXT_VARS.index(name)]
    frame_mask = v < value if op == "<" else v > value
    fires = np.repeat(frame_mask, counts)
    slow = prf(target, fires & eligible)
    if abs(slow["f1"] - rule["f1"]) > 1e-9:
        raise SystemExit(f"{tag}: fast path {rule['f1']:.12f} != prf {slow['f1']:.12f} "
                         f"for `{rule['condition']}` — the frame arithmetic is wrong")


def check_alignment(tag: str, y: np.ndarray) -> dict:
    """Reproduce the recorded all-positive baseline and contingency count."""
    p = float(y.mean())
    all_pos = 2 * p / (1 + p) if p else 0.0
    if len(y) != RECORDED_CONTINGENCIES[tag]:
        raise SystemExit(f"{tag}: {len(y):,} contingencies, recorded "
                         f"{RECORDED_CONTINGENCIES[tag]:,} — rows are misaligned")
    if abs(all_pos - RECORDED_ALL_POSITIVE[tag]) > 5e-4:
        raise SystemExit(f"{tag}: all-positive F1 {all_pos:.4f}, recorded "
                         f"{RECORDED_ALL_POSITIVE[tag]} — labels are misaligned")
    return {"contingencies": int(len(y)), "all_positive_f1": round(all_pos, 4),
            "recorded_best_single_rule_baseline": RECORDED_BEST_RULE[tag]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--pair-cuts", type=int, default=PAIR_CUTS)
    # ── arm A: the F-beta crossover on the served rule's threshold ───────────
    ap.add_argument("--served-threshold", type=float, default=SERVED_THRESHOLD,
                    help="the standards' loading_pct cutoff that the corpus ships")
    ap.add_argument("--beta-min", type=float, default=BETA_MIN)
    ap.add_argument("--beta-max", type=float, default=BETA_MAX)
    ap.add_argument("--beta-steps", type=int, default=BETA_STEPS)
    ap.add_argument("--sweep-json", default=SWEEP_OUT,
                    help="sidecar for the per-cutpoint loading_pct curves")
    # ── arm B: held-out rule selection ──────────────────────────────────────
    ap.add_argument("--select-split", default="train", choices=["train", "val"],
                    help="which neurips2020 rows arm B selects its single rule on; "
                         "both are disjoint from the test rows it is then scored on")
    # ── arm C: pair-grid resolution ─────────────────────────────────────────
    ap.add_argument("--pair-resolutions", default="64,128",
                    help="comma-separated finer pair grids to re-run; empty to skip")
    ap.add_argument("--max-pair-matrix-gb", type=float, default=MAX_PAIR_MATRIX_GB)
    args = ap.parse_args()
    resolutions = [int(c) for c in args.pair_resolutions.split(",") if c.strip()]
    betas = np.geomspace(args.beta_min, args.beta_max, args.beta_steps)
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

    print("Re-deriving the decision threshold on the neurips2020 VAL split...")
    val_logits, val_y = score_model(HOME, val_idx, home_meta, norm)
    _, thr = best_f1_and_thr(val_logits, val_y)
    if abs(thr - RECORDED_THRESHOLD) > 5e-4:
        raise SystemExit(f"threshold {thr:.6f} does not reproduce the recorded "
                         f"{RECORDED_THRESHOLD} — the protocol has silently changed")
    print(f"  threshold {thr:.6f}")

    def build(tag: str, idx: np.ndarray, meta) -> dict:
        logits, y = score_model(tag, idx, meta, norm)
        X, counts = frame_contexts(tag, idx, meta)
        if counts.sum() != len(y):
            raise ValueError(f"{tag}: {counts.sum()} context rows vs {len(y)} labels")
        eligible = logits <= thr
        return {"X": X, "counts": counts, "y": y, "eligible": eligible,
                "target": (eligible & (y == 1)).astype(int)}

    report = {
        "protocol": {
            "target": "model predicted secure AND the contingency was a violation",
            "semantics": "gate — a rule can only act where the model predicted secure",
            "model_threshold": float(thr),
            "eval_batch_size": EVAL_BATCH_SIZE,
            "variables": list(CONTEXT_VARS),
            "pair_cuts_per_variable": args.pair_cuts,
            "selection": ("arms 1 and 2 pick the winner WITH the answer key on the "
                          "grid being scored — an oracle selection, and deliberately "
                          "generous to the challenger"),
            "supersedes": ("§28's greedy depth-4 tree as the ceiling instrument; the "
                           "tree's figures are carried alongside for continuity"),
            "seed": SEED,
            # the three arms added 2026-09-20 — every flag they run under
            "fbeta_arm": {
                "served_threshold": args.served_threshold,
                "beta_grid": {"min": args.beta_min, "max": args.beta_max,
                              "steps": args.beta_steps, "spacing": "geometric"},
                "question": ("the thesis DEFENDS 100 over the F1-optimal ~97 by "
                             "argument; this prices the argument in beta, where "
                             "beta^2 is how many false blocks one missed violation "
                             "is worth"),
                "sweep_sidecar": args.sweep_json,
            },
            "held_selection_arm": {
                "select_split": args.select_split,
                "question": ("arms 1 and 2 select with the answer key on the grid "
                             "being scored; this selects on neurips2020 alone and "
                             "then freezes the rule, which is what a rule-writer "
                             "could actually have deployed"),
            },
            "pair_resolution_arm": {
                "resolutions": resolutions,
                "baseline_cuts": args.pair_cuts,
                "max_pair_matrix_gb": args.max_pair_matrix_gb,
                "question": ("arm 2 quantises to 32 cutpoints per variable; does a "
                             "finer grid find a pair the coarse one could not see?"),
            },
        },
        "grids": {},
    }

    gbt, gbt_thr = _fit_gbt(build, train_idx, val_idx, home_meta)
    report["protocol"]["gbt"] = {"max_depth": None, "max_iter": GBT_ITERS,
                                 "held_threshold": float(gbt_thr),
                                 "fit_on": f"neurips2020 train ({len(train_idx)} frames)"}

    sel_idx = train_idx if args.select_split == "train" else val_idx
    sel = select_rule_on_home(build, sel_idx, home_meta, args.select_split)
    sel["scoring_rows"] = ("neurips2020 HELD-OUT TEST split for neurips2020; every "
                           "frame for case14 and wcci2022 — disjoint from selection")
    report["held_out_selected_rule"] = sel
    print(f"  selected: {sel['condition']}  "
          f"(F1 {sel['f1_on_selection_rows']:.4f} on the selection rows)")

    sweeps = {}
    for tag in TAGS:
        meta = home_meta if tag == HOME else load_meta(tag)
        idx = test_idx if tag == HOME else np.arange(frame_count(tag))
        scope = "held-out test split" if tag == HOME else "all frames (unseen topology)"
        print(f"Scoring {tag} ({len(idx):,} frames)...")
        d = build(tag, idx, meta)
        report["grids"][tag], sweeps[tag] = _score_grid(
            tag, scope, d, gbt, gbt_thr, args, sel, betas, resolutions)
        del d

    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    with open(args.sweep_json, "w") as fh:
        json.dump({"protocol": report["protocol"]["fbeta_arm"],
                   "note": ("per-cutpoint `loading_pct > t` curves, parallel arrays; "
                            "the decision numbers derived from them live in "
                            f"{args.json}"),
                   "grids": sweeps}, fh, separators=(",", ":"))
    _print(report)
    print(f"\n  written: {args.json}")
    print(f"  written: {args.sweep_json}  (full per-cutpoint sweeps)")


def _fit_gbt(build, train_idx: np.ndarray, val_idx: np.ndarray, meta):
    """Arm 3 — the uncapped ensemble, plus its held threshold from the val split."""
    print(f"Fitting the UNCAPPED gradient-boosted ceiling "
          f"({len(train_idx):,} frames)...")
    tr = build(HOME, train_idx, meta)
    X = np.repeat(tr["X"], tr["counts"], axis=0)
    gbt = HistGradientBoostingClassifier(max_depth=None, max_iter=GBT_ITERS,
                                         random_state=SEED)
    gbt.fit(X, tr["target"])
    del tr, X

    va = build(HOME, val_idx, meta)
    proba = gbt.predict_proba(np.repeat(va["X"], va["counts"], axis=0))[:, 1]
    _, gbt_thr = best_f1_and_thr(np.where(va["eligible"], proba, -np.inf), va["target"])
    del va
    print(f"  gbt held threshold {gbt_thr:.6f}")
    return gbt, gbt_thr


def _score_grid(tag: str, scope: str, d: dict, gbt, gbt_thr: float, args,
                sel: dict, betas: np.ndarray, resolutions: list) -> tuple:
    """All arms plus both baselines on one grid. Returns (report, sweep arrays)."""
    X, counts, target, eligible = d["X"], d["counts"], d["target"], d["eligible"]
    n_target, n_eligible = frame_aggregates(target, eligible, counts)
    total = float(target.sum())

    singles = [r for r in (best_single_rule(X[:, j], v, n_target, n_eligible, total)
                           for j, v in enumerate(CONTEXT_VARS)) if r]
    singles.sort(key=lambda r: -r["f1"])
    verify_fast_path(singles[0], X, counts, target, eligible, tag)

    masks, labels, terms = build_pair_masks(X, args.pair_cuts)
    print(f"  {len(labels)} frame masks -> {len(labels) * (len(labels) - 1) // 2:,} "
          f"pairs x 2 operators")
    pair = best_pair_rule(masks, labels, n_target, n_eligible, total, terms)
    del masks
    if pair:  # arm 2 was never verified against the slow path; it is now
        pair["verification"] = verify_pair_fast_path(pair, X, counts, target,
                                                     eligible, tag)

    proba = gbt.predict_proba(np.repeat(X, counts, axis=0))[:, 1]
    gated = np.where(eligible, proba, -np.inf)
    out = {
        "scope": scope,
        "alignment": check_alignment(tag, d["y"]),
        "target_contingencies": int(total),
        "flag_all_eligible": prf(target, eligible),
        "best_single_rule": singles[0],
        "top_single_rules": singles[:5],
        "best_pair_rule": pair,
        "gbt_held": prf(target, (proba >= gbt_thr) & eligible),
        "gbt_oracle_f1": float(best_f1_and_thr(gated, target)[0]),
        "shield_on_same_target": shield_reference(tag),
    }
    del proba, gated

    out["loading_pct_fbeta"] = loading_pct_fbeta_arm(
        X, counts, target, eligible, n_target, n_eligible, total, betas,
        args.served_threshold, tag)
    out["held_out_selected_rule"] = apply_held_rule(
        sel, X, counts, target, eligible, n_target, n_eligible, total, tag)
    out["pair_resolution"] = pair_resolution_arm(
        X, counts, target, eligible, n_target, n_eligible, total, pair,
        resolutions, args.max_pair_matrix_gb, tag)
    # The question arm C actually answers is not "does the winning pair change"
    # (a finer quantile grid moves every cutpoint, so of course it does) but
    # "does a finer grid find a pair that beats the best SINGLE rule".
    scored = [r for r in out["pair_resolution"] if not r.get("skipped")]
    best_single_f1 = singles[0]["f1"]
    out["pair_resolution_summary"] = {
        "best_single_rule_f1": best_single_f1,
        "best_pair_f1_at_32_cuts": (pair or {}).get("f1"),
        "best_pair_f1_over_all_resolutions": max(
            [r["f1"] for r in scored] + ([pair["f1"]] if pair else []), default=None),
        "resolutions_scored": [r["cuts"] for r in scored],
        "any_resolution_beats_best_single": any(
            r["f1"] > best_single_f1 for r in scored) or bool(
            pair and pair["f1"] > best_single_f1),
        "spread_across_resolutions": (
            float(max(r["f1"] for r in scored) - min(r["f1"] for r in scored))
            if scored else None),
    }

    cuts, tp, pred = gt_sweep(X[:, CONTEXT_VARS.index("loading_pct")],
                              n_target, n_eligible)
    f1 = f1_from_counts(tp, pred, total)
    sweep = {"variable": "loading_pct", "operator": ">",
             "target_contingencies": int(total),
             "threshold": [round(float(c), 6) for c in cuts],
             "tp": [int(x) for x in tp], "predicted": [int(x) for x in pred],
             "precision": [round(float(a / b), 8) if b > 0 else 0.0
                           for a, b in zip(tp, pred)],
             "recall": [round(float(x / total), 8) for x in tp],
             "f1": [round(float(x), 8) for x in f1]}
    return out, sweep


def _print(rep: dict) -> None:
    print(f"\n{'=' * 80}")
    print("  §24.2 by ENUMERATION — every 1- and 2-term rule in the language")
    print(f"  {rep['protocol']['target']}")
    print(f"  gate semantics; winner chosen WITH the answer key (an upper bound)")
    print(f"{'=' * 80}")
    for tag, g in rep["grids"].items():
        s, fa = g["shield_on_same_target"], g["flag_all_eligible"]
        print(f"\n  {tag}  —  {g['scope']}")
        print(f"    alignment             : {g['alignment']['contingencies']:,} "
              f"contingencies, all-positive {g['alignment']['all_positive_f1']} OK")
        print(f"    target                : {g['target_contingencies']:,}")
        print(f"    flag-all-eligible     : F1 {fa['f1']:.4f}   <- NO RULE AT ALL")
        b = g["best_single_rule"]
        print(f"    BEST SINGLE RULE      : F1 {b['f1']:.4f}   fires on "
              f"{b['frames_firing']:.1%} of frames   [{b['condition']}]")
        p = g["best_pair_rule"]
        if p:
            print(f"    BEST PAIR RULE        : F1 {p['f1']:.4f}   fires on "
                  f"{p['frames_firing']:.1%} of frames\n"
                  f"                            [{p['condition']}]")
        print(f"    GBT uncapped (held)   : F1 {g['gbt_held']['f1']:.4f}   "
              f"(oracle {g['gbt_oracle_f1']:.4f})")
        if s:
            print(f"    EXTRACTED SHIELD      : F1 {s['f1']:.4f}   "
                  f"prec {s['precision']:.3f}  rec {s['recall']:.3f}")
        print("    top single rules:")
        for r in g["top_single_rules"]:
            print(f"      F1 {r['f1']:.4f}  fires {r['frames_firing']:.1%}  "
                  f"{r['condition']}")
        _print_new_arms(g)
    sel = rep.get("held_out_selected_rule")
    if sel:
        print(f"\n  HELD-OUT SELECTION — one rule, chosen on {sel['selection_rows']}")
        print(f"    rule                  : {sel['condition']}")
        for tag, g in rep["grids"].items():
            h = g["held_out_selected_rule"]
            s = g["shield_on_same_target"]
            print(f"    {tag:<13}: F1 {h['f1']:.4f}  prec {h['precision']:.3f}  "
                  f"rec {h['recall']:.3f}   vs oracle "
                  f"{g['best_single_rule']['f1']:.4f}"
                  + (f"   vs shield {s['f1']:.4f}" if s else ""))


def _print_new_arms(g: dict) -> None:
    fb = g.get("loading_pct_fbeta") or {}
    if "served_point" in fb:
        sp, op = fb["served_point"], fb["f1_optimum_point"]
        print(f"    loading_pct > {sp['threshold']:g} (SERVED) : F1 {sp['f1']:.4f}  "
              f"prec {sp['precision']:.3f}  rec {sp['recall']:.3f}")
        print(f"    loading_pct > {op['threshold']:.4f} (F1 opt): F1 {op['f1']:.4f}  "
              f"prec {op['precision']:.3f}  rec {op['recall']:.3f}   "
              f"(+{fb['f1_gain_of_optimum']:.4f} F1, "
              f"-{fb['precision_cost_of_optimum']:.3f} precision)")
        b, a = fb["critical_beta_vs_f1_optimum"], fb["critical_beta_any_lower_cutpoint"]
        bs = "n/a" if b is None else f"{b:.4f}  (beta^2 = {b * b:.4f})"
        as_ = "n/a" if a is None else f"{a:.4f}  (beta^2 = {a * a:.4f})"
        print(f"    CRITICAL BETA         : {bs}   vs the F1 optimum")
        print(f"      any lower cutpoint  : {as_}   first at loading_pct > "
              f"{fb['first_cutpoint_to_overtake_served']}")
    for r in g.get("pair_resolution", []):
        if r.get("skipped"):
            print(f"    pair grid {r['cuts']:>4} cuts   : SKIPPED — {r['reason']}")
        else:
            print(f"    pair grid {r['cuts']:>4} cuts   : F1 {r['f1']:.4f}  "
                  f"(delta vs 32 cuts {r.get('f1_delta_vs_32_cuts', 0.0):+.4f}, "
                  f"same rule: {r.get('same_condition_as_32_cuts')})")
    ps = g.get("pair_resolution_summary")
    if ps and ps["resolutions_scored"]:
        print(f"    pair grid VERDICT     : best pair over "
              f"{[32] + ps['resolutions_scored']} cuts = "
              f"{ps['best_pair_f1_over_all_resolutions']:.4f}  vs best single "
              f"{ps['best_single_rule_f1']:.4f}  -> beats single: "
              f"{ps['any_resolution_beats_best_single']}")



# ── §30 extensions — three measurement arms added 2026-09-20 ─────────────────
# Arm A (issue A2): the F-beta crossover on the served rule's own threshold.
# Arm B (issue A3.1): held-out rule selection, replacing the oracle selection.
# Arm C (issue A3.2): does the 32-cut pair grid hide a pair a finer grid finds?


def fbeta_from_counts(tp, predicted, total: float, beta):
    """F-beta from counts, exactly as `f1_from_counts` does for beta = 1.

    P = tp/predicted and R = tp/total, so
        (1+b^2) P R / (b^2 P + R)  ==  (1+b^2) tp / (b^2 * total + predicted).
    Broadcasts, so a (B,1) column of betas against a (1,N) row of cutpoints
    scores the whole beta x threshold plane in one expression.
    """
    b2 = np.asarray(beta, dtype=float) ** 2
    denom = b2 * float(total) + np.asarray(predicted, dtype=float)
    return np.where(denom > 0,
                    (1.0 + b2) * np.asarray(tp, dtype=float) / np.maximum(denom, 1e-12),
                    0.0)


def gt_sweep(values: np.ndarray, n_target: np.ndarray, n_eligible: np.ndarray):
    """(cuts, tp, predicted) for `values > t` at EVERY distinct cutpoint.

    The same sorted-cumsum identity `best_single_rule` uses, exposed so the whole
    curve is reportable and not just its argmax.
    """
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.empty(0), np.empty(0), np.empty(0)
    order = np.argsort(values[finite], kind="stable")
    ordered = values[finite][order]
    cum_t = np.cumsum(n_target[finite][order])
    cum_e = np.cumsum(n_eligible[finite][order])
    bounds = np.flatnonzero(np.diff(ordered))
    if bounds.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    cuts = 0.5 * (ordered[bounds] + ordered[bounds + 1])
    return cuts, cum_t[-1] - cum_t[bounds], cum_e[-1] - cum_e[bounds]


def counts_for_threshold(values: np.ndarray, thr: float, op: str,
                         n_target: np.ndarray, n_eligible: np.ndarray):
    mask = values > thr if op == ">" else values < thr
    return float(n_target[mask].sum()), float(n_eligible[mask].sum()), mask


def _row(thr: float, tp: float, pred: float, total: float) -> dict:
    return {"threshold": float(thr), "tp": int(round(tp)), "predicted": int(round(pred)),
            "precision": float(tp / pred) if pred > 0 else 0.0,
            "recall": float(tp / total) if total > 0 else 0.0,
            "f1": float(f1_from_counts(tp, pred, total))}


def verify_gt_point(thr: float, op: str, X: np.ndarray, counts: np.ndarray,
                    target: np.ndarray, eligible: np.ndarray, expected: dict,
                    tag: str, var: str = "loading_pct",
                    beta: float | None = None) -> None:
    """Re-derive one operating point from the raw contingency vectors.

    `expected` carries precision/recall/f1 computed from frame counts; this
    recomputes them with `prf` over every contingency and refuses to continue on
    any disagreement. Extends `verify_fast_path`'s discipline to the new arms.
    """
    v = X[:, CONTEXT_VARS.index(var)]
    fires = np.repeat(v > thr if op == ">" else v < thr, counts)
    slow = prf(target, fires & eligible)
    for key in ("precision", "recall", "f1"):
        if abs(slow[key] - expected[key]) > 1e-9:
            raise SystemExit(f"{tag}: fast path {key} {expected[key]:.12f} != prf "
                             f"{slow[key]:.12f} for `{var} {op} {thr:.10g}`")
    if beta is not None and "fbeta" in expected:
        p, r = slow["precision"], slow["recall"]
        b2 = beta * beta
        slow_fb = (1 + b2) * p * r / max(b2 * p + r, 1e-12)
        if abs(slow_fb - expected["fbeta"]) > 1e-9:
            raise SystemExit(f"{tag}: F-beta({beta}) {expected['fbeta']:.12f} != "
                             f"{slow_fb:.12f} from precision/recall — identity wrong")


def crossover_beta(tp_a: float, pred_a: float, tp_b: float, pred_b: float,
                   total: float) -> float | None:
    """The beta at which operating point A overtakes B, in closed form.

    F_beta = (1+b^2) tp / (b^2 T + pred), so equality of A and B gives
        b^2 = (tp_B * pred_A - tp_A * pred_B) / (T * (tp_A - tp_B)).
    A is assumed the higher-recall, lower-precision point. Returns 0.0 when A
    dominates B at every beta, and None when A can never overtake B.
    """
    den = total * (tp_a - tp_b)
    if den <= 0:
        return None
    num = tp_b * pred_a - tp_a * pred_b
    return float(np.sqrt(max(num, 0.0) / den))


def loading_pct_fbeta_arm(X: np.ndarray, counts: np.ndarray, target: np.ndarray,
                          eligible: np.ndarray, n_target: np.ndarray,
                          n_eligible: np.ndarray, total: float, betas: np.ndarray,
                          served: float, tag: str) -> dict:
    """ARM A — the whole `loading_pct >` curve, and where 100 stops being optimal.

    §30 reported that the F1 optimum sits ~3 points BELOW the standards' round
    100 and that it buys that F1 by spending precision. Whether the trade is
    worth taking is a statement about how a missed violation is priced against a
    false block — which is exactly what beta is. So price it, rather than
    asserting that a gate "should not spend precision lightly".
    """
    v = X[:, CONTEXT_VARS.index("loading_pct")]
    cuts, tp, pred = gt_sweep(v, n_target, n_eligible)
    if cuts.size == 0:
        return {"note": "loading_pct is constant on this grid — no sweep exists"}

    s_tp, s_pred, _ = counts_for_threshold(v, served, ">", n_target, n_eligible)
    served_row = _row(served, s_tp, s_pred, total)
    verify_gt_point(served, ">", X, counts, target, eligible, served_row, tag)

    f1 = f1_from_counts(tp, pred, total)
    k = int(f1.argmax())
    opt_row = _row(cuts[k], tp[k], pred[k], total)
    verify_gt_point(cuts[k], ">", X, counts, target, eligible, opt_row, tag)

    # the beta plane: rows are betas, columns are cutpoints
    fb = fbeta_from_counts(tp[None, :], pred[None, :], total, betas[:, None])
    fb_served = fbeta_from_counts(s_tp, s_pred, total, betas)
    arg = fb.argmax(axis=1)
    table = []
    for i, b in enumerate(betas):
        j = int(arg[i])
        table.append({"beta": float(b), "threshold": float(cuts[j]),
                      "precision": float(tp[j] / pred[j]) if pred[j] > 0 else 0.0,
                      "recall": float(tp[j] / total) if total > 0 else 0.0,
                      "fbeta": float(fb[i, j]), "fbeta_at_served": float(fb_served[i]),
                      "argmax_below_served": bool(cuts[j] < served)})
    j0 = int(arg[0])
    verify_gt_point(table[0]["threshold"], ">", X, counts, target, eligible,
                    {"precision": table[0]["precision"], "recall": table[0]["recall"],
                     "f1": float(f1_from_counts(tp[j0], pred[j0], total)),
                     "fbeta": table[0]["fbeta"]},
                    tag, beta=float(betas[0]))

    # exact crossovers against the served 100 — closed form, no grid search
    b_any, thr_any = None, None
    for j in np.flatnonzero(cuts < served):
        b = crossover_beta(tp[j], pred[j], s_tp, s_pred, total)
        if b is not None and (b_any is None or b < b_any):
            b_any, thr_any = b, float(cuts[j])
    b_opt = (crossover_beta(tp[k], pred[k], s_tp, s_pred, total)
             if cuts[k] < served else None)

    flags = [t["argmax_below_served"] for t in table]
    first = flags.index(True) if True in flags else None
    monotone = all(flags[first:]) if first is not None else True
    return {
        "served_threshold": float(served),
        "served_point": served_row,
        "f1_optimum_point": opt_row,
        "f1_gain_of_optimum": float(opt_row["f1"] - served_row["f1"]),
        "precision_cost_of_optimum": float(served_row["precision"]
                                           - opt_row["precision"]),
        "cutpoints": int(cuts.size),
        "beta_grid": {"min": float(betas[0]), "max": float(betas[-1]),
                      "steps": int(betas.size)},
        "argmax_by_beta": table,
        "first_beta_on_grid_with_argmax_below_served": (
            float(betas[first]) if first is not None else None),
        # the decision-relevant numbers
        "critical_beta_vs_f1_optimum": b_opt,
        "critical_beta_squared_vs_f1_optimum": (b_opt ** 2 if b_opt is not None
                                                else None),
        "critical_beta_any_lower_cutpoint": b_any,
        "critical_beta_squared_any_lower_cutpoint": (b_any ** 2 if b_any is not None
                                                     else None),
        "first_cutpoint_to_overtake_served": thr_any,
        "argmax_is_monotone_in_beta": bool(monotone),
        "shield_cross_check": {
            "sweep_predicted_at_100": served_row["predicted"],
            "sweep_tp_at_100": served_row["tp"],
            "note": ("the served corpus's only firing predicate is loading_pct > 100, "
                     "so these should track shield_on_same_target's blocked/caught"),
        },
    }


def select_rule_on_home(build, idx: np.ndarray, meta, split_name: str) -> dict:
    """ARM B, part 1 — pick one rule on neurips2020 WITHOUT the answer key elsewhere.

    Arms 1 and 2 of this script choose the winner on the grid being scored, which
    no deployable gate can do. This mirrors the model's own protocol instead:
    everything is chosen on neurips2020, then frozen and carried unchanged.
    """
    print(f"Selecting ONE rule on the neurips2020 {split_name} split "
          f"({len(idx):,} frames)...")
    d = build(HOME, idx, meta)
    n_target, n_eligible = frame_aggregates(d["target"], d["eligible"], d["counts"])
    total = float(d["target"].sum())
    singles = [r for r in (best_single_rule(d["X"][:, j], v, n_target, n_eligible, total)
                           for j, v in enumerate(CONTEXT_VARS)) if r]
    singles.sort(key=lambda r: -r["f1"])
    win = singles[0]
    verify_fast_path(win, d["X"], d["counts"], d["target"], d["eligible"],
                     f"{HOME}/{split_name}")

    # A rule is deployed as it is WRITTEN, so the 6-significant-figure threshold in
    # the condition string IS the rule — not the full-precision midpoint behind it.
    written = float(f"{win['threshold']:.6g}")
    v = d["X"][:, CONTEXT_VARS.index(win["variable"])]
    tp, pred, _ = counts_for_threshold(v, written, win["operator"], n_target, n_eligible)
    if tp != win["tp"] or pred != win["predicted"]:
        raise SystemExit(f"rounding `{win['condition']}` to 6 significant figures "
                         f"moves its counts: ({tp},{pred}) vs "
                         f"({win['tp']},{win['predicted']})")
    out = {"selection_rows": f"neurips2020 {split_name} split ({len(idx)} frames, "
                             f"{len(d['y'])} contingencies)",
           "selection_is_disjoint_from_scoring_rows": split_name != "test",
           "condition": win["condition"], "variable": win["variable"],
           "operator": win["operator"], "threshold": written,
           "f1_on_selection_rows": win["f1"],
           "precision_on_selection_rows": float(tp / pred) if pred else 0.0,
           "recall_on_selection_rows": float(tp / total) if total else 0.0,
           "runners_up_on_selection_rows": [
               {"condition": r["condition"], "f1": r["f1"]} for r in singles[1:4]]}
    del d
    return out


def apply_held_rule(sel: dict, X: np.ndarray, counts: np.ndarray, target: np.ndarray,
                    eligible: np.ndarray, n_target: np.ndarray, n_eligible: np.ndarray,
                    total: float, tag: str) -> dict:
    """ARM B, part 2 — that one frozen rule, scored here, verified both ways."""
    v = X[:, CONTEXT_VARS.index(sel["variable"])]
    tp, pred, mask = counts_for_threshold(v, sel["threshold"], sel["operator"],
                                          n_target, n_eligible)
    fast = _row(sel["threshold"], tp, pred, total)
    slow = prf(target, np.repeat(mask, counts) & eligible)
    if abs(slow["f1"] - fast["f1"]) > 1e-9:
        raise SystemExit(f"{tag}: held rule fast F1 {fast['f1']:.12f} != prf "
                         f"{slow['f1']:.12f}")
    return {"condition": sel["condition"], "f1": slow["f1"],
            "precision": slow["precision"], "recall": slow["recall"],
            "tp": slow["tp"], "fp": slow["fp"], "fn": slow["fn"],
            "frames_firing": float(np.mean(mask)),
            "selected_on": sel["selection_rows"]}


def _cut_mask(name: str, op: str, cut: float, X: np.ndarray) -> np.ndarray:
    v = X[:, CONTEXT_VARS.index(name)]
    return v < cut if op == "<" else v > cut


def _term_mask(term: str, X: np.ndarray) -> np.ndarray:
    name, op, raw = term.split()
    return _cut_mask(name, op, float(raw), X)


def _pair_mask(rule: dict, X: np.ndarray, exact: bool) -> np.ndarray:
    op = rule["operator"]
    if exact and "terms" in rule:
        a, b = (_cut_mask(n, o, c, X) for n, o, c in rule["terms"])
    else:
        left, right = rule["condition"].split(f") {op} (")
        a = _term_mask(left.lstrip("("), X)
        b = _term_mask(right.rstrip(")"), X)
    return (a & b) if op == "and" else (a | b)


def verify_pair_fast_path(rule: dict, X: np.ndarray, counts: np.ndarray,
                          target: np.ndarray, eligible: np.ndarray, tag: str) -> dict:
    """The pair equivalent of `verify_fast_path` — the matrix products are checked.

    Re-derives the winner's mask from `X` and re-scores it over every contingency
    with `prf`, refusing to continue on any disagreement. It also scores the rule
    AS WRITTEN — thresholds at the 6 significant figures `condition` prints — and
    returns the gap, because that gap is a property of the reported string and not
    an error in the search.
    """
    slow = prf(target, np.repeat(_pair_mask(rule, X, exact=True), counts) & eligible)
    if abs(slow["f1"] - rule["f1"]) > 1e-9:
        raise SystemExit(f"{tag}: pair fast path {rule['f1']:.12f} != prf "
                         f"{slow['f1']:.12f} for `{rule['condition']}`")
    written = prf(target, np.repeat(_pair_mask(rule, X, exact=False), counts) & eligible)
    return {"f1_as_written_at_6_significant_figures": written["f1"],
            "rounding_gap": float(written["f1"] - slow["f1"])}


def pair_resolution_arm(X: np.ndarray, counts: np.ndarray, target: np.ndarray,
                        eligible: np.ndarray, n_target: np.ndarray,
                        n_eligible: np.ndarray, total: float, base: dict | None,
                        resolutions: list, max_gb: float, tag: str) -> list:
    """ARM C — re-run the pair search at finer grids and see if the winner moves.

    Arm 2 quantises to 32 cutpoints per variable because the space squares, which
    leaves open "the grid was too coarse to see it". Nothing else in this project
    measures that; this does.
    """
    out = []
    for n_cuts in resolutions:
        masks, labels, terms = build_pair_masks(X, n_cuts)
        gb = (len(labels) ** 2 * 8 * 4) / 1e9  # tp/pred x and/or — four such matrices
        if gb > max_gb:
            out.append({"cuts": n_cuts, "n_masks": len(labels), "skipped": True,
                        "reason": f"pair matrices would need ~{gb:.2f} GB "
                                  f"(cap {max_gb} GB)"})
            del masks
            continue
        t0 = time.time()
        pair = best_pair_rule(masks, labels, n_target, n_eligible, total, terms)
        del masks
        if pair is None:
            out.append({"cuts": n_cuts, "n_masks": len(labels), "skipped": True,
                        "reason": "fewer than two usable masks"})
            continue
        pair["verification"] = verify_pair_fast_path(pair, X, counts, target,
                                                     eligible, f"{tag}@{n_cuts}")
        rec = {"cuts": n_cuts, "n_masks": len(labels),
               "pairs_scored": len(labels) * (len(labels) - 1) // 2 * 2,
               "seconds": round(time.time() - t0, 1), **pair}
        if base:
            rec["same_condition_as_32_cuts"] = pair["condition"] == base["condition"]
            rec["f1_delta_vs_32_cuts"] = float(pair["f1"] - base["f1"])
        out.append(rec)
        print(f"    pair grid {n_cuts:>4} cuts ({len(labels)} masks): "
              f"F1 {pair['f1']:.4f}  {pair['condition']}  [{rec['seconds']}s]")
    return out


if __name__ == "__main__":
    main()
