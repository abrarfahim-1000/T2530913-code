"""Arm A1 — DC/LODF contingency screening, scored against the GNN's own labels.

`supplimentary_docs/results_comparisons.md` §A1. This is the incumbent method the
power-systems discipline runs, and the direct test of the claim that producing an
N-1 label needs a power-flow solve: LODF is the linear approximation of that
solve, so the gap between it and the model is what the nonlinear AC physics buys.

ADMISSIBILITY (§0). This arm produces a score vector aligned to the same `y` the
GNN harness scores, and hands it to the same `best_f1_and_thr`. Alignment is by
construction rather than by hope: `y` is rebuilt here with `build_line_targets`,
the function the model's DataLoader uses. Eval batch size does not apply — there
is no forward pass and no BatchNorm — and the all-positive and rule baselines are
recomputed here as a check that the alignment held (they must reproduce the
recorded 0.4639 / 0.5392 / 0.4915).

WHAT THE SCORE IS
-----------------
For each in-service line k, remove it, redistribute its base-case flow onto the
rest of the network by the LODF factors, and report the largest resulting loading
ratio. Two channels, reported separately because they fail differently:

  flow      the linear redistribution itself.
  topology  whether the outage strands a load or generator. A DC screen ALWAYS
            solves, so §A1 q4 argued it cannot produce the game-over class of
            positive at all. That is true of the flow channel and false of the
            topology channel, which is why both are measured here instead of the
            ceiling being assumed.

Thermal limits are backed out per frame as `S_base / rho_base` (§A1 q3), which
makes the denominator by construction the same limit that produced `y` — immune
to a definitional mismatch with Grid2Op.

Usage:
    python evaluation/eval_lodf_n1.py --tag neurips2020   # run this one FIRST
    python evaluation/eval_lodf_n1.py --tag case14
    python evaluation/eval_lodf_n1.py --tag wcci2022
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.append(".")

from evaluation.eval_n1_cross_topology import best_f1_and_thr
from evaluation.lodf import LodfCache, load_branch_model
from scripts.pyg_data import GridEnvMetadata, build_line_targets

DATA_DIR = "data"
OUT_DIR = os.path.join("results", "lodf")
HOME = "neurips2020"

# The held-threshold protocol (§0 constraint 2): chosen once on the neurips2020
# val split, applied unchanged to every grid. Written by the neurips2020 run.
HELD_PATH = os.path.join(OUT_DIR, "lodf_held_threshold.json")

# A line carrying no flow has no recoverable limit (limit = S/rho is 0/0). It is
# 0.02% of in-service entries at worst; the per-line median stands in, and the
# count is reported so the substitution is auditable rather than silent.
RHO_EPS = 1e-9

# Score assigned where the topology channel fires. Any value above every
# achievable loading ratio ranks these first and trips a >= 1.0 decision rule.
STRANDED_SCORE = 1e6

FIELDS = ("line_status", "rho", "p_or", "q_or", "n1_violation", "n1_post_rho")


def load_frames(jsonl: str, indices: np.ndarray) -> list[dict]:
    """Read the scored frames, keeping only the fields this arm needs."""
    wanted = set(indices.tolist())
    out: list[dict] = []
    with open(jsonl) as fh:
        for i, line in enumerate(fh):
            if i in wanted and line.strip():
                r = json.loads(line)
                out.append({k: r[k] for k in FIELDS})
    if len(out) != len(indices):
        raise ValueError(f"expected {len(indices)} frames, read {len(out)} from {jsonl}")
    return out


def per_line_median_limits(frames: list[dict], n_line: int) -> np.ndarray:
    """Fallback thermal limit per line, for frames where the back-out degenerates."""
    table = np.full((len(frames), n_line), np.nan)
    for i, r in enumerate(frames):
        rho = np.asarray(r["rho"], dtype=float)
        s = np.hypot(np.asarray(r["p_or"], dtype=float), np.asarray(r["q_or"], dtype=float))
        ok = np.asarray(r["line_status"], dtype=bool) & (rho > RHO_EPS)
        table[i, ok] = s[ok] / rho[ok]
    with np.errstate(all="ignore"):
        med = np.nanmedian(table, axis=0)
    # A line that never carries flow anywhere in the file has no recoverable
    # limit at all. It cannot be overloaded either, so a large stand-in keeps it
    # out of the max without inventing a violation.
    return np.where(np.isfinite(med) & (med > 0), med, np.inf)


def score_frames(frames: list[dict], model, meta: GridEnvMetadata,
                 fallback: np.ndarray) -> dict[str, np.ndarray]:
    """Score every in-service line of every frame. Returns aligned 1-D vectors."""
    cache = LodfCache(model)
    flow, stranded, ys, post_rho, base_rho, noredist = [], [], [], [], [], []
    degenerate = 0

    for r in frames:
        status = np.asarray(r["line_status"], dtype=bool)
        line_y, line_mask, line_id = build_line_targets(r, meta)
        # Measured across all three datasets: no in-service line is ever
        # unevaluated, so the mask is a no-op. Honour it anyway — it is the
        # harness's contract, and a future regeneration could reintroduce -1.
        if not line_mask.all():
            keep = line_mask
        else:
            keep = np.ones(len(line_id), dtype=bool)

        p = np.asarray(r["p_or"], dtype=float)[status]
        q = np.asarray(r["q_or"], dtype=float)[status]
        rho = np.asarray(r["rho"], dtype=float)[status]
        s_base = np.hypot(p, q)

        limit = np.where(rho > RHO_EPS, s_base / np.maximum(rho, RHO_EPS), fallback[line_id])
        degenerate += int(np.sum(rho <= RHO_EPS))

        lodf, _bridge, strands = cache.get(status)

        # post[m, k] = flow on m after removing k. Reactive power is held at its
        # base value: DC redistributes active power only.
        post_p = p[:, None] + lodf * p[None, :]
        rho_hat = np.hypot(post_p, q[:, None]) / limit[:, None]
        np.fill_diagonal(rho_hat, -np.inf)  # a removed line cannot overload itself
        score = rho_hat.max(axis=0) if rho_hat.shape[0] > 1 else np.zeros(len(p))

        # CONTROL: the same statistic with the redistribution switched off —
        # "is any OTHER line already loaded?". Everything `flow` scores above
        # this is what the LODF factors themselves contribute, and without it a
        # reader cannot tell the physics from a frame-level max.
        held_out = np.broadcast_to(rho[:, None], rho_hat.shape).copy()
        np.fill_diagonal(held_out, -np.inf)
        control = held_out.max(axis=0) if held_out.shape[0] > 1 else np.zeros(len(p))

        flow.append(score[keep])
        noredist.append(control[keep])
        stranded.append(strands[keep])
        ys.append(line_y[keep])
        post_rho.append(np.asarray(r["n1_post_rho"], dtype=float)[line_id][keep])
        base_rho.append(rho[keep])

    return {
        "flow": np.concatenate(flow),
        "no_redistribution": np.concatenate(noredist),
        "stranded": np.concatenate(stranded),
        "y": np.concatenate(ys).astype(int),
        "post_rho": np.concatenate(post_rho),
        "base_rho": np.concatenate(base_rho),
        "degenerate_limits": degenerate,
        "patterns": cache.misses,
    }


def f1_at(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    pred = score >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "threshold": float(thr),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "precision": float(prec),
        "recall": float(rec),
        "tp": tp, "fp": fp, "fn": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="neurips2020")
    ap.add_argument("--json", default=None, help="output path (default results/lodf/lodf_<tag>.json)")
    args = ap.parse_args()

    jsonl = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_n1.jsonl")
    meta_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_n1_meta.json")
    for p in (jsonl, meta_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}")

    with open(meta_path) as fh:
        meta = GridEnvMetadata(json.load(fh))
    model = load_branch_model(args.tag, meta_path)

    n_records = sum(1 for line in open(jsonl) if line.strip())
    if args.tag == HOME:
        test_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_test_idx.npy"))
        val_idx = np.load(os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy"))
        scope = f"held-out test split ({len(test_idx):,} of {n_records:,} frames)"
    else:
        test_idx, val_idx = np.arange(n_records), None
        scope = f"all {n_records:,} frames (topology never seen in training)"

    t0 = time.time()
    print(f"Reading {args.tag} frames...")
    frames = load_frames(jsonl, test_idx)
    fallback = per_line_median_limits(frames, model.n_line)
    print(f"Scoring {len(frames):,} frames...")
    res = score_frames(frames, model, meta, fallback)
    elapsed = time.time() - t0

    y, flow, stranded = res["y"], res["flow"], res["stranded"]
    topo = np.where(stranded, STRANDED_SCORE, flow)

    # ── the held threshold: selected once, on the home grid's VAL split ───────
    if args.tag == HOME:
        val_frames = load_frames(jsonl, val_idx)
        val = score_frames(val_frames, model, meta, fallback)
        held = {
            "flow": best_f1_and_thr(val["flow"], val["y"])[1],
            "topo": best_f1_and_thr(
                np.where(val["stranded"], STRANDED_SCORE, val["flow"]), val["y"])[1],
            "noredist": best_f1_and_thr(val["no_redistribution"], val["y"])[1],
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(HELD_PATH, "w") as fh:
            json.dump(held, fh, indent=2)
        print(f"  held thresholds selected on the val split -> {HELD_PATH}")
    elif os.path.exists(HELD_PATH):
        with open(HELD_PATH) as fh:
            held = json.load(fh)
    else:
        sys.exit(f"Missing {HELD_PATH}. Run --tag {HOME} first: the held threshold "
                 f"must be selected on the home grid's val split (§0 constraint 2).")

    # ── the two label views (§A1 q4) ─────────────────────────────────────────
    # A game-over contingency carries post_rho == 0.0 exactly and y == 1; the
    # decomposition is one-sided and exact, verified on all three grids.
    solved = res["post_rho"] > 0.0
    gameover = (y == 1) & ~solved

    p_rate = float(y.mean())
    report: dict = {
        "tag": args.tag,
        "scope": scope,
        "contingencies": int(len(y)),
        "violation_rate": p_rate,
        "distinct_topology_patterns": res["patterns"],
        "degenerate_limit_substitutions": res["degenerate_limits"],
        "seconds": round(elapsed, 1),
        "labels": {
            "positives": int(y.sum()),
            "game_over_positives": int(gameover.sum()),
            "game_over_share_of_positives": float(gameover.sum() / max(y.sum(), 1)),
        },
        "baselines": {
            "all_positive": float(2 * p_rate / (1 + p_rate)) if p_rate else 0.0,
            "best_rule_rho_of_removed_line": best_f1_and_thr(res["base_rho"], y)[0],
        },
        "arms": {},
    }

    arms = (
        ("no_redistribution_control", res["no_redistribution"], "noredist"),
        ("lodf_flow", flow, "flow"),
        ("lodf_flow_plus_topology", topo, "topo"),
    )
    for name, score, key in arms:
        oracle_f1, oracle_thr = best_f1_and_thr(score, y)
        arm = {
            "doctrinal_rho_ge_1": f1_at(score, y, 1.0),
            "held": f1_at(score, y, held[key]),
            "oracle": {"f1": oracle_f1, "threshold": oracle_thr},
        }
        # The fair test of a linear flow method: drop the contingencies whose
        # label exists only because the episode ended.
        if solved.any():
            arm["overload_only"] = {
                "contingencies": int(solved.sum()),
                "violation_rate": float(y[solved].mean()),
                "doctrinal_rho_ge_1": f1_at(score[solved], y[solved], 1.0),
                "oracle": {"f1": best_f1_and_thr(score[solved], y[solved])[0]},
            }
        report["arms"][name] = arm

    # ── the topology channel on its own ──────────────────────────────────────
    report["topology_channel"] = {
        "fired": int(stranded.sum()),
        "precision_vs_any_violation": float(y[stranded].mean()) if stranded.any() else None,
        "share_of_game_over_caught": float(
            np.sum(stranded & gameover) / max(gameover.sum(), 1)),
    }

    # ── DC vs AC fidelity: does the linear estimate track the real solve? ─────
    if solved.sum() > 2:
        a, b = flow[solved], res["post_rho"][solved]
        report["dc_vs_ac"] = {
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "median_abs_error": float(np.median(np.abs(a - b))),
            "median_signed_error": float(np.median(a - b)),
        }

    out = args.json or os.path.join(OUT_DIR, f"lodf_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    _print_report(report, meta)
    print(f"\n  written: {out}")


def _print_report(rep: dict, meta: GridEnvMetadata) -> None:
    print(f"\n{'=' * 74}")
    print(f"  A1 — DC/LODF screening — {rep['tag']}   "
          f"({meta.n_line} lines, {meta.n_sub} buses)")
    print(f"  {rep['scope']}")
    print(f"{'=' * 74}")
    print(f"  contingencies scored : {rep['contingencies']:,}")
    print(f"  violation rate       : {rep['violation_rate']:.1%}")
    print(f"  game-over positives  : {rep['labels']['game_over_positives']:,} "
          f"({rep['labels']['game_over_share_of_positives']:.1%} of positives)")
    print(f"  topology patterns    : {rep['distinct_topology_patterns']}")
    b = rep["baselines"]
    print(f"\n  all-positive baseline            : {b['all_positive']:.4f}")
    print(f"  best rule on rho of removed line : {b['best_rule_rho_of_removed_line']:.4f}"
          f"   <- must match the recorded value; it is the alignment check")
    for name, arm in rep["arms"].items():
        print(f"\n  {name}")
        print(f"    rho_hat >= 1.0 (no tuning) : F1 {arm['doctrinal_rho_ge_1']['f1']:.4f}"
              f"   prec {arm['doctrinal_rho_ge_1']['precision']:.3f}"
              f"  rec {arm['doctrinal_rho_ge_1']['recall']:.3f}")
        print(f"    held threshold             : F1 {arm['held']['f1']:.4f}"
              f"   (thr {arm['held']['threshold']:.4f})")
        print(f"    oracle (best on this grid) : F1 {arm['oracle']['f1']:.4f}")
        if "overload_only" in arm:
            oo = arm["overload_only"]
            print(f"    overload-only subset       : F1 {oo['doctrinal_rho_ge_1']['f1']:.4f}"
                  f" @1.0, {oo['oracle']['f1']:.4f} oracle"
                  f"   ({oo['contingencies']:,} contingencies)")
    tc = rep["topology_channel"]
    print(f"\n  topology channel: fired {tc['fired']:,}, "
          f"precision {tc['precision_vs_any_violation']}, "
          f"caught {tc['share_of_game_over_caught']:.1%} of game-overs")
    if "dc_vs_ac" in rep:
        d = rep["dc_vs_ac"]
        print(f"  DC vs AC on solved contingencies: r = {d['pearson_r']:.4f}, "
              f"median |err| {d['median_abs_error']:.4f}, "
              f"median signed {d['median_signed_error']:+.4f}")


if __name__ == "__main__":
    main()
