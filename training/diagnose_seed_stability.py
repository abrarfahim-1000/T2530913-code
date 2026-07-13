"""Seed-stability diagnostic (Round 3 follow-up).

Round 3 found the baseline is seed-unstable: per-seed macro 0.8386 / 0.7162 / 0.6538. But `--seed`
varied BOTH weight init AND the train/val repartition at once, so the collapse source was not
localized. This script separates them with a 2-arm design on the baseline (lambda_f1=0, small config):

  * Arm A — vary INIT, hold the partition fixed (split_seed=42):  init in {42,43,44}
  * Arm B — vary the PARTITION, hold init fixed (init_seed=42):   split in {42,43,44}

The (init=42, split=42) run is the shared anchor (== the Round 3 "good" seed), so only 5 unique
trainings run. Comparing the two arms' std:

  * sigma(Arm A) >> sigma(Arm B)  => initialization / training instability dominates
  * sigma(Arm B) >> sigma(Arm A)  => train/val PARTITION luck dominates (the in-memory frame-level
                                     reshuffle is producing bad val/checkpoint-selection sets)
  * both large                    => both contribute

Usage:  python training/diagnose_seed_stability.py --seeds 42 43 44
"""
import argparse
import json
import os
import sys

# Make the repo root importable when run as `python training/diagnose_seed_stability.py`
# (which otherwise only puts training/ on sys.path).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.run_round3_multiseed import REPO_ROOT, train_and_calibrate, _mean_std  # noqa: E402


def _summarize(label, results, classes):
    macro = [r["test_macro_f1"] for r in results]
    nrec  = [r["test_normal_recall"] for r in results]
    m_mean, m_std = _mean_std(macro)
    n_mean, n_std = _mean_std(nrec)
    print(f"\n--- {label} (n={len(results)}) ---")
    print(f"  per-run macro : {[round(x, 4) for x in macro]}")
    print(f"  macro         : {m_mean:.4f} +/- {m_std:.4f}")
    print(f"  normal_recall : {n_mean:.4f} +/- {n_std:.4f}")
    return {"macro_mean": m_mean, "macro_std": m_std,
            "normal_recall_mean": n_mean, "normal_recall_std": n_std,
            "per_run_macro": macro}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="seed values swept in each arm; first is the fixed anchor")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--outdir", default="round3_runs")
    args = ap.parse_args()

    anchor = args.seeds[0]
    outdir = os.path.join(REPO_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Cache runs by (init, split) so the shared anchor trains only once.
    cache = {}

    def run(init_seed, split_seed):
        key = (init_seed, split_seed)
        if key not in cache:
            label = f"i{init_seed}_s{split_seed}"
            cache[key] = train_and_calibrate(
                "seeddiag", label, 0.0, outdir, args.epochs, None, None,
                init_seed=init_seed, split_seed=split_seed)
        return cache[key]

    print(f"=== Seed-stability diagnostic: anchor={anchor}, sweep={args.seeds} ===")
    print(">>> Arm A: vary INIT, fixed split_seed=%d" % anchor)
    arm_a = [run(s, anchor) for s in args.seeds]
    print(">>> Arm B: vary PARTITION, fixed init_seed=%d" % anchor)
    arm_b = [run(anchor, s) for s in args.seeds]

    classes = list(arm_a[0]["test_per_class_f1"].keys())
    a = _summarize("Arm A (vary init, fixed partition)", arm_a, classes)
    b = _summarize("Arm B (vary partition, fixed init)", arm_b, classes)

    # Interpretation.
    print("\n=== Variance decomposition ===")
    print(f"  sigma(macro) from INIT      (Arm A): {a['macro_std']:.4f}")
    print(f"  sigma(macro) from PARTITION (Arm B): {b['macro_std']:.4f}")
    if max(a["macro_std"], b["macro_std"]) < 0.02:
        verdict = ("Neither arm shows large variance at these seeds — the earlier collapse may need "
                   "more seeds to characterize, or was specific to the (init==split) coupling.")
    elif a["macro_std"] >= 2 * b["macro_std"]:
        verdict = ("INITIALIZATION / training instability dominates. Fixes to try: longer warm-up, "
                   "lower lr, gentler BN, or seed-averaged init — the partition is not the problem.")
    elif b["macro_std"] >= 2 * a["macro_std"]:
        verdict = ("Train/val PARTITION luck dominates. The in-memory frame-level reshuffle is making "
                   "bad val/checkpoint-selection sets. Revisit the split (chronic-level, or fix the "
                   "val set) rather than the optimizer.")
    else:
        verdict = ("BOTH init and partition contribute comparably. Address the larger first, then "
                   "re-measure; expect to need both fixes for a stable baseline.")
    print(f"  -> {verdict}")

    out = {
        "anchor": anchor, "seeds": args.seeds,
        "arm_a_vary_init": a, "arm_b_vary_partition": b,
        "verdict": verdict,
    }
    out_path = os.path.join(outdir, "agg_seeddiag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> saved {out_path}")


if __name__ == "__main__":
    main()
