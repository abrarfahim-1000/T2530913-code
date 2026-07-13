"""Round 3 multi-seed experiment driver (I1/I4/I5).

Trains one "lever" (a lambda_f1 setting) across several seeds, calibrates each run
with training/calibrate_margin.py, aggregates test metrics as mean +/- std, and — when
given a baseline aggregate — applies the NOISE-AWARE strict bar:

  * macro-F1: lever mean must beat baseline mean by MORE than baseline std
  * normal recall: lever mean >= 0.88 floor AND >= baseline mean - baseline std
  * every class F1 (incl. normal): lever mean must not drop below baseline mean - baseline std

Usage
-----
  # 1) establish the baseline noise floor (pure CE, lambda_f1=0)
  python training/run_round3_multiseed.py --tag baseline --lambda_f1 0.0 --seeds 42 43 44

  # 2) the soft-F1 lever, judged against the baseline aggregate
  python training/run_round3_multiseed.py --tag softf1_0p1 --lambda_f1 0.1 --seeds 42 43 44 \
         --compare_to round3_runs/agg_baseline.json

Runs on the SMALL config (GRID_CONFIG=small is forced for every child process) so results
are comparable to the deployed Lever A baseline regardless of the machine's device.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NORMAL_RECALL_FLOOR = 0.88  # absolute floor from the deployed baseline (0.8806)


def _run(cmd, log_path, env):
    """Run a child process, tee stdout+stderr to log_path, return (ok, tail)."""
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    ok = proc.returncode == 0
    tail = ""
    if not ok:
        with open(log_path, encoding="utf-8") as log:
            tail = "".join(log.readlines()[-25:])
    return ok, tail


def train_and_calibrate(tag, label, lambda_f1, outdir, epochs, warmup_frac, ramp_epochs,
                        init_seed, split_seed, gsat=False, beta=None, r=None, gsat_tau=None,
                        beta_warmup_frac=None):
    env = dict(os.environ, GRID_CONFIG="small")
    py = sys.executable
    ckpt   = os.path.join(outdir, f"ckpt_{tag}_{label}.pt")
    margin = os.path.join(outdir, f"margin_{tag}_{label}.json")

    train_cmd = [py, "training/train_gnn.py",
                 "--init_seed", str(init_seed), "--split_seed", str(split_seed),
                 "--lambda_f1", str(lambda_f1), "--ckpt_out", ckpt]
    if epochs is not None:
        train_cmd += ["--epochs", str(epochs)]
    if warmup_frac is not None:
        train_cmd += ["--f1_warmup_frac", str(warmup_frac)]
    if ramp_epochs is not None:
        train_cmd += ["--lambda_ramp_epochs", str(ramp_epochs)]
    # Round 3 GSAT: forward the edge-gate flags so the multi-seed harness can drive it.
    if gsat:
        train_cmd += ["--gsat"]
        if beta is not None:             train_cmd += ["--beta", str(beta)]
        if r is not None:                train_cmd += ["--r", str(r)]
        if gsat_tau is not None:         train_cmd += ["--gsat_tau", str(gsat_tau)]
        if beta_warmup_frac is not None: train_cmd += ["--beta_warmup_frac", str(beta_warmup_frac)]

    print(f"  [{tag} {label}] training (init={init_seed} split={split_seed})...", flush=True)
    ok, tail = _run(train_cmd, os.path.join(outdir, f"train_{tag}_{label}.log"), env)
    if not ok:
        print(f"  [{tag} {label}] TRAIN FAILED:\n{tail}")
        sys.exit(1)

    print(f"  [{tag} {label}] calibrating...", flush=True)
    # GSAT checkpoints carry edge_gate.* params — calibrate must build a gate-matched model.
    cal_cmd = [py, "training/calibrate_margin.py", "--checkpoint", ckpt, "--out", margin]
    if gsat:
        cal_cmd += ["--gsat"]
    ok, tail = _run(cal_cmd, os.path.join(outdir, f"cal_{tag}_{label}.log"), env)
    if not ok:
        print(f"  [{tag} {label}] CALIBRATE FAILED:\n{tail}")
        sys.exit(1)

    with open(margin, encoding="utf-8") as f:
        m = json.load(f)
    print(f"  [{tag} {label}] macro={m['test_macro_f1']:.4f}  normal_recall={m['test_normal_recall']:.4f}")
    return m


def _mean_std(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def aggregate(tag, results, classes):
    macro = [r["test_macro_f1"] for r in results]
    nrec  = [r["test_normal_recall"] for r in results]
    agg = {
        "tag": tag,
        "seeds": [r.get("_seed") for r in results],
        "macro_f1":      dict(zip(("mean", "std"), _mean_std(macro))),
        "normal_recall": dict(zip(("mean", "std"), _mean_std(nrec))),
        "per_class_f1":  {},
    }
    for c in classes:
        vals = [r["test_per_class_f1"][c] for r in results]
        agg["per_class_f1"][c] = dict(zip(("mean", "std"), _mean_std(vals)))
    return agg


def noise_aware_verdict(lever, base):
    """Apply the noise-aware strict bar. Returns (keep: bool, lines: list[str])."""
    lines, keep = [], True
    b_macro, b_macro_s = base["macro_f1"]["mean"], base["macro_f1"]["std"]
    l_macro = lever["macro_f1"]["mean"]
    macro_win = l_macro > b_macro + b_macro_s
    keep &= macro_win
    lines.append(f"  macro: lever {l_macro:.4f} vs baseline {b_macro:.4f}+/-{b_macro_s:.4f}"
                 f"  -> {'WIN' if macro_win else 'no win (within noise)'}")

    l_nrec = lever["normal_recall"]["mean"]
    b_nrec, b_nrec_s = base["normal_recall"]["mean"], base["normal_recall"]["std"]
    nrec_ok = (l_nrec >= NORMAL_RECALL_FLOOR) and (l_nrec >= b_nrec - b_nrec_s)
    keep &= nrec_ok
    lines.append(f"  normal_recall: lever {l_nrec:.4f} (floor {NORMAL_RECALL_FLOOR}, "
                 f"baseline {b_nrec:.4f}+/-{b_nrec_s:.4f}) -> {'OK' if nrec_ok else 'FAIL'}")

    for c, cs in lever["per_class_f1"].items():
        lm = cs["mean"]
        bm, bs = base["per_class_f1"][c]["mean"], base["per_class_f1"][c]["std"]
        holds = lm >= bm - bs
        keep &= holds
        lines.append(f"  {c} F1: lever {lm:.4f} vs baseline {bm:.4f}+/-{bs:.4f}"
                     f"  -> {'holds' if holds else 'REGRESSED'}")
    return keep, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="short label for this lever (used in filenames)")
    ap.add_argument("--lambda_f1", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--f1_warmup_frac", type=float, default=None)
    ap.add_argument("--lambda_ramp_epochs", type=int, default=None)
    ap.add_argument("--outdir", default="round3_runs")
    ap.add_argument("--compare_to", default=None, help="baseline agg_*.json for the noise-aware bar")
    # Round 3 GSAT lever (handoff §4/§5). --gsat turns the edge gate on for every seed.
    ap.add_argument("--gsat", action="store_true", default=False, help="enable the GSAT edge gate")
    ap.add_argument("--beta", type=float, default=None, help="IB-KL weight")
    ap.add_argument("--r", type=float, default=None, help="Bernoulli prior (edge-keep rate)")
    ap.add_argument("--gsat_tau", type=float, default=None, help="Gumbel temperature")
    ap.add_argument("--beta_warmup_frac", type=float, default=None, help="pure-task warm-up fraction for beta")
    # Init control (handoff §2/§5.2 minimum): fix weight-init seed, vary only the train/val
    # split across --seeds (Arm-B isolation) so the lever is judged free of init luck.
    ap.add_argument("--fixed_init", type=int, default=None,
                    help="if set, hold init_seed at this value and vary only split_seed across --seeds")
    args = ap.parse_args()

    outdir = os.path.join(REPO_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    init_desc = f"fixed_init={args.fixed_init}" if args.fixed_init is not None else "init=split=seed"
    gsat_desc = (f" gsat beta={args.beta} r={args.r} tau={args.gsat_tau}" if args.gsat else "")
    print(f"=== Round 3 multi-seed: tag={args.tag} lambda_f1={args.lambda_f1} seeds={args.seeds} "
          f"[{init_desc}]{gsat_desc} ===")
    results = []
    for seed in args.seeds:
        # Arm-B isolation when --fixed_init is set: init held constant, split = seed.
        init_seed = args.fixed_init if args.fixed_init is not None else seed
        m = train_and_calibrate(args.tag, f"s{seed}", args.lambda_f1, outdir,
                                args.epochs, args.f1_warmup_frac, args.lambda_ramp_epochs,
                                init_seed=init_seed, split_seed=seed,
                                gsat=args.gsat, beta=args.beta, r=args.r, gsat_tau=args.gsat_tau,
                                beta_warmup_frac=args.beta_warmup_frac)
        m["_seed"] = seed
        results.append(m)

    classes = list(results[0]["test_per_class_f1"].keys())
    agg = aggregate(args.tag, results, classes)
    agg_path = os.path.join(outdir, f"agg_{args.tag}.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"\n=== Aggregate ({args.tag}, n={len(results)} seeds) ===")
    print(f"  macro_f1      : {agg['macro_f1']['mean']:.4f} +/- {agg['macro_f1']['std']:.4f}")
    print(f"  normal_recall : {agg['normal_recall']['mean']:.4f} +/- {agg['normal_recall']['std']:.4f}")
    for c in classes:
        cs = agg["per_class_f1"][c]
        print(f"  {c:10s} F1 : {cs['mean']:.4f} +/- {cs['std']:.4f}")
    print(f"  -> saved {agg_path}")

    if args.compare_to:
        with open(os.path.join(REPO_ROOT, args.compare_to), encoding="utf-8") as f:
            base = json.load(f)
        keep, lines = noise_aware_verdict(agg, base)
        print(f"\n=== Noise-aware strict bar vs {base['tag']} ===")
        print("\n".join(lines))
        print(f"\n  VERDICT: {'KEEP (beats baseline beyond noise)' if keep else 'REJECT (revert to baseline)'}")


if __name__ == "__main__":
    main()
