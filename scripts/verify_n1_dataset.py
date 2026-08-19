"""
Sanity-check an N-1 dataset before spending an hour training on it.

Run this the moment generation finishes. It answers, in order:

  1. Is the file structurally sound? (vector lengths, sentinel alignment)
  2. Is there enough SCENARIO diversity? Effective sample size is bounded by
     distinct chronics, not by label count - frames from one chronic share a
     load profile and all n_line contingencies in a frame share a base state.
  3. Is the task still non-degenerate ON THIS DATA? The whole reason N-1 was
     adopted is that the global current state does not determine the answer.
     If the `rho_max` lift comes back high, something has gone wrong with
     generation and the task has collapsed into the closed-form trap that
     killed the classify target.
  4. What must the GNN beat? Prints the best-threshold F1 of the strongest
     single rule (flow on the line being removed), which is the number that
     decides whether the model earned anything.

Usage:
    python scripts/verify_n1_dataset.py --tag neurips2020
    python scripts/verify_n1_dataset.py --tag case14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def best_f1(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score)
    ys = y[order]
    tp, fp, P = np.cumsum(ys), np.cumsum(1 - ys), ys.sum()
    if P == 0:
        return 0.0
    prec, rec = tp / np.maximum(tp + fp, 1), tp / P
    return float(np.max(np.where(prec + rec > 0,
                                 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="neurips2020")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    path = os.path.join(args.data_dir, f"grid_dataset_{args.tag}_n1.jsonl")
    meta_path = os.path.join(args.data_dir, f"grid_dataset_{args.tag}_n1_meta.json")
    if not os.path.exists(path):
        sys.exit(f"Missing {path} — generate it with `--task n1` first.")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n_line = meta.get("n_line")

    frames = 0
    scenarios: Counter = Counter()
    episodes: set = set()
    n_eval = n_unevaluated = n_violation = 0
    mixed = allsecure = allviolation = 0
    bad_len = bad_sentinel = 0
    rho_k, y_k, rho_global = [], [], []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            frames += 1
            scenarios[r["chronic_id"]] += 1
            if "episode_id" in r:
                episodes.add(r["episode_id"])

            v = np.asarray(r["n1_violation"], dtype=int)
            st = np.asarray(r["line_status"], dtype=bool)
            rho = np.asarray(r["rho"], dtype=float)

            if n_line and (len(v) != n_line or len(st) != n_line):
                bad_len += 1
                continue
            # every de-energised line must carry the sentinel, and no other
            if ((~st) & (v != -1)).any():
                bad_sentinel += 1

            ev = v >= 0
            n_eval += int(ev.sum())
            n_unevaluated += int((~ev).sum())
            n_violation += int((v == 1).sum())

            got = v[ev]
            if got.size:
                if got.all():
                    allviolation += 1
                elif not got.any():
                    allsecure += 1
                else:
                    mixed += 1

            sel = ev & st
            rho_k.append(rho[sel])
            y_k.append(v[sel])
            rho_global.append(np.full(int(sel.sum()), rho.max()))

    rho_k = np.concatenate(rho_k) if rho_k else np.zeros(0)
    y = np.concatenate(y_k).astype(int) if y_k else np.zeros(0, dtype=int)
    rho_global = np.concatenate(rho_global) if rho_global else np.zeros(0)

    print(f"\n{'=' * 72}\n{path}\n{'=' * 72}")
    print(f"frames                : {frames:,}")
    print(f"distinct scenarios    : {len(scenarios):,}"
          + (f"   episodes: {len(episodes):,}" if episodes else ""))
    if scenarios:
        fpc = np.array(list(scenarios.values()))
        print(f"frames per scenario   : median {int(np.median(fpc))}  "
              f"min {fpc.min()}  max {fpc.max()}")
    print(f"contingency labels    : {n_eval:,} evaluated, "
          f"{n_unevaluated:,} unevaluated ({n_unevaluated / max(n_eval + n_unevaluated, 1):.2%})")
    print(f"violation rate        : {n_violation / max(n_eval, 1):.1%}")

    print(f"\nstructure")
    print(f"  wrong-length vectors            : {bad_len}   (must be 0)")
    print(f"  de-energised lines w/ real label: {bad_sentinel}   (must be 0)")

    print(f"\nper-frame outcome mix   (the property that makes this a GRAPH task)")
    tot = max(mixed + allsecure + allviolation, 1)
    print(f"  mixed (some violate, some do not): {mixed / tot:.0%}")
    print(f"  all-secure frames                : {allsecure / tot:.0%}")
    print(f"  all-violation frames             : {allviolation / tot:.0%}")

    if y.size:
        p = y.mean()
        allpos = 2 * p / (1 + p) if p else 0.0
        f1_global = best_f1(rho_global, y)
        f1_local = best_f1(rho_k, y)
        print(f"\nis the task still non-degenerate?")
        print(f"  all-positive baseline           : {allpos:.4f}")
        print(f"  best rule on GLOBAL rho_max     : {f1_global:.4f}  "
              f"({f1_global / max(allpos, 1e-9):.2f}x)")
        print(f"  best rule on rho of REMOVED line: {f1_local:.4f}  "
              f"({f1_local / max(allpos, 1e-9):.2f}x)   <- the GNN must beat THIS")

    print(f"\n{'-' * 72}")
    problems = []
    if bad_len or bad_sentinel:
        problems.append("structural errors above are non-zero")
    if len(scenarios) < 100:
        problems.append(f"only {len(scenarios)} scenarios — thin for a chronic-level split")
    if mixed / tot < 0.5:
        problems.append(f"only {mixed / tot:.0%} of frames are mixed — the per-line "
                        "distinction is weak, check generation")
    if y.size and f1_global / max(allpos, 1e-9) > 1.5:
        problems.append(f"global rho_max reaches {f1_global / max(allpos, 1e-9):.2f}x the "
                        "baseline — the current state is predicting the label, which is "
                        "the closed-form trap that killed the classify task")
    if problems:
        print("REVIEW BEFORE TRAINING:")
        for p_ in problems:
            print(f"  - {p_}")
    else:
        print("Looks good — safe to preprocess and train.")


if __name__ == "__main__":
    main()
