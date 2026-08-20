"""
dump_base_kv.py — Per-line base (nominal) kV sidecar for voltage_pu conversion.
================================================================================
The symbolic shield converts Grid2Op line voltages (kV) to per-unit via
    voltage_pu[i] = v_or[i] / base_kv_or[i]        (energized lines only)
A flat nominal (e.g. 150 kV) is WRONG off the 36-bus grid: l2rpn_case14_sandbox
has lines at ~20 kV and ~138 kV, so a flat divisor produces ~0.13 pu on healthy
frames and a 100% false block rate. This script writes the per-line base kV for
an environment tag to data/grid_dataset_<tag>_basekv.json.

Two methods:
  backend    (default) — instantiate the grid2op env and read the backend's
              per-line pu→kV conversion arrays. Requires the env to be
              downloadable/installed on this machine.
  --empirical — no grid2op needed: per-line median of v_or / v_ex over
              ground-truth "normal" frames of the existing dataset JSONL.
              Median of the healthy voltage distribution ≈ nominal.

Usage:
    python scripts/dump_base_kv.py --tag neurips2020
    python scripts/dump_base_kv.py --tag case14 --empirical
"""

import argparse
import json
import os
import sys

import numpy as np

DATA_DIR = "data"

# tag → grid2op env name (mirrors scripts/generate_dataset.py)
ENV_NAMES = {
    "neurips2020": "l2rpn_neurips_2020_track1_small",
    "case14":      "l2rpn_case14_sandbox",
    "wcci2022":    "l2rpn_wcci_2022",
}


def base_kv_from_backend(env_name: str) -> tuple[list[float], list[float]]:
    """Read per-line origin/extremity base kV from the grid2op backend."""
    import grid2op
    from lightsim2grid import LightSimBackend

    env = grid2op.make(env_name, backend=LightSimBackend())
    try:
        backend = env.backend
        kv_or = np.asarray(backend.lines_or_pu_to_kv, dtype=float)
        kv_ex = np.asarray(backend.lines_ex_pu_to_kv, dtype=float)
    finally:
        env.close()
    return kv_or.tolist(), kv_ex.tolist()


def base_kv_empirical(jsonl_path: str, n_line: int) -> tuple[list[float], list[float]]:
    """Per-line median v_or / v_ex over ground-truth 'normal' frames (energized lines only)."""
    v_or_samples = [[] for _ in range(n_line)]
    v_ex_samples = [[] for _ in range(n_line)]
    n_normal = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("label") != "normal":
                continue
            n_normal += 1
            status = rec["line_status"]
            for i in range(n_line):
                if status[i]:
                    v_or_samples[i].append(rec["v_or"][i])
                    v_ex_samples[i].append(rec["v_ex"][i])

    if n_normal == 0:
        sys.exit(f"No ground-truth 'normal' frames in {jsonl_path} — cannot estimate base kV.")

    kv_or, kv_ex = [], []
    for i in range(n_line):
        if not v_or_samples[i]:
            sys.exit(f"Line {i} has no energized 'normal' samples — cannot estimate its base kV.")
        kv_or.append(float(np.median(v_or_samples[i])))
        kv_ex.append(float(np.median(v_ex_samples[i])))

    print(f"[empirical] {n_normal:,} normal frames used")
    return kv_or, kv_ex


def main():
    ap = argparse.ArgumentParser(description="Dump per-line base kV sidecar for an env tag")
    ap.add_argument("--tag", required=True, choices=sorted(ENV_NAMES),
                    help="dataset tag → data/grid_dataset_<tag>_basekv.json")
    ap.add_argument("--empirical", action="store_true",
                    help="estimate from the dataset JSONL instead of the grid2op backend")
    args = ap.parse_args()

    env_name  = ENV_NAMES[args.tag]
    # The two methods write to DIFFERENT files. They disagree by design — the
    # backend reports nominal kV, the empirical scan reports what the grid is
    # actually running at (~6% above nominal) — and the shield reads the backend
    # sidecar. Sharing one path meant a single --empirical run silently replaced
    # the authoritative file with the counterfactual arm.
    _suffix   = "_basekv_empirical" if args.empirical else "_basekv"
    out_path  = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}{_suffix}.json")

    # Only `n_line` is read from the meta, and every task's meta carries it.
    # The suffix is resolved ONCE and reused for the --empirical JSONL, so meta
    # and samples always describe the same run.
    #
    # N-1 comes FIRST. The classify sets were deleted on 2026-08-20, but their
    # `_meta.json` sidecars outlived them; with the old ("", "_n1") order this
    # loop resolved to a suffix whose .jsonl no longer exists, so --empirical
    # died on a missing file while the backend path silently read n_line from a
    # stale meta.
    for suffix in ("_n1", "_forecast"):
        meta_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}{suffix}_meta.json")
        if os.path.exists(meta_path):
            break
    else:
        sys.exit(f"No meta for tag '{args.tag}' in {DATA_DIR} — generate the dataset first.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n_line = meta["n_line"]

    if args.empirical:
        jsonl_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}{suffix}.jsonl")
        if not os.path.exists(jsonl_path):
            sys.exit(f"Missing {jsonl_path} — needed for --empirical.")
        kv_or, kv_ex = base_kv_empirical(jsonl_path, n_line)
        method = "empirical"
    else:
        try:
            kv_or, kv_ex = base_kv_from_backend(env_name)
            method = "backend"
        except Exception as exc:
            sys.exit(f"Backend method failed ({exc}).\n"
                     f"If the env is not available on this machine, rerun with --empirical.")

    if len(kv_or) != n_line:
        sys.exit(f"Line count mismatch: backend gave {len(kv_or)}, meta says {n_line}.")

    sidecar = {
        "env_name":   env_name,
        "method":     method,
        "n_line":     n_line,
        "base_kv_or": kv_or,
        "base_kv_ex": kv_ex,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    levels = sorted(set(round(v) for v in kv_or))
    print(f"[{args.tag}] method={method}  n_line={n_line}")
    print(f"  base_kv_or levels (rounded): {levels}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
