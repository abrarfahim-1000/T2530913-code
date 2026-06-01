"""
Grid2Op Dual-Environment Data Generation Pipeline
==================================================
Supports both thesis environments from a single script.
Run via CLI:

    python generate_dataset.py --env neurips        # 36 subs, 59 lines  [PRIMARY]
    python generate_dataset.py --env wcci           # 118 subs, 186 lines [STRETCH]
    python generate_dataset.py --env neurips --smoke  # 3 chronics, 200 steps — quick sanity check
    python generate_dataset.py --env wcci   --smoke

Outputs (per run):
    grid_dataset_<env_tag>.jsonl      — one JSON record per line (streamable)
    grid_dataset_<env_tag>_meta.json  — env dims + label distribution (for GNN constructor)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time
import numpy as np
from collections import Counter
from pathlib import Path

import grid2op
from grid2op.Parameters import Parameters
from tqdm.auto import tqdm

import ssl
# This restores the old behavior of not verifying certificates
ssl._create_default_https_context = ssl._create_unverified_context

# ── ENV CONFIG ──────────────────────────────────────────────────────────────
ENV_NAME = "l2rpn_neurips_2020_track1_small" # ← change this to switch env
ENV_TAG  = "neurips2020"
ENV_DESC = "NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY]"

FAULT_PROB  = 0.05
RECONNECT_PROB = 0.20
SEED            = 42
RHO_CLIP        = 2.0    


# Undersampling: only log a normal step with this probability
# Fault/overload/cascade steps are ALWAYS logged
NORMAL_KEEP_PROB = 0.02 # discard 70% of normal steps → ~70% fault in final set (was 0.3, then 0.2, now 0.15 for more balance)
LINE_TRIP_KEEP_PROB = 1.0  # ← NEW: Discard 80% of redundant N-1 cooldown frames

# Smoke-test overrides (--smoke flag)
SMOKE_MAX_CHRONICS = 3
# SMOKE_MAX_STEPS    = 200
SMOKE_MAX_STEPS    = 500

LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3}

LINE_KEYS = ("rho", "p_or", "q_or", "p_ex", "q_ex", "v_or", "v_ex")
BUS_KEYS  = ("load_p", "load_q", "gen_p", "gen_q", "topo_vect")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Grid2Op dataset generator")
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Quick sanity check: {SMOKE_MAX_CHRONICS} chronics × {SMOKE_MAX_STEPS} steps"
    )
    parser.add_argument(
        "--max-chronics", type=int, default=None,
        help="Override: max number of chronics to run (default: all)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override: max steps per episode (default: full episode)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Output directory (default: 'data' directory)"
    )
    parser.add_argument(
        "--target-records",
        type=int,
        default=300000,
        help="Stop after this many records (default: 300000)"
    )
    return parser.parse_args()


def load_backend():
    try:
        from lightsim2grid import LightSimBackend
        b = LightSimBackend()
        print("[backend] LightSimBackend loaded (~10x faster)")
        return b
    except Exception:
        print("[backend] LightSimBackend unavailable — falling back to PandaPower (slow)")
        return None


def safe_tolist(arr, fill=0.0):
    """Convert array to list, replacing NaN/inf with fill value."""
    arr = np.array(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr.tolist()


def extract_features(obs):
    feats = {}
    for key in LINE_KEYS:
        arr = safe_tolist(getattr(obs, key))
        if key == "rho":
            arr = [min(v, RHO_CLIP) for v in arr]
        feats[key] = arr
    feats["line_status"] = obs.line_status.tolist()
    for key in BUS_KEYS:
        feats[key] = safe_tolist(getattr(obs, key))
    return feats

def get_state_label(obs, env):
    """Pure physical labeling based strictly on the current frame's topology and power flow."""
    max_rho = obs.rho.max() if len(obs.rho) > 0 else 0.0
    active_lines = int(np.sum(obs.line_status))
    
    # Priority 1: Overloads supersede everything
    if max_rho >= 1.0:
        line_id = int(obs.rho.argmax())
        return "overload", int(env.line_or_to_subid[line_id])
        
    # Priority 2: Full Topology
    if active_lines == env.n_line:
        return "normal", -1
        
    # Priority 3: N-1 Topology
    if active_lines == env.n_line - 1:
        line_id = int(np.where(~obs.line_status)[0][0])
        return "line_trip", int(env.line_or_to_subid[line_id])
        
    # Priority 4: N-k Topology (Cascade)
    return "cascade", -1

def validate_record(record):
    """Raise immediately if any numeric field contains NaN/inf."""
    for key in LINE_KEYS + BUS_KEYS:
        for v in record.get(key, []):
            if not np.isfinite(v):
                raise ValueError(f"Non-finite value in '{key}': {v}")


def build_meta(env, label_counts, total_records, total_time, smoke):
    """Serialisable metadata dict — consumed by GridDataset and GNN constructor."""
    present_labels = {k: v for k, v in label_counts.items() if v > 0}
    return {
        "env_name":      ENV_NAME,
        "smoke_run":     smoke,
        "n_sub":         int(env.n_sub),
        "n_line":        int(env.n_line),
        "n_load":        int(env.n_load),
        "n_gen":         int(env.n_gen),
        "topology": {
            "line_or_bus": env.line_or_to_subid.tolist(),
            "line_ex_bus": env.line_ex_to_subid.tolist(),
            "load_to_sub": env.load_to_subid.tolist(),
            "gen_to_sub":  env.gen_to_subid.tolist(),
        },
        "n_classes":     len(present_labels),           # dynamic — use this in GNN
        "label_map":     {k: LABEL_MAP[k] for k in present_labels},
        "rho_clip":      RHO_CLIP,
        "total_records": total_records,
        "label_counts":  dict(label_counts),
        "label_pct":     {k: round(100 * v / max(total_records, 1), 2)
                          for k, v in label_counts.items()},
        "throughput_steps_per_sec": round(total_records / max(total_time, 1e-9)),
        "total_time_sec": round(total_time, 1),
        # Shapes for GridDataset / GNN constructor — no magic numbers needed
        "node_feature_dim": 4,   # load_p, gen_p, mean_v_or, max_rho — built in GridDataset
        "edge_feature_dim": 3,   # rho, p_or, q_or — per line, bidirectional
    }

def print_summary(meta, out_jsonl, out_meta):
    total   = meta["total_records"]
    print(f"\n{'='*56}")
    print(f"  Env      : {meta['env_name']}")
    print(f"  Records  : {total:,}")
    print(f"  Time     : {meta['total_time_sec']:.1f}s  "
          f"({meta['throughput_steps_per_sec']} steps/sec)")
    print(f"  n_classes: {meta['n_classes']}  →  {list(meta['label_map'].keys())}")
    print(f"\n  Label distribution:")
    # Fixed order: normal → overload → line_trip → cascade
    label_order = ["normal", "overload", "line_trip", "cascade", "maintenance"]
    for lbl in label_order:
        if lbl in meta["label_counts"]:
            count = meta["label_counts"][lbl]
            pct = meta["label_pct"][lbl]
            bar = "█" * int(pct / 2)
            print(f"    {lbl:<12} {count:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"\n  Data   → {out_jsonl}")
    print(f"  Meta   → {out_meta}")
    print(f"{'='*56}\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    smoke = args.smoke

    max_chronics = SMOKE_MAX_CHRONICS if smoke else args.max_chronics
    max_steps    = SMOKE_MAX_STEPS    if smoke else args.max_steps

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag      = ENV_TAG + ("_smoke" if smoke else "")
    out_jsonl = out_dir / f"grid_dataset_{tag}.jsonl"
    out_meta  = out_dir / f"grid_dataset_{tag}_meta.json"

    print(f"\n[run]  {ENV_DESC}")
    if smoke:
        print(f"[run]  SMOKE MODE — {max_chronics} chronics × {max_steps} steps\n")

    backend = load_backend()
    make_kwargs = {"backend": backend} if backend else {}
    env = grid2op.make(ENV_NAME, **make_kwargs)

    params = Parameters()
    # params.NO_OVERFLOW_DISCONNECTION = True
    params.NO_OVERFLOW_DISCONNECTION = False 
    env.change_parameters(params)
    env.seed(SEED)
    np.random.seed(SEED)

    n_chronics = len(env.chronics_handler.subpaths)
    if max_chronics:
        n_chronics = min(max_chronics, n_chronics)

    print(f"[env]  n_sub={env.n_sub}  n_line={env.n_line}  "
          f"n_load={env.n_load}  n_gen={env.n_gen}")
    print(f"       chronics: {len(env.chronics_handler.subpaths)} available "
          f"→ {n_chronics} running\n")

    TARGET_RECORDS = args.target_records # Increase this to whatever you need
    do_nothing   = env.action_space({})
    label_counts = Counter({k: 0 for k in LABEL_MAP})
    total_written = 0
    t_start = time.time()

    with out_jsonl.open("w") as out_f:
        # We replace the chronic progress bar with a record-based progress bar
        with tqdm(total=TARGET_RECORDS, desc="Generating Dataset", unit="rec") as pbar:
            chronic_idx = 0
            
            # Loop indefinitely until we hit the exact target
            while total_written < TARGET_RECORDS:
                # Cycle through the available chronics repeatedly (epochs)
                env.set_id(chronic_idx % max(1, n_chronics))
                obs   = env.reset()
                chronic_idx += 1

                steps = min(env.max_episode_duration(),
                            max_steps if max_steps else int(1e9))

                # prev_line_status = obs.line_status.copy()
                # tripped_lines = set()

                for t in range(steps):
                    action = do_nothing
                    
                    connected = np.where(obs.line_status)[0]
                    disconnected = np.where(~obs.line_status)[0]

                    # ── Reconnect a previously tripped line (grid recovery) ──────────────
                    if len(disconnected) > 0 and np.random.rand() < RECONNECT_PROB:
                        line_id = int(np.random.choice(disconnected))
                        action  = env.action_space({"set_line_status": [(line_id, 1)]})

                    # ── Inject a new fault (only if not already reconnecting) ────────────
                    elif np.random.rand() < FAULT_PROB:
                        if len(connected) > env.n_line * 0.7:
                            line_id = int(np.random.choice(connected))
                            action  = env.action_space({"set_line_status": [(line_id, -1)]})

                    obs, reward, done, _info = env.step(action)
                    
                    # 🚨 THE PURE LABEL FIX 🚨
                    fault_label, fault_loc = get_state_label(obs, env)

                    is_normal = (fault_label == "normal")
                    is_line_trip = (fault_label == "line_trip")
                    
                    # 🚨 NEW: Hard quotas for PERFECT balancing
                    MAX_NORMAL_RECORDS = TARGET_RECORDS * 0.35    # Cap normal at 35%
                    MAX_TRIP_RECORDS = TARGET_RECORDS * 0.25      # Cap line_trip at 25%
                    MAX_CASCADE_RECORDS = TARGET_RECORDS * 0.20   # Cap cascade at 20%
                    
                    if is_normal and (np.random.rand() > NORMAL_KEEP_PROB or label_counts["normal"] >= MAX_NORMAL_RECORDS):
                        pass  
                    elif is_line_trip and (np.random.rand() > LINE_TRIP_KEEP_PROB or label_counts["line_trip"] >= MAX_TRIP_RECORDS):
                        pass  
                    elif fault_label == "cascade" and label_counts["cascade"] >= MAX_CASCADE_RECORDS:
                        pass
                    else:
                        record = {
                            # ... (Keep your existing extraction logic here) ...
                            **extract_features(obs),
                            "label":      fault_label,
                            "label_int":  LABEL_MAP[fault_label],
                            "fault_loc":  fault_loc,
                            "timestep":   t,
                            "chronic_id": chronic_idx - 1,
                            "reward":     float(reward),
                        }

                        validate_record(record)
                        out_f.write(json.dumps(record) + "\n")
                        label_counts[fault_label] += 1
                        total_written += 1
                        pbar.update(1)

                        if total_written >= TARGET_RECORDS:
                            break

                    if done:
                        break

    total_time = time.time() - t_start
    meta = build_meta(env, label_counts, total_written, total_time, smoke)

    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print_summary(meta, out_jsonl, out_meta)


if __name__ == "__main__":
    main()