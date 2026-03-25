"""
Grid2Op Data Generation Pipeline
=================================
Toy script: runs on rte_case5_example (bundled, no download needed).
Switch ENV_NAME to 'l2rpn_wcci_2022' for thesis-scale generation.

Output: grid_dataset.json — one record per timestep, ready for PyG conversion.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
from collections import Counter

import numpy as np
import grid2op
from grid2op.Parameters import Parameters
from tqdm.auto import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Toggle this one line to switch from toy → thesis scale
ENV_NAME   = "rte_case5_example"         # toy (bundled, instant)
# ENV_NAME = "l2rpn_wcci_2022"           # thesis primary (bundled dev cut, 2 chronics)
# ENV_NAME = "/path/to/l2rpn_wcci_2022"  # thesis primary (full, requires download and unpacking)

MAX_CHRONICS      = None    # None = all available chronics
MAX_STEPS_PER_EP  = None    # None = full episode; set e.g. 200 for quick smoke test
FAULT_PROB        = 0.08    # probability of injecting a line trip each step
SEED              = 42
OUTPUT_PATH       = "grid_dataset.json"
# ─────────────────────────────────────────────────────────────────────────────

# Try LightSimBackend for ~10x speed; fall back to default if not available
try:
    from lightsim2grid import LightSimBackend
    backend = LightSimBackend()
    print("[backend] LightSimBackend loaded")
except Exception:
    backend = None
    print("[backend] LightSimBackend unavailable — using default PandaPower backend")

# ── ENV SETUP ─────────────────────────────────────────────────────────────────
pkg_dir  = os.path.dirname(grid2op.__file__)
env_path = os.path.join(pkg_dir, "data", ENV_NAME) if not os.path.isabs(ENV_NAME) else ENV_NAME

env = grid2op.make(env_path, test='true', backend=backend) if backend else grid2op.make(env_path)

# Keep overloaded lines alive so we can observe and label them
params = Parameters()
params.NO_OVERFLOW_DISCONNECTION = True
env.change_parameters(params)
env.seed(SEED)
np.random.seed(SEED)

n_chronics = len(env.chronics_handler.subpaths)
if MAX_CHRONICS:
    n_chronics = min(MAX_CHRONICS, n_chronics)

print(f"[env]  {ENV_NAME}")
print(f"       n_sub={env.n_sub}  n_line={env.n_line}  "
      f"n_load={env.n_load}  n_gen={env.n_gen}")
print(f"       chronics available: {len(env.chronics_handler.subpaths)}"
      f"  →  running: {n_chronics}")
print()

# ── LABEL CONSTANTS ───────────────────────────────────────────────────────────
LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3, "maintenance": 4}

LINE_FEATURE_KEYS = ("rho", "p_or", "q_or", "p_ex", "q_ex", "v_or", "v_ex", "line_status")
BUS_FEATURE_KEYS = ("load_p", "load_q", "gen_p", "gen_q", "topo_vect")


def extract_features(obs):
    features = {}
    for key in LINE_FEATURE_KEYS + BUS_FEATURE_KEYS:
        val = getattr(obs, key)
        features[key] = val.tolist() if hasattr(val, "tolist") else val
    return features


def derive_label(obs, prev_line_status, current_label, current_loc):
    label, loc = current_label, current_loc

    # Overload has priority because it signals immediate thermal violation.
    if obs.rho.max() > 1.0:
        label, loc = "overload", int(obs.rho.argmax())

    # Cascade: a new trip without explicit injected line-trip intent.
    new_trips = (~obs.line_status) & prev_line_status
    if new_trips.any() and label == "normal":
        label, loc = "cascade", int(np.where(new_trips)[0][0])

    return label, loc

do_nothing = env.action_space({})
records    = []
t_start    = time.time()

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
for chronic_id in tqdm(range(n_chronics), desc="Chronics", unit="chronic"):
    obs     = env.reset()
    ep_len  = env.max_episode_duration()
    steps   = min(ep_len, MAX_STEPS_PER_EP) if MAX_STEPS_PER_EP else ep_len

    prev_line_status = obs.line_status.copy()   # track trips across steps

    for t in tqdm(range(steps), desc=f"Steps (chronic {chronic_id + 1})", unit="step", leave=False):
        action      = do_nothing
        fault_label = "normal"
        fault_loc   = None

        # ── Fault injection ───────────────────────────────────────────────────
        if np.random.rand() < FAULT_PROB:
            # Pick a line that is currently connected
            connected = np.where(obs.line_status)[0]
            if len(connected) > 0:
                line_id = int(np.random.choice(connected))
                action  = env.action_space({"set_line_status": [(line_id, -1)]})
                fault_label = "line_trip"
                fault_loc   = line_id

        obs, reward, done, _info = env.step(action)

        fault_label, fault_loc = derive_label(obs, prev_line_status, fault_label, fault_loc)

        prev_line_status = obs.line_status.copy()

        # ── Record ────────────────────────────────────────────────────────────
        records.append({
            **extract_features(obs),
            "label":        fault_label,
            "label_int":    LABEL_MAP[fault_label],
            "fault_loc":    fault_loc,
            "timestep":     t,
            "chronic_id":   chronic_id,
            "reward":       float(reward),
        })

        if done:
            break

    # elapsed = time.time() - t_start
    # tqdm.write(
    #     f"  chronic {chronic_id + 1}/{n_chronics}  |  "
    #     f"steps recorded: {len(records)}  |  elapsed: {elapsed:.1f}s"
    # )

# ── SAVE ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(records, f)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
labels     = [r["label"] for r in records]
dist       = Counter(labels)
total_time = time.time() - t_start

print()
print("=" * 50)
print(f"Dataset saved → {OUTPUT_PATH}")
print(f"Total records : {len(records)}")
print(f"Total time    : {total_time:.1f}s")
print(f"Throughput    : {len(records)/max(total_time, 1e-9):.0f} steps/sec")
print()
print("Label distribution:")
for lbl, count in sorted(dist.items(), key=lambda x: -x[1]):
    pct = 100 * count / len(records)
    bar = "█" * int(pct / 2)
    print(f"  {lbl:<12} {count:>5}  ({pct:5.1f}%)  {bar}")
print("=" * 50)