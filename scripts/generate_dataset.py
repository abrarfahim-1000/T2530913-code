"""
Grid2Op Data Generation Pipeline — N-1 contingency screening
=============================================================
    python generate_dataset.py --env neurips --n_records 12000   # 36 subs,  59 lines [TRAINING]
    python generate_dataset.py --env case14  --n_records 6000    # 14 subs,  20 lines [TEST]
    python generate_dataset.py --env wcci    --n_records 4000    # 118 subs, 186 lines [TEST]
    python generate_dataset.py --env neurips --smoke             # 3 chronics — quick sanity check

Outputs (per run):
    grid_dataset_<env_tag>_n1.jsonl      — one JSON record per line (streamable)
    grid_dataset_<env_tag>_n1_meta.json  — env dims + label distribution

⚠️ Windows: run under the repo venv (`.venv/Scripts/python.exe`) — a bare `python` is the
Microsoft Store stub. Set `PYTHONIOENCODING=utf-8` when piping or redirecting: this script
prints `→`, and the cp1252 pipe encoding kills it mid-run on a UnicodeEncodeError.

RETIRED TASK PATHS
------------------
The 4-class `classify` generator was removed on 2026-08-16, mirroring the same cleanup in
`training/train_gnn.py`. Its target was a CLOSED-FORM function of the observation — four
threshold rules on `rho_max` and `n_tripped_lines` reproduce the stored labels with ZERO
disagreements across all 315,000 records of both datasets — so it could not distinguish a learned
model from a threshold, and rigged the neuro-symbolic comparison before it ran
(supplimentary_docs/thesis_findings.md §9.1).
Its code is in git history; the datasets it produced stay frozen on disk and are still cited.

`--task forecast` is DELIBERATELY RETAINED even though it was also rejected. No forecast
dataset was ever written to disk, so deleting the generator would make the negative result of
supplimentary_docs/thesis_findings.md §9.2 irreproducible. Do not "finish the cleanup" by
removing it.
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
ENV_CONFIGS = {
    "neurips": {
        "name": "l2rpn_neurips_2020_track1_small",
        "tag":  "neurips2020",
        "desc": "NeurIPS 2020 L2RPN — 36 subs, 59 lines [PRIMARY TRAINING]",
    },
    "case14": {
        "name": "l2rpn_case14_sandbox",
        "tag":  "case14",
        "desc": "L2RPN case14 sandbox — 14 subs, 20 lines [CROSS-TOPOLOGY TEST]",
    },
    "wcci": {
        "name": "l2rpn_wcci_2022",
        "tag":  "wcci2022",
        "desc": "WCCI 2022 L2RPN — 118 subs, 186 lines [CROSS-TOPOLOGY TEST]",
    },
}

# Defaults — overridden at runtime by --env arg
ENV_NAME = ENV_CONFIGS["neurips"]["name"]
ENV_TAG  = ENV_CONFIGS["neurips"]["tag"]
ENV_DESC = ENV_CONFIGS["neurips"]["desc"]

FAULT_PROB  = 0.05
RECONNECT_PROB = 0.20
SEED            = 42
RHO_CLIP        = 2.0    

# Smoke-test overrides (--smoke flag).
# SMOKE_MAX_RECORDS is load-bearing, not cosmetic: the generation loop cycles
# chronics until it reaches the record target, so capping chronics ALONE makes a
# smoke run replay the same 3 scenarios toward --n_records (default 300,000) and
# never terminate. Measured the hard way on the 118-bus grid at 2 rec/s -> 41h.
SMOKE_MAX_CHRONICS = 3
SMOKE_MAX_STEPS    = 500
SMOKE_MAX_RECORDS  = 200

# Current-state label, retained on every record for REFERENCE ONLY — it is not a
# training target for any surviving task. See the RETIRED TASK PATHS note above
# for why the 4-class task built on it was withdrawn.
LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3}

# ── FORECAST TASK ─────────────────────────────────────────────────────────────
# The 4-class `label` above is a CLOSED-FORM function of the observation:
#     rho_max >= 1.0 -> overload;  n_tripped == 0 -> normal;  == 1 -> line_trip;
#     else cascade
# Verified: 4 rules reproduce it across ALL 315,000 records of both topologies
# with zero disagreements. A model cannot demonstrate anything on that target
# that a threshold does not already do, which makes the neuro-symbolic
# comparison degenerate (supplimentary_docs/thesis_findings.md §9.1).
#
# The forecast task asks a question the present state does NOT determine:
#     will the grid leave the `normal` state within the next HORIZON steps?
# Faults are injected stochastically (FAULT_PROB) and load evolves, so this is
# genuinely predictive. It also un-rigs the shield: rules see only the present,
# the label depends on the future, so no rule can restate the labeller.
RISK_LABEL_MAP = {"not_at_risk": 0, "at_risk": 1}
DEFAULT_HORIZON = 6            # steps; 5 min/step in Grid2Op -> 30 minutes ahead

# Which future faults make a frame `at_risk`. See flush_forecast_chronic() for
# the measurement that fixes the default: line-trip ONSET is injected by an
# unconditional Bernoulli draw below (FAULT_PROB), so an ANY-fault target is
# dominated by irreducible noise and no model beats the base rate on it.
RISK_SOURCES = {
    "overload":           ("overload",),
    "overload_cascade":   ("overload", "cascade"),
    "any":                ("overload", "line_trip", "cascade"),
}
DEFAULT_RISK_SOURCE = "overload"

# n1 task: one power-flow solve per energized line per labelled frame (~170/s
# measured), so consecutive frames are expensive AND nearly duplicates.
#
# Sizing measured on the 36-bus env: 576 chronics available (a hard diversity
# ceiling), episodes surviving ~190 steps under fault injection. At stride 4 that
# is ~48 frames/chronic, so a 10k-frame run would spend its whole budget on ~210
# scenarios - a third of what exists - sampling near-duplicate frames. At stride
# 12 it is ~16 frames/chronic, so the same budget covers ALL 576. Effective
# sample size is bounded by scenario count, not by label count: frames 12 steps
# apart still share a load profile, and all n_line contingencies in one frame
# share a base state.
DEFAULT_N1_STRIDE = 12
N1_LABEL_MAP = {"secure": 0, "violation": 1}

LINE_KEYS = ("rho", "p_or", "q_or", "p_ex", "q_ex", "v_or", "v_ex")
BUS_KEYS  = ("load_p", "load_q", "gen_p", "gen_q", "topo_vect")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Grid2Op dataset generator")
    parser.add_argument(
        "--env", type=str, default="neurips", choices=list(ENV_CONFIGS.keys()),
        help="Which Grid2Op environment to use: neurips (default), case14, wcci"
    )
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
        "--task", choices=("n1", "forecast"), default="n1",
        help="n1 (DEFAULT, and the only supported task): per-line N-1 contingency "
             "screening - for each energized line, does tripping it violate a thermal "
             "limit? Its label is not reproducible by any rule over the present "
             "observation, because producing it requires a power-flow solve. "
             "forecast: binary 'fault within --horizon steps' - REJECTED as degenerate, "
             "retained only so that negative result stays reproducible; do not train "
             "against it (see label_n1 docstring and thesis_findings.md §9.2).",
    )
    parser.add_argument(
        "--n1-stride", type=int, default=DEFAULT_N1_STRIDE,
        help=f"n1 only: label every Nth step (default {DEFAULT_N1_STRIDE}). Each "
             "labelled frame costs one power-flow solve PER energized line, so the "
             "stride buys chronic diversity per unit of compute rather than near-"
             "duplicate consecutive frames.",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"forecast only: lookahead in steps (default {DEFAULT_HORIZON} = 30 min)",
    )
    parser.add_argument(
        "--risk-source", choices=tuple(RISK_SOURCES), default=DEFAULT_RISK_SOURCE,
        help="forecast only: which future faults set at_risk. 'overload' (default) is "
             "the only one with measurable signal - line-trip onset is an unconditional "
             "coin flip, so 'any' is dominated by noise. Per-class distances are written "
             "on every record, so this can be changed by relabelling, not regenerating.",
    )
    parser.add_argument(
        "--n_records", type=int, default=300000,
        help="Target number of records to write (default: 300000). Use a smaller value for "
             "cross-topology TEST sets, e.g. --n_records 15000."
    )
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Output directory (default: 'data' directory)"
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

def flush_forecast_chronic(frames: list[dict], horizon: int, out_f,
                           label_counts: Counter, budget: int,
                           risk_source: str = DEFAULT_RISK_SOURCE) -> int:
    """Attach lookahead labels to one buffered chronic and write it out.

    `at_risk` is True when a fault of a kind selected by `risk_source` occurs in
    the next `horizon` frames. The final `horizon` frames are DROPPED: their
    lookahead window runs past the end of the episode, so their label is
    undefined rather than negative. Keeping them would systematically mislabel
    the end of every chronic as safe.

    ⚠️ `risk_source` is NOT just a knob - it decides whether the target is
    learnable at all. Line trips are injected in main() by an unconditional
    `np.random.rand() < FAULT_PROB` on a uniformly random line, so their ONSET
    is exogenous noise that no observation can predict. Measured on 1,995
    unbiased chronics of the 36-bus training set, the best possible threshold on
    the current `rho_max` scores F1 1.00x the all-positive baseline against an
    ANY-fault target at every H >= 3 - i.e. zero signal - but 2.0x-9.0x against
    an OVERLOAD target. Overload onset is produced by the load/generation
    trajectory and is genuinely forecastable. Hence the default.

    Every per-class distance is written on EVERY record (`steps_to_overload`,
    `steps_to_trip`, `steps_to_cascade`, `steps_to_fault`), so both the horizon
    AND the choice of risk source can be changed later by RELABELLING the JSONL
    rather than regenerating it. -1 means "does not occur again in this chronic".

    Returns the number of records written.
    """
    n = len(frames)
    if n <= horizon or budget <= 0:
        return 0

    # next_at[kind][t] = index of the next frame at/after t carrying that label.
    # Backward sweep so each frame costs O(1) per kind, not O(horizon).
    kinds = ("overload", "line_trip", "cascade")
    next_at: dict[str, list[int | None]] = {}
    for kind in kinds:
        seq: list[int | None] = [None] * n
        nxt: int | None = None
        for t in range(n - 1, -1, -1):
            if frames[t]["label"] == kind:
                nxt = t
            seq[t] = nxt
        next_at[kind] = seq

    def first_of(kinds_: tuple[str, ...], t: int) -> int | None:
        """Earliest occurrence of any of `kinds_` strictly after t."""
        hits = [next_at[k][t + 1] for k in kinds_ if next_at[k][t + 1] is not None]
        return min(hits) if hits else None

    trigger = RISK_SOURCES[risk_source]

    written = 0
    for t in range(n - horizon):
        nxt = first_of(trigger, t)
        at_risk = nxt is not None and nxt <= t + horizon
        risk = "at_risk" if at_risk else "not_at_risk"

        def dist(kinds_: tuple[str, ...]) -> int:
            hit = first_of(kinds_, t)
            return (hit - t) if hit is not None else -1

        record = {
            **frames[t],
            "risk_label":        risk,
            "risk_int":          RISK_LABEL_MAP[risk],
            "horizon":           horizon,
            "risk_source":       risk_source,
            "steps_to_fault":    dist(kinds),
            "steps_to_overload": dist(("overload",)),
            "steps_to_trip":     dist(("line_trip",)),
            "steps_to_cascade":  dist(("cascade",)),
        }
        validate_record(record)
        out_f.write(json.dumps(record) + "\n")
        label_counts[frames[t]["label"]] += 1
        written += 1
        if written >= budget:
            break
    return written


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

def label_n1(obs, env) -> tuple[list[int], list[float]]:
    """N-1 contingency screening labels for one frame.

    For every currently-energized line k: trip it and ask whether the resulting
    power flow violates a thermal limit. Returns two per-line vectors aligned to
    `line_id`:

        violation[k]  1 = post-contingency violation, 0 = secure,
                     -1 = not evaluated (line already out, or the solver failed)
        post_rho[k]   max rho after removing k (-1.0 where not evaluated)

    Why this target and not "fault within H steps": measured on this generator,
    the answer to *this* question is NOT determined by the current state - the
    best threshold on the current `rho_max` scores 1.04x the all-positive
    baseline, and 100% of sampled frames are mixed (some lines violate, some do
    not). It depends on how flow REDISTRIBUTES through the remaining network,
    which is a property of the topology - exactly the inductive bias a GNN has
    and a threshold rule does not. A network-aware model reaches F1 0.868 where
    the best per-line rule (flow on the removed line) reaches 0.608.

    It also un-rigs the shield permanently: no rule over the present observation
    can restate this label, because producing it requires a power-flow solve.
    """
    n_line = int(env.n_line)
    violation = [-1] * n_line
    post_rho = [-1.0] * n_line

    for k in range(n_line):
        if not obs.line_status[k]:
            continue
        try:
            act = env.action_space({"set_line_status": [(k, -1)]})
            sim_obs, _r, sim_done, _i = obs.simulate(act)
        except Exception:
            continue
        if sim_obs is None:
            # solver diverged; leave as -1 rather than guessing a class
            continue
        rho = float(sim_obs.rho.max()) if len(sim_obs.rho) else 0.0
        # a game-over is the most severe violation there is, not a missing label
        violation[k] = int(bool(sim_done) or rho >= 1.0)
        post_rho[k] = rho

    return violation, post_rho


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
        # Shapes built by scripts/pyg_data.py — kept in sync with NODE_FEATURES /
        # EDGE_FEATURES there. Both went 4 -> 8 when the N-1 features landed; these
        # two fields were left stale until 2026-08-16. Nothing reads them (the
        # trainer imports the constants directly), so they are descriptive only.
        "node_feature_dim": 8,   # load_p, mean_v, max_rho, connected_line_frac,
                                 # global_trip_frac, sum_headroom, sum_abs_p, degree
        "edge_feature_dim": 8,   # rho, p_or, q_or, near_limit, |p_or|, |q_or|,
                                 # apparent_s, headroom — bidirectional per active line
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
    global ENV_NAME, ENV_TAG, ENV_DESC

    args = parse_args()
    smoke = args.smoke

    # Apply --env selection
    cfg      = ENV_CONFIGS[args.env]
    ENV_NAME = cfg["name"]
    ENV_TAG  = cfg["tag"]
    ENV_DESC = cfg["desc"]

    max_chronics = SMOKE_MAX_CHRONICS if smoke else args.max_chronics
    max_steps    = SMOKE_MAX_STEPS    if smoke else args.max_steps
    # Without this the smoke run never exits — see the SMOKE_MAX_RECORDS comment.
    n_records    = min(args.n_records, SMOKE_MAX_RECORDS) if smoke else args.n_records

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every task carries its own suffix so it can never overwrite the frozen
    # classify datasets, which have no suffix and are still cited in the thesis.
    TASK_SUFFIX = {"n1": "_n1", "forecast": "_forecast"}
    tag      = ENV_TAG + ("_smoke" if smoke else "") + TASK_SUFFIX[args.task]
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

    TARGET_RECORDS = n_records  # default 300000; smaller for cross-topology test sets
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
                # Cycle through the available chronics repeatedly (epochs).
                # The id RECORDED on each frame must be the scenario (load
                # profile), never the episode counter: this loop wraps around
                # n_chronics, so a second pass over the same week would get a
                # fresh id, and a chronic-level split would then place the same
                # load profile in BOTH train and test. `episode_id` keeps the
                # pass counter for debugging.
                scenario_id = chronic_idx % max(1, n_chronics)
                env.set_id(scenario_id)
                obs   = env.reset()
                episode_id = chronic_idx
                chronic_idx += 1

                steps = min(env.max_episode_duration(),
                            max_steps if max_steps else int(1e9))

                # prev_line_status = obs.line_status.copy()
                # tripped_lines = set()

                # forecast mode buffers the whole chronic: the label for frame t
                # depends on frames t+1..t+H, so nothing can be written until the
                # chronic finishes. This is also why forecast mode must not
                # subsample - a dropped frame is a hole in someone's lookahead window.
                chronic_buffer: list[dict] = []

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

                    if args.task == "n1":
                        # One record per FRAME carrying a per-line label vector,
                        # not one record per contingency: the frame features are
                        # shared by all n_line contingencies, so duplicating them
                        # would inflate the file ~n_line-fold. The vector layout
                        # also lines up 1:1 with an edge-level GNN head.
                        if t % max(args.n1_stride, 1) != 0:
                            if done:
                                break
                            continue

                        violation, post_rho = label_n1(obs, env)
                        n_eval = sum(1 for v in violation if v >= 0)
                        if n_eval == 0:
                            if done:
                                break
                            continue

                        record = {
                            **extract_features(obs),
                            "n1_violation":   violation,
                            "n1_post_rho":    post_rho,
                            "n1_n_evaluated": n_eval,
                            "n1_n_violation": sum(1 for v in violation if v == 1),
                            "label":          fault_label,
                            "label_int":      LABEL_MAP[fault_label],
                            "fault_loc":      fault_loc,
                            "timestep":       t,
                            "chronic_id":     scenario_id,
                            "episode_id":     episode_id,
                            "reward":         float(reward),
                        }
                        validate_record(record)
                        out_f.write(json.dumps(record) + "\n")
                        label_counts[fault_label] += 1
                        total_written += 1
                        pbar.update(1)
                        if total_written >= TARGET_RECORDS or done:
                            break
                        continue

                    # forecast is the only remaining alternative to n1, and it
                    # buffers rather than writes — the label for frame t depends on
                    # t+1..t+H, so nothing can be emitted until the chronic ends.
                    chronic_buffer.append({
                        **extract_features(obs),
                        "label":      fault_label,
                        "label_int":  LABEL_MAP[fault_label],
                        "fault_loc":  fault_loc,
                        "timestep":   t,
                        "chronic_id": scenario_id,
                        "episode_id": episode_id,
                        "reward":     float(reward),
                    })
                    if done:
                        break

                # ── forecast: label the buffered chronic by looking ahead ────────
                if args.task == "forecast" and chronic_buffer:
                    written = flush_forecast_chronic(
                        chronic_buffer, args.horizon, out_f,
                        label_counts, TARGET_RECORDS - total_written,
                        risk_source=args.risk_source,
                    )
                    total_written += written
                    pbar.update(written)

    total_time = time.time() - t_start
    meta = build_meta(env, label_counts, total_written, total_time, smoke)
    meta["task"] = args.task
    if args.task == "n1":
        meta["state_label_map"] = meta.get("label_map")
        meta["state_label_counts"] = meta.get("label_counts")
        meta["label_map"] = dict(N1_LABEL_MAP)
        meta["n_classes"] = len(N1_LABEL_MAP)
        meta["label_key"] = "n1_violation"     # per-LINE vector, not a scalar
        meta["label_level"] = "edge"
        meta["n1_stride"] = args.n1_stride
        meta["note"] = (
            "N-1 contingency screening. Each record is ONE FRAME; `n1_violation` is "
            "a per-line vector aligned to line_id: 1 = tripping that line violates a "
            "thermal limit (or ends the episode), 0 = secure, -1 = not evaluated "
            "(line already out, or the power flow diverged). -1 entries MUST be "
            "masked out of the loss - they are missing labels, not negatives. "
            "`n1_post_rho` carries max rho after each outage, so a severity/"
            "regression variant is derivable without regenerating. This target is "
            "not reproducible by any rule over the present observation: producing it "
            "requires a power-flow solve."
        )
    if args.task == "forecast":
        meta["horizon"] = args.horizon
        # train_gnn.py reads label_map / n_classes straight from the meta, so
        # overriding them here is what switches training to the binary target —
        # no flag needed at the training end.
        meta["state_label_map"] = meta.get("label_map")     # keep the 4-class one for reference
        meta["state_label_counts"] = meta.get("label_counts")
        meta["label_map"] = dict(RISK_LABEL_MAP)
        meta["n_classes"] = len(RISK_LABEL_MAP)
        meta["label_key"] = "risk_int"
        meta["risk_label_map"] = RISK_LABEL_MAP
        meta["risk_source"] = args.risk_source
        meta["risk_trigger_labels"] = list(RISK_SOURCES[args.risk_source])
        meta["note"] = (
            "Binary lookahead target: at_risk = a fault of kind "
            f"{list(RISK_SOURCES[args.risk_source])} occurs within `horizon` steps. "
            "Frames are contiguous within each chronic and the last `horizon` frames "
            "of each chronic are dropped (undefined lookahead). `label`/`label_int` "
            "retain the legacy 4-class current-state label for reference; they are NOT "
            "the training target for this task. Every record also carries "
            "steps_to_{fault,overload,trip,cascade}, so BOTH the horizon and the risk "
            "source can be changed by relabelling this file - no regeneration needed."
        )

    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print_summary(meta, out_jsonl, out_meta)


if __name__ == "__main__":
    main()