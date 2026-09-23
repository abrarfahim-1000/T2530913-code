#!/usr/bin/env python
"""AC power flow vs the GNN — inference cost on N-1 contingency screening.

The thesis claim this measures: exhaustive AC screening is accurate but does not
scale to real-time operation, while the GNN reaches most of the answer at a
fraction of the cost. arXiv:2310.04213 reports 100-400x for NNs over an AC power
flow solution; this is the same measurement on this pipeline.

THE UNIT IS ONE CONTINGENCY, NOT ONE FRAME. The GNN emits one verdict per
(frame, energized line). The AC equivalent is one power-flow solve per
(frame, energized line) — which is exactly `label_n1`. Timing one solve per frame
would understate AC's cost by ~n_line (59x on neurips2020, 186x on wcci2022).

The AC arm calls `scripts.generate_dataset.label_n1` verbatim rather than
reimplementing it, so the number describes the pipeline that produced every
dataset in `data/`, not a benchmark written to be timed.

WARNING: `env.backend.runpf()` is NOT a valid AC arm. It re-solves whatever state
the backend already holds and never loads the snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.append(".")

import grid2op  # noqa: E402
from grid2op.Parameters import Parameters  # noqa: E402

from evaluation.eval_shield_n1 import (  # noqa: E402
    EVAL_BATCH_SIZE,
    best_f1_and_thr,
    score_topology_indices,
)
from scripts.generate_dataset import (  # noqa: E402
    ENV_CONFIGS,
    extract_features,
    label_n1,
)
from scripts.pyg_data import (  # noqa: E402
    GridEnvMetadata,
    PreloadedGridDataset,
    build_data,
)
from training.config import (  # noqa: E402
    DATA_DIR,
    DEVICE,
    EDGE_FEATURES,
    NODE_FEATURES,
    TRAIN_CONFIG,
)
from training.train_gnn import GridGNN, compute_normalization_stats  # noqa: E402

CKPT = "gnn_checkpoint_n1.pt"
HOME = "neurips2020"
RESULTS_DIR = Path("results")
TAG_TO_ENV = {cfg["tag"]: cfg["name"] for cfg in ENV_CONFIGS.values()}

# Frames per grid. wcci2022 runs 186 lines per frame, so fewer frames buy the
# same number of contingencies at a fraction of the wall clock.
DEFAULT_FRAMES = {"neurips2020": 500, "case14": 500, "wcci2022": 200}

BACKENDS = ("lightsim", "pandapower")


# ── device plumbing ─────────────────────────────────────────────────────────
def device_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return False


def sync(dev: torch.device) -> None:
    """Block until queued kernels finish.

    LOAD-BEARING. There is no synchronize call anywhere else in this repo — the
    eval loops get away without one only because `.float().cpu()` is an implicit
    sync point. A timed region must not rely on that: without this the forward
    returns before the kernels run and the measured speedup is fantasy.
    """
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def cpu_model_name() -> str:
    """Friendly CPU name. platform.processor() returns a family/model string on
    Windows, which is not a name anyone can look up."""
    try:
        if platform.system() == "Windows":
            key = r"HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-ItemProperty '{key}').ProcessorNameString"],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def machine_info(devices: list[str]) -> dict:
    """Recorded in the artifact ONLY — never in prose docs.

    CLAUDE.md states no hardware specifications are recorded anywhere in this
    repository. That rule exists because the specs were never load-bearing. A
    timing ratio is the one result that cannot be read without them, so this is a
    deliberate, scoped exception confined to this JSON.
    """
    def ver(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "not installed"

    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model_name(),
        "logical_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "devices_timed": devices,
        "grid2op": ver("grid2op"),
        "lightsim2grid": ver("lightsim2grid"),
        "pandapower": ver("pandapower"),
    }


# ── setup (never timed) ─────────────────────────────────────────────────────
def make_env(env_name: str, backend: str, seed: int):
    """Build the env with the REQUESTED backend, explicitly.

    `generate_dataset.load_backend()` silently prefers LightSim via try/except and
    has no flag, so it cannot be used to select PandaPower. A silent fallback here
    would void the headline, hence the assertion.
    """
    if backend == "lightsim":
        from lightsim2grid import LightSimBackend as Cls
    else:
        from grid2op.Backend import PandaPowerBackend as Cls

    env = grid2op.make(env_name, backend=Cls())
    # grid2op subclasses the backend per environment, so the runtime class name is
    # mangled ("LightSimBackend_l2rpn_case14_sandboxLightSimBackend"). isinstance
    # is the check that survives that; a name comparison would reject a correct
    # backend, and a substring test would be looser than it looks.
    if not isinstance(env.backend, Cls):
        raise RuntimeError(f"asked for {Cls.__name__}, grid2op built "
                           f"{type(env.backend).__name__}")

    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = False  # mirrors generate_dataset.py
    env.change_parameters(params)
    env.seed(seed)
    np.random.seed(seed)
    return env


def load_model(checkpoint: str) -> GridGNN:
    """One load serves every device arm.

    TRAIN_CONFIG only switches architecture on DEVICE.type == "cuda", so xpu and
    cpu both resolve to the deployed [16,32,32]/[4,4,1] branch that matches the
    checkpoint. On a CUDA box it resolves to [64,128,128] and would not load — the
    shape check below says so in words rather than raising a size mismatch.
    """
    state = torch.load(checkpoint, map_location="cpu")
    ckpt_hidden = state["conv1.att"].shape[-1]
    want_hidden = TRAIN_CONFIG["hidden_channels"][0]
    if ckpt_hidden != want_hidden:
        raise SystemExit(
            f"Checkpoint has hidden_channels[0]={ckpt_hidden} but TRAIN_CONFIG on "
            f"this device ({DEVICE.type}) gives {want_hidden}. TRAIN_CONFIG "
            f"branches on DEVICE.type (training/config.py); the deployed "
            f"checkpoint is the non-CUDA branch."
        )
    model = GridGNN(
        node_features=NODE_FEATURES, edge_features=EDGE_FEATURES,
        hidden_channels=TRAIN_CONFIG["hidden_channels"], heads=TRAIN_CONFIG["heads"],
        dropout=TRAIN_CONFIG["dropout"],
    )
    model.load_state_dict(state)
    model.eval()
    return model


def collect_snapshots(env, n_frames: int) -> tuple[list, dict]:
    """Step with do-nothing, keeping live observations.

    Observations are bound to the env that produced them (`obs.simulate` runs on
    that env's own forecast backend), which is why each backend gets its own
    snapshot set rather than sharing one.

    Returns the snapshots and a health record. The health record is not
    bookkeeping: measured 2026-09-20, PandaPower cannot hold wcci2022 alive for
    more than THREE steps under this policy (99 game-overs in 200 frames) while
    LightSim runs the whole collection without one. The frames that survive that
    are the opening steps of ~100 dying scenarios, sitting at median rho 1.24
    where LightSim sits at 0.78 — so 99.6% of contingencies violate and the
    population is nothing like the other backend's. Both arms still see the same
    frames, so the run is internally valid; it is the CROSS-BACKEND comparison
    that is void. Nothing about the output looks wrong without this record.
    """
    do_nothing = env.action_space({})
    snaps: list = []
    episode_lengths: list[int] = []
    since_reset = 0
    env.reset()
    while len(snaps) < n_frames:
        obs, _r, done, _i = env.step(do_nothing)
        since_reset += 1
        if done:
            episode_lengths.append(since_reset)
            since_reset = 0
            env.reset()
            continue
        if not obs.line_status.any():
            continue
        snaps.append(obs)

    n_go = len(episode_lengths)
    shortest = min(episode_lengths) if episode_lengths else None
    # An episode that dies every few steps is not sampling the grid, it is
    # sampling the same collapse repeatedly.
    degenerate = n_go > 0 and (sum(episode_lengths) / n_go) < 10
    return snaps, {
        "game_overs_during_collection": n_go,
        "shortest_episode": shortest,
        "mean_episode_length": (
            round(sum(episode_lengths) / n_go, 2) if n_go else None),
        "frames_are_independent": not degenerate,
    }


def summarize(xs: list[float]) -> dict:
    return {
        "median": round(statistics.median(xs), 4),
        "p95": round(sorted(xs)[min(int(0.95 * len(xs)), len(xs) - 1)], 4),
        "mean": round(statistics.fmean(xs), 4),
    }


# ── arm 1: AC ───────────────────────────────────────────────────────────────
def time_ac(snapshots: list, env, backend: str) -> tuple[dict, list]:
    """Time the incumbent path: one power-flow solve per energized line."""
    per_frame_ms: list[float] = []
    violations: list = []
    t0 = time.perf_counter()
    for obs in snapshots:
        f0 = time.perf_counter()
        viol, _post_rho = label_n1(obs, env)
        per_frame_ms.append((time.perf_counter() - f0) * 1e3)
        violations.append(viol)
    seconds = time.perf_counter() - t0

    n_cont = sum(1 for viol in violations for v in viol if v != -1)
    return {
        "backend": backend,
        "backend_class": type(env.backend).__name__,
        "method": "grid2op obs.simulate per energized line "
                  "(scripts.generate_dataset.label_n1)",
        "seconds": round(seconds, 4),
        "contingencies": n_cont,
        "per_contingency_ms": round(seconds * 1e3 / max(n_cont, 1), 6),
        "contingencies_per_sec": round(n_cont / seconds, 2) if seconds else None,
        "per_frame_ms": summarize(per_frame_ms),
    }, violations


# ── arm 2: GNN ──────────────────────────────────────────────────────────────
def build_graphs(snapshots: list, violations: list, meta) -> tuple[list, float]:
    """Observation -> PyG graph. Timed: it is real per-frame inference cost.

    The AC labels ride along in `n1_violation`, which is what lets the same graphs
    serve the accuracy check for free.
    """
    t0 = time.perf_counter()
    graphs = [
        build_data({**extract_features(obs), "n1_violation": viol}, meta)
        for obs, viol in zip(snapshots, violations)
    ]
    return graphs, time.perf_counter() - t0


def forward_pass(model, graphs: list, norm: tuple, dev: torch.device,
                 batch_size: int) -> dict:
    """One scored pass. Mirrors eval_shield_n1.score_topology_indices."""
    node_mean, node_std, edge_mean, edge_std = norm
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    lg, ys, ids, frames = [], [], [], []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            batch.x = (batch.x - node_mean) / node_std
            batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_fwd)
            m = batch.line_mask
            graph_of_edge = batch.batch[batch.edge_index[0]][batch.edge_fwd]
            lg.append(out[m].float().cpu())
            ys.append(batch.line_y[m].cpu())
            ids.append(batch.line_id[m].cpu())
            frames.append(graph_of_edge[m].cpu() + cursor)
            cursor += int(batch.num_graphs)
    return {
        "logits": torch.cat(lg).numpy(),
        "y": torch.cat(ys).numpy().astype(int),
        "line_id": torch.cat(ids).numpy(),
        "frame": torch.cat(frames).numpy(),
    }


def time_forward(model, graphs: list, norm: tuple, dev: torch.device,
                 batch_size: int, repeat: int) -> float:
    """Median of `repeat` timed passes, after one untimed warm-up."""
    forward_pass(model, graphs, norm, dev, batch_size)  # warm up
    runs = []
    for _ in range(repeat):
        sync(dev)
        t0 = time.perf_counter()
        forward_pass(model, graphs, norm, dev, batch_size)
        sync(dev)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


def time_batch1(model, graphs: list, norm: tuple, dev: torch.device,
                limit: int = 100) -> dict:
    """Single-frame latency — what an online loop would actually see.

    The batched figure is throughput under an offline workload; a control room
    scores one frame at a time. Reported next to it, never instead of it.
    """
    subset = graphs[:limit]
    forward_pass(model, subset[:2], norm, dev, 1)  # warm up
    per_frame_ms = []
    for g in subset:
        sync(dev)
        t0 = time.perf_counter()
        forward_pass(model, [g], norm, dev, 1)
        sync(dev)
        per_frame_ms.append((time.perf_counter() - t0) * 1e3)
    return summarize(per_frame_ms)


def gnn_arm(model, graphs: list, norm_cpu: tuple, dev_name: str,
            feat_seconds: float, batch_size: int,
            repeat: int) -> tuple[dict, dict]:
    """Time the GNN on one device. Returns (report, scored)."""
    dev = torch.device(dev_name)
    model = model.to(dev)
    norm = tuple(t.to(dev) for t in norm_cpu)

    fwd = time_forward(model, graphs, norm, dev, batch_size, repeat)
    end_to_end = feat_seconds + fwd
    scored = forward_pass(model, graphs, norm, dev, batch_size)
    n_cont = len(scored["y"])

    report = {
        "device": dev_name,
        "batch_size": batch_size,
        "repeats": repeat,
        "feature_build_seconds": round(feat_seconds, 4),
        "forward_seconds": round(fwd, 4),
        "end_to_end_seconds": round(end_to_end, 4),
        "contingencies": n_cont,
        "per_contingency_ms": round(end_to_end * 1e3 / max(n_cont, 1), 6),
        "contingencies_per_sec": round(n_cont / end_to_end, 2) if end_to_end else None,
        "forward_only_contingencies_per_sec": (
            round(n_cont / fwd, 2) if fwd else None
        ),
        "batch1_per_frame_ms": time_batch1(model, graphs, norm, dev),
    }
    return report, scored


# ── fairness assertions ─────────────────────────────────────────────────────
def assert_same_contingencies(scored: dict, violations: list) -> None:
    """Both arms must have screened the identical (frame, line) set.

    The classic way a speed benchmark lies is that the fast arm quietly does less
    work. This is the guard, and it is an assertion rather than a printed note.
    """
    ac = {(i, k) for i, viol in enumerate(violations)
          for k, v in enumerate(viol) if v != -1}
    gnn = set(zip(scored["frame"].tolist(), scored["line_id"].tolist()))
    if ac != gnn:
        raise AssertionError(
            f"arms scored different contingency sets: AC {len(ac):,}, "
            f"GNN {len(gnn):,}, symmetric difference {len(ac ^ gnn):,}"
        )


def assert_devices_agree(scoreds: dict[str, dict], thr: float) -> None:
    """Every device arm must compute the same function, or the timings compare
    two different models rather than two placements of one."""
    names = list(scoreds)
    base = (scoreds[names[0]]["logits"] > thr).astype(int)
    for name in names[1:]:
        other = (scoreds[name]["logits"] > thr).astype(int)
        n_diff = int((base != other).sum())
        if n_diff:
            raise AssertionError(
                f"{names[0]} and {name} disagree on {n_diff:,} of {len(base):,} "
                f"predictions at threshold {thr:.4f}"
            )


def accuracy(scored: dict, thr: float) -> dict:
    """The GNN against the AC labels on these very snapshots.

    AC is ground truth by construction (F1 1.0), so this is what turns a speed
    table into a speed/accuracy TRADE table. A speed-only table is not a finding.
    """
    y, pred = scored["y"], (scored["logits"] > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    p = float(y.mean()) if len(y) else 0.0
    return {
        "threshold": round(thr, 6),
        "threshold_source": "selected on the neurips2020 val split, held fixed",
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "positives": int(y.sum()),
        "all_positive_f1": round(2 * p / (1 + p), 4) if p else 0.0,
        "ac_f1": 1.0,
    }


CAVEATS = [
    "Two AC backends give two different ratios. LightSim generated every dataset "
    "in data/ and is the honest headline; PandaPower is the reference "
    "implementation and yields a ~10x larger ratio. Never quote one without "
    "naming which.",
    "Snapshot sets differ between backend runs: same seed and same do-nothing "
    "policy, but not bit-identical trajectories. Comparisons are exact WITHIN an "
    "artifact and indicative across two.",
    "obs.simulate is a pipeline, not a raw solver — it includes grid2op action "
    "construction and observation building, and it solves the t+1 forecast "
    "injections rather than the current instant.",
    "The GNN was timed on every available device; AC is CPU-bound and "
    "single-threaded. Quote the CPU row whenever an accelerator row is quoted, "
    "or the comparison confounds 'GNN vs AC' with 'GPU vs CPU'.",
    "Timing is hardware-dependent. The ratio is the result; the absolute seconds "
    "describe the one machine recorded in `environment`.",
    "The GNN batches; AC is serial by construction. That is a real architectural "
    "advantage, which is why batch-1 latency is reported beside it.",
    "Batch size 64 is load-bearing: BatchNorm(track_running_stats=False) uses "
    "live batch statistics, so the accuracy figures here hold at 64 only. Never "
    "mix them with figures taken at another batch size.",
    "The speed buys a measured accuracy loss. AC is ground truth (F1 1.0); the "
    "GNN scores 0.8956 / 0.4167 / 0.5577 at the held threshold, and on case14 it "
    "loses to answering 'violation' every time.",
]


# ── orchestration ───────────────────────────────────────────────────────────
LOCK = RESULTS_DIR / "timing" / ".bench.lock"


class _Lock:
    """Refuse to run while another benchmark is running.

    Two concurrent runs contend for the same cores and each measures the other's
    load. Nothing about the output looks wrong when that happens — the numbers are
    simply too slow, uniformly, and a reader cannot tell. This has already
    happened once during development, which is why it is enforced rather than
    documented.
    """

    def __enter__(self) -> "_Lock":
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(
                f"Another benchmark holds {LOCK} — concurrent runs contend for "
                f"the CPU and silently corrupt BOTH results.\n"
                f"Held by: {LOCK.read_text().strip()}\n"
                f"If no benchmark is running, that lock is stale: delete it."
            )
        with os.fdopen(fd, "w") as f:
            f.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self

    def __exit__(self, *exc) -> None:
        LOCK.unlink(missing_ok=True)


def select_threshold(checkpoint: str, batch_size: int, norm: tuple) -> float:
    """Chosen on the neurips2020 val split and held fixed — the same protocol as
    eval_shield_n1.py. Never selected on the set being reported."""
    val_path = os.path.join(DATA_DIR, "split_neurips2020_n1_val_idx.npy")
    if not os.path.exists(val_path):
        raise SystemExit(f"No val split at {val_path}; cannot hold a threshold.")
    val = score_topology_indices(HOME, checkpoint, batch_size, norm,
                                 np.load(val_path))
    _f1, thr = best_f1_and_thr(val["logits"], val["y"])
    return float(thr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="case14", choices=sorted(TAG_TO_ENV))
    ap.add_argument("--backend", default="lightsim", choices=BACKENDS,
                    help="AC solver to time. lightsim generated every dataset in "
                         "data/ and is the honest headline; pandapower is the "
                         "reference implementation (~10x slower).")
    ap.add_argument("--frames", type=int, default=None,
                    help="Snapshots to score (default per grid: "
                         f"{DEFAULT_FRAMES}).")
    ap.add_argument("--repeat", type=int, default=3,
                    help="Timed forward passes; the median is reported.")
    ap.add_argument("--devices", default="xpu,cpu",
                    help="Comma-separated torch devices for the GNN arm. "
                         "Unavailable ones are skipped.")
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE,
                    help=f"Eval batch size (default {EVAL_BATCH_SIZE}). "
                         "LOAD-BEARING for the accuracy columns — BatchNorm uses "
                         "live batch stats. See gnn_n1_tightening.md §8.")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ac", action="store_true",
                    help="GNN arm only. No speedup or accuracy is reported, "
                         "because both need the AC labels.")
    ap.add_argument("--smoke", action="store_true",
                    help="3 frames, 1 repeat — a wiring check, not a result.")
    return ap.parse_args()


def print_report(out: dict) -> None:
    ac = out.get("ac")
    print(f"\n{'=' * 72}")
    print(f"  INFERENCE COST — {out['tag']}  ({out['env']})")
    print(f"  {out['frames']:,} frames -> {out['contingencies']:,} contingencies")
    print(f"{'=' * 72}")
    if ac:
        print(f"\n  AC ({ac['backend']} / {ac['backend_class']})")
        print(f"    total            : {ac['seconds']:>12,.2f} s")
        print(f"    per contingency  : {ac['per_contingency_ms']:>12.4f} ms")
        print(f"    throughput       : {ac['contingencies_per_sec']:>12,.1f} /s")
        print(f"    per frame (ms)   : median {ac['per_frame_ms']['median']:.1f}"
              f"  p95 {ac['per_frame_ms']['p95']:.1f}")
    for dev, g in out["gnn"].items():
        print(f"\n  GNN [{dev}]  batch {g['batch_size']}")
        print(f"    feature build    : {g['feature_build_seconds']:>12,.3f} s")
        print(f"    forward          : {g['forward_seconds']:>12,.3f} s")
        print(f"    end to end       : {g['end_to_end_seconds']:>12,.3f} s")
        print(f"    throughput       : {g['contingencies_per_sec']:>12,.1f} /s "
              f"(forward only {g['forward_only_contingencies_per_sec']:,.1f} /s)")
        print(f"    batch-1 latency  : median "
              f"{g['batch1_per_frame_ms']['median']:.2f} ms  "
              f"p95 {g['batch1_per_frame_ms']['p95']:.2f} ms")
        sp = out["speedup"].get(dev)
        if sp:
            print(f"    SPEEDUP vs AC    : {sp['end_to_end']:>12,.1f}x end-to-end"
                  f"   ({sp['forward_only']:,.1f}x forward only)")
    acc = out.get("accuracy_on_these_frames")
    if acc:
        print(f"\n  ACCURACY on these frames (AC is ground truth, F1 1.0)")
        print(f"    threshold {acc['threshold']:.4f} — {acc['threshold_source']}")
        print(f"    GNN F1 {acc['f1']:.4f}   precision {acc['precision']:.4f}   "
              f"recall {acc['recall']:.4f}")
        print(f"    all-positive baseline F1 {acc['all_positive_f1']:.4f}")
    print(f"\n{'=' * 72}\n")


def main() -> None:
    args = parse_args()
    if args.batch_size != EVAL_BATCH_SIZE:
        print(f"\n  !! WARNING: --batch-size {args.batch_size} != "
              f"{EVAL_BATCH_SIZE}. BatchNorm uses live batch statistics, so the "
              f"accuracy columns will NOT match any recorded result. "
              f"(gnn_n1_tightening.md §8)\n")

    frames = 3 if args.smoke else (args.frames or DEFAULT_FRAMES[args.tag])
    repeat = 1 if args.smoke else args.repeat
    devices = [d for d in (x.strip() for x in args.devices.split(",")) if d]
    skipped = [d for d in devices if not device_available(d)]
    devices = [d for d in devices if device_available(d)]
    if skipped:
        print(f"[dev]  skipping unavailable: {', '.join(skipped)}")
    if not devices:
        sys.exit("No usable device in --devices.")
    print(f"[dev]  timing the GNN on: {', '.join(devices)}")

    meta_path = os.path.join(DATA_DIR, f"grid_dataset_{args.tag}_n1_meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"Missing {meta_path} — generate the dataset first.")
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Missing {args.checkpoint} — train the N-1 model first.")
    with open(meta_path) as f:
        meta = GridEnvMetadata(json.load(f))

    # ── setup, never timed ──────────────────────────────────────────────────
    print("Computing 36-bus N-1 train-split normalization stats...")
    home = PreloadedGridDataset(
        os.path.join(DATA_DIR, "processed_grid_data_n1.pt"), device=DEVICE)
    train_idx = np.load(
        os.path.join(DATA_DIR, "split_neurips2020_n1_train_idx.npy"))
    norm_dev = compute_normalization_stats(home, train_idx)
    norm_cpu = tuple(t.cpu() for t in norm_dev)
    del home

    model = load_model(args.checkpoint)

    env_name = TAG_TO_ENV[args.tag]
    print(f"[env]  {env_name} with {args.backend} backend...")
    env = make_env(env_name, args.backend, args.seed)
    print(f"[env]  n_line={env.n_line}  collecting {frames:,} snapshots...")
    snapshots, health = collect_snapshots(env, frames)
    if not health["frames_are_independent"]:
        print(f"\n  !! WARNING: {health['game_overs_during_collection']} "
              f"game-overs while collecting {frames:,} frames (mean episode "
              f"{health['mean_episode_length']} steps). This backend cannot hold "
              f"this grid up under a do-nothing policy, so the frames are the "
              f"opening steps of many dying scenarios rather than a sample of "
              f"normal operation. The two arms still see the SAME frames, so the "
              f"speedup here is internally valid — but this population is NOT "
              f"comparable to another backend's, and the accuracy column is "
              f"measured on a near-collapse grid. Do not table it against one.\n")

    # ── arm 1: AC ───────────────────────────────────────────────────────────
    if args.skip_ac:
        ac_report = None
        violations = [[0] * env.n_line for _ in snapshots]
    else:
        print(f"[ac]   screening {frames:,} frames x ~{env.n_line} lines...")
        ac_report, violations = time_ac(snapshots, env, args.backend)
        print(f"[ac]   {ac_report['seconds']:,.1f} s "
              f"({ac_report['contingencies_per_sec']:,.1f} contingencies/s)")

    # ── arm 2: GNN, same snapshots ──────────────────────────────────────────
    graphs, feat_seconds = build_graphs(snapshots, violations, meta)
    print(f"[gnn]  built {len(graphs):,} graphs in {feat_seconds:.2f} s")

    gnn_reports: dict[str, dict] = {}
    scoreds: dict[str, dict] = {}
    for dev_name in devices:
        print(f"[gnn]  timing on {dev_name} ({repeat} repeat(s))...")
        rep, scored = gnn_arm(model, graphs, norm_cpu, dev_name, feat_seconds,
                              args.batch_size, repeat)
        gnn_reports[dev_name], scoreds[dev_name] = rep, scored
        print(f"[gnn]  {dev_name}: {rep['end_to_end_seconds']:.3f} s end-to-end "
              f"({rep['contingencies_per_sec']:,.1f} contingencies/s)")

    # ── fairness gates ──────────────────────────────────────────────────────
    n_cont = len(scoreds[devices[0]]["y"])
    if not args.skip_ac:
        for dev_name, scored in scoreds.items():
            assert_same_contingencies(scored, violations)
        print(f"[gate] both arms screened the same {n_cont:,} contingencies")

    speedup = {}
    if ac_report:
        for dev_name, rep in gnn_reports.items():
            speedup[dev_name] = {
                "end_to_end": round(
                    ac_report["seconds"] / rep["end_to_end_seconds"], 2),
                "forward_only": round(
                    ac_report["seconds"] / rep["forward_seconds"], 2),
            }

    acc = None
    if not args.skip_ac:
        print("Selecting the decision threshold on the neurips2020 VAL split...")
        thr = select_threshold(args.checkpoint, args.batch_size, norm_dev)
        assert_devices_agree(scoreds, thr)
        if len(scoreds) > 1:
            print(f"[gate] all {len(scoreds)} device arms agree on every "
                  f"prediction")
        acc = accuracy(scoreds[devices[0]], thr)

    out = {
        "tag": args.tag,
        "env": env_name,
        "frames": len(snapshots),
        "contingencies": n_cont,
        "ac": ac_report,
        "gnn": gnn_reports,
        "speedup": speedup,
        "accuracy_on_these_frames": acc,
        "snapshot_health": health,
        "environment": machine_info(devices),
        "caveats": CAVEATS,
    }

    print_report(out)

    json_path = Path(args.json) if args.json else (
        RESULTS_DIR / "timing" / f"timing_{args.tag}_{args.backend}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  artifact -> {json_path}\n")
    env.close()


if __name__ == "__main__":
    with _Lock():
        main()
