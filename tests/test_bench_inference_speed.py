"""Guards for the AC-vs-GNN inference speed benchmark.

A timing benchmark is unusually easy to get wrong in a way that still produces a
plausible number. The three failure modes worth pinning:

  1. The fast arm quietly screens fewer contingencies than the slow one, so the
     ratio is measuring workload rather than speed.
  2. An accelerator arm returns before its kernels finish, so the forward pass
     looks free.
  3. grid2op silently hands back a different backend than the one asked for, so
     the "AC" figure is the other solver's.

Each has a test here. Elapsed times are deliberately NOT asserted — they are
hardware-dependent and would make the suite flaky.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.append(".")

from evaluation.bench_inference_speed import (
    BACKENDS,
    CAVEATS,
    DEFAULT_FRAMES,
    TAG_TO_ENV,
    accuracy,
    assert_devices_agree,
    assert_same_contingencies,
    device_available,
    machine_info,
    summarize,
    sync,
    time_ac,
)

GRID2OP_ENV_DIR = os.path.expanduser("~/data_grid2op")
needs_env = pytest.mark.skipif(
    not os.path.isdir(GRID2OP_ENV_DIR),
    reason="grid2op environment data not present on this machine",
)


# ── the AC arm must be the pipeline, not a copy of it ───────────────────────
def test_ac_arm_calls_the_generator_verbatim():
    """If the benchmark reimplements label_n1, it stops describing the pipeline
    that produced every dataset in data/ and starts describing itself."""
    import evaluation.bench_inference_speed as bench
    from scripts.generate_dataset import label_n1

    assert bench.label_n1 is label_n1


def test_tags_cover_all_three_grids():
    assert set(TAG_TO_ENV) == {"neurips2020", "case14", "wcci2022"}
    assert set(DEFAULT_FRAMES) == set(TAG_TO_ENV)
    assert set(BACKENDS) == {"lightsim", "pandapower"}


def test_caveats_name_both_backends_and_the_accuracy_cost():
    """The number is not citable without them, so they ship inside the artifact."""
    blob = " ".join(CAVEATS).lower()
    for token in ("lightsim", "pandapower", "batch", "hardware", "ground truth"):
        assert token in blob


# ── gate 1: both arms screened the same work ────────────────────────────────
def _scored(pairs, logits=None):
    frame = np.array([p[0] for p in pairs], dtype=np.int64)
    line_id = np.array([p[1] for p in pairs], dtype=np.int64)
    n = len(pairs)
    return {
        "frame": frame,
        "line_id": line_id,
        "logits": np.zeros(n) if logits is None else np.asarray(logits),
        "y": np.zeros(n, dtype=int),
    }


def test_same_contingencies_passes_when_arms_agree():
    violations = [[1, 0, -1], [-1, 1, 0]]
    pairs = [(0, 0), (0, 1), (1, 1), (1, 2)]
    assert_same_contingencies(_scored(pairs), violations)


def test_same_contingencies_raises_when_gnn_screens_fewer():
    """The classic lie: the fast arm does less work and looks faster for it."""
    violations = [[1, 0, -1], [-1, 1, 0]]
    pairs = [(0, 0), (0, 1), (1, 1)]  # (1, 2) missing
    with pytest.raises(AssertionError, match="different contingency sets"):
        assert_same_contingencies(_scored(pairs), violations)


def test_same_contingencies_counts_unevaluated_as_not_screened():
    """-1 means the solver never answered; it is not a contingency either arm
    scored, and counting it would inflate AC's denominator."""
    violations = [[-1, -1, -1]]
    assert_same_contingencies(_scored([]), violations)


# ── gate 2: every device arm computes the same function ─────────────────────
def test_devices_agree_passes_on_identical_logits():
    a = _scored([(0, 0), (0, 1)], logits=[0.9, 0.1])
    b = _scored([(0, 0), (0, 1)], logits=[0.91, 0.11])  # same side of 0.5
    assert_devices_agree({"xpu": a, "cpu": b}, thr=0.5)


def test_devices_agree_raises_when_predictions_differ():
    a = _scored([(0, 0), (0, 1)], logits=[0.9, 0.1])
    b = _scored([(0, 0), (0, 1)], logits=[0.9, 0.7])
    with pytest.raises(AssertionError, match="disagree on 1"):
        assert_devices_agree({"xpu": a, "cpu": b}, thr=0.5)


# ── gate 3: the accelerator must actually be waited on ──────────────────────
@pytest.mark.parametrize("dev_type", ["xpu", "cuda"])
def test_sync_dispatches_to_the_accelerator(monkeypatch, dev_type):
    """Without this the forward returns before the kernels run and the speedup
    is fantasy. There is no other synchronize call in this repo."""
    called = []
    mod = getattr(torch, dev_type)
    monkeypatch.setattr(mod, "synchronize", lambda *a: called.append(dev_type))
    sync(torch.device(dev_type))
    assert called == [dev_type]


def test_sync_is_a_noop_on_cpu(monkeypatch):
    called = []
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "synchronize", lambda *a: called.append("x"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: called.append("c"))
    sync(torch.device("cpu"))
    assert called == []


def test_cpu_is_always_available_and_nonsense_is_not():
    assert device_available("cpu")
    assert not device_available("tpu")


# ── reporting maths ─────────────────────────────────────────────────────────
def test_summarize_reports_median_p95_mean():
    s = summarize([float(i) for i in range(1, 101)])
    assert s["median"] == pytest.approx(50.5)
    assert s["mean"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(96.0)


def test_accuracy_records_ac_as_ground_truth():
    """AC produced the labels, so its F1 is 1.0 by construction. The artifact has
    to say so, or a reader sees a speed table and no cost."""
    scored = {"logits": np.array([0.9, 0.9, 0.1, 0.1]),
              "y": np.array([1, 0, 1, 0])}
    acc = accuracy(scored, thr=0.5)
    assert acc["ac_f1"] == 1.0
    assert acc["tp"] == 1 and acc["fp"] == 1 and acc["fn"] == 1
    assert acc["f1"] == pytest.approx(0.5)
    assert acc["precision"] == pytest.approx(0.5)
    assert "val split" in acc["threshold_source"]


def test_accuracy_reports_the_all_positive_baseline():
    """A model that loses to 'violation every time' must be visible as such."""
    scored = {"logits": np.array([0.9, 0.1, 0.1, 0.1]),
              "y": np.array([1, 1, 1, 0])}
    acc = accuracy(scored, thr=0.5)
    assert acc["all_positive_f1"] > acc["f1"]


def test_lock_refuses_a_concurrent_run(monkeypatch, tmp_path):
    """Two benchmarks on one machine each measure the other's CPU load. Nothing
    about the output looks wrong — it is just uniformly slow. This happened once
    during development and silently corrupted a result, so it is enforced."""
    import evaluation.bench_inference_speed as bench

    monkeypatch.setattr(bench, "LOCK", tmp_path / "t" / ".bench.lock")
    with bench._Lock():
        assert bench.LOCK.exists()
        with pytest.raises(SystemExit, match="concurrent runs contend"):
            with bench._Lock():
                pass
    assert not bench.LOCK.exists()  # released on the way out


@needs_env
def test_collection_health_flags_a_grid_the_backend_cannot_hold_up():
    """Measured 2026-09-20: PandaPower game-overs on wcci2022 after THREE steps,
    99 times in 200 frames, while LightSim runs the collection without one. The
    surviving frames sit at median rho 1.24 against LightSim's 0.78, so 99.6% of
    contingencies violate and the population is nothing like the other backend's.
    Nothing in the timing output looks wrong when that happens, which is why the
    health record exists."""
    from evaluation.bench_inference_speed import collect_snapshots, make_env

    env = make_env("l2rpn_wcci_2022", "pandapower", seed=42)
    try:
        _snaps, health = collect_snapshots(env, 20)
    finally:
        env.close()

    assert health["game_overs_during_collection"] > 0
    assert health["mean_episode_length"] < 10
    assert not health["frames_are_independent"], (
        "wcci2022 + PandaPower used to collapse every 3 steps; if this now holds "
        "up, the cross-backend comparison on wcci2022 may be recoverable — "
        "see thesis_findings.md §25"
    )


def test_machine_info_records_what_a_timing_ratio_needs():
    info = machine_info(["cpu"])
    for key in ("platform", "cpu_model", "logical_cores", "torch_version",
                "torch_threads", "grid2op", "pandapower"):
        assert info[key] not in (None, "")


# ── env-backed: the backend must be the one requested ───────────────────────
@needs_env
@pytest.mark.parametrize("backend", BACKENDS)
def test_make_env_builds_the_requested_backend(backend):
    """grid2op subclasses the backend per env, so the runtime class name is
    mangled and a silent fallback would be easy to miss."""
    from evaluation.bench_inference_speed import make_env

    env = make_env("l2rpn_case14_sandbox", backend, seed=42)
    try:
        name = type(env.backend).__name__
        assert ("LightSim" if backend == "lightsim" else "PandaPower") in name
        assert not env.parameters.NO_OVERFLOW_DISCONNECTION
    finally:
        env.close()


@needs_env
def test_time_ac_evaluates_every_energized_line():
    """The unit is one contingency, not one frame. If this ever reports one solve
    per frame the whole comparison is off by ~n_line."""
    from evaluation.bench_inference_speed import collect_snapshots, make_env

    env = make_env("l2rpn_case14_sandbox", "lightsim", seed=42)
    try:
        snaps, health = collect_snapshots(env, 2)
        assert health["frames_are_independent"], health
        report, violations = time_ac(snaps, env, "lightsim")
        energized = sum(int(o.line_status.sum()) for o in snaps)
        assert report["contingencies"] <= energized
        assert report["contingencies"] > len(snaps)  # not one-per-frame
        assert report["contingencies_per_sec"] > 0
        assert len(violations) == len(snaps)
    finally:
        env.close()


# ── end to end ──────────────────────────────────────────────────────────────
@needs_env
def test_smoke_run_writes_a_complete_artifact(tmp_path):
    """Runs the real CLI. Slow, but it is the only test that proves the gates fire
    against a live grid rather than against synthetic arrays."""
    out = tmp_path / "timing.json"
    proc = subprocess.run(
        [sys.executable, "evaluation/bench_inference_speed.py", "--tag", "case14",
         "--backend", "lightsim", "--smoke", "--json", str(out)],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "both arms screened the same" in proc.stdout

    art = json.loads(out.read_text())
    assert art["tag"] == "case14"
    assert art["contingencies"] > 0
    assert art["ac"]["contingencies"] == art["contingencies"]
    assert art["caveats"] and art["environment"]["cpu_model"]

    for dev, g in art["gnn"].items():
        assert g["contingencies"] == art["contingencies"], dev
        assert g["end_to_end_seconds"] >= g["forward_seconds"] > 0
        assert g["batch1_per_frame_ms"]["median"] > 0
    for dev, sp in art["speedup"].items():
        assert np.isfinite(sp["end_to_end"]) and sp["end_to_end"] > 0, dev
        assert sp["forward_only"] >= sp["end_to_end"], dev


@needs_env
def test_nonstandard_batch_size_warns(tmp_path):
    """Batch size changes the logits (live BatchNorm stats), so the accuracy
    columns silently stop matching any recorded result."""
    proc = subprocess.run(
        [sys.executable, "evaluation/bench_inference_speed.py", "--tag", "case14",
         "--backend", "lightsim", "--smoke", "--batch-size", "32",
         "--json", str(tmp_path / "t.json")],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert "WARNING" in proc.stdout and "batch-size 32" in proc.stdout
