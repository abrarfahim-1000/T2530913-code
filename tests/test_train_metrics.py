"""
Tests for the training run record (training/checkpoint.py + train_gnn.py).

WHAT THESE PROTECT
------------------
1. **The deployed checkpoint must keep loading.** `gnn_checkpoint_n1.pt` is the
   thesis deliverable and five call sites load it with the bare

       model.load_state_dict(torch.load(path, map_location=DEVICE))

   `load_state_dict` is strict, so a single extra non-tensor key in the saved
   mapping breaks all of them. The enrichment therefore goes in SIBLING files
   and the deliverable's format is unchanged. If someone ever "improves" that
   by saving a wrapped dict under the main name, these fail.

2. **Both forms load through one call.** `load_checkpoint()` accepts the bare
   form and the enriched form and returns a state_dict either way, so a caller
   never has to know which it was handed.

3. **The history survives an interrupted run.** History is appended per epoch,
   not written at the end. A run killed at epoch 7 leaves 7 epochs. That is the
   difference between having a record and not.

Run: pytest tests/test_train_metrics.py -v
"""
import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    EpochRecord,
    RunMetadata,
    RunRecorder,
    describe,
    enriched_path_for,
    history_path_for,
    load_checkpoint,
    load_history,
    meta_path_for,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)


def _metadata(ckpt, epochs=3, **over):
    base = dict(
        format_version=CHECKPOINT_FORMAT_VERSION,
        task="n1",
        checkpoint_file=str(ckpt),
        best_epoch=None,
        best_val_contingency_f1=None,
        epochs_requested=epochs,
        epochs_run=0,
        early_stopped=False,
        config={"epochs": epochs, "batch_size": 128, "lr": 3e-4,
                "hidden_channels": [16, 32, 32], "heads": [4, 4, 1]},
        node_features=8,
        edge_features=8,
        seed=42,
        device="cpu",
        torch_version=torch.__version__,
        python_version="3.11.0",
        git_commit="abc123",
        git_dirty=False,
        started_utc="2026-09-20T00:00:00+00:00",
    )
    base.update(over)
    return RunMetadata(**base)


def _record(epoch, val_f1, is_best, train_f1=None):
    return EpochRecord(
        epoch=epoch, train_loss=1.0 / epoch, val_contingency_f1=val_f1,
        lr=3e-4, is_best=is_best, seconds=1.0, val_ap=0.9,
        train_contingency_f1=train_f1,
    )


# ---------------------------------------------------------------------------
# 1. BACKWARD COMPATIBILITY — the deliverable's format does not change
# ---------------------------------------------------------------------------

def test_the_main_checkpoint_is_still_a_bare_state_dict(tmp_path):
    """A plain torch.load must yield something load_state_dict accepts directly.

    This is the exact call the five existing eval loaders make. It is not
    routed through any helper, deliberately — the point is that they work
    untouched.
    """
    ckpt = tmp_path / "gnn_checkpoint_n1.pt"
    model = _Tiny()
    rec = RunRecorder(ckpt, _metadata(ckpt))
    rec.log_epoch(_record(1, 0.80, True))
    rec.save_best(model)

    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)

    assert isinstance(loaded, dict)
    assert all(isinstance(v, torch.Tensor) for v in loaded.values()), \
        "a non-tensor value in the deliverable breaks every strict load_state_dict"
    assert set(loaded) == set(model.state_dict()), \
        "extra or missing keys in the deliverable break strict loading"

    fresh = _Tiny()
    fresh.load_state_dict(loaded)  # strict=True — must not raise


def test_a_genuinely_old_bare_checkpoint_still_loads(tmp_path):
    """Simulates gnn_checkpoint_n1.pt as saved in Aug 2026: no sidecar at all."""
    ckpt = tmp_path / "legacy.pt"
    model = _Tiny()
    torch.save(model.state_dict(), ckpt)  # the pre-change save, verbatim

    state, meta = load_checkpoint(ckpt, map_location="cpu")

    assert meta is None, "a bare checkpoint with no sidecar has no metadata — say so"
    _Tiny().load_state_dict(state)


@pytest.mark.skipif(not os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "gnn_checkpoint_n1.pt")),
    reason="deployed checkpoint not on disk")
def test_the_real_deployed_checkpoint_loads_both_ways():
    """The actual deliverable, not a fixture. Loads bare and through the helper."""
    path = os.path.join(os.path.dirname(__file__), "..", "gnn_checkpoint_n1.pt")

    bare = torch.load(path, map_location="cpu", weights_only=False)
    assert all(isinstance(v, torch.Tensor) for v in bare.values())

    state, _meta = load_checkpoint(path, map_location="cpu")
    assert set(state) == set(bare)
    for k in bare:
        assert torch.equal(state[k], bare[k])


# ---------------------------------------------------------------------------
# 2. The enriched form, and one load path for both
# ---------------------------------------------------------------------------

def test_the_enriched_sibling_is_self_describing(tmp_path):
    ckpt = tmp_path / "gnn_checkpoint_n1.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt))
    rec.log_epoch(_record(1, 0.70, True))
    rec.log_epoch(_record(2, 0.88, True))
    rec.save_best(_Tiny())
    rec.finish(early_stopped=False)

    state, meta = load_checkpoint(enriched_path_for(ckpt), map_location="cpu")

    _Tiny().load_state_dict(state)
    assert meta["best_epoch"] == 2
    assert meta["best_val_contingency_f1"] == pytest.approx(0.88)
    assert meta["config"]["lr"] == 3e-4
    assert meta["config"]["hidden_channels"] == [16, 32, 32]
    assert meta["seed"] == 42
    assert meta["node_features"] == 8 and meta["edge_features"] == 8
    assert meta["git_commit"] == "abc123"
    assert len(meta["history"]) == 2


def test_the_bare_deliverable_plus_sidecar_reads_as_one_object(tmp_path):
    """The point of the sidecar: the bare file becomes self-describing via the helper."""
    ckpt = tmp_path / "gnn_checkpoint_n1.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt))
    rec.log_epoch(_record(1, 0.91, True))
    rec.save_best(_Tiny())
    rec.finish(early_stopped=False)

    state, meta = load_checkpoint(ckpt, map_location="cpu")

    _Tiny().load_state_dict(state)
    assert meta is not None and meta["best_epoch"] == 1


def test_best_epoch_is_the_saved_epoch_not_the_last_epoch(tmp_path):
    """Best-F1 checkpointing means those differ, and the distinction is the point."""
    ckpt = tmp_path / "c.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt, epochs=5))
    rec.log_epoch(_record(1, 0.50, True))
    rec.log_epoch(_record(2, 0.90, True))
    rec.save_best(_Tiny())
    rec.log_epoch(_record(3, 0.60, False))
    rec.log_epoch(_record(4, 0.55, False))
    rec.finish(early_stopped=False)

    meta = json.loads(meta_path_for(ckpt).read_text(encoding="utf-8"))
    assert meta["best_epoch"] == 2
    assert meta["epochs_run"] == 4
    assert meta["epochs_requested"] == 5


# ---------------------------------------------------------------------------
# 3. History survives an interrupted run
# ---------------------------------------------------------------------------

def test_history_is_written_per_epoch_not_at_the_end(tmp_path):
    ckpt = tmp_path / "c.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt, epochs=30))
    for e in range(1, 8):
        rec.log_epoch(_record(e, 0.5 + e / 100, e == 1))
    # No finish() — the run is killed here.

    hist = load_history(ckpt)
    assert len(hist) == 7
    assert [h["epoch"] for h in hist] == list(range(1, 8))
    assert hist[3]["val_contingency_f1"] == pytest.approx(0.54)


def test_history_from_a_previous_run_is_not_read_as_part_of_this_one(tmp_path):
    ckpt = tmp_path / "c.pt"
    first = RunRecorder(ckpt, _metadata(ckpt))
    for e in range(1, 4):
        first.log_epoch(_record(e, 0.5, False))
    assert len(load_history(ckpt)) == 3

    second = RunRecorder(ckpt, _metadata(ckpt))
    second.log_epoch(_record(1, 0.9, True))
    assert len(load_history(ckpt)) == 1, "a stale history was appended to, not replaced"


def test_every_epoch_record_carries_the_required_fields(tmp_path):
    """epoch, train loss, val_contingency_f1 — the minimum the task needs."""
    ckpt = tmp_path / "c.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt))
    rec.log_epoch(_record(1, 0.8, True))
    h = load_history(ckpt)[0]
    for field in ("epoch", "train_loss", "val_contingency_f1", "lr",
                  "is_best", "seconds", "val_ap"):
        assert field in h, field


# ---------------------------------------------------------------------------
# 4. --report-train becomes a recorded quantity
# ---------------------------------------------------------------------------

def test_the_train_val_gap_is_computed_and_stored(tmp_path):
    """--report-train used to print two numbers and compute nothing."""
    ckpt = tmp_path / "c.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt, report_train=True))
    rec.log_epoch(_record(1, 0.80, True, train_f1=0.95))

    h = load_history(ckpt)[0]
    assert h["train_contingency_f1"] == pytest.approx(0.95)
    assert h["train_val_gap"] == pytest.approx(0.15)


def test_the_gap_is_absent_when_report_train_was_not_passed(tmp_path):
    """No train slice was scored, so no gap — not a zero, which would read as 'no gap'."""
    ckpt = tmp_path / "c.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt))
    rec.log_epoch(_record(1, 0.80, True))

    h = load_history(ckpt)[0]
    assert h["train_contingency_f1"] is None
    assert h["train_val_gap"] is None


# ---------------------------------------------------------------------------
# 5. Ablation runs must not touch the deliverable
# ---------------------------------------------------------------------------

def test_a_headonly_run_writes_only_its_own_files(tmp_path):
    """The existing guard renames the checkpoint; the sidecars must follow it."""
    deliverable = tmp_path / "gnn_checkpoint_n1.pt"
    torch.save(_Tiny().state_dict(), deliverable)
    before = deliverable.read_bytes()

    ablation = tmp_path / "gnn_checkpoint_n1_headonly.pt"
    rec = RunRecorder(ablation, _metadata(ablation, head_only=True))
    rec.log_epoch(_record(1, 0.84, True))
    rec.save_best(_Tiny())
    rec.finish(early_stopped=False)

    assert deliverable.read_bytes() == before, "the ablation overwrote the deliverable"
    assert meta_path_for(ablation).exists()
    assert history_path_for(ablation).exists()
    assert not meta_path_for(deliverable).exists(), \
        "the ablation wrote a sidecar against the deliverable's name"


def test_sidecar_paths_are_derived_from_the_checkpoint_name(tmp_path):
    c = tmp_path / "gnn_checkpoint_n1_headonly.pt"
    assert meta_path_for(c).name == "gnn_checkpoint_n1_headonly_meta.json"
    assert history_path_for(c).name == "gnn_checkpoint_n1_headonly_history.jsonl"
    assert enriched_path_for(c).name == "gnn_checkpoint_n1_headonly_full.pt"


# ---------------------------------------------------------------------------
# 6. describe() is honest about a checkpoint with no record
# ---------------------------------------------------------------------------

def test_describe_says_so_when_there_is_no_metadata(tmp_path):
    ckpt = tmp_path / "legacy.pt"
    torch.save(_Tiny().state_dict(), ckpt)
    text = describe(ckpt)
    assert "NONE" in text
    assert "NOT recoverable" in text


def test_describe_reports_the_epoch_when_there_is(tmp_path):
    ckpt = tmp_path / "c.pt"
    rec = RunRecorder(ckpt, _metadata(ckpt))
    rec.log_epoch(_record(1, 0.4, True))
    rec.log_epoch(_record(2, 0.93, True))
    rec.save_best(_Tiny())
    rec.finish(early_stopped=False)
    text = describe(ckpt)
    assert "best epoch     : 2" in text


# ---------------------------------------------------------------------------
# 7. train_gnn.py still imports, and evaluate() reports what it prints
# ---------------------------------------------------------------------------

def test_train_gnn_imports_clean():
    import training.train_gnn as tg
    assert hasattr(tg, "GridGNN")
    assert hasattr(tg, "compute_normalization_stats")


def test_evaluate_returns_the_numbers_it_prints():
    """They used to go to stdout only, which is why none were ever recorded."""
    import inspect

    import training.train_gnn as tg
    src = inspect.getsource(tg.evaluate)
    for key in ("contingency_f1", "ap", "all_positive_baseline", "rho_rule_baseline"):
        assert f'"{key}"' in src, key
