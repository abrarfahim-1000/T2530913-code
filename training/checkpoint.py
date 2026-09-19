"""
checkpoint.py — make a checkpoint on disk say what produced it.

THE PROBLEM THIS CLOSES
-----------------------
`train_gnn.py` saved a bare `state_dict` and wrote no metrics file. The
consequences are recorded in CLAUDE.md and `study.md` §7: the deployed
`gnn_checkpoint_n1.pt`'s **epoch count is not recoverable from disk**, no
per-epoch history for the N-1 model exists anywhere in the repo, and the four
`training/*.log` files that do exist are classify-era (2026-06-24/25, 4-class
`val_macro_f1`, 5/4 features) and describe a retired task.

The past is gone. This module closes the going-forward half.

THE BACKWARD-COMPATIBILITY CONSTRAINT, AND WHY THE DELIVERABLE STAYS BARE
------------------------------------------------------------------------
Five call sites load the checkpoint as

    model.load_state_dict(torch.load(path, map_location=DEVICE))

`load_state_dict` is strict by default, so ANY non-tensor key added to the
saved mapping — an epoch number, a config block — raises on every one of them.
There is no format that is simultaneously an enriched dict and a drop-in for
that call. So:

  * `gnn_checkpoint_n1.pt` (and every `*_headonly.pt` / seed variant) stays a
    **bare state_dict**. Byte-compatible with every existing loader, which is
    not edited. `tests/test_train_metrics.py` pins that.

  * beside it go three artifacts that carry everything the bare file cannot:

        <stem>_meta.json      the run: config actually used, the epoch the
                              saved weights came from, git commit, timestamp,
                              device, seed, feature counts, and the full
                              per-epoch history.
        <stem>_history.jsonl  one line per epoch, flushed as the epoch ends,
                              so a run killed at epoch 7 still leaves 7 epochs.
        <stem>_full.pt        the enriched checkpoint — the same weights under
                              a "state_dict" key, with the metadata beside it.
                              Self-describing on its own, and NOT loadable by
                              a bare `load_state_dict` call, which is exactly
                              why it is a separate file.

`load_checkpoint()` below accepts either form. Handed the bare deliverable it
picks up `<stem>_meta.json` if it is there, so the pair reads as one
self-describing object through a single call.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

CHECKPOINT_FORMAT_VERSION = 1


def meta_path_for(checkpoint_file: str | os.PathLike) -> Path:
    return Path(str(checkpoint_file)[:-3] + "_meta.json")


def history_path_for(checkpoint_file: str | os.PathLike) -> Path:
    return Path(str(checkpoint_file)[:-3] + "_history.jsonl")


def enriched_path_for(checkpoint_file: str | os.PathLike) -> Path:
    return Path(str(checkpoint_file)[:-3] + "_full.pt")


def git_commit() -> Optional[str]:
    """Current HEAD, or None outside a work tree / without git on PATH.

    Never raises: a missing commit is a gap in the record, not a reason to lose
    a finished training run.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_is_dirty() -> Optional[bool]:
    """Whether the work tree had uncommitted changes. None when undeterminable.

    A commit hash beside a dirty tree does not identify the code that ran, and
    silently implying it does is worse than saying so.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


@dataclass
class EpochRecord:
    """One epoch. `train_*` stay None unless --report-train was passed."""
    epoch: int                      # 1-based, as printed
    train_loss: float
    val_contingency_f1: float
    lr: float
    is_best: bool
    seconds: float
    val_ap: Optional[float] = None
    val_f1_at_half: Optional[float] = None
    val_all_positive_baseline: Optional[float] = None
    val_rho_rule_baseline: Optional[float] = None
    train_contingency_f1: Optional[float] = None
    # val minus train. --report-train used to print both numbers to stdout and
    # compute nothing; the gap is the quantity that separates memorisation from
    # an optimiser that never fitted the signal, so it is recorded, not left to
    # the reader's arithmetic on a lost terminal.
    train_val_gap: Optional[float] = None


@dataclass
class RunMetadata:
    """Everything needed to say what produced a set of weights."""
    format_version: int
    task: str
    checkpoint_file: str
    # The epoch the SAVED weights came from — not the last epoch run. Best-F1
    # checkpointing means those differ whenever the run kept going after its
    # best, which is the normal case.
    best_epoch: Optional[int]
    best_val_contingency_f1: Optional[float]
    epochs_requested: int
    epochs_run: int
    early_stopped: bool
    config: Dict[str, Any]
    node_features: int
    edge_features: int
    seed: int
    device: str
    torch_version: str
    python_version: str
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    started_utc: str
    finished_utc: Optional[str] = None
    argv: List[str] = field(default_factory=list)
    data_file: Optional[str] = None
    n_train_frames: Optional[int] = None
    n_val_frames: Optional[int] = None
    train_violation_rate: Optional[float] = None
    pos_weight: Optional[float] = None
    head_only: bool = False
    report_train: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)


class RunRecorder:
    """Collects per-epoch metrics and writes them where they survive the run.

    The history file is appended and flushed at the end of every epoch, so an
    interrupted run still leaves a usable record. The meta file and the
    enriched checkpoint are rewritten whenever a new best appears and again at
    the end, so the metadata on disk always describes the weights on disk
    rather than an epoch that has since been superseded.
    """

    def __init__(self, checkpoint_file: str | os.PathLike, metadata: RunMetadata):
        self.checkpoint_file = str(checkpoint_file)
        self.meta = metadata
        self.records: List[EpochRecord] = []
        self.history_file = history_path_for(checkpoint_file)
        self.meta_file = meta_path_for(checkpoint_file)
        self.enriched_file = enriched_path_for(checkpoint_file)
        # Truncate: a history from a previous run of the same checkpoint name
        # would otherwise be read as part of this one.
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text("", encoding="utf-8")

    def log_epoch(self, record: EpochRecord) -> None:
        if record.train_contingency_f1 is not None and record.train_val_gap is None:
            record.train_val_gap = record.train_contingency_f1 - record.val_contingency_f1
        self.records.append(record)
        with self.history_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

    def _refresh_meta(self) -> None:
        best = max((r for r in self.records if r.is_best),
                   key=lambda r: r.epoch, default=None)
        self.meta.best_epoch = best.epoch if best else None
        self.meta.best_val_contingency_f1 = best.val_contingency_f1 if best else None
        self.meta.epochs_run = len(self.records)
        self.meta.history = [asdict(r) for r in self.records]

    def save_best(self, model: torch.nn.Module) -> None:
        """Write the bare deliverable AND the enriched sibling, in that order.

        The bare file is written first and on its own: if the enriched write
        fails for any reason, what is on disk is still exactly what every
        existing loader expects.
        """
        torch.save(model.state_dict(), self.checkpoint_file)
        self._refresh_meta()
        self.meta_file.write_text(
            json.dumps(asdict(self.meta), indent=2), encoding="utf-8")
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "state_dict": model.state_dict(),
                "metadata": asdict(self.meta),
            },
            self.enriched_file,
        )

    def finish(self, early_stopped: bool) -> None:
        self.meta.early_stopped = early_stopped
        self.meta.finished_utc = datetime.now(timezone.utc).isoformat()
        self._refresh_meta()
        self.meta_file.write_text(
            json.dumps(asdict(self.meta), indent=2), encoding="utf-8")


def _looks_enriched(obj: Any) -> bool:
    return isinstance(obj, dict) and "state_dict" in obj and isinstance(
        obj.get("state_dict"), dict)


def load_checkpoint(
    path: str | os.PathLike,
    map_location: Any = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load either checkpoint form. Returns `(state_dict, metadata_or_None)`.

    Accepts:
      * the **bare** form — a plain `{param_name: tensor}` mapping, which is
        what `gnn_checkpoint_n1.pt` and every seed variant on disk are. If a
        `<stem>_meta.json` sits beside it, that metadata comes back too, so the
        deployed pair is self-describing through this one call.
      * the **enriched** form — `{"format_version", "state_dict", "metadata"}`.

    The returned `state_dict` goes straight into `model.load_state_dict()` in
    both cases, so a caller never has to know which form it was handed.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)

    if _looks_enriched(obj):
        return obj["state_dict"], obj.get("metadata")

    # Bare. Pick up the sidecar if the run that wrote it left one.
    meta = None
    sidecar = meta_path_for(path)
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = None
    return obj, meta


def load_history(checkpoint_file: str | os.PathLike) -> List[Dict[str, Any]]:
    """Per-epoch records for a checkpoint, newest run only. Empty if none."""
    path = history_path_for(checkpoint_file)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def describe(path: str | os.PathLike) -> str:
    """One-screen summary of what produced a checkpoint. `python -m training.checkpoint <path>`."""
    state, meta = load_checkpoint(path, map_location="cpu")
    lines = [f"{path}", f"  tensors: {len(state)}"]
    if meta is None:
        lines += [
            "  metadata: NONE — this is a bare checkpoint with no sidecar.",
            "  The epoch count and config that produced it are NOT recoverable",
            "  from disk. That is the pre-2026-09 state; see training/checkpoint.py.",
        ]
        return "\n".join(lines)
    lines += [
        f"  best epoch     : {meta.get('best_epoch')} of {meta.get('epochs_run')} run "
        f"({meta.get('epochs_requested')} requested)",
        f"  best val F1    : {meta.get('best_val_contingency_f1')}",
        f"  early stopped  : {meta.get('early_stopped')}",
        f"  seed / device  : {meta.get('seed')} / {meta.get('device')}",
        f"  features       : {meta.get('node_features')} node / {meta.get('edge_features')} edge",
        f"  git            : {meta.get('git_commit')}"
        f"{' (DIRTY)' if meta.get('git_dirty') else ''}",
        f"  started        : {meta.get('started_utc')}",
        f"  config         : {json.dumps(meta.get('config'))}",
        f"  history        : {len(meta.get('history') or [])} epochs recorded",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["gnn_checkpoint_n1.pt"]:
        print(describe(arg))
        print()
