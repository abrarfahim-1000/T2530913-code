"""Unit tests for the Round 3 soft-macro-F1 loss (training/train_gnn.py).

Run:  python -m pytest tests/test_soft_f1_loss.py -v
"""
import os
import sys

import pytest
import torch
from sklearn.metrics import f1_score

# Make the repo root importable when pytest is invoked from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_gnn import soft_f1_loss, f1_ramp_factor  # noqa: E402

N_CLASSES = 4


def _one_hot(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(targets, n_classes).float()


def _hard_logits(targets: torch.Tensor, n_classes: int, scale: float = 20.0) -> torch.Tensor:
    """Near-one-hot logits that argmax to `targets` (soft probs ~ hard predictions)."""
    return _one_hot(targets, n_classes) * scale


@pytest.mark.unit
def test_perfect_prediction_is_near_zero():
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    logits = _hard_logits(targets, N_CLASSES)
    loss = soft_f1_loss(logits, targets, N_CLASSES)
    # perfect macro-F1 == 1.0 -> loss == 1 - 1 == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


@pytest.mark.unit
def test_matches_sklearn_macro_f1_on_hard_predictions():
    # A deliberately imperfect prediction set so macro-F1 != 1.
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    preds   = torch.tensor([0, 1, 1, 1, 2, 3, 3, 3])  # a few errors across classes
    logits  = _hard_logits(preds, N_CLASSES)

    loss = soft_f1_loss(logits, targets, N_CLASSES).item()
    sklearn_macro = f1_score(
        targets.numpy(), preds.numpy(),
        average="macro", labels=list(range(N_CLASSES)), zero_division=0,
    )
    # With near-one-hot logits, soft-F1 collapses to hard macro-F1.
    assert loss == pytest.approx(1.0 - sklearn_macro, abs=1e-3)


@pytest.mark.unit
def test_loss_is_bounded_and_differentiable():
    torch.manual_seed(0)
    targets = torch.randint(0, N_CLASSES, (32,))
    logits = torch.randn(32, N_CLASSES, requires_grad=True)

    loss = soft_f1_loss(logits, targets, N_CLASSES)
    assert 0.0 <= loss.item() <= 1.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.unit
def test_worse_predictions_give_higher_loss():
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    good = soft_f1_loss(_hard_logits(targets, N_CLASSES), targets, N_CLASSES).item()

    # Collapse everything onto a single class -> macro-F1 tanks -> loss rises.
    all_zero = torch.zeros_like(targets)
    bad = soft_f1_loss(_hard_logits(all_zero, N_CLASSES), targets, N_CLASSES).item()

    assert bad > good


@pytest.mark.unit
def test_ramp_factor_zero_during_warmup():
    # Before warm-up ends, factor is always 0 regardless of ramp length.
    assert f1_ramp_factor(0, warmup_epochs=3, ramp_epochs=0) == 0.0
    assert f1_ramp_factor(2, warmup_epochs=3, ramp_epochs=3) == 0.0


@pytest.mark.unit
def test_ramp_factor_hard_step_preserves_original_behavior():
    # ramp_epochs<=0 == the original hard on/off step at warm-up boundary.
    assert f1_ramp_factor(3, warmup_epochs=3, ramp_epochs=0) == 1.0
    assert f1_ramp_factor(9, warmup_epochs=3, ramp_epochs=0) == 1.0


@pytest.mark.unit
def test_ramp_factor_linear_and_clamped():
    # Linear 1/3, 2/3, 1 over the first 3 post-warm-up epochs, then clamped at 1.
    assert f1_ramp_factor(3, warmup_epochs=3, ramp_epochs=3) == pytest.approx(1 / 3)
    assert f1_ramp_factor(4, warmup_epochs=3, ramp_epochs=3) == pytest.approx(2 / 3)
    assert f1_ramp_factor(5, warmup_epochs=3, ramp_epochs=3) == pytest.approx(1.0)
    assert f1_ramp_factor(8, warmup_epochs=3, ramp_epochs=3) == pytest.approx(1.0)
