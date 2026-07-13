"""Unit tests for the Round 3 GSAT edge gate + information-bottleneck KL
(training/train_gnn.py). See supplimentary_docs/gsat_lsgat_handoff.md §4.

Run:  python -m pytest tests/test_gsat.py -v
"""
import os
import sys

import pytest
import torch

# Make the repo root importable when pytest is invoked from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_gnn import StochasticEdgeGate, info_bottleneck_kl  # noqa: E402

EDGE_DIM = 4  # rho, p_or, q_or, line_status (EDGE_FEATURES)


@pytest.mark.unit
def test_ib_kl_is_non_negative():
    torch.manual_seed(0)
    gate = torch.rand(128)  # arbitrary gates in (0, 1)
    kl = info_bottleneck_kl(gate, r=0.6)
    assert kl.item() >= -1e-6  # KL divergence is >= 0


@pytest.mark.unit
def test_ib_kl_is_zero_when_gate_equals_prior():
    # KL(Bernoulli(r) || Bernoulli(r)) == 0 exactly.
    for r in (0.5, 0.6, 0.7):
        gate = torch.full((64,), r)
        kl = info_bottleneck_kl(gate, r=r)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_ib_kl_grows_as_gate_leaves_prior():
    r = 0.6
    near = info_bottleneck_kl(torch.full((64,), 0.6), r=r).item()
    far  = info_bottleneck_kl(torch.full((64,), 0.99), r=r).item()
    assert far > near


@pytest.mark.unit
def test_ib_kl_is_differentiable():
    gate = torch.rand(32, requires_grad=True)
    kl = info_bottleneck_kl(gate, r=0.6)
    kl.backward()
    assert gate.grad is not None
    assert torch.isfinite(gate.grad).all()


@pytest.mark.unit
def test_gate_output_in_unit_interval_eval():
    torch.manual_seed(1)
    gate_mod = StochasticEdgeGate(EDGE_DIM, temp=1.0)
    edge_attr = torch.randn(200, EDGE_DIM)
    g = gate_mod(edge_attr, training=False)  # deterministic sigmoid
    assert g.shape == (200,)
    assert (g >= 0).all() and (g <= 1).all()


@pytest.mark.unit
def test_gate_output_in_unit_interval_train():
    torch.manual_seed(2)
    gate_mod = StochasticEdgeGate(EDGE_DIM, temp=1.0)
    edge_attr = torch.randn(200, EDGE_DIM)
    g = gate_mod(edge_attr, training=True)  # Gumbel-softmax relaxation
    assert g.shape == (200,)
    # gumbel_softmax(hard=False) yields a probability in [0, 1].
    assert (g >= 0).all() and (g <= 1).all()


@pytest.mark.unit
def test_gate_is_differentiable_train():
    gate_mod = StochasticEdgeGate(EDGE_DIM, temp=1.0)
    edge_attr = torch.randn(50, EDGE_DIM, requires_grad=True)
    g = gate_mod(edge_attr, training=True)
    g.sum().backward()
    assert edge_attr.grad is not None
    assert torch.isfinite(edge_attr.grad).all()
