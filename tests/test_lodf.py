"""Validation for the DC/LODF comparator (results_comparisons.md §A1).

The arm's whole value depends on the factors being right. A wrong branch index or
a wrong reactance produces numbers that look entirely plausible and are entirely
meaningless, so the hand-rolled algebra in `evaluation/lodf.py` is pinned against
pandapower's own `makePTDF`/`makeLODF` on every grid.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(".")

from evaluation.lodf import (ENV_NAMES, GRID2OP_ENV_DIR, LodfCache,
                             _components, load_branch_model, lodf_for_pattern)

TAGS = sorted(ENV_NAMES)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(GRID2OP_ENV_DIR),
    reason="grid2op environment data not present on this machine",
)


@pytest.fixture(scope="module")
def models():
    return {tag: load_branch_model(tag) for tag in TAGS}


@pytest.mark.parametrize("tag", TAGS)
def test_branch_model_matches_dataset_metadata(models, tag):
    """load_branch_model validates endpoints internally; assert it stayed strict."""
    m = models[tag]
    assert m.n_line == len(m.f_bus) == len(m.t_bus) == len(m.b)
    assert m.f_bus.max() < m.n_bus and m.t_bus.max() < m.n_bus
    assert np.all(m.b > 0)


@pytest.mark.parametrize("tag", TAGS)
def test_matches_pandapower_on_intact_topology(models, tag):
    """The reference check: our factors vs pandapower's, intact grid.

    pandapower solves a slack-reduced system; we use a pseudo-inverse. They must
    agree wherever pandapower is defined, which is every non-bridge column.
    """
    import pandapower as pp
    from pandapower.pypower.makeLODF import makeLODF
    from pandapower.pypower.makePTDF import makePTDF

    net = pp.from_json(os.path.join(GRID2OP_ENV_DIR, ENV_NAMES[tag], "grid.json"))
    pp.rundcpp(net)
    ppc = net._ppc
    ref_ptdf = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"], slack=0)
    ref = np.asarray(makeLODF(ppc["branch"], ref_ptdf))

    model = models[tag]
    live = np.ones(model.n_line, dtype=bool)
    ours, bridge = lodf_for_pattern(model, live)

    finite = np.isfinite(ref)
    compare = finite & ~bridge[None, :]
    assert compare.any(), "nothing to compare against"
    np.testing.assert_allclose(ours[compare], ref[compare], atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("tag", TAGS)
def test_self_factor_is_minus_one(models, tag):
    model = models[tag]
    live = np.ones(model.n_line, dtype=bool)
    ours, _ = lodf_for_pattern(model, live)
    np.testing.assert_allclose(np.diag(ours), -1.0, atol=1e-10)


@pytest.mark.parametrize("tag", TAGS)
def test_bridges_agree_with_graph_connectivity(models, tag):
    """The vanishing LODF denominator must mean exactly what it claims to mean.

    Checked against an independent union-find on several outaged topologies, not
    only the intact one — the outaged case is where it matters.
    """
    model = models[tag]
    rng = np.random.default_rng(0)
    patterns = [np.ones(model.n_line, dtype=bool)]
    for _ in range(6):
        live = np.ones(model.n_line, dtype=bool)
        live[rng.choice(model.n_line, size=3, replace=False)] = False
        patterns.append(live)

    for live in patterns:
        _, bridge = lodf_for_pattern(model, live)
        idx = np.nonzero(live)[0]
        base_components = np.unique(_components(model, live)).size
        for pos, k in enumerate(idx):
            cut = live.copy()
            cut[k] = False
            disconnects = np.unique(_components(model, cut)).size > base_components
            assert bool(bridge[pos]) == disconnects, (
                f"{tag}: line {k} bridge={bridge[pos]} but connectivity says "
                f"{disconnects}"
            )


@pytest.mark.parametrize("tag", TAGS)
def test_flow_is_conserved_under_redistribution(models, tag):
    """Physical sanity: redistributed flow still obeys Kirchhoff at every bus.

    Catches a transposed or mis-signed incidence matrix, which the pandapower
    comparison alone could miss if both were built from the same wrong ordering.
    """
    model = models[tag]
    live = np.ones(model.n_line, dtype=bool)
    lodf, bridge = lodf_for_pattern(model, live)
    rng = np.random.default_rng(1)

    incidence = np.zeros((model.n_line, model.n_bus))
    rows = np.arange(model.n_line)
    np.add.at(incidence, (rows, model.f_bus), 1.0)
    np.add.at(incidence, (rows, model.t_bus), -1.0)

    # A base flow vector that satisfies DC power flow for some injection.
    angles = rng.normal(size=model.n_bus)
    base = model.b * (incidence @ angles)
    injection = incidence.T @ base

    for k in np.nonzero(~bridge)[0]:
        post = base + lodf[:, k] * base[k]
        post[k] = 0.0
        np.testing.assert_allclose(incidence.T @ post, injection, atol=1e-6)


@pytest.mark.parametrize("tag", TAGS)
def test_cache_returns_identical_factors(models, tag):
    model = models[tag]
    cache = LodfCache(model)
    live = np.ones(model.n_line, dtype=bool)
    live[0] = False
    first = cache.get(live)
    second = cache.get(live.copy())
    assert cache.hits == 1 and cache.misses == 1
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
