"""Guards for the static DC branch model.

`evaluation/branch_model.py` is what outlived the retired DC/LODF arm. It is now
read by exactly one consumer — `evaluation/reactance_transfer.py`, which feeds the
model each grid's branch susceptances (`thesis_findings.md` §27). A single
consumer is precisely when a module stops being tested, so the checks that
mattered are kept here rather than deleted with the arm.

The one failure mode worth guarding: pandapower's branch table and the dataset
metadata could disagree about a line's endpoints. A silent index mismatch there
produces reactances attached to the wrong lines — plausible-looking nonsense that
nothing downstream would catch.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.append(".")

from evaluation.branch_model import ENV_NAMES, GRID2OP_ENV_DIR, load_branch_model

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
def test_susceptances_are_usable_as_a_feature(models, tag):
    """reactance_transfer.py computes log1p(b / median(b)); that must be finite
    and non-degenerate, or the experiment silently trains on a constant."""
    b = models[tag].b
    assert np.all(np.isfinite(b))
    feat = np.log1p(b / np.median(b))
    assert np.all(np.isfinite(feat))
    assert feat.std() > 0, "susceptance feature is constant — nothing to learn from"


def test_unknown_tag_is_rejected():
    with pytest.raises(KeyError, match="unknown tag"):
        load_branch_model("not_a_grid")


def test_endpoint_disagreement_is_a_hard_error(tmp_path):
    """The one mismatch that would produce plausible nonsense rather than a crash:
    reactances bound to the wrong line indices."""
    tag = "case14"
    real = os.path.join("data", f"grid_dataset_{tag}_n1_meta.json")
    if not os.path.exists(real):
        pytest.skip(f"{real} not generated on this machine")

    meta = json.loads(open(real).read())
    meta["topology"]["line_or_bus"] = list(
        reversed(meta["topology"]["line_or_bus"]))
    bad = tmp_path / "bad_meta.json"
    bad.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="branch ordering disagrees"):
        load_branch_model(tag, str(bad))
