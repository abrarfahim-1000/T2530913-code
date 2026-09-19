"""DC power-flow / LODF contingency screening — the machinery for arm A1.

`supplimentary_docs/results_comparisons.md` §A1 specifies this as the priority
comparator: the incumbent method the discipline actually runs, and a direct test
of the thesis's load-bearing claim that producing an N-1 label needs a power-flow
solve. LODF *is* the linear approximation of that solve, so its score is the
measurement of what the nonlinear AC physics is worth.

This module builds the factors. `eval_lodf_n1.py` scores them against the same
`y` the GNN harness sees.

WHY THE ALGEBRA IS HERE AND NOT `pandapower.pypower.makeLODF`
-------------------------------------------------------------
§A1 q1 settled on pandapower's builders, and they are the right default. Two
requirements the plan did not anticipate rule them out as the *primary* path:

  1. `makePTDF` takes an explicit `slack=` bus and solves a reduced system, so it
     cannot express a topology whose in-service subgraph is DISCONNECTED. 19-32%
     of frames have lines out and some of those outages island the grid.
  2. We need to know WHICH outages island the grid — that is a reportable result
     (§A1 q4), not an implementation detail to be swallowed.

So the factors are computed here from a pseudo-inverse, which is block-diagonal
across islands and therefore handles both. `tests/test_lodf.py` pins the result
against `makePTDF`/`makeLODF` on the intact topology of all three grids, so the
hand-rolled path is verified rather than trusted.

A useful consequence: LODF depends only on flow DIFFERENCES between the endpoints
of the outaged branch, and those are invariant to the slack choice. §A1's warning
to "record which bus you pass as slack" is therefore moot for this arm — there is
no slack to record.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# Grid2Op ships each environment as a pandapower `grid.json`. Nothing here needs
# grid2op itself — only the network, which is static.
GRID2OP_ENV_DIR = os.path.join(os.path.expanduser("~"), "data_grid2op")

ENV_NAMES = {
    "case14": "l2rpn_case14_sandbox",
    "neurips2020": "l2rpn_neurips_2020_track1_small",
    "wcci2022": "l2rpn_wcci_2022",
}

# A branch is treated as a bridge when its self-factor drives the LODF
# denominator to zero: removing it has nowhere to send its flow. Verified against
# an independent graph-connectivity test in `tests/test_lodf.py`.
BRIDGE_TOL = 1e-8

# Relative cutoff for the pseudo-inverse. The nullspace of Bbus has exactly one
# dimension per connected component and must be truncated cleanly; the physical
# singular values sit many orders above this.
PINV_RCOND = 1e-9


@dataclass(frozen=True)
class BranchModel:
    """Static DC model of one grid, indexed in GRID2OP LINE ORDER.

    That ordering is verified, not assumed: grid2op enumerates pandapower's
    `line` table first and its `trafo` table second, and
    `load_branch_model` refuses to return a model whose endpoints disagree with
    the dataset metadata on any branch.
    """

    tag: str
    n_line: int
    n_bus: int
    f_bus: np.ndarray  # origin substation per line
    t_bus: np.ndarray  # extremity substation per line
    b: np.ndarray  # DC susceptance, 1 / (x_pu * tap)
    load_bus: np.ndarray
    gen_bus: np.ndarray


def load_branch_model(tag: str, meta_path: str | None = None) -> BranchModel:
    """Build the DC model for `tag` and validate it against the dataset metadata.

    Raises if the pandapower network and the metadata disagree about any line's
    endpoints. A silent index mismatch is the one failure mode that would produce
    plausible-looking nonsense, so it is a hard error rather than a warning.
    """
    import pandapower as pp

    if tag not in ENV_NAMES:
        raise KeyError(f"unknown tag {tag!r}; expected one of {sorted(ENV_NAMES)}")

    grid_path = os.path.join(GRID2OP_ENV_DIR, ENV_NAMES[tag], "grid.json")
    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"missing {grid_path}\nThe grid2op environment data is needed for the "
            f"branch reactances. Nothing else in this module needs grid2op."
        )
    net = pp.from_json(grid_path)

    if meta_path is None:
        meta_path = os.path.join("data", f"grid_dataset_{tag}_n1_meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    topo = meta["topology"]
    meta_or = np.asarray(topo["line_or_bus"], dtype=int)
    meta_ex = np.asarray(topo["line_ex_bus"], dtype=int)

    # pandapower's own conversion, so tap ratios and transformer impedances are
    # handled by the library rather than restated here.
    pp.rundcpp(net)
    ppc = net._ppc
    branch = ppc["branch"]
    f_bus = branch[:, 0].real.astype(int)
    t_bus = branch[:, 1].real.astype(int)
    x_pu = branch[:, 3].real
    tap = branch[:, 8].real.copy()
    tap[tap == 0.0] = 1.0

    n_line = int(meta["n_line"])
    if len(f_bus) != n_line:
        raise ValueError(
            f"{tag}: pandapower gives {len(f_bus)} branches, metadata says {n_line}"
        )
    mismatch = np.nonzero((f_bus != meta_or) | (t_bus != meta_ex))[0]
    if mismatch.size:
        k = int(mismatch[0])
        raise ValueError(
            f"{tag}: branch ordering disagrees with the dataset metadata at line "
            f"{k}: pandapower says ({f_bus[k]}, {t_bus[k]}), metadata says "
            f"({meta_or[k]}, {meta_ex[k]}). Refusing to build a model whose line "
            f"indices do not match the labels."
        )
    if np.any(x_pu <= 0):
        raise ValueError(f"{tag}: non-positive reactance on {int(np.sum(x_pu <= 0))} branches")
    if np.any(branch[:, 9].real != 0.0):
        raise ValueError(
            f"{tag}: phase-shifting transformers present; the DC model here assumes "
            f"no phase shift and would silently misattribute their flow"
        )

    return BranchModel(
        tag=tag,
        n_line=n_line,
        n_bus=int(meta["n_sub"]),
        f_bus=f_bus,
        t_bus=t_bus,
        b=1.0 / (x_pu * tap),
        load_bus=np.asarray(topo["load_to_sub"], dtype=int),
        gen_bus=np.asarray(topo["gen_to_sub"], dtype=int),
    )


def lodf_for_pattern(model: BranchModel, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Line Outage Distribution Factors for one in-service pattern.

    Returns `(lodf, bridge)`, both indexed over the LIVE branches only, in
    ascending line order — the same order `build_line_targets` uses.

      lodf[m, k]  fraction of branch k's pre-outage flow that lands on branch m
                  when k is removed. `lodf[k, k]` is -1 by convention.
      bridge[k]   True where removing k disconnects the network, which makes the
                  factors for that outage undefined (the denominator vanishes).
                  Those columns are zeroed; callers must branch on `bridge`.
    """
    idx = np.nonzero(live)[0]
    n = idx.size
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0, dtype=bool)

    f, t, b = model.f_bus[idx], model.t_bus[idx], model.b[idx]

    incidence = np.zeros((n, model.n_bus))
    rows = np.arange(n)
    np.add.at(incidence, (rows, f), 1.0)
    np.add.at(incidence, (rows, t), -1.0)

    b_flow = incidence * b[:, None]  # diag(b) @ A
    b_bus = incidence.T @ b_flow
    # Pseudo-inverse: singular by construction (one nullspace dimension per
    # connected component) and block-diagonal across islands, which is exactly
    # the behaviour an outaged topology needs.
    ptdf = b_flow @ np.linalg.pinv(b_bus, rcond=PINV_RCOND, hermitian=True)

    # Transfer factors for a unit injection at each live branch's own endpoints.
    # Slack-invariant, which is why no slack bus appears anywhere in this module.
    d = ptdf[:, f] - ptdf[:, t]
    denom = 1.0 - np.diag(d)
    bridge = np.abs(denom) < BRIDGE_TOL

    safe = np.where(bridge, 1.0, denom)
    lodf = d / safe[None, :]
    lodf[:, bridge] = 0.0
    lodf[rows, rows] = -1.0
    return lodf, bridge


def isolates_supply(model: BranchModel, live: np.ndarray, k: int) -> bool:
    """Would removing live branch `k` cut a load or generator off from the rest?

    The topology pre-screen every real contingency tool runs before any flow
    calculation. It is separated from `lodf_for_pattern` because it answers a
    different question: `bridge` says the linear flow model has nothing to say,
    this says the outage strands supply or demand — which is what ends a Grid2Op
    episode and therefore what produces the game-over label class (§A1 q4).
    """
    remaining = live.copy()
    remaining[k] = False
    comp = _components(model, remaining)
    served = np.concatenate([model.load_bus, model.gen_bus])
    if served.size == 0:
        return False
    # The outage strands supply iff the load/gen buses no longer share one
    # component. A bus with neither load nor generator may be isolated freely.
    return np.unique(comp[served]).size > 1


def _components(model: BranchModel, live: np.ndarray) -> np.ndarray:
    """Connected-component label per bus, over the live branches. Union-find."""
    parent = np.arange(model.n_bus)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in zip(model.f_bus[live], model.t_bus[live]):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    return np.array([find(i) for i in range(model.n_bus)])


class LodfCache:
    """Memoizes the factors per distinct `line_status` pattern.

    §A1 q2 measured 140 / 745 / 402 distinct patterns against 6,000 / 12,000 /
    4,000 frames, so this turns the dominant cost into a rounding error.
    """

    def __init__(self, model: BranchModel):
        self.model = model
        self._store: dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, live: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(lodf, bridge, strands_supply)` for this pattern, over live branches."""
        key = np.packbits(live).tobytes()
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        lodf, bridge = lodf_for_pattern(self.model, live)
        idx = np.nonzero(live)[0]
        strands = np.zeros(idx.size, dtype=bool)
        # Only a bridge can strand anything, so the connectivity walk runs on a
        # handful of branches per pattern rather than all of them.
        for pos in np.nonzero(bridge)[0]:
            strands[pos] = isolates_supply(self.model, live, int(idx[pos]))
        entry = (lodf, bridge, strands)
        self._store[key] = entry
        return entry
