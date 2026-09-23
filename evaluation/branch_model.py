"""Static DC branch model of one grid — reactances, endpoints and bus maps.

This is the half of the retired DC/LODF arm that outlived it. The LODF
comparator (`evaluation/lodf.py`, `evaluation/eval_lodf_n1.py`) was removed on
2026-09-20 and replaced by the inference-cost benchmark in
`evaluation/bench_inference_speed.py`; recover it with
`git show 6546fc2:evaluation/lodf.py`.

What survives is `load_branch_model`, because the reactance experiment
(`evaluation/reactance_transfer.py`, `thesis_findings.md` §27) feeds the model
each grid's branch susceptances and has nothing else to read them from. Only the
branch model is kept — the factor machinery (`lodf_for_pattern`,
`isolates_supply`, `LodfCache`) went with the arm rather than being left behind
as dead code.

Grid2Op ships each environment as a pandapower `grid.json`. Nothing here needs
grid2op itself — only the network, which is static.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

GRID2OP_ENV_DIR = os.path.join(os.path.expanduser("~"), "data_grid2op")

ENV_NAMES = {
    "case14": "l2rpn_case14_sandbox",
    "neurips2020": "l2rpn_neurips_2020_track1_small",
    "wcci2022": "l2rpn_wcci_2022",
}


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
        raise ValueError(
            f"{tag}: non-positive reactance on {int(np.sum(x_pu <= 0))} branches")
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
