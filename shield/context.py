"""
context.py — the binding voltage contract (component_d_plan.md §5).

Turns one raw telemetry frame into the exact namespace a rule condition is
evaluated against. This is the single definition of "what a context is"; the
polarity guard imports it too, so guard and shield can never disagree about
what a healthy grid looks like.

Two invariants earn their keep here:

1. **Per-line base kV, never a flat nominal.** `study3(integration).md` §4.5
   prescribes `v_or / 150.0`. Measured reality: case14 runs lines at ~20 kV and
   ~138 kV, and even the 36-bus training grid has 7 lines at ~365 kV. A flat
   divisor reads those as 0.13 pu / 2.4 pu and blocks every healthy frame.
2. **Energized lines only.** A tripped line reports `v_or = 0`, so an unmasked
   `min(v_or)` fires every undervoltage rule on any frame with a disconnection.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

# The six variables a rule condition may reference (extraction.common.CONDITION_VOCABULARY).
# Duplicated as a plain tuple so shield/ imports nothing from extraction/ at inference time.
CONTEXT_VARIABLES = (
    "voltage_pu_min",
    "voltage_pu_max",
    "loading_pct",
    "rho_max",
    "n_tripped_lines",
    "any_line_tripped",
    "active_power_mw_max",
    "reactive_power_mvar_max",
    "apparent_power_mva_max",
    "power_factor_at_max_load",
    "current_a_max",
    "total_generation_mw",
    "total_load_mw",
    "generation_load_imbalance_pct",
)

SQRT3 = np.sqrt(3.0)


def load_base_kv(tag: str, data_dir: str = "data") -> np.ndarray:
    """Read the per-line base-kV sidecar written by scripts/dump_base_kv.py."""
    path = os.path.join(data_dir, f"grid_dataset_{tag}_basekv.json")
    with open(path) as f:
        sidecar = json.load(f)
    return np.asarray(sidecar["base_kv_or"], dtype=float)


def _field(source: Any, name: str) -> Any:
    """Read `name` from a JSONL record (dict) or a live Grid2Op observation.

    Key names match across both by construction — `rho`, `v_or`, `line_status`.
    """
    if isinstance(source, dict):
        return source[name]
    return getattr(source, name)


def _has(source: Any, name: str) -> bool:
    """Whether the frame carries `name`. Optional fields (p_or, gen_p, ...) are
    absent from hand-written test fixtures and from older dataset generations;
    the variables derived from them are then simply omitted, which makes rules
    referencing them NOT_EVALUABLE rather than wrong."""
    return name in source if isinstance(source, dict) else hasattr(source, name)


def build_context(
    record: Any,
    base_kv: np.ndarray,
    *,
    fault_type: Optional[str] = None,
    confidence: Optional[float] = None,
) -> dict:
    """Build the rule-evaluation namespace from one raw frame.

    `record` is a dataset record dict or a Grid2Op observation. Values are RAW
    physical quantities — never the z-scored tensors the GNN consumes.

    Blackout edge case: when no line is energized the per-unit voltages are
    undefined, so `voltage_pu_min` / `voltage_pu_max` are **omitted from the
    returned dict** rather than defaulted. A voltage rule then raises NameError
    and the evaluator classifies it NOT_EVALUABLE, which never blocks. Defaulting
    them to 0.0 would instead read as a catastrophic undervoltage and block a
    frame the shield has no evidence about.
    """
    rho = np.asarray(_field(record, "rho"), dtype=float)
    v_or = np.asarray(_field(record, "v_or"), dtype=float)
    status = np.asarray(_field(record, "line_status"), dtype=bool)

    if base_kv.shape[0] != v_or.shape[0]:
        raise ValueError(
            f"base_kv has {base_kv.shape[0]} lines but the frame has {v_or.shape[0]} "
            f"— wrong sidecar for this topology?"
        )

    # rho_max stays UNMASKED to match get_state_label()'s definition of the labels
    # this is scored against (scripts/generate_dataset.py). Tripped lines report
    # rho = 0, so they cannot inflate the maximum anyway.
    rho_max = float(rho.max()) if rho.size else 0.0

    context: dict = {
        "loading_pct": rho_max * 100.0,
        "rho_max": rho_max,
        "n_tripped_lines": int((~status).sum()),
        "any_line_tripped": bool(not status.all()),
    }

    # ── power flow, over energized lines only ─────────────────────────────────
    # Grid2Op exposes a_or (amperes) natively, but our JSONL predates that field,
    # so current is reconstructed as I = S*1000 / (sqrt(3) * V_kV). Same quantity,
    # one extra assumption (three-phase, line-to-line kV).
    p_or = np.asarray(_field(record, "p_or"), dtype=float) if _has(record, "p_or") else None
    q_or = np.asarray(_field(record, "q_or"), dtype=float) if _has(record, "q_or") else None

    if p_or is not None and q_or is not None and status.any():
        p, q = np.abs(p_or[status]), np.abs(q_or[status])
        s = np.hypot(p, q)
        context["active_power_mw_max"] = float(p.max())
        context["reactive_power_mvar_max"] = float(q.max())
        context["apparent_power_mva_max"] = float(s.max())

        # Power factor is taken on the MOST LOADED line, not as a minimum across
        # lines. Measured on case14 normal frames, the per-line minimum has median
        # 0.000 - lightly loaded lines carry near-pure reactive flow, so the min is
        # dominated by lines that are electrically irrelevant. At the most loaded
        # line it is stable (median 0.930, p5-p95 0.926-0.936), and that is the line
        # a power-factor requirement is actually about.
        if s.max() > 1e-6:
            context["power_factor_at_max_load"] = float(p[s.argmax()] / s[s.argmax()])

        v_kv = v_or[status]
        live = v_kv > 1e-6
        if live.any():
            context["current_a_max"] = float((s[live] * 1000.0 / (SQRT3 * v_kv[live])).max())

    # ── system balance ────────────────────────────────────────────────────────
    if _has(record, "gen_p"):
        context["total_generation_mw"] = float(np.asarray(_field(record, "gen_p"), float).sum())
    if _has(record, "load_p"):
        context["total_load_mw"] = float(np.asarray(_field(record, "load_p"), float).sum())
    gen, load = context.get("total_generation_mw"), context.get("total_load_mw")
    if gen is not None and load is not None and abs(load) > 1e-6:
        context["generation_load_imbalance_pct"] = (gen - load) / load * 100.0

    if status.any():
        v_pu = v_or[status] / base_kv[status]
        context["voltage_pu_min"] = float(v_pu.min())
        context["voltage_pu_max"] = float(v_pu.max())

    if fault_type is not None:
        context["fault_type"] = fault_type
    if confidence is not None:
        context["confidence"] = float(confidence)

    return context
