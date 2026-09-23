"""
fig_shield_threshold_sweep.py — Figure: the shield across the full threshold range.

Reads results/shield_sweep/shield_threshold_analysis.json and the cached scores
it was built from (data/shield_sweep_scores_<tag>.npz), writes
figures/shield_threshold_sweep.{svg,png}. Styling matches
notebooks/thesis_figures.ipynb (same model/shield colours, same rcParams).

x-axis is the share of contingencies the model calls secure at that threshold,
not the raw logit: logit ranges differ per grid (-10..35 on wcci2022), so the
share is the only axis on which the three panels are comparable.

Usage:
    python evaluation/fig_shield_threshold_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(".")
from evaluation.shield_threshold_analysis import (  # noqa: E402
    MIN_BLOCKS_FOR_PRECISION, OUT as ANALYSIS, TAGS, cache_path,
)

FIG_DIR = Path("figures")
NICE = {"neurips2020": "36-bus NeurIPS 2020 (trained on)",
        "case14": "14-bus case14 (unseen)",
        "wcci2022": "118-bus WCCI 2022 (unseen)"}
C_MODEL, C_SHIELD, C_MUTED, C_INK = "#2f4b7c", "#d45087", "#b4b2a9", "#5f5e5a"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
})


def share_secure(logits: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Fraction of contingencies with logit <= t, i.e. predicted secure."""
    s = np.sort(logits)
    return np.searchsorted(s, thresholds, side="right") / len(s)


def main() -> None:
    res = json.loads(ANALYSIS.read_text())
    held = res["held_threshold"]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})

    for j, tag in enumerate(TAGS):
        curve = res["grids"][tag]["sweep"]["curve"]
        logits = np.load(cache_path(tag))["logits"]
        t = np.array([r["threshold"] for r in curve])
        x = share_secure(logits, t)
        f_raw = np.array([r["f1_raw"] for r in curve])
        f_gat = np.array([r["f1_gated"] for r in curve])
        x_held = share_secure(logits, np.array([held]))[0]

        ax = axes[0, j]
        ax.plot(x, f_raw, color=C_MODEL, lw=2, label="model alone")
        ax.plot(x, f_gat, color=C_SHIELD, lw=2, ls=(0, (5, 2)), label="model + shield")
        ax.fill_between(x, f_raw, f_gat, color=C_SHIELD, alpha=0.12, lw=0)
        ax.axvline(x_held, color=C_INK, lw=0.9, ls=":")
        i_h = int(np.argmin(np.abs(t - held)))
        ax.text(x_held - 0.02, 0.04, f"held threshold\nshield +{f_gat[i_h] - f_raw[i_h]:.4f}",
                ha="right", va="bottom", fontsize=7.5, color=C_INK)
        ax.set_ylim(0, 1)
        ax.set_title(NICE[tag], fontsize=9.5)
        if j == 0:
            ax.set_ylabel("F1")
            ax.legend(fontsize=8, loc="upper left")

        ax = axes[1, j]
        ok = np.array([r["blocks"] >= MIN_BLOCKS_FOR_PRECISION for r in curve])
        prec = np.array([r["intervention_precision"] if o else np.nan
                         for r, o in zip(curve, ok)], dtype=float)
        bar = f_raw / 2
        ax.fill_between(x, 0, bar, color=C_MUTED, alpha=0.35, lw=0)
        ax.plot(x, bar, color=C_INK, lw=0.9, ls="--")
        ax.plot(x, prec, color=C_SHIELD, lw=2)
        ax.axvline(x_held, color=C_INK, lw=0.9, ls=":")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of contingencies the model calls secure")
        worst = res["grids"][tag]["sweep"]["min_intervention_precision"]
        ax.text(0.03, 0.62, f"lowest {worst:.3f}", transform=ax.transAxes,
                ha="left", fontsize=7.5, color=C_SHIELD)
        if j == 0:
            ax.set_ylabel("intervention precision")
            ax.text(0.03, 0.03, "shield would lower F1 here\n(precision < F1 / 2)",
                    transform=ax.transAxes, fontsize=7.5, color=C_INK)

    fig.suptitle("The shield never lowers F1, at any threshold, on any grid",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        p = FIG_DIR / f"shield_threshold_sweep.{ext}"
        fig.savefig(p)
        print(f"Wrote {p}")


if __name__ == "__main__":
    if not ANALYSIS.exists() or not all(os.path.exists(cache_path(t)) for t in TAGS):
        sys.exit("Run evaluation/shield_threshold_analysis.py first.")
    main()
