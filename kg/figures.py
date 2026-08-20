"""
figures.py — static thesis figures for the knowledge graph.

Two figures, vector SVG plus a 200-dpi PNG, sized for a thesis page:

  1. `kg_provenance`     the whole graph, laid out in its six layers. Small enough
                         to read every node, which was never true of the 587-node
                         first iteration.
  2. `kg_corroboration`  how many clauses, in how many documents, state each
                         predicate. This is the figure that carries the finding:
                         one thermal check, stated ten times, across four
                         documents from two standards bodies on two continents.

No interactive output. The first iteration shipped pyvis HTML that nobody opened.
"""
from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from kg.build import kg_stats, nodes_of  # noqa: E402
from kg.schema import (  # noqa: E402
    CLAUSE,
    DOCUMENT,
    LAYERS,
    PREDICATE,
    RULE,
    SERVED_RULE,
    VARIABLE,
    doc_id,
)

#: SVG for the vector original, PNG for anything that will not embed one.
#: PDF was dropped 2026-08-20 — it duplicated the SVG and nothing consumed it.
EXPORT_FORMATS = ("svg", "png")

#: One colour per layer. Chosen to stay distinguishable in greyscale print.
LAYER_COLOR = {
    DOCUMENT: "#2f4b7c",
    CLAUSE: "#665191",
    RULE: "#a05195",
    SERVED_RULE: "#d45087",
    PREDICATE: "#f95d6a",
    VARIABLE: "#ff7c43",
}


def _save_figure_formats(
    fig,
    output_path: str | Path,
    *,
    export_formats: tuple[str, ...] = EXPORT_FORMATS,
    dpi: int = 200,
) -> list[str]:
    """Save a Matplotlib figure to multiple formats.

    `output_path` may be given with or without an extension; outputs are written
    next to it using its stem. Lifted from the retired `extraction/build_kg.py`,
    which was the one piece of those 1,203 lines worth keeping.
    """
    base = Path(output_path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for fmt in export_formats:
        fmt = str(fmt).lstrip(".").lower().strip()
        if not fmt:
            continue
        out_path = base.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        written.append(str(out_path))
    return written


# ── FIGURE 1 — the graph ──────────────────────────────────────────────────────

def plot_provenance(
    kg: nx.DiGraph,
    output_path: str | Path = "kg/kg_provenance",
    export_formats: tuple[str, ...] = EXPORT_FORMATS,
) -> list[str]:
    """The whole graph in six layers, left to right."""
    layer_of = {t: i for i, t in enumerate(LAYERS)}
    graph = kg.copy()
    for node, attrs in graph.nodes(data=True):
        attrs["layer"] = layer_of.get(attrs.get("type"), 0)

    pos = nx.multipartite_layout(graph, subset_key="layer")
    # Widen horizontally: labels sit to the right of their node, so the columns
    # need room for the longest label in each or the text bleeds into the next
    # layer. 5.2 is what stops Clause text landing on top of the Rule nodes.
    pos = {n: (x * 5.2, y) for n, (x, y) in pos.items()}

    fig, ax = plt.subplots(figsize=(15.5, 6.8))

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color="#b9b9c6", width=0.9,
        arrows=True, arrowsize=8, node_size=340,
    )
    for layer_type in LAYERS:
        members = nodes_of(graph, layer_type)
        if not members:
            continue
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=members, node_size=340,
            node_color=LAYER_COLOR[layer_type], linewidths=0.0, label=layer_type,
        )

    for node, (x, y) in pos.items():
        ax.text(
            x + 0.13, y, _truncate(str(graph.nodes[node].get("label", node)), 26),
            fontsize=6.4, va="center", ha="left", color="#22222c",
        )

    stats = kg_stats(kg)
    for layer_type in LAYERS:
        xs = [pos[n][0] for n in nodes_of(graph, layer_type)]
        if not xs:
            continue
        ax.text(
            xs[0], 1.015, f"{layer_type} ({stats['by_type'][layer_type]})",
            fontsize=9, ha="left", va="bottom", color=LAYER_COLOR[layer_type],
            fontweight="bold", transform=ax.get_xaxis_transform(),
        )

    # suptitle, not set_title: the layer headers occupy the axes' own top strip.
    fig.suptitle(
        f"Rule provenance — {stats['nodes']} nodes, {stats['edges']} edges. "
        f"No topology: one graph serves all three grids.",
        fontsize=11, y=0.99,
    )
    ax.margins(x=0.10, y=0.05)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    written = _save_figure_formats(fig, output_path, export_formats=export_formats)
    plt.close(fig)
    return written


# ── FIGURE 2 — corroboration ──────────────────────────────────────────────────

def plot_corroboration(
    kg: nx.DiGraph,
    output_path: str | Path = "kg/kg_corroboration",
    export_formats: tuple[str, ...] = EXPORT_FORMATS,
) -> list[str]:
    """Clauses per predicate, segmented by source document."""
    rows = []
    for pid in sorted(nodes_of(kg, PREDICATE)):
        per_doc: Counter = Counter()
        for snode in kg.predecessors(pid):
            for rnode in kg.predecessors(snode):
                if kg.nodes[rnode].get("type") != RULE:
                    continue
                for cnode in kg.predecessors(rnode):
                    if kg.nodes[cnode].get("type") == CLAUSE:
                        per_doc[kg.nodes[cnode]["document"]] += 1
        rows.append((kg.nodes[pid], per_doc))

    documents = sorted({d for _, c in rows for d in c})
    palette = plt.get_cmap("tab20")
    doc_color = {d: palette(i % 20) for i, d in enumerate(documents)}

    fig, ax = plt.subplots(figsize=(11.0, 1.5 + 1.1 * len(rows)))

    for i, (attrs, per_doc) in enumerate(rows):
        left = 0
        for doc in documents:
            n = per_doc.get(doc, 0)
            if not n:
                continue
            body = kg.nodes[doc_id(doc)].get("issuing_body")
            ax.barh(i, n, left=left, height=0.42, color=doc_color[doc],
                    edgecolor="white", linewidth=0.8,
                    label=f"{_short_doc(doc)}" + (f" ({body})" if body else ""))
            if n >= 2:
                ax.text(left + n / 2, i, str(n), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += n
        ax.text(
            left + 0.18, i,
            f"{left} clause(s), {len(per_doc)} document(s)",
            va="center", fontsize=8.5, color="#44444e",
        )

    ax.set_yticks(range(len(rows)))
    # Wrap rather than truncate: cutting "is at least 0.9 and ..." mid-clause
    # misstates the rule, which is the one thing a provenance figure must not do.
    ax.set_yticklabels(
        [textwrap.fill(a["plain_english"], 46)
         + f"\n({textwrap.fill(a['canonical'], 46)}) — {a['role']}"
         for a, _ in rows],
        fontsize=7.6,
    )
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("clauses stating this predicate", fontsize=9)
    ax.set_title(
        "Independent corroboration per predicate\n"
        "the same physical check, restated across separate standards",
        fontsize=10.5,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.margins(x=0.20)

    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    # Below the axes: inside the plot it lands on the per-row annotations.
    ax.legend(seen.values(), seen.keys(), fontsize=7.2, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22 / len(rows) - 0.10))

    fig.tight_layout()
    written = _save_figure_formats(fig, output_path, export_formats=export_formats)
    plt.close(fig)
    return written


def _truncate(text: str, limit: int) -> str:
    """Length-only. Unlike `build._short` this keeps underscores: node labels are
    identifiers (`R_1717`, `loading_pct`) and stripping them corrupts the name."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short_doc(stem: str, limit: int = 34) -> str:
    """Document stems are filenames, so underscores here ARE word separators."""
    return _truncate(stem.replace("_", " "), limit)


def write_all(
    kg: nx.DiGraph,
    out_dir: str | Path = "kg",
    export_formats: tuple[str, ...] = EXPORT_FORMATS,
) -> list[str]:
    out_dir = Path(out_dir)
    return (
        plot_provenance(kg, out_dir / "kg_provenance", export_formats)
        + plot_corroboration(kg, out_dir / "kg_corroboration", export_formats)
    )
