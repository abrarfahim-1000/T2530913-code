"""
figures.py — static thesis figures for the knowledge graph.

Two figures, vector SVG plus a 200-dpi PNG, sized for a thesis page:

  1. `kg_provenance`     the whole graph, laid out in its seven layers. It no
                         longer labels every node: the graph grew from 36 nodes
                         to 171 when shield v2's four-channel corpus came in as
                         an `ExplanatoryRule` layer, and 58 labels in a column do
                         not fit on a page. The dense layers are drawn unlabelled
                         and SAID to be unlabelled, on the figure itself.
  2. `kg_corroboration`  how many clauses, in how many documents, state each
                         predicate. This is the figure that carries the finding:
                         one thermal check, stated ten times, across four
                         documents from two standards bodies on two continents —
                         and now also the 24 predicates that nothing corroborates.

No interactive output. The first iteration shipped pyvis HTML that nobody opened.

⚠ Layout is computed here, not by `nx.multipartite_layout`. That helper orders
the nodes inside a layer through a hash-backed collection, so every rebuild
reshuffled the columns vertically and turned `git diff` on kg/*.svg into noise;
`scripts/build_kg.py` re-execs with PYTHONHASHSEED=0 to suppress it. The layout
below sorts each layer on an explicit key, so it is deterministic with or without
that re-exec — belt and braces, and the sort keys also do real work (clauses group
under their document, explanatory rules group by channel).
"""
from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from kg.build import _clauses_behind, kg_stats, nodes_of  # noqa: E402
from kg.schema import (  # noqa: E402
    CLAUSE,
    DOCUMENT,
    EXPLANATORY_RULE,
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

#: Matplotlib derives the SVG's internal clip-path/glyph ids from this salt, and
#: from `id()` values when it is unset — so two builds of the IDENTICAL figure
#: produced different SVG bytes. The PNGs were always stable; this is what makes
#: `git diff kg/*.svg` mean something. Paired with `Date: None` below.
matplotlib.rcParams["svg.hashsalt"] = "kg-provenance"

#: One colour per layer, walking a single hue ramp left to right so the columns
#: stay ordered — and therefore distinguishable — in greyscale print.
LAYER_COLOR = {
    DOCUMENT: "#2f4b7c",
    CLAUSE: "#665191",
    RULE: "#a05195",
    SERVED_RULE: "#d45087",
    EXPLANATORY_RULE: "#f95d6a",
    PREDICATE: "#ff7c43",
    VARIABLE: "#ffa600",
}

#: Channel order for the ExplanatoryRule column, most consequential first. BLOCK
#: is the only channel that can veto (shield/channels.py); the rest explain.
CHANNEL_ORDER = ("BLOCK", "WARN", "NORMAL", "NOT_APPLICABLE", "INERT")

#: Layers whose nodes are drawn without labels because they no longer fit. Every
#: one of these is named on the figure with its count, so nothing is dropped
#: silently — see `_omitted_note`.
UNLABELLED_LAYERS = (CLAUSE, RULE, EXPLANATORY_RULE, PREDICATE)

#: Column centres, in data units. NOT evenly spaced: the gap to the right of
#: ExplanatoryRule carries the channel brackets, and the gap to the left of
#: Clause carries the document labels, so both are widened deliberately.
COLUMN_X = {
    DOCUMENT: 0.00,
    CLAUSE: 1.05,
    RULE: 2.00,
    SERVED_RULE: 2.85,
    EXPLANATORY_RULE: 3.70,
    PREDICATE: 5.55,
    VARIABLE: 6.45,
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
        elif fmt in ("svg", "svgz", "pdf"):
            # Date=None drops the <dc:date> stamp, which otherwise changes every
            # build and is the only reason a rebuilt-but-unchanged figure shows
            # up in `git status`.
            fig.savefig(out_path, bbox_inches="tight", metadata={"Date": None})
        else:
            fig.savefig(out_path, bbox_inches="tight")
        written.append(str(out_path))
    return written


# ── FIGURE 1 — the graph ──────────────────────────────────────────────────────

def _sort_key(graph: nx.DiGraph, node: str) -> tuple:
    """Deterministic vertical order within a layer, chosen to make the edges
    readable: clauses sit under their own document, explanatory rules sit in
    their channel group, everything else falls back to its id."""
    attrs = graph.nodes[node]
    ntype = attrs.get("type")
    if ntype == CLAUSE:
        return (str(attrs.get("document", "")), str(node))
    if ntype == EXPLANATORY_RULE:
        channel = str(attrs.get("channel", ""))
        rank = CHANNEL_ORDER.index(channel) if channel in CHANNEL_ORDER else len(CHANNEL_ORDER)
        return (f"{rank:02d}", str(node))
    if ntype == DOCUMENT:
        return (str(attrs.get("stem", node)).lower(), str(node))
    return (str(node),)


def _layered_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """One column per layer, each column spread over the full height."""
    pos: dict[str, tuple[float, float]] = {}
    for column, layer_type in enumerate(LAYERS):
        members = sorted(nodes_of(graph, layer_type), key=lambda n: _sort_key(graph, n))
        n = len(members)
        x = COLUMN_X.get(layer_type, float(column))
        for i, node in enumerate(members):
            y = 0.5 if n == 1 else 1.0 - i / (n - 1)
            pos[node] = (x, y)
    # Any node whose type is not a known layer would otherwise vanish; park it in
    # column 0 rather than raise inside a figure.
    for node in graph.nodes:
        pos.setdefault(node, (0.0, 0.5))
    return pos


def _omitted_note(stats: dict) -> str:
    parts = [f"{t} ({stats['by_type'].get(t, 0)})" for t in UNLABELLED_LAYERS
             if stats["by_type"].get(t)]
    return (
        "Node labels are drawn for " + ", ".join(
            f"{t} ({stats['by_type'].get(t, 0)})"
            for t in (DOCUMENT, SERVED_RULE, VARIABLE) if stats["by_type"].get(t)
        )
        + ". No node is omitted from the figure, but the names in "
        + ", ".join(parts)
        + " are not drawn — they do not fit at this size.\n"
        "Every predicate is named in kg_corroboration; every clause and rule name is in "
        "kg/knowledge_graph.json and in the console output of scripts/build_kg.py.\n"
        "Edges are undirected in appearance only: the graph is strictly layered, so every "
        "edge runs left to right."
    )


def plot_provenance(
    kg: nx.DiGraph,
    output_path: str | Path = "kg/kg_provenance",
    export_formats: tuple[str, ...] = EXPORT_FORMATS,
) -> list[str]:
    """The whole graph in its layers, left to right.

    Rewritten 2026-09-17 for the 171-node graph. The old version labelled all 36
    nodes and packed them into 6.8 inches; at 58 nodes in the tallest column that
    produces overlapping text and touching markers, so the dense layers are now
    drawn as unlabelled columns with their counts in the header and an explicit
    note on the figure saying which names were left off.
    """
    graph = kg
    stats = kg_stats(kg)
    pos = _layered_positions(graph)
    tallest = max((stats["by_type"].get(t, 0) for t in LAYERS), default=1)

    # Height follows the tallest column so the markers never touch: ~0.16in per
    # node in that column, floored so a small graph does not come out squat.
    height = max(7.0, min(16.0, 2.4 + 0.155 * tallest))
    fig, ax = plt.subplots(figsize=(15.0, height))

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color="#c9c9d4", width=0.55, alpha=0.75,
        arrows=False,
    )

    for layer_type in LAYERS:
        members = sorted(nodes_of(graph, layer_type), key=lambda n: _sort_key(graph, n))
        if not members:
            continue
        # Dense columns get small markers; sparse ones stay big enough to read.
        size = 190.0 if len(members) <= 15 else 60.0
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=members, node_size=size,
            node_color=LAYER_COLOR[layer_type], linewidths=0.0, label=layer_type,
        )

    # Labels: only the layers whose names carry information at this size.
    for node in sorted(nodes_of(graph, DOCUMENT), key=lambda n: _sort_key(graph, n)):
        x, y = pos[node]
        body = graph.nodes[node].get("issuing_body")
        stem = _short_doc(str(graph.nodes[node].get("stem", node)), 40)
        ax.text(x - 0.10, y, f"{stem}  [{body or 'body not identified'}]",
                fontsize=6.6, va="center", ha="right", color="#22222c")

    for layer_type, dx, fontsize in ((SERVED_RULE, 0.05, 7.0), (VARIABLE, 0.06, 8.0)):
        for node in sorted(nodes_of(graph, layer_type), key=lambda n: _sort_key(graph, n)):
            x, y = pos[node]
            label = graph.nodes[node].get("name") or graph.nodes[node].get("rule_id") or node
            ax.text(x + dx, y, _demojibake(str(label)), fontsize=fontsize,
                    va="center", ha="left", color="#22222c")

    _annotate_channels(ax, graph, pos)

    # Column headers.
    for column, layer_type in enumerate(LAYERS):
        n = stats["by_type"].get(layer_type, 0)
        if not n:
            continue
        ax.text(
            COLUMN_X.get(layer_type, float(column)), 1.015, f"{layer_type}\n({n})",
            fontsize=8.5, ha="center", va="bottom", color=LAYER_COLOR[layer_type],
            fontweight="bold", transform=ax.get_xaxis_transform(),
        )

    fig.suptitle(
        f"Rule provenance — {stats['nodes']} nodes, {stats['edges']} edges, "
        f"{len(LAYERS)} layers. No topology: one graph serves all three grids.",
        fontsize=12, y=0.995,
    )
    ax.set_xlim(-2.35, COLUMN_X[VARIABLE] + 0.10)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    fig.text(0.012, 0.012, _omitted_note(stats), fontsize=7.0, ha="left",
             va="bottom", color="#55555f", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))

    written = _save_figure_formats(fig, output_path, export_formats=export_formats)
    plt.close(fig)
    return written


def _annotate_channels(ax, graph: nx.DiGraph, pos: dict) -> None:
    """Bracket the ExplanatoryRule column by channel.

    The column is 58 unlabelled dots; without this it says nothing. Sorted by
    `_sort_key`, each channel is a contiguous run, so one bracket per run turns
    the column into the corpus's channel breakdown — which is the only thing a
    reader can use from 58 rule ids anyway.
    """
    members = sorted(nodes_of(graph, EXPLANATORY_RULE), key=lambda n: _sort_key(graph, n))
    if not members:
        return
    column = COLUMN_X[EXPLANATORY_RULE]
    groups: dict[str, list[float]] = {}
    for node in members:
        channel = str(graph.nodes[node].get("channel", "unlabelled"))
        groups.setdefault(channel, []).append(pos[node][1])

    ordered = [c for c in CHANNEL_ORDER if c in groups]
    ordered += [c for c in sorted(groups) if c not in CHANNEL_ORDER]
    note = {"BLOCK": "\nthe only channel that can veto"}
    x = column + 0.06
    for channel in ordered:
        ys = groups[channel]
        top, bottom = max(ys), min(ys)
        ax.plot([x, x], [bottom, top], color=LAYER_COLOR[EXPLANATORY_RULE],
                linewidth=1.2, solid_capstyle="butt")
        # White backing: the label sits in the gap between two columns, which is
        # where the instantiates edges fan out.
        ax.text(x + 0.05, (top + bottom) / 2,
                f"{channel} ({len(ys)}){note.get(channel, '')}",
                fontsize=7.0, va="center", ha="left", linespacing=1.3,
                color=LAYER_COLOR[EXPLANATORY_RULE],
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.82))


# ── FIGURE 2 — corroboration ──────────────────────────────────────────────────

def _corroboration_rows(kg: nx.DiGraph) -> list[dict]:
    """One row per predicate: clauses per document, and whether it can veto."""
    rows: list[dict] = []
    for pid in sorted(nodes_of(kg, PREDICATE)):
        served = 0
        # A CLAUSE, not a (clause, rule) pair. One clause reaches a predicate by
        # several paths once the explanatory layer is in — the same section is
        # stated by a Rule that deduped into a ServedRule *and* by an
        # ExplanatoryRule — and counting the paths inflates the corroboration
        # figure. `kg_stats` takes a set here; so does this, or the figure
        # disagrees with the number the thesis quotes (20 against 10).
        clauses: set[str] = set()
        for rule_node in kg.predecessors(pid):
            rtype = kg.nodes[rule_node].get("type")
            if rtype == SERVED_RULE:
                served += 1
            elif rtype != EXPLANATORY_RULE:
                continue
            # A ServedRule sits one hop further from its clauses than an
            # ExplanatoryRule (Clause->Rule->ServedRule against Clause->
            # ExplanatoryRule). Reuse build's walk rather than assume the served
            # shape — assuming it is why every explanatory predicate would draw
            # an empty bar.
            clauses.update(_clauses_behind(kg, rule_node))
        per_doc: Counter = Counter(kg.nodes[c]["document"] for c in clauses)
        rows.append({
            "attrs": kg.nodes[pid],
            "per_doc": per_doc,
            "clauses": sum(per_doc.values()),
            "n_served": served,
        })
    # Served predicates first — they are the ones the shield can act on — then
    # by how much corroborates them, so the headline row lands at the top.
    rows.sort(key=lambda r: (-int(bool(r["n_served"])), -r["clauses"],
                             r["attrs"]["canonical"]))
    return rows


def plot_corroboration(
    kg: nx.DiGraph,
    output_path: str | Path = "kg/kg_corroboration",
    export_formats: tuple[str, ...] = EXPORT_FORMATS,
) -> list[str]:
    """Clauses per predicate, segmented by source document.

    Rewritten 2026-09-17: 2 predicates became 26, and the old row label (wrapped
    plain-English gloss plus wrapped condition, at 1.1 inches per row) would have
    produced a 30-inch figure. Rows are now one predicate each, labelled with the
    canonical condition, split into the served block and the explanatory-only
    block. The plain-English glosses are still printed by scripts/build_kg.py.
    """
    rows = _corroboration_rows(kg)
    if not rows:
        rows = []
    n_served = sum(1 for r in rows if r["n_served"])

    documents = sorted({d for r in rows for d in r["per_doc"]})
    palette = plt.get_cmap("tab20")
    doc_color = {d: palette(i % 20) for i, d in enumerate(documents)}

    fig, ax = plt.subplots(figsize=(11.5, 2.9 + 0.40 * max(len(rows), 1)))

    for i, row in enumerate(rows):
        left = 0
        for doc in documents:
            n = row["per_doc"].get(doc, 0)
            if not n:
                continue
            body = kg.nodes[doc_id(doc)].get("issuing_body")
            ax.barh(i, n, left=left, height=0.56, color=doc_color[doc],
                    edgecolor="white", linewidth=0.8,
                    label=f"{_short_doc(doc)}" + (f" ({body})" if body else ""))
            if n >= 2:
                ax.text(left + n / 2, i, str(n), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
            left += n
        ax.text(
            left + 0.12, i,
            f"{left} clause(s), {len(row['per_doc'])} document(s)",
            va="center", fontsize=7.5, color="#44444e",
        )

    ax.set_yticks(range(len(rows)))
    # Wrap rather than truncate: cutting "is at least 0.9 and ..." mid-clause
    # misstates the rule, which is the one thing a provenance figure must not do.
    ax.set_yticklabels(
        [textwrap.fill(r["attrs"]["canonical"], 42) + f"\n[{r['attrs']['role']}]"
         for r in rows],
        fontsize=7.0,
    )
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("clauses stating this predicate", fontsize=9)

    # The two blocks, and the line between them. Without it the figure implies
    # all 26 predicates are equally live; only the served ones can veto.
    if 0 < n_served < len(rows):
        ax.axhline(n_served - 0.5, color="#8a8a96", linewidth=0.9, linestyle=(0, (4, 3)))
    _block_label(ax, 0, n_served, "SERVED — the shield acts on these")
    _block_label(ax, n_served, len(rows),
                 "EXPLANATORY ONLY — citable, never a veto (shield v2 channels)")

    ax.set_title(
        f"Independent corroboration per predicate — all {len(rows)} predicates in the graph\n"
        "the same physical check, restated across separate standards; "
        "and the long tail that nothing restates",
        fontsize=10.5,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.margins(x=0.22)

    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    # Below the axes: inside the plot it lands on the per-row annotations.
    ax.legend(seen.values(), seen.keys(), fontsize=6.8, frameon=False, ncol=2,
              loc="upper center", title="source document (issuing body)",
              title_fontsize=7.4,
              bbox_to_anchor=(0.5, -1.30 / max(len(rows), 1) - 0.035))

    fig.tight_layout()
    written = _save_figure_formats(fig, output_path, export_formats=export_formats)
    plt.close(fig)
    return written


def _block_label(ax, start: int, stop: int, text: str) -> None:
    """Name a contiguous block of rows in the right-hand margin."""
    if stop <= start:
        return
    ax.text(
        1.012, (start + stop - 1) / 2, text,
        transform=ax.get_yaxis_transform(which="grid"),
        rotation=90, rotation_mode="anchor",
        ha="center", va="bottom", fontsize=7.4, color="#55555f",
        fontweight="bold", clip_on=False,
    )


def _truncate(text: str, limit: int) -> str:
    """Length-only. Unlike `build._short` this keeps underscores: node labels are
    identifiers (`R_1717`, `loading_pct`) and stripping them corrupts the name."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _demojibake(text: str) -> str:
    """Undo a cp1252-read-as-UTF-8 round trip in a document stem.

    One stem reaches the graph as `TPL-001-5.1 â€” Transmission …`. The graph is
    an input here and is not ours to rewrite, but printing the mojibake into a
    thesis figure is not on either.
    """
    if "â" not in text:
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _short_doc(stem: str, limit: int = 34) -> str:
    """Document stems are filenames, so underscores here ARE word separators."""
    return _truncate(_demojibake(stem).replace("_", " "), limit)


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
