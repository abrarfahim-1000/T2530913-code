"""
build_kg.py — Knowledge Graph construction and visualization
============================================================
Builds a NetworkX DiGraph from:
  - all_rules_deduped.jsonl  (LLM-extracted rules from extraction pipeline)
  - GridEnvMetadata           (topology: buses, lines from meta JSON)

Graph schema:
  Nodes: Bus_N | Line_N | Grid | Rule nodes (R_001, ...)
  Edges: connected_to (Bus↔Line), has_rule (Entity→Rule)

Usage:
    from kg.build_kg import build_knowledge_graph, load_knowledge_graph
    KG = build_knowledge_graph("rules/all_rules_deduped.jsonl", meta, save_path="kg/knowledge_graph.pkl")
    KG = load_knowledge_graph("kg/knowledge_graph.pkl")

Visualization:
    from kg.build_kg import visualize_kg, visualize_rule_subgraph, print_kg_stats
    visualize_kg(KG, output_path="kg/kg_full.html")
    visualize_rule_subgraph(KG, rule_id="R_001", output_path="kg/rule_R001.html")

2D Figures (reports / slides):
    from kg.build_kg import plot_kg_overview_2d, plot_rule_neighborhood_2d
    plot_kg_overview_2d(KG, output_path="kg/kg_overview_2d")
    plot_rule_neighborhood_2d(KG, rule_id="R_003", output_path="kg/rule_R_003_2d")
"""

import json
import pickle
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np

# ── OPTIONAL IMPORTS (visualization) ─────────────────────────────────────────
try:
    import matplotlib
    # Use a non-interactive backend when possible (safe for servers/CI).
    # If a backend is already active (e.g., interactive notebooks), keep it.
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# ── NODE TYPE → COLOR MAP ─────────────────────────────────────────────────────
NODE_COLORS = {
    "Bus":             "#4a90d9",   # blue
    "Line":            "#7ed321",   # green
    "Grid":            "#f5a623",   # orange
    "Rule_critical":   "#d0021b",   # red
    "Rule_high":       "#e88a00",   # amber
    "Rule_medium":     "#f5d020",   # yellow
    "Rule_low":        "#9b9b9b",   # grey
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ══════════════════════════════════════════════════════════════════════════════
#  BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(
    rules_path: str,
    meta,                        # GridEnvMetadata instance
    save_path: Optional[str] = None,
) -> nx.DiGraph:
    """
    Build KG from rules JSONL + grid topology metadata.

    Args:
        rules_path : path to all_rules_deduped.jsonl
        meta       : GridEnvMetadata (n_sub, n_line, line_or_bus, line_ex_bus)
        save_path  : optional .pkl path to persist the graph

    Returns:
        nx.DiGraph
    """
    KG = nx.DiGraph()

    # ── 1. Add topology: Bus nodes ────────────────────────────────────────────
    for bus_id in range(meta.n_sub):
        KG.add_node(
            f"Bus_{bus_id}",
            type="Bus",
            bus_id=bus_id,
            label=f"Bus {bus_id}",
        )

    # ── 2. Add topology: Line nodes + Bus↔Line edges ──────────────────────────
    for line_id in range(meta.n_line):
        or_bus = int(meta.line_or_bus[line_id])
        ex_bus = int(meta.line_ex_bus[line_id])

        KG.add_node(
            f"Line_{line_id}",
            type="Line",
            line_id=line_id,
            or_bus=or_bus,
            ex_bus=ex_bus,
            label=f"Line {line_id}",
        )
        KG.add_edge(f"Bus_{or_bus}", f"Line_{line_id}", type="connected_to", direction="origin")
        KG.add_edge(f"Bus_{ex_bus}", f"Line_{line_id}", type="connected_to", direction="extremity")

    # ── 3. Add Grid node ──────────────────────────────────────────────────────
    KG.add_node("Grid", type="Grid", label="Grid")

    # ── 4. Load and add rules ─────────────────────────────────────────────────
    n_rules = 0
    entity_rule_counts = defaultdict(int)

    with open(rules_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rule = json.loads(line)
            except json.JSONDecodeError:
                continue

            rule_id   = rule["rule_id"]
            entity    = rule.get("entity", "Grid").lower()
            severity  = rule.get("severity", "medium").lower()

            # Add Rule node — store all fields
            KG.add_node(
                rule_id,
                type="Rule",
                severity=severity,
                label=f"{rule_id}\n({severity})",
                **{k: v for k, v in rule.items() if k != 'severity'},
            )

            # Connect rule to every matching topology entity
            if entity == "bus":
                for bus_id in range(meta.n_sub):
                    KG.add_edge(f"Bus_{bus_id}", rule_id, type="has_rule")
                entity_rule_counts["bus"] += 1

            elif entity == "line":
                for line_id in range(meta.n_line):
                    KG.add_edge(f"Line_{line_id}", rule_id, type="has_rule")
                entity_rule_counts["line"] += 1

            elif entity == "generator":
                # Generator nodes — not in topology by default; attach to Grid
                KG.add_edge("Grid", rule_id, type="has_rule")
                entity_rule_counts["generator"] += 1

            elif entity == "transformer":
                KG.add_edge("Grid", rule_id, type="has_rule")
                entity_rule_counts["transformer"] += 1

            elif entity == "protectiondevice":
                KG.add_edge("Grid", rule_id, type="has_rule")
                entity_rule_counts["protectiondevice"] += 1

            else:  # "grid" or unknown → attach to Grid node
                KG.add_edge("Grid", rule_id, type="has_rule")
                entity_rule_counts["grid"] += 1

            n_rules += 1

    print(f"[KG] Built graph:")
    print(f"     Nodes  : {KG.number_of_nodes():,}  "
          f"(buses={meta.n_sub}, lines={meta.n_line}, rules={n_rules}, +Grid)")
    print(f"     Edges  : {KG.number_of_edges():,}")
    print(f"     Rules per entity type: {dict(entity_rule_counts)}")

    # ── 5. Persist ────────────────────────────────────────────────────────────
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(KG, f)
        print(f"[KG] Saved → {save_path}")

    return KG


def load_knowledge_graph(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY HELPERS  (used by the Shield at inference time)
# ══════════════════════════════════════════════════════════════════════════════

def get_rules_for_entity(KG: nx.DiGraph, entity_node: str) -> list[dict]:
    """
    Return all Rule dicts reachable from entity_node via 'has_rule' edges.
    entity_node : e.g. "Bus_5", "Line_12", "Grid"
    """
    rules = []
    if entity_node not in KG:
        return rules
    for _, rule_id, edge_data in KG.out_edges(entity_node, data=True):
        if edge_data.get("type") == "has_rule":
            rule_attrs = KG.nodes[rule_id]
            if rule_attrs.get("type") == "Rule":
                rules.append(dict(rule_attrs))
    # Sort by severity (critical first)
    rules.sort(key=lambda r: SEVERITY_ORDER.get(r.get("severity", "low"), 3))
    return rules


def get_rules_for_fault_type(KG: nx.DiGraph, fault_type: str) -> list[dict]:
    """
    Return all rules applicable to a given fault type prediction.
    Maps fault_type → entity node + Grid (grid-level rules always apply).
    fault_type : "overload" | "line_trip" | "cascade" | "normal"
    """
    fault_entity_map = {
        "overload":  ["Grid"],      # overload rules on lines → attached to Grid in our schema
        "line_trip": ["Grid"],
        "cascade":   ["Grid"],
        "normal":    ["Grid"],
    }
    # Also pull rules directly attached to line/bus entities
    entity_nodes = fault_entity_map.get(fault_type, ["Grid"])
    # For overload/line_trip, line-level rules are also relevant
    if fault_type in ("overload", "line_trip", "cascade"):
        # Sample a Line node to discover line-level rules
        entity_nodes = entity_nodes + ["Line_0"]

    seen_ids = set()
    all_rules = []
    for entity_node in entity_nodes:
        for rule in get_rules_for_entity(KG, entity_node):
            if rule["rule_id"] not in seen_ids:
                seen_ids.add(rule["rule_id"])
                all_rules.append(rule)

    all_rules.sort(key=lambda r: SEVERITY_ORDER.get(r.get("severity", "low"), 3))
    return all_rules


def get_all_rules(KG: nx.DiGraph) -> list[dict]:
    """Return all Rule nodes as a list of dicts, sorted by rule_id."""
    rules = [
        dict(attrs)
        for _, attrs in KG.nodes(data=True)
        if attrs.get("type") == "Rule"
    ]
    rules.sort(key=lambda r: r.get("rule_id", ""))
    return rules


# ══════════════════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════════════════

def print_kg_stats(KG: nx.DiGraph):
    """Print a structured summary of the KG."""
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    severity_counts = defaultdict(int)

    for _, attrs in KG.nodes(data=True):
        t = attrs.get("type", "unknown")
        node_types[t] += 1
        if t == "Rule":
            severity_counts[attrs.get("severity", "unknown")] += 1

    for _, _, attrs in KG.edges(data=True):
        edge_types[attrs.get("type", "unknown")] += 1

    print("\n" + "=" * 56)
    print("  KNOWLEDGE GRAPH STATS")
    print("=" * 56)
    print(f"  Total nodes : {KG.number_of_nodes():,}")
    print(f"  Total edges : {KG.number_of_edges():,}")
    print(f"\n  Node types:")
    for t, count in sorted(node_types.items()):
        print(f"    {t:<20} {count:>6,}")
    print(f"\n  Edge types:")
    for t, count in sorted(edge_types.items()):
        print(f"    {t:<20} {count:>6,}")
    print(f"\n  Rule severity breakdown:")
    for sev in ["critical", "high", "medium", "low"]:
        if sev in severity_counts:
            bar = "█" * severity_counts[sev]
            print(f"    {sev:<12} {severity_counts[sev]:>4}  {bar}")
    print("=" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — PyVis (interactive HTML)
# ══════════════════════════════════════════════════════════════════════════════

def _node_color(attrs: dict) -> str:
    t = attrs.get("type", "")
    if t == "Rule":
        sev = attrs.get("severity", "medium")
        return NODE_COLORS.get(f"Rule_{sev}", NODE_COLORS["Rule_medium"])
    return NODE_COLORS.get(t, "#cccccc")


def _node_size(attrs: dict) -> int:
    t = attrs.get("type", "")
    if t == "Rule":
        return 18
    if t == "Grid":
        return 30
    if t == "Bus":
        return 12
    if t == "Line":
        return 10
    return 10


def _node_title(attrs: dict) -> str:
    """Tooltip text shown on hover."""
    t = attrs.get("type", "")
    if t == "Rule":
        return (
            f"<b>{attrs.get('rule_id')}</b> [{attrs.get('severity','').upper()}]<br>"
            f"<b>Entity:</b> {attrs.get('entity','')}<br>"
            f"<b>Condition:</b> {attrs.get('condition','')}<br>"
            f"<b>Action:</b> {attrs.get('action','')}<br>"
            f"<b>Source:</b> {attrs.get('source','')}<br>"
            f"<b>Explanation:</b> {attrs.get('explanation','')}"
        )
    if t == "Bus":
        return f"<b>Bus {attrs.get('bus_id')}</b>"
    if t == "Line":
        return (
            f"<b>Line {attrs.get('line_id')}</b><br>"
            f"Or-bus: {attrs.get('or_bus')}  Ex-bus: {attrs.get('ex_bus')}"
        )
    return f"<b>{t}</b>"


def visualize_kg(
    KG: nx.DiGraph,
    output_path: str = "kg/kg_full.html",
    max_nodes: int = 200,
    show_lines: bool = True,
    show_buses: bool = True,
) -> str:
    """
    Interactive HTML visualization using PyVis.
    Opens in any browser. Hover nodes for rule details.

    Args:
        KG          : built KG
        output_path : where to save the HTML
        max_nodes   : cap total nodes to prevent browser freeze on large KGs
        show_lines  : include Line nodes
        show_buses  : include Bus nodes

    Returns:
        output_path (str)
    """
    if not HAS_PYVIS:
        print("[VIZ] pyvis not installed. Run: pip install pyvis")
        _visualize_kg_matplotlib(KG, output_path.replace(".html", ".png"))
        return output_path

    net = Network(
        height="900px", width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    # Filter which node types to include
    skip_types = set()
    if not show_lines:
        skip_types.add("Line")
    if not show_buses:
        skip_types.add("Bus")

    # Count per type to respect max_nodes
    type_counts = defaultdict(int)
    for _, attrs in KG.nodes(data=True):
        type_counts[attrs.get("type")] += 1

    added_nodes = set()

    for node_id, attrs in KG.nodes(data=True):
        t = attrs.get("type", "")
        if t in skip_types:
            continue
        if len(added_nodes) >= max_nodes:
            break

        net.add_node(
            node_id,
            label=attrs.get("label", str(node_id)),
            color=_node_color(attrs),
            size=_node_size(attrs),
            title=_node_title(attrs),
            shape="dot" if t != "Rule" else "diamond",
        )
        added_nodes.add(node_id)

    for src, dst, edge_attrs in KG.edges(data=True):
        if src not in added_nodes or dst not in added_nodes:
            continue
        etype = edge_attrs.get("type", "")
        color = "#555577" if etype == "connected_to" else "#ff6b6b"
        net.add_edge(src, dst, color=color, width=1.5 if etype == "has_rule" else 0.8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(output_path)
    print(f"[VIZ] Interactive KG → {output_path}  ({len(added_nodes)} nodes shown)")
    return output_path


def visualize_rule_subgraph(
    KG: nx.DiGraph,
    rule_id: str,
    output_path: Optional[str] = None,
    hop: int = 2,
) -> str:
    """
    Visualize the ego-graph around a single rule node (N-hop neighbourhood).
    Useful for inspecting one specific rule and all entities it governs.
    """
    if rule_id not in KG:
        print(f"[VIZ] Rule '{rule_id}' not found in KG.")
        return ""

    # Build undirected ego-graph for neighbourhood traversal
    sub_nodes = nx.ego_graph(KG.to_undirected(), rule_id, radius=hop).nodes()
    subKG = KG.subgraph(sub_nodes)

    if output_path is None:
        output_path = f"kg/rule_{rule_id}.html"

    return visualize_kg(subKG, output_path=output_path)


def visualize_rules_only(
    KG: nx.DiGraph,
    output_path: str = "kg/rules_only.html",
) -> str:
    """
    Minimal view: only Rule nodes, colored by severity.
    Fastest way to inspect the extracted rule set.
    """
    return visualize_kg(
        KG,
        output_path=output_path,
        show_lines=False,
        show_buses=False,
    )


def visualize_severity_breakdown(
    KG: nx.DiGraph,
    output_path: str = "kg/severity_breakdown.png",
):
    """
    Matplotlib bar chart: rule count by severity and entity type.
    Saved as PNG.
    """
    if not HAS_MPL:
        print("[VIZ] matplotlib not installed.")
        return

    rules = get_all_rules(KG)
    if not rules:
        print("[VIZ] No rules in KG.")
        return

    # Severity counts
    sev_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    for r in rules:
        sev_counts[r.get("severity", "unknown")] += 1
        entity_counts[r.get("entity", "unknown")] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # Left: severity
    sev_order = ["critical", "high", "medium", "low"]
    sev_vals  = [sev_counts.get(s, 0) for s in sev_order]
    sev_colors = [NODE_COLORS[f"Rule_{s}"] for s in sev_order]
    bars = axes[0].bar(sev_order, sev_vals, color=sev_colors, edgecolor="#333355")
    axes[0].set_title("Rules by Severity", fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, sev_vals):
        if val > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", color="white", fontsize=10,
            )

    # Right: entity type
    entities = sorted(entity_counts.keys())
    e_vals   = [entity_counts[e] for e in entities]
    e_colors = ["#4a90d9", "#7ed321", "#f5a623", "#9b9b9b", "#c0392b", "#8e44ad", "#1abc9c"]
    axes[1].bar(entities, e_vals, color=e_colors[:len(entities)], edgecolor="#333355")
    axes[1].set_title("Rules by Entity Type", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"[VIZ] Severity chart → {output_path}")


def _save_figure_formats(
    fig,
    output_path: str,
    *,
    export_formats: tuple[str, ...],
    dpi: int,
) -> list[str]:
    """Save a Matplotlib figure to multiple formats.

    `output_path` can be provided with or without an extension. Outputs are
    written next to `output_path`, using its stem.

    Returns:
        List of written file paths (as strings).
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
            # Vector formats (pdf/svg) ignore dpi for most elements.
            fig.savefig(out_path, bbox_inches="tight")

        written.append(str(out_path))

    return written


def plot_kg_overview_2d(
    KG: nx.DiGraph,
    output_path: str = "kg/kg_overview_2d",
    *,
    export_formats: tuple[str, ...] = ("png", "pdf", "svg"),
    max_rules: int = 200,
    include_grid: bool = True,
    include_buses: bool = False,
    include_lines: bool = False,
    max_topology_nodes: int = 50,
    hop_from_grid: int = 1,
    layout: str = "spring",
    seed: int = 42,
    figsize: tuple[float, float] = (14.0, 9.0),
    dpi: int = 200,
    label_rules: bool = False,
    label_top_rules: int = 30,
):
    """Render a static 2D KG overview figure (Matplotlib).

    Designed for slides/reports: readable styling, deterministic layout, and
    safe defaults that avoid trying to render the entire dense topology.

    Defaults to plotting Grid + up to `max_rules` Rule nodes. Optionally includes
    a limited number of Bus/Line nodes from the neighbourhood of Grid.

    Args:
        KG: Knowledge graph.
        output_path: Output path *stem* or file path; if no extension is provided,
            files are written as `<output_path>.<format>`.
        export_formats: Tuple of formats to write (e.g., ("png","pdf","svg")).
        max_rules: Maximum number of Rule nodes to include (sorted by severity then rule_id).
        include_grid: Whether to include the Grid node.
        include_buses: Whether to include Bus nodes (capped by max_topology_nodes).
        include_lines: Whether to include Line nodes (capped by max_topology_nodes).
        max_topology_nodes: Maximum number of Bus/Line nodes to include.
        hop_from_grid: If including topology nodes, only include nodes within this hop distance
            from Grid in the undirected graph.
        layout: "spring" | "kamada_kawai".
        seed: Random seed for deterministic layouts.
        figsize: Matplotlib figure size.
        dpi: Figure DPI.
        label_rules: If True, label Rule nodes (may clutter on large graphs).
        label_top_rules: If labeling is enabled, cap the number of rule labels.
    """
    if not HAS_MPL:
        print("[VIZ] matplotlib not installed.")
        return

    # --- Choose nodes to include -------------------------------------------------
    rule_nodes = [
        n
        for n, d in KG.nodes(data=True)
        if d.get("type") == "Rule"
    ]
    # Prefer critical/high, then rule_id for stable ordering
    def _rule_sort_key(node_id: str) -> tuple[int, str]:
        attrs = KG.nodes[node_id]
        sev = attrs.get("severity", "low")
        return (SEVERITY_ORDER.get(sev, 3), str(attrs.get("rule_id", node_id)))

    rule_nodes = sorted(rule_nodes, key=_rule_sort_key)[: max(0, int(max_rules))]

    # IMPORTANT: keep node ordering stable for deterministic layouts.
    ordered_nodes: list[str] = []
    seen: set[str] = set()

    def _push(node_id: str):
        if node_id in KG and node_id not in seen:
            ordered_nodes.append(node_id)
            seen.add(node_id)

    for node_id in rule_nodes:
        _push(node_id)
    if include_grid:
        _push("Grid")

    # Topology nodes are not connected to Grid in this schema.
    # If requested, sample topology nodes from the neighbourhood of the selected
    # rules (and Grid, if present), then cap for readability.
    if include_buses or include_lines:
        UG = KG.to_undirected()
        anchors = list(rule_nodes)
        if include_grid and "Grid" in KG:
            anchors.append("Grid")

        neighbourhood: set[str] = set()
        radius = max(0, int(hop_from_grid))
        for anchor in anchors:
            if anchor in UG:
                neighbourhood.update(nx.ego_graph(UG, anchor, radius=radius).nodes())

        topo_candidates: list[str] = []
        for node_id in neighbourhood:
            t = KG.nodes[node_id].get("type")
            if include_buses and t == "Bus":
                topo_candidates.append(node_id)
            elif include_lines and t == "Line":
                topo_candidates.append(node_id)

        topo_candidates = sorted(topo_candidates)[: max(0, int(max_topology_nodes))]
        for node_id in topo_candidates:
            _push(node_id)

    subKG = KG.subgraph(ordered_nodes).copy()

    # --- Layout ------------------------------------------------------------------
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subKG)
    else:
        pos = nx.spring_layout(subKG, seed=seed, k=1.5)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Draw edges first (behind nodes) -----------------------------------------
    edge_colors = []
    edge_widths = []
    for _, _, attrs in subKG.edges(data=True):
        etype = attrs.get("type")
        if etype == "has_rule":
            edge_colors.append("#b23b3b")
            edge_widths.append(1.2)
        else:
            edge_colors.append("#9aa0a6")
            edge_widths.append(0.8)

    nx.draw_networkx_edges(
        subKG,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.35,
        arrows=False,
    )

    # --- Draw nodes by type/severity --------------------------------------------
    node_ids = list(subKG.nodes())
    node_shapes = {
        "Rule": "D",
        "Grid": "o",
        "Bus": "o",
        "Line": "o",
    }

    # Draw each type separately to control marker shape
    for node_type, marker in node_shapes.items():
        subset = [n for n in node_ids if KG.nodes[n].get("type") == node_type]
        if not subset:
            continue
        subset_colors = [_node_color(KG.nodes[n]) for n in subset]
        subset_sizes = [_node_size(KG.nodes[n]) * 35 for n in subset]
        nx.draw_networkx_nodes(
            subKG,
            pos,
            nodelist=subset,
            node_color=subset_colors,
            node_size=subset_sizes,
            ax=ax,
            alpha=0.95,
            linewidths=0.8,
            edgecolors="#2b2b2b",
            node_shape=marker,
        )

    # --- Labels ------------------------------------------------------------------
    if label_rules:
        labeled_rules = [n for n in rule_nodes if n in subKG]
        labeled_rules = labeled_rules[: max(0, int(label_top_rules))]
        labels = {
            n: str(KG.nodes[n].get("rule_id", n))
            for n in labeled_rules
        }
        nx.draw_networkx_labels(
            subKG,
            pos,
            labels=labels,
            font_size=7,
            font_color="#111111",
            ax=ax,
        )

    # --- Legend ------------------------------------------------------------------
    legend_patches = [
        mpatches.Patch(color=NODE_COLORS["Grid"], label="Grid"),
        mpatches.Patch(color=NODE_COLORS["Rule_critical"], label="Rule: critical"),
        mpatches.Patch(color=NODE_COLORS["Rule_high"], label="Rule: high"),
        mpatches.Patch(color=NODE_COLORS["Rule_medium"], label="Rule: medium"),
        mpatches.Patch(color=NODE_COLORS["Rule_low"], label="Rule: low"),
    ]
    if include_buses:
        legend_patches.insert(1, mpatches.Patch(color=NODE_COLORS["Bus"], label="Bus"))
    if include_lines:
        legend_patches.insert(1, mpatches.Patch(color=NODE_COLORS["Line"], label="Line"))

    ax.legend(handles=legend_patches, loc="upper left", frameon=True, fontsize=9)
    ax.set_title("Knowledge Graph — Overview", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    written = _save_figure_formats(fig, output_path, export_formats=export_formats, dpi=dpi)
    plt.close()
    if written:
        print(f"[VIZ] 2D KG overview → {', '.join(written)}")
    else:
        print(f"[VIZ] 2D KG overview → {output_path}")


def plot_rule_neighborhood_2d(
    KG: nx.DiGraph,
    rule_id: str,
    output_path: Optional[str] = None,
    *,
    export_formats: tuple[str, ...] = ("png", "pdf", "svg"),
    hop: int = 2,
    layout: str = "spring",
    seed: int = 42,
    figsize: tuple[float, float] = (12.0, 8.0),
    dpi: int = 220,
    label_entities: bool = True,
):
    """Render a static 2D figure of a single rule's local neighbourhood.

    This is useful for report callouts: one rule, the entities it governs, and
    nearby topology context (N-hop ego graph).

    Args:
        KG: Knowledge graph.
        rule_id: Rule node id (e.g. "R_001").
        output_path: Output path stem or file path. Defaults to "kg/rule_<rule_id>_2d".
        export_formats: Tuple of formats to write (e.g., ("png","pdf","svg")).
        hop: Ego-graph radius in the undirected projection.
        layout: "spring" | "kamada_kawai".
        seed: Layout seed.
        figsize: Matplotlib figure size.
        dpi: Figure DPI.
        label_entities: Label Grid/Bus/Line nodes (Rule is always labeled).
    """
    if not HAS_MPL:
        print("[VIZ] matplotlib not installed.")
        return

    if rule_id not in KG:
        print(f"[VIZ] Rule '{rule_id}' not found in KG.")
        return

    if output_path is None:
        output_path = f"kg/rule_{rule_id}_2d"

    sub_nodes = nx.ego_graph(KG.to_undirected(), rule_id, radius=max(1, int(hop))).nodes()
    subKG = KG.subgraph(sub_nodes).copy()

    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subKG)
    else:
        pos = nx.spring_layout(subKG, seed=seed, k=1.2)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Edges
    edge_colors = []
    edge_widths = []
    for _, _, attrs in subKG.edges(data=True):
        etype = attrs.get("type")
        if etype == "has_rule":
            edge_colors.append("#b23b3b")
            edge_widths.append(1.4)
        else:
            edge_colors.append("#9aa0a6")
            edge_widths.append(0.9)

    nx.draw_networkx_edges(
        subKG,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.5,
        arrows=False,
    )

    # Nodes by type
    for node_type, marker in (("Rule", "D"), ("Grid", "o"), ("Bus", "o"), ("Line", "o")):
        subset = [n for n, d in subKG.nodes(data=True) if d.get("type") == node_type]
        if not subset:
            continue
        nx.draw_networkx_nodes(
            subKG,
            pos,
            nodelist=subset,
            node_color=[_node_color(KG.nodes[n]) for n in subset],
            node_size=[_node_size(KG.nodes[n]) * 55 for n in subset],
            ax=ax,
            alpha=0.96,
            linewidths=0.9,
            edgecolors="#2b2b2b",
            node_shape=marker,
        )

    # Labels
    labels: dict[str, str] = {rule_id: str(KG.nodes[rule_id].get("rule_id", rule_id))}
    if label_entities:
        for n, d in subKG.nodes(data=True):
            if n == rule_id:
                continue
            t = d.get("type")
            if t == "Grid":
                labels[n] = "Grid"
            elif t == "Bus":
                labels[n] = f"Bus {d.get('bus_id', '')}".strip()
            elif t == "Line":
                labels[n] = f"Line {d.get('line_id', '')}".strip()

    nx.draw_networkx_labels(
        subKG,
        pos,
        labels=labels,
        font_size=9,
        font_color="#111111",
        ax=ax,
    )

    # Title includes severity for quick context
    sev = KG.nodes[rule_id].get("severity", "medium")
    ax.set_title(f"Rule Neighbourhood — {rule_id} ({str(sev).upper()})", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    written = _save_figure_formats(fig, output_path, export_formats=export_formats, dpi=dpi)
    plt.close()
    if written:
        print(f"[VIZ] 2D rule neighbourhood → {', '.join(written)}")
    else:
        print(f"[VIZ] 2D rule neighbourhood → {output_path}")


# ── MATPLOTLIB FALLBACK (when pyvis unavailable) ──────────────────────────────
def _visualize_kg_matplotlib(KG: nx.DiGraph, output_path: str = "kg/kg_full.png"):
    if not HAS_MPL:
        print("[VIZ] matplotlib not installed.")
        return

    # Use only Rule + Grid nodes for readability (full KG is too dense for mpl)
    rule_nodes = [n for n, d in KG.nodes(data=True) if d.get("type") in ("Rule", "Grid")]
    subKG = KG.subgraph(rule_nodes)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(subKG, seed=42, k=2.0)
    colors = [_node_color(KG.nodes[n]) for n in subKG.nodes()]
    sizes  = [_node_size(KG.nodes[n]) * 30 for n in subKG.nodes()]

    nx.draw_networkx_nodes(subKG, pos, node_color=colors, node_size=sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(
        subKG, pos,
        labels={n: KG.nodes[n].get("rule_id", n) for n in subKG.nodes()},
        font_size=6, font_color="white", ax=ax,
    )
    nx.draw_networkx_edges(
        subKG, pos, ax=ax,
        edge_color="#ff6b6b", alpha=0.4, arrows=True,
        arrowsize=8, width=0.8,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=NODE_COLORS["Rule_critical"], label="Rule: critical"),
        mpatches.Patch(color=NODE_COLORS["Rule_high"],     label="Rule: high"),
        mpatches.Patch(color=NODE_COLORS["Rule_medium"],   label="Rule: medium"),
        mpatches.Patch(color=NODE_COLORS["Rule_low"],      label="Rule: low"),
        mpatches.Patch(color=NODE_COLORS["Grid"],          label="Grid"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              facecolor="#333355", labelcolor="white", fontsize=8)
    ax.set_title("Knowledge Graph — Rules", color="white", fontsize=13, fontweight="bold")
    ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"[VIZ] Matplotlib KG → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from scripts.pyg_data import GridEnvMetadata
    from training.config import DATA_FILE

    parser = argparse.ArgumentParser(description="Build and visualize the Knowledge Graph")
    parser.add_argument("--rules",   default="rules/all_rules_deduped.jsonl",
                        help="Path to all_rules_deduped.jsonl")
    parser.add_argument("--meta",    default=None,
                        help="Path to _meta.json (default: inferred from DATA_FILE)")
    parser.add_argument("--out-dir", default="kg",
                        help="Output directory for KG + visualizations")
    parser.add_argument("--no-viz",  action="store_true",
                        help="Skip visualization (build only)")
    args = parser.parse_args()

    # Load metadata
    meta_path = args.meta or DATA_FILE.replace(".jsonl", "_meta.json")
    import json as _json
    with open(meta_path) as f:
        meta_dict = _json.load(f)
    meta = GridEnvMetadata(meta_dict)

    # Build
    kg_path = os.path.join(args.out_dir, "knowledge_graph.pkl")
    KG = build_knowledge_graph(args.rules, meta, save_path=kg_path)
    print_kg_stats(KG)

    # Visualize
    if not args.no_viz:
        out = args.out_dir
        visualize_kg(KG, output_path=f"{out}/kg_full.html")
        visualize_rules_only(KG, output_path=f"{out}/rules_only.html")
        visualize_severity_breakdown(KG, output_path=f"{out}/severity_breakdown.png")
        print("\n[DONE] All outputs written to:", out)