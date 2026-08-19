import torch
import numpy as np
import json
import grid2op
from torch_geometric.data import Data, Dataset, InMemoryDataset
import os
import linecache

ENV_NAME = "l2rpn_neurips_2020_track1_small"

LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3}
RHO_CLIP  = 2.0

# â”€â”€ UPDATED DIMENSIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# node: load_p, mean_v, max_rho, connected_line_frac, global_trip_frac
# edge: rho, p_or, q_or, near_limit
# ── FEATURE SET ───────────────────────────────────────────────────────────────
#   node: load_p, mean_v, max_rho, connected_line_frac, global_trip_frac,
#         sum_headroom, sum_abs_p, degree
#   edge: rho, p_or, q_or, near_limit, |p_or|, |q_or|, apparent_s, headroom
#
# The last three of each were added for N-1 screening, chosen from what actually
# predicted post-contingency violation in the probe (component_d_plan.md §1.1):
#
#   |p_or| — the single strongest predictor there (F1 ~0.61 alone). p_or itself
#     is RAW AND SIGNED; flow direction is arbitrary and only magnitude predicts,
#     so a signed feature centred near zero forces the model to synthesise |x|,
#     which the linear part of the readout cannot express at all.
#   headroom = max(0, 1 - rho) — spare thermal capacity, the quantity that
#     decides whether a neighbour can absorb redistributed flow.
#   node sum_headroom / sum_abs_p / degree — the same idea aggregated per bus:
#     spare capacity at this bus, flow through it, and how many alternative paths
#     exist. In the probe these lifted F1 from 0.61 to 0.87.
NODE_FEATURES = 8
EDGE_FEATURES = 8


class GridEnvMetadata:
    def __init__(self, meta_dict=None):
        if meta_dict:
            self._init_from_dict(meta_dict)
        else:
            self.env_name = ENV_NAME
            print(f"[meta] Initializing Grid2Op env: {self.env_name}...")
            env = grid2op.make(self.env_name)
            self.n_sub = env.n_sub
            self.n_line = env.n_line
            self.n_load = env.n_load
            self.n_gen = env.n_gen
            self.line_or_bus = env.line_or_to_subid.copy()
            self.line_ex_bus = env.line_ex_to_subid.copy()
            self.load_to_sub = env.load_to_subid.copy()
            self.gen_to_sub  = env.gen_to_subid.copy()
            self.n_classes = len(LABEL_MAP)
            self.label_map = LABEL_MAP.copy()
            env.close()

    def _init_from_dict(self, meta_dict):
        self.env_name = meta_dict["env_name"]
        self.n_sub    = meta_dict["n_sub"]
        self.n_line   = meta_dict["n_line"]
        self.n_load   = meta_dict["n_load"]
        self.n_gen    = meta_dict["n_gen"]
        topo = meta_dict["topology"]
        self.line_or_bus = np.array(topo["line_or_bus"])
        self.line_ex_bus = np.array(topo["line_ex_bus"])
        self.load_to_sub = np.array(topo["load_to_sub"])
        self.gen_to_sub  = np.array(topo["gen_to_sub"])
        self.n_classes = meta_dict.get("n_classes", len(LABEL_MAP))
        self.label_map = LABEL_MAP


def build_node_features(r, meta: GridEnvMetadata):
    """
    Per-bus node features — 8 per node, matching NODE_FEATURES above.

      0  load_p              - total active load at bus
      1  mean_v              - mean voltage of connected lines, divided by 150.0
      2  max_rho             - max line loading ratio at bus
      3  connected_line_frac - fraction of lines at this bus still connected
                               0.0 = all tripped, 1.0 = all healthy
      4  global_trip_frac    - fraction of ALL lines in the graph that are tripped,
                               broadcast uniformly to every node
      5  sum_headroom        - spare thermal capacity at this bus (energized lines)
      6  sum_abs_p           - active-power throughput at this bus
      7  degree              - number of energized lines, i.e. alternative paths

    Features 5-7 were added for N-1 screening; see the FEATURE SET note at the top of
    this module for why each one is present.

    ⚠️ Feature 1 divides by a flat 150.0. That is fine as a GNN input scale factor, but
    it is NOT the per-unit conversion — the shield uses per-line base kV from
    `data/grid_dataset_<tag>_basekv.json` (component_d_plan.md §5). The two must not be
    conflated. It is also the most likely reason case14 transfers badly: its 14/20 kV
    lines enter here at ~0.13 where the 138/345 kV training grid gave ~1.0 (§7.6).

    HISTORICAL: a 6th binary `overloaded` feature (max_rho >= 1.0) was tried during the
    retired 4-class classify task and collapsed the `normal` class to 0.0 F1 on the small
    [16,32,32] model. That result pertains to the classify target and its 5-feature input,
    not to the 8-feature N-1 setup here.
    """
    # 1. Extract raw arrays (gen_p removed)
    load_p      = np.array(r["load_p"],      dtype=np.float32)
    v_or        = np.array(r["v_or"],        dtype=np.float32)
    rho         = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)
    line_status = np.array(r["line_status"], dtype=np.float32)  # 1=connected, 0=tripped

    # Global trip fraction â€” cleanly separates:
    #   normal/overload: 0.0
    #   line_trip:       1/n_line â‰ˆ 0.017 (exactly 1 line tripped)
    #   cascade:         â‰¥2/n_line â‰ˆ 0.034+
    # Broadcast uniformly to every node; primary discriminator for line_trip vs cascade.
    trip_frac = np.float32((1.0 - line_status).sum() / len(line_status))
    node_trip_frac = np.full(meta.n_sub, trip_frac, dtype=np.float32)

    for arr in [v_or, rho]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Initialize node arrays (node_gen removed)
    node_load    = np.zeros(meta.n_sub, dtype=np.float32)
    node_v       = np.zeros(meta.n_sub, dtype=np.float32)
    node_rho     = np.zeros(meta.n_sub, dtype=np.float32)
    node_connected = np.zeros(meta.n_sub, dtype=np.float32)
    node_total     = np.zeros(meta.n_sub, dtype=np.float32)
    v_count        = np.zeros(meta.n_sub, dtype=np.float32)

    # 3. Map features to buses
    np.add.at(node_load, meta.load_to_sub, load_p)

    # Voltage: mean of connected lines at each bus (or-side)
    np.add.at(node_v,   meta.line_or_bus, v_or)
    np.add.at(v_count,  meta.line_or_bus, 1)
    np.add.at(node_v,   meta.line_ex_bus, v_or)
    np.add.at(v_count,  meta.line_ex_bus, 1)

    # Max rho per bus
    np.maximum.at(node_rho, meta.line_or_bus, rho)
    np.maximum.at(node_rho, meta.line_ex_bus, rho)

    # Connected line fraction per bus
    np.add.at(node_connected, meta.line_or_bus, line_status)
    np.add.at(node_connected, meta.line_ex_bus, line_status)
    np.add.at(node_total,     meta.line_or_bus, 1.0)
    np.add.at(node_total,     meta.line_ex_bus, 1.0)

    # 4. Normalize
    node_v    = np.divide(node_v, v_count,
                          out=np.zeros_like(node_v), where=v_count > 0)
    node_v    = node_v / 150.0  # normalize kV â†’ ~[0,1]

    node_conn_frac = np.divide(node_connected, node_total,
                               out=np.ones_like(node_connected),   # default 1.0
                               where=node_total > 0)

    # Per-bus capacity accounting. Energized lines only: a tripped line has
    # rho = 0 and would otherwise contribute a full 1.0 of "spare capacity"
    # it cannot actually provide.
    live = np.array(r["line_status"], dtype=bool)
    p_abs = np.abs(np.array(r["p_or"], dtype=np.float32))
    headroom = np.maximum(0.0, 1.0 - rho).astype(np.float32)
    np.nan_to_num(p_abs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    node_headroom = np.zeros(meta.n_sub, dtype=np.float32)
    node_absp     = np.zeros(meta.n_sub, dtype=np.float32)
    node_degree   = np.zeros(meta.n_sub, dtype=np.float32)
    for bus in (meta.line_or_bus, meta.line_ex_bus):
        np.add.at(node_headroom, bus[live], headroom[live])
        np.add.at(node_absp,     bus[live], p_abs[live])
        np.add.at(node_degree,   bus[live], 1.0)

    return np.stack([node_load, node_v, node_rho, node_conn_frac, node_trip_frac,
                     node_headroom, node_absp, node_degree], axis=1)


def build_edges(r, meta):
    # 1. Convert line_status to a boolean mask (True = connected, False = disconnected)
    line_status = np.array(r["line_status"], dtype=bool)

    # 2. Filter to connected lines only
    or_bus = meta.line_or_bus[line_status]
    ex_bus = meta.line_ex_bus[line_status]

    rho        = np.array(r["rho"])[line_status]
    p_or       = np.array(r["p_or"])[line_status]
    q_or       = np.array(r["q_or"])[line_status]
    # overload flag: 1 if line is at â‰¥100% thermal capacity (rho â‰¥ 1.0).
    # Threshold set to exactly the normal/overload labeling boundary (max_rho â‰¥ 1.0).
    # At 0.9 this fired on borderline NORMAL samples (rho âˆˆ [0.9, 1.0)), giving them the
    # same edge signal as genuine overloads and blurring the only boundary that separates
    # normal from overload. At 1.0 it is a clean separator: every normal sample gets 0,
    # every overload has â‰¥1 line with the flag set.
    # Replaces the old `line_status` constant (always 1.0 here since tripped lines are
    # filtered out), which had stdâ‰ˆ0 and became identically 0 after z-score normalization.
    near_limit = (rho >= 1.0).astype(np.float32)

    # |p_or| is the load-bearing addition — see the FEATURE SET note at the top.
    # `headroom` gives the readout spare capacity directly rather than requiring
    # it to synthesise 1 - rho through a z-scored feature.
    abs_p, abs_q = np.abs(p_or), np.abs(q_or)
    edge_attr_fwd = np.column_stack([
        rho, p_or, q_or, near_limit,
        abs_p, abs_q, np.hypot(abs_p, abs_q), np.maximum(0.0, 1.0 - rho),
    ])

    # 3. Bidirectional edges: orâ†’ex AND exâ†’or
    # Without this, 5 buses (those that only appear as or_bus, never ex_bus)
    # receive zero messages from neighbors â€” permanently isolated in message passing.
    src = np.concatenate([or_bus, ex_bus])
    dst = np.concatenate([ex_bus, or_bus])
    edge_index = np.array([src, dst])
    edge_attr  = np.concatenate([edge_attr_fwd, edge_attr_fwd], axis=0)

    return edge_index.tolist(), edge_attr.tolist()


N1_UNEVALUATED = -1


def build_line_targets(r, meta):
    """Per-LINE N-1 targets, aligned to the energized lines in `build_edges` order.

    `build_edges` keeps `line_status` lines in ascending line_id and emits them as
    the FIRST half of `edge_index` (the second half is the same lines reversed).
    These three vectors are aligned to that first half, so the model can read a
    per-line prediction straight off the forward edges.

      line_y     1 = tripping this line violates a thermal limit, 0 = secure
      line_mask  False where the label is MISSING (solver diverged). Never a
                 negative - masked entries must be excluded from the loss, not
                 taught as 'secure'.
      line_id    original grid2op line index, for per-line reporting

    De-energized lines carry -1 in the file and are absent from the graph
    entirely, so they drop out here without needing the mask.
    """
    status = np.array(r["line_status"], dtype=bool)
    line_id = np.nonzero(status)[0]
    viol = np.array(r["n1_violation"], dtype=np.int64)[line_id]
    return (np.maximum(viol, 0).astype(np.float32),
            viol != N1_UNEVALUATED,
            line_id)


def build_data(r, meta):
    """One JSONL record -> one PyG graph. Single definition shared by the lazy
    GridDataset and the preprocessor, so the two can never drift apart."""
    node_feats = build_node_features(r, meta)
    edge_index, edge_attr = build_edges(r, meta)

    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    line_y, line_mask, line_id = build_line_targets(r, meta)
    n_lines = len(line_id)

    # Marks the or->ex half of the bidirectional edge list. Batching concatenates
    # it per graph, so edge_index[:, edge_fwd] stays aligned with the
    # concatenated line_y across the whole batch.
    edge_fwd = torch.zeros(edge_index_t.shape[1], dtype=torch.bool)
    edge_fwd[:n_lines] = True

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=edge_index_t,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        line_y=torch.tensor(line_y, dtype=torch.float),
        line_mask=torch.tensor(line_mask, dtype=torch.bool),
        line_id=torch.tensor(line_id, dtype=torch.long),
        edge_fwd=edge_fwd,
        # frame-level summary: is ANY contingency insecure? reporting only
        y=torch.tensor(int(line_y[line_mask].any()) if line_mask.any() else 0,
                       dtype=torch.long),
    )


class GridDataset(Dataset):
    """Lazy-loading dataset — reads from .jsonl on demand."""
    def __init__(self, file_path, indices, meta: GridEnvMetadata):
        super().__init__()
        self.file_path = os.path.abspath(file_path)
        self.idx       = indices
        self.meta      = meta
        self.n_classes = 2

    def len(self):
        return len(self.idx)

    def get(self, idx):
        line_num = self.idx[idx] + 1
        line = linecache.getline(self.file_path, line_num)
        if not line:
            raise IndexError(f"Line {line_num} not found in {self.file_path}")
        return build_data(json.loads(line), self.meta)


class PreloadedGridDataset(InMemoryDataset):
    """In-memory dataset loaded from preprocessed .pt file."""
    def __init__(self, pt_file_path, device=None):
        super().__init__(root=None)
        _data, slices = torch.load(pt_file_path, weights_only=False)
        if device is not None:
            _data = _data.to(device)
        self._data  = _data
        self.slices = slices


if __name__ == "__main__":
    meta = GridEnvMetadata()
    print(f"Metadata loaded: n_sub={meta.n_sub}, n_line={meta.n_line}")
    print(f"NODE_FEATURES={NODE_FEATURES}, EDGE_FEATURES={EDGE_FEATURES}")
