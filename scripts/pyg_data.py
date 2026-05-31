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

# ── UPDATED DIMENSIONS ────────────────────────────────────────────────────────
# node: load_p, mean_v, max_rho, connected_line_frac  → 4 features
# edge: rho, p_or, q_or, line_status                  → 4 features
NODE_FEATURES = 4
EDGE_FEATURES = 4


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
    Per-bus node features (36 buses for NeurIPS env).

    Features:
      0  load_p              — total active load at bus
      1  mean_v              — mean voltage of connected lines (normalized)
      2  max_rho             — max line loading ratio at bus (key fault signal)
      3  connected_line_frac — fraction of lines at this bus that are still connected
                               0.0 = all lines tripped (cascade/trip signal)
                               1.0 = all lines healthy (normal signal)
    """
    # 1. Extract raw arrays (gen_p removed)
    load_p      = np.array(r["load_p"],      dtype=np.float32)
    v_or        = np.array(r["v_or"],        dtype=np.float32)
    rho         = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)
    line_status = np.array(r["line_status"], dtype=np.float32)  # 1=connected, 0=tripped

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
    node_v    = node_v / 150.0  # normalize kV → ~[0,1]

    node_conn_frac = np.divide(node_connected, node_total,
                               out=np.ones_like(node_connected),   # default 1.0
                               where=node_total > 0)

    # 5. Return stacked 4 features (node_gen removed)
    return np.stack([node_load, node_v, node_rho, node_conn_frac], axis=1)


def build_edges(r, meta):
    # 1. Convert line_status to a boolean mask (True = connected, False = disconnected)
    line_status = np.array(r["line_status"], dtype=bool)
    
    # 2. Get the full static topology from the meta object
    or_bus = meta.line_or_bus
    ex_bus = meta.line_ex_bus
    
    # 3. 🚨 CRITICAL: Prune the graph using the line_status mask
    # This physically removes the dead line from the message-passing path
    edge_index = np.array([or_bus[line_status], ex_bus[line_status]])
    
    # 4. Filter edge attributes so tripped lines provide zero signal
    rho = np.array(r["rho"])
    p_or = np.array(r["p_or"])
    q_or = np.array(r["q_or"])
    status_float = np.array(r["line_status"])
    
    # Stack features and apply the exact same active mask
    edge_attr = np.column_stack((rho, p_or, q_or, status_float))[line_status]
    
    return edge_index.tolist(), edge_attr.tolist()


class GridDataset(Dataset):
    """Lazy-loading dataset — reads from .jsonl on demand."""
    def __init__(self, file_path, indices, meta: GridEnvMetadata):
        super().__init__()
        self.file_path = os.path.abspath(file_path)
        self.idx       = indices
        self.meta      = meta
        self.n_classes = len(LABEL_MAP)

    def len(self):
        return len(self.idx)

    def get(self, idx):
        line_num = self.idx[idx] + 1
        line = linecache.getline(self.file_path, line_num)
        if not line:
            raise IndexError(f"Line {line_num} not found in {self.file_path}")
        r = json.loads(line)
        node_feats            = build_node_features(r, self.meta)
        edge_index, edge_attr = build_edges(r, self.meta)
        label     = LABEL_MAP.get(r["label"], 0)
        fault_loc = r["fault_loc"]
        return Data(
            x          = torch.tensor(node_feats,  dtype=torch.float),
            edge_index = torch.tensor(edge_index,  dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,   dtype=torch.float),
            y          = torch.tensor(label,        dtype=torch.long),
            fault_loc  = torch.tensor(fault_loc if fault_loc is not None else -1, dtype=torch.long),
        )


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