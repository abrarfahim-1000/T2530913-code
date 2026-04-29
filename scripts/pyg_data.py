import torch
import numpy as np
import json
import grid2op
from torch_geometric.data import Data, Dataset, InMemoryDataset
import os
import linecache

ENV_NAME = "l2rpn_neurips_2020_track1_small"

# 4-class map — maintenance excluded (0 samples in current dataset)
LABEL_MAP = {"normal": 0, "overload": 1, "line_trip": 2, "cascade": 3}
RHO_CLIP  = 2.0


class GridEnvMetadata:
    def __init__(self, meta_dict=None):
        if meta_dict:
            self._init_from_dict(meta_dict)
        else:
            # Fallback to Grid2Op initialization (slow)
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

        # n_classes from meta (4 — no maintenance in current dataset)
        self.n_classes = meta_dict.get("n_classes", len(LABEL_MAP))
        # label_map from meta is always string-keyed (from JSON); use module LABEL_MAP
        # as the authoritative source so type consistency is guaranteed
        self.label_map = LABEL_MAP


def build_node_features(r, meta: GridEnvMetadata):
    load_p = np.array(r["load_p"], dtype=np.float32)
    gen_p  = np.array(r["gen_p"],  dtype=np.float32)
    v_or   = np.array(r["v_or"],   dtype=np.float32)
    rho    = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)

    v_or = np.nan_to_num(v_or, nan=0.0, posinf=0.0, neginf=0.0)
    rho  = np.nan_to_num(rho,  nan=0.0, posinf=0.0, neginf=0.0)

    node_load = np.zeros(meta.n_sub, dtype=np.float32)
    node_gen  = np.zeros(meta.n_sub, dtype=np.float32)
    node_v    = np.zeros(meta.n_sub, dtype=np.float32)
    node_rho  = np.zeros(meta.n_sub, dtype=np.float32)
    v_count   = np.zeros(meta.n_sub, dtype=np.float32)

    np.add.at(node_load, meta.load_to_sub, load_p)
    np.add.at(node_gen,  meta.gen_to_sub,  gen_p)

    np.add.at(node_v,     meta.line_or_bus, v_or)
    np.add.at(v_count,    meta.line_or_bus, 1)
    np.add.at(node_v,     meta.line_ex_bus, v_or)
    np.add.at(v_count,    meta.line_ex_bus, 1)

    np.maximum.at(node_rho, meta.line_or_bus, rho)
    np.maximum.at(node_rho, meta.line_ex_bus, rho)

    node_v = np.divide(node_v, v_count, out=np.zeros_like(node_v), where=v_count > 0)
    node_v = node_v / 150.0  # normalize kV → ~[0,1] range for NeurIPS env

    return np.stack([node_load, node_gen, node_v, node_rho], axis=1)


def build_edges(r, meta: GridEnvMetadata):
    rho  = np.clip(r["rho"], 0, RHO_CLIP).astype(np.float32)
    p_or = np.array(r["p_or"], dtype=np.float32)
    q_or = np.array(r["q_or"], dtype=np.float32)

    for arr in [rho, p_or, q_or]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    src = np.concatenate([meta.line_or_bus, meta.line_ex_bus])
    dst = np.concatenate([meta.line_ex_bus, meta.line_or_bus])
    edge_index = np.stack([src, dst], axis=0)

    feats     = np.stack([rho, p_or, q_or], axis=1)
    edge_attr = np.concatenate([feats, feats], axis=0)  # bidirectional — same features both dirs

    return edge_index, edge_attr


class GridDataset(Dataset):
    """
    Lazy-loading dataset — reads from .jsonl on demand.
    Use this when the .pt preprocessed file is not available.
    """
    def __init__(self, file_path, indices, meta: GridEnvMetadata):
        super().__init__()
        self.file_path = os.path.abspath(file_path)
        self.idx       = indices
        self.meta      = meta
        self.n_classes = len(LABEL_MAP)

    def len(self):
        return len(self.idx)

    def get(self, idx):
        line_num = self.idx[idx] + 1  # linecache is 1-based
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
    """
    Loads preprocessed .pt file into memory (and optionally onto GPU).
    Use after running preprocess.py.
    """
    def __init__(self, pt_file_path, device=None):
        super().__init__(root=None)
        
        # Load into _data first, then move — avoids triggering __getattr__ 
        # on an incomplete object and avoids weights_only warnings in newer PyTorch.
        _data, slices = torch.load(pt_file_path, weights_only=False)
        if device is not None:
            _data = _data.to(device)
        
        self._data   = _data
        self.slices  = slices


if __name__ == "__main__":
    meta = GridEnvMetadata()
    print(f"Metadata loaded for NeurIPS 2020: n_sub={meta.n_sub}, n_line={meta.n_line}")