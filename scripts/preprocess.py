import os
import json
import sys
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.pyg_data import GridEnvMetadata, build_node_features, build_edges, LABEL_MAP
from training.config import DATA_FILE, DATA_DIR


def preprocess_data():
    meta_path = DATA_FILE.replace(".jsonl", "_meta.json")
    if os.path.exists(meta_path):
        print(f"Loading metadata from {meta_path}...")
        with open(meta_path, 'r') as f:
            meta_dict = json.load(f)
        meta = GridEnvMetadata(meta_dict)
    else:
        print("Warning: meta JSON not found — falling back to Grid2Op init (slow).")
        meta = GridEnvMetadata()

    # Always use the module-level LABEL_MAP (string-keyed, 4-class, no maintenance).
    # Do NOT use meta_dict["label_map"] directly — it may contain the 5-class
    # definition from an older generation run that included maintenance.
    label_map = LABEL_MAP

    out_file = os.path.join(DATA_DIR, "processed_grid_data.pt")
    print(f"Processing {DATA_FILE}...")
    print(f"Label map in use: {label_map}")

    data_list = []
    skipped   = 0

    with open(DATA_FILE, 'r') as f:
        for line in tqdm(f):
            r = json.loads(line)

            label_str = r["label"]
            if label_str not in label_map:
                # Skip any records with labels not in the current map
                # (e.g. 'maintenance' records from mixed-run data files)
                skipped += 1
                continue

            node_feats            = build_node_features(r, meta)
            edge_index, edge_attr = build_edges(r, meta)

            label     = label_map[label_str]
            fault_loc = r["fault_loc"]

            data = Data(
                x          = torch.tensor(node_feats,  dtype=torch.float),
                edge_index = torch.tensor(edge_index,  dtype=torch.long),
                edge_attr  = torch.tensor(edge_attr,   dtype=torch.float),
                y          = torch.tensor(label,        dtype=torch.long),
                fault_loc  = torch.tensor(fault_loc if fault_loc is not None else -1, dtype=torch.long),
            )
            data_list.append(data)

    if skipped:
        print(f"Skipped {skipped} records with unknown labels.")

    print(f"Collating {len(data_list)} graphs...")
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), out_file)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    preprocess_data()