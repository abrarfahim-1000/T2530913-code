import os
import json
import sys
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.pyg_data import GridEnvMetadata, build_data, LABEL_MAP
from training.config import DATA_FILE, DATA_DIR, PROCESSED_PT


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

    out_file = os.path.join(DATA_DIR, PROCESSED_PT)
    print(f"Processing {DATA_FILE}...")

    data_list = []
    skipped = 0
    n_lines = n_masked = 0

    with open(DATA_FILE, 'r') as f:
        for line in tqdm(f):
            r = json.loads(line)

            # `label` is the 4-class current-state label, retained on every record
            # for reference only — it is NOT the training target. Records carrying
            # an unknown one (e.g. 'maintenance' from an older run) are skipped.
            if r["label"] not in LABEL_MAP:
                skipped += 1
                continue

            if "n1_violation" not in r:
                raise KeyError(
                    f"{DATA_FILE} has no `n1_violation` — it was not generated with "
                    "`--task n1`. Regenerate:\n"
                    "  python scripts/generate_dataset.py --env neurips --task n1 "
                    "--n_records 12000"
                )

            data = build_data(r, meta)
            n_lines += int(data.line_mask.numel())
            n_masked += int((~data.line_mask).sum())
            data_list.append(data)

    if skipped:
        print(f"Skipped {skipped} records with unknown labels.")
    if n_lines:
        print(f"N-1 contingencies: {n_lines:,}   unevaluated (masked out of the "
              f"loss): {n_masked:,} ({n_masked / n_lines:.2%})")

    print(f"Collating {len(data_list)} graphs...")
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), out_file)
    print(f"Saved -> {out_file}")

if __name__ == "__main__":
    preprocess_data()