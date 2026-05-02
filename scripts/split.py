import json
import numpy as np
import argparse
import sys

def load_labels(file_path):
    """Load only labels to save memory for splitting."""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            labels.append(record["label"])
    return labels

def get_splits(file_path, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42):
    labels = load_labels(file_path)
    n = len(labels)
    
    # Strict chronological split to prevent temporal leakage
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    idx = np.arange(n)
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    
    return train_idx, val_idx, test_idx

def compute_class_weights(labels, label_map):
    n_classes = len(label_map)
    counts = np.zeros(n_classes, dtype=np.float64)
    for l in labels:
        if l in label_map:
            counts[label_map[l]] += 1
    
    # Standard Inverse Class Frequency (ICF) weighting
    # Gives higher weight to minority classes proportional to their scarcity
    total_samples = counts.sum()
    weights = np.zeros_like(counts)
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total_samples / (n_classes * counts[i])
        else:
            weights[i] = 0.0
            
    return weights.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/grid_dataset_neurips2020.jsonl', help='Path to .jsonl dataset')
    parser.add_argument('--output_prefix', type=str, default='split_neurips2020', help='Prefix for output files')
    args = parser.parse_args()

    import os
    if not os.path.exists(args.input):
        print(f"Error: Dataset {args.input} not found.")
        sys.exit(1)

    print(f"Loading labels from {args.input}...")
    train_idx, val_idx, test_idx = get_splits(args.input)
    
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Save indices in the data directory
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, f"{args.output_prefix}_train_idx.npy"), train_idx)
    np.save(os.path.join(out_dir, f"{args.output_prefix}_val_idx.npy"), val_idx)
    np.save(os.path.join(out_dir, f"{args.output_prefix}_test_idx.npy"), test_idx)
    print(f"Split indices saved in {out_dir}/: {args.output_prefix}_*.npy")
