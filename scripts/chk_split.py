from collections import Counter
from split import load_labels
import numpy as np

all_labels = load_labels("data/grid_dataset_neurips2020.jsonl")
train_idx = np.load("data/split_neurips2020_train_idx.npy")
train_labels = [all_labels[i] for i in train_idx]
print(Counter(train_labels))