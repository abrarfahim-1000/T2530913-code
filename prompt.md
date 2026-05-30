# Project Context: Grid2Op GNN Anomaly Detection

**Objective:** Train a Graph Neural Network (PyTorch Geometric) to detect and localize power grid anomalies (normal, line_trip, overload, cascade) using the `l2rpn_neurips_2020_track1_small` environment.

## Phase 1: Data Diagnosis (EDA)

**Initial Problem:** The model was stuck in mode collapse, classifying almost all states as `overload` while missing `normal` and `line_trip`. F1 scores were near zero.
**Investigation:** We built an Exploratory Data Analysis (EDA) script to extract scalar features (`max_rho`, `connected_line_frac`) and check for physical contradictions in the labels.
**Discoveries:**

1. **False Overloads:** 47% of states labeled `overload` had a maximum line capacity (`max_rho`) of `< 1.0`, meaning they were physically safe.
2. **Ghost Line Trips:** 100% of `line_trip` states looked exactly like `normal` states in terms of continuous power flow features.

## Phase 2: Data Generation Fixes (`generate_dataset.py`)

To fix the corrupt labels, we transitioned from "Event-Based" labeling to "Topology-Based" (N-1) labeling.

* **Removed Predictive Heuristic:** Deleted a bug where a line trip that caused a `rho > 0.75` was preemptively relabeled as an `overload`.
* **Topology Baseline:** Modified the step loop so that if a line is currently disconnected (`len(tripped_lines) > 0`), the baseline state defaults to `line_trip` instead of `normal`.
* **Balanced the Distribution:** Adjusted environment sampling to prevent the grid from getting stuck in permanent N-1 cooldown loops.
* `FAULT_PROB = 0.02`
* `RECONNECT_PROB = 0.20`
* `NORMAL_KEEP_PROB = 0.10` (Undersampled)
* `LINE_TRIP_KEEP_PROB = 0.20` (Undersampled redundant cooldown frames)


* **Result:** A mathematically pure dataset containing ~300k records (approx. 54% normal, 34% line_trip, 7% overload, 4% cascade) with **zero** contradictory labels.

## Phase 3: Graph Construction Fixes (`pyg_data.py`)

To ensure the PyTorch Geometric data loaders accurately reflected the physical grid:

* **The "Phantom Edge" Bug:** Tripped lines (`line_status = 0`) were still physically present in `edge_index`, allowing the GATConv to pass messages across severed power lines. We added a boolean mask (`connected_mask`) to physically delete tripped edges from the graphs.
* **Outlier Clipping:** Restored `np.clip` for active (`p_or`, $\pm 500$) and reactive (`q_or`, $\pm 300$) power flows to prevent massive voltage spikes during cascades from destroying Z-score normalization.

## Phase 4: GNN Architecture & Training Fixes (`train_gnn.py`)

Despite clean data, the model experienced extreme mode collapse during validation. We solved three consecutive PyTorch optimization traps:

1. **The Class Weight Trap:** Extreme loss multipliers (e.g., 10x penalty for cascade) terrified the model into never predicting `normal`. **Fix:** Removed `compute_class_weights` entirely. The natural 54/34/7/4 distribution is perfectly healthy for unweighted cross-entropy.
2. **The Pooling Blindspot:** `global_max_pool` could find overload spikes, but was blind to the *drops* in capacity caused by line trips. **Fix:** Concatenated `global_mean_pool`, `global_max_pool`, and `global_min_pool` (`-global_max_pool(-x)`) into the classifier.
3. **The Validation Magnitude Explosion:** The model achieved 0.48 training loss but crashed to 0.04 macro F1 during validation, even after a diagnostic dataset shuffle proved there was no chronological domain shift.
* **Root Cause 1:** `model.eval()` turned off the 30% dropout, causing 100% of nodes to fire. This exponentially increased the magnitude of the `global_max_pool` output, terrifying the classifier.
* **Root Cause 2:** Passing `dropout` to `GATConv` randomly severed topological attention edges during training, destroying the physical graph structure.
* **Root Cause 3:** `autocast` (Float16) caused softmax underflows to `0.0` inside the GAT attention mechanism.
* **The Final Fix:** * Stripped out `autocast` and `GradScaler` for pure FP32 training.
* Removed `dropout` from the `GATConv` layers.
* Added `nn.LayerNorm` after every convolution.
* **Crucial:** Added a master `nn.LayerNorm` *immediately after the concatenated pooling step* to guarantee the classifier receives a perfectly scaled `N(0,1)` distribution regardless of train/eval mode.


## Phase 5: The Validation Mode Collapse Debugging (Iterative Architecture Fixes)

Despite the Phase 4 diagnoses, the model continued to experience severe Train vs. Eval distribution shifts, successfully achieving excellent training losses (~0.11) while the validation Macro F1 repeatedly crashed to ~0.04 (collapsing entirely into the `overload` prediction).

Through an iterative debugging process, we identified three hidden mathematical traps and applied the following final fixes:

### 1. Removing All Dropout (Making the Model 100% Deterministic)

* **Root Cause A (The Min-Pool Trap):** `F.dropout(x, p=0.1)` applied to node features during training randomly forced 10% of node features to exactly `0.0`. The `global_min_pool` memorized these artificial zeros. During `.eval()`, the dropout turned off, the `0.0`s vanished, and the pooled embeddings experienced a massive upward shift in magnitude, instantly breaking the classifier.
* **Root Cause B (The 1.25x Scaling Trap):** The dense `nn.Dropout(p=0.2)` inside the classifier scaled active neurons by 1.25x during training. During evaluation, this scaling dropped to 1.0x. Because the decision boundaries for continuous power flow anomalies are extremely sharp, dropping the magnitude of the hidden layers by 20% in `eval()` pushed the logits completely out of bounds and into the `overload` threshold.
* **Fix:** Completely removed node, attention, and dense dropouts. The network is now 100% deterministic (`model.train()` and `model.eval()` execute identical math).

### 2. The Phantom Shuffle (Forced In-Memory Randomization)

* **Root Cause:** Despite writing a shuffle mechanism in `split.py`, the training script was loading stale, chronological `.npy` split arrays from a different disk path. This quietly forced the model to train exclusively on low-load "winter" scenarios and validate entirely on high-load "summer" scenarios, guaranteeing mode collapse.
* **Fix:** Implemented an **unconditional in-memory shuffle** directly inside `train_gnn.py` that concatenates `train_idx` and `val_idx`, shuffles them with a fixed seed (`42`), and redistributes them *before* applying the normalizations. This permanently annihilated the chronological domain shift.

### 3. The "Scale Destroyer" (Removing LayerNorm)

* **Root Cause:** In Phase 4, we introduced `nn.LayerNorm` to stabilize the network. However, in a power grid, an `overload` is defined entirely by its absolute magnitude (e.g., line capacity `rho > 1.0`). `LayerNorm` was standardizing extreme power flows down to `N(0,1)` and scaling up low flows, mathematically erasing the absolute magnitude. It made heavily overloaded lines look identical to safe ones, forcing the model to memorize topological artifacts to minimize training loss.
* **Fix:** Stripped all `LayerNorm` layers from the network entirely. The raw physical amplitudes now flow directly through the `GATConv` layers (activated by `ELU`) into the `global_max_pool`, preserving the extreme numeric spikes necessary for the linear classifier to detect overloads.

### Final Current Architecture (`GridGNN`)

* **Convolutions:** 3x `GATConv` layers (no dropout).
* **Activations:** `F.elu` for smooth continuous gradients.
* **Normalization:** Completely removed from the GNN (handled entirely by Z-score preprocessing on the dataset level).
* **Pooling:** Concatenated `global_mean_pool`, `global_max_pool`, and `global_min_pool` (`-global_max_pool(-x)`).
* **Precision & Loss:** Pure FP32 training with unweighted Cross-Entropy Loss.