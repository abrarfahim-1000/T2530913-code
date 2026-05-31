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

## Phase 6: Resolving Ghost Topology & Class Imbalance

**Initial Problem:** Despite forcing a deterministic architecture and removing normalization scaling traps, the model completely failed to predict `normal` and `line_trip` (0% recall), predicting only `overload` or `cascade`.
**Investigation:** We ran three new EDA scripts focusing on graph structure and discovered the **"Ghost Line Trip" bug**. A `line_trip` state had 37 nodes and 59 edges—the exact same topology as a `normal` state. The dataset generator was labeling lines as tripped without physically removing them from the graph. Because `GATConv` was still passing messages across the "dead" lines, a `line_trip` was mathematically indistinguishable from a `normal` state.

**Fixes Implemented:**

### 1. Topological Pruning (`scripts/pyg_data.py`)

* **The Fix:** Completely rewrote the `build_edges` function to dynamically prune the graph. It now uses the JSON's `line_status` array as a boolean mask to filter both `edge_index` and `edge_attr`.
* **Result:** A `line_trip` state now physically drops to 58 edges. This physically breaks the message-passing path in the `GATConv` layers, providing the GNN with the hard mathematical boundary it needs to separate a tripped grid from a normal grid.

### 2. The Label Enforcer (`scripts/generate_dataset.py`)

* **The Fix:** Added physical verification logic inside the simulation step loop.
* **Mechanism:** Grid2Op sometimes rejects fault injections (e.g., due to line cooldowns). The script now checks `obs.line_status` *after* the step. If the label says `line_trip` but the line is still physically connected, the script forcefully reverts the label to `normal`.

### 3. Aggressive Dataset Rebalancing (`scripts/generate_dataset.py`)

* **The Trap:** Implementing the Label Enforcer revealed the true distribution: because the script was throwing away 80% of successful line trips but keeping 10% of the vastly more common normal states, the generated dataset became 85% `normal`.
* **The Fix:**
* Increased `FAULT_PROB` to `0.05`.
* Crushed `NORMAL_KEEP_PROB` down to `0.02`.
* Increased `LINE_TRIP_KEEP_PROB` to `1.0` (hoarding 100% of successful trips).
* **Hard Quota:** Implemented a hard cap (`MAX_NORMAL_RECORDS = TARGET_RECORDS * 0.40`). Once `normal` states reach 40% of the dataset, the generator drops all subsequent normal steps and loops until it fills the remaining 60% with anomalies.

## Phase 7: Resolving Ghost Topology & The Cooldown Trap

**Initial Problem:** Despite forcing a deterministic architecture, the model completely failed to predict `normal` and `line_trip` (0% recall). An EDA topology check revealed the **"Ghost Line Trip" bug**: a `line_trip` state had 37 nodes and 59 edges—the exact same topology as a `normal` state. The GATConv layers were passing messages across dead lines, making the states mathematically indistinguishable. 

Furthermore, the dataset generator suffered from the **Cooldown Trap**. Grid2Op maintains line disconnections for many consecutive steps. Keeping 100% of these frames resulted in massive class imbalances (e.g., dataset skewing to 65% redundant line trips or 85% normal states depending on the sampling math).

**Fixes Implemented:**

### 1. Pure Topological Labeling (`generate_dataset.py`)
* **The Fix:** Removed predictive heuristics and complex history tracking (`prev_line_status`, `tripped_lines`). The label is now derived *exclusively* from the current physical frame:
  * N-0 (59 Lines) = `normal`
  * N-1 (58 Lines) = `line_trip`
  * N-k (≤57 Lines) = `cascade`
  * Any `rho >= 1.0` = `overload` (Supersedes topology)

### 2. Topological Pruning (`pyg_data.py`)
* **The Fix:** Completely rewrote the `build_edges` function to dynamically prune the graph. It uses the JSON's `line_status` array as a boolean mask to filter the static `edge_index` and `edge_attr`.
* **Result:** A `line_trip` state now physically drops to 58 edges. This breaks the message-passing path in the `GATConv` layers, providing the GNN with the hard mathematical boundary it needs to separate a tripped grid from a normal grid.

### 3. Aggressive Dataset Quotas (`generate_dataset.py`)
* **The Fix:** Implemented hard record quotas during generation to force perfect class balance and prevent cooldown/normal frame dominance:
  * `MAX_NORMAL_RECORDS = TARGET_RECORDS * 0.35`
  * `MAX_TRIP_RECORDS = TARGET_RECORDS * 0.25`
  * `MAX_CASCADE_RECORDS = TARGET_RECORDS * 0.20`
* **Result:** The generator smoothly fills these buckets, explicitly dropping redundant frames once a class quota is met, ensuring a perfectly distributed dataset for training.