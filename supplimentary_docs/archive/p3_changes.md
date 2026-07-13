# Phase 3 Codebase Changes

**Date:** 2026-06-24  
**Session goal:** Diagnose and fix the GNN class-collapse bug (model only predicting cascade + overload, never line_trip or normal) and improve overall F1 scores.

---

## Root Cause Analysis

Three compounding issues were identified:

| # | Issue | Impact |
|---|-------|--------|
| 1 | **Unidirectional edges** in `pyg_data.py` | 5 buses (0, 1, 2, 5, 30) had zero incoming edges — permanently isolated in message passing |
| 2 | **Localization loss disabled** in `train_gnn.py` | No node-level gradient signal to distinguish line_trip (1 specific fault node) from cascade (fault_loc=-1) |
| 3 | **GATConv rank collapse** | Static attention on a 36-node graph collapses to near-uniform weights — reduced expressiveness |

The cascade+overload survival is explained by their detectability through global pooling alone (max_rho spike for overload, mean conn_frac drop for cascade), while line_trip and normal required local topology propagation that was broken.

---

## File Changes

### 1. `scripts/pyg_data.py` — Bidirectional edge construction

**What changed:** `build_edges()` was building only `or_bus → ex_bus` edges. Five buses that only appear as `or_bus` in the NeurIPS 2020 topology had zero incoming edges in 100% of records.

**Before:**
```python
edge_index = np.array([or_bus[line_status], ex_bus[line_status]])
edge_attr = np.column_stack((rho, p_or, q_or, status_float))[line_status]
return edge_index.tolist(), edge_attr.tolist()
```

**After:**
```python
or_bus = meta.line_or_bus[line_status]
ex_bus = meta.line_ex_bus[line_status]
rho        = np.array(r["rho"])[line_status]
p_or       = np.array(r["p_or"])[line_status]
q_or       = np.array(r["q_or"])[line_status]
stat_float = np.ones(line_status.sum(), dtype=np.float32)

edge_attr_fwd = np.column_stack((rho, p_or, q_or, stat_float))

# Bidirectional: or→ex AND ex→or
src = np.concatenate([or_bus, ex_bus])
dst = np.concatenate([ex_bus, or_bus])
edge_index = np.array([src, dst])
edge_attr  = np.concatenate([edge_attr_fwd, edge_attr_fwd], axis=0)

return edge_index.tolist(), edge_attr.tolist()
```

**Verified:** edge_index shape is now (2, 116) for 58 active lines. Zero buses with no incoming edges.

**Side effect:** `edge_feature_dim` in `build_meta()` corrected from 3 → 4, and node feature comment corrected (`gen_p` removed, `connected_line_frac` added).

**Action required before training:** Delete `data/processed_grid_data.pt` and rerun `scripts/preprocess.py` — the stale `.pt` file has old unidirectional edges baked in.

---

### 2. `training/train_gnn.py` — Three fixes

#### 2a. GATConv → GATv2Conv

**What changed:** All three GAT layers swapped from `GATConv` to `GATv2Conv`.

`GATConv` computes attention scores before concatenating source and target embeddings (static attention). On a small 36-node graph this causes rank collapse — attention weights flatten toward uniform. `GATv2Conv` computes attention after concatenation (dynamic attention), maintaining expressive capacity per the gnn_upgrade_assessment recommendation.

**Before:**
```python
from torch_geometric.nn import GATConv, ...
self.conv1 = GATConv(node_features, hidden_channels[0], heads=heads[0], edge_dim=edge_features)
self.conv2 = GATConv(hidden_channels[0] * heads[0], hidden_channels[1], heads=heads[1], edge_dim=edge_features)
self.conv3 = GATConv(hidden_channels[1] * heads[1], hidden_channels[2], heads=heads[2], edge_dim=edge_features)
```

**After:**
```python
from torch_geometric.nn import GATv2Conv, ...
self.conv1 = GATv2Conv(node_features, hidden_channels[0], heads=heads[0], edge_dim=edge_features)
self.conv2 = GATv2Conv(hidden_channels[0] * heads[0], hidden_channels[1], heads=heads[1], edge_dim=edge_features)
self.conv3 = GATv2Conv(hidden_channels[1] * heads[1], hidden_channels[2], heads=heads[2], edge_dim=edge_features)
```

#### 2b. Localization loss re-enabled

**What changed:** The training loop previously had `loss = cls_loss` with `loc_loss` silently discarded. The `build_loc_targets_fast()` function existed but was never called.

Without localization loss, the classification head receives no gradient signal encoding *which specific bus* is the fault source. Line_trip and cascade have structurally similar global pool embeddings when loc_loss is absent — the model collapses them.

**Before:**
```python
cls_loss = F.cross_entropy(class_logits, batch.y, weight=class_weights)
loss = cls_loss  # loc_loss detached
```

**After:**
```python
cls_loss = F.cross_entropy(
    class_logits, batch.y,
    weight=class_weights,
    label_smoothing=0.1,
)
loc_targets = build_loc_targets_fast(batch)
loc_loss    = loc_loss_fn(loc_logits, loc_targets)
loss        = cls_loss + TRAIN_CONFIG["loc_loss_weight"] * loc_loss
```

#### 2c. Label smoothing added (smoothing=0.1)

**What changed:** `label_smoothing=0.1` added to `F.cross_entropy`.

Without smoothing, the model becomes overconfident on majority-class boundaries (overload, cascade) in early epochs. Once those logits saturate, gradient for normal and line_trip becomes negligibly small and those classes never recover.

---

### 3. `scripts/generate_dataset.py` — Multi-environment support + metadata fixes

**What changed:**

- Added `ENV_CONFIGS` dict with entries for `neurips`, `case14`, and `wcci`.
- Added `--env` argparse argument (choices: neurips, case14, wcci; default: neurips).
- Wired `--env` selection into `main()` via `global ENV_NAME, ENV_TAG, ENV_DESC`.
- Fixed `build_meta()`: `edge_feature_dim` corrected from 3 → 4; node feature comment corrected to reflect actual 4 features (load_p, mean_v, max_rho, connected_line_frac).

**Usage after change:**
```bash
python scripts/generate_dataset.py --env neurips   # 36-bus training set
python scripts/generate_dataset.py --env case14    # 14-bus cross-topology test
python scripts/generate_dataset.py --env wcci      # 118-bus cross-topology test
```

---

### 4. `CLAUDE.md` — Full rewrite

Rewritten from scratch to reflect the finalized system architecture documented in `study.md`. Key additions:

- Correct GNN architecture table (GATv2Conv, triple pooling, BatchNorm track_running_stats=False, no dropout)
- LLM pipeline finalized results (469 unique rules, extraction/validation counts)
- KG statistics (587 nodes, 6,632 edges, edge type breakdown)
- Shield validation pseudocode (`validate()`, `evaluate_condition()`)
- Voltage translation constant (150.0 kV nominal)
- Hardware allocation table (Research PC vs Personal PC)
- Cross-topology failure mode taxonomy
- Key implementation constants table

---

## Steps Required to Apply Changes

```bash
# 1. Delete stale preprocessed data (has old unidirectional edges)
del data\processed_grid_data.pt

# 2. Regenerate with bidirectional edges
python scripts/preprocess.py

# 3. Train with all fixes applied
python training/train_gnn.py
```

The normalization stats (`data/normalization_stats.pt`) will be regenerated during training from the new training split.
