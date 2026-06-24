# GNN Architecture Upgrade Assessment
**Codebase:** `~/NeuroSym/T2530913-code` | **Target:** Macro F1 improvement

---

## Current State

**File:** `training/train_gnn.py`  
**Architecture:** 3-layer `GATConv` + `BatchNorm(track_running_stats=False)` + 3-pool concat (mean/max/min)  
**Config:** `hidden_channels=[64,128,128]`, `heads=[2,2,1]`, `dropout=0.0`  
**Known issue:** `GATConv` uses static attention (dot-product before concat) → rank collapse on small graphs (36 nodes)

---

## Candidate Architectures

### 1. GATv2Conv ✅ PRIORITY 1 — Do this first

**Why:** Fixes rank collapse. GATv2 computes attention *after* concatenation → dynamic, expressive attention on small graphs. Identical API to `GATConv`.

**Risk:** None. Edge features preserved (`edge_dim` param identical). SCU unaffected.

**Diff — `training/train_gnn.py`:**
```python
# FIND (3 occurrences):
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm

# REPLACE WITH:
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, BatchNorm

# FIND:
self.conv1 = GATConv(node_features, hidden_channels[0], heads=heads[0], edge_dim=edge_features)
self.conv2 = GATConv(hidden_channels[0] * heads[0], hidden_channels[1], heads=heads[1], edge_dim=edge_features)
self.conv3 = GATConv(hidden_channels[1] * heads[1], hidden_channels[2], heads=heads[2], edge_dim=edge_features)

# REPLACE WITH:
self.conv1 = GATv2Conv(node_features, hidden_channels[0], heads=heads[0], edge_dim=edge_features)
self.conv2 = GATv2Conv(hidden_channels[0] * heads[0], hidden_channels[1], heads=heads[1], edge_dim=edge_features)
self.conv3 = GATv2Conv(hidden_channels[1] * heads[1], hidden_channels[2], heads=heads[2], edge_dim=edge_features)
```

**Also try in `training/config.py`:** Scale up heads after confirming GATv2 trains stably:
```python
"hidden_channels": [64, 128, 128],
"heads": [4, 4, 1],   # was [2, 2, 1]
```

---

### 2. GraphSAGE ⚠️ PRIORITY 2 — Only if GATv2 fails

**Why:** No attention rank collapse, fast, strong inductive generalization.  
**Blocker:** `SAGEConv` has no `edge_dim` parameter. Your 4 edge features (`rho`, `p_or`, `q_or`, `line_status`) must be folded into node features manually.

**Edge feature workaround:** For each node, aggregate edge attrs from incident edges and concatenate to node feature vector before passing to SAGEConv. Increases `NODE_FEATURES` from 4 → 8.

**Diff — `scripts/pyg_data.py`:** Add edge-to-node aggregation in `build_graph()`.  
**Diff — `training/train_gnn.py`:**
```python
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

# In GridGNN.__init__:
self.conv1 = SAGEConv(node_features, hidden_channels[0])   # no edge_dim
self.conv2 = SAGEConv(hidden_channels[0], hidden_channels[1])
self.conv3 = SAGEConv(hidden_channels[1], hidden_channels[2])

# forward() drops edge_attr arg to convs:
x_emb = F.elu(self.conv1(x, edge_index))
x_emb = F.elu(self.conv2(x_emb, edge_index))
x_emb = F.elu(self.conv3(x_emb, edge_index))
```

**Config:** `hidden_channels=[64,128,128]`, no heads param needed.

---

### 3. RGATv2 (GATv2 + GRU) ❌ SKIP

**Why skip:** Your dataset is i.i.d. graph snapshots — no temporal ordering exists between records. GRU hidden state would carry meaningless carryover. Adds complexity with no signal gain.

---

### 4. DACDFE-GNN ❌ SKIP

**Why skip:** No PyG implementation exists. Requires custom CUDA message-passing kernels. Weeks of work. Out of thesis scope.

---

## Execution Order

```
Step 1: Apply GATv2Conv swap (3-line change)
Step 2: Train on Research PC — compare val macro F1 vs current baseline
Step 3: If F1 improves → tune heads [4,4,1], then hidden_channels [128,256,256]
Step 4: If F1 stagnates → apply GraphSAGE with edge-fold workaround
Step 5: Log results in iterations.md
```

---

## What NOT to touch

- Pooling strategy (mean+max+min concat) — already optimal for this task
- Z-score normalization pipeline — already correct
- BCEWithLogitsLoss localization head — independent of conv type
- SCU / Knowledge Graph — no changes needed for GATv2 or SAGEConv swap
- `dropout=0.0` — keep; determinism is load-bearing
- `autocast` — keep disabled; FP32 only
