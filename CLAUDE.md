# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A neuro-symbolic fault detection system for power grids combining:
- **GNN (Neural Layer):** Graph Attention Network trained on 36-bus NeurIPS 2020 topology
- **LLM Pipeline (Component B):** Qwen3-14B + Nemotron-3 Nano 30B extract rules from IEEE/NERC standards
- **Knowledge Graph (Component C):** 587 nodes, 6,632 edges — Bus, Line, Generator, Grid, and 469 Rule nodes
- **Symbolic Shield (Component D):** Hard inference-time gate — every GNN prediction validated against KG rules before output

The GNN is trained on one topology (36-bus) and evaluated on unseen topologies (14-bus, 118-bus) to measure cross-topology generalization. The shield's rule compliance rate is expected to remain stable while GNN accuracy degrades.

---

## Development Commands

### Environment Setup
```bash
# Install PyTorch with CUDA support (RTX 4080 Super)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch Geometric dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install remaining dependencies
pip install -r requirements.txt
```

### Dataset Generation
```bash
# Generate training dataset (NeurIPS 2020, 36-bus, ~300k records)
python scripts/generate_dataset.py --env neurips --n_records 300000

# Generate cross-topology test sets (~15-20k records each)
python scripts/generate_dataset.py --env case14
python scripts/generate_dataset.py --env wcci
```

### LLM Rule Extraction
```bash
# Phase 1: Extract candidate rules (Qwen3-14B)
python kg/extract_rules.py --docs_dir data/documents/

# Phase 2: Validate candidates (Nemotron-3 Nano 30B) — run after extraction completes
python kg/validate_rules.py --candidates data/rules_candidates.jsonl

# Build knowledge graph from validated rules
python kg/build_kg.py --rules data/all_rules_deduped.jsonl
```

### Model Training
```bash
# Train GNN (36-bus NeurIPS 2020 — only topology used for training)
python training/train_gnn.py

# Key flags
python training/train_gnn.py --epochs 100 --batch_size 256 --lr 5e-4
```

### Cross-Topology Evaluation
```bash
# Evaluate trained GNN + Shield on unseen topologies (no retraining)
python evaluation/eval_cross_topology.py --env case14 --checkpoint gnn_checkpoint_neurips_best.pt
python evaluation/eval_cross_topology.py --env wcci   --checkpoint gnn_checkpoint_neurips_best.pt
```

### Data Inspection
```bash
# Inspect dataset statistics
python scripts/inspect.py [data_file_path]
# Default: data/grid_dataset_neurips2020.jsonl
```

### Code Quality
```bash
black .
isort .
ruff check .
```

---

## Project Architecture

### Directory Structure
```
data/
  grid_dataset_neurips2020.jsonl   # 300k training records (36-bus)
  grid_dataset_case14.jsonl        # ~15k test records (14-bus, unseen)
  grid_dataset_wcci2022.jsonl      # ~20k test records (118-bus, unseen)
  normalization_stats.pt           # z-score stats from 36-bus training split — required at inference
  documents/                       # IEEE/NERC/FERC/AEMO PDFs for LLM extraction

kg/
  extract_rules.py                 # Qwen3-14B extraction pass
  validate_rules.py                # Nemotron-3 Nano 30B validation pass
  build_kg.py                      # Assembles NetworkX DiGraph from validated rules
  shield.py                        # Symbolic validation logic — validate(), evaluate_condition()
  knowledge_graph.graphml          # Serialized KG (587 nodes, 6,632 edges)
  all_rules_deduped.jsonl          # 469 unique rules after dedup

scripts/
  generate_dataset.py              # Grid2Op simulation → JSONL records
  inspect.py                       # Dataset statistics
  pyg_data.py                      # PyTorch Geometric dataset wrapper
  split.py                         # Chronic-level train/val/test splitting + class weights

training/
  train_gnn.py                     # Main training script

evaluation/
  eval_cross_topology.py           # Cross-topology inference + shield + failure mode logging
  failures_case14.jsonl            # BLOCK log for 14-bus evaluation
  failures_wcci.jsonl              # BLOCK log for 118-bus evaluation
```

---

## Component A — GNN Architecture

### Model: GridGNN (Graph Attention Network)

```
Input: (n_nodes × 4 node features, n_edges × 4 edge features)
    ↓
GATConv(4 → 64, heads=2)   + BatchNorm(track_running_stats=False) + ELU
    ↓
GATConv(128 → 128, heads=2) + BatchNorm(track_running_stats=False) + ELU
    ↓
GATConv(256 → 128, heads=1) + BatchNorm(track_running_stats=False) + ELU
    ↓
global_mean_pool ‖ global_max_pool ‖ global_min_pool  →  (384,)
    ↓
Classifier MLP  →  (4,) class logits       [normal / overload / line_trip / cascade]
Localizer MLP   →  (n_nodes,) per-bus fault probability  [DISABLED for cross-topology eval]
```

**Critical implementation notes:**
- `BatchNorm(track_running_stats=False)` — live batch stats, not running stats. Running stats flatten overload spikes (rho > 1.0) during eval mode.
- **No dropout** — dropout severs attention edges and creates train/eval scaling gaps on power flow features.
- **Triple pooling** — max captures overload spikes, min captures connectivity drops, mean captures baseline state.
- **Tripped lines are pruned from `edge_index`** using `line_status` boolean mask at graph construction. Without pruning, line_trip states are structurally identical to normal states.

### Node Features (4 per bus)

| Feature | Construction | Signal |
|---|---|---|
| `load_p` | Sum of active loads at bus | Demand |
| `mean_v` | Mean voltage of connected lines / 150.0 | Voltage health |
| `max_rho` | Max loading ratio of connected lines | Overload |
| `connected_line_frac` | Fraction of lines still connected | Trip/cascade |

`gen_p` was removed — EDA showed 1.00 correlation with `load_p` (generation matches load by power flow law).

### Edge Features (4 per line)

`[rho, p_or, q_or, line_status]`

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR, T_max=50 |
| Batch size | 256 |
| Loss | Weighted CrossEntropy (ICF weights, sqrt-smoothed) |
| Early stopping | patience=15, min_delta=0.001 |
| Primary metric | **Macro F1** (not accuracy — dataset is imbalanced) |
| Normalization | Z-score from training split only → saved to `data/normalization_stats.pt` |

**`normalization_stats.pt` must be saved after training.** It is loaded at inference time for all topologies (including 14-bus and 118-bus). Foreign topologies are normalized with 36-bus stats — intentional, physical quantities have the same scale.

**In-memory shuffle required** before training to eliminate chronological domain shift:
```python
combined_idx = np.concatenate([train_idx, val_idx])
np.random.seed(42)
np.random.shuffle(combined_idx)
```

### Cross-Topology Inference

The classification head runs unchanged on foreign topologies (global pooling is topology-agnostic). The localization head is **disabled** for cross-topology evaluation (outputs `n_nodes` logits, which vary by topology).

---

## Component B — LLM Knowledge Extraction

### Models

- **Extractor:** Qwen3-14B via Ollama — `/no_think` prefix enforced, outputs JSON rule arrays
- **Validator:** Nemotron-3 Nano 30B (A3B MoE) via Ollama — `think=False` enforced, confirms/corrects/rejects each candidate

Sequential only — never load both models simultaneously. `keep_alive=0` on every Ollama call to release VRAM immediately.

### Pipeline

```
PDF chunks → Qwen3-14B → JSON candidates → Pydantic validation → Nemotron-3 Nano 30B → CONFIRM/CORRECT/REJECT → dedup → KG
```

### Finalized Results

| Stage | Count |
|---|---|
| Candidate rules extracted | 1,372 |
| Confirmed (CONFIRM) | 50 |
| Corrected and retained (CORRECT) | 451 |
| Rejected (REJECT) | 766 |
| Flagged for review | 105 |
| **Unique rules after dedup** | **469** |

### Rule Schema

```json
{
  "rule_id": "R_042",
  "source": "IEEE Std 1547-2018, Section 7.4",
  "entity": "Bus",
  "condition": "voltage_pu > 1.05 OR voltage_pu < 0.95",
  "action": "BLOCK",
  "severity": "critical",
  "explanation": "Voltage deviation beyond ±5% nominal violates IEEE 1547 interconnection standards."
}
```

### Implementation Notes

- `strip_think()` regex applied to all model output before JSON parsing — Qwen3 occasionally leaks `<think>` tokens despite `/no_think`
- GBNF grammar enforcement on array output prevents malformed JSON
- Rule IDs are globally sequential (`R_001`, `R_002`, ...) across all chunks/documents before validation

---

## Component C — Knowledge Graph

**Backend:** NetworkX DiGraph (pure Python, zero setup, sufficient at this scale)

### Statistics

| Metric | Value |
|---|---|
| Total nodes | 587 |
| Bus nodes | 36 |
| Line nodes | 59 |
| Generator nodes | 22 |
| Grid node | 1 (singleton — anchors system-level rules) |
| Rule nodes | 469 |
| Total directed edges | 6,632 |
| `connected_to` edges | 118 |
| `part_of` edges | 139 |
| `has_rule` edges | 6,375 |

### Edge Types

| Edge | Meaning |
|---|---|
| `connected_to` | Bus ↔ Line (physical topology) |
| `part_of` | Bus → Grid, Line → Grid, Generator → Bus |
| `has_rule` | Entity → Rule |

The `Grid` node anchors system-level rules. At inference, the shield retrieves rules from both the predicted fault entity AND the Grid node:
```python
applicable = rules_of(predicted_entity) + rules_of("Grid")
```

The KG is built from IEEE standards — it is **not** built from the NeurIPS 2020 topology. Rules are topology-agnostic and apply to 14-bus and 118-bus environments unchanged.

---

## Component D — Symbolic Validation Shield

Every GNN prediction passes through the shield. There is no bypass mode.

### Validation Flow

```python
def validate(prediction_context: dict, KG) -> dict:
    fault_type       = prediction_context["fault_type"]
    applicable_rules = rules_of(KG, fault_type) + rules_of(KG, "Grid")

    violated = [r for r in applicable_rules if evaluate_condition(r["condition"], prediction_context)]

    if not violated:
        return {"status": "PASS", "fault_type": fault_type, ...}

    violated.sort(key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}[r["severity"]])
    return {"status": "BLOCK", "violated_rules": violated, "explanation": build_explanation(violated)}


def evaluate_condition(condition: str, context: dict) -> bool:
    namespace = {
        "voltage_pu":      min(context["v_or"]) / 150.0,   # Grid2Op kV → per-unit
        "loading_pct":     context["rho_max"] * 100,
        "rho":             context["rho_max"],
        "n_tripped_lines": context["n_tripped_lines"],
        "line_status":     not all(context["line_status"]),
    }
    try:
        return bool(eval(condition, {"__builtins__": {}}, namespace))
    except Exception:
        return False  # malformed condition → do not block
```

**Voltage translation:** `voltage_pu = v_or_kv / 150.0` (nominal for NeurIPS 2020 environment). Update this constant from the environment's meta JSON when evaluating on other environments.

### Cross-Topology Failure Mode Logging

Every BLOCK from cross-topology evaluation is logged to `failures_<topo>.jsonl` with failure mode classification:

| Failure Mode | Definition |
|---|---|
| Overconfident wrong | GNN predicts "normal" with confidence > 0.85 during a physical fault |
| Class confusion | GNN detects fault but confuses type (e.g., overload predicted as cascade) |
| Threshold failure | Correct fault type, but predicted entity violates physical thresholds |
| Novel topology state | Sub-graph structure never seen during training |

---

## Data Collection

### Environment Role Assignment

| Environment | Role | Buses | Lines | Records |
|---|---|---|---|---|
| `l2rpn_neurips_2020_track1_small` | **Training only** | 36 | 59 | 300,000 |
| `rte_case14_sandbox` | **Test only — unseen smaller** | 14 | 20 | ~15,000 |
| `l2rpn_wcci_2022` | **Test only — unseen larger** | 118 | 186 | ~20,000 |

No training on case14 or WCCI 2022. The trained checkpoint is applied directly.

### Training Dataset Class Distribution

| Label | Count | % |
|---|---|---|
| Normal | 61,444 | 20.48% |
| Overload | 103,556 | 34.52% |
| Line Trip | 75,000 | 25.00% |
| Cascade | 60,000 | 20.00% |

**Chronic-level splitting only** (70/15/15). Frame-level splitting leaks cascade sequences across splits.

### Labeling Logic

```python
def get_state_label(obs, env):
    max_rho      = obs.rho.max() if len(obs.rho) > 0 else 0.0
    active_lines = int(np.sum(obs.line_status))

    if max_rho >= 1.0:
        return "overload", int(obs.rho.argmax())
    if active_lines == env.n_line:
        return "normal", -1
    if active_lines == env.n_line - 1:
        return "line_trip", int(np.where(~obs.line_status)[0][0])
    return "cascade", -1
```

**`NO_OVERFLOW_DISCONNECTION = False`** — must be passed at `grid2op.make()` via `param=params`, not set post-make. Enables natural cascades.

---

## Hardware

### Research PC — All Production Runs

- RTX 4080 Super (16GB VRAM), i7-14700K, 64GB DDR5
- Runs: GNN training, all dataset generation, LLM extraction (Qwen3-14B then Nemotron-3 Nano 30B, sequential), cross-topology evaluation
- Schedule: Saturdays, Mondays, Wednesdays

### Personal PC — Development Only

- Intel Arc B580 (12GB VRAM), Ryzen 5 7500F, 16GB DDR5
- Runs: Code development, unit tests, shield logic — no training, no LLM inference

**Nemotron-3 Nano 30B note:** ~18–20GB total (partial CPU offload into 64GB RAM). Qwen3-14B runs first (~8–10GB, fully GPU-resident), then Nemotron-3, never concurrent.

---

## Key Implementation Constants

| Constant | Value | Context |
|---|---|---|
| `num_workers` | `0` | Windows PyG DataLoader constraint |
| Nominal voltage | `150.0 kV` | NeurIPS 2020 environment — used in voltage_pu conversion |
| FAULT_PROB | `0.10` | Grid2Op fault injection rate |
| RECONNECT_PROB | `0.09` | Grid2Op reconnect probability |
| Random seed | `42` | In-memory shuffle before training |
| Checkpoint filename | `gnn_checkpoint_neurips_best.pt` | Saved on best macro F1 |
| Normalization stats | `data/normalization_stats.pt` | Must exist before cross-topology eval |

---

## Dependencies

- PyTorch 2.8.0 + CUDA 12.8
- PyTorch Geometric (GAT, global pooling, DataLoader)
- grid2op + lightsim2grid (LightSimBackend — ~10x faster than PandaPower)
- ollama (local LLM inference — Qwen3-14B, Nemotron-3 Nano 30B)
- networkx (KG backend — DiGraph)
- pydantic (rule and verdict schema validation)
- pdfplumber (PDF ingestion for LLM pipeline)
- numpy, scikit-learn (data processing)
- wandb (experiment tracking — macro F1, per-class F1, confusion matrices)
- tqdm
