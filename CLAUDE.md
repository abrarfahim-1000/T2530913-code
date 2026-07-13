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

## Current Status (2026-07-13)

**Component A (GNN) — CLOSED.** No further training-side experimentation is planned. The deployed
result is **Lever A** (post-hoc per-class logit-margin calibration): calibrated test macro F1
**0.8277**, `normal` recall **0.8806** (36-bus, in-distribution). Three rounds of architecture/loss/
attention experiments (data regeneration, resampling, alternate pooling, a two-stage head, soft-F1
loss, GSAT stochastic edge gating) were all tried and rejected — full index in
[`supplimentary_docs/gnn_final_results.md`](supplimentary_docs/gnn_final_results.md),
which is written to double as the thesis negative-results section. Deployed artifacts
(`gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt`, `gnn_logit_margin.json`,
`data/normalization_stats.pt`) are frozen — do not retrain or recalibrate against them without a new,
previously-untried hypothesis.

**Component B (LLM extraction) — DONE.** 469 deduplicated rules in `rules/all_rules_deduped.jsonl`
(see §Component B for the extraction pipeline stats). Not being revisited.

**Component C (Knowledge Graph) — DONE.** Built from the 469 rules; artifact is `kg/knowledge_graph.pkl`.

**Component D (Symbolic Shield) — NOT YET STARTED. This is the active next phase.** No `shield.py` or
equivalent validation module exists anywhere in this repo yet — the shield pseudocode in this file and
the worked examples in `supplimentary_docs/inference.md` are design references, not implemented code.
`evaluation/eval_cross_topology.py` currently only runs GNN classification metrics on foreign
topologies; it does not call a shield, and the `failures_<topo>.jsonl` BLOCK logs described later in
this file do not exist yet either. Design blueprint: `supplimentary_docs/study3(integration).md`
(full walkthrough — observation→GNN→shield wiring, KG rule retrieval, condition evaluation) and
`supplimentary_docs/shield_necessity_analysis_report.md` (the cross-topology generalization framing
for why this phase matters to the thesis). Build this against the frozen GNN (Component A) and the
existing KG (Component C) — neither should need to change to support it.

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
# Phase 1: Extract candidate rules (Qwen3-14B) — writes *_candidates.jsonl to --out (default "rules")
python extraction/extract.py --docs data/documents/ --out rules/

# Phase 2: Validate candidates (Nemotron-3 Nano 30B) — run after extraction completes
python extraction/validate.py --candidates rules/

# Build knowledge graph from validated rules (writes into --out-dir, default "kg")
python extraction/build_kg.py --rules rules/all_rules_deduped.jsonl --out-dir kg/
```
Already run once — outputs are the committed `rules/all_rules_deduped.jsonl` (469 rules) and
`kg/knowledge_graph.pkl`. Not being re-run unless new source documents are added.

### Model Training
```bash
# Train GNN (36-bus NeurIPS 2020 — only topology used for training)
python training/train_gnn.py

# Key flags
python training/train_gnn.py --epochs 100 --batch_size 256 --lr 5e-4
```

### Cross-Topology Evaluation
```bash
# GNN-only classification metrics on unseen topologies (no retraining, no shield — see Current Status)
python evaluation/eval_cross_topology.py --tag case14
python evaluation/eval_cross_topology.py --tag wcci2022
# checkpoint defaults to gnn_checkpoint_best.pt; override with --checkpoint / margin with --margin
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
  grid_dataset_*_meta.json         # per-environment metadata (n_sub, n_line, ...)
  processed_grid_data.pt           # preprocessed PyG tensors (scripts/preprocess.py)
  split_neurips2020_{train,val,test}_idx.npy   # chronic-level split indices
  normalization_stats.pt           # z-score stats from 36-bus training split — required at inference
  documents/                       # IEEE/NERC/FERC/AEMO PDFs for LLM extraction (not present on every machine)

extraction/                        # Component B — LLM rule extraction pipeline
  extract.py                       # Qwen3-14B extraction pass → rules/*_candidates.jsonl
  validate.py                      # Nemotron-3 Nano 30B validation pass → *_confirmed/*_flagged
  build_kg.py                      # Assembles NetworkX DiGraph from validated rules → kg/
  common.py                        # Shared prompt/schema helpers

rules/                             # Component B outputs (one *_candidates/_confirmed/_flagged.jsonl per source doc)
  all_rules_deduped.jsonl          # 469 unique rules after dedup — the artifact Component C is built from
  validation_run_summary.json      # per-document confirm/correct/reject counts from the validation pass

kg/                                # Component C — knowledge graph artifacts (built by extraction/build_kg.py)
  knowledge_graph.pkl              # Serialized KG (587 nodes, 6,632 edges)
  kg_full.html / rules_only.html / rule_R_001_subgraph.html   # pyvis visualizations
  severity_breakdown.png, rule_local_sample_2d.{png,svg,pdf}  # EDA plots
  # NOTE: no shield.py here (or anywhere) yet — Component D is not implemented, see Current Status.

scripts/
  generate_dataset.py              # Grid2Op simulation → JSONL records
  preprocess.py                    # JSONL → processed_grid_data.pt
  pyg_data.py                      # PyTorch Geometric dataset wrapper (GridDataset, feature builders)
  split.py                         # Chronic-level train/val/test splitting + class weights
  inspect_data.py, audit_datasets.py, chk_split.py, diag.py   # ad hoc inspection utilities

training/
  train_gnn.py                     # Main training script (GridGNN) — reverted to pre-Round-3 state; GSAT/soft-F1 removed, see closure doc
  config.py                        # TRAIN_CONFIG, auto-selected by device (cuda vs personal-PC)
  calibrate_margin.py              # Lever A post-hoc logit-margin calibration/report tool

evaluation/
  eval_cross_topology.py           # GNN-only classification metrics on unseen topologies (no shield yet)
```

---

## Component A — GNN Architecture

### Model: GridGNN (Graph Attention Network v2)

```
Input: (n_nodes × 5 node features, n_edges × 4 edge features)
    ↓
GATv2Conv(5 → h0, heads=k0, edge_dim=4)  + BatchNorm(track_running_stats=False) + ELU
    ↓
GATv2Conv(h0*k0 → h1, heads=k1, edge_dim=4) + BatchNorm(track_running_stats=False) + ELU
    ↓
GATv2Conv(h1*k1 → h2, heads=k2, edge_dim=4) + BatchNorm(track_running_stats=False) + ELU
    ↓
global_mean_pool ‖ global_max_pool ‖ global_min_pool  →  (h2*3,)
    ↓
Classifier MLP  →  (4,) class logits       [normal / overload / line_trip / cascade]
Localizer MLP   →  (n_nodes,) per-bus fault probability  [DISABLED for cross-topology eval]
    ↓
Lever A: argmax(class_logits + per-class_margin)   [inference-time only, see below]
```
`(h0,h1,h2)` and head counts `(k0,k1,k2)` come from `TRAIN_CONFIG` — see **Training Configuration**
below; the DEPLOYED checkpoint uses `[16,32,32]`/`heads=[4,4,1]` (the personal-PC/non-CUDA branch).

**Critical implementation notes:**
- **GATv2Conv, not GATConv.** GATConv's static attention (scored before concatenation) rank-collapses
  on small graphs like this 36-node grid; GATv2Conv scores attention after concatenation, preserving
  expressiveness. This is load-bearing, confirmed by `supplimentary_docs/archive/gnn_upgrade_assessment.md`.
- `BatchNorm(track_running_stats=False)` — live batch stats, not running stats. Running stats flatten overload spikes (rho > 1.0) during eval mode.
- **No dropout** — dropout severs attention edges and creates train/eval scaling gaps on power flow features.
- **Triple pooling** — max captures overload spikes, min captures connectivity drops, mean captures baseline state. This pooling operator is also the diagnosed bottleneck for `normal`/`line_trip` confusion — see `supplimentary_docs/gnn_final_results.md` §3.
- **Tripped lines are pruned from `edge_index`** using `line_status` boolean mask at graph construction. Without pruning, line_trip states are structurally identical to normal states.
- **Lever A (deployed, inference-time only):** a fixed per-class logit offset (`normal +0.30`,
  `line_trip −0.10`, others `0`, in `gnn_logit_margin.json`) is added to the raw logits before argmax.
  This is what takes macro F1 from 0.7830 → 0.8277 and `normal` recall from 0.65 → 0.88 — see
  `training/calibrate_margin.py` and the closure doc for why this, and not an architecture change, is
  the thing that worked. **Does not transfer cross-topology** — apply only to 36-bus in-distribution eval.
- **GSAT (stochastic edge gating) and a soft-F1 loss term were tried and rejected in Round 3** — the
  code has since been removed from `train_gnn.py`/`config.py` (reverted to pre-Round-3 state) now that
  the results doc has captured the results. Retrieve from git history before revisiting; see
  `supplimentary_docs/gnn_final_results.md`.

### Node Features (5 per bus)

| Feature | Construction | Signal |
|---|---|---|
| `load_p` | Sum of active loads at bus | Demand |
| `mean_v` | Mean voltage of connected lines / 150.0 | Voltage health |
| `max_rho` | Max loading ratio of connected lines | Overload |
| `connected_line_frac` | Fraction of lines still connected | Trip/cascade |
| `global_trip_frac` | Fraction of ALL lines in the graph that are tripped | Graph-level topology signal |

`gen_p` was removed — EDA showed 1.00 correlation with `load_p` (generation matches load by power flow law).

### Edge Features (4 per line)

`[rho, p_or, q_or, near_limit]` — `near_limit = (rho >= 1.0)`, a boolean overload flag (this superseded
a `>= 0.9` threshold; see Exp 1 in the closure doc, the one Round-1 lever that was kept).

### Training Configuration — auto-selected by device in `training/config.py`

`TRAIN_CONFIG` is picked automatically: `DEVICE.type == "cuda"` gets the "Research PC" branch, anything
else (the personal-PC XPU/CPU dev machine) gets the smaller, **deployed** branch below.

| Parameter | Personal-PC branch (**deployed config**) | CUDA branch (untested at scale — see note) |
|---|---|---|
| `hidden_channels` | `[16, 32, 32]` | `[64, 128, 128]` |
| `heads` | `[4, 4, 1]` | `[2, 2, 1]` |
| Epochs | 10 | 50 |
| Batch size | 512 | 256 |
| Learning rate | 1e-4 | 5e-4 |
| Weight decay | 1e-5 | 1e-5 |
| Dropout | 0.0 (determinism) | 0.0 |
| Loss | Weighted CrossEntropy (ICF, sqrt-smoothed) + label_smoothing 0.1 + 0.5×loc_loss | same |
| Primary metric | **Macro F1** (not accuracy — dataset is imbalanced) | same |
| Normalization | Z-score from training split only → saved to `data/normalization_stats.pt` | same |

**The deployed `gnn_checkpoint_best.pt` was trained on the small `[16,32,32]`/`heads=[4,4,1]` config**
— verified from checkpoint tensor shapes, not assumed. The `[64,128,128]` CUDA-branch scale-up was
tried (Round 1) and **overfits/collapses** under its schedule (train loss falls while val F1 falls;
`normal`+`cascade` go to 0.0) — it is *not* a validated alternative, just what auto-selects on a CUDA
machine. If training on a CUDA (research) machine, be aware the auto-selected config there has never
produced a working checkpoint; the small config is what's proven.

**`normalization_stats.pt` must be saved after training.** It is loaded at inference time for all topologies (including 14-bus and 118-bus). Foreign topologies are normalized with 36-bus stats — intentional, physical quantities have the same scale.

**Seed sensitivity (diagnosed in Round 3, tooling since removed):** the deployed init (`seed=42`) is a
good, reproducible init (0.82–0.90 macro F1 across different train/val partitions) — but a *different*
init can degrade to ~0.72–0.82. Localized to initialization, not the data split; full data in
`supplimentary_docs/gnn_final_results.md` §5. Do not change `SEED` (still hardcoded to `42` in
`training/config.py`) without being aware of this.

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

**NOT YET IMPLEMENTED — this is the design, not existing code.** No `shield.py` (or equivalent) exists
in this repo yet; `evaluation/eval_cross_topology.py` does not call anything like `validate()` today.
See **Current Status** at the top of this file. The pseudocode below is the intended design (from
`supplimentary_docs/study3(integration).md`, the implementation blueprint) — treat it as the spec for
the next phase, not a description of what runs today. Once built: every GNN prediction passes through
the shield, with no bypass mode.

### Validation Flow (design)

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

### Cross-Topology Failure Mode Logging (planned, not yet built)

Once the shield exists, every BLOCK from cross-topology evaluation should be logged to
`failures_<topo>.jsonl` (these files do not exist yet) with failure mode classification:

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
| `l2rpn_case14_sandbox` | **Test only — unseen smaller** | 14 | 20 | ~15,000 |
| `l2rpn_wcci_2022` | **Test only — unseen larger** | 118 | 186 | ~20,000 |

(`rte_case14_sandbox` was the original env name and does not exist in this grid2op version — corrected
to `l2rpn_case14_sandbox` in `scripts/generate_dataset.py`; noted here so it isn't reintroduced.)

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
| FAULT_PROB | `0.05` | Grid2Op fault injection rate (`scripts/generate_dataset.py`) |
| RECONNECT_PROB | `0.20` | Grid2Op reconnect probability |
| NORMAL_KEEP_PROB | `0.02` | Fraction of `normal` frames kept (dataset is capped/quota'd, see script) |
| LINE_TRIP_KEEP_PROB | `1.0` | All N-1 cooldown frames kept — this is the Round-2 `line_trip` magnet's data-side root cause (see closure doc); a rebalanced alternative was tried (Lever E) and reverted |
| Random seed | `42` | Model init + in-memory shuffle before training — deployed init is seed-sensitive, see Component A |
| Checkpoint filename | `gnn_checkpoint_best.pt` (deployed, == `gnn_checkpoint_leverA.pt`) | Saved on best macro F1 |
| Logit margin | `gnn_logit_margin.json` | Lever A per-class calibration offsets, applied at inference only |
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
