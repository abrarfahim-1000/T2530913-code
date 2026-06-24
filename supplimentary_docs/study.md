# Methodology & Implementation Guide

> Everything here is decided. This is not a discussion document — it is a reference for how we are building the system, why we made each choice, and how to collect the data we need.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Component A — Graph Neural Network (Neural Layer)](#2-component-a--graph-neural-network-neural-layer)
3. [Component B — LLM Knowledge Extraction Pipeline](#3-component-b--llm-knowledge-extraction-pipeline)
4. [Component C — Knowledge Graph (Symbolic Rule Store)](#4-component-c--knowledge-graph-symbolic-rule-store)
5. [Component D — Symbolic Validation Shield](#5-component-d--symbolic-validation-shield)
6. [Data Collection Strategy](#6-data-collection-strategy)
7. [GNN Training Pipeline](#7-gnn-training-pipeline)
8. [End-to-End Integration & Testing](#8-end-to-end-integration--testing)
9. [Technology Stack](#9-technology-stack)
10. [Hardware & Compute Allocation](#10-hardware--compute-allocation)
11. [Evaluation Plan](#11-evaluation-plan)
12. [Out of Scope](#12-out-of-scope)
13. [Glossary](#13-glossary)
14. [Disclaimer — Living Document](#14-disclaimer--living-document)

---

## 1. System Architecture

### The Full Pipeline

```
[Domain Documents]                        [Grid2Op Simulation]
(IEEE standards, grid                     (TRAINING: l2rpn_neurips_2020_track1_small
 manuals, NERC/FERC/                       36 substations, 59 lines,
 AEMO standards)                           300,000 labeled records)
        │                                           │
        ▼                                           ▼
[LLM Knowledge Extraction]              [Dataset Generation]
 Qwen3-14B (Extractor) parses            Observations logged per step:
 documents → JSON rules                  rho, voltages, flows, topology,
 Nemotron-3 Nano 30B (Validator)         fault labels (4-class)
 verifies each rule
 → 469 unique rules retained
        │                                           │
        ▼                                           ▼
[Knowledge Graph]                       [GNN Training]
 587 nodes, 6,632 edges                  PyTorch Geometric GAT model
 (36 Bus, 59 Line, 22 Generator,         learns fault detection on
  1 Grid, 469 Rule nodes)                36-bus home topology
 Rules linked via has_rule edges
        │                                           │
        └──────────────────┬────────────────────────┘
                           │
                           ▼
              [Symbolic Validation — THE SHIELD]
              Every GNN prediction is checked against KG rules.
              PASS → output forwarded
              BLOCK → prediction blocked + explanation generated
                           │
                           ▼
                    [Final Output]
              Validated prediction + traceable explanation
              (cites rule ID and source document)
                           │
                           ▼
              [Cross-Topology Evaluation]
              Same trained GNN + same KG tested on:
              • rte_case14_sandbox  (14 buses — unseen, smaller)
              • l2rpn_wcci_2022     (118 buses — unseen, larger)
              Measures GNN degradation vs Shield stability
              across topologies it was never trained on
```

### The Non-Negotiable Principle

**GNN outputs are never final without symbolic validation.** The GNN is the perception engine. The shield is the safety officer. They are always both active. There is no mode where the GNN output bypasses the shield — including during cross-topology evaluation on unseen environments.

### The Research Thesis

The system addresses two tightly linked research questions:

**Generalization (The "Where"):** Neural networks trained on one grid topology typically fail when deployed on a different one because node degrees and spatial relationships change. Physical safety rules do not — the IEEE standard that voltage must not drop below 0.95 pu applies equally to a 14-bus grid and a 118-bus grid. We train the GNN on the 36-bus NeurIPS 2020 environment and evaluate the full GNN+Shield system on unseen 14-bus and 118-bus environments. The GNN's classification accuracy is expected to degrade. The shield's rule compliance rate is expected to remain near-constant because the KG rules are topology-agnostic.

**Epistemology (The "Why"):** Every prediction the shield blocks on an unseen topology is a structured diagnostic signal. Blocked predictions are clustered into four failure modes: overconfident wrong predictions, class confusion between fault types, localization failure, and novel topology states the GNN has never seen. This turns the shield from a safety layer into an explainability instrument — it tells us exactly where and why the neural component fails.

**The open-ended thesis question:** Is the symbolic shield necessary for cross-topology safety, or does the GNN intrinsically generalize to unseen topologies? Both outcomes are publishable. If the shield is necessary, we prove that neural models cannot be trusted with topological changes and that our decoupled symbolic layer is a mandatory safety requirement. If the GNN generalizes, we discover the empirical boundaries of where neural perception succeeds on grid physics.

---

## 2. Component A — Graph Neural Network (Neural Layer)

### What It Does

Takes a snapshot of the grid state as a graph and outputs:
- Fault classification: normal / overload / line_trip / cascade (4 classes)
- Fault localization: which substation is the fault source (node-level)

### Why GNN and Not Something Else

A power grid **is** a graph. Buses are nodes. Lines are edges. Electrical quantities (voltage, current, power) are node and edge features. Every other architecture ignores this:

- **MLP/DNN:** Flattens the grid into a vector. Loses all topology information. A fault on Line 3-4 looks identical whether Bus 3 has 2 neighbors or 10.
- **LSTM:** Models time sequences well but is completely blind to graph structure. Cannot localize a fault spatially.
- **CNN:** Designed for grid-like spatial data (images). Power grids are irregular, sparse graphs — CNN kernels have no meaning here.

A GNN propagates information along edges, so it natively understands that a fault on Line 3-4 affects Bus 3 and Bus 4 differently depending on their local topology. Global pooling (mean/max/min) is topology-agnostic by construction — this is why the same trained model can run inference on graphs with different node counts without architectural changes.

### Architecture (Finalized)

**Graph Attention Network (GAT)** with three GATConv layers, BatchNorm after each layer (`track_running_stats=False` — critical for eval mode stability), and triple global pooling.

```
Input graph (n_nodes × 4 node features, n_edges × 4 edge features)
    ↓
GATConv(4 → 64, heads=2)  + BatchNorm + ELU
    ↓
GATConv(128 → 128, heads=2) + BatchNorm + ELU
    ↓
GATConv(256 → 128, heads=1) + BatchNorm + ELU
    ↓
global_mean_pool ‖ global_max_pool ‖ global_min_pool  → (128×3,)
    ↓
Classifier MLP → (4,) class logits
Localizer MLP  → (n_nodes,) fault probability per bus
```

**Why triple pooling:** `global_max_pool` captures overload spikes. `global_min_pool` (implemented as `-global_max_pool(-x)`) captures line-trip drops in connectivity. `global_mean_pool` captures the baseline state. Each fault type requires a different pooling signal.

**Why `track_running_stats=False`:** Standard BatchNorm tracks running statistics during training and applies them during eval. On power grid data where 90%+ of nodes are operating safely, these running stats flatten anomalous spikes (rho > 1.0) during validation — destroying the model's ability to detect overloads. Live batch statistics avoid this.

**Why no dropout:** Dropout in GATConv randomly severs attention edges during training, destroying the physical graph structure. Dropout in the classifier creates a 1.25x scaling gap between train and eval mode — sufficient to push logits across decision boundaries on continuous power flow features. The model is 100% deterministic: `model.train()` and `model.eval()` execute identical math.

### Node Features (per bus, 4 features)

| Feature | Construction | Rationale |
|---|---|---|
| `load_p` | Sum of active loads at bus | Demand signal |
| `mean_v` | Mean voltage of connected lines, normalized by 150.0 kV | Voltage health |
| `max_rho` | Max loading ratio of connected lines | Overload signal |
| `connected_line_frac` | Fraction of lines at bus still connected | Trip/cascade signal |

**Why these four:** `gen_p` was removed after EDA revealed 1.00 correlation with `load_p` (generation matches load by power flow law). Removing it eliminates 100% redundant information. The remaining four features are orthogonal and capture the four fault types directly.

### Edge Features (per line, 4 features)

`[rho, p_or, q_or, line_status]`

**Critical:** Tripped lines (`line_status = 0`) are physically pruned from `edge_index` at graph construction time using a boolean mask. This gives GATConv a hard topological boundary between a tripped grid and a normal grid — without pruning, a line_trip state has the same number of edges as a normal state and is mathematically indistinguishable.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR, T_max=50 |
| Batch size | 256 |
| Loss | Weighted CrossEntropy (ICF weights, sqrt-smoothed) |
| Early stopping | patience=15, min_delta=0.001 |
| Validation metric | Macro F1 (not accuracy — dataset is imbalanced) |
| Normalization | Z-score from training split only, saved to `normalization_stats.pt` |

**Normalization stats must be saved to disk after training.** They are required at inference time for cross-topology evaluation. The foreign topology graphs (14-bus, 118-bus) are normalized using the 36-bus training stats — this is intentional. The GNN sees the same feature scale it was trained on regardless of topology.

### Cross-Topology Inference

The trained GNN runs on foreign topology graphs without modification because:
- Global pooling is topology-agnostic (operates over variable node counts)
- Node and edge features are physical quantities that exist in any Grid2Op environment
- The classification head outputs 4 logits regardless of input graph size

The localization head is **disabled** for cross-topology evaluation. It outputs `n_nodes` logits, which varies by topology, and localization on an unseen graph is not a thesis claim. Classification accuracy and shield compliance rate are the cross-topology metrics.

---

## 3. Component B — LLM Knowledge Extraction Pipeline

### What It Does

An LLM reads domain-authoritative documents — IEEE standards, NERC/FERC operational standards, grid operation manuals — and extracts symbolic rules in structured JSON format. These rules are loaded into the knowledge graph and used by the shield at inference time.

This replaces manual rule encoding, which is the core knowledge bottleneck in all existing Neuro-Symbolic systems for power grids. The closest validated precedent is **Chen et al. (2025, 2026)**, who used an LLM to extract rules from USGS geology textbooks into a KG, achieving 99.06% accuracy vs an 84.3% baseline. We apply the same pipeline to power grid documentation.

### Why Local LLM

- Grid operational documents may be institution-sensitive
- API calls introduce non-determinism (model updates, rate limits)
- Local inference is fully reproducible
- RTX 4080 Super (16GB VRAM) + 64GB RAM handles both models comfortably

### Models (Finalized)

**Extractor: Qwen3-14B** via Ollama. Fits fully in 16GB VRAM at 4-bit quantization (~8–10GB). Superior technical comprehension on dense IEEE prose. Thinking mode disabled via `/no_think` prefix — essential to prevent chain-of-thought tokens from breaking JSON output.

**Validator: Nemotron-3 Nano 30B (A3B)** via Ollama. Mixture-of-Experts with 3B active parameters — reasoning depth of a 30B model at the efficiency of a smaller one. Independent training lineage from NVIDIA provides architecturally distinct failure modes from Qwen3. Thinking suppressed via `think=False` in `ollama.chat()`.

**Why this split:** Extraction is a high-recall discovery task on dense prose. Validation is a high-precision verification task — the validator receives the original chunk plus the candidate rules, so it only needs to confirm, not discover. Architecturally distinct models provide stronger validation confidence than two Qwen models agreeing.

**Sequential execution:** Qwen3-14B runs first across the full document set, results saved to `*_candidates.jsonl`. Nemotron-3 Nano 30B then loads for the validation pass. No concurrent model loading.

### Pipeline Architecture

```
[Document Chunk]
        │
        ▼
[Qwen3-14B — Extractor]
 /no_think prefix enforced
 Output: JSON array of rule candidates
        │
        ▼
[Pydantic schema validation]
 Rule schema: rule_id, source, entity, condition,
              action, severity, explanation
 Invalid schema → dropped, logged
        │
        ▼
[Nemotron-3 Nano 30B — Validator]
 think=False enforced
 Input: original chunk + candidates
 Output: CONFIRM / REJECT / CORRECT per rule
        │
        ├── CONFIRM → load into KG as-is
        ├── CORRECT → merge corrected_fields, load into KG
        ├── REJECT  → discard
        └── NO_VERDICT / error → flag for manual review
        │
        ▼
[Deduplication]
 Key: (entity, condition) pair
 Duplicates: sources merged, rule dropped
        │
        ▼
[all_rules_deduped.jsonl → Knowledge Graph]
```

### Finalized Extraction Results

| Stage | Count |
|---|---|
| Candidate rules extracted | 1,372 |
| Confirmed as-is (CONFIRM) | 50 (3.64%) |
| Corrected and retained (CORRECT) | 451 (32.87%) |
| Rejected (REJECT) | 766 (55.83%) |
| Flagged for review | 105 (7.65%) |
| Retained before dedup | 501 |
| **Unique rules after dedup** | **469** |

### Output Schema

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

### Document Tier System

Rules are sourced from documents classified by authority:

| Tier | Source Type | Treatment |
|---|---|---|
| A | IEEE/IEC/NERC/FERC/ENTSO-E standards | Mandatory inclusion |
| B | Official operational manuals, simulation docs | Included |
| C | Academic papers | Conditional — only if citing Tier A |
| D | Textbooks, theses, blog posts, AI-generated | Rejected |

Source documents used: IEEE Std 1547-2018, IEEE Std C37.2, NERC FAC-001/FAC-002, AEMO PSR 2020.

### Key Implementation Notes

- `keep_alive=0` in every Ollama call — releases VRAM immediately after inference, preventing overlap between Extractor and Validator
- GBNF grammar enforcement on array output prevents malformed JSON from reaching Pydantic
- `strip_think()` regex applied to all model output before JSON parsing — Qwen3 occasionally leaks `<think>` tokens despite `/no_think`
- Rule IDs are rewritten to globally unique sequential IDs (`R_001`, `R_002`, ...) across all chunks and documents before validation

---

## 4. Component C — Knowledge Graph (Symbolic Rule Store)

### What It Does

Stores the extracted rules and the grid topology as a queryable graph. During inference, the shield traverses `has_rule` edges from the predicted fault entity to collect all applicable rules and evaluate them against the current grid context.

### Finalized KG Statistics

| Metric | Value |
|---|---|
| Total nodes | 587 |
| Bus nodes | 36 |
| Line nodes | 59 |
| Generator nodes | 22 |
| Grid node | 1 |
| Rule nodes | 469 |
| Total directed edges | 6,632 |
| `connected_to` edges | 118 |
| `part_of` edges | 139 |
| `has_rule` edges | 6,375 |

Rule severity breakdown: majority critical/high. Largest entity category: generator constraints.

### Graph Schema

**Nodes:**

| Node Type | Key Attributes |
|---|---|
| `Bus` | bus_id, voltage_nominal, bus_type |
| `Line` | line_id, max_current_A, max_loading_pct |
| `Generator` | gen_id, p_min_kw, p_max_kw |
| `Grid` | (singleton — anchors system-level rules) |
| `Rule` | rule_id, condition, action, severity, source, explanation |

**Edges:**

| Edge Type | Meaning |
|---|---|
| `connected_to` | Bus ↔ Line (physical topology) |
| `part_of` | Bus → Grid, Line → Grid, Generator → Bus |
| `has_rule` | Entity → Rule |

### The Grid Node

A single global `Grid` node anchors all system-level rules that apply regardless of which specific entity faulted. At inference time, the shield retrieves rules from both the predicted fault entity AND the Grid node. This ensures system-wide constraints (frequency response, N-1 security) are always evaluated, not just entity-specific ones.

**Retrieval pattern:**
```
applicable = rules_of(predicted_entity) + rules_of("Grid")
```

### Why NetworkX (Not Neo4j)

NetworkX is pure Python, zero setup, in-process. At 587 nodes and 6,632 edges, the full rule set is small enough that iterating all Rule nodes per inference step completes in under 1ms. Neo4j requires a running server process and Cypher queries — unnecessary overhead at this scale. Switch only if the graph grows beyond 50,000+ nodes.

### Cross-Topology Note

The KG is built from IEEE standards and grid documentation — it is **not built from the NeurIPS 2020 training topology**. The 36/59/22 Bus/Line/Generator counts in the KG reflect the training environment's topology nodes, but the Rule nodes (469) are sourced from physical standards that apply universally. When evaluating on 14-bus or 118-bus environments, the shield uses the same KG. Only the shield's context dict (rho, voltage, line_status) changes to reflect the foreign topology's observation. Rule conditions evaluate against physical thresholds regardless of topology size.

---

## 5. Component D — Symbolic Validation Shield

### What It Does

Intercepts every GNN prediction. Evaluates all applicable rules from the KG against the current grid state. Returns PASS (with the validated prediction) or BLOCK (with a structured explanation citing rule IDs and source documents).

### Why "Shield" and Not a Loss Function Penalty

This is the distinction between Generation 2 and Generation 3 NeSy systems:

- **Gen 2 (PINNs, soft constraints):** Loss function penalizes rule violations during training. The model is discouraged from breaking physics but not prevented. It can still output a physically impossible prediction at inference time.
- **Gen 3 (Shield — this system):** Rule violations are detected at inference time, after the model outputs. The output is structurally blocked from reaching execution. Hard constraint, not a soft suggestion.

Directly inspired by **Younesi et al. (2026)**: their two-step microgrid system validated actions against a fixed rule set and achieved 91.7% safe power restoration. Our improvement: their rules were manually written. Ours are LLM-generated from IEEE standards.

### Shield as Diagnostic Tool (Cross-Topology Extension)

In the cross-topology evaluation, every BLOCK decision is logged with structured metadata and clustered into one of four failure modes:

| Failure Mode | Definition |
|---|---|
| **Overconfident wrong** | GNN predicts "normal" with confidence > 0.85 during a physical fault |
| **Class confusion** | GNN detects a fault but confuses fault type (e.g., overload predicted as cascade) |
| **Threshold failure** | GNN correctly identifies fault type but predicted entity violates physical thresholds |
| **Novel topology state** | Fault state involves sub-graph structure never seen during training |

This clustering converts the shield's BLOCK log into a structured analysis of where and why the neural component fails on unseen topologies.

### Voltage Translation

Grid2Op provides voltage in kV. IEEE rules use per-unit. For the NeurIPS 2020 environment, nominal is ~150kV:

```
voltage_pu = v_or_kv / 150.0
```

Loading percentage:
```
loading_pct = rho_max * 100
```

These translations are fixed constants in the shield's condition evaluation namespace. If evaluating on a different environment with a different nominal voltage, this constant must be updated from the environment's meta JSON.

### Validation Logic

```python
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

def validate(prediction_context: dict, KG) -> dict:
    fault_type       = prediction_context["fault_type"]
    applicable_rules = get_applicable_rules(KG, fault_type)

    violated = []
    for rule in applicable_rules:
        if evaluate_condition(rule["condition"], prediction_context):
            violated.append(rule)

    if not violated:
        return {
            "status":     "PASS",
            "fault_type": fault_type,
            "confidence": prediction_context["confidence"],
        }

    violated.sort(key=lambda r: SEVERITY_ORDER.get(r.get("severity", "low"), 3))

    return {
        "status":           "BLOCK",
        "fault_type":       fault_type,
        "confidence":       prediction_context["confidence"],
        "violated_rules":   violated,
        "highest_severity": violated[0]["severity"],
        "explanation":      build_explanation(violated),
    }


def evaluate_condition(condition: str, context: dict) -> bool:
    namespace = {
        "voltage_pu":      min(context["v_or"]) / 150.0,
        "loading_pct":     context["rho_max"] * 100,
        "rho":             context["rho_max"],
        "rho_max":         context["rho_max"],
        "n_tripped_lines": context["n_tripped_lines"],
        "line_status":     not all(context["line_status"]),
    }
    try:
        return bool(eval(condition, {"__builtins__": {}}, namespace))
    except Exception:
        return False  # malformed condition → do not block


def build_explanation(violated_rules: list) -> str:
    return " | ".join(
        f"Rule {r['rule_id']} ({r['severity'].upper()}): {r['explanation']} [Source: {r['source']}]"
        for r in violated_rules
    )
```

### Example BLOCK Output

```
Rule R_042 (CRITICAL): Voltage at point of common coupling is 0.91 pu.
Minimum permissible is 0.95 pu per IEEE Std 1547-2018, Section 7.4.
[Source: IEEE Std 1547-2018, Section 7.4]
|
Rule R_019 (HIGH): N-1 security constraint violated — Bus 7 has no alternate
feed path after proposed disconnection.
[Source: NERC FAC-002, Section 3.1]
```

---

## 6. Data Collection Strategy

### Why Simulation, Not Real Data

Real operational data from utilities is proprietary, unavailable without NDAs, and typically anonymized in ways that remove the topology information a GNN needs. This is a known limitation across the entire field — Ahmadi et al. (2026) explicitly identifies this as the "simulation-to-reality gap." Simulation on standard benchmarks is the accepted methodology for this research domain.

### Environment Role Assignment

| Environment | Role | Buses | Lines | Records |
|---|---|---|---|---|
| `l2rpn_neurips_2020_track1_small` | **Training only** | 36 | 59 | 300,000 |
| `rte_case14_sandbox` | **Test only — unseen smaller** | 14 | 20 | ~15,000 |
| `l2rpn_wcci_2022` | **Test only — unseen larger** | 118 | 186 | ~20,000 |

No training occurs on case14 or WCCI 2022. These are evaluation-only datasets used to measure cross-topology generalization. The GNN checkpoint from NeurIPS 2020 training is applied directly without fine-tuning.

### Training Dataset (NeurIPS 2020, Finalized)

**300,000 records, 4 classes:**

| Label | Count | Percentage |
|---|---|---|
| Normal | 61,444 | 20.48% |
| Overload | 103,556 | 34.52% |
| Line Trip | 75,000 | 25.00% |
| Cascade | 60,000 | 20.00% |

Generated using **chronic-level splitting** (not frame-level) to prevent cascade sequence leakage across train/val splits. Runtime: ~2h38m at ~32 steps/sec.

### Labeling Logic (Pure Topology-Based)

```python
def get_state_label(obs, env):
    max_rho      = obs.rho.max() if len(obs.rho) > 0 else 0.0
    active_lines = int(np.sum(obs.line_status))

    if max_rho >= 1.0:                          # overload supersedes topology
        line_id = int(obs.rho.argmax())
        return "overload", int(env.line_or_to_subid[line_id])
    if active_lines == env.n_line:              # all lines up → normal
        return "normal", -1
    if active_lines == env.n_line - 1:          # exactly one line down → trip
        line_id = int(np.where(~obs.line_status)[0][0])
        return "line_trip", int(env.line_or_to_subid[line_id])
    return "cascade", -1                        # two or more lines down → cascade
```

**Key implementation facts:**
- `NO_OVERFLOW_DISCONNECTION = False` — allows Grid2Op to auto-disconnect overloaded lines, enabling natural cascades. Must be passed at `grid2op.make()` time via `param=params`, not set post-`make()`.
- FAULT_PROB = 0.10, RECONNECT_PROB = 0.09 — tuned to achieve the target distribution while preventing grid collapse within episodes.
- Hard record quotas enforced during generation to cap each class.

### Cross-Topology Test Datasets

Generated from the same labeling logic above, applied to case14 and WCCI 2022 environments. Same fault injection parameters. No class balancing required — these are test sets, not training sets, so natural distribution is acceptable. ~15k–20k records each; generation time ~30 minutes per environment.

### Train/Val/Test Split

70% train / 15% validation / 15% test. **Chronic-level splitting only** — frame-level splitting leaks cascade sequences across splits, making val/test metrics artificially inflated.

---

## 7. GNN Training Pipeline

### Feature Construction

**Node features** are constructed per substation by aggregating line-level observations to bus level. **Edge features** are per line. Tripped lines are physically pruned from `edge_index` using the `line_status` boolean mask before graph construction.

Node feature vector (4 features):
```
x_v = [load_p_v, mean_v_v, max_rho_v, connected_line_frac_v]
```

Edge feature vector (4 features):
```
e_uv = [rho_uv, p_or_uv, q_or_uv, line_status_uv]
```

### Normalization

Z-score normalization computed from the training split only:

```python
node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(dataset, train_idx)

# Applied to entire dataset in-place before split
full_dataset._data.x         = (full_dataset._data.x         - node_mean) / node_std
full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std

# Saved to disk — required for cross-topology inference
torch.save({
    "node_mean": node_mean, "node_std": node_std,
    "edge_mean": edge_mean, "edge_std": edge_std,
}, "data/normalization_stats.pt")
```

All features are normalized uniformly. No physics exemptions for rho or connected_line_frac — exempting individual features creates scale mismatches inside GATConv attention weight computation.

### In-Memory Shuffle (Required)

The training script enforces an unconditional in-memory shuffle before training to eliminate chronological domain shift:

```python
combined_idx = np.concatenate([train_idx, val_idx])
np.random.seed(42)
np.random.shuffle(combined_idx)
train_idx = combined_idx[:len(train_idx)]
val_idx   = combined_idx[len(train_idx):]
```

Without this, the model trains on low-load chronics and validates on high-load chronics, guaranteeing mode collapse.

### Class Weights

Inverse Class Frequency with sqrt smoothing:
```
w_c = sqrt(N / (C × N_c))
```

Applied to CrossEntropyLoss. Prevents the model from ignoring cascade (rarest class).

### Validation Metric

**Macro F1, not accuracy.** On an imbalanced 4-class dataset, accuracy is dominated by the majority class. Macro F1 treats all four fault classes equally — a model that ignores cascade completely will show high accuracy but near-zero macro F1.

---

## 8. End-to-End Integration & Testing

### Phase 1 — Toy Scenario Smoke Test

Three hand-crafted scenarios on `rte_case5_example` (5-bus) verify the full pipeline runs end-to-end without integration failures. Must all pass before proceeding to Phase 2.

**Scenario 1 — Clean pass:** All lines at 60–80% load, voltages nominal. GNN predicts "normal". Expected: PASS.

**Scenario 2 — Overload block:** Line 3 at 115% thermal limit. GNN predicts "disconnect Line 3". Shield evaluates voltage and N-1 rules. Expected: BLOCK citing rule IDs and sources.

**Scenario 3 — Multi-rule cascade block:** Lines 2 and 5 at 90% load. GNN predicts "disconnect Line 2". Shield evaluates thermal limit AND N-1 security rules simultaneously. Expected: BLOCK with multi-rule explanation.

### Phase 2 — Home Topology Formal Evaluation

Held-out test set from `l2rpn_neurips_2020_track1_small` (15% of 300,000 records). Full pipeline: obs → PyG graph → GNN → class logits → prediction context → shield → PASS/BLOCK + explanation. Metrics logged per step and aggregated.

### Phase 3 — Cross-Topology Evaluation

The same trained GNN checkpoint and the same KG are evaluated on case14 and WCCI 2022 test sets. No retraining, no fine-tuning, no KG modification.

**Information flow per step:**
```
obs (foreign topology)
    → build_node_features(obs, meta_foreign)   ← uses foreign env's topology meta
    → build_edges(obs, meta_foreign)
    → normalize using normalization_stats.pt    ← from 36-bus training split
    → GNN inference (classification head only)  ← localization head disabled
    → prediction context (raw obs values)
    → shield validate()                         ← same KG, topology-agnostic rules
    → PASS or BLOCK + failure mode classification
```

**Why the same normalization stats work on foreign topologies:** The physical quantities (rho, voltage in kV, power in MW) have the same physical meaning and scale regardless of how many buses the environment has. The training distribution of these quantities is a reasonable prior for any Grid2Op environment.

**What changes per topology:** The `meta` object passed to `build_node_features` and `build_edges` must come from the foreign environment's meta JSON, not the training environment's. This ensures node/edge construction uses the correct bus-to-line mappings for the foreign topology.

### Phase 4 — Failure Mode Clustering

Every BLOCK decision from Phase 3 is logged to `failures_<topo>.jsonl` with: true label, predicted label, confidence, violated rule IDs, highest severity, and failure mode classification. Aggregated across all steps to produce the failure mode distribution per topology — a key thesis table.

---

## 9. Technology Stack

| Component | Tool | Justification |
|---|---|---|
| Grid simulation | **Grid2Op + LightSimBackend** | RTE-built; L2RPN benchmark environments; ~10x faster than PandaPower backend |
| GNN framework | **PyTorch + PyTorch Geometric** | De facto standard for GNN research; GAT, global pooling, DataLoader |
| LLM inference | **Ollama + Qwen3-14B + Nemotron-3 Nano 30B** | Local, reproducible; JSON format enforcement; think mode control |
| LLM orchestration | **pdfplumber + custom chunker** | PDF ingestion; sliding-window chunking with paragraph-boundary detection |
| Schema validation | **Pydantic** | Rule and Verdict schemas; malformed outputs dropped before KG ingestion |
| Knowledge graph | **NetworkX DiGraph** | Zero setup; in-process; sufficient at 587 nodes / 6,632 edges |
| Shield logic | **Python (custom)** | Fully auditable; `eval()` with whitelist namespace; no external dependencies |
| Experiment tracking | **Weights & Biases** | Training metrics, per-class F1, confusion matrices |
| Version control | **Git + GitHub** | Standard |

---

## 10. Hardware & Compute Allocation

### Research PC — All Primary Workloads

- **CPU:** Intel Core i7-14700K (20 cores / 28 threads)
- **GPU:** NVIDIA RTX 4080 Super — 16GB VRAM
- **RAM:** DDR5 64GB
- **Schedule:** Saturdays, Mondays, Wednesdays, 1 PM–1 AM
- **Runs:** GNN training, all dataset generation (training + cross-topology test sets), LLM extraction (Qwen3-14B then Nemotron-3 Nano 30B, sequential), cross-topology evaluation

### Personal PC — Development & Smoke Tests Only

- **CPU:** AMD Ryzen 5 7500F
- **GPU:** Intel Arc B580 — 12GB VRAM (no PyG XPU support, no LLM inference)
- **RAM:** DDR5 16GB
- **Runs:** Code development, unit tests, shield logic, `num_workers=0` DataLoader tests

### Allocation Rules

- All production runs execute on the Research PC only
- Personal PC never runs training, dataset generation at scale, or LLM inference
- LLM extraction: Qwen3-14B (~8–10GB VRAM, fully GPU-resident) runs first; Nemotron-3 Nano 30B (~18–20GB, partial CPU offload into 64GB RAM) runs second. Sequential, never concurrent.
- GNN training target: under 2 hours per run on RTX 4080 Super
- Cross-topology dataset generation: ~30 minutes per environment (inference only, no training)
- `num_workers=0` enforced on Windows DataLoader (PyG multiprocessing constraint)

---

## 11. Evaluation Plan

### GNN — Home Topology (NeurIPS 2020 Test Split)

| Metric | Description |
|---|---|
| Macro F1 | Primary metric — F1 averaged across all 4 fault classes equally |
| Per-class F1 | Breakdown by normal / overload / line_trip / cascade |
| Confusion matrix | Reveals which fault types are confused with each other |

Baseline comparisons: flat MLP on same data, LSTM on time-series version of same data. GNN should outperform both on macro F1 due to topology-aware message passing.

### LLM Rule Extraction

| Metric | Value (finalized) |
|---|---|
| Candidates extracted | 1,372 |
| Confirmed + corrected (retained) | 501 (36.5%) |
| Rejected | 766 (55.8%) |
| Flagged | 105 (7.6%) |
| Unique rules after dedup | 469 |

These numbers are reported as-is. The extraction pipeline is complete.

### Shield — Home Topology

| Metric | Description |
|---|---|
| Rule compliance rate | % of PASS decisions that satisfy all applicable rules (should be 100% by construction) |
| False block rate | % of ground-truth normal/valid states incorrectly blocked |
| Block precision | % of blocked predictions that were genuinely rule-violating |

Reference target: Younesi et al. (2026) achieved 91.7% safe restoration with manually written rules. Our target is to match or exceed this with LLM-generated rules.

### Cross-Topology Generalization (New — Core Thesis Evaluation)

Evaluated on case14 (14-bus) and WCCI 2022 (118-bus) using the GNN trained on NeurIPS 2020 (36-bus). No retraining.

| Metric | Description |
|---|---|
| GNN macro F1 per topology | Expected to degrade as topology diverges from training |
| Shield rule compliance rate per topology | Expected to remain near-constant — rules are topology-agnostic |
| BLOCK rate per topology | Fraction of predictions the shield intercepts |
| False block rate per topology | Shield over-blocking valid states on unseen topology |
| Failure mode distribution | Count of each failure mode (overconfident wrong / class confusion / threshold failure / novel topology state) |

**The core thesis result** is the comparison of these two curves: GNN macro F1 (drops with topology distance) vs Shield compliance rate (stays flat). The gap between them is the safety contribution of the symbolic layer — the empirical proof of why a decoupled neuro-symbolic architecture is necessary for cross-topology deployment.

### Explanation Quality

For blocked predictions, the explanation cites `rule_id` and `source`. Team members rate each explanation 1–5 on correctness, specificity, and actionability. Reported as mean ± std per topology.

---

## 12. Out of Scope

These are decided exclusions. Do not prototype or propose these.

- Real utility deployment or hardware-in-the-loop
- Training any foundation model (LLMs are used pre-trained, quantized only)
- Fine-tuning the GNN on cross-topology test environments
- Multi-agent / multi-microgrid coordination
- Reinforcement Learning agent (future work only)
- Blockchain, digital twins, or metaverse integration
- Softening the shield into a loss penalty — the hard inference gate is non-negotiable

---

## 13. Glossary

| Term | Definition |
|---|---|
| **GNN** | Graph Neural Network — processes graph-structured input natively |
| **GAT** | Graph Attention Network — GNN variant where edges have learned attention weights; our chosen architecture |
| **Knowledge Graph (KG)** | Graph of grid entities (Bus, Line, Generator, Grid) and Rule nodes linked by `has_rule` edges |
| **Shield** | The symbolic validation layer — hard-blocks rule-violating GNN outputs; also acts as diagnostic tool for cross-topology failure analysis |
| **Cross-topology evaluation** | Running the trained GNN+Shield on grid environments it was never trained on, to measure generalization and safety guarantees |
| **Failure mode** | Structured category of why the shield blocked a prediction on an unseen topology: overconfident wrong / class confusion / threshold failure / novel topology state |
| **Grid2Op** | Python power grid simulation framework by RTE; manages episodes, chronics, and fault injection for AI research |
| **Chronic** | A time-series of load and generation values used as one simulation episode in Grid2Op |
| **Chronic-level splitting** | Assigning entire chronic sequences to train or test — prevents cascade sequence frames from leaking across splits |
| **`obs.rho`** | Grid2Op observation attribute — ratio of current flow to thermal limit per line; ≥1.0 means overload |
| **`rte_case14_sandbox`** | Grid2Op 14-bus environment — used as unseen smaller topology in cross-topology evaluation |
| **`l2rpn_neurips_2020_track1_small`** | Grid2Op NeurIPS 2020 environment — 36 buses, 59 lines; training topology |
| **`l2rpn_wcci_2022`** | Grid2Op WCCI 2022 environment — 118 buses, 186 lines; used as unseen larger topology in cross-topology evaluation |
| **LightSimBackend** | Fast power flow backend for Grid2Op (~10x faster than default); from `lightsim2grid` package |
| **Macro F1** | F1 score averaged equally across all classes — primary GNN validation metric; robust to class imbalance |
| **Normalization stats** | Per-feature mean and std computed from the 36-bus training split; applied to all topologies at inference time; saved to `normalization_stats.pt` |
| **Topology-agnostic** | Property of the shield's KG rules and the GNN's global pooling — both operate correctly regardless of how many buses or lines the environment has |
| **`track_running_stats=False`** | BatchNorm setting that forces live batch statistics during both train and eval — prevents running stats from flattening anomaly signals in eval mode |
| **Voltage pu** | Voltage in per-unit — normalized so 1.0 = nominal; IEEE rules use pu; Grid2Op provides kV; conversion: divide by 150.0 for NeurIPS 2020 environment |
| **Gen-3 NeSy** | Generation 3 Neuro-Symbolic — hard inference-time symbolic gate, as opposed to Gen-2 soft loss penalty approaches (PINNs) |
| **Ollama** | Local LLM runtime used for Qwen3-14B and Nemotron-3 Nano 30B inference; `keep_alive=0` releases VRAM immediately after each call |
| **NetworkX** | Python graph library used as KG backend; `DiGraph` for directed edges |

---

## 14. Disclaimer — Living Document

> **Nothing in this document is final.**
>
> All model names, dataset choices, simulation environments, library selections, and architectural decisions recorded here reflect the best available information at the time of writing. Any of these may change as new findings, benchmarks, hardware constraints, or implementation realities emerge.
>
> Specific items subject to change without notice:
> - GNN hidden channel dimensions and head configuration
> - Cross-topology evaluation environments (case14 / WCCI 2022 / others)
> - Failure mode taxonomy and clustering method
> - Normalization strategy for foreign topologies
> - Evaluation metric thresholds and baseline comparisons
> - LLM model versions (Qwen3 / Nemotron-3 Nano)
> - Knowledge graph backend (NetworkX vs Neo4j if scale increases)
>
> When a decision changes, the relevant section is updated to reflect the new choice and reasoning. Previous decisions are not preserved — this document describes the current state, not the history.

---

*Last updated: June 2026.*
