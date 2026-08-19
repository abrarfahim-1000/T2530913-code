# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A neuro-symbolic fault detection system for power grids combining:
- **GNN (Neural Layer):** Graph Attention Network trained on 36-bus NeurIPS 2020 topology
- **LLM Pipeline (Component B):** Qwen3-14B + Nemotron-3 Nano 30B extract rules from IEEE/NERC standards
- **Knowledge Graph (Component C):** to be redesigned from scratch (the v1 graph — 587 nodes / 469 rules — is retired)
- **Symbolic Shield (Component D):** Post-hoc gate — every GNN prediction validated against the rule corpus before output

The GNN is trained on one topology (36-bus) and evaluated on unseen topologies (14-bus, 118-bus) to measure cross-topology generalization. The shield's rule compliance rate is expected to remain stable while GNN accuracy degrades.

---

## Current Status (2026-08-15)

> **The operative plan is [`supplimentary_docs/component_d_plan.md`](supplimentary_docs/component_d_plan.md).**
> Read it before starting work — it carries the task redesign, the runbook, and the landmines.
> This file describes the *architecture*; the plan describes *what to do next*.
> `component_d_handoff_archive.md` is retired — do not follow it.

> **2026-08-16 — the task is now N-1 CONTINGENCY SCREENING (`--task n1`).** Two earlier task
> designs were built, probed and rejected *before* generating against them; the full evidence chain
> is [`component_d_plan.md`](supplimentary_docs/component_d_plan.md) §1.1. Short version: the
> 4-class `classify` target is closed-form (4 rules = 100% agreement), and the binary `forecast`
> target is either unlearnable (any-fault: 1.00× the all-positive baseline at every H ≥ 3, because
> trip onset is an unconditional coin flip) or exhausted by one threshold (overload-only: model
> 0.163 vs rule 0.160). N-1 screening is neither — global `rho_max` scores 1.04× baseline, 100% of
> frames are mixed, and a network-aware model reaches 0.868 against a best-rule 0.608. **No rule
> over the present observation can restate an N-1 label, because producing it needs a power-flow
> solve** — which is what permanently un-rigs the shield comparison.
>
> **TRAINED (2026-08-16).** Edge-level head, 12,000 frames / 576 chronics / 702,618 contingency
> labels. Held-out test **F1 0.8872, AP 0.9549 — 1.91× the best single-rule baseline (0.4639)**;
> ablation shows message passing contributes **+0.058** over endpoint features alone. Full account
> in [`gnn_n1_tightening.md`](supplimentary_docs/gnn_n1_tightening.md).
>
> **CROSS-TOPOLOGY, measured 2026-08-16 — generalisation is PARTIAL and ASYMMETRIC.** Full table,
> caveats and an untested hypothesis in [`component_d_plan.md`](supplimentary_docs/component_d_plan.md) §7.6.
>
> | topology | lines | contingencies | best rule | **model** | AP |
> |---|---:|---:|---:|---:|---:|
> | neurips2020 *(in-dist)* | 59 | 113,205 | 0.4639 | **0.8972** — 1.93× | 0.9615 |
> | case14 *(unseen, smaller)* | 20 | 118,502 | 0.5392 | **0.4477** — **0.83×, FAILS** | 0.4270 |
> | wcci2022 *(unseen, larger)* | 186 | 742,472 | 0.4915 | **0.5721** — 1.16× | 0.6419 |
>
> Scaling **up** costs far less than scaling **down** — the opposite of the naive expectation.
> On case14 the model loses to the single-rule baseline outright; report it as such. Every figure
> is best-threshold *on its own topology* (agenda item 6 would lower all three). Note the eval
> script reads 0.8972/0.9615 in-distribution where the tightening doc records 0.8872/0.9549 —
> the checkpoint on disk is from a different run than the one written up; treat the checkpoint as
> authoritative.
>
> 🚨 **That doc also records a normalization bug affecting EVERY task, including the frozen
> classify checkpoint.** `compute_normalization_stats()` populates PyG's `_data_list` cache, so the
> subsequent in-place `_data.x` normalization never reached the DataLoaders — all training ran on
> raw unnormalized features while `normalization_stats.pt` was saved as if it had been applied.
> the (now removed) classify cross-topology script *did* normalize at inference, so the frozen
> classify checkpoint has a **train/inference mismatch**, and some share of its reported
> cross-topology degradation is attributable to that rather than to topology transfer. Fixed for
> N-1 (`_data_list = None` plus an assertion); read §3 of the tightening doc before citing any
> classify cross-topology number.

**Component A (GNN) — N-1 contingency screening, TRAINED and DONE.** The trainer, config and
feature builders carry the N-1 path *only*; the 4-class classifier and the binary forecast model
were retired on 2026-08-16 and their code lives in git history. Rationale for the demotion, and the
one page of methodology that survives into the thesis, are in
[`revised_thesis_claim.md`](supplimentary_docs/revised_thesis_claim.md) §2.

For the record, the finding that motivated it: the 4-class label was a **closed-form function of
the observation** (`rho_max >= 1.0 → overload`, `n_tripped_lines == 0 → normal`, `== 1 →
line_trip`, else `cascade`) — four ordered rules reproduce the stored labels with **100% agreement
on 55,000 records** across both topologies, so a symbolic layer scored 100% where the trained GATv2
reached 0.8277.

⚠️ **Retired classify artifacts remain on disk and must not be overwritten** —
`gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt` (**not** the same file — see the constants
table), `gnn_logit_margin.json`, `normalization_stats.pt`, `data/grid_dataset_neurips2020.jsonl`,
`data/processed_grid_data.pt`, `data/split_neurips2020_*.npy`. Every N-1 artifact carries an `_n1`
suffix precisely so it cannot collide with them. The three rounds of rejected architecture
experiments are documented in
[`supplimentary_docs/gnn_final_results.md`](supplimentary_docs/gnn_final_results.md) — they pertain
to the retired classify task.

**Component B (LLM extraction) — stage 1 DONE, stages 2–3 pending.** 2,463 candidates across 16
documents in `rules_35b/` (kept pristine as the stage-1 archive). The pipeline is now **four
stages**: `extract.py` (open vocabulary) → `translate.py` → `polarity_guard.py` (stage 2.5,
deterministic) → `validate.py`. The closed vocabulary is enforced at stages 2–3 **only**.

The vocabulary is **14 variables**, not 6 — it previously used only `rho`, `v_or`, `line_status`
and ignored `p_or/q_or/gen_p/load_p`, which every record already carries. Measured against what is
actually derivable, **693 of 2,463 candidates (28.1%)** are expressible, not 195 (7.9%).

Rules carry a **role**: `CONSTRAINT` (true ⇒ violation ⇒ BLOCK) or `AFFIRMATION` (true ⇒ telemetry
consistent with the class in `affirms`). Forcing everything into constraint polarity was itself
generating inverted-band defects.

⚠️ **Do not re-close `EXTRACT_PROMPT` against `CONDITION_VOCABULARY`.** Tried twice; extraction
returned *zero* rules both times — on frequency/timing-heavy standards the model correctly answers
`[]` for nearly every chunk, starving stage 2. Guarded by
`tests/test_condition_lint.py::test_extract_prompt_stays_open_vocabulary`. Related: thinking must
be suppressed via `generate_no_think()` (API-level `think=False`), not the `/no_think` prefix
alone — newer Qwen builds ignore the prefix and the reasoning trace breaks JSON parsing.

v1 outputs are archived at `rules/v1_archive/` (retained as a thesis negative result).

**Component C (Knowledge Graph) — to be REDESIGNED from scratch.** `kg/knowledge_graph.pkl` is a
v1 artifact and is not used by anything current. Do **not** rebuild with the existing mechanism —
the new graph is designed against the rules that actually survive translation, guard, and
validation. Note for that redesign: `Line` (59) and `Bus` (28) account for only **3.5%** of the
corpus, so entity-based retrieval carries almost no information.

**Component D (Symbolic Shield) — BUILT and tested; not yet run against a real ruleset.**
`shield/` (context, evaluator, shield), `extraction/polarity_guard.py`,
`evaluation/eval_shield.py`, `evaluation/summarize_shield_results.py`. **118 tests green.**
Rule retrieval sits behind a `RuleProvider` protocol, **not** `build_kg.py::get_rules_for_entity`,
so the KG redesign cannot invalidate it. Pending: the binary/asymmetric update for the forecast
task (plan §6.1).

Binding voltage contract (plan §5): per-line base kV from `data/grid_dataset_<tag>_basekv.json`
with energized-line masking. The flat `v_or/150.0` conversion in `study3(integration).md` §4.5 is
**superseded** — case14 runs lines at ~20 kV and ~138 kV, and even the 36-bus grid has 7 lines at
~365 kV.

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
# ⚠️ Windows: use the repo venv — a bare `python` is the Microsoft Store stub.
#    Set PYTHONIOENCODING=utf-8 when piping/redirecting (these scripts print `→`,
#    and the cp1252 pipe encoding kills them mid-run on a UnicodeEncodeError).
#      $env:PYTHONIOENCODING = "utf-8"
#      .venv\Scripts\python.exe scripts\generate_dataset.py ...

# N-1 contingency screening — the ONLY task; `--task n1` is the default.
# Per-line "if this line trips, is a thermal limit violated?"; writes *_n1.jsonl
# with a per-line label vector per frame.
#
# SIZING: effective sample size is bounded by SCENARIO count, not label count.
# --n1-stride 12 is the protocol on ALL THREE topologies — identical generation,
# only the environment differs. At the old stride 4 a 12k-frame budget would have
# covered only ~210 of neurips's 576 chronics instead of all of them.
#
# All three sets below are GENERATED and verified (component_d_plan.md §7.6).
# Measured throughput, not estimates:
python scripts/generate_dataset.py --env neurips --n_records 12000   # 20 min, 10 rec/s, 702k labels
python scripts/generate_dataset.py --env case14  --n_records 6000    #  5 min, 21 rec/s, 119k labels
python scripts/generate_dataset.py --env wcci    --n_records 4000    # 30 min,  2 rec/s, 742k labels

# --smoke: 3 chronics x 500 steps, capped at 200 records (~8 s). The record cap is
# load-bearing — the loop cycles chronics toward --n_records, so before it existed a
# smoke run replayed 3 scenarios toward 300,000 records and never terminated.
python scripts/generate_dataset.py --env case14 --smoke

# ⚠️ Forecast task — BUILT, PROBED, REJECTED. Retained ONLY to reproduce the
# negative result; do not train against it, and do not delete it to "finish" the
# classify cleanup — no forecast dataset exists on disk, so the code is the only
# way to reproduce component_d_plan.md §1.1 steps 1-2. Any-fault scores 1.00x the
# all-positive baseline at every H >= 3 (line-trip onset is an unconditional coin
# flip); overload-only is exhausted by a single rho threshold (model 0.163 vs rule
# 0.160, trend features do not help).
python scripts/generate_dataset.py --env neurips --task forecast --horizon 6 --n_records 300000

# The legacy 4-class `classify` generator was REMOVED on 2026-08-16 (closed-form
# target — see Current Status), mirroring the same cleanup in train_gnn.py. Its
# code is in git history; the datasets it produced stay frozen on disk.
```

### LLM Rule Extraction
```bash
# Stage 1: Extract candidate rules (Qwen3 extractor) — open vocabulary; writes *_candidates.jsonl
# Add --debug-raw to dump full model responses to <out>/_raw/ whenever JSON parsing fails
python extraction/extract.py --docs data/documents/ --out rules/     # DONE - do not re-run

# Stage 2: Translate conditions onto CONDITION_VOCABULARY (same model, reloaded)
#          → *_translated.jsonl + *_untranslatable.jsonl
python extraction/translate.py --candidates rules_35b/ --out rules/

# Stage 2.5: polarity guard (deterministic, no LLM/GPU) — --report first to pick the cutoff
python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14 --report
python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14

# Stage 3: Validate guarded rules (Nemotron-3 Nano 30B) — reads guarded/*_translated.jsonl
python extraction/validate.py --candidates rules/guarded/

# NOTE: build_kg.py is NOT part of the current flow — Component C is being redesigned.
```
Stage 1 is complete (`rules_35b/`, 2,463 candidates) — do not re-run it. v1 outputs are archived
at `rules/v1_archive/`. ⚠️ `deduplicate_rules` merges every `*_confirmed.jsonl` in the out dir —
keep old outputs out of `rules/`, and never name a guard output `*_confirmed.jsonl`.

```bash
# Per-line base kV sidecar (needed by the shield's voltage_pu conversion)
python scripts/dump_base_kv.py --tag case14              # backend method (research PC)
python scripts/dump_base_kv.py --tag case14 --empirical  # from existing JSONL (any machine)
```

### Model Training
```bash
# Train the N-1 GNN (36-bus NeurIPS 2020 — only topology used for training).
# No task switch: N-1 screening is the only model. GRID_TASK is gone.
python scripts/preprocess.py
python training/train_gnn.py --epochs 30 --batch_size 128 --lr 3e-4

# Ablation: drop message passing from the readout (writes *_headonly.pt, never
# overwrites the deliverable). Reported result: 0.8411 vs 0.8987 with it.
python training/train_gnn.py --head-only

# --report-train scores a train slice each epoch; the train/val gap is what
# separates memorisation from an optimiser that never fitted the signal.
python training/train_gnn.py --report-train

# GRID_DEVICE forces a backend (cpu/xpu/cuda) — added to distinguish backend
# bugs from modelling problems.
$env:GRID_DEVICE = "cpu"; python training/train_gnn.py
```

### Verification and Cross-Topology Evaluation
```bash
# Sanity-check a generated N-1 set BEFORE training on it: structure, scenario
# diversity, mixed-frame fraction, and whether the task is still non-degenerate.
python scripts/verify_n1_dataset.py --tag neurips2020

# Per-contingency metrics vs both baselines. neurips2020 scores the held-out
# test split; foreign topologies score every frame. All three RUN — results and
# caveats in component_d_plan.md §7.6.
python evaluation/eval_n1_cross_topology.py --tag neurips2020   # F1 0.8972  (1.93x rule)
python evaluation/eval_n1_cross_topology.py --tag case14        # F1 0.4477  (0.83x rule — FAILS)
python evaluation/eval_n1_cross_topology.py --tag wcci2022      # F1 0.5721  (1.16x rule)
```

### Data Inspection
```bash
# Inspect dataset statistics
python scripts/verify_n1_dataset.py --tag neurips2020
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
  grid_dataset_*_basekv.json       # per-line base kV sidecars (scripts/dump_base_kv.py) — shield voltage_pu conversion
  documents/                       # IEEE/NERC/FERC/AEMO PDFs for LLM extraction (not present on every machine)

# NOTE: normalization_stats.pt (z-score stats from the 36-bus training split, required at
# inference) lives at the REPO ROOT. Written by training/train_gnn.py.

extraction/                        # Component B — LLM rule extraction pipeline
  extract.py                       # Qwen3-14B extraction pass → rules/*_candidates.jsonl
  validate.py                      # Nemotron-3 Nano 30B validation pass → *_confirmed/*_flagged
  build_kg.py                      # Assembles NetworkX DiGraph from validated rules → kg/
  common.py                        # Shared prompt/schema helpers

rules/                             # Component B outputs (one *_candidates/_confirmed/_flagged.jsonl per source doc)
  all_rules_deduped.jsonl          # produced by validate.py; does not exist yet (stages 2-3 pending)
  validation_run_summary.json      # per-document confirm/correct/reject counts from the validation pass

kg/                                # Component C — knowledge graph artifacts (built by extraction/build_kg.py)
  knowledge_graph.pkl              # RETIRED v1 artifact - not used by the shield
  kg_full.html / rules_only.html / rule_R_001_subgraph.html   # pyvis visualizations
  severity_breakdown.png, rule_local_sample_2d.{png,svg,pdf}  # EDA plots
  # NOTE: the shield lives in shield/ at the repo root, NOT here. This KG is a retired v1 artifact.

scripts/
  generate_dataset.py              # Grid2Op simulation → JSONL records (--task n1 | forecast | classify)
  preprocess.py                    # JSONL → processed_grid_data_n1.pt
  pyg_data.py                      # PyG wrapper: GridDataset, feature builders, build_line_targets
  verify_n1_dataset.py             # pre-training sanity check on a generated N-1 set
  dump_base_kv.py                  # per-line base kV sidecar for the shield

training/
  train_gnn.py                     # Trains the N-1 GNN — the only model. Chronic-level split built inline.
  config.py                        # TRAIN_CONFIG (auto-selected by device), artifact paths, GRID_DEVICE override

evaluation/
  eval_n1_cross_topology.py        # per-contingency metrics vs the all-positive and rule baselines
  eval_shield.py                   # ⚠️ classify-era harness, awaiting rewrite for N-1 (plan §7.5 item 5)
  summarize_shield_results.py
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
  the closure doc for why this, and not an architecture change, is
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
| Normalization | Z-score from training split only → saved to `normalization_stats.pt` (repo root) | same |

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

- **Extractor / Translator:** Qwen3 via Ollama, auto-selected by device in `extraction/common.py`
  — `qwen3.6:35b` when `DEVICE.type == "cuda"` (research PC), else `qwen3.5:9b` (personal PC
  testing). Override with the `EXTRACTOR_MODEL` env var. Thinking is suppressed via
  `generate_no_think()` (`think=False`), with `/no_think` kept only as belt-and-braces.
- **Validator:** Nemotron-3 Nano 30B (A3B MoE) via Ollama — `think=False` enforced, confirms/corrects/rejects each candidate

Sequential only — never load both models simultaneously. `keep_alive=0` on every Ollama call to release VRAM immediately.

### Pipeline

```
PDF chunks → Qwen3-14B → JSON candidates → Pydantic validation → Nemotron-3 Nano 30B → CONFIRM/CORRECT/REJECT → dedup → KG
```

### v1 Results (SUPERSEDED — kept as the negative-result record)

| Stage | Count |
|---|---|
| Candidate rules extracted | 1,372 |
| Confirmed (CONFIRM) | 50 |
| Corrected and retained (CORRECT) | 451 |
| Rejected (REJECT) | 766 |
| Flagged for review | 105 |
| **Unique rules after dedup** | **469** |

v2 stage 1 produced **2,463 candidates** across 16 documents (`rules_35b/`); stages 2–3 pending.

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

### v1 Statistics (RETIRED — the graph is being redesigned, see Current Status)

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

**BUILT (2026-08-15) — but the pseudocode below is the OLD design and no longer matches the code.**
The real implementation is `shield/{context,evaluator,shield}.py`; read those and
`supplimentary_docs/component_d_plan.md` §5–§6 instead. Three differences that matter:

1. **Rule retrieval is not KG-based.** `validate(context, rules)` takes a plain rule list behind a
   `RuleProvider` protocol, so the Component C redesign cannot invalidate the shield.
2. **`voltage_pu = min(v_or)/150.0` is wrong** — per-line base kV with energized-line masking (§5).
3. **Rules have roles.** Only `CONSTRAINT` violations block; `AFFIRMATION` rules supply supporting
   evidence and never block on their own (Option A).

Every GNN prediction passes through the shield; there is no bypass mode.

### Validation Flow (SUPERSEDED design — historical)

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

⚠️ **This labelling function is CLOSED-FORM over the observation** — four threshold rules on
`rho_max` and `n_tripped_lines` reproduce it with 100% agreement on 55,000 records. That is why
Component A was reopened for the binary forecast task; see Current Status and plan §2. The
`classify` target below cannot distinguish a learned model from a threshold.

**`NO_OVERFLOW_DISCONNECTION = False`** — must be passed at `grid2op.make()` via `param=params`, not set post-make. Enables natural cascades.

---

## Hardware

### Research PC — LLM stages only

- RTX 4080 Super (16GB VRAM), i7-14700K, 64GB DDR5
- Runs: **LLM stages only** — `translate.py`, then `validate.py` (sequential, never both models loaded)
- Schedule: Saturdays, Mondays, Wednesdays

### Personal PC — Data Generation, Training, Shield

- Intel Arc B580 (12GB VRAM), Ryzen 5 7500F, 16GB DDR5
- Runs: forecast dataset generation, **GNN training**, polarity guard, shield, all tests — no LLM inference.
  `training/config.py` auto-selects the small `[16,32,32]` config here, which is the proven one;
  the CUDA branch `[64,128,128]` has never produced a working checkpoint.

**Nemotron-3 Nano 30B note:** ~18–20GB total (partial CPU offload into 64GB RAM). Qwen3-14B runs first (~8–10GB, fully GPU-resident), then Nemotron-3, never concurrent.

---

## Key Implementation Constants

| Constant | Value | Context |
|---|---|---|
| `num_workers` | `0` | Windows PyG DataLoader constraint |
| ~~Nominal voltage~~ | ~~`150.0 kV`~~ | **SUPERSEDED** — the shield uses per-line base kV from `data/grid_dataset_<tag>_basekv.json`. A flat divisor gives ~100% false blocks. Still used only for the GNN's `mean_v` node feature. |
| Base kV method | `backend`, all 3 tags | **Settled 2026-08-16** (`component_d_plan.md` §5.1). Backend nominals: neurips 138/345, case14 14/20/138, wcci2022 138/161/345. These grids *operate* ~6% above nominal, so healthy frames read ~1.06 pu, **not ~1.00**. Empirical originals kept as `*_basekv_empirical.json` — do not delete, they are the counterfactual arm. Never pass `--empirical` again. |
| FAULT_PROB | `0.05` | Grid2Op fault injection rate (`scripts/generate_dataset.py`) |
| RECONNECT_PROB | `0.20` | Grid2Op reconnect probability |
| `--n1-stride` | `12` | Labels every 12th step. The protocol on **all three** topologies — see the sizing note under Dataset Generation |
| ~~NORMAL_KEEP_PROB~~ / ~~LINE_TRIP_KEEP_PROB~~ | — | **REMOVED 2026-08-16** with the classify generator. They drove that task's subsampling quotas; N-1 does not subsample by class. In git history only. |
| Random seed | `42` | Model init + in-memory shuffle before training — deployed init is seed-sensitive, see Component A |
| Checkpoint filename | `gnn_checkpoint_best.pt` — ⚠️ **NOT** identical to `gnn_checkpoint_leverA.pt` (verified: byte-different, all 35 tensors differ by up to ~5e-3; two separate runs). Which one produced 0.8277 is not determinable from the artifacts. | Saved on best macro F1 |
| Logit margin | `gnn_logit_margin.json` | Lever A per-class calibration offsets, applied at inference only |
| Normalization stats | `normalization_stats.pt` (repo root) | Must exist before cross-topology eval |

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
