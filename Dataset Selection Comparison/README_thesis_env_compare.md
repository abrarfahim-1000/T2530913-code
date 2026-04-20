# Thesis-Focused Grid2Op Environment Comparison Guide

## What this comparison is trying to prove

This is **not** a generic “which environment gives better accuracy?” benchmark.
It is a **thesis-fit benchmark** for a neuro-symbolic smart-grid thesis where the final pipeline is:

1. Grid simulation generates graph-structured states
2. A GNN learns fault-related patterns from those states
3. A symbolic layer / shield checks whether a recommendation violates rules
4. The chosen environment should therefore be:
   - rich enough to produce meaningful fault states
   - manageable enough to finish the full pipeline
   - compatible with the current symbolic scope of the thesis
   - computationally practical on the available hardware

## What the script measures

The script compares the two environments using five score families:

### 1) Symbolic fit score
This is **thesis-specific**, not universal.
It penalizes environments that increase immediate symbolic integration burden.

Why?
Because your current methodology explicitly models buses, lines, generators, loads, rules, and the shield.
WCCI 2022 adds storage, which is a real extra modeling burden for the first full prototype.

### 2) Compute efficiency score
This checks whether an environment is realistic to use for the whole thesis pipeline.
It combines:
- collection throughput
- epoch time
- RAM use
- GPU memory use

### 3) Data richness score
This measures whether, under the same sampling budget, the environment gives enough useful event diversity.
It combines:
- how many non-normal fault classes actually appear
- class diversity / class entropy

### 4) Pilot model score
This is the **learning** part, but it is not just raw accuracy.
It combines:
- macro F1
- balanced accuracy
- minority fault recall

This matters because a fault dataset is usually imbalanced.

### 5) Graph build success score
This checks how reliably the environment can be turned into the graph representation used by the pilot GNN.

## Final score weighting

The final thesis score intentionally underweights pure predictive performance.

```text
Final Thesis Score =
  0.25 * symbolic_fit_score
+ 0.25 * compute_efficiency_score
+ 0.20 * data_richness_score
+ 0.20 * pilot_model_score
+ 0.10 * graph_build_success_score
```

So even if one environment gets slightly better classification metrics,
it might still lose overall if it is much harder to integrate into the full thesis.

## Install

You need Python 3.10+ recommended.

Core packages:

```bash
pip install numpy pandas scikit-learn psutil
pip install grid2op lightsim2grid
pip install torch torchvision torchaudio
pip install torch_geometric
```

If `torch_geometric` gives installation trouble on your machine,
use the official PyG install page for the exact wheel matching your Torch/CUDA version.

## Run

Default run:

```bash
python thesis_env_compare.py --outdir thesis_env_compare_results
```

A faster smoke test:

```bash
python thesis_env_compare.py --n-episodes 8 --steps-per-episode 128 --epochs 8 --outdir quick_compare
```

A stronger pilot run:

```bash
python thesis_env_compare.py --n-episodes 24 --steps-per-episode 384 --epochs 30 --hidden-dim 64 --outdir stronger_compare
```

## What files you will get

- `env_static_summary.csv`
  - number of substations, lines, generators, loads, storage units

- `collection_summary.csv`
  - sample collection throughput
  - graph build success rate
  - RAM usage
  - class counts

- `model_summary.csv`
  - macro F1
  - balanced accuracy
  - minority fault recall
  - per-class metrics
  - training time
  - GPU memory

- `label_distribution.csv`
  - class distribution observed during collection

- `thesis_scorecard.csv`
  - all final comparison scores

- `final_recommendation.txt`
  - short thesis-focused recommendation text

- `<env_name>_confusion_matrix.csv`
  - confusion matrix for each environment

## How to explain this to your supervisor

Use this logic:

> We did not choose the environment based on accuracy alone.
> We benchmarked both candidate environments under the same sampling budget,
> using the same graph construction method and the same pilot GNN,
> then evaluated them with thesis-relevant criteria:
> symbolic integration burden, compute efficiency, data richness,
> minority-fault learning quality, and graph-construction stability.
> The final decision was based on the weighted thesis score,
> where raw model quality is only one part of the decision.

## Important limitation to say clearly

This is a **pilot environment-selection benchmark**.
It is not the final thesis result and not the final production GNN.
Its purpose is to justify which environment is the better primary platform
for building the full neuro-symbolic pipeline.
