# Tightening the N-1 GNN — diagnosis and fixes

> ### ✅ STATUS 2026-08-20 — CURRENT. One number to reconcile before citing.
>
> This is the live record of the N-1 model and its normalization fix. ⚠️ The held-out figures
> here (**F1 0.8872 / AP 0.9549**) come from a different run than the checkpoint now on disk,
> which the evaluation harness scores at **0.8972 / 0.9615**. **Treat the checkpoint as
> authoritative** and cite the harness numbers; the gap is run-to-run variance, not a
> methodological difference.
>
> §3's normalization bug is the important part and applies to **every GNN run in this project**,
> including the retired classify checkpoints — read it before citing any pre-2026-08-16 result.


**Date:** 2026-08-16
**Outcome:** val contingency F1 **0.4751 → 0.8987**; held-out test **0.8872** (AP 0.9549).
**Root cause:** a silent normalization failure that meant *every GNN run in this project, on every
task, trained on raw unnormalized features.*

---

## 1. Result

Held-out test split (87 chronics, 113,205 contingencies, never used for training or model
selection):

| | F1 | note |
|---|---|---|
| all-positive baseline | 0.3104 | is the target non-vacuous? |
| best rule on `rho` of the removed line | 0.4639 | the symbolic bar |
| skip features only, **no message passing** | 0.8411 *(val)* | ablation |
| **full GNN** | **0.8872** | **1.91× the rule**, AP 0.9549 |

At the best threshold: recall 0.870, precision 0.905, **2,714 missed violations** out of 20,801 —
the error class the shield's asymmetric gate exists to catch.

**Message passing contributes +0.058** (0.8411 → 0.8987 on val). This matters for the thesis: it is
direct evidence that the graph structure carries information beyond the endpoint features, which is
the claim the whole architecture rests on.

---

## 2. The bug that mattered

### What happened

```python
node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(full_dataset, train_idx)
full_dataset._data.x         = (full_dataset._data.x         - node_mean) / node_std
full_dataset._data.edge_attr = (full_dataset._data.edge_attr - edge_mean) / edge_std
```

This looks correct and is not.

`compute_normalization_stats` iterates `dataset[idx]` over **every training index**. PyG's
`InMemoryDataset` memoises each materialised graph in `_data_list` on first access, and `get()`
serves from that cache in preference to `_data`. So by the time the two assignment lines run, the
cache is fully populated with **unnormalized copies**, and mutating `_data` changes nothing that the
DataLoaders will ever yield.

The network trained on raw physical magnitudes — `load_p` in MW, `p_or` in MW, voltages in kV, with
per-feature standard deviations spanning 0.013 to 98 — while `normalization_stats.pt` was written to
disk as though normalization had been applied.

### Verification

```
before normalization, graph0 x[:3,:3]:   [[ 3.18  0.985  0.126] ...]
after  normalization, graph0 x[:3,:3]:   [[ 3.18  0.985  0.126] ...]
expected if applied:                     [[-0.359 -0.398 -1.623] ...]

normalization visible through dataset[i]: False
dataset[i] still returns RAW values      : True
```

### Fix

Invalidate the cache after mutating `_data`, and assert that the change is actually visible:

```python
full_dataset._data_list = None
_check = full_dataset[0].x[0, 0].item()
assert abs(_check - full_dataset._data.x[0, 0].item()) < 1e-4
```

The assertion is the important half. The failure mode is silent — the code runs, the loss goes down,
a checkpoint is written — so only an explicit check prevents it recurring.

### Why it took so long to find

Because trees are scale-invariant. Gradient boosting on the same features reached 0.908, which made
the ceiling look reachable and pointed suspicion at the network. Every component tested in isolation
passed:

| Hypothesis | Test | Verdict |
|---|---|---|
| labels misaligned with edges | rule baseline through the batched path (0.4725) vs from raw JSONL (0.4757) | ruled out |
| features insufficient | GB on the readout inputs → 0.9083 | ruled out |
| XPU backend bug in scatter-add | identical run on CPU → bit-identical trajectory | ruled out |
| batch composition (graph-grouped vs random) | both samplers → 0.84–0.86 | ruled out |
| train/val leakage | split audit: 403/86/87 chronics, zero overlap | ruled out |
| optimizer settings | same MLP standalone with the trainer's exact lr/clip/pos_weight → 0.8650 | ruled out |

The decisive step was training a plain reference MLP **inside the trainer process, on the trainer's
own loaders**. It failed there (0.42) while succeeding on data normalized by hand (0.86). Identical
model, identical split — so the difference had to be the tensors themselves.

---

## 3. Impact on the frozen classify artifacts — read this before citing them

`gnn_checkpoint_best.pt` (macro F1 0.8277) was trained through this same code path, so **it was
trained on unnormalized features.**

`evaluation/eval_n1_cross_topology.py` normalizes correctly, per batch, at inference:

```python
batch.x         = (batch.x         - node_mean) / node_std
batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
```

That is a **train/inference mismatch**: the frozen checkpoint was fitted on raw magnitudes and is
then evaluated on z-scored ones. Cross-topology degradation was a headline finding of Component A,
and some unknown share of it is attributable to this mismatch rather than to topology transfer.

**Do not silently re-run and replace the classify numbers.** Options, in order of preference:

1. Retrain classify with the fix and report both, framing the difference as a methodological
   finding. This is honest and adds a result rather than removing one.
2. If the frozen numbers must stand, state the mismatch explicitly wherever cross-topology
   degradation is discussed.

The in-distribution classify number is less affected — train and eval were both unnormalized there,
so it is at least self-consistent. It is the *cross-topology* comparison that is compromised.

---

## 4. Other defects fixed in the same pass

| Defect | Consequence if unfixed |
|---|---|
| **Train/val remix** — `Force Shuffle` concatenated train+val and re-split at random, destroying chronic separation | Val contained near-duplicate frames of training chronics; every reported score was partly memorised. Kept for `classify` only, to preserve reproducibility of the frozen deliverable. |
| **`chronic_id` was the episode counter**, not the scenario id, while `set_id` used `chronic_idx % n_chronics` | Past 576 episodes the loop wraps; the same load profile would be written under a fresh id and land in both train and test. Now records the true scenario, with `episode_id` kept for debugging. |
| **`--task n1` wrote to `grid_dataset_<tag>.jsonl`** | Would have overwritten the frozen classify dataset. Every task now carries its own suffix. |
| **Ablation checkpoints overwrote the main one** | `--n1-head-only` has a narrower `line_head`, so the eval script died on a shape mismatch. Ablations now write `*_headonly.pt`. |
| **Stale split files** | A split from a smaller run silently mis-indexes a larger regeneration. Now detected by record count and rebuilt. |

---

## 5. Feature changes (kept, but not the fix)

Added for `n1` only — `classify`/`forecast` stay at 5/4 so the frozen checkpoint keeps loading:

- edge: `|p_or|`, `|q_or|`, apparent power, `headroom = max(0, 1-rho)`
- node: `sum_headroom`, `sum_abs_p`, `degree` — per-bus spare capacity, throughput, alternative paths

The reasoning was sound — `p_or` was raw and **signed**, so magnitude (the actual predictor) had to
be synthesised through a z-scored feature centred near zero, which a linear readout cannot express.
But measured on their own, before the normalization fix, these features moved val F1 by **+0.003**
(0.4884 → 0.4911). They were not the bottleneck. They are retained because they are physically
motivated and cost nothing, but the write-up should not claim they caused the improvement.

A **skip connection** from the raw endpoint features to the line readout was added at the same time,
for the same reason, and is likewise not the cause of the gain.

---

## 6. Method note worth carrying forward

The general lesson, and it generalises past this bug:

> **When a network underperforms a single-feature rule, do not tune it. Fit a scale-invariant model
> (gradient boosting) on the exact tensors the network receives.** If that model succeeds, the
> information is present and the fault is in the network or its inputs — not in the task, the
> features, or the architecture.

That one comparison converted an open-ended "the GNN is underperforming" into a bounded search, and
every subsequent step was elimination against a known-reachable ceiling of 0.908.

The second lesson: **a metric that is never asserted on will eventually be wrong.** The normalization
was computed, printed to the console, and saved to disk — all of which looked like evidence it was
working. Only comparing `dataset[i]` against `_data` revealed otherwise.

---

## 7. Still open

- **Cross-topology evaluation** — `case14` n1 set not yet generated; `wcci2022` env not downloaded.
- **Classify retrain** under the normalization fix (§3) — a decision for the thesis, not a bug fix.
- **Shield integration** on real n1 predictions — `validate_n1` is built and tested (9 tests) but has
  not run against model output, because the rule corpus still needs the LLM stages.
- **Threshold selection** — all model numbers are best-threshold. A deployment threshold should be
  chosen on val and reported on test, especially since the shield's asymmetric gate cares about the
  missed-violation count specifically.
