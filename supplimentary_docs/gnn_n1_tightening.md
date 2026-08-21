# Tightening the N-1 GNN — diagnosis and fixes

> ### ✅ STATUS 2026-08-21 — CURRENT. Two things to know before citing.
>
> This is the live record of the N-1 model and its normalization fix.
>
> **1. EVERY ABSOLUTE MODEL F1 IN THIS PROJECT IS CONDITIONAL ON THE EVAL BATCH SIZE.**
> Measured 2026-08-21 on the tracked checkpoint — see §8. `BatchNorm(track_running_stats=False)`
> uses **live batch statistics at inference**, so batch composition changes the logits. The same
> weights on the same data score **0.8972 at batch 64 and 0.9255 at batch 512**. Both eval scripts
> default to 64, which is where every reported figure comes from.
> **Report deltas, not levels** — the shield's delta and its intervention precision are stable
> (§8); the absolute F1 is not. Quote any model F1 with its batch size attached.
>
> This supersedes the note that stood here, which said the §1/harness gap was "run-to-run variance,
> not a methodological difference." **That was wrong** — it is methodological, and the batch-size
> effect (0.028) is larger than the gap it was explaining (0.010). §1's **0.8872 / AP 0.9549** does
> not reproduce from the tracked checkpoint at any batch size tested; treat it as the record of the
> run that produced the fix, not as a citable figure.
>
> **2. §3's normalization bug is the important part.** It applies to **every GNN run in this
> project**, including the retired classify checkpoints — read it before citing any
> pre-2026-08-16 result. ⚠️ §3 was corrected on 2026-08-21: it previously named
> `eval_n1_cross_topology.py` as the script with the train/inference mismatch. That is wrong.
> That script scores the **N-1** checkpoint, which was trained *after* the fix and carries no
> mismatch. The affected script was the classify cross-topology script, since deleted.
>
> §4, §5 and §7 were also updated on 2026-08-21; §1, §2 and §6 are unchanged.


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

⚠ **These are best-threshold (oracle) figures**, chosen with the answer key on this split. Every
table outside this section now reports the **held** threshold instead (0.8849, selected once on
val) — see §7. For the tracked checkpoint the held in-distribution figure is **0.8956**. Do not
mix the two protocols in one table.

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

> ⚠️ **RESOLVED 2026-08-20 — the decision below was taken, and option 1 is no longer available.**
> Classify was demoted to a page of methodology (`revised_thesis_claim.md` §2) and **all its
> artifacts were deleted**: `gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt`,
> `normalization_stats.pt` and both datasets. They were untracked, so a retrain would no longer
> reproduce them. Option 2 is what stands, and this section is the statement it requires.
> The analysis below is retained because it is the reason for that decision.

`gnn_checkpoint_best.pt` (macro F1 0.8277) was trained through this same code path, so **it was
trained on unnormalized features.**

The **classify cross-topology script** normalized correctly, per batch, at inference:

```python
batch.x         = (batch.x         - node_mean) / node_std
batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
```

That is a **train/inference mismatch**: the frozen checkpoint was fitted on raw magnitudes and was
then evaluated on z-scored ones. Cross-topology degradation was a headline finding of Component A,
and some unknown share of it is attributable to this mismatch rather than to topology transfer.

⚠️ **That script no longer exists** — it was deleted on 2026-08-20 and is recoverable only from
`git show a5c5199:`. Do not read the surviving `evaluation/eval_n1_cross_topology.py` as the
script described here: it carries the same per-batch normalization, but it scores the **N-1**
checkpoint, which was trained *after* the fix. Train and inference agree there, so **the N-1 arm
carries no mismatch.**

The in-distribution classify number is less affected — train and eval were both unnormalized there,
so it is at least self-consistent. It is the *cross-topology* comparison that is compromised, and
it must not be presented as evidence about generalisation.

---

## 4. Other defects fixed in the same pass

| Defect | Consequence if unfixed |
|---|---|
| **Train/val remix** — `Force Shuffle` concatenated train+val and re-split at random, destroying chronic separation | Val contained near-duplicate frames of training chronics; every reported score was partly memorised. Kept for `classify` only, to preserve reproducibility of the frozen deliverable. |
| **`chronic_id` was the episode counter**, not the scenario id, while `set_id` used `chronic_idx % n_chronics` | Past 576 episodes the loop wraps; the same load profile would be written under a fresh id and land in both train and test. Now records the true scenario, with `episode_id` kept for debugging. |
| **`--task n1` wrote to `grid_dataset_<tag>.jsonl`** | Would have overwritten the frozen classify dataset. Every task now carries its own suffix. |
| **Ablation checkpoints overwrote the main one** | `--head-only` has a narrower `line_head`, so the eval script died on a shape mismatch. Ablations now write `*_headonly.pt`. |
| **Stale split files** | A split from a smaller run silently mis-indexes a larger regeneration. Now detected by record count and rebuilt. |

---

## 5. Feature changes (kept, but not the fix)

Added for `n1`. ⚠️ The note that stood here — *"`classify`/`forecast` stay at 5/4 so the frozen
checkpoint keeps loading"* — is **obsolete**. The classify generator and its checkpoints were
removed on 2026-08-16/20, and `scripts/pyg_data.py` now declares `NODE_FEATURES = 8` /
`EDGE_FEATURES = 8` unconditionally, with no task branch. Nothing is held back for compatibility.

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

## 7. Still open — reviewed 2026-08-21

Three of the four items below closed. Kept with their outcomes rather than deleted, so the
sequence stays legible.

| item | status |
|---|---|
| **Cross-topology evaluation** | ✅ **DONE 2026-08-16**, re-reported at the held threshold 2026-08-21. neurips2020 **0.8956** · case14 **0.4167 — loses to the rule baseline (0.77×) and to the all-positive baseline (0.96×)** · wcci2022 **0.5577**. Oracle (best-threshold) figures were 0.8972 / 0.4477 / 0.5721; they are retained only as a labelled ceiling. Table and caveats: `thesis_findings.md` §14. |
| **Classify retrain** under the normalization fix | ✅ **DECIDED 2026-08-20 — not retrained.** Classify was demoted and its artifacts deleted; see the §3 banner. |
| **Shield integration** on real n1 predictions | ✅ **DONE 2026-08-20.** Ran on all three topologies against the validated 4-rule corpus: +0.0082 / +0.0021 / **+0.0676** F1 at 0.938 / 0.922 / 0.934 intervention precision. `thesis_findings.md` §13. |
| **Threshold selection** | ✅ **CLOSED 2026-08-21** for reporting. One held threshold — 0.8849, selected on the neurips2020 val split — is now used for the shield arm *and* the raw-model tables everywhere (`thesis_findings.md` §14, `revised_thesis_claim.md` §4.2, `study.md`, `CLAUDE.md`). Best-threshold figures are retained only as a labelled oracle ceiling. The recorded prediction that this "would lower all three rows, and lower case14 and wcci2022 most" held: −0.0015 / −0.0310 / −0.0144. **case14 falls to 0.77× the rule baseline and 0.96× the all-positive baseline** — it loses to answering "violation" every time. ✅ The selection *objective* is closed too (2026-08-21): the full F-beta trade curve is measured in `thesis_findings.md` §14.2 via `evaluation/sweep_threshold.py`. **F1 is retained** — choosing any beta > 1 needs an operator cost ratio nobody has measured — with the alternative reported alongside. Verified that intervention precision stays flat (0.932–0.937) at an F2 threshold, so the headline claim does not depend on the objective. ⚠ Note §14.2's incidental finding: on case14 a recall-weighted threshold *improves* F1 (0.4167 → 0.4428), so that grid's held-vs-oracle gap is largely threshold mis-transfer. §1's figures remain best-threshold and are labelled as such. |

See §8: the absolute figures are not a stable property of the checkpoint at all.

---

## 8. The eval is not batch-invariant — measured 2026-08-21

`gnn_checkpoint_n1.pt` is tracked (committed at `f9c5328`) and both `eval_n1_cross_topology.py` and
`eval_shield_n1.py` load it, so every reported number does come from one fixed set of weights.
**The scores are still not a stable property of those weights.**

Scored on the held-out neurips2020 test split, varying **only** the eval batch size:

| eval batch size | model F1 | AP | recall | precision | missed violations |
|---:|---:|---:|---:|---:|---:|
| **64** — both scripts' default, and the source of every reported figure | **0.8972** | 0.9615 | 0.885 | 0.910 | 2,397 |
| 128 | 0.9088 | 0.9684 | 0.905 | 0.913 | 1,979 |
| 256 | 0.9241 | 0.9772 | 0.925 | 0.924 | 1,590 |
| 512 — `TRAIN_CONFIG` batch size, what training evaluated at | **0.9255** | 0.9780 | 0.918 | 0.934 | 1,716 |

A spread of **0.028 F1** on identical weights and identical data.

**Cause: `BatchNorm(track_running_stats=False)`.** The choice is deliberate and defensible —
running statistics flatten the overload spikes at `rho > 1.0` that the task is about — but the
consequence is that the network uses **live batch statistics at inference**, so batch composition
changes the logits. It is not a bug; it is a property of the architecture that was never written
down.

**Isolated, not assumed.** Across all four runs the positive count is constant at 20,801 and the
rule baseline is constant at 0.4639, so the split, the labels and the label alignment are
identical. The variation is entirely in the model's forward pass.

### What survives, and what does not

**Does not survive:** any absolute model F1 quoted without its batch size. That includes §1, the
cross-topology table, and the raw-model column everywhere else.

**Survives:** the shield comparison, because both arms share one forward pass.

| | model | +shield | **delta** | **intervention precision** |
|---|---:|---:|---:|---:|
| batch 64 | 0.8956 | 0.9038 | **+0.0082** | **0.938** |
| batch 512 | 0.9216 | 0.9288 | **+0.0072** | **0.930** |

The delta moves by 0.001 and the precision by 0.008 across a batch-size change that moves the raw
model by 0.026. **The thesis claim is a claim about deltas and about precision, and both are
stable.** Report it that way.

### DECIDED 2026-08-21 — keep 64, and it is now pinned in code

The batch size **stays at 64**. It is what every recorded result was measured at, and the shield
deltas — the quantities the thesis actually claims — are stable regardless.

It is no longer an incidental argparse default. Both `evaluation/eval_n1_cross_topology.py` and
`evaluation/eval_shield_n1.py` now declare:

```python
EVAL_BATCH_SIZE = 64
```

with the table above in a comment beside it, and both **print a warning if the value is
overridden** on the command line, saying that absolute F1 will not match any recorded result while
deltas remain comparable. Pinned by
`tests/test_eval_shield_n1.py::test_eval_batch_size_is_pinned_at_64`.

Standing consequences:

1. **Quote the batch size with any model F1.** Everything in the docs is batch 64.
2. **This is a second protocol axis, on top of threshold selection (§7).** Threshold is closed for
   the shield arm and open for the raw-model table; batch size is now closed for both.
3. **Do not "fix" this by switching to running statistics** without re-measuring — that changes
   the model's behaviour on exactly the overload spikes the task is about.
4. If the batch size is ever changed, **every recorded number must be regenerated together.** A
   mixed table is the failure this pinning exists to prevent.
