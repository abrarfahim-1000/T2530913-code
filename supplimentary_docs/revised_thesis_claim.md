# Revised Thesis Claim — Component A and Component D

**Status:** supersedes the fault-classification framing throughout the draft.
**Date:** 2026-08-16
**Audience:** whoever is editing the thesis text. Everything below is measured, not projected;
the numbers are reproducible from the probes described in §6.

---

## 1. The claim, in one paragraph

> We train a graph attention network to perform **N-1 contingency screening** on a power grid: for
> every energized line, the model predicts whether tripping that line would drive the network
> outside its thermal limits. The task is chosen because its label cannot be reproduced by any rule
> over the observed state — producing it requires a post-contingency power-flow solve — which makes
> the comparison between the neural model and the symbolic layer non-degenerate by construction.
> Model predictions are then gated by a **symbolic shield** built from rules extracted from IEEE,
> NERC, FERC and AEMO standards by an LLM pipeline. The shield does not re-derive the contingency
> analysis; it enforces the operating principle that **N-1 security presupposes a secure base case**,
> blocking any prediction that declares the grid safe against a further outage while the present
> state already violates a standard. The gate is deliberately asymmetric: over-permissive
> predictions are blocked, over-cautious ones are always allowed through.

---

## 2. What changed from the previous draft, and why

> **Editorial decision (2026-08-16): the fault-classification work is DEMOTED from a results chapter
> to roughly one page of methodology.** It is not deleted — the closed-form finding below is what
> *justifies* the N-1 task design, and without it section 3 reads as an arbitrary choice that any
> examiner will challenge with "why not fault classification, like the rest of this literature?"
> What goes is the apparatus built to *optimise* a target we have since proven degenerate: the
> Lever A logit-margin calibration, the seed-sensitivity study, the three rounds of rejected
> architecture experiments (GSAT, soft-F1, the `[64,128,128]` scale-up), the triple-pooling
> bottleneck diagnosis, and the cross-topology classification results.
>
> Two independent reasons, either of which is sufficient:
>
> 1. **The target is closed-form** (below), so no amount of tuning on it can answer the research
>    question. A chapter of hyperparameter work performed *after* that was established reads as a
>    failure to act on your own finding.
> 2. **The numbers are compromised.** The frozen checkpoint was trained on unnormalized features
>    (see §6.1), while `eval_cross_topology.py` normalizes at inference — a train/inference
>    mismatch. Keeping the chapter would force either a retrain (hours, to better answer a rigged
>    question) or a caveat on every cross-topology number. Demotion disposes of both.
>
> Crucially, **the closed-form argument is robust to reason 2**: it does not depend on the network
> being any good, only on rules reaching 100%. So it can be stated without defending 0.8277 as a
> rigorous figure.

The thesis previously claimed a **4-class fault classifier** (`normal` / `overload` / `line_trip` /
`cascade`) validated by a symbolic shield. That claim does not survive scrutiny, and the reason is
worth stating in the paper rather than hiding:

**The 4-class label was a closed-form function of the observation.** Four threshold rules on
`rho_max` and `n_tripped_lines` reproduce the stored labels with **100% agreement on 55,000 records**
across both topologies. A symbolic layer therefore scores 100% where the trained GATv2 reached
0.8277 macro F1. The question "does the shield add value over the GNN" was answered before the
experiment ran.

Two replacement targets were then built and **rejected by measurement before any full training run**:

| Candidate target | Why it was rejected |
|---|---|
| **Any-fault forecast** — will *any* fault occur within H steps? | Line-trip onset is injected by an unconditional Bernoulli draw on a uniformly random line, so it is unpredictable **by construction**, and trips are 71–88% of all positives. The best possible threshold on current `rho_max` scores **1.00× the all-positive baseline at every H ≥ 3**. No model can beat "always say at-risk". |
| **Overload-only forecast** — will `rho ≥ 1.0` within H steps? | Learnable, but exhausted by a single rule. Gradient boosting on the full feature vector reached **F1 0.163 against the single threshold's 0.160**, with *worse* average precision (0.125 vs 0.127). Adding trend features over 1/3/6 frames on contiguous pilots changed nothing on either topology (lift 0.67–1.02× across H ∈ {2…18}). A single snapshot carries no information about future exogenous load. |

**N-1 screening fails neither test**, which is why it was adopted. See §3.

### The methodological point worth making in the paper

These two rejections are a contribution, not wasted effort. They establish that **a task must pass
two independent checks** before it can support a neuro-symbolic comparison:

1. **Non-vacuity** — the target must not be predictable from the base rate alone.
2. **Non-exhaustion** — a *model* must beat the best *rule*, not merely beat the base rate.

The overload-forecast target passed check 1 with a 2.76× lift and failed check 2 outright. A
literature that only reports check 1 will accept targets on which the neural component is doing no
work. We recommend reporting both.

---

## 3. Why N-1 screening is the right task

Measured on the 36-bus L2RPN NeurIPS 2020 training topology:

| Predictor | F1 |
|---|---|
| All-positive baseline (2p/(1+p)) | 0.311 |
| Best threshold on the **global** current `rho_max` | 0.289 — **1.04×**, i.e. nothing |
| Best single rule: **flow on the line being removed** | 0.608 |
| Model with **network context** | **0.868** (AP 0.937 vs 0.628) |

**These were pre-generation probe figures. The trained model has since confirmed them — see §4.1
for the measured result on the real dataset and held-out test split.**

Three properties follow, and each maps onto a claim the paper can make:

1. **The present state does not determine the answer.** Global `rho_max` is worthless here (1.04×),
   in direct contrast to the classify task where four such thresholds were sufficient.
2. **The answer is topological.** **100% of sampled frames are mixed** — within a single frame, some
   contingencies violate and others do not. A graph-level prediction cannot express this; the
   quantity being predicted lives on edges.
3. **There is a real gap for the model to occupy.** A strong local rule reaches 0.608 and a
   network-aware model reaches 0.868. Neither end is degenerate: the symbolic baseline is
   respectable rather than trivial, and the model's advantage is substantial rather than noise.

Base rate is ~16–26% violation depending on topology and stress, so the task is not a rare-event
problem. Labels cost ~170 power-flow solves per second.

---

## 4. What the GNN does now

**Input:** unchanged — one graph per frame of the *pre-contingency* state. Nodes are buses
(5 features), edges are energized lines (4 features), tripped lines pruned from `edge_index`.
The GATv2 backbone, live-batch-statistics normalization, and zero-dropout choices all carry over
with their original justifications intact.

**Output — this is the architectural change.** The global mean/max/min pooling and the graph-level
classifier head are replaced by an **edge-level head**: one logit per line, read off the or→ex edge
as `MLP([h_or ‖ h_ex ‖ edge_attr])`. Nothing is pooled, because the answer to "is losing line *k*
safe" is local to *k* and its neighbourhood, and pooling would erase precisely the per-line
distinction the task is about.

**Two consequences worth noting in the text:**

- **Cross-topology transfer becomes more natural, not less.** An edge-level head has no dependence
  on `n_line`, so 20-line case14 and 186-line WCCI 2022 run unchanged. The previous localization
  head emitted `n_nodes` logits and had to be *disabled* for cross-topology evaluation; this one
  does not.
- **The triple-pooling bottleneck is dissolved rather than fixed.** The `normal`/`line_trip`
  confusion previously diagnosed as a pooling limitation is no longer expressible, because there is
  no pooling operator on the prediction path.

**Loss:** masked binary cross-entropy over individual contingencies. Entries where the power flow
could not be evaluated (line already out of service, or solver divergence — under 1% of cells)
carry a `-1` sentinel and are **excluded from the loss**. They are missing labels, not negatives;
folding them in as `secure` would teach the model that losing an already-dead line is safe.

**Metric:** per-contingency F1, reported against both baselines from §3 at matched thresholds.

### 4.1 Measured result

Dataset: 12,000 frames over **all 576 available chronics** of the 36-bus grid, 702,618 contingency
labels, 19.5% violation rate. Chronic-level 70/15/15 split (403 / 86 / 87 chronics, zero overlap).
Held-out **test** split, never used for training or model selection:

| | F1 | |
|---|---|---|
| all-positive baseline | 0.3104 | is the target non-vacuous? |
| best rule on `rho` of the removed line | 0.4639 | the symbolic bar |
| skip features only, **message passing disabled** | 0.8411 *(val)* | ablation |
| **full GNN** | **0.8872** | **1.91× the rule**, AP 0.9549 |

At the operating threshold: recall 0.870, precision 0.905, **2,714 missed violations of 20,801** —
the error class §5 exists to catch.

**Message passing contributes +0.058** (0.8411 → 0.8987 on val). Report this ablation: it is the
direct evidence that graph structure carries information beyond the two endpoint buses, which is the
premise the architecture rests on. Without it, a reviewer is entitled to ask whether an MLP on
endpoint features would have done just as well.

---

## 5. What the shield does now — and the limit on that claim

### 5.1 The mechanism

The shield gates each per-contingency verdict. Its logic:

```
predicted "secure"    + base case violates a CONSTRAINT rule   ->  BLOCK
predicted "secure"    + base case clean                        ->  PASS
predicted "violation" + anything                               ->  PASS
```

The justification is a standard operating principle: **N-1 security presupposes a secure base
case.** A grid already outside its thermal or voltage limits cannot be declared safe against the
loss of a further element. A `secure` verdict issued over such a base case is therefore
unsupportable regardless of what the contingency analysis would have shown.

The asymmetry is deliberate. Predicting `violation` on a calm grid is a false alarm — wasteful, but
it fails toward caution, and a safety gate must never suppress it. Predicting `secure` on a grid
already outside its limits is the failure that endangers a network. Only that direction blocks.

AFFIRMATION rules retain Option A semantics: they supply supporting evidence for a `secure` verdict
and never block on their own. Absence of support is recorded (`unsupported`) so the harness can
report what a gating shield *would* have done without paying its false-block rate.

### 5.2 The limit — state this explicitly in the paper

**The shield does not verify the contingency analysis.** Establishing whether losing line *k* truly
violates a limit requires a post-contingency power flow, which is exactly the computation the GNN
exists to replace. If the shield could run it, the GNN would be unnecessary.

What the shield verifies is that the verdict is **consistent with the rules governing the present
state**. The defensible sentence is:

> The shield catches unsafe-permissive predictions using present-state standards.

and **not**:

> The shield independently verifies the contingency analysis.

This is a narrower claim than the previous draft implied, and it is the claim the extracted rule
corpus can actually support. It remains a real result: it is a genuine safety property, enforced by
rules traceable to named standards, and it is the property an operator would actually want.

---

## 6. Reproducibility

| Finding | How it was measured |
|---|---|
| Classify label is closed-form | 4 threshold rules vs stored labels, 55,000 records, both topologies |
| Any-fault forecast is unlearnable | 1,995 quota-unbiased chronics / 321,966 reconstructed steps; best threshold on `rho_max` vs all-positive F1, calm frames only |
| Overload forecast is exhausted by one rule | HistGradientBoosting on 30 summary features, chronic-level split, 56,441 calm frames / 2,227 positives |
| Trend features do not help | 60,000-frame contiguous pilots on both topologies, deltas at lags 1/3/6, H ∈ {2…18} |
| N-1 numbers in §3 | 17,652 contingency labels from 300 frames, frame-level split, `obs.simulate()` per line |

Reconstruction of the unsubsampled fault timeline is valid because `generate_dataset.py` drops
frames only in the `normal` / `line_trip` / `cascade` branches — **overload frames carry no
keep-probability and no quota**, so the overload timeline is exact across the whole file. Kept
`normal` frames are a uniform 2% random sample, so conditional statistics estimated on them are
unbiased.

The rejected task designs remain in the codebase (`--task forecast`, `--risk-source`, and per-class
`steps_to_*` distances on every record) specifically so §2 can be reproduced by a reader.

---

### 6.1 A defect that affects what may be claimed about the retired work

`compute_normalization_stats()` indexes every training graph, which populates PyG's `_data_list`
cache; the in-place normalization that follows mutates `_data` and therefore never reaches the
DataLoaders. **Every GNN run in this project, on every task, trained on raw unnormalized features**
while `normalization_stats.pt` was saved as though it had been applied. Measured cost on N-1: val F1
0.44 versus 0.90. Fixed for N-1 (cache invalidation plus an assertion); full account in
[`gnn_n1_tightening.md`](gnn_n1_tightening.md).

Consequences for the retired classification work, which the write-up must respect:

- `eval_cross_topology.py` **does** normalize at inference, so the frozen checkpoint was fitted on
  raw magnitudes and evaluated on z-scored ones. Cross-topology degradation was a headline finding;
  an unknown share of it is this mismatch rather than topology transfer. **Do not present those
  numbers as evidence about generalisation.**
- The in-distribution figure is at least self-consistent (train and eval both unnormalized).
- **The two checkpoints are not the same file.** `CLAUDE.md` asserted
  `gnn_checkpoint_best.pt == gnn_checkpoint_leverA.pt`; they are byte-different, and all 35 tensors
  differ (max ~5e-3 per tensor) — two separate training runs of the same architecture.
  `gnn_logit_margin.json` records `test_macro_f1: 0.8277` but does not name the checkpoint it was
  calibrated against, and `eval_cross_topology.py` defaults to `best` while the *older* file is the
  one named for Lever A. **Which run produced 0.8277 is not determinable from the artifacts.** Cite
  it as "the reported figure" and lean on the closed-form argument, which does not depend on it.

## 7. Wording that must be removed from the current draft

- Any statement that the system "classifies faults into four categories" as the headline
  contribution — that task is retained only as a documented negative result.
- Any claim that the shield "validates" or "verifies" GNN predictions without qualification. Replace
  with the §5.2 formulation.
- Any figure or table reporting macro F1 0.8277 as the primary result, and the whole
  classification **results chapter** — Lever A, seed sensitivity, the rejected architecture rounds,
  the pooling-bottleneck diagnosis, cross-topology classification. Compress to the one-page
  methodology note described at the head of §2.
- Any claim that `gnn_checkpoint_best.pt` and `gnn_checkpoint_leverA.pt` are the same artifact
  (§6.1), and any attribution of 0.8277 to a specific checkpoint.
- The `voltage_pu = v_or / 150.0` conversion, wherever it appears. Superseded by per-line base kV
  with energized-line masking; a flat divisor produces ~100% false blocks on case14.
