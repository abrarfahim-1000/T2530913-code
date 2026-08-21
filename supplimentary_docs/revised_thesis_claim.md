# Revised Thesis Claim — what to write, and what not to

**Status:** current as of 2026-08-21. All four components are built and measured. This document
supersedes the fault-classification framing throughout the draft.
**Audience:** whoever is editing the thesis text.

**Background reading:** [`thesis_walkthrough.md`](thesis_walkthrough.md) explains what each pipeline stage
does and why, in plain English. This document assumes you already know that.

Everything below is measured, not projected. This document states **the claim and its limits**;
the derivations behind every number live in [`thesis_findings.md`](thesis_findings.md) §9–§18,
whose section numbers are cited inline. The narrative version, readable without the codebase
open, is that same document's §1–§8.

> **History.** The core of this file was written 2026-08-16, before Components C and D were
> finished. The N-1 framing was correct then and is unchanged now; §5 and §8 are new, and §1's
> claim paragraph has been extended to cover the gate's measured behaviour and the provenance
> graph. The build plan that carried the operational record was retired on 2026-08-21 and folded
> into `thesis_findings.md`; it survives at `archive/component_d_plan.md`.

---

## 1. The claim, in one paragraph

> We train a graph attention network to perform **N-1 contingency screening** on a power grid: for
> every energized line, the model predicts whether tripping that line would drive the network
> outside its thermal limits. The task is chosen because its label cannot be reproduced by any rule
> over the observed state — producing it requires a post-contingency power-flow solve — which makes
> the comparison between the neural model and the symbolic layer non-degenerate by construction.
> Model predictions are then gated by a **symbolic shield** built from rules extracted from IEEE,
> NERC, FERC and national grid codes by an LLM pipeline. The shield does not re-derive the
> contingency analysis; it enforces the operating principle that **N-1 security presupposes a
> secure base case**, blocking any prediction that declares the grid safe against a further outage
> while the present state already violates a standard. The gate is deliberately asymmetric:
> over-permissive predictions are blocked, over-cautious ones are always allowed through. Measured
> across three topologies at a threshold selected once and held fixed, **the gate's accuracy when
> it intervenes is flat at 92–94% regardless of topology, while the model's own accuracy collapses
> off-distribution** — and every block it issues can name the clause, document and standards body
> that authorise it.

---

## 2. What changed from the previous draft, and why

> **Editorial decision (2026-08-16): the fault-classification work is DEMOTED from a results chapter
> to roughly one page of methodology.** It is not deleted — the closed-form finding below is what
> *justifies* the N-1 task design, and without it §3 reads as an arbitrary choice that any examiner
> will challenge with "why not fault classification, like the rest of this literature?"
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
>    (see §6.1), while the cross-topology script normalized at inference — a train/inference
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
`rho_max` and `n_tripped_lines` reproduce the stored labels with **zero disagreements across all
315,000 records** of both topologies — every record, not a sample. A symbolic layer therefore
scores 100% where the trained GATv2 reached 0.8277 macro F1. The question "does the shield add
value over the GNN" was answered before the experiment ran. *(Derivation: findings §9.1.)*

Two replacement targets were then built and **rejected by measurement before any full training run**:

| Candidate target | Why it was rejected |
|---|---|
| **Any-fault forecast** — will *any* fault occur within H steps? | Line-trip onset is injected by an unconditional Bernoulli draw on a uniformly random line, so it is unpredictable **by construction**, and trips are 71–88% of all positives. The best possible threshold on current `rho_max` scores **1.00× the all-positive baseline at every H ≥ 3**. No model can beat "always say at-risk". |
| **Overload-only forecast** — will `rho ≥ 1.0` within H steps? | Learnable, but exhausted by a single rule. Gradient boosting on the full feature vector reached **F1 0.163 against the single threshold's 0.160**, with *worse* average precision (0.125 vs 0.127). Adding trend features over 1/3/6 frames on contiguous pilots changed nothing on either topology (lift 0.67–1.02× across H ∈ {2…18}). A single snapshot carries no information about future exogenous load. |

**N-1 screening fails neither test**, which is why it was adopted. See §3.
*(Derivation for both rows: findings §9.2.)*

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
2. **The answer is topological.** **96–99% of frames are strictly mixed** — within a single frame,
   some contingencies violate and others do not. A graph-level prediction cannot express this; the
   quantity being predicted lives on edges.
   ⚠ **Quote 96–99%, not 100%.** An earlier draft said "100% of sampled frames are mixed"; measured
   over all 22,000 frames the figure is 98.80 / 98.48 / 96.43%. What *is* 100% is that **no frame
   on any grid is entirely secure**. The argument is unaffected — a per-frame label is still wrong
   for most lines in 96%+ of frames — but the number must be stated correctly. *(findings §9.3.)*
3. **There is a real gap for the model to occupy.** A strong local rule reaches 0.608 and a
   network-aware model reaches 0.868. Neither end is degenerate: the symbolic baseline is
   respectable rather than trivial, and the model's advantage is substantial rather than noise.

Base rate is ~16–26% violation depending on topology and stress, so the task is not a rare-event
problem. Labels cost ~170 power-flow solves per second.

---

## 4. What the GNN does now

**Input:** unchanged — one graph per frame of the *pre-contingency* state. Nodes are buses,
edges are energized lines, tripped lines pruned from `edge_index`. The GATv2 backbone,
live-batch-statistics normalization, and zero-dropout choices all carry over with their original
justifications intact.

**Output — this is the architectural change.** The global mean/max/min pooling and the graph-level
classifier head are replaced by an **edge-level head**: one logit per line, read off the or→ex edge
as `MLP([h_or ‖ h_ex ‖ edge_attr ‖ x_or ‖ x_ex])`. Nothing is pooled, because the answer to "is
losing line *k* safe" is local to *k* and its neighbourhood, and pooling would erase precisely the
per-line distinction the task is about.

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

### 4.2 Cross-topology — report the failure

**Held threshold — 0.8849, selected once on the neurips2020 validation split and applied unchanged
to all three grids. This is the protocol to write up.**

| topology | lines | all-positive | best rule | **model (held)** | recall | precision |
|---|---:|---:|---:|---:|---:|---:|
| neurips2020 *(in-dist)* | 59 | 0.3104 | 0.4639 | **0.8956** — 1.93× rule | 0.902 | 0.889 |
| case14 *(unseen, smaller)* | 20 | 0.4345 | 0.5392 | **0.4167** — **0.77× rule, FAILS** | 0.435 | 0.400 |
| wcci2022 *(unseen, larger)* | 186 | 0.3969 | 0.4915 | **0.5577** — 1.13× rule | 0.629 | 0.501 |

**On case14 the model fails outright, and harder than earlier drafts said.** At 0.4167 it scores
**0.77× the single-rule baseline** and **0.96× the all-positive baseline** — on that topology it is
beaten not only by one threshold on one feature, but by answering "violation" unconditionally.
Report it plainly; it is a stronger and more honest statement than the 0.83× figure it replaces.

Scaling **up** costs far less than scaling **down**, which is the opposite of the naive expectation
and is the interesting result. That conclusion is unaffected by the protocol change.

✅ **Protocol closed 2026-08-21.** This table was previously best-threshold on each grid
individually — the cutoff chosen using the answer key, which is not obtainable in deployment. The
oracle figures are retained **only** as a labelled ceiling, because §5 measures against them:

| topology | oracle (best-threshold) | AP | optimism vs held |
|---|---:|---:|---:|
| neurips2020 | 0.8972 | 0.9615 | +0.0015 |
| case14 | 0.4477 | 0.4270 | **+0.0310** |
| wcci2022 | 0.5721 | 0.6419 | +0.0144 |

The optimism is negligible at home, where the threshold was selected, and twenty times larger on
case14 — a foreign grid shifts the score distribution, so a fixed cutoff sits further from that
grid's optimum. **Never put the two protocols in one column.** *(findings §14.)*

🚨 **Every absolute model F1 in this document is conditional on eval batch size 64.** Measured
2026-08-21: the tracked checkpoint scores 0.8972 at batch 64 and **0.9255 at batch 512** on the
identical split, because `BatchNorm(track_running_stats=False)` uses live batch statistics at
inference. **Never quote a model F1 without its batch size**, and do not present the raw-model
column as a property of the model alone.

**The claim in §5 is unaffected**, and this is precisely why it must be framed as a claim about
*deltas and precision* rather than levels: across the same batch-size change the shield's delta
moves +0.0082 to +0.0072 and its intervention precision 0.938 to 0.930, because both arms share one
forward pass. Write the thesis around those two quantities. Derivation:
`thesis_findings.md` §14 caveat 3 and `gnn_n1_tightening.md` §8.

---

## 5. What the shield does — measured, and the limit on the claim

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
report what a gating shield *would* have done without paying its false-block rate. ⚠ In practice
the Option B counterfactual is **unmeasurable on this corpus** — affirmation coverage is `normal`
only, across three independent extraction runs. State that as a settled property, not an omission.

### 5.2 The measured result

Threshold **0.8849, selected on the neurips2020 validation split and held fixed across all three
topologies.** Rules: the validated four-rule corpus.

| topology | model F1 | **+shield F1** | Δ | missed violations | **intervention precision** |
|---|---:|---:|---:|---|---:|
| neurips2020 *(in-dist)* | 0.8956 | **0.9038** | +0.0082 | 2,041 → 1,710 | **0.938** |
| case14 *(unseen)* | 0.4167 | **0.4188** | +0.0021 | 18,592 → 18,497 | **0.922** |
| wcci2022 *(unseen)* | 0.5577 | **0.6253** | **+0.0676** | 68,166 → 47,101 (**−30.9%**) | **0.934** |

Two claims the paper can make from this, both measured:

1. **On the large unseen grid, a post-hoc gate pushed the result past the model's own oracle
   ceiling.** The shielded 0.6253 exceeds the 0.5721 the raw model reaches on that grid *when
   allowed to tune its threshold against the answers* (§4.2). Nothing inside the model changed.
   Note this comparison is **deliberately** against the oracle figure — it is the stronger form of
   the claim, and the honest-protocol raw score there is lower still (0.5577), so the margin is
   +0.0532 against the ceiling and +0.0676 against the like-for-like arm.
2. **The gate's precision is topology-invariant while the model's collapses.** 0.938 / 0.922 /
   0.934 against the model's 0.90 → 0.42 → 0.56. What changes off-distribution is not how *right*
   the gate is, but how *often it gets to speak* (0.31% → 0.087% → 3.04% of contingencies) and how
   much of the model's error sits where the doctrine can reach it (16.2% / 0.51% / 30.9%).

**On the threshold's selection objective — disclose it, do not quietly switch it.** The cutoff is
chosen by maximising **F1**, which prices one missed violation equal to one false alarm. §5.1's gate
is deliberately asymmetric on the opposite premise, so the two are in tension and an examiner will
find it. The resolution is measured rather than argued: the full F-beta trade curve is reported in
`thesis_findings.md` §14.2. Two results from it belong in the text.

- **The trade is real and priced.** Halving the training grid's missed violations costs 3.2 extra
  false alarms each and 0.04 F1; on wcci2022 the rate is 4.2. Choosing a point on that curve needs
  an operator cost ratio nobody has measured for these grids, which is why **F1 is retained and the
  alternative disclosed** rather than silently swapped.
- **The claim below survives the change.** Re-scored at an F2-selected threshold, intervention
  precision is **0.937 / 0.933 / 0.932** against F1's 0.938 / 0.922 / 0.934. The gate's accuracy is
  invariant to topology *and* to the threshold objective. The delta shrinks (+0.0052 / +0.0011 /
  +0.0508) for a mechanical reason worth stating: a lower cutoff issues fewer `secure` verdicts, so
  the gate has less to veto. **Reach changes; precision does not.**

🚨 **Do not write "the shield's precision rises off-distribution."** That phrasing appears in
drafts built from the pre-2026-08-20 numbers and was **measured to be false** — it was an artifact
of averaging misextracted voltage rules (0.13–0.20 precision) with thermal rules (0.92–0.94). The
corrected finding is flat precision, and it is the stronger claim. *(findings §13.2.)*

**A result about pipelines, worth its own paragraph:** dropping 28 of 32 rules at the validation
stage *improved or held every metric on every grid*. A rulebook that is mostly bad rules does not
perform at a discount to the good version — it performs at the average, and the average hides the
good rules completely. *(findings §13.1.)*

### 5.3 The limits — state all four explicitly in the paper

**(a) The shield does not verify the contingency analysis.** Establishing whether losing line *k*
truly violates a limit requires a post-contingency power flow, which is exactly the computation the
GNN exists to replace. If the shield could run it, the GNN would be unnecessary. The defensible
sentence is:

> The shield catches unsafe-permissive predictions using present-state standards.

and **not**:

> The shield independently verifies the contingency analysis.

**(b) The surviving rule is close to a physical tautology, and the distance was measured.**
P(violation | base overloaded) is **91–96%, not 100%**, so the predicate carries real but partial
information. If it were a label restatement the conditional would be 100%. The honest reading:
the shield enforces N-1 doctrine and the data confirms that doctrine 91–96% of the time. It is not
discovering new physics. *(findings §13.4.)*

**(c) The doctrine is hand-written; the LLM supplied only the threshold.** The asymmetric gating
logic in §5.1 lives in `validate_n1`, not in the extracted corpus — and it belongs to the
operational-standards class that the corpus provably lacks. **Volunteer this** rather than
defending it under question.

**(d) The end-to-end yield is 0.16%** — 2,463 candidates become 4 rules. That is a finding about
the standards/simulator mismatch, not a success metric, and must be reported as one. At least 3 of
the 21 validation rejections are wrong and 4 more should have been CORRECT verdicts; the validated
corpus is better than the guarded one, not clean. *(findings §12.4, §12.5, §15.)*

This is a narrower claim than the previous draft implied, and it is the claim the extracted rule
corpus can actually support. It remains a real result: a genuine safety property, enforced by rules
traceable to named standards, and the property an operator would actually want.

### 5.4 The convergence argument — the strongest methodological result

Two entirely independent filters selected the same four rules:

- **Empirical** — no LLM. Ran every rule against real grid data: *does it fire on healthy grids,
  and does it discriminate?* Answer, across two independent extraction runs: `loading_pct > 100` is
  the only useful family, 10/10 both times.
- **Textual** — no grid data. Read the source standards: *is this a faithful reading of the
  document?* Answer: keep `loading_pct > 100`, reject the rest, because the voltage rules came from
  ride-through tables and time-bound operating envelopes misread as instantaneous limits.

Each result alone invites an easy objection — *"your fire-rate cutoff is mistuned"* or *"your prompt
is wrong"* — and the strict-arm A/B shows the second objection can genuinely be correct.
**Convergence defeats both.** *(findings §10.3, §12.2.)*

---

## 6. Reproducibility

| Finding | How it was measured |
|---|---|
| Classify label is closed-form | 4 threshold rules vs stored labels, **all 315,000 records**, both topologies |
| Any-fault forecast is unlearnable | 1,995 quota-unbiased chronics / 321,966 reconstructed steps; best threshold on `rho_max` vs all-positive F1, calm frames only |
| Overload forecast is exhausted by one rule | HistGradientBoosting on 30 summary features, chronic-level split, 56,441 calm frames / 2,227 positives |
| Trend features do not help | 60,000-frame contiguous pilots on both topologies, deltas at lags 1/3/6, H ∈ {2…18} |
| N-1 probe numbers in §3 | 17,652 contingency labels from 300 frames, frame-level split, `obs.simulate()` per line |
| Shield results in §5.2 | `evaluation/eval_shield_n1.py`, all three tags, `--rules validated_translated/all_rules_deduped.jsonl` |
| Validation A/B | same 32-rule input, same model, temperature 0.0, `--prompt-variant {strict,translated}` into separate output directories |

Reconstruction of the unsubsampled fault timeline is valid because `generate_dataset.py` drops
frames only in the `normal` / `line_trip` / `cascade` branches — **overload frames carry no
keep-probability and no quota** — and kept `normal` frames are a uniform 2% random sample, so
conditional statistics estimated on them are unbiased.

The rejected task designs remain in the codebase (`--task forecast`, `--risk-source`, and per-class
`steps_to_*` distances on every record) specifically so §2 can be reproduced by a reader.

⚠ **Two reproducibility limits to state.** The classify artifacts were deleted on 2026-08-20 and
were mostly untracked, so §2's 0.8277 is **not** reproducible — cite it as "the reported figure"
and lean on the closed-form argument, which does not depend on it. And the eval harness is not
bit-reproducible on this hardware: `threshold` and `ap` drift at ~1e-6 / ~1e-8 between any two runs
of the identical path. **No count has ever moved.** *(findings §16.4.)*

### 6.1 A defect that affects what may be claimed about the retired work

`compute_normalization_stats()` indexes every training graph, which populates PyG's `_data_list`
cache; the in-place normalization that follows mutates `_data` and therefore never reaches the
DataLoaders. **Every GNN run in this project, on every task, trained on raw unnormalized features**
while `normalization_stats.pt` was saved as though it had been applied. Measured cost on N-1: val F1
0.44 versus 0.90. Fixed for N-1 (cache invalidation plus an assertion); full account in
[`gnn_n1_tightening.md`](gnn_n1_tightening.md) §3.

Consequences for the retired classification work, which the write-up must respect:

- The classify cross-topology script **did** normalize at inference, so the frozen checkpoint was
  fitted on raw magnitudes and evaluated on z-scored ones. Cross-topology degradation was a headline
  finding; an unknown share of it is this mismatch rather than topology transfer. **Do not present
  those numbers as evidence about generalisation.**
- The in-distribution figure is at least self-consistent (train and eval both unnormalized).
- **The two checkpoints were not the same file.** `CLAUDE.md` asserted
  `gnn_checkpoint_best.pt == gnn_checkpoint_leverA.pt`; they were byte-different, and all 35 tensors
  differed (max ~5e-3) — two separate training runs of the same architecture. **Which run produced
  0.8277 is not determinable from the artifacts**, and both were deleted on 2026-08-20.
- **The N-1 arm is not affected.** That checkpoint was trained after the fix, so train and inference
  agree. The caveat applies only to the retired classify work.

---

## 7. Wording that must be removed from the current draft

- Any statement that the system "classifies faults into four categories" as the headline
  contribution — that task is retained only as a documented negative result.
- Any claim that the shield "validates" or "verifies" GNN predictions without qualification. Replace
  with the §5.3(a) formulation.
- 🚨 **Any claim that the shield's override precision *rises* off-distribution** (§5.2). Measured
  false. Replace with topology-invariant precision.
- Any figure or table reporting macro F1 0.8277 as the primary result, and the whole
  classification **results chapter** — Lever A, seed sensitivity, the rejected architecture rounds,
  the pooling-bottleneck diagnosis, cross-topology classification. Compress to the one-page
  methodology note described at the head of §2.
- **"100% of frames are mixed"** — the measured figure is 96–99% (§3, item 2).
- **"100% agreement on 55,000 records"** for the closed-form finding — superseded by zero
  disagreements across all 315,000.
- Any claim that `gnn_checkpoint_best.pt` and `gnn_checkpoint_leverA.pt` are the same artifact
  (§6.1), and any attribution of 0.8277 to a specific checkpoint.
- The `voltage_pu = v_or / 150.0` conversion, wherever it appears. Superseded by per-line base kV
  with energized-line masking; a flat divisor produces ~100% false blocks on case14. It survives
  legitimately in exactly one place — the GNN's `mean_v` node feature, where it is an input scale
  factor, not a per-unit conversion.
- Any framing of the knowledge graph as a topology graph (buses, lines, rules hung off components).
  That was v1, it was deleted, and §8 describes what replaced it.

---

## 8. What the knowledge graph contributes to the claim

The graph is **provenance, not topology**:
`Document → Clause → Rule → ServedRule → Predicate → Variable`, 36 nodes and 40 edges, and it is
topology-agnostic by construction so one graph serves all three grids.

It answers exactly one question — *when the gate blocks, on whose authority?* — and it makes one
measured claim visible that a flat rule list cannot express:

> **The thermal check the shield enforces is stated in 10 clauses across 4 documents, by two
> identified standards bodies on two continents.**

Two things not to overclaim, both of which an examiner will probe:

- **Corroboration is not independent evidence about physics.** Four documents restating a thermal
  rating limit is four documents agreeing on standard practice — which §5.3(b) already
  characterises as close to a tautology. It strengthens the **provenance** claim, not the novelty
  claim.
- **Two identified bodies, not three.** One source document does not name its issuing body, and the
  lookup table records `None` rather than guessing. The body count is a reported figure.

**Retrieval through the graph is opt-in and returns the identical rule set** — verified field by
field on all three topologies and pinned by a test. So the graph is a citation layer, not a result;
it must not be presented as having changed any number, because it did not. *(findings §16.)*

---

## 9. What was designed and never run — say so

Three interpretability controls were specified and **not implemented**: a ceiling analysis (a
decision tree on the 14 context variables targeting *"the model was wrong"*, which upper-bounds
what any symbolic gate could catch), an expert-written rule baseline as a never-merged third arm,
and a rule-count ablation.

The rule-count ablation was answered two other ways — structurally, by measuring what share of the
model's errors any present-state rule could reach at all (16.2% / 0.51% / 30.9%), and empirically,
by the finding that a corpus can lose 29% of its rules and change nothing downstream. **The ceiling
analysis and the expert baseline were never run.**

That matters for exactly one reason, and the write-up should own it: these controls exist to stop
**(b)** *"we couldn't build much of a symbolic layer, so we can't tell whether it would help"*
being reported as **(a)** *"we built a sound symbolic layer and it didn't help."* Only (a) answers
the thesis question. Here the gate demonstrably *does* help, so the risk is inverted and lower —
but the expert baseline is what would separate "extraction is the bottleneck" from "gating is the
bottleneck", and without it that question stays open. State it as an open question rather than
letting an examiner find it. *(findings §17.1.)*
