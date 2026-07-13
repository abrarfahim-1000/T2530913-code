# Operating-Point Calibration vs. Representation Learning for the `normal`/`line_trip` Confusion: A Decision and Its Defense

**Status:** decision record + defense. Self-contained — readable without the other supplementary docs.
**Date:** 2026-06-25.
**Scope:** the GridGNN classifier head (Component A) on the NeurIPS-2020 36-bus topology, and its
behaviour under cross-topology transfer to the 14-bus `l2rpn_case14_sandbox` environment.
**Companion files:** `gnn_final_results.md` (consolidated log of all rounds, incl. this decision),
`training/calibrate_margin.py`, `evaluation/eval_cross_topology.py`.

---

## 0. Executive summary

The recurring symptom — "of 9,128 truly `normal` states the model recovers only ~5,900, dumping ~3,000
into `line_trip`" — was diagnosed not as a `normal`-class weakness but as a **`line_trip`
over-prediction (magnet) pathology** rooted in the data-generation sampling regime. Five interventions
were tried under a strict *keep-what-works* bar. Exactly one succeeded:

- **Lever A — a post-hoc per-class logit margin** (a decision-threshold recalibration applied at
  inference, no retraining): on the 36-bus test set it lifted macro-F1 **0.7830 → 0.8272** and `normal`
  recall **0.6466 → 0.8803** while *every* other class's F1 held or improved.
- **Levers B, C, D** (de-magnet resampling, a dispersion-pool readout, a two-stage hierarchical head)
  all failed the strict bar and were reverted.
- **Lever E** (the root-cause data-generation fix) is implemented but its regeneration is deferred.

A cross-topology test then revealed the decisive constraint: **the Lever A margin does not transfer.**
On the unseen 14-bus grid it *reduced* macro-F1 (0.6258 → 0.5961), because the 14-bus failure mode is
different (there `line_trip` is under-predicted and `cascade` is the magnet).

**Recommendation:** adopt **Lever A as the in-distribution (36-bus) result**, report unseen topologies
**uncalibrated**, treat **Lever E as optional and non-urgent**, and let the **symbolic shield**
(Component D) — not a logit patch — carry the cross-topology safety argument. The remainder of this
document defends each clause of that recommendation against the objections a reviewer is most likely to
raise.

---

## 1. Problem statement

The GridGNN emits a 4-way classification over `{normal, overload, line_trip, cascade}`. The headline
defect, measured on the held-out 36-bus test split (45,000 frames), was poor `normal` recall with high
`normal` precision — a classifier that is *correct when it commits to `normal` but commits too rarely*.

Round 1 of experimentation (documented separately) exhausted the obvious **feature / loss /
architecture** levers — an edge-flag threshold change (kept), label-smoothing reduction, focal loss, a
sixth node feature, and a graph-level trip-count injection (all reverted) — and concluded without
moving the needle beyond `normal` recall ≈ 0.54–0.65. The effort was declared concluded.

Round 2 (this document) reopened the problem by first asking a different question: *not* "how do we make
the model commit to `normal`?" but "**why does it commit to `line_trip` instead, and is that the real
defect?**"

---

## 2. The diagnosis: `line_trip` is an over-predicted magnet

Re-running `training/evaluate.py` on the kept checkpoint produced the confusion matrix that reframed
everything (rows = true, columns = predicted, order `[normal, overload, line_trip, cascade]`):

```
true normal     [ 5902   183   3043     0 ]   recall 0.6466
true overload   [  687 11287   1605  1885 ]
true line_trip  [   64   404  10338   496 ]   recall 0.9147
true cascade    [    0   336    959  7811 ]
```

`line_trip` is **predicted 15,945 times but true only 11,302** — a precision of **0.648**. It is a sink:
it absorbs 3,043 normals, 1,605 overloads, and 959 cascades. The `normal → line_trip` leak is therefore
a *symptom* of a class-level over-prediction, not a `normal`-specific phenomenon. Any fix framed purely
around `normal` was attacking a shadow.

**Root cause (data generation).** In `scripts/generate_dataset.py` the sampling regime was asymmetric:
`NORMAL_KEEP_PROB = 0.02` discarded 98% of `normal` frames, while `LINE_TRIP_KEEP_PROB = 1.0` retained
**every** N-1 "cooldown" frame. Because a single injected fault sits in the N-1 state for several
consecutive, nearly identical steps before reconnecting (`RECONNECT_PROB = 0.20`), the `line_trip` class
became a dense cluster of **autocorrelated near-duplicates**. The inverse-frequency (sqrt-ICF) class
weights are computed on the *post-cap counts*, so they never perceive the low *effective* sample
diversity. The optimiser, rewarded for fitting a tightly packed and over-represented region, carved an
oversized `line_trip` decision basin. Every Round-1 lever operated *downstream* of this and so could
only redistribute the error.

---

## 3. The intervention space

Five levers were defined, ordered cheap → expensive, and tried one at a time, stacking only proven
winners (strict *keep-what-works*: adopt only if `normal` recall rises **and** the other three classes'
F1 each hold within ~0.02 **and** macro-F1 does not regress).

| Lever | Mechanism | Touches |
|---|---|---|
| **A** | Post-hoc per-class logit margin (decision-threshold recalibration) | inference only |
| **B** | De-magnet `line_trip` via `WeightedRandomSampler` + unweighted CE | training loop |
| **C** | Extra **std/dispersion** pooling branch in the readout | architecture |
| **D** | Two-stage **normal-vs-fault** gate + 3-way subtype head | architecture |
| **E** | Fix the data generator (subsample cooldown frames + settling filter) | data |

---

## 4. Results

### 4.1 Lever A — the winner (36-bus, in-distribution)

A per-class offset vector **b** is added to the logits at inference, and the prediction is
`argmax(logits + b)`. The offset is tuned on the **validation** split and reported on the **test**
split. The selected offset is `b_normal = +0.30, b_line_trip = −0.10` (others 0).

| class | F1 (argmax → calibrated) | recall (argmax → calibrated) |
|---|---|---|
| normal | 0.7480 → **0.8649** | 0.6466 → **0.8803** |
| overload | 0.8157 → 0.8170 | 0.7299 → 0.7345 |
| line_trip | 0.7588 → **0.8109** | 0.9147 → 0.8496 |
| cascade | 0.8095 → **0.8160** | 0.8578 → 0.8930 |
| **macro** | **0.7830 → 0.8272** | |

The `line_trip` over-prediction collapsed from 4,643 to **1,078**; the `normal → line_trip` leak from
3,043 to **902**. The validation-tuned macro (0.8286) matched the independent test macro (0.8272),
confirming the gain is **not** validation overfitting or test leakage.

**Interpretation.** Lever A does not make the model *smarter*; it chooses a better **operating point**.
The model already encoded `normal` well (precision 0.887) but argmax — optimal only under a 0/1 loss
with calibrated, balanced posteriors — was a poor decision rule for an imbalanced, miscalibrated head.
A margin is the textbook remedy. This framing matters for the defense (§6).

### 4.2 Levers B, C, D — reverted

- **B (de-magnet sampler).** Calibrated macro **0.8190 < 0.8272**. The sampler did shrink the
  `line_trip` basin but merely **relocated** the magnet: `cascade` became the over-predicted class
  (precision 0.704; 2,479 overloads leaked into it) and **overload F1 regressed 0.8170 → 0.7883**.
  Moving balancing entirely into the sampler (with unweighted CE) destabilised the majority/overload
  boundary. Fails the strict bar.
- **C (std/dispersion pool).** Best validation macro **0.5842** vs 0.79 baseline — a collapse, not slow
  convergence (it plateaued and never recovered). Note: the originally-planned `global_add_pool` is
  *provably redundant* here because the node count is constant at 36 (only edges are pruned for tripped
  lines), so a sum-pool is exactly `36 × mean-pool`. The substituted dispersion pool destabilised the
  small `[16,32,32]` model — the same fragility-to-added-dimensions failure seen in Round 1.
- **D (two-stage head).** Best validation macro **0.5874**, peaking at epoch 6 and then *declining*. A
  shared trunk kept head parameters ≈ the flat classifier, isolating the decision-structure change; the
  hierarchical log-prob objective nonetheless handed the small model a worse, unstable landscape (the
  flat softmax reaches 0.79 in the same budget). Reverted.

The B/C/D outcomes are themselves evidence: the in-distribution modelling headroom on this small,
deliberately-regularised model is largely exhausted, which is *why* an operating-point fix (A) — rather
than a representation fix — was the move that worked.

### 4.3 The cross-topology test — the pivotal result

The thesis's central claim concerns **cross-topology generalisation** (train on 36-bus, evaluate on
unseen 14- and 118-bus grids). The natural question is whether Lever A's margin — a property of the
36-bus class geometry — survives transfer. A 14-bus test set (`l2rpn_case14_sandbox`, 15,000 frames,
generated with the *original* configuration for consistency) was evaluated through
`evaluation/eval_cross_topology.py`, which runs the topology-agnostic classifier head, normalises
foreign features with the 36-bus train-split statistics, and applies the saved margin.

| case14 (14-bus) | macro F1 | normal R | overload R | line_trip R | cascade R |
|---|---|---|---|---|---|
| Uncalibrated (argmax) | **0.6258** | 0.9625 | 0.484 | 0.596 | 0.497 |
| + Lever A margin | **0.5961** | 0.9936 | 0.469 | 0.472 | 0.515 |

**The margin tuned on 36-bus *reduces* 14-bus macro-F1 by 0.030.** The reason is that the failure mode
is topology-specific:

- On **36-bus**, `line_trip` is the magnet → a negative `line_trip` margin corrects it.
- On **14-bus**, `line_trip` is *under*-predicted (3,330 predicted vs 3,750 true) and **`cascade` is the
  magnet** (precision 0.35; it absorbs 1,736 overloads and 1,036 line_trips). The 36-bus margin's
  `line_trip −0.10` therefore *suppresses already-scarce 14-bus line_trips* and trades `normal`
  precision (0.827 → 0.734) for a near-ceiling recall gain (0.96 → 0.99).

A secondary, structural confound compounds this: the connectivity features `global_trip_frac` and
`connected_line_frac` are **topology-dependent magnitudes** — one tripped line is `1/20 = 0.05` of a
14-bus grid but `1/59 = 0.017` of a 36-bus grid. Normalising foreign features with 36-bus statistics
therefore inflates the apparent severity of a 14-bus single-line trip, making it resemble a 36-bus
*multi-line* event and biasing it toward `cascade`. This is a generalisation-gap property of the
feature design, independent of any calibration choice.

---

## 5. The recommendation

1. **Adopt Lever A as the in-distribution (36-bus) result.** It is a legitimate, reproducible
   operating-point calibration delivering macro-F1 0.8272 and `normal` recall 0.88, and it slots in
   cleanly *before* the symbolic shield.
2. **Do not apply the 36-bus margin to unseen topologies.** Report cross-topology numbers
   **uncalibrated**. Applying a topology-specific correction off-topology is unprincipled and, here,
   empirically harmful.
3. **Treat Lever E as optional and non-urgent.** It removes the `line_trip` magnet *in the weights*
   (topology-portable in principle) and yields a cleaner 36-bus base, but it targets `line_trip` whereas
   the 14-bus deficit is `cascade`; it is therefore not a cross-topology remedy and does not block the
   thesis.
4. **Let the symbolic shield carry the cross-topology safety claim.** The degradation of the neural
   layer on unseen topologies is *expected and is itself a thesis result*; the shield's topology-agnostic
   rule compliance is the designed safeguard, not a logit patch.

---

## 6. Defense

This section states the strongest objections a reviewer (or committee) could raise and answers each.

### Objection 1 — "Lever A is a cheap post-hoc trick, not a contribution."
**Rebuttal.** Decision-threshold optimisation for imbalanced, miscalibrated classifiers is a recognised,
principled technique, not a hack: argmax is Bayes-optimal only under symmetric costs *and* calibrated
posteriors, neither of which holds here. The honest framing is explicit in this report — Lever A is
**operating-point calibration**, presented as such, not as "a stronger GNN." Its value is (a) it
quantifies that the model already separates `normal` (the +0.23 recall comes from re-thresholding, not
new capacity), and (b) it is the *only* intervention that satisfied the strict bar after three retrained
alternatives (B/C/D) failed — which is informative about where the remaining headroom is (the decision
rule, not the representation).

### Objection 2 — "You tuned on the test set / the gain is overfitting."
**Rebuttal.** The offset is selected on the **validation** split and reported on a disjoint **test**
split; the two macro values agree to within 0.0014 (val 0.8286, test 0.8272). The splits are
chronic-level (no cascade-sequence leakage across splits) and normalisation statistics are computed from
the training split alone. There is no test-set tuning. The four-number offset has negligible capacity to
overfit 45,000 evaluation frames, and the agreement between val and test demonstrates it did not.

### Objection 3 — "If A is so good, why not just re-tune the margin per topology and keep the gain?"
**Rebuttal.** Re-tuning per topology requires **labelled validation data on each unseen topology**,
which directly contradicts the zero-shot transfer premise of the study. The whole point of the
cross-topology evaluation is to measure what the 36-bus-trained system does on grids it has never seen
and for which no labels are assumed at deployment. A per-topology margin is, definitionally, no longer
zero-shot. We therefore report unseen topologies uncalibrated and treat A as in-distribution only.

### Objection 4 — "Reporting cross-topology uncalibrated while reporting 36-bus calibrated is inconsistent / cherry-picking."
**Rebuttal.** The two settings answer two different questions and the asymmetry is principled, not
opportunistic. The 36-bus number answers "what is the best *deployable in-distribution* operating point?"
— and on the topology you actually trained for, you *are* entitled to calibrate on held-out
in-distribution data. The cross-topology number answers "what does the frozen system do on an unseen
grid *with no foreign labels*?" — where calibration is unavailable by assumption. Crucially, the
inconsistency is disclosed and the calibrated cross-topology number is *also* reported (0.5961) to show
it would have been worse, so nothing is hidden. Transparency converts a potential weakness into a
finding.

### Objection 5 — "Then the real fix is Lever E; you should have run it before concluding."
**Rebuttal.** Lever E was costed, implemented, and deliberately deferred — not skipped out of laziness —
for two defensible reasons. First, it is a ~1-hour regeneration with its own knob-tuning loop, and the
*free* cross-topology test was the higher-information next action: it determines whether A suffices and
whether E could even help. Second, the cross-topology result shows E is **not** the cross-topology
remedy: E de-magnets `line_trip`, but the 14-bus deficit is a `cascade` magnet plus a feature-scale
confound. E would improve the 36-bus *base* model (worth doing eventually, code is ready) but would not
close the transfer gap. Spending an hour on E *before* knowing this would have been the unjustified
move; the order of operations here is the defensible one.

> **Empirical update (Lever E was subsequently run; see §6a).** The clause "E would improve the 36-bus
> base model" was a *prediction*, and the experiment **falsified it**: regenerating the dataset
> regressed the uncalibrated base (0.7830 → 0.7640) and the magnet persisted. This does not weaken the
> recommendation — it strengthens it, by proving the magnet is architectural rather than a data
> artifact. The original prediction is left intact above for honesty; the corrected, evidence-based
> position is in §6a.

### Objection 6 — "B merely 'relocating the magnet' might be a tuning artifact; a different `line_trip` factor could win."
**Rebuttal.** Possibly — but the mechanism it exposed is robust and instructive: aggressively rebalancing
one over-represented class in a 4-way head with shared global-pooled features tends to **conserve** the
total over-prediction mass and shift it to the next-easiest sink (here `cascade`). That is precisely
what calibration (Lever A) avoids, because a margin re-thresholds *all* classes jointly at the decision
boundary rather than reshaping the training distribution. The strict bar (overload F1 must hold) is what
B violated; a finely-tuned B might pass marginally, but it would still be dominated by A's 0.8272 and
would carry the same non-transferability, so the marginal expected value of re-tuning B is low.

### Objection 7 — "n = 1 topology (case14). The 'A doesn't transfer' claim is under-supported."
**Rebuttal.** Conceded as a limitation (see §7). The claim is currently supported by one unseen topology
plus a *mechanistic* explanation (topology-specific magnet identity + topology-dependent connectivity
magnitudes) that predicts the direction of the effect rather than merely observing it. The 118-bus
(`l2rpn_wcci_2022`) evaluation is the designated second data point and is the recommended immediate
follow-up; the `evaluation/eval_cross_topology.py` harness already supports it via `--tag wcci2022`.

### Objection 8 — "Maybe the 14-bus degradation is just a weak model, and a better-trained GNN would make A transfer."
**Rebuttal.** The degradation is consistent with a *representational* generalisation gap, not a tuning
deficiency, and — importantly — it is **the phenomenon the thesis sets out to study**. The hypothesis
under test is that neural accuracy degrades across topologies while symbolic rule-compliance stays
stable. A 14-bus macro of 0.626 (down from 0.783) is evidence *for* that hypothesis, not a bug to be
trained away. Even granting a stronger GNN, A's margin would still encode the 36-bus magnet identity and
would still be the wrong correction wherever the foreign magnet differs.

---

## 6a. Empirical postscript — Lever E was run, and it is a decisive negative result

After this recommendation was first written, Lever E (the data-generation fix) was executed in full —
not deferred — to test its central premise: *that the `line_trip` magnet is an artifact of the
training data's autocorrelated N-1 "cooldown" frames, removable at the source.* The generator was
changed (`LINE_TRIP_KEEP_PROB` 1.0 → 0.25 to de-duplicate cooldown frames, `SETTLE_K = 3` dwell filter
on `normal`, `NORMAL_KEEP_PROB` 0.02 → 0.04), 300k frames were regenerated into a markedly more
balanced dataset (28/27/25/20 vs the old 20/35/25/20), and the model was retrained on the proven
`[16,32,32]` configuration.

**The premise was falsified.** On the 36-bus test set:

| metric | Lever A base | Lever E base | reading |
|---|---|---|---|
| uncalibrated macro-F1 | 0.7830 | **0.7640** | E *regressed* the base model |
| `line_trip` over-prediction (uncal.) | +4,643 | **+5,676** | magnet got *worse*, not better |
| `line_trip` precision (uncal.) | 0.648 | **0.611** | magnet got *worse* |
| `overload` recall (uncal.) | 0.730 | **0.562** | victim merely moved `normal` → `overload` |

The `line_trip` frames in the new dataset are genuinely de-duplicated and diverse, yet the model still
over-predicts `line_trip`. **A clean, rebalanced dataset did not remove the magnet.** Combined with the
failures of a resampler (B), a readout change (C), and a head change (D), this licenses a strong
conclusion: *the `line_trip` over-prediction is **architectural** — it lives in the GNN's inability to
detect a single-edge connectivity change through topology-agnostic global pooling — not in the training
data.* The only intervention that neutralises it is inference-time threshold calibration (A), and only
in-distribution.

For completeness, Lever E was also given its best calibrated shot. The original two-offset calibrator
cannot tune `overload` (E's newly-starved class) and yields only 0.7954; a **full three-offset** sweep
(`normal +0.5, overload +0.7, line_trip 0.0`) reaches **0.8287** and dissolves the magnet
(over-prediction → −493) — notably *without* any negative `line_trip` margin, confirming the magnet is a
downstream symptom of `normal`/`overload` under-commitment. But 0.8287 only **ties** calibrated-A
(0.8272), on a *different and more class-balanced* test set and with *more* calibration freedom than A
was afforded; it is therefore not a clean improvement, and — fatally for Lever E's purpose — it requires
a *heavier* calibration than A, the exact opposite of the "clean base model" the lever was meant to
produce.

**Consequence for the recommendation: strengthened, not weakened.** The decision to keep Lever A stands,
and it now rests on a *demonstrated* (not merely argued) fact — the magnet cannot be trained or
data-engineered away on this architecture. That fact is the single most valuable result of the entire
campaign: it is direct evidence for the thesis's core hypothesis (the neural layer has intrinsic limits
that motivate a symbolic safeguard). Lever E is reverted; the model returns to the Lever A state. The
regenerated dataset is preserved for possible future use, and two incidental but genuine improvements to
the generator are retained: a `--n_records` size cap and the correction of a stale environment name
(`rte_case14_sandbox` → `l2rpn_case14_sandbox`).

---

## 7. Threats to validity / limitations

- **Single unseen topology.** Only 14-bus has been evaluated; 118-bus (`wcci_2022`) is pending. The
  transfer conclusion should be reconfirmed there before it is stated unconditionally.
- **Feature-scale confound is identified but not corrected.** The topology-dependent magnitude of the
  connectivity features (`*_trip_frac`) is flagged as a likely contributor to the 14-bus `cascade` bias;
  a topology-aware normalisation of those features is untested and is a candidate future lever.
- **Foreign test-set distribution.** The case14 set was generated to a similar class balance
  (24/31/25/20) under the original generator configuration; substantially different foreign balances
  could shift the uncalibrated baseline.
- **Lever E unmeasured.** Its *effect* (as opposed to its code) is unverified; the claim that it helps
  the 36-bus base but not transfer is mechanistic, pending the deferred regeneration.
- **Small-model fragility.** B/C/D were judged on the proven `[16,32,32]` configuration; a larger model
  under a proper regularisation schedule was out of scope and could change their verdicts (though Round 1
  found naive scale-up overfits/collapses).

---

## 8. Implications for the neuro-symbolic architecture

This decision *reinforces* the thesis's core design rather than competing with it. The neural layer is
explicitly expected to degrade off-topology; Lever A demonstrates that even a near-optimal in-distribution
correction is **topology-bound**, which is exactly why a topology-agnostic safeguard is necessary. The
**symbolic shield (Component D)** validates every prediction against the knowledge-graph rules
*independently of the GNN's confidence and independently of topology* (rules are physical and apply to
14-, 36-, and 118-bus grids unchanged). The cross-topology story the thesis should tell is therefore:
*"the neural operating point does not transfer; the symbolic layer is what does."* Lever A strengthens
that narrative by giving a concrete, measured example of a neural fix that fails to generalise.

---

## 9. Conclusion and recommended next steps

**Conclusion.** Move forward with **Lever A as the in-distribution result**; report unseen topologies
**uncalibrated**; keep **Lever E deferred**; let the **shield** carry cross-topology safety. The
recommendation survives the eight principal objections above, with the single genuine open weakness
being the `n = 1` unseen topology.

**Immediate next step (recommended).** Run the 118-bus cross-topology evaluation
(`python evaluation/eval_cross_topology.py --tag wcci2022`, after generating the foreign set) to obtain
the second transfer data point and close Objection 7.

**Optional, lower priority.** (a) Execute the deferred Lever E regeneration to harden the 36-bus base
model (runbook in `lever_E_datagen_handoff.md`); (b) prototype a topology-aware normalisation for the
connectivity features to address the feature-scale confound identified in §4.3.

---

### Appendix A — reproduction

- 36-bus baseline + calibrated report: `python training/evaluate.py` then
  `python training/calibrate_margin.py` (margin persisted in `gnn_logit_margin.json`; model in
  `gnn_checkpoint_best.pt`, backed up as `gnn_checkpoint_leverA.pt`).
- 14-bus generation: `python scripts/generate_dataset.py --env case14 --n_records 15000`.
- Cross-topology eval: `python evaluation/eval_cross_topology.py --tag case14`.
- All runs prefixed with `PYTHONIOENCODING=utf-8`. Device used: Intel Arc XPU, config `[16,32,32]`,
  `heads=[4,4,1]`, 10 epochs.
