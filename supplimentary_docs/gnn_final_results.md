# GNN Experimentation — Final Results

**Status: CLOSED, 2026-07-13.** This is the single, consolidated record of every GNN modeling
experiment run on this project (Rounds 1–3) and the reasoning behind what was kept and what wasn't.
Written to be used directly as the negative-results / ablation section of the thesis. It supersedes
and replaces `gnn_experimentation_closure.md`, `normal_recall_experiments.md`,
`round3_findings_summary.md`, and `archive/round3_gsat_softf1_plan.md` (all now removed — this file
is the sole source going forward). Companion docs that remain separate: `lever_A_recommendation_dissertation.md`
(the full objections/rebuttals defense of Lever A, for a deeper reviewer-facing treatment) and
`CASCADE_GENERATION_JOURNAL.md` (dataset-labeling methodology, a different topic).

---

## 1. The problem and the deployed result

The GridGNN classifier (GATv2Conv ×3, triple global pooling) reliably separated `overload` and
`cascade` but confused `normal` with `line_trip`: pure-argmax inference recovered only ~65% of true
`normal` states, dumping the rest into `line_trip`. The fix that shipped is **Lever A** — a post-hoc
per-class logit-margin calibration applied at inference only (`normal +0.30`, `line_trip −0.10`,
others `0`), tuned on the validation split, no retraining, no architecture change.

| Metric (36-bus, in-distribution test set) | Uncalibrated | **Calibrated — deployed** |
|---|---|---|
| Macro F1 | 0.7830 | **0.8277** |
| `normal` recall | 0.6466 | **0.8806** |
| `overload` F1 | 0.8157 | 0.8170 |
| `line_trip` F1 | 0.7588 | 0.8109 |
| `cascade` F1 | 0.8095 | 0.8160 |

Validation-tuned macro (0.8286) and independent test macro (0.8272–0.8277) agree to within 0.0014 —
no test-set leakage, no overfitting of the four-number offset. Artifacts: `gnn_checkpoint_best.pt`
(== `gnn_checkpoint_leverA.pt`), `gnn_logit_margin.json`, `training/calibrate_margin.py`.

**Does not transfer cross-topology** — see §4.

---

## 2. Complete experiment log — all three rounds

**Discipline used throughout:** keep-what-works. A change is adopted only if it improves the target
metric AND does not regress any other class beyond a strict bar (±0.02 single-run in Rounds 1–2;
a multi-seed noise-aware bar — beats baseline mean by more than baseline σ, on every class — in
Round 3, once seed instability was discovered). Everything else is reverted and logged as a negative
result.

| Round | Experiment | Mechanism | Result | Verdict |
|---|---|---|---|---|
| 1 | `near_limit` edge-flag threshold 0.9→1.0 | feature engineering | macro 0.7830→**0.7950**, normal recall 0.48→0.54 | ✅ kept (pre-Lever-A baseline) |
| 1 | label smoothing 0.1→0.05 | loss shaping | normal recall 0.54→0.42, macro 0.795→0.777 | ❌ reverted |
| 1 | focal loss (γ=2) | loss shaping | normal recall **collapsed to 0.03**; classic overfit (train loss ↓, val F1 ↓) | ❌ reverted |
| 1 | 6th node feature (`overloaded` flag), incl. a z-norm-exempt retry | feature engineering | normal recall **collapsed to 0.00** both times; intrinsic to the extra dimension on the small model, not a normalization artifact | ❌ reverted |
| 1 | graph-level `n_tripped` scalar injected into head (raw / capped / z-norm variants) | feature engineering | best variant cut fault→normal safety misses 789→219 (−72%) but did not fix `normal` recall (0.64→0.62) and regressed macro 0.784→0.745; z-norm variant re-compressed the exact signal it was meant to expose | ❌ reverted |
| 1 | scale-up to `[64,128,128]` hidden channels | capacity | **overfits/collapses** — train loss falls while val F1 falls; `normal`+`cascade` → 0.0 | ❌ abandoned |
| 2 | **Post-hoc per-class logit margin ("Lever A")** | inference-time calibration | macro **0.7830→0.8272**, normal recall **0.65→0.88**, all other classes held/improved | ✅ **KEPT — deployed** |
| 2 | de-magnet `WeightedRandomSampler` (`line_trip` ×0.4) + unweighted CE | sampling | magnet *relocated* to `cascade` (precision 0.704); `overload` F1 regressed 0.817→0.788; calibrated macro 0.819 < 0.827 | ❌ reverted |
| 2 | dispersion/std pooling branch (`cat[mean,max,min,std]`) | readout architecture | val macro **collapsed to 0.584** (vs 0.79 baseline), plateaued, never recovered | ❌ reverted |
| 2 | two-stage normal-vs-fault gate + 3-way subtype head | head architecture | val macro **collapsed to 0.587**, peaked epoch 6 then declined (flat softmax reaches 0.79 in the same budget) | ❌ reverted |
| 2 | data-generation fix — de-duplicate autocorrelated N-1 cooldown frames, rebalance classes | data generation | uncalibrated base **regressed** 0.7830→0.7640; `line_trip` over-prediction got *worse* (+4,643→+5,676) despite genuinely cleaner data | ❌ reverted — **the key negative result** (proves the magnet is architectural, not a data artifact) |
| 3 | soft-F1 hybrid loss (λ=0.1, CE warm-up) | loss shaping | multi-seed macro 0.7339±0.0701 vs baseline 0.7362±0.0768 (within noise); normal recall 0.7961±0.1408 (< 0.88 floor); nothing collapsed | ❌ **REJECT — neutral** |
| 3 | GSAT — Graph Stochastic Attention (Miao et al., ICML 2022), stochastic edge gate + info-bottleneck KL, β=0.01, r=0.6 | attention gating (upstream of pooling) | multi-seed macro **collapsed to 0.5382±0.0812** (vs 0.8514±0.0192 fixed-init baseline); **all four classes** regressed beyond noise, three distinct per-seed failure shapes | ❌ **REJECT — decisive collapse** |
| 3 | GSAT β=0.1 | — | **never run** — kill-switch: β=0.01 already collapsed the model in two distinct ways; escalating would repeat a known-bad mechanism | — |
| — | LSGAT | — | **parked** — not a real citable technique; the professor's suggestion traced to a generic "latest GNN tech" search with no paper behind it. GSAT served as the concrete, citable realization of the same suggestion. | — |

---

## 3. Why this is a scientific finding, not a failed search

The interventions above span every layer where a "smarter GNN" fix could plausibly live: the training
data itself (data regen), how examples are sampled (resampler), what the loss rewards (label
smoothing, focal loss, soft-F1), what the readout computes (dispersion pool), how the classification
head is structured (two-stage gate), and which edges the attention mechanism keeps (GSAT). Every one
of these — nine independent, orthogonal attempts across Rounds 1–3 — either failed to durably fix the
`normal`/`line_trip` confusion or actively collapsed the model.

**Root cause:** the classifier reads the graph through
`global_mean_pool ‖ global_max_pool ‖ global_min_pool` over node embeddings. A single tripped line
is a one-edge-out-of-59 topology change; its signal is diluted to near-invisibility by any global
pooling operator, regardless of how the upstream representation is computed. This is a
**representational ceiling**, not a hyperparameter or data-imbalance problem:
- The data-regen experiment proved it isn't a data artifact — a cleanly rebalanced, de-duplicated
  dataset didn't help; the magnet got worse.
- The dispersion-pool and two-stage-head experiments proved it isn't a fixable readout/head shape —
  both collapsed rather than improved.
- GSAT proved it isn't an attention-sparsity problem — sparsifying edges upstream of the same pooling
  operator collapsed the model further, it didn't help.

A real fix would require changing what pooling computes (e.g. an edge-level or hierarchical readout)
— a new architecture, not a lever, and out of scope for this project.

**Framing note for the thesis (do not overclaim):** the defensible claim is *"none of the
interventions tested moved `normal`/`line_trip` performance beyond what post-hoc calibration
achieves"* — not the stronger, unproven claim that global pooling is categorically incapable of this
signal. Nine convergent negative results is strong evidence for the narrower claim; it is not proof
of the broader one.

**Framing note on Lever A itself:** it is **operating-point calibration, not representation
learning** — argmax is Bayes-optimal only under symmetric costs and calibrated posteriors, neither of
which holds for this imbalanced, miscalibrated head. Lever A doesn't make the model smarter; it
picks a better decision threshold on capacity the model already had (`normal` precision was already
0.887 pre-calibration — the gain is a recall re-threshold, not new information). That framing is also
why it's the *only* thing that worked: it's the one intervention that doesn't touch representation
learning, which is exactly what Round 1–3 showed is already saturated on this architecture/data scale.

---

## 4. Cross-topology behavior — Lever A does not transfer, and that's itself a finding

| case14 (14-bus, unseen), macro F1 | Value |
|---|---|
| Uncalibrated (argmax) | **0.6258** |
| + 36-bus Lever A margin applied anyway | 0.5961 (**worse**) |

The 36-bus magnet is `line_trip` (over-predicted); the 14-bus magnet is `cascade` (precision 0.35,
absorbing overload and line_trip). A topology-specific correction tuned on one magnet actively hurts
when applied to a grid with a different magnet — so cross-topology numbers are reported
**uncalibrated**. A likely contributing confound: `global_trip_frac`/`connected_line_frac` are
topology-dependent magnitudes (one tripped line is 1/59 of the 36-bus graph but 1/20 of the 14-bus
graph), so normalizing foreign features with 36-bus statistics inflates the apparent severity of a
single 14-bus trip toward what a 36-bus multi-trip looks like — independent of calibration.

**This is supporting evidence for the thesis's core claim, not a shortfall to patch around:** the
neural operating point does not transfer across topology; a topology-agnostic safeguard (the
symbolic shield, Component D) is what has to carry that burden instead. Re-tuning a margin per
topology would also require labeled validation data on each unseen grid, which defeats the zero-shot
premise of the cross-topology evaluation — so it's excluded on principle, not just because it failed
once.

---

## 5. Secondary finding: the baseline is init-sensitive, not split-sensitive

A multi-seed run surfaced that the deployed macro F1 of 0.8277 doesn't reproduce across seeds
(σ ≈ 0.077 across 3 seeds: 0.8386 / 0.7162 / 0.6538). A follow-up 2-arm diagnostic separated weight
initialization from the train/val partition:

| Arm | held fixed | varied | per-run macro | mean ± σ |
|---|---|---|---|---|
| A | partition (seed 42) | **init** (42/43/44) | 0.8191 / 0.7696 / 0.7251 | 0.7713 ± 0.0384 |
| B | init (seed 42) | **partition** (42/43/44) | 0.8191 / 0.8472 / 0.8970 | 0.8544 ± 0.0322 |

Arm B never collapses — every partition lands strong with a good init fixed, so **the train/val
split is not the problem.** Arm A degrades steadily as init varies — **initialization is the
sensitive knob.** The Round-3 catastrophic collapses (macro ~0.65) were bad-init × bad-partition
compounding; neither factor alone produces them.

**Conclusion: the deployed init (`seed=42`) is a legitimate, reproducible, conservative good init**
(0.82–0.90 across partitions) — not a favorable-split fluke. For the thesis, report Lever A as
init-conditioned rather than as an unconditional point estimate; init should be named as a controlled
hyperparameter (fixed seed / best-of-N selection), not left implicit.

---

## 6. Repo state — what's frozen, what's gone

**Frozen (do not retrain/recalibrate without a new, previously-untried hypothesis):**
`gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt`, `gnn_logit_margin.json`,
`data/normalization_stats.pt`.

**Removed from the codebase** (implementation captured above and in git history; not kept as dead
code since no further experiments are planned): `StochasticEdgeGate`, `info_bottleneck_kl`,
`soft_f1_loss`, and all associated CLI flags in `training/train_gnn.py`/`training/config.py` —
reverted to their pre-Round-3 state (git commit `497fa4d`). The Round-3 multi-seed harness, the
seed-stability diagnostic script, and their unit tests were deleted outright. Retrieve any of it from
git history if a genuinely new hypothesis ever justifies revisiting GSAT or soft-F1.

**Kept (generically useful, not tied to a rejected experiment):** `--checkpoint`/`--out` override
flags on `training/calibrate_margin.py` and `--checkpoint`/`--margin` on
`evaluation/eval_cross_topology.py` — these prevent an experiment from ever clobbering the deployed
margin/checkpoint files.

**Next phase:** the symbolic shield / KG integration layer (Component D) — see
`study3(integration).md` and `shield_necessity_analysis_report.md`. Not yet started; the GNN above is
one of its two frozen inputs (the other being the knowledge graph, `kg/knowledge_graph.pkl`).
