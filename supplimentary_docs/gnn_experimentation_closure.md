# GNN Experimentation — Closure Report

**Status: CLOSED, 2026-07-13.** No further GNN architecture, loss, or attention-mechanism
experimentation is planned. This document is the single entry point for "what did we try on the
GNN, and why did we stop" — written to be usable near-verbatim as a negative-results section in
the thesis. It indexes, not replaces, the full logs (§6).

---

## 1. The deployed result (what shipped)

**Lever A — post-hoc per-class logit-margin calibration.** A fixed offset vector is added to the
GNN's output logits at inference (`normal +0.30`, `line_trip −0.10`, others `0`), tuned on the
validation split, applied at `argmax(logits + offset)`. No retraining, no architecture change.

| Metric (36-bus, in-distribution) | Uncalibrated | Calibrated (deployed) |
|---|---|---|
| Macro F1 | 0.7830 | **0.8277** |
| `normal` recall | 0.6466 | **0.8806** |
| `overload` F1 | 0.8157 | 0.8170 |
| `line_trip` F1 | 0.7588 | 0.8109 |
| `cascade` F1 | 0.8095 | 0.8160 |

Files: `gnn_checkpoint_best.pt` (== `gnn_checkpoint_leverA.pt`), `gnn_logit_margin.json`,
`training/calibrate_margin.py`. **Does not transfer cross-topology** — on the unseen 14-bus grid
the same margin *reduces* macro F1 (0.6258 → 0.5961), because the off-topology failure mode is
different (there `cascade`, not `line_trip`, is the over-predicted class). Cross-topology numbers
are therefore reported uncalibrated (§4).

---

## 2. Headline finding: nothing tried since Lever A has beaten it

Across three rounds and **five distinct mechanism categories** — data generation, sampling,
readout architecture, classification-head architecture, loss shaping, and attention gating — every
attempt to raise `normal`/`line_trip` performance through training or architecture changes either
failed to move the needle or collapsed the model outright. Only an inference-time decision-rule
fix (Lever A) worked, and only in-distribution.

| Round | Experiment | Mechanism category | Result | Verdict |
|---|---|---|---|---|
| 1 | Exp 1 — `near_limit` edge-flag threshold 0.9→1.0 | feature engineering | macro 0.7950, normal recall 0.54 | ✅ **kept** (pre-Lever-A baseline) |
| 1 | Exp 2 — label smoothing 0.1→0.05 | loss shaping | normal recall 0.54→0.42 | ❌ reverted |
| 1 | Exp 3a — focal loss (γ=2) | loss shaping | normal recall **collapsed to 0.03** | ❌ reverted |
| 1 | Exp 3b — 6th node feature (`overloaded` flag) | feature engineering | normal recall **collapsed to 0.00** | ❌ reverted |
| 1 | Exp 4 — inject `n_tripped` scalar into head | feature engineering | fault-class F1 regressed, macro 0.784→0.745 | ❌ reverted |
| 2 | **Lever A — post-hoc logit margin** | inference-time calibration | macro **0.7830→0.8272**, normal recall **0.65→0.88**, all other classes held/improved | ✅ **KEPT — deployed** |
| 2 | Lever B — de-magnet resampler (`WeightedRandomSampler`) | sampling | magnet *relocated* to `cascade`; overload F1 0.817→0.788 | ❌ reverted |
| 2 | Lever C — dispersion/std pooling branch | readout architecture | val macro **collapsed to 0.584** (vs 0.79) | ❌ reverted |
| 2 | Lever D — two-stage normal-vs-fault gate head | head architecture | val macro **collapsed to 0.587**, peaked ep.6 then declined | ❌ reverted |
| 2 | Lever E — data-generation fix (de-duplicate N-1 cooldown frames) | data generation | uncalibrated base **regressed** 0.7830→0.7640; magnet got *worse*, not better | ❌ reverted (negative result) |
| 3 | soft-F1 hybrid loss (λ=0.1) | loss shaping | multi-seed macro 0.7339 vs baseline 0.7362 — within noise; normal recall 0.796 < 0.88 floor | ❌ **REJECT — neutral** |
| 3 | GSAT (Graph Stochastic Attention, β=0.01) | attention gating | multi-seed macro **collapsed to 0.538** (vs 0.851 fixed-init baseline); **all four classes** regressed beyond noise | ❌ **REJECT — decisive collapse** |
| 3 | GSAT β=0.1 | — | **never run** — kill-switch: β=0.01 already collapsed the model, escalating would repeat a known-bad mechanism | — |
| — | LSGAT | — | **parked** — not a real citable technique; professor's suggestion was a generic "latest GNN tech" search with no paper behind it. GSAT stood in as the concrete realization of the same suggestion. | — |

Full per-experiment data, confusion matrices, and reasoning are in the source logs (§6) — this
table is the index, not the evidence.

---

## 3. Why this is a *scientific* finding, not a failure to find the right knob

The interventions above touch every layer of the pipeline where a "smarter GNN" fix could plausibly
live: what data it trains on (Lever E), how examples are weighted during training (Lever B), what
the loss function rewards (Exp 2/3a, soft-F1), what the readout computes (Lever C), how the
classification head is structured (Lever D), and which edges the attention mechanism attends to
(GSAT). All nine independent, orthogonal attempts converge on the same outcome: none of them
durably fix the `normal`/`line_trip` confusion, and several actively destabilize the small
`[16,32,32]` model.

**Root cause (established across Lever C, D, E and reconfirmed by GSAT):** the classifier reads
the graph through `global_mean_pool ‖ global_max_pool ‖ global_min_pool` over node embeddings. A
single tripped line is a topology change affecting one edge out of 59 — its signal is diluted to
near-invisibility by any global pooling operator, regardless of how the upstream representation is
computed. This is a **representational ceiling**, not a hyperparameter or data-imbalance problem.
Lever E proved it isn't a data artifact (a cleanly rebalanced dataset didn't help); Levers C and D
proved it isn't a fixable readout/head shape (both collapsed rather than improved); GSAT proved it
isn't an attention-sparsity problem (sparsifying edges upstream of the same pooling operator
collapsed the model further, it didn't help). A real fix would require changing what pooling
computes (e.g. an edge-level or hierarchical readout) — a new architecture, not a lever, and out of
scope for this project's remaining timeline.

**Framing note for the thesis:** the defensible claim is *"none of the interventions tested moved
`normal`/`line_trip` performance beyond what post-hoc calibration achieves"* — not the stronger
(and unproven) claim that global pooling is categorically incapable of this signal. Nine negative
results is strong evidence for the narrower claim; it does not constitute a proof for the broader
one.

---

## 4. Cross-topology behavior (what does and doesn't generalize)

The Lever A margin is an in-distribution (36-bus) fix and is **not applied** to unseen topologies:

| case14 (14-bus, unseen) | macro F1 |
|---|---|
| Uncalibrated (argmax) | **0.6258** |
| + 36-bus Lever A margin applied anyway | 0.5961 (worse) |

The 36-bus magnet (`line_trip` over-predicted) and the 14-bus magnet (`cascade` over-predicted) are
different classes, so a topology-specific correction actively hurts off-topology. This is reported
as a finding, not patched around — see `lever_A_recommendation_dissertation.md` §4.3 and §6
(Objections 3–4) for the full argument on why per-topology re-tuning would undercut the zero-shot
premise of the cross-topology evaluation. **This is itself supporting evidence for the thesis's
core claim**: the neural operating point does not transfer across topology; a topology-agnostic
safeguard (the symbolic shield) is the thing that has to carry that burden instead.

---

## 5. A secondary finding: the baseline is init-sensitive, not split-sensitive

A multi-seed run surfaced that the deployed macro F1 of 0.8277 doesn't reproduce across seeds
(σ ≈ 0.077 across 3 seeds). A follow-up 2-arm diagnostic (`training/diagnose_seed_stability.py`)
localized this to **weight initialization**, not the train/val split: fixing a good init and
varying the split never collapses (0.82–0.90 across 3 partitions); fixing the split and varying
init degrades steadily (0.82→0.77→0.72). **Conclusion: the deployed init (`seed=42`) is a
legitimate, reproducible, conservative good init — not a lucky data-split fluke.** This
rehabilitates Lever A's reported number (it isn't cherry-picked on the split) while flagging that
init should be reported as a controlled hyperparameter, not implied to be split-independent luck.
Full detail: `round3_findings_summary.md` §4.

---

## 6. Source documents (full detail lives here — this report is the index)

| Document | Contents |
|---|---|
| `normal_recall_experiments.md` | Master lever-by-lever log, Rounds 1–3, all confusion matrices and per-class numbers — the primary source of truth |
| `lever_A_recommendation_dissertation.md` | Full decision record + defense for Lever A, written to be directly usable as a thesis chapter (objections/rebuttals format) |
| `round3_findings_summary.md` | Round 3 soft-F1 experiment + the seed-instability diagnostic, in full |
| `archive/gsat_lsgat_handoff.md` | GSAT implementation runbook and design correction notes (the session it describes is closed; archived, not deleted) |
| `archive/round3_gsat_softf1_plan.md` | Original Round 3 planning/rationale doc, superseded by the executed results above (archived, not deleted) |
| `archive/p3_changes.md` | Historical record of the Phase 3 architecture fixes (bidirectional edges, GATv2Conv, localization loss) now baked into the current codebase and `CLAUDE.md` |
| `CASCADE_GENERATION_JOURNAL.md` | Dataset-generation labeling fixes (separate from the GNN-modeling experiments above; still load-bearing, not archived) |

---

## 7. What is frozen going forward

`gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt`, `gnn_logit_margin.json`, and
`data/normalization_stats.pt` are the final GNN artifacts. No further training runs, calibration
sweeps, or architecture edits are planned against them. GSAT and soft-F1 code remain on disk,
default-off and fully reversible, in case a genuinely new hypothesis (not a retune of what already
failed) arises later.

**Next phase:** the symbolic shield / KG integration layer (Component D) — see `study3(integration).md`
and `shield_necessity_analysis_report.md`. This is a separate body of work, not yet started; the GNN
above is one of its two frozen inputs (the other being the knowledge graph, `kg/knowledge_graph.pkl`).
