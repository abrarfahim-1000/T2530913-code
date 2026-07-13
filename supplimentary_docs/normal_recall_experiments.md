# Normal-class recall experiments — handoff

**Goal:** raise `normal` recall (was ~0.48–0.50, high precision but under-committed) WITHOUT
regressing `overload` / `line_trip` / `cascade`. Methodology: try levers one at a time,
**keep-what-works** — adopt a change only if `normal` improves AND the other three hold; stack
the next on the retained state; revert anything that regresses.

Baseline before any of this (committed `main`): macro F1 ≈ 0.7701, `normal` recall ≈ 0.48.

---

## ROUND 2 (2026-06-25, REOPENED) — attack the `line_trip` magnet

**New diagnosis (from `training/evaluate.py` on `gnn_checkpoint_best.pt`, full 45k test split):**
the real problem is NOT normal-specific. `line_trip` is an **over-predicted magnet class**
(precision **0.648**, recall 0.915): 15,945 frames predicted line_trip vs 11,302 true → it absorbs
3,043 normals + 1,605 overloads + 959 cascades. normal→line_trip is a *symptom*. Test baseline:
macro **0.7830**, normal recall **0.6466** (P 0.887), others F1 0.8157 / 0.7588 / 0.8095.

Root cause traced to **data generation** (`scripts/generate_dataset.py:67-68`, never examined in
round 1): `NORMAL_KEEP_PROB=0.02` decimates normals while `LINE_TRIP_KEEP_PROB=1.0` keeps every
autocorrelated N-1 cooldown frame → oversized line_trip basin; sqrt-ICF weights (computed on
post-cap counts) don't see the low *effective* diversity.

All B–E rows are reported **calibrated** (own val-tuned margin) for apples-to-apples vs Lever A.

| # | Change | Test macro F1 | normal recall | other 3 F1 | Verdict |
|---|--------|---------------|---------------|------------|---------|
| **Lever A** | post-hoc logit margin `normal +0.30`, `line_trip −0.10` (tuned on val, `training/calibrate_margin.py`) | **0.8272** | 0.6466 → **0.8803** | 0.8170 / **0.8109** / **0.8160** (all ↑/held) | ✅ **KEPT** |
| Lever B | de-magnet WeightedRandomSampler (line_trip ×0.4) + unweighted CE (retrain, calibrated) | 0.8190 | 0.8522 | **overload 0.7883 ↓** / 0.8050 / 0.8191 | ❌ reverted |
| Lever C | extra **std (dispersion) pool** in readout, `cat[mean,max,min,std]` (retrain) | 0.5842 (val, uncal) | — | all collapsed | ❌ reverted |
| Lever D | **two-stage head** (normal-vs-fault gate + 3-way subtype, shared trunk, hierarchical log-prob NLL) | 0.5874 (val, uncal) | — | peaked ep6 then declined | ❌ reverted |
| Lever E | **data regen** (LINE_TRIP_KEEP 1.0→0.25, SETTLE_K=3, NORMAL_KEEP 0.02→0.04); full retrain | 0.7640 uncal / **0.8287 full-cal** | 0.73 uncal | overload **0.69 uncal** ↓↓ | ❌ reverted (negative result) |

Lever B detail: the sampler just **moved the magnet** — line_trip basin shrank but `cascade` became
the new over-predicted class (precision 0.704; 2,479 overloads leaked to cascade) and **overload F1
regressed 0.8170 → 0.7883**. Calibration could not recover overload; macro 0.8190 < 0.8272. Removing
the loss weighting + balancing entirely via sampling destabilized the majority/overload boundary.
Checkpoint restored from `gnn_checkpoint_leverA.pt`; code reverted to pristine.

Lever C detail: **`global_add_pool` would be redundant** — node count is constant at 36 (only edges
are pruned for tripped lines), so add-pool ≡ 36×mean-pool, zero new info. Substituted a **std/dispersion
pool** (captures "2 of 36 nodes deviate"). It destabilized the small `[16,32,32]` model — best val
macro 0.5842 vs 0.79 baseline, plateaued, never recovered. Same fragility-to-added-dimensions failure
mode as Exp 3b (6th feature) and the big-model scale-up. Reverted to 3-pool; checkpoint restored.

Lever D detail: shared trunk kept head params ≈ the flat `128→4` classifier (so this isolates the
*decision structure*, not capacity), and the hierarchical log-prob assembly kept `evaluate.py`/
`calibrate_margin.py` unchanged. Still, the gate+subtype objective gave the small model a worse,
unstable landscape — best val macro 0.5874, peaked epoch 6 then *declined* (the flat softmax reaches
0.79 in the same 10 epochs). Not slow convergence; reverted. Could be revisited with a longer
schedule, but out of scope here. Code reverted to pristine; checkpoint restored.

### Lever E — data-generation fix (RAN 2026-06-25 — NEGATIVE RESULT, the key scientific finding)
Implemented in `scripts/generate_dataset.py` and fully executed: `LINE_TRIP_KEEP_PROB` 1.0→0.25
(de-duplicate N-1 cooldown frames), `NORMAL_KEEP_PROB` 0.02→0.04, new `SETTLE_K=3` dwell filter on
`normal`. Regenerated 300k → re-split → reprocess → retrain (proven `[16,32,32]` config) on a now
much more balanced dataset (train: normal 28% / overload 27% / line_trip 25% / cascade 20%, vs the old
20/35/25/20). **Hypothesis tested: the line_trip magnet is a data-autocorrelation artifact. Result: it
is NOT — the magnet is architectural.**

Evidence (36-bus test):
- **Uncalibrated base REGRESSED: 0.7830 → 0.7640.** The magnet *persisted and worsened* — line_trip
  over-prediction +4,643 → **+5,676**, precision 0.648 → **0.611** — even though the line_trip frames
  are now genuinely de-duplicated/diverse. Cleaner data did NOT remove the magnet ⇒ the cause is the
  representation (single-missing-edge connectivity is hard to detect through global pooling), not the
  training-data composition.
- **E only moved the victim from `normal` to `overload`** (same pattern as Lever B): normal recall rose
  0.65→0.73 (cleaner normals from the settling filter) but **overload recall collapsed 0.73→0.56** and
  overload F1 0.8157→0.6905, because overload's data share fell to ~27% so the model now under-commits
  to it (overload leaks 2,555→line_trip, 2,102→cascade).
- **Even fully calibrated, E does not clearly beat A.** A 2-offset margin (the original tool) gives E
  only 0.7954 (overload untunable there). A *full 3-offset* sweep (normal +0.5, overload +0.7,
  line_trip 0.0) reaches **0.8287** and dissolves the magnet (over-pred → −493) — i.e. boosting the two
  starved classes drains the magnet, no negative line_trip margin needed. But this **ties** calibrated-A
  (0.8272) on a *different, more class-balanced* test set and with *more* calibration freedom than A was
  given, so it is not a clean win — and E still needs a *heavier* calibration than A, the opposite of
  the "clean base model" the lever was meant to deliver.

**Verdict (user decision): keep A, document E as a negative result.** E failed its purpose (clean base
model); its enduring value is the proof that the magnet is **architectural**, which is a stronger thesis
result than a data fix would have been (it motivates the symbolic shield). Reverted to the A state
(old data restored, `gnn_checkpoint_leverA.pt`). The new balanced dataset is preserved (not deleted) for
possible future use. The `--n_records` flag and the `rte_case14_sandbox`→`l2rpn_case14_sandbox` env-name
fix in `generate_dataset.py` are kept (genuine improvements).

---

## ROUND 2 OUTCOME

**Winner = Lever A** (post-hoc logit margin, inference-time only). Test macro **0.7830 → 0.8272**,
normal recall **0.6466 → 0.8803**, all other three F1 held/improved — clears the strict bar by a wide
margin. Artifacts: model `gnn_checkpoint_best.pt` (= `gnn_checkpoint_leverA.pt`), margin in
`gnn_logit_margin.json`, calibration/report tool `training/calibrate_margin.py`.

B (de-magnet sampler), C (std-pool), D (two-stage head) all FAILED the strict bar and were reverted —
B moved the magnet to cascade and hurt overload; C and D destabilized the fragile `[16,32,32]` model.
**Lever E (data regen) was subsequently RAN and also FAILED** (negative result, see its section above):
the regenerated/rebalanced dataset did **not** remove the magnet (uncalibrated base regressed
0.7830→0.7640, magnet got worse), proving the `line_trip` over-prediction is **architectural, not a
data artifact**. Fully calibrated E only ties A. Kept = **Lever A**; A state restored.

**The single most important round-2 finding** is therefore not a number but a mechanism: across five
levers — a sampler (B), a readout change (C), a head change (D), and a from-scratch data regeneration
(E) — *nothing in training or data removed the `line_trip` magnet from the base model.* Only inference-
time threshold calibration (A) neutralizes it, and only in-distribution. This pins the cause on the
GNN's representational capacity to detect a single-edge connectivity change via global pooling, and is
the empirical backbone of the "neural layer degrades / symbolic shield holds" thesis argument.

---

## CROSS-TOPOLOGY CHECK (2026-06-25) — does the Lever A margin transfer? **NO.**

Built `evaluation/eval_cross_topology.py` (topology-agnostic head + 36-bus normalization stats +
the saved A margin) and a 14-bus test set (`grid_dataset_case14.jsonl`, 15k, original gen config,
`l2rpn_case14_sandbox`). Also added a `--n_records` cap to `generate_dataset.py` and fixed a stale
env name (`rte_case14_sandbox` → `l2rpn_case14_sandbox`, the former isn't in this grid2op).

**case14 (14-bus) result — macro F1:**
- **Uncalibrated (pure argmax):** 0.6258  (normal R 0.9625, overload R 0.484, line_trip R 0.596, cascade R 0.497)
- **Calibrated (36-bus A margin):** 0.5961  ← **A HURTS off-topology** (−0.030)

**Why A doesn't transfer:** the failure mode is topology-specific. On 36-bus `line_trip` is the
over-predicted magnet → a negative line_trip margin helps. On 14-bus `line_trip` is *under*-predicted
(3,330 pred vs 3,750 true) and **cascade is the new magnet** (precision 0.35; overload→cascade 1,736,
line_trip→cascade 1,036). So the 36-bus-tuned `line_trip −0.10` is the wrong correction — it suppresses
already-scarce 14-bus line_trips and pushes normal precision down (0.827→0.734) for a near-ceiling
recall gain (0.96→0.99). A is an **in-distribution (36-bus) calibration**, not a generalizable fix.

Likely contributing confound (worth a note in the thesis): `global_trip_frac` / `connected_line_frac`
are topology-dependent magnitudes (1 trip = 1/20=0.05 on 14-bus vs 1/59=0.017 on 36-bus), so
normalizing foreign features with 36-bus stats distorts the connectivity signal — a 14-bus line_trip
looks like a 36-bus multi-trip. This is a generalization-gap issue independent of A/E.

**Implication for A vs E:** keep A as the **36-bus headline** but do **not** apply it cross-topology
(report uncalibrated there, or re-tune a per-topology margin on labeled foreign val — which undercuts
zero-shot). Lever E de-magnets line_trip *in the weights* (topology-portable in principle) so it's the
better path if the goal is transfer without per-topology tuning — BUT E targets line_trip, while the
14-bus problem is cascade, so E alone won't close the cross-topology gap. Still code-ready & deferred.

Lever A detail: inference-time only, no retrain, no regen. Reuses `gnn_checkpoint_best.pt`
(backed up as `gnn_checkpoint_leverA.pt`); offset saved to `gnn_logit_margin.json`. line_trip
over-prediction 4,643 → 1,078; normal→line_trip leak 3,043 → 902. Tuned on val (macro 0.8286),
confirmed on independent test (0.8272) → no leakage. Confirms the model already encodes `normal`
(precision 0.887); argmax just under-committed. **New best to beat for B–E (calibrated): 0.8272.**

Remaining levers (B–E) are evaluated WITH the same val-tuned margin applied (apples-to-apples),
and kept only if calibrated test macro beats 0.8272 under the strict bar.

---

## Results so far (all on personal-PC XPU smoke runs, 10 epochs)

| # | Change | Best macro F1 | normal recall | other 3 F1 | Verdict |
|---|--------|---------------|---------------|------------|---------|
| **Exp 1** | `near_limit` edge flag threshold **0.9 → 1.0** (`scripts/pyg_data.py`) | **0.7950** | 0.48 → **0.54** | 0.84 / 0.76 / 0.86 (intact) | ✅ **KEPT — FINAL** |
| Exp 2 | `label_smoothing` 0.1 → 0.05 (`train_gnn.py`) | 0.7774 | ↓ 0.42 | slightly down | ❌ reverted |
| Exp 3a | focal loss (γ=2), alpha=class_weights | 0.6634 | **0.03 collapsed** | overfit, all down | ❌ reverted (2026-06-25) |
| Exp 3b | +6th node feature `overloaded=(node_rho≥1.0)`, **exempt from z-norm** | 0.4805 | **0.00 collapsed** | all down | ❌ dropped (2026-06-25) |
| Exp 4 | inject graph-level `n_tripped` scalar into classifier head (3 variants) | 0.745 (raw) | — | line_trip/cascade ↓ | ❌ reverted (2026-06-25) |

### Exp 4 — n_tripped scalar injection (3 variants, all reverted)

Motivation: on the TEST split the baseline confuses `normal → line_trip` (3,075 of 9,128 normals).
Root cause: `normal` vs `line_trip` is decided by exactly one fact — is one of 59 lines gone — and
that signal lives only in `global_trip_frac` (0.0 vs 1/59=0.017), which after z-norm is ~0.15 std
apart and gets smeared by the GAT/BatchNorm. Idea: concat a graph-level **trip-count scalar**
straight to the pooled embedding before the classifier MLP (bypasses GAT smearing). Count is
topology-invariant ("1 tripped line"=1 everywhere) and computed from `line_status` (`count_tripped_lines`),
so NO hardcoded n_line — correct for 14/36/118-bus. Mechanism is graph-level, NOT a per-node input,
so it is NOT the Exp 3b failure (GAT input untouched; only the head widens by 1).

TEST-set results (rows=variant): macro F1 / normal P / normal R / normal→line_trip / faults→normal:
- **baseline**: 0.784 / 0.88 / 0.64 / 3,075 / 789  (line_trip R 0.92, cascade R 0.87)
- **raw count** (0..15): 0.745 / 0.96 / 0.62 / 2,673 / **219**  (line_trip R 0.79, cascade R 0.64)
- **capped @2**: 0.627 / 0.99 / **0.22** / 6,602 / 20  (normal recall collapsed)
- **z-norm** (mean 2.13, std **6.74**): 0.626 / 0.94 / 0.41 / 4,481 / 225

Why all reverted:
- **z-norm failed for a clean reason**: the cascade long tail inflates std to 6.74, which
  re-compresses normal(0)/line_trip(1) back to ~0.15 std — exactly the problem the injection was
  meant to fix. (Non-obvious; don't retry z-norm here.)
- **capped @2** made the head over-conservative — `normal` recall collapsed to 0.22.
- **raw** was the best injection variant and is genuinely interesting: it cuts faults-misclassified-as-
  normal 789→219 (−72%, the "overconfident wrong" safety failure mode) and lifts normal precision
  0.88→0.96. BUT it does NOT fix the original problem (normal *recall* flat 0.64→0.62 — missed normals
  just move line_trip→overload), and it regresses fault-type discrimination (cascade R 0.87→0.64,
  line_trip R 0.92→0.79) and macro F1 0.784→0.745. It also injects a **topology-dependent magnitude**
  (z-stats are 36-bus-derived; 118-bus cascades trip far more lines) — a confound for the cross-topology
  study. Net: not an "overall better state" for the thesis. Reverted to Exp 1 baseline.

**If revisited:** the raw-count injection is a viable lever IF the thesis prioritizes minimizing
fault→normal (safety) over per-class balance — its 789→219 drop is real. But weigh the cross-topology
magnitude confound. Code is fully reverted; see git history of this session for the implementation.

\* **3b crash, updated understanding:** scaling the model up to `[64,128,128]` to "fix" the
6-feature crash was a dead end — the big model **overfits and collapses** under this schedule
(train loss falls while val F1 falls; `normal` AND `cascade` go to 0.0; best epoch-1 macro only
~0.36). Big model abandoned. The proven `[16,32,32]` config (macro 0.795) is the best to date and
is what we work from.

**CONCLUSION (2026-06-25): both remaining levers FAILED; Exp 1 is the final state.**
- **Exp 3a (focal loss):** `normal` recall collapsed to ~0.03 (precision 0.98 — model almost never
  *commits* to `normal`); best macro 0.66; classic overfit (train loss → 0.076 while val F1 falls
  0.66 → 0.62). Confirms loss-side levers (cf. Exp 2) do not move `normal` here. Reverted to weighted-CE.
- **Exp 3b (6th feature, z-norm-exempt retry):** the z-scored-spike hypothesis was **wrong** — keeping
  the flag raw 0/1 did NOT prevent the collapse. `normal` recall went 0.0 → 0.0004 → 0.0; best macro
  only 0.48; overload/line_trip/cascade all regressed too. The collapse is **intrinsic to the extra
  input dimension on the small `[16,32,32]` model**, not a normalization artifact. Dropped per plan —
  Exp 1 already encodes the overload boundary at the edge level, so the node-level duplicate is redundant.
- After reverting both, a clean 5-feature weighted-CE retrain reproduced the reference: **best macro
  0.787, `normal` recall 0.54, others 0.84/0.75/0.86** — restored `gnn_checkpoint_best.pt`.

---

## Current code state (what's on disk right now)

- ✅ **Exp 1 KEPT**: `scripts/pyg_data.py` `near_limit = (rho >= 1.0)` (was 0.9).
- ✅ `data/processed_grid_data.pt` is the **5-feature** version (regenerated, matches Exp 1).
- ✅ `training/config.py` else/XPU branch is the **proven 0.795 config**:
  `hidden_channels=[16,32,32]`, `heads=[4,4,1]`, lr 1e-4, 10 epochs, batch 512. `NODE_FEATURES=5`.
  (The `[64,128,128]` upgrade was tried and reverted — it overfits/collapses; see results note.)
- ✅ Two latent **Unicode-on-cp1252 crash bugs fixed**:
  - `scripts/preprocess.py`: the `print("Saved → ...")` arrow crashed *before* `torch.save`,
    so older reprocess runs silently never wrote the `.pt`. Moved save before print, ASCII `->`.
  - `training/train_gnn.py:363`: the `✓` in the "New best saved" print crashed mid-training.
    Now `[best]`.
  - **Always run training/preprocess with `PYTHONIOENCODING=utf-8`** as a belt-and-suspenders.

---

## STATUS: CONCLUDED (2026-06-25)

All documented levers are exhausted. **Final state = Exp 1** (`near_limit=1.0`, 5 features,
small `[16,32,32]` weighted-CE config). Macro F1 ~0.787, `normal` recall 0.54, others
0.84/0.75/0.86. `gnn_checkpoint_best.pt` holds this model. Code on disk is the Exp 1 baseline
plus a documentation note in `train_gnn.py` recording the focal-loss failure.

**If picking this up again, do NOT re-run 3a/3b on the small model — both decisively failed
(see results table). New ideas would have to come from outside this plan**, e.g.: a `normal`-vs-rest
decision-threshold / two-stage head, oversampling `normal`, or revisiting capacity only with a
proper regularization schedule (the naive big-model scale-up overfits — see note above).

---

## (archived) original TODO — all items DONE

All commands assume: `source .venv/Scripts/activate` and prefix runs with `PYTHONIOENCODING=utf-8`.

**Reference baseline = Exp 1 small model, macro F1 0.795, `normal` recall ~0.54.**

### 1. Exp 3a — focal loss (no reprocess; stacks on the 0.795 state) — ❌ DONE, FAILED
Add this helper to `training/train_gnn.py` (e.g. after `build_loc_targets_fast`):
```python
def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Multi-class focal loss with per-class alpha. Down-weights easy (high p_t) samples
    by (1 - p_t)**gamma so gradient concentrates on hard, misclassified ones (the
    under-recalled `normal` examples predicted as `overload`). alpha = existing class_weights."""
    logp   = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t    = logp_t.exp()
    alpha_t = alpha.gather(0, targets)
    return (-alpha_t * (1.0 - p_t).pow(gamma) * logp_t).mean()
```
Replace the `cls_loss = F.cross_entropy(...)` block (currently `label_smoothing=0.1`) with:
```python
cls_loss = focal_loss(class_logits, batch.y, alpha=class_weights, gamma=2.0)
```
Retrain (no reprocess), compare to the 0.795 reference. **Caveat:** loss-side levers (Exp 2)
haven't moved `normal` so far — low expected value, but cheap. Revert to weighted-CE if it doesn't
beat baseline.

### 2. Exp 3b — 6th node feature `overloaded` (OPTIONAL; reprocess needed) — ❌ DONE, DROPPED
On the small model this collapsed `normal` to 0.0 (twice). Do NOT go bigger to fix it (big model
overfits). Instead retry ONCE with the suspected fix: **add the binary flag but exempt it from
z-normalization** (a z-scored sparse binary becomes a dominant spike — likely the destabilizer).
- `scripts/pyg_data.py`: `NODE_FEATURES = 6`; in `build_node_features` add
  `node_overloaded = (node_rho >= 1.0).astype(np.float32)` as the 6th stacked feature.
- `training/config.py`: `NODE_FEATURES = 6`.
- In `train_gnn.py` normalization step, leave column 5 raw (e.g. set `node_mean[5]=0`,
  `node_std[5]=1` after `compute_normalization_stats`) so the flag stays 0/1.
- Reprocess + retrain. If `normal` still collapses, **drop 3b** — Exp 1 already encodes the
  overload boundary at the edge level, so it's likely redundant, not essential.

### 3. Decide + finalize
Keep whichever of 3a/3b beats the 0.795 reference (both if they stack and each independently helps
`normal`). Revert losers. Regenerate the `.pt` to match the final `NODE_FEATURES`. Commit the winner.

---

## Gotchas / invariants
- **Changing `NODE_FEATURES`/`EDGE_FEATURES` ⇒ must reprocess `data/processed_grid_data.pt`.**
  The asserts at `train_gnn.py:258-261` catch a stale `.pt` — if they fire, delete the `.pt` and
  rerun `scripts/preprocess.py`.
- Per-class report prints every 5 epochs (`evaluate()` with `classification_report`). Watch
  `normal` **recall** and confirm overload/line_trip/cascade F1 don't drop more than ~0.01–0.02.
- `normal` signature to track: high precision (~0.88) + low/volatile recall = model under-commits
  to `normal`. The fix that worked (Exp 1) sharpened the normal↔overload boundary feature.

---

## Round 3 (2026-07-12) — soft-F1 hybrid loss ❌ REJECTED + a seed-instability discovery

Full plan/rationale: `round3_gsat_softf1_plan.md`. Ran the first multi-seed evaluation this project
has done (3 seeds), on the personal PC / XPU / small `[16,32,32]` config.

| Change | Test macro F1 (mean±σ, 3 seeds) | normal recall | other 3 F1 (norm/over/lt/casc) | Verdict |
|---|---|---|---|---|
| baseline λ=0 (multi-seed) | **0.7362 ± 0.0768** | 0.7787 ± 0.145 | 0.7794 / 0.7789 / 0.6464 / 0.7400 | reference |
| soft-F1 λ=0.1 (CE warm-up) | 0.7339 ± 0.0701 | 0.7961 ± 0.141 | 0.7860 / 0.7761 / 0.6470 / 0.7266 | **❌ REJECT — neutral, within noise** |

Per-seed macro — baseline **0.8386 / 0.7162 / 0.6538**, soft-F1 0.8278 / 0.7146 / 0.6594.

- **soft-F1 verdict:** neutral. No win beyond noise, normal recall fails the 0.88 floor, nothing
  collapsed (kill switch not tripped), nothing regressed beyond σ. Confirms once more: *loss-side
  levers do not move `normal` here.* Deployed Lever A untouched.
- **⚠ Bigger finding — the baseline itself is seed-unstable.** Only seed 42 (the original hardcoded
  seed) reproduces the good regime (0.8386 ≈ deployed 0.8277); seeds 43/44 collapse to ~0.72/0.65.
  **The deployed 0.8277 is a favorable-seed draw, not the config's expected value (~0.74 ± 0.08).**
  Every single-run comparison above (incl. Lever A and the ±0.02 bars) may be seed luck, not signal.
  New tooling: `--seed` flag + `training/run_round3_multiseed.py` (noise-aware bar). `--seed` drives
  both weight init AND the train/val repartition, so the collapse source is not yet localized.
- **Next diagnostic (higher priority than GSAT/I6):** separate init-variance from partition-variance
  (fix one, vary the other) to find why 2 of 3 seeds collapse. A model that only trains well on 1
  seed in 3 makes every lever comparison untrustworthy until fixed.

### Seed-stability diagnostic (2026-07-12) — it's INIT, not the data split ✅ localized

`training/diagnose_seed_stability.py` (separate `--init_seed`/`--split_seed`, baseline λ=0, small).

| Arm | fixed | varied | per-run macro | mean ± σ |
|---|---|---|---|---|
| A | partition s=42 | **init** 42/43/44 | 0.8191 / 0.7696 / 0.7251 | 0.7713 ± 0.0384 |
| B | init=42 | **partition** 42/43/44 | 0.8191 / 0.8472 / 0.8970 | 0.8544 ± 0.0322 |

- **Arm B never collapses** → the train/val split is fine; the eval is robust to partition. **Do not
  touch the split.**
- **Arm A degrades** (0.82→0.72) → **initialization is the sensitive knob.** Round-3's 0.65 collapses
  were bad-init × bad-partition compounding.
- ~0.02 residual XPU run-to-run nondeterminism (anchor 0.8191 here vs 0.8386 at same seed earlier).
- **Corrected read of the "0.8277 is lucky" claim:** it's a lucky-**init** draw, but reproducible and
  conservative (init=42 → 0.82–0.90 across partitions). Init is controllable ⇒ deployed Lever A is a
  legitimate good-init result, not a favorable-split fluke.
- **Fix (later):** treat init as a controlled hyperparam (fix seed / best-of-N via the harness);
  optionally reduce init sensitivity (lr-warmup / more epochs / init scheme) until Arm-A σ ≈ Arm-B σ.
  Only then are GSAT/soft-F1 comparisons trustworthy.

---

## Round 3 continued (2026-07-12/13) — GSAT (Graph Stochastic Attention) ❌ REJECTED — decisive collapse, kill switch tripped

Full plan/rationale: `gsat_lsgat_handoff.md`. Implemented per the handoff's **corrected placement**
(§4.2): a `StochasticEdgeGate` scores each edge from `edge_attr`, samples a Gumbel-softmax gate in
training (sigmoid at eval), and multiplies `edge_attr` **before conv1** so the sparsified topology
propagates into the pooled embedding the classifier reads. Loss adds `beta * KL(Bernoulli(gate) ||
Bernoulli(r))`, ramped in after a pure-task warm-up (30% of epochs) — the §4.3 decision made on paper
before coding: let the gate MLP learn useful edge scores under CE+loc first, then apply IB pressure,
rather than let three competing objectives fight from epoch 0.

**Init control:** fixed-init Arm-B isolation (`init_seed=42`, `split_seed ∈ {42,43,44}`), per the
handoff's minimum bar — isolates the lever from the init-luck confound documented above.

| Change | Test macro F1 (mean±σ, 3 seeds) | normal recall | other 3 F1 (norm/over/lt/casc) | Verdict |
|---|---|---|---|---|
| baseline_fixinit (init=42, split=42/43/44) | **0.8514 ± 0.0192** | 0.8975 ± 0.0140 | 0.8818 / 0.8396 / 0.8361 / 0.8483 | reference |
| GSAT β=0.01, r=0.6, τ=1.0 | **0.5382 ± 0.0812** | 0.5486 ± 0.1821 | 0.5126 / 0.7175 / 0.3243 / 0.5981 | ❌ **REJECT — all 4 classes regressed beyond noise** |

Per-seed macro/normal-recall (β=0.01): **0.4245/0.7799** (s42), **0.5805/0.3350** (s43),
**0.6095/0.5308** (s44) — three different failure shapes, none close to baseline.

**Noise-aware verdict (strict bar, `run_round3_multiseed.noise_aware_verdict`):** macro FAIL (0.538 vs
0.851±0.019), normal_recall FAIL (0.549 < 0.88 floor and < baseline−σ), and **every one of the four
per-class F1s REGRESSED beyond baseline mean−σ** — not just `normal`. This is not the narrow
normal-recall-only failure mode seen in earlier loss-side levers; GSAT broke the model wholesale.

**β=0.1 was NOT run.** Per the handoff's kill-switch discipline (§5.2): *"If beta=0.01 or 0.1
collapses a class... STOP — do NOT escalate beta. Escalating a mechanism already trending wrong is the
mistake the whole keep-what-works discipline forbids."* β=0.01 already collapsed all four classes in
two distinct failure modes (seed 42: `line_trip` F1 → 0.0004, essentially never predicted; seed 43:
`normal` recall → 0.335). A larger IB penalty (β=0.1) would only push the gate harder toward the same
r=0.6 keep-rate prior that already produced this collapse — there is no plausible mechanism by which
*more* sparsification pressure recovers a model that failed this badly at the smaller value. Explicit
user decision (2026-07-13): kill the β=0.1 run, close out, defer any further GSAT tuning to a future
session. The orchestrator process was terminated before it could auto-launch β=0.1; verified no stray
processes remained and deployed Lever A files (`gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt`,
`gnn_logit_margin.json`) were untouched (mtimes unchanged from before this session).

**Reasoning — why GSAT collapsed this model:** the handoff's own honest prior (§1) called this
correctly: *"a new mechanism ... shown fragile to upstream-of-pooling changes (Levers C/D collapsed
it)."* GSAT is architecturally the same category of change as Round-2 Levers C (extra dispersion pool)
and D (two-stage head) — it inserts new learned structure **upstream of / feeding into** the pooling
operator that the small `[16,32,32]` model already turns out to be extremely fragile around. Two
compounding mechanisms plausibly explain the severity (both consistent with the data, not
independently verified — worth noting as such):
1. **Stochastic edge dropout during training destabilizes an already-fragile small model.** Even with
   the beta warm-up, the Gumbel-softmax gate injects per-edge sampling noise into the very topology the
   3-layer GATv2 stack convolves over — on a 36-node/59-edge graph this is a large relative
   perturbation per batch, unlike the loss-side levers (soft-F1, focal, label-smoothing) which never
   touched the graph structure itself and only ever moved `normal` by ±0.02–0.15, never collapsed
   *all four* classes simultaneously.
2. **The prior r=0.6 may be a poor target for this graph.** Tripped lines are already pruned from
   `edge_index` at construction (per CLAUDE.md); a further ~40% random-ish drop of the *remaining*
   intact edges risks disconnecting exactly the single-edge signal (line_trip's defining feature) that
   Round 2 already proved is the hardest thing for this architecture's pooling to preserve (see the
   `line_trip`-magnet diagnosis and Lever-E finding that the magnet is architectural, not data-driven).
   Seed 42's near-total `line_trip` F1 collapse (0.0004) is the most direct evidence for this reading.

**Thesis value:** a third independent architecture-side mechanism (after C's readout change and D's
head change) that collapses this model when it touches anything upstream of global pooling — while
loss-side levers (soft-F1, focal, label-smoothing, GSAT's IB-KL) either do nothing (soft-F1) or actively
hurt (focal) `normal` alone, never the whole model. This strengthens, not weakens, the
"pooling-is-the-bottleneck / architecture is fragile / symbolic shield is load-bearing" thesis argument
— GSAT joins C and D as a documented negative result, and unlike them, quantifies the failure with a
proper multi-seed noise-aware bar rather than a single run.

**Files:** GSAT implementation is on disk, default-OFF (`gsat_enabled=False`/`beta=0.0` in
`training/config.py`), fully reversible with `--gsat` unset. `StochasticEdgeGate` / `info_bottleneck_kl`
in `training/train_gnn.py`; `--gsat/--beta/--r/--gsat_tau/--beta_warmup_frac` CLI on `train_gnn.py`,
`--gsat` on `calibrate_margin.py` and `evaluation/eval_cross_topology.py`; harness support
(`--gsat/--beta/--r/--gsat_tau/--fixed_init`) in `training/run_round3_multiseed.py`; 7 unit tests in
`tests/test_gsat.py` (KL≥0, KL=0 at gate==r, gate∈[0,1] train+eval, differentiable) — all passing.
Deployed Lever A untouched. Cross-topology (14-bus) OOD check (handoff §5.1) was **not run** — moot
given the in-distribution collapse; no plausible transfer benefit survives a broken 36-bus base model.

**If picking GSAT up again:** do not simply retune beta/r on this same architecture — the failure
pattern (all 4 classes, seed-dependent shape) suggests the stochastic edge-dropping mechanism itself is
incompatible with the small `[16,32,32]` model's already-thin margins, not a hyperparameter-tuning
problem. Higher-value directions: (a) test GSAT on a model that has first been stabilized against
Round-3's init-sensitivity (see the seed-stability section above) in case part of the collapse is
init×GSAT interaction, not GSAT alone; (b) try a much gentler gate (e.g. r closer to 0.9, minimal
dropout) as a diagnostic rather than a real lever, purely to see if the collapse is graded or a cliff;
(c) deprioritize GSAT — three independent architecture-side experiments (C, D, GSAT) have now failed,
which is itself a strong, already-documented thesis result.
