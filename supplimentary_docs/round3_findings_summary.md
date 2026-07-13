# Round 3 findings summary — soft-F1 loss + seed-stability diagnostic

**Written 2026-07-12.** Consolidated record of the Round 3 session: what was tried, the data, the
reasoning, and what it means. Companion to the runbook `round3_gsat_softf1_plan.md` and the lever log
`normal_recall_experiments.md` (the single source of truth for "what's been tried"). GSAT/LSGAT are
**not** covered here — see the separate `gsat_lsgat_handoff.md`.

All runs: personal PC, Intel Arc **XPU**, `SMALL_CONFIG` `[16,32,32]`/`heads[4,4,1]`, 10 epochs,
batch 512, lr 1e-4, weighted-CE + label_smoothing 0.1 + loc_loss (0.5). This is the config that
produced the deployed baseline (see §2).

---

## 0. TL;DR

1. **soft-F1 hybrid loss (λ=0.1) → REJECTED (neutral).** Multi-seed mean macro 0.7339 vs baseline
   0.7362 — within noise; normal recall below the 0.88 floor; nothing collapsed, nothing improved
   beyond noise. Confirms once more: *loss-side levers do not move `normal` on this architecture.*
2. **The real discovery: the baseline is seed-unstable — and it's the INITIALIZATION, not the data
   split.** A follow-up diagnostic showed a fixed good init tolerates every train/val partition
   (0.82–0.90, never collapses), while varying the init degrades performance (0.82→0.72).
3. **Corrected headline:** the deployed `0.8277` is a lucky-**init** draw, but a *reproducible and
   conservative* one — not a favorable-split fluke. Deployed Lever A stands; init is a controllable knob.
4. **Deployed Lever A files were NOT touched** (REJECT verdict). Everything is working-tree only.

---

## 1. Baseline to beat (deployed Lever A, in-distribution 36-bus)

| Metric | Value |
|---|---|
| Calibrated test macro F1 | **0.8277** |
| normal recall (calibrated) | **0.8806** |
| Uncalibrated base macro | 0.7830 |
| Offsets (`gnn_logit_margin.json`) | normal +0.30, line_trip −0.10, others 0 |
| Cross-topology 14-bus (uncalibrated) | 0.6258 (margin does NOT transfer) |

Lever A = post-hoc per-class logit-margin calibration (Round 2 winner). Files: `gnn_checkpoint_best.pt`
== `gnn_checkpoint_leverA.pt`, `gnn_logit_margin.json`. Diagnosis behind it: `line_trip` is an
over-predicted "magnet" (precision 0.648); the margin de-biases it. Root cause of the `normal` problem
is **architectural** — global pooling can't resolve a single-tripped-edge signal — proven across
Round-2 levers B/C/D/E. This is why loss-side levers keep failing.

## 2. Config resolution (was an open question, now closed)

Inspected `gnn_checkpoint_best.pt` tensor shapes: `conv1.att (1,4,16)` (heads=4, hidden[0]=16),
`classifier.0.weight (128,96)` → `96 = 32*3` (hidden[2]=32). **The deployed checkpoint is the SMALL
config `[16,32,32]`/`[4,4,1]`, not the CUDA `[64,128,128]` branch.** Consequences: all Round-1/2
fragility evidence transfers (measured on this same config), and Round-3 runs must use it. Enforced via
`GRID_CONFIG=small` (`training/config.py`), which forces the small config even on a CUDA machine. On
the XPU personal PC the small config is auto-selected (device is not cuda).

---

## 3. Experiment: soft-F1 hybrid loss (Phase 1) — REJECTED

### 3.1 Mechanism
Differentiable macro soft-F1 surrogate added as an **additive** term on top of CE, with a pure-CE
warm-up (first ~30% of epochs), gated behind `lambda_f1` (default 0.0 = pure CE = baseline). Optional
linear ramp of λ after warm-up. Implementation: `soft_f1_loss()` + `f1_ramp_factor()` in
`training/train_gnn.py`; unit-tested in `tests/test_soft_f1_loss.py` (7 tests, all pass — soft-F1
matches sklearn macro-F1 at hard predictions, 0 at perfect, bounded [0,1], differentiable; ramp factor
correct).

### 3.2 Data (multi-seed, noise-aware — seeds 42/43/44)

| Lever | macro-F1 mean±σ | normal recall mean±σ | normal F1 | overload F1 | line_trip F1 | cascade F1 |
|---|---|---|---|---|---|---|
| baseline (λ=0) | 0.7362 ± 0.0768 | 0.7787 ± 0.1450 | 0.7794 | 0.7789 | 0.6464 | 0.7400 |
| soft-F1 (λ=0.1) | 0.7339 ± 0.0701 | 0.7961 ± 0.1408 | 0.7860 | 0.7761 | 0.6470 | 0.7266 |

Per-seed macro — baseline **0.8386 / 0.7162 / 0.6538**; soft-F1 0.8278 / 0.7146 / 0.6594.

### 3.3 Verdict + reasoning
**REJECT (neutral).** macro within noise; normal recall 0.796 < 0.88 floor; no class improves or
regresses beyond baseline σ. soft-F1 did NOT collapse anything (kill-switch not tripped) — it simply
didn't move the needle. This is the predicted negative result: soft-F1 is in the same family as the
already-failed focal loss / label-smoothing / resampling levers (all push the objective toward
minority recall), and the root cause is architectural, not gradient-shaped. **A documented negative is
still thesis value** — it's another independent confirmation of the neural-degrades / shield-holds
argument. Deployed Lever A untouched.

### 3.4 Note on the numbers
The single deployed 0.8277 does NOT match the multi-seed baseline mean 0.7362 — because 0.8277/0.8386
is the *good-seed* run and the mean is dragged down by 2 collapsed seeds. That discrepancy is exactly
what triggered the seed-stability diagnostic (§4).

---

## 4. Discovery: seed instability, localized to INITIALIZATION

### 4.1 Why we looked
The multi-seed baseline (§3.2) had **σ ≈ 0.077** on macro — enormous. Per-seed: 0.8386 / 0.7162 /
0.6538. Only seed 42 (the original hardcoded seed) reproduced the deployed regime; seeds 43/44
collapsed. But `--seed` drove BOTH weight init AND the in-memory shuffle that repartitions train/val
(val = checkpoint-selection set), so the collapse source was ambiguous.

### 4.2 The 2-arm design
Split the seed into `--init_seed` (weight init + batch order) and `--split_seed` (train/val
repartition), each defaulting to `--seed`. Ran (`training/diagnose_seed_stability.py`, baseline λ=0):
- **Arm A** — vary init, fix partition (s=42)
- **Arm B** — vary partition, fix init (=42)
- Shared anchor i42/s42 → 5 unique trainings.

### 4.3 Data

| Arm | held fixed | varied | per-run macro | mean ± σ |
|---|---|---|---|---|
| **A** | partition s=42 | **init** 42/43/44 | 0.8191 / 0.7696 / 0.7251 | 0.7713 ± 0.0384 |
| **B** | init=42 | **partition** 42/43/44 | 0.8191 / 0.8472 / 0.8970 | 0.8544 ± 0.0322 |

### 4.4 Reasoning — structure beats raw σ
The automated verdict read "both contribute" (σ 0.038 vs 0.032, ratio < 2). But the **shape** of the
variance is decisive:
- **Arm B never collapses.** With a good init fixed, *every* partition lands strong (0.82/0.85/0.90) —
  its σ is "all-good, varying degrees of good." → **the train/val split is fine; the eval is robust to
  partition. Do NOT touch the split.**
- **Arm A degrades** (0.82→0.77→0.72). Even with the good partition, a different init underperforms. →
  **initialization is the sensitive knob.**
- The Round-3 catastrophic 0.65 collapses were **bad-init × bad-partition compounding** — neither
  factor alone produces them.
- Underneath everything sits **~0.02 of pure XPU run-to-run nondeterminism** (the anchor read 0.8191
  here vs 0.8386 at the same seed in §3.2). Non-deterministic XPU atomics; a floor on reproducibility.

### 4.5 What it means for the deployed baseline
`0.8277` is a lucky-**init** draw — but reproducible and even conservative: init=42 gives 0.82–0.90
across partitions. The fragility is in initialization, which is **controllable** (fix the seed /
best-of-N), unlike partition luck (which would have undermined the whole evaluation — and doesn't).
**Deployed Lever A is a legitimate good-init result, not a favorable-split fluke.** For the thesis,
report it as init-conditioned, not as a robust point estimate.

---

## 5. Tooling added this session (all default-OFF / behavior-preserving)

| File | What |
|---|---|
| `training/config.py` | `SMALL_CONFIG`/`CUDA_CONFIG` split + `GRID_CONFIG` env override; new keys `lambda_f1`, `f1_warmup_frac`, `lambda_ramp_epochs` (all inert by default) |
| `training/train_gnn.py` | `soft_f1_loss()`, `f1_ramp_factor()`; CLI `--lambda_f1 --f1_warmup_frac --lambda_ramp_epochs --seed --init_seed --split_seed --ckpt_out` |
| `training/calibrate_margin.py` | `--checkpoint`/`--out` (no longer clobbers deployed margin); adds `test_per_class_f1` to output |
| `training/run_round3_multiseed.py` | multi-seed driver; `train_and_calibrate`, `aggregate`, `noise_aware_verdict`, `_mean_std`, `REPO_ROOT`; noise-aware strict bar |
| `training/diagnose_seed_stability.py` | 2-arm init-vs-partition variance decomposition |
| `tests/test_soft_f1_loss.py`, `pytest.ini` | 7 unit tests (soft-F1 + ramp) |
| `.gitignore` | `round3_runs/` (generated artifacts) |

Deployed behavior is unchanged unless a flag is set: `lambda_f1=0.0`, `lambda_ramp_epochs=0`,
`--seed` default = `SEED=42` (both init and split), `--ckpt_out` default = `gnn_checkpoint_best.pt`.

### 5.1 Noise-aware strict bar (the win condition, for any future lever)
Implemented in `run_round3_multiseed.noise_aware_verdict()`. A lever is KEPT only if, over ≥3 seeds:
- mean macro-F1 beats baseline mean by **more than baseline σ**, AND
- mean normal recall ≥ 0.88 floor AND ≥ baseline mean − σ, AND
- **no** class mean F1 drops below its baseline mean − σ.

This replaces the old single-run ±0.02 bar, which — given σ≈0.077 — could not distinguish signal from
seed luck. **Every future lever (GSAT included) must clear this multi-seed bar, not a single run.**

---

## 6. Reproduce / rerun commands

```bash
# soft-F1 multi-seed (baseline then lever, noise-aware verdict)
GRID_CONFIG=small python training/run_round3_multiseed.py --tag baseline   --lambda_f1 0.0 --seeds 42 43 44
GRID_CONFIG=small python training/run_round3_multiseed.py --tag softf1_0p1 --lambda_f1 0.1 --seeds 42 43 44 \
    --compare_to round3_runs/agg_baseline.json

# seed-stability diagnostic (init vs partition)
GRID_CONFIG=small python training/diagnose_seed_stability.py --seeds 42 43 44

# unit tests
python -m pytest tests/test_soft_f1_loss.py -q
```
Artifacts land in `round3_runs/` (gitignored). Always run with `PYTHONIOENCODING=utf-8` if console
encoding errors appear.

---

## 7. GSAT (Graph Stochastic Attention) — REJECTED (2026-07-12/13, decisive collapse)

Implemented per `gsat_lsgat_handoff.md` §4 corrected placement (edge gate feeds `edge_attr` BEFORE
conv1, so pooling sees the sparsified graph) with an IB-KL term (`beta*KL(Bernoulli(gate)||
Bernoulli(r))`) ramped in after a pure-task warm-up. Judged with fixed-init Arm-B isolation
(`init_seed=42`, `split_seed ∈ {42,43,44}`) against the same multi-seed noise-aware bar as soft-F1.

| Lever | macro-F1 mean±σ | normal recall mean±σ | normal F1 | overload F1 | line_trip F1 | cascade F1 |
|---|---|---|---|---|---|---|
| baseline_fixinit | 0.8514 ± 0.0192 | 0.8975 ± 0.0140 | 0.8818 | 0.8396 | 0.8361 | 0.8483 |
| GSAT β=0.01, r=0.6 | 0.5382 ± 0.0812 | 0.5486 ± 0.1821 | 0.5126 | 0.7175 | 0.3243 | 0.5981 |

**Verdict: REJECT — every one of the four per-class F1s regressed beyond baseline mean−σ**, not just
`normal`. This is qualitatively different from soft-F1 (neutral, nothing collapsed) — GSAT broke the
whole model. Per-seed macro/normal-recall: 0.4245/0.7799 (s42, `line_trip` F1→0.0004), 0.5805/0.3350
(s43, `normal` recall→0.335), 0.6095/0.5308 (s44) — three distinct failure shapes.

**β=0.1 was deliberately NOT run.** The handoff's kill-switch (§5.2) forbids escalating beta once a
class collapses at the smaller value — it already had, in two different ways, at β=0.01. User
confirmed the decision explicitly (2026-07-13): stop, do not run β=0.1 this session, defer further
GSAT work to a future session. The background orchestrator was killed before it could auto-launch the
β=0.1 phase; deployed Lever A files were verified untouched afterward.

**Reasoning:** GSAT is architecturally the same category of change as Round-2 Levers C (dispersion
pool) and D (two-stage head) — new learned structure upstream of / feeding the pooling operator this
small `[16,32,32]` model is fragile around (the handoff's own honest prior called this risk before
running anything). Two plausible compounding causes: (1) stochastic edge-dropout noise during training
is a large relative perturbation on a 36-node/59-edge graph, unlike loss-side levers which never touch
graph structure; (2) the r=0.6 prior may disconnect exactly the single-edge `line_trip` signal Round 2
already showed is the hardest thing for this architecture to preserve through pooling (seed 42's
`line_trip` F1→0.0004 is the clearest evidence). See `normal_recall_experiments.md` for the full
lever-table entry and "if picking this up again" notes.

**Thesis value:** GSAT is now a third independent architecture-side negative result (after C, D),
using a completely different mechanism (stochastic edge gating vs. readout/head changes), all failing
the same way — this strengthens the "pooling is the bottleneck, not attention or loss shape" argument
that underlies the shield-holds thesis narrative.

**Cross-topology (14-bus) OOD check was NOT run** — moot given the in-distribution collapse.

**Code status:** GSAT implementation stays on disk, default-OFF (`gsat_enabled=False`/`beta=0.0`),
fully reversible. 7 unit tests in `tests/test_gsat.py`, all passing.

## 8. Recommended next steps (priority order)

1. **Stabilize initialization before trusting any lever comparison.** Options: fix seed / best-of-N
   init via the harness (honest, cheap); reduce init sensitivity via lr-warmup, more epochs, or a
   different init scheme (a tiny model at 10 epochs lands different inits in different basins).
   Success metric: Arm-A σ shrinks toward Arm-B's.
2. **Do NOT touch the train/val split** — Arm B shows it's robust.
3. **GSAT is REJECTED** (§7) — do not retune beta/r on this architecture without a new hypothesis for
   why the collapse happened (see "if picking this up again" in `normal_recall_experiments.md`).
4. **LSGAT** — parked; no canonical technique (professor's generic Google suggestion). Revisit only if
   a real citation appears.
5. Three independent architecture-side experiments (Levers C, D, GSAT) have now failed — this is
   itself a strong, documented thesis result about the small model's fragility around pooling.
6. Keep appending every outcome to `normal_recall_experiments.md` in the lever-table format.
