# Round 3 candidate levers — GSAT/attention-gating + soft-F1 hybrid loss

**Status (updated 2026-07-12): Phase 0 (config-pin guardrails) + Phase 1 (soft-F1 hybrid loss)
are now IMPLEMENTED and unit-tested, default-OFF. No training RUN has been executed yet — the
research-PC run is still deferred. Phase 2 (GSAT) and Phase 3 (LSGAT) remain unimplemented; this
doc is now the handoff for those.** Originally written 2026-07-06 after a professor consult
suggested (a) trying "LSGAT and GSAT variants" of the GNN and (b) feeding F1 into a hybrid loss
function instead of standard CE.

> **What changed on 2026-07-12 (code, not runs):**
> - **Config question (§3.4) RESOLVED.** Inspected `gnn_checkpoint_best.pt` tensor shapes
>   (`conv1.att (1,4,16)`, `classifier.0` in-dim `96 = 32*3`): the deployed Lever A checkpoint was
>   trained on the **SMALL config `[16,32,32]`/`[4,4,1]`**, NOT the CUDA `[64,128,128]` branch. So
>   ALL Round 1/2 fragility evidence transfers directly, and Round 3 runs must use the small config.
> - **`training/config.py`** now splits into named `SMALL_CONFIG`/`CUDA_CONFIG` with a `GRID_CONFIG`
>   env override (`GRID_CONFIG=small` forces the baseline config even on a CUDA machine — closes the
>   Risk R1 footgun where a research-PC run would silently train the unvalidated big config).
> - **`training/train_gnn.py`** gained `soft_f1_loss()` (§3.1) + a CE-warm-up blend gated behind
>   `lambda_f1` (default `0.0` = pure-CE baseline; `--lambda_f1`/`--f1_warmup_frac` CLI flags).
> - **`training/calibrate_margin.py`** now takes `--checkpoint`/`--out` so an experiment calibrates
>   its own margin file without clobbering the deployed `gnn_logit_margin.json` (Risk R4).
> - **`tests/test_soft_f1_loss.py`** (4 tests, all passing): soft-F1 matches sklearn macro-F1 at hard
>   predictions, is 0 at perfect prediction, bounded [0,1], and differentiable.
> - **LSGAT (§2) DOWNGRADED to parked.** Professor confirmed it was a generic "latest GNN tech"
>   Google result, not a specific paper — intent was exploratory ("try modern attention variants and
>   see if `normal` recall improves"). No canonical technique exists to implement; GSAT carries that
>   exploratory intent concretely. Do not chase an LSGAT acronym.

**Companion files:** `normal_recall_experiments.md` (Round 1/2 lever log — read this first),
`lever_A_recommendation_dissertation.md` (why Lever A was kept and what "strict bar" means),
`training/train_gnn.py`, `training/config.py`.

**Baseline to beat:** calibrated Lever A, test macro F1 **0.8272**, normal recall **0.8803**,
other three F1 0.8170 / 0.8109 / 0.8160 (36-bus, in-distribution). Uncalibrated base macro **0.7830**.
Cross-topology (14-bus) uncalibrated macro **0.6258** — no calibration applied there (A does not
transfer, see dissertation doc §3).

---

## 0. Executive summary / verdict up front

| Idea | Verdict | Why |
|---|---|---|
| **GSAT** (Graph Stochastic Attention, Miao et al. ICML 2022) | **Worth a scoped trial, NOT a confident win** | Real, citable, targets OOD generalization — matches the thesis's cross-topology framing. But the diagnosed root cause (global pooling can't resolve a single-edge change, `normal_recall_experiments.md` Lever C/E) is a *pooling* problem, not a *spurious-edge* problem. GSAT sparsifies which edges the attention keeps; it does not change what pooling does with the survivors. Plausible, unproven. |
| **LSGAT** | **Blocked — do not implement from a guess** | No canonical "LSGAT" is co-cited with GSAT in the graph classification literature the way GATv2/GSAT are. Best-guess candidates (spatio-temporal traffic-forecasting GAT+LSTM variant, or a sparsified-attention variant) don't obviously fit this problem shape (single-snapshot classification, not time series). Need the professor's actual citation before spending implementation time. |
| **Soft-F1 hybrid loss** | **Low expected value given prior evidence — try last, small, reversible** | Three loss/sampling-side levers already tried and failed for this exact imbalance problem: label-smoothing tune (regressed), **focal loss γ=2 (catastrophic — normal recall 0.65→0.03)**, WeightedRandomSampler de-magnet (moved the magnet to cascade). `normal_recall_experiments.md`'s own conclusion: *"loss-side levers do not move normal here."* Soft-F1 is a different mechanism from focal loss but the same family (reweight objective toward minority-class recall) and inherits the same risk. Also: `evaluate()` already selects checkpoints by val macro F1, not accuracy — the "wrong metric" gap a hybrid-F1 loss is normally sold to close is narrower here than usual. |

None of this should touch the kept Lever A state until proven better under the same strict bar used
for B/C/D/E: **adopt only if it beats 0.8272 calibrated macro F1 AND does not regress any of the
other three classes' F1.** Same keep-what-works discipline as Round 2.

---

## 1. GSAT — Graph Stochastic Attention

### 1.1 What it actually is
Miao, Liu, Li, *"Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism"*,
ICML 2022. Mechanism:
1. Compute attention logits over edges (already have this — `GATv2Conv` produces per-edge attention
   weights internally, just not exposed/used downstream today).
2. Convert each edge's attention weight into a Bernoulli/Gumbel-softmax sampling probability — the
   edge is stochastically **kept or dropped** per forward pass, not just down-weighted.
3. Add an **information bottleneck (IB) regularizer** — a KL divergence between the learned edge-keep
   distribution and a fixed prior (e.g. Bernoulli(r), r≈0.5–0.7) — to the loss. This penalizes the
   model for keeping edges that don't earn their place, forcing it toward a **minimal sufficient
   subgraph** for the label.

Net effect in the published results: better OOD generalization on graph classification benchmarks
where the shift is "spurious motifs correlated with the label in training but not in test" — and
improved interpretability (the surviving edges *are* the explanation).

### 1.2 Why it's tempting here
This project's whole thesis argument is GNN-degrades-on-unseen-topology / shield-holds. GSAT is
explicitly an OOD-generalization technique, which is the closest published match to "make the neural
layer generalize better across 14/36/118-bus" of anything the professor could plausibly have meant.
It also composes naturally with the current architecture: `GATv2Conv` → attention weights already
exist, so exposing them and adding a stochastic gate is an extension, not a rewrite.

### 1.3 Why it's not a confident win — the mismatch
GSAT's OOD setting is spurious-but-fixed-topology correlations (e.g., a benzene ring that happens to
correlate with mutagenicity in training molecules but shouldn't). Our OOD setting is **topology and
scale shift itself** (14→36→118 buses, different edge_index shapes entirely). The two Round-2 findings
that matter here:
- `normal_recall_experiments.md` Lever C: adding *any* new pooling statistic (`std`/dispersion pool)
  collapsed the small `[16,32,32]` model (val macro 0.584 vs 0.79 baseline) — confirms this specific
  architecture is fragile to changes in what gets pooled, which is exactly what GSAT's edge-gate would
  perturb upstream of pooling.
- Lever E (data regen) proved the `line_trip` magnet is **architectural** — "single-missing-edge
  connectivity is hard to detect through global pooling" — not a spurious-correlation artifact. GSAT
  sharpens *which* edges the attention attends to; it does not change the fact that
  `global_mean/max/min_pool` still has to compress whatever survives into 3 fixed-size vectors. A
  sparser input to a lossy pooling operator is not guaranteed to fix a lossy-pooling problem.

So: plausible upside (forces attention onto the tripped edge instead of averaging it away — might
actually help the exact failure mode diagnosed), real downside risk (another dimension/mechanism
change on a model already shown to be fragile to exactly this kind of change), unproven either way.

### 1.4 Implementation sketch (when picked up)

> **CORRECTION 2026-07-12 (I2) — the original sketch below has a placement bug; read this first.**
> The sketch computes the gate from "the final GAT layer" and applies it "before the final pooling
> stage." But `GridGNN.forward` is `conv1 → conv2 → conv3 → pool(x_emb)` (`train_gnn.py:62-78`):
> `conv3` is the **last** conv and pooling operates on the **node** embeddings `x_emb`, which are
> already computed by the time you'd gate. Gating edges after conv3 therefore changes **nothing that
> reaches the classifier** — it would be a near no-op for the classification head. Testing it as-is
> would burn ~1–2 days validating a gate that does nothing.
>
> **Correct placement: the stochastic edge mask must feed the conv stack, so the gated topology
> shapes the node embeddings that get pooled** (`gate → convs → pool`). Recommended shape: derive
> per-edge gate logits from a first cheap attention pass (or a small MLP on `edge_attr`), sample the
> gate, then run the MAIN conv stack on the gated edges — e.g. pass `edge_attr * gate.unsqueeze(-1)`
> (or hard-mask `edge_index`) into conv1/2/3 so pooling sees the sparsified graph. The IB-KL term then
> regularizes that gate. When you pick this up, replace the `return gate` comment and the "before
> final pooling stage" wording in the code below accordingly, and judge it with the multi-seed harness
> (§3.5), not a single seed.

No PyG-native module exists. Port from the [official GSAT repo](https://github.com/Graph-COM/GSAT)
(PyTorch/PyG-based, MIT-style research code). Rough shape (placement per the correction above),
targeting `training/train_gnn.py`:

```python
# New: an edge-attention extractor + stochastic gate sitting between conv layers and pooling.
# GATv2Conv exposes attention via return_attention_weights=True — currently unused (see
# GridGNN.forward, train_gnn.py:62-78). Would need conv1/2/3 calls changed to capture it.

class StochasticEdgeGate(nn.Module):
    def __init__(self, in_dim, temp=1.0):
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Linear(in_dim // 2, 1))
        self.temp = temp

    def forward(self, edge_attn_logits, training):
        # edge_attn_logits: raw attention logit per edge (pre-softmax), from the final GAT layer
        gate_logits = self.gate_mlp(edge_attn_logits)
        if training:
            # Gumbel-softmax relaxation for a differentiable Bernoulli sample
            gate = F.gumbel_softmax(torch.cat([gate_logits, -gate_logits], dim=-1), tau=self.temp, hard=False)[:, 0]
        else:
            gate = torch.sigmoid(gate_logits).squeeze(-1)
        return gate  # multiply into edge_attr or mask edge_index before final pooling stage

def info_bottleneck_kl(gate, r=0.6):
    # KL(Bernoulli(gate) || Bernoulli(r)) per edge, averaged
    eps = 1e-7
    gate = gate.clamp(eps, 1 - eps)
    return (gate * torch.log(gate / r) + (1 - gate) * torch.log((1 - gate) / (1 - r))).mean()
```

Loss becomes: `loss = cls_loss + loc_loss_weight * loc_loss + beta * info_bottleneck_kl(gate)`, with
`beta` a new hyperparameter (GSAT paper uses ~1e-2 to 1e-1 range, needs tuning here).

**New hyperparameters to sweep:** bottleneck prior `r`, KL weight `beta`, Gumbel temperature. This is
a genuinely new axis of tuning, not a drop-in — budget for it accordingly (estimate: 0.5–1 day to
port + wire in, another 0.5–1 day to get a stable sweep given the small model's known fragility).

### 1.5 Gate before implementing
Do NOT start this without first confirming on paper (a short note, not code) whether the IB gate is
expected to interact with the **loc_loss** (`build_loc_targets_fast` / `loc_loss_fn`, train_gnn.py:342-349)
— that auxiliary loss already forces node embeddings to encode fault location, which is a different
mechanism pushing on the same edges GSAT would be gating. Sequencing/weighting the two losses matters
and should be thought through before writing code.

---

## 2. LSGAT — PARKED (no canonical technique; professor's suggestion was exploratory)

**Resolution 2026-07-12:** asked whether the professor had a specific LSGAT citation. He did not —
the name came from a general Google search for "latest technologies in GNN," offered as an
exploratory pointer ("experiment with these and see whether the `normal`-class issue gets solved"),
not a specific paper to reproduce. There is therefore nothing concrete to implement under "LSGAT."
GSAT (§1) already carries the same exploratory intent — "try a modern attention-mechanism variant to
improve the neural layer" — and unlike LSGAT it maps to a specific, citable, PyG-portable technique.
**Action: do NOT implement an LSGAT acronym from a guess. Treat GSAT as the concrete realization of
this suggestion.** The original candidate analysis is retained below for the record.

Candidates considered and why each is a weak guess:
- **"Long Short-term Graph ATtention Network"** (traffic-forecasting literature, GAT + LSTM/GRU over
  time): doesn't fit — this project's records are independent snapshots (each JSONL row is one grid
  state), not a time series per node. Retrofitting temporal modeling would require restructuring the
  dataset around chronic sequences instead of i.i.d. classification instances, which is a much bigger
  change than "swap the layer."
- **A local/sparse-attention GAT variant** (for scaling attention to large graphs): could be relevant
  for the 118-bus WCCI topology if attention compute becomes a bottleneck, but that hasn't been
  reported as a problem — 118 buses / 186 lines is still small for a GAT.
- Possible the professor conflated/misspoke a name (e.g. meant GATv2, which is already in use).

**Action item before any code:** ask the professor for the exact paper/arXiv link. Don't burn
implementation time on a guessed acronym.

---

## 3. Soft-F1 hybrid loss

### 3.1 The mechanism, precisely
Macro-F1 is non-differentiable (needs hard argmax counts of TP/FP/FN). The standard differentiable
surrogate is **soft-F1**: use softmax probabilities in place of hard predictions to compute soft
per-class TP/FP/FN, then average per-class soft-F1 and use `1 − macro_soft_F1` as a loss term.

```python
def soft_f1_loss(logits: torch.Tensor, targets: torch.Tensor, n_classes: int, eps: float = 1e-7) -> torch.Tensor:
    """Differentiable macro soft-F1 surrogate. Not a replacement for CE — additive term only,
    per the prior-evidence discussion in normal_recall_experiments.md (loss-side levers have not
    moved `normal` on this architecture; focal loss collapsed it). Small weight, CE-warm-up first."""
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, n_classes).float()

    tp = (probs * targets_onehot).sum(dim=0)
    fp = (probs * (1 - targets_onehot)).sum(dim=0)
    fn = ((1 - probs) * targets_onehot).sum(dim=0)

    soft_f1_per_class = 2 * tp / (2 * tp + fp + fn + eps)
    return 1.0 - soft_f1_per_class.mean()
```

Hybrid loss: `cls_loss = ce_loss + lambda_f1 * soft_f1_loss(...)`, with:
- `lambda_f1` small (start 0.1–0.2 — this is a regularizer on top of CE, not a replacement).
- **CE-only warm-up** for the first ~30–40% of epochs before adding the soft-F1 term — early-epoch
  softmax outputs are near-uniform, so soft-F1 gradients are noisy/uninformative before the model has
  learned anything; this mirrors why label_smoothing=0.1 exists in the current loss (train_gnn.py:326-335,
  "prevents overconfidence collapse in early epochs").

### 3.2 The prior-evidence case against this (read before running)
From `normal_recall_experiments.md`, three loss/sampling-side interventions already tried on this
exact problem (raise `normal` recall / fix the `line_trip` magnet without regressing the other three):

| Lever | Mechanism | Result |
|---|---|---|
| Exp 2 — label_smoothing 0.1→0.05 | less smoothing | normal recall 0.54→0.42, macro 0.795→0.777 — **regressed** |
| Exp 3a — focal loss γ=2 | reweight toward hard/misclassified examples | **normal recall collapsed 0.65→0.03**, model stopped committing to the class; train loss fell while val F1 fell (pure overfit); macro 0.66 |
| Lever B — WeightedRandomSampler (line_trip ×0.4) + unweighted CE | reweight sampling toward minority classes | didn't fix the magnet, **relocated it to cascade**; overload F1 regressed 0.817→0.788; macro 0.819 < Lever A's 0.827 |

Stated conclusion in that doc: *"loss-side levers do not move normal here"* — diagnosis is
architectural (global pooling can't resolve a single-tripped-edge signal, confirmed independently by
Lever C readout change and Lever E full data regen) and data-driven (line_trip autocorrelation
inflating an oversized decision basin), not a gradient-weighting problem.

Soft-F1 is mechanistically different from focal loss (it's a class-aggregate differentiable metric
surrogate, not a per-example hard/easy reweighting), so it is **not guaranteed to fail identically** —
but it's in the same family (push the objective harder toward minority-class recall) and the honest
prior, given 3/4 loss-side levers failed or made things worse for a diagnosed non-loss root cause, is
**skepticism, not optimism.**

One more point worth keeping in mind: `evaluate()` (train_gnn.py:174-196) already selects the
checkpoint to save by **val macro F1**, not accuracy or val loss (`if val_f1 > best_val_f1`). So the
usual pitch for a metric-aware loss — "you're training for the wrong objective and only checking the
right one at eval time" — doesn't apply cleanly here; model *selection* already targets F1. What a
soft-F1 loss term would change is only the **per-step gradient shape** during training, which is a
narrower, less certain win than the standard pitch implies.

### 3.3 Experiment protocol (when picked up)
Follow the exact Round-2 discipline — one lever, isolated, strict bar, cheap to revert:

1. Start from the **Lever-A-restored state** (`gnn_checkpoint_leverA.pt` == current
   `gnn_checkpoint_best.pt`, offset in `gnn_logit_margin.json`). Do not stack on top of an unproven
   change.
2. Add `soft_f1_loss` as in §3.1, gated behind a CLI flag or config toggle so it's a one-line revert.
3. Warm-up schedule: pure CE (current behavior) for the first ~30% of epochs, then blend in
   `lambda_f1 * soft_f1_loss`. Try `lambda_f1 ∈ {0.1, 0.2}` — do not sweep wildly, this is meant to be
   a cheap probe given the negative prior, not a full hyperparameter search.
4. Retrain (no reprocess needed — no feature/architecture change), evaluate on the same 45k 36-bus
   test split used for Lever A/B/C/D/E, then re-run `training/calibrate_margin.py` fresh (do not reuse
   the old Lever A offset — a different loss landscape needs its own calibration).
5. **Keep only if:** calibrated test macro F1 > 0.8272 AND normal recall does not regress below 0.88
   AND none of overload/line_trip/cascade F1 drops. Otherwise revert to Lever A and log as a negative
   result (like Lever E) — a documented failure that confirms the architectural diagnosis is still
   valuable evidence for the thesis, even if the lever itself doesn't win.
6. If `lambda_f1=0.1` regresses like focal loss did, **stop — do not escalate lambda.** Exp 3a already
   showed this class of intervention fails hard, not gradually; escalating weight on a mechanism that's
   already trending wrong is the same mistake Exp 2 (smoothing tune) avoided by not chasing sunk cost.

### 3.4 Pre-flight open question: which config/device — RESOLVED 2026-07-12

**RESOLVED: the deployed baseline is the SMALL `[16,32,32]`/`[4,4,1]` config.** Verified by
inspecting `gnn_checkpoint_best.pt` tensor shapes (`conv1.att (1,4,16)` → heads=4, hidden[0]=16;
`classifier.0.weight (128,96)` → `96 = 32*3` → hidden[2]=32). `gnn_logit_margin.json` confirms this
checkpoint yields the baseline (test macro 0.8277, normal recall 0.8806). Consequences: (1) all
Round 1/2 fragility evidence in §3.2 transfers, it was measured on this exact config; (2) Round 3
runs MUST use the small config — now enforceable on any machine via `GRID_CONFIG=small`
(`training/config.py`). The stale/self-contradictory CUDA-branch comment noted below has been
superseded by the named `SMALL_CONFIG`/`CUDA_CONFIG` split. Original reasoning retained:
`training/config.py` currently has two divergent hyperparameter sets: the CUDA (research-PC,
production) config uses `hidden_channels=[64,128,128]`, 50 epochs, batch 256, lr 5e-4; the
non-CUDA/personal-PC config uses the proven `[16,32,32]`, 10 epochs, batch 512, lr 1e-4 — the one all
of Round 1/2's fragility findings (Lever C/D collapse, Exp 3b collapse) were measured against. Confirm
**which config produced the committed `gnn_checkpoint_best.pt` / Lever A numbers** before running this
experiment on the other one — the fragility evidence above may not transfer between the two scales.
(Quick check: `training/config.py` DEVICE branch + whichever machine `gnn_checkpoint_best.pt` was last
written on.)

### 3.5 Multi-seed, noise-aware evaluation (I1/I4/I5) — added 2026-07-12

**Why:** Round 1/2 (and the `0.8277` baseline) are single-run point estimates on a `[16,32,32]`,
10-epoch model the doc repeatedly calls fragile. A lever reading `0.831` could be seed noise; a real
win could be hidden by an unlucky seed. Judging any Round 3 lever on one run is statistically weak —
exactly the kind of shaky inference this project's discipline is meant to avoid.

**Protocol (now tooled):** `training/run_round3_multiseed.py` forces `GRID_CONFIG=small`, trains a
lever across ≥3 seeds (each via `--seed`, now a real flag — previously `SEED` was hardcoded so
multi-seed was impossible without editing the file), calibrates each run to its own margin file, and
aggregates test metrics as **mean ± std**.

1. **Establish the noise floor first (I4):**
   `python training/run_round3_multiseed.py --tag baseline --lambda_f1 0.0 --seeds 42 43 44`
   The baseline σ this produces *is* the bar every lever must clear.
2. **Run the lever against it:**
   `python training/run_round3_multiseed.py --tag softf1_0p1 --lambda_f1 0.1 --seeds 42 43 44
   --compare_to round3_runs/agg_baseline.json`
3. **Noise-aware strict bar (the win condition):** a lever is KEPT only if
   - mean macro-F1 beats baseline mean by **more than baseline σ**, AND
   - mean normal recall ≥ 0.88 floor AND ≥ baseline mean − σ, AND
   - **no** class's mean F1 drops below its baseline mean − σ.
   Otherwise REJECT and revert. (Implemented in `noise_aware_verdict()`.)
4. **Keep-what-works tree (I5):** `baseline(3 seeds) → soft-F1(3 seeds)`; keep soft-F1 only if it
   clears the bar; then GSAT (§1) is judged **on whichever base is currently best-kept**, never
   stacked on an unproven change.
5. **Kill switch still applies:** if `lambda_f1=0.1` *collapses* a class (focal-style), STOP — do not
   escalate to `0.2` or the ramp. Only escalate if `0.1` is neutral/promising (§3.3 step 6).

Caveat worth recording: `soft_f1_loss` is computed **per batch**, a slightly biased estimator of
dataset-level soft-F1. At `batch_size=512` with ~20% per class each batch has ~100 examples/class, so
the bias is small — acceptable, but note it when interpreting a marginal result.

### 3.6 Optional fallback probe (I6) — only if plain soft-F1 is NEUTRAL, never if it collapses

Plain macro soft-F1 weights all four classes equally. If it comes back *neutral* (clears no bar but
does not collapse any class), one cheap follow-up is a **class-targeted soft-F1** that up-weights the
diagnosed problem classes `{normal, line_trip}` in the mean instead of averaging all four equally —
i.e. `sum(w_c * soft_f1_c) / sum(w_c)` with larger `w` on those two. This is a *different knob* from
`lambda_f1` (which scales the whole term); it reshapes *within* the term. **Respect the kill switch:**
do NOT run this if plain soft-F1 *collapses* a class — that would be escalating a mechanism already
trending wrong, the exact mistake §3.3 step 6 forbids. Keep this documented as an option, not a
default next step; decide only after seeing the plain-soft-F1 aggregate.

---

## 4. Runbook checklist

**Done (2026-07-12, code only — no runs):**
- [x] Confirm which `TRAIN_CONFIG` branch produced `gnn_checkpoint_best.pt` / Lever A — **SMALL
      `[16,32,32]`/`[4,4,1]`** (verified from checkpoint tensor shapes). §3.4 resolved.
- [x] Resolve LSGAT — **parked**, no canonical technique (§2). GSAT is the concrete realization.
- [x] Soft-F1 (§3): implemented as a flagged additive term with CE warm-up, `lambda_f1` default 0.0
      (`training/train_gnn.py:soft_f1_loss` + blend). Unit-tested (`tests/test_soft_f1_loss.py`).
- [x] Config-pin guardrail: `GRID_CONFIG=small` override (`training/config.py`, Risk R1).
- [x] Calibration isolation: `calibrate_margin.py --checkpoint/--out` (Risk R4).

**RUN the soft-F1 experiment — now tooled via the multi-seed harness (§3.5):**
1. Baseline noise floor: `python training/run_round3_multiseed.py --tag baseline --lambda_f1 0.0
      --seeds 42 43 44`  (forces `GRID_CONFIG=small`, writes `round3_runs/agg_baseline.json`).
2. Soft-F1 lever vs the floor: `python training/run_round3_multiseed.py --tag softf1_0p1
      --lambda_f1 0.1 --seeds 42 43 44 --compare_to round3_runs/agg_baseline.json`.
3. Read the printed **noise-aware VERDICT** (KEEP/REJECT). Kill switch: if `0.1` *collapses* a class,
      STOP — do not escalate to `0.2`/ramp (§3.3 step 6, §3.5 step 5).
4. If KEEP: promote the best-seed checkpoint to `gnn_checkpoint_best.pt` and its margin to
      `gnn_logit_margin.json`. If REJECT: leave the deployed Lever A files untouched.
5. Either way append the aggregate (mean±σ) to `normal_recall_experiments.md` in the lever-table
      format so the Round 2 record stays the single source of truth.

**Results (2026-07-12, personal PC / XPU / small config, seeds 42/43/44):**

| Lever | macro-F1 (mean±σ) | normal recall (mean±σ) | normal F1 | overload F1 | line_trip F1 | cascade F1 |
|---|---|---|---|---|---|---|
| baseline (λ=0) | 0.7362 ± 0.0768 | 0.7787 ± 0.1450 | 0.7794 | 0.7789 | 0.6464 | 0.7400 |
| soft-F1 (λ=0.1) | 0.7339 ± 0.0701 | 0.7961 ± 0.1408 | 0.7860 | 0.7761 | 0.6470 | 0.7266 |

Per-seed macro — baseline: **0.8386 / 0.7162 / 0.6538**; soft-F1: 0.8278 / 0.7146 / 0.6594.

**VERDICT on soft-F1: REJECT (neutral).** macro 0.7339 vs 0.7362 is well within noise; normal recall
0.7961 fails the 0.88 floor; no class improves *or* regresses beyond baseline σ. This is the predicted
negative result (§3.2) — soft-F1 did NOT collapse anything (kill switch not tripped), it simply did not
move the needle. Loss-side lever confirms the architectural-diagnosis thesis once more. Deployed Lever A
files left untouched.

**BIGGER FINDING — the baseline is seed-unstable, and this is the real headline.** Only seed 42 (the
original hardcoded seed) reproduces the good regime (macro 0.8386 ≈ deployed 0.8277). Seeds 43 and 44
**collapse** to 0.7162 and 0.6538 (σ ≈ 0.077 on macro — enormous). **The deployed "0.8277" is a
favorable-seed draw, not the expected value of this config (~0.74 ± 0.08).** Implication: every
single-run comparison in Round 1/2 — including Lever A itself and the ±0.02 strict bars — may reflect
seed luck rather than signal. The multi-seed harness (I1) surfaced this on its first use; without it we
would still be treating 0.8277 as ground truth. This dominates the soft-F1 question: **with a noise
floor this high, no lever can be distinguished from noise at 3 seeds** — the priority shifts from
"which lever wins" to "why does this model train stably on 1 seed in 3."

Note: `--seed` drives BOTH weight init AND the in-memory shuffle that repartitions train/val (val =
checkpoint-selection set). So the collapse could be init instability, an "unlucky" val partition, or
both. Recommended next diagnostic — separate the two: vary init with a FIXED partition vs vary the
partition with FIXED init, to localize the instability before spending effort on GSAT or I6.

### Seed-stability diagnostic RESULT (2026-07-12) — it's INIT, not partition

Ran `training/diagnose_seed_stability.py` (separate `--init_seed`/`--split_seed`; baseline λ=0, small
config; anchor i42/s42, 5 unique runs):

| Arm | held fixed | varied | per-run macro | mean ± σ |
|---|---|---|---|---|
| **A** | partition (s=42) | **init** (42/43/44) | 0.8191 / 0.7696 / 0.7251 | 0.7713 ± 0.0384 |
| **B** | init (42) | **partition** (42/43/44) | 0.8191 / 0.8472 / 0.8970 | 0.8544 ± 0.0322 |

**Localization — the structure matters more than the raw σ (0.038 vs 0.032):**
- **Arm B never collapses.** With a good init fixed, *every* train/val partition lands strong
  (0.82 / 0.85 / 0.90). → **Partition luck is NOT the problem; the eval is robust to the split.**
- **Arm A degrades** (0.82 → 0.77 → 0.72). Even with the good partition, a different init underperforms.
  → **Initialization is the sensitive knob.**
- The Round-3 catastrophic collapses (0.65) were **bad-init × bad-partition compounding**; neither
  factor alone produces them.
- Underneath everything: ~0.02 of pure XPU run-to-run nondeterminism (anchor read 0.8191 here vs 0.8386
  at the same seed in the multi-seed run).

**Corrected headline:** 0.8277 is a lucky-**init** draw — but a *reproducible and conservative* one
(init=42 gives 0.82–0.90 across partitions). The fragility is in initialization, which is
**controllable** (fix the seed / best-of-N), unlike partition luck which would have undermined the
whole evaluation and does not. This partially **rehabilitates** the deployed Lever A: it is a
legitimately good, reproducible init, not a favorable-split fluke.

**Recommended fixes (for a later session — no more compute this session):**
1. **Treat init as a controlled hyperparameter.** Keep init=42 for deployment (justified), and/or use
   the multi-seed harness for explicit **best-of-N init** selection — honest and cheap.
2. **Reduce init sensitivity** so results don't depend on a lucky seed: try lr-warmup, more epochs, or
   a different init scheme (tiny model + 10 epochs ⇒ different inits settle in different basins).
   Re-measure Arm A σ after each — success = Arm A σ shrinks toward Arm B's.
3. **Leave the split alone** — Arm B says it's fine. Do NOT spend effort on chronic-split changes for
   this instability.
4. Only after the baseline is init-stable does comparing GSAT / soft-F1 variants become trustworthy.

**After soft-F1 is resolved — Phase 2 (GSAT), still to be written:**
- [ ] Paper-first note on IB-gate ↔ `loc_loss` interaction (§1.5) BEFORE any code.
- [ ] Port `StochasticEdgeGate` + `info_bottleneck_kl` from the official repo (§1.4). Expose
      `conv3` attention via `return_attention_weights=True` (currently discarded, `train_gnn.py:44,67`).
      New knobs (`beta`, `r`, `tau`) go in `SMALL_CONFIG`/`CUDA_CONFIG` alongside `lambda_f1`.
      Budget ~1–2 days incl. a first `r`/`beta` pass. Validate on the strict bar AND on 14-bus
      cross-topology (baseline uncalibrated macro 0.6258) — that OOD number is GSAT's actual claim.

---

## 5. Decision rationale — how this document reached its verdicts

Documenting the reasoning, not just the conclusions, since "why X over Y" is exactly what this
project's other supplementary docs already preserve (see `lever_A_recommendation_dissertation.md`,
written for the same reason).

### 5.1 GSAT vs LSGAT — a confidence asymmetry, not a coin flip
GSAT matched a specific, nameable paper: title, authors (Miao, Liu, Li), venue (ICML 2022), and a
mechanism I could describe precisely (stochastic edge attention + information-bottleneck KL term).
That's a high-confidence recall — specific enough that misattribution risk is low. LSGAT did not match
any single well-known paper commonly co-cited with GSAT the way GATv2 is co-cited with GAT. The
candidates I could generate (spatio-temporal traffic-forecasting GAT+LSTM, sparse-attention variants)
were domain mismatches or generic guesses, not a specific recall. Rather than hedge both ideas at the
same confidence level (which would either overstate LSGAT or understate GSAT), the two got different
verdicts on purpose: **differential confidence should produce differential recommendations.**
Presenting a low-confidence guess as implementable would risk burning real implementation time chasing
the wrong paper — the honest move was to gate it on the professor's citation instead of picking a
candidate and hoping.

### 5.2 Why "soft-F1" specifically, and not some other F1 surrogate
"Feed F1 into the loss" is underspecified — real candidates include soft macro-F1, differentiable
F-beta, and the Dice/Tversky loss family (common in segmentation). Soft macro-F1 was chosen because:
(a) it's the most direct, most commonly implemented translation of "put F1 in the loss" in applied
imbalanced-classification work; (b) it maps onto the exact metric this codebase already uses for
checkpoint selection (`evaluate()` picks the best checkpoint by val macro F1), so "CE vs CE+soft-F1"
is an apples-to-apples comparison against the metric that already matters here; (c) Dice/Tversky is
more natural for dense per-pixel/per-node prediction than for a single whole-graph 4-way label, which
is what the classifier head actually produces — so it was a weaker fit for this architecture and was
set aside rather than presented as an equal option.

### 5.3 Why the focal-loss failure was treated as a strong prior against soft-F1, not a separate case
Could have argued soft-F1 is a different enough mechanism from focal loss to deserve a clean, unbiased
trial. Decided against that framing because both interventions pull the same causal lever: increase
gradient pressure toward under-recalled minority classes at the expense of majority-class stability.
Three independent implementations of that lever — label-smoothing reduction, focal reweighting,
resampling — all failed on this exact architecture (`normal_recall_experiments.md`, Exp 2 / 3a / Lever
B). Generalizing "this family of intervention is risky here" from three independent failures is a
reasonable inductive step, not proof of a fourth failure — but it's enough to downgrade soft-F1 from
"promising new idea" to "cheap bounded probe with an explicit kill switch" (§3.3 step 6), rather than
treating it as a clean slate.

### 5.4 Why soft-F1 is sequenced before GSAT
Cost asymmetry, not idea quality. Soft-F1 needs no architecture change, no reprocessing, and is a
single flagged loss-function addition — cheap to fail fast. GSAT needs new modules ported, attention
internals exposed that aren't currently used, and 2–3 new hyperparameters tuned on a model already
shown fragile to structural changes (Levers C and D both collapsed the small model). Given comparable
uncertainty about payoff for both, the cheaper falsification goes first: if soft-F1 fails as expected,
that's a fast, low-cost confirmation of the "loss-side levers don't work here" pattern before
committing the 1–2 days GSAT would cost.

### 5.5 Why the CUDA/personal-PC config split was flagged instead of silently resolved
`training/config.py` has two divergent hyperparameter branches with a comment that partially
contradicts itself (the CUDA branch says "Reduced to prevent memorization" while listing the *larger*
`[64,128,128]` config — likely a stale copy-paste), and CLAUDE.md states production training runs on
the CUDA research PC, while the fragility evidence in memory (`normal_recall_experiments.md`) was
measured on "personal-PC XPU smoke runs" using the smaller `[16,32,32]` config. Rather than assume that
evidence transfers to whichever branch is actually authoritative, this was surfaced as an open question
(§3.4) to resolve before running Round 3 — silently picking one risks invalidating the entire
prior-evidence argument in §3.2 if the committed checkpoint actually came from the other branch.

### 5.6 Why neither idea was rejected outright
Both got "worth trying under constraints," not "don't do this." The professor's suggestions aren't
wrong on their face — GSAT genuinely is a real OOD-generalization technique, and soft-F1 genuinely is
the standard way to put F1 in a loss function. The point of this document is to calibrate expectations
and sequence effort against the project's own accumulated evidence, not to overrule the suggestion —
hence "try last, small, reversible" rather than "skip it."
