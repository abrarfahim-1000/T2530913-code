# Round 3 candidate levers — GSAT/attention-gating + soft-F1 hybrid loss

**Status: PLAN ONLY. Nothing in this document has been implemented or run.** Written 2026-07-06
after a professor consult suggested (a) trying "LSGAT and GSAT variants" of the GNN and (b) feeding
F1 into a hybrid loss function instead of standard CE. This is the pre-flight dissection + runbook
for whoever picks this up next; execution deliberately deferred (not enough time in the session that
produced this doc).

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
No PyG-native module exists. Port from the [official GSAT repo](https://github.com/Graph-COM/GSAT)
(PyTorch/PyG-based, MIT-style research code). Rough shape, targeting `training/train_gnn.py`:

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

## 2. LSGAT — blocked on citation

No implementation plan until the professor's reference is confirmed. Candidates considered and why
each is a weak guess:
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

### 3.4 Pre-flight open question: which config/device
`training/config.py` currently has two divergent hyperparameter sets: the CUDA (research-PC,
production) config uses `hidden_channels=[64,128,128]`, 50 epochs, batch 256, lr 5e-4; the
non-CUDA/personal-PC config uses the proven `[16,32,32]`, 10 epochs, batch 512, lr 1e-4 — the one all
of Round 1/2's fragility findings (Lever C/D collapse, Exp 3b collapse) were measured against. Confirm
**which config produced the committed `gnn_checkpoint_best.pt` / Lever A numbers** before running this
experiment on the other one — the fragility evidence above may not transfer between the two scales.
(Quick check: `training/config.py` DEVICE branch + whichever machine `gnn_checkpoint_best.pt` was last
written on.)

---

## 4. Runbook checklist (for the follow-up session)

- [ ] Confirm which `TRAIN_CONFIG` branch (CUDA vs personal-PC) produced the current
      `gnn_checkpoint_best.pt` / Lever A numbers.
- [ ] Get the professor's actual LSGAT citation. Do not implement from a guess.
- [ ] Soft-F1 (§3): implement as a flagged additive term, CE warm-up, `lambda_f1 ∈ {0.1, 0.2}`, fresh
      calibration, strict-bar comparison against 0.8272. Expect a negative result given prior evidence
      — that's still a usable finding, document it either way.
- [ ] GSAT (§1): only after soft-F1 is resolved (cheaper, faster probe first). Port the edge-gate +
      IB-KL term from the official repo, budget ~1–2 days including a first tuning pass on `r`/`beta`.
      Think through interaction with `loc_loss` before writing code (§1.5).
- [ ] Whatever the outcome, append results to `normal_recall_experiments.md` in the same
      lever-table format (Change / Test macro F1 / normal recall / other 3 F1 / Verdict) so the
      Round 2 record stays the single source of truth for "what's been tried."

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
