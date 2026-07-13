# GSAT / LSGAT handoff — start-here for the next session

**Written 2026-07-12. Cold-start runbook.** You are picking this up fresh. Read this top-to-bottom
before writing code. Context lives in two companions:
- `round3_findings_summary.md` — all Round 3 results/data/reasoning (soft-F1 REJECTED, seed instability).
- `round3_gsat_softf1_plan.md` — the original dissection; §1 (GSAT), §1.4 (CORRECTED sketch), §1.5.
- `normal_recall_experiments.md` — the lever log; append every outcome here.

---

## 0. The one-paragraph orientation

We have a GNN fault classifier (36-bus, small config `[16,32,32]`/`heads[4,4,1]`). Deployed baseline =
"Lever A" (post-hoc logit-margin calibration), test macro **0.8277**, normal recall **0.8806**. A
professor suggested trying **GSAT/LSGAT** attention variants and an **F1-in-loss**. The F1 loss was
tried and **REJECTED** (neutral). GSAT is the remaining idea. **Before trusting any GSAT number you
must know:** the baseline is **initialization-unstable** — a good init reaches 0.82–0.90, a bad init
falls to ~0.72 — so single-run comparisons are meaningless. GSAT must be judged with the **multi-seed
noise-aware harness** already built.

## 1. Verdicts up front (do not re-litigate)

| Idea | Status | Action |
|---|---|---|
| **GSAT** (Graph Stochastic Attention, Miao et al. ICML 2022) | Real, citable, worth a scoped trial — NOT a confident win | Implement per §4, judge with multi-seed bar |
| **LSGAT** | **PARKED** — professor's generic Google suggestion, no real paper | Do NOT implement from a guess. See §6 |

**Honest prior on GSAT:** the diagnosed root cause of the `normal` problem is *pooling* (global
mean/max/min can't resolve a single-tripped-edge signal), proven across Round-2 levers B/C/D/E. GSAT
sparsifies *which edges attention keeps*; it does not change what pooling does with survivors. It also
adds a new mechanism to a model shown fragile to upstream-of-pooling changes (Levers C/D collapsed it).
Plausible upside (it could force attention onto the tripped edge instead of averaging it away), real
downside risk, unproven. Treat as a scoped experiment with an explicit kill switch, not a sure thing.

## 2. Prerequisites BEFORE starting GSAT (important)

1. **Decide how to control initialization** (the baseline is init-unstable — see
   `round3_findings_summary.md` §4). Two acceptable options:
   - **Minimum:** run GSAT and baseline both at **fixed `--init_seed 42`**, varying only `--split_seed`
     across 3 seeds (Arm-B style — the partition is robust, so this isolates the lever from init luck).
   - **Better:** first do the init-stabilization work (lr-warmup / more epochs / init scheme) so the
     baseline is init-robust, then compare GSAT against that stabilized baseline across full seeds.
   Do NOT compare a single GSAT run to the single deployed 0.8277 — that is exactly the mistake the
   multi-seed harness exists to prevent.
2. **Confirm environment:** `GRID_CONFIG=small` on every run (forces the deployed config). On the XPU
   personal PC small is auto-selected; on a CUDA machine you MUST set it. Expect ~0.02 XPU run-to-run
   nondeterminism as a reproducibility floor.
3. **Snapshot the restore point:** `gnn_checkpoint_leverA.pt` == `gnn_checkpoint_best.pt`. Never let an
   experiment overwrite these until it wins the bar. Write GSAT checkpoints to `round3_runs/` via
   `--ckpt_out` (the harness already does this).

## 3. The current architecture (what you're modifying)

`GridGNN.forward` in `training/train_gnn.py`:
```
x -> conv1(GATv2Conv, edge_dim) -> bn1 -> ELU
  -> conv2 -> bn2 -> ELU
  -> conv3 -> bn3 -> ELU          # x_emb  (node embeddings)
loc_logits = localizer(x_emb)      # auxiliary per-node fault-location head (BCE, weight 0.5)
graph_emb  = [mean_pool | max_pool | min_pool](x_emb)   # (last_dim*3,)
class_logits = classifier(graph_emb)                    # 4-way
```
`GATv2Conv` supports `return_attention_weights=True` (currently unused). Edge features are 4:
`[rho, p_or, q_or, line_status]`. Tripped lines are pruned from `edge_index` at graph construction.

## 4. GSAT implementation — the CORRECTED design (read the bug note)

> **⚠ The original sketch (`round3_gsat_softf1_plan.md` §1.4, pre-correction) had a placement bug:** it
> gated edges *after* `conv3`, but pooling operates on the node embeddings that `conv3` already
> produced — so gating there is a **no-op for the classifier**. You would spend a day testing nothing.
> The gate MUST feed the conv stack so the gated topology shapes the pooled embeddings.

### 4.1 Mechanism (Miao et al. 2022)
1. Score each edge (from `edge_attr`, or a first cheap attention pass) → per-edge keep-logit.
2. Sample a per-edge **gate ∈ [0,1]** (Gumbel-softmax relaxation in training; sigmoid at eval).
3. Run the MAIN conv stack on the **gated** graph (`edge_attr * gate`, or hard-mask `edge_index`), so
   pooling sees the sparsified graph.
4. Add an **information-bottleneck KL** term: `KL(Bernoulli(gate) || Bernoulli(r))`, averaged over
   edges, weighted by `beta`. Pushes the model toward a minimal sufficient subgraph.

### 4.2 Correct placement (pseudocode, gate BEFORE convs)
```python
class StochasticEdgeGate(nn.Module):
    def __init__(self, edge_in_dim, temp=1.0):
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(edge_in_dim, edge_in_dim), nn.ReLU(),
                                      nn.Linear(edge_in_dim, 1))
        self.temp = temp
    def forward(self, edge_score_input, training):
        gl = self.gate_mlp(edge_score_input)
        if training:
            return F.gumbel_softmax(torch.cat([gl, -gl], dim=-1), tau=self.temp, hard=False)[:, 0]
        return torch.sigmoid(gl).squeeze(-1)               # [E] in [0,1]

def info_bottleneck_kl(gate, r=0.6, eps=1e-7):
    g = gate.clamp(eps, 1 - eps)
    return (g*torch.log(g/r) + (1-g)*torch.log((1-g)/(1-r))).mean()

# in forward(), BEFORE conv1:
#   gate = self.edge_gate(edge_attr, self.training)          # [E]
#   ea   = edge_attr * gate.unsqueeze(-1)                    # gated edge features
#   x = ELU(bn1(conv1(x, edge_index, ea)))                   # pooled embeddings now depend on gate
#   ... conv2, conv3 on ea ...
#   return class_logits, loc_logits, gate                    # return gate for the KL term
```
Loss: `loss = cls_loss + loc_loss_weight*loc_loss + beta*info_bottleneck_kl(gate)`.

### 4.3 §1.5 gate — think through the loc_loss interaction ON PAPER FIRST
The auxiliary `loc_loss` (`build_loc_targets_fast` + `BCEWithLogitsLoss`, weight 0.5) already forces
node embeddings to encode WHERE the fault is — pushing on the same edges GSAT would gate. Before coding,
write a short note deciding: does the IB gate fight or reinforce loc_loss? Consider (a) reducing
`loc_loss_weight` while GSAT is on, or (b) a warm-up for `beta` like soft-F1's λ warm-up. Sequencing
and relative weights matter on this fragile model — do NOT wire three competing losses blindly.

### 4.4 New hyperparameters (add to BOTH `SMALL_CONFIG` and `CUDA_CONFIG`, default OFF)
- `gsat_enabled` (bool, default False) or `beta=0.0` as the off switch — keep it a one-line revert.
- `beta` (KL weight; paper range ~1e-2 to 1e-1) — first tuning axis.
- `r` (Bernoulli prior, ~0.5–0.7) — second tuning axis.
- `gsat_tau` (Gumbel temperature, ~0.5–1.0).
- Optional `beta_warmup_frac` mirroring `f1_warmup_frac`.
Wire a `--gsat`/`--beta`/`--r`/`--gsat_tau` CLI in `train_gnn.py` exactly like the soft-F1 flags, so the
multi-seed harness can drive it.

## 5. Run protocol (use the existing harness — do NOT hand-run single seeds)

1. **Extend the harness** (`training/run_round3_multiseed.py`): add GSAT flags to the `train_cmd` it
   builds (alongside `--lambda_f1`). Keep the noise-aware bar unchanged.
2. **Baseline first** (the noise floor). If you did NOT stabilize init, use Arm-B isolation
   (fixed init, vary split):
   ```bash
   GRID_CONFIG=small python training/run_round3_multiseed.py --tag baseline_fixinit \
       --lambda_f1 0.0 --seeds 42 43 44   # (then set init fixed in the harness, or vary split only)
   ```
   (If you stabilized init, use the stabilized baseline agg instead.)
3. **GSAT sweep** — start SMALL: `beta ∈ {0.01, 0.1}`, `r=0.6`, `tau=1.0`. One knob at a time.
   ```bash
   GRID_CONFIG=small python training/run_round3_multiseed.py --tag gsat_b01 --gsat --beta 0.1 --r 0.6 \
       --seeds 42 43 44 --compare_to round3_runs/agg_baseline_fixinit.json
   ```
4. **Read the printed noise-aware VERDICT** (KEEP/REJECT). Fresh calibration is already run per-seed by
   the harness (`calibrate_margin.py --checkpoint ... --out ...`).

### 5.1 Strict bar (unchanged) + the OOD check that actually matters for GSAT
- In-distribution: must clear the multi-seed noise-aware bar vs baseline (macro beats mean+σ; normal
  recall ≥ 0.88 and ≥ mean−σ; no class drops below mean−σ).
- **Cross-topology (GSAT's real claim):** GSAT is an *OOD generalization* technique. Even if
  in-distribution 36-bus is flat, check **14-bus** (`evaluation/eval_cross_topology.py`, baseline
  uncalibrated macro **0.6258**). A GSAT win would most plausibly show as improved cross-topology
  transfer, not in-distribution macro. Report both.

### 5.2 Kill switch / discipline
- If `beta=0.01` or `0.1` **collapses** a class (normal recall craters like focal loss did), STOP — do
  NOT escalate `beta`. Escalating a mechanism already trending wrong is the mistake the whole
  keep-what-works discipline forbids.
- If GSAT is **neutral** (within noise), REJECT and log it — a documented negative is thesis evidence
  for the architectural-diagnosis argument (pooling, not attention, is the bottleneck).
- **Never overwrite `gnn_checkpoint_best.pt` / `gnn_checkpoint_leverA.pt` / `gnn_logit_margin.json`**
  until GSAT wins the bar. Promote only on a KEEP.

## 6. LSGAT — parked

No canonical "LSGAT" technique exists in the graph-classification literature co-cited with GSAT. The
professor confirmed the name came from a generic Google search for "latest GNN tech," offered as
exploratory ("try modern attention variants, see if `normal` improves"), not a paper to reproduce.
Weak-guess candidates (traffic-forecasting GAT+LSTM = wrong problem shape, single-snapshot not time
series; sparse-attention GAT = no reported compute bottleneck) are all mismatches. **Action:** treat
GSAT as the concrete realization of "modern attention variant." Only revisit LSGAT if the professor
supplies a real arXiv/paper link — then re-plan from the actual mechanism, do not implement from the
acronym.

## 7. Definition of done for the GSAT session — CLOSED 2026-07-13, REJECTED

- [x] Init-control decided: **fixed-init isolation** (`init_seed=42`, `split_seed ∈ {42,43,44}`,
      Arm-B). Baseline agg: macro **0.8514 ± 0.0192**, normal recall **0.8975 ± 0.0140**.
- [x] GSAT implemented with the CORRECTED placement (gate feeds convs BEFORE conv1), default-OFF
      (`gsat_enabled=False`/`beta=0.0`), CLI-flagged (`--gsat --beta --r --gsat_tau --beta_warmup_frac`
      on `train_gnn.py`; `--gsat` on `calibrate_margin.py`/`eval_cross_topology.py`).
- [x] loc_loss ↔ IB-gate interaction reasoned on paper (§4.3): kept `loc_loss_weight=0.5` unchanged,
      added a `beta_warmup_frac` (pure-task warm-up before IB pressure ramps in) so the gate MLP
      learns useful edge scores under CE+loc first, mirroring soft-F1's warm-up pattern.
- [x] `beta=0.01` swept via the multi-seed harness (fixed-init) — **decisively REJECTED**: macro
      **0.5382 ± 0.0812** vs baseline 0.8514±0.0192, normal recall **0.5486 ± 0.1821**, and **all
      four** per-class F1s regressed beyond baseline mean−σ (not just `normal`).
- [x] Kill switch applied per §5.2: `beta=0.1` **deliberately NOT run** — β=0.01 already collapsed a
      class (two different ways across seeds), so escalating beta was explicitly avoided. User
      confirmed stopping here (2026-07-13); background orchestrator killed before it could
      auto-launch the β=0.1 phase.
- [ ] Cross-topology 14-bus check (§5.1) — **not run**, moot given the in-distribution collapse.
- [x] Outcome appended to `normal_recall_experiments.md` (lever-table format) — **REJECTED**, with
      full reasoning and a "if picking this up again" note. Also mirrored in
      `round3_findings_summary.md` §7.
- [x] Deployed Lever A untouched (verified: checkpoint/margin file mtimes unchanged from before this
      session; no experiment ever wrote to the deployed filenames).

**Verdict: GSAT REJECTED.** See `normal_recall_experiments.md` "Round 3 continued — GSAT" section for
the full data table, per-seed breakdown, and reasoning on why it collapsed (joins Levers C/D as a third
independent architecture-side negative result). GSAT code remains on disk, default-OFF, for future
revisiting with a new hypothesis — do not simply retune beta/r without one.

## 8. Quick reference — files & flags already in place
- `training/train_gnn.py`: `soft_f1_loss`, `f1_ramp_factor`; flags `--lambda_f1 --f1_warmup_frac
  --lambda_ramp_epochs --seed --init_seed --split_seed --ckpt_out`. Add GSAT flags here.
- `training/config.py`: `SMALL_CONFIG`/`CUDA_CONFIG`, `GRID_CONFIG` override. Add GSAT keys here.
- `training/calibrate_margin.py`: `--checkpoint --out`; emits `test_per_class_f1`.
- `training/run_round3_multiseed.py`: multi-seed driver + `noise_aware_verdict`. Extend `train_cmd`.
- `training/diagnose_seed_stability.py`: `--init_seed`/`--split_seed` 2-arm diagnostic (reuse if needed).
- `tests/test_soft_f1_loss.py` + `pytest.ini`: add GSAT unit tests (e.g. `info_bottleneck_kl ≥ 0`,
  `== 0` when `gate == r`; gate output in [0,1]).
- Artifacts dir `round3_runs/` is gitignored. Run with `PYTHONIOENCODING=utf-8`.
