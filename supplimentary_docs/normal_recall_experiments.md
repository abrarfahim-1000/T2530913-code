# Normal-class recall experiments — handoff

**Goal:** raise `normal` recall (was ~0.48–0.50, high precision but under-committed) WITHOUT
regressing `overload` / `line_trip` / `cascade`. Methodology: try levers one at a time,
**keep-what-works** — adopt a change only if `normal` improves AND the other three hold; stack
the next on the retained state; revert anything that regresses.

Baseline before any of this (committed `main`): macro F1 ≈ 0.7701, `normal` recall ≈ 0.48.

---

## Results so far (all on personal-PC XPU smoke runs, 10 epochs)

| # | Change | Best macro F1 | normal recall | other 3 F1 | Verdict |
|---|--------|---------------|---------------|------------|---------|
| **Exp 1** | `near_limit` edge flag threshold **0.9 → 1.0** (`scripts/pyg_data.py`) | **0.7950** | 0.48 → **0.54** | 0.84 / 0.76 / 0.86 (intact) | ✅ **KEPT** |
| Exp 2 | `label_smoothing` 0.1 → 0.05 (`train_gnn.py`) | 0.7774 | ↓ 0.42 | slightly down | ❌ reverted |
| Exp 3b | +6th node feature, binary `overloaded=(node_rho≥1.0)` | 0.4985 | **0.00 collapsed** | broken | ❌ reverted* |
| Exp 3a | focal loss (γ=2) | not run | — | — | ⏳ pending |

\* **3b crash, updated understanding:** scaling the model up to `[64,128,128]` to "fix" the
6-feature crash was a dead end — the big model **overfits and collapses** under this schedule
(train loss falls while val F1 falls; `normal` AND `cascade` go to 0.0; best epoch-1 macro only
~0.36). Big model abandoned. The proven `[16,32,32]` config (macro 0.795) is the best to date and
is what we work from. The 6-feature crash mechanism on the small model is **unconfirmed** — leading
suspicion is z-scoring the sparse binary flag (~3% ones) produces a dominant spike that
destabilizes training. 3b is **optional**: Exp 1 already encodes the overload boundary at the edge
level, so a node-level duplicate may be redundant. Retry once with the no-normalize tweak (below);
if it still crashes, drop it.

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

## TODO tomorrow (in order)

All commands assume: `source .venv/Scripts/activate` and prefix runs with `PYTHONIOENCODING=utf-8`.

**Reference baseline = Exp 1 small model, macro F1 0.795, `normal` recall ~0.54.** No new baseline
run needed — the big-model detour is abandoned and the config is already back to the proven setup.

### 1. Exp 3a — focal loss (no reprocess; stacks on the 0.795 state) — MAIN remaining lever
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

### 2. Exp 3b — 6th node feature `overloaded` (OPTIONAL; reprocess needed)
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
