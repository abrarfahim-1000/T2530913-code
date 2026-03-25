# Data Issues & Gaps

> Tracking known issues, expected anomalies, and action items for the dataset generation pipeline.
> Updated as new environments and runs are added.

---

## Issue 1 — Short Episode Lengths

**Severity:** Expected / Low  
**Environment:** `rte_case5_example` (toy), likely present in all environments  
**Status:** No fix needed — mitigation documented

**Observation:**  
Several chronics terminate well before their natural duration (e.g. 11–29 steps out of a possible ~70+). Per-chronic step counts are uneven across the 20 chronics.

**Cause:**  
Fault injection (`set_line_status` line trip) can drive the grid into an unrecoverable state, triggering `done=True` from Grid2Op before the chronic's natural end. This is correct Grid2Op behavior — the environment terminates when no valid power flow solution exists.

**Impact on training:**  
Per-chronic step counts are uneven. Splitting by chronic ID would produce unbalanced train/val/test sets.

**Mitigation:**  
Split by individual timestep record stratified by label — not by chronic ID. Use `sklearn.model_selection.train_test_split` with `stratify=labels` on the full flat record list.

---

## Issue 2 — No Cascade Labels

**Severity:** Expected / Low  
**Environment:** `rte_case5_example` only  
**Status:** Will resolve automatically on environment upgrade

**Observation:**  
The dataset contains 0 records with `label = "cascade"` despite cascade detection logic being present in the generation script.

**Cause:**  
`rte_case5_example` has only 5 substations and 8 lines. The grid is too small and sparse for a single line trip to propagate and overload adjacent lines within 1–3 timesteps. Cascading requires a denser, more interconnected topology.

**Impact on training:**  
The label set is effectively 3-class (`normal`, `overload`, `line_trip`) instead of the intended 5-class. Class weight computation must not reserve a weight slot for `cascade` or `maintenance` until those labels are present in the data — doing so will cause a zero-division or misleading weight assignment.

**Mitigation:**  
- For toy runs: train as 3-class, hardcode `n_classes=3` or derive dynamically from `set(labels)`
- For thesis runs on `l2rpn_neurips_2020_track1` (59 lines): cascade events will appear naturally — re-check label distribution after first generation run and update `n_classes` accordingly

---

## Issue 3 — Extreme rho Values

**Severity:** Expected / Informational  
**Environment:** `rte_case5_example`  
**Status:** No fix needed — behavior is intentional

**Observation:**  
`rho_max` reaches 7.962 in some overload records (7× thermal limit).

**Cause:**  
`Parameters.NO_OVERFLOW_DISCONNECTION = True` is set intentionally — this prevents Grid2Op from auto-disconnecting overloaded lines so the GNN can observe the full overload range rather than only states just above 1.0. Without this flag, lines would be disconnected at the first overflow and the overload label would rarely be observed.

**Impact on training:**  
`rho` features will have a long right tail. Apply per-feature normalization (e.g. `StandardScaler` or clip at a sensible max like 2.0) before feeding into the GNN to prevent extreme values from dominating attention weights.

**Action item:**  
Add a normalization step in `GridDataset.get()` — clip `rho` at 2.0 or apply min-max scaling using statistics computed on the training split only.

---

## Environment Upgrade Checklist

When moving from `rte_case5_example` → `l2rpn_neurips_2020_track1`:

- [ ] Re-run generation script (change `ENV_NAME` only)
- [ ] Re-check label distribution — expect cascade labels to appear
- [ ] Update `n_classes` in `GridGNN` constructor accordingly
- [ ] Recompute class weights from new distribution
- [ ] Re-check `rho_max` range and update normalization clip threshold if needed
- [ ] Verify episode lengths — full chronics should run longer before `done=True`

---

*Last updated: March 2026*
