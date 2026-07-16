# Component D Handoff — Issues Fixed, Re-Extraction Runbook, and Build Plan

**Audience:** a future Claude (Sonnet 5) session tasked with building Component D (the Symbolic
Shield) and the final cross-topology test harness. This doc is self-contained: it records the
blocking issues found during pre-implementation exploration (2026-07-16), the fixes already
applied, what the user runs manually before the build can start, and the full build plan.

**Read alongside:** `CLAUDE.md` (project overview, frozen artifacts),
`supplimentary_docs/study3(integration).md` (original shield design — parts of it are
superseded by this doc, noted below), `supplimentary_docs/shield_necessity_analysis_report.md`
(thesis framing).

---

## 1. Thesis framing (governs how results are reported)

The thesis question is open-ended: **does the GNN alone suffice on unseen topologies, or does it
need the symbolic shield?** The headline result is therefore the **GNN-only vs GNN+shield delta**
per topology — accuracy/safety of raw GNN outputs vs shield-gated outputs, plus false block rate
and missed-fault catch rate by failure mode. A large, well-characterized delta (in either
direction) is a complete result. No external benchmark is a binding target; Younesi et al.
(91.7% safe restoration, manual rules) may be cited as context only.

---

## 2. The three issues that were found (and why the v1 rules were scrapped)

### Issue 1 — v1 rules were largely not machine-evaluable
Of the 469 rules in the v1 ruleset (now archived at `rules/v1_archive/all_rules_deduped.jsonl`):
- only **~64** used exclusively Grid2Op-observable variables;
- **~386** referenced unobservables — `frequency_hz` (Grid2Op does not simulate frequency),
  `droop_pct`, `verification_months`, `power_factor`, `fault_clearance_time_ms`, ...;
- **~19** mixed observables with non-Python syntax (`BETWEEN a AND b`, `within 80ms`, `FOR`).

Root cause: the old `EXTRACT_PROMPT` gave an open-ended variable hint ("voltage_pu, loading_pct,
rho, p_mw, **etc.**") with no syntax constraint, and nothing in the pipeline linted conditions.
This is a documented negative result for the thesis (LLM extraction without a closed vocabulary
yields mostly non-operationalizable rules).

### Issue 2 — semantics inversion hazard
Some v1 conditions described the *compliant* state, not the violation — e.g.
`R_354 'voltage_pu >= 0.95 AND voltage_pu <= 1.05'` (a healthy band) and ride-through rules like
`R_370 'voltage_pu >= 0.9'`. A shield treating condition-true as violation would BLOCK every
healthy frame. The old prompts never specified condition polarity.

### Issue 3 — flat 150 kV nominal breaks voltage_pu everywhere (including in-distribution)
`study3(integration).md` §4.5 prescribes `voltage_pu = min(v_or) / 150.0`. Measured reality:
- `l2rpn_case14_sandbox` has lines at **~20 kV and ~138 kV** → flat 150 gives ~0.13 pu on healthy
  frames → 100% false block rate.
- Even `l2rpn_neurips_2020_track1_small` (the training grid) has **52 lines at ~150 kV and
  7 lines at ~365 kV** (line ids 45–47, 55–58) → those lines read ~2.4 pu → false overvoltage.
- Additionally, **tripped lines report `v_or = 0`**, so an unmasked `min(v_or)` falsely triggers
  every undervoltage rule on any frame with a disconnected line.

**The §4.5 `evaluate_condition` namespace in study3(integration).md is superseded** by the
voltage contract in §5 below.

---

## 3. Fixes already applied (this session, 2026-07-16)

| File | Change |
|---|---|
| `extraction/common.py` | `CONDITION_VOCABULARY` (6 variables, single source of truth); rewritten `EXTRACT_PROMPT` (closed vocabulary, Python-only syntax, violation polarity with worked example, skip-instruction for inexpressible constraints); rewritten `VALIDATE_PROMPT` (checks 4: vocabulary/syntax, 5: polarity — corrects inverted conditions); `lint_condition()` AST linter + smoke-eval; `Rule.check_condition` field validator wired to it |
| `extraction/validate.py` | Startup guard: warns + prompts if stale `*_confirmed.jsonl` files (older than newest candidates) would be merged into the dedup |
| `tests/test_condition_lint.py` | 27 tests covering accept/reject cases, `Rule`-schema integration, prompt rendering — all passing |
| `scripts/dump_base_kv.py` | New: writes per-line base kV sidecar per env tag (backend method on the research PC, `--empirical` from JSONL normal-frame medians anywhere) |
| `data/grid_dataset_{neurips2020,case14}_basekv.json` | Generated (empirical method) |
| `rules/v1_archive/` | All v1 extraction outputs moved here (out of the dedup glob's reach); `rules/` is clean for the re-run |

### The condition vocabulary (fixed, enforced by prompt AND linter)

| Variable | Meaning | Derivation from a record/obs |
|---|---|---|
| `voltage_pu_min` | lowest per-unit voltage across energized lines | `min(v_or[i]/base_kv_or[i] for i where line_status[i])` |
| `voltage_pu_max` | highest per-unit voltage across energized lines | same, `max` |
| `loading_pct` | max line loading, % of thermal limit | `max(rho) * 100` |
| `rho_max` | max line loading ratio | `max(rho)` |
| `n_tripped_lines` | count of disconnected lines | `sum(not s for s in line_status)` |
| `any_line_tripped` | at least one line disconnected | `not all(line_status)` |

Any rule whose condition parses to anything else is dropped by the Pydantic schema at extraction
time (`lint_condition` in `extraction/common.py` — AST whitelist + sandboxed smoke-eval; result
must be `bool`).

---

## 4. Manual steps before the build (user, research PC)

1. **Re-run extraction** (Ollama models live there; sequential, never both loaded):
   ```bash
   python extraction/extract.py --docs data/documents/ --out rules/
   python extraction/validate.py --candidates rules/
   ```
   Expected artifacts: fresh `rules/*_candidates.jsonl`, `*_confirmed.jsonl`, `*_flagged.jsonl`,
   `rules/all_rules_deduped.jsonl`, run summaries. Expect a much smaller but ~100% evaluable
   ruleset (the linter + skip-instruction cut everything inexpressible). Keep `rules/v1_archive/`
   untouched — if the stale-output guard fires, something from v1 leaked back into `rules/`.
2. **Rebuild the KG** from the new ruleset (overwrites `kg/knowledge_graph.pkl`, which is still
   the v1-based graph until this runs):
   ```bash
   python extraction/build_kg.py --rules rules/all_rules_deduped.jsonl --out-dir kg/
   ```
3. **Generate the 118-bus test set** (whenever convenient, before final testing):
   ```bash
   python scripts/generate_dataset.py --env wcci
   python scripts/dump_base_kv.py --tag wcci2022        # backend method works there
   ```
   Optionally re-run `dump_base_kv.py` for neurips2020/case14 without `--empirical` on the
   research PC to replace the empirical sidecars with exact backend values (methods should agree
   to within ~2%).

**Build precondition checklist for the future session:** new `rules/all_rules_deduped.jsonl`
exists and every condition passes `lint_condition`; `kg/knowledge_graph.pkl` rebuilt after it;
`data/grid_dataset_<tag>_basekv.json` exists for every tag being evaluated.

---

## 5. The voltage contract (binding for the shield implementation)

```python
sidecar = json.load(open(f"data/grid_dataset_{tag}_basekv.json"))
base_kv = np.array(sidecar["base_kv_or"])          # shape (n_line,)

mask = np.array(record["line_status"], dtype=bool)  # energized lines ONLY
v_pu = np.array(record["v_or"])[mask] / base_kv[mask]
voltage_pu_min = float(v_pu.min())
voltage_pu_max = float(v_pu.max())
```
Edge case: if `mask.sum() == 0` (total blackout frame), voltage_pu_min/max are undefined —
treat voltage-based rules as NOT_EVALUABLE for that frame, never as violated-by-default.

Sidecar format: `{"env_name", "method": "backend"|"empirical", "n_line", "base_kv_or": [...],
"base_kv_ex": [...]}`.

---

## 6. Component D build plan (the shield)

New top-level `shield/` package. The GNN (`gnn_checkpoint_best.pt`), Lever A margin
(`gnn_logit_margin.json`), `normalization_stats.pt` (repo root), and the rebuilt KG are inputs —
none of them change. Interaction is strictly post-hoc: GNN never sees the KG; shield only sees
the prediction + raw telemetry. No bypass path.

- **`shield/context.py`** — `build_context(record: dict, meta, base_kv) -> dict`. Produces
  exactly the six vocabulary variables (per §5 voltage contract) plus `fault_type` (predicted
  label) and `confidence`. Works identically for a JSONL record or a live Grid2Op obs (key names
  match: `rho`, `v_or`, `line_status`). Raw values only — never the z-scored features.
- **`shield/evaluator.py`** — `evaluate_condition(condition, context) -> Verdict` with verdict
  taxonomy **VIOLATED / SATISFIED / NOT_EVALUABLE / ERROR**. Sandboxed
  `eval(cond, {"__builtins__": {}}, namespace)`. `NameError`/undefined-context → NOT_EVALUABLE;
  any other exception → ERROR. **Neither NOT_EVALUABLE nor ERROR ever blocks** — only VIOLATED
  does. (Since re-extracted rules are pre-linted, NOT_EVALUABLE should be ≈0; a nonzero count is
  a regression signal.) No polarity handling needed — polarity is enforced at extraction.
- **`shield/shield.py`** — `validate(context, KG) -> ShieldResult`. Rule retrieval reuses
  `extraction/build_kg.py::get_rules_for_entity` with a fault_type→entity map:
  `overload | line_trip | cascade` → Line + Grid (+ Load if present in the new ruleset);
  `normal` → Grid. Dedupe by rule_id; evaluate all; sort violations by severity
  (critical > high > medium > low); return
  `{"status": "PASS"|"BLOCK", "fault_type", "confidence", "violated_rules",
  "highest_severity", "explanation", "n_evaluated", "n_not_evaluable"}`.
  `build_explanation` cites rule_id, severity, source, explanation per violated rule.
  The localization head is **not** used (disabled cross-topology per CLAUDE.md) — the shield
  gates the classification only.
- **`tests/test_shield.py`** — build test-first: context builder (tripped-line masking, per-line
  base-kV division, blackout edge case), evaluator (verdict taxonomy, sandbox, never-block-on-
  failure), `validate()` on synthetic contexts (clean PASS; single-overload BLOCK; multi-rule
  BLOCK with severity ordering — the three scenarios of study3 §4.10, as fixtures rather than
  the doc's `rte_case5_example`, which predates this grid2op version), and a real-KG smoke test
  (load rebuilt `kg/knowledge_graph.pkl`, one healthy + one overloaded context → PASS/BLOCK).
  All of this runs on the personal PC (no GPU needed).

---

## 7. Test-harness build plan (final testing phase)

### `evaluation/eval_shield.py`
Extends `evaluation/eval_cross_topology.py`'s setup pattern — reuse its checkpoint loading,
36-bus train-split normalization (`compute_normalization_stats`), foreign-meta/`GridDataset`
loading, and dual-arm reporting. Per record:
1. GNN forward (z-scored features) → logits → prediction arms: **raw argmax** (all tags) and
   **Lever A margin** (`--tag neurips2020` only — the margin is val-tuned in-distribution and
   does not transfer; CLAUDE.md).
2. Build context from the **raw** record (§5/§6) → `shield.validate()`.
3. Log every step to `results/shield_eval_<tag>.jsonl`: true label, predicted label (per arm),
   confidence, shield status, violated rule ids, highest severity, n_not_evaluable.
4. Every BLOCK also goes to `results/failures_<tag>.jsonl` with the failure-mode classification
   (CLAUDE.md Component D table): **overconfident-wrong** (pred normal, conf > 0.85,
   true ≠ normal), **class-confusion** (fault detected, wrong type), **threshold-failure**
   (right type, physical threshold violated), **novel-topology-state** (residual).

Evaluation matrix (one command each): `neurips2020` (held-out test split via
`data/split_neurips2020_test_idx.npy` — in-distribution baseline, both arms), `case14`,
`wcci2022`. Deterministic: model in eval mode, no sampling — reruns must be bit-identical.

### Metrics (aggregated per topology, GNN-only arm vs GNN+shield arm)
- **GNN degradation:** macro F1 vs the 0.8277 36-bus calibrated baseline.
- **False block rate:** % of ground-truth `normal` frames blocked (shield over-constraint).
- **Missed-fault catch rate:** of frames where the GNN prediction is wrong, % blocked by the
  shield — split by the four failure modes. This is the necessity evidence.
- **Rule compliance rate:** % of PASS states satisfying all evaluable rules on post-hoc re-check
  (must be 100% by construction; less means evaluator bugs).
- **Shield applicability:** % of ruleset evaluating without NOT_EVALUABLE per topology
  (expected ≈100% with the linted ruleset).
- **GNN-only vs GNN+shield safety delta:** rate of rule-violating outputs shipped by each arm —
  the thesis headline (§1).

### `evaluation/summarize_shield_results.py`
Reads the three `shield_eval_*.jsonl`, emits thesis tables (metric grid rows=topology,
failure-mode distribution, per-severity block counts) as markdown →
`supplimentary_docs/shield_final_results.md` (companion to `gnn_final_results.md`). Also dumps
N=30 random BLOCK explanations per topology to a review sheet for the manual 1–5
explanation-quality rating.

### Verification gates (before trusting any numbers)
1. `pytest tests/` green (linter + shield suites).
2. case14 sanity: false block rate on ground-truth normal frames must be far below 100%
   (catches base-kV regressions); zero ERROR verdicts; rule compliance = 100%.
3. neurips2020 in-distribution: classification columns reproduce the known macro F1 (0.7830 raw
   / 0.8277 Lever A) — the shield must not perturb the underlying predictions.
4. Rerun any tag twice → identical aggregates.

### Build order
`shield/` + `tests/test_shield.py` (test-first) → `eval_shield.py` → smoke-run case14 +
neurips2020 → full matrix once `wcci2022` data exists → `summarize_shield_results.py` →
results doc → update `CLAUDE.md` Current Status.

---

## 8. Deferred / out of scope for the build session

- Retraining or recalibrating the GNN (Component A closed — see `gnn_final_results.md`).
- Re-running extraction or KG build (user does this per §4 before the build starts).
- Localization-head evaluation (disabled cross-topology).
