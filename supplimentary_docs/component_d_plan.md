# Neuro-Symbolic Grid Project — Current Plan

**Supersedes `component_d_handoff_archive.md`** (retired 2026-08-15, kept for the v1/v2
negative-results record only). Everything still operative is in this file.

**Read alongside:** `CLAUDE.md` (project overview), `gnn_final_results.md` (Component A
negative results — note §2 below reopens Component A).

---

## 1. Status and sequencing

| Component | State |
|---|---|
| **A — GNN** | **REOPENED** (§2). Classify task closed; forecast task implemented, not yet trained. |
| **B — Extraction** | Stage 1 done: **2,463 candidates / 16 documents** in `rules_35b/`. Stages 2–3 not run. |
| **C — KG** | To be **redesigned from scratch** against the rules that survive. Not started. |
| **D — Shield** | Built and tested. Needs the binary/asymmetric update (§6) once the task lands. |

| When | Where | What |
|---|---|---|
| **Next** | personal PC | `generate_dataset.py --task forecast` → check class balance (§2.4) → `preprocess.py` → train |
| Then | research PC | `translate.py`, then `validate.py` |
| Then | personal PC | polarity guard on the translated corpus → KG redesign → shield binary update |
| Finally | personal PC | `eval_shield.py` matrix + the §7 controls → `summarize_shield_results.py` |

### 1.1 THE TASK DECISION — settled 2026-08-16, by measurement

**Decision: Component A becomes N-1 contingency screening (`--task n1`).** The forecast task was
built, probed, and rejected before generating against it. Supervisor was unavailable and delegated
the call; the evidence chain below is the record, and is itself a thesis result.

**What was measured, in order.**

1. *Any-fault forecast is unlearnable.* Line-trip onset is an unconditional coin flip —
   `np.random.rand() < FAULT_PROB` (0.05) on a uniformly random connected line — and trips are
   71–88% of all `at_risk` events. On 1,995 quota-unbiased chronics / 321,966 reconstructed steps of
   the 36-bus training set, the best possible threshold on the current `rho_max` scores **1.00× the
   all-positive baseline at every H ≥ 3**. No rule, and therefore no model, beats "always say
   at_risk".
2. *Overload-only forecast is learnable but exhausted by one rule.* Restricting the target to
   overload gave a 2.0–9.0× threshold lift, which looked promising — but a gradient-boosting model
   on the full feature vector reached **F1 0.163 vs the single threshold's 0.160 (1.02×)**, with
   *worse* average precision (0.125 vs 0.127). Adding trend features over a contiguous 60k-frame
   pilot did not change this on either topology (lift 0.67–1.02× across H ∈ {2…18}). A snapshot
   carries no information about future exogenous load, so there is nothing for a GNN to add.
3. *N-1 screening is neither.* Per line k: "if k trips now, does a thermal limit go?" Measured on
   the 36-bus grid:

   | | F1 |
   |---|---|
   | all-positive baseline | 0.311 |
   | best single rule (flow on the removed line) | **0.608** |
   | model with network context | **0.868** (AP 0.937 vs 0.628) |

   Global `rho_max` scores 1.04× the baseline — the current state does *not* determine the answer —
   and **100% of sampled frames are mixed**: some contingencies violate, others do not. The label
   depends on how flow redistributes through the remaining network, i.e. on topology. Cost is
   ~170 power-flow solves/sec, so a 1M-label set is roughly 110 minutes.

**Why this un-rigs the comparison permanently.** The classify target failed because four rules
reproduced it exactly. N-1 fails the same test in the opposite direction: **no rule over the present
observation can restate the label, because producing it requires a power-flow solve.** The symbolic
layer can no longer trivially win, and the 0.608 → 0.868 gap is where the GNN's contribution lives.

**What survives unchanged:** Component B extraction (thermal/voltage limit rules are *more* on-point
for post-contingency screening, not less), the shield machinery in `shield/` behind `RuleProvider`,
and the Component C redesign plan. What changes is Component A's head (global pooling → edge-level)
and the shield's unit of validation (a frame → a contingency).

**Retained for the write-up.** `--task forecast` and `--risk-source` stay in the code with their
per-class distances, so §1.1 steps 1–2 are reproducible. Three probed-and-rejected task designs is a
methodology contribution, not wasted work.

---

#### Superseded — the forecast-horizon analysis that led here

> `H` was the *second* question; the first was **which future faults set `at_risk`**, because
> line-trip onset in `generate_dataset.py` is an unconditional coin flip —
> `np.random.rand() < FAULT_PROB` (0.05) on a uniformly random connected
> line — and therefore **cannot be predicted from any observation, by construction**.
>
> Probe on the **36-bus training** set, 1,995 quota-unbiased chronics / 321,966 reconstructed steps,
> scoring the best possible single threshold on the current `rho_max` against the all-positive
> baseline F1 = 2p/(1+p), on currently-calm frames only:
>
> | H | min | ANY-fault base | ANY lift | OVERLOAD base | OVERLOAD lift |
> |---:|---:|---:|---:|---:|---:|
> | 1 | 5 | 5.1% | 1.12× | 0.7% | **9.03×** |
> | 2 | 10 | 9.9% | 1.02× | 1.4% | **7.02×** |
> | 3 | 15 | 14.8% | 1.00× | 2.3% | 3.88× |
> | 4 | 20 | 18.8% | 1.00× | 3.0% | 3.11× |
> | **6** | **30** | 27.0% | **1.00×** | **3.6%** | **2.76×** |
> | 8 | 40 | 34.3% | 1.00× | 4.2% | 2.48× |
> | 12 | 60 | 46.3% | 1.00× | 4.9% | 2.00× |
> | 18 | 90 | 59.9% | 1.00× | 5.6% | 1.96× |
> | 24 | 120 | 69.5% | 1.00× | 6.3% | 1.59× |
>
> **The ANY-fault target has lift 1.00× at every H ≥ 3** — no rule, and by extension no model, beats
> "always say at_risk". That is not the triviality failure (§2.1); it is the opposite one, an
> *unlearnable* target, and it is equally fatal. The overload target carries real, physically
> sourced signal (rho rises out of the load/generation trajectory) that decays smoothly with `H`.
>
> Acted on already: `--risk-source {overload,overload_cascade,any}` now **defaults to `overload`**,
> and every record carries `steps_to_overload / steps_to_trip / steps_to_cascade / steps_to_fault`,
> so **both the source and the horizon are now re-derivable by relabelling** — verified end-to-end
> on a smoke run. The *only* thing still fixed at generation time is binary-vs-four-classes.

> The 2.76× lift at `H = 6` looked like enough headroom, and `risk_source = overload, H = 6` was
> the standing recommendation for about an hour — until the ceiling analysis showed a full model
> could not beat the single threshold it was measured against (step 2 above). **A lift over the
> all-positive baseline says the target is not vacuous; it says nothing about whether a model can
> beat the rule.** Both checks are needed. That is the transferable lesson from this episode.

**2. The closed-form finding, and that it reopened Component A.** Still worth telling the supervisor
when they surface, and it is the first link in the chain above. The substance:

> The 4-class label is a closed-form function of the observation. Four threshold rules on
> `rho_max` and `n_tripped_lines` reproduce it with **100% agreement on 55,000 records**, across
> both topologies. A symbolic layer therefore scores 100% where the trained GATv2 reaches 0.8277 —
> so "does the shield add value over the GNN" was rigged before it ran. Component A is reopened to
> replace the target with a genuinely predictive one.

Why it needs a conversation rather than a note:

- It **reopens a component that was declared closed**, and the three rounds of rejected
  architecture experiments in `gnn_final_results.md` now describe the *old* task.
- It changes what the thesis claims. The defensible framing is a **diagnosis and a fix** — "we
  built the classifier, proved the target was closed-form with a 4-rule baseline, and redesigned
  the task" — which is a stronger arc than either half alone, but it is a change of story.
- If the supervisor would resist a partly negative-results framing, that is far cheaper to learn
  now than after the eval matrix runs.

**Hardware split changed.** Supersedes `CLAUDE.md`'s "research PC — all production runs":

| Machine | Runs |
|---|---|
| **Personal PC** (Arc B580 12 GB, 16 GB RAM) | dataset generation, GNN training, guard, shield, tests |
| **Research PC** (RTX 4080 Super, 64 GB) | LLM stages only — `translate.py`, `validate.py` |

`training/config.py` auto-selects the small `[16,32,32]` / `heads=[4,4,1]` branch on non-CUDA
devices — the **proven deployed** config. The CUDA branch `[64,128,128]` has never produced a
working checkpoint, so training on the personal PC is the safer choice, not a compromise.

---

## 2. TASK REDESIGN — why Component A reopened

### 2.1 The 4-class target is closed-form

`get_state_label()` defines the labels as:

```python
if max_rho >= 1.0:                  return "overload"
if active_lines == env.n_line:      return "normal"
if active_lines == env.n_line - 1:  return "line_trip"
return "cascade"
```

Every branch is expressible in the shield's own vocabulary (`rho_max`, `n_tripped_lines`).
**Measured: four ordered rules reproduce the stored labels with 100% agreement on 55,000
records** — 15,000/15,000 case14, 40,000/40,000 neurips2020.

Consequences:

1. A symbolic layer scores **100%** where the trained GATv2 reaches **0.8277**. "Does the shield
   add value over the GNN" is rigged before it runs.
2. The GNN receives `max_rho` and `global_trip_frac` as node features, so after max-pooling the
   answer is directly available. The documented `normal`/`line_trip` confusion is exactly the
   `n_tripped_lines == 0` vs `== 1` distinction that triple pooling averages away — three rounds
   of architecture experiments were fighting to learn a threshold already in the input.
3. `fault_loc` is `obs.rho.argmax()` — the localization target is closed-form too.

Not a bug; a property of the labelling scheme, and worth reporting **as a finding**: *the
fault-labelling scheme commonly used on Grid2Op-derived datasets yields a target that is a
closed-form function of the observation, and therefore cannot distinguish a learned model from a
threshold.*

### 2.2 The fix: a binary forecast task

> **`at_risk`** = does the grid leave the `normal` state within the next `H` steps?

Faults are injected stochastically (`FAULT_PROB = 0.05`) and load evolves, so this is genuinely
predictive. It also **un-rigs the shield**: rules see only the present, the label depends on the
future, so no rule can restate the labeller.

**Both tasks are kept and reported** — classify carries the closed-form finding, forecast is the
corrected experiment. The narrative is a diagnosis and a fix, not just a system.

### 2.3 What was implemented (2026-08-15)

`scripts/generate_dataset.py --task forecast --horizon H`

| Change | Detail |
|---|---|
| **No subsampling** | `NORMAL_KEEP_PROB` / quotas bypassed. A dropped frame is a hole in an earlier frame's lookahead window. Balance handled by class weights at training. |
| **Per-chronic buffering** | Frame `t`'s label depends on `t+1..t+H`, so a chronic is buffered and labelled on completion (`flush_forecast_chronic`). |
| **Trailing frames dropped** | The last `H` frames have an undefined window. Labelling them `not_at_risk` would mark the end of every chronic as safe. |
| **Separate tag** | Writes `grid_dataset_<env>_forecast.jsonl` — never overwrites the classify set. |
| **Extra fields** | `risk_label`, `risk_int`, `horizon`, `steps_to_fault`. `label`/`label_int` retained for reference, NOT the target. |
| **Meta override** | `label_map` → `RISK_LABEL_MAP`, `n_classes` → 2. `train_gnn.py` reads both from the meta, so training switches with no flag. |

Training side: `training/config.py` gains `TASK` (env var `GRID_TASK`) and per-task
`DATA_FILE` / `CHECKPOINT_FILE` / `NORM_STATS_FILE` / `PROCESSED_PT`. `train_gnn.py` **disables
the localization loss when `n_classes == 2`** — `fault_loc` is a present-state signal that would
leak into a future-state objective. `preprocess.py` / `pyg_data.py` use `risk_int` when present.

Architecture is **unchanged**. Only the target changes.

`steps_to_fault` is on every frame, so **the horizon can be retuned by relabelling — no
regeneration needed.**

⚠️ **The old data cannot be relabelled.** Measured contiguity is 61.6% (case14) / 88.0%
(neurips2020) — `NORMAL_KEEP_PROB = 0.02` punched holes throughout (first chronic timesteps:
`[5, 6, 9, 10, 11, 20, 33, ...]`). Relabelling intact stretches only would bias toward dense,
fault-heavy regions.

Tests: `tests/test_forecast_labels.py` — 12 tests including a brute-force cross-check at four
horizons. This code runs unattended for hours; an off-by-one is expensive.

### 2.4 Risks to watch at generation time

1. **Class balance.** Without subsampling, and with faults persisting until reconnect
   (`RECONNECT_PROB = 0.20`), `at_risk` may dominate. Check immediately after generation.
2. **Horizon too short → triviality returns.** Small `H` makes `at_risk` correlate with the
   *current* state. Early warning: the shield-alone baseline (§7.1) scoring near-perfectly.
3. **Dataset size.** No subsampling means far more frames per chronic. Cap with `--n_records` /
   `--max-chronics`; 16 GB RAM and `PreloadedGridDataset` loads the whole `.pt` into memory.

---

## 3. Extraction pipeline (Component B)

Four stages. The closed vocabulary is enforced at **stages 2–3 only**.

| Stage | Script | Job |
|---|---|---|
| 1 | `extract.py` | open vocabulary, syntax lint only → `*_candidates.jsonl` |
| 2 | `translate.py` | map engineering variables onto `CONDITION_VOCABULARY` → `*_translated.jsonl` + `*_untranslatable.jsonl` |
| 2.5 | `polarity_guard.py` | fire-rate filter → `guarded/*_translated.jsonl` + `*_polarity_rejected.jsonl` |
| 3 | `validate.py` | verify against source, correct roles → `*_confirmed.jsonl`, then dedup |

Keeping stage 1 open is load-bearing: it makes `*_untranslatable.jsonl` a meaningful audit trail
(and the thesis negative result — "most grid-code content is not Grid2Op-observable") rather than
an empty file.

### 3.1 Full-corpus measurements (all 16 documents)

| Measure | Value |
|---|---|
| Stage-1 candidates | **2,463** (v1: 1,372) |
| All variables derivable from a record | **693 (28.1%)** |
| Mixed (some derivable, some not) | 571 (23.2%) |
| Distinct variable names | 2,083 |

Blocked families (overlapping): **FREQ 625 (25.4%)**, **TIME 582 (23.6%)**, **OTHER 610
(24.8%)**, **PROT 204 (8.3%)**. Frequency is irreducible — Grid2Op targets human-timescale
control and quasi-static power flow assumes uniform system frequency.

Entity skew, relevant to the KG redesign: `Line` 59 and `Bus` 28 out of 2,463 — **3.5%**.
Everything else is module-, facility-, or system-level, so entity-based retrieval carries almost
no information. Do not assume it does.

**Projection: ~200–250 final rules**, roughly balanced constraints/affirmations. Still a
projection — `*_untranslatable.jsonl` and `polarity_guard_summary.json` replace it with
measurement.

### 3.2 Rules carry two ROLES

The shield cannot read the corpus at inference, so the KG must carry both directions of evidence:

- **CONSTRAINT** — condition TRUE ⇒ the standard is violated ⇒ BLOCK.
- **AFFIRMATION** — condition TRUE ⇒ telemetry is consistent with the class in `affirms`.
  Supporting evidence.

Both prompts previously *ordered* the model to invert any "required or normal operating range"
into a violation predicate. That instruction was itself generating the inverted-band defect:
"voltage shall remain within 0.95–1.05" is an affirmation of `normal`, and inverting it yields a
predicate that fires at nominal.

`Rule` has `role` (default CONSTRAINT) and `affirms`; an AFFIRMATION without a class is rejected,
a CONSTRAINT with one is rejected.

### 3.3 The condition vocabulary (14 variables)

Enforced by prompt AND by the `lint_condition` AST linter. Anything else is dropped at schema
time.

| Group | Variables |
|---|---|
| voltage / loading / topology | `voltage_pu_min`, `voltage_pu_max`, `loading_pct`, `rho_max`, `n_tripped_lines`, `any_line_tripped` |
| power flow | `active_power_mw_max`, `reactive_power_mvar_max`, `apparent_power_mva_max`, `power_factor_at_max_load`, `current_a_max` |
| system balance | `total_generation_mw`, `total_load_mw`, `generation_load_imbalance_pct` |

⚠️ **`power_factor_min` was a derivation bug**, caught by the guard on real data. As a minimum
across energized lines it has median **0.000** on healthy case14 frames — lightly loaded lines
carry near-pure reactive flow, so the minimum is dominated by electrically irrelevant lines, and
`pf < 0.9` fired on 100% of healthy frames. Replaced with `power_factor_at_max_load` (median
0.930, p5–p95 0.926–0.936). Evidence that the guard catches *context* defects, not only rule
defects.

### 3.4 Translation cases the prompt handles

Synonym; unit conversion; aggregation binding (under-limit → `min`, over-limit → `max`); proxy
substitution (rating-relative → `rho`); **scope vs predicate split** (an applicability filter like
"modules above 50 MW" is not part of the condition); role assignment.

**Compound conditions are DISCARDED, not trimmed.** `voltage_pu_min < 0.9 and time_seconds > 3`
must not become `voltage_pu_min < 0.9` — the standard tolerates a brief dip, so the trimmed rule
would flag states the standard permits while still citing that standard. The reason field names
the unmeasurable variable.

---

## 4. The polarity guard (stage 2.5)

A violation predicate must be FALSE on a healthy grid; an affirmation must be TRUE on frames of
the class it affirms. The guard measures this deterministically — no LLM, no GPU.

**Fire rate over real frames, never a synthetic band.** Settled by data:

- `voltage_pu_min < 0 or voltage_pu_max > 1` (real rule, severity critical) evaluates **False** at
  a synthetic 1.0 pu point — a band-based filter ships it. Against real healthy case14 frames it
  fires at **0.714** (`voltage_pu_max` exceeds 1.00 on 71.8% of them).
- `loading_pct > 80` is legitimate yet fires on **14.1%** of healthy case14 frames (that grid runs
  hot — normal-frame `rho_max` median 0.72, max 0.99). A cutoff near zero would cull it.

Role-aware scoring: CONSTRAINTs on `normal` frames (reject above `--max-fire-rate`, default 0.5);
AFFIRMATIONs on frames of their own class (reject below `--min-support`, default 0.5). Worst-case
across topologies, not mean.

⚠️ The observed distribution is **not cleanly bimodal** — the two rules above land at 0.71 and
0.15. The guard warns and lists everything between 0.1 and 0.9; inspect that before trusting the
cutoff.

Outputs carry per-rule `fire_rate_<tag>` metadata — this *is* the rule's false block rate, so it
predicts the shield's overall false block rate before the shield runs. Rejects go to
`*_polarity_rejected.jsonl` (audit trail, enables the filtered-vs-unfiltered counterfactual).

---

## 5. The voltage contract (BINDING)

```python
sidecar = json.load(open(f"data/grid_dataset_{tag}_basekv.json"))
base_kv = np.array(sidecar["base_kv_or"])            # shape (n_line,)

mask = np.array(record["line_status"], dtype=bool)   # energized lines ONLY
v_pu = np.array(record["v_or"])[mask] / base_kv[mask]
voltage_pu_min, voltage_pu_max = float(v_pu.min()), float(v_pu.max())
```

Two invariants, both measured, both non-negotiable:

1. **Per-line base kV, never a flat nominal.** Backend nominals: case14 runs lines at 14 / 20 /
   138 kV, and the 36-bus training grid at 138 kV with **7 lines at 345 kV** (ids 45–47, 55–58).
   A flat 150 divisor reads the 345 kV lines as 2.4 pu and case14's 20 kV lines as 0.13 pu →
   ~100% false block rate. (Earlier revisions of this section — and `CLAUDE.md` — quote ~150 kV
   and ~365 kV. Those are the grids' *operating* voltages, not their base kV. See §5.1.)
2. **Energized lines only.** Tripped lines report `v_or = 0`, so an unmasked `min` fires every
   undervoltage rule on any frame with a disconnection.

Blackout (`mask.sum() == 0`): voltages are **omitted from the context**, so voltage rules raise
NameError → NOT_EVALUABLE → never block. Defaulting to 0.0 would read as catastrophic undervoltage.

### 5.1 Base kV is the BACKEND nominal, not the empirical median — settled 2026-08-16

This section previously carried a TODO: the case14/neurips2020 sidecars were built with
`--empirical` (per-line medians over normal frames of the same dataset), with an instruction to
re-run the backend method and re-check the guard if the two disagreed by more than ~2%. All three
environments are now installed on the personal PC, so this was measured rather than deferred.

**They disagree far past that tolerance.**

| tag | backend nominal (kV) | empirical median (kV) | median disagreement | lines > 2% |
|---|---|---|---|---|
| neurips2020 | 138, 345 | 145.9–149.0, 361.2–369.6 | 6.63% (max 7.39%) | **59 / 59** |
| case14 | 14, 20, 138 | 14.9, 21.2–22.0, 139.5–142.1 | 4.23% (max 9.09%) | **16 / 20** |

The disagreement is near-uniform across both voltage levels — 147/138 = 1.066, 365/345 = 1.058 —
which is the signature of generator voltage setpoints fixed at ~1.05–1.06 pu, the IEEE test-case
convention these Grid2Op environments inherit. **It is not a defect in the grids**, and it is not
noise in the empirical estimate.

**Decision: the backend nominal is the contract.** `data/grid_dataset_<tag>_basekv.json` now holds
backend values for all three tags. The empirical originals are preserved as
`data/grid_dataset_<tag>_basekv_empirical.json` for the counterfactual, and must not be deleted.

Consequence, stated plainly so nobody rediscovers it as a bug: healthy frames read **~1.06 pu, not
~1.00 pu**, so a standards band like `voltage_pu_max > 1.05` fires on essentially every healthy
frame, and the guard (§4) will reject it at a fire rate near 1.0.

Why that is the right trade:

1. **The empirical base is centred by construction.** It sets the healthy median to 1.00 pu using
   the same data the rules are then tested against. This project's spine is the discovery that the
   classify target was rigged before it ran (§2.1); shipping a second, subtler self-calibration
   inside the shield's own context is the first thing an examiner would go for.
   ⚠️ Note the empirical base is **not degenerate** — it fixes the *centre*, not the *spread*, and
   §4's measured fire rates of 0.714 and 0.141 came from it. The objection is provenance, not lack
   of signal. Do not restate this as "the empirical sidecar makes every voltage rule pass."
2. **The cost is small, because the N-1 label is thermal.** `label_n1` sets
   `violation[k] = int(sim_done or rho >= 1.0)`. The rules that can bear on that target are the
   `rho` / `loading_pct` ones — dimensionless, and unaffected by the base entirely. Voltage rules
   reach the label only through the weak indirect channel of voltage stress correlating with
   thermal stress. This choice moves corpus composition and false block rate; it barely touches the
   headline comparison.
3. **A rejected voltage rule is a measured finding, not a gap.** Rejects are logged to
   `*_polarity_rejected.jsonl` — which is exactly §7.3's distinction between a quantified
   limitation and an unmeasured hole.

The reportable form:

> Grid-code voltage bands are calibrated to utility operating practice, not to IEEE test-case
> setpoint conventions, so they do not transfer to Grid2Op reference grids without rebasing.

Same class of result as §3.1's blocked families (FREQ 25.4%, TIME 23.6%): the standards corpus and
the simulator do not share a frame of reference, and this quantifies where and by how much.

**When the corpus lands, run the guard under BOTH sidecars** and report both fire-rate tables —
backend as the deployed contract, empirical as the sensitivity analysis.

⚠️ **Revisit if voltage rules turn out to be a large share of what survives translation.** The
argument above rests on them being a minority. If they are not, backend kV is thinning the corpus
enough to matter, which pushes toward the §7 failure mode of reporting (b) as (a). That share is
measurable the moment stage 2 finishes — **check it before committing this choice in writing.**

Code change made alongside: `dump_base_kv.py` resolved its meta path as
`grid_dataset_<tag>_meta.json` only, which does not exist for wcci2022 (never generated with
`--task classify`). It now falls back to `<tag>_n1_meta.json`, resolving the suffix once and
reusing it for the `--empirical` JSONL path so meta and samples always describe the same run.

---

## 6. The shield (Component D)

`shield/context.py` (§5 contract) · `evaluator.py` (VIOLATED / SATISFIED / NOT_EVALUABLE / ERROR —
**only VIOLATED blocks**) · `shield.py` (`validate()`, `RuleProvider`, `JsonlRuleProvider`).

Rule retrieval sits behind `RuleProvider`, **not** `build_kg.py::get_rules_for_entity`, so the KG
can be redesigned without touching the shield or the eval harness.

**Option A semantics: affirmations never block.** Absence of support is recorded as `unsupported`
and reported; only a constraint violation produces BLOCK. Option B (gating on missing support) was
rejected as deployed behaviour because its false block rate would be a function of how evenly the
corpus covers the classes — a property of the standards, not of the grid. `eval_shield.py`
computes the Option B counterfactual every run.

### 6.1 Pending change for the forecast task

The shield's role shifts from **verify** to **challenge** — it cannot confirm a claim about the
future, only ask whether the present is plausible grounds for it.

| GNN says | Telemetry | Shield |
|---|---|---|
| `not_at_risk` | stressed | **BLOCK** |
| `not_at_risk` | nominal | pass |
| `at_risk` | stressed | pass — they agree |
| `at_risk` | nominal | **pass — must not block** |

That last row is load-bearing: predicting risk from a calm grid is exactly what learning adds
(early warning), and blocking it destroys the model's only real value. It is also the right safety
posture — the costly error is a missed fault, not a false alarm.

Code impact: `FAULT_CLASSES` → `{"at_risk", "not_at_risk"}`; `validate()` applies constraint
blocking only on the permissive prediction. `context.py`, `evaluator.py`, guard core, tests
unchanged.

**Affirmation remap** (deterministic, lossless, no LLM): `normal` → `not_at_risk`;
`overload | line_trip | cascade` → `at_risk`. Re-run the guard afterwards to re-measure fire rates.

⚠️ **Keep the 4-class vocabulary at translation time and collapse afterwards.** "Does this
standard describe an overload?" is a more natural judgment for a model reading a grid code. The
collapse is one-way and lossless. **This is why translation is not blocked on the task decision.**

---

## 7. Interpretability controls — designed, NOT implemented

These prevent one specific failure: reporting **(b)** *"we couldn't build much of a symbolic
layer, so we can't tell whether it would help"* as if it were **(a)** *"we built a sound symbolic
layer and it didn't help"*. Only (a) answers the thesis question.

### 7.1 Ceiling analysis — the decisive one

Fit a small decision tree on the 14 context variables. Two targets: **→ `risk_label`** (can *any*
function of present observables predict a fault within `H`? — also the shield-alone baseline), and
**→ "GNN was wrong"** (the upper bound on what any symbolic gate could catch).

- tree can't beat chance → no rule system over observable state could catch these errors. Answer
  **(a)**, established without reference to rule count.
- tree does well, shield doesn't → the information is there and the rules missed it.

Degenerate under the classify task (a tree would hit 100% rediscovering the labeller); a real
measurement against a lookahead target.

### 7.2 Expert baseline

Hand-write ~10–15 rules against the same 14 variables. **Third arm, never merged.**

- expert rules also give ~0 delta → gating doesn't help regardless of provenance; extraction
  thinness exonerated.
- expert rules work, extracted don't → **extraction is the bottleneck**. Sharper than either alone.

Under the classify task this control was contaminated (the "expert" rules would have been the 4
label-definition rules). Against a lookahead target they become a legitimate **persistence
baseline** — non-circular.

### 7.3 Rule-count ablation

Subsets of 5, 10, 20, 50, all N → plot catch rate vs ruleset size. **Flat** → more rules wouldn't
have helped, supports (a). **Still rising at N** → genuinely truncated, but now *quantified*: a
measured (b) is a legitimate limitation; an unmeasured one is the hole.

### 7.4 The three-way comparison the redesign unlocks

**shield-alone vs GNN-alone vs GNN+shield**, all predicting a fault within `H`. Neither side sees
the future, so neither can cheat. Every outcome is reportable — this was impossible under the
classify task, where shield-alone was 100% by construction.

### 7.5 Watch for label-definition restatements

Any rule that restates the labelling function inflates results while measuring nothing. Far less
likely under the forecast task, but check and flag rather than quietly count.

---

## 7.5 AGENDA — what is next, in order (as of 2026-08-16)

Component A is **done and trained**: N-1 screening, held-out test F1 **0.8874**, AP 0.9550, 1.91×
the best single-rule baseline (0.4639); message passing contributes +0.058 over endpoint features.
See [`gnn_n1_tightening.md`](gnn_n1_tightening.md). Everything below is what remains.

| # | Task | Blocked on | Notes |
|---|---|---|---|
| 1 | ~~`case14` N-1 set + cross-topology eval~~ | **DONE 2026-08-16** | 6,000 frames / 581 scenarios / 118,502 labels, 289 s. Result in §7.6 — the model **does not** transfer. |
| 2 | ~~`wcci2022`~~ | **DONE 2026-08-16** | 4,000 frames / 609 scenarios / 742,472 labels, 1,807 s. Result in §7.6 — transfers **partially**. |
| 3 | **Component B stages 2–3** — `translate.py` → `polarity_guard.py` → `validate.py` | research PC | The long pole. Everything in 4 and 5 waits on it. |
| 4 | **Component C — KG redesign** | 3 | Build against the rules that actually survive translation + guard + validation, not the v1 mechanism. |
| 5 | **Shield on real predictions** | 3 | `validate_n1` is built and covered by 9 tests but has never seen model output or a real rule corpus. |
| 6 | **Threshold selection** | nothing | Every reported figure is best-threshold. Choose on val, report on test. The **missed-violation count** is the quantity the asymmetric gate keys on (§6.1), so select against that, not against F1. |
| 7 | **Decision, not a task:** retrain `classify` under the normalization fix, or leave it demoted and caveated | — | See `revised_thesis_claim.md` §2 and §6.1. Recommendation: leave it demoted. |

**1 and 2 are done (§7.6). 6 is unblocked and can be done in any session.** 3–5 need the research PC.

---

## 7.6 CROSS-TOPOLOGY RESULT — measured 2026-08-16

Generation protocol identical on all three topologies (`--task n1 --n1-stride 12`); only the
environment differs. All three sets verify clean under `verify_n1_dataset.py`: zero structural
errors, zero all-secure frames, and a global-`rho_max` lift of 1.08–1.16×, so **the task stays
non-degenerate off the training grid** — it has not collapsed into the closed-form trap of §2.1.

| topology | lines | contingencies | viol. rate | all-positive | best rule | **MODEL** | AP | recall | precision | missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neurips2020 *(in-dist, test split)* | 59 | 113,205 | 18.4% | 0.3104 | 0.4639 | **0.8972** — 1.93× rule | 0.9615 | 0.885 | 0.910 | 2,397 |
| **case14** *(unseen, smaller)* | 20 | 118,502 | 27.8% | 0.4345 | 0.5392 | **0.4477** — **0.83× rule** | 0.4270 | 0.935 | 0.294 | 2,150 |
| **wcci2022** *(unseen, larger)* | 186 | 742,472 | 24.8% | 0.3969 | 0.4915 | **0.5721** — 1.16× rule | 0.6419 | 0.531 | 0.620 | 86,273 |

**The headline: generalisation is real but partial, and it is ASYMMETRIC in grid size.**

- On **case14 the model loses to the single-rule baseline** (0.83×) and clears the all-positive
  baseline by only 1.03×. Report it as a failure — on that topology the neural component adds
  nothing a rule cannot state. Recall stays high (0.935) while precision collapses to 0.294
  (73,679 false alarms): it still *ranks* violations, it floods.
- On **wcci2022 the model beats the rule** (1.16×) despite the grid being 3× larger than training,
  but at 0.531 recall — 86,273 missed violations, the error class the asymmetric gate (§6.1)
  exists to catch.
- Scaling **up** (59 → 186 lines) costs far less than scaling **down** (59 → 20). That ordering is
  the opposite of the naive expectation and is the interesting result here.

### Hypothesis for the asymmetry — NOT YET TESTED

Voltage class is the obvious suspect, and §5.1 supplies the numbers. Backend base kV:

| grid | levels (kV) | overlaps training grid? |
|---|---|---|
| neurips2020 (training) | 138, 345 | — |
| wcci2022 | 138, **161**, 345 | almost entirely |
| case14 | **14, 20**, 138 | mostly **not** |

The `mean_v` node feature is `v/150.0` ([pyg_data.py](../scripts/pyg_data.py) `build_node_features`),
so a 20 kV case14 line enters at ≈0.13 where training saw ≈1.0 — a feature the model has never
observed in that range, and foreign features are normalized with 36-bus stats. wcci sits inside the
training distribution on this axis; case14 does not.

⚠️ **This is a hypothesis, not a finding.** It is cheap to test — compare per-feature normalized
distributions against the 36-bus training range, or re-score case14 with `mean_v` ablated — and it
should be tested before any of it is written up as explanation. Do not report the mechanism as
established.

### Caveats that must travel with these numbers

1. **Every figure is best-threshold on its own topology.** That is optimistic and it is agenda
   item 6. A threshold picked on val and held fixed across topologies would lower all three rows,
   and would lower case14 and wcci2022 most, since their score distributions have shifted.
2. Harness verified, not assumed: the eval's rule baseline reproduces
   `verify_n1_dataset.py`'s independent computation from raw JSONL exactly (case14 0.5392,
   wcci2022 0.4915), and neurips2020's 0.4639 / 0.3104 match `gnn_n1_tightening.md` to 4 dp.
   This is the same batched-path-vs-raw check that doc used to confirm edge alignment (§2).
3. The model figure for neurips2020 reads **0.8972 / AP 0.9615** here against the **0.8872 /
   0.9549** recorded in `gnn_n1_tightening.md` §1 (and 0.8874 in §7.5 above). Counts, baselines
   and threshold protocol all match, so `gnn_checkpoint_n1.pt` on disk is simply from a different
   run than the one written up. Take the checkpoint as authoritative and reconcile the docs.
4. This arm is **not** affected by the normalization bug of `gnn_n1_tightening.md` §3 — the N-1
   checkpoint was trained after the `_data_list = None` fix, so train and inference agree. That
   caveat applies only to the frozen classify checkpoint.

---

## 8. Metrics and verification gates

Per topology, GNN-only arm vs GNN+shield arm:

- **False block rate** — % of ground-truth safe frames blocked (over-constraint).
- **Missed-fault catch rate** — of frames the GNN got wrong, % blocked, split by failure mode
  (overconfident-wrong / class-confusion / threshold-failure / novel-topology-state).
- **Shield applicability** — % evaluating without NOT_EVALUABLE (expected ≈100%, pre-linted).
- **Rule compliance on PASS** — 100% by construction; less means an evaluator bug.
- **GNN-only vs GNN+shield safety delta** — the headline.

Gates before trusting any numbers:

1. `pytest tests/` green.
2. case14 sanity: false block rate far below 100% (catches base-kV regressions); **zero ERROR
   verdicts**.
3. Rerun any tag twice → identical aggregates (eval mode, no sampling).

---

## 9. Landmines — read before editing

- **Never re-close `EXTRACT_PROMPT` against `CONDITION_VOCABULARY`.** Tried twice, both times
  extraction returned **zero rules**: on frequency/timing-heavy standards the model correctly
  answers `[]` for nearly every chunk, starving stage 2. Guarded by
  `tests/test_condition_lint.py::test_extract_prompt_stays_open_vocabulary`.
- **Never name a guard output `*_confirmed.jsonl`** — `deduplicate_rules` globs that pattern and
  would merge rejects back into `all_rules_deduped.jsonl`.
- **Do not delete `--task forecast` from `generate_dataset.py`.** The `classify` generator was
  removed on 2026-08-16 and it is tempting to "finish the job". Don't: **no forecast dataset was
  ever written to disk**, so that code is the only way to reproduce §1.1 steps 1–2. Classify was
  safe to remove precisely because its datasets *are* on disk and frozen. Guarded by a RETIRED
  TASK PATHS note in the module docstring.
- **`--smoke` used to never terminate.** It caps chronics and steps but the loop cycles chronics
  toward `--n_records`, so it replayed 3 scenarios toward 300,000 records — 41 hours on the
  118-bus grid. Fixed 2026-08-16 by `SMOKE_MAX_RECORDS = 200`. If that constant is ever removed,
  the hang comes back.
- **Forecast training must not overwrite the frozen classify artifacts.** It writes
  `gnn_checkpoint_forecast.pt` / `normalization_stats_forecast.pt` /
  `processed_grid_data_forecast.pt`. `gnn_checkpoint_best.pt` (macro F1 0.8277) stays frozen.
- **`gnn_logit_margin.json` is tuned to the 4-class distribution** — do not reuse it for forecast.
- **Thinking must be suppressed via `generate_no_think()`** (API-level `think=False`), not the
  `/no_think` prefix — newer Qwen builds ignore the prefix and the reasoning trace breaks JSON
  parsing.
- **`DEVICE` is a `torch.device`** — compare `.type`, never the object against `"cuda"`.

---

## 10. Runbook

```powershell
# 1. personal PC — forecast dataset + training
python scripts/generate_dataset.py --env neurips --task forecast --horizon 6 --n_records 300000
python scripts/preprocess.py
$env:GRID_TASK = "forecast"; python training/train_gnn.py

# 2. research PC — LLM stages (sequential, never both models loaded)
python extraction/translate.py --candidates rules_35b/ --out rules/
python extraction/validate.py  --candidates rules/guarded/

# 3. personal PC — guard (--report first to choose the cutoff, then apply)
python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14 --report
python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14

# 4. base kV — backend method, DONE for all three tags on the personal PC (§5.1).
#    Re-run only if an environment is reinstalled. Never pass --empirical again.
python scripts/dump_base_kv.py --tag neurips2020
python scripts/dump_base_kv.py --tag case14
python scripts/dump_base_kv.py --tag wcci2022
```

⚠️ Every command above runs under the repo venv — `.venv/Scripts/python.exe`. A bare `python` on
the personal PC resolves to the Microsoft Store stub and fails with "Python was not found".
Set `PYTHONIOENCODING=utf-8` when redirecting or piping output: these scripts print `→`, and the
Windows pipe encoding is cp1252, which crashes them mid-run on a `UnicodeEncodeError`.

`rules_35b/` stays pristine as the stage-1 archive. The guard runs between stages 2 and 3 —
`validate.py` reads `rules/guarded/`, where the kept files reuse the `*_translated.jsonl` name so
its glob picks them up unchanged while the unfiltered originals survive for the counterfactual.
