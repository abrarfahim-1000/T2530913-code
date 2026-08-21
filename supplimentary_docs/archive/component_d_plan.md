> # ⚠️ RETIRED 2026-08-21 — FOLDED INTO `thesis_findings.md`. DO NOT FOLLOW, DO NOT CITE.
>
> This was the operative build plan from 2026-08-15 to 2026-08-21. **All four components are now
> built and measured, so there is nothing left to plan.** Every section that carried live content
> was moved into a document that is still maintained:
>
> | what was here | where it lives now |
> |---|---|
> | §1.1, §2 — task selection, the two rejected forecast targets, the closed-form classify finding | `thesis_findings.md` §9 |
> | §3, §3.1–§3.5 — extraction pipeline, corpus measurements, the stage-2 parsing bug | `thesis_findings.md` §10.1–§10.2 |
> | §4, §4.2 — the polarity guard, the audit, variable coverage | `thesis_findings.md` §10.3–§10.4 |
> | §5, §5.1 — the binding voltage contract and base-kV derivation | `thesis_findings.md` §11 |
> | §6 — the shield's gating logic and Option A semantics | `revised_thesis_claim.md` §5.1 |
> | §6.1 — the forecast-task shield change | **dropped** — that task was rejected and never trained |
> | §7 — interpretability controls, designed but never run | `thesis_findings.md` §17.1 · `revised_thesis_claim.md` §9 |
> | §7.5 — the agenda | `thesis_findings.md` §17 (only the open items survive) |
> | §7.6 — cross-topology raw-model table and the untested hypothesis | `thesis_findings.md` §14 |
> | §8 — metrics and verification gates | `thesis_findings.md` §13.5 |
> | §9 — landmines | `thesis_findings.md` §18 |
> | §10, §13.1 — runbooks | `CLAUDE.md` (Development Commands) |
> | §11, §11.3, §11.4 — guarded-corpus shield result, structural ceiling, tautology caveat | `thesis_findings.md` §13.1, §13.3, §13.4 |
> | §12 — document alignment, and why re-extraction is not recommended | `thesis_findings.md` §15 |
> | §13, §14 — decisions, run-3 guard, the invariance result | `thesis_findings.md` §10.3, §13.6 |
> | §15 — the validation A/B and the fabricated reasoning | `thesis_findings.md` §12 |
> | §16 — the provenance knowledge graph | `thesis_findings.md` §16 · `revised_thesis_claim.md` §8 |
> | §17 — the pre-commit cleanup ledger | **stays here only** — a one-time historical record, cited by nothing |
>
> **Two sections below are actively WRONG and were corrected during the fold.** §11.2's
> "the shield's overrides are coin flips in-distribution" reading is **void** (see
> `thesis_findings.md` §13.2), and §1.1/§2 quote "100% agreement on 55,000 records" and "100% of
> frames are mixed", both superseded (315,000 records / 0 disagreements, and 96–99% mixed).
> The §6.1 forecast design describes a task that was rejected and never built.
>
> Retained unedited as the working journal: it records the order things were discovered in, the
> predictions made before the fact, and the reasoning behind decisions that the folded documents
> state only as conclusions.

---

# Neuro-Symbolic Grid Project — Current Plan

**Supersedes `archive/component_d_handoff_archive.md`** (retired 2026-08-15, kept for the v1/v2
negative-results record only). Everything still operative is in this file.

**Read alongside:** `CLAUDE.md` (project overview), `archive/gnn_final_results.md` (Component A
negative results — note §2 below reopens Component A).

**For the narrative account** — what was found and what it means, readable without the
codebase open — see [`thesis_findings.md`](../thesis_findings.md). This file is the
operational record: runbooks, derivations, landmines. Where the two disagree, this one
is authoritative and the findings doc needs correcting.

---

## 1. Status and sequencing

| Component | State |
|---|---|
| **A — GNN** | **DONE and TRAINED** (2026-08-16). N-1 screening; test F1 0.8872, 1.91× the best single rule. Classify and forecast tasks probed and rejected — §1.1, §2. |
| **B — Extraction** | **ALL FOUR STAGES DONE** (2026-08-20). 2,463 candidates → 58 translated → guard 32 → **stage 3: 11 confirmed, 4 distinct** (§15). End-to-end yield **0.16%**. |
| **C — KG** | **BUILT 2026-08-20 — §16.** Provenance graph, 36 nodes / 40 edges, topology-agnostic. Opt-in retrieval verified identical to the JSONL path on all three grids. v1 deleted. |
| **D — Shield** | **RUN END-TO-END on all three topologies against the VALIDATED corpus, 2026-08-20 — §15.6.** wcci2022 F1 0.5577 → **0.6253**, missed violations **−30.9%** at **0.934** intervention precision. Precision is **flat at 0.92–0.94 across all three grids** (§15.7). |

⚠ The sequencing table that stood here described the **forecast** phase and every row of it was
retired on 2026-08-16 (§1.1). Current sequence — commands in §13.1:

| When | Where | What |
|---|---|---|
| ~~done~~ | personal PC | N-1 datasets ×3 → training → cross-topology eval (§7.6) |
| ~~done~~ | research PC | `translate.py` run 2 → 82 translated (§3.5) |
| ~~done~~ | personal PC | `polarity_guard.py` → 33 kept · `audit_rules.py` → 11 can work (§4.2) |
| ~~done~~ | personal PC | `eval_shield_n1.py` on all three grids against the guarded corpus (**§11**) |
| ~~done~~ | research PC | `translate.py` run 3 with the proxy fix → 58 translated (§14.1–14.2) |
| ~~done~~ | personal PC | guard run 3 → **32 kept**; audit → 21 can work; **shield re-run reproduces §11 exactly** (§14.4) |
| ~~done~~ | research PC | `validate.py` run 1 → 1/32, **no evidence persisted** (§15.1) |
| ~~done~~ | research PC | `validate.py` A/B — `strict` reproduces 1/32, `translated` → **11 confirmed, 4 distinct** (§15.2) |
| ~~done~~ | personal PC | `eval_shield_n1.py` ×3 against the validated corpus — **better on every metric** (§15.6) |
| ~~done~~ | personal PC | Component C — provenance KG built, citation path live, equality verified (§16) |
| Open | personal PC | §7 controls · agenda item 6 for the §7.6 table · `translated` multi-seed (§15.9 caveat 4) |

**All four components are built and measured as of 2026-08-20.** What remains is optional
robustness, not construction — see §16.8.

**That expectation was recorded before the fact and then tested:** "the §11 numbers should move
only slightly — `loading_pct > 100` is doing all the work." Measured on the run-3 corpus, they
did not move **at all** (§14.4). The prediction about *yield* in §13 item 4 was wrong in
direction (58, not ~82); the prediction about *mechanism* held exactly. Both are recorded.

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
   and **96.4-98.8% of frames are strictly mixed** (⚠ corrected 2026-08-20 from "100% of sampled
   frames" — measured over all 22,000 frames, not a sample; what IS 100% is that **no frame on any
   grid is entirely secure**, 0.00% everywhere): some contingencies violate, others do not. The label
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
  architecture experiments in `archive/gnn_final_results.md` now describe the *old* task.
- It changes what the thesis claims. The defensible framing is a **diagnosis and a fix** — "we
  built the classifier, proved the target was closed-form with a 4-rule baseline, and redesigned
  the task" — which is a stronger arc than either half alone, but it is a change of story.
- If the supervisor would resist a partly negative-results framing, that is far cheaper to learn
  now than after the eval matrix runs.

**Machine split changed.** Supersedes `CLAUDE.md`'s "one machine runs everything":

| role | runs |
|---|---|
| **workstation** | dataset generation, GNN training, guard, KG build, shield, tests |
| **LLM host** (CUDA + Ollama) | LLM stages only — `translate.py`, `validate.py` |

⚠️ Hardware specifications were removed from every document on 2026-08-20 — they were not
load-bearing and the two places that recorded them disagreed. Read "LLM host" as *the machine
with CUDA and Ollama on it*, nothing more.

`training/config.py` auto-selects the small `[16,32,32]` / `heads=[4,4,1]` branch on non-CUDA
devices — the **proven deployed** config. The CUDA branch `[64,128,128]` has never produced a
working checkpoint, so training off the LLM host is the safer choice, not a compromise.

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

### 3.5 STAGE 2 RUN 1 — a parsing bug that presented as a corpus property (2026-08-19)

Stage 2 ran to completion on all 16 documents (32 min, `qwen3.6:35b-a3b`) and reported **6
translatable of 2,463 — 0.24%**, against a measured expressible share of 693 (28.1%). Output is
preserved at `translated_rules/` as evidence. **The run is void; do not read a yield from it.**

| outcome | count | share |
|---|---:|---:|
| `NO_VERDICT` — rule never assessed | 2,190 | 88.9% |
| `TRANSLATOR_FAIL` — response was an object, parser wanted an array | 40 | 1.6% |
| **actual model verdict** | **233** | **9.5%** |
| └ translated | 6 | — |

**Cause.** Every worked example in `TRANSLATE_PROMPT` showed the return shape *without*
`rule_id` — `{"translatable": true, "condition": ..., "role": ...}` — while `TranslationResult`
declares `rule_id` required and `translate_batch` keyed its `result_map` on it. The model complied
with its instructions; every entry then failed Pydantic validation and was discarded by a bare
`except ValidationError: pass`. `result_map` came back empty, so every rule in the chunk fell
through to `NO_VERDICT`. Reproduced directly: feeding the prompt's own examples to the parser
yields `missing ('rule_id',)` on each.

The 233 verdicts that did land look correct — `unobservable` 79, `ride-through` 36, `compound` 30,
`frequency` 29, naming specific missing variables (`rocof_hz_per_sec`, `time_seconds`). The model
was doing the job; the pipeline was throwing the answers away.

**The failure was invisible in the totals.** A void run and a genuinely inexpressible corpus
produce the same summary JSON. That is the property worth remembering: it would have been reported
as the §7 failure mode — a pipeline artifact presented as a finding about standards.

**Fixed.**
- `common.py` — `rule_id` in all 7 worked examples, plus an explicit output contract (one entry per
  rule, same order, `rule_id` copied verbatim, never skip a hard rule).
- `translate.py::build_result_map()` — keys on `rule_id`; falls back to **positional** zip when
  entries omit it *and* the array length matches the rules sent *and* nothing was keyed. Refuses to
  guess on a partial or mixed response, because attaching a translated condition to the wrong
  standard is worse than a NO_VERDICT.
- No silent drops: malformed entries are logged and counted (`n_unparseable`), and a chunk where
  *no* rule got a verdict logs ERROR naming it a parsing failure.
- Per-file and per-run `n_no_verdict` in the summary JSON; ERROR above a 20% file-level rate.
- `--debug-raw` dumps unparseable responses to `<out>/_raw/`.
- `tests/test_translate_mapping.py` — 8 tests, including one asserting every prompt example carries
  `rule_id`, so prompt and parser cannot drift apart again.

**Deliberately NOT changed:** the translation semantics. The strict clauses (compound-discard,
ride-through-is-neither, scope-vs-predicate) stay exactly as they were. Run 2 changes one thing, so
its yield is a clean measurement of what this prompt actually produces. Only then is there evidence
about whether the prompt is too strict — run 1 provides none, since 90% of the corpus was never
assessed.

**RUN 2 (2026-08-19, after the fix) — the parser fix is confirmed and the yield is now real.**
`total_no_verdict: 0`, `total_unparseable: 0` across all 2,463 rules — every rule received a model
verdict. TRANSLATOR_FAIL fell 40 → 5. **82 translated (3.3%)**, 49 distinct `(entity, condition)`
pairs, 54 CONSTRAINT / 28 AFFIRMATION, and **zero out-of-vocabulary variable names**. Qwen3.6-35B
obeyed the output contract exactly once the prompt stopped contradicting the parser — there is no
remaining instruction-following problem, and **no case for swapping in Nemotron for stage 2**
(doing so would also cost stage 3 its independence, since Nemotron is the validator).

The 82 → guard → audit chain is §4.2. The 693 (28.1%) expectation in §3.1 is a **variable-level**
bound — it asks whether a rule's variable *names* map onto the vocabulary — and cannot see the
blocker that dominates in practice: the variable is derivable but the **entity it is measured on**
is not (`power_factor` of one module vs the grid-wide aggregate). Do not cite 693 as a yield
target; cite it as what a static analysis over-counts, and §4.2 as why.

**Quality signal from the 6 survivors** (all that run 1 supports): two fail the prompt's *own*
self-check, firing on the healthy grid it defines —
`power_factor_at_max_load < 0.95 or ... > 1.0` and `... < -0.95 or ... > 0.90` (the second is
"0.90 lagging to 0.95 leading" with the sign convention lost). Both would be caught by the stage
2.5 guard at fire rate 1.0, which is the guard doing its job. Separately, R_746's source chunk is
unreadable OCR and the model tagged its own source *"Reconstructed from garbled text"* — a stage-1
PDF-extraction concern, tracked separately from this one.

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

### 4.1 The guard reads the N-1 sets, and defaults to all three topologies — 2026-08-19

`--tag` defaulted to `neurips2020 case14`, and `sample_contexts` resolved `grid_dataset_<tag>.jsonl`
— the classify-era filename. **wcci2022 has no classify set** (never generated; the classify
generator was removed 2026-08-16), so passing `--tag wcci2022` hit the skip-on-missing branch and
logged a warning. The default silently excluded the largest and most structurally distant grid from
a measurement whose entire purpose is cross-topology.

No wcci2022 classify set is needed. The guard reads *observations* plus the ground-truth `label` to
bucket frames by class; it never reads the task target. The N-1 records carry both — `label` and
`label_int` are still written by the N-1 generator alongside `n1_violation` — as well as every field
`build_context` consumes (`rho`, `v_or`, `p_or`, `q_or`, `line_status`, `load_p`, `gen_p`).

Changed: `resolve_dataset_path()` prefers `grid_dataset_<tag>_n1.jsonl` and falls back to the
classify file; `DEFAULT_TAGS` is all three. Same precedence `dump_base_kv.py` already uses for its
meta path (§5.1), for the same reason.

Preferring `_n1` **uniformly**, not just for wcci2022, is the load-bearing half. Classify generation
applied per-class keep-probability quotas (`NORMAL_KEEP_PROB`, `LINE_TRIP_KEEP_PROB`); N-1 does not
subsample by class. A fire rate is a claim about how often a rule fires in normal operation, so it
wants the natural class distribution — and mixing quota-subsampled rates for two grids with a
natural rate for the third would put three non-comparable numbers in one comparison table.

The context cache is now keyed on the source filename as well as the sampling parameters, so a
cache written under one source is not silently reused under the other.

⚠️ **Minority-class resolution.** The N-1 sets are 12k / 6k / 4k records, well under the
`--sample-size` default of 5,000 per class. Measured availability: `normal` 8,129 / 4,815 / 2,781,
but `overload` only 250 / 138 / 251 (neurips2020 / case14 / wcci2022). Constraint fire rates are
scored on `normal` and are fine. **AFFIRMATION support figures for `overload` rest on ~150–250
frames** — report them as such, and do not read their trailing digits as comparable in precision to
a constraint's. If affirmations turn out to be a large share of what survives translation, raise
`--n1-stride` density or generate more frames before trusting those rates.

Verified end-to-end on all three topologies (300-frame probe): all four classes sample, and
`voltage_pu_max > 1` fires at **1.00 on every topology** — the ~6% above-nominal operating point
from §5.1, now visible on wcci2022 too.

---

### 4.2 GUARD + AUDIT RESULTS — measured 2026-08-19

First run of the guard against a real ruleset (stage-2 run 2, 82 translated rules), and of a new
deterministic audit, `evaluation/audit_rules.py`. Both offline, no LLM, no GPU.

#### Guard: 82 → 33 kept, 49 rejected (59.8% contaminated)

**The fire-rate distribution is cleanly bimodal** — 43 rules at ≤ 0.01, 39 at ≥ 0.75, **nothing
between 0.1 and 0.9**. §4 warned this might not hold and told the reader to inspect the middle
before trusting the cutoff. There is no middle: the 0.5 cutoff is not a judgment call on this
corpus. Report it as measured, not chosen.

**Cross-topology: 38 rules differ by > 0.25 between grids**, and the pattern is systematic rather
than noisy. A textbook ±5% grid-code voltage band:

| topology | v_min (median) | v_max (median) | healthy frames inside 0.95–1.05 |
|---|---:|---:|---:|
| neurips2020 | 1.047 | 1.080 | **0.0%** |
| case14 | 1.011 | 1.100 | **0.0%** |
| wcci2022 | 0.982 | 1.050 | **100.0%** |

The same rule is polarity-contaminated on two grids and clean on the third. This is §5.1's
prediction — grid-code bands are calibrated to utility operating practice, not to IEEE test-case
setpoint conventions — now measured on real frames. It is also the strongest single argument for
guarding against **all three** topologies (§4.1): guarding on wcci2022 alone would have passed
every one of those rules.

#### Audit: of the 33 survivors, 11 distinct rules can do work

The guard asks *"is this rule wrong?"*. `audit_rules.py` asks *"can it do anything?"* — the two
come apart. A constraint that fires on 0% of healthy frames passes the guard; if it also fires on
0% of overload, line_trip and cascade frames it can never block, and contributes nothing.

| verdict | rules | representative |
|---|---:|---|
| INERT | 5 | `voltage_pu_max > 1.5`, `generation_load_imbalance_pct > 25` — never fires on any class on any grid |
| UNINFORMATIVE | 9 | wide voltage bands holding on `normal` **and** `overload` alike (best margin **+0.006**) |
| TOPOLOGY_DEPENDENT | 9 | voltage constraints firing on abnormal frames on some grids only (0.008–0.275) |
| USEFUL | 10 | **all ten are `loading_pct > 100`** |

Voltage cannot discriminate here because Grid2Op faults are **thermal**: voltage stays in band
through overload and cascade alike. That is what the +0.006 margin says.

⚠️ **Two caveats that must travel with the audit.**
1. `loading_pct > 100` scores 1.000 on abnormal frames partly because the `overload` label *is*
   `rho_max >= 1.0`. This is the §7.5 label-restatement trap; the USEFUL verdict is contaminated
   by it and must not be quoted as independent validation.
2. The audit scores against **classify** labels while the live task is **N-1 screening**. It is
   triage — which rules separate grid states at all — not a shield evaluation. The N-1 shield
   evaluation is still agenda item 5.

#### The mismatch runs BOTH ways — the headline result

The surviving rules touch **4 of the 14 vocabulary variables**:

| covered | uncovered (no surviving rule) |
|---|---|
| `voltage_pu_min` (18), `voltage_pu_max` (16), `loading_pct` (10), `generation_load_imbalance_pct` (2) | `rho_max`, `n_tripped_lines`, `any_line_tripped`, `active_power_mw_max`, `reactive_power_mvar_max`, `apparent_power_mva_max`, `power_factor_at_max_load`, `current_a_max`, `total_generation_mw`, `total_load_mw` |

**Nothing in the surviving corpus governs topology** — no rule mentions `n_tripped_lines` or
`any_line_tripped`, which is the entire subject of the N-1 task. The one INERT variable
(`generation_load_imbalance_pct`) leaves **3 of 14** with an informative rule.

So the standards/simulator gap is not one-directional, and the honest framing is:

> **(a)** The observation is coarser than the standards regulate. Grid codes govern *equipment* —
> a generator's power factor, a module's terminal voltage, a connection point's demand. Grid2Op
> simulates a *network*: buses, lines, aggregate flows. **1,332 candidates (54.1% of the corpus)
> are blocked on exactly two missing capabilities, frequency and time**, and would become
> *assessable* if the observation carried them. That is a measurement, not a hope — but
> "assessable" is not "useful": those rules would still have to survive the guard and the audit,
> and this corpus rejected 59.8% at the guard alone.
>
> **(b)** The standards are also silent about most of what the simulator *does* provide. Ten of
> fourteen observable quantities have no rule at all — including line outages, the subject of the
> live task. Grid codes govern **connection and equipment compliance**; contingency screening is
> an *operational* activity written as process requirements ("shall perform studies annually"),
> not as state predicates over telemetry. A shield built from connection codes cannot govern
> operations, because the corpus was never about operations.

Do **not** write "what the simulator provides can be governed by the rules that survived" — it is
measurably false at 4/14 coverage, and an examiner can check it in one command.

Blocked-capability counts from the run-2 verdicts (overlapping, 2,381 untranslatable):

| missing capability | rules | share |
|---|---:|---:|
| TIME / duration | 782 | 32.8% |
| FREQ / ROCOF / droop | 683 | 28.7% |
| PROT / relay / settings | 528 | 22.2% |
| ENTITY resolution | 364 | 15.3% |
| other | 564 | 23.7% |

#### What this implies for the remaining prompt work

The emergency-rating family (`current_amps > 580 or power_mva > 132`, rejected as compound for a
"10 m duration") is a PROXY SUBSTITUTION case that should become `rho_max > 1.0` — the one family
the audit shows does work. The ride-through family is voltage-shaped, which the audit shows is
inert or uninformative. **Fix proxy substitution; do not loosen ride-through** — it would add
rules that measurably cannot contribute.

And note the ceiling: more restatements of the thermal limit add no information. Ten copies of
`loading_pct > 100` is one rule. The deliverable is not a bigger corpus.

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

Rule retrieval sits behind `RuleProvider`: `JsonlRuleProvider` by default,
`kg.provider.KgRuleProvider` opt-in via `--rules-kg` (§16). Nothing under `shield/` imports `kg`,
so the gate keeps working with no graph present — which is what let Component D be measured
before Component C existed.

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
See [`gnn_n1_tightening.md`](../gnn_n1_tightening.md). Everything below is what remains.

| # | Task | Blocked on | Notes |
|---|---|---|---|
| 1 | ~~`case14` N-1 set + cross-topology eval~~ | **DONE 2026-08-16** | 6,000 frames / 581 scenarios / 118,502 labels, 289 s. Result in §7.6 — the model **does not** transfer. |
| 2 | ~~`wcci2022`~~ | **DONE 2026-08-16** | 4,000 frames / 609 scenarios / 742,472 labels, 1,807 s. Result in §7.6 — transfers **partially**. |
| 3 | **Component B stages 2–3** — `translate.py` → `polarity_guard.py` → `validate.py` | research PC | The long pole. Everything in 4 and 5 waits on it. |
| 4 | **Component C — KG redesign** | 3 | Build against the rules that actually survive translation + guard + validation, not the v1 mechanism. |
| 5 | ~~**Shield on real predictions**~~ | **DONE 2026-08-19** | Ran against the 33 guarded rules on all three topologies — **§11**. New harness `evaluation/eval_shield_n1.py` (+17 tests); `eval_shield.py` retired. Re-run after stage 3 and diff (§13.1). |
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

The `mean_v` node feature is `v/150.0` ([pyg_data.py](../../scripts/pyg_data.py) `build_node_features`),
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
#    Defaults to all three tags; reads the _n1 sets (§5.2).
python extraction/polarity_guard.py --translated rules/ --report
python extraction/polarity_guard.py --translated rules/

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

---

## 11. SHIELD RESULT — measured 2026-08-19, all three topologies

> ⚠ **SUPERSEDED BY §15.6 for the headline numbers.** This section reports the **guarded-32**
> corpus, before stage 3 ran. The validated-4 corpus beats it on every metric on every topology.
> §11 is retained because §11.3 (structural ceiling) and §11.4 (the tautology caveat) still
> stand, and because the guarded/validated diff is itself the evidence in §15.6.
> **§11.2's "coin flip in-distribution" reading is VOID — see §15.7 for what replaces it.**

**The first end-to-end run of the symbolic gate against real model output and a real rule
corpus** (agenda item 5, previously blocked). Harness: `evaluation/eval_shield_n1.py` (new;
the classify-era `eval_shield.py` is banner-marked RETIRED and must not be run). Rules: the
**33 stage-2.5 guarded** rules from run 2 — stage 3 has not run yet, so these numbers will
move slightly when it does.

Threshold **0.8849, selected on the neurips2020 val split and held fixed across all three
topologies** — this closes agenda item 6 for the shield arm. Every figure below is therefore
honest-protocol, not best-on-its-own-topology. The best-on-topology raw model F1 is printed
alongside as a reference line so these reconcile with §7.6.

### 11.1 The result

| topology | GNN F1 | **+shield F1** | Δ | missed violations | Δ missed | false alarms | Δ FA |
|---|---:|---:|---:|---:|---:|---:|---:|
| neurips2020 *(in-dist, test)* | 0.8956 | **0.8982** | +0.0026 | 2,041 → 1,614 | **−20.9%** | 2,331 → 2,735 | +404 |
| case14 *(unseen, smaller)* | 0.4167 | **0.4188** | +0.0021 | 18,592 → 18,497 | −0.51% | 21,431 → 21,439 | +8 |
| wcci2022 *(unseen, larger)* | 0.5577 | **0.6232** | **+0.0655** | 68,166 → 46,860 | **−31.3%** | 115,283 → 118,754 | +3,471 |

**On wcci2022 the shielded model (0.6232) beats the raw model's own best-on-topology ceiling
(0.5721)** and lifts it from 1.13× to 1.27× the single-rule baseline (0.4915). A post-hoc
gate that never touches training moved an unseen-topology result past what the model could
reach with an oracle threshold on that topology.

### 11.2 Where it acts, and how precisely

The gate can only speak where the base case violates a CONSTRAINT and the model still said
`secure`. That set is small, and its size is the whole story:

| topology | eligible (of all scored) | blocked | corrections | regressions | **intervention precision** |
|---|---:|---:|---:|---:|---:|
| neurips2020 | 831 (0.73%) | 831 | 427 | 404 | **0.514** |
| case14 | 103 (0.087%) | 103 | 95 | 8 | **0.922** |
| wcci2022 | 24,777 (3.34%) | 24,777 | 21,306 | 3,471 | **0.860** |

> ⚠ **THE READING BELOW IS VOID — measured 2026-08-20, superseded by §15.7.** The 0.514 is an
> artifact of averaging two rule families: 478 of the 831 neurips blocks fired on voltage rules
> at 0.13–0.20 precision against thermal's 0.92–0.94. Stage 3 rejected exactly those voltage
> rules on textual grounds, and intervention precision then reads **0.938 / 0.922 / 0.934 —
> flat across all three grids**. Do not restate "precision rises off-distribution" anywhere.

**This is the thesis claim, measured.** `CLAUDE.md` predicted "rule compliance stable while
GNN accuracy degrades". What the numbers show is sharper:

> **On the training topology the shield's overrides are coin flips (0.514) — when a good
> model disagrees with the rule, its disagreement carries information. Off-distribution that
> stops being true (0.860, 0.922) and the rule is right to override.** The symbolic layer
> does not get better; the neural layer stops earning the benefit of the doubt.

Note neurips2020 against its own conditional base rate: unconditionally **95.67%** of
contingencies on an overloaded base case are violations (§11.4), but among the ones this
model calls secure, only **51.4%** are. That gap **is** the model's contribution, and it is
exactly what vanishes on foreign grids.

### 11.3 The structural ceiling — what no corpus could reach

Independent of rule count (the §7.3 ablation answered structurally rather than by
subsetting): of every missed violation the model commits, what share sits on a base case a
present-state rule over `CONDITION_VOCABULARY` can even see?

| topology | missed violations | reachable by any present-state rule |
|---|---:|---:|
| neurips2020 | 2,041 | 427 (**20.9%**) |
| case14 | 18,592 | 95 (**0.51%**) |
| wcci2022 | 68,166 | 21,306 (**31.3%**) |

The complement is the hard bound: **no rule over this observation can reach the other 79% /
99.5% / 69%**, because those failures occur on base cases within every limit. Establishing
them needs a post-contingency power flow — the computation the GNN stands in for. This is the
answer to "would more rules have helped": on case14, essentially no; on wcci2022, up to a
third more, but only if a rule existed that fired where `loading_pct > 100` does not.

### 11.4 ⚠ THE CAVEAT THAT MUST TRAVEL WITH §11

`loading_pct > 100` on the base case is **not** a restatement of the N-1 labeller, but it is
close to a physical tautology, and the distance was measured rather than assumed:

| topology | P(violation) | P(violation given base overloaded) | lift |
|---|---:|---:|---:|
| neurips2020 | 19.54% | **95.67%** | 4.90× |
| case14 | 27.75% | **91.16%** | 3.28× |
| wcci2022 | 24.76% | **94.74%** | 3.83× |

If this were the §7.5 label-restatement trap the conditional would be 100%; at 91–96% the
predicate carries real but partial information. The honest reading: **the shield is
validating N-1 doctrine — a base case outside its limits cannot be declared secure against
further loss — and the data confirms that doctrine 91–96% of the time.** The symbolic layer
is not discovering new physics; it enforces a known operational principle the GNN failed to
learn, and failed to learn *worse* the further it got from its training topology. That is
legitimate and defensible. It is **not** a claim that the extracted corpus found something
novel, and must not be written up as one.

Equally important: **the doctrine being enforced is hand-written in `validate_n1`'s
docstring, not extracted from the corpus.** The LLM pipeline supplied only the threshold.
That doctrine belongs to the operational-standards class (NERC FAC-011, EU SO GL) that §12
shows is absent from the corpus — so the loop closes, but volunteer it in the write-up
rather than defending it under question.

### 11.5 Shield health — every §8 gate passes

| check | neurips2020 | case14 | wcci2022 |
|---|---|---|---|
| ERROR verdicts (must be 0) | **0** | **0** | **0** |
| NOT_EVALUABLE verdicts | 0 | 0 | 0 |
| Option B counterfactual (`unsupported`) | 0 | 0 | 0 |
| false block rate (base-kV regression check) | 0.36% | 0.007% | 0.47% |

Applicability is 100% on all three grids — every rule evaluated on every frame, which is what
the stage-2 AST lint plus the §5 voltage contract were built to guarantee. The Option B
counterfactual is 0 everywhere because **no AFFIRMATION survived the guard for the two N-1
classes**; Option A vs Option B is therefore not a live choice on this corpus, and the
comparison §6 promised cannot be made until affirmations survive.

Per-contingency failure records go to `results/failures/failures_<tag>.jsonl` (capped by `--max-failures`,
default 50,000 — wcci2022 misses ~68k).

---

## 12. DOCUMENT ALIGNMENT — why the corpus is thin, and what would fix it

Established while auditing the run-2 yield (§4.2). The per-document numbers show the problem
is not uniform across the 16 documents:

| document | candidates | translated | guard-kept |
|---|---:|---:|---:|
| NERC Reliability Standards (complete set) | 707 | 19 | 13 |
| GB Grid Code | 571 | 14 | 3 |
| Bangladesh Grid Code | 228 | 15 | 4 |
| EU NC RfG | 218 | 11 | 7 |
| IBR performance guideline | 237 | 5 | 1 |
| **TPL-001-5.1** | **11** | 1 | 1 |
| **NERC FAC-008-5** | **4** | 0 | 0 |

**The two documents actually about post-contingency performance contributed 15 candidates
between them.** TPL-001-5.1 yielded only **5 distinct chunks**, so the PDF on disk is a
summary, not the standard carrying Table 1's P0–P7 contingency categories.

What TPL-001 did produce is the diagnostic:

```
Line | loading_pct > relay_loadability_limit_pct
Grid | power_oscillation_damping < acceptable_damping_threshold
```

Free variables, not numbers — **and that is structural, not an extraction failure.** NERC
standards are criteria-referencing by design: they say "within its applicable Facility
Rating", and FAC-008 then requires each utility to hold a *documented methodology* for
computing that rating. The number is deliberately never in the standard. No prompt work
extracts a threshold that was never written down.

### 12.1 Where the numbers actually live

| source | what it supplies | status |
|---|---|---|
| **PJM Manual 14B** | post-contingency voltage as literal numbers (**0.92–1.05 pu** across TPL-001 P1–P7), thermal as "within the applicable emergency rating", plus a **voltage-drop** criterion | verified by search; strongest candidate |
| **NERC FAC-011-4** | the *predicate structure*: an SOL is thermal facility ratings + voltage limits + stability limits, pre- or post-contingency | verified; non-numeric, but it is the schema |
| **EU SO GL (Reg. 2017/1485)** | the *operation* sibling of the RfG **connection** code already in the corpus; Art. 25 operational security limits, formal N-1 criterion | ⚠ Article 25 text **not** verified — EUR-Lex PDF fetch returned empty. Open it before citing |
| ISO-NE PP3, NYISO Reliability Criteria, WECC TPL-001-WECC-CRT | same category — regional criteria with explicit numeric post-contingency limits | not investigated |

The corpus is biased toward **connection** codes (RfG, IEEE 1547, Order 842, PRC-024/025/029
= generator protection settings) and away from **operational security** codes. That is the
one-sentence explanation for §4.2's finding, and it is more defensible than "the simulator is
too coarse" because it does not require the simulator to be at fault.

### 12.2 The ceiling on re-extraction — decide with this in hand

A perfect document set supplies two predicates: post-contingency thermal (the corpus
**already** yields it as `loading_pct > 100`, the only USEFUL family) and post-contingency
voltage (which §4.2 measured as inert/uninformative here, since these grids operate at
1.05–1.08 pu and never enter the ±5% band). **Most of the gain would be more restatements of
the one rule already in hand**, and §11.3 bounds what that rule can reach.

Exactly **one** genuinely new capability appears in §12.1: PJM's **voltage-drop** criterion.
It is a delta rather than a level, so §4.2's band-inertness does not apply, and it is
physically computable. But `label_n1()` stores only `n1_post_rho` and discards
`sim_obs.v_or`, so it needs a generator change, a dataset regeneration (~55 min across all
three topologies) and a vocabulary extension before any such rule could be evaluated.

**Recommendation: do not re-run extraction on new documents for the thesis.** §11 is a
complete, positive, honest result on the corpus in hand, and §11.3 bounds the headroom.
Record §12 as the "what would extend this" section instead. Revisit only if the voltage-drop
channel is wanted, and treat that as a scoped experiment, not a re-extraction.

---

## 13. DECISIONS TAKEN 2026-08-19

1. **Proxy substitution fixed in `TRANSLATE_PROMPT`** (§4.2's remaining prompt item). R_769
   ORed an offshore circuit's rating (`580 A`, `132 MVA`) with `loading_pct > 100`; both
   absolute terms are grid-wide maxima while the rating belongs to one named circuit, so it
   fired on **100% of healthy frames on all three topologies**. The prompt now states the
   ratio REPLACES the absolute figure, carries the rating-schedule case as a worked example,
   and names the defective string as a `NEVER`. **The SELF-CHECK was the deeper hole** — it
   supplied healthy values for only 7 of 14 variables, so a rule naming
   `apparent_power_mva_max` had nothing to substitute and passed vacuously. It now covers all
   14 using measured healthy readings. Guarded by 3 tests in `tests/test_condition_lint.py`,
   two of which evaluate the defective condition against the prompt's own numbers rather than
   grepping for text.
2. **Ride-through loosening: NOT done, deliberately.** It is voltage-shaped, and §4.2
   measured voltage as inert or uninformative here. It would add rules that cannot contribute.
3. **Nemotron for stage 2: rejected** (§3.5). No obedience gap remains after the run-2 fix,
   and reusing the stage-3 validator at stage 2 would destroy validation independence.
4. **Stage 2 re-run: worth one pass, on the same research-PC trip as stage 3** — shipping the
   final ruleset with a known defect is worse than one extra run. Expect ~82 translated again,
   **not** a yield jump; the fix is precision, not recall.
5. **`eval_shield.py` retired, `eval_shield_n1.py` is the live harness.** 17 tests in
   `tests/test_eval_shield_n1.py`. Suite: **190 green.**
6. **Threshold protocol settled for the shield arm**: selected on the neurips2020 val split,
   held fixed across topologies. Agenda item 6 stays open for §7.6's raw-model table, which is
   still best-on-own-topology.

### 13.1 Runbook — the remaining sequence

```powershell
# research PC — stage 2 with the proxy fix, into a FRESH folder (translated_rules/ is run 2)
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe extraction\translate.py --candidates rules_35b\ --out translated_rules_run3\ --debug-raw

# personal PC — guard (needs the datasets)
.venv\Scripts\python.exe extraction\polarity_guard.py --translated translated_rules_run3\ --report
.venv\Scripts\python.exe extraction\polarity_guard.py --translated translated_rules_run3\

# research PC — stage 3
.venv\Scripts\python.exe extraction\validate.py --candidates translated_rules_run3\guarded\ --out rules\

# personal PC — re-run §11 against the validated corpus and diff the tables
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag neurips2020 --json results\shield\shield_neurips2020_v2.json
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag case14     --json results\shield\shield_case14_v2.json
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag wcci2022   --json results\shield\shield_wcci2022_v2.json
```

⚠ `--rules` now defaults to `validated_translated/all_rules_deduped.jsonl` (stage-3 output).
  It used to default to `rules/all_rules_deduped.jsonl`, which has not existed since the v2
  pipeline — the default resolved to a missing file. Fixed 2026-08-20. To reproduce §11
exactly, pass `--rules translated_rules/guarded/` — the 33-rule guarded corpus these numbers
were measured on.

---

## 14. STAGE 2 RUN 3 + GUARD — measured 2026-08-19, and the shield result is INVARIANT

Run 3 carried one change: the proxy-substitution fix and the extended SELF-CHECK (§13 item 1).
32 min, `qwen3.6:35b-a3b`, `total_no_verdict: 0`, `total_unparseable: 0` — the run-1 parser class
of failure stays fixed.

### 14.1 The defect is gone, by rejection rather than substitution

`grep` for an absolute rating ORed into a condition returns **nothing** across the run-3 corpus.
R_769 — the offshore rating schedule that produced
`loading_pct > 100 or current_a_max > 580 or apparent_power_mva_max > 132` and fired on 100% of
healthy frames on all three grids — is now in `*_untranslatable.jsonl` with the reason
*"compound: requires time_seconds (post-fault condition)"*.

**That is a defensible call, and arguably better than the substitution the fix was aiming for.**
The source table carries 6hr / 20m / 10m / 5m / 3m duration columns, so "Post-Fault Continuous"
genuinely does have a duration attached, and the compound-discard rule (§3.4) applies. The
contaminated rule is eliminated either way; it simply left by a different door.

### 14.2 Yield fell 82 → 58, and the fall is entirely in the worthless part

| | run 2 | run 3 |
|---|---:|---:|
| translated | 82 | **58** |
| distinct (entity, condition) | 49 | 38 |
| CONSTRAINT / AFFIRMATION | 54 / 28 | 43 / 15 |
| **guard-kept** | **33** | **32** |
| guard rejection rate | 59.8% | **44.8%** |
| `loading_pct > 100` rules (the only USEFUL family) | **10** | **10** |

§13 item 4 predicted "~82 again, not a yield jump; the fix is precision, not recall." The count
fell instead of holding — **record that the prediction was wrong in direction** — but the
composition explains it and vindicates the mechanism: the working family is unchanged at exactly
10, and the 24 lost rules are voltage and power-factor restatements, which §4.2 measured as inert
or uninformative. The guard rejection rate falling 15 points on an independent run is the
cleanest evidence that the prompt now emits a cleaner stream rather than a smaller one.

### 14.3 Independent replication of two §4.2 findings

Both were single-run observations before; run 3 is an independent draw and both hold.

1. **The fire-rate distribution is again cleanly bimodal** — 33 rules at ≤ 0.01, 25 at ≥ 0.75,
   **nothing between 0.1 and 0.9**. The 0.5 cutoff remains measured, not chosen.
2. **The ±5% voltage band is again contaminated on two grids and clean on the third** — 19 rules
   differ by > 0.25 across grids, all in the same direction (`neurips2020=1.00 case14=1.00
   wcci2022=0.00`). §5.1's operating-point explanation reproduces.

Audit verdicts (`evaluation/audit_rules.py`, `results/audit/audit_run3.json`):

| verdict | run 2 | run 3 |
|---|---:|---:|
| INERT | 5 | 4 |
| UNINFORMATIVE | 9 | 7 |
| TOPOLOGY_DEPENDENT | 9 | 11 |
| **USEFUL** | **10** | **10** — still all `loading_pct > 100` |
| **can do work** | 19/33 (11 distinct) | **21/32 (13 distinct)** |

Affirmation coverage is again `normal` and `overload` only — **still nothing for `line_trip` or
`cascade`**, so Option B (§6) remains unmeasurable on this corpus for the second run running.
That is now a property of the corpus, not an accident of one run.

### 14.4 THE RESULT THAT MATTERS — the shield is invariant to the translation run

`eval_shield_n1.py` re-run on all three grids against the **run-3** guarded corpus reproduces
§11 **exactly, to every reported digit**:

| grid | GNN | +shield | Δ | blocked | intervention precision |
|---|---:|---:|---:|---:|---:|
| neurips2020 | 0.8956 | 0.8982 | +0.0026 | 831 | 0.514 |
| case14 | 0.4167 | 0.4188 | +0.0021 | 103 | 0.922 |
| wcci2022 | 0.5577 | **0.6232** | **+0.0655** | 24,777 | 0.860 |

Not "close to" §11 — **identical**, including every count of corrections, regressions, missed
violations and false alarms. Zero ERROR and zero NOT_EVALUABLE verdicts, as before.

⚠ **THE MECHANISM SENTENCE BELOW IS FALSE — measured 2026-08-19, pending rewrite.** 478 of
neurips2020's 831 blocks (58%) fire on *voltage* rules with no thermal rule involved. The
blocking DECISIONS are still identical across both corpora — that part stands — but not for
the reason given. Voltage blocks were then measured at 0.13–0.20 precision against thermal's
0.92–0.94, and a thermal-only shield scores strictly better on every axis. §11.2's
"coin flip in-distribution" reading is an artifact of mixing the two and is also void.

The mechanism is stated in §11.2 and is now demonstrated rather than argued: **`loading_pct > 100`
is the only rule that ever fires as a constraint on a base case, and both corpora contain exactly
ten copies of it.** Every other rule is silent at inference. Two independently generated rulesets,
differing by 24 rules, produce a bit-identical gate.

Two things follow, and both are worth reporting:

1. **The §11 result does not depend on the extraction run that produced it.** For a pipeline with
   an LLM in the middle, run-to-run stability of the *end* result is not a given, and this is the
   evidence for it.
2. **It also bounds the corpus's contribution honestly.** A ruleset can lose 29% of its rules and
   change nothing downstream, because the contribution was concentrated in one predicate all
   along. This is §4.2's "the deliverable is not a bigger corpus", now closed experimentally.

⚠ **Reproduction path corrected.** Run 2's `translated_rules/` and its `guarded/` subfolder were
**overwritten** when run 3 was copied into the same directory (the runbook asked for
`translated_rules_run3/`). Neither is recoverable — they were untracked. This would have been a
genuine hole in §11's reproducibility, and it is closed only because §14.4 shows the run-3 corpus
gives the identical result. **`translated_rules/guarded/` now holds run 3, and that is the corpus
to cite for §11.** Keep the next run out of this directory.

### 14.5 Next — RESOLVED 2026-08-20, see §15.10

> ✅ **The prediction recorded below was correct, and it paid off.** Validation *did* reject
> `loading_pct > 100` itself — all ten copies — and it was a finding about the validator, not a
> numerical drift. Full account in §15; the resolution is §15.10.

Stage 3 (`validate.py`) has still never run. It is the last unexecuted stage in the pipeline.

```powershell
.venv\Scripts\python.exe extraction\validate.py --candidates translated_rules\guarded\ --out rules\
```

Then re-run §11 a third time against `validated_translated/all_rules_deduped.jsonl` and diff. **Recorded
expectation, before the fact:** validation removes or corrects rules and never adds them, and
§14.4 shows the gate is insensitive to everything except the `loading_pct` family. So the shield
numbers should be identical again *unless validation rejects `loading_pct > 100` itself* — which
would be a substantive finding about the validator, not a numerical drift, and must be
investigated rather than absorbed.

---

## 15. STAGE 3 — VALIDATION RAN, AND THE PROMPT WAS THE VARIABLE (2026-08-20)

The last unexecuted stage (§14.5) executed. It ran **three times**: once as written, then twice
more as a controlled A/B after run 1 came back uninterpretable. The outcome resolves §14.5's
pre-registered prediction, voids §11.2's central reading, and produces a corpus that beats the
guarded-32 set on every topology.

### 15.1 Run 1 — 31 of 32 rejected, with no evidence of why

`validate.py --candidates translated_rules/guarded/` against the run-3 guarded corpus:
**1 confirmed, 31 rejected, 0 flagged, 0 NO_VERDICT.** The mechanism was sound — every rule got
a parseable verdict with a matching `rule_id`, so this was *not* §3.5's stage-2 failure class
repeating.

But `validate.py` incremented `n_rejected` and discarded the `Verdict`, including the model's
`reason` field, which the schema carries and the model had populated. 31 rejections, zero
explanations. **A rejection without its reason is not a finding**, and this one could not be
told apart from a broken prompt.

Two things were established before re-running, both without an LLM:

1. **All 32 rules pass the validator's own mechanical criteria.** Criteria 4 (vocabulary) and 7
   (healthy-grid self-check) are deterministically checkable; a substitution pass over the corpus
   found 32/32 passing both. Only 2 rules (R_741, R_1319) trip criterion 6's ride-through
   heuristic. **At most 2–4 of the 31 rejections rested on the prompt's explicit criteria.**
2. **At least one rejection is demonstrably wrong.** TPL-001-5.1 Table 1 (f) reads *"Applicable
   Facility Ratings shall not be exceeded"*; R_167 renders that `loading_pct > 100`, CONSTRAINT.
   Rejected.

The suspected cause was criterion 1 — *"Is this constraint actually stated in the source text?"* —
written for stage-1 candidates whose conditions were still in the standard's own terms. Applied to
a stage-2 condition it is a category error: translation's entire job is to move the condition out
of the standard's language, so a translated condition is *never* stated in the source text.

`validate.py` now writes `<stem>_rejected.jsonl` with the full verdict, carries a `reject_rate`
plus an ERROR tripwire above 80%, and takes `--prompt-variant`. Guarded by
`tests/test_validate_rejections.py`. **Never merge a validation run whose rejections were not
persisted.**

### 15.2 The A/B — one variable, perfectly nested outcome

Two arms over the identical 32-rule input, identical model (`nemotron-3-nano:30b`), differing
only in criteria 1–3. Criteria 4–7 and the output contract are a shared string
(`_VALIDATE_BODY` in `common.py`), and `VALIDATE_PROMPT` is byte-identical to the run-1 arm —
verified against `git show HEAD:extraction/common.py`.

| arm | question asked | confirmed | rejected | unique after dedup |
|---|---|---:|---:|---:|
| `strict` | is the constraint **stated** in the source text? | 1 | 31 | 1 |
| `translated` | is the condition a faithful **operationalization**? | **11** | 21 | **4** |

`strict` reproduced run 1 exactly — same count, same survivor (R_1154). The outcome is **perfectly
nested**: every rule `strict` confirmed, `translated` also confirmed; nothing was confirmed by
`strict` alone; all 21 rules `translated` rejected were also rejected by `strict`. **The entire
difference is the ten `loading_pct > 100` rules, and they moved as a block.**

### 15.3 Why `strict` rejected the thermal family — the stated grounds are false

The persisted reasons are not merely strict, they are incoherent, and in a specific way:

| rule | `strict` reason (verbatim) |
|---|---|
| R_167 | "uses an undefined variable `loading_pct` and the threshold 100 is not explicitly stated" |
| R_1443 | "uses '100' which is not a valid variable" |
| R_858 | "includes a unit (100.0), violating the allowed syntax" |
| R_260 | "includes non-Python syntax such as 'or' without proper parentheses" |
| R_1717 | "the correct variable is `loading_pct`, but ... incorrectly uses `loading_pct > 100`" |

`loading_pct` is in the vocabulary block injected into that same prompt. Numeric literals and
`or` are explicitly permitted by criterion 4. `100.0` is a float, not a unit. R_1717's reason
contradicts itself inside one sentence.

**The mechanism: forced by criterion 1 into a rejection it could not justify on the merits, the
auditor reached for the vocabulary criterion and hallucinated a violation of it.** Eleven of the
31 rejections cite a criterion-4 breach that the deterministic pass (§15.1) proves does not exist.
This is worth reporting as a methodological result in its own right — a validator's *verdict* was
stable across runs while its *reasoning* was fabricated, and only persisting the reasons exposed
it.

### 15.4 The 21 shared rejections are substantive — and sharper than the guard

With the framing corrected, what survives as a rejection is mostly correct, and several catches
are ones the deterministic guard could not make:

| rules | validator's ground | assessment |
|---|---|---|
| R_1833, R_1835, R_2282 | PRC-024 Attachments "only provide minimum time durations for voltage excursions, not a continuous limit" | **correct** — ride-through curves, criterion 6 |
| R_116, R_120, R_121, R_123 | ENTSO-E NC RfG Table 6.1 gives "a minimum operating time of 60 minutes for the 0.85–0.90 pu range"; AFFIRMATION of `normal` is inconsistent with a time-bound requirement | **correct**, and sharper than §4.2's fire-rate reading, which only saw them firing at 1.00 |
| R_1108, R_069 | "0.9–1.1 pu is an affirmation of normal, not a constraint to be inverted" | **correct** role catch |
| R_797, R_798 | source states ±5% for 400 kV and +10%/−15% for 230 kV, not flat 0.90/1.10 | **correct** threshold catch |
| R_1319 | STATCOM blocks at 0.2–0.3 pu — a ride-through range | **correct**, matches §15.1's deterministic flag |
| R_1926, R_1927, R_1098 | `voltage_pu_min < 0.95` is "the opposite logical requirement" of "at least 0.95 pu" | **WRONG** — for a CONSTRAINT that inversion is correct; the validator applied affirmation semantics to a constraint |
| R_797, R_798, R_434, R_741 | reason text says "requiring correction" | **verdict-selection defect** — diagnosed as fixable, then discarded rather than returned as CORRECT |

Roughly 17 of 21 rest on a defensible reading. **At least 3 are false rejections and 4 more should
have been CORRECT verdicts** — carry both when citing the yield.

### 15.5 The validated corpus — 4 rules

`validated_translated/all_rules_deduped.jsonl` (11 confirmed → 4 distinct by condition+role;
dedup merges sources):

| rule_id | role | severity | condition |
|---|---|---|---|
| R_858 | CONSTRAINT | high | `loading_pct > 100.0` |
| R_916 | CONSTRAINT | medium | `loading_pct > 100` |
| R_1443 | CONSTRAINT | high | `loading_pct > 100` |
| R_1154 | AFFIRMATION | medium | `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1` |

Pipeline yield end to end: **2,463 candidates → 4 rules (0.16%)**. Affirmation coverage is
`normal` only, so **Option B (§6) remains unmeasurable for the third run running** — now a settled
property of the corpus.

### 15.6 SHIELD RESULT against the validated corpus — strictly better everywhere

`eval_shield_n1.py --rules validated_translated/all_rules_deduped.jsonl`, same checkpoint, same
held threshold 0.8849 selected on the neurips2020 val split.

| topology | corpus | GNN F1 | +shield F1 | Δ | blocked | corrections | regressions | **int. precision** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| neurips2020 | guarded-32 | 0.8956 | 0.8982 | +0.0026 | 831 | 427 | 404 | 0.514 |
| neurips2020 | **validated-4** | 0.8956 | **0.9038** | **+0.0082** | 353 | 331 | 22 | **0.938** |
| case14 | guarded-32 | 0.4167 | 0.4188 | +0.0021 | 103 | 95 | 8 | 0.922 |
| case14 | **validated-4** | 0.4167 | 0.4188 | +0.0021 | 103 | 95 | 8 | **0.922** |
| wcci2022 | guarded-32 | 0.5577 | 0.6232 | +0.0655 | 24,777 | 21,306 | 3,471 | 0.860 |
| wcci2022 | **validated-4** | 0.5577 | **0.6253** | **+0.0676** | 22,559 | 21,065 | 1,494 | **0.934** |

**Dropping 28 of 32 rules improved or held every metric on every grid.** Zero ERROR and zero
NOT_EVALUABLE throughout; blocks are now `high` severity only — every `critical` block in the
guarded runs came from a rejected voltage rule.

The trade is explicit and worth stating: validation cost a little coverage (neurips corrections
427 → 331, wcci 21,306 → 21,065) and bought a large cut in bad blocks (neurips regressions
404 → 22, wcci 3,471 → 1,494). Net F1 rises in both cases. On case14 the result is **bit-identical**
— the voltage rules never fired there, so nothing was lost.

Structural ceiling, recomputed: missed violations reachable by any present-state rule are now
**16.2% / 0.51% / 30.9%** (was 20.9% / 0.51% / 31.3%). The §11.3 bound is unchanged in substance.

**Artifacts:** `shield_{neurips2020,case14,wcci2022}_validated.json`. ⚠ These runs overwrote
`results/failures/failures_*.jsonl`, which now describe the validated corpus, not the guarded one.

### 15.7 ⚠ §11.2's "coin flip in-distribution" reading is VOID — this replaces it

§14.4 already flagged it, pending a rewrite. This is the rewrite.

The claim was: *the shield's overrides are coin flips in-distribution (0.514) and reliable
off-distribution (0.860, 0.922) — the symbolic layer holds while the neural layer stops earning
the benefit of the doubt.* That reading was an artifact of mixing two rule families. 478 of
neurips2020's 831 blocks fired on voltage rules at 0.13–0.20 precision against thermal's
0.92–0.94; averaging them produced the 0.514.

Validation removed exactly those voltage rules **on textual grounds, having never seen the
fire-rate data**. With them gone:

> **Intervention precision is 0.938 / 0.922 / 0.934 — flat across all three topologies.**

The corrected claim is different, and stronger:

> **The gate's precision is topology-invariant, because it enforces a physical doctrine rather
> than a learned pattern. What changes off-distribution is not how *right* the gate is, but how
> *often it gets to speak* — 0.31% → 0.087% → 3.04% of contingencies — and how much of the model's
> error sits where the doctrine can reach it: 16.2% / 0.51% / 30.9%.**

The wcci2022 result that carried the thesis survives intact and improves: the shielded score
**0.6253** still exceeds the raw model's own best-on-topology oracle ceiling (0.5721).

**Do not restate the "precision rises off-distribution" version anywhere.** It is measured to be
false once the misextracted rules are removed.

### 15.8 The convergence claim — two independent methods, same four rules

This is the strongest methodological result in the component, and it is worth foregrounding:

- **§4.2 / §14.3 — empirical.** No LLM. Ran every guarded rule against real grid data and asked
  *does it fire, and does it discriminate?* Answer across two independent extraction runs:
  `loading_pct > 100` is the only USEFUL family (10/10 both runs).
- **§15.2 — textual.** No grid data. Read the source standards and asked *is this a faithful
  reading of the document?* Answer: keep `loading_pct > 100`, reject the rest, because the voltage
  rules came from ride-through tables and time-bound operating envelopes misread as instantaneous
  limits.

Each result alone invites an easy objection — "your fire-rate cutoff is mistuned", "your prompt is
wrong" (and the `strict` arm shows the second objection can be *correct*). Convergence defeats
both. **The four surviving rules are a property of the standards/simulator mismatch, not an
artifact of either filter.** §12's document-alignment analysis is the explanation for *why*.

### 15.9 Caveats that must travel with §15

1. **§11.4 still applies in full.** `loading_pct > 100` on the base case remains close to a
   physical tautology (P(violation | base overloaded) = 91–96%), and the N-1 doctrine being
   enforced is hand-written in `validate_n1`, not extracted. Validation did not change that; it
   removed the rules that were *diluting* it.
2. **The yield is 0.16%.** 2,463 candidates → 4 rules. That is a finding about the corpus (§12),
   not a success metric, and must be reported as such.
3. **At least 3 of the 21 rejections are wrong** (§15.4), and 4 more should have been CORRECT.
   The validated corpus is better, not clean.
4. **Both arms ran a single seed at temperature 0.0.** Verdict stability across runs is
   established for `strict` (run 1 ≡ run 2) but not for `translated`.

### 15.10 §14.5's pre-registered prediction — resolved, and it was right

§14.5 recorded, before the fact:

> "the shield numbers should be identical again *unless validation rejects `loading_pct > 100`
> itself* — which would be a substantive finding about the validator, not a numerical drift, and
> must be investigated rather than absorbed."

That is exactly what happened. The `strict` arm rejected all ten copies of `loading_pct > 100`;
it was a finding about the validator, not about the corpus; and it was investigated rather than
absorbed. **Record this as a pre-registration that paid off** — had the run been accepted at face
value, Component D would have been left with one AFFIRMATION, which under Option A can never
block, and the shield would have been silently reduced to a no-op.

### 15.11 Next — DONE 2026-08-20, see §16

1. The corpus is complete. `validated_translated/all_rules_deduped.jsonl` is the corpus to cite.
2. ✅ Component C built against those 4 rules — §16. The graph shows the thermal predicate is
   stated in 10 clauses across 4 documents, which the deduped file hides by concatenating sources.
3. Still open, cheap: re-run `translated` at 2–3 seeds to close caveat 4.

---

## 16. COMPONENT C — THE PROVENANCE KNOWLEDGE GRAPH (built 2026-08-20)

The last planned piece of code. Built **after** the corpus was settled, deliberately: the first
attempt guessed at what the graph would hold and guessed wrong.

### 16.1 The first iteration was deleted, not archived

`extraction/build_kg.py` (1,203 lines) and its eight artifacts are gone. Recoverable from
`git show a5c5199:extraction/build_kg.py`; do not resurrect them.

It built the graph around *grid topology* — Bus/Line/Generator nodes read from one grid's
metadata, with rules hung off them by `entity`. Two independent failures:

1. **It carried no information.** `Line` (59) and `Bus` (28) were 3.5% of the corpus, so
   **6,375 of its 6,632 edges (96%) were `has_rule` noise** onto a single `Grid` node.
2. **It was welded to one topology.** The shield is evaluated on three. A graph keyed to 36-bus
   indices cannot serve case14 or wcci2022 without being rebuilt, and the rules are
   topology-agnostic by construction anyway.

Nothing in the project imported it — verified before deletion; the only `get_all_rules` matches
in the tree were inside `.venv`. Its output `kg/knowledge_graph.pkl` had already vanished from
disk months earlier and nothing noticed, which is the sharpest available evidence that it was
load-bearing for nothing.

### 16.2 Schema — provenance, not topology

```
Document --contains--> Clause --states--> Rule --deduped_into--> ServedRule
    --instantiates--> Predicate --reads--> Variable
```

| node | n | source |
|---|---:|---|
| `Document` | 5 | filename stems of `validated_translated/*_confirmed.jsonl` |
| `Clause` | 11 | each rule's `source` string, deduplicated within a document |
| `Rule` | 11 | the validated rules |
| `ServedRule` | 4 | `all_rules_deduped.jsonl`, **carried verbatim** — what the shield receives |
| `Predicate` | 2 | normalized condition semantics |
| `Variable` | 3 | the `CONDITION_VOCABULARY` entries actually read |

**36 nodes, 40 edges**, spread across five edge types. No edge type exceeds 60% of the total —
guarded by `test_no_node_type_dominates_the_edge_count`, which exists solely to stop a future
change regressing to v1's 96%.

**The `Predicate` layer is the graph's own contribution.** `deduplicate_rules` keys on
`(entity, condition)` (`common.py:728`), which fragments **one physical check into three records**
purely on LLM noise: the extractor labelled it `Line` in the Bangladesh grid code and `Facility`
in the NERC set, and wrote `100` in one clause and `100.0` in another. `normalize_predicate()`
parses with `ast` and canonicalizes numeric literals, so those collapse. **This is a view, not a
substitution** — it never changes what `rules_for()` returns, because collapsing three served
rules into one would move `highest_severity` (critical/high/medium differ across them) and the
blocks-by-severity histogram with it.

### 16.3 THE MEASURED FINDING — independent corroboration

| predicate | role | served rules | clauses | documents | identified bodies |
|---|---|---:|---:|---:|---:|
| `loading_pct > 100` | CONSTRAINT | 3 | **10** | **4** | **2** |
| `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1` | AFFIRMATION | 1 | 1 | 1 | 1 |

The thermal check is stated **ten times across four documents**, by NERC (7 clauses, in the
Reliability Standards set and TPL-001-5.1) and the Bangladesh grid code (2 clauses), plus one
"EMO Dispatch Computer Constraints" clause in `power-system-requirements`.

This reframes §15.5's "the corpus is 4 rules", which undersells it in one direction and oversells
it in another. Honestly stated: **the symbolic layer is one thermal predicate and one voltage
affirmation — but the thermal predicate is independently restated by two standards bodies on two
continents.** A flat JSONL file cannot express that; the deduped file actively hides it by
concatenating sources into one string.

⚠ Two things not to overclaim:
- `power-system-requirements` does not identify its issuing body. `kg/schema.py::ISSUING_BODY` is
  **hand-labelled** and records `None` rather than guessing, because the body count is a reported
  figure. Two identified bodies, not three.
- Corroboration is not independent *evidence* about physics. Four documents restating a thermal
  rating limit is four documents agreeing on standard practice, which §11.4 already characterised
  as close to a physical tautology. It strengthens the provenance claim, not the novelty claim.

### 16.4 Retrieval is opt-in, and the guarantee is a test

`KgRuleProvider` satisfies `RuleProvider` structurally, so it drops into `validate_n1()` and the
harness unchanged. `--rules-kg kg/knowledge_graph.json` selects it; `JsonlRuleProvider` stays the
default.

`tests/test_kg.py::test_kg_provider_serves_exactly_what_the_jsonl_provider_serves` compares **full
rule dicts as sets**, not ids — a changed threshold, severity or role would move measured numbers
and must fail loudly.

**Verified end to end on all three topologies.** Every field of `shield_<tag>_kg.json` equals
`shield_<tag>_validated.json`: F1, precision, recall, tp, false alarms, missed violations, blocked,
corrections, regressions, eligible, severities, and all four shield-health counters.

⚠ **`threshold` and `ap` do NOT compare bit-equal, and that is not the graph.** Two runs of the
*identical JSONL path* reproduce the same drift — threshold at ~1e-6, AP at ~1e-8 — so it is
forward-pass nondeterminism on this hardware (Intel Arc XPU), not a retrieval difference. **No
count moved in any run.** This is worth recording independently of Component C: **the eval harness
is not bit-reproducible on this machine**, and §14.4's "identical to every reported digit" holds
at the 4 decimal places actually reported, not at full float precision. A logit sitting exactly on
the threshold could in principle flip a count; none has.

### 16.5 Never route by entity

`R_1443` is labelled `Facility`; `R_916` states the same check and is labelled `Line`. Retrieving
rules by walking entity edges — the classic KG pattern, and what v1 did — silently drops one of
them from any line-level query. Both providers therefore serve **every rule for every prediction**;
discrimination comes from the conditions. `test_graph_is_topology_agnostic` bans bus/line indices
and grid tags from node ids and structural attributes.

One deliberate exemption in that test: the verbatim `rule` payload carries the polarity guard's
`fire_rate_<tag>` fields, which do name all three grids. Those are *measurements about a rule*,
not structure, and cannot be stripped without breaking the set-equality guarantee. Indexed
entities remain banned inside the payload.

### 16.6 Citation

`KgRuleProvider.cite(rule_id)` accepts either a served rule id (what `ShieldResult.violated_rules`
carries) or any member rule id, and returns a `Citation` with every clause, document and issuing
body. `kg/cite.py::explain(result, provider)` renders a `ShieldResult` with the full chain.

`kg/cite.py` lives in `kg/`, **not** in `shield/`, on purpose — nothing under `shield/` imports
`kg`, so the gate keeps working with no graph present. That property is what allowed Component D
to be built and measured before Component C existed, and it is worth keeping.

`--citations <path>` on the harness writes one provenance record **per distinct rule that fired**,
not per contingency; the failure log already runs to 50,000 records.

### 16.7 Artifacts and runbook

```powershell
.venv\Scripts\python.exe scripts\build_kg.py --out kg\knowledge_graph.json --figures
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag wcci2022 --rules-kg kg\knowledge_graph.json --json results\shield\shield_wcci2022_kg.json --citations results\citations\citations_wcci2022.json
```

| artifact | what |
|---|---|
| `kg/knowledge_graph.json` | the graph — **JSON, not pickle**; v1's `.pkl` vanished unnoticed |
| `kg/kg_provenance.{svg,png}` | all 36 nodes in six layers |
| `kg/kg_corroboration.{svg,png}` | clauses per predicate, segmented by document |
| `shield_<tag>_kg.json` | the KG-path runs, for the equality diff |
| `results/citations/citations_<tag>.json` | provenance chains for the rules that fired |

**222 tests green** (25 new in `tests/test_kg.py`).

### 16.8 Next

The coding is complete — all four components built and measured. Remaining items are optional
robustness, not construction:

1. `translate`-arm validation at 2–3 seeds (§15.9 caveat 4).
2. Agenda item 6 for the §7.6 table — re-report the raw model under the held-threshold protocol
   used everywhere else.
3. If bit-reproducibility is wanted for the thesis, pin the forward pass (`GRID_DEVICE=cpu` plus
   deterministic flags) and re-run the three evaluations once. §16.4 says why this is cosmetic.

---

## 17. PRE-COMMIT CLEANUP LEDGER (2026-08-20)

Everything below happened in one pass, after the last component was measured and before the
first commit of the finished state. It is recorded because several entries are **not
recoverable** and because three of them were latent bugs, not tidying.

### 17.1 Deleted — untracked, therefore permanent

| what | size | why |
|---|---:|---|
| `data/grid_dataset_neurips2020.jsonl` | 3.42 GB | classify-era; task retired 2026-08-16 |
| `data/processed_grid_data.pt` | 1.32 GB | classify tensors |
| `data/grid_dataset_case14.jsonl` | 57 MB | classify-era |
| `data/split_neurips2020_{train,val,test}_idx.npy` | — | classify splits |
| `gnn_checkpoint_best.pt`, `gnn_checkpoint_leverA.pt` | — | classify checkpoints |
| `normalization_stats.pt` | — | nothing read it; stats are recomputed inline |
| `data/grid_dataset_{neurips2020,case14}_meta.json` | — | see §17.4 — these were **not inert** |
| `lib/` (vis-9.1.2, tom-select, bindings) | 740 KB | pyvis assets for the deleted v1 KG HTML renders |

**The closed-form result was measured before the datasets were deleted, not after** — 315,000
records, 0 disagreements. That table is now the only record of it, which is why it was taken
first.

### 17.2 Deleted — tracked, recoverable from history

`extraction/build_kg.py` (1,203 lines, v1 KG) · `evaluation/eval_shield.py` (classify harness) ·
`evaluation/summarize_shield_results.py` · `EDA_final.py` (958 lines) · `gnn_logit_margin.json` ·
`validated_rules/` (22 files, stage-3 run 1) · 8 v1 KG artifacts under `kg/` ·
`supplimentary_docs/{inference,thesis_overview_plain_english}.md`.

Recovery point for the classify-era tree: `git show a5c5199:<path>`.

`summarize_shield_results.py` is worth singling out. It was already orphaned — it read a
`results/` directory that did not exist. §17.3 **created** that directory, at which point the
script would have half-worked against a retired schema and produced a plausible-looking wrong
table. Being orphaned was what made it safe; it stopped being orphaned, so it went.

### 17.3 Moved — the repo root was the results directory

Twelve `shield_*.json`, three `failures_*.jsonl`, two `citations_*.json` and `audit_run3.json`
were loose in the root. They now live under `results/`:

| path | tracked? |
|---|---|
| `results/shield/shield_<tag>[_run3\|_validated\|_kg].json` | **yes** — small, and they are the recorded result |
| `results/citations/citations_<tag>.json` | **yes** |
| `results/audit/audit_run3.json` | **yes** |
| `results/failures/failures_<tag>.jsonl` | **no** — ~14 MB, rebuilt by re-running the harness |

`eval_shield_n1.py` and `audit_rules.py` now create their output's parent directory, so a fresh
clone does not fail on a missing folder. `.gitignore` moved from `failures_*.jsonl` to
`results/failures/`.

**Verified, not assumed.** `eval_shield_n1.py --tag case14` was re-run end to end through the new
paths with no `--rules` argument. Every count reproduces the recorded run exactly; `threshold` and
`ap` differ at ~1e-6 / ~1e-8, which is the forward-pass nondeterminism already documented in
§16.4. The 15-figure notebook also re-executes clean against the new locations.

### 17.4 Three latent bugs found while moving things

These are the reason this section exists rather than a commit message.

1. **`eval_shield_n1.py --rules` defaulted to `rules/all_rules_deduped.jsonl`** — a path that has
   not existed since the v2 pipeline. Every documented invocation passes `--rules` explicitly, so
   it never fired; anyone running the harness bare would have hit a missing file. Now defaults to
   `validated_translated/all_rules_deduped.jsonl`.

2. **`dump_base_kv.py` preferred a meta file whose dataset was gone.** It resolved its suffix as
   `("", "_n1")`, so for `neurips2020` and `case14` it picked the classify-era
   `grid_dataset_<tag>_meta.json` sidecars — which outlived the datasets they described.
   `--empirical` then died looking for a `.jsonl` that no longer existed, and the backend path
   silently read `n_line` from a stale file. Suffix order is now `("_n1", "_forecast")` and the
   two orphaned sidecars are deleted.

3. **`dump_base_kv.py --empirical` overwrote the authoritative sidecar.** Both methods wrote to
   `grid_dataset_<tag>_basekv.json`. The two disagree *by design* — backend reports nominal kV,
   the empirical scan reports what the grid actually runs at, ~6% higher — and the shield reads
   the backend file. One `--empirical` run would have silently replaced the file the shield
   depends on with the counterfactual arm, with nothing to detect it afterwards. Empirical output
   now goes to `*_basekv_empirical.json`, which is where the retained originals already sat.

### 17.5 Hardware specifications removed everywhere

`study.md` recorded one GPU, `CLAUDE.md` recorded a different one, and no result depends on
either. Rather than adjudicate, all CPU/GPU/RAM model numbers were removed from every document
and code comment. What survives is the split by **capability** — one machine has CUDA and Ollama,
the other does not — because that is a real constraint on the pipeline's shape: the two LLMs are
never co-resident, so Component B is three sequential passes rather than one.

### 17.6 Documents moved to `supplimentary_docs/archive/`

`gnn_final_results.md` · `lever_A_recommendation_dissertation.md` ·
`CASCADE_GENERATION_JOURNAL.md` · `shield_necessity_analysis_report.md` ·
`study3(integration).md` · `component_d_handoff_archive.md`.

Each was already banner-marked historical or superseded; the move makes the live set
self-evident. **Live:** this plan, `study.md`, `formula.md`, `thesis_findings.md`,
`gnn_n1_tightening.md`, `revised_thesis_claim.md`.

**`Architecture.svg` was redrawn rather than archived.** The old one depicted Qwen3-14B,
two-stage extraction and a `Bus / Line / Rule` knowledge graph — none of which describe what
was built. The replacement is two parallel tracks converging on the gate: the neural track
(simulator → model → a call on every line) and the symbolic track (16 documents → four
extraction passes → the provenance graph), meeting at the shield, then the verdict with its
citation, then the measured result on all three grids. It carries the thesis claim as its
closing line — *the model gets worse on a grid it has never seen; the gate does not.*

The April environment-selection study moved from `Dataset Selection Comparison/` (repo root) to
`supplimentary_docs/env_selection/` and was bannered. It is **not** dead weight — it is the
methodology answer to *"why this grid?"*, and its conclusion is the one that was built. Its
symbolic-schema criterion is void (it scores against the deleted topology KG) and it predates
`case14` entirely.

### 17.7 `data/` — reproducible, with one exception that matters

Every file in `data/` (423 MB, gitignored in full) is regenerable from committed code:

| artifact | produced by |
|---|---|
| `grid_dataset_<tag>_n1.jsonl` + `_n1_meta.json` | `scripts/generate_dataset.py` (seeded 42) |
| `processed_grid_data_n1.pt` | `scripts/preprocess.py` |
| `split_neurips2020_n1_{train,val,test}_idx.npy` | `training/train_gnn.py`, chronic-level, inline |
| `grid_dataset_<tag>_basekv.json` | `scripts/dump_base_kv.py` (backend method) |
| `label_contexts_<tag>.json` | `extraction/polarity_guard.py` (context cache) |

Two caveats, and the second is the one to act on:

- Regeneration needs the Grid2Op environments downloaded (~2.6 GB across the three).
  `grid_dataset_wcci2022_basekv_empirical.json` does not exist and never did — wcci2022 postdates
  the empirical method. Do not "restore" it.
- 🚨 **`gnn_checkpoint_n1.pt` is NOT bit-reproducible, and is therefore now tracked.** The
  forward pass is nondeterministic on this backend and the init is seed-sensitive (§16.4;
  `CLAUDE.md` records 0.82–0.90 across partitions at seed 42, degrading to ~0.72 at other inits).
  Retraining yields *a* good model, not *this* model — and every reported figure is this
  checkpoint's output. `.gitignore` keeps the blanket `*.pt` rule (the preprocessed tensor set is
  92 MB and regenerable) with a single explicit exception, `!gnn_checkpoint_n1.pt`, at 205 KB.
  **Do not remove that exception.** Losing this file makes the results unreproducible in a way no
  amount of code preservation fixes.

### 17.8 `rules/` — kept, and here is the argument

`rules/` now contains exactly one thing: `v1_archive/` (2.8 MB), the v1 extraction corpus. It is
**not** a live output directory, despite several scripts' usage strings still showing
`--out rules/` as a generic example; the live stage-1 corpus is `rules_35b/`.

It is **gitignored and untracked**, so deleting it is permanent — and it is the only evidence
behind the v1 table in `CLAUDE.md` (1,372 candidates → 50 CONFIRM / 451 CORRECT / 766 REJECT →
469 unique).

**Measured 2026-08-20, over all 469 v1 rules — this is the finding the archive exists for.**
The v1 pipeline extracted and validated rules but never checked whether they could *run*:

| | count | share |
|---|---:|---:|
| v1 rules surviving extraction + validation + dedup | 469 | 100% |
| …that parse as a Python expression at all | 293 | 62% |
| …that use only variables the simulator actually reports | **4** | **1%** |

The 176 that do not parse are not code — `droop_pct BETWEEN 2 AND 12`,
`power_factor_tolerance NOT DEFINED BY RELEVANT NETWORK OPERATOR`. The 289 that parse but fail
invent their own vocabulary — `frequency_hz`, `initial_delay_seconds`, `delta_f1` — all real
quantities in the standards, none of them in the 14 the observation carries.

**That is the whole argument for the four-stage v2 pipeline in one table.** Adding a translation
stage and a closed vocabulary looks like it destroyed the yield (469 → 4); what it actually did
was stop counting rules that could never have been evaluated. The executable yield went 4 → 4.
The 469 was never real. (The two fours are a coincidence of count, not the same rules.)

Recommendation: **keep the archive.** 2.8 MB is not the cleanup that matters, and the table above
cannot be re-derived from anything else in the repo. One command if the call goes the other way:

```bash
rm -rf rules/v1_archive
```
