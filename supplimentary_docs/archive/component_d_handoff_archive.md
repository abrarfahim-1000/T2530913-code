# ARCHIVED — Component D Handoff (2026-07-16 to 2026-08-15)

> **RETIRED 2026-08-15. Do not follow this document.**
> Everything still operative moved to **`component_d_plan.md`** — read that instead.
>
> Kept only for the historical record, which has thesis value:
> the v1 ruleset post-mortem (§2), the 2-stage extraction failure and the escape-hatch
> finding (§3, §3b), and the original fire-rate design argument (§3c). Several sections
> describe designs that were subsequently superseded or reversed.

**Audience:** a future Claude session tasked with building Component D (the Symbolic Shield) and
the final cross-topology test harness. This doc is self-contained: it records the blocking issues
found during pre-implementation exploration (2026-07-16), the extraction defects found and fixed
(2026-08-12), the fixes already applied, what the user runs manually before the build can start,
and the full build plan.

> **START HERE (as of 2026-08-15, end of day).**
>
> **Read §9 first — the task was redesigned and Component A is REOPENED.** The 4-class label
> turned out to be a closed-form function of the observation (4 rules reproduce it on 55,000
> records with 100% agreement), which made the neuro-symbolic comparison degenerate. The fix is a
> binary lookahead target (`at_risk` within H steps); the code for it is implemented and tested.
> **§11 is the sequencing table** — read it to know what runs where and next.
>
> Then, for background in the order it happened:
> §3d (full-corpus extraction measurements), §3e (rules carry two ROLES — CONSTRAINT and
> AFFIRMATION — and the vocabulary is now 14 variables, not 6), §10 (the three interpretability
> controls, designed but NOT yet implemented).
>
> **State:** stage-1 extraction done — 2,463 candidates in `rules_35b/`. Polarity guard, shield
> package, eval harness, and the forecast-task code all built; **118 tests green**. Nothing
> downstream of translation has run — stages 2–3 need Ollama on the research PC.
>
> ⚠️ §3's 6-variable vocabulary table and §6's `get_rules_for_entity` retrieval are **superseded**
> (by §3e and §3d respectively). §8's "Component A closed" is superseded by §9.

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
| `extraction/common.py` | `CONDITION_VOCABULARY` (6 variables, single source of truth); rewritten `EXTRACT_PROMPT` (Python-only syntax, violation polarity with worked example); rewritten `VALIDATE_PROMPT` (checks 4: vocabulary/syntax, 5: polarity — corrects inverted conditions); `lint_condition()` AST linter + smoke-eval; `Rule.check_condition` field validator wired to it |
| `extraction/validate.py` | Startup guard: warns + prompts if stale `*_confirmed.jsonl` files (older than newest candidates) would be merged into the dedup |
| `tests/test_condition_lint.py` | 27 tests covering accept/reject cases, `Rule`-schema integration, prompt rendering — all passing |
| `scripts/dump_base_kv.py` | New: writes per-line base kV sidecar per env tag (backend method on the research PC, `--empirical` from JSONL normal-frame medians anywhere) |
| `data/grid_dataset_{neurips2020,case14}_basekv.json` | Generated (empirical method) |
| `rules/v1_archive/` | All v1 extraction outputs moved here (out of the dedup glob's reach); `rules/` is clean for the re-run |

> **Superseded (2026-08-10).** This section originally described a **2-stage** pipeline in which
> `EXTRACT_PROMPT` itself enforced the closed vocabulary and instructed the model to skip
> inexpressible constraints. That was tried and **failed**: on frequency/timing-heavy standards
> (PRC-006, PRC-024, PRC-025, PRC-029) the model correctly answered `[]` for almost every chunk
> and extraction yielded **zero rules**. The pipeline is now **3 stages** — see below.

### The three-stage pipeline (current)

| Stage | Script | Schema | Job |
|---|---|---|---|
| 1 | `extraction/extract.py` | `RawRule` / `lint_condition_raw` — Python syntax only, **any** variable name | cast a wide net; write `*_candidates.jsonl` |
| 2 | `extraction/translate.py` | `Rule` / `lint_condition` | map arbitrary engineering variables → the 6 Grid2Op variables; write `*_translated.jsonl` + `*_untranslatable.jsonl` |
| 3 | `extraction/validate.py` | `Rule` | verify against source text, correct polarity; write `*_confirmed.jsonl` + `*_flagged.jsonl`, then dedup |

The closed vocabulary is enforced at **Stage 2 and 3 only**. Keeping Stage 1 open is load-bearing:
it is what makes `*_untranslatable.jsonl` a meaningful audit trail (and the thesis negative result
— "most grid-code content is not Grid2Op-observable") rather than an empty file.
`tests/test_condition_lint.py::test_extract_prompt_stays_open_vocabulary` guards this.

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

## 3b. Session of 2026-08-12 — extraction unblocked, regen IN PROGRESS

**Status at time of writing: the v2 extraction run is still executing on the research PC.**
Numbers below are from the first 2 of 10 documents plus a 4-document validation run on the
personal PC. `rules/` is otherwise empty (cleaned; `v1_archive/` untouched).

### What was broken (three defects, all fixed)

1. **`EXTRACT_PROMPT` had been re-closed against `CONDITION_VOCABULARY`** — reintroducing the
   2-stage design that §3's superseded note already records as failing. Zero rules.
2. **`_strip_think` was silently corrupted** — `<think>.*?</think>` had become
   `" thinking.*? response"` (angle-bracket tags eaten by an editor pass). Latent until the
   extractor moved to a newer Qwen; reasoning traces then reached `extract_json_array`, whose
   `find("[")` latched onto a bracket inside the trace → `No JSON array found`.
3. **`DEVICE == "cuda"` is always False** (`DEVICE` is a `torch.device`) — the research PC was
   silently selecting the 9B personal-PC model.

Fixes: `generate_no_think()` in `common.py` suppresses thinking at the API level (`think=False`,
the way `validate.py` always did) rather than via the `/no_think` prefix, which newer Qwen builds
ignore; `_strip_think` restored and hardened against unclosed blocks; `DEVICE.type` +
`EXTRACTOR_MODEL` env override; `num_predict` 4096 → 6144 (rule-dense chunks were truncating
mid-array); `extract.py --debug-raw` dumps full responses to `<out>/_raw/` on parse failure.
Tests: 47 passing.

### The fourth defect — the escape hatch, not just the vocabulary

Opening the vocabulary was **necessary but not sufficient**. Extraction still returned `[]` until
`EXTRACT_PROMPT` rule 4 was reworded. At `temperature=0.0`, greedy decoding on a smaller model
locks onto the shortest defensible answer, and *"if no safety constraint is present, return []"*
makes `[]` always defensible. Measured on a chunk containing an unambiguous limit
(NC RfG p.20, "50.2 Hz and 50.5 Hz with a Droop in a range of 2–12%"):

| Prompt variant | Result |
|---|---|
| escape hatch present | `[]` |
| escape hatch removed | extracts correctly, right polarity, right citation |
| `/no_think` removed | `[]` (no effect) |

Rule 4 is now an **exhaustive directive** with a narrow carve-out (front-matter and purely
administrative clauses only). 9B validation run: 4 PDFs, 423 chunks → **155 candidates**
(previously 0), 1 tautology, 0 contradictions.

⚠️ The current wording is tuned for the 35B (unit-list crutch removed, escape hatch slightly
widened, anti-overlap line added). The blunter 9B-validated variant is in git history if the
35B under-yields.

### Measured quality of the v2 candidates (first 2 documents, 35B)

| Metric | Value |
|---|---|
| Rules extracted | **789** (NC RfG 218, Complete Grid Code 571) — v1 was 469 from *all ten* docs |
| Fully translatable (vocabulary-only variables) | **75 (9.5%)** — v1 was 64 of 469 (13.6%) |
| Partial (mappable + unmappable mixed) | 60 |
| Not translatable | 651 |

The 9.5% is a denominator artifact of much higher raw yield, not a regression — in absolute terms
2 documents already beat v1's entire corpus. Dominant variables: `frequency_hz` (215),
`voltage_pu` (207), `time_seconds` (87), `power_factor` (36), `loading_pct` (18). Frequency
dominance is inherent to grid codes and is the thesis negative result, not a pipeline defect.

### ⚠️ BLOCKING FINDING — Issue 2 has recurred in a new form

**14 of the 75 translatable rules (19%) evaluate TRUE on a healthy grid** (voltage 1.0 pu, no
trips, 40% loading). Examples straight from the candidates:

```
voltage_pu < 0.7   or voltage_pu > 0.9
voltage_pu < 1.05  or voltage_pu > 1.10
voltage_pu < 1.118 or voltage_pu > 1.15
voltage_pu < 0.05  or voltage_pu > 0.15
voltage_pu < 0     or voltage_pu > 1        # severity: critical
```

These are **ride-through bands inverted as though they were normal-operating bands**. This is a
subtler failure than v1's: prompt rule 3 correctly says *"if the source states a required or
normal operating range, INVERT it"* — right for "voltage shall remain within 0.95–1.05", wrong
for "the module shall remain connected between 0.7 and 0.9 pu for 3 seconds", which is an
*abnormal-but-must-survive* range. Inverting the latter yields a predicate that fires at nominal.
Grid codes are full of the second kind, and two different models have now made this mistake.

**Neither `translate.py` nor `validate.py` will catch this as currently written.** Both prompts
test for "does this describe the healthy state"; these conditions do not — they already look like
violation predicates and will pass through unchanged into `all_rules_deduped.jsonl`.

---

## 3d. Session of 2026-08-15 — stage 1 complete, guard + shield built

Extraction finished: **2,463 candidates across all 16 documents**, in `rules_35b/` (kept pristine
as the stage-1 archive; `rules/` stays empty until translation writes into it).

### Full-corpus measurements

The 2-document sample in §3b generalised. Measured across all 16:

| Measure | Value |
|---|---|
| Stage-1 candidates | 2,463 (v1: 1,372) |
| Conditions using only vocabulary-mappable variables | 195 (7.9%) |
| Distinct (entity, condition) pairs in that pool | **128** — the realistic post-dedup ceiling |
| Of that pool, fires at synthetic nominal (1.0 pu / 40%) | 27 (13.8%) |
| Distinct variable names across the corpus | 2,083 |

Dominant variables remain `voltage_pu` (354), `frequency_hz` (346), `time_seconds` (268),
`power_factor` (57), `droop_pct` (33). Frequency dominance is inherent to grid codes and is the
thesis negative result, not a pipeline defect.

Entity skew worth knowing before the KG redesign: of the 195 usable rules, **113 are
`PowerGeneratingModule`**, 21 `Bus`, 12 `ParkModule`, and only **2 `Line`**. `build_kg.py`'s current
else-branch routes all the module/facility entities to the `Grid` node, so entity-based retrieval
is close to degenerate — nearly every rule applies to every prediction. That is not harmful (the
shield gates a graph-level classification, and conditions do the discriminating), but it means a KG
redesign should not assume entity routing carries information.

### The fire-rate criterion, now settled by data

§3c's choice of a fire rate over real frames instead of a synthetic healthy band was the right one,
and the numbers now prove it rather than argue it:

- `voltage_pu_min < 0 or voltage_pu_max > 1` — a real extracted rule, severity `critical` —
  evaluates **False** at a synthetic 1.0 pu point, so a band-based filter ships it. Measured against
  real healthy case14 frames it fires at **0.714**: `voltage_pu_max` exceeds 1.00 on **71.8%** of
  them. Only the fire rate catches it.
- `loading_pct > 80` is a legitimate near-limit rule, yet fires on **14.1%** of healthy case14
  frames, because that grid genuinely runs hot (normal-frame `rho_max` median 0.72, max 0.99). A
  cutoff anywhere near zero would cull it.

Both figures were reproduced end-to-end by the guard itself against the real dataset.

⚠️ The observed distribution is **not cleanly bimodal** — those two rules land at 0.71 and 0.15,
in the middle §3c hoped would be empty. The guard prints a warning listing everything between 0.1
and 0.9 rather than pretending the gap exists; inspect that list before trusting the cutoff.

### What now exists in code

| Path | What it is |
|---|---|
| `extraction/polarity_guard.py` | **Stage 2.5.** Library-first (`fire_rate`, `assess_rules`, `partition`, `assert_no_rule_fires`) plus a CLI. Reservoir-samples ~5k `normal` frames per topology, caches them to `data/healthy_contexts_<tag>.json` |
| `shield/context.py` | `build_context` — the §5 voltage contract, single definition shared with the guard |
| `shield/evaluator.py` | `evaluate_condition` → VIOLATED / SATISFIED / NOT_EVALUABLE / ERROR; only VIOLATED blocks |
| `shield/shield.py` | `validate()`, `RuleProvider`, `JsonlRuleProvider` (+ `assert_clean` load-time backstop) |
| `evaluation/eval_shield.py` | Dual-arm GNN-only vs GNN+shield eval, failure-mode logging |
| `evaluation/summarize_shield_results.py` | §7 thesis tables + explanation review sheet |
| `tests/test_polarity_guard.py`, `tests/test_shield.py` | 42 new tests; 89 total green |

Two deliberate departures from this document, both recorded here so they are not read as drift:

1. **The guard is a standalone stage 2.5, not inline in `translate.py`** as §3c item 1 specified.
   Reasons: the cutoff is meant to be chosen *after* plotting the distribution, and inline welds
   that choice to the LLM pass; `wcci2022` does not exist yet, so the guard must be re-runnable on
   a new topology without re-invoking the translator; and the filtered-vs-unfiltered counterfactual
   needs both rulesets on disk. It writes into `<translated>/guarded/`, reusing the
   `*_translated.jsonl` name so `validate.py`'s glob picks it up unchanged while the unfiltered
   originals survive alongside.
2. **The shield does not use `build_kg.py::get_rules_for_entity`**, superseding §6. Rule retrieval
   sits behind the `RuleProvider` protocol so the KG can be rebuilt from scratch against the
   surviving ruleset without touching the shield or the eval harness.

### Prompt fixes applied (§3c item 2 — done)

`TRANSLATE_PROMPT` and `VALIDATE_PROMPT` now name the ride-through case explicitly, contrasting a
normal-operating range (invert it) against a ride-through / no-trip-zone range (do **not** invert —
the violation is tripping *inside* the zone, which needs protection status and duration, neither
observable). Both prompts also carry a healthy-grid self-check instructing the model to substitute
nominal values and reject anything that comes out TRUE. `EXTRACT_PROMPT` was **not** touched —
stage 1 is finished, and §3 records that editing it has twice produced zero-yield runs.

---

## 3e. Session of 2026-08-15 (later) — rules carry TWO roles; vocabulary expanded

Two corrections to the design, both from the user, both larger than they look.

### 1. Rules are not only prohibitions

The shield cannot read the corpus at inference, so the KG must carry **both**
directions of evidence. A rule is now one of:

- **CONSTRAINT** — condition TRUE ⇒ the standard is violated ⇒ BLOCK.
- **AFFIRMATION** — condition TRUE ⇒ telemetry is consistent with the class named in
  `affirms` (`normal | overload | line_trip | cascade`). Supporting evidence.

Both prompts previously *ordered* the model to invert any "required or normal
operating range" into a violation predicate. That instruction was itself generating
the defect §3d blamed on the model: "voltage shall remain within 0.95–1.05" is an
affirmation of `normal`, and inverting it yields a predicate that fires at nominal.

**Semantics chosen: Option A — affirmations never block.** Absence of support is
recorded as `unsupported` and reported, but only a constraint violation produces
BLOCK. Option B (gating on missing support) was rejected as the *deployed* behaviour
because its false block rate would be a function of how evenly the corpus happens to
cover the four classes, which is a property of the standards, not of the grid.
`eval_shield.py` computes the Option B counterfactual on every run
(`summary["affirmation"]`), so the gating variant can be judged on measured coverage.

Consequences, all implemented and tested:
- `Rule` gains `role` (default CONSTRAINT) and `affirms`; an AFFIRMATION without a
  class is rejected, a CONSTRAINT with one is rejected.
- The guard is **role-aware**: constraints are scored on `normal` frames and must not
  fire; affirmations are scored on frames **of the class they affirm** and must fire
  (`--min-support`, default 0.5). `sample_contexts()` now reservoirs all four classes.
- `validate()` is two-channel, returning `supporting_rules`, `contradicting_rules`,
  `n_affirmations_for_class`, `unsupported`.
- The guard reports affirmation coverage per class and names classes with none —
  precisely the gap that would make Option B unusable.

### 2. The 6-variable vocabulary was the wrong bar

The vocabulary used only `rho`, `v_or`, `line_status` — ignoring `p_or, q_or, p_ex,
q_ex, v_ex, load_p, load_q, gen_p, gen_q`, which every record already carries.
Measured against what is actually derivable, **693 of 2,463 candidates (28.1%) are
expressible, not 195 (7.9%)**. The earlier figure measured the vocabulary, not the
simulator.

Vocabulary expanded 6 → 14, adding `active_power_mw_max`, `reactive_power_mvar_max`,
`apparent_power_mva_max`, `power_factor_at_max_load`, `current_a_max`,
`total_generation_mw`, `total_load_mw`, `generation_load_imbalance_pct`.

Confirmed against the Grid2Op docs: `a_or`/`a_ex` are native (current needs no
derivation once the dataset carries them), `timestep_overflow` + `delta_time` make
*sustained-overload* durations expressible, and **frequency is genuinely absent** —
Grid2Op targets human-timescale control and quasi-static power flow assumes uniform
system frequency, so the 625 FREQ rules are irreducible.

⚠️ **`power_factor_min` was a derivation bug, caught by the guard on real data.**
Defined as the minimum across energized lines it has median **0.000** on healthy
case14 frames — lightly loaded lines carry near-pure reactive flow, so the minimum is
dominated by electrically irrelevant lines and a rule `pf < 0.9` fired on 100% of
healthy frames. Replaced with `power_factor_at_max_load` (median 0.930, p5–p95
0.926–0.936). Worth noting as evidence that the guard catches *context* defects, not
only rule defects.

**Revised projection: ~200–250 final rules** (was ~50), roughly balanced between
constraints and affirmations. Still a projection — `*_untranslatable.jsonl` and
`polarity_guard_summary.json` replace it with measurement.

### Not pre-filtering before translation

`translate.py` sends **all 2,463** candidates; there is no gate. Cost is per *chunk*
(862 calls), not per rule, so filtering saves nothing. And a keyword heuristic
deciding translatability is exactly the error that produced 195 — it would also
corrupt `*_untranslatable.jsonl`, where "the model judged this inexpressible" is a
result and "we never asked" is not.

---

## 3c. TODO — items 1 and 2 are DONE (see §3d); items 3 and 4 remain

Items 1 (the guard) and 2 (the prompts) are implemented and tested. Item 3 is superseded: the KG is
being **redesigned from scratch** after translation and validation, against the rules that actually
survive, rather than rebuilt with the existing mechanism. Item 4's shield is built; only its
real-KG smoke test is outstanding. The original text is kept below as the design record.

1. **Build the healthy-frame guard** (the fix for the finding above). Deterministic,
   LLM-independent, mirrors `lint_condition`'s smoke-eval: evaluate every candidate condition
   against *healthy* observations. Any condition returning TRUE is wrong-polarity or
   wrong-threshold **by definition** — a violation predicate must be False on a healthy grid.
   Would have caught all 14 of the above and every v1 inversion.

   **Where it lives:** an explicit filter step in `translate.py` (after translation, before the
   rule is written to `*_translated.jsonl`), plus a load-time assertion in the shield's KG
   loader as backstop. Not a `Rule`-schema validator — that protects everything downstream
   automatically but rejects silently inside Pydantic with nowhere to record what was dropped.

   **DECIDED (2026-08-12): use per-rule fire rate over real frames. Do NOT use a synthetic
   healthy band.** An earlier draft of this section specified a hand-written band
   (voltage_pu 0.98–1.02, loading_pct 20–60). Those numbers were *invented while writing the
   doc* and are not derived from the data — do not reinstate them as a filter criterion.

   The criterion is instead:

   ```python
   # ground-truth healthy states: frames get_state_label() labelled "normal"
   # (max rho < 1.0 AND every line in service) — 61,444 of them in the training set
   contexts  = [build_context(f, base_kv) for f in sample(normal_frames, 5000)]
   fire_rate = sum(evaluate(rule.condition, c) for c in contexts) / len(contexts)
   ```

   | fire rate | interpretation | action |
   |---|---|---|
   | ~1.0 | fires on essentially every healthy frame — polarity inverted | **reject** |
   | 0.01–0.5 | fires occasionally; likely a legitimate near-limit rule (`loading_pct > 80`) | keep, record the rate |
   | 0.0 | never fires on a healthy grid | pass |

   Pick the cutoff **after** plotting the distribution — it will almost certainly be bimodal
   (clustered near 0 and near 1) with an empty middle to cut in. This trades six guessed numbers
   for one threshold chosen from observed data.

   Why this beats a band:
   - **No invented parameters.** "Rejected because it fires on 97% of simulator-labelled normal
     frames" is evidence; "rejected because it fires at 1.0 pu" is an assumption with extra steps.
   - **It separates broken rules from correct-but-tight ones.** A band rejects both
     `voltage_pu < 0.7 or voltage_pu > 0.9` (broken) and `loading_pct > 80` (valid) or neither,
     depending where the edges were drawn. The fire rate distinguishes them without being asked.
   - **It yields the Phase-7 metric for free.** A rule's fire rate on ground-truth `normal`
     frames *is* its per-rule false block rate. Combining them predicts the shield's overall
     false block rate **before the shield is built**, and names the rules that drive it. Persist
     the per-rule rate as metadata — it feeds §7's false-block analysis directly.

   **Run it on every available topology, not just the training one.** `neurips2020` and `case14`
   have different base-kV profiles, and a rule that stays quiet on one can fire on the other —
   which is precisely the cross-topology failure the thesis measures. Re-run once `wcci2022`
   exists. Treat any rule whose fire rate jumps sharply between topologies as a **finding worth
   reporting**, not merely something to filter.

   **Keep a synthetic band for one purpose only:** hand-written contexts (clean / overloaded /
   tripped-line) as fixtures in `tests/test_shield.py`, so the suite need not load a 300k-record
   dataset. That is the band's sole legitimate remaining job — never the filter criterion.

   Cost: ~5,000 frames × ~300 rules ≈ 1.5M evaluations, well under a minute with `compile()`
   caching per condition. No GPU, no LLM — runs on the personal PC from the dataset and the
   existing base-kV sidecars.

   **Reject to an audit file, do not drop silently.** Write rejects to
   `*_polarity_rejected.jsonl` in the same shape `translate.py` already uses for untranslatables
   (`{"rule": ..., "reason": ...}`), with the failing probe recorded in `reason`. The rule is
   excluded from the KG either way — the only question is whether a copy survives, and it should,
   for four reasons:

   - **It converts a silent filter into a measurable result.** "LLM rule extraction inverts
     fault-ride-through bands at ~19%, caught by a deterministic post-check" is a reproducible
     finding, but only with the numerator and denominator on disk. Consistent with how
     `gnn_final_results.md` and §3's open-vocabulary result are already used.
   - **It supplies the counterfactual that justifies the guard.** Not "we filter bad rules" but
     "the guard removed N of M; unfiltered, the shield's false block rate is X% vs Y% filtered."
     Same delta structure as §1's GNN-only vs GNN+shield headline, one layer down — and
     computable only if the rejected rules can still be loaded.
   - **It separates two defects that are indistinguishable after rejection.** Wrong *polarity*
     (ride-through inverted — mostly unrecoverable) vs right polarity with a wrong *threshold*
     (`voltage_pu > 0.9` where a digit was lost — often recoverable by a correction pass). The
     second class may be salvageable, which matters at this pool size.
   - **It is the only way to detect over-filtering** of the guard itself (see the empirical-band
     point above).

   ⚠️ Do **not** name the file `*_confirmed.jsonl` — `deduplicate_rules` globs that pattern and
   would merge the rejects straight back into `all_rules_deduped.jsonl`.
2. **Update `TRANSLATE_PROMPT` / `VALIDATE_PROMPT`** to name the ride-through case explicitly, so
   the models stop generating it rather than relying solely on the guard to catch it.
3. **Rebuild the KG** — `python extraction/build_kg.py --rules rules/all_rules_deduped.jsonl
   --out-dir kg/`. Verify the Rule-node count matches the post-guard ruleset, not the raw dedup.
4. **Build the shield** per §6, with `tests/test_shield.py` first. Add a KG-load-time assertion
   that no loaded rule fires on a healthy context — cheap insurance against a contaminated
   ruleset reaching inference.

**Verification gate before the shield is trusted:** false block rate on ground-truth `normal`
frames must be near zero. With the current 19% contamination it would be near 100%, which is
exactly the v1 failure mode this whole regeneration exists to eliminate.

---

## 4. Manual steps before the build (user, research PC)

> **Updated 2026-08-15.** Step 1's extraction is **done** — do not re-run it. The remaining
> pipeline is translate → guard → validate, with the guard (stage 2.5) inserted between them:
> ```bash
> python extraction/translate.py     --candidates rules_35b/ --out rules/
> python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14 --report
> python extraction/polarity_guard.py --translated rules/ --tag neurips2020 --tag case14
> python extraction/validate.py      --candidates rules/guarded/
> ```
> Run the guard with `--report` first: it writes nothing and prints the fire-rate distribution, from
> which the `--max-fire-rate` cutoff should be chosen. The second call applies it. The guard needs
> no GPU and no Ollama, so it can equally run on the personal PC. The KG build that used to follow
> is deliberately omitted — see §3d.

1. **Re-run extraction** — all three stages, in order (Ollama models live there; sequential,
   never both loaded):
   ```bash
   python extraction/extract.py   --docs data/documents/ --out rules/
   python extraction/translate.py --candidates rules/
   python extraction/validate.py  --candidates rules/
   ```
   Expected artifacts: fresh `rules/*_candidates.jsonl`, `*_translated.jsonl`,
   `*_untranslatable.jsonl`, `*_confirmed.jsonl`, `*_flagged.jsonl`,
   `rules/all_rules_deduped.jsonl`, run summaries. Expect a much smaller but ~100% evaluable
   ruleset (Stage 2 + the linter cut everything inexpressible, into `*_untranslatable.jsonl`).
   Keep `rules/v1_archive/` untouched — if the stale-output guard fires, something from v1
   leaked back into `rules/`.

   If Stage 1 reports `raw=0 valid=0` across the board, `EXTRACT_PROMPT` has been re-closed
   against `CONDITION_VOCABULARY` — see the superseded note in §3. If it reports
   `Extractor failed: No JSON array found`, reasoning traces are leaking into the parser;
   re-run with `--debug-raw` to dump full responses to `rules/_raw/`.
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
exists and every condition passes `lint_condition`; **every condition also passes the
healthy-frame guard (§3c item 1) — this is the one that is currently failing at 19%**;
`kg/knowledge_graph.pkl` rebuilt *after* the guard has filtered the ruleset;
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

## 6. Component D build plan (the shield) — BUILT 2026-08-15

> **Status: implemented**, with one deviation. The `shield/shield.py` bullet below specifies reusing
> `extraction/build_kg.py::get_rules_for_entity` with a fault_type→entity map. That is
> **superseded**: `validate(context, rules)` takes a plain rule list behind a `RuleProvider`
> protocol, and `JsonlRuleProvider` serves `all_rules_deduped.jsonl` with no graph at all. This
> exists so the KG can be redesigned from scratch (§3d) without touching the shield or the eval
> harness — a `KgRuleProvider` slots in later. The entity map is moot in any case: 113 of the 195
> usable rules are `PowerGeneratingModule`, which the current builder routes to `Grid` regardless.
> Everything else in this section was implemented as written.

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

## 7. Test-harness build plan (final testing phase) — BUILT 2026-08-15, not yet run

> **Status: implemented as `evaluation/eval_shield.py` and
> `evaluation/summarize_shield_results.py`.** Both import cleanly and expose their CLIs; neither has
> been run, because they need `rules/all_rules_deduped.jsonl`. Rule retrieval goes through
> `RuleProvider` (§6), so the KG redesign lands without editing either file. `--limit N` was added
> to `eval_shield.py` for a cheap first smoke run.

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

---

# 9. TASK REDESIGN (2026-08-15) — Component A REOPENED

> **This section supersedes §8's "Component A closed".** Read it before touching
> anything in `training/`, `scripts/generate_dataset.py`, or the shield's class names.

## 9.1 Why: the 4-class target is closed-form

`get_state_label()` in `scripts/generate_dataset.py` defines the labels as:

```python
if max_rho >= 1.0:                  return "overload"
if active_lines == env.n_line:      return "normal"
if active_lines == env.n_line - 1:  return "line_trip"
return "cascade"
```

Every branch is expressible in the shield's own vocabulary (`rho_max`,
`n_tripped_lines`). **Measured: four ordered rules reproduce the stored labels with
100% agreement on 55,000 records** — 15,000/15,000 on case14 and 40,000/40,000 on
neurips2020.

Consequences, all of them bad for the thesis as originally framed:

1. A symbolic layer can score **100%** where the trained GATv2 reaches **0.8277**.
   The comparison "does the shield add value over the GNN" is rigged before it runs.
2. The GNN already receives `max_rho` and `global_trip_frac` as node features, so
   after max-pooling the answer is directly available. The documented
   `normal`/`line_trip` confusion is exactly the `n_tripped_lines == 0` vs `== 1`
   distinction that triple pooling averages away. Three rounds of architecture
   experiments were fighting to learn a threshold already present in the input.
3. `fault_loc` is `obs.rho.argmax()` — the localization target is closed-form too.

This is not a bug in the code; it is a property of the labelling scheme. It is worth
reporting **as a finding**: *the fault-labelling scheme commonly used on
Grid2Op-derived datasets yields a target that is a closed-form function of the
observation, and therefore cannot distinguish a learned model from a threshold.*

## 9.2 The fix: a binary forecast task

Ask a question the present state does **not** determine:

> **`at_risk`** = does the grid leave the `normal` state within the next `H` steps?

Faults are injected stochastically (`FAULT_PROB = 0.05`) and load evolves, so this is
genuinely predictive. Crucially it also **un-rigs the shield**: rules see only the
present, the label depends on the future, so no rule can restate the labeller.

**Both tasks are kept and reported.** The classify task carries the closed-form
finding; the forecast task is the corrected experiment. The narrative is "we built the
classifier, proved the target was closed-form with a 4-rule baseline, and redesigned
the task" — a diagnosis and a fix, not just a system.

## 9.3 Data generation — IMPLEMENTED

`scripts/generate_dataset.py --task forecast --horizon H`

| Change | Detail |
|---|---|
| **No subsampling** | `NORMAL_KEEP_PROB` / quotas are bypassed in forecast mode. A dropped frame is a hole in an earlier frame's lookahead window. Balance is handled by class weights at training instead. |
| **Per-chronic buffering** | Frame `t`'s label depends on `t+1..t+H`, so a chronic is buffered and labelled on completion (`flush_forecast_chronic`). |
| **Trailing frames dropped** | The last `H` frames of each chronic have an undefined lookahead window. Labelling them `not_at_risk` would mark the end of every chronic as safe. |
| **Separate tag** | Writes `grid_dataset_<env>_forecast.jsonl` — never overwrites the classify set. |
| **Extra fields** | `risk_label`, `risk_int`, `horizon`, `steps_to_fault`. `label`/`label_int` are retained for reference but are NOT the target. |
| **Meta override** | `label_map` becomes `RISK_LABEL_MAP`, `n_classes` becomes 2. `train_gnn.py` reads both from the meta, so training switches automatically with no flag. |

`steps_to_fault` is recorded on every frame, so **the horizon can be re-tuned without
regenerating** — relabel from the existing file.

⚠️ **Why the old data cannot be relabelled:** measured contiguity is 61.6% (case14)
and 88.0% (neurips2020) — `NORMAL_KEEP_PROB = 0.02` punched holes throughout. First
chronic timesteps read `[5, 6, 9, 10, 11, 20, 33, ...]`. Relabelling only intact
stretches would bias toward dense (fault-heavy) regions.

Tests: `tests/test_forecast_labels.py` (12 tests, incl. a brute-force cross-check at
four horizons). This code runs unattended for hours — an off-by-one is expensive.

## 9.4 GNN — IMPLEMENTED

Architecture is **unchanged**. Only the target changes.

| File | Change |
|---|---|
| `training/config.py` | `TASK` (env var `GRID_TASK`, default `classify`); per-task `DATA_FILE`, `CHECKPOINT_FILE`, `NORM_STATS_FILE`, `PROCESSED_PT` |
| `training/train_gnn.py` | reads per-task artifact names; **localization loss disabled when `n_classes == 2`** — `fault_loc` is a present-state signal and would leak into a future-state objective |
| `scripts/preprocess.py`, `scripts/pyg_data.py` | use `risk_int` as the target when present |

**The frozen Component A artifacts are protected.** Forecast training writes
`gnn_checkpoint_forecast.pt` / `normalization_stats_forecast.pt` /
`processed_grid_data_forecast.pt`. `gnn_checkpoint_best.pt` (macro F1 0.8277) is never
touched.

Still to do at training time: recalibrate a Lever A margin for the new task if wanted
(the existing `gnn_logit_margin.json` is tuned to the 4-class distribution and must
not be reused), and re-check class balance — see §9.7.

## 9.5 Shield — small change, better job

The shield's role shifts from **verify** to **challenge**. It cannot confirm a
prediction about the future; it can ask whether the present state is plausible grounds
for it.

**Asymmetric gating — the core design point:**

| GNN says | Telemetry | Shield |
|---|---|---|
| `not_at_risk` | stressed (over limit, voltage sagging) | **BLOCK** — model called it clear on a stressed grid |
| `not_at_risk` | nominal | pass |
| `at_risk` | stressed | pass — they agree |
| `at_risk` | nominal | **pass — must not block** |

That last row is load-bearing: predicting risk from a currently-calm grid is exactly
what learning is supposed to add (early warning). Blocking it destroys the model's
only real value. It is also the right safety posture — the costly error in grid
operation is a missed fault, not a false alarm.

Code impact:
- `FAULT_CLASSES` becomes `{"at_risk", "not_at_risk"}`
- `validate()` gains the asymmetry: apply constraint blocking only on the permissive
  prediction. A few lines.
- `context.py`, `evaluator.py`, guard core, tests: **unchanged**.

**Affirmation remap** (deterministic, lossless, no LLM):
```
normal                          -> not_at_risk
overload | line_trip | cascade  -> at_risk
```
Re-run the polarity guard afterwards to re-measure fire rates against the new labels —
minutes, personal PC. CONSTRAINT rules are untouched (no class binding).

⚠️ **Keep the 4-class vocabulary at translation time and collapse afterwards.**
"Does this standard describe an overload?" is a more natural judgment for a model
reading a grid code than "does this describe at-risk?". The collapse is one-way and
lossless; translating straight to binary would need a re-run to recover the finer
classes. **This is why the translation run is NOT blocked on the task decision.**

## 9.6 KG — unchanged by this, still to be redesigned

The task change does not affect the KG plan (§3d): still rebuilt from scratch against
the surviving ruleset, after translation and validation. One thing improves — with a
binary target the affirmation channel needs coverage for 2 classes instead of 4, which
was the gap the guard flagged (`no affirmations for ['line_trip', 'cascade']`).

## 9.7 Risks to watch

1. **Class balance.** Without subsampling, and with faults persisting until reconnect
   (`RECONNECT_PROB = 0.20`), `at_risk` could dominate. Check the split immediately
   after generation; tune `H` if it is extreme. `steps_to_fault` allows retuning
   without regenerating.
2. **Horizon too short → triviality creeps back.** If `H` is small, `at_risk`
   correlates strongly with the *current* state and the closed-form problem partially
   returns. The early-warning signal is the shield-alone baseline (§10.1) scoring
   near-perfectly.
3. **Dataset size.** No subsampling means far more frames per chronic. Cap with
   `--n_records` / `--max-chronics`; the workstation is RAM-limited and
   `PreloadedGridDataset` loads the whole `.pt` into memory.

## 9.8 Hardware change — personal PC now runs generation AND training

Supersedes `CLAUDE.md`'s "Research PC — all production runs".

| Machine | Now runs |
|---|---|
| **Workstation** | forecast dataset generation, GNN training, polarity guard, shield, all tests |
| **LLM host** | LLM stages only — `translate.py`, `validate.py` |

`training/config.py` auto-selects the small `[16,32,32]` / `heads=[4,4,1]` branch on
non-CUDA devices, which is the **proven deployed** config. The CUDA branch
(`[64,128,128]`) has never produced a working checkpoint. Training on the personal PC
is therefore the *safer* choice, not a compromise.

---

# 10. INTERPRETABILITY CONTROLS — carried forward, re-validated

These exist to prevent one specific failure: reporting **(b)** *"we couldn't build much
of a symbolic layer, so we can't tell whether it would help"* as if it were **(a)**
*"we built a sound symbolic layer and it didn't help"*. Only (a) answers the thesis
question. All three were designed before the task redesign; all three survive it, and
two improve.

**Not yet implemented — a later session.**

## 10.1 Ceiling analysis — the decisive one *(still valid; now meaningful)*

Fit a small decision tree on the 14 context variables. Two targets:

- **tree → `risk_label`**: can *any* function of present observables predict a fault
  within `H`? This is the upper bound on what any symbolic layer could achieve, and it
  is also the **shield-alone baseline**.
- **tree → "GNN was wrong"**: the upper bound on what any symbolic *gate* could catch.

Interpretation:
- tree can't beat chance → no rule system over observable state could catch these
  errors. Answer **(a)**, established without reference to rule count at all.
- tree does well but the shield doesn't → the information is there and the rules
  missed it; quantifies exactly how much was left on the table.

*Changed by the redesign:* under the 4-class task this was degenerate (a tree would hit
100% by rediscovering the labeller). Against a lookahead target it is a real
measurement. **Improved.**

## 10.2 Expert baseline *(still valid; substantially improved)*

Hand-write ~10–15 rules against the same 14 variables. Run as a **third arm, never
merged** into the corpus.

- expert rules also give ~0 delta → symbolic gating doesn't help regardless of
  provenance; extraction thinness is exonerated as the cause.
- expert rules give a real delta, extracted don't → the idea works, **extraction is
  the bottleneck**. Sharper and more publishable than either alone.

*Changed by the redesign:* under the 4-class task this control was contaminated — the
"expert" rules would have been the 4 label-definition rules, guaranteeing a perfect
score for trivial reasons, and a separate "trivial arm" was needed to quarantine them.
Against a lookahead target those 4 rules no longer reproduce the label; they become a
legitimate **persistence baseline** ("does the current state predict the future?").
The control is now non-circular. **Improved, and the trivial-arm split is no longer
needed.**

## 10.3 Rule-count ablation *(still valid, unchanged)*

Sample random subsets of the final ruleset — 5, 10, 20, 50, all N — and plot catch rate
against ruleset size.

- **flat** → 5 rules did as well as N; more rules would not have helped. Supports (a).
- **still rising at N** → genuinely truncated. Still (b), but **quantified**: "the
  delta was still increasing at N rules, so the corpus bound, not the hypothesis,
  limits this result." A measured (b) is a legitimate limitation; an unmeasured one is
  the hole.

Cheap — re-runs of `eval_shield.py` with rule subsets, no new rules needed.

## 10.4 The three-way comparison the redesign unlocks

With a lookahead target, **shield-alone vs GNN-alone vs GNN+shield** becomes a fair
comparison — neither side can see the future, so neither can cheat. Any outcome is
reportable:

| Outcome | Reading |
|---|---|
| shield-alone strong | simple thresholds predict faults; the GNN must beat that to justify itself |
| GNN-alone strong, shield adds little | learning captures something rules don't |
| GNN+shield beats both | the neuro-symbolic claim is supported, on a task it could not have won trivially |

This was impossible under the 4-class task, where shield-alone was 100% by construction.

## 10.5 Watch for label-definition restatements

Any surviving rule that restates the labelling function inflates results while
measuring nothing. Under the forecast task this is far less likely (the labeller
depends on the future), but check anyway and flag such rules in the results rather than
quietly counting them.

---

# 11. Sequencing

| When | Where | What |
|---|---|---|
| **Done (2026-08-15)** | personal PC | Guard, shield, eval harness, role/affirmation model, vocabulary 6→14, forecast task code. 118 tests green. |
| **Next** | personal PC | `generate_dataset.py --task forecast` → check class balance (§9.7) → `preprocess.py` → `train_gnn.py` with `GRID_TASK=forecast` |
| **Then** | research PC | `translate.py` (4-class vocabulary — not blocked on the task decision), then `validate.py` |
| **Then** | personal PC | polarity guard on the translated corpus → KG redesign → shield binary/asymmetric update |
| **Finally** | personal PC | `eval_shield.py` matrix + the §10 controls → `summarize_shield_results.py` |

```powershell
# personal PC — forecast dataset + training
python scripts/generate_dataset.py --env neurips --task forecast --horizon 6 --n_records 300000
python scripts/preprocess.py
$env:GRID_TASK = "forecast"; python training/train_gnn.py
```
