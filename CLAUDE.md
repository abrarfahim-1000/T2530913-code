# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A neuro-symbolic fault detection system for power grids combining:
- **GNN (Neural Layer):** Graph Attention Network trained on 36-bus NeurIPS 2020 topology
- **LLM Pipeline (Component B):** Qwen3-14B + Nemotron-3 Nano 30B extract rules from IEEE/NERC standards
- **Knowledge Graph (Component C):** BUILT 2026-08-20 — provenance graph (36 nodes / 40 edges) citing which standard authorises each block. The v1 topology graph (587 nodes / 6,632 edges) was DELETED, not archived.
- **Symbolic Shield (Component D):** Post-hoc gate — every GNN prediction validated against the rule corpus before output

The GNN is trained on one topology (36-bus) and evaluated on unseen topologies (14-bus, 118-bus) to measure cross-topology generalization. The shield's rule compliance rate is expected to remain stable while GNN accuracy degrades.

---

## Current Status (2026-08-15)

> **Start here for results:** [`supplimentary_docs/thesis_findings.md`](supplimentary_docs/thesis_findings.md)
> — the plain-language account of every measured finding, with the caveats that must travel
> with them. Written for a reader without the codebase open.
>
> **ALL FOUR COMPONENTS ARE BUILT as of 2026-08-20.** Every Component B stage has run; the
> corpus to cite is `validated_translated/all_rules_deduped.jsonl` (4 rules), and Component C is
> `kg/knowledge_graph.json` — see `component_d_plan.md` §16. What remains is optional robustness,
> not construction (§16.8).
>
> **The operative plan is [`supplimentary_docs/component_d_plan.md`](supplimentary_docs/component_d_plan.md).**
> Read it before starting work — it carries the task redesign, the runbook, and the landmines.
> This file describes the *architecture*; the plan describes *what to do next*.
> `archive/component_d_handoff_archive.md` is retired — do not follow it.

> **2026-08-16 — the task is now N-1 CONTINGENCY SCREENING (`--task n1`).** Two earlier task
> designs were built, probed and rejected *before* generating against them; the full evidence chain
> is [`component_d_plan.md`](supplimentary_docs/component_d_plan.md) §1.1. Short version: the
> 4-class `classify` target is closed-form (4 if/else rules = **100% agreement, 0 disagreements
> in 315,000 records** — see "Why classification was abandoned" below), and the binary `forecast`
> target is either unlearnable (any-fault: 1.00× the all-positive baseline at every H ≥ 3, because
> trip onset is an unconditional coin flip) or exhausted by one threshold (overload-only: model
> 0.163 vs rule 0.160). N-1 screening is neither — global `rho_max` scores 1.04× baseline, 100% of
> frames are mixed, and a network-aware model reaches 0.868 against a best-rule 0.608. **No rule
> over the present observation can restate an N-1 label, because producing it needs a power-flow
> solve** — which is what permanently un-rigs the shield comparison.
>
> **TRAINED (2026-08-16).** Edge-level head, 12,000 frames / 576 chronics / 702,618 contingency
> labels. Held-out test **F1 0.8872, AP 0.9549 — 1.91× the best single-rule baseline (0.4639)**;
> ablation shows message passing contributes **+0.058** over endpoint features alone. Full account
> in [`gnn_n1_tightening.md`](supplimentary_docs/gnn_n1_tightening.md).
>
> **CROSS-TOPOLOGY, measured 2026-08-16 — generalisation is PARTIAL and ASYMMETRIC.** Full table,
> caveats and an untested hypothesis in [`component_d_plan.md`](supplimentary_docs/component_d_plan.md) §7.6.
>
> | topology | lines | contingencies | best rule | **model** | AP |
> |---|---:|---:|---:|---:|---:|
> | neurips2020 *(in-dist)* | 59 | 113,205 | 0.4639 | **0.8972** — 1.93× | 0.9615 |
> | case14 *(unseen, smaller)* | 20 | 118,502 | 0.5392 | **0.4477** — **0.83×, FAILS** | 0.4270 |
> | wcci2022 *(unseen, larger)* | 186 | 742,472 | 0.4915 | **0.5721** — 1.16× | 0.6419 |
>
> Scaling **up** costs far less than scaling **down** — the opposite of the naive expectation.
> On case14 the model loses to the single-rule baseline outright; report it as such. Every figure
> is best-threshold *on its own topology* (agenda item 6 would lower all three). Note the eval
> script reads 0.8972/0.9615 in-distribution where the tightening doc records 0.8872/0.9549 —
> the checkpoint on disk is from a different run than the one written up; treat the checkpoint as
> authoritative.
>
> 🚨 **That doc also records a normalization bug affecting EVERY task, including the frozen
> classify checkpoint.** `compute_normalization_stats()` populates PyG's `_data_list` cache, so the
> subsequent in-place `_data.x` normalization never reached the DataLoaders — all training ran on
> raw unnormalized features while `normalization_stats.pt` was saved as if it had been applied.
> the (now removed) classify cross-topology script *did* normalize at inference, so the frozen
> classify checkpoint has a **train/inference mismatch**, and some share of its reported
> cross-topology degradation is attributable to that rather than to topology transfer. Fixed for
> N-1 (`_data_list = None` plus an assertion); read §3 of the tightening doc before citing any
> classify cross-topology number.

**Component A (GNN) — N-1 contingency screening, TRAINED and DONE.** The trainer, config and
feature builders carry the N-1 path *only*; the 4-class classifier and the binary forecast model
were retired on 2026-08-16 and their code lives in git history. Rationale for the demotion, and the
one page of methodology that survives into the thesis, are in
[`revised_thesis_claim.md`](supplimentary_docs/revised_thesis_claim.md) §2.

### WHY CLASSIFICATION WAS ABANDONED — the training was unnecessary, and it lost

This goes in the report. Two claims, both measured, and the second only lands because of the first:

**1. Four `if`/`else` statements reproduce the label with 100% accuracy, so there was nothing to
learn.** The 4-class target was a **closed-form function of the observation** — not approximable
by rules, *literally computable* from them:

```python
if max(rho) >= 1.0:                 return "overload"
if n_active_lines == n_line:        return "normal"
if n_active_lines == n_line - 1:    return "line_trip"
return "cascade"
```

**Re-verified on the FULL datasets 2026-08-20, immediately before deleting them:**

| dataset | records | rule == stored label | disagreements |
|---|---:|---:|---:|
| neurips2020 (36-bus) | 300,000 | **300,000 (100.000000%)** | **0** |
| case14 (14-bus) | 15,000 | **15,000 (100.000000%)** | **0** |
| **total** | **315,000** | **315,000 (100%)** | **0** |

⚠️ This supersedes the older "100% agreement on 55,000 records" figure quoted elsewhere in the
docs — the check was re-run over every record, not a sample. **Zero disagreements in 315,000.**

**2. The trained GATv2 scored 0.8277 — worse than the if/else.** Best macro F1 after three rounds
of architecture work and the Lever A logit calibration. The symbolic layer scored **100%**. A
neural network was beaten by four threshold comparisons on the task it was built for.

**The honest reading, which must travel with the claim.** This is not evidence that symbolic
methods beat neural ones at fault classification. It is evidence that **the benchmark was
mis-specified**: the labelling function was defined over the same observation the model was given,
so the label carried no information the input did not already contain. Any measurement of
"does the symbolic layer add value over the GNN?" on this task was rigged before it ran — the
symbolic layer *is* the labeller. That is exactly why the task was replaced with N-1 screening,
where producing the label requires a power-flow solve and no rule over the present observation can
restate it.

🚨 **ALL CLASSIFY ARTIFACTS WERE DELETED on 2026-08-20** (4.81 GB) — `gnn_checkpoint_best.pt`,
`gnn_checkpoint_leverA.pt`, `gnn_logit_margin.json`, `normalization_stats.pt`,
`data/grid_dataset_{neurips2020,case14}.jsonl`, `data/processed_grid_data.pt`,
`data/split_neurips2020_{train,val,test}_idx.npy`, and the retired `evaluation/eval_shield.py`.
The two `.json`/`.py` files were git-tracked and are recoverable from history; **the rest were
untracked and are gone permanently.** The table above is now the only record of the closed-form
result, which is why it was measured before deletion rather than after. Do not cite the 0.8277
figure as reproducible — it is not, and the checkpoint that produced it was already
undeterminable between two separate runs before they were removed. The three rounds of rejected
architecture experiments remain documented in
[`supplimentary_docs/archive/gnn_final_results.md`](supplimentary_docs/archive/gnn_final_results.md).

> **2026-08-20 — STAGE 3 RAN. THE PIPELINE IS COMPLETE, AND THE PROMPT WAS THE VARIABLE.**
> Validation rejected 31 of 32 guarded rules and recorded no reason for any of them, because
> `validate.py` counted rejections and discarded the `Verdict`. A controlled A/B over the same
> input then isolated the cause: the auditor's criterion 1 asked whether the constraint is
> *stated* in the source text — a category error against a stage-2 condition, whose whole
> purpose is to leave the standard's language. Outcome is perfectly nested:
>
> | arm | question | confirmed | rejected | distinct |
> |---|---|---:|---:|---:|
> | `strict` (reproduces run 1) | is it **stated** in the text? | 1 | 31 | 1 |
> | `translated` | is it a faithful **operationalization**? | **11** | 21 | **4** |
>
> The whole difference is the ten `loading_pct > 100` rules. `strict` rejected them citing
> "an undefined variable `loading_pct`" — a variable that is in the vocabulary block of that
> very prompt. **The verdicts were stable across runs while the reasoning was fabricated;
> only persisting the reasons exposed it.** The 21 shared rejections are substantive (PRC-024
> ride-through curves, ENTSO-E time-bound envelopes, role inversions) — though ≥3 are wrong
> and 4 more should have been CORRECT. Full account: `component_d_plan.md` §15.
>
> **The validated 4-rule corpus BEATS the guarded 32 on every metric on every topology:**
>
> | topology | guarded-32 Δ / precision | **validated-4 Δ / precision** |
> |---|---:|---:|
> | neurips2020 | +0.0026 / 0.514 | **+0.0082 / 0.938** |
> | case14 | +0.0021 / 0.922 | +0.0021 / 0.922 |
> | wcci2022 | +0.0655 / 0.860 | **+0.0676 / 0.934** |
>
> 🚨 **The "override precision *rises* off-distribution" claim is VOID.** It was an artifact of
> averaging voltage rules (0.13–0.20 precision) with thermal rules (0.92–0.94). Stage 3 removed
> the voltage rules on textual grounds, having never seen fire-rate data, and precision is then
> **flat at 0.938 / 0.922 / 0.934**. What changes off-distribution is how *often* the gate can
> speak (0.31% → 0.087% → 3.04%) and how much model error it can reach (16.2% / 0.51% / 30.9%),
> not how right it is. See `component_d_plan.md` §15.7 — do not restate the old version.
>
> **Convergence (§15.8):** an empirical filter (fire rates on real grid data, no LLM) and a
> textual auditor (source standards, no grid data) independently kept the same family. Neither
> "your cutoff is mistuned" nor "your prompt is wrong" survives both.

> **2026-08-19 — stage 2 FAILED on a parsing bug, was fixed, and RE-RAN clean. Guard + audit done.**
> Run 2: `total_no_verdict: 0`, `total_unparseable: 0` — every one of the 2,463 rules got a verdict.
> **82 translated (3.3%) → guard 33 kept (59.8% rejected) → audit 11 distinct rules can do work**,
> of which the only unambiguously useful family is `loading_pct > 100`. Surviving rules cover
> **4 of 14** observable variables and **nothing about topology** — see `component_d_plan.md` §4.2
> for the headline finding (the standards/simulator mismatch runs both ways) and the two caveats
> that must travel with it. New: `evaluation/audit_rules.py`. History of the original bug follows.
>
> **2026-08-19 — stage 2 ran and FAILED on a parsing bug; the fix is in, the re-run is pending.**
> 2,190 of 2,463 rules (88.9%) came back `NO_VERDICT` and only **6 translated**. Cause: every
> worked example in `TRANSLATE_PROMPT` omitted `rule_id` while `TranslationResult` requires it, so
> every entry failed validation and was dropped by a bare `except ValidationError: pass`. The model
> was answering correctly and the answers were being discarded. **The 0.24% yield says nothing
> about the corpus** — only 233 rules (9.5%) ever received a verdict. Fixed in
> `common.py` (rule_id in every example) and `translate.py` (`build_result_map`, positional
> fallback, no silent drops, `--debug-raw`, NO_VERDICT tripwire). Guarded by
> `tests/test_translate_mapping.py`. Full account: `component_d_plan.md` §3.5.
> ⚠️ `translated_rules/` on disk is the FAILED run — evidence only, do not feed it to stage 2.5.

**Component B (LLM extraction) — ALL FOUR STAGES DONE (2026-08-20).** 2,463 candidates across 16
documents in `rules_35b/` (kept pristine as the stage-1 archive). The pipeline is **four
stages**: `extract.py` (open vocabulary) → `translate.py` → `polarity_guard.py` (stage 2.5,
deterministic) → `validate.py`. The closed vocabulary is enforced at stages 2–3 **only**.

**End-to-end yield: 2,463 candidates → 58 translated → 32 guarded → 4 distinct validated rules
(0.16%).** The corpus to cite is `validated_translated/all_rules_deduped.jsonl`: three
`loading_pct > 100` CONSTRAINTs and one `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1`
AFFIRMATION of `normal`. Affirmation coverage is `normal` only for the third run running, so
the shield's Option B counterfactual stays unmeasurable — a settled property of the corpus.
`validated_strict/` is the counterfactual arm; keep it, do not feed it to anything.

The vocabulary is **14 variables**, not 6 — it previously used only `rho`, `v_or`, `line_status`
and ignored `p_or/q_or/gen_p/load_p`, which every record already carries. Measured against what is
actually derivable, **693 of 2,463 candidates (28.1%)** are expressible, not 195 (7.9%).

Rules carry a **role**: `CONSTRAINT` (true ⇒ violation ⇒ BLOCK) or `AFFIRMATION` (true ⇒ telemetry
consistent with the class in `affirms`). Forcing everything into constraint polarity was itself
generating inverted-band defects.

⚠️ **Do not re-close `EXTRACT_PROMPT` against `CONDITION_VOCABULARY`.** Tried twice; extraction
returned *zero* rules both times — on frequency/timing-heavy standards the model correctly answers
`[]` for nearly every chunk, starving stage 2. Guarded by
`tests/test_condition_lint.py::test_extract_prompt_stays_open_vocabulary`. Related: thinking must
be suppressed via `generate_no_think()` (API-level `think=False`), not the `/no_think` prefix
alone — newer Qwen builds ignore the prefix and the reasoning trace breaks JSON parsing.

v1 outputs are archived at `rules/v1_archive/` (retained as a thesis negative result).

**Component C (Knowledge Graph) — BUILT 2026-08-20, provenance-centric.**
`kg/knowledge_graph.json` — **36 nodes, 40 edges**:
`Document --contains--> Clause --states--> Rule --deduped_into--> ServedRule --instantiates-->
Predicate --reads--> Variable`. Topology-agnostic by construction, so **one graph serves all
three grids**.

**What it is for:** citation. `KgRuleProvider.cite(rule_id)` returns every clause, in every
standard, that states the rule the shield acted on. The measured headline it makes visible:
the thermal check is **one predicate, stated in 10 clauses across 4 documents from 2 identified
standards bodies** — corroboration that a flat rule list cannot express. The `Predicate` layer is
the graph's own contribution: dedup keys on `(entity, condition)`, which fragments one physical
check into three records because the extractor labelled it `Line` in one standard and `Facility`
in another, and wrote `100` in one place and `100.0` in another.

⚠️ **Retrieval through the KG is OPT-IN and returns the identical rule set.** `--rules-kg` was
verified against all three topologies: every field of `shield_<tag>_kg.json` matches
`shield_<tag>_validated.json`. Pinned by `tests/test_kg.py`. **Never let the graph filter by
entity** — `R_1443` is labelled `Facility` while an identical rule is labelled `Line`, so entity
routing silently drops rules. That, plus binding rules to one grid's bus/line indices, is what
made v1 useless (`Line` + `Bus` were 3.5% of the corpus; 6,375 of 6,632 edges carried nothing).
Guarded by `test_graph_is_topology_agnostic`.

🚨 **The v1 graph and `extraction/build_kg.py` (1,203 lines) were DELETED on 2026-08-20**, not
archived — recoverable from `git show a5c5199:extraction/build_kg.py`, the last commit that
carries them. Do not resurrect them.

**Component D (Symbolic Shield) — BUILT, tested, and RUN END-TO-END against the VALIDATED
corpus (2026-08-20).** All three topologies, threshold selected on the neurips2020 val split
and held fixed: **wcci2022 F1 0.5577 → 0.6253 (+0.0676), missed violations 68,166 → 47,101
(−30.9%) at 0.934 intervention precision**; neurips2020 +0.0082 (precision 0.938);
case14 +0.0021 (precision 0.922). The shielded wcci2022 score still exceeds the raw model's
own best-on-topology oracle ceiling (0.5721). Zero ERROR and zero NOT_EVALUABLE verdicts
everywhere; blocks are `high` severity only.

**Intervention precision is flat at 0.92–0.94 across all three grids** — the gate enforces a
physical doctrine, not a learned pattern, so its accuracy does not depend on topology. What
changes off-distribution is how *often* it can act and how much of the model's error sits
where the doctrine can reach it. ⚠ Two caveats travel with this: `loading_pct > 100` on the
base case is close to a physical tautology (P(violation | base overloaded) = 91–96%, not
100%), and the N-1 doctrine it enforces is hand-written in `validate_n1`, not extracted.
Full account and both caveats: `component_d_plan.md` §15.6–§15.9. The guarded-32 numbers in
§11 are superseded but retained — the diff between the two corpora is itself the evidence.

**Live harness is `evaluation/eval_shield_n1.py`.** The classify-era `eval_shield.py` was
DELETED on 2026-08-20 along with the artifacts it needed (recoverable from git history).
`shield/` (context, evaluator, shield), `extraction/polarity_guard.py`, `kg/` (Component C).
`evaluation/summarize_shield_results.py` was DELETED on 2026-08-20 — it read a retired schema
from a `results/` directory that did not then exist. Now that `results/` *does* exist it would
have half-worked, which is worse than being orphaned; recoverable from git history.
**222 tests green.**
Rule retrieval sits behind a `RuleProvider` protocol (`JsonlRuleProvider` by default,
`kg.provider.KgRuleProvider` opt-in),
so the KG redesign cannot invalidate it. Pending: the binary/asymmetric update for the forecast
task (plan §6.1).

Binding voltage contract (plan §5): per-line base kV from `data/grid_dataset_<tag>_basekv.json`
with energized-line masking. The flat `v_or/150.0` conversion in `archive/study3(integration).md` §4.5 is
**superseded** — case14 runs lines at ~20 kV and ~138 kV, and even the 36-bus grid has 7 lines at
~365 kV.

---

## Development Commands

### Environment Setup
```bash
# Install PyTorch with CUDA support (LLM host)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch Geometric dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install remaining dependencies
pip install -r requirements.txt
```

### Dataset Generation
```bash
# ⚠️ Windows: use the repo venv — a bare `python` is the Microsoft Store stub.
#    Set PYTHONIOENCODING=utf-8 when piping/redirecting (these scripts print `→`,
#    and the cp1252 pipe encoding kills them mid-run on a UnicodeEncodeError).
#      $env:PYTHONIOENCODING = "utf-8"
#      .venv\Scripts\python.exe scripts\generate_dataset.py ...

# N-1 contingency screening — the ONLY task; `--task n1` is the default.
# Per-line "if this line trips, is a thermal limit violated?"; writes *_n1.jsonl
# with a per-line label vector per frame.
#
# SIZING: effective sample size is bounded by SCENARIO count, not label count.
# --n1-stride 12 is the protocol on ALL THREE topologies — identical generation,
# only the environment differs. At the old stride 4 a 12k-frame budget would have
# covered only ~210 of neurips's 576 chronics instead of all of them.
#
# All three sets below are GENERATED and verified (component_d_plan.md §7.6).
# Measured throughput, not estimates:
python scripts/generate_dataset.py --env neurips --n_records 12000   # 20 min, 10 rec/s, 702k labels
python scripts/generate_dataset.py --env case14  --n_records 6000    #  5 min, 21 rec/s, 119k labels
python scripts/generate_dataset.py --env wcci    --n_records 4000    # 30 min,  2 rec/s, 742k labels

# --smoke: 3 chronics x 500 steps, capped at 200 records (~8 s). The record cap is
# load-bearing — the loop cycles chronics toward --n_records, so before it existed a
# smoke run replayed 3 scenarios toward 300,000 records and never terminated.
python scripts/generate_dataset.py --env case14 --smoke

# ⚠️ Forecast task — BUILT, PROBED, REJECTED. Retained ONLY to reproduce the
# negative result; do not train against it, and do not delete it to "finish" the
# classify cleanup — no forecast dataset exists on disk, so the code is the only
# way to reproduce component_d_plan.md §1.1 steps 1-2. Any-fault scores 1.00x the
# all-positive baseline at every H >= 3 (line-trip onset is an unconditional coin
# flip); overload-only is exhausted by a single rho threshold (model 0.163 vs rule
# 0.160, trend features do not help).
python scripts/generate_dataset.py --env neurips --task forecast --horizon 6 --n_records 300000

# The legacy 4-class `classify` generator was REMOVED on 2026-08-16 (closed-form
# target — see Current Status), mirroring the same cleanup in train_gnn.py. Its
# code is in git history; the datasets it produced stay frozen on disk.
```

### LLM Rule Extraction
```bash
# Stage 1: Extract candidate rules (Qwen3 extractor) — open vocabulary; writes *_candidates.jsonl
# Add --debug-raw to dump full model responses to <out>/_raw/ whenever JSON parsing fails
python extraction/extract.py --docs data/documents/ --out rules/     # DONE - do not re-run

# Stage 2: Translate conditions onto CONDITION_VOCABULARY (same model, reloaded)
#          → *_translated.jsonl + *_untranslatable.jsonl
# Add --debug-raw to dump responses that fail to parse to <out>/_raw/.
# ⚠️ Check `total_no_verdict` in translation_run_summary.json before believing the
#    yield: NO_VERDICT means the rule was never assessed (a parsing failure), which
#    is a different claim from "not expressible" and looks identical in the totals.
#    The run logs an ERROR above 20%. See component_d_plan.md §3.5.
python extraction/translate.py --candidates rules_35b/ --out rules/

# Stage 2.5: polarity guard (deterministic, no LLM/GPU) — --report first to pick the cutoff
# Defaults to ALL THREE topologies, reading grid_dataset_<tag>_n1.jsonl (falling back
# to the classify set). Narrow with repeated --tag only for a sensitivity run.
python extraction/polarity_guard.py --translated rules/ --report
python extraction/polarity_guard.py --translated rules/

# Stage 3: Validate guarded rules (Nemotron-3 Nano 30B) — reads guarded/*_translated.jsonl
# DONE 2026-08-20 — do not re-run without reading component_d_plan.md §15 first.
#
# ⚠️ --prompt-variant is LOAD-BEARING. 'strict' asks whether the constraint is *stated* in
#    the source text, which a translated condition never is; it rejected 31 of 32 rules,
#    including every `loading_pct > 100`, citing hallucinated vocabulary violations.
#    'translated' asks whether the condition faithfully *operationalizes* the requirement.
#    Default is 'strict' ONLY so the 2026-08-20 run stays reproducible — use 'translated'.
# ⚠️ Give each arm its OWN --out dir. deduplicate_rules merges every *_confirmed.jsonl it
#    finds, so a shared dir silently blends the two arms.
# ⚠️ Set VALIDATOR_MODEL if your Ollama tag differs (the LLM host runs nemotron-3-nano:30b).
python extraction/validate.py --candidates translated_rules/guarded/ --out validated_translated/ --prompt-variant translated
python extraction/validate.py --candidates translated_rules/guarded/ --out validated_strict/     --prompt-variant strict

# Rejections now persist to <stem>_rejected.jsonl WITH the validator's reason, and the run
# logs an ERROR above an 80% reject rate. A rejection without its reason is not a finding —
# run 1 rejected 31 of 32 and recorded nothing, which is what cost a round trip.

# NOTE: the knowledge graph is built by scripts/build_kg.py, NOT from this pipeline —
#       it reads stage-3 output only. See Component C below.
```
Stage 1 is complete (`rules_35b/`, 2,463 candidates) — do not re-run it. v1 outputs are archived
at `rules/v1_archive/`. ⚠️ `deduplicate_rules` merges every `*_confirmed.jsonl` in the out dir —
keep old outputs out of `rules/`, and never name a guard output `*_confirmed.jsonl`.

```bash
# Per-line base kV sidecar (needed by the shield's voltage_pu conversion)
python scripts/dump_base_kv.py --tag case14              # backend method (needs grid2op)
python scripts/dump_base_kv.py --tag case14 --empirical  # from existing JSONL (any machine)
```

### Model Training
```bash
# Train the N-1 GNN (36-bus NeurIPS 2020 — only topology used for training).
# No task switch: N-1 screening is the only model. GRID_TASK is gone.
python scripts/preprocess.py
python training/train_gnn.py --epochs 30 --batch_size 128 --lr 3e-4

# Ablation: drop message passing from the readout (writes *_headonly.pt, never
# overwrites the deliverable). Reported result: 0.8411 vs 0.8987 with it.
python training/train_gnn.py --head-only

# --report-train scores a train slice each epoch; the train/val gap is what
# separates memorisation from an optimiser that never fitted the signal.
python training/train_gnn.py --report-train

# GRID_DEVICE forces a backend (cpu/xpu/cuda) — added to distinguish backend
# bugs from modelling problems.
$env:GRID_DEVICE = "cpu"; python training/train_gnn.py
```

### Verification and Cross-Topology Evaluation
```bash
# Sanity-check a generated N-1 set BEFORE training on it: structure, scenario
# diversity, mixed-frame fraction, and whether the task is still non-degenerate.
python scripts/verify_n1_dataset.py --tag neurips2020

# Per-contingency metrics vs both baselines. neurips2020 scores the held-out
# test split; foreign topologies score every frame. All three RUN — results and
# caveats in component_d_plan.md §7.6.
python evaluation/eval_n1_cross_topology.py --tag neurips2020   # F1 0.8972  (1.93x rule)
python evaluation/eval_n1_cross_topology.py --tag case14        # F1 0.4477  (0.83x rule — FAILS)
python evaluation/eval_n1_cross_topology.py --tag wcci2022      # F1 0.5721  (1.16x rule)

# Shield (Component D) vs the raw model, per contingency. --rules MUST point at the
# validated corpus; the default is the stale rules/ path and the guarded folder is
# superseded (component_d_plan.md §15.6). Threshold is selected on the neurips2020 val
# split and held fixed across all three — do not pass --threshold.
python evaluation/eval_shield_n1.py --tag neurips2020 --rules validated_translated/all_rules_deduped.jsonl --json results/shield/shield_neurips2020_validated.json  # +0.0082, prec 0.938
python evaluation/eval_shield_n1.py --tag case14      --rules validated_translated/all_rules_deduped.jsonl --json results/shield/shield_case14_validated.json       # +0.0021, prec 0.922
python evaluation/eval_shield_n1.py --tag wcci2022    --rules validated_translated/all_rules_deduped.jsonl --json results/shield/shield_wcci2022_validated.json     # +0.0676, prec 0.934

# ⚠️ Each run OVERWRITES results/failures/failures_<tag>.jsonl. They currently describe the
#    validated corpus. `--rules` now DEFAULTS to the validated corpus — it used to default to
#    `rules/all_rules_deduped.jsonl`, a path that has not existed since the v2 pipeline.
```

### Component C — build the knowledge graph
```bash
# Reads stage-3 output only. No grid, no checkpoint, no topology.
python scripts/build_kg.py --out kg/knowledge_graph.json --figures

# Same shield run, but served from the graph and citing its sources. --rules-kg
# returns the identical rule set (tests/test_kg.py), so the numbers must not move;
# --citations writes one provenance chain per rule that fired, not per contingency.
python evaluation/eval_shield_n1.py --tag wcci2022 --rules-kg kg/knowledge_graph.json --json results/shield/shield_wcci2022_kg.json --citations results/citations/citations_wcci2022.json
```

### Data Inspection
```bash
# Inspect dataset statistics
python scripts/verify_n1_dataset.py --tag neurips2020
# Default: data/grid_dataset_neurips2020.jsonl
```

### Code Quality
```bash
black .
isort .
ruff check .
```

---

## Project Architecture

### Directory Structure
```
data/                              # gitignored in full — 423 MB, all of it regenerable
  grid_dataset_<tag>_n1.jsonl      # the three N-1 sets (scripts/generate_dataset.py)
  grid_dataset_<tag>_n1_meta.json  # env dims + label distribution, written beside each set
  grid_dataset_<tag>_basekv.json   # per-line base kV sidecars (scripts/dump_base_kv.py) — the
                                   #   shield's voltage_pu conversion. BACKEND method, all 3 tags.
  grid_dataset_<tag>_basekv_empirical.json  # the counterfactual arm — do not delete, do not use
  label_contexts_<tag>.json        # polarity-guard context cache (extraction/polarity_guard.py)
  processed_grid_data_n1.pt        # preprocessed PyG tensors (scripts/preprocess.py)
  split_neurips2020_n1_{train,val,test}_idx.npy   # chronic-level split indices

# The classify-era `grid_dataset_{neurips2020,case14}_meta.json` sidecars were deleted on
# 2026-08-20 along with the datasets they described. They were not inert: dump_base_kv.py
# resolved its suffix as ("", "_n1") and so preferred them, which made --empirical die on a
# .jsonl that no longer existed. The suffix order is now ("_n1", "_forecast").
#
# NOTE: there is no normalization stats file. Z-score stats are recomputed inline from the
# 36-bus training split by every eval script — see Component A below.

results/                           # every artifact the eval harness writes (was the repo root)
  shield/     shield_<tag>[_run3|_validated|_kg].json   # tracked — the recorded result
  citations/  citations_<tag>.json                      # tracked — provenance chains
  audit/      audit_run3.json                           # tracked — evaluation/audit_rules.py
  failures/   failures_<tag>.jsonl                      # GITIGNORED — ~14 MB, rebuilt by re-running

docs/                              # the 16 source PDFs (gitignored)

extraction/                        # Component B — LLM rule extraction pipeline
  extract.py                       # Qwen3-14B extraction pass → rules/*_candidates.jsonl
  validate.py                      # Nemotron-3 Nano 30B validation pass → *_confirmed/*_flagged
  common.py                        # Shared prompt/schema helpers

rules/                             # v1 archive ONLY (gitignored). Not an output directory —
                                   #   several scripts' usage strings still say `--out rules/`
                                   #   as a generic example; the live stage-1 corpus is rules_35b/
rules_35b/                         # stage 1, 2,463 candidates across 16 docs (PRISTINE — never re-run)
translated_rules/                  # stage 2 run 3 output; guarded/ is the stage-2.5 corpus (32 rules)
validated_translated/              # ⭐ STAGE 3, the CORPUS TO CITE — all_rules_deduped.jsonl = 4 rules
validated_strict/                  # stage 3 counterfactual arm (1 rule) — evidence, never an input
# NOTE: validated_rules/ was stage 3 run 1 (rejections not persisted). DELETED 2026-08-20 —
# validated_strict/ reproduces it exactly and keeps the reasons. In git history at a5c5199.
  all_rules_deduped.jsonl          # merged, deduplicated confirmed rules
  *_rejected.jsonl                 # REJECT verdicts WITH reasons (added 2026-08-20)
  validation_run_summary.json      # per-document confirm/correct/reject counts + reject_rate

kg/                                # Component C — the provenance knowledge graph (a PACKAGE, not a dump)
  schema.py                        # node/edge vocabulary, Citation dataclasses, ISSUING_BODY table
  build.py                         # build_knowledge_graph(), save/load, predicate normalization
  provider.py                      # KgRuleProvider — RuleProvider protocol + cite()
  cite.py                          # ShieldResult -> citation chain (keeps shield/ free of kg imports)
  figures.py                       # static thesis figures (SVG + PNG)
  knowledge_graph.json             # the graph itself — JSON, not pickle (v1's .pkl vanished unnoticed)
  kg_provenance.{svg,png}          # the 36-node graph in its six layers
  kg_corroboration.{svg,png}       # clauses per predicate, segmented by source document
  # NOTE: the shield lives in shield/ at the repo root, NOT here.

scripts/
  generate_dataset.py              # Grid2Op simulation → JSONL records (--task n1 | forecast | classify)
  preprocess.py                    # JSONL → processed_grid_data_n1.pt
  pyg_data.py                      # PyG wrapper: GridDataset, feature builders, build_line_targets
  verify_n1_dataset.py             # pre-training sanity check on a generated N-1 set
  dump_base_kv.py                  # per-line base kV sidecar for the shield
  build_kg.py                      # Component C — builds kg/knowledge_graph.json from stage-3 output

training/
  train_gnn.py                     # Trains the N-1 GNN — the only model. Chronic-level split built inline.
  config.py                        # TRAIN_CONFIG (auto-selected by device), artifact paths, GRID_DEVICE override

evaluation/
  eval_n1_cross_topology.py        # per-contingency metrics vs the all-positive and rule baselines
  eval_shield_n1.py                # the live shield harness — writes into results/
  audit_rules.py                   # offline rule audit (INERT/UNINFORMATIVE/USEFUL), no LLM

notebooks/thesis_figures.ipynb     # 14 report figures -> figures/*.{svg,png}
sanity/                            # standalone environment checks + the LLM throughput bench
supplimentary_docs/
  component_d_plan.md · study.md · formula.md · thesis_findings.md ·
  gnn_n1_tightening.md · revised_thesis_claim.md      # LIVE
  env_selection/                   # why these Grid2Op environments (Apr 2026, bannered)
  Architecture.svg                 # the four components on one page — REDRAWN 2026-08-20
  archive/                         # retired: gnn_final_results, lever_A, CASCADE journal,
                                   #   shield_necessity, study3(integration),
                                   #   component_d_handoff_archive
```

---

## Component A — GNN Architecture

### Model: GridGNN (Graph Attention Network v2)

```
Input: (n_nodes × 8 node features, n_edges × 8 edge features)
    ↓
GATv2Conv(8 → h0, heads=k0, edge_dim=8)  + BatchNorm(track_running_stats=False) + ELU
    ↓
GATv2Conv(h0*k0 → h1, heads=k1, edge_dim=8) + BatchNorm(track_running_stats=False) + ELU
    ↓
GATv2Conv(h1*k1 → h2, heads=k2, edge_dim=8) + BatchNorm(track_running_stats=False) + ELU
    ↓
Edge-level readout, per LINE k, on its or→ex edge — NOTHING IS POOLED:
    [ h_or ‖ h_ex ‖ edge_attr_k ‖ x_or ‖ x_ex ]   (learned embeddings + raw skip)
    ↓
line_head MLP (→128 →64 →1)  →  ONE logit per line = P(losing line k violates a limit)
```
`(h0,h1,h2)` and head counts `(k0,k1,k2)` come from `TRAIN_CONFIG` — see **Training Configuration**
below; the deployed checkpoint uses `[16,32,32]`/`heads=[4,4,1]` (the personal-PC/non-CUDA branch).

**Why no global pooling.** ⚠️ **Corrected 2026-08-20 — the earlier "100% of frames are mixed"
was an overstatement**, measured over every frame of all three datasets (22,000 frames):

| grid | strictly mixed | all contingencies violate | none violate |
|---|---:|---:|---:|
| neurips2020 | **98.80%** | 1.20% | **0.00%** |
| case14 | **98.48%** | 1.52% | **0.00%** |
| wcci2022 | **96.43%** | 3.57% | **0.00%** |

The 100% figure was true of a *different* statistic: **no frame anywhere is entirely secure**.
Strictly mixed frames are 96–99%. The argument is unaffected — a per-frame label is still wrong
for most lines in 96%+ of frames — but quote 96–99%, not 100%. Within one grid state, some contingencies
violate and others do not. Pooling to a graph-level vector erases exactly the per-line distinction
the task is about. It is also what makes the checkpoint topology-agnostic: one logit per line
present, so 20-line case14 and 186-line WCCI run on it unchanged.

**Why the raw endpoint features are concatenated (the skip).** Three rounds of attention, batch
norm and ELU are free to turn `sum_headroom` and `sum_abs_p` into something unrecognisable — but
those two quantities *are* the physics of post-contingency redistribution: line k's flow has to be
absorbed by the spare capacity at its endpoints. Handing them to the readout unmodified lets the
head form the flow/spare ratio directly instead of hoping it survived the encoder.

⚠️ The 4-class `classifier` head, the per-bus `localizer` head and the Lever A logit margin below
belonged to the RETIRED classify model. They are described further down for the record only —
`GridGNN` no longer has them, and their checkpoints were deleted on 2026-08-20.

**Critical implementation notes:**
- **GATv2Conv, not GATConv.** GATConv's static attention (scored before concatenation) rank-collapses
  on small graphs like this 36-node grid; GATv2Conv scores attention after concatenation, preserving
  expressiveness. This is load-bearing, confirmed by `supplimentary_docs/archive/gnn_upgrade_assessment.md`.
- `BatchNorm(track_running_stats=False)` — live batch stats, not running stats. Running stats flatten overload spikes (rho > 1.0) during eval mode.
- **No dropout** — dropout severs attention edges and creates train/eval scaling gaps on power flow features.
- ~~**Triple pooling**~~ — CLASSIFY-ONLY, and gone. The N-1 model pools nothing; it reads a logit
  off each line's own edge. Kept here as the record: pooling was the diagnosed bottleneck for
  `normal`/`line_trip` confusion (`archive/gnn_final_results.md` §3), and removing it is part of why the
  N-1 head works.
- **Tripped lines are pruned from `edge_index`** using `line_status` boolean mask at graph construction. Without pruning, line_trip states are structurally identical to normal states.
- ~~**Lever A**~~ — CLASSIFY-ONLY, and gone (`gnn_logit_margin.json` deleted 2026-08-20;
  recoverable from git history). For the record: a fixed per-class logit offset (`normal +0.30`,
  `line_trip −0.10`, others `0`) added before argmax took macro F1 from 0.7830 → 0.8277 and
  `normal` recall from 0.65 → 0.88 — a calibration tweak, not an architecture change, was the only
  thing that worked on that task. It never transferred cross-topology. **The N-1 model has no
  logit margin and needs none.**
- **GSAT (stochastic edge gating) and a soft-F1 loss term were tried and rejected in Round 3** — the
  code has since been removed from `train_gnn.py`/`config.py` (reverted to pre-Round-3 state) now that
  the results doc has captured the results. Retrieve from git history before revisiting; see
  `supplimentary_docs/archive/gnn_final_results.md`.

### Node Features (5 per bus)

| Feature | Construction | Signal |
|---|---|---|
| `load_p` | Sum of active loads at bus | Demand |
| `mean_v` | Mean voltage of connected lines / 150.0 | Voltage health |
| `max_rho` | Max loading ratio of connected lines | Overload |
| `connected_line_frac` | Fraction of lines still connected | Trip/cascade |
| `global_trip_frac` | Fraction of ALL lines in the graph that are tripped | Graph-level topology signal |

`gen_p` was removed — EDA showed 1.00 correlation with `load_p` (generation matches load by power flow law).

### Edge Features (4 per line)

`[rho, p_or, q_or, near_limit]` — `near_limit = (rho >= 1.0)`, a boolean overload flag (this superseded
a `>= 0.9` threshold; see Exp 1 in the closure doc, the one Round-1 lever that was kept).

### Training Configuration — auto-selected by device in `training/config.py`

`TRAIN_CONFIG` is picked automatically: `DEVICE.type == "cuda"` gets the CUDA branch, anything
else (XPU/CPU) gets the smaller, **deployed** branch below.

| Parameter | Non-CUDA branch (**deployed config**) | CUDA branch (untested at scale — see note) |
|---|---|---|
| `hidden_channels` | `[16, 32, 32]` | `[64, 128, 128]` |
| `heads` | `[4, 4, 1]` | `[2, 2, 1]` |
| Epochs | 10 | 50 |
| Batch size | 512 | 256 |
| Learning rate | 1e-4 | 5e-4 |
| Weight decay | 1e-5 | 1e-5 |
| Dropout | 0.0 (determinism) | 0.0 |
| Loss | Weighted CrossEntropy (ICF, sqrt-smoothed) + label_smoothing 0.1 + 0.5×loc_loss | same |
| Primary metric | **Macro F1** (not accuracy — dataset is imbalanced) | same |
| Normalization | Z-score from the training split only, recomputed inline at eval time | same |

**The deployed `gnn_checkpoint_n1.pt` uses the small `[16,32,32]`/`heads=[4,4,1]` config**
— verified from checkpoint tensor shapes, not assumed. The `[64,128,128]` CUDA-branch scale-up was
tried (Round 1) and **overfits/collapses** under its schedule (train loss falls while val F1 falls;
`normal`+`cascade` go to 0.0) — it is *not* a validated alternative, just what auto-selects on a CUDA
machine. If training on a CUDA (research) machine, be aware the auto-selected config there has never
produced a working checkpoint; the small config is what's proven.

**Normalization is recomputed inline, not loaded from disk.** Every live eval calls
`compute_normalization_stats(home, train_idx)` over the 36-bus training split at startup; foreign
topologies are then normalized with those 36-bus stats — intentional, physical quantities have the
same scale. `normalization_stats.pt` was deleted on 2026-08-20 because nothing read it, and
`training/config.py`'s `NORM_STATS_FILE` is dead config referenced by nothing.

**Seed sensitivity (diagnosed in Round 3, tooling since removed):** the deployed init (`seed=42`) is a
good, reproducible init (0.82–0.90 macro F1 across different train/val partitions) — but a *different*
init can degrade to ~0.72–0.82. Localized to initialization, not the data split; full data in
`supplimentary_docs/archive/gnn_final_results.md` §5. Do not change `SEED` (still hardcoded to `42` in
`training/config.py`) without being aware of this.

**In-memory shuffle required** before training to eliminate chronological domain shift:
```python
combined_idx = np.concatenate([train_idx, val_idx])
np.random.seed(42)
np.random.shuffle(combined_idx)
```

### Cross-Topology Inference

The classification head runs unchanged on foreign topologies (global pooling is topology-agnostic). The localization head is **disabled** for cross-topology evaluation (outputs `n_nodes` logits, which vary by topology).

---

## Component B — LLM Knowledge Extraction

### Models

- **Extractor / Translator:** Qwen3 via Ollama, auto-selected by device in `extraction/common.py`
  — `qwen3.6:35b` when `DEVICE.type == "cuda"`, else `qwen3.5:9b` for local testing.
  Override with the `EXTRACTOR_MODEL` env var. Thinking is suppressed via
  `generate_no_think()` (`think=False`), with `/no_think` kept only as belt-and-braces.
- **Validator:** Nemotron-3 Nano 30B (A3B MoE) via Ollama — `think=False` enforced, confirms/corrects/rejects each candidate

Sequential only — never load both models simultaneously. `keep_alive=0` on every Ollama call to release VRAM immediately.

### Pipeline

```
PDF chunks → Qwen3-14B → JSON candidates → Pydantic validation → Nemotron-3 Nano 30B → CONFIRM/CORRECT/REJECT → dedup → KG
```

### v1 Results (SUPERSEDED — kept as the negative-result record)

| Stage | Count |
|---|---|
| Candidate rules extracted | 1,372 |
| Confirmed (CONFIRM) | 50 |
| Corrected and retained (CORRECT) | 451 |
| Rejected (REJECT) | 766 |
| Flagged for review | 105 |
| **Unique rules after dedup** | **469** |

v2 ran all four stages: **2,463 candidates** (`rules_35b/`) → 58 translated → 32 guarded →
**4 distinct validated rules** (`validated_translated/all_rules_deduped.jsonl`). 0.16% yield —
a finding about the standards/simulator mismatch, not a success metric. See
`component_d_plan.md` §12 for why, and §15 for the validation run.

### Rule Schema

```json
{
  "rule_id": "R_042",
  "source": "IEEE Std 1547-2018, Section 7.4",
  "entity": "Bus",
  "condition": "voltage_pu > 1.05 OR voltage_pu < 0.95",
  "action": "BLOCK",
  "severity": "critical",
  "explanation": "Voltage deviation beyond ±5% nominal violates IEEE 1547 interconnection standards."
}
```

### Implementation Notes

- `strip_think()` regex applied to all model output before JSON parsing — Qwen3 occasionally leaks `<think>` tokens despite `/no_think`
- GBNF grammar enforcement on array output prevents malformed JSON
- Rule IDs are globally sequential (`R_001`, `R_002`, ...) across all chunks/documents before validation

---

## Component C — Knowledge Graph

**Backend:** NetworkX DiGraph, persisted as **JSON** (`nx.node_link_data`), not pickle. v1 used a
`.pkl`; it vanished from disk and nothing noticed. 36 nodes cost nothing to store readably, and
JSON survives a networkx upgrade that would break an unpickle.

### Schema — provenance, not topology

```
Document --contains--> Clause --states--> Rule --deduped_into--> ServedRule
    --instantiates--> Predicate --reads--> Variable
```

| node | n | what it is |
|---|---:|---|
| `Document` | 5 | one per source standard that contributed a validated rule |
| `Clause` | 11 | the cited section within a document |
| `Rule` | 11 | one per record in `validated_translated/*_confirmed.jsonl` |
| `ServedRule` | 4 | one per record in `all_rules_deduped.jsonl` — **carries the rule dict verbatim**; this is what the shield receives |
| `Predicate` | 2 | the physical check, normalized across noisy `entity` labels and numeric formatting |
| `Variable` | 3 | the `CONDITION_VOCABULARY` entries actually read |

Total **40 edges**, spread across five types. Compare v1: 6,632 edges of which 6,375 (96%) were
`has_rule` noise. `tests/test_kg.py::test_no_node_type_dominates_the_edge_count` guards against
regressing to that shape.

### The corroboration finding

| predicate | role | served rules | clauses | documents | identified bodies |
|---|---|---:|---:|---:|---:|
| `loading_pct > 100` | CONSTRAINT | 3 | **10** | **4** | 2 |
| `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1` | AFFIRMATION | 1 | 1 | 1 | 1 |

The thermal check is stated ten times, in four documents, by NERC and the Bangladesh grid code —
independent restatement of the same physical limit. `power-system-requirements` carries an
"EMO Dispatch Computer Constraints" clause whose issuing body the document does not identify; it
is recorded as `None` rather than guessed, because the body count is a reported figure
(`kg/schema.py::ISSUING_BODY` is hand-labelled and says so).

### Retrieval — opt-in, and provably identical

`KgRuleProvider` satisfies the `RuleProvider` protocol structurally, so it drops into the gate and
the harness unchanged. **Every rule applies to every prediction, exactly as with
`JsonlRuleProvider`.** Do not add entity routing: `R_1443` is labelled `Facility` while an
identical rule from another standard is labelled `Line`, so routing by entity silently drops
rules — the v1 mistake.

Verified on all three topologies: every field of `shield_<tag>_kg.json` equals
`shield_<tag>_validated.json`. ⚠️ `threshold` and `ap` wobble at ~1e-6 / ~1e-8 between *any* two
runs on this hardware — two runs of the identical JSONL path reproduce the same drift, so it is
forward-pass nondeterminism, not the graph. No count ever moved.

## Component D — Symbolic Validation Shield

**BUILT (2026-08-15) — but the pseudocode below is the OLD design and no longer matches the code.**
The real implementation is `shield/{context,evaluator,shield}.py`; read those and
`supplimentary_docs/component_d_plan.md` §5–§6 instead. Three differences that matter:

1. **Rule retrieval is not KG-based.** `validate(context, rules)` takes a plain rule list behind a
   `RuleProvider` protocol. The KG is an opt-in provider, never a dependency — nothing under
   `shield/` imports anything from `kg/`, and citation rendering lives in `kg/cite.py`.
2. **`voltage_pu = min(v_or)/150.0` is wrong** — per-line base kV with energized-line masking (§5).
3. **Rules have roles.** Only `CONSTRAINT` violations block; `AFFIRMATION` rules supply supporting
   evidence and never block on their own (Option A).

Every GNN prediction passes through the shield; there is no bypass mode.

### Validation Flow (SUPERSEDED design — historical)

```python
def validate(prediction_context: dict, KG) -> dict:
    fault_type       = prediction_context["fault_type"]
    applicable_rules = rules_of(KG, fault_type) + rules_of(KG, "Grid")

    violated = [r for r in applicable_rules if evaluate_condition(r["condition"], prediction_context)]

    if not violated:
        return {"status": "PASS", "fault_type": fault_type, ...}

    violated.sort(key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}[r["severity"]])
    return {"status": "BLOCK", "violated_rules": violated, "explanation": build_explanation(violated)}


def evaluate_condition(condition: str, context: dict) -> bool:
    namespace = {
        "voltage_pu":      min(context["v_or"]) / 150.0,   # Grid2Op kV → per-unit
        "loading_pct":     context["rho_max"] * 100,
        "rho":             context["rho_max"],
        "n_tripped_lines": context["n_tripped_lines"],
        "line_status":     not all(context["line_status"]),
    }
    try:
        return bool(eval(condition, {"__builtins__": {}}, namespace))
    except Exception:
        return False  # malformed condition → do not block
```

**Voltage translation:** `voltage_pu = v_or_kv / 150.0` (nominal for NeurIPS 2020 environment). Update this constant from the environment's meta JSON when evaluating on other environments.

### Cross-Topology Failure Mode Logging (planned, not yet built)

Once the shield exists, every BLOCK from cross-topology evaluation should be logged to
`failures_<topo>.jsonl` (these files do not exist yet) with failure mode classification:

| Failure Mode | Definition |
|---|---|
| Overconfident wrong | GNN predicts "normal" with confidence > 0.85 during a physical fault |
| Class confusion | GNN detects fault but confuses type (e.g., overload predicted as cascade) |
| Threshold failure | Correct fault type, but predicted entity violates physical thresholds |
| Novel topology state | Sub-graph structure never seen during training |

---

## Data Collection

### Environment Role Assignment

| Environment | Role | Buses | Lines | Records |
|---|---|---|---|---|
| `l2rpn_neurips_2020_track1_small` | **Training only** | 36 | 59 | 300,000 |
| `l2rpn_case14_sandbox` | **Test only — unseen smaller** | 14 | 20 | ~15,000 |
| `l2rpn_wcci_2022` | **Test only — unseen larger** | 118 | 186 | ~20,000 |

(`rte_case14_sandbox` was the original env name and does not exist in this grid2op version — corrected
to `l2rpn_case14_sandbox` in `scripts/generate_dataset.py`; noted here so it isn't reintroduced.)

No training on case14 or WCCI 2022. The trained checkpoint is applied directly.

### Training Dataset Class Distribution

| Label | Count | % |
|---|---|---|
| Normal | 61,444 | 20.48% |
| Overload | 103,556 | 34.52% |
| Line Trip | 75,000 | 25.00% |
| Cascade | 60,000 | 20.00% |

**Chronic-level splitting only** (70/15/15). Frame-level splitting leaks cascade sequences across splits.

### Labeling Logic

```python
def get_state_label(obs, env):
    max_rho      = obs.rho.max() if len(obs.rho) > 0 else 0.0
    active_lines = int(np.sum(obs.line_status))

    if max_rho >= 1.0:
        return "overload", int(obs.rho.argmax())
    if active_lines == env.n_line:
        return "normal", -1
    if active_lines == env.n_line - 1:
        return "line_trip", int(np.where(~obs.line_status)[0][0])
    return "cascade", -1
```

⚠️ **This labelling function is CLOSED-FORM over the observation** — four threshold rules on
`rho_max` and `n_tripped_lines` reproduce it with **100% agreement, 0 disagreements in 315,000
records** (every record of both datasets, re-measured 2026-08-20). That is why
Component A was reopened for the binary forecast task; see Current Status and plan §2. The
`classify` target below cannot distinguish a learned model from a threshold.

**`NO_OVERFLOW_DISCONNECTION = False`** — must be passed at `grid2op.make()` via `param=params`, not set post-make. Enables natural cascades.

---

## Machine roles

Two machines, split by capability rather than by size. **No hardware specifications are
recorded anywhere in this repository** — they were never load-bearing, and the two documents
that carried them disagreed with each other. What matters is which capability a stage needs.

| role | needs | runs |
|---|---|---|
| **LLM host** | CUDA + Ollama | `extract.py` → `translate.py` → `validate.py`, strictly sequential |
| **Workstation** | any torch device | dataset generation, GNN training, polarity guard, KG build, shield, all tests |

**Never load both LLM models at once.** Qwen3 runs to completion first, then Nemotron-3; every
Ollama call passes `keep_alive=0` so VRAM is released between stages. The validator is the larger
of the two and relies on partial CPU offload.

`training/config.py` auto-selects by `DEVICE.type`. The non-CUDA branch (`[16,32,32]`) is the
**proven, deployed** config; the CUDA branch (`[64,128,128]`) has never produced a working
checkpoint, so training off the LLM host is the safer choice, not a compromise.

---

## Key Implementation Constants

| Constant | Value | Context |
|---|---|---|
| `num_workers` | `0` | Windows PyG DataLoader constraint |
| ~~Nominal voltage~~ | ~~`150.0 kV`~~ | **SUPERSEDED** — the shield uses per-line base kV from `data/grid_dataset_<tag>_basekv.json`. A flat divisor gives ~100% false blocks. Still used only for the GNN's `mean_v` node feature. |
| Base kV method | `backend`, all 3 tags | **Settled 2026-08-16** (`component_d_plan.md` §5.1). Backend nominals: neurips 138/345, case14 14/20/138, wcci2022 138/161/345. These grids *operate* ~6% above nominal, so healthy frames read ~1.06 pu, **not ~1.00**. Empirical originals kept as `*_basekv_empirical.json` — do not delete, they are the counterfactual arm. Never pass `--empirical` again. |
| FAULT_PROB | `0.05` | Grid2Op fault injection rate (`scripts/generate_dataset.py`) |
| RECONNECT_PROB | `0.20` | Grid2Op reconnect probability |
| `--n1-stride` | `12` | Labels every 12th step. The protocol on **all three** topologies — see the sizing note under Dataset Generation |
| ~~NORMAL_KEEP_PROB~~ / ~~LINE_TRIP_KEEP_PROB~~ | — | **REMOVED 2026-08-16** with the classify generator. They drove that task's subsampling quotas; N-1 does not subsample by class. In git history only. |
| Random seed | `42` | Model init + in-memory shuffle before training — deployed init is seed-sensitive, see Component A |
| Checkpoint filename | `gnn_checkpoint_n1.pt` (repo root) — the ONLY model. Loads into the current `GridGNN` (8 node / 8 edge features, `line_head`). | Saved on best F1 during N-1 training |
| ~~Logit margin~~ | ~~`gnn_logit_margin.json`~~ | **DELETED 2026-08-20** — Lever A was classify-only; recoverable from git history |
| ~~Normalization stats~~ | ~~`normalization_stats.pt`~~ | **DELETED 2026-08-20** — nothing read it. Every live eval calls `compute_normalization_stats()` inline from the training split. `training/config.py` still defines `NORM_STATS_FILE`, which nothing references — dead config |

---

## Dependencies

- PyTorch 2.8.0 + CUDA 12.8
- PyTorch Geometric (GAT, global pooling, DataLoader)
- grid2op + lightsim2grid (LightSimBackend — ~10x faster than PandaPower)
- ollama (local LLM inference — Qwen3-14B, Nemotron-3 Nano 30B)
- networkx (KG backend — DiGraph)
- pydantic (rule and verdict schema validation)
- pdfplumber (PDF ingestion for LLM pipeline)
- numpy, scikit-learn (data processing)
- wandb (experiment tracking — macro F1, per-class F1, confusion matrices)
- tqdm
