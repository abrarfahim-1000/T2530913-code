# Findings — Neuro-Symbolic Fault Screening for Power Grids

**Status as of 2026-08-20 — all four components are built and measured.** A plain-language account
of what was built, what was measured, and what it means. Written to be readable without the
codebase open.

**Confirmed by replication:** the shield numbers were re-measured against a second, independently
generated rule corpus and came back **identical to the digit** (§13.6). The result does not
depend on which extraction run produced the rules.

⚠ **Corrected 2026-08-20.** An earlier version of this document said the gate's override
precision *rises* off-distribution (51% at home, 86–92% away). That was measured to be false. It
was an artifact of averaging two rule families; once the misextracted ones were removed the
precision is **flat at 92–94% on all three grids**. §2.1 carries the corrected reading, and it is
a better result than the one it replaces. Derivation: §13.2.

**If you are new to the project, read
[`thesis_walkthrough.md`](thesis_walkthrough.md) first** — it walks through every stage in build order and
explains the power-grid terms this document assumes.

**This document is self-contained.** §1–§8 are the narrative, readable without the codebase open.
§9–§18 are an appendix carrying the measurement record behind every claim — the exact tables,
protocols, caveats and landmines. Each narrative claim cites the appendix section that derives it.
The appendix was folded in from the retired build plan on 2026-08-21; that plan now lives at
[`archive/component_d_plan.md`](archive/component_d_plan.md) and nothing here depends on it.

---

## 1. The question

A grid operator continually asks a question called **N-1 screening**: *for every transmission
line in the network, if that line failed right now, would the grid break?*

Answering it properly means running a power-flow simulation for each line — expensive, and the
reason operators cannot check every contingency continuously on a large network. A neural
network that approximates the answer from the current state of the grid would be valuable.

The thesis question is not "can a neural network do this" — it can. It is:

> **Can a symbolic layer, built automatically from published engineering standards, catch the
> neural network when it is wrong — and does that help more, or less, as the network moves onto
> grids it was never trained on?**

The architecture is a **post-hoc gate**. The model makes a prediction; a rule checker reads the
same raw telemetry, consults rules extracted from IEEE / NERC / national grid codes, and may
veto. The model never sees the rules, and there is no bypass path.

---

## 2. The headline result

**Yes, it catches the model — and its value grows the further the model gets from home.**

Three grids. One is the training grid; the other two the model has never seen. The decision
threshold was chosen on a validation split of the training grid and then **held fixed** across
all three, so none of these numbers are tuned to the grid they are reported on.

| grid | size | model alone | **+ symbolic gate** | change |
|---|---|---:|---:|---:|
| NeurIPS 2020 *(trained on)* | 36 buses, 59 lines | 0.8956 | **0.9038** | +0.0082 |
| case14 *(unseen, smaller)* | 14 buses, 20 lines | 0.4167 | **0.4188** | +0.0021 |
| WCCI 2022 *(unseen, larger)* | 118 buses, 186 lines | 0.5577 | **0.6253** | **+0.0676** |

*(F1 score, against the final validated four-rule corpus. Full table, including the earlier
32-rule corpus for comparison: §13.1.)*

On the large unseen grid the gate is worth a great deal more than a rounding error:

- **Dangerous errors — cases where the model wrongly declared a contingency safe — fell from
  68,166 to 47,101. A 31% reduction.**
- The cost was about 1,500 extra false alarms, a 1.3% increase. In grid operations a missed fault
  is far more expensive than a false alarm, so that is a favourable trade.
- The gated score, **0.6253**, is higher than the raw model reaches on that grid *even when
  allowed to tune its threshold against the answers* (0.5721). A layer bolted on after training,
  changing nothing inside the model, pushed an unseen-topology result past the model's own
  oracle ceiling.

### 2.1 Why this is the interesting result and not just a number

The gate can only intervene in a narrow situation: the grid is **already** past a limit, and the
model still called a contingency safe. Two things vary across the three grids, and only one of
them is what you would expect:

| grid | how often it can intervene | **when it does, how often it is right** |
|---|---:|---:|
| NeurIPS 2020 *(trained on)* | 0.31% of cases | **93.8%** |
| case14 *(unseen)* | 0.087% | **92.2%** |
| WCCI 2022 *(unseen)* | 3.04% | **93.4%** |

**The right-hand column is flat.** The gate is right about 93% of the time on the grid the model
was trained on, and about 93% of the time on grids it has never seen. That is the point:

> **The rulebook's accuracy does not depend on the topology, because it is not pattern-matching
> — it is enforcing a physical principle that is equally true on every grid. The neural network's
> accuracy collapses off-distribution (0.90 → 0.42 → 0.56). The rulebook's does not move.**

What *does* change off-distribution is how much work there is for the gate to do. On the training
grid the model is good, so it rarely hands the gate a wrong "this is safe" claim to veto — 0.31%
of cases. On WCCI 2022 it is wrong ten times as often in exactly the way the gate can see — 3.04%
— which is why the same rulebook is worth +0.0676 there and +0.0082 at home.

So the finding is not "rules help." It is:

> **A symbolic layer built from published standards is topology-invariant in a way a trained
> model is not. Its value scales with how badly the model is failing, not with how well the
> symbolic layer is tuned.**

That is precisely the case for keeping a symbolic layer in the loop, and it is measured rather
than asserted.

### 2.2 A correction worth reporting, because of how it was found

An earlier version of this document reported a different and more dramatic story: that the gate
was a **coin flip** on the training grid (51.4%) and reliable only off-distribution (86–92%). The
reading built on it — *"the symbolic layer holds while the neural layer stops earning the benefit
of the doubt"* — was wrong, and the way it was caught is itself a result.

That 51.4% was an average over two very different kinds of rule. The voltage rules, which came
from ride-through tables misread as instantaneous limits, were right only 13–20% of the time when
they fired. The thermal rules were right 92–94% of the time. On the training grid the voltage
rules happened to dominate — 478 of 831 interventions — and dragged the average to a coin flip.

The final validation stage removed exactly those voltage rules. It did so by **reading the source
standards**, having never seen a single measurement of how the rules behave on real grid data. The
number it was implicitly predicting — that the remaining rules would be right ~93% of the time
everywhere — is what came out.

The lesson is a general one about this kind of pipeline: **a rulebook that is 90% wrong rules and
10% good ones does not perform at 90% of the good version. It performs at the average, and the
average hides the good rules completely.** Filtering was not cosmetic; it was the difference
between a coin flip and a reliable gate.

---

## 3. Why the effect is dramatic on one grid and invisible on two

Not a contradiction — the three grids sit in different regimes, and in all three the gate is
right roughly 93% of the time when it speaks. What differs is how often it gets to speak.

**Training grid:** the model is genuinely good (0.90 F1), so there were only 2,041 dangerous
errors to begin with, and only 16% of those sit on a grid state a rule can see. The gate is
working — and at 93.8% precision it is working well — but there is little left for it to work on.

**case14:** the model is broken here — it fails against a single-threshold baseline outright
(§14) and floods the output with false alarms. Because it declares almost everything
dangerous, it almost never hands the gate a "this is safe" claim to veto. Reach collapses to
0.087%. The gate is *nearly always right* in that sliver (92.2%); there is just no sliver.

**WCCI 2022:** the sweet spot. The model fails often **and** fails on states the rules can see.
3.04% reach, 93.4% precision, 31% of the dangerous errors removed.

---

## 4. The honest limits

These are load-bearing. They belong in the write-up **volunteered**, not extracted under
questioning.

### 4.1 The surviving rulebook is essentially one rule

Of **2,463** candidate rules extracted from 16 standards documents — shown for both extraction
runs, because the agreement between them is itself informative:

| stage | run 2 | run 3 |
|---|---:|---:|
| extracted from the PDFs | 2,463 | 2,463 |
| expressible against simulator telemetry | 82 (3.3%) | 58 (2.4%) |
| passing the polarity check (does not fire on a healthy grid) | 33 | 32 |
| able to do any work at all | 11 distinct | 13 distinct |
| **surviving final validation against the source standards** | — | **4 distinct** |
| **rules that actually fire at inference** | **10** | **10** |
| — all of them saying the same thing (`is the line over 100% loaded?`) | ✔ | ✔ |

*(§10.3, §12.)* End to end: **2,463 candidate rules become 4.** The symbolic layer
distils to **one** physically meaningful predicate, and that survives unchanged across two
independent extraction runs whose totals differ by 29%. The second run also produced a
**cleaner** stream — the polarity check rejected 44.8% of it rather than 59.8% — after a prompt
fix described in §10.3.

This is why the shield result was identical on both extraction corpora: everything except that
one predicate is silent at inference. It is also why removing 28 of the 32 rules at the
validation stage *improved* the result rather than degrading it (§2.2) — the discarded rules were
not contributing coverage, they were contributing noise.

**The four survivors are worth naming, since the whole symbolic layer is these:** three phrasings
of *"is any line loaded past 100% of its thermal rating?"*, plus one statement that *"voltage
between 0.9 and 1.1 per unit is consistent with normal operation"*, which supplies supporting
evidence but can never block on its own.

**Four is also the least flattering way to count it.** Those three thermal rules are the same
physical check; they are stored as three records only because the extraction model labelled the
entity `Line` in one standard and `Facility` in another, and wrote `100` in one clause and `100.0`
in another. The knowledge graph (§7) normalizes that away and shows what is actually there:

> **one thermal check, stated in 10 separate clauses across 4 documents, by two different
> standards bodies on two continents** — NERC and the Bangladesh grid code.

That is a stronger claim than "we extracted 4 rules", from the same evidence counted more
carefully. It does not make the rule less obvious — §4.2 still applies, and four documents
agreeing that you should not exceed a thermal rating is four documents agreeing on standard
practice. What it does establish is that the surviving rule is not an artefact of one document or
one parse.

### 4.2 That rule is close to common sense

If a grid is already over its thermal limit, then losing another line obviously keeps it over the
limit. How obvious was measured rather than assumed:

| grid | violations overall | violations *given* an already-overloaded grid |
|---|---:|---:|
| NeurIPS 2020 | 19.5% | **95.7%** |
| case14 | 27.8% | **91.2%** |
| WCCI 2022 | 24.8% | **94.7%** |

If this were a circular result — the rule secretly restating the labelling function — the right
column would read 100%. At **91–96%** the rule carries real but partial information.

The honest framing: **the gate is confirming established N-1 doctrine — you cannot declare a grid
secure against further loss when it is already outside its limits — and the data bears that
doctrine out 91–96% of the time.** It is not discovering new physics. It is enforcing a known
operational principle that the network failed to learn, and failed to learn *worse* the further
it moved from its training topology.

### 4.3 The reasoning is hand-written; the language model supplied only the number

The principle in 4.2 is encoded in our own code (`shield/shield.py::validate_n1`). The LLM
extraction pipeline contributed the threshold — 100% loading — and nothing else.

This is not a flaw so much as a finding, and it closes a loop with §5: that doctrine belongs to
the class of **operational** standards (NERC FAC-011, EU System Operation Guideline) which §5
shows is missing from the document corpus. The pipeline could not supply the reasoning because
nobody fed it a document containing the reasoning.

### 4.4 There is a hard ceiling, and more rules will not lift it

Of every dangerous error the model makes, what share happens on a grid state that **any**
present-state rule could even notice?

| grid | dangerous errors | reachable by any rule of this kind |
|---|---:|---:|
| NeurIPS 2020 | 2,041 | 331 (**16.2%**) |
| case14 | 18,592 | 95 (**0.51%**) |
| WCCI 2022 | 68,166 | 21,065 (**30.9%**) |

The complement is a hard bound. **The other 69–99% of failures happen on grids that look
perfectly healthy.** Detecting those requires actually simulating the line failure — which is the
exact computation the neural network exists to replace. No larger rulebook closes that gap; it is
a property of the problem, not of the corpus.

This answers "would more rules have helped?" structurally, without needing to build progressively
larger rulesets and plot the curve.

---

## 5. Why the rulebook came out thin — the standards/simulator mismatch

The low yield in 4.1 is not an extraction failure. **The documents and the simulator are about
different things, and the mismatch runs both ways.**

### 5.1 The simulator is coarser than the standards regulate

Grid codes govern **equipment**: a generator's power factor, a connection point's frequency
response, how fast a protection relay must trip, how long a module must ride through a voltage
dip. The simulator models a **network**: buses, lines, aggregate power flows, quasi-static
physics.

| what the rule needs that the simulator lacks | rules blocked | share of corpus |
|---|---:|---:|
| time / duration | 782 | 32.8% |
| frequency / rate-of-change-of-frequency / droop | 683 | 28.7% |
| protection relay settings | 528 | 22.2% |
| which specific piece of equipment is meant | 364 | 15.3% |

**1,332 rules — 54.1% of the entire corpus — are blocked on just two missing capabilities:
frequency and time.** Those would become *assessable* if the simulator carried them. Assessable
is not useful: they would still have to survive the quality checks, and this corpus lost 60% of
its rules at that step.

### 5.2 The standards are silent about most of what the simulator provides

The reverse direction is less obvious and more damaging. Of fourteen quantities the simulator
measures, only **four** have any surviving rule — three once an inert one is dropped.

**Nothing in the surviving corpus governs topology.** No rule mentions line outages, which is the
entire subject of the task.

The reason is structural: grid codes govern **connection and equipment compliance**. Contingency
screening is an **operational** activity, and where standards address it they do so as process
requirements ("the operator shall perform studies annually"), not as conditions on live telemetry.

> **A safety layer built from connection codes cannot govern operations, because the corpus was
> never about operations.**

That is a stronger and more defensible claim than "the simulator is too coarse," because it does
not require the simulator to be at fault, and it generalises to any quasi-static power-flow
simulator rather than to Grid2Op specifically.

### 5.3 A structural reason the American standards in particular yield nothing

NERC standards are **criteria-referencing by design**. They say *"within its applicable Facility
Rating"*, and a separate standard (FAC-008) then requires each utility to maintain its own
documented method for computing that rating. **The number is deliberately never printed in the
standard.** No amount of prompt engineering extracts a threshold that does not exist on the page.

Visible directly in the output: the two documents in the corpus that are genuinely about
post-contingency performance produced rules like `loading_pct > relay_loadability_limit_pct` —
correct in form, with the threshold left as a free variable, because that is how the source is
written.

### 5.4 Where numeric rules would actually be found

Investigated but **not acted on** — see 5.5.

| source | what it would supply |
|---|---|
| **PJM Manual 14B** | post-contingency voltage as literal numbers (0.92–1.05 pu), thermal as "within the emergency rating", plus a *voltage-drop* criterion |
| **NERC FAC-011-4** | the predicate *structure*: an operating limit is thermal rating + voltage limit + stability limit, pre- or post-contingency |
| **EU System Operation Guideline (2017/1485)** | the *operational* sibling of the connection code already in the corpus — operational security limits, and a formal N-1 criterion. ⚠ Article 25 text not yet verified directly |
| ISO-NE, NYISO, WECC planning criteria | same category: regional criteria with explicit numeric post-contingency limits |

The corpus is biased toward **connection** codes and away from **operational security** codes.
That single sentence explains the yield.

### 5.5 Why we are not re-running extraction

A perfect document set supplies two predicates: post-contingency **thermal** (already in hand —
it is the one rule that works) and post-contingency **voltage** (measured as unable to
discriminate on these grids, because they operate at 1.05–1.08 pu and never enter the ±5% band a
grid code specifies). Most of the gain would be more restatements of the rule already in hand,
and §4.4 bounds what that rule can reach.

Exactly one genuinely new capability appears above — PJM's **voltage-drop** criterion, a change
rather than a level, so the band problem does not apply. It is physically computable, but the
generated datasets discard post-contingency voltages, so it needs a data regeneration and a
vocabulary extension. **Treat it as a scoped future experiment, not a reason to re-extract.**

---

## 6. Negative results that are findings

Three task designs were built, probed, and rejected **before** committing compute to them. This
is a methodology contribution, not wasted work.

**The original 4-class task was solvable by four if-statements.** The labels (`normal`,
`overload`, `line_trip`, `cascade`) turned out to be a closed-form function of the observation:
four ordered threshold rules reproduce them with **zero disagreements across all 315,000 records**
of both grids. A symbolic layer scored 100% where the trained network reached 0.83 — meaning "does the
symbolic layer add value over the network?" was rigged before it ran. *(§9.1.)*

**Forecasting faults was unlearnable.** Line-trip onset in the simulator is an unconditional coin
flip by construction, so the best possible rule scores **1.00× the trivial all-positive baseline**
at every horizon of three steps or more. No model can beat "always say at-risk". *(§9.2.)*

**Forecasting overloads was learnable but exhausted by one threshold.** A gradient-boosting model
on the full feature vector reached F1 0.163 against a single rho threshold's 0.160 — with *worse*
average precision. *(§9.2.)*

**N-1 screening is neither.** It cannot be restated by a rule, because producing the label
requires a power-flow solve. That is what permanently un-rigs the comparison, and it is why the
result in §2 means something.

There is also a pair of instructive ones, both cases of **a pipeline failure that looked exactly
like a finding.**

*First:* a stage in the extraction pipeline reported that only 6 of 2,463 rules were expressible
— a dramatic negative result about standards. It was a parsing bug: 89% of the corpus was never
assessed at all. A void run and a genuinely inexpressible corpus produce identical summary
statistics. It is now guarded by tests and a tripwire. *(§10.2.)*

*Second, and subtler:* the final validation stage rejected 31 of 32 rules and looked like a
verdict on the corpus. It was a verdict on the question being asked. The auditor was asked whether
each rule was **stated** in the source text — but the preceding stage's entire job is to rewrite
rules out of the standard's language into simulator variables, so nothing it produces is ever
stated in the source text. Asked instead whether each rule was a faithful **operationalization**,
the same model on the same input confirmed 11 instead of 1.

What makes this worth reporting is *how* the two runs differed. The rejections in the strict run
cited reasons that were fabricated — rules were rejected for "using an undefined variable
`loading_pct`", where `loading_pct` was listed in the model's own instructions two paragraphs
earlier. **The verdicts were perfectly stable across runs while the stated reasoning was
invented.** That is invisible unless you store the reasons, and the first run had thrown them
away. *(§12.1–§12.3.)*

### 6.1 The strongest methodological result: two independent filters, same answer

Two entirely separate methods were used to decide which extracted rules were worth keeping, and
neither had access to what the other used:

- **An empirical filter.** No language model. It ran every candidate rule against real grid data
  and asked *does this fire, and does it discriminate?*
- **A textual auditor.** No grid data. It read the source standards and asked *is this a faithful
  reading of the document?*

Both kept the same family — *"is any line loaded past 100% of its rating?"* — and both discarded
the voltage rules, the auditor because they came from ride-through tables and time-bound operating
envelopes that had been misread as instantaneous limits.

Either result alone invites an easy objection: *your empirical cutoff is mistuned*, or *your
prompt is wrong* — and the strict run shows the second objection can genuinely be correct.
Convergence defeats both. **The four surviving rules are a property of the mismatch between what
standards regulate and what the simulator models, not an artifact of either filter.**
*(Empirical arm: §10.3. Textual arm: §12.2.)*

---

## 7. What exists, and what is left

**Built and measured:**

- The N-1 model itself, trained on the 36-bus grid, evaluated on two unseen topologies
- A four-stage extraction pipeline over 16 standards documents
- A deterministic quality filter that rejects rules firing on healthy grids
- A deterministic audit distinguishing "this rule is wrong" from "this rule can't do anything"
- The gate, with asymmetric semantics: it blocks over-permissive verdicts and never blocks
  cautious ones
- An evaluation harness reporting both arms, the conditional view, and the structural ceiling
- The full four-stage pipeline, run end to end, including the final validation stage and a
  controlled A/B over the question it asks
- A knowledge graph recording where every surviving rule came from, so the gate can cite the
  clause and standard that authorise each block (below)
- **222 tests, all passing**

### 7.1 The knowledge graph, and why it was built last

The graph answers one question: *when the gate blocks, on whose authority?* It records
**Document → Clause → Rule → Predicate**, so a block can name the exact section of the exact
standard behind it — and every other standard that says the same thing.

It was built last on purpose. A first attempt, made before the corpus was settled, organised the
graph around the physical grid: a node per bus, a node per transmission line, rules attached to
whichever component they mentioned. That failed twice over. Only 3.5% of the extracted rules named
a specific bus or line at all, so **96% of its 6,632 connections carried no information** — they
were every rule attached to a single catch-all node. And because it was built from one grid's
wiring, it could not be used on the other two grids the model is tested on, even though the rules
themselves are grid-independent. That version has been deleted.

The replacement holds **36 nodes and 40 connections**, is grid-independent, and every connection
means something. Using it changes no result: the gate reads exactly the same rules whether it is
handed the flat file or the graph, which is checked by a test and was verified by re-running all
three evaluations and comparing every reported number.

**Remaining:** two optional robustness items — repeating the validation run at several random
seeds, and re-reporting the raw model under the same held-threshold protocol this document uses.
Neither changes the shape of the result. Sequence: §17.

---

## 8. Where the evidence lives

| claim | derivation | artifacts |
|---|---|---|
| shield results, all three grids | §13.1 | `results/shield/shield_<tag>_validated.json` · `results/failures/failures_<tag>.jsonl` |
| the result replicates across extraction runs | §13.6 | `results/shield/shield_<tag>_run3.json` |
| validation A/B, and the fabricated reasoning | §12.1–§12.4 | `validated_{strict,translated}/*_rejected.jsonl` |
| filtering the corpus improved the gate | §13.1 (both corpora, same table) | — |
| the "precision rises off-distribution" correction | §13.2 | — |
| two independent filters converge | §6.1 · §10.3 · §12.2 | — |
| the knowledge graph, and the v1 that was deleted | §16.1–§16.2 | `kg/knowledge_graph.json` |
| 10 clauses / 4 documents / 2 bodies behind one rule | §16.3 | `kg/kg_corroboration.svg` |
| the graph changes no measured number | §16.4 | `results/shield/shield_<tag>_kg.json` · `tests/test_kg.py` |
| the rule is not circular (91–96%, not 100%) | §13.4 | — |
| structural ceiling on what rules can reach | §13.3 | — |
| extraction yield and the audit | §10.1, §10.3 | `evaluation/audit_rules.py` · `results/audit/audit_run3.json` |
| standards/simulator mismatch, both directions | §10.4, §15 | — |
| the 4-class task was closed-form | §9.1 | — |
| forecast tasks rejected | §9.2 | — |
| raw model cross-topology performance | §14 | `gnn_n1_tightening.md` |
| voltage base-kV contract and why it matters | §11 | `data/grid_dataset_<tag>_basekv.json` |
| the four surviving rules themselves | §12.5 | `validated_translated/all_rules_deduped.jsonl` |
| what was designed but never run | §17.1 | — |

**Reproducing §2:** the evaluation harness is `evaluation/eval_shield_n1.py`. To reproduce the
numbers in this document exactly, point it at the validated four-rule corpus:

```powershell
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag wcci2022 --rules validated_translated\all_rules_deduped.jsonl
```

⚠ Pointing it at `translated_rules\guarded\` instead reproduces the *earlier* 32-rule numbers
(+0.0655 on WCCI, 51.4% precision at home). Those are superseded — see §2.2 — but kept in §13.1
alongside the new ones, because the difference between the two corpora is the evidence for
§2.2.

⚠ Two threshold protocols appear in this document and must never be mixed in one column.
**Every headline figure — §2, §13 and §14 alike — uses the held threshold**: 0.8849, selected once
on the neurips2020 validation split and applied unchanged to all three grids. §14 additionally
reports a **best-threshold-on-each-grid** figure in a separate, explicitly labelled table, kept
only as an oracle ceiling because §2's headline claim is measured against it. Held and oracle give
different numbers for the same model on the same grid (0.8956 vs 0.8972 in-distribution, and
0.4167 vs 0.4477 on case14); that is the size of the oracle's optimism, not a discrepancy.
*(This note previously said §14 was best-threshold and tracked as an open item. That item was
closed on 2026-08-21 — see §17 item 2.)*

---

# Appendix — derivations

The narrative ends at §7. Everything below is the measurement record behind it: the exact tables,
protocols and counts each claim was derived from. It was folded in from the retired build plan
(`archive/component_d_plan.md`) on 2026-08-21 so that this document stands on its own. Section
numbers here are what §1–§8 cite.

---

## 9. Task selection — why N-1 screening, and what it replaced

Three task designs were built or probed. Two were rejected **by measurement, before any full
training run against them**. The rejections are a contribution, not wasted effort: they establish
that a task must pass **two independent checks** before it can support a neuro-symbolic
comparison — **non-vacuity** (the target is not predictable from the base rate alone) and
**non-exhaustion** (a *model* beats the best *rule*, not merely the base rate).

### 9.1 The 4-class target was closed-form

`get_state_label()` defined the labels as:

```python
if max_rho >= 1.0:                  return "overload"
if active_lines == env.n_line:      return "normal"
if active_lines == env.n_line - 1:  return "line_trip"
return "cascade"
```

Every branch is expressible in the shield's own vocabulary (`rho_max`, `n_tripped_lines`).
Re-verified over **every record of both datasets** on 2026-08-20, immediately before the datasets
were deleted:

| dataset | records | rule == stored label | disagreements |
|---|---:|---:|---:|
| neurips2020 (36-bus) | 300,000 | **300,000 (100.000000%)** | **0** |
| case14 (14-bus) | 15,000 | **15,000 (100.000000%)** | **0** |
| **total** | **315,000** | **315,000 (100%)** | **0** |

⚠ This supersedes an earlier "100% agreement on 55,000 records" figure that survives in older
drafts and some code comments — the check was re-run over every record, not a sample.

The trained GATv2 reached **0.8277 macro F1** on the same target after three rounds of
architecture work. A symbolic layer scores **100%**. `fault_loc` was `obs.rho.argmax()`, so the
localization target was closed-form too.

**The honest reading, which must travel with the claim:** this is not evidence that symbolic
methods beat neural ones at fault classification. It is evidence that the **benchmark was
mis-specified** — the labelling function was defined over the same observation the model was
given, so the label carried no information the input did not already contain. Any measurement of
"does the symbolic layer add value?" on that task was rigged before it ran, because the symbolic
layer *is* the labeller.

⚠ **Not reproducible.** All classify artifacts were deleted on 2026-08-20 (4.81 GB, mostly
untracked). The table above is the only surviving record, which is why it was measured before
deletion rather than after. Do not cite 0.8277 as a reproducible figure — and the checkpoint that
produced it was already undeterminable between two separate runs before removal.

### 9.2 Both forecast targets failed, in opposite directions

**Any-fault forecast — unlearnable.** Line-trip onset is an unconditional Bernoulli draw
(`np.random.rand() < FAULT_PROB`, 0.05) on a uniformly random connected line, so it is
unpredictable *by construction*, and trips are 71–88% of all positives. Probe on the 36-bus
training set, **1,995 quota-unbiased chronics / 321,966 reconstructed steps**, scoring the best
possible single threshold on current `rho_max` against the all-positive baseline F1 = 2p/(1+p),
on currently-calm frames only:

| H | ANY-fault base | ANY lift | OVERLOAD base | OVERLOAD lift |
|---:|---:|---:|---:|---:|
| 1 | 5.1% | 1.12× | 0.7% | **9.03×** |
| 2 | 9.9% | 1.02× | 1.4% | **7.02×** |
| 3 | 14.8% | **1.00×** | 2.3% | 3.88× |
| 4 | 18.8% | **1.00×** | 3.0% | 3.11× |
| **6** | 27.0% | **1.00×** | 3.6% | 2.76× |
| 8 | 34.3% | **1.00×** | 4.2% | 2.48× |
| 12 | 46.3% | **1.00×** | 4.9% | 2.00× |
| 18 | 59.9% | **1.00×** | 5.6% | 1.96× |
| 24 | 69.5% | **1.00×** | 6.3% | 1.59× |

Lift is **1.00× at every H ≥ 3** — no rule, and therefore no model, beats "always say at-risk".

**Overload-only forecast — learnable, but exhausted by one rule.** The 2.76× lift at H = 6 looked
like headroom. It was not: HistGradientBoosting on 30 summary features (chronic-level split,
56,441 calm frames / 2,227 positives) reached **F1 0.163 against the single threshold's 0.160**,
with *worse* average precision (0.125 vs 0.127). Trend features over lags 1/3/6 on 60,000-frame
contiguous pilots changed nothing on either topology (lift 0.67–1.02× across H ∈ {2…18}). A
snapshot carries no information about future exogenous load.

> **The transferable lesson:** a lift over the all-positive baseline says the target is not
> vacuous. It says nothing about whether a model can beat the rule. Both checks are needed, and a
> literature reporting only the first will accept targets on which the neural component does no
> work.

### 9.3 N-1 screening fails neither test

Pre-generation probe on the 36-bus grid, 17,652 contingency labels from 300 frames,
`obs.simulate()` per line:

| Predictor | F1 |
|---|---:|
| all-positive baseline | 0.311 |
| best threshold on **global** current `rho_max` | 0.289 — **1.04×, i.e. nothing** |
| best single rule: **flow on the line being removed** | 0.608 |
| model with **network context** | **0.868** (AP 0.937 vs 0.628) |

Three properties follow, one per thesis claim:

1. **The present state does not determine the answer** — global `rho_max` is worthless (1.04×), in
   direct contrast to classify, where four such thresholds sufficed.
2. **The answer is topological** — within a single frame some contingencies violate and others do
   not, so a graph-level prediction cannot express it.
3. **There is a real gap for the model to occupy** — 0.608 → 0.868. Neither end is degenerate: the
   symbolic baseline is respectable rather than trivial, and the model's advantage is substantial
   rather than noise.

**Mixed-frame statistics, measured over all 22,000 frames of all three datasets** (⚠ corrected
2026-08-20 from an earlier "100% of sampled frames"):

| grid | strictly mixed | all contingencies violate | none violate |
|---|---:|---:|---:|
| neurips2020 | **98.80%** | 1.20% | **0.00%** |
| case14 | **98.48%** | 1.52% | **0.00%** |
| wcci2022 | **96.43%** | 3.57% | **0.00%** |

Quote 96–99% for strictly mixed. What *is* 100% is that **no frame on any grid is entirely
secure**. Base rate is 16–26% violation depending on topology, so this is not a rare-event
problem; labels cost ~170 power-flow solves per second.

**Why this un-rigs the comparison permanently:** classify failed because four rules reproduced it
exactly. N-1 fails the same test in the opposite direction — **no rule over the present
observation can restate the label, because producing it requires a post-contingency power-flow
solve.**

---

## 10. Extraction — the pipeline, its yields, and one bug worth reporting

### 10.1 Corpus measurements, all 16 documents

| Measure | Value |
|---|---|
| Stage-1 candidates | **2,463** (v1: 1,372) |
| All variables derivable from a record | **693 (28.1%)** |
| Mixed (some derivable, some not) | 571 (23.2%) |
| Distinct variable names | 2,083 |

Blocked families (overlapping): **FREQ 625 (25.4%)**, **TIME 582 (23.6%)**, **OTHER 610 (24.8%)**,
**PROT 204 (8.3%)**. Frequency is irreducible — Grid2Op targets human-timescale control, and
quasi-static power flow assumes uniform system frequency.

**Entity skew:** `Line` 59 and `Bus` 28 out of 2,463 — **3.5%**. Everything else is module-,
facility- or system-level, so entity-based retrieval carries almost no information. This is the
number that killed the v1 knowledge graph (§16.1).

Blocked-capability counts from the run-2 verdicts (overlapping, 2,381 untranslatable):

| missing capability | rules | share |
|---|---:|---:|
| TIME / duration | 782 | 32.8% |
| FREQ / ROCOF / droop | 683 | 28.7% |
| PROT / relay / settings | 528 | 22.2% |
| ENTITY resolution | 364 | 15.3% |
| other | 564 | 23.7% |

**1,332 candidates (54.1% of the corpus) are blocked on exactly two missing capabilities,
frequency and time**, and would become *assessable* if the observation carried them. That is a
measurement, not a hope — but "assessable" is not "useful": they would still have to survive the
guard and the audit, and this corpus rejected 59.8% at the guard alone.

### 10.2 Stage 2 run 1 — a parsing bug that presented as a corpus property

Stage 2 ran to completion and reported **6 translated rules out of 2,463 (0.24%)**, which read as
a devastating finding about the standards. It was not. **2,190 of 2,463 rules (88.9%) came back
`NO_VERDICT`** — only 233 (9.5%) were ever assessed.

Cause: every worked example in `TRANSLATE_PROMPT` omitted `rule_id` while the `TranslationResult`
schema required it, so every entry failed Pydantic validation and was dropped by a bare
`except ValidationError: pass`. **The model was answering correctly and the answers were being
discarded.**

Fixed in `common.py` (rule_id in every example) and `translate.py` (`build_result_map`, positional
fallback, no silent drops, `--debug-raw`, a NO_VERDICT tripwire that logs an ERROR above 20%).
Guarded by `tests/test_translate_mapping.py`.

> **The lesson, which recurs at stage 3 (§12.1):** a pipeline that cannot distinguish *"assessed
> and rejected"* from *"never assessed"* will report the second as the first, and the totals look
> identical either way. Check `total_no_verdict` before believing any yield.

### 10.3 The guard and the audit — two questions that come apart

The polarity guard (stage 2.5, deterministic, no LLM) asks *"is this rule wrong?"* — does it fire
on healthy grids? `evaluation/audit_rules.py` asks *"can it do anything?"* — a constraint firing
on 0% of healthy frames passes the guard, but if it also fires on 0% of abnormal frames it can
never block and contributes nothing.

| | run 2 | run 3 |
|---|---:|---:|
| translated | 82 | **58** |
| distinct (entity, condition) | 49 | 38 |
| CONSTRAINT / AFFIRMATION | 54 / 28 | 43 / 15 |
| **guard-kept** | **33** | **32** |
| guard rejection rate | 59.8% | **44.8%** |
| `loading_pct > 100` rules | **10** | **10** |

**What changed between the runs — the proxy-substitution fix.** Run 2 produced R_769, which ORed
an offshore circuit's rating schedule into the thermal check:
`loading_pct > 100 or current_a_max > 580 or apparent_power_mva_max > 132`. Both absolute terms
are **grid-wide maxima** while the rating belongs to **one named circuit**, so the rule fired on
**100% of healthy frames on all three topologies**. `TRANSLATE_PROMPT` now states that the ratio
*replaces* the absolute figure, carries the rating-schedule case as a worked example, and names
the defective string as a `NEVER`.

The deeper hole was the prompt's SELF-CHECK, which supplied healthy values for only **7 of 14**
variables — so a rule naming `apparent_power_mva_max` had nothing to substitute against and passed
vacuously. It now covers all 14 using measured healthy readings. Guarded by three tests in
`tests/test_condition_lint.py`, two of which evaluate the defective condition against the prompt's
own numbers rather than grepping for text.

In run 3 a `grep` for an absolute rating ORed into a condition returns **nothing**. R_769 left by
a different door than the fix intended — it is now in `*_untranslatable.jsonl` as *"compound:
requires time_seconds (post-fault condition)"*, which is a defensible call and arguably the better
one, since the source table does carry 6hr/20m/10m/5m/3m duration columns.

⚠ The yield fell 82 → 58 where the recorded prediction was "~82 again; the fix is precision, not
recall." **The prediction was wrong in direction** and that is recorded rather than quietly
dropped. The composition vindicates the mechanism: the working family is unchanged at exactly 10,
and the 24 lost rules are voltage and power-factor restatements the audit had already measured as
inert or uninformative.

Audit verdicts (`results/audit/audit_run3.json`):

| verdict | run 2 | run 3 | representative |
|---|---:|---:|---|
| INERT | 5 | 4 | `voltage_pu_max > 1.5` — never fires on any class on any grid |
| UNINFORMATIVE | 9 | 7 | wide voltage bands holding on `normal` **and** `overload` alike (best margin **+0.006**) |
| TOPOLOGY_DEPENDENT | 9 | 11 | voltage constraints firing on abnormal frames on some grids only |
| **USEFUL** | **10** | **10** | **all ten are `loading_pct > 100`** |
| can do work | 19/33 (11 distinct) | **21/32 (13 distinct)** | |

**The fire-rate distribution is cleanly bimodal on both runs** — run 2: 43 rules at ≤ 0.01, 39 at
≥ 0.75, **nothing between 0.1 and 0.9**; run 3: 33 and 25, again nothing between. The 0.5 cutoff
is measured, not chosen.

⚠ **Two caveats on the audit.** (1) `loading_pct > 100` scores 1.000 on abnormal frames partly
because the `overload` label *is* `rho_max >= 1.0` — a label-restatement effect, so the USEFUL
verdict must not be quoted as independent validation. (2) The audit scores against *classify*
labels while the live task is N-1. It is triage — which rules separate grid states at all — not a
shield evaluation.

**The ±5% voltage band is contaminated on two grids and clean on the third:**

| topology | v_min (median) | v_max (median) | healthy frames inside 0.95–1.05 |
|---|---:|---:|---:|
| neurips2020 | 1.047 | 1.080 | **0.0%** |
| case14 | 1.011 | 1.100 | **0.0%** |
| wcci2022 | 0.982 | 1.050 | **100.0%** |

This is the strongest single argument for guarding against **all three** topologies: guarding on
wcci2022 alone would have passed every one of those rules. It reproduced independently on run 3
(19 rules differing by > 0.25 across grids, all in the same direction).

**A context bug the guard caught, not a rule bug:** `power_factor_min`, as a minimum across
energized lines, has median **0.000** on healthy case14 frames — lightly loaded lines carry
near-pure reactive flow, so the minimum is dominated by electrically irrelevant lines and
`pf < 0.9` fired on 100% of healthy frames. Replaced with `power_factor_at_max_load` (median
0.930, p5–p95 0.926–0.936).

### 10.4 Variable coverage — the mismatch runs both ways

The surviving rules touch **4 of the 14 vocabulary variables**:

| covered | uncovered (no surviving rule) |
|---|---|
| `voltage_pu_min` (18), `voltage_pu_max` (16), `loading_pct` (10), `generation_load_imbalance_pct` (2) | `rho_max`, `n_tripped_lines`, `any_line_tripped`, `active_power_mw_max`, `reactive_power_mvar_max`, `apparent_power_mva_max`, `power_factor_at_max_load`, `current_a_max`, `total_generation_mw`, `total_load_mw` |

**Nothing in the surviving corpus governs topology** — no rule mentions `n_tripped_lines` or
`any_line_tripped`, which is the entire subject of the N-1 task. `generation_load_imbalance_pct`
is INERT, leaving **3 of 14** with an informative rule.

⚠ Do **not** write "what the simulator provides can be governed by the rules that survived." It is
measurably false at 4/14 coverage, and an examiner can check it in one command.

---

## 11. The voltage contract, and why base kV is not a constant

### 11.1 The contract (binding)

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
   **~100% false block rate**.
2. **Energized lines only.** Tripped lines report `v_or = 0`, so an unmasked `min` fires every
   undervoltage rule on any frame with a disconnection.

On a total blackout (`mask.sum() == 0`) voltages are **omitted from the context**, so voltage
rules raise NameError → NOT_EVALUABLE → never block. Defaulting to 0.0 would read as catastrophic
undervoltage.

⚠ The `v_or / 150.0` conversion that appears in older design notes is **superseded**. It survives
in exactly one legitimate place: the GNN's `mean_v` node feature, where it is an input scale
factor and not a per-unit conversion. The two must not be conflated.

### 11.2 Backend nominal, not empirical median — and why healthy frames read 1.06 pu

The sidecars were originally built with `--empirical` (per-line medians over normal frames). The
two methods disagree far past tolerance:

| tag | backend nominal (kV) | empirical median (kV) | median disagreement | lines > 2% |
|---|---|---|---|---|
| neurips2020 | 138, 345 | 145.9–149.0, 361.2–369.6 | 6.63% (max 7.39%) | **59 / 59** |
| case14 | 14, 20, 138 | 14.9, 21.2–22.0, 139.5–142.1 | 4.23% (max 9.09%) | **16 / 20** |

The disagreement is near-uniform across both voltage levels — 147/138 = 1.066, 365/345 = 1.058 —
the signature of generator voltage setpoints fixed at ~1.05–1.06 pu, the IEEE test-case convention
these Grid2Op environments inherit. **It is not a defect in the grids** and not noise in the
estimate.

**The backend nominal is the contract.** The empirical originals are preserved as
`*_basekv_empirical.json` as the counterfactual arm and must not be deleted or used.

The consequence, stated plainly so nobody rediscovers it as a bug: **healthy frames read ~1.06 pu,
not ~1.00 pu**, so a standards band like `voltage_pu_max > 1.05` fires on essentially every
healthy frame and the guard rejects it at a fire rate near 1.0.

Why that is the right trade:

1. **The empirical base is centred by construction** — it sets the healthy median to 1.00 pu using
   the same data the rules are then tested against. This project's spine is the discovery that the
   classify target was rigged before it ran (§9.1); shipping a second, subtler self-calibration
   inside the shield's own context is the first thing an examiner would go for. ⚠ Note the
   empirical base is **not degenerate** — it fixes the *centre*, not the *spread*. The objection is
   provenance, not lack of signal.
2. **The cost is small, because the N-1 label is thermal.** `label_n1` sets
   `violation[k] = int(sim_done or rho >= 1.0)`, so the rules that can bear on the target are the
   dimensionless `rho` / `loading_pct` ones, unaffected by the base entirely.
3. **A rejected voltage rule is a measured finding, not a gap** — rejects are logged with their
   fire rates.

The reportable form:

> Grid-code voltage bands are calibrated to utility operating practice, not to IEEE test-case
> setpoint conventions, so they do not transfer to Grid2Op reference grids without rebasing.

---

## 12. Validation — the A/B, and a validator that fabricated its reasoning

### 12.1 Run 1: 31 of 32 rejected, with no evidence of why

`validate.py` against the run-3 guarded corpus returned **1 confirmed, 31 rejected, 0 flagged, 0
NO_VERDICT**. The mechanism was sound — every rule got a parseable verdict with a matching
`rule_id`, so this was *not* §10.2's failure class repeating — but `validate.py` incremented
`n_rejected` and **discarded the `Verdict`, including the model's `reason` field**. 31 rejections,
zero explanations.

**A rejection without its reason is not a finding**, and this one could not be told apart from a
broken prompt. Two things were established before re-running, both without an LLM:

1. **All 32 rules pass the validator's own mechanical criteria.** Criteria 4 (vocabulary) and 7
   (healthy-grid self-check) are deterministically checkable; a substitution pass found **32/32
   passing both**. Only 2 rules trip criterion 6's ride-through heuristic. At most 2–4 of the 31
   rejections rested on the prompt's explicit criteria.
2. **At least one rejection is demonstrably wrong.** TPL-001-5.1 Table 1 (f) reads *"Applicable
   Facility Ratings shall not be exceeded"*; R_167 renders that `loading_pct > 100`, CONSTRAINT.
   Rejected.

`validate.py` now persists `<stem>_rejected.jsonl` with the full verdict, carries a `reject_rate`
with an ERROR tripwire above 80%, and takes `--prompt-variant`. Guarded by
`tests/test_validate_rejections.py`. **Never merge a validation run whose rejections were not
persisted.**

### 12.2 The A/B — one variable, perfectly nested outcome

Two arms over the identical 32-rule input, identical model, temperature 0.0, differing **only in
criteria 1–3**. Criteria 4–7 and the output contract are a shared string.

| arm | question asked | confirmed | rejected | unique after dedup |
|---|---|---:|---:|---:|
| `strict` | is the constraint **stated** in the source text? | 1 | 31 | 1 |
| `translated` | is the condition a faithful **operationalization**? | **11** | 21 | **4** |

`strict` reproduced run 1 exactly — same count, same survivor. The outcome is **perfectly nested**:
every rule `strict` confirmed, `translated` also confirmed; nothing was confirmed by `strict`
alone; all 21 rules `translated` rejected were also rejected by `strict`. **The entire difference
is the ten `loading_pct > 100` rules, and they moved as a block.**

The root cause is a category error. Criterion 1 was written for stage-1 candidates, whose
conditions were still in the standard's own terms. Applied to a **stage-2** condition it cannot be
satisfied: translation's entire job is to move the condition out of the standard's language, so a
translated condition is *never* stated in the source text.

### 12.3 The fabricated reasoning — a methodological result in its own right

The persisted `strict` reasons are not merely strict, they are incoherent:

| rule | `strict` reason (verbatim) |
|---|---|
| R_167 | "uses an undefined variable `loading_pct` and the threshold 100 is not explicitly stated" |
| R_1443 | "uses '100' which is not a valid variable" |
| R_858 | "includes a unit (100.0), violating the allowed syntax" |
| R_260 | "includes non-Python syntax such as 'or' without proper parentheses" |
| R_1717 | "the correct variable is `loading_pct`, but ... incorrectly uses `loading_pct > 100`" |

`loading_pct` is in the vocabulary block injected into that same prompt. Numeric literals and `or`
are explicitly permitted by criterion 4. `100.0` is a float, not a unit. R_1717's reason
contradicts itself inside one sentence.

**The mechanism: forced by criterion 1 into a rejection it could not justify on the merits, the
auditor reached for the vocabulary criterion and hallucinated a violation of it.** Eleven of the 31
rejections cite a criterion-4 breach that the deterministic pass (§12.1) proves does not exist.

> **The verdicts were stable across runs while the reasoning was fabricated, and only persisting
> the reasons exposed it.** A pipeline that logs counts and discards justifications cannot tell a
> correct filter from a broken one.

### 12.4 The 21 shared rejections are substantive — and sharper than the guard

| rules | validator's ground | assessment |
|---|---|---|
| R_1833, R_1835, R_2282 | PRC-024 Attachments "only provide minimum time durations for voltage excursions, not a continuous limit" | **correct** — ride-through curves |
| R_116, R_120, R_121, R_123 | ENTSO-E NC RfG Table 6.1 gives "a minimum operating time of 60 minutes for the 0.85–0.90 pu range"; an AFFIRMATION of `normal` is inconsistent with a time-bound requirement | **correct**, and sharper than the fire-rate reading, which only saw them firing at 1.00 |
| R_1108, R_069 | "0.9–1.1 pu is an affirmation of normal, not a constraint to be inverted" | **correct** role catch |
| R_797, R_798 | source states ±5% for 400 kV and +10%/−15% for 230 kV, not flat 0.90/1.10 | **correct** threshold catch |
| R_1319 | STATCOM blocks at 0.2–0.3 pu — a ride-through range | **correct**, matches the deterministic flag |
| R_1926, R_1927, R_1098 | `voltage_pu_min < 0.95` is "the opposite logical requirement" of "at least 0.95 pu" | **WRONG** — for a CONSTRAINT that inversion is correct; the validator applied affirmation semantics to a constraint |
| R_797, R_798, R_434, R_741 | reason text says "requiring correction" | **verdict-selection defect** — diagnosed as fixable, then discarded rather than returned as CORRECT |

Roughly 17 of 21 rest on a defensible reading. **At least 3 are false rejections and 4 more should
have been CORRECT verdicts** — carry both when citing the yield.

### 12.5 The validated corpus

`validated_translated/all_rules_deduped.jsonl` — 11 confirmed → 4 distinct by condition + role:

| rule_id | role | severity | condition |
|---|---|---|---|
| R_858 | CONSTRAINT | high | `loading_pct > 100.0` |
| R_916 | CONSTRAINT | medium | `loading_pct > 100` |
| R_1443 | CONSTRAINT | high | `loading_pct > 100` |
| R_1154 | AFFIRMATION | medium | `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1` |

End-to-end yield: **2,463 candidates → 4 rules (0.16%)**. Affirmation coverage is `normal` only,
so the shield's Option B counterfactual (gating on *missing* support rather than on violation)
remains unmeasurable — now a settled property of the corpus across three runs, not an accident of
one.

`validated_strict/` is retained as the counterfactual arm. It is evidence, never an input.

---

## 13. The shield — full results, both corpora

Protocol: threshold **0.8849, selected on the neurips2020 validation split and held fixed across
all three topologies.** Every figure below is honest-protocol, not best-on-its-own-topology.
**Eval batch size 64** — load-bearing for the *levels* in this table but not for the *deltas*; see
§14 caveat 3 and `gnn_n1_tightening.md` §8. Read the delta and precision columns as the result.

### 13.1 Headline table

| topology | corpus | GNN F1 | +shield F1 | Δ | blocked | corrections | regressions | **int. precision** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| neurips2020 | guarded-32 | 0.8956 | 0.8982 | +0.0026 | 831 | 427 | 404 | 0.514 |
| neurips2020 | **validated-4** | 0.8956 | **0.9038** | **+0.0082** | 353 | 331 | 22 | **0.938** |
| case14 | guarded-32 | 0.4167 | 0.4188 | +0.0021 | 103 | 95 | 8 | 0.922 |
| case14 | **validated-4** | 0.4167 | 0.4188 | +0.0021 | 103 | 95 | 8 | **0.922** |
| wcci2022 | guarded-32 | 0.5577 | 0.6232 | +0.0655 | 24,777 | 21,306 | 3,471 | 0.860 |
| wcci2022 | **validated-4** | 0.5577 | **0.6253** | **+0.0676** | 22,559 | 21,065 | 1,494 | **0.934** |

**Dropping 28 of 32 rules improved or held every metric on every grid.** The trade is explicit:
validation cost a little coverage (neurips corrections 427 → 331, wcci 21,306 → 21,065) and bought
a large cut in bad blocks (neurips regressions 404 → 22, wcci 3,471 → 1,494). On case14 the result
is **bit-identical** — the voltage rules never fired there, so nothing was lost.

Blocks are `high` severity only under the validated corpus; every `critical` block in the guarded
runs came from a rejected voltage rule.

Reach — how often the gate can speak at all — is **0.31% / 0.087% / 3.04%** of contingencies.

### 13.2 ⚠ The superseded reading — do not restate it

An earlier version of this analysis claimed *the shield's overrides are coin flips in-distribution
(0.514) and reliable off-distribution (0.860, 0.922) — the symbolic layer holds while the neural
layer stops earning the benefit of the doubt.* **That reading is void.** It was an artifact of
averaging two rule families: 478 of neurips2020's 831 blocks fired on voltage rules at 0.13–0.20
precision against thermal's 0.92–0.94.

Validation removed exactly those voltage rules **on textual grounds, having never seen the
fire-rate data**. Intervention precision is then **0.938 / 0.922 / 0.934 — flat**. The corrected
claim is in §2.1, and it is stronger than the one it replaces.

### 13.3 The structural ceiling — what no corpus could reach

Of every missed violation the model commits, what share sits on a base case that a present-state
rule over the 14-variable vocabulary can even see?

| topology | missed violations | reachable, guarded-32 | reachable, validated-4 |
|---|---:|---:|---:|
| neurips2020 | 2,041 | 427 (20.9%) | **16.2%** |
| case14 | 18,592 | 95 (0.51%) | **0.51%** |
| wcci2022 | 68,166 | 21,306 (31.3%) | **30.9%** |

The complement is a hard bound: **no rule over this observation can reach the other 84% / 99.5% /
69%**, because those failures occur on base cases within every limit. Establishing them needs a
post-contingency power flow — the computation the GNN stands in for.

This answers "would more rules have helped?" **structurally** rather than by subsetting: on case14,
essentially no; on wcci2022, up to a third more, but only if a rule existed that fires where
`loading_pct > 100` does not.

### 13.4 The rule is close to a tautology — and the distance was measured

| topology | P(violation) | P(violation given base overloaded) | lift |
|---|---:|---:|---:|
| neurips2020 | 19.54% | **95.67%** | 4.90× |
| case14 | 27.75% | **91.16%** | 3.28× |
| wcci2022 | 24.76% | **94.74%** | 3.83× |

If this were the label-restatement trap the conditional would be 100%. At 91–96% the predicate
carries real but partial information. The honest reading: **the shield is validating N-1 doctrine
— a base case outside its limits cannot be declared secure against further loss — and the data
confirms that doctrine 91–96% of the time.** It is not discovering new physics.

⚠ Equally important: **the doctrine being enforced is hand-written in `validate_n1`, not
extracted.** The LLM pipeline supplied only the threshold, and that doctrine belongs to the
operational-standards class (NERC FAC-011, EU SO GL) that §15 shows is absent from the corpus.
Volunteer this in the write-up rather than defending it under question.

### 13.5 Shield health — every verification gate passes

| check | neurips2020 | case14 | wcci2022 |
|---|---|---|---|
| ERROR verdicts (must be 0) | **0** | **0** | **0** |
| NOT_EVALUABLE verdicts | 0 | 0 | 0 |
| Option B counterfactual (`unsupported`) | 0 | 0 | 0 |
| false block rate (base-kV regression check) | 0.36% | 0.007% | 0.47% |

Applicability is 100% on all three grids — every rule evaluated on every frame, which is what the
stage-2 AST lint plus the §11.1 voltage contract were built to guarantee.

The gates that must pass before trusting any of these numbers: `pytest tests/` green; case14 false
block rate far below 100% (catches base-kV regressions) with **zero ERROR verdicts**; and rerunning
any tag twice giving identical aggregates.

### 13.6 The result replicates across extraction runs

`eval_shield_n1.py` re-run against the **run-3** guarded corpus reproduces the run-2 numbers
**exactly, to every reported digit** — including every count of corrections, regressions, missed
violations and false alarms. Two independently generated rulesets, differing by 24 rules, produce
a bit-identical gate.

The mechanism: **`loading_pct > 100` is the only rule that ever fires as a constraint on a base
case, and both corpora contain exactly ten copies of it.** Every other rule is silent at inference.

Two things follow: the result **does not depend on the extraction run that produced it** (not a
given for a pipeline with an LLM in the middle), and it **bounds the corpus's contribution
honestly** — a ruleset can lose 29% of its rules and change nothing downstream.

⚠ Run 2's `translated_rules/` was overwritten when run 3 was copied into the same directory, and
neither is recoverable. That would have been a genuine hole in reproducibility; it is closed only
because the run-3 corpus gives the identical result. `translated_rules/guarded/` now holds run 3.

### 13.7 A pre-registration that paid off

Before stage 3 ran, this was recorded:

> "the shield numbers should be identical again *unless validation rejects `loading_pct > 100`
> itself* — which would be a substantive finding about the validator, not a numerical drift, and
> must be investigated rather than absorbed."

That is exactly what happened (§12.2). Had run 1 been accepted at face value, Component D would
have been left with one AFFIRMATION, which under Option A can never block, and **the shield would
have been silently reduced to a no-op.**

---

## 14. Raw model cross-topology performance

Generation protocol identical on all three topologies (`--task n1 --n1-stride 12`); only the
environment differs. All three sets verify clean: zero structural errors, zero all-secure frames,
and a global-`rho_max` lift of 1.08–1.16×, so **the task stays non-degenerate off the training
grid** — it has not collapsed into the closed-form trap of §9.1.

> **Protocol closed 2026-08-21.** This table previously reported the model at the **best threshold
> on each grid individually**, which uses the answer key to pick the cutoff and is not obtainable in
> deployment. It now reports the **held threshold** — 0.8849, selected once on the neurips2020
> validation split and applied unchanged to all three grids, the same protocol §13 uses. The
> best-threshold column is retained as an explicitly-labelled *oracle ceiling*, because §13's
> headline claim is measured against it.

**Headline figures — held threshold, the honest protocol:**

| topology | lines | contingencies | viol. rate | all-positive | best rule | **MODEL (held)** | recall | precision | missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neurips2020 *(in-dist, test)* | 59 | 113,205 | 18.4% | 0.3104 | 0.4639 | **0.8956** — 1.93× rule | 0.902 | 0.889 | 2,041 |
| **case14** *(unseen, smaller)* | 20 | 118,502 | 27.8% | 0.4345 | 0.5392 | **0.4167** — **0.77× rule, FAILS** | 0.435 | 0.400 | 18,592 |
| **wcci2022** *(unseen, larger)* | 186 | 742,472 | 24.8% | 0.3969 | 0.4915 | **0.5577** — 1.13× rule | 0.629 | 0.501 | 68,166 |

**Oracle ceiling — best threshold on each grid individually. Not obtainable in deployment; quote
only as a ceiling:**

| topology | model (oracle) | AP | gap over held |
|---|---:|---:|---:|
| neurips2020 | 0.8972 — 1.93× | 0.9615 | +0.0015 |
| case14 | 0.4477 — 0.83× | 0.4270 | **+0.0310** |
| wcci2022 | 0.5721 — 1.16× | 0.6419 | +0.0144 |

**The gap is the size of the optimism, and it grows off-distribution** — 0.0015 at home, where the
threshold was chosen, against 0.0310 on case14, twenty times larger. That ordering is expected: a
foreign grid shifts the model's score distribution, so a threshold fixed on the training grid sits
further from that grid's optimum. It is also why the old protocol flattered exactly the comparison
the thesis rests on.

**Generalisation is real but partial, and ASYMMETRIC in grid size.**

- **case14 is a harder failure than the old table showed.** At the held threshold the model scores
  **0.77× the single-rule baseline** — and **0.96× the all-positive baseline** (0.4167 vs 0.4345).
  On that grid the model is beaten not only by one threshold on one feature, but by answering
  "violation" unconditionally. Report it plainly.
- **wcci2022 still clears the rule**, 1.13× rather than the 1.16× previously quoted, at 0.629
  recall.
- **Scaling up costs far less than scaling down** — the opposite of the naive expectation, and the
  interesting result here. The conclusion is unchanged by the protocol fix; only the margins move.

⚠ **Caveats that must travel with this table.**

1. ✅ **Threshold protocol — CLOSED 2026-08-21.** The headline table above is now held-threshold,
   matching §2 and §13, so the same model on the same grid reads 0.8956 everywhere. The prediction
   recorded when this was an open item — *"a threshold picked on val and held fixed would lower all
   three rows, and would lower case14 and wcci2022 most"* — was **correct in direction and in
   ordering**: −0.0015 / −0.0310 / −0.0144. Both figures are kept, with the oracle row labelled as
   a ceiling. Do not mix the two in one column.
2. Harness verified, not assumed: the eval's rule baseline reproduces `verify_n1_dataset.py`'s
   independent computation from raw JSONL exactly (case14 0.5392, wcci2022 0.4915).
3. **Every figure in this table is conditional on the eval batch size, which is 64.** Measured
   2026-08-21: the tracked checkpoint scores **0.8972 at batch 64 and 0.9255 at batch 512** on the
   same split — a spread of 0.028 on identical weights and identical data. The cause is
   `BatchNorm(track_running_stats=False)`, which uses live batch statistics at inference, so batch
   composition changes the logits. Isolated rather than assumed: the positive count (20,801) and
   the rule baseline (0.4639) are constant across every batch size, so it is the forward pass and
   not the data. **Quote any model F1 with its batch size.** The shield's delta and intervention
   precision are stable across the same change (+0.0082 to +0.0072; 0.938 to 0.930), because both
   arms share one forward pass, so §13's claims are unaffected. Full table:
   `gnn_n1_tightening.md` §8.
   ⚠ This supersedes an earlier note here attributing the 0.8972 / 0.8872 gap to "a different
   run". It is a protocol difference, and 0.8872 does not reproduce from the tracked checkpoint at
   any batch size tested.
4. This arm is **not** affected by the normalization bug described in `gnn_n1_tightening.md` §3 —
   the N-1 checkpoint was trained after the fix. That caveat applies only to the frozen classify
   checkpoint.

### 14.2 Selecting the threshold against recall instead of F1 — measured 2026-08-21

The held threshold is chosen on the val split by maximising **F1**, which is symmetric: it prices
one missed violation exactly equal to one false alarm. §13's gate is built on the opposite premise
— it blocks only over-permissive verdicts because a missed violation can cascade while a false
alarm wastes an operator's time. An examiner is entitled to ask why a symmetric metric was used to
tune a deliberately asymmetric system.

Rather than invent a cost ratio, the whole trade curve was measured. Protocol is unchanged in
shape: the threshold is **always** selected on the neurips2020 val split and then held fixed across
all three grids. Only the *selection objective* varies — F-beta with beta > 1 prices a missed
violation at beta² false alarms. Reproduce with `evaluation/sweep_threshold.py`; recorded at
`results/threshold/threshold_sweep.json`.

| grid | objective | threshold | F1 | recall | precision | **missed** | false alarms |
|---|---|---:|---:|---:|---:|---:|---:|
| neurips2020 | **F1 (current)** | 0.8849 | **0.8956** | 0.902 | 0.889 | **2,041** | 2,331 |
| neurips2020 | F2 | −0.7896 | 0.8558 | 0.953 | 0.777 | **987** | 5,689 |
| neurips2020 | F5 | −2.6781 | 0.7076 | 0.984 | 0.552 | **324** | 16,596 |
| case14 | **F1 (current)** | 0.8849 | **0.4167** | 0.435 | 0.400 | **18,592** | 21,431 |
| case14 | F2 | −0.7896 | **0.4330** | 0.502 | 0.381 | **16,390** | 26,826 |
| case14 | F5 | −2.6781 | **0.4428** | 0.600 | 0.351 | **13,168** | 36,453 |
| wcci2022 | **F1 (current)** | 0.8849 | **0.5577** | 0.629 | 0.501 | **68,166** | 115,283 |
| wcci2022 | F2 | −0.7896 | 0.5361 | 0.689 | 0.439 | **57,132** | 162,165 |
| wcci2022 | F5 | −2.6781 | 0.5004 | 0.767 | 0.371 | **42,793** | 238,795 |

**The exchange rate — extra false alarms bought per missed violation avoided, against F1:**

| grid | F1.5 | F2 | F3 | F5 |
|---|---:|---:|---:|---:|
| neurips2020 | 2.2 | 3.2 | 4.8 | 8.3 |
| case14 | 2.4 | 2.5 | 2.6 | 2.8 |
| wcci2022 | 4.0 | 4.2 | 4.5 | 4.9 |

Three things this establishes.

**1. On the training grid the choice is cheap; off-distribution it is not.** Halving neurips2020's
missed violations (2,041 → 987) costs 3.2 false alarms each and 0.04 F1. Cutting wcci2022's by 16%
costs 4.2 each and 47,000 extra alarms in absolute terms.

**2. ⚠ On case14 the F1-selected threshold is simply mis-transferred.** F1 there *improves*
monotonically as the objective weights recall — 0.4167 → 0.4330 → 0.4428 — approaching the oracle
0.4477. The cutoff that is F1-optimal on the training grid is too high for case14 **by F1's own
standard**, so a large part of case14's 0.0310 held-vs-oracle gap (§14) is threshold transfer, not
a cost-preference question at all. This is a second, independent symptom of the same
distribution shift the §14.1 hypothesis is about.

**3. The gate's headline property survives the change.** Re-running §13 at the F2 threshold:

| grid | objective | model | +shield | Δ | blocked | **intervention precision** |
|---|---|---:|---:|---:|---:|---:|
| neurips2020 | F1 | 0.8956 | 0.9038 | +0.0082 | 353 | **0.938** |
| neurips2020 | F2 | 0.8558 | 0.8610 | +0.0052 | 238 | **0.937** |
| case14 | F1 | 0.4167 | 0.4188 | +0.0021 | 103 | **0.922** |
| case14 | F2 | 0.4330 | 0.4341 | +0.0011 | 60 | **0.933** |
| wcci2022 | F1 | 0.5577 | 0.6253 | +0.0676 | 22,559 | **0.934** |
| wcci2022 | F2 | 0.5361 | 0.5869 | +0.0508 | 18,794 | **0.932** |

The delta shrinks, for a mechanical reason worth stating: a lower cutoff means fewer `secure`
predictions, so the gate has less to veto — blocks fall 353 → 238 and 22,559 → 18,794. **But
intervention precision stays flat at 0.932–0.937 across every grid and both objectives.** §2.1's
claim is that the gate's accuracy is a property of the physics rather than of the setup; it now
survives a change of topology *and* a change of threshold objective.

**Decision: keep F1 as the reported protocol, and report this table alongside it.** F1 is the
standard metric, it keeps the work comparable, and both shield arms are scored identically so the
delta is fair either way. Moving to F2 would mean optimising something other than the number being
reported, and choosing beta at all requires an operator cost ratio nobody has measured for these
grids. The honest position is to disclose the trade rather than to pick a point on it silently.

---

### 14.1 Hypothesis for the asymmetry — NOT TESTED

| grid | base kV levels | overlaps training grid? |
|---|---|---|
| neurips2020 (training) | 138, 345 | — |
| wcci2022 | 138, **161**, 345 | almost entirely |
| case14 | **14, 20**, 138 | mostly **not** |

The `mean_v` node feature is `v / 150.0`, so a 20 kV case14 line enters at ≈0.13 where training saw
≈1.0 — a range the model has never observed, and foreign features are normalized with 36-bus stats.
wcci sits inside the training distribution on this axis; case14 does not.

⚠ **This is a hypothesis, not a finding.** It is cheap to test — compare per-feature normalized
distributions against the 36-bus training range, or re-score case14 with `mean_v` ablated — and it
should be tested before any of it is written up as explanation.

---

## 15. Document alignment — why the corpus is thin

The per-document yield shows the problem is not uniform across the 16 documents:

| document | candidates | translated | guard-kept |
|---|---:|---:|---:|
| NERC Reliability Standards (complete set) | 707 | 19 | 13 |
| GB Grid Code | 571 | 14 | 3 |
| Bangladesh Grid Code | 228 | 15 | 4 |
| EU NC RfG | 218 | 11 | 7 |
| IBR performance guideline | 237 | 5 | 1 |
| **TPL-001-5.1** | **11** | 1 | 1 |
| **NERC FAC-008-5** | **4** | 0 | 0 |

**The two documents actually about post-contingency performance contributed 15 candidates between
them.** TPL-001-5.1 yielded only 5 distinct chunks, so the PDF on disk is a summary, not the
standard carrying Table 1's P0–P7 contingency categories.

What TPL-001 did produce is the diagnostic — free variables, not numbers:

```
Line | loading_pct > relay_loadability_limit_pct
Grid | power_oscillation_damping < acceptable_damping_threshold
```

**That is structural, not an extraction failure.** NERC standards are criteria-referencing by
design: they say "within its applicable Facility Rating", and FAC-008 then requires each utility to
hold a *documented methodology* for computing that rating. **The number is deliberately never in
the standard.** No prompt work extracts a threshold that was never written down.

### 15.1 Where the numbers actually live

| source | what it supplies | status |
|---|---|---|
| **PJM Manual 14B** | post-contingency voltage as literal numbers (**0.92–1.05 pu** across TPL-001 P1–P7), thermal as "within the applicable emergency rating", plus a **voltage-drop** criterion | verified by search; strongest candidate |
| **NERC FAC-011-4** | the *predicate structure*: an SOL is thermal facility ratings + voltage limits + stability limits | verified; non-numeric, but it is the schema |
| **EU SO GL (Reg. 2017/1485)** | the *operation* sibling of the RfG **connection** code; Art. 25 operational security limits, formal N-1 criterion | ⚠ Article 25 text **not** verified — open it before citing |
| ISO-NE PP3, NYISO Reliability Criteria, WECC TPL-001-WECC-CRT | regional criteria with explicit numeric post-contingency limits | not investigated |

The corpus is biased toward **connection** codes (RfG, IEEE 1547, Order 842, PRC-024/025/029) and
away from **operational security** codes. That is the one-sentence explanation, and it is more
defensible than "the simulator is too coarse" because it does not require the simulator to be at
fault.

### 15.2 The ceiling on re-extraction

A perfect document set supplies two predicates: post-contingency thermal (the corpus **already**
yields it as `loading_pct > 100`) and post-contingency voltage (measured as inert/uninformative
here, since these grids operate at 1.05–1.08 pu and never enter the ±5% band). **Most of the gain
would be more restatements of the one rule already in hand**, and §13.3 bounds what that rule can
reach.

Exactly **one** genuinely new capability appears above: PJM's **voltage-drop** criterion. It is a
delta rather than a level, so band-inertness does not apply, and it is physically computable. But
`label_n1()` stores only `n1_post_rho` and discards `sim_obs.v_or`, so it needs a generator change,
a dataset regeneration (~55 min across all three topologies) and a vocabulary extension before any
such rule could be evaluated.

**Recommendation: do not re-run extraction on new documents for the thesis.** §2 is a complete,
positive, honest result on the corpus in hand, and §13.3 bounds the headroom. Treat the
voltage-drop channel as a scoped experiment if it is wanted, not a re-extraction.

---

## 16. The knowledge graph

### 16.1 v1 — deleted, not archived, and why

`extraction/build_kg.py` (1,203 lines) and its eight artifacts are gone. Recoverable from
`git show a5c5199:extraction/build_kg.py`; **do not resurrect them.**

It built the graph around *grid topology* — Bus/Line/Generator nodes read from one grid's metadata,
with rules hung off them by `entity`. Two independent failures:

1. **It carried no information.** `Line` (59) and `Bus` (28) were 3.5% of the corpus, so **6,375 of
   its 6,632 edges (96%) were `has_rule` noise** onto a single catch-all node.
2. **It was welded to one topology.** The shield is evaluated on three. A graph keyed to 36-bus
   indices cannot serve case14 or wcci2022 — and the rules are topology-agnostic anyway.

Nothing in the project imported it. Its output `knowledge_graph.pkl` had already vanished from disk
months earlier and nothing noticed — the sharpest available evidence that it was load-bearing for
nothing.

### 16.2 v2 schema — provenance, not topology

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
| `Variable` | 3 | the vocabulary entries actually read |

**36 nodes, 40 edges** across five edge types. No edge type exceeds 60% of the total — guarded by
`test_no_node_type_dominates_the_edge_count`, which exists solely to stop a regression to v1's 96%.

**The `Predicate` layer is the graph's own contribution.** `deduplicate_rules` keys on
`(entity, condition)`, which fragments **one physical check into three records** purely on LLM
noise: the extractor labelled it `Line` in the Bangladesh grid code and `Facility` in the NERC set,
and wrote `100` in one clause and `100.0` in another. `normalize_predicate()` parses with `ast` and
canonicalizes numeric literals, so those collapse. **This is a view, not a substitution** — it never
changes what `rules_for()` returns, because collapsing three served rules into one would move
`highest_severity` (critical/high/medium differ across them) and the blocks-by-severity histogram
with it.

### 16.3 The corroboration finding

| predicate | role | served rules | clauses | documents | identified bodies |
|---|---|---:|---:|---:|---:|
| `loading_pct > 100` | CONSTRAINT | 3 | **10** | **4** | **2** |
| `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1` | AFFIRMATION | 1 | 1 | 1 | 1 |

The thermal check is stated **ten times across four documents**, by NERC (7 clauses, in the
Reliability Standards set and TPL-001-5.1) and the Bangladesh grid code (2 clauses), plus one
"EMO Dispatch Computer Constraints" clause in `power-system-requirements`.

Honestly stated: **the symbolic layer is one thermal predicate and one voltage affirmation — but
the thermal predicate is independently restated by two standards bodies on two continents.** A flat
JSONL file cannot express that; the deduped file actively hides it by concatenating sources into
one string.

⚠ Two things not to overclaim:
- `power-system-requirements` does not identify its issuing body. `kg/schema.py::ISSUING_BODY` is
  **hand-labelled** and records `None` rather than guessing, because the body count is a reported
  figure. **Two** identified bodies, not three.
- Corroboration is not independent *evidence about physics*. Four documents restating a thermal
  rating limit is four documents agreeing on standard practice, which §13.4 already characterised as
  close to a physical tautology. It strengthens the **provenance** claim, not the novelty claim.

### 16.4 Retrieval is opt-in, and the guarantee is a test

`KgRuleProvider` satisfies `RuleProvider` structurally, so it drops into `validate_n1()` and the
harness unchanged. `--rules-kg` selects it; `JsonlRuleProvider` stays the default.
`tests/test_kg.py::test_kg_provider_serves_exactly_what_the_jsonl_provider_serves` compares **full
rule dicts as sets**, not ids — a changed threshold, severity or role would move measured numbers
and must fail loudly.

**Verified end to end on all three topologies.** Every field of `shield_<tag>_kg.json` equals
`shield_<tag>_validated.json`: F1, precision, recall, tp, false alarms, missed violations, blocked,
corrections, regressions, eligible, severities, and all four shield-health counters.

⚠ **`threshold` and `ap` do NOT compare bit-equal, and that is not the graph.** Two runs of the
*identical JSONL path* reproduce the same drift — threshold ~1e-6, AP ~1e-8 — so it is forward-pass
nondeterminism on this hardware, not a retrieval difference. **No count moved in any run.** Worth
recording independently: the eval harness is **not bit-reproducible on this machine**, and §13.6's
"identical to every reported digit" holds at the 4 decimal places actually reported, not at full
float precision. A logit sitting exactly on the threshold could in principle flip a count; none has.

### 16.5 Never route by entity

`R_1443` is labelled `Facility`; `R_916` states the same check and is labelled `Line`. Retrieving
rules by walking entity edges — the classic KG pattern, and what v1 did — **silently drops one of
them** from any line-level query. Both providers therefore serve **every rule for every
prediction**; discrimination comes from the conditions. `test_graph_is_topology_agnostic` bans
bus/line indices and grid tags from node ids and structural attributes.

One deliberate exemption in that test: the verbatim `rule` payload carries the polarity guard's
`fire_rate_<tag>` fields, which do name all three grids. Those are *measurements about a rule*, not
structure, and cannot be stripped without breaking the set-equality guarantee. Indexed entities
remain banned inside the payload.

### 16.6 Citation

`KgRuleProvider.cite(rule_id)` accepts either a served rule id (what `ShieldResult.violated_rules`
carries) or any member rule id, and returns a `Citation` with every clause, document and issuing
body. `kg/cite.py::explain(result, provider)` renders a `ShieldResult` with the full chain.

`kg/cite.py` lives in `kg/`, **not** in `shield/`, on purpose — nothing under `shield/` imports
`kg`, so the gate keeps working with no graph present. That property is what allowed Component D to
be built and measured before Component C existed, and it is worth keeping.

`--citations <path>` writes one provenance record **per distinct rule that fired**, not per
contingency; the failure log already runs to 50,000 records.

---

## 17. Open items

All four components are built and measured. What remains is **optional robustness, not
construction**:

1. **`translate`-arm validation at 2–3 seeds.** Both A/B arms ran a single seed at temperature 0.0.
   Verdict stability across runs is established for `strict` (run 1 ≡ run 2) but not for
   `translated`. Cheap, and it closes the last open caveat on §12.
2. ✅ **DONE 2026-08-21 — §14's raw-model table is now held-threshold.** It reports the model at
   the threshold selected once on the neurips2020 validation split (0.8849) and applied unchanged
   to all three grids, with the best-threshold figures retained as a labelled oracle ceiling. The
   held figures were not re-measured: they are the `gnn` arm already recorded in
   `results/shield/shield_<tag>_validated.json`. **case14 gets worse under the honest protocol** —
   0.77× the rule baseline and 0.96× the all-positive baseline. ✅ The sub-item about the selection
   *objective* is also closed: the full F-beta trade curve is measured and reported in §14.2, F1 is
   retained as the protocol with the alternative disclosed alongside, and the gate's intervention
   precision was verified to stay flat (0.932–0.937) under a recall-weighted threshold.
3. **Pin the forward pass** (`GRID_DEVICE=cpu` plus deterministic flags) and re-run the three
   evaluations once, if bit-reproducibility is wanted. §16.4 says why this is cosmetic.
4. **Test the case14 asymmetry hypothesis** (§14.1) before writing the mechanism up as explanation.

### 17.1 Interpretability controls — designed, never run

These were specified to prevent one specific failure: reporting **(b)** *"we couldn't build much of
a symbolic layer, so we can't tell whether it would help"* as if it were **(a)** *"we built a sound
symbolic layer and it didn't help"*. Only (a) answers the thesis question.

| control | what it would establish |
|---|---|
| **Ceiling analysis** | Fit a small decision tree on the 14 context variables, target *"the model was wrong"*. This is the upper bound on what **any** symbolic gate could catch. Tree can't beat chance → answer (a), established without reference to rule count. Tree does well and the shield doesn't → the information was there and the rules missed it. |
| **Expert baseline** | Hand-write 10–15 rules against the same variables, as a **third arm, never merged**. Expert rules also give ~0 delta → gating doesn't help regardless of provenance, and extraction thinness is exonerated. Expert rules work, extracted don't → **extraction is the bottleneck**. Sharper than either alone. |
| **Rule-count ablation** | Subsets of 5/10/20/all → catch rate vs ruleset size. Flat → more rules wouldn't have helped. Still rising at N → genuinely truncated, but now *quantified*: a measured (b) is a legitimate limitation, an unmeasured one is the hole. |

**§13.3 answers the rule-count ablation structurally rather than by subsetting**, and §13.6 answers
it a second way — a corpus can lose 29% of its rules and change nothing. The ceiling analysis and
the expert baseline were **never run**, and that is a real gap in the write-up rather than a settled
question. State it as such.

---

## 18. Landmines

Read before editing anything in the pipeline. Each of these cost a round trip at least once.

- **Never re-close `EXTRACT_PROMPT` against `CONDITION_VOCABULARY`.** Tried twice, both times
  extraction returned **zero rules**: on frequency/timing-heavy standards the model correctly
  answers `[]` for nearly every chunk, starving stage 2. Guarded by
  `tests/test_condition_lint.py::test_extract_prompt_stays_open_vocabulary`.
- **Never name a guard output `*_confirmed.jsonl`** — `deduplicate_rules` globs that pattern and
  would merge rejects back into `all_rules_deduped.jsonl`.
- **Never merge a validation run whose rejections were not persisted** (§12.1).
- **Give each validation arm its own `--out` directory** — `deduplicate_rules` merges every
  `*_confirmed.jsonl` it finds, so a shared directory silently blends the two arms.
- **Do not delete `--task forecast` from `generate_dataset.py`.** The `classify` generator was
  removed and it is tempting to "finish the job". Don't: **no forecast dataset was ever written to
  disk**, so that code is the only way to reproduce §9.2. Classify was safe to remove precisely
  because its datasets *were* on disk.
- **`--smoke` used to never terminate.** It caps chronics and steps, but the loop cycles chronics
  toward `--n_records`, so it replayed 3 scenarios toward 300,000 records — 41 hours on the 118-bus
  grid. Fixed by `SMOKE_MAX_RECORDS = 200`. If that constant is removed, the hang comes back.
- **Thinking must be suppressed via `generate_no_think()`** (API-level `think=False`), not the
  `/no_think` prefix — newer Qwen builds ignore the prefix and the reasoning trace breaks JSON
  parsing.
- **`DEVICE` is a `torch.device`** — compare `.type`, never the object against `"cuda"`.
- **Never pass `dump_base_kv.py --empirical` again** (§11.2). It writes to `*_basekv_empirical.json`
  now, but the backend file is the contract.
- **Every command runs under the repo venv** — `.venv\Scripts\python.exe`. A bare `python` on
  Windows resolves to the Microsoft Store stub. Set `PYTHONIOENCODING=utf-8` when redirecting or
  piping: these scripts print `→` and the cp1252 pipe encoding kills them mid-run.
