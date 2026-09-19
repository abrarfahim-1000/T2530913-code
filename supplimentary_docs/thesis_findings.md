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
| NeurIPS 2020 | 18.4% | **96.7%** |
| case14 | 27.8% | **91.2%** |
| WCCI 2022 | 24.8% | **94.7%** |

⚠ **Corrected 2026-09-20** — this row previously read 19.5% / 95.7%, computed over the full
12,000-frame NeurIPS 2020 dataset rather than the 113,205-contingency held-out test split every
other NeurIPS 2020 figure in this document uses. Recomputed on that same test split, the baseline
reconciles with the "18.4%" violation rate quoted elsewhere for this grid, and the lift is if
anything larger. case14 and WCCI 2022 needed no correction — their entire generated dataset is
already the evaluation scope.

If this were a circular result — the rule secretly restating the labelling function — the right
column would read 100%. At **91–97%** the rule carries real but partial information.

The honest framing: **the gate is confirming established N-1 doctrine — you cannot declare a grid
secure against further loss when it is already outside its limits — and the data bears that
doctrine out 91–97% of the time.** It is not discovering new physics. It is enforcing a known
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
exact computation the neural network exists to replace.

⚠️ **UPDATED 2026-09-20 — the figures above are the SERVED RULE's reach, not the rule language's.**
This paragraph previously ended *"No larger rulebook closes that gap; it is a property of the
problem, not of the corpus."* The second clause stands; the first is too strong as written.
**That sentence is withdrawn and must not be quoted in that form.** §30's enumeration measures the
language's reach directly: the best rule
it contains reaches **23.0%** of NeurIPS 2020's dangerous errors and **38.5%** of WCCI 2022's,
against the served rule's 16.2% and 30.9%. The unreachable share is therefore about **77% and 62%**,
not 84% and 69%. What no rulebook lifts is the **order of magnitude** — most dangerous errors sit on
base states no present-state rule can see, and that is a property of the problem — but the exact
share is a property of where the gate is placed on its precision–recall curve, and a larger rulebook
buys a little of it back at a large cost in precision (§30.3).

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
| time / duration | **752** | **30.5%** |
| frequency / rate-of-change-of-frequency / droop | **617** | **25.1%** |
| protection relay settings | 528 | 22.2% |
| which specific piece of equipment is meant | 364 | 15.3% |

⚠️ **The time/frequency rows were corrected 2026-09-19.** They previously read 782/32.8% and
683/28.7% — an uncomputed hand tally that never reconciled with §19.1's own partition table two
sections over. `evaluation/capability_gap.py` scans every untranslatable candidate's stated reason
directly (the same regexes §19.1 uses, applied inclusively instead of first-match-wins) and gets
752 / 617, union 1,252 — which reconciles exactly with §19.1's exclusive buckets (635 + 617). The
protection-relay and equipment rows were not part of that recheck and are unverified by any script;
treat them as reported, not reproduced.

**1,252 rules — 50.8% of the entire corpus — are blocked on just two missing capabilities:
frequency and time** (`results/audit/capability_gap.json`). Those would become *assessable* if the
simulator carried them. Assessable is not useful: they would still have to survive the quality
checks, and this corpus lost 60% of its rules at that step.

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

The replacement was built holding **36 nodes and 40 connections**, is grid-independent, and every
connection means something. It was later rebuilt to also carry the shield's 58-rule explanation
corpus and now holds 171 nodes and 236 edges (§16.2a) — additively, so this changes no result: the
gate reads exactly the same rules whether it is handed the flat file or the graph, which is checked
by a test and was verified by re-running all three evaluations and comparing every reported number.

**Remaining:** two optional robustness items — repeating the validation run at several random
seeds, and re-reporting the raw model under the same held-threshold protocol this document uses.
Neither changes the shape of the result. Sequence: §17.

**Added 2026-09-17 (§19):** an exhaustive accounting that assigns all 2,463 candidates to one
terminal bucket each, and a deterministic re-partition of the 58 expressible rules into four
output channels. **46 rules (20 distinct predicates) now speak, against 5 today**, and 54 of 58
are documented rather than dropped. No measured result moved: BLOCK is deliberately unchanged, so
§13's 92–94% intervention precision stands as reported. The accounting also found that
`rule_id` is **not unique** across the corpus (§19.2) — the served four-rule corpus is unaffected.

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
| the rule is not circular (91–97%, not 100%) | §13.4 | — |
| structural ceiling on what rules can reach | §13.3 | — |
| extraction yield and the audit | §10.1, §10.3 | `evaluation/audit_rules.py` · `results/audit/audit_run3.json` |
| standards/simulator mismatch, both directions | §10.4, §15 | — |
| the 4-class task was closed-form | §9.1 | — |
| forecast tasks rejected | §9.2 | — |
| raw model cross-topology performance | §14 | `gnn_n1_tightening.md` |
| voltage base-kV contract and why it matters | §11 | `data/grid_dataset_<tag>_basekv.json` |
| the four surviving rules themselves | §12.5 | `validated_translated/all_rules_deduped.jsonl` |
| what was designed, and the three controls that have since run | §17.1 ➡ §28, §29, §30 | `results/ceiling/` |
| every candidate accounted for, one terminal bucket each | §19.1 | `evaluation/corpus_accounting.py` · `results/audit/corpus_accounting.json` |
| stage-1 `rule_id` collisions, and the served corpus is clean | §19.2 | same artifact |
| the four-channel re-admission (41 rules speak, 17 distinct after §21) | §19.3 | `evaluation/readmit_rules.py` · `results/audit/readmission.json` |
| the loading-band cliff — why 100 is not arbitrary | §20.1 | `evaluation/loading_band_calibration.py` · `results/audit/loading_band_calibration.json` |
| ⚠ NOT RUN — is the thin corpus the task's fault or the pipeline's? | §24 | *(spec only, no artifact — this is the disclosed gap)* |
| model-conditional WARN calibration; no predicate changed sign | §23.1 | `evaluation/warn_rule_calibration_conditional.py` · `results/audit/warn_n1_calibration_conditional.json` |
| why no dynamic simulator (standalone, plain-language) | §20.2 | `supplimentary_docs/andes_investigation.md` |
| shield v2: channels implemented, 58 rules reproduce the 4-rule numbers | §22.3 | `shield/channels.py` · `shield_corpus/all_rules_channels.jsonl` · `results/shield/shield_<tag>_v2channels.json` · `tests/test_shield_channels.py` |
| every WARN rule carries a measured N-1 rate; 2 predicates INVERTED | §21.2 | `evaluation/warn_rule_calibration.py` · `results/audit/warn_n1_calibration.json` |
| ANDES investigated and rejected on evidence | §20.2 | `sanity/andes_{frequency,voltage}_spike.py` · `results/audit/andes_*_spike.json` |
| the live forward plan | §20.3 | — |

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

⚠️ **The table above is frozen run-2 history (2,381 untranslatable, the pre-fix parsing-bug run —
§10.2) and is kept as the record of what that run reported.** It is not the corpus to cite for a
current TIME/FREQ count. On the live, clean run-3 data (2,405 untranslatable,
`translated_rules/*_untranslatable.jsonl`, the same files §19.1 and `corpus_accounting.py` use),
`evaluation/capability_gap.py` gets **TIME 752, FREQ 617**, not 782/683 — the run-2 figure was
never reproduced on run-3 data and should not be quoted as current. PROT/ENTITY/other were not
rechecked and remain unverified by any script either way.

**1,332 candidates (54.1% of the corpus)** was the run-2 reading of "blocked on exactly two missing
capabilities, frequency and time". ⚠️ **Superseded 2026-09-19.** The current, script-verified figure
on run-3 data is **1,252 candidates (50.8%)** (`results/audit/capability_gap.json`), which
reconciles exactly with §19.1's own exclusive partition (635 + 617 = 1,252). Both readings agree on
the qualitative point: they would become *assessable* if the observation carried them, and
"assessable" is not "useful" — they would still have to survive the guard and the audit, and this
corpus rejected 59.8% at the guard alone.

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
| neurips2020 | 18.37% | **96.66%** | 5.26× |
| case14 | 27.75% | **91.16%** | 3.28× |
| wcci2022 | 24.76% | **94.74%** | 3.83× |

⚠ **Corrected 2026-09-20** — the neurips2020 row previously read 19.54% / 95.67% / 4.90×, computed
over the full 12,000-frame dataset rather than the 113,205-contingency held-out test split that
every other neurips2020 figure in this document is scored on (`results/audit/loading_band_calibration.json`
scans all frames by default). Recomputed on that test split, the baseline reconciles with the
"18.4%" violation rate quoted elsewhere for this grid, and the lift is larger, not smaller. case14
and wcci2022 needed no correction — their whole generated dataset is already the evaluation scope,
so both populations coincide. The corrected `>= 1.00` band rests on only 34 frames rather than 250,
because few held-out frames land in an already-overloaded base state; §20.1's fuller per-band curve
is deliberately left on the larger, full-dataset sample for stability, and its neurips2020 `>= 1.00`
figure (0.957) differs from this one (0.9666) for that sampling reason alone.

If this were the label-restatement trap the conditional would be 100%. At 91–97% the predicate
carries real but partial information. The honest reading: **the shield is validating N-1 doctrine
— a base case outside its limits cannot be declared secure against further loss — and the data
confirms that doctrine 91–97% of the time.** It is not discovering new physics.

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

| node | n (as built here) | n (current file, §16.2a) | source |
|---|---:|---:|---|
| `Document` | 5 | 12 | filename stems of `validated_translated/*_confirmed.jsonl` |
| `Clause` | 11 | 55 | each rule's `source` string, deduplicated within a document |
| `Rule` | 11 | 11 | the validated rules |
| `ServedRule` | 4 | 4 | `all_rules_deduped.jsonl`, **carried verbatim** — what the shield receives |
| `Predicate` | 2 | 26 | normalized condition semantics |
| `Variable` | 3 | 5 | the vocabulary entries actually read |

**As built here: 36 nodes, 40 edges** across five edge types. No edge type exceeds 60% of the
total — guarded by `test_no_node_type_dominates_the_edge_count`, which exists solely to stop a
regression to v1's 96%.

#### 16.2a The live file has since grown to 171 nodes, 236 edges

`kg/knowledge_graph.json` was later rebuilt (`scripts/build_kg.py --channels`) to add an
`ExplanatoryRule` layer — 58 nodes, one per record in the shield's explanation-channel corpus
(§19.3), each carrying the same provenance chain as a `ServedRule` but for a rule that can now
*speak* (BLOCK/WARN/NORMAL) rather than only the four that can *veto*. Because those 58 rules draw
on more source documents and clauses than the 11 validated ones, `Document`, `Clause`, `Predicate`
and `Variable` all grew alongside it (second column above). The edge count grew to 236 for the
same reason; no edge type dominates it either (the largest, `states`, is 69 of 236 — 29%).

This is additive, not a substitution: `ServedRule` and the edges into it are untouched, so §16.3's
corroboration table and every shield result below are unaffected. Re-tracing §16.3 from the current,
larger file returns the identical figures — 3 served records, 10 clauses, 4 documents, 2 identified
bodies for the thermal predicate — which is the check that matters, not the total node count.

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

### 17.1 Interpretability controls — designed here, and RUN (see §28, §29, §30)

These were specified to prevent one specific failure: reporting **(b)** *"we couldn't build much of
a symbolic layer, so we can't tell whether it would help"* as if it were **(a)** *"we built a sound
symbolic layer and it didn't help"*. Only (a) answers the thesis question.

| control | what it would establish |
|---|---|
| **Ceiling analysis** | Fit a small decision tree on the 14 context variables, target *"the model was wrong"*. This is the upper bound on what **any** symbolic gate could catch. Tree can't beat chance → answer (a), established without reference to rule count. Tree does well and the shield doesn't → the information was there and the rules missed it. |
| **Expert baseline** | Hand-write 10–15 rules against the same variables, as a **third arm, never merged**. Expert rules also give ~0 delta → gating doesn't help regardless of provenance, and extraction thinness is exonerated. Expert rules work, extracted don't → **extraction is the bottleneck**. Sharper than either alone. |
| **Rule-count ablation** | Subsets of 5/10/20/all → catch rate vs ruleset size. Flat → more rules wouldn't have helped. Still rising at N → genuinely truncated, but now *quantified*: a measured (b) is a legitimate limitation, an unmeasured one is the hole. |

**§13.3 answers the rule-count ablation structurally rather than by subsetting**, and §13.6 answers
it a second way — a corpus can lose 29% of its rules and change nothing.

🚨 **UPDATED 2026-09-19 — the ceiling analysis and the expert baseline HAVE RUN, and a third, stronger
instrument has since replaced the first.** The paragraph that stood here said they were *never run*
and called it a real gap in the write-up; that is no longer true and must not be quoted. The ceiling
analysis ran as a greedy depth-4 tree (**§28**), the expert baseline as fourteen pre-registered
hand-written rules (**§29**, status open — see §30.8), and the tree was then superseded by an
exhaustive enumeration of the entire rule language (**§30**, the one to cite). All three support
reading **(a)**: the task is rule-poor. The rule-count ablation remains answered structurally, as
above, rather than by subsetting. ⚠️ §29 is deliberately left unrevised pending the §29.1
blindness decision, and §30.8 records why that question is no longer load-bearing.

➡️ **§24 was the runnable spec for both, and it has been executed** — written 2026-09-17 to be picked
up cold, it fixed the exact target and split discipline for the tree, the contamination trap on the
expert arm, and what each outcome would license the thesis to claim. Its clause on what must be said
if neither is run no longer applies; its clause on the expert arm's contamination trap does, and was
triggered (§29.1).

---

## 18. Landmines

Read before editing anything in the pipeline. Each of these cost a round trip at least once.

- **Never key on `rule_id` alone** (§19.2). It is not unique: five documents restarted numbering,
  so `R_001`..`R_172` are reused across five documents — 2,463 records, 2,291 distinct ids. Use
  `(document, rule_id)`. A bare-`rule_id` join reports 12 rules as both translated *and*
  untranslatable, which looks like a pipeline bug and is not one.
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

---

## 19. Corpus accounting and the four-channel shield

**Added 2026-09-17.** Two deterministic passes over artifacts already on disk. No stage was
re-run, no model was called, and **no measured result moved**.

This section exists to answer one objection directly: *a pipeline that turns 2,463 candidates
into 4 rules did not find a signal, it found noise.* The answer is not a better yield. It is that
the yield was never the finding — the **partition** is.

### 19.1 Every candidate is accounted for

`evaluation/corpus_accounting.py` assigns all 2,463 stage-1 candidates to exactly one terminal
bucket and **asserts the partition** at every stage rather than assuming it. Artifact:
`results/audit/corpus_accounting.json`.

| terminal fate | n | share |
|---|---:|---:|
| not expressible — time / dynamics | 635 | 25.8% |
| not expressible — frequency | 617 | 25.1% |
| not expressible — quantity not modelled by the simulator | 562 | 22.8% |
| not expressible — equipment-internal | 315 | 12.8% |
| **not expressible — scope / aggregation  [RECOVERABLE]** | **142** | **5.8%** |
| not expressible — administrative / process | 115 | 4.7% |
| expressible, rejected by the polarity guard | 26 | 1.1% |
| expressible and clean, rejected by the validator | 21 | 0.9% |
| not expressible — unclassified | 13 | 0.5% |
| **VALIDATED (served to the shield)** | **11** | **0.4%** |
| not expressible — free variable / no number | 6 | 0.2% |
| **TOTAL** | **2,463** | **100.0%** |

Buckets are first-match-wins in a declared order, so they are disjoint by construction. The
ordering is load-bearing and documented in the script: frequency precedes time (Grid2Op models no
frequency at all, whereas a duration could in principle be mapped onto steps), and
equipment-internal precedes scope (*"stator current, not grid-wide observable"* mentions scope, but
the real blocker is that the simulator has no stator).

**The distinction that carries the argument is the emphasized row.** `scope / aggregation` marks
rules whose quantity **is in every record** and which the 14-variable vocabulary flattens to a
grid-wide min/max — per-generator power factor from `gen_p`/`gen_q`, per-line loading from `rho`,
per-voltage-level bands from the existing `*_basekv.json` sidecars. Those 142 are blocked by
**vocabulary design, not by physics**, and are the only bucket recoverable without a new simulator.
Everything above them is a property of what a quasi-static power-flow simulator represents.

### 19.2 A stage-1 invariant is broken — and the headline corpus is clean

The accounting asserts identity before it counts, and the assertion failed. `rules_35b/` holds
**2,463 records but only 2,291 distinct `rule_id`s**: five documents restarted numbering, so
**R_001..R_172 are reused across five documents**. The documented invariant — *"rule IDs are
globally sequential across all chunks and documents"* — does not hold.

Consequences, traced rather than assumed:

| stage | rules carrying a colliding id |
|---|---:|
| translated (58) | 12 |
| guarded (32) | 6 |
| confirmed (11) | 1 — `R_167` |
| **served (4)** | **0** |

**No served rule carries a colliding id, and `R_167`'s provenance is correct.** Each record keeps
its own `source` string and the per-document files never merge, so the confirmed `R_167` is the
TPL-001-5.1 §5.1.f rule (`loading_pct > 100`), not the ENTSO-E `power_factor_step` rule that shares
its number. The KG chain `Clause:C_010 → Rule:R_167 → ServedRule:R_1443` was checked directly and
is right.

⚠ **Nothing downstream may key on `rule_id` alone.** Use `(document, rule_id)`. This is why §19.1's
partition verifies only under a composite key; under a bare `rule_id` it reports 12 rules as
simultaneously translated and untranslatable, which is an artifact of the collision and not a
pipeline defect.

### 19.3 The shield had one channel, which is why 54 of 58 rules were on the floor

`evaluation/readmit_rules.py` re-partitions all 58 expressible rules by **measured behaviour**,
deterministically. Artifact: `results/audit/readmission.json`.

⚠️ **The WARN row below is PRE-CALIBRATION and is superseded by §21.** It was assigned on
classify-label fire rates — the retired 4-class target. Calibrated against the N-1 label, 5 of the
25 WARN records do not survive (2 predicates INVERTED, 1 INSUFFICIENT), so the figure to quote is
**41 speaking / 17 distinct**, not 46 / 20. The other four channels are unchanged.

A rule was discarded whenever it could not justify a veto — because vetoing was the only thing a
rule was permitted to do. That is a property of the gate's design, not of the rules.

| channel | rules | distinct | today | what it does |
|---|---:|---:|---:|---|
| **BLOCK** | 10 | 2 | 4 | vetoes an over-permissive prediction |
| **WARN** | 25 | 11 | 0 | annotates, never vetoes; carries its measured rate |
| **NORMAL** | 11 | 9 | 1 | affirms telemetry is consistent with normal operation |
| **NOT_APPLICABLE** | 8 | 8 | 0 | correct rule whose calibration does not transfer (§11.2) |
| **INERT** | 4 | 4 | 0 | cannot fire on any class on any grid; excluded and counted |
| **rules that SPEAK** | **46** | **20** | **5** | BLOCK + WARN + NORMAL |
| **documented** | **54** | **25** | **5** | + NOT_APPLICABLE, cited rather than dropped |

**Quote the `distinct` column.** Ten BLOCK records are two distinct conditions
(`loading_pct > 100` and `loading_pct > 100.0`) restated across standards, which the KG's
`Predicate` layer collapses to one. Rule records are not predicates, and a count of records
inflates.

Three properties of the assignment:

**BLOCK is deliberately unchanged.** It is the set the 92–94% intervention-precision result is
measured on. Admitting anything to it moves that number, so nothing was admitted. §13's results
stand exactly as reported.

**Channel membership is decided per grid, not per corpus.** A rule that discriminates on two
topologies and misfires on the third is a WARN scoped to those two, not a rejection. `R_128` fires
on 0.1% of healthy neurips frames and 98.6% of healthy wcci2022 frames — one statement about the
rule, three different statements about the grids. Collapsing to a single verdict loses it; an
earlier version of this pass did exactly that and misfiled 14 rules into NOT_APPLICABLE.

**NOT_APPLICABLE is a finding, not a bin.** Its eight members are almost all ±5% voltage bands,
degenerate on every grid because these grids operate ~6% above nominal (§11.2). They are correct
readings of their standards. Recording them with that reason converts eight silent drops into
eight cited exclusions with a measured cause.

### 19.4 What this does and does not change

**Does not change:** F1, blocking precision, the shield's measured deltas, or any number in §13
or §14. The served corpus is the same four rules.

**Does change what the system says.** The shield currently speaks on 0.31% / 0.087% / 3.04% of
contingencies — it is silent on 97–99.9% of predictions. Four channels let it annotate every one,
with three verdict types, each traceable to a clause and each carrying a measured firing rate.
For a neuro-symbolic system that is the contribution; blocking is one of its modes, not the whole.

**Two guardrails, both load-bearing.** Channels must be structurally incapable of crossing — a
WARN rule that can reach the veto path moves the headline number. And **every WARN must carry its
measured rate**: a warning without a rate is an alarm, a warning with one is evidence. §13.4's
conditional table, extended per band, is the calibration source.

### 19.5 Reproduce

```powershell
.venv\Scripts\python.exe evaluation\corpus_accounting.py   # exits non-zero if the partition breaks
.venv\Scripts\python.exe evaluation\readmit_rules.py
```

---

## 20. The loading-band cliff, the ANDES investigation, and what is next

**Added 2026-09-17.** §20.1 is a new measurement. §20.2 records a route that was investigated
and **rejected on evidence**, so it is not re-opened. §20.3 is the live plan.

### 20.1 The threshold 100 sits on a physical cliff

`evaluation/loading_band_calibration.py`, every frame of all three N-1 datasets. Artifact:
`results/audit/loading_band_calibration.json`. This is close to, but not identical to, §13.4's
single number (P(violation | base overloaded) = 91–97%, held-out test split for neurips2020)
extended to the full curve — this curve uses every frame (including train/val) for neurips2020,
so its `>= 1.00` figure (0.957) is not the same measurement as §13.4's 0.9666.

| base-case `rho_max` | neurips2020 | case14 | wcci2022 |
|---|---:|---:|---:|
| 0.60 – 0.70 | 0.127 | 0.176 | 0.129 |
| 0.70 – 0.80 | 0.180 | 0.254 | 0.161 |
| 0.80 – 0.90 | 0.228 | 0.396 | 0.205 |
| 0.90 – 0.95 | 0.284 | 0.531 | 0.253 |
| 0.95 – 1.00 | 0.432 | 0.616 | 0.382 |
| **≥ 1.00 — the served rule** | **0.957** | **0.912** | **0.947** |

**It is a cliff, not a slope.** In the 0.95–1.00 band, immediately below the threshold, the
predicate is near a coin flip. Crossing 1.00 it jumps to 92–97% — which is where the shield's
measured 0.938 / 0.922 / 0.934 intervention precision comes from.

Two consequences, and both belong in the write-up:

**Lower-threshold thermal rules cannot be admitted to BLOCK.** The corpus carries them at 84, 90,
95, 110, 116 and 125%. The 110/116/125 rules are logically subsumed by `> 100` and add no block.
The 84/90/95 rules would add coverage at 38–62% precision, trading the headline result for volume.
They belong in WARN, carrying the row above as their rate.

**The corpus converged on the one threshold that is physically load-bearing.** Four standards
bodies wrote 100; the pipeline independently kept only 100; and the grid data shows 100 is where
the predicate starts working. This reframes §5: the yield is not thin because extraction was weak,
it is thin because **the task has approximately one governing predicate and the pipeline found it.**

### 20.2 ANDES / dynamic simulation — INVESTIGATED AND REJECTED, do not re-open

A dynamic simulator (ANDES 2.0, installed, `sanity/andes_*_spike.py`) was proposed to recover the
1,300+ candidates blocked on frequency and sub-second time. It was measured, not argued about.

**Frequency arm** (`results/audit/andes_frequency_spike.json`) — IEEE-14 with TGOV1 governors,
permanent trips at t=1.0 s, 20 s window:

| disturbance | frequency excursion | fires a corpus threshold? |
|---|---:|---|
| 8 most-loaded **line** trips (the task) | 0.014 – **0.329 Hz** | **no** |
| **generator** trip, N-1 (positive control) | 0.315 Hz | **no** |
| shipped islanding case | 0.125 Hz | **no** |
| 2 generators out (N-2) | 0.798 Hz | yes — 59.4 |
| 3 loads out (N-3) | 1.623 Hz | yes — 60.6, 61.0 |

**No N-1 contingency of any kind reaches the mildest of the 457 numeric frequency thresholds.**
The corpus tests 47–52 Hz and 57–63 Hz — system-wide imbalance bars, ~1% off nominal at their
mildest. N-1 produces at most 0.55%. Firing them requires N-2 or worse. The positive control
works (a generator trip produces a sustained offset), so this is physics, not instrumentation.

**Voltage arm** (`results/audit/andes_voltage_spike.json`) — 103 voltage-and-time rules, 94 with
numeric thresholds on both sides. Clean line opening enters **no** envelope (deepest dip 0.9557 pu
rebased against a mildest bar of 0.95). Under a bolted three-phase fault the deep envelopes *are*
entered — but voltage recovers above 0.9 pu within ~50 ms of clearing, and the surviving
post-clearing crossings clear only 0.0016–0.033 s durations, missing the corpus's own
0.14/0.15/0.16 s band entirely. **The ride-through envelopes are reachable by the EVENT, never by
a STATE.** A shield that reads telemetry sees the post-event state, not the fault window.

The one genuine positive is a post-fault **over-voltage overshoot** surviving clearing (1.05–1.15 pu
for up to 0.65 s rebased; R_1081, R_1831, R_1832). Three rules — and observing them requires ANDES
at inference, because Grid2Op is a 5-minute steady-state snapshot in which a 0.65 s overshoot does
not exist.

**Verdict: 1,252 evaluable → 3 usable, at the cost of a 2–4 week dynamic pipeline, describing a
fault event while the task is a clean outage.** (⚠️ Corrected 2026-09-19 from an unscripted 726 —
`evaluation/capability_gap.py` gives 1,252, reconciling exactly with §19.1's partition. The 726
never reconciled with anything and is retracted, not merely re-caveated.) The finding that replaces
it is sharper than the rules would have been: *grid-code frequency limits are calibrated for
N-2-and-beyond emergencies and grid-code ride-through limits for fault transients; N-1 thermal
screening reaches neither, so no simulator upgrade makes them applicable — the gap is subject
matter, not instrumentation.*

### 20.3 What is next

Ordered. Nothing here changes a measured result.

1. ~~**N-1 calibration per WARN rule**~~ — **DONE 2026-09-17, see §21.** 8 of 11 predicates
   calibrated (+0.23 to +0.73 discrimination); **2 came back INVERTED** — the entire power-factor
   family fires more often when N-1 risk is *lower* — and 1 INSUFFICIENT. Those 5 records left the
   WARN channel, so the banked figure is **17 distinct conditions speaking, not 20**. What remains
   is the model-conditional version (§21.4), which needs a forward pass.
2. ~~**Implement the channels in `shield/`**~~ — **DONE 2026-09-17, see §22.** Original note:
   `ShieldResult` already carries `violated_rules`,
   `supporting_rules`, `contradicting_rules` and `unsupported`; what is missing is a
   `warning_rules` channel and a `build_explanation` that renders the PASS path (it currently
   returns "No applicable rule was violated."). Pin with a test that a non-BLOCK rule can never
   reach the veto path — if it can, §13's precision moves.
3. **Optional: stage-2/3 re-run** (user-sanctioned, not yet started). Target is the 142
   `scope / aggregation` rules of §19.1 plus the unexplained gap between the documented
   693-expressible estimate and the 58 stage 2 produced. Needs a vocabulary extension for
   per-generator (`gen_p`/`gen_q`), per-line (`rho`) and per-kV-level (`*_basekv.json`) accessors,
   and a channel assigned at translation time. Projected **+15–25 distinct conditions**, low
   confidence. Stage 1 stays frozen.
4. **Still open from §17.1 — now the LARGEST remaining hole, spec'd in §24.** The expert-written
   rule baseline and the ceiling analysis. Neither was run. Together they are the only external
   check on whether the thin corpus is the task's doing or the pipeline's — everything else in this
   document argues the former from *inside* the pipeline that produced it.

**Where the count stands.** Served today: 4 records, **3 distinct conditions**, 2 KG predicates.
After §19.3's re-admission *as calibrated in §21*: **41 rules speaking, 17 distinct conditions**,
54 documented. ⚠ §19.3's 46/20 is the pre-calibration figure and is superseded — quote 41/17. The
re-run would roughly double the distinct count; it does not change the order of magnitude, and the
**5.7×** is already banked and measured.

---

## 21. Calibrating the WARN channel against the N-1 label

§19.3 admitted 25 rule records to a WARN channel on the strength of their **classify-label** fire
rates — the rule separates `normal` frames from `overload`/`line_trip`/`cascade` frames. That is a
statement about the *retired* 4-class target (§9.1), not about the task the shield gates. §20.3
listed calibrating them against the N-1 label as the highest-value remaining item, on the grounds
that an uncalibrated warning is an alarm wearing a citation. It has now been run.

`evaluation/warn_rule_calibration.py`, every frame of all three N-1 datasets (12,000 / 6,000 /
4,000), no model and no LLM. Artifact: `results/audit/warn_n1_calibration.json`.

### 21.1 The statistic, and two corrections made while measuring

For each of the **11 distinct WARN predicates** (the 25 records are restatements; calibrating each
record would report one measurement as though it were 25), the base-case context is built exactly
as the shield builds it, the predicate is evaluated, and the frame's per-line N-1 labels are
assigned to the firing or the silent arm.

**The reported statistic is `P(violation | fires) − P(violation | silent)`, not lift against the
base rate.** §20.1 uses lift-against-base, which is correct for a partition into seven `rho_max`
bands and wrong for a binary predicate. The first full run showed why: the power-factor rule fires
on **97.5%** of wcci2022 frames at P|fire 0.242 against P|silent 0.478, and lift-against-base still
reads −0.006 — because a predicate that fires on nearly every frame *is* the base rate. Against its
own silent arm it is **−0.236**. Both are persisted; the verdict is taken on the difference.

**Pooled values are stratified by grid, and weighted by frames.** Two traps, both caught by
measurement rather than by argument:

- *Simpson.* `voltage_pu_min <= 0.90 or voltage_pu_max >= 1.10` fires on 100% of case14 frames and
  on ~0% of the other two. Its within-case14 discrimination is undefined — it never stays silent —
  yet naive pooling scored it **+0.067**, purely because case14's base violation rate (0.278) is
  higher than neurips2020's (0.195). The predicate was being credited for identifying a topology.
- *Frames, not contingencies.* Weighting a grid's contribution by contingencies let three firing
  wcci2022 frames carry 558 votes against neurips2020's 71 frames. Lines in one frame share a grid
  state. The pool now weights by **frames**, and a grid that did not earn its own verdict gets no
  vote at all.

A predicate needs **≥30 firing frames and ≥30 silent frames on a grid** to be scored there. A rule
that fires on everything has no silent arm, and "it is always true" is a statement about the grid's
operating point, not about risk.

### 21.2 The result

| predicate | grids scoring | P(violation \| fires) | P \| silent | **P\|f − P\|q** | verdict |
|---|---:|---:|---:|---:|---|
| `voltage_pu_min < 0.917` | case14 | **0.843** | 0.226 | **+0.726** | ELEVATED |
| `voltage_pu_min < 0.95` | case14, wcci2022 | **0.592** | 0.225 | **+0.537** | ELEVATED |
| `voltage_pu_min < 0.9 or voltage_pu_max > 1.1` | neurips2020 | **0.562** | 0.225 | **+0.297** | ELEVATED |
| `voltage_pu_min < 0.90 or voltage_pu_max > 1.10` | neurips2020 | 0.562 | 0.225 | +0.297 | ELEVATED |
| `voltage_pu_min <= 0.90 or voltage_pu_max >= 1.10` | neurips2020 | 0.287 | 0.221 | +0.297 | ELEVATED |
| `voltage_pu_min < 0.85 or voltage_pu_max > 1.10` | neurips2020 | 0.522 | 0.226 | +0.297 | ELEVATED |
| `voltage_pu_max > 1.10` | neurips2020 | 0.491 | 0.226 | +0.297 | ELEVATED |
| `voltage_pu_min < 0.95 or voltage_pu_max > 1.05` | wcci2022 | 0.474 | 0.245 | +0.229 | ELEVATED |
| `power_factor_at_max_load < −0.95 or > 0.95` | all three | 0.247 | 0.208 | **−0.163** | **INVERTED** |
| `power_factor_at_max_load < −0.95 or > 0.90` | neurips2020, wcci2022 | 0.252 | 0.195 | **−0.070** | **INVERTED** |
| `voltage_pu_min < 0.90` | *none* | 0.855 | 0.226 | — | INSUFFICIENT |

**8 of 11 predicates (20 of 25 records) are calibrated.** Firing genuinely predicts a higher N-1
violation rate, by between +0.23 and +0.73 against the predicate's own silent arm — against a base
rate of 0.195–0.278. These may quote P|fire in their warning text.

**Two predicates are INVERTED, and they are the whole power-factor family** (4 records: `R_909`,
`R_128`, `R_1291`, `R_157`). On wcci2022, `power_factor_at_max_load > 0.95` fires on 97.5% of
frames and firing is associated with a violation rate **0.236 lower** than staying silent. This is
not a weak warning; it is a warning that points the wrong way, and no phrasing repairs it. They are
factually true statements about the telemetry and stay documented — they must not be shown as risk
warnings. Note the classify-label audit had them as discriminating: this is the clearest case in
the corpus of a rule that separates the *retired* target and not the live one.

**One is INSUFFICIENT** (`voltage_pu_min < 0.90`, `R_1835`): 26 firing frames pooled, under the
bar. Its two-sided sibling `voltage_pu_min < 0.9 or voltage_pu_max > 1.1` clears it at 97 frames —
on neurips2020 the overvoltage arm does all the work, which is §11.2 again (these grids operate
~1.06 pu, so it is the upper band that trips, not the lower).

### 21.3 What this does to the count

| channel | §19.3 records | **calibrated** | §19.3 distinct | **calibrated** |
|---|---:|---:|---:|---:|
| BLOCK | 10 | **10** | 2 | **2** |
| WARN | 25 | **20** | 11 | **8** |
| NORMAL | 11 | **11** | 9 | **9** |
| NOT_APPLICABLE | 8 | **13** | 8 | **11** |
| INERT | 4 | 4 | 4 | 4 |
| **speaking** | **46** | **41** | **20** | **17** |
| documented | 54 | 54 | 25 | 25 |

**3 → 17 distinct conditions speaking (5.7×), not 6.7×.** The calibration cost three distinct
conditions and five records. It bought the thing those 20 records did not have: every WARN rule now
quotes a number measured on the task the shield actually gates, and the five that could not earn
one were found *before* a reviewer found them. Nothing moved in BLOCK, so §13's 92–94%
intervention precision is untouched.

### 21.4 The limit that remains — CLOSED 2026-09-17, see §23

This is a **grid-conditional** calibration: P(violation | rule fires) over base-case frames. The
WARN channel's semantic is narrower — *"this prediction was let through, but according to this
rule, that could happen"* — which is P(violation | rule fires **and** the model predicted secure).
That needs a forward pass and is a strictly smaller conditioning set. The direction is unlikely to
flip, but the magnitudes will move, and the honest phrasing until it is run is the grid-conditional
one. Do not attribute the numbers above to the model's error set.

```
.venv\Scripts\python.exe evaluation\warn_rule_calibration.py
.venv\Scripts\python.exe evaluation\warn_rule_calibration.py --channel NORMAL
```

---

## 22. Shield v2 — the explanation channels, implemented

§19.3 diagnosed the problem and §21 calibrated the fix. This is the implementation:
`SHIELD_VERSION = "2.0"`, new module `shield/channels.py`, new corpus
`shield_corpus/all_rules_channels.jsonl`, new suite `tests/test_shield_channels.py`
(22 tests). **245 tests pass.**

### 22.1 What v1 could say, and what v2 says

v1 had one channel. A rule vetoed a prediction or it was discarded — which is why 54 of the 58
expressible rules never spoke. On a PASS it emitted a single string, `"No applicable rule was
violated."`, which is true and carries no information: it cannot distinguish a grid where nine
standards affirmatively hold from a grid no rule could evaluate.

v2 renders three registers, in the order an operator reads them:

| register | v2 output |
|---|---|
| **BLOCK** | `Prediction blocked by N rule violation(s): [HIGH] R_858 (Section 8.3.1.2): ... — condition: loading_pct > 100.0` |
| **WARN** | `Warning - let through, but 10 rule(s) report a condition that raises N-1 risk: [WARN] R_351 (...): ... — 47.4% of contingencies violated a limit when this fired, against 24.5% when it did not; measured on wcci2022` |
| **NORMAL** | `Environment normal according to N rule(s): [OK] R_1154 (NERC ...): ... — condition holds: ...` |

The rate in the WARN line is not decoration — it is §21.2's measurement, and a warning that cannot
produce one renders `no measured rate on this task` rather than implying it has one.

### 22.2 The veto path is barred structurally, not by convention

`partition_by_channel()` buckets rules before anything is evaluated, and the veto loop iterates
`buckets[BLOCK]` and nothing else. A WARN rule cannot reach `violated_rules` by being mislabelled,
mis-sorted or mis-ordered, because it is not in the list being iterated. An unrecognised `channel`
value resolves to `NOT_APPLICABLE`, never to `BLOCK` — a typo silences a rule, it never promotes
one.

This is worth the ceremony because the failure would be **silent**: a WARN rule in the veto path
still returns `BLOCK`, the harness still computes an F1, and nothing looks broken — but §13's
92–94% intervention precision would be measured on a wider set than the one it was validated on.
`tests/test_shield_channels.py::test_no_non_block_rule_can_ever_veto` and
`::test_shipped_corpus_never_vetoes_outside_the_block_channel` pin it.

Back-compatibility is exact: a rule with **no** `channel` key keeps v1 behaviour
(CONSTRAINT → BLOCK, AFFIRMATION → NORMAL), so `validated_translated/all_rules_deduped.jsonl`
behaves identically under v2.

### 22.3 The proof: 58 rules, same numbers

`evaluation/eval_shield_n1.py` run on all three grids against the 58-rule channel corpus, compared
field by field with the recorded 4-rule results:

| grid | rules | F1 model → shielded | delta | every metric arm |
|---|---:|---|---:|---|
| neurips2020 | 4 → **58** | 0.8956 → 0.9038 | +0.0082 | **identical** |
| case14 | 4 → **58** | 0.4167 → 0.4188 | +0.0021 | **identical** |
| wcci2022 | 4 → **58** | 0.5577 → 0.6253 | +0.0676 | **identical** |

`arms`, `shield_health`, and every scalar (`eligible`, `blocked`, `corrections`, `regressions`,
`missed_violations`, `missed_reachable_by_rule`, `violation_rate`, `f1_rule_baseline`) match
exactly. `ap` and `threshold` differ at 1e-9 and 1e-6 — the forward-pass nondeterminism already
documented in CLAUDE.md, reproducible between two runs of the *same* file. Artifacts:
`results/shield/shield_<tag>_v2channels.json`.

**One reported figure does move, and it is not a metric.** Block severity reads `critical` instead
of `high` (353 / 103 / 22 559 blocks — the same blocks). The served corpus deduplicated ten
validated restatements of `loading_pct > 100` down to three records, none of which was
`R_2165`; serving all ten surfaces that **`R_2165` (EMO Dispatch Computer Constraints, §3.1)
labels the same physical limit `critical` where nine other clauses call it `high` or `medium`**.
That is a real disagreement between standards about the severity of one thermal limit, not a
defect. The block *decisions* are identical. Supersedes "blocks are `high` severity only".

### 22.4 Two defects found by looking at the rendered output

Both were mine, both were invisible in the tests, and both were caught only by reading an
explanation rendered from a real case14 frame.

1. **The warning quoted the confounded rate.** The corpus builder shipped
   `pooled.p_violation_given_fires` next to `pooled.discrimination`. Only the discrimination is
   stratified (§21.1), so an ELEVATED rule rendered as *"21.0% when this fired against 24.5% when
   it did not"* — the pair backwards, on a rule whose whole claim is that firing raises risk. Rates
   are now taken from the grids that actually scored the predicate: **47.4% against 24.5%**, which
   is §21.2's wcci2022 row.
2. **A vocabulary mismatch rendered as contradiction.** Nine NORMAL rules affirm `normal` — the
   retired 4-class label — while an N-1 verdict is `secure`. Same physical state, different task
   vocabulary, so v1 filed all nine as *contradicting* and v2 dutifully printed "affirm a state
   other than the one predicted". The labels are **not** aliased (that would move
   `supporting_rules` and therefore the Option B counterfactual §13 reports); instead the text now
   says what they are: *"affirm a state in the retired 4-class vocabulary, which does not compare
   with an N-1 verdict — reported, not counted either way."*

### 22.5 The limit a reader should know — the model-conditional half is now CLOSED (§23)

A warning names the grid its rate was measured on, and that grid may not be the grid it is running
on. `voltage_pu_min < 0.95 or voltage_pu_max > 1.05` is calibrated on wcci2022 only — on
neurips2020 and case14 it fires on 100% of frames and has no silent arm (§21.2). The shield does
not know which topology it is deployed on, so it cannot suppress the rule there; naming the
measurement grid in the text is the honest minimum, not a full answer. The remaining item is
§21.4's model-conditional calibration.

```
.venv\Scripts\python.exe evaluation\build_channel_corpus.py
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag wcci2022 --rules shield_corpus\all_rules_channels.jsonl --json results\shield\shield_wcci2022_v2channels.json
.venv\Scripts\python.exe -m pytest tests\test_shield_channels.py -v
```

---

## 23. The model-conditional calibration — the last open item on the WARN channel

§21 measured P(N-1 violation | rule fires) over *all* base-case frames. §21.4 flagged that as
narrower than the channel's actual semantic: a warning says *"this prediction was **let through**,
but that could happen"*, so the conditioning set is the contingencies the model predicted `secure`
— not every contingency. §22.5 carried it as the last open item. This is it, run.

`evaluation/warn_rule_calibration_conditional.py`, every frame of all three grids, held threshold
0.8849, eval batch size 64. Artifact: `results/audit/warn_n1_calibration_conditional.json`. All
statistics are **imported** from `warn_rule_calibration.py` rather than copied, so the two passes
cannot drift apart.

### 23.1 No predicate changed sign

| condition | shipped | grid-cond. | **model-cond.** | verdict |
|---|---|---:|---:|---|
| `voltage_pu_min < 0.95` | Y | +0.537 | **+0.290** | ELEVATED |
| `voltage_pu_min < 0.95 or voltage_pu_max > 1.05` | Y | +0.229 | **+0.216** | ELEVATED |
| the five ±10%/±15% voltage bands | Y | +0.297 | **+0.152** | ELEVATED |
| `voltage_pu_min < 0.917` | Y | +0.726 | *sample lost* | **ELEVATED → INSUFFICIENT** |
| `power_factor_at_max_load > 0.95` | n | −0.163 | **−0.151** | INVERTED |
| `power_factor_at_max_load > 0.90` | n | −0.070 | **−0.051** | INVERTED |

**Every ELEVATED predicate that keeps a sample stays ELEVATED; both INVERTED predicates stay
INVERTED.** §21.4's stated expectation — *direction unlikely to flip, magnitudes will move* — held.
**19 of the 20 shipped WARN records, and 7 of the 8 shipped distinct conditions, survive.**

**The absolute rates a warning may quote drop sharply and must be restated.** The ±10% family goes
from "49.1% when it fired against 19.4% when it did not" to **18.2% against 3.0%**. But the
baseline falls further than the rate does: the violation rate among model-secure contingencies is
0.030 / 0.225 / 0.133 against 0.195 / 0.278 / 0.248 grid-wide. On neurips2020 the warning is
therefore **6.0× the base rate** where the grid-conditional pair was 2.5×. The *relative* signal
gets stronger; the number printed to an operator gets smaller. Both must be restated together.

### 23.2 Why one rule lost its sample — the substantive finding

`voltage_pu_min < 0.917` scored only on case14 in §21. Under model conditioning its firing frames
go **35 → 2**, and the whole undervoltage family collapses the same way on that grid (`< 0.9 or
> 1.1`: 23 → 0; `< 0.95`: 38 → 2).

The cause is not arithmetic. **180 case14 frames drop out because the model predicted `violation`
on every single contingency in them** — and those 180 frames contain essentially all the
undervoltage-firing frames. On case14, when the base case is deeply undervolted, the model already
flags everything. There is no prediction being "let through" for the rule to warn about: *its
information is already inside the model's output.*

That is the most interesting thing this pass found, and it cuts both ways. It is a limit on the
rule (nothing left to add) and a credit to the model (it saw what the standard saw). The voltage
predicates that survive are the ones whose signal sits in the model's **blind spot** rather than
beside its strength — which is the only place a shield can add anything at all, and is the same
mechanism §13.3's structural ceiling describes.

### 23.3 Three caveats that travel with this

1. **`voltage_pu_min < 0.95` clears the bar on wcci2022 with exactly 30 firing frames**, the
   minimum. Its ELEVATED verdict is one frame from INSUFFICIENT. Do not present it as robust.
2. **~70% of the neurips2020 frames scanned are in-sample for the checkpoint.** `score_topology`
   restricts the home grid to its held-out test split (1,800 frames), which at a 30-frame bar
   reports INSUFFICIENT for arithmetic reasons and would confound "model-conditional" with
   "different frames". All 12,000 were scanned instead, and the leak is disclosed rather than
   hidden. The uncontaminated arm (`--neurips-scope test`) was run as a sensitivity check:
   **every sign is preserved and the magnitudes are larger, not smaller** (+0.209 against +0.152 on
   the voltage family). The all-frames choice is therefore conservative, not flattering.
3. **Frames with no model-secure contingency contribute to neither arm**, and are counted
   (`frames_no_secure_contingency`: 15 / 180 / 0). That counter turned out to be the entire
   explanation of §23.2, which is why it is reported rather than swept into a denominator.

### 23.4 The one decision this left open — SETTLED: `R_1098` stays in WARN

`R_1098` (`voltage_pu_min < 0.917`) ships in the WARN channel on its grid-conditional verdict.
The model-conditional pass cannot score it. Two readings were available:

- **Keep it in WARN.** Its grid-conditional discrimination is **+0.726, the strongest in the
  corpus**, and the model-conditional pass did not *contradict* it — it ran out of sample. Absence
  of evidence under a narrower test is not evidence of absence, and the same reasoning already
  governs `partition()` in the polarity guard, which keeps rules that cannot be measured anywhere.
- *(rejected)* Demote it to NOT_APPLICABLE, on the ground that §21's rule was that an uncalibrated
  warning does not ship.

**Decision, 2026-09-17: it stays.** The shipped figures therefore remain **41 speaking /
17 distinct**. Two things must travel with that choice and are not optional:

1. `R_1098`'s quoted rate is the **grid-conditional** one (P|fire 0.843 on case14), and the
   explanation text names the grid, as it does for every warning.
2. §23.2 is the reason it cannot be scored model-conditionally — on case14 the model already flags
   every contingency when voltage is that low. A reader who asks "why is this one not in the
   §23.1 table?" must be able to find that answer, which is why it is stated there and not only
   here.

No measured result depends on this either way.

```
.venv\Scripts\python.exe evaluation\warn_rule_calibration_conditional.py
.venv\Scripts\python.exe evaluation\warn_rule_calibration_conditional.py --neurips-scope test
```

---

## 24. The unanswered question — is the thin corpus the pipeline's fault or the task's?

**Status: BOTH ARMS RUN 2026-09-19 — §28 (ceiling tree) and §29 (expert rules), and the
ceiling arm SUPERSEDED the same day by §30 (enumeration).** §24.1 is answered in favour of
reading **(a)**, now from three directions rather than two — and §30 is the one to cite,
because it is the only one that does not depend on how well something searched:

| route | question | answer |
|---|---|---|
| §28 → **§30** | could *any* rule over these variables beat the corpus? | Exhaustively: only the same rule at a looser threshold, +0.045 F1 and −0.37 precision. Nothing else, and no pair. |
| §30.6 | could an **uncapped, unreadable** model beat it? | No — it loses to one threshold comparison on all three grids. |
| §29 | could a **human writing from doctrine** beat it? | ⚠️ **OPEN** — not a blind baseline (§29.1), and §30.8 explains why the question is no longer load-bearing. |

⚠️ **§28's headline is corrected by §30.5** — "the shield is at the fitted ceiling" is true
of a greedy tree and false of the enumerated language. ⚠️ §29 is **not a blind baseline**
and is **left unrevised pending that decision** — do not quote it as final.

§28 executes §24.2 across all
three grids and answers §24.1 in favour of reading **(a)**: a tree fitted directly on the answer,
over the same 14 variables, does not beat the extracted corpus. The spec below is kept as written
because §28 follows it verbatim and the reasoning is part of the record.

**Original status: NOT RUN. This was the largest remaining hole in the write-up, and it is the
question a reviewer is most likely to ask.** §17.1 named the controls; this section is the pickup-ready spec,
written so a cold reader can run it without reconstructing the reasoning.

### 24.1 The question, stated so it can be answered

The thesis reports a 0.16% end-to-end extraction yield and a shield built on 3 distinct predicates
(17 after §21–§23). Two very different readings produce that same number:

- **(a) The task is genuinely rule-poor.** The standards do not say much that is both expressible
  over present-state telemetry and predictive of an N-1 contingency. A thin corpus is then a
  *finding* about the standards/simulator mismatch, and the thesis's central claim survives.
- **(b) The extraction pipeline is the bottleneck.** The knowledge is there and the LLM stages
  failed to get it out. A thin corpus is then a *limitation of the method*, and every downstream
  measurement inherits it.

**Everything in this document argues (a) and nothing measures it.** §5, §10.4, §15 and §20.2 all
support (a) by different routes — but every one of them is an argument about *why* the yield is
low, made from inside the pipeline that produced it. None is an external check. That asymmetry is
exactly what a sceptical reader will find, and reporting (b) dressed as (a) is the specific failure
§17.1 was written to prevent.

⚠️ **Do not let the §13.6 replication or the §13.3 ceiling stand in for this.** They establish that
the result is *stable* under corpus change and that a *structural* limit exists — neither says
whether a better corpus was obtainable.

### 24.2 Arm 1 — the ceiling analysis (answers it without any rules at all)

Fit a small decision tree (depth ≤ 4, so it stays a readable rule set) on the **14 context
variables** from `shield/context.py`, targeting the error the shield is built to catch.

- **Target: `model predicted secure AND the contingency was a violation`.** Not "any model error".
  The gate is asymmetric (§13 / `validate_n1`) and only blocks the over-permissive direction, so
  scoring against symmetric error would measure something the shield never tries to do.
- **Fit on the neurips2020 *training* split only; evaluate on its test split and on the two foreign
  grids.** Fitting and scoring on the same frames produces an oracle and answers nothing. This is
  the same held-out discipline as the 0.8849 threshold, for the same reason.
- Held threshold **0.8849**, eval batch size **64**. Non-negotiable — see CLAUDE.md.
- Note the honest constraint on the tree: the context is **per frame** and the error is **per
  contingency**. The tree therefore cannot distinguish two contingencies in the same frame — but
  neither can any extracted rule, so this is the fair upper bound for frame-level symbolic gating,
  not a handicap. Say so when reporting it.

**Reading the outcome:**

| tree result | conclusion |
|---|---|
| cannot beat chance | **(a), established without reference to rule count at all.** The strongest possible version of the result: the information simply is not in the present-state telemetry. |
| does well, shield does not | the information *was* there and the extracted rules missed it — **(b)**, and the tree's splits name exactly which variables were missed |
| does about as well as the shield | the shield is at the ceiling; extraction was not the binding constraint |

### 24.3 Arm 2 — the expert baseline (sharper, and easy to contaminate)

Hand-write **10–15 rules** against the same 14 variables, from domain knowledge and the source
standards, and run them as a **third arm that is NEVER merged into the corpus**. Merging them
destroys the provenance claim that is the thesis's actual contribution — every served rule must
trace to a clause in a document.

🚨 **The contamination trap, and it is easy to fall into.** Rules written *after* looking at fire
rates, audit verdicts, or the §21 calibration are not an independent baseline — they are the
extracted corpus laundered through a human. The expert rules must be written **before** consulting
`results/audit/*`, and the write-up must record that ordering explicitly, or the arm proves
nothing. If that ordering cannot be honoured, report the arm as *"a hand-written comparison, not a
blind baseline"* and claim less.

| expert-rule result | conclusion |
|---|---|
| ~0 delta, like the extracted rules | gating does not help *regardless of provenance* — **extraction thinness is exonerated**, and this is the cleanest possible (a) |
| clearly beats the extracted corpus | **extraction is the bottleneck — (b)**, stated plainly and prominently |

### 24.4 What already exists, and what a preview of the answer looks like

Nothing needs generating. On disk: the three N-1 datasets, `gnn_checkpoint_n1.pt`,
`shield/context.py`'s builder, and `evaluation/eval_shield_n1.py`'s scoring — the ceiling arm is a
scikit-learn fit wrapped around machinery that already runs.

**One result already points toward (a), and it arrived by accident.** §23.2: on case14, when the
base case is deeply undervolted the model *already flags every contingency*, which is why
`voltage_pu_min < 0.917` lost its sample. That is a single instance of the ceiling argument
measured on real data — the rule's information was already inside the model's output. It is
suggestive, it is one predicate on one grid, and it is **not** a substitute for the controls above.

### 24.5 Effort, and the honest fallback

The ceiling arm is roughly **a day** and needs no new data. The expert arm is a day of writing plus
a harness run, and its cost is mostly the discipline of writing the rules before looking at the
answers.

**If neither is run, the thesis must say so in those words** — that the controls were specified and
not executed, and that (a) is therefore argued rather than demonstrated. A disclosed gap is a
limitation; an undisclosed one is the hole a reviewer falls into. §17.1 already states this; do not
soften it.

---

## 25. Arm A1 — DC/LODF contingency screening. The incumbent method beats the model on every grid.

**Run 2026-09-19.** `results_comparisons.md` §A1 specified this as the priority comparator and named
three possible outcomes. The one that occurred is the third: *"above the model — report it plainly.
The neural component would not be earning its place on this task."*

Code: `evaluation/lodf.py` (factors), `evaluation/eval_lodf_n1.py` (arm),
`tests/test_lodf.py` (validation). Artifacts: `results/lodf/lodf_<tag>.json`.

### 25.1 The result

Every figure is at the **oracle** column — best threshold chosen on each grid with the answer key —
because that is the model's most favourable presentation. The rightmost column is a *zero-tuning*
decision rule, predicted loading >= 1.0, with no threshold selection of any kind.

| grid | all-positive | best rule | **GNN (oracle)** | LODF flow (oracle) | **LODF + topology (oracle)** | LODF + topology **untuned at 1.0** |
|---|---:|---:|---:|---:|---:|---:|
| neurips2020 *(trained on)* | 0.3104 | 0.4639 | **0.8972** | 0.8906 | **0.9550** | 0.9509 |
| case14 *(unseen, smaller)* | 0.4345 | 0.5392 | **0.4477** | 0.7797 | **0.9150** | 0.9095 |
| wcci2022 *(unseen, larger)* | 0.3969 | 0.4915 | **0.5721** | 0.8208 | **0.9207** | 0.9137 |

Held-threshold figures (selected once on the neurips2020 val split, applied unchanged) are in the
artifacts and move nothing: LODF + topology scores **0.9547 / 0.9147 / 0.9199**.

**The model matches DC screening on the grid it was trained on and loses heavily on both grids it
was not.** On case14 the gap is 0.4477 against 0.9150 — the model is not merely below the linear
method, it is below it by more than it is above the all-positive baseline.

### 25.2 Why the numbers can be trusted

Three independent checks, all of which had to pass before any of the above was written down.

1. **The labels are the harness's labels, not a reconstruction.** The arm calls
   `scripts.pyg_data.build_line_targets` — the same function the model's DataLoader calls — and
   rebuilds `y` frame by frame. It then recomputes the all-positive and best-single-rule baselines
   from that `y`. They reproduce **0.3104 / 0.4639, 0.4345 / 0.5392, 0.3969 / 0.4915** and the
   contingency counts **113,205 / 118,502 / 742,472** exactly, against the recorded values. A
   misalignment of even one line would move them.
2. **The factors are pinned against pandapower.** `tests/test_lodf.py` compares every LODF entry
   against `makePTDF` + `makeLODF` on all three intact grids, checks the bridge detection against an
   independent union-find on outaged topologies, and verifies Kirchhoff's law holds for the
   redistributed flows. 18 tests, all green; suite total 302.
3. **The linear estimate tracks the real AC solve.** Against `n1_post_rho` — the actual
   post-contingency maximum loading from Grid2Op's solver — the predicted value has median absolute
   error **0.0095 / 0.0132 / 0.0110** and Pearson *r* **0.914 / 0.873 / 0.707**.

Eval batch size does not enter: there is no forward pass and no BatchNorm. §0's constraints 2 and 3
are honoured (held threshold selected on the home val split; neurips2020 scored on its test split
only, foreign grids on every frame).

### 25.3 The redistribution is doing the work — the control that proves it

A DC screen could look strong for a trivial reason: if some *other* line is already at rho >= 1.0 in
the base case, almost any contingency is a violation. The arm therefore carries a control that is
the identical statistic with the LODF factors switched off — max rho over lines other than k:

| grid | no-redistribution control | LODF flow | contribution of the factors |
|---|---:|---:|---:|
| neurips2020 | 0.3430 | 0.8906 | **+0.548** |
| case14 | 0.4474 | 0.7797 | **+0.332** |
| wcci2022 | 0.4582 | 0.8208 | **+0.363** |

The control sits at or below the single-rule baseline on all three. The strength is the physics, not
a frame-level artifact.

### 25.4 🚨 §A1 q4's ceiling does not exist — a topology pre-screen catches most game-overs

`results_comparisons.md` §A1 q4 argued that because a DC screen always solves, it **cannot produce
the game-over class of positive at all**, putting a hard ceiling of roughly 75–79% recall on any
linear method. **That is false, and it is the reason the doc's headline prediction was wrong.**

A game-over is overwhelmingly a *structural* event — the outage strands a load or a generator — and
structure is exactly what a topology pre-screen reads. Every real contingency tool runs one before
touching a flow calculation. Implemented as `isolates_supply()`:

| grid | fired | **precision** | share of game-over positives caught |
|---|---:|---:|---:|
| neurips2020 | 2,464 | **1.000** | 55.1% |
| case14 | 7,359 | **1.000** | 88.5% |
| wcci2022 | 33,240 | **1.000** | 71.7% |

**Precision is exactly 1.000 on all three grids** — 43,063 predictions, zero false alarms. If an
outage disconnects supply, the episode ends; there is no counterexample in 974,179 contingencies.
The remaining game-overs are non-structural (an overload cascade that trips further lines), and the
flow channel reaches part of those.

The correct statement is therefore the narrower one: **the flow channel alone cannot produce the
game-over class; the method as a whole can, because topology screening is part of it.** §A1 q4
conflated the channel with the method.

### 25.5 What this does and does not overturn

**It does not touch the claim that producing an N-1 label requires a power-flow solve.** That claim
is why `classify` was abandoned (§9) and it survives intact — LODF is a *linear solve over the
network*, not a rule over the present observation. The no-redistribution control is the
rule-over-the-observation version, and it scores 0.34–0.46.

**It does overturn the inference drawn from that claim.** "The label needs a solve, therefore a
learned network-aware model is the right tool" does not follow. A linear solve is also a solve, it
is cheap, it is a century old, and on this benchmark it is better. The thesis may no longer present
the GNN as earning its place on N-1 screening accuracy.

**It explains the case14 failure that §14.1 could not.** ⚠️ **Weakened by §27, which tested
this directly.** Handing the model the reactances raises its case14 ranking (oracle +0.068 over
four seeds) but leaves it below the single-rule baseline and nowhere near LODF. The diagnosis
captured something real and is **not** the whole explanation; do not state it as the cause.
The original wording follows. §14.1 recorded the 0.77x rule-baseline
failure and offered an untested hypothesis about voltage feature scaling. The simpler explanation is
now measurable: the GNN learned an approximation of linear flow redistribution *on one network's
parameters*, and redistribution depends on reactances the model was never given. LODF transfers
because it is recomputed from each grid; the GNN cannot, because it has nothing to recompute from.
**The 0.4477 -> 0.9150 gap on case14 is the size of that missing input.**

**Two asymmetries, stated in both directions.**

- ⚠️ **In LODF's favour:** it is given the branch reactances and the load/generator bus assignments.
  The GNN receives topology and operating state but **never the electrical parameters**. This is a
  real informational advantage — and it is also freely available data that any operator has, which
  is precisely why it is the incumbent method. The honest reading is not "the comparison is unfair"
  but "the model was denied an input the baseline uses, and the cost of that is now measured."
- ⚠️ **In the model's favour, and it is the sharper caveat:** the labels come from Grid2Op's own
  power-flow solver on this same network. A linear approximation of that solver is approximating the
  labelling process analytically, which no learned model can do. This is a milder version of the
  defect that killed `classify` (§9): there the labelling function was *exactly* four thresholds;
  here it is *approximately* a linear solve. Approximately, not exactly — flow-channel recall is
  0.63–0.87 and DC–AC *r* is 0.71–0.91, so the task is not closed-form. But it is far closer to
  closed-form than the thesis assumed, and that must be said.

### 25.6 A counting correction to `results_comparisons.md` §A1 q4

The doc reports positives of 31,390 / 131,933 / 182,312 and game-over shares of 26.5% / 21.7% /
25.4%. Those came from `y.sum()` over the raw per-line vectors, which sums `-1` for every
de-energized line and therefore undercounts. Over in-service lines only — the entities actually
scored — the figures are:

| grid | positives | game-over positives | share |
|---|---:|---:|---:|
| case14 | 32,888 | 8,313 | **25.3%** |
| neurips2020 | 137,315 | 28,616 | **20.8%** |
| wcci2022 | 183,840 | 46,373 | **25.2%** |

The decomposition itself is exact and one-sided, which the doc got right: `n1_violation == 1` holds
**iff** `n1_post_rho >= 1.0` or `n1_post_rho == 0.0`, with **zero** contingencies in either
off-diagonal on any grid, and **zero** masked in-service entries anywhere — so `line_mask` is a
no-op on all three datasets.

### 25.7 What follows

This changes what the remaining arms in `results_comparisons.md` §A are for. A3 (tabular) and A2
(Pavão MLP) were specified to answer "is the GNN over-parameterised for this task?" The prior
question is now "does a learned model belong on this task at all?", and A1 has answered it for
accuracy. The arms are still worth running — they locate where on the ladder the GNN sits — but
their framing must change, and the results section must lead with §25.1, not bury it.

⚠️ **Do not soften this into "LODF is a strong baseline."** It beats the model on all three grids,
at the model's most favourable threshold, with no tuning of its own. That is the finding.

---

## 26. Arm A3 — a model with no graph at all collapses the same way. The transfer claim broadens.

**Run 2026-09-19.** `results_comparisons.md` §A3, re-scoped from its original
in-distribution framing to the question the thesis turns on: **when the GNN fails on an
unseen grid, does a non-graph learned model fail with it?**

It does. Code `evaluation/eval_tabular_n1.py`, artifact
`results/tabular/tabular_baselines.json`.

### 26.1 Setup

Features are the **raw portion of the GNN's own readout** — for line k, its 8 edge
features plus the 8 node features of each endpoint, 24 columns. That is the readout's
skip connection with the three GATv2 layers deleted, so the comparison is "same
information, no message passing, no graph" rather than "some other feature set".

Fit on the neurips2020 **train** split only (491,684 contingencies, 19.6% positive).
Threshold selected on its **val** split and held unchanged across all three grids.
Features standardized with train-split statistics, applied unmodified to the foreign
grids — the GNN's own normalization rule. Scored by `best_f1_and_thr`. The recomputed
rule baseline reads **0.4639 / 0.5392 / 0.4915** on the three grids, which is the
alignment check: these are the same contingencies the model is scored on.

### 26.2 The result

Held threshold throughout; oracle in brackets.

| grid | all-positive | best rule | logistic | **GBT** | **GNN** | LODF + topology |
|---|---:|---:|---:|---:|---:|---:|
| neurips2020 *(trained on)* | 0.3104 | 0.4639 | 0.7121 | **0.9190** *(0.9202)* | **0.8956** *(0.8972)* | 0.9547 *(0.9550)* |
| case14 *(unseen)* | 0.4345 | 0.5392 | 0.4554 | **0.5845** *(0.6320)* | **0.4167** *(0.4477)* | 0.9147 *(0.9150)* |
| wcci2022 *(unseen)* | 0.3969 | 0.4915 | 0.4705 | **0.5177** *(0.5665)* | **0.5577** *(0.5721)* | 0.9199 *(0.9207)* |

**Degradation from the home grid, at oracle:**

| method | neurips2020 | case14 | wcci2022 | worst drop |
|---|---:|---:|---:|---:|
| logistic regression | 0.7125 | 0.4895 | 0.4725 | **−34%** |
| gradient-boosted trees | 0.9202 | 0.6320 | 0.5665 | **−38%** |
| GNN | 0.8972 | 0.4477 | 0.5721 | **−50%** |
| **DC/LODF** | 0.9550 | 0.9150 | 0.9207 | **−4%** |

### 26.3 What it establishes

**The transfer failure is not about graphs. It is about learning.** Three learned
methods — a linear model, a tree ensemble, and a graph attention network — lose
31–50% of their home-grid score on an unseen topology. The analytical method loses 4%.
That is the strongest available form of the thesis's central claim, and it now rests on
three independent architectures rather than one.

⚠️ **State it as "learned screeners trained on one topology did not transfer", not as
"neural networks cannot generalise".** Nothing here tests a model trained on several
grids, and §25.5's diagnosis stands: none of these methods was given the branch
reactances that determine where power actually goes, so none of them had anything to
recompute on a new grid. The claim is about this training regime, not about neural
networks as a class.

**The GNN degrades worst of the three.** On case14 it loses 50% where the tree loses
31%, and it ends up *below the single-rule baseline* while the tree stays above it.
Message passing over a topology-specific graph appears to make transfer worse, not
better — consistent with §25.5: the model encodes one network's redistribution pattern
and carries it to a grid where it does not hold, while a tree over local features has
less topology-specific structure to be wrong about.

### 26.4 The in-distribution comparison is a tie, and saying otherwise would be an artifact

The GBT's 0.9190 is above the GNN's 0.8956 on the home grid, and it would be easy to
write "a gradient-boosted tree beats the GNN in-distribution". **Do not.** The GNN's
absolute F1 moves with eval batch size — 0.8972 at 64, **0.9255 at 512**, same
checkpoint, same split (§8 of `gnn_n1_tightening.md`). The GBT's 0.9202 sits inside
that band. At the project's pinned batch 64 the tree is ahead; at 512 the GNN is. The
honest statement is that **in-distribution they are indistinguishable at the precision
this protocol supports**, which is itself the answer to §A3's original question: the
graph machinery is not buying measurable accuracy on the grid it was trained on.

Off-distribution there is no such ambiguity — the gaps are 0.13–0.19, far outside any
batch-size band.

### 26.5 What was dropped, and why

**A2 (replicating the Pavão MLP) is dropped.** It was specified to build an
architecture ladder — plain net, no message passing, full GNN — which measures model
capacity, not transfer. A3 answers the transfer question with the same class of model,
at lower cost, on features we actually have; A2's feature list includes maintenance
schedules and cumulative overload duration that our records do not carry. If the ladder
is wanted later for a different reason, §A2 is still written.

⚠️ **The head-only ablation is not "already half-built" as `results_comparisons.md` §A5
claims.** `gnn_checkpoint_n1_headonly.pt` is **not on disk** — only
`gnn_checkpoint_n1.pt` exists. The recorded 0.8411 cannot be reproduced without
re-running `train_gnn.py --head-only`.

### 26.6 The open question this sharpens

Every learned arm here was denied the same input: the electrical properties of the
lines. The decisive experiment is now to give them to the model and re-measure —
retrain with per-line reactance as an edge feature and score the same three grids. If
transfer still fails, §26.3's claim survives its strongest attack. If it does not, the
claim was an artifact of feature selection, and that is far better discovered here than
in a viva. **Nothing else outstanding is worth as much.**

---

## 27. The reactance experiment — testing §25.5's diagnosis directly

**Design fixed 2026-09-19; result in §27.3.** §25.5 explained the cross-topology
collapse by a missing input: where power goes when a line trips depends on the branch
reactances, the model was never shown them, so it had nothing to recompute on an unseen
grid. §26 strengthened the circumstantial case — a logistic regression and a
gradient-boosted tree, denied the same inputs, collapse the same way — but the
diagnosis had still never been tested. It is the strongest available attack on §26.3's
claim, and it is cheap to run, so it is run here rather than left as an argument.

Code: `evaluation/reactance_transfer.py`. Artifact `results/reactance/reactance_transfer.json`.

### 27.1 Two arms, one difference

**The deployed checkpoint cannot be the control.** `train_gnn.py` saves a bare
state_dict with no epoch field and writes no metrics file, so what actually produced
`gnn_checkpoint_n1.pt` is not recoverable from disk (CLAUDE.md; `study.md` §7).
Comparing a new 9-feature model against it would confound the added feature with an
unknown training run. Both arms are therefore trained here:

| | edge features | everything else |
|---|---|---|
| **control** | 8 — the deployed set | identical |
| **physics** | 9 — the deployed set plus the reactance column | identical |

Same 30 epochs, same batch size 128, same learning rate 3e-4, same chronic-level split,
same graphs. **Repeated over four seeds (42, 0, 1, 2)** — §27.3.3 explains why one was
not enough. The graphs are built **once** with nine columns and
the control arm slices the ninth off, so the two arms cannot differ in graph
construction, ordering or split — only in what the model is allowed to read.

⚠️ The control is not expected to reproduce `gnn_checkpoint_n1.pt` exactly, and does not
need to. The comparison that matters is control-vs-physics, both trained by this script.
The control's own cross-topology behaviour is reported so a reader can see whether this
training run reproduces the deployed model's *pattern*.

### 27.2 The feature, and why it is not the raw reactance

Susceptance `b = 1 / (x · tap)` is **not comparable across these grids.** Medians are
**5.13 (case14), 985.22 (neurips2020), 1156.77 (wcci2022)** — a ~200x spread caused by
differing `baseMVA` conventions in the source networks (100 for case14, 1 for the other
two), not by physics. Feeding that raw would guarantee a collapse on the foreign grids
and the experiment would have measured unit conventions.

The feature is therefore

```
log1p( b / median(b) )
```

- **Dividing by the grid median is lossless for this purpose.** LODF is invariant under
  a uniform scaling of every susceptance, so the median divisor discards exactly the
  quantity the analytical method also ignores. What survives is each line's electrical
  strength *relative to its own network*, which is what sets the redistribution shares.
- **The log is for conditioning.** `b/median(b)` spans [0.36, 4.63] / [0.31, 10.80] /
  [0.21, 21.34] — comparable centres but tails that differ threefold across grids.
  After `log1p` the ranges are [0.31, 1.73] / [0.27, 2.47] / [0.19, 3.11], which is a
  far better match for a feature that will be z-scored with the home grid's statistics
  and fed to attention.

**`scripts/pyg_data.py` is not modified.** The column is appended after `build_data`
returns, so the deployed 8-feature path cannot regress and the 302-test suite is
untouched. Checkpoints are written to `gnn_checkpoint_n1_reactance_{control,physics}.pt`
and never to the deliverable.

### 27.3 Result — four seeds. It helps the ranking, it does not repair transfer.

Run 2026-09-19, seeds **42, 0, 1, 2**, everything else identical. Four seeds rather than
one because CLAUDE.md records seed sensitivity on this architecture, and the first
seed's case14 delta was exactly the size of thing that dissolves under replication.
It did. Artifacts `results/reactance/reactance_transfer[_seed{0,1,2}].json`.

**Physics minus control, mean over four seeds, with the observed range:**

| grid | held threshold | oracle |
|---|---|---|
| neurips2020 | **+0.0020**  [−0.0044, +0.0063] | **+0.0009**  [−0.0047, +0.0072] |
| case14 | **+0.0089**  [−0.1009, +0.0836] | **+0.0677**  [+0.0429, +0.0854] |
| wcci2022 | **−0.0051**  [−0.0151, +0.0081] | **−0.0103**  [−0.0225, +0.0112] |

**Levels, mean ± sd over the four seeds:**

| grid | col | control | physics | rule baseline | LODF |
|---|---|---|---|---:|---:|
| neurips2020 | held | 0.8835 ± 0.0092 | 0.8855 ± 0.0065 | 0.4639 | 0.9547 |
| | oracle | 0.8865 ± 0.0083 | 0.8874 ± 0.0065 | | 0.9550 |
| case14 | held | 0.4318 ± 0.0177 | 0.4407 ± **0.0672** | 0.5392 | 0.9147 |
| | oracle | 0.4558 ± 0.0141 | **0.5235 ± 0.0162** | | 0.9150 |
| wcci2022 | held | 0.5641 ± 0.0191 | 0.5590 ± 0.0127 | 0.4915 | 0.9199 |
| | oracle | 0.5708 ± 0.0189 | 0.5605 ± 0.0119 | | 0.9207 |

### 27.3.1 What is real, and what is noise

**Real: the reactance improves RANKING on case14.** The oracle delta is **+0.068 and
positive on all four seeds** (+0.043 to +0.085), against a within-arm sd of ~0.016. §25.5's
diagnosis captured something genuine — the model can use the electrical parameters, and
on the grid where it failed worst they measurably improve the order it puts
contingencies in.

**Noise: everything else.** neurips2020 and wcci2022 move by less than a hundredth in
either direction, inside their own seed spread. And at the **held** threshold the case14
delta ranges from **−0.1009 to +0.0836** — the sign is not stable, so the single-seed
+0.0651 that seed 42 produced is not a result and must not be quoted as one.

**Not enough, and this is the part that settles the question.** The best case14 oracle
any physics seed reached is **0.5339**, still below the rule baseline of **0.5392**, and
against LODF's **0.9150**. Every seed, every arm, every column: **case14 still fails.**
Handing the model the reactances closes perhaps a tenth of the gap to the analytical
method and none of the gap to a single threshold on `rho`.

### 27.3.2 The held-vs-oracle split is itself a finding

The two columns disagree on case14 and the disagreement is systematic:

| arm | case14 oracle | case14 held | gap |
|---|---:|---:|---:|
| control | 0.4558 | 0.4318 | 0.024 |
| physics | 0.5235 | 0.4407 | **0.083** |

**The added feature improves the ranking and makes the threshold transfer worse** — the
physics arm's held-to-oracle gap is 3.5x the control's. Giving the model a new input
shifts its score distribution, and a cutoff chosen on the home grid lands somewhere else
on a foreign one. This is a concrete instance of a general problem: an off-distribution
improvement that only shows up at the oracle column is not deployable, because the
oracle threshold is chosen with the answer key.

### 27.3.3 A second finding: off-distribution scores are seed-unstable

| grid | control sd | physics sd (held) |
|---|---:|---:|
| neurips2020 | 0.0092 | 0.0065 |
| wcci2022 | 0.0191 | 0.0127 |
| **case14** | 0.0177 | **0.0672** |

In-distribution the architecture is stable to within a hundredth. On case14 the physics
arm swings across a **0.18-wide range** on initialization alone. **Off-distribution
behaviour is partly a property of the random seed, not of the architecture or the data** —
which means any single-seed cross-topology number in this project carries an error bar
roughly the size of the effects being discussed. CLAUDE.md's seed-sensitivity note was
recorded for the retired `classify` task; it applies here, and more sharply off
distribution than on it.

⚠️ **This applies retroactively.** §14's cross-topology table, §25 and §26 all report
single-seed model numbers. The LODF and tabular arms are unaffected (LODF has no seed;
the GBT was fit once with seed 42 and trees are far less init-sensitive), but every GNN
cross-topology figure in this project should be read as ±0.02 at least, and ±0.07 on
case14.

### 27.3.4 Verdict against §27.4's pre-registration

This is **outcome 3**, written down before the run: *"physics > control but still far
below LODF — the input helps and does not rescue."*

**§26.3's claim survives.** The transfer failure is not simply a missing-input problem.
The model was handed the electrical parameters, in a form that transfers cleanly across
all three grids, and it recovered a tenth of the gap to a method that computes those
parameters directly. The honest statement is:

> Learned screeners trained on one topology did not transfer, and giving the model the
> branch reactances did not repair it. The parameters improve the model's ranking on the
> worst grid (oracle +0.068, four seeds) without making it deployable there — it remains
> below a single threshold on the removed line's loading.

### 27.3.5 A side result worth keeping: the deployed checkpoint is reproducible

The seed-42 control arm scores oracle **0.8978 / 0.4473 / 0.5727**. The deployed
`gnn_checkpoint_n1.pt` scores **0.8972 / 0.4477 / 0.5721** — within 0.0006 on all three
grids.

CLAUDE.md records that the deployed model's epoch count "is NOT RECOVERABLE FROM DISK"
because `train_gnn.py` saves a bare state_dict and writes no metrics. This says the
documented intended command — **30 epochs, batch 128, lr 3e-4** — reproduces its
cross-topology behaviour. It does not prove the weights match, and §27.3.3 says a
four-seed spread of ±0.018 on case14 makes an exact match unfalsifiable from one run
anyway. What it does close is the practical worry: the deployed result is reproducible,
and there is now a second checkpoint that behaves like it.

### 27.4 How to read whichever result arrives

Fixing this in advance, because both outcomes are publishable and the temptation to
narrate whichever one occurs as the expected one is real.

| outcome | reading |
|---|---|
| **physics ≈ control off-distribution** | §25.5's diagnosis is wrong or incomplete. The reactance was available and the model still could not use it, so the failure is about *learning to transfer*, not about a missing input. §26.3's claim survives its strongest attack and becomes considerably harder to dismiss. |
| **physics ≫ control off-distribution** | The diagnosis was right and the claim was an artifact of feature selection. The thesis must say so: the earlier arms measured a model that had been denied a necessary input. Far better found here than in a viva. |
| **physics > control but still far below LODF** | The most likely outcome on the evidence so far, and the most awkward to write. The input helps and does not rescue: the model can be told the physics and still not recover what computing the physics gives you for free. State both halves. |

⚠️ Whatever the result, it does **not** license "neural networks cannot generalise".
One architecture, one training regime, one added feature, three grids, one seed. §26.3's
wording — *learned screeners trained on one topology did not transfer* — remains the
ceiling.

---

## 28. §24.2 ceiling tree, all three grids — extraction was NOT the bottleneck

**Run 2026-09-19.** §24 has been the largest open item in this document since it was
written: *is the thin rule corpus the task's doing, or the pipeline's?* Arm 1 is now
executed, under §24.2's protocol rather than a new one. **Arm 2 (the expert baseline,
§24.3) is still not run.**

Code `evaluation/ceiling_tree_n1.py`, artifacts `results/ceiling/ceiling_tree*.json`.

> 🚨 **§28's HEADLINE IS CORRECTED BY §30. Read §30 before quoting anything below.**
> The tree is a *greedy* search, so this section establishes only that greedy trees of
> depth ≤ 8 do not beat the corpus. §30 replaces the instrument with an exhaustive
> enumeration of every 1- and 2-term rule in the language plus an uncapped ensemble,
> and finds that **a better rule does exist**: `loading_pct > ~97`, the same predicate
> with a threshold three points below the standards' round 100, worth +0.0504 on
> neurips2020 and +0.0435 on wcci2022 — bought by spending precision (0.566 against
> the corpus's 0.938). The claim this section is cited for — **extraction was not the
> bottleneck** — survives and is better supported (§30.6). The sentences that do not
> survive are flagged inline in §28.2 and §28.4.
>
> ⚠️ "Arm 2 … is still not run" above is stale: arm 2 ran and is §29. Its status is
> **open**, not settled — see §29.1 and §30.8.

### 28.1 What was run

A depth-≤4 decision tree fitted on the **14 variables of `shield/context.py`** — the
exact namespace every extracted rule is written over, listed explicitly in the script
so the tree cannot gain a feature the rules never had. Target, per contingency:
**`the model predicted secure AND the contingency was a violation`** — the asymmetric
error `validate_n1` exists to catch, not symmetric error.

Fitted on the neurips2020 **train** split only (8,397 frames, 491,684 contingencies).
Scored on its test split and every frame of both foreign grids. Eval batch size 64. The
decision threshold was **re-derived on the val split and asserted against the recorded
0.8849** before anything else ran — the script refuses to continue if it drifts.

The tree is fitted **directly on the answer**, so it is the best a frame-level symbolic
gate could do on those variables. What the extracted corpus achieves against it is the
measurement.

### 28.2 The result — the fitted ceiling does not beat the extracted rules

> ⚠️ **CORRECTED TWICE.** (1) §29.7 — this table originally scored the tree over *all*
> rows where a gate acts only on predicted-secure rows; the like-for-like figures are
> **0.2508 / 0.0179 / 0.0025**. (2) §30.5 — the heading is wrong as written. The
> *greedy* fitted ceiling does not beat the extracted rules; the **enumerated** ceiling
> does, by 0.04–0.05 F1, at roughly half the precision.

Identical target, identical rows, both scored as precision/recall/F1.

| grid | all-positive | **tree (held)** | tree (oracle) | **extracted shield** |
|---|---:|---:|---:|---:|
| neurips2020 | 0.0354 | **0.1755**  p .192 / r .162 | 0.1825 | **0.2765**  p .938 / r .162 |
| case14 | 0.2712 | **0.0154**  p .049 / r .009 | *0.2712* | **0.0102**  p .922 / r .005 |
| wcci2022 | 0.1682 | **0.0025**  p .457 / r .001 | *0.1682* | **0.4644**  p .934 / r .309 |

**On the home grid the tree and the shield catch exactly the same number of errors —
331 of 2,041, recall 0.162 each — and the shield does it with five times the
precision** (0.938 against 0.192). The tree flags 1,724 contingencies to find those
331; the shield flags 353.

⚠️ **CORRECTED — see §29.7.** This table scores the tree over ALL rows, while a gate can
only act where the model predicted secure. The like-for-like figures are
**0.2508 / 0.0179 / 0.0025** and the right baseline is *flag-all-eligible*, not
all-positive. The conclusion holds and is sharper under the correction; the sentence
below is kept because the numbers above are the ones it describes.

**On both unseen grids the italicised oracle figures are exactly the all-positive
baseline.** That is not a coincidence or a rounding: the tree's best achievable F1 at
*any* threshold is obtained by flagging everything, which means its ranking carries **no
usable information at all** off-distribution. The extracted rules, meanwhile, hold
precision 0.922 and 0.934 there.

### 28.3 Robustness — the tree was not simply fitted badly

The obvious objection is that a shallow unweighted tree on a 1.8%-positive target is a
strawman. Two further configurations, both more generous than §24.2 specifies:

| configuration | neurips2020 held | case14 held | case14 **oracle** | wcci2022 held | wcci2022 oracle |
|---|---:|---:|---:|---:|---:|
| depth 4, unweighted *(§24.2 spec)* | 0.1755 | 0.0154 | **0.2712** | 0.0025 | 0.1682 |
| depth 4, class-balanced | 0.1867 | 0.0157 | **0.2712** | 0.3749 | 0.3749 |
| depth 8, class-balanced | 0.1758 | 0.0141 | **0.2712** | 0.1080 | 0.2068 |

- **case14 is immovable.** Every configuration lands at exactly 0.2712 oracle — the
  all-positive baseline. No depth, no weighting recovers any signal.
- **More capacity makes transfer worse.** Depth 8 beats depth 4 nowhere and is
  substantially worse on wcci2022 (0.108 against 0.375 held). Textbook overfitting to
  the home grid, and it closes the "too shallow" objection from the other direction.
- **The best tree anywhere still loses to the shield on F1 on two of three grids**
  (0.1867 vs 0.2765; 0.3749 vs 0.4644). It edges case14 at 0.0157 against 0.0102, where
  both numbers are indistinguishable from zero.

⚠️ The comparison **favours the tree** and it still loses: the tree gets a threshold
tuned on the val split, while the shield's predicate has no tunable parameter at all.

### 28.4 The answer to §24.1

**Reading (a): the task is genuinely rule-poor.** §24.2's table called this outcome
"tree does about as well as the shield → the shield is at the ceiling; extraction was
not the binding constraint". That is what happened, and rather more strongly than the
table anticipated — the shield is *above* the fitted ceiling on precision everywhere and
on F1 on two grids of three.

> ⚠️ **The second sentence is CORRECTED by §30.5.** Against the *enumerated* ceiling
> the shield is **below** the optimum on F1 on those same two grids (0.2765 vs 0.3269;
> 0.4644 vs 0.5079) and above it on precision everywhere (0.938 vs 0.566; 0.934 vs
> 0.744). Reading (a) — the task is rule-poor — is unaffected and is established
> independently from above in §30.6: an ensemble with no depth limit, fitted on the
> answer, loses to a single threshold comparison on all three grids.

**The thin corpus is a finding about the standards/simulator mismatch, not a limitation
of the extraction pipeline.** §5, §10.4, §15 and §20.2 all argued this from inside the
pipeline that produced the corpus. This is the external check they lacked. A reader no
longer has to take the argument on trust: fitting a model *directly on the answer*, over
the same variables, does not beat the four rules the pipeline recovered.

⚠️ **This exonerates extraction. It does not make the shield good.** Recall is 0.162 /
0.005 / 0.309 — the gate is precise and narrow. On case14 it catches **95 of 18,592**
reachable errors, which is 0.5%, and no framing makes that a success. What §28 licenses
is "the rules that exist are close to the best obtainable from these variables", not
"the rules that exist are sufficient".

### 28.5 A fourth independent route to the convergence result

The tree's split variables, by importance:

| variable | importance |
|---|---:|
| **`loading_pct`** | **0.614** |
| **`voltage_pu_max`** | **0.152** |
| `current_a_max` | 0.068 |
| `active_power_mw_max` | 0.064 |
| `voltage_pu_min` | 0.050 |
| `power_factor_at_max_load` | 0.027 |
| `total_generation_mw` | 0.019 |
| `total_load_mw` | 0.006 |

**The two dominant variables are exactly the two families the extraction pipeline
recovered** — the thermal check (`loading_pct > 100`, the most corroborated rule in the
corpus at 10 clauses across 4 documents) and the voltage band
(`voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1`). Together they carry 77% of the
tree's importance.

§15.8 recorded a convergence between an empirical filter and a textual auditor. This is
a third and fourth route to the same place: a tree fitted on the answer picks the same
two variable families that (a) an LLM reading standards documents extracted, (b) an
empirical fire-rate filter kept, and (c) the L2RPN challenge's top teams chose by hand
(§B1 of `results_comparisons.md`). Four unrelated procedures, one pair of variables.

> ✅ **§30.6 upgrades this from suggestive to overwhelming.** Split importance is weak
> evidence; exhaustive enumeration is not. Out of *every* single-variable rule
> expressible in this namespace, the best is the thermal check on both grids where a
> gate is meaningful, and the runner-up is 2.2× worse on neurips2020 and 1.7× worse on
> wcci2022.

### 28.6 Caveats

- ⚠️ **The per-frame constraint, which §24.2 requires be restated.** The context is per
  frame and the error is per contingency, so the tree cannot separate two contingencies
  within one frame. **Neither can any extracted rule** — the shield reads the same
  per-frame namespace — so this is the fair upper bound for frame-level symbolic gating,
  not a handicap imposed on the tree. Frames containing at least one target contingency:
  37.4% / 96.3% / 100.0%.
- The tree is fitted once, seed 42. Depth-4 gini is near-deterministic and trees are far
  less initialization-sensitive than the GNN (§27.3.3), but it is one fit.
- The target depends on the model's own predictions at threshold 0.8849, so it inherits
  the ±0.02 cross-topology error bar of §27.3.3. The case14 result (no configuration
  beats all-positive) is far outside that band; the neurips margin (0.2765 vs 0.1867) is
  not, and should be reported as "the shield is at or above the fitted ceiling" rather
  than as a precise gap.
- ~~**§24.3, the expert-rule arm, is still not run.**~~ **Stale.** Arm 2 ran and is §29.
  Its status is **open** rather than settled — §29.1's blindness limitation is still
  under consideration — and §30.8 explains why the question it asks is no longer
  load-bearing: enumeration answers *"could anyone have written a better rule?"*, which
  strictly dominates *"could a human have?"*.
- ⚠️ **"Too shallow" is closed here; "too greedy" is NOT**, and §30 is where that is
  dealt with. This section's search is greedy by construction, so it cannot see a rule
  that only works as a pair. §30 enumerates ~600,000 two-term rules per grid and finds
  **no such pair exists** — which retrospectively vindicates the tree as an instrument
  even though it corrects the tree's headline.

---

## 29. §24.3 expert rules — the hand-written set converges on the extracted rule

**Run 2026-09-19.** Arm 2 of §24. With §28 this closes §24, the document's largest open
item. Code `evaluation/expert_rules_n1.py`, rules `expert_rules/expert_rules.jsonl`,
artifact `results/ceiling/expert_rules.json`.

### 29.1 🚨 This is not a blind baseline, and §24.3 requires that be said first

§24.3 demands the rules be written **before** consulting `results/audit/*`, the fire
rates or the §21 calibration, because rules written afterwards are *"the extracted
corpus laundered through a human"*. **That ordering was not available.** The author had
already read the served corpus, §13's precision figures and §28's tree splits in the
same session. §24.3's own fallback therefore applies: this is reported as **"a
hand-written comparison, not a blind baseline"**, and it claims less.

Two mitigations, neither of which repairs the above:

- The 14 rules were written to disk and **fingerprinted (md5
  `05a29bd74e0ba1722e64f0c3579f6cd8`) at 08:14:58Z, before any evaluation ran.** The
  runner asserts that hash and refuses to proceed if it changed, so the rules cannot
  have been tuned against the result.
- Every family the 14 variables can express is included rather than a chosen few, and
  thresholds are the standards' own wherever one exists. Each rule carries a `basis`
  field: `standard-stated` (6), `operating-practice` (3), `doctrine` (5).

What this arm can still show: whether a **broader, doctrine-derived** rule set does
better than the four that survived extraction. What it cannot show: that a human
working blind would have written these.

### 29.2 The result

Gate semantics throughout — a rule can only act where the model predicted secure.
Target as §28: *the model predicted secure AND the contingency was a violation*.

| grid | all-positive | flag-all-eligible *(no rule)* | expert, all 14 (OR) | **expert, best single** | tree (§28, gate) | **extracted shield** |
|---|---:|---:|---:|---:|---:|---:|
| neurips2020 | 0.0354 | 0.0434 | 0.0434 | **0.2765** | 0.2508 | **0.2765** |
| case14 | 0.2712 | 0.3668 | 0.3668 | *0.3668* | 0.0179 | 0.0102 |
| wcci2022 | 0.1682 | 0.2352 | 0.2358 | **0.4644** | 0.0025 | **0.4644** |

**The best hand-written rule is the extracted rule, and it scores identically.** On
neurips2020 and wcci2022 the winner is `X_001: loading_pct > 100` at F1 **0.2765** and
**0.4644**, precision 0.938 and 0.934 — the same figures, to four decimals, as the
served corpus. A human writing from doctrine and a language model reading standards
documents produced the same predicate and therefore the same gate.

*The italicised case14 entry is not a result.* Its "best" rule is
`X_005: voltage_pu_max > 1.05`, which **fires on 100.0% of case14 frames** and is
therefore numerically identical to the flag-all-eligible baseline (0.3668 vs 0.3668).
It is the absence of a gate, not a gate. This independently reproduces the degenerate-
WARN finding already recorded in CLAUDE.md: that same voltage band fires on every frame
of neurips2020 and case14 and discriminates only on wcci2022.

### 29.3 The finding that is worth more than the comparison: more rules make it worse

**The union of all 14 rules is indistinguishable from having no rule at all.**

| grid | flag-all-eligible | expert union of 14 | difference |
|---|---:|---:|---:|
| neurips2020 | 0.0434 | 0.0434 | **0.0000** |
| case14 | 0.3668 | 0.3668 | **0.0000** |
| wcci2022 | 0.2352 | 0.2358 | **+0.0006** |

Fourteen rules OR-ed together fire on essentially every frame, so the gate blocks every
contingency the model called secure and its precision falls to the base rate. The single
best rule scores **6.4x** the union on neurips2020 and **2.0x** on wcci2022.

**This inverts the natural reading of the 0.16% extraction yield.** A thin corpus has
been treated throughout this document as a limitation to be explained away. Measured
against a deliberately broad hand-written set, the thinness is **load-bearing**: the four
rules that survived are close to the best obtainable from these variables, and the
rules that did not survive would have made the gate worse by diluting it. Precision is
the scarce resource in a disjunctive gate, and every additional imprecise rule spends it.

### 29.4 §24 is now answered, from both directions

| arm | question | answer |
|---|---|---|
| §28 (ceiling tree) | could a model *fitted on the answer* beat the corpus? | **No** — same recall at a fifth of the precision on the home grid |
| §29 (expert rules) | could a *human writing from doctrine* beat the corpus? | **No** — the best rule they write IS the corpus's rule, scoring identically |

**Reading (a) of §24.1 — the task is genuinely rule-poor — is established from two
independent directions.** §5, §10.4, §15 and §20.2 all argued it from inside the
pipeline that produced the corpus. §28 and §29 are the external checks they lacked.

⚠️ **Neither arm makes the shield good.** Recall is 0.162 / 0.005 / 0.309 and on case14
no frame-level gate of any provenance — extracted, fitted, or hand-written — beats
blocking everything. The licensed claim is *"the rules that exist are close to the best
obtainable from these variables"*, never *"the rules are sufficient"*.

### 29.5 Fifth independent route to the convergence result

§28.5 recorded four unrelated procedures landing on `loading_pct` and the voltage band.
This is the fifth: **hand-written doctrine**. The full list, all converging on
`loading_pct > 100`:

1. An LLM reading published standards (Component B — 10 clauses, 4 documents, 2 bodies)
2. An empirical fire-rate filter over grid data, no LLM (§15.8)
3. A textual auditor over source standards, no grid data (§15.8)
4. A depth-4 tree fitted directly on the answer (§28.5 — 61% of its importance)
5. Hand-written expert rules (this section — the best of 14 on two grids of three)
6. The L2RPN challenge's top teams, independently (`results_comparisons.md` §B1)

### 29.6 Caveats

- ⚠️ **Not blind.** §29.1. This is the binding limitation and it must travel with the
  result.
- ⚠️ **`X_010` never evaluated.** `abs(generation_load_imbalance_pct) > 5` returns
  NOT_EVALUABLE on every frame of every grid: the shield's evaluator runs with no
  builtins, so `abs` raises `NameError`. It was **reported rather than repaired** —
  fixing it would mean editing a pre-registered file. 13 of 14 rules were effective.
  This is a real constraint the extracted corpus operates under too, and worth knowing:
  a natural expert formulation is silently unevaluable in the gate.
- The target depends on the model's predictions at threshold 0.8849 and inherits
  §27.3.3's cross-topology error bar. The neurips2020 and wcci2022 identities
  (expert-best == shield, to four decimals) are exact rather than approximate, because
  both arms evaluate the same predicate over the same rows.
- Ranking rules by F1 *after* seeing the grid is an oracle selection. The "expert best
  single" column is therefore optimistic for the expert arm — and it still only ties
  the shield rather than beating it.

### 29.7 Correction to §28.2 — gate semantics

§28 scored the tree over **all** rows while the shield can only ever act on rows the
model called secure, which charged the tree for false alarms no gate would raise. The
corrected figures (`tree_held_gate` in the artifact) are in §29.2's table and are better
than those in §28.2: **0.2508 / 0.0179 / 0.0025** against the originally reported
0.1755 / 0.0154 / 0.0025.

§28.2's conclusion is unchanged but its supporting sentence needs restating. The claim
was that the tree's oracle equals the all-positive baseline on both unseen grids. Under
gate semantics the right baseline is **flag-all-eligible**, and against it the finding is
sharper, not weaker:

| grid | tree oracle (gate) | flag-all-eligible | difference |
|---|---:|---:|---:|
| neurips2020 | 0.2667 | 0.0434 | +0.2233 |
| case14 | 0.3669 | 0.3668 | **+0.0001** |
| wcci2022 | 0.2352 | 0.2352 | **0.0000** |

**On both unseen grids the fitted tree's best achievable score is the flag-everything
baseline to within one ten-thousandth.** It carries real signal on the grid it was fitted
on (+0.22) and none at all on either grid it was not.

---

## 30. The ceiling by ENUMERATION — §28's instrument replaced, and §28's headline corrected

**Run 2026-09-19.** §28 answered §24.1 with a depth-4 decision tree fitted on the
answer key. This section removes the search altogether and enumerates the rule
language instead. It **strengthens §28's load-bearing claim and falsifies §28's
headline sentence**, and both matter.

Code `evaluation/exhaustive_rules_n1.py`, artifact
`results/ceiling/exhaustive_rules.json`.

### 30.1 Why §28 needed replacing

§28's argument had a hole that §28.3 did not close. CART is **greedy**: it takes the
single best split, then the best split beneath it. A rule that only works as a *pair*
— where neither half is useful alone but the conjunction is — is structurally
invisible to it. §28.3 closed the "too shallow" objection from the other direction
(depth 8 and class-balancing are both *worse*), but nothing closed "too greedy".

So what §28 actually licensed was: *greedy trees of depth ≤ 8 over these variables do
not beat the corpus.* That is a much narrower claim than the one §28.4 stated, and it
is the kind of gap a viva finds.

### 30.2 What was run

Three arms, all under §28's protocol unchanged — same target (*the model predicted
secure AND the contingency was a violation*), same gate semantics, same held
threshold 0.8849 re-derived on the val split, eval batch size 64.

| arm | what it enumerates | per grid |
|---|---|---|
| **1. every single-variable rule** | every **distinct cutpoint present in the data**, both directions, all 14 variables — not a sampled grid | exhaustive |
| **2. every two-variable rule** | all pairs of masks on a 32-quantile grid, both directions per side, joined by **AND** and by **OR** | 252,405–299,925 pairs × 2 operators |
| **3. an uncapped ensemble** | `HistGradientBoostingClassifier`, **no depth limit**, 300 iterations, on the same 14 variables | — |

Arm 3 is deliberately **not a rule**. It is unreadable, so no gate could ship it.
That is the point: it bounds the corpus from above by something strictly *more
expressive* than anything in the rule language.

**Why this is cheap enough to do exhaustively.** Rules read **per-frame** variables
while the target is **per contingency**, so a rule is a mask over 1,932–6,000 frames
rather than over 113,205–742,472 contingencies. Aggregating each frame's target and
eligible counts once reduces scoring any rule to two dot products, and
`F1 = 2·tp / (predicted + total)` follows exactly. Arm 1 then needs one sort and two
cumulative sums per variable to score *every* threshold at once; arm 2 needs two
matrix products to score *every* pair at once. The winning rule on each grid is
re-scored the slow way through `prf` over the full contingency vectors and the script
aborts if the two disagree by more than 1e-9.

⚠️ **Arms 1 and 2 select the winner WITH the answer key, on the grid being scored.**
That is an oracle selection and it is intentional — this is an upper bound, not a
deployable gate. It is deliberately generous to the challenger.

### 30.3 The result

| grid | flag-all-eligible *(no rule)* | **best rule in the language** | best 2-term rule | uncapped ensemble (held / oracle) | **extracted shield** |
|---|---:|---:|---:|---:|---:|
| neurips2020 | 0.0434 | **0.3269** | 0.3140 | 0.2771 / 0.3052 | 0.2765 |
| case14 | 0.3668 | *0.3937* | *0.4017* | 0.0052 / *0.3668* | 0.0102 |
| wcci2022 | 0.2352 | **0.5079** | 0.5066 | 0.2911 / 0.3172 | 0.4644 |

The winners:

| grid | the best rule in the entire language |
|---|---|
| neurips2020 | `loading_pct > 96.79` — fires on 2.4% of frames |
| wcci2022 | `loading_pct > 97.21` — fires on 9.0% of frames |
| case14 | `active_power_mw_max > 33.92` — fires on **71.1%** of frames, i.e. degenerate |

🚨 **A better rule than the extracted one exists, and it is the same rule.** On both
grids where a gate means anything, the optimum is `loading_pct` with the threshold
about three points *below* the standards' round 100. The margin is **+0.0504** on
neurips2020 and **+0.0435** on wcci2022.

### 30.4 How it wins — and why "better" needs qualifying

The two rules are not doing the same job at different quality. They are two points on
one curve.

| grid | arm | precision | recall | flagged to find them |
|---|---|---:|---:|---:|
| neurips2020 | best rule (`> 96.79`) | 0.566 | 0.230 | 828 |
| neurips2020 | **extracted shield (`> 100`)** | **0.938** | 0.162 | **353** |
| wcci2022 | best rule (`> 97.21`) | 0.744 | 0.385 | 35,293 |
| wcci2022 | **extracted shield (`> 100`)** | **0.934** | 0.309 | 22,559 |

The enumerated optimum buys recall by spending precision. On neurips2020 it raises
**nearly half its alarms falsely** (precision 0.566) against the served corpus's 6%.
F1 weights precision and recall equally and therefore prefers the looser cutoff; a
gate that *overrides a model's output* plausibly should not.

**The defensible statement is therefore narrow and should be quoted in these words:**
*the standards' threshold of 100% is slightly conservative for this task, and the cost
of that conservatism is about 0.045 F1, bought back as a 0.37 gain in precision.* Not
"the corpus was beaten by a better rule" — the predicate is identical and only the
constant differs.

### 30.5 🚨 Correction to §28.2 and §28.4

Two statements in §28 are **wrong as written** and are superseded here:

| §28 said | corrected |
|---|---|
| "the fitted ceiling does not beat the extracted rules" (§28.2 heading) | The *greedy* ceiling does not. The **enumerated** ceiling does, by 0.04–0.05 F1, at much lower precision. |
| "the shield is *above* the fitted ceiling … on F1 on two grids of three" (§28.4) | The shield is **below** the language's oracle optimum on those same two grids, and above it on precision everywhere. |

§28's *load-bearing* claim — **extraction was not the bottleneck** — survives, and is
better supported than it was, for the reasons in §30.6. The gap that exists is a
constant on the rule the pipeline already found, not a rule the pipeline missed. No
variable, no threshold and no pair recovered anything the corpus does not already
name.

### 30.6 What got stronger — three results, and the third is the strongest

**1. The greedy objection is dead.** Across ~600,000 two-term rules per grid, **pairs
buy nothing**: neurips2020's best pair (0.3140) is *worse* than its best single
(0.3269), and wcci2022's best pair (0.5066 vs 0.5079) is
`loading_pct > 66.47 and loading_pct > 97.15` — the single rule wearing a hat, an
artifact of the coarser quantile grid arm 2 uses. There is no interaction a tree
missed, because there is no interaction.

**2. Unlimited capacity loses to one threshold comparison.** The uncapped ensemble,
fitted directly on the answer, scores **below a single `loading_pct >` rule on all
three grids** (0.3052 vs 0.3269; 0.3668 vs 0.3937; 0.3172 vs 0.5079 at oracle). A
model that may split as deep as it likes, is not constrained to be readable, and
could never be shipped as a rule, cannot extract more from these 14 variables than one
comparison does. This is §24.1's reading (a) established **from above**.

**3. The convergence result is now overwhelming rather than suggestive.** §28.5
recorded that the tree's importance concentrated on `loading_pct`. Enumeration is
stronger evidence than importance:

| grid | best variable | runner-up | ratio |
|---|---|---|---:|
| neurips2020 | `loading_pct` 0.3269 | `current_a_max` 0.1455 | **2.2×** |
| wcci2022 | `loading_pct` 0.5079 | `voltage_pu_min` 0.3007 | **1.7×** |

Out of *every* single-variable rule expressible in this namespace, the best is the
thermal check and the runner-up is not close. A clean internal check that the search
works: it independently returned `loading_pct > 96.7897` and `rho_max > 0.967897` with
**identical** F1 — the same predicate found twice, in the two variables that encode
it.

### 30.7 case14 is confirmed dead, by exhaustion

Every previous arm found nothing on case14; this one proves there was nothing to find.
The best single rule fires on **71.1% of frames** for F1 0.3937 against 0.3668 for
flagging everything, and the best pair (0.4017) fires on 63.7%. Both are degenerate in
the §29.2 sense — they approximate "block everything the model called secure" and are
scored as such.

**No rule, of any provenance, at any threshold, in any pairwise combination, works on
case14.** Extracted, fitted, hand-written and now exhaustively enumerated have all
returned the same answer. This is the strongest form of the finding and it should
replace any softer statement about case14 elsewhere in this document.

### 30.8 What this does to §29 — OPEN, not resolved

§29.2 states that *the best hand-written rule **is** the extracted rule, and it scores
identically*. That remains true of the two arms it compared — the expert wrote
`loading_pct > 100` and so did the pipeline — but it is no longer the whole picture:
**both wrote 100, and the optimum is ~97.** The shared error is itself informative
(two independent procedures inherited the standards' round number), but §29's framing
does not currently say so.

⚠️ **§29 is deliberately left unrevised.** Its binding limitation (§29.1 — not a blind
baseline) is still under consideration, and how §29 should be framed depends on that
decision. **Nothing in §29 should be quoted as final until it is reconciled with this
section.** What §30 does settle is that §29's question is no longer load-bearing:
enumeration answers *"could anyone have written a better rule?"*, which strictly
dominates *"could a human have?"*.

### 30.9 Caveats

- ⚠️ **The oracle selection.** Arms 1 and 2 choose the winner using the answer key on
  the grid being scored. The extracted rule has no tunable parameter at all. The
  comparison is therefore generous to the challenger by construction, which is what
  makes "and it only wins by 0.045, on precision it loses badly" meaningful.
- ⚠️ **Arm 2 quantises.** Arm 1 uses every distinct cutpoint; arm 2 uses 32 quantiles
  per variable, because the pair space squares. A pair needing a threshold between two
  quantiles could in principle be missed. Given that the best pair is *worse* than the
  best single on two grids of three, and is a restatement of it on the third, this is
  unlikely to be hiding anything.
- ⚠️ **Three-term rules were not enumerated.** The space cubes. The evidence against
  them is indirect but consistent: two terms already add nothing over one.
- The per-frame constraint of §24.2 binds here exactly as it binds the tree and every
  extracted rule — no rule in this language can separate two contingencies inside one
  frame.
- The target depends on the model's predictions at 0.8849 and inherits §27.3.3's
  cross-topology error bar (±0.02 at least). The neurips2020 margin of 0.0504 is
  comparable to that bar and should be reported as *"about 0.05"*, not to four
  decimals. The wcci2022 margin and the precision gaps are far outside it.
- Alignment is proven, not assumed: the arm reproduces 113,205 / 118,502 / 742,472
  contingencies and all-positive F1 0.3104 / 0.4345 / 0.3969 on the three grids, and
  aborts if either drifts.
