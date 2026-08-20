# Findings — Neuro-Symbolic Fault Screening for Power Grids

**Status as of 2026-08-20 — all four components are built and measured.** A plain-language account
of what was built, what was measured, and what it means. Written to be readable without the
codebase open.

**Confirmed by replication:** the shield numbers were re-measured against a second, independently
generated rule corpus and came back **identical to the digit** (plan §14.4). The result does not
depend on which extraction run produced the rules.

⚠ **Corrected 2026-08-20.** An earlier version of this document said the gate's override
precision *rises* off-distribution (51% at home, 86–92% away). That was measured to be false. It
was an artifact of averaging two rule families; once the misextracted ones were removed the
precision is **flat at 92–94% on all three grids**. §2.1 carries the corrected reading, and it is
a better result than the one it replaces. Derivation: plan §15.7.

This is the *narrative* record. The operational detail, runbooks and landmines live in
[`component_d_plan.md`](component_d_plan.md); every number below cites the section there that
carries its derivation. Where the two disagree, the plan doc is authoritative.

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
32-rule corpus for comparison: plan §15.6.)*

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
(plan §7.6) and floods the output with false alarms. Because it declares almost everything
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

*(plan §4.2, §14, §15.)* End to end: **2,463 candidate rules become 4.** The symbolic layer
distils to **one** physically meaningful predicate, and that survives unchanged across two
independent extraction runs whose totals differ by 29%. The second run also produced a
**cleaner** stream — the polarity check rejected 44.8% of it rather than 59.8% — after a prompt
fix described in plan §13.

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
four ordered threshold rules reproduce them with **100% agreement on 55,000 records**, across two
grids. A symbolic layer scored 100% where the trained network reached 0.83 — meaning "does the
symbolic layer add value over the network?" was rigged before it ran. *(plan §2.1.)*

**Forecasting faults was unlearnable.** Line-trip onset in the simulator is an unconditional coin
flip by construction, so the best possible rule scores **1.00× the trivial all-positive baseline**
at every horizon of three steps or more. No model can beat "always say at-risk". *(plan §1.1.)*

**Forecasting overloads was learnable but exhausted by one threshold.** A gradient-boosting model
on the full feature vector reached F1 0.163 against a single rho threshold's 0.160 — with *worse*
average precision. *(plan §1.1.)*

**N-1 screening is neither.** It cannot be restated by a rule, because producing the label
requires a power-flow solve. That is what permanently un-rigs the comparison, and it is why the
result in §2 means something.

There is also a pair of instructive ones, both cases of **a pipeline failure that looked exactly
like a finding.**

*First:* a stage in the extraction pipeline reported that only 6 of 2,463 rules were expressible
— a dramatic negative result about standards. It was a parsing bug: 89% of the corpus was never
assessed at all. A void run and a genuinely inexpressible corpus produce identical summary
statistics. It is now guarded by tests and a tripwire. *(plan §3.5.)*

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
away. *(plan §15.1–§15.3.)*

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
standards regulate and what the simulator models, not an artifact of either filter.** *(plan
§15.8.)*

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
Neither changes the shape of the result. Sequence: plan §16.8.

---

## 8. Where the evidence lives

| claim | evidence |
|---|---|
| shield results, all three grids | plan §15.6 · `results/shield/shield_<tag>_validated.json` · `results/failures/failures_<tag>.jsonl` |
| the result replicates across extraction runs | plan §14.4 · `shield_<tag>_run3.json` |
| validation A/B, and the fabricated reasoning | plan §15.1–§15.4 · `validated_{strict,translated}/*_rejected.jsonl` |
| filtering the corpus improved the gate | plan §15.6 (both corpora, same table) |
| the "precision rises off-distribution" correction | plan §15.7 |
| two independent filters converge | plan §15.8 |
| the knowledge graph, and the v1 that was deleted | plan §16 · `kg/knowledge_graph.json` |
| 10 clauses / 4 documents / 2 bodies behind one rule | plan §16.3 · `kg/kg_corroboration.svg` |
| the graph changes no measured number | plan §16.4 · `shield_<tag>_kg.json` · `tests/test_kg.py` |
| the rule is not circular (91–96%, not 100%) | plan §11.4 |
| structural ceiling on what rules can reach | plan §11.3, §15.6 |
| extraction yield and the audit | plan §4.2 · `evaluation/audit_rules.py` |
| standards/simulator mismatch, both directions | plan §4.2, §12 |
| the 4-class task was closed-form | plan §2.1 |
| forecast tasks rejected | plan §1.1 |
| raw model cross-topology performance | plan §7.6 · `gnn_n1_tightening.md` |
| voltage base-kV contract and why it matters | plan §5, §5.1 |

**Reproducing §2:** the evaluation harness is `evaluation/eval_shield_n1.py`. To reproduce the
numbers in this document exactly, point it at the validated four-rule corpus:

```powershell
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag wcci2022 --rules validated_translated\all_rules_deduped.jsonl
```

⚠ Pointing it at `translated_rules\guarded\` instead reproduces the *earlier* 32-rule numbers
(+0.0655 on WCCI, 51.4% precision at home). Those are superseded — see §2.2 — but kept in plan
§15.6 alongside the new ones, because the difference between the two corpora is the evidence for
§2.2.

⚠ Two threshold protocols are in circulation and must not be mixed. This document and plan §11
use a threshold **selected on a validation split and held fixed across grids** — the honest
protocol. Plan §7.6 reports the raw model at the **best threshold on each grid individually**,
which is optimistic and is tracked as an open item. The two therefore give slightly different
figures for the same model on the same grid (0.8956 vs 0.8972 on the training grid); this is
expected, not a discrepancy.
