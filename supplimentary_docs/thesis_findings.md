# Findings — Neuro-Symbolic Fault Screening for Power Grids

**Status as of 2026-08-19.** A plain-language account of what was built, what was measured, and
what it means. Written to be readable without the codebase open.

**Confirmed by replication:** every shield number below was re-measured against a second,
independently generated rule corpus and came back **identical to the digit** (plan §14.4). The
result does not depend on which extraction run produced the rules.

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
| NeurIPS 2020 *(trained on)* | 36 buses, 59 lines | 0.8956 | **0.8982** | +0.0026 |
| case14 *(unseen, smaller)* | 14 buses, 20 lines | 0.4167 | **0.4188** | +0.0021 |
| WCCI 2022 *(unseen, larger)* | 118 buses, 186 lines | 0.5577 | **0.6232** | **+0.0655** |

*(F1 score. Full table, including baselines and false-alarm counts: plan §11.1.)*

On the large unseen grid the gate is worth a great deal more than a rounding error:

- **Dangerous errors — cases where the model wrongly declared a contingency safe — fell from
  68,166 to 46,860. A 31% reduction.**
- The cost was about 3,500 extra false alarms, a 3% increase. In grid operations a missed fault
  is far more expensive than a false alarm, so that is a favourable trade.
- The gated score, **0.6232**, is higher than the raw model reaches on that grid *even when
  allowed to tune its threshold against the answers* (0.5721). A layer bolted on after training,
  changing nothing inside the model, pushed an unseen-topology result past the model's own
  oracle ceiling.

### 2.1 Why this is the interesting result and not just a number

The gate can only intervene in a narrow situation: the grid is **already** past a limit, and the
model still called a contingency safe. How often the gate is *right* when it intervenes is the
diagnostic:

| grid | how often it can intervene | **when it does, how often it is right** |
|---|---:|---:|
| NeurIPS 2020 *(trained on)* | 0.73% of cases | **51.4%** |
| case14 *(unseen)* | 0.087% | **92.2%** |
| WCCI 2022 *(unseen)* | 3.34% | **86.0%** |

Read that as a senior engineer looking over a junior's shoulder.

**On familiar ground, when the junior disagrees with the rulebook, the junior is right about half
the time.** They are seeing something real that the crude rule misses — overruling them is a coin
flip.

**On unfamiliar ground the junior's disagreements stop being insightful.** They are simply wrong,
86–92% of the time. There, the rulebook should win.

The sharpest single number: on the training grid, **95.7%** of contingencies sitting on an
already-overloaded base case are genuine violations — but among the ones *this model* calls safe,
only **51.4%** are. That 44-point gap **is** the model's contribution, quantified. And it is
exactly what disappears on foreign grids.

So the finding is not "rules help." It is:

> **The symbolic layer does not get smarter off-distribution. The neural layer stops earning the
> benefit of the doubt.**

That is precisely the case for keeping a symbolic layer in the loop at all, and it is now
measured rather than asserted.

---

## 3. Why the effect is dramatic on one grid and invisible on two

Not a contradiction — the three grids sit in different regimes.

**Training grid:** the model is genuinely good (0.90 F1), so there were only 2,041 dangerous
errors to begin with. Cutting 20% of a small number barely moves an aggregate score. The gate is
working; there is little left for it to work on.

**case14:** the model is broken here — it fails against a single-threshold baseline outright
(plan §7.6) and floods the output with false alarms. Because it declares almost everything
dangerous, it almost never hands the gate a "this is safe" claim to veto. Reach collapses to
0.087%. Note the gate is *nearly always right* in that sliver (92.2%); there is just no sliver.

**WCCI 2022:** the sweet spot. The model fails often **and** fails on states the rules can see.
3.34% reach, 86% precision, 31% of the dangerous errors removed.

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
| **rules that actually fire at inference** | **10** | **10** |
| — all of them saying the same thing (`is the line over 100% loaded?`) | ✔ | ✔ |

*(plan §4.2, §14.)* The symbolic layer distils to **one** physically meaningful predicate, and
that survives unchanged across two independent extraction runs whose totals differ by 29%. The
second run also produced a **cleaner** stream — the polarity check rejected 44.8% of it rather
than 59.8% — after a prompt fix described in plan §13.

This is why the shield result in §2 is identical on both corpora: everything except that one
predicate is silent at inference.

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
| NeurIPS 2020 | 2,041 | 427 (**20.9%**) |
| case14 | 18,592 | 95 (**0.51%**) |
| WCCI 2022 | 68,166 | 21,306 (**31.3%**) |

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

There is also a smaller but instructive one: **a pipeline failure that looked exactly like a
finding.** A stage in the extraction pipeline reported that only 6 of 2,463 rules were
expressible — a dramatic negative result about standards. It was a parsing bug: 89% of the corpus
was never assessed at all. A void run and a genuinely inexpressible corpus produce identical
summary statistics. It is now guarded by tests and a tripwire. *(plan §3.5.)*

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
- **190 tests, all passing**

**Remaining, and mechanical:** one more translation pass carrying a prompt fix made today, the
validation stage on the research PC, then re-running the same three evaluation commands and
diffing the tables. The expectation is that the numbers shift slightly and the shape holds, since
the one working rule is not in question. Sequence: plan §13.1.

---

## 8. Where the evidence lives

| claim | evidence |
|---|---|
| shield results, all three grids | plan §11 · `shield_<tag>.json` · `failures_<tag>.jsonl` |
| the result replicates across extraction runs | plan §14.4 · `shield_<tag>_run3.json` |
| the rule is not circular (91–96%, not 100%) | plan §11.4 |
| structural ceiling on what rules can reach | plan §11.3 |
| extraction yield and the audit | plan §4.2 · `evaluation/audit_rules.py` |
| standards/simulator mismatch, both directions | plan §4.2, §12 |
| the 4-class task was closed-form | plan §2.1 |
| forecast tasks rejected | plan §1.1 |
| raw model cross-topology performance | plan §7.6 · `gnn_n1_tightening.md` |
| voltage base-kV contract and why it matters | plan §5, §5.1 |

**Reproducing §2:** the evaluation harness is `evaluation/eval_shield_n1.py`. To reproduce the
numbers in this document exactly, point it at the 33-rule guarded corpus:

```powershell
.venv\Scripts\python.exe evaluation\eval_shield_n1.py --tag wcci2022 --rules translated_rules\guarded\
```

⚠ Two threshold protocols are in circulation and must not be mixed. This document and plan §11
use a threshold **selected on a validation split and held fixed across grids** — the honest
protocol. Plan §7.6 reports the raw model at the **best threshold on each grid individually**,
which is optimistic and is tracked as an open item. The two therefore give slightly different
figures for the same model on the same grid (0.8956 vs 0.8972 on the training grid); this is
expected, not a discrepancy.
