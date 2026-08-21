# Methodology & Implementation Guide

> Everything here is decided. This is not a discussion document — it is a reference for how we are building the system, why we made each choice, and how to collect the data we need.

---

## ⚠️ REVISION — what changed between the agreed design and what was built (2026-08-20)

**All four components are now built and measured.** Several things in this document were
superseded by measurement rather than by preference. They are listed here in one place so a reader
who knew the old design is not misled, and each is corrected in its own section below.

| # | Agreed here | What was actually built | Why |
|---|---|---|---|
| 1 | GNN does **4-class fault classification** + bus localization | GNN does **N-1 contingency screening** — one binary logit per line | The 4-class label was a **closed-form function of the observation**: four `if`/`else` rules reproduce it with **100% agreement, 0 disagreements in 315,000 records**. The task could not distinguish a model from a threshold. §2 |
| 2 | Graph-level readout via 3-pool concat | **No pooling at all** — a logit read off each line's own edge | 96–99% of frames are *mixed*: some contingencies violate, others do not, in the same grid state. Pooling erases exactly that distinction. §2 |
| 3 | Two-stage extraction (Extract → Validate) | **Four stages**: extract → translate → polarity guard → validate | Raw extracted conditions are in the standard's language; they must be translated onto simulator-observable variables before anything can evaluate them. §3 |
| 4 | KG stores **rules + grid topology**, shield queries it by entity | KG stores **provenance only** (document → clause → rule → predicate). Retrieval through it is **opt-in and returns an identical rule set** | A topology-keyed graph was built first and failed twice over: `Line`+`Bus` were 3.5% of the corpus so 96% of its edges carried nothing, and it was welded to one grid while the shield runs on three. §4 |
| 5 | Shield does **PASS / CORRECT / BLOCK** over the GNN's **top-k** candidates | Shield is an **asymmetric binary gate**: it blocks an over-permissive `secure` verdict and never blocks a cautious one. No top-k, no CORRECT | Top-k selection presupposes a multi-class label. N-1 screening is binary per line, so there is no candidate set to select over. §5 |
| 6 | Cross-topology on IEEE **14 / 57 / 118-bus** | **case14 (14-bus), neurips2020 (36-bus, trained), wcci2022 (118-bus)** | 57-bus was never generated. The three used are the Grid2Op environments that ship with usable chronics. §6 |
| 7 | **KCL residual** as a physics-tier metric | **Not implemented.** No physics-invariant check exists | The shield's constraint checks are thermal and voltage limits from the corpus. KCL was never needed and never written; it should not appear in the report as a delivered metric. §11 |
| 8 | Normalization persisted to `normalization_stats.pt` | Stats are **recomputed inline** from the 36-bus training split at every eval | The *principle* here survives untouched — foreign topologies are still normalized with 36-bus stats, never their own. Only the persistence mechanism changed; the file was deleted because nothing read it. §2 |
| 9 | §8 end-to-end **toy scenario testing** on `rte_case5_example` | **Never built.** Replaced by 222 automated tests plus three full-topology evaluation runs | The toy harness was a pre-integration confidence check; the real harness overtook it. §8 |
| 10 | LangChain for document loading; W&B for tracking | **Neither is used.** `pdfplumber` for PDFs; metrics printed and written to JSON | Both appear only in a stack-verification script. §9 |

**The core claim (§0) survives, with one correction.** An earlier reading — that the shield's
override precision *rises* off-distribution — was measured to be **false**; it was an artifact of
averaging two rule families. Corrected finding, and it is stronger: **the gate's precision is
topology-invariant (93.8% / 92.2% / 93.4%) while the model's collapses (0.90 → 0.42 → 0.56).**
See §0 and §11.

---

## 0. Core Claim

**This is not a full-loop autonomous system as an end in itself. The full loop is the instrument; the claim it exists to prove is empirical.**

A GNN trained on one grid topology is a statistical approximator, not a guarantee. It generalizes imperfectly to topologies it has never seen — this is expected, not a flaw to eliminate. In a safety-critical system, an ungated GNN acting on a degraded, out-of-distribution prediction can recommend or trigger a physically invalid or destabilizing action. The claim under test:

> **A GNN alone is not general enough to be trusted across grid topologies. The symbolic shield is a necessary safety layer — it prevents the GNN from acting on unhinged, physically invalid predictions when generalization fails, rather than assuming the GNN's confidence is reliable.**

Everything in this document — the GNN (§2), the LLM-derived rule base (§3–4), and the shield (§5) — is built to generate the evidence for this claim, primarily via the cross-topology evaluation in §11: measuring how badly the frozen GNN degrades on unseen topologies (14/57/118-bus), and how much of that degradation the shield catches, corrects, or blocks before it reaches the grid. The three-tier table in §11 (Trained / Unseen-No-Shield / Unseen-With-Shield) is the direct empirical evidence for this claim, not a supplementary metric.

**Consequence for every other section:** the GNN's job is detection, classification, and localization — never a final, unchecked action recommendation (see §12). Any wording elsewhere implying the GNN outputs an autonomous control decision is legacy phrasing from the pre-pivot framing and should be read as superseded by this section.

### The claim, now measured (2026-08-20)

The claim held. The mechanism is not the one anticipated here, and the corrected version is
stronger.

| grid | model alone | + symbolic gate | when the gate acts, it is right |
|---|---:|---:|---:|
| 36-bus *(trained on)* | 0.8956 | **0.9038** | **93.8%** |
| 14-bus *(unseen)* | 0.4167 | **0.4188** | **92.2%** |
| 118-bus *(unseen)* | 0.5577 | **0.6253** | **93.4%** |

*(F1, threshold selected on a 36-bus validation split and held fixed across all three.)*

The model degrades exactly as predicted — 0.90 → 0.42 → 0.56 — and on the 14-bus grid it loses
outright to a single-threshold baseline. What was **not** anticipated:

> **The gate's accuracy does not vary with topology.** It is right ~93% of the time on the grid
> the model knows and ~93% of the time on grids it has never seen, because it enforces a physical
> principle rather than a learned pattern. What changes off-distribution is not how *right* the
> gate is, but how *often it gets to speak* — 0.31% → 0.087% → 3.04% of contingencies — and how
> much of the model's error sits where a present-state rule can reach it: 16.2% / 0.51% / 30.9%.

That is a cleaner statement of "the symbolic layer is the necessary safety layer" than the
degradation-and-recovery framing this document was written around: the symbolic layer is not
*recovering* the model's losses, it is *unaffected* by whatever caused them.

⚠️ Two limits belong with this claim wherever it appears. The surviving rule (`loading_pct > 100`
on the base case) is close to a physical tautology — P(violation | base already overloaded) is
91–96%, informative but not a discovery — and the N-1 doctrine the gate enforces is hand-written
in `shield/shield.py::validate_n1`, not extracted. The LLM pipeline supplied the threshold.

---

## Table of Contents

0. [Core Claim](#0-core-claim)
1. [System Architecture](#1-system-architecture)
2. [Component A — Graph Neural Network (Neural Layer)](#2-component-a--graph-neural-network-neural-layer)
3. [Component B — LLM Knowledge Extraction Pipeline](#3-component-b--llm-knowledge-extraction-pipeline)
4. [Component C — Knowledge Graph (Symbolic Rule Store)](#4-component-c--knowledge-graph-symbolic-rule-store)
5. [Component D — Symbolic Validation Shield](#5-component-d--symbolic-validation-shield)
6. [Data Collection Strategy](#6-data-collection-strategy)
7. [GNN Training Pipeline](#7-gnn-training-pipeline)
8. [End-to-End Testing](#8-end-to-end-testing)
9. [Technology Stack](#9-technology-stack)
10. [Hardware & Compute Allocation](#10-hardware--compute-allocation)
11. [Evaluation Plan](#11-evaluation-plan)
12. [Out of Scope](#12-out-of-scope)
13. [Glossary](#13-glossary)
14. [Disclaimer — Living Document](#14-disclaimer--living-document)

---


## 1. System Architecture

### The Full Pipeline

```
[16 standards PDFs]                       [Grid2Op — 3 topologies]
 IEEE / NERC / FERC /                     l2rpn_neurips_2020_track1_small (36-bus, TRAIN)
 ENTSO-E / national grid codes            l2rpn_case14_sandbox            (14-bus, test)
        │                                 l2rpn_wcci_2022                 (118-bus, test)
        ▼                                           │
[Component B — 4-stage extraction]                  ▼
 1  extract.py    qwen3.6:35b-a3b, open vocab   [N-1 dataset generation]
    2,463 candidates                             per frame, per line:
 2  translate.py  → 14 observable variables       "if line k trips now,
    58 expressible                                 is a limit violated?"
 3  polarity_guard.py (deterministic)             12k / 6k / 4k frames
    32 kept — drops rules that fire on a          702k / 119k / 742k labels
    healthy grid                                          │
 4  validate.py   nemotron-3-nano:30b                     ▼
    11 confirmed → 4 distinct                 [Component A — GNN, 36-bus ONLY]
        │                                      GATv2 ×3, NO pooling,
        ▼                                      one logit per LINE off its own edge
[Component C — provenance KG]                  weights frozen after training
 Document → Clause → Rule                              │
   → ServedRule → Predicate → Variable                 │
 36 nodes, 40 edges, topology-agnostic     ┌───────────┴───────────┐
 Retrieval is OPT-IN and returns the       ▼                       ▼
 identical rule set (test-pinned)    [36-bus, in-dist]     [14/118-bus, frozen]
        │                                  │                       │
        └──────────────────┬───────────────┴───────────┬───────────┘
                           ▼                           ▼
        [Component D — THE SHIELD, identical logic on all three paths]
        ASYMMETRIC. For each contingency the model calls `secure`, check the
        BASE CASE against the rule corpus:
            predicted `secure`    + base case violates a CONSTRAINT  → BLOCK
            predicted `violation` + anything                          → PASS
        A false alarm fails toward caution and is never suppressed. Declaring a
        grid secure while it is already outside its limits is the failure worth
        blocking. AFFIRMATION rules supply evidence and never block alone.
                           │
                           ▼
                  [Final output]
        Per-line secure/violation verdict + citation: which rule fired, which
        clause of which standard states it, and how many other standards agree.
```
### The Non-Negotiable Principle

**GNN outputs are never final without symbolic validation.** The GNN is the perception engine — fault detection, classification, and localization only. The shield is the safety officer. They are always both active, on both the 36-bus and cross-topology inference paths. There is no mode where the GNN output bypasses the shield, and no mode where the GNN proposes a corrective action (see §12).

### The Decoupling Invariant

This is the structural property the novelty claim rests on, stated explicitly so it isn't lost as components change:

- The GNN has no knowledge of the rules. It does not see the KG, and its weights are never affected by the KG.
- The shield has no knowledge of GNN internals. It only sees `(predicted_label, top_buses, confidence)` plus raw observation context — never logits, weights, or architecture.
- Consequence: the GNN can be retrained, swapped, or upgraded (e.g. GATConv → GATv2Conv) without touching the shield. Rules can be added or removed without retraining the GNN.
- This decoupling is what separates this system from prior work that either fuses LLM-derived knowledge into training (loss-based) or never closes the loop into a runtime gate at all. It is verified, not assumed — see the interaction contract in the integration document (`study3(integration).md`, §4.7).

---

## 2. Component A — Graph Neural Network (Neural Layer)

### What It Does

⚠️ **CHANGED (revision item 1).** Takes a snapshot of the grid state as a graph and answers, **for
every transmission line independently**:

> *if line k were lost right now, would the grid violate a limit?*

One binary logit per line. This is **N-1 contingency screening**, not the 4-class fault
classification originally agreed here.

**Why the original task was abandoned, in one measurement.** The 4-class label
(normal / overload / line_trip / cascade) turned out to be a **closed-form function of the very
observation the model was given** — four ordered threshold rules on `rho_max` and the count of
active lines. Re-verified over **every record of both datasets** before they were deleted:
**315,000 records, 100.000000% agreement, zero disagreements.** The trained GATv2 reached macro F1
0.8277 on the same task. Four threshold comparisons beat it.

The honest reading — which must travel with this claim — is not "symbolic beats neural". It is
that **the benchmark was mis-specified**: the labelling function was defined over the same
observation the model consumed, so the label carried no information the input did not already
contain. Any measurement of "does the symbolic layer add value over the GNN?" on that task was
rigged before it ran, because the symbolic layer *was* the labeller.

N-1 screening is not: **producing the label requires a power-flow solve**, so no rule over the
present observation can restate it. That is what makes the §0 comparison meaningful.

Does not output or recommend a corrective/control action — that is explicitly out of scope (§12). The GNN's job ends at detection; the shield gates what happens to that output.

### Why GNN and Not Something Else

A power grid **is** a graph. Buses are nodes. Lines are edges. Electrical quantities (voltage, current, power) are node and edge features. Every other architecture ignores this:

- **MLP/DNN:** Flattens the grid into a vector. Loses all topology information. A fault on Line 3-4 looks identical whether Bus 3 has 2 neighbors or 10.
- **LSTM:** Models time sequences well but is completely blind to graph structure. Cannot localize a fault spatially.
- **CNN:** Designed for grid-like spatial data (images). Power grids are irregular, sparse graphs — CNN kernels have no meaning here.

A GNN propagates information along edges, so it natively understands that losing Line 3-4 affects
Bus 3 and Bus 4 differently depending on their local topology.

⚠️ **The argument is now stronger than when it was written, for a reason worth stating.** Under N-1
screening the question is *where does line k's flow go when line k is removed* — literally a
question about the neighbourhood structure around k. A flattened representation cannot express it
at all, and the measured ablation confirms message passing is load-bearing: removing it from the
readout costs 0.8987 to 0.8411. The original framing (fault *localization*) is retired with the
localizer head, but the reason for choosing a GNN survives intact.

### Architecture Decision

**GATv2Conv**, 3-layer, `hidden_channels=[16,32,32]`, `heads=[4,4,1]`, dropout 0.0, BatchNorm
(`track_running_stats=False`). Migrated from static GATConv — GATConv's fixed attention ranking
caused rank collapse on 36-node graphs; GATv2Conv fixes this with near-identical API. GCN baseline
dropped.

⚠️ **CHANGED (revision item 2) — there is no pooling.** The agreed 3-pool concat
(mean/max/min → graph vector → classifier) is gone. The readout is **edge-level**: for each line
*k*, the head reads that line's own or→ex edge and emits one logit.

```
[ h_or ‖ h_ex ‖ edge_attr_k ‖ x_or ‖ x_ex ]  →  MLP(→128 →64 →1)
```

**Why pooling had to go.** Measured over every frame of all three datasets (22,000 frames):

| grid | strictly mixed frames | all contingencies violate | none violate |
|---|---:|---:|---:|
| neurips2020 | **98.80%** | 1.20% | **0.00%** |
| case14 | **98.48%** | 1.52% | **0.00%** |
| wcci2022 | **96.43%** | 3.57% | **0.00%** |

Within a single grid state, some contingencies violate and others do not — in 96–99% of frames.
A graph-level vector cannot represent that; pooling erases exactly the per-line distinction the
task is about. *(Note: no frame on any grid is entirely secure — that 0.00% column is the "100%"
figure quoted in earlier drafts. Strictly-mixed is 96–99%, not 100%.)*

Removing pooling also makes the checkpoint **topology-agnostic by construction**: one logit per
line present, so 20-line case14 and 186-line wcci2022 run on the frozen 36-bus weights unchanged,
with no architectural special-casing.

**The raw endpoint features are concatenated alongside the learned embeddings** (the skip in the
readout above). Three rounds of attention, batch norm and ELU are free to turn the endpoint
headroom and flow magnitudes into something unrecognisable — but those quantities *are* the physics
of post-contingency redistribution: line k's flow has to be absorbed by spare capacity at its
endpoints. Handing them to the head unmodified lets it form the flow/spare ratio directly. Ablation:
dropping message passing from the readout costs **0.8987 → 0.8411**.

### Cross-Topology Inference (Frozen Weights)

Weights are trained once on the 36-bus environment and frozen. No fine-tuning on other topologies.
Frozen model is evaluated zero-shot on **case14 (14-bus) and wcci2022 (118-bus)** — see §11.
⚠️ 57-bus was never generated (revision item 6); do not cite it.

**Normalization — decided: reuse 36-bus stats, fixed, unchanged.** `node_mean`/`node_std`/`edge_mean`/`edge_std` are computed once from the 36-bus training split and applied unchanged to 14/57/118-bus observations at inference. Not recomputed per topology — recomputing would launder away the exact distribution shift the cross-topology evaluation is designed to measure, and mirrors real zero-shot deployment (no target-domain stats available in production). Stated explicitly here so degradation isn't misattributed to normalization mismatch.

⚠️ **CHANGED (revision item 8) — the principle above is unchanged; only the mechanism is.** The
stats are **recomputed inline** from the 36-bus training split at the start of every evaluation
(`compute_normalization_stats(home, train_idx)`), not loaded from a persisted file. Foreign
topologies are still normalized with 36-bus stats and never their own, so nothing about the
measurement changes. `normalization_stats.pt` was deleted because nothing read it.

### Library

**PyTorch Geometric (PyG)** — the standard library for GNN research. Direct integration with PyTorch, supports GCN, GAT, GraphSAGE, and all standard message-passing architectures out of the box.

### Input/Output Specification

⚠️ **CHANGED — 8 node and 8 edge features, built for N-1 redistribution rather than for
classification.** The exact builders are `scripts/pyg_data.py`; the shapes are what the deployed
checkpoint expects and are asserted at load time.

| | count | what they carry |
|---|---:|---|
| **Node features** (per bus) | 8 | demand, voltage health, local and global loading, connectivity fraction, and the endpoint **spare-capacity / flow** aggregates the readout skip depends on |
| **Edge features** (per line) | 8 | loading ratio, active and reactive flow, and an at-limit flag |

`gen_p` was dropped from the node features — EDA measured a correlation of **1.00** with `load_p`
(generation matches load by power-flow law), so it was pure duplication.

**Output:** one logit per line — P(losing line k violates a limit). There is no graph-level head
and no per-bus localizer; both belonged to the retired classify model.

### Measured result

Trained on 12,000 frames / 576 chronics / 702,618 contingency labels, chronic-level 70/15/15 split.

Threshold **0.8849, selected once on the neurips2020 validation split and applied unchanged to all
three grids** — the same protocol the shield table uses.

| topology | lines | contingencies | all-positive | best single rule | **model (held)** | *oracle* |
|---|---:|---:|---:|---:|---:|---:|
| neurips2020 *(in-dist, test split)* | 59 | 113,205 | 0.3104 | 0.4639 | **0.8956** — 1.93× | *0.8972* |
| case14 *(unseen, smaller)* | 20 | 118,502 | 0.4345 | 0.5392 | **0.4167** — **0.77×, FAILS** | *0.4477* |
| wcci2022 *(unseen, larger)* | 186 | 742,472 | 0.3969 | 0.4915 | **0.5577** — 1.13× | *0.5721* |

**Generalisation is partial and asymmetric, and the direction is the opposite of the naive
expectation:** scaling *up* (36 → 118 buses) costs far less than scaling *down* (36 → 14). On the
smaller grid the model loses to a single-threshold baseline outright — and to the all-positive
baseline as well (0.4167 vs 0.4345, **0.96×**), so on case14 it is beaten by answering "violation"
unconditionally. Report it as such: it is the degradation the §0 claim predicts, and it is what the
shield is measured against.

⚠ The *oracle* column is best-threshold on each grid individually, selected with the answer key and
therefore unobtainable in deployment. It is kept only because the shield's headline claim is
measured against it. Its optimism grows off-distribution — +0.0015 at home, +0.0310 on case14 —
which is why the held column is the one to quote.

---

## 3. Component B — LLM Knowledge Extraction Pipeline

### What It Does

An LLM reads domain-authoritative documents — IEEE standards, grid operation manuals, safety protocols — and extracts symbolic rules in structured JSON format. These rules are then loaded into the knowledge graph.

This replaces manual rule encoding, which is the core knowledge bottleneck in all existing Neuro-Symbolic systems for power grids.

### Why LLM for This

The alternative is having a domain expert read every document and manually translate prose into code. That is:
- Slow (weeks per document set)
- Error-prone (rules buried in dense technical prose are easy to miss)
- Not scalable (adding a new standard means starting over)

The closest validated precedent is **Chen et al. (2025, 2026)**, who used an LLM to extract rules from USGS geology textbooks into a Knowledge Graph, then used those rules to constrain a Random Forest — achieving 99.06% accuracy vs an 84.3% baseline. We apply the same pipeline to power grid documents.

### Why Local LLM (Not OpenAI API)

- Grid operational documents may be institution-sensitive
- API calls introduce non-determinism across runs (model updates, rate limits)
- Local inference is fully reproducible
- One local machine with a CUDA GPU and Ollama runs both models at 4-bit/5-bit quantization, sequentially. The validator is the larger of the two and relies on partial CPU offload

### Model

**Qwen3.6-35B-A3B (MoE, Q4_K_M/Q5_K_M quantized)** via **Ollama**. This is the Extractor stage of the multi-LLM pipeline. As a Mixture-of-Experts model (35B total parameters, ~3B active per token), it stays GPU-resident at these quant levels on a single consumer card.

**Why Qwen3.6-35B-A3B:**
- Superior technical comprehension and recall on dense IEEE standards prose — MoE routing gives 30B+-class reasoning depth at a fraction of the active-parameter compute cost of a same-size dense model.
- Stays GPU-resident at Q4_K_M/Q5_K_M, allowing fast iteration across large document sets.
- Superseded an earlier Qwen3-14B dense-model choice; retained for comprehension quality, not swapped back.

### Multi-LLM Hybrid Pipeline

A single-model extraction pipeline places competing demands on one model simultaneously: reading comprehension over dense technical prose, strict schema compliance, and precise constraint boundary detection. A two-LLM architecture separates these concerns across models optimized for each role.

Four patterns were evaluated (Extractor+Validator, Dual Extraction+Consensus, Extractor+Formalizer, Self-Consistency). The chosen approach combines Pattern 1 and Pattern 2.

#### Architecture as built — FOUR stages, not two

⚠️ **CHANGED (revision item 3).** The agreed Extract → Validate pair could not work as specified.
A rule extracted from a standard is phrased in the standard's language ("Applicable Facility
Ratings shall not be exceeded"); nothing can *evaluate* that against a simulator observation. A
translation stage had to be inserted, and a deterministic polarity check after it.

```
[Document chunk]  (pdfplumber, 1200 chars, 200 overlap)
        │
        ▼
1  extract.py     qwen3.6:35b-a3b — OPEN vocabulary, high recall
        │         2,463 candidates across 16 documents
        ▼
2  translate.py   qwen3.6:35b-a3b reloaded — rewrite each condition onto the
        │         14 simulator-observable variables, or mark it untranslatable
        │         58 expressible (2.4%)
        ▼
2.5 polarity_guard.py   DETERMINISTIC, no LLM. Evaluate every rule against real
        │         grid frames; drop any that fires on a healthy grid.
        │         32 kept (26 rejected)
        ▼
3  validate.py    nemotron-3-nano:30b — audit each translated rule against the
        │         source chunk it came from
        │         11 confirmed → 4 distinct after dedup
        ▼
[Knowledge graph — Component C]
```

**The closed vocabulary is enforced at stages 2–3 only.** Closing stage 1 against it was tried
twice and returned *zero* rules both times: on frequency- and timing-heavy standards the model
correctly answers `[]` for nearly every chunk, starving the pipeline. Guarded by a test.

**Rules carry a role, decided after inverted-band defects kept appearing:**

| role | meaning | shield behaviour |
|---|---|---|
| `CONSTRAINT` | condition TRUE ⇒ the standard is violated | can BLOCK |
| `AFFIRMATION` | condition TRUE ⇒ telemetry is consistent with the named class | evidence only, never blocks alone |

Forcing every clause into constraint polarity is what mangled healthy-band rules like "voltage
shall remain within 0.95–1.05 pu" — inverting it produced predicates that fire at nominal.

#### End-to-end yield, and why it is a finding rather than a failure

**2,463 → 58 → 32 → 11 → 4 distinct rules (0.16%).** The surviving corpus is three phrasings of
*"is any line loaded past 100% of its thermal rating?"* plus one voltage-band affirmation of
`normal`.

The 2,405 rules that could not be expressed were categorised by the translator's own stated
reasons — the largest buckets are **quantities the simulator does not model** (~700),
**frequency** (~590; Grid2Op simulates none at all), and **time-bound / ride-through requirements**
(~550; needs a duration term the observation has no room for). This is the standards/simulator
mismatch, measured, and it runs in both directions: the corpus reaches only **5 of the 14**
observable variables and says nothing whatsoever about topology.

#### ⚠️ The validator's question was load-bearing — a controlled A/B

The first validation run rejected **31 of 32** rules and recorded no reason for any of them. The
cause was not the corpus. The auditor's first criterion asked whether the constraint is *stated in
the source text* — a category error against a stage-2 condition, whose entire purpose is to leave
the standard's language behind. Same model, same 32 rules, one changed question:

| arm | question asked | confirmed | rejected | distinct |
|---|---|---:|---:|---:|
| `strict` | is it **stated** in the text? | 1 | 31 | 1 |
| `translated` | is it a faithful **operationalization**? | **11** | 21 | **4** |

Perfectly nested — the whole difference is the ten `loading_pct > 100` rules. The strict arm
rejected them citing *"an undefined variable `loading_pct`"*, a variable listed in the vocabulary
block of that same prompt. **The verdicts were stable across runs while the stated reasoning was
fabricated.** That is invisible unless rejections are persisted with their reasons, which the first
run did not do.

Two practices are now permanent, and both belong in the methodology chapter: **persist every
rejection with its reason**, and **check an LLM's mechanically-checkable criteria in code first** —
a deterministic pass showed all 32 rules satisfied the vocabulary and healthy-grid criteria before
any re-run, which is what proved the stated grounds were false.

### Extraction Pipeline (Step by Step)

**Step 1 — Document ingestion**
⚠️ **CHANGED (revision item 10) — `pdfplumber`, not LangChain.** LangChain was never used; it
appears only in a stack-verification script. Chunks are 1,200 characters with 200 overlap.

**Step 2 — Prompted extraction**
Each chunk is passed to the LLM with a structured prompt:

```
You are a power systems engineer extracting safety rules from grid documentation.
From the text below, extract ALL operational constraints as JSON objects.
Each rule must have: rule_id, source, entity, condition, action, severity, explanation.
If no rule is present, return an empty list.
Respond ONLY with a JSON array. No preamble.

Text:
{chunk}
```

**Step 3 — Parsing and validation**
JSON output is parsed and validated against a Pydantic schema. Malformed outputs are retried once, then logged as failures for manual review.

**Step 4 — Deduplication**
Rules with identical conditions across overlapping chunks are merged. Source references are preserved.

### Output Schema

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

### Source Documents (as used — 16 PDFs, `docs/`)

NERC Reliability Standards (complete set), TPL-001-5.1, NERC FAC-008-5, PRC-006-5, PRC-023-6,
PRC-024-4, PRC-025-2, PRC-029-1, IEEE 1547-2018 (BPS perspectives), the NERC inverter-based
resource performance guideline, ENTSO-E NC RfG, the GB Complete Grid Code, the Bangladesh
Electricity Grid Code, FERC Order 842, FERC RM16-1-000, and an operator requirements document.

⚠️ **The yield is not uniform across them, and the pattern is the finding.** The two documents
most directly about post-contingency performance — TPL-001-5.1 and NERC FAC-008-5 — produced
**15 candidates between them**, because the PDFs on disk are summary versions rather than the
full standards carrying the contingency tables. Meanwhile the 707-candidate NERC set yielded 6
validated rules. See the per-document figure in `notebooks/thesis_figures.ipynb`.

---

## 4. Component C — Knowledge Graph (Symbolic Rule Store)

### What It Does

⚠️ **CHANGED (revision item 4).** It does **not** store grid topology, and the shield does **not**
depend on it. It answers exactly one question:

> *when the shield blocks a prediction, where did that rule come from, and who else says the same
> thing?*

### The first attempt, and why it was deleted rather than archived

A topology-centric graph was built first, exactly as agreed in this section: `Bus` / `Line` /
`Generator` / `Grid` nodes taken from the 36-bus network, with rules hung off whichever entity
they mentioned. It failed twice over, and both failures are worth reporting:

1. **It carried almost no information.** `Line` (59) and `Bus` (28) accounted for **3.5%** of the
   corpus — grid codes speak about generating modules and facilities, not about line 17. So
   **6,375 of its 6,632 edges (96%)** were `has_rule` links onto one catch-all `Grid` node.
2. **It was welded to one grid.** The shield is evaluated on three topologies. A graph keyed to
   36-bus indices cannot serve case14 or wcci2022 — while the rules themselves are
   topology-agnostic and need no such binding.

It was deleted, not archived. Its `.pkl` had already vanished from disk months earlier and nothing
noticed, which is the sharpest available evidence that it was load-bearing for nothing.

### Graph Schema — provenance

```
Document --contains--> Clause --states--> Rule --deduped_into--> ServedRule
    --instantiates--> Predicate --reads--> Variable
```

| Node type | n | What it is |
|---|---:|---|
| `Document` | 5 | one per source standard that contributed a validated rule |
| `Clause` | 11 | the cited section within a document |
| `Rule` | 11 | one per validated rule record |
| `ServedRule` | 4 | one per deduplicated rule — **carries the rule dict verbatim**; this is what the shield receives |
| `Predicate` | 2 | the physical check, normalized across noisy `entity` labels and numeric formatting |
| `Variable` | 3 | the vocabulary entries actually read |

**36 nodes, 40 edges**, spread across five edge types — compare v1's 6,632 edges of which 96% were
noise. Topology-agnostic by construction, so **one graph serves all three grids**.

### The Predicate layer is the graph's own contribution

Deduplication keys on `(entity, condition)`. That splits **one physical check into three records**
purely because of noise in LLM output: the extractor labelled it `Line` in the Bangladesh grid code
and `Facility` in the NERC set, and wrote `100` in one clause and `100.0` in another. The graph
parses each condition and canonicalises numeric literals, so those collapse back into one node.

This is a **view, not a substitution** — it never changes what the shield is served. Collapsing
three served rules into one would move `highest_severity` (they carry critical/high/medium) and the
blocks-by-severity histogram with it.

### The measured corroboration finding

| predicate | role | served rules | clauses | documents | identified bodies |
|---|---|---:|---:|---:|---:|
| `loading_pct > 100` | CONSTRAINT | 3 | **10** | **4** | **2** |
| `voltage_pu_min >= 0.9 and voltage_pu_max <= 1.1` | AFFIRMATION | 1 | 1 | 1 | 1 |

The thermal check is stated **ten times across four documents**, by NERC and the Bangladesh grid
code — independent restatement of the same physical limit by two standards bodies on two
continents. A flat rule list cannot express that; the deduplicated file actively hides it by
concatenating sources into one string.

This reframes "the corpus is 4 rules", which undersells it. ⚠️ Two things not to overclaim: one
source document does not identify its issuing body and is recorded as `None` rather than guessed
(so: **two** identified bodies, not three), and corroboration is not independent evidence about
physics — four documents agreeing that you should not exceed a thermal rating is four documents
agreeing on standard practice.

### Retrieval is opt-in, and the guarantee is a test

`KgRuleProvider` satisfies the same `RuleProvider` protocol the JSONL provider does, so it drops
into the gate unchanged. **The JSONL path remains the default.** A test compares full rule dicts as
sets between the two providers; re-running all three evaluations through the graph reproduced every
reported field exactly.

⚠️ **Never route by entity.** `R_1443` is labelled `Facility` while a rule stating the identical
check is labelled `Line`. Retrieving by walking entity edges — the classic KG pattern, and what v1
did — silently drops one of them from any line-level query. Both providers serve **every rule for
every prediction**; discrimination comes from the conditions.

### Why NetworkX, and why JSON rather than pickle

NetworkX is a pure Python library — zero setup, in-process, and trivially sufficient at 36 nodes.
Persisted as **JSON** (`node_link_data`), not pickle: v1 used a `.pkl` and it disappeared without
anyone noticing. JSON is diffable, git-friendly, survives a NetworkX version bump that would break
an unpickle, and costs nothing at this scale.

---

## 5. Component D — Symbolic Validation Shield

### What It Does

Intercepts every GNN prediction. Runs it against all applicable rules in the knowledge graph. Returns either a PASS (with the validated prediction) or a BLOCK (with a structured explanation).

### Why "Shield" and Not a Loss Function Penalty

This is the distinction between Generation 2 and Generation 3 NeSy systems:

- **Gen 2 (PINNs):** Loss function penalizes rule violations. The model is discouraged from breaking physics but not prevented. It can still output a physically impossible voltage.
- **Gen 3 (Shield):** The rule violation is detected at inference time, after the model outputs. The output is structurally blocked from reaching execution. This is a **hard constraint**, not a soft suggestion.

Directly inspired by **Younesi et al. (2026)**: their two-step microgrid system validated actions against a fixed rule set and achieved 91.7% safe power restoration. Our improvement: their rules were manually written. Ours are LLM-generated.

### Why the Shield Is Necessary, Not Optional (Core Claim, §0)

The Gen2/Gen3 distinction above is the mechanism. The reason it matters for this thesis specifically: a GNN's accuracy on a topology it wasn't trained on is a statistical claim, not a deterministic guarantee — a GNN is capable of running message-passing math on any graph size, but that says nothing about whether its output is correct on that graph. §11's cross-topology evaluation is designed to force this failure mode into the open (frozen 36-bus weights, zero-shot on 14/57/118-bus) specifically so the shield's intervention rate on those failures is the direct evidence that gating is necessary — not a defensive fallback bolted on for robustness, but the component the whole empirical argument rests on.

### ⚠️ CHANGED (revision item 5) — the gate is asymmetric and binary. There is no top-k.

The Correction Mechanism agreed in this section — retain the GNN's top-k candidates, let the shield
select a lower-ranked one that satisfies the rules, and log that as CORRECT — **presupposes a
multi-class label**. N-1 screening is binary per line: the answer is `secure` or `violation`, and
there is no third candidate to fall back to. Top-k selection, the CORRECT verdict, Correction
Delta and Catastrophic Failure Rate all go with it.

What replaced it is simpler and, for a safety argument, stronger:

```
for each contingency the model scored:
    model says `violation`  →  PASS unconditionally
    model says `secure`     →  evaluate the BASE CASE against the CONSTRAINT rules
                                 any rule fires  →  BLOCK  (escalate to `violation`)
                                 none fires      →  PASS
```

**The asymmetry is deliberate and is the whole design.** The gate can only ever move a prediction
in the cautious direction. It cannot suppress an alarm, and it never blocks a prediction that was
already conservative.

**Why that is the right shape here.** The two errors are not symmetric in cost. A false alarm
wastes an operator's attention; declaring a grid secure against further loss while it is *already*
outside its limits is the failure that ends in a cascade. A symmetric gate would need to be right
about both directions to be worth deploying; an asymmetric one only has to be right about the
direction that matters, which is a far weaker requirement to defend.

**AFFIRMATION rules never block on their own** (Option A). When affirmations for the predicted
class exist but none fires, that absence is *recorded* — so the harness can report what a
stricter gating policy would have done — but it does not veto. Blocking on missing support would
make the false-block rate a function of how evenly the corpus happens to cover the classes, which
is a property of the extraction run and not of the grid.

### The voltage contract (binding)

Rules speak in per-unit voltage; the simulator reports kV. The conversion is **per-line base kV**
from a generated sidecar, with energized-line masking — not a flat divisor.

⚠️ A flat `v_or / 150.0` conversion, which earlier drafts specified, gives ~100% false blocks:
case14 runs lines at ~20 kV *and* ~138 kV, and even the 36-bus grid has 7 lines at ~365 kV. Base kV
comes from the backend nominals. **These grids operate ~6% above nominal**, so a healthy frame
reads ~1.06 pu, not ~1.00 — which is exactly the sort of thing that turns a correct rule into a
false alarm if assumed rather than measured.

### Validation Logic (as built)

```python
def validate_n1(context, rules, predicted):
    """context: raw telemetry for the frame. predicted: SECURE | VIOLATION."""
    if predicted == VIOLATION:
        return ShieldResult(status="PASS")          # never block a cautious call

    violated, supporting, contradicting = [], [], []
    for rule in rules:
        fired = evaluate_condition(rule["condition"], context)
        if rule["role"] == "CONSTRAINT" and fired:
            violated.append(rule)                   # base case already out of limits
        elif rule["role"] == "AFFIRMATION" and fired:
            (supporting if rule["affirms"] == predicted else contradicting).append(rule)

    if not violated:
        return ShieldResult(status="PASS", supporting_rules=supporting)

    violated.sort(key=lambda r: SEVERITY_ORDER[r["severity"]])
    return ShieldResult(status="BLOCK", violated_rules=violated,
                        highest_severity=violated[0]["severity"],
                        explanation=build_explanation(violated))
```

A malformed condition returns `False` and is counted as `NOT_EVALUABLE` rather than blocking —
a rule the gate cannot understand must not be allowed to veto. Both counters read **zero** across
all three topologies.

### Example Output (blocked prediction, with citation)

```
BLOCKED: model called contingency (frame 118, line 42) secure.

  R_1443 [HIGH] CONSTRAINT: maximum line loading as percent of thermal limit exceeds 100
    condition: loading_pct > 100
    stated in 7 clause(s) across 2 document(s), 1 standards body
      - TPL-001-5.1, Section 5.1.f — TPL-001-5.1 Transmission System Planning [R_167]
      - NERC PRC-005, Section 6.2.1 — NERC Reliability Standards (Complete Set) [R_1492]
      - TPL-008-1, Table 1 (f) — NERC Reliability Standards (Complete Set) [R_2018]
      ...
```

Every block cites the rule, the clause, the document, and **every other standard that states the
same requirement**. The last of those is what the knowledge graph adds over a flat rule list.

### Measured result

| topology | model | + shield | Δ | blocked | corrections | regressions | **intervention precision** |
|---|---:|---:|---:|---:|---:|---:|---:|
| neurips2020 | 0.8956 | **0.9038** | +0.0082 | 353 | 331 | 22 | **0.938** |
| case14 | 0.4167 | **0.4188** | +0.0021 | 103 | 95 | 8 | **0.922** |
| wcci2022 | 0.5577 | **0.6253** | +0.0676 | 22,559 | 21,065 | 1,494 | **0.934** |

On the large unseen grid the shielded score (0.6253) **exceeds what the raw model reaches with an
oracle threshold on that same grid** (0.5721) — a layer bolted on after training, changing nothing
inside the model, pushing an unseen-topology result past the model's own ceiling.

Zero ERROR and zero NOT_EVALUABLE verdicts everywhere. Blocks are `high` severity only.

⚠️ **A validation note that belongs with these numbers.** Running the gate against the *unvalidated*
32-rule corpus gives visibly worse results (neurips +0.0026 at 0.514 precision). The difference is
the voltage rules, which came from ride-through tables and time-bound envelopes misread as
instantaneous limits: they fired at 0.13–0.20 precision and, on the training grid, outnumbered the
thermal blocks. Stage 3 removed them on textual grounds having never seen a single fire-rate
measurement. **A rulebook that is mostly wrong rules does not perform at a fraction of the good
version — it performs at the average, and the average hides the good rules entirely.**

---

## 6. Data Collection Strategy

### Why Simulation, Not Real Data

Real operational data from utilities is proprietary, unavailable without NDAs, and typically anonymized in ways that remove the topology information a GNN needs. This is a known limitation across the entire field — Ahmadi et al. (2026) explicitly identifies this as the "simulation-to-reality gap."

We use **Grid2Op** — a Python framework developed by RTE (the French transmission system operator) specifically for sequential decision-making and AI research on power grids. It implements full power flow equations via a backend (PandaPower by default), enforces thermal limits and N-1 security constraints natively, and is the environment used in the L2RPN (Learn to Run a Power Network) challenge — the most prominent AI-for-grids benchmark in the literature. Simulation on standard benchmarks is the accepted methodology for this research domain.

### The Environments: l2rpn_neurips_2020_track1 and l2rpn_wcci_2022

Grid2Op ships several competition environments. Environment choice directly determines the scale and complexity of the problem the GNN must solve. `l2rpn_case14_sandbox` — which we were previously using — is explicitly a development/sandbox environment with only 14 buses and 20 lines. It is useful for quick checks, but it does not cover the same scale as the benchmark environments used for the thesis.

**Smaller benchmark: `l2rpn_neurips_2020_track1`**
- 36 substations, 59 powerlines, 22 generators, 37 loads
- Subset of the IEEE 118-bus grid (the standard large-scale benchmark)
- Used in the NeurIPS 2020 L2RPN competition robustness track
- Ships with two data tiers: `_small` (900MB, ~48 years of 5-min data) and `_large` (4.5GB, ~240 years)
- The `_small` variant is convenient for quick iteration; the `_large` variant provides broader coverage
- Includes stochastic line disconnections and maintenance events — realistic fault conditions out of the box

**Larger benchmark: `l2rpn_wcci_2022`**
- 118 substations, 186 powerlines, 91 loads, 62 generators — full IEEE 118 scale
- Supports `chronix2grid` for infinite synthetic data generation
- Useful for scalability checks or for experiments that benefit from a larger graph and storage nodes
- A good candidate when comparing how the pipeline behaves at a broader problem scale

### ⚠️ CHANGED — what is generated is an N-1 label vector, not a fault class

**Normal operation:** Grid2Op's chronics provide 48–240 years of realistic load and generation
time-series at 5-minute resolution, including renewable variability and demand peaks. No synthetic
load profiles needed.

**The label.** For each logged frame, the generator loops over **every line in the network**,
simulates its outage, and records whether the resulting power flow violates a thermal limit:

```
n1_violation[k] =  1   losing line k violates a limit (or ends the episode)
                   0   losing line k is secure
                  -1   not evaluated (line already out, or the flow diverged)
```

`-1` entries are **missing labels, not negatives**, and are masked out of the loss. `n1_post_rho`
records the maximum rho after each outage, so a severity or regression variant is derivable without
regenerating anything.

**This is the property that makes the thesis comparison meaningful:** producing this label requires
a power-flow solve per line. No rule over the present observation can restate it — unlike the
retired 4-class target, which four `if`/`else` statements reproduced exactly (§2).

**The three sets, as generated and verified:**

| environment | role | subs | lines | frames | contingency labels | wall time |
|---|---|---:|---:|---:|---:|---:|
| `l2rpn_neurips_2020_track1_small` | **train** | 36 | 59 | 12,000 | 702,618 | 20 min |
| `l2rpn_case14_sandbox` | test — unseen, smaller | 14 | 20 | 6,000 | 118,502 | 5 min |
| `l2rpn_wcci_2022` | test — unseen, larger | 118 | 186 | 4,000 | 742,472 | 30 min |

**Sizing note that is easy to get wrong:** effective sample size is bounded by **scenario count**,
not label count. A stride of 12 (label every 12th step) is the protocol on all three topologies —
identical generation, only the environment differs. At the earlier stride of 4, a 12,000-frame
budget would have covered ~210 of the 36-bus environment's 576 chronics instead of all of them,
buying correlated frames rather than diversity.

**`NO_OVERFLOW_DISCONNECTION = False`** must be passed at `grid2op.make()` via `param=params`, not
set after — it is what allows natural cascades to develop.

### Data Generation Script (Skeleton)

```python
import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend  # faster backend — use this always
import numpy as np
import json

# Example setup using the NeurIPS 2020 track1 small environment (36 subs, 59 lines)
env = grid2op.make(
    "l2rpn_neurips_2020_track1_small",
    backend=LightSimBackend()  # ~10x faster than default PandaPowerBackend
)

params = Parameters()
params.NO_OVERFLOW_DISCONNECTION = True  # keep overloads alive so we can label them
env.change_parameters(params)

do_nothing = env.action_space({})
records = []

for chronic_id in range(len(env.chronics_handler.subpaths)):
    obs = env.reset()

    for t in range(env.max_episode_duration()):
        action = do_nothing
        fault_label = "normal"
        fault_loc = None

        # Inject line trip with 8% probability
        if np.random.rand() < 0.08:
            line_id = np.random.randint(0, env.n_line)
            action = env.action_space({"set_line_status": [(line_id, -1)]})
            fault_label = "line_trip"
            fault_loc = int(line_id)

        obs, reward, done, info = env.step(action)

        # Overload detection from observation
        if obs.rho.max() > 1.0:
            fault_label = "overload"
            fault_loc = int(obs.rho.argmax())

        records.append({
            "rho":         obs.rho.tolist(),        # (59,) line loading
            "p_or":        obs.p_or.tolist(),        # (59,) active power origin
            "q_or":        obs.q_or.tolist(),
            "v_or":        obs.v_or.tolist(),        # (59,) voltage origin bus
            "load_p":      obs.load_p.tolist(),      # (37,) load active power
            "gen_p":       obs.gen_p.tolist(),       # (22,) gen active power
            "topo_vect":   obs.topo_vect.tolist(),   # topology
            "line_status": obs.line_status.tolist(), # (59,) bool
            "label":       fault_label,
            "fault_loc":   fault_loc,
            "timestep":    t,
            "chronic":     chronic_id
        })

        if done:
            break

with open("grid_dataset.json", "w") as f:
    json.dump(records, f)
```

> **Note:** `LightSimBackend` from the `lightsim2grid` package replaces the default PandaPower backend. It is the recommended backend for any serious data collection — approximately 10x faster for power flow computation, which matters when iterating over thousands of chronic steps.

### Why Not These Alternatives

| Alternative | Why Rejected |
|---|---|
| **Real utility datasets** | Proprietary; topology usually stripped; unavailable |
| **Pandapower (standalone)** | Excellent power flow solver but no built-in episode/chronic management, no RL-compatible step API, and no native fault/maintenance injection framework — we would have to rebuild what Grid2Op already provides |
| **PSCAD / MATLAB** | Commercial license required; not Python-native; no direct PyTorch integration |
| **Random synthetic graphs** | Not comparable to published benchmarks; unreproducible by reviewers |

### Cross-Topology Evaluation Sets

⚠️ **CHANGED (revision item 6) — two unseen topologies, not three.** `case14` and `wcci2022`.
**57-bus was never generated** and must not be cited.

Eval-only, no training, identical generation protocol. The prerequisite flagged here — that the
feature builders run unmodified on non-36-bus graphs — was **verified, and it holds**: removing
global pooling (§2) made the readout per-line, so a 20-line and a 186-line grid both run on the
frozen 36-bus checkpoint with no special-casing.

### Dataset Targets

⚠️ **CHANGED — frames, not fault-class samples.**

- **12,000 frames on the training grid**, yielding **702,618 contingency labels**. The unit that
  matters is the label, but the unit that bounds diversity is the frame.
- **Split: 70/15/15, chronic-level — never frame-level.** Splitting by frame leaks: consecutive
  frames within a chronic are near-duplicates and a cascade sequence would straddle the boundary.
- **No stratification by fault class** — there are no fault classes. Class balance is handled by
  masking unevaluated contingencies and by the base rate of violations (~18–28% per grid).
- **In-memory shuffle before training** to remove chronological domain shift, seeded for
  reproducibility.

---

## 7. GNN Training Pipeline

**Scope:** training happens on 36-bus data only. IEEE 14/57/118-bus data (§6) is never used to fit weights — frozen model is evaluated on it, see §11.

### Cross-Topology Evaluation — WRITTEN AND RUN

⚠️ **CHANGED.** The script exists (`evaluation/eval_n1_cross_topology.py`, and
`evaluation/eval_shield_n1.py` for the gated arm) and has been run on all three topologies. The
top-k retention step agreed here is gone with the top-k mechanism itself (revision item 5).

What it does per topology:

1. Load the frozen `gnn_checkpoint_n1.pt`.
2. Build PyG graphs via the same feature builders used in training — verified to run unmodified on
   foreign topologies.
3. Normalize with **36-bus training-split statistics**, recomputed inline (§2). Never the target
   topology's own.
4. Forward pass → one logit per line.
5. Threshold at a value **selected on the 36-bus validation split and held fixed across all three
   grids** — so no number is tuned to the grid it is reported on.
6. Pass each `secure` verdict through the shield (§5) → PASS or BLOCK.
7. Log per contingency: truth, prediction, shield verdict, failure mode, and which rules fired.

**⚠️ Two threshold protocols are in circulation and must never be mixed.** The headline tables use
the *held* threshold (honest protocol). A second, *best-on-its-own-topology* figure is reported
alongside as a reference ceiling — it is optimistic and is labelled as such. They give slightly
different numbers for the same model on the same grid (0.8956 vs 0.8972 in-distribution); that is
expected, not a discrepancy.

### Model Architecture (as built)

See §2 for the design and the reasoning. The agreed two-headed classifier/localizer below is
**retired** — kept only so the change is legible.

```python
class GridGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, heads, dropout,
                 line_head_use_mp=True):
        # 3x GATv2Conv + BatchNorm(track_running_stats=False), ELU between
        # NO pooling, NO classifier head, NO localizer head
        line_in = edge_features + node_features * 2      # raw endpoint skip
        if line_head_use_mp:
            line_in += last_dim * 2                       # learned embeddings
        self.line_head = nn.Sequential(
            nn.Linear(line_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),                             # ONE logit per line
        )

    def forward(self, x, edge_index, edge_attr, edge_fwd):
        # ... message passing ...
        # then, for each line k, read its own or->ex edge:
        #   [h_or || h_ex || edge_attr_k || x_or || x_ex] -> line_head
```

`line_head_use_mp=False` is the ablation that answers "is message passing helping here?" — it is a
reported result (0.8987 → 0.8411), not a private toggle.

### Training

Binary cross-entropy over the **masked** label vector (`-1` entries excluded). Primary metric is
F1 on the held-out chronics, with average precision alongside. Deployed configuration:
`hidden_channels=[16,32,32]`, `heads=[4,4,1]`, dropout 0.0, seed 42.

⚠️ **The larger `[64,128,128]` configuration overfits and collapses under its schedule** — it is
what auto-selects on a CUDA machine and has never produced a working checkpoint. The small
configuration is the proven one. ⚠️ **The initialisation is seed-sensitive**: seed 42 gives a
reproducible 0.82–0.90 across data partitions, but a different init can degrade to ~0.72. This is
localised to initialisation, not to the split.

### What Gets Logged

⚠️ **CHANGED (revision item 10) — no Weights & Biases.** W&B appears only in a stack-verification
script and was never wired into training. Metrics are printed per epoch and written to JSON
alongside the checkpoint: loss, F1, precision, recall, average precision, and the train/validation
gap — the last being what separates memorisation from an optimiser that never fitted the signal.

---

## 8. End-to-End Testing

### ⚠️ CHANGED (revision item 9) — the toy-scenario harness was never built

This section originally specified three hand-crafted scenarios on `rte_case5_example` as a
pre-integration confidence check. It was overtaken and never written. Nothing in the results
depends on it, and it should not be described in the report as a delivered artifact.

**What replaced it, and does the same job better:**

| | what it covers |
|---|---|
| **222 automated tests** | the shield's evaluator and role semantics, the polarity guard, N-1 label construction, the translation mapping, rejection persistence, the knowledge graph and its retrieval equality, and the eval harness itself |
| **Three full-topology evaluation runs** | the entire pipeline end to end on real data — 974,000 contingencies across three grids, not five hand-checked states |
| **Regression tests that encode past failures** | each pins a specific bug that already happened once: the stage-2 parsing failure, the discarded-rejection failure, the topology-keyed knowledge graph |

The last of those is the part worth describing in the methodology chapter. Every test in the suite
that guards a known failure carries a docstring stating **what failure it encodes** — so a future
change that reintroduces the bug fails with an explanation rather than an assertion error.

**What the toy harness was going to validate, and where that evidence now comes from:**

1. *GNN outputs are well-formed and convertible from an observation* → covered by the cross-topology
   runs, which do exactly this 22,000 times.
2. *The knowledge graph is built correctly and queryable* → `tests/test_kg.py`, including the
   structural checks that would have caught the v1 design.
3. *The shield passes valid states and blocks invalid ones* → measured, with intervention precision
   reported per grid rather than asserted on three examples.
4. *Explanations are traceable to rule IDs and source documents* → the citation chain in §5, which
   goes further than the toy scenarios promised: it names every standard that states the rule, not
   just the one it came from.
5. *The full pipeline runs without integration failures* → zero ERROR and zero NOT_EVALUABLE
   verdicts across all three topologies.

---

## 9. Technology Stack

| Component | Tool | Justification |
|---|---|---|
| Grid simulation | **Grid2Op** | Built by RTE (French TSO); native chronic/episode management; thermal limit enforcement; L2RPN benchmark environment; RL-compatible step API |
| Grid2Op backend | **lightsim2grid** | ~10x faster power flow solver than Grid2Op's default PandaPower backend; mandatory for any serious data generation run |
| ~~Physics solver (KCL residuals)~~ | ⚠️ **NOT IMPLEMENTED** (revision item 7) | The KCL residual metric was specified and never written. The gate's checks are the thermal and voltage limits carried by the rule corpus. **Do not describe KCL as a delivered metric.** |
| GNN framework | **PyTorch + PyTorch Geometric** | De facto standard for GNN research; full architecture flexibility |
| LLM inference | **Ollama + Qwen3.6-35B-A3B / Nemotron 3 Nano (30B-A3B)** | Local, reproducible; sequential two-stage pipeline — Extractor then Validator; native JSON format enforcement |
| PDF ingestion | **pdfplumber** | ⚠️ **CHANGED** — LangChain was specified but never used; it appears only in a stack-verification script. Chunking is 1,200 chars / 200 overlap, done directly. |
| Knowledge graph | **NetworkX**, persisted as **JSON** | Zero-setup at 36 nodes. JSON rather than pickle — the v1 `.pkl` vanished from disk unnoticed (§4). Neo4j was never needed. |
| Shield logic | **Python (custom)** | Fully auditable, no external dependencies, easy to unit test |
| Data validation | **Pydantic** | Schema enforcement on LLM JSON outputs |
| ~~Experiment tracking~~ | ⚠️ **W&B NOT USED** (revision item 10) | Specified but never wired in. Metrics print per epoch and are written to JSON. |
| Version control | **Git + GitHub** | Standard |

---

## 10. Compute Allocation

⚠️ **CHANGED — hardware specifications were removed from this document, and from the repository,
on 2026-08-20.** They were never load-bearing, this section and `CLAUDE.md` had drifted into
recording *different* GPUs, and nothing in the results depends on which card ran which stage.
What is worth recording is the split by **capability**, which is a genuine design constraint: one
machine has CUDA and Ollama on it, the other does not, and the pipeline is arranged so that
neither one blocks the other.

| role | capability it needs | what runs there |
|---|---|---|
| **LLM host** | CUDA + Ollama | `extract.py` → `translate.py` → `validate.py`, strictly sequential with `keep_alive=0` between stages |
| **Workstation** | any torch device | all three N-1 dataset generations, GNN training, the polarity guard, the knowledge-graph build, every shield evaluation, the full test suite |

⚠️ **CHANGED from the agreed plan, which put every production run on one machine.** Training moved
off the LLM host, and that turned out to be the safer choice rather than a compromise:
`training/config.py` auto-selects `[64,128,128]` on a CUDA device, and that configuration has
never produced a working checkpoint. The proven `[16,32,32]` config is what auto-selects
everywhere else.

### The models are never co-resident

Qwen3.6-35B-A3B runs to completion, is unloaded, and only then does the Nemotron validator load.
The validator is the larger of the two and relies on partial CPU offload. This is a hard
constraint on the pipeline's shape, not a preference — it is why Component B is three sequential
passes over the corpus rather than one.

### A reproducibility note, found while verifying the knowledge graph

The forward pass is **not bit-deterministic** on the workstation's backend: two runs of an
identical evaluation differ at ~1e-6 in the selected threshold and ~1e-8 in average precision.
**No count has ever moved** — every F1, block, correction and regression reproduces exactly — but
"identical to the digit" holds at the four decimal places actually reported, not at full float
precision. Worth stating rather than discovering under questioning.

### Implications for implementation

- Cross-topology *inference* is cheap; **dataset generation** for those topologies (§6) is the
  heavier cost — 20 / 5 / 30 minutes for the three grids. Estimate before scheduling.
- No multi-GPU; keep everything single-device.
- LLM extraction is the only stage that cannot move between machines, so it is the one to schedule
  around.

---

## 11. Evaluation Plan

⚠️ **This section is substantially rewritten.** The agreed plan was built around the 4-class task
and a three-tier accuracy/hop-error/KCL table. Several of its metrics have no meaning under N-1
screening, and two were never implemented. What is below is what was measured.

### Metrics that were retired, and why

| Agreed metric | Status |
|---|---|
| Fault accuracy, macro F1 over 4 classes | **Gone with the task** (§2). Binary F1 per contingency replaces it. |
| Mean hop error (predicted vs true fault bus) | **Gone with the localizer head.** There is no bus-level prediction. |
| KCL violations | ⚠️ **Never implemented** (revision item 7). Do not report. |
| Prediction entropy shift | Not measured. A binary logit's entropy is a rescaling of its distance from the threshold — it adds nothing here. |
| Shield Activation Rate | Superseded by **reach** (share of contingencies the gate is eligible to act on), which is the same quantity stated so it cannot be confused with block *count*. |
| Correction Delta, Catastrophic Failure Rate | **Gone with top-k** (revision item 5). Binary screening has no candidate set. |

### The model, alone

| Metric | What it measures |
|---|---|
| F1 per contingency | at the **held** threshold — selected on the 36-bus validation split, fixed across all three grids |
| Average precision | threshold-free ranking quality |
| Ratio to best single rule | the honest baseline: the best achievable threshold on the removed line's own loading |
| Message-passing ablation | head-only readout vs full — isolates what the graph structure contributes |

### The gate

| Metric | What it measures |
|---|---|
| **Intervention precision** | when the gate overrules the model, how often it is right. **The headline.** |
| **Reach** | share of contingencies where the gate is eligible to act at all |
| **Corrections vs regressions** | violations caught vs secure contingencies wrongly escalated |
| **Structural ceiling** | share of the model's missed violations that sit on a base case *any* present-state rule could see |
| Health counters | ERROR and NOT_EVALUABLE verdicts — both must be zero, and are |

### Results

| | 36-bus *(trained)* | 14-bus *(unseen)* | 118-bus *(unseen)* |
|---|---:|---:|---:|
| best single rule | 0.4639 | 0.5392 | 0.4915 |
| model alone | 0.8956 | 0.4167 | 0.5577 |
| **model + gate** | **0.9038** | **0.4188** | **0.6253** |
| model at its own oracle threshold | 0.8972 | 0.4477 | 0.5721 |
| **intervention precision** | **93.8%** | **92.2%** | **93.4%** |
| reach | 0.31% | 0.087% | 3.04% |
| structural ceiling | 16.2% | 0.51% | 30.9% |

**Read the last three rows together — that is the finding.** The gate's precision is flat; its
reach and the share of model error it can touch are not. The symbolic layer does not become more
accurate off-distribution; the neural layer stops being reliable, so there is more for the gate to
do.

### The structural ceiling — the honest limit on the whole approach

Of every violation the model misses, what share happens on a base case a present-state rule could
even *see*? **16.2% / 0.51% / 30.9%.** The complement is a hard bound: **69–99% of the model's
dangerous errors occur on grids that look entirely healthy.** Detecting those requires simulating
the outage — which is the exact computation the model exists to avoid.

This answers "would more rules have helped?" structurally, without building progressively larger
rulesets and plotting a curve. On the 14-bus grid, essentially no. On the 118-bus grid, up to a
third more — but only if a rule existed that fires where the thermal check does not.

### Topology distance analysis — still open

The agreed analysis (node-count ratio, average degree, diameter, density vs F1, to find the
breakpoint) was **not run**; only three topologies exist, which is too few to fit a trend. The one
thing the three points do show is that the naive predictor is wrong: **scaling up cost far less
than scaling down** (0.5577 at 118 buses vs 0.4167 at 14, from a 36-bus model, at the held
threshold). Do not present node count as the driver.

---

## 12. Out of Scope

These are decided exclusions. Do not prototype or propose these.

- Real utility deployment or hardware-in-the-loop
- Training any foundation model (LLMs are used pre-trained, quantized only)
- Transmission-level grids (we work on distribution level via Grid2Op's built-in environments only)
- Multi-agent / multi-microgrid coordination
- Blockchain, digital twins, or metaverse integration
- Reinforcement Learning agent (future work only)
- **Corrective action (auto-repair of unsafe states).** The shield blocks unsafe GNN predictions; it does not propose or execute a fix. This is a deliberate scope boundary, not an oversight — redispatch/repair is a different problem (see Younesi et al. 2026, OPF/redispatch literature) and adding it would conflate detection-and-gating novelty with optimization novelty. Flagged explicitly here so it reads as scoped, not missing, at defense. ⚠️ **CHANGED — the CORRECT carve-out no longer exists and no longer needs defending.** The shield
is now a binary asymmetric gate (§5): it escalates an over-permissive `secure` verdict to
`violation` and does nothing else. It never selects among candidates, never proposes a fault
location, and never computes a repair. The scope boundary is now structural rather than argued.
- **Retraining or fine-tuning on non-36-bus topologies.** Cross-topology evaluation (§11) uses
  frozen 36-bus weights only. Any retraining on case14 or wcci2022 data would defeat the
  generalization measurement — explicitly excluded, not an oversight.
- **Entity-based rule routing through the knowledge graph.** Not merely unimplemented — actively
  forbidden and test-guarded (§4). Identical rules carry different `entity` labels depending on
  which standard they came from, so routing by entity silently drops rules.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **GNN** | Graph Neural Network — processes graph-structured input natively |
| **GCN** | Graph Convolutional Network — a specific GNN architecture using spectral convolution |
| **GAT** | Graph Attention Network — GNN variant where edges have learned attention weights |
| **Knowledge Graph (KG)** | Graph of entities, relationships, and rules — our symbolic rule store |
| **Shield** | The symbolic validation layer — hard-blocks rule-violating GNN outputs |
| **Grid2Op** | Python power grid simulation framework by RTE; manages episodes, chronics, and fault injection for AI research |
| **Chronic** | A time-series of load and generation values used as one simulation episode in Grid2Op |
| **`obs.rho`** | Grid2Op observation attribute — ratio of current flow to thermal limit per line; >1.0 means overload |
| **`rte_case5_example`** | Grid2Op's 5-bus, 8-line minimal environment — used exclusively for toy scenario integration testing |
| **`l2rpn_neurips_2020_track1`** | Grid2Op's NeurIPS 2020 competition environment — 36 substations, 59 lines, subset of IEEE 118. One benchmark option used for the thesis |
| **`l2rpn_wcci_2022`** | Grid2Op's WCCI 2022 environment — full IEEE 118 scale (118 subs, 186 lines). Another benchmark option used for larger-scale testing |
| **LightSimBackend** | Fast power flow backend for Grid2Op (~10x faster than default); from the `lightsim2grid` package |
| **Power flow** | Mathematical solution for voltage/current/power at every grid node |
| **Fault injection** | Programmatically forcing a fault condition (overload, undervoltage, line trip) in simulation |
| **Voltage pu** | Voltage in per-unit — normalized so 1.0 = nominal voltage |
| **Knowledge bottleneck** | The problem that symbolic rules in NeSy systems must be written by hand — LLM extraction solves this |
| ~~**LangChain**~~ | ⚠️ Specified but **never used** (revision item 10) — `pdfplumber` handles ingestion directly |
| **Ollama** | Local LLM runtime that simplifies model management, serving, and inference with structured output support |
| **NetworkX** | Python graph library used as our initial knowledge graph backend |
| **4-bit/5-bit quantization** | Model compression (Q4_K_M/Q5_K_M) that reduces LLM memory footprint; what allows Qwen3.6-35B-A3B to stay GPU-resident on a single consumer card |
| **Frozen weights** | GNN weights fixed after 36-bus training; used unchanged for cross-topology inference, no fine-tuning |
| ~~**Geodesic / hop distance**~~ | ⚠️ Retired with the localizer head — there is no bus-level prediction to measure distance against |
| ~~**KCL residual**~~ | ⚠️ Specified but **never implemented** (revision item 7). Do not report as a delivered metric |
| ~~**Shield Activation Rate (SAR)**~~ | Superseded by **reach**, below |
| ~~**Correction Delta**~~ / ~~**Top-k candidate set**~~ | ⚠️ Retired with the top-k mechanism (revision item 5). Binary screening has no candidate set to select over |
| **N-1 screening** | For every line: *if that line were lost right now, would the grid violate a limit?* The task the GNN actually performs. Producing the label requires a power-flow solve, which is what makes the neuro-symbolic comparison meaningful |
| **Contingency** | One (frame, line) pair — one hypothetical outage evaluated against one grid state |
| **Mixed frame** | A grid state where some contingencies violate a limit and others do not. 96–99% of all frames; the reason there is no graph-level pooling (§2) |
| **CONSTRAINT / AFFIRMATION** | A rule's role. CONSTRAINT true ⇒ the standard is violated (can block); AFFIRMATION true ⇒ telemetry is consistent with the named class (evidence only, never blocks alone) |
| **Asymmetric gate** | The shield only ever moves a prediction toward caution: it escalates `secure` → `violation`, never the reverse (§5) |
| **Intervention precision** | When the gate overrules the model, how often it is right. Measured flat at 92–94% across all three grids — the headline finding |
| **Reach** | Share of contingencies where the gate is eligible to act at all (0.087%–3.04%). What actually varies with topology |
| **Structural ceiling** | Share of the model's missed violations sitting on a base case *any* present-state rule could see (16.2% / 0.51% / 30.9%). The complement is a hard bound on the whole approach |
| **Served rule** | One record in the deduplicated corpus — what the shield is actually handed. Four of them |
| **Predicate** | The physical check underneath one or more served rules, normalized across noisy entity labels and numeric formatting (§4) |
| **Closed-form target** | A label computable directly from the observation the model is given. The retired 4-class task was one — which is why it was retired (§2) |
| **Held threshold** | A decision threshold selected on the 36-bus validation split and applied unchanged to every grid, so no reported number is tuned to the grid it appears on |

---

## 14. Disclaimer — Living Document

> **Nothing in this document is final.**
>
> All model names, dataset choices, simulation environments, library selections, and architectural decisions recorded here reflect the best available information at the time of writing. Any of these may change as new findings, benchmarks, hardware constraints, or implementation realities emerge during the course of the thesis.
>
> Specific items subject to change without notice:
> - LLM model selection (e.g. Qwen3.6-35B-A3B, Nemotron 3 Nano 30B-A3B)
> - GNN architecture choice (GCN vs GAT vs alternatives)
> - Grid2Op environments and their version-specific behavior
> - Knowledge graph backend (NetworkX vs Neo4j)
> - Dataset size targets and split ratios
> - Evaluation metrics and baseline comparisons
>
> When a decision changes, the relevant section of this document is updated to reflect the new choice and the reasoning behind the change. Previous decisions are not preserved — this document describes the current state, not the history.
>
> **Exception, added 2026-08-20:** where a design was *superseded by measurement* rather than by
> preference, the old design is retained alongside the new one and marked ⚠️. Those changes are
> results in their own right — "we agreed X, measured Y, and here is why" is a stronger methodology
> chapter than a document that only ever shows its final state. The change table at the top is the
> index to all of them.

---

---

*Last updated: 2026-08-20 — **major revision, all four components now built and measured.** Added
the change table (top) covering ten design changes between the agreed architecture and what was
built. Rewritten: §0 (claim now measured; corrected the "precision rises off-distribution" reading,
which was false), §1 (pipeline diagram — four extraction stages, provenance KG, asymmetric gate),
§2 (Component A is N-1 screening, not 4-class classification; no pooling; edge-level readout;
8/8 features; measured cross-topology results), §3 (four-stage extraction; the validator
prompt-framing A/B), §4 (Component C is provenance-centric; the topology-keyed v1 and why it was
deleted; the corroboration finding), §5 (asymmetric binary gate; top-k and CORRECT retired; the
per-line base-kV voltage contract; measured results), §6 (N-1 label generation; three environments,
not four; chronic-level splitting), §7 (cross-topology script written and run; architecture as
built; no W&B), §8 (toy harness never built — replaced by 222 tests and three full evaluation
runs), §9 (no LangChain, no W&B, KCL never implemented, KG persisted as JSON), §10 (machine roles
swapped; ⚠️ unresolved GPU conflict flagged; non-determinism note), §11 (evaluation plan rebuilt
around the metrics that exist), §12 (CORRECT carve-out removed; entity routing added as forbidden),
§13 (glossary: retired six terms, added twelve).*

*Amended the same day: **all hardware specifications were removed** from §9, §10 and §13. This
document and `CLAUDE.md` recorded different GPUs for the same machine; rather than adjudicate a
detail that no result depends on, §10 now records the split by capability — which is the part that
actually constrained the design.*
