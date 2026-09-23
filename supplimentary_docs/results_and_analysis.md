# 5 Results and Analysis

**Two protocol constraints govern every number below, and are stated once, here.**

**First, the decision threshold is 0.8849, selected once on the NeurIPS 2020 validation split and
applied unchanged to all three topologies** — three grids differing in substation and line count
and connectivity. A cutoff chosen once and applied everywhere is a **held** threshold, the reported
protocol here because it is the only one obtainable in practice. Choosing the best cutoff for each
grid individually instead gives **best-case** figures: they use that grid's answer key, unavailable
in deployment, so they appear only in labelled columns and are quoted as ceilings — never mixed with
held values in the same column.

**Second, the evaluation batch size is pinned at 64.** This model uses
`BatchNorm(track_running_stats=False)`, which rescales each inference batch using that batch's own
statistics rather than stored training statistics — so *which other examples share the batch*
changes the output slightly. The same trained model on the same data scores F1 0.8972 at batch 64
and 0.9255 at batch 512 (F1 defined in §5.1.2, 1.0 = perfect). No model score is quoted here without
its batch size (findings §8, §13, §14 caveat 3).

---

## 5.1 Performance Evaluation

### 5.1.1 Experimental design

The task is **N-1 contingency screening**. A *contingency* is the loss of one component — here,
one transmission line switching out — and "N-1" means one component gone out of *N*. The task is
therefore to decide, for every line in turn, whether losing that one line right now would push the
grid past an operating limit. When the answer is yes, that line's contingency is a **violation**:
losing it drives some other line past its thermal rating, or leaves the grid unable to solve at
all.

One property of this task makes the entire comparison meaningful, and it is worth stating before
anything else. **The correct answer cannot be worked out from the present observation.** Deciding
whether losing line *k* causes a violation requires solving the power flow for the network with
line *k* removed — a computation, not a lookup. No rule written over the grid's current readings
can restate that answer.

Three topologies were used, generated under an identical protocol (`--task n1 --n1-stride 12`) so
that only the environment differs:

| topology | role | buses | lines | scored contingencies | violation rate |
|---|---|---:|---:|---:|---:|
| NeurIPS 2020 track 1 | **training**; scored on its held-out test split | 36 | 59 | 113,205 | 18.4% |
| case14 | unseen, **smaller** | 14 | 20 | 118,502 | 27.8% |
| WCCI 2022 | unseen, **larger** | 118 | 186 | 742,472 | 24.8% |

The learned component is a **graph neural network** (GNN): it treats the grid as a graph, with
substations as nodes and lines as edges, and exchanges information between neighbouring nodes
before predicting. It was trained on the 36-bus grid only. The other two grids are scored with that
same trained model applied unchanged, never having been fitted to them.

All three datasets verify clean. A *frame* is one snapshot of a grid at one moment — every line
flow, voltage and switch position — and each frame yields one label per line. Across all three
there are no structural errors, no frames in which every contingency is secure, and a global
`rho_max` lift of 1.08–1.16x. (*Loading*, written `rho`, is a line's flow divided by its rating, so
`rho = 1.0` is exactly at the limit and `rho_max` is the worst-loaded line in a frame.) The task
therefore does not become trivial or degenerate off the training grid (§14).

The symbolic layer was built by a four-stage pipeline driven by large language models, reading
**16 published standards documents** (IEEE, NERC, ENTSO-E and national grid codes). It produced
**2,463 candidate rules** and, after translation into the simulator's vocabulary, a **58-rule
corpus** — every rule expressible over what the simulator actually measures. Channel assignment
sorts those 58 into BLOCK (vetoes a dangerous prediction), WARN and NORMAL (explain a prediction
the gate lets through), or documents them as not currently applicable (§5.2.5, §5.2.9).

### 5.1.2 Metrics and evaluation regimes

The primary metric is **F1** on the per-contingency binary label — violation or not. Two
quantities compose it: *precision* is, of everything flagged as a violation, the share that really
was one; *recall* is, of all the violations that existed, the share that was flagged. F1 is their
harmonic mean, so a method cannot score well by neglecting either — flagging everything gives
perfect recall and poor precision, and flagging almost nothing does the reverse. F1 is used here
because the classes are heavily skewed, and because it is the quantity the decision threshold is
selected against.

Two reference points accompany every grid, and both are floors rather than competitors:

- the **all-positive baseline** — the score obtained by predicting "violation" for every
  contingency, which requires no model at all;
- the **best single rule** — a threshold on the loading of the line being removed. The L2RPN
  challenge organisers record that all top teams adopted exactly this predicate for overload
  alerting, so it represents what practitioners actually do.

The symbolic layer — referred to throughout as the **shield**, or the **gate** — reads each of the
model's predictions, checks it against the rules extracted from published standards, and may
overturn it. It carries three metrics of its own, because an F1 delta alone conceals *how* an
improvement was earned:

| metric | definition |
|---|---|
| **intervention precision** | when the gate overrides the model, how often the override is correct |
| **reach** | share of predictions on which the gate is eligible to act at all |
| **structural ceiling** | share of the model's missed violations occurring on a base state that any present-state rule could observe |

The distinction between the first two carries much of §5.4. A gate can be highly accurate and still
change very little, if it is rarely in a position to speak at all.

### 5.1.3 Cross-topology performance of the learned model

**Held threshold — the reported protocol:**

| topology | all-positive | best single rule | **model (held)** | recall | precision | missed violations |
|---|---:|---:|---:|---:|---:|---:|
| NeurIPS 2020 *(in-distribution)* | 0.3104 | 0.4639 | **0.8956** — 1.93x rule | 0.902 | 0.889 | 2,041 |
| **case14** *(unseen, smaller)* | 0.4345 | 0.5392 | **0.4167** — **0.77x rule; fails** | 0.435 | 0.400 | 18,592 |
| **WCCI 2022** *(unseen, larger)* | 0.3969 | 0.4915 | **0.5577** — 1.13x rule | 0.629 | 0.501 | 68,166 |

(*In-distribution* means measured on the grid the model was trained on; the two unseen grids are
*off-distribution*, and that regime is what this thesis is about.)

**Best-case ceiling — best threshold chosen on each grid with the answer key. Not obtainable in
deployment; quoted only as a ceiling.** The third column, *average precision*, summarises
performance across every possible cutoff at once, so unlike F1 it measures the quality of the
model's ranking rather than of one particular decision.

| topology | model (best case) | average precision | gap over held |
|---|---:|---:|---:|
| NeurIPS 2020 | 0.8972 | 0.9615 | +0.0015 |
| case14 | 0.4477 | 0.4270 | **+0.0310** |
| WCCI 2022 | 0.5721 | 0.6419 | +0.0144 |

Three readings follow, stated plainly.

**Transfer is real but partial, and it is asymmetric in grid size.** Moving *up* to a grid three
times larger costs far less than moving *down* to one three times smaller. This is the opposite of
the intuitive expectation — a smaller grid looks like an easier problem — and it is the more
interesting half of the result.

**On case14 the model fails outright.** At the held threshold it scores 0.77x the single-rule
baseline and 0.96x the all-positive baseline (0.4167 against 0.4345). On that grid it is beaten not
only by one threshold on one feature, but by the strategy of answering "violation" every single
time. It is reported here as a failure rather than softened.

**The best-case gap measures the optimism a per-grid threshold would buy, and it grows
off-distribution** — 0.0015 at home, where the threshold was chosen, against 0.0310 on case14,
twenty times larger. The reason is mechanical: a foreign grid shifts the range of scores the model
produces, so a cutoff fixed on the training grid lands further from that grid's best choice.
§5.3.1 quantifies this.

> ⚠ **Every figure in this table comes from a single training run.** A *seed* is the random number
> that sets a model's starting weights, so two runs differing only in seed are the same experiment
> run twice. A four-seed replication (§5.2.4, §5.3.6) puts this checkpoint's cross-topology spread
> at **±0.02, and ±0.07 on case14** — full derivation there. The three conclusions above survive
> that band comfortably — case14's 0.12 shortfall against the rule baseline and WCCI 2022's 0.066
> margin over it are both several times the spread — but no narrower comparison may be read off
> this table.

### 5.1.4 The cost of the accurate method

The labels this chapter has been scoring against are not estimates. Each one is the answer a full
AC power-flow solver gave when asked what happens if a particular line is removed — the same
non-linear physics an operator's own control-room tools solve. On accuracy that method cannot be
beaten, because it *is* the answer key (§5.1.1, §5.1.2). The question it raises is a different one:
what does it cost to run?

That matters because contingency screening is not an offline exercise. A control room re-runs it as
the grid state changes, and the number of questions it must ask is the number of lines — every one
of which could fail. A method that answers correctly but too slowly to finish before the state has
moved on is not usable, however accurate it is. Nothing in this project had measured that cost, so
it was measured directly (`evaluation/bench_inference_speed.py`, validated by
`tests/test_bench_inference_speed.py`, artifacts in `results/timing/`).

The benchmark asks both methods the identical question on the identical grid snapshots — for each
line, does removing it violate a thermal limit? — and times them. Two properties keep the
comparison honest rather than flattering:

- **The unit is one contingency, not one frame.** The network emits one verdict per line per
  snapshot, so the solver is timed for one solve per line per snapshot too. Timing a single solve
  per snapshot would have understated the solver's cost by a factor of the line count — 59x on
  NeurIPS 2020, 186x on WCCI 2022.
- **The solver arm is this project's own labelling code, called unmodified** — the same
  `scripts.generate_dataset.label_n1` routine that produced every dataset in `data/`, not a
  reimplementation written to be timed. Both arms are asserted to have screened the identical
  (frame, line) set before anything is reported.

The AC solver timed throughout is **LightSim2Grid**, a C++ implementation of Newton-Raphson power
flow — the same backend that generated every dataset in this thesis. It is the fastest solver
actually available, and the one a deployed screening system would have to beat, so it is the only
comparison reported here.

**Throughput, contingencies screened per second:**

| grid | contingencies | AC — LightSim (C++) | GNN — CPU | GNN — XPU |
|---|---:|---:|---:|---:|
| NeurIPS 2020 *(trained on)* | 29,213 | 597.3 | **34,864** | **36,096** |
| case14 *(unseen, smaller)* | 10,000 | 715.0 | **18,427** | **16,366** |
| WCCI 2022 *(unseen, larger)* | 37,200 | 393.5 | **111,104** | **136,305** |

**Speedup, end-to-end (feature build + forward pass) against LightSim:**

| grid | GNN CPU vs LightSim | GNN XPU vs LightSim |
|---|---:|---:|
| NeurIPS 2020 | **58x** | **60x** |
| case14 | **26x** | **23x** |
| WCCI 2022 | **282x** | **346x** |

**Against the solver that actually built this project's data, the network is 23–346x faster.** The
comparison that matters is against the fastest solver actually available — which is also the one
used to build the datasets — so no slower reference implementation is quoted alongside it.

**The advantage grows with the size of the grid.** GNN-CPU-vs-LightSim speedup rises monotonically
with line count: 26x on case14 (20 lines), 58x on NeurIPS 2020 (59 lines), 282x on WCCI 2022
(186 lines). This is the more interesting half of the result, and the opposite of how a speed
advantage usually behaves. The solver's cost scales with the number of lines, because each is a
separate power-flow solve; the network answers all of a snapshot's lines in one batched pass. The
larger and more realistic the grid, the wider the gap — which is exactly the property that would
matter in deployment.

**What the speed buys, and what it does not:**

| grid | AC per-frame, median (ms) | GNN batch-1, median (ms) | GNN F1 (these frames) | all-positive F1 (these frames) | AC F1 |
|---|---:|---:|---:|---:|---:|
| NeurIPS 2020 | 83.5 | 2.35 | 0.8858 | 0.2531 | 1.0000 |
| case14 | 24.4 | 2.21 | 0.3120 | 0.5111 | 1.0000 |
| WCCI 2022 | 382.5 | 2.75 | 0.4999 | 0.3014 | 1.0000 |

AC is ground truth by construction — it produced the labels, so its F1 is 1.0 and cannot be
anything else — so the trade is only meaningful stated in both directions. On the grid it was
trained on, the network reaches F1 0.8858 for roughly 58x less work than the solver that would
otherwise answer the question. On case14 it reaches 0.3120 — below the all-positive baseline on
these frames (0.5111), meaning it loses to answering "violation" every time, consistent with
§5.1.3's held-threshold result on the recorded dataset — and **speed does not repair that**. A wrong
answer produced an order of magnitude faster is still wrong. The cost measurement licenses one
claim and not the other: *where the model is accurate, it is accurate at a small fraction of the
cost of the method that produced its labels; where it is not accurate, no speed advantage changes
that.*

⚠ **Three caveats travel with these figures.**

1. **The accuracy column is measured on these benchmark snapshots, not on the recorded datasets
   §5.1.2 scores**, and their violation rates differ — the all-positive baseline here is 0.5111 for
   case14 against the dataset's 0.4345. These numbers must not be tabled against §5.1.3's; they are
   a paired control for the timing, sharing one set of frames with it, not a repeat of the primary
   protocol.
2. **Timing is hardware-dependent.** The ratio is the result; the absolute seconds describe the one
   machine each artifact's `environment` block records — a deliberate, scoped exception to this
   project's rule that no hardware specification is recorded anywhere else in the repository.
3. **The faster device is not the same on every grid, and does not track grid size cleanly.** The
   accelerator out-throughputs the CPU on NeurIPS 2020 (36,096 vs 34,864) and on WCCI 2022 (136,305
   vs 111,104), while the CPU wins on case14, the smallest grid (18,427 vs 16,366). The NeurIPS 2020
   margin is under 4% and close enough to be dispatch-cost noise rather than a clean trend; the WCCI
   2022 margin (23%) is not. The CPU figure is quoted as the headline throughout regardless, because
   it is the like-for-like comparison: the AC solver is itself CPU-bound and single-threaded.

### Relation to the literature

Nakiganda and Chatzivasileiadis (arXiv:2310.04213, *Graph Neural Networks for Fast Contingency
Analysis of Power Systems*) report graph networks screening contingencies "100–400 times faster
than the Newton-Raphson power flow solver" on test cases from 6 to 118 buses — the same task as
this one. The comparison here only partly agrees with that range: WCCI 2022, the largest grid,
sits inside it at 282x (CPU, end-to-end), but NeurIPS 2020 (58x) and case14 (26x) both fall below
their reported floor. Read against the grid-size trend above, that is consistent with their test
cases skewing smaller than WCCI 2022's 186 lines — but it is a partial disagreement with the
literature, not a confirmation of it, and is reported as such.

### 5.1.5 Learned baselines without a graph

The previous section places a non-learned method above the model. This section asks a different
question: within the learned methods, how much is the *graph* contributing?

**Why a graph was chosen in the first place.** The choice was a primitive one — made early, and
from a general pattern noticed across the literature rather than a benchmarked comparison against
simpler alternatives. Graph neural networks were visibly being applied to power-related problems
across several independent lines of work: power-flow balancing (Hansen, Anfinsen & Bianchi, *IEEE
Trans. Power Systems* 38(3), 2023), optimal power flow (Yang et al., *IEEE Trans. Industrial
Informatics* 20(9), 2024), unsupervised power-flow solving (Lopez-Garcia & Domínguez-Navarro,
*Engineering Applications of AI* 117, 2023), and state estimation (Ringsquandl et al., CIKM '21).
That visible pattern — GNNs being reached for across power-systems problems generally — is what
motivated GATv2 as Component A's architecture (Architecture, `CLAUDE.md`; the choice of GATv2 over
the plain GATConv is a separate, load-bearing decision documented in
`archive/gnn_upgrade_assessment.md`).

None of the four papers above performs line-level violation or contingency classification — three
solve power flow as a regression problem, one performs state estimation — so none establishes a
GNN as literature-favoured for *this* task specifically. The one near-comparator doing structurally
the same task this thesis does, Pavão et al. (*Energy and AI* 22, 2025, 100564: per-line binary
violation prediction, multilabel per frame), reports the opposite: *"all top teams chose a
rule-based approach for raising alerts if the power line is overloaded with rho ≥ 1.0"* (§B1 of
`results_comparisons.md`) — on the one directly comparable benchmark, practitioners reached for a
threshold rule, not a graph model.

Two things in the broader literature are worth noting in hindsight, because they anticipate what
this section goes on to measure rather than excuse it. Yang et al. and Hansen et al. both treat
cross-topology transfer as something a plain GNN does *not* get for free — it is the specific
problem their added machinery (a physics-guided Lagrangian plus online transfer learning; a
line-graph representation plus localized deep layers) exists to solve, over and above a standard
message-passing network. And Ringsquandl et al. found that power grids do not show the usual
2–3-layer oversmoothing consensus of standard GNN benchmarks — their best-performing models needed
up to 13 layers to capture the long-range dependence power-flow redistribution requires. Component
A's GATv2 has three. Both observations are consistent with, and offer a literature-grounded
candidate explanation for, the transfer failures measured below — not something the original,
primitive choice of architecture had reason to anticipate.

To test it, **three models with no graph structure, spanning three different ways of fitting a
function**, were trained on the raw portion of the GNN's own readout — the readout being the final
stage that turns the network's internal representations into one number per line. For each line,
the raw portion is its 8 edge features plus the 8 node features of each endpoint, 24 columns in
total, reaching the readout by a *skip connection*: a path that hands the original inputs to the
final stage directly, bypassing the three message-passing layers. Fitting on them alone is
therefore the full model with its graph deleted — the comparison is "the same local information,
without the graph" rather than "some other feature set". All three were fitted on the NeurIPS 2020
*training* split only, with the threshold selected on its validation split and held unchanged, and
features standardised using training-split statistics applied unmodified to the foreign grids
(§26.1) — identical protocol to the GNN and to each other.

| model | what it is fed | scope of what it can see | what it outputs |
|---|---|---|---|
| logistic regression | the same 24 columns, standardised | that line's two endpoints only — no connectivity, no other line or bus | one probability per line, from a linear decision boundary |
| MLP | the same 24 columns, standardised | identical scope to the above | one probability per line, from a nonlinear function fitted by gradient descent — the same optimisation family as the GNN's own readout head, minus the graph |
| random forest | the same 24 columns, standardised | identical scope to the above | one probability per line, from the vote share of 300 bagged decision trees — a different inductive bias again, with no gradient descent at all |
| GNN (GATv2) | the **entire grid snapshot** — every node's and edge's features, plus connectivity | the whole graph, three hops — each line's prediction is informed by attention-weighted aggregation from its neighbours, its neighbours' neighbours, and a third hop out | one logit per line, all produced from a single forward pass over shared graph embeddings |

Every row of the first three is computed independently of every other row; the GNN's outputs for
every line in a frame come from one shared pass and are therefore correlated with each other in a
way the tabular models' predictions are not.

Held threshold throughout; best-case figures in brackets.

| grid | all-positive | best rule | logistic | MLP | **random forest** | **GNN** |
|---|---:|---:|---:|---:|---:|---:|
| NeurIPS 2020 *(trained on)* | 0.3104 | 0.4639 | 0.7121 *(0.7125)* | 0.9127 *(0.9131)* | **0.9246** *(0.9251)* | 0.8956 *(0.8972)* |
| case14 *(unseen)* | 0.4345 | 0.5392 | 0.4554 *(0.4895)* | 0.4215 *(0.4798)* | **0.6192** *(0.6348)* | 0.4167 *(0.4477)* |
| WCCI 2022 *(unseen)* | 0.3969 | 0.4915 | 0.4705 *(0.4725)* | 0.4926 *(0.5179)* | 0.5261 *(0.5507)* | **0.5577** *(0.5721)* |

**In-distribution, none of the three non-graph models can be told apart from the GNN at this
protocol's precision.** MLP (0.9127) and random forest (0.9246) both sit inside the batch-size band
the GNN's own checkpoint spans on identical weights and identical data (0.8972–0.9255, established
at the head of this chapter) — the same reasoning that made the earlier tree comparison a tie
applies here to both newer models. The honest statement, again, is a tie on the grid the model was
trained on: **the graph machinery buys no measurable accuracy at home, against three different
non-graph families, not just one.**

Off-distribution the picture splits by grid rather than resolving cleanly for the GNN.

- **On case14, random forest is the only model in this table — the GNN included — that beats the
  best single rule.** 0.6192 against a rule baseline of 0.5392, a margin of 0.08, while the GNN
  falls to 0.4167, *below* the all-positive baseline (0.4345). No batch-size or seed-noise band
  measured anywhere in this project is close to 0.20 wide; this is not an artefact of measurement
  precision. Logistic regression and the MLP also fail to clear the rule there.
- **On WCCI 2022, the GNN leads** — 0.5577 against random forest's 0.5261, a margin of 0.0316.

The underlying hypothesis was that a power grid, being inherently graph-structured data, was
naturally suited to a graph-structured model, and that the advantage would show up most where the
data's relational structure has the most room to matter — larger topologies, where a contingency's
effects redistribute across more of the network before reaching the line being scored. The WCCI
2022 result is the one outcome in this comparison consistent with that hypothesis. **It is not
proof of it.** The margin sits under two standard deviations of this project's measured single-seed
noise band (±0.02 on unseen grids, §5.3.6) — nowhere near the "several times the spread" bar the
rule-baseline margins in §5.1.3 cleared. The deployed WCCI 2022 checkpoint has only ever been
trained at one seed; whether this margin survives a multi-seed re-run has not been measured. A
hypothesis with one unconfirmed, noise-adjacent data point in its favour is a lead worth stating,
not a result worth claiming as settled.

Random forest does not collapse on case14 — it is the only model of any kind, gated or ungated,
graph or not, that improves on the rule baseline there. §5.1.7 reflects this: transfer failure is
common among the architectures tested, not universal, and not specific to graph structure.

### 5.1.6 Symbolic shield performance

The shield is **asymmetric by design**: it may convert an over-permissive prediction ("this
contingency is secure") into a block, and it can never veto a cautious one. The reasoning is
operational rather than statistical — a missed violation can cascade into an outage, whereas a
false alarm costs an operator's attention. The two are not symmetric costs, so the gate is not
symmetric either.

The corpus rows marked *superseded* below are an earlier, unvalidated 32-rule corpus. They are
retained because the difference between the two corpora is itself evidence (§5.2.6).

| topology | corpus | model F1 | **+ shield** | Δ | blocked | corrections | regressions | **intervention precision** | reach |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NeurIPS 2020 | *guarded-32 (superseded)* | 0.8956 | 0.8982 | +0.0026 | 831 | 427 | 404 | 0.514 | — |
| NeurIPS 2020 | **58-rule** | 0.8956 | **0.9038** | **+0.0082** | 353 | 331 | 22 | **0.938** | 0.31% |
| case14 | *guarded-32 (superseded)* | 0.4167 | 0.4188 | +0.0021 | 103 | 95 | 8 | 0.922 | — |
| case14 | **58-rule** | 0.4167 | **0.4188** | **+0.0021** | 103 | 95 | 8 | **0.922** | 0.087% |
| WCCI 2022 | *guarded-32 (superseded)* | 0.5577 | 0.6232 | +0.0655 | 24,777 | 21,306 | 3,471 | 0.860 | — |
| WCCI 2022 | **58-rule** | 0.5577 | **0.6253** | **+0.0676** | 22,559 | 21,065 | 1,494 | **0.934** | 3.04% |

("Corrections" are blocks that fixed a wrong prediction; "regressions" are blocks that spoiled a
right one.)

On the large unseen grid the gate is worth considerably more than a rounding error:

- **Missed violations fell from 68,166 to 47,101 — a 31% reduction** — at the cost of roughly
  1,500 additional false alarms, an increase of 1.3%. In grid operations a missed fault is far more
  expensive than a false alarm, so this trade is favourable.
- The gated score, **0.6253, exceeds what the raw model reaches on that grid even when it is
  allowed to tune its threshold against the answers (0.5721).** A layer bolted on after training,
  changing nothing inside the model, pushed an unseen-topology result past the model's own
  best-case ceiling.

**Intervention precision is flat — 0.938 / 0.922 / 0.934.** The gate is right about 93% of the time
on the grid the model was trained on, and about 93% of the time on grids it has never seen. What
varies instead is **reach**. On the training grid the model is good, so it rarely hands the gate a
wrong "secure" claim to overturn (0.31%); on WCCI 2022 it is wrong roughly ten times as often in
exactly the way the gate can observe (3.04%), which is why the same rulebook is worth +0.0676 there
and +0.0082 at home. §5.4.1 develops this relationship, which is the central one in the chapter.

Every verification check passes on all three grids: **zero** evaluation errors, **zero** rules that
could not be evaluated, 100% rule applicability, and a false-block rate on healthy frames of
0.36% / 0.007% / 0.47% (§13.5).

### 5.1.7 What the three comparisons establish jointly

Read together, §5.1.3–§5.1.6 support a narrower and more carefully bounded claim than the raw model
result alone would suggest.

1. **The neural component's accuracy is uneven across topology, and speed does not repair that.**
   It matches the single-rule baseline at home and falls below it off-distribution, most severely on
   case14, where it loses to answering "violation" every time (§5.1.3). What it earns instead is
   cost: where it is accurate, it reaches that accuracy at a small fraction of what the AC solver
   that produced its labels would have cost to run directly, and the gap widens with grid size
   (§5.1.4). Speed and accuracy are separate properties; the model has the first unconditionally and
   the second only some of the time.
2. **Learned architectures are not fully transferable off-distribution.** Logistic regression, the
   MLP, and the GNN each fail to clear the best single rule on at least one unseen grid — most
   severely the GNN on case14, where it falls below even the all-positive baseline (§5.1.3,
   §5.1.5). The analytical control (a threshold with nothing to transfer) does not fail this way
   (§5.4.4). Random forest is the exception: it clears the rule baseline on both unseen grids
   (§5.1.5), so the failure is common among the architectures tested here, not universal — and it
   is not specific to graph structure. The worst transfer result (the GNN, on case14) and the best
   (random forest, with no graph at all) sit at opposite ends of the same comparison.
3. **The symbolic gate's accuracy does not depend on topology, and a learned model's does.** That
   is the contribution. It is measured rather than asserted, and — importantly — it does not depend
   on whether the model being gated is the best available screener. The gate was evaluated against
   the model it guards, not against the state of the art, and its property would hold over any model
   whose errors take the same form. What the gate is bounded *against* is measured separately in
   §5.2.11: a rule set fitted directly on the answer, every rule the gate's language can express,
   and a set written by hand from doctrine.

---

## 5.2 Analysis of Design Solutions

### 5.2.2 Model architecture: what was removed, and why

**Global pooling was removed.** Pooling means summarising the whole grid into a single vector
before predicting. That destroys the very distinction the task is about. Measured over every frame
of all three datasets (22,000 frames), **96.4–98.8% of frames are strictly mixed** — meaning that
within one grid state, some contingencies violate and others do not:

| grid | strictly mixed | all contingencies violate | none violate |
|---|---:|---:|---:|
| NeurIPS 2020 | **98.80%** | 1.20% | **0.00%** |
| case14 | **98.48%** | 1.52% | **0.00%** |
| WCCI 2022 | **96.43%** | 3.57% | **0.00%** |

(No frame anywhere is entirely secure, but that is a different statistic from the one above and
should not be conflated with it.) A single per-frame answer is wrong for most lines in the large
majority of frames, so the model instead reads **one output per line, taken from that line's
own edge**. The same choice is what makes the trained model topology-agnostic, since a 20-line grid
and a 186-line grid both run on it without modification — there is no layer whose size depends on
how big the grid is.

**Raw endpoint features are concatenated into the readout.** Three rounds of attention are free to
transform the endpoint headroom and absolute-power sums into something unrecognisable, but those
two quantities *are* the physics of post-contingency redistribution: the outaged line's flow has to
be absorbed by spare capacity at its endpoints. Passing them to the final stage unmodified lets it
form the ratio of flow to spare capacity directly, rather than relying on that information having
survived the encoder.

**GATv2 was used rather than GAT.** Both are attention mechanisms — ways of letting a node weigh
its neighbours differently rather than treating them all alike. The difference is when the scoring
happens: GAT scores attention before combining the two nodes' features, which on a graph this small
causes the scores to collapse towards uniformity; GATv2 scores after combining, and preserves the
distinction.

**A normalisation defect was found and fixed, and it was worth more than any architectural
change.** Feature normalisation rescales inputs so that quantities measured in different units are
comparable. The function computing those statistics populated an internal cache in the graph
library, with the result that the normalisation applied afterwards never reached the data loaders:
training ran on raw, unnormalised features while the statistics file was written as though it had
been applied. Clearing the cache raised validation F1 from **0.4751 to 0.8987**
(`gnn_n1_tightening.md` §3). The N-1 model was trained after the fix.

### 5.2.3 The message-passing ablation, and the caveat §5.1.5 places on it

An ablation removes one component and re-measures, to find out what that component was worth. The
component here is *message passing*: the step in which each node updates itself using its
neighbours' information, which is what makes a graph network a graph network rather than an
ordinary one. Removing it — so that each line is scored from its own features alone, with nothing
from its neighbours — drops validation F1 from **0.8987 to 0.8411, a contribution of +0.058** from
the graph structure.

⚠ **This is the one headline figure in the chapter that is not currently traceable to an artifact on
disk.** The head-only checkpoint was not retained, so the pair above cannot be re-scored; restoring
it requires a retraining run rather than a re-measurement. It is also a single-seed figure. The
in-distribution four-seed standard deviation for this architecture is 0.007 to 0.009 (§5.3.6), so a
contribution of +0.058 has roughly six standard deviations of room and is very likely to reproduce —
but it has not been reproduced, and it is reported here with that stated rather than implied. The
claim that rests on it is bounded accordingly: §5.4.4's transfer result is carried by the tabular
baselines and the analytical screen, not by this ablation.

That result stands as measured, but §5.1.5 constrains what may be concluded from it.
In-distribution, a random forest on the same raw features reaches a score indistinguishable
from the full model's; and off-distribution the graph model degrades **worse** than the forest (−50%
against −40% at the best-case column). Message passing over a topology-specific graph contributes
accuracy on the grid it was fitted to, and appears to make transfer *worse* rather than better.
That is consistent with the diagnosis developed in §5.2.4 — that the model learns one network's
particular redistribution pattern and carries it to a grid where that pattern does not hold — a
diagnosis §5.2.4 then tests over four seeds and finds real but insufficient.

### 5.2.4 The reactance experiment — testing the diagnosis rather than arguing it

§5.1.4 explained the cross-topology collapse by pointing to a missing input. Where power goes when
a line trips is determined by the branch reactances; the model was never shown them; so on an
unseen grid it has nothing from which to recompute the redistribution. That diagnosis was
circumstantial, and it is the strongest available attack on §5.4.4's claim, so it was tested rather
than left as an argument.

Two versions of the model were trained from scratch under identical conditions — 30 epochs, batch
size 128, learning rate 3e-4, the same split, and the same graphs built once with nine edge
columns, the control version simply slicing the ninth off. (The split is by *chronic* — one
continuous simulated scenario — rather than by frame, so that no scenario can appear on both sides
of it.) Constructed this way, the two
cannot differ in graph construction, example ordering or data split. The deployed model could not
serve as the control, because the training run that produced it is not recoverable from disk.

The added feature is `log1p(b / median(b))`, where `b` is the line's *susceptance* — the reciprocal
of its reactance, and so the same physical information in inverted form. It is used in preference
to the raw value, and the reason is worth stating because it is what makes the experiment portable
across grids. Raw medians span
5.13 / 985.22 / 1156.77 across the three grids — a roughly 200x spread caused by differing
`baseMVA` conventions rather than by any physical difference. DC redistribution is invariant under a uniform
scaling of every susceptance, so dividing by each grid's median discards exactly the quantity the
analytical method also ignores, leaving only the relative differences that carry the physics.

**The experiment was repeated over four seeds (42, 0, 1, 2), and that decision changed the
result.** Physics arm minus control arm, mean over four seeds with the observed range:

| grid | held threshold | best case |
|---|---|---|
| NeurIPS 2020 | **+0.0020**  [−0.0044, +0.0063] | **+0.0009**  [−0.0047, +0.0072] |
| case14 | **+0.0089**  [−0.1009, +0.0836] | **+0.0677**  [+0.0429, +0.0854] |
| WCCI 2022 | **−0.0051**  [−0.0151, +0.0081] | **−0.0103**  [−0.0225, +0.0112] |

**What is real: the reactance improves the model's *ranking* on case14.** The best-case delta is
+0.068 and positive on all four seeds (+0.043 to +0.085), against a within-arm standard deviation
of about 0.016. Ranking is what the best-case column measures — whether the model orders
contingencies correctly, independently of where the cutoff is placed.

**What is noise: everything else.** The two other grids move by less than a hundredth in either
direction, inside their own seed spread. And at the *held* threshold the case14 delta ranges from
**−0.1009 to +0.0836** — the sign is not stable, so a single seed's apparent +0.065 is not a result
and is not quoted as one here.

**What settles the question: it is not enough.** The best case14 score any physics seed
reached is **0.5339, still below the single-rule baseline of 0.5392**.
Every seed, every arm, every column: **case14 still fails.** Handing the model the reactances
closes none of the gap to a single
threshold on the removed line's loading.

**A third finding falls out of the held-versus-best-case split, and it is systematic.**

| arm | case14 best case | case14 held | gap |
|---|---:|---:|---:|
| control | 0.4558 | 0.4318 | 0.024 |
| physics | 0.5235 | 0.4407 | **0.083** |

**The added feature improves the ranking and makes threshold transfer worse** — the physics arm's
held-to-best-case gap is 3.5x the control's. The mechanism is the one described in §5.1.3: giving the
model a new input shifts the range of scores it produces, so a cutoff chosen on the home grid lands
somewhere else on a foreign one. This is a concrete instance of a general problem worth stating in
its own right: **an off-distribution improvement visible only in the best-case column is not
deployable**, because the best-case threshold is chosen with the answer key.

The reading was fixed in advance of the run, and this is the third of three anticipated outcomes —
*the input helps and does not rescue*. §5.4.4's claim therefore survives its strongest attack: the
transfer failure is not simply a missing-input problem. The model was handed the electrical
parameters, in a form that transfers cleanly across all three grids, and recovered a tenth of the
gap to a method that computes those parameters directly.

> **Learned screeners trained on one topology did not transfer, and giving the model the branch
> reactances did not repair it. The parameters improve the model's ranking on the worst grid
> (best case +0.068, four seeds) without making it deployable there — it remains below a single
> threshold on the removed line's loading.**

The experiment also produced a side result worth keeping. The seed-42 control arm scores best case
**0.8978 / 0.4473 / 0.5727** against the deployed model's **0.8972 / 0.4477 / 0.5721** — within
0.0006 on all three grids. The deployed model's epoch count is not recoverable from disk, because
the trainer saves only a bare weight dictionary and writes no metrics. This does not prove the
weights match, but it does establish that **the documented intended command reproduces the deployed
model's cross-topology behaviour**, and there is now a second model that behaves like it.

⚠ **The seed instability this run exposed is not confined to it.** §5.3.6 carries the consequence,
which applies retroactively to every single-seed model figure in this chapter.

### 5.2.5 The extraction pipeline, and an exhaustive account of its yield

The pipeline runs in four stages, and it is worth naming what each one does before quoting what
came out of them.

1. **Extraction** (Qwen3). Reads the standards documents and pulls out anything that looks like a
   rule, in whatever words the document used. Deliberately open-ended: no restriction on vocabulary.
2. **Translation.** Rewrites each candidate as a condition over the **fourteen quantities the
   simulator actually reports** — loading, voltages, powers, line status and so on. A rule that
   names something outside that list cannot be translated, and is recorded as such with the reason.
3. **Polarity guard.** A deterministic check, with no language model involved, that a rule fires in
   the right direction — that a rule meant to detect danger does not in fact fire on healthy grids.
4. **Validation** (Nemotron-3 Nano). Reads the original clause and the translated condition and
   judges whether the second is a faithful rendering of the first.

End to end the translation yield is **2,463 candidates → 58 expressible against the simulator's
vocabulary, 2.4%** — that 58-rule set is the corpus the shield holds. The guard and validation
stages that follow do not shrink it further; they decide what each of the 58 is allowed to do at
inference. The guard stage keeps **32** of the 58 (rejects anything that fires on a healthy grid),
and validation against the source standards is what earns a rule the **BLOCK** channel — only the
thermal-loading family clears that bar. Everything else the 58 contains speaks in **WARN** or
**NORMAL**, explaining a prediction the gate lets through rather than staying silent on it.

A low BLOCK yield invites an objection immediately: *a pipeline that turns 2,463 candidates into
a corpus where only the thermal-loading check ever blocks did not find a signal, it found noise.*
The answer is
not a better yield. It is that the yield was never the finding — the **partition** is. Every
candidate is assigned to exactly one terminal bucket, and the partition is asserted at each stage
rather than assumed (§19.1):

| terminal fate | n | share |
|---|---:|---:|
| not expressible — time / dynamics | 635 | 25.8% |
| not expressible — frequency | 617 | 25.1% |
| not expressible — quantity not modelled by the simulator | 562 | 22.8% |
| not expressible — equipment-internal | 315 | 12.8% |
| **not expressible — scope / aggregation  [recoverable]** | **142** | **5.8%** |
| not expressible — administrative / process | 115 | 4.7% |
| expressible, rejected by the polarity guard | 26 | 1.1% |
| expressible and clean, rejected by the validator | 21 | 0.9% |
| not expressible — unclassified | 13 | 0.5% |
| **validated and served to the shield** | **11** | **0.4%** |
| not expressible — free variable, no number | 6 | 0.2% |
| **total** | **2,463** | **100.0%** |

**The emphasised row carries the argument.** `scope / aggregation` marks rules whose quantity *is*
present in every record, but which the flat fourteen-variable vocabulary collapses into a grid-wide
minimum or maximum — a rule about one particular substation, for instance, when the vocabulary only
offers the worst substation anywhere. Those 142 are blocked by vocabulary design, not by physics,
and they are the only bucket recoverable without changing simulator. Everything above them is a
property of what a quasi-static power-flow simulator represents at all.

**The mismatch runs in both directions, and the second direction is the more damaging.**

*Forward*: **1,252 rules — 50.8% of the corpus — are blocked on just two missing capabilities,
frequency and time.** Counted two ways, and the two now reconcile exactly
(`evaluation/capability_gap.py`): scanning every untranslatable candidate's stated reason for a
frequency pattern finds 617, for a time/duration pattern finds 752, and 117 candidates carry both
— a union of 1,252. That union is identical to the exclusive partition above (635 time-bucketed +
617 frequency-bucketed, first-match-wins with frequency checked first), because a candidate that
mentions both is exactly the 117 the exclusive count folds into "frequency".

*Reverse*: of the fourteen quantities the simulator *does* measure, the 58-rule corpus reads five
(loading and a handful of voltage and power-factor variables), and only loading is calibrated
precisely enough to reach BLOCK. **Nothing in the corpus governs topology** — no rule mentions line outages,
which is the entire subject of the task. The reason is structural rather than technical. Grid codes
govern **connection and equipment compliance**: what a generator must do to be allowed to connect,
what a protection relay must be set to. Contingency screening is an **operational** activity, and
where standards address it at all they do so as process requirements — *"the operator shall perform
studies annually"* — rather than as conditions on live telemetry.

> **A safety layer built from connection codes cannot govern operations, because the corpus was
> never about operations.** That claim is stronger and more defensible than "the simulator is too
> coarse", because it does not require the simulator to be at fault, and it generalises to any
> quasi-static power-flow simulator.

A further structural reason applies to the American standards specifically. NERC standards are
**criteria-referencing by design**: they say *"within its applicable Facility Rating"*, and a
separate standard requires each utility to maintain its own documented method for computing that
rating. The number is deliberately never printed on the page, and no amount of prompt engineering
extracts a threshold that does not exist. This is directly visible in the output — the two
documents genuinely about post-contingency performance produced rules of the form
`loading_pct > relay_loadability_limit_pct`: correct in form, with the threshold left free.

#### Why the frequency and time rules could not be used

Slightly more than half the corpus — **457 frequency rules** (e.g. *"if system frequency falls
below 59.4 Hz, disconnect within 2 seconds"*) and **103 voltage ride-through rules** (a depth-and-
duration pair, e.g. *"must not disconnect for a dip to 0.45 pu lasting less than 0.15 s"*) — could
not be translated, because Grid2Op is *quasi-static*: it solves one snapshot of the grid every 300
seconds and represents neither system frequency nor sub-second transients at all. That is a
statement about expressibility, and it invites the obvious rejoinder: *use a simulator that does.*

**ANDES 2.0**, a dynamic simulator that does represent both, was the specific remedy, and building a
pipeline around it was estimated at two to four weeks. Rather than spend that, the question was
reduced to an afternoon's measurement: **does an N-1 event move these quantities enough to matter at
all?** Being evaluable and being capable of firing are different questions — 1,252 of the 2,463
candidates (50.8%) are blocked for naming frequency or sub-second timing, so this was the single
largest lever available if it worked.

It did not. On IEEE-14 with governors present (60 Hz base, permanent trips, a generator-trip
positive control to confirm the instrument could detect an effect when one exists):

| check | result |
|---|---|
| worst frequency excursion from any N-1 line trip | 0.33 Hz |
| mildest frequency threshold anywhere in the corpus | 0.60 Hz — **not reached** |
| what it would take to reach it | an N-2 event (0.80 Hz) — one full contingency order beyond the task |
| deepest voltage dip from a clean line opening (no fault) | 0.0057 pu short of the mildest ride-through bar — **not reached** |
| voltage envelopes entered under an actual short-circuit fault | yes, but recover within ~50 ms of clearing — the corpus's bands need 140–160 ms **post-clearing**, which none reach |
| candidates blocked on frequency/time → would ever fire under ANDES | **1,252 → 3** |

**Frequency:** losing a line redistributes flow but changes neither generation nor demand, so the
generation/demand balance that drives frequency is untouched — these rules are calibrated for
N-2-and-beyond emergencies, not N-1. **Voltage:** a ride-through envelope describes a *moment* during
a fault, not the *settled state after* a line is lost, which is what a shield reading telemetry
actually observes — right physics, wrong instant. The voltage margin is the thinner of the two
findings (under 1% from being reached by a clean opening, against frequency's roughly 2x miss), so a
heavier loading condition could plausibly cross it; frequency should be read as settled, voltage as
measured-but-narrow. Three rules (a post-fault over-voltage overshoot) are the one genuine positive,
and even those need ANDES running at inference — a five-minute snapshot cannot represent a
one-second transient.

> **Grid-code frequency limits are calibrated for N-2-and-beyond emergencies, and ride-through
> limits for fault transients. N-1 thermal screening reaches neither, and no simulator upgrade makes
> them applicable, because the gap is subject matter, not instrumentation.**

This generalises beyond this project: the same argument applies to any quasi-static contingency
screen built from connection-code rules. Full experimental arms, the escalation ladder, and every
supporting number: `thesis_findings.md` §20.2.

### 5.2.6 Filtering the corpus improved the result — and removing 28 of 32 rules is why

The validated BLOCK channel beats the unvalidated 32-rule guard-stage set on **every metric on
every grid** (§5.1.6). The trade is explicit: validation cost a little coverage (corrections
427 → 331 at home, and 21,306 → 21,065 on WCCI 2022) and bought a large reduction in bad blocks
(regressions 404 → 22, and 3,471 → 1,494). On case14 the outcome is bit-identical, because the
removed rules never fired there in the first place.

The mechanism is worth stating as a general property of this kind of pipeline. The 32-rule corpus
averaged two very different families: voltage rules, which were correct only 13–20% of the time
when they fired, and thermal rules, correct 92–94% of the time. On the training grid the voltage
rules happened to dominate — 478 of 831 interventions — dragging intervention precision down to
0.514, which is a coin flip.

> **A rulebook that is 90% bad rules and 10% good ones does not perform at 90% of the good version.
> It performs at the average, and the average hides the good rules completely.**

> ⚠ **Intervention precision does not rise off-distribution — it is flat once the voltage rules are
> removed.** The 0.514-at-home figure is an averaging artefact of a corpus that mixes a 92–94%
> family with a 13–20% one; once the voltage rules are removed, precision is flat at 0.92–0.94 on
> every grid (§13.2), which is the stronger and more defensible reading.

This is not a property of the extracted corpus alone. §5.2.11 measures the same effect on a
deliberately broad set of hand-written rules: fourteen of them combined score no better than having
no rule at all, while the single best scores 6.4x their union. It measures it a second way by
exhaustive enumeration — of every rule the gate's language can express, and every two-term
combination of them, none improves on one thermal threshold except by loosening that same threshold
and giving up precision. Precision is the scarce resource in a gate that fires if *any* rule fires,
and filtering is how it is conserved.

### 5.2.7 The voltage contract, and a defect it exposed

The shield converts line voltages into per unit using **per-line base kV from a backend-derived
sidecar, with energised lines masked out**, rather than dividing everything by one nominal value.
The *base kV* is the nominal voltage of that particular line, and it is the divisor that turns its
measured kilovolts into a per-unit figure. A
single constant is wrong by an order of magnitude on these grids: case14 runs lines at roughly
20 kV and 138 kV, and even the 36-bus grid carries seven lines near 365 kV. A flat divisor produces
close to 100% false blocks.

The measurement that followed is a finding rather than a fix. **These grids operate at 1.05–1.08
per unit, so a healthy frame reads about 1.06, not about 1.00.** Grid codes specify a ±5% band
around nominal. The official rule, applied unmodified to the simulator, therefore fires on
perfectly healthy grids — which is exactly what the voltage rules of §5.2.6 were doing. §5.4.6
returns to this, because the same predicate turns out to be informative on one grid and vacuous on
another.

### 5.2.8 A validator that fabricated its reasoning, and why rejection reasons are now persisted

The validation stage first rejected **31 of 32** rules and recorded no reason for any of them,
because the implementation counted rejections and discarded the verdict object that explained them.
That looked like a verdict on the corpus. A controlled A/B comparison over identical input showed
it was a verdict on the question being asked:

| arm | question put to the validator | confirmed | rejected | distinct |
|---|---|---:|---:|---:|
| `strict` (reproduces run 1) | is the constraint **stated** in the source text? | 1 | 31 | 1 |
| `translated` | is it a faithful **operationalisation** of the requirement? | **11** | 21 | **4** |

The outcome is perfectly nested — every rule the strict arm kept, the translated arm kept too — and
the whole difference is the ten `loading_pct > 100` rules. Criterion 1 of the strict prompt was a
category error when applied to a translated condition, whose entire purpose is to leave the
standard's language behind. Nothing the preceding stage produces is ever *stated* in the source
text, so the strict question could only ever be answered "no".

**What makes this a methodological result rather than a bug report is how the two runs differed.**
The strict arm rejected those rules citing *"an undefined variable `loading_pct`"* — a variable
listed in the vocabulary block of that very prompt, two paragraphs earlier. The verdicts were
perfectly stable across runs while the stated reasoning was invented. **That is invisible unless
the reasons are stored, and the first run had thrown them away.** Rejections now persist with their
reason attached, and the run logs an error if the rejection rate exceeds 80%.

**The fabrication is not confined to the strict arm, and that was established only by auditing the
stored reasons.** Every one of the 21 rejections shared by both arms was re-read against its source
chunk and its condition, and each rejection's stated reason was checked mechanically where a
mechanical check exists. Two of the 21 rejections in the **`translated`** arm — the arm that
produced the served corpus — assert a vocabulary breach that provably does not occur: `R_069` and
`R_1926` both reject conditions over `voltage_pu_min`, stating that *"the condition uses an invalid
variable"*, when `voltage_pu_min` is a member of the closed vocabulary. This is the same failure
mode as the strict arm's fabricated `loading_pct` complaint, in the arm the thesis relies on. The
distinction to carry is therefore not *"the strict prompt fabricated and the translated prompt did
not"*, but **the sharper and less comfortable one: verdicts were stable across both arms while the
stated reasoning was unreliable in both.**

A second failure is visible without any vocabulary check. `R_1444` rejects
`voltage_pu_min < 0.9 or voltage_pu_max > 1.1` on the grounds that the source specifies *"a voltage
deviation of ±10% (i.e., 0.90–1.10 pu)"* and that the condition *"incorrectly sets the lower bound
at 0.9 and upper bound at 1.1, deviating from the ±10% range"* — a sentence that contradicts itself
across its own clauses. A third, `R_797`, concedes the thresholds are *"correct"*, identifies the
repair needed as a role change, states that the rule is *"requiring correction"*, and returns REJECT
regardless.

Adjudicating all 21 gives **11 sound rejections, 6 wrong, and 4 that should have been corrected
rather than rejected**. The substantive majority is real — PRC-024 ride-through curves misread as
instantaneous limits, ENTSO-E time-bound operating envelopes, and rules whose direction had been
inverted — but it is a slimmer majority than eleven of twenty-one suggests when stated as *most*.
⚠ These counts are a **reading**, not a measurement, and the artifact separates the two: the
validator's verdict and verbatim reason are recorded mechanically, and the classification is
recorded beside them with its basis, so a reader may reject an individual call without discarding
the audit. The vocabulary-breach finding above is the exception — that one is set membership, and it
is not a matter of judgement.

**A pre-registration paid off.** Before the stage ran it was recorded in advance that the shield
numbers should come out identical again *unless validation rejected `loading_pct > 100` itself* —
which would be a substantive finding about the validator rather than numerical drift, and would
have to be investigated rather than absorbed. That is exactly what happened. Had run 1 been
accepted at face value, the served corpus would have held a single affirmation rule, which can
never block, and **the shield would have been silently reduced to doing nothing at all.**

### 5.2.9 Gate design: asymmetry, and the channels that let non-blocking rules speak

The gate blocks only over-permissive verdicts. Affirmation rules — those that confirm a state looks
normal — supply supporting evidence and can never block on their own.

Version 1 of the shield had exactly one output channel, which is why **54 of 58 expressible rules
were discarded**: a rule was thrown away whenever it could not justify a veto, because vetoing was
the only thing a rule was permitted to do. That is a property of the gate's design, not of the
rules. Version 2 re-partitions all 58 by measured behaviour into five channels — BLOCK, WARN,
NORMAL, NOT_APPLICABLE and INERT — yielding **41 rules speaking across 17 distinct conditions**,
up from the handful of distinct conditions validation alone had confirmed. That is a 5.7x increase
in what the system can say, and 54 of 58 rules documented rather than dropped.

Two safeguards make this additive rather than risky.

- **Only the BLOCK channel can veto, and that is structural rather than conventional.** Rules are
  sorted into channels before evaluation, and the veto loop iterates the BLOCK group alone. An
  unrecognised channel resolves to NOT_APPLICABLE and never to BLOCK, and a rule carrying no
  channel at all retains version-1 behaviour exactly. This is pinned by a test, because if a
  warning rule could reach the veto path the 92–94% intervention precision would move.
- **A warning may not ship without its calibration.** The build refuses any corpus whose warnings
  carry no measured firing rate, or whose rate does not distinguish risky states from safe ones.
  This caught real defects: **two predicates came back inverted**, with the entire power-factor
  family firing *more* often when N-1 risk was *lower*, so five records were demoted out of the
  warning channel on the strength of the measurement.

**Every metric arm is identical between the earlier BLOCK-only file and the 58-rule corpus on all
three grids** — every scalar, every count — so nothing in §5.1.6 depends on the channel work.

### 5.2.10 The knowledge graph, and the version of it that was deleted

The knowledge graph answers one question: *when the gate blocks, on whose authority?* It records
the chain `Document → Clause → Rule → ServedRule → Predicate → Variable`, so a block can name the
exact section of the exact standard behind it, together with every other standard that says the
same thing.

It was built last, deliberately. A first attempt, made before the corpus was settled, organised the
graph around the physical grid — a node per bus, a node per line, and rules attached to whatever
component they mentioned. That failed twice over. Only 3.5% of extracted rules named a specific bus
or line at all, so **6,375 of its 6,632 edges carried no information**; and because it was built
from one grid's wiring, it could not serve the other two topologies even though the rules
themselves are grid-independent. It was deleted rather than archived.

The replacement was built holding **36 nodes and 40 edges**: one `ServedRule` node per record in the
validated BLOCK set, and the `Document → Clause → Rule → ServedRule → Predicate → Variable` chain
behind each. It is topology-agnostic by construction so that one graph serves all three grids, and
every edge means something. **Retrieval through it is optional and returns a provably identical rule
set**: every field of the graph-served results matches the flat-file results on all three
topologies, which is pinned by a test. The one prohibition is routing by entity — the same physical
check is labelled `Facility` in one standard and `Line` in another, so routing on that label
silently drops rules. That, plus binding rules to one grid's component indices, is what made
version 1 useless.

**The graph has since grown, and `kg/knowledge_graph.json` on disk today holds 171 nodes and 236
edges, not 36 and 40.** It was rebuilt with an `ExplanatoryRule` layer that gives the same
provenance chain to all 58 rules in the shield's explanation-channel corpus (§5.2.9), not just the
records that can block, which is what pulled in more documents, clauses and predicates than the
BLOCK-only file alone cites. The corroboration claim in §5.4.5 does not depend on the smaller
count — re-tracing it from the current, larger file returns the identical figures: three served
records, 10 clauses, 4 documents, 2 identified bodies for the thermal predicate. The growth is
additive provenance for rules that can now speak but never veto, not a change to the 58-rule
corpus the shield acts on.

---

### 5.2.11 Was the thin corpus the pipeline's fault or the task's? Three controls

Everything above argues that the thin BLOCK channel is a property of the standards rather than a
failure of the extraction. Every one of those arguments, however, is made from *inside* the pipeline that
produced the corpus — which is precisely the asymmetry a sceptical reader finds first. Two controls
were specified to settle the question from outside, both have been run, and a third and stronger
instrument has since replaced the first. None of the three uses the extraction pipeline at all.

All three are scored on the error the gate exists to catch — *the model predicted secure and the
contingency was in fact a violation* — over identical rows, at the held threshold, at batch 64.
Because a gate can only act where the model predicted secure, the reference point is not "flag
everything" but **flag every eligible contingency**, which is what a gate carrying no rule at all
would do.

#### A decision tree fitted directly on the answer

A decision tree is a sequence of threshold tests arranged as a branching diagram; "depth 4" means
at most four tests before reaching a verdict. Fitting one directly on the answer estimates the best
any rule-shaped method could do over the available variables, because the tree gets to choose its
own thresholds with the labels in hand — an advantage no extracted rule enjoys.

This tree was restricted to the **fourteen variables of the shield's own context namespace** — the
exact vocabulary every extracted rule is written in, enumerated in the script so the tree cannot
acquire a feature the rules never had. It was fitted on the NeurIPS 2020 **training** split alone
(8,397 frames, 491,684 contingencies) and scored on its test split and on every frame of both
foreign grids, with its decision threshold tuned on the validation split.

| grid | flag every eligible *(no rule)* | tree, held | tree, best case *(ceiling)* | **extracted shield** |
|---|---:|---:|---:|---:|
| NeurIPS 2020 | 0.0434 | 0.2508 | 0.2667 | **0.2765** |
| case14 | 0.3668 | 0.0179 | *0.3669* | 0.0102 |
| WCCI 2022 | 0.2352 | 0.0025 | *0.2352* | **0.4644** |

**This fitted ceiling does not beat the extracted shield's BLOCK rule.** On the home grid the two reach the
same errors — 330 and 331 out of 2,041 — and the shield reaches them at precision 0.938 against the
tree's 0.558, flagging 353 contingencies where the tree needs 591.

⚠ **That result is real, but narrower than it first appears, and the subsection after next replaces
this instrument entirely.** A decision tree of this kind is fitted *greedily*: it takes the single
best test, then the best test beneath it, and never reconsiders. A rule that works only as a pair —
where neither half is useful alone but the conjunction is — is structurally invisible to such a
search. What this arm licenses is therefore the narrow statement *greedy trees over these variables
do not beat the corpus*, and not the general one.

**On both unseen grids the italicised ceiling equals the no-rule baseline to within one
ten-thousandth.** The tree's best achievable score at *any* threshold is obtained by flagging
everything, which is to say its ranking carries no usable information off-distribution whatever. It
holds real signal on the grid it was fitted on — +0.22 over the baseline — and none at all on
either grid it was not.

The obvious objection is that a shallow, unweighted tree on a target with only 1.8% positives is a
strawman. Two further configurations answer it, both more generous than the specification:

| configuration | NeurIPS 2020 | case14 | case14 *best case* | WCCI 2022 | WCCI 2022 *best case* |
|---|---:|---:|---:|---:|---:|
| depth 4, unweighted | 0.1755 | 0.0154 | *0.2712* | 0.0025 | *0.1682* |
| depth 4, class-balanced | 0.1867 | 0.0157 | *0.2712* | 0.3749 | *0.3749* |
| depth 8, class-balanced | 0.1758 | 0.0141 | *0.2712* | 0.1080 | *0.2068* |

> These three rows are scored over **all** rows rather than under gate semantics, so they compare
> with one another and not with the table above. They are kept in that form because the question
> they answer is about the tree's own fitting rather than about the gate.

**case14 is immovable**: every configuration lands on exactly the all-positive baseline, 0.2712.
**More capacity makes transfer worse** — depth 8 beats depth 4 nowhere and is substantially worse
on WCCI 2022, which closes the "too shallow" objection from the opposite direction. The comparison
also favours the tree throughout, since the tree is granted a tuned threshold while the shield's
predicate has no tunable parameter at all.

The variables the tree chose are a result in their own right. ("Importance" here is the share of
the tree's discriminating power attributable to each variable.)

| variable | share of the tree's importance |
|---|---:|
| **`loading_pct`** | **0.614** |
| **`voltage_pu_max`** | **0.152** |
| `current_a_max` | 0.068 |
| `active_power_mw_max` | 0.064 |
| `voltage_pu_min` | 0.050 |
| three remaining variables | 0.052 |

The two dominant variables, carrying 77% of the tree's importance between them, are exactly the two
families the extraction pipeline recovered: the thermal check and the voltage band (§5.4.7).

#### Every rule in the language, enumerated

The objection a fitted tree cannot answer is that it searched badly. The way to close it is to stop
searching and enumerate instead: score *every* rule the shield's language can express, and read the
best one off the list. That is feasible here for a reason particular to this gate — rules read
**per-frame** quantities while the target is per-contingency, so a rule is a mask over 1,932–6,000
frames rather than over 113,205–742,472 contingencies. Aggregating each frame's target and eligible
counts once reduces the score of any rule to two dot products, and every threshold on a variable can
then be scored at once with a sort and two running sums.

Three arms, all under the protocol above unchanged:

| arm | what it covers | scale |
|---|---|---|
| every single-variable rule | every distinct cutpoint present in the data, both directions, all fourteen variables | exhaustive, not sampled |
| every two-variable rule | all pairs on a 32-quantile grid, both directions per side, joined by AND and by OR | 252,405–299,925 pairs per operator per grid |
| an uncapped ensemble | a gradient-boosted ensemble with **no depth limit**, 300 iterations, on the same fourteen variables | — |

The third arm is deliberately *not* a rule. It is unreadable, so no gate could ship it. That is the
point: it bounds the corpus from above by something strictly more expressive than anything the rule
language contains. The winning rule on each grid is re-scored the slow way over the full
contingency vectors, and the procedure aborts if the two disagree beyond one part in a billion.

⚠ The first two arms select their winner **using the answer key, on the grid being scored**. That
is a best-case selection and it is deliberate: the comparison is meant to be generous to the
challenger, which is what makes a narrow victory informative.

| grid | flag every eligible *(no rule)* | **best rule in the language** | best two-term rule | uncapped ensemble, held / best case | **extracted shield** |
|---|---:|---:|---:|---:|---:|
| NeurIPS 2020 | 0.0434 | **0.3269** | 0.3140 | 0.2771 / 0.3052 | 0.2765 |
| case14 | 0.3668 | *0.3937* | *0.4017* | 0.0052 / *0.3668* | 0.0102 |
| WCCI 2022 | 0.2352 | **0.5079** | 0.5066 | 0.2911 / 0.3172 | 0.4644 |

| grid | the best rule in the entire language | frames on which it fires |
|---|---|---:|
| NeurIPS 2020 | `loading_pct > 96.79` | 2.4% |
| WCCI 2022 | `loading_pct > 97.21` | 9.0% |
| case14 | `active_power_mw_max > 33.92` | **71.1%** — degenerate |

**A better rule than the extracted one exists, and it is the same rule.** On both grids where a
gate means anything the optimum is `loading_pct`, with its threshold about three points *below* the
standards' round 100. The margin is +0.0504 on NeurIPS 2020 and +0.0435 on WCCI 2022.

That margin needs qualifying rather than reporting bare, because the two rules are not the same job
done at different quality — they are two points on one curve.

| grid | rule | precision | recall | contingencies flagged |
|---|---|---:|---:|---:|
| NeurIPS 2020 | best in language, `> 96.79` | 0.566 | 0.230 | 828 |
| NeurIPS 2020 | **served corpus, `> 100`** | **0.938** | 0.162 | **353** |
| WCCI 2022 | best in language, `> 97.21` | 0.744 | 0.385 | 35,293 |
| WCCI 2022 | **served corpus, `> 100`** | **0.934** | 0.309 | **22,559** |

The enumerated optimum buys recall by spending precision: on NeurIPS 2020 it raises nearly half its
alarms falsely, against six percent for the served rule. F1 weights precision and recall equally and
therefore prefers the looser cutoff, but a gate whose function is to *override a model's output*
plausibly should not. **The defensible statement is narrow: the standards' threshold of 100% is
slightly conservative for this task, and the cost of that conservatism is about 0.045 F1, bought
back as a 0.37 gain in precision.** It is not that the corpus was beaten by a better rule — the
predicate is identical, and only the constant differs.

**That preference has since been priced, and it is not a neutral default.** The claim that a gate
overriding a model should favour precision was, until it was measured, an assertion. Sweeping the
F-beta objective over every cutpoint on `loading_pct` locates the exact exchange rate at which the
two operating points swap places. F-beta weights recall β² times as heavily as precision, so the
critical β is the point of indifference between the served 100 and the F1 optimum near 97:

| grid | served 100 (P / R) | optimum near 97 (P / R) | critical β | β² | 100 is preferable only if one false block costs more than |
|---|---|---|---:|---:|---:|
| NeurIPS 2020 | 0.938 / 0.162 | 0.566 / 0.230 | **0.621** | 0.385 | **2.60 missed violations** |
| WCCI 2022 | 0.934 / 0.309 | 0.744 / 0.385 | **0.652** | 0.425 | **2.35 missed violations** |

The crossover lies **below** β = 1, which is the uncomfortable direction: at equal weighting the
looser cutoff has already won, and holding 100 requires weighting precision roughly two and a half
times recall. The honest formulation is therefore that **the served threshold is defensible, but
only under a stated preference, and the preference is now quantified rather than asserted.** An
operator who prices a false block below about 2.4 missed violations should prefer 97; one who prices
it above should keep 100. This thesis keeps 100 for reasons given in §5.5.2, and states the rate.

Two further details bound the claim. At β = 0.25 the F-beta argmax sits *above* the served value
(100.07 on NeurIPS 2020, 100.76 on WCCI 2022), so 100 is genuinely near-optimal only in a heavily
precision-weighted regime. And the first sub-100 cutpoint to overtake 100 does so at β = 0.444 and
β = 0.237 respectively — meaning that even a strongly precision-weighted operator would shave the
threshold to about 99.15 or 99.67 rather than hold it at 100. **That refinement is reported and not
adopted**: 99.15 is a number with no provenance, whereas 100 is the figure four standards bodies
independently wrote down, and the provenance is the property this corpus exists to demonstrate.

An incidental check falls out of the same sweep and is worth recording. Evaluated at exactly 100,
the sweep reproduces the shield's own counts on every grid — 331 caught of 353 blocked, 95 of 103,
and 21,065 of 22,559. The BLOCK channel is therefore *behaviourally* the single predicate
`loading_pct > 100`, confirmed from an independent code path.

Three results follow, and the third is the strongest.

**The greedy objection is dead.** Across roughly 600,000 two-term rules per grid, pairs buy nothing.
NeurIPS 2020's best pair (0.3140) is *worse* than its best single (0.3269), and WCCI 2022's best
pair is `loading_pct > 66.47 and loading_pct > 97.15` — the single rule wearing a hat, an artefact
of the coarser grid the pair search uses. There is no interaction the tree missed, because there is
no interaction.

**Unlimited capacity loses to one threshold comparison.** The uncapped ensemble, fitted directly on
the answer, scores below a single `loading_pct >` rule on all three grids (0.3052 against 0.3269;
0.3668 against 0.3937; 0.3172 against 0.5079, at the best-case column). A model free to split as deep
as it likes, under no obligation to be readable, and which could never be shipped as a rule, cannot
extract more from these fourteen variables than one comparison does. This establishes the
rule-poverty reading **from above** rather than by argument.

**The convergence result becomes overwhelming rather than suggestive.** The tree offered variable
importances; enumeration offers the actual best rule of each kind:

| grid | best variable | runner-up | ratio |
|---|---|---|---:|
| NeurIPS 2020 | `loading_pct` 0.3269 | `current_a_max` 0.1455 | **2.2x** |
| WCCI 2022 | `loading_pct` 0.5079 | `voltage_pu_min` 0.3007 | **1.7x** |

Out of every single-variable rule expressible in this vocabulary, the best is the thermal check and
the runner-up is not close. A clean internal check that the search is working: it independently
returned `loading_pct > 96.7897` and `rho_max > 0.967897` with *identical* F1 — the same predicate
found twice, in the two variables that encode it.

**case14 is confirmed dead by exhaustion.** Its best single rule fires on 71.1% of frames for F1
0.3937 against 0.3668 for flagging everything, and its best pair (0.4017) fires on 63.7%. Both
approximate "block everything the model called secure" and are scored as such. No rule, of any
provenance, at any threshold, in any pairwise combination, works on case14 — extracted, fitted,
hand-written and now exhaustively enumerated have all returned the same answer.

Three limits bounded what the enumeration proved, and two have since been closed by measurement:

| objection | check | result |
|---|---|---|
| the winning rule was picked **with the answer key** — no rule-writer could do that | re-select the best single rule on the NeurIPS 2020 **training** split alone, apply unchanged to both unseen grids | `loading_pct > 97.13` — still beats the served corpus everywhere (+0.027 / +0.008 / +0.042), bought the same way, in precision (0.611 / 0.642 / 0.731 against 0.938 / 0.922 / 0.934) |
| the pair search's 32-quantile grid might hide an interaction | re-run at 64 and 128 cutpoints (3,081 masks on the largest grid) | best pair F1 moves by ≤0.012; **no resolution beats the best single rule** on either non-degenerate grid |

Both selection rows independently found the same predicate twice (`rho_max`/`loading_pct` at
identical F1), the same internal check that appeared in the best-case-selected arm above.
**Three-term rules were not enumerated** — the space cubes — so the evidence against them is
indirect: two terms already add nothing over one, at three separate resolutions.

#### Fourteen rules written by hand

⚠ **This arm is a hand-written comparison, not a blind baseline, and that must be said before the
result.** The specification required the rules to be written before consulting any firing-rate or
calibration artifact, because rules written afterwards are the extracted corpus laundered through a
human. That ordering was not available: the served corpus, the gate's precision figures and the
tree's splits had all been seen in the same session. The arm therefore claims less than it would
have. Two mitigations, neither of which repairs that: the fourteen rules were written to disk and
**fingerprinted before any evaluation ran**, with the runner asserting that fingerprint and
refusing to proceed if it changed; and every family the fourteen variables can express is included
rather than a chosen few, with each rule labelled by basis — six `standard-stated`, three
`operating-practice`, five `doctrine`. What the arm can still show is whether a broader,
doctrine-derived rule set does better than the BLOCK-calibrated rule that survived extraction. What
it cannot show is that a human working blind would have written these.

| grid | flag every eligible *(no rule)* | all 14, OR-ed | **best single hand-written** | **extracted shield** |
|---|---:|---:|---:|---:|
| NeurIPS 2020 | 0.0434 | 0.0434 | **0.2765** | **0.2765** |
| case14 | 0.3668 | 0.3668 | *0.3668* | 0.0102 |
| WCCI 2022 | 0.2352 | 0.2358 | **0.4644** | **0.4644** |

**The best hand-written rule is the extracted rule, and it scores identically.** On both grids where
a gate is possible at all, the winner is `loading_pct > 100`, at F1 0.2765 and 0.4644 and precision
0.938 and 0.934 — the same figures to four decimal places as the served corpus, because the two
arms are evaluating the same predicate over the same rows. A human writing from doctrine and a
language model reading standards documents produced one predicate between them.

**They also produced the same small error, and that is informative.** Both wrote 100, and the
enumerated optimum is about 97. Two procedures with nothing in common — one reading published
standards, one writing from operating doctrine — independently inherited the standards' round
number rather than the task's best cutoff. The agreement is therefore evidence about where the
predicate comes from, not evidence that the predicate is optimally tuned.

*The italicised case14 entry is not a result.* Its best rule is `voltage_pu_max > 1.05`, which fires
on **100% of case14 frames** and is therefore numerically identical to having no rule at all. It is
the absence of a gate rather than a gate — and it independently reproduces the degeneracy reported
in §5.4.6, where that same voltage band is informative on one grid and vacuous on two others.

**The finding worth more than the comparison is that more rules make the gate worse.**

| grid | flag every eligible | all 14, OR-ed | difference |
|---|---:|---:|---:|
| NeurIPS 2020 | 0.0434 | 0.0434 | **0.0000** |
| case14 | 0.3668 | 0.3668 | **0.0000** |
| WCCI 2022 | 0.2352 | 0.2358 | **+0.0006** |

Fourteen rules combined with OR fire on essentially every frame, so the gate blocks every
contingency the model called secure and its precision falls to the base rate. The single best rule
scores **6.4x** the union on NeurIPS 2020 and **2.0x** on WCCI 2022.

**This inverts the natural reading of the thin BLOCK channel.** It has been treated throughout as a
limitation to be explained away. Measured against a deliberately broad hand-written set, the
thinness is *load-bearing*: the candidates that were filtered out on the way to BLOCK would have
diluted the gate rather than strengthened it. In a gate that fires if any rule fires, precision is
the scarce resource, and every additional imprecise rule spends it. §5.2.6 observed this on the
corpus the pipeline itself produced; this measures it against rules the pipeline never saw.

**One pre-registered rule, `abs(generation_load_imbalance_pct) > 5`, was NOT_EVALUABLE as run** — the
shield's evaluator had no built-in functions, so `abs` raised an error on every frame. Reported
rather than silently patched around, since repairing a pre-registered file mid-arm would defeat the
point; thirteen of fourteen rules were effective as run, and the constraint applied to the extracted
corpus equally.

The evaluator and condition linter have since admitted `abs`/`min`/`max` behind a closed whitelist,
and the rule was re-scored as a supplementary run without touching the pre-registered file or its
fingerprint. It becomes evaluable, fires on 0–7 frames per grid (F1 0.0000 / 0.0000 / 0.0084), and
loses on every grid — the same verdict as its thirteen siblings. Nothing else moves, and the finding
strengthens: the best hand-written rule is the extracted rule over all fourteen rules, not thirteen,
closed by measurement rather than by an unevaluated gap.

#### What the three controls establish, and what they do not

| control | question | answer |
|---|---|---|
| fitted tree | could a greedy search fitted *on the answer* beat the corpus? | **No** — the same recall at a fraction of the precision on the home grid, and no usable ranking on either unseen grid |
| exhaustive enumeration | could *any* rule in this language beat the corpus? | **Marginally, and only by loosening the same predicate's constant** — +0.045 F1 at half the precision; no other variable, threshold or pair comes near |
| expert rules | could a human writing *from doctrine* beat the corpus? | **No** — the best rule they write is the corpus's rule, scoring identically |

**The thin corpus is a finding about the standards/simulator mismatch rather than a limitation of
the extraction method, and this is now measured from outside the pipeline rather than argued from
within it.** §5.2.5, §5.3.4 and §5.4.7 all supported that reading by different routes, but each was
an argument about *why* the yield is low, made by the pipeline's own authors. These three are not.
The enumeration is the strongest of them, because it answers the question by exhaustion rather than
by search: what the pipeline missed is a constant on a rule it already found, and not a rule.

⚠ **No control makes the gate good, and none is offered as evidence that it does.** Recall remains
0.162 / 0.005 / 0.309, and the best rule obtainable at any threshold lifts it only to 0.230 and
0.385 at precisions of 0.566 and 0.744. On case14 no frame-level gate of any provenance — extracted,
fitted, hand-written or exhaustively enumerated — beats blocking everything the model called secure.
What the controls license is the statement *the rules that exist are close to the best obtainable
from these variables*; they never license *the rules are sufficient*.

Three caveats travel with them. The expert arm was not blind, as stated above. The enumeration
selects with the answer key on the grid it scores, where the served predicate has no tunable
parameter at all, so the comparison is generous to the challenger by construction — which is what
makes a 0.045 margin, lost on precision, the meaningful reading of it. And every target here depends
on the model's own predictions at the held threshold and therefore inherits the ±0.02
cross-topology band of §5.3.6: the case14 and WCCI 2022 findings sit far outside it, but the
NeurIPS 2020 enumeration margin of 0.0504 is comparable to it and should be quoted as *about 0.05*
rather than to four decimals. The precision gaps and the WCCI 2022 margin are far outside the
band.

---

## 5.3 Statistical Analysis

### 5.3.1 Threshold protocol, and the measured size of the optimism it avoids

Selecting a decision threshold on each grid individually requires that grid's answer key, which is
unavailable in deployment — if the answers were already known, no screening method would be needed.
The reported protocol therefore selects **once**, on the NeurIPS 2020 validation split, and holds
the value fixed everywhere. The cost of the honest protocol is directly measurable as the gap to
the best case:

| grid | held | best case | gap |
|---|---:|---:|---:|
| NeurIPS 2020 | 0.8956 | 0.8972 | +0.0015 |
| case14 | 0.4167 | 0.4477 | **+0.0310** |
| WCCI 2022 | 0.5577 | 0.5721 | +0.0144 |

**The gap grows off-distribution, and the ordering was predicted before it was measured.** When
this table was still being reported at the best-case column, the recorded prediction was that moving
to a held threshold would lower all three rows, and would lower case14 and WCCI 2022 most. The
outcome was correct in both direction and ordering: −0.0015 / −0.0310 / −0.0144. A foreign grid
shifts the range of scores the model produces, so a threshold fixed on the training grid sits
further from that grid's optimum — which also means the earlier protocol flattered precisely the
comparison the thesis rests on.

### 5.3.2 Evaluation batch size moves the levels and not the deltas

As noted in the protocol at the head of this chapter, the model uses batch normalisation without
stored running statistics, so at inference it computes statistics over whatever examples happen to
share the batch. Batch composition therefore changes the outputs. On identical data with an identical model:

| eval batch size | model F1 | shield Δ | intervention precision |
|---:|---:|---:|---:|
| **64 (pinned)** | **0.8972** | +0.0082 | 0.938 |
| 512 | 0.9255 | +0.0072 | 0.930 |

A spread of **0.028 on identical weights and identical data**. This was isolated rather than
assumed: the positive count (20,801) and the rule baseline (0.4639) are constant across every batch
size tested, so the variation is in the forward pass and not in the data.

Two consequences follow. **No model F1 may be quoted without its batch size** — which is why §5.1.5
reports the in-distribution tree-versus-graph comparison as a tie rather than a win for either.
And **the shield's delta and intervention precision are stable across the same change**, because
both the gated and ungated arms share one forward pass, so every claim in §5.1.6 and §5.4 is
unaffected. The reported protocol pins batch size at 64 and emits a warning if it is overridden.

### 5.3.3 The cost of catching more violations — the F-beta trade curve

The held threshold is chosen by maximising **F1**, which is symmetric: it prices one missed
violation exactly equal to one false alarm. The gate, by contrast, is built on the premise that
those two errors are not equally bad. Rather than invent an operator cost ratio, the entire trade
curve was measured.

The protocol is unchanged in shape — the threshold is always selected on the NeurIPS 2020
validation split and held fixed across all three grids — and only the *selection objective* varies.
F-beta with beta > 1 prices one missed violation at beta² false alarms, so beta = 2 means an
operator who would accept four false alarms to avoid one miss.

| grid | objective | threshold | F1 | recall | precision | **missed** | false alarms |
|---|---|---:|---:|---:|---:|---:|---:|
| NeurIPS 2020 | **F1 (reported)** | 0.8849 | **0.8956** | 0.902 | 0.889 | **2,041** | 2,331 |
| NeurIPS 2020 | F2 | −0.7896 | 0.8558 | 0.953 | 0.777 | **987** | 5,689 |
| NeurIPS 2020 | F5 | −2.6781 | 0.7076 | 0.984 | 0.552 | **324** | 16,596 |
| case14 | **F1 (reported)** | 0.8849 | **0.4167** | 0.435 | 0.400 | **18,592** | 21,431 |
| case14 | F2 | −0.7896 | **0.4330** | 0.502 | 0.381 | **16,390** | 26,826 |
| case14 | F5 | −2.6781 | **0.4428** | 0.600 | 0.351 | **13,168** | 36,453 |
| WCCI 2022 | **F1 (reported)** | 0.8849 | **0.5577** | 0.629 | 0.501 | **68,166** | 115,283 |
| WCCI 2022 | F2 | −0.7896 | 0.5361 | 0.689 | 0.439 | **57,132** | 162,165 |
| WCCI 2022 | F5 | −2.6781 | 0.5004 | 0.767 | 0.371 | **42,793** | 238,795 |

**The exchange rate — extra false alarms bought per missed violation avoided, measured against
F1:**

| grid | F1.5 | F2 | F3 | F5 |
|---|---:|---:|---:|---:|
| NeurIPS 2020 | 2.2 | 3.2 | 4.8 | 8.3 |
| case14 | 2.4 | 2.5 | 2.6 | 2.8 |
| WCCI 2022 | 4.0 | 4.2 | 4.5 | 4.9 |

Three findings follow.

**On the training grid the choice is cheap; off-distribution it is not.** Halving the training
grid's missed violations (2,041 → 987) costs 3.2 false alarms each and 0.04 F1. Cutting WCCI 2022's
by 16% costs 4.2 each, and 47,000 extra alarms in absolute terms.

**On case14 the F1-selected threshold is simply mis-transferred.** F1 there *improves*
monotonically as the objective weights recall — 0.4167 → 0.4330 → 0.4428 — approaching the best case
0.4477. The cutoff that is F1-optimal on the training grid is too high for case14 *by F1's own
standard*, so a large part of case14's 0.0310 held-versus-best-case gap is threshold transfer rather
than a question of cost preference. It is worth stating explicitly that **threshold choice does not
rescue case14**: even the best of these still sits below the single-rule baseline of 0.5392 and
barely above all-positive's 0.4345.

**The gate's headline property survives the change of objective.** Re-running the shield at the F2
threshold, intervention precision holds at **0.937 / 0.933 / 0.932** against F1's
0.938 / 0.922 / 0.934. The delta shrinks — a lower cutoff means fewer "secure" predictions and so
less for the gate to overturn, with blocks falling 353 → 238 and 22,559 → 18,794 — but the accuracy
of each intervention does not move. §5.4.1's claim therefore survives a change of topology *and* a
change of threshold objective.

**F1 is retained as the reported protocol**, with this table published alongside it. It is the
standard metric; both shield arms are scored identically so the delta is fair either way; and
choosing a beta at all would require an operator cost ratio nobody has measured for these grids.
The honest position is to disclose the trade rather than to pick a point on it silently.

### 5.3.4 The served threshold sits on a physical cliff, not a slope

The gate's intervention precision is not a tuned quantity, and this section shows where it comes
from. Measuring the probability of a violation given the base-case `rho_max` band, over every frame
of all three datasets:

| base-case `rho_max` | NeurIPS 2020 | case14 | WCCI 2022 |
|---|---:|---:|---:|
| 0.60 – 0.70 | 0.127 | 0.176 | 0.129 |
| 0.70 – 0.80 | 0.180 | 0.254 | 0.161 |
| 0.80 – 0.90 | 0.228 | 0.396 | 0.205 |
| 0.90 – 0.95 | 0.284 | 0.531 | 0.253 |
| 0.95 – 1.00 | 0.432 | 0.616 | 0.382 |
| **≥ 1.00 — the served rule** | **0.957** | **0.912** | **0.947** |

**It is a cliff, not a slope.** Immediately below the threshold the predicate is close to a coin
flip; crossing 1.00 it jumps to 92–97% — which is precisely where the measured
0.938 / 0.922 / 0.934 intervention precision comes from.

Two consequences follow. **Lower-threshold thermal rules cannot be admitted to the blocking
channel.** The corpus carries them at 84, 90, 95, 110, 116 and 125%. The 110/116/125 rules are
logically subsumed by `> 100` and add nothing, since anything above 110 is already above 100; the
84/90/95 rules would add coverage at 38–62% precision, trading the headline result for volume. They
belong in the warning channel, carrying the row above as their measured rate.

And **the corpus converged on the one threshold that is physically load-bearing**: four standards
bodies wrote 100, the pipeline independently kept only 100, and the grid data shows 100 is where
the predicate starts working. That reframes the thin yield — it is not thin because extraction was
weak, but because **the task has approximately one governing predicate, and the pipeline found it**.

### 5.3.5 The rule is close to a tautology, and the distance was measured

If a grid is already past its thermal limit, then losing another line obviously keeps it past the
limit. How obvious was measured rather than assumed:

| grid | P(violation) | P(violation given base overloaded) | lift |
|---|---:|---:|---:|
| NeurIPS 2020 | 18.37% | **96.66%** | 5.26x |
| case14 | 27.75% | **91.16%** | 3.28x |
| WCCI 2022 | 24.76% | **94.74%** | 3.83x |

⚠ **The NeurIPS 2020 row is scored on the same 113,205-contingency held-out test split as every
other model/shield table in this chapter**, not the full 12,000-frame generated dataset — the two
populations differ, and the baseline computed on the test split (18.37%) reconciles with §5.1.1's
18.4% violation rate. case14 and WCCI 2022 need no such distinction: their entire generated dataset
is already the evaluation scope, so both populations coincide there.

⚠ **The `>= 1.00` figure above rests on a much smaller sample, and that should travel with it.**
The held-out test split puts only 34 frames (1,947 contingencies) in this band, against 250 frames
(14,297 contingencies) in the full dataset — few of the model's held-out frames land in an
already-overloaded base state. §5.3.4's cliff table is left on the larger, full-dataset sample
deliberately: its purpose is to characterise the shape of the whole curve stably, not to match one
particular evaluation's scope, and its neurips2020 `>= 1.00` figure of 0.957 differs from this
table's 0.9666 for that sampling reason, not because the underlying relationship changed.

**If this were a circular result — the rule secretly restating the labelling function — the middle
column would read 100%.** At 91–97% the predicate carries real but partial information. The honest
framing is that the gate confirms established N-1 doctrine — a base case outside its limits cannot
be declared secure against further loss — and the data bears that doctrine out 91–97% of the time.
It is not discovering new physics; it is enforcing a known operational principle that the network
failed to learn, and failed to learn *worse* the further it moved from its training topology.

**The threshold itself is slightly conservative for this task, and by a measured amount.**
Enumerating every threshold on every variable in the gate's vocabulary (§5.2.11) puts the F1-optimal
cutoff at about 97% loading rather than 100% on both grids where a gate operates at all. Moving
there would gain roughly 0.045 F1 and lose 0.37 precision — from 0.938 to 0.566 on NeurIPS 2020,
which is to say from six false alarms in a hundred to nearly half. The served rule is therefore not
the best rule by the F1 criterion. Whether it is the better rule for a component whose function is
to override a model is no longer left as an assertion: §5.2.11 prices it, and 100 is preferable only
to an operator who treats one false block as costing more than about 2.4 to 2.6 missed violations.
That is a real and defensible preference for a gate with veto power, and this thesis adopts it — but
it is adopted explicitly, at a stated rate, rather than assumed. What matters most for the argument
is the direction and size of the gap: the standards' round number is close to, and slightly inside,
the task's own optimum.

### 5.3.6 Replication, seed variance, and what remains unmeasured

**The shield result is bit-identical across two independently generated rule corpora.** The
evaluation was re-run against a second extraction run whose rule totals differ by **29%**, and it
reproduced every reported digit — every count of corrections, regressions, missed violations and
false alarms. The mechanism is that `loading_pct > 100` is the only rule that ever fires as a
constraint on a base case, and both corpora contain exactly ten copies of it.

Two things follow. The result **does not depend on the extraction run that produced it**, which is
not a given for a pipeline with a language model in the middle. And it **bounds the corpus's
contribution honestly**: a ruleset can lose 29% of its rules and change nothing downstream.

**Seed variance is now partly measured, and the measurement is unflattering.** The four-seed
reactance experiment (§5.2.4) is the first arm in this project trained at more than one
initialisation, and it shows that cross-topology behaviour is partly a property of the random seed
rather than of the method:

| grid | control arm, sd over 4 seeds | physics arm, sd (held) |
|---|---:|---:|
| NeurIPS 2020 | 0.0092 | 0.0065 |
| WCCI 2022 | 0.0191 | 0.0127 |
| **case14** | 0.0177 | **0.0672** |

In-distribution the architecture is stable to within a hundredth. On case14 the physics arm swings
across a **0.18-wide range on initialisation alone**. This is a finding rather than a nuisance:
**off-distribution behaviour is partly a property of the seed, not of the architecture or the
data** — which is the same reason the single-seed case14 delta in §5.2.4 had to be discarded.

> **This applies retroactively to this chapter.** §5.1.3, §5.1.5, §5.2.3 and §5.4.4 all report
> **single-seed** model figures. **Every GNN cross-topology figure should be read as carrying at
> least ±0.02, and ±0.07 on case14.** Three consequences follow, and none is cosmetic. The case14
> failure survives comfortably — the gap to the single-rule baseline is 0.12, several times the
> band. The WCCI 2022 margin over the rule baseline, 0.5577 against 0.4915, survives at roughly
> three standard deviations. But **no narrow cross-topology comparison may be drawn**, and none is
> drawn here. **The tabular arm is unaffected** — the tree
> was fit once with far less initialisation sensitivity — so §5.4.4's 0.13–0.19 off-distribution
> gaps remain far outside any band.
>
> **The shield's deltas are barely affected, and this has now been measured rather than argued.**
> The structural reason is that each delta is a difference between two arms sharing one forward pass
> through one model, so it is deterministic *given* that model; a seed band moves the level the
> delta is measured from, not the delta itself. Scoring four architecturally identical seeds through
> the gate confirms it, and is reported in full below.

**The gate's own figures now carry a measured band.** The four control-arm checkpoints are
architecturally identical to the deployed model, so running each of them through the gate on all
three grids gives the shield's headline quantities a seed distribution rather than a transferred
caveat. Each seed selects its own threshold on the NeurIPS 2020 validation split and holds it fixed
across grids, which is the deployed protocol applied honestly to a different model:

| grid | shield delta, mean ± sd | range | intervention precision, mean ± sd |
|---|---|---|---|
| NeurIPS 2020 | **+0.0082 ± 0.0003** | +0.0077 to +0.0085 | 0.940 ± 0.008 |
| case14 | +0.0023 ± 0.0015 | +0.0003 to +0.0037 | **0.881 ± 0.050** |
| WCCI 2022 | **+0.0687 ± 0.0050** | +0.0640 to +0.0757 | 0.931 ± 0.004 |

**The delta keeps its sign on all four seeds on all three grids** — twelve of twelve positive. The
NeurIPS 2020 delta, the smallest headline number in the chapter and the one most open to the charge
that it is seed noise, turns out to be the *tightest* quantity measured anywhere in this work: a
standard deviation of 0.0003 on a mean of +0.0082. The WCCI 2022 delta never falls below +0.0640.

**The deployed checkpoint is not a lucky seed.** Scored under the identical protocol it reproduces
its published figures exactly, and it sits inside the four-seed band on every grid on both metrics —
on WCCI 2022 slightly *below* the mean (+0.0676 against +0.0687), which is to say the reported
result is marginally conservative rather than flattering. Its raw F1 is near the top of the band
in-distribution and at the bottom on both unseen grids, and the delta is insensitive to that, which
is exactly the structural property claimed above.

⚠ **Intervention precision is flat where the gate fires often enough to measure it, and not on
case14.** On WCCI 2022 it is 0.931 ± 0.004 across four independently initialised models whose
selected thresholds span 0.745 to 0.907 — strong support for the claim that the gate enforces a
doctrine rather than a learned pattern. On case14 it ranges from 0.816 to 0.924, a seed-to-seed
spread wider than the entire between-grid spread of the other two. That is small-sample noise, not
a doctrine failure: case14 admits between 15 and 217 blocks depending on the seed, and the lowest
figure is thirteen correct out of fifteen. **The case14 precision figure should be quoted with its
range and never as a point estimate**, and the same applies to its delta, whose sign is safe and
whose magnitude is not. A third quantity turns out to be more volatile than either: the selected
threshold itself varies from 0.745 to 0.907 across identical architectures, and most of case14's
swing in block count traces to that rather than to the gate.

**What is still unmeasured.** No per-epoch training history exists for the N-1 model, and the
deployed model's training run is not recoverable from disk — though §5.2.4's control arm establishes
that the documented intended command reproduces its cross-topology behaviour to within 0.0006. The
deployed model has still never itself been retrained at several seeds, so the *model* bands in the
table above this one remain transferred from an architecturally identical arm; it is the *gate*
figures that are now measured directly. **No confidence interval is claimed on any headline raw-model
figure**; those bands are a caveat, not an error bar.

> ⚠ **Validator-seed stability is also unmeasured.** Both arms of the §5.2.8 A/B ran at a single
> seed and temperature 0.0. Verdict stability across repeated runs is established for the `strict`
> arm and not for the `translated` one — which is the arm that produced the served corpus, and the
> arm §5.2.8 now shows was also capable of fabricating a reason. A replication harness exists and is
> specified to report verdict stability and reason stability **separately**, since §5.2.8 is
> precisely the case where the two diverge; it has not been executed, because it requires the
> language-model host. One limit is structural rather than practical: the validation stage writes
> confirmed rules without their verdict object, so reason stability is recoverable for the 21
> rejections and **not** for the 11 confirmations.

---

## 5.4 Comparisons and Relationships

### 5.4.1 The central relationship: the model's accuracy is topology-dependent and the gate's is not

| grid | model F1 (held) | gate: how often it can act | **gate: when it acts, how often right** |
|---|---:|---:|---:|
| NeurIPS 2020 *(trained on)* | 0.8956 | 0.31% | **0.938** |
| case14 *(unseen)* | 0.4167 | 0.087% | **0.922** |
| WCCI 2022 *(unseen)* | 0.5577 | 3.04% | **0.934** |

⚠ **The case14 precision figure is a point estimate on a very small sample and should be read as a
range.** Across four architecturally identical seeds it spans 0.816 to 0.924, because case14 admits
only 15 to 217 blocks depending on which model is guarded (§5.3.6). The other two figures are stable
across the same four seeds — WCCI 2022 to within ±0.004 — so the invariance claim below rests on
them, and case14 neither supports nor undermines it.

**The left column collapses and the right column does not move.** That is the relationship the
thesis turns on:

> The rulebook's accuracy does not depend on the topology, because it is not recognising a learned
> pattern — it is enforcing a physical principle that is equally true on every grid. The learned
> model's accuracy collapses off-distribution, from 0.90 to 0.42 to 0.56. The rulebook's does not.

What varies instead is **how much work there is for the gate to do**, and that varies with the
model's failure rate rather than with anything about the rules. The finding is therefore not "rules
help", but:

> **A symbolic layer built from published standards is topology-invariant in a way a trained model
> is not. Its value scales with how badly the model is failing, not with how well the symbolic
> layer is tuned.**

That relationship survives three separate perturbations: a change of topology (§5.1.6), a change of
threshold objective (§5.3.3), and a change of rule corpus (§5.3.6).

### 5.4.2 Why the effect is large on one grid and invisible on two

The three grids sit in different regimes, and this is a consequence of §5.4.1 rather than a
contradiction of it.

- **NeurIPS 2020.** The model is genuinely good, so there were only 2,041 dangerous errors to begin
  with, and only 16% of those sit on a state the served rule can observe — 23% for the best rule the
  language contains, at half the precision (§5.4.3). The gate works, and works well at 0.938
  precision; there is simply little for it to work on.
- **case14.** The model is broken here, and it fails in the *direction the gate cannot use*. It
  declares almost everything dangerous, so it almost never hands the gate a "secure" claim to
  overturn. Reach collapses to 0.087%. The gate is nearly always right in that sliver; there is
  just no sliver.
- **WCCI 2022.** The model fails often **and** fails on states the rules can see: 3.04% reach, 0.934
  precision, 31% of dangerous errors removed.

### 5.4.3 The structural ceiling — a bound that rule quantity does not lift

Of every dangerous error the model makes, what share occurs on a base state that **any**
present-state rule over the fourteen-variable vocabulary could even notice?

| grid | dangerous errors | reachable by any such rule | **unreachable** |
|---|---:|---:|---:|
| NeurIPS 2020 | 2,041 | 331 (16.2%) | **83.8%** |
| case14 | 18,592 | 95 (0.51%) | **99.5%** |
| WCCI 2022 | 68,166 | 21,065 (30.9%) | **69.1%** |

**The complement is a bound of the same order whatever rule is used, and it is large.** Between
69% and 99% of the model's dangerous errors sit beyond the reach of the *served* rule, on base
states that look perfectly healthy before the line is removed. Detecting those requires actually
simulating the failure — the exact computation the neural network exists to replace.

⚠ **The figures in that table are properties of this corpus, not limits of the rule language, and
the distinction was measured rather than assumed.** Enumerating every rule the fourteen-variable
vocabulary can express (§5.2.11) finds that the best of them reaches 23.0% of the NeurIPS 2020
errors and 38.5% of the WCCI 2022 errors, against the served rule's 16.2% and 30.9% — a looser
threshold on the same variable, bought with half the precision. The unreachable share is therefore
at most about 77% and 62% rather than exactly 84% and 69%. On case14 the question does not arise:
every rule that reaches more is one that fires on most frames, which is to say it is not a gate.
**The bound remains large and remains structural; its precise value depends on which point of the
precision–recall curve the gate is placed at.**

This answers the obvious question — *would more rules have helped?* — **structurally**, without
building progressively larger rulesets and plotting a curve. §5.3.6 answers it a second way: a
corpus can lose 29% of its rules and change nothing downstream. §5.2.11 answers it a third way, and
from outside the pipeline: a tree fitted directly on the answer over the same fourteen variables
reaches 330 of the 2,041 errors where the extracted BLOCK rule reaches 331; fourteen hand-written
rules combined reach no more than blocking everything; and an exhaustive enumeration of the language
recovers no variable, threshold or pair that the corpus does not already name. More rules would not
have helped. A slightly lower number on the rule already in hand would have traded recall for
precision, and nothing else was available.

### 5.4.4 The transfer failure is about learning, not about graphs

Degradation from the home grid, at the best-case column:

| method | NeurIPS 2020 | case14 | WCCI 2022 | worst drop |
|---|---:|---:|---:|---:|
| logistic regression | 0.7125 | 0.4895 | 0.4725 | **−34%** |
| random forest | 0.9251 | 0.6348 | 0.5507 | **−40%** |
| graph attention network | 0.8972 | 0.4477 | 0.5721 | **−50%** |
| *single-rule baseline (no learning)* | *0.4639* | *0.5392* | *0.4915* | *none* |

**Three learned methods — a linear model, a tree ensemble and a graph attention network — lose
34–50% of their home-grid score on an unseen topology.** That is the strongest available form of
the thesis's central claim, and it rests on three independent architectures rather than one.

**The graph model degrades worst of the three.** On case14 it loses 50% where the forest loses 31%,
and it ends *below* the single-rule baseline while the forest stays above it. Message passing over a
topology-specific graph appears to make transfer worse rather than better — consistent with the
diagnosis that the model encodes one network's redistribution pattern and carries it to a grid where
that pattern does not hold. §5.2.4 measured how much of that is a missing input, over four seeds:
supplying the branch reactance improves case14 ranking by +0.068 at best case — positive on every seed
— but leaves the model below the single-rule baseline on every seed. **A missing input is part of
the case14 failure and not the whole of it.**

**The last row is the control that keeps this claim about learning rather than about the task.** A
threshold on the removed line's own loading involves no fitting of any kind and does not degrade:
0.4639 / 0.5392 / 0.4915, with case14 its *best* grid. An unseen topology therefore does not make
the question intrinsically harder — a method with nothing to carry over carries nothing over and is
unharmed. What degrades is the learned part.

> ⚠ **This control is a weaker claim than an unlearned network-aware method transferring would be.**
> An unlearned *threshold* transferring is a much smaller claim than that, and 0.46–0.54 is not a
> competitive score in absolute terms — the control establishes only that a method with nothing to
> carry over is not harmed by an unseen topology, not that a stronger analytical method would do
> better still.

> ⚠ **The claim must be worded as *learned screeners trained on one topology did not transfer*,
> never as *neural networks cannot generalise*.** No arm here trains on several grids, and until
> §5.2.4 no arm was given the branch reactances that determine where power actually goes. The claim
> is about this training regime, not about neural networks as a class.

### 5.4.5 The thermal-loading check: ten clauses, four documents, two standards bodies

The shield's corpus is 58 rules, and judging it by how few of those reach the BLOCK channel is the
least flattering reading available — and not the most accurate one. Several of the BLOCK records
are the same physical check, stored separately only because the extraction model labelled the
entity `Line` in one standard and `Facility` in another, and wrote `100` in one clause and `100.0`
in another. Normalising that away:

| predicate | role | served records | clauses | documents | identified bodies |
|---|---|---:|---:|---:|---:|
| line loading above 100% of rating | constraint | 3 | **10** | **4** | 2 |
| voltage within 0.9–1.1 per unit | affirmation | 1 | 1 | 1 | 1 |

**The thermal check is stated in ten separate clauses across four documents by two standards
bodies on two continents** — NERC and the Bangladesh grid code. That is a stronger claim than a
bare BLOCK-record count, derived from the same evidence counted more carefully. It does not make
the check less obvious, but it does establish that it is not an artefact of one document or one
parse. Where a document does not identify its issuing body, it is recorded as unknown rather than
guessed, because the body count is a reported figure.

### 5.4.6 The same rule is informative on one grid and vacuous on another

The voltage predicate *"minimum below 0.95 or maximum above 1.05 per unit"* discriminates usefully
on WCCI 2022 at 1.12% coverage, and fires on **100% of frames** on both other grids — an alarm that
never stops, which carries exactly as much information as printing a warning unconditionally.

This is not a mis-extraction. That predicate is stated in ten clauses across four documents, tied
with the thermal check as the most corroborated in the corpus. The cause is the one identified in
§5.2.7: these grids rest at 1.05–1.08 per unit, so a band written about the *post-disturbance*
state is breached before anything has happened. Such rules are therefore silenced per grid at
serving time rather than dropped, and carry an explicit record of the demotion:

| grid | warning rules speaking | silenced | firing on 100% of frames |
|---|---:|---:|---:|
| NeurIPS 2020 | 20 → **7** | 9 | 9 → **0** |
| case14 | 20 → **9** | 10 | 10 → **0** |
| WCCI 2022 | 20 → **19** | 0 | 0 → **0** |

No reported number moves, and that is structural: warnings never reach the veto path, and a test
asserts that the blocking group is identical before and after on all three grids.

**The finding is worth more than the fix.** The same unmodified rule, stated by four standards
bodies, is informative on one grid and vacuous on another, and which one it is depends on how the
grid is *operated* rather than on anything written in the standard.

### 5.4.7 Seven unrelated procedures converged on the same check

Two entirely separate methods decided which extracted rules were worth keeping, and neither had
access to what the other used:

- **An empirical filter.** No language model. It ran every candidate against real grid data and
  asked *does this fire, and does it discriminate?*
- **A textual auditor.** No grid data. It read the source standards and asked *is this a faithful
  reading of the document?*

Both kept the same family — *is any line loaded past 100% of its rating?* — and both discarded the
voltage rules, the auditor because they came from ride-through tables and time-bound operating
envelopes misread as instantaneous limits.

Either result alone invites an easy objection: *your empirical cutoff is mistuned*, or *your prompt
is wrong* — and §5.2.8 shows that the second objection can genuinely be correct. **Convergence
defeats both.** The thermal-loading check is the survivor because of the mismatch between what
standards regulate and what the simulator models, not because of an artefact of either filter.

The controls of §5.2.11 extend that convergence considerably. Counting every procedure in this
project that selected variables or predicates for this task, independently of the others:

| # | procedure | what it had access to | what it selected |
|---|---|---|---|
| 1 | LLM extraction over published standards | 16 documents, no grid data | `loading_pct > 100` — 10 clauses, 4 documents, 2 bodies |
| 2 | empirical fire-rate filter | grid data, no language model | the same thermal family |
| 3 | textual auditor | source standards, no grid data | the same thermal family |
| 4 | depth-4 tree fitted on the answer | 14 context variables, the labels | `loading_pct` (0.614) and `voltage_pu_max` (0.152) |
| 5 | fourteen hand-written expert rules | doctrine and the same 14 variables | `loading_pct > 100`, best of 14 on two grids of three |
| 6 | **exhaustive enumeration of the rule language** | every cutpoint and every pair, plus the labels | `loading_pct` — best single-variable rule on both grids, at 2.2x and 1.7x the runner-up |
| 7 | the L2RPN challenge's leading teams | their own competition entries | a rule-based alert at `rho ≥ 1.0` |

**Seven procedures with different inputs, different failure modes and different authors arrive at
one predicate.** Routes 4, 5 and 6 are the ones that matter most for the argument, because they are
the only three that had the answer key and could therefore have gone anywhere else.

**Route 6 is the strongest, and it is different in kind.** The other six searched, chose or
inherited. Enumeration does not choose: it scores every rule the language can express and reads off
the maximum. When that maximum is the thermal check by a factor of 1.7 to 2.2 over the next best
variable, the convergence stops being a coincidence of six procedures and becomes a statement about
the task — there was nothing else in the vocabulary to find. The one thing it does add is that the
optimal *constant* is about 97 rather than 100, which routes 1 and 5 both inherited from the
standards' round number (§5.2.11).

⚠ Route 5 is not blind (§5.2.11), and route 7 is an observation drawn from a published challenge
report rather than a comparison run here; neither is presented as an independent replication of the
extraction pipeline's output. The claim is convergence, not confirmation.

---

## 5.5 Discussion

### 5.5.1 What the results establish

**A symbolic layer built from published standards catches a neural screener when it is wrong, at
accuracy independent of topology.** Intervention precision holds flat at 0.92–0.94 across the
training grid, a grid a third its size, and one three times its size. The gate's value scales with
the model's failure rate, not its own tuning — +0.0676 F1 and a 31% cut in dangerous errors on the
largest unseen grid, where the gated score exceeds even the raw model's own best-case ceiling.

**A learned screener trained on one topology does not transfer, and the architecture is not the
cause.** A linear model, a tree ensemble and a graph attention network all lose 34–50% of their
home-grid score off-distribution; an analytical method loses 4%. The graph model degrades worst.

**The rule corpus's thinness belongs to the task, established by exhaustion rather than argument.**
Every rule the shield's fourteen-variable language can express was enumerated, along with every
two-term combination and an unlimited-depth ensemble. Nothing beats the extracted predicate except
itself at a slightly lower constant — at roughly half the precision; the best variable beats the
runner-up by 1.7–2.2x; and an unreadable ensemble free to split as deep as it likes still loses to
one threshold comparison, on all three grids. A tree fitted on the answer and fourteen hand-written
rules return the same verdict (§5.2.11). **The thin BLOCK channel is a finding about the
standards/simulator mismatch: what the pipeline missed was a better constant on a rule it already
had, not a rule.**

**The largest blocked families were closed by measurement, not argument.** Just over half the
corpus is unusable because it names frequency or sub-second time, and the obvious remedy — a
dynamic simulator representing both — was tested rather than assumed. No N-1 contingency moves
frequency far enough to reach even the mildest of 457 thresholds, and voltage ride-through envelopes
are entered by the fault event, never by any state a shield could observe. Of 1,252 candidates
blocked on frequency or sub-second time, **3** would have fired under a dynamic simulator (§5.2.5) —
a finding about subject matter, not instrumentation, that generalises beyond this project.

**Reporting standards matter as much as the results.** Three findings here are methodological: a
rulebook of mostly bad rules performs at its average and hides the good rules completely (§5.2.6); a
pipeline failure and a genuine negative result can produce identical summary statistics, so verdicts
must be persisted with their reasons (§5.2.8); and "can this rule be computed?" and "will this rule
ever be true?" are different questions whose answers differed by 1,249 rules (§5.2.5).

### 5.5.2 What the results do not establish

**The neural component does not earn its place on cross-topology accuracy.** On case14 it scores
0.4167 at the held threshold — 0.77x the best single-rule baseline and below all-positive, i.e. it
loses to answering "violation" every time (§5.1.3). The graph network cannot be presented as the
right tool for screening an unseen grid. What survives is the narrower claim about the gate, which
holds over *whatever* model it guards.

**What the model does earn is cost, and only where it is already accurate.** §5.1.4 measures the
exhaustive AC screen that produced every label in this chapter at 665 contingencies/second on the
training grid, against the network's 88,260 — a gap that widens further on the larger unseen grid.
That buys F1 0.8858 for roughly 133x less work at home; on case14 it just buys a wrong answer
faster. **Speed is a property the model has; off-training accuracy is one it does not, and the first
cannot substitute for the second.**

**The claim that producing an N-1 label requires a power-flow solve survives, untested here.**
Nothing in this chapter contradicts it, but it doesn't follow that a learned model is therefore the
right tool: a *linear* solve — neither a rule nor a learned model — can also produce the label, and
score well doing so. This thesis measured what the *non-linear* solve costs, not what the cheapest
sufficient solve is.

**The controls do not make the gate good, nor are they offered as evidence it is.** They bound what
any frame-level symbolic gate over these variables could achieve — not whether that bound is useful.
The bound is now known exactly, from enumeration rather than search, and it is low: the best rule at
any threshold reaches 23.0% and 38.5% of the model's dangerous errors, at precisions of 0.566 and
0.744. Served-rule recall is 0.162 / 0.005 / 0.309, and on case14 no gate — extracted, fitted,
hand-written or enumerated — beats blocking every contingency the model called secure. The licensed
claim is that the rules that exist are close to the best obtainable from these variables, never that
they are sufficient.

**The served threshold is not F1-optimal, and the thesis does not claim it is.** The optimum over
the whole rule language sits near 97% loading rather than 100%, worth about 0.045 F1 (§5.2.11) —
and it survives selection on a held-out split rather than the answer key, still worth +0.027 and
+0.042 there. The served rule is kept for two reasons: it is what the standards state, and its
provenance is the property this corpus exists to demonstrate. The second is now quantified — the
0.045 is bought by surrendering 0.37 precision, which a component with veto power over a model
should not spend lightly. "Lightly" now has a number: 100 is the better operating point above an
exchange rate of about 2.4–2.6 missed violations per false block, and the worse one below it.

**This is a design choice, and stating it honestly is uncomfortable, not neutral.** The crossover
falls *below* equal weighting, so at plain F1 the served threshold has already lost. The thesis's
position is that a gate overriding a model belongs on the precision side of that crossover — and
states the rate at which that position would flip, rather than simply asserting precision matters
more.

### 5.5.3 Limitations, volunteered

**Only the thermal-loading check reaches BLOCK.** The shield's corpus is 58 rules, translated from
2,463 candidates; of those, every record calibrated precisely enough to veto a dangerous prediction
restates the same thermal check. §5.4.5 counts it more generously — stated in ten clauses across
four documents by two bodies — but the BLOCK channel is still that one physical check. Everything
else the 58 rules do is WARN or NORMAL: explaining a prediction, not vetoing it.

**That predicate is close to common sense, and the distance was measured.** P(violation | an
already-overloaded base case) is 91–97%, not 100% — real but partial information (§5.3.5). The
gate confirms established N-1 doctrine; it discovers no new physics.

**Its threshold is inherited, not fitted, and is about three points conservative.** The F1-optimal
cutoff over the whole rule language is near 97% loading, not 100% (§5.2.11). The served number came
from the standards; the hand-written expert arm independently inherited the same round number, so
their agreement is partly agreement about a convention. The thesis reports 100% as what the
standards state and the higher-precision point on the curve — not as the optimum.

**The doctrine is hand-written; the language model supplied only the number.** The asymmetric N-1
reasoning is encoded in the project's own code — extraction contributed the threshold (100%
loading) and nothing else. That doctrine belongs to the class of *operational* standards §5.2.5
shows is absent from the document corpus, so the pipeline could not have supplied the reasoning:
nobody fed it a document containing it.

**Slightly over half the corpus was closed off by the simulator's representation — evidence-backed,
but narrow in one place.** The frequency finding is wide: the worst N-1 excursion, 0.3293 Hz, misses
the mildest threshold of 0.6 Hz by a factor of ~1.8, and firing needs N-2 or worse. The voltage
finding is narrower: the deepest clean-opening dip clears the mildest bar by under 1% once rebased,
so a heavier loading condition or deeper contingency could plausibly cross it. Nine of the 103
ride-through rules carry free-variable thresholds and could be neither fired nor shown inert, and
eight distinct duration thresholds — 21, 30, 180, 300, 900, 1200, 1800, 3600 seconds, behind ~27
rule mentions — exceed the 20-second simulation window. Quote frequency as settled; voltage as
measured but thin-margin (§5.2.5).

**There is a hard ceiling, and more rules will not lift it.** Most of the model's dangerous errors
occur on base states that look perfectly healthy, reachable only by the post-contingency solve the
model exists to replace (§5.4.3). The exact share depends on where the gate sits on the
precision–recall curve: 84% and 69% unreachable by the served rule, ~77% and 62% by the best rule
the language contains, at half its precision. The order of magnitude is fixed either way —
established by enumerating the language, not by asserting it.

**No confidence interval is claimed on any headline model figure, and the reason is now measured.**
A four-seed replication (§5.3.6) puts cross-topology spread at ±0.02 generally, ±0.07 on case14 —
larger than several effects this chapter discusses. The deployed model itself was never retrained at
multiple seeds, so these bands transfer from an architecturally identical arm rather than being
measured on it. Every conclusion here clears the band; no narrow cross-topology comparison is drawn.

**The expert-rule control was not run blind, and the thesis claims less for it accordingly.** The
central objection to this work: a thin BLOCK channel might mean the task is rule-poor **(a)** or the
pipeline is the bottleneck **(b)** — and until recently every argument for (a) came from inside the
pipeline that produced the corpus. Three controls have now run, and all three support (a)
(§5.2.11). Two are clean: the fitted tree and the exhaustive enumeration see only the fourteen
context variables and the labels. The third is not — the fourteen expert rules were written after
their author had already seen the served corpus, the gate's precision figures and the tree's splits,
so the specification's own fallback applies and the arm is reported as a hand-written comparison,
not a blind baseline. It was fingerprinted before evaluation, which prevents tuning against the
result but does not reconstruct blindness.

**That limitation has since been made largely moot, not repaired.** The expert arm asks *could a
human have written a better rule?*; enumeration asks *could anyone have?* — which strictly contains
it, and answers no. The blindness objection now bears only on how much independent weight the
expert arm carries as corroboration, not on whether reading (a) stands. **Reading (a) is
demonstrated by exhaustion, corroborated by a fitted search, and corroborated again — with a
disclosed contamination — by hand.**

**One further item is a disclosed gap, not a measurement.** A "zero-shot capability retention"
check has no analogue here: the gate is a post-hoc filter that modifies no weight and can cause no
catastrophic forgetting — the model is unchanged by construction, not measured and found stable.

⚠ **Yet to run:** the message-passing ablation retrain (§5.2.3), the voltage ride-through stress
arm (§5.2.5), validator replication (§5.3.6), and three-term rule enumeration (§5.2.11). None
blocks a claim made here — each is a matter of time or hardware, not of method.

### 5.5.4 The claim this chapter supports

The neural screener is not the contribution. Nor is raw accuracy. The contribution is what the gate
adds on top of either: a verdict that is traceable, blockable and explainable.

> **The contribution is explainability — a safety layer that turns a black-box screener's verdict
> into a decision with an authority, a mechanism, and a reason, not simply a more accurate one.**
> **Traceable:** every block cites the exact clause of the exact standard behind it —
> `Document → Clause → Rule → ServedRule → Predicate → Variable` — so nothing is asserted without
> its provenance (§5.2.10). **Blockable:** only the BLOCK channel may veto, structurally rather than
> by convention, at a measured 0.92–0.94 intervention precision (§5.2.9). **Explainable:** rules
> that version 1 silently discarded because they could not justify a veto now speak instead — 41 of
> 58 rules across 17 distinct conditions, up from the 4 that could act at all, a 5.7x increase in
> what the system can say about a prediction it lets through (§5.2.9). This layer is also
> topology-invariant — its accuracy is a property of the physics it enforces rather than of the
> data it was tuned on, and it survives a change of grid, of threshold objective and of rule corpus
> — and distils to a single, corroborated predicate that an exhaustive enumeration of its own rule
> language confirms is the best one available. That thinness belongs to the task rather than to the
> method, established from outside the extraction pipeline rather than argued from within it.**

---

## Artifacts behind every figure

| claim | findings § | artifact |
|---|---|---|
| shield results, all three grids | §13.1 | `results/shield/shield_<tag>_validated.json` |
| bit-identical replication across extraction runs | §13.6 | `results/shield/shield_<tag>_run3.json` |
| structural ceiling | §13.3 | same |
| distance from tautology (91–97%, NeurIPS 2020 corrected to the test-split scope) | §13.4 | same |
| raw model cross-topology, held and best case | §14 | `gnn_n1_tightening.md` |
| F-beta trade curve | §14.2 | `results/threshold/threshold_sweep.json` |
| inference cost, AC (LightSim) vs the model, three grids | §5.1.4 | `results/timing/timing_<tag>_lightsim.json` · `tests/test_bench_inference_speed.py` |
| non-graph learned baselines (logistic, random forest, MLP) | §26 | `results/tabular/tabular_baselines.json` · `results/tabular/baseline_ladder.json` |
| **reactance experiment, four seeds** | §27.3 | `results/reactance/reactance_transfer[_seed{0,1,2}].json` |
| seed instability off-distribution, and its retroactive reach | §27.3.3 | same |
| deployed model reproduces from the documented command | §27.3.5 | same |
| corpus accounting, all 2,463 candidates | §19.1 | `results/audit/corpus_accounting.json` |
| loading-band cliff | §20.1 | `results/audit/loading_band_calibration.json` |
| warning-channel calibration, two inverted predicates | §21.2 | `results/audit/warn_n1_calibration.json` |
| validation A/B and the fabricated reasoning | §12.1–§12.4 | `validated_{strict,translated}/*_rejected.jsonl` |
| the 58-rule shield corpus and its BLOCK/WARN/NORMAL channels | §12.5, §5.2.9 | `shield_corpus/all_rules_channels.jsonl` |
| provenance and corroboration | §16.3 | `kg/knowledge_graph.json` · `kg/kg_corroboration.svg` |
| **frequency rules inert under N-1; escalation ladder** | §20.2 | `results/audit/andes_frequency_spike.json` · `sanity/andes_frequency_spike.py` |
| **ride-through envelopes reachable by the event, not by a state** | §20.2 | `results/audit/andes_voltage_spike.json` · `sanity/andes_voltage_spike.py` |
| **1,252 candidates blocked on frequency/time (replaces the unscripted 726 and 1,332/782/683 figures)** | §5.2.5 | `results/audit/capability_gap.json` · `evaluation/capability_gap.py` |
| the dynamic-simulator rejection, in plain language | §20.2 | `supplimentary_docs/andes_investigation.md` |
| **ceiling tree, three grids and two robustness configurations** | §28 | `results/ceiling/ceiling_tree*.json` · `evaluation/ceiling_tree_n1.py` |
| gate-semantics correction to the tree's figures | §29.7 | same (`tree_held_gate`, `tree_oracle_gate_f1`) |
| **hand-written expert baseline, pre-registered and fingerprinted** | §29 | `results/ceiling/expert_rules.json` · `expert_rules/expert_rules.jsonl` (md5 `05a29bd7…`) · `evaluation/expert_rules_n1.py` |
| **exhaustive enumeration of the rule language, three arms** | §30 | `results/ceiling/exhaustive_rules.json` · `evaluation/exhaustive_rules_n1.py` |
| correction to the tree's headline, and the ~97 optimum | §30.5, §30.3 | same |
| **the F-beta exchange rate between 100 and ~97, every cutpoint** | §5.2.11, §5.3.5, §5.5.2 | `results/ceiling/loading_pct_sweep.json` · `grids.<tag>.loading_pct_fbeta` in `exhaustive_rules.json` |
| **held-out rule selection, replacing the best-case selection** | §5.2.11 | `results/ceiling/exhaustive_rules.json` (`held_out_selection`) |
| **pair search at 32 / 64 / 128 cutpoints** | §5.2.11 | same (`pair_resolutions`) |
| **shield seed band, four control seeds x three grids** | §5.3.6 | `results/seeds/shield_seed_band.json` |
| **expert arm re-scored with `abs`/`min`/`max` admitted** | §5.2.11 | `results/ceiling/expert_rules_builtins_repaired.json` |
| **adjudication of all 21 shared rejections, mechanical and read fields separated** | §5.2.8 | `results/audit/rejection_adjudication.json` · `evaluation/audit_rejections.py` |
| validator replication harness — **written, not executed** | §5.3.6 | `evaluation/replicate_validation.py` |
| retired claims guarded as a test | `tests/test_docs_hygiene.py` docstring | `tests/test_docs_hygiene.py` |
