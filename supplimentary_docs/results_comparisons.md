# Comparison plan for the results section

**Status: SPEC — two arms run, one dropped, one RETIRED.** Written 2026-09-18 to be executed in a
later session. **A1 was executed 2026-09-19 and then RETIRED on 2026-09-20** — its code and
artifacts were deleted and the results section now carries an inference-cost benchmark in its place
(`thesis_findings.md` §25). §S7 and §A1 are kept as written, under retirement banners, because the
reasoning is part of the record and because A1's finding was never shown to be wrong. A2–A5 remain
unrun. This document catalogues every comparison the results section could carry, ranks them by
what they buy against what they cost, and records the traps for each. It does not report results;
when an arm is executed, its numbers go to `thesis_findings.md` and this file's entry is marked
DONE with a pointer.

---

# START HERE — session pickup

**Written 2026-09-18 for a session starting cold with no prior context.** Everything in this block
was verified on disk on that date, not recalled. If a fact here is load-bearing for what you are
about to do, re-run its check (§S4) rather than trusting it.

## S1. Read order

1. **This block**, then **§0** (the admissibility rule — it decides what counts as a comparison).
2. **§A1** if you are doing the priority arm. Its four open questions are now **RESOLVED**; the
   fourth resolution changes what the arm measures, so do not skip it.
3. `CLAUDE.md` "Current Status" for the protocol constants. **Do not re-derive them.**
4. `thesis_findings.md` §24 only if you are running §A4.

## S2. Environment — verified 2026-09-18

```bash
# Windows: the repo venv. A bare `python` is the Microsoft Store stub and will fail.
.venv/Scripts/python.exe
# When piping or redirecting, these scripts print → and die on cp1252:
#   $env:PYTHONIOENCODING = "utf-8"
```

| package | version | needed by |
|---|---|---|
| `pandapower` | 3.4.0 | **A1** — `makePTDF`, `makeLODF`, both import clean |
| `lightsim2grid` | 0.12.2 | A1 alternative (not chosen — §A1 q1) |
| `grid2op` | 1.12.3 | not needed if limits are backed out per §A1 q3 |
| `scikit-learn` | 1.8.0 | **A3** — `HistGradientBoostingClassifier` |
| `torch` | 2.9.0+xpu | non-CUDA → **deployed `[16,32,32]` config auto-selects.** Correct |
| `torch_geometric` | 2.7.0 | dataset loading |
| `xgboost` | **MISSING** | do not reach for it; A3 says use sklearn |

**All inputs present:** `gnn_checkpoint_n1.pt` (206 KB), `data/processed_grid_data_n1.pt` (92.7 MB),
the three `grid_dataset_<tag>_n1.jsonl` (156 / 27 / 157 MB), all three
`split_neurips2020_n1_{train,val,test}_idx.npy`, and the base-kV sidecars. **Nothing needs
regenerating.**

## S3. The four constants that make results comparable — never vary these

| | value | why |
|---|---|---|
| eval batch size | **64** | BatchNorm uses live batch stats; same checkpoint scores 0.8972 @ 64 and 0.9255 @ 512 |
| held threshold | **0.8849** | chosen on the neurips2020 val split, applied unchanged everywhere |
| scorer | `best_f1_and_thr` | `evaluation/eval_n1_cross_topology.py:76`. Every arm uses this one function |
| scope | neurips2020 = test split; foreign grids = every frame | already how `main()` behaves |

An arm that violates any of these is **not a comparison** (§0). It can be context, in prose.

## S4. Re-derive the §A1 measurements

Everything claimed in §A1's resolved questions comes from these. Run them first — they take about a
minute and they are the difference between building on a measurement and building on a memory.

```bash
# q1 — PTDF/LODF builders import?
.venv/Scripts/python.exe -c "from pandapower.pypower.makePTDF import makePTDF; from pandapower.pypower.makeLODF import makeLODF; print('ok')"
```

```bash
# q2 + q4 — bus splitting, topology patterns, and the label decomposition
.venv/Scripts/python.exe -c "
import json, numpy as np
for tag in ['case14','neurips2020','wcci2022']:
    n=intact=split=P=D=Z=N=0; pats=set()
    for line in open(f'data/grid_dataset_{tag}_n1.jsonl'):
        if not line.strip(): continue
        r=json.loads(line); n+=1
        ls=np.array(r['line_status']); tv=np.array(r['topo_vect'])
        intact+=bool(ls.all()); split+=bool((tv==2).any()); pats.add(tuple(ls))
        y=np.array(r['n1_violation']); pr=np.array(r['n1_post_rho'])
        P+=int(y.sum()); N+=len(y)
        D+=int(((y==1)&(pr==0.0)).sum()); Z+=int(((y==0)&(pr>=1.0)).sum())
    print(f'{tag:12s} frames={n:6d} intact={intact/n:5.1%} bussplit={split/n:5.1%} patterns={len(pats):4d} | positives={P} solvefail={D} ({D/P:.1%}) y0_but_rho1={Z}')
"
```

Expected: bus-split **0.0%** everywhere; patterns 140 / 745 / 402; `y0_but_rho1` **0** everywhere;
solve-failure share **26.5% / 21.7% / 25.4%** of positives.

## S5. Status of every task in this file

| # | task | § | state |
|---|---|---|---|
| 1 | Resolve A1's open questions | A1 | ✅ **DONE 2026-09-18** — all four, by measurement |
| 2 | **A1 — DC-LODF screening** | A1 | 🚫 **RETIRED 2026-09-20** — ran, then removed with its code and artifacts. Replaced by the inference-cost benchmark, `thesis_findings.md` §25. Its finding (LODF above the model on all three grids) is recorded in §A1, not withdrawn |
| 3 | A3 — tabular baselines | A3 | ✅ **DONE 2026-09-19, re-scoped cross-topology** — `thesis_findings.md` §26. Non-graph models collapse too |
| 4 | A2 — Pavão MLP replication | A2 | ❌ **DROPPED 2026-09-19** — A3 answers the transfer question at lower cost; §26.5 |
| 5 | A5 — collect existing arms; re-measure train/val/test | A5 | ⬜ not started. No new code; one re-measurement |
| 6 | A4 — §24 tree + expert rules | A4 | ⬜ not started. **Spec lives in `thesis_findings.md` §24 — run it there, do not restate** |
| 7 | B5 — read Ahmadi Appendix B | B5 | ⬜ not started. Decides whether §B5's argument survives |
| 8 | Write Table 2 | D | ⬜ not started. Needs B1–B4, B7 |
| 9 | Verify the two abstract-only papers | B8 | ⬜ not started. **Blocks quoting either number** |
| 10 | Novelty statement wording | F | ✅ **DONE 2026-09-18** — F1 is the ceiling until a systematic search exists |

**#2 was done, and it was the one that mattered — then it was retired.** It tested a load-bearing
claim and the claim did not survive; on 2026-09-20 the arm itself was removed from the thesis's
scope rather than the finding being rebutted. Read §S7 and §A1 before picking the next task, and
read them as history rather than as live status.

## S6. What changed on 2026-09-18, in one paragraph

A targeted search outside `docs/lit` found the two papers now at §A1 *Precedent* and §B7, and is
recorded at §B8. §F fixes the novelty wording for Components B/C/D. Most importantly, **§A1's four
open questions were resolved by measurement**, and the fourth turned up something the plan did not
anticipate: **21.7–26.5% of positive labels are solve-failure cases, which a linear method cannot
produce even in principle.** A1 must therefore report two F1s — overload-only and full-label — and
the gap between them *is* the result. Read §A1 q4 before writing any A1 code.

## S7. What changed on 2026-09-19 — read this before choosing a task

> 🚫 **SUPERSEDED 2026-09-20 — A1 WAS RETIRED.** Everything below is an accurate record of what
> was measured on 2026-09-19, and none of it was rebutted. But the arm's code and artifacts are
> gone (`git show 6546fc2:evaluation/lodf.py`), the results section carries an inference-cost
> benchmark instead (`thesis_findings.md` §25), and point 1's instruction to "lead with it" no
> longer describes the chapter. Read this as history. §A1 carries the full retirement note.

**A1 ran. DC/LODF screening beats the GNN on every grid**, at the model's most favourable threshold
and with no tuning of its own: **0.9550 / 0.9150 / 0.9207** against the model's **0.8972 / 0.4477 /
0.5721**. Full account, validation and caveats: `thesis_findings.md` §25. Three things follow.

1. **§A1's "what a result would mean" table resolved to its third row** — *"above the model. The
   neural component would not be earning its place on this task."* That is now the finding, and the
   results section must lead with it rather than bury it.
2. **Two claims in this file are wrong and are corrected in place below** — §A1 q4's solve-failure
   ceiling (a topology pre-screen catches 55–88% of that class at precision 1.000, so the ceiling
   does not exist) and its positive counts (they summed `-1` for de-energized lines).
3. **A2 and A3 still deserve to run, but not for the reason they were written.** They were specified
   to ask "is the GNN over-parameterised?"; the live question is now where the GNN sits on a ladder
   whose top is occupied by a linear method from the 1960s.

---

**Why this exists.** The thesis currently reports its model against two internally-computed
baselines and nothing external. A reviewer's first question is "compared to what?" — and the
literature review folder (`docs/lit`, 18 PDFs) was surveyed on 2026-09-18 and contains **no paper
reporting a directly comparable number**. One paper comes close enough to cite carefully (§B1);
the rest are positioning material. The gap therefore has to be closed by building comparators
ourselves, which is §A.

**The folder is not the only place that was checked.** A targeted web search on 2026-09-18 went
outside `docs/lit` for the two literatures most likely to hold a comparable number — GNN-based
contingency screening and static security assessment. It **did not overturn the conclusion above**,
and it produced two papers that change how §A1 and Component D are positioned: they are recorded at
§A1 (*Precedent*) and §B7. The useful negative is that per-line N-1 violation scores, at a threshold
held fixed across several grids, are essentially unreported — which is the real reason our
comparison has to be internally constructed, and a better one than "our folder lacks it."

---

## 0. The admissibility rule — read before adding any arm

Every arm in §A produces **a score vector aligned to the same `y`** and is scored by the same
function. That is the whole design. Concretely, from `evaluation/eval_n1_cross_topology.py`:

| piece | where | note |
|---|---|---|
| `best_f1_and_thr(score, y)` | `eval_n1_cross_topology.py:76` | the single scorer; every arm uses it |
| all-positive baseline | `allpos = 2p / (1 + p)` | closed form, `p` = violation rate |
| rule baseline | `best_f1_and_thr(rho, y)` | best threshold on the base-case `rho` of the **removed** line |
| `y`, `logits`, `rho` | built in `main()` | de-normalized, so the rule is a rule on a physical quantity |

**Four constraints that are non-negotiable, because every recorded number depends on them:**

1. **Eval batch size 64.** `EVAL_BATCH_SIZE` in both harnesses.
   `BatchNorm(track_running_stats=False)` uses live batch statistics, so the same checkpoint scores
   0.8972 at 64 and 0.9255 at 512. A baseline scored against a model number from a different batch
   size is not a comparison.
2. **Held threshold 0.8849**, selected on the neurips2020 val split, applied unchanged everywhere.
   Any arm that picks its threshold per-grid is an **oracle** arm and must be labelled as such in
   its own column. Never mix held and oracle columns in one column.
3. **neurips2020 scores its held-out test split; foreign grids score every frame.** Already how
   `main()` behaves. A baseline fitted on data must honour the same train/test separation.
4. **Report the delta, not only the level.** Levels move with batch size and forward-pass
   nondeterminism; deltas between arms sharing one forward pass do not.

> ⚠️ **An arm that cannot satisfy 1–3 is not admissible as a comparison.** It can still appear as
> context, in prose, clearly marked as not comparable. That is how §B1 is handled.

---

## A. Comparators we build ourselves

These are the substance. Ranked by value per unit of effort.

### A1. DC power-flow / LODF contingency screening — 🚫 RAN 2026-09-19, **RETIRED 2026-09-20**

> 🚫 **THIS ARM NO LONGER EXISTS.** `evaluation/lodf.py`, `evaluation/eval_lodf_n1.py`,
> `tests/test_lodf.py` and `results/lodf/` were deleted on 2026-09-20 and replaced by the
> inference-cost benchmark — `evaluation/bench_inference_speed.py`, `thesis_findings.md` §25.
> The results section now asks what the incumbent AC method *costs* rather than how accurate a
> linear approximation of it is. **Recover any of it with `git show 6546fc2:<path>`.**
>
> ⚠️ **What it measured, so that nobody re-derives it by accident.** LODF + topology screening
> scored **0.9550 / 0.9150 / 0.9207** (oracle) against the model's **0.8972 / 0.4477 / 0.5721** —
> above the model on all three grids, at the model's most favourable threshold, with no tuning of
> its own. That result is not withdrawn and was never shown to be wrong; it was removed from the
> thesis's scope. **Anyone reinstating this arm should expect that finding to reappear.**
>
> ⚠️ **One piece survived the deletion.** `load_branch_model` now lives in
> `evaluation/branch_model.py`, because `evaluation/reactance_transfer.py` (§27) feeds the model
> each grid's branch susceptances and has nothing else to read them from. Only the factor machinery
> (`lodf_for_pattern`, `isolates_supply`, `LodfCache`) went. Guarded by `tests/test_branch_model.py`.
>
> The spec below is kept as written, with its two errors corrected in place, because the reasoning
> that led here is part of the record — and because it is the specification anyone reinstating the
> arm would need.


**What.** The discipline's incumbent method. Line Outage Distribution Factors give post-contingency
line flows in closed form from the base case: removing line *k* redistributes its flow onto the
rest of the network by a fixed linear factor. Screen by "does any resulting flow exceed its
limit?", producing one score per contingency — exactly the shape of `y`.

**Why it is the priority.**

- It is what practitioners actually run. Comparing against it is comparing against the field, not
  against a strawman we invented.
- It is reproducible by anyone, unlike a number lifted from a paper.
- **It directly tests the thesis's load-bearing claim.** The argument for abandoning `classify` was
  that "no rule over the present observation can restate an N-1 label, because producing it needs a
  power-flow solve" (CLAUDE.md, `thesis_findings.md` §9). LODF is the *linear approximation* of
  that solve. Its F1 therefore measures precisely how much the nonlinear AC physics is worth — and
  how much of the model's advantage over the rho-rule is "it learned linear redistribution" versus
  "it learned something beyond it".

**Precedent — DCPF is the field's standard comparator, not one we invented.**

> Alcántara, Chatzivasileiadis. "Trustworthiness Layer for Foundation Models in Power Systems:
> Application to N-k Contingency Screening." arXiv:2602.07995 (submitted Feb 2026, rev. Apr 2026).

They screen contingencies with a learned model and benchmark it **directly against DC power flow**,
reporting that their calibrated approach captures **over 90% of critical violations at up to 5×
fewer false alarms than DCPF**, on the IEEE 24- and 118-bus systems. Found by targeted web search
2026-09-18; **it is not in `docs/lit`** (see §B7's scope note).

- **This is what A1 is for.** Cite it as the reason the arm exists: benchmarking a learned screener
  against DC power flow is the move this field already makes, so building it is meeting the
  standard rather than inventing a convenient strawman.
- ⚠️ **Not tabulable against us, and for the usual reasons.** Different grids, different metric
  (violation capture and false-alarm ratio, not F1), no cross-topology arm, and no score vector we
  could align to our `y`. It fails §0 constraints 1–3. **Prose only** — the same discipline §B1 gets.
- ⚠️ **Read at abstract level only.** Verify their DCPF configuration and what counts as a "critical
  violation" before quoting the 5× figure, or we will be repeating a number whose denominator we
  have not seen.

**What a result would mean:**

| LODF result | reading |
|---|---|
| well below the model | the model captures nonlinear redistribution the linear method misses — the strongest version of the claim |
| comparable to the model | the model has learned approximately DC power flow; still a finding, but the framing changes from "learns physics" to "learns a fast surrogate for a known method" |
| above the model | report it plainly. The neural component would not be earning its place on this task. |

**What exists.** `lightsim2grid 0.12.2`, `pandapower 3.4.0`, `grid2op 1.12.3`, `scipy 1.16.3` are
all installed. No new dependency. The three datasets and their `_meta.json` are on disk.

**Open questions — RESOLVED 2026-09-18 by direct measurement. Read this before writing code.**

All four were settled against the datasets on disk. **Re-derivation commands are in the START HERE
block at the top of this file (§S4)** — re-run them rather than trusting this table.

**1. Where do the PTDF/LODF matrices come from? — SETTLED: pandapower.**

```
from pandapower.pypower.makePTDF import makePTDF   # (baseMVA, bus, branch, slack=None, ...)
from pandapower.pypower.makeLODF import makeLODF   # (branch, PTDF)
```

Both import clean in the repo venv (pandapower 3.4.0). Use these; do not hand-roll the algebra.
⚠️ `makePTDF` takes an explicit `slack=` — **record which bus you pass**, because the earlier
concern about slack-bus convention is real and a silent default is exactly how it bites.

**2. Topology changes between frames — SETTLED, and easier than feared.**

| grid | frames | all lines in service | **any bus split** | distinct `line_status` patterns |
|---|---:|---:|---:|---:|
| case14 | 6,000 | 80.4% | **0.0%** | 140 |
| neurips2020 | 12,000 | 67.8% | **0.0%** | 745 |
| wcci2022 | 4,000 | 72.8% | **0.0%** | 402 |

- **No substation bus-splitting occurs anywhere** — `topo_vect` never takes value 2 in any frame of
  any grid. Topology is therefore *fully described by `line_status`*, which is the good case:
  LODF needs to handle line outages only, not substation reconfiguration.
- **Do not restrict to intact-topology frames.** That would discard 20–32% of frames and breaks §0
  constraint 3.
- **Recompute LODF per distinct `line_status` pattern.** At 140 / 745 / 402 patterns this is
  cheap — build a cache keyed on the pattern, not one solve per frame.

**3. Thermal limits — SETTLED: back them out of the records, do not fetch them from Grid2Op.**

Each record carries `rho` (loading ratio) and `p_or`/`q_or` (flows). The per-line limit is
recoverable as `limit = |flow_base| / rho_base` wherever `rho_base > 0`. Doing it this way makes the
comparison immune to a definitional mismatch with Grid2Op — it is *by construction* the same limit
that produced `y`. Prefer this over re-deriving limits from the environment.

**4. Islanding — SUBSUMED BY A LARGER FINDING. Read this one carefully; it changes the arm.**

🚨 **The label is not "post-contingency overload". It is "overload OR the power flow fails to
solve."** Measured over every record of all three grids:

- `n1_violation == 1` and `n1_post_rho >= 1.0` → the overload class.
- `n1_violation == 1` and `n1_post_rho == 0.0` → **the solve-failure class** (divergence, islanding,
  game-over). `post_rho` is exactly `0.0` in **100%** of these.
- `n1_violation == 0` and `n1_post_rho >= 1.0` → **zero cases, on all three grids.** The label is
  one-sided and clean.

| grid | positives | **solve-failure share of positives** | share of all labels |
|---|---:|---:|---:|
| case14 | ~~31,390~~ **32,888** | ~~26.5%~~ **25.3%** | 6.9% |
| neurips2020 | ~~131,933~~ **137,315** | ~~21.7%~~ **20.8%** | 4.0% |
| wcci2022 | ~~182,312~~ **183,840** | ~~25.4%~~ **25.2%** | 6.2% |

⚠️ **Counts corrected 2026-09-19.** The struck figures came from §S4's `y.sum()`, which adds `-1`
for every de-energized line and so undercounts by exactly the number of out-of-service entries. The
bold figures count in-service lines only — the entities actually scored. Also measured: **no
in-service line is ever unevaluated on any grid**, so `line_mask` is a no-op everywhere.

🚨 **CORRECTED 2026-09-19 — the paragraph below was wrong, and it was this file's central
prediction.** A game-over is overwhelmingly a *structural* event: the outage strands a load or a
generator. A topology pre-screen — which every real contingency tool runs before any flow
calculation — reads exactly that, and it caught **55.1% / 88.5% / 71.7%** of the game-over class at
**precision 1.000** on all three grids (43,063 predictions, zero false alarms). The ceiling is real
only for the *flow channel*; the method as a whole is not bounded by it. The error was conflating
one channel with the method. See `thesis_findings.md` §25.4. The original argument follows.

**Why this is the most important thing in this section.** A DC/LODF screen **always solves** — that
is the point of a linear model. It therefore *cannot produce* the solve-failure class of positive at
all. Roughly **a fifth to a quarter of every positive label is structurally unreachable by LODF**,
which puts a hard ceiling on its recall before a single line of code is written.

That is not a flaw in the arm. **It is the thesis's central claim, made quantitative and falsifiable
for the first time** — "producing an N-1 label needs a power-flow solve" now has a specific,
measured referent: the ~22–26% of positives that exist *because* the solve failed. Report A1 with
this decomposition, not as one aggregate F1:

- **F1 on the overload-only subset** — the fair test of LODF as a screening method.
- **F1 on the full label** — the honest test, showing the ceiling the linear method cannot pass.
- The gap between the two is the measurement, and it is a better result than either number alone.

⚠️ **These counts are over raw records — every line in every frame.** The harness scores only lines
present in that frame (`batch.line_mask`) and neurips2020 only on its test split, so these figures
are **not** the reported violation rates (0.3104 / 0.4345 / 0.3969 all-positive) and must be
**re-derived under the harness's masking before being cited.** Treat them as sizing, not as results.

**Cost.** The largest arm here. Half a day to a day if the topology question resolves cleanly, more
if it does not.

---

### A2. Replicate the Pavão alert module on our data — the honest head-to-head

**What.** Pavão et al. (§B1) trained a fully-connected network, two layers of 256 units, on
hand-engineered features, for structurally our task. Reimplement that architecture and feature set
against **our** labels, **our** splits, **our** scorer. This converts a paper we can only cite into
a baseline arm we can actually table.

**Their feature list, verbatim from §5.3:** line loads of each transmission line; upcoming
maintenance on the critical lines; topological distance (how much the grid has been changed from
its initial configuration); sum of how long each line has been overloaded; datetime features. They
also report trying measured and forecasted load and generation per area and finding "little to no
benefit" — which corroborates our own finding that trend features did not help the forecast task
(`thesis_findings.md` §9.2).

**Traps:**

- ⚠️ **We do not have all their features.** Maintenance schedules and cumulative overload duration
  are not in our records; datetime may be recoverable from the chronic. **Report the substitution
  explicitly** — a partial-feature replication is a weaker claim than a replication, and must be
  labelled as one.
- ⚠️ This is **not** "we beat Pavão". Their task, label and grid differ (§B1). It is "their
  architecture, applied to our task, scores X against our GNN's Y" — a statement about
  architectures, on our benchmark only.
- Must honour the neurips2020 train/test split and be scored at held threshold as well as best.

**What it buys.** Together with the existing head-only ablation (0.8411 vs 0.8987 on val, `+0.058`
for message passing), it turns a two-point story into a three-point one: no graph + hand features
(A2) → no message passing (existing ablation) → full GNN. That is a proper architecture ladder, and
it is the cleanest available answer to "is the graph structure earning its keep?"

**Cost.** Low-to-moderate. The data loading already exists; it is an MLP and a feature builder.

---

### A3. Plain tabular baseline

**What.** Logistic regression and a gradient-boosted tree on the per-contingency endpoint features
already assembled in the readout (`h_or ‖ h_ex ‖ edge_attr_k ‖ x_or ‖ x_ex`, or just the raw
portion of it). `sklearn 1.8.0` is installed; **`xgboost` is NOT** — use
`HistGradientBoostingClassifier`, no new dependency.

**What it buys.** Forecloses "your GNN is an overparameterised tabular model". Cheap insurance
against a reviewer question, and if the tree does well it is a genuine finding rather than an
embarrassment — it would say the task is mostly local.

**Cost.** Low. Half of it is already done by the head-only ablation.

---

### A4. The `thesis_findings.md` §24 controls — already specced, do not duplicate

§24 is a pickup-ready spec for two arms written to answer a *different* question (is the thin rule
corpus the pipeline's fault or the task's?), but both double as comparison baselines:

- **§24.2 — depth-≤4 decision tree** on the 14 context variables, targeting `predicted secure AND
  was a violation`, fit on the neurips2020 **train** split only. As a comparator it is the ceiling
  for frame-level symbolic gating.
- **§24.3 — 10–15 hand-written expert rules**, run as a third arm **never merged into the corpus**.
  🚨 Its contamination trap is real and documented there: rules written after looking at
  `results/audit/*` are the extracted corpus laundered through a human, not a baseline.

**Action for the execution session: read §24 and run those arms under §24's protocol, not a new
one.** Cross-reference, do not restate. If they are run, update both documents.

---

### A5. Cheap arms already half-built

Not new work, but they belong in the comparison table and are currently scattered:

- **Head-only ablation** — 0.8411 vs 0.8987, `+0.058` for message passing. ⚠️ **The checkpoint
  does NOT exist on disk** (checked 2026-09-19 — only `gnn_checkpoint_n1.pt`). This is a retrain
  with `--head-only`, not a re-measurement.
- **All-positive and best-single-rule baselines** — computed by every cross-topology run. Exist.
- **Oracle (best-threshold-per-grid) ceiling** — 0.8972 / 0.4477 / 0.5721. Exists. Must stay in its
  own column, never mixed with held.
- **Batch-size sensitivity** — 0.8972 @ 64 vs 0.9255 @ 512, same checkpoint, same split. Exists
  (`gnn_n1_tightening.md` §8). Belongs in the results section as a *methodological* disclosure,
  because it is the reason every quoted figure carries a batch size.
- **Train / val / test fit** — measured 2026-09-18 at batch 64: **0.8876 / 0.8924 / 0.8972**,
  train−val gap **−0.0048**. Train scores *worse* than val despite a higher positive rate (22.5% vs
  20.4%). Evidence of no overfitting; worth one line in the results section, and it currently
  appears in no document. ⚠️ Measured once, by one script, not independently re-derived —
  re-measure before citing.

---

## B. Literature positioning — what `docs/lit` can and cannot support

Surveyed 2026-09-18, all 18 PDFs. **B1–B6 are the folder.**

⚠️ **B7 is not.** It was found by targeted web search on the same date and is the closest published
antecedent to Component D — closer than anything the folder holds. It is recorded here because it
belongs with the positioning material, not because the survey found it. Keep the distinction
visible: a claim of the form "no prior work does X" is only as good as the search behind it, and
the folder was assembled for a different purpose.

### B1. Pavão et al. — the one near-comparator

> Pavão, Marot, Sintes, Eriksson Möllerstedt, Crochepierre, Chaouache, Donnot, Dang, Guyon.
> "AI challenge for safe and low carbon power grid operation." *Energy and AI* 22 (2025) 100564.

**§5.3 describes structurally our task**, in their words: *"Our goal was to predict the probability
that a disconnection of any of the 21 critical lines would cause a blackout within the next hour…
This is a multilabel classification problem: for each time step t, predict a 0 or 1 for each of the
21 contingencies."*

| | Pavão §5.3 | ours |
|---|---|---|
| shape | per-line binary, multilabel per frame | same |
| model | fully-connected, 2 × 256 | GATv2, 3 layers + edge readout |
| training samples | 252,000 (scaled from 48,000) | 702,618 contingency labels |
| label | blackout within 1 h, **under the La Javaness agent's control** | thermal violation, controller-free |
| metric | TP rate 93.9% (89.6–97.3), TN 96.3% (89.7–98.6) | F1 0.8956, AP 0.9549 @ batch 64 |
| cross-topology | **not evaluated** | three grids, held threshold |

**The two things worth more than the numbers:**

1. *"All top teams chose a rule-based approach for raising alerts if the power line is overloaded
   with rho ≥ 1.0."* — **that is exactly our best-single-rule baseline, and exactly the predicate
   the shield enforces (`loading_pct > 100`).** Independent confirmation from the challenge
   organisers that the baseline we benchmark against is the one practitioners reach for. This is the
   single most useful sentence in the folder and it should be quoted in the results section.
2. Their learned module scored **77.1** assistant score against the best rule-based contestant's
   **61.13** (Artelys; La Javaness 46.55, N-side 48.97) — learning beats the rho-rule by ~1.26×. We
   report ~1.93× in-distribution. Different scoring systems, same direction, larger margin for us.

**Also useful:** they scaled training data 48k → 252k and gained ≈+2% TP / ≈+3% TN, and "suspect
that this trend would continue". Context for our own data-scale discussion.

> 🚨 **Three caveats that must travel with any use of this paper, every time:**
>
> - **Their label is agent-conditional** — it measures the safety of one specific control program.
>   Ours is controller-free physics. These are different questions.
> - **TP/TN rates on a self-described "skewed" label distribution are not comparable to F1 at an
>   18–43% positive rate.** **Never put 93.9% next to 0.8956 in a table.** Prose only.
> - Different grid (118-node IDF-2023, 21 critical lines) and different horizon (1 hour ahead,
>   cascading to blackout, vs immediate post-contingency thermal violation).

**Verdict: cite prominently in prose as the closest published analogue; do not table against it.**
The admissible way to get a number out of this paper is A2 — rebuild their architecture on our data.

---

### B2. Younesi et al. — the related-work table, and a shield contrast (B7 is the nearest antecedent)

> Younesi, Siano, Moradpour, Mehrizi-Sani. "Hybrid neuro-symbolic learning and reasoning for
> resilient load restoration in smart microgrids." *Renewable Energy* 256 (2026) 124401.

- **Its Table 1 is a ready-made related-work table** — 14 neuro-symbolic power-systems studies,
  2020–2025, columns Method/Model · Validation · Key contribution. Mirror this structure.
- **Not one of the 14 reports a metric comparable to another.** That is itself a citable finding:
  this subfield has no shared quantitative benchmark, which is why §A exists and why our comparison
  is internally constructed. Say this explicitly rather than apologising for it.
- **Its ref [12] (2024, "shielded safe reinforcement learning")** — the paper distinguishes it in a
  sentence we can use: that shield *"acts reactively on a continuous reinforcement learning
  policy."* Ours gates a supervised per-line classifier post-hoc, with rules **extracted from
  standards documents**, not hand-specified. That is the difference to state.
  ⚠️ **This was written as "the nearest published antecedent to Component D" — it is not.**
  **§B7 is nearer on every axis** (same simulator family, explicit ρ ≥ 1.0 gate, reported deltas).
  Ref [12] is still worth citing, but as the *reactive-on-RL* contrast, not as the closest relative.

### B3. Giunchiglia et al., CCN+ — the methodological foil

> Giunchiglia, Tatomir, Stoian, Lukasiewicz. "CCN+: A neuro-symbolic framework for deep learning
> with requirements." *International Journal of Approximate Reasoning* 171 (2024) 109124.

Constraint enforcement over neural predictions — our shield's design space. The contrast is clean
and worth a paragraph: they inject requirements **at training time**; we gate **post-hoc**, which is
what lets us swap the corpus without retraining (and is why §22.3's identical-metrics result across
two corpora is even measurable).

### B4. Bohne et al. — structural analogue for B+C+D

> Bohne, Windler, Atzmueller. "A Neuro-Symbolic Approach for Anomaly Detection and Complex Fault
> Diagnosis Exemplified in the Automotive Domain." K-CAP '23. DOI 10.1145/3587259.3627546.

Neuro-symbolic fault diagnosis with a knowledge graph, different domain. Cite for "this
architecture pattern exists outside power systems". No numbers.

### B5. Ahmadi, Aly & Gu — the review, with a caveat

> Ahmadi, Aly, Gu. "A comprehensive review of AI-driven approaches for smart grid stability and
> reliability." *Renewable and Sustainable Energy Reviews* 226 (2026) 116424.

Large per-subdomain tables (Table 7; Appendices A and B list "typical datasets, common metrics").
Mineable for context on what the fault-detection literature reports.

⚠️ **A tempting argument that is NOT yet verified:** that the high accuracies in this literature are
mostly on fault *classification* tasks structurally like our abandoned closed-form `classify`
target — which would make abandoning that task a contribution rather than a retreat. **This is a
hypothesis.** It requires actually reading Appendix B's task definitions before it can be asserted.
If it survives that check it is a strong paragraph; if it does not, drop it silently.

### B6. Not usable for comparison — 11 of 18

Solar forecasting, electricity theft, load management, smart-city DNN modelling, supply-chain AI, an
AI-ethics framework for smart grids, nano-grid/digital-twin review, energy-storage integration,
neural-SVM energy productivity, and the two geoscience neuro-symbolic KG papers (Chen et al.,
*Applied Computing and Geosciences* 2025 and 2026 — architecturally analogous, wrong domain, cite in
related work at most). Wrong task, wrong metric, wrong system. Do not stretch them.

---

### B7. Malik — the closest published antecedent to Component D  ⚠️ NOT in `docs/lit`

> Malik. "Hierarchical Reinforcement Learning with Runtime Safety Shielding for Power Grid
> Operation." arXiv:2604.14032v1, 15 April 2026. Delhi Technological University.

Found by targeted web search 2026-09-18, not by the folder survey. **It shares our experimental
substrate** — Grid2Op, with `l2rpn_case14_sandbox` for training and ablation and
`l2rpn_icaps_2021_large` for zero-shot evaluation. No other shield antecedent we know of runs on the
same simulator family, which is what makes this the one to position against.

**Their shield.** One-step forward simulation of each candidate action; the action is admissible if
the predicted `max ρ ≤ ρ_max`, *"typically set to 1.0"*. On a violation it searches a restricted
neighbourhood of pre-computed safe primitives — line disconnections, conservative redispatch — for a
corrective action.

**Reported (stress test with forced line outages):**

| arm | avg steps survived | max line loading | vetoes |
|---|---:|---:|---:|
| flat RL, unshielded | 50.35 | 1.21 | 0 |
| hierarchy + shield | 200.0 | 0.85 | 0.25 |

**Three differences, and together they are the Component D contribution stated precisely:**

1. **Their rules are hand-designed. Ours are extracted from published standards by an LLM.** This is
   the only one of the three that is a novelty claim rather than a design choice, and it is the
   whole of Components B and C. State it as the difference.
2. **They gate a reinforcement-learning control policy; we gate a supervised per-line classifier,
   post-hoc.** Different object, different failure mode, different thing at risk.
3. **Their gate decides by running a simulation; ours evaluates a stated predicate against observed
   telemetry** — cheaper, and citable back to the clause it came from.
   ⚠️ **Do not present 3 as a straight win.** A simulation-based gate is *strictly better informed*
   than a predicate over the present observation; ours is cheaper and auditable, not more accurate.
   The honest framing is a cost/provenance trade, and it is the same asymmetry §A1 probes.

**🚨 Third independent confirmation of the `loading_pct > 100` predicate.** After (a) Pavão's *"all
top teams chose a rule-based approach for raising alerts if the power line is overloaded with
rho ≥ 1.0"* (§B1), and (b) our own corpus, where it is the most-corroborated rule in the extraction
(10 clauses, 4 documents, 2 identified bodies). Malik arrives at the same threshold from a third
direction — hand-designed, for control rather than screening. That three unrelated routes converge
on one number is worth a sentence in the results section.

⚠️ **Read at abstract / page-summary level only.** Verify the shield mechanism and the stress-test
protocol from the full paper before citing the 50.35 → 200.0 figures.

⚠️ **Not a comparison arm and cannot become one.** Different task (control, not screening),
different metric (episode length and peak loading, not F1 over a score vector). **Table 2 and prose
only** — it fails §0 the same way §B1 does.

---

### B8. The targeted search itself — what was looked for, and what came back

Run 2026-09-18, outside `docs/lit`. Recorded because **§F's novelty claim is only as good as this
record**, and because a reader is entitled to know where we looked before we say nobody has done
something. Everything below was read at **abstract / landing-page level only** unless stated.

**Queries, in five directions:** GNN N-1 contingency screening with F1; GNN static security
assessment and contingency classification; ML contingency screening benchmarked against DC power
flow with false-negative rates; GNN power-grid generalisation to unseen topologies; rule-based
shielding and runtime verification of neural predictions in power systems. A sixth — LLM extraction
of rules from power-system standards — is covered in §F.

**What it found that mattered:** §A1's *Precedent* (Alcántara & Chatzivasileiadis) and §B7 (Malik).

**What it found that did not, and why — keep this, it is the answer to "did you check?":**

| work | reports | why it is not our number |
|---|---|---|
| Salako et al., arXiv:2609.04300 (Sep 2026) | **F1 0.97** (IEEE-30), **0.86** (IEEE-14) | 3-class severity (safe/moderate/severe) from an Overall Performance Index computed from a Newton-Raphson solve the model's own inputs come from. ⚠️ **Verify from the full paper** — if it holds, this is the closed-form-target trap that killed our `classify` task, and it is §B5's hypothesis with a name |
| ICNN screening, arXiv:2410.00796 | **zero** false negatives, 2–5% false positives | feasibility of an injection, not per-line overload; IEEE 39-bus only; no F1; no cross-grid arm |
| Contingency-case ML study, arXiv:2008.09384 | 97–98% accuracy, 0.0–0.64% false negatives | classifies whether a **time step** is critical, not whether a given outage violates a limit |
| GNN contingency analysis, arXiv:2310.04213 | ~126–400× faster than Newton-Raphson; MAE on line flows | predicts voltages and angles — a regression surrogate. No violation classification at all |
| Donnot et al., arXiv:1805.02608 (2018) | ranks N-1/N-2 by presumed severity | a **ranking** to prioritise simulation, not a per-contingency decision |
| AC-OPF generalisation (HH-MPNN et al.) | <1% gap 14→2,000 bus; <3% zero-shot on unseen topologies | optimal power flow, not contingency screening |

> 🚨 **Correct a claim a search summary will hand you.** A snippet asserts that arXiv:2310.04213
> beat DCPF on line-flow violations. **It does not.** The paper was fetched: its only baseline is
> Newton-Raphson and it reports MAE/MSE, never violations. Do not let that into the thesis.

**The useful negative, and it is the load-bearing one.** Across all five directions, **no work was
found that reports per-line binary N-1 violation scores, at a threshold selected once and held fixed
across several different grids.** The nearest neighbours each drop one of those three properties:
they score per time step, or per grid with its own tuning, or on one grid only. *That* is why our
comparison must be internally constructed — a better reason than "our folder lacks one," and it also
explains why our case14 failure has no counterpart in the literature: almost nobody runs the test
that would expose it.

⚠️ **This is a targeted search, not a systematic review.** It is enough to say "we looked here and
found none"; it is **not** enough to say "none exists." §F holds that line.

---

## C. Traps — things that look like comparisons and are not

1. **Quoting any external accuracy next to our F1.** Different label, different base rate, different
   grid. Every one of them. Prose, never a table row.
2. **Mixing held-threshold and oracle columns.** Already a standing warning in CLAUDE.md. A new
   baseline that tunes per-grid belongs in the oracle column.
3. **Comparing arms scored at different eval batch sizes.** See §0, constraint 1.
4. **Treating the case14 threshold sweep as a rescue.** Measured (`results/threshold/threshold_sweep.json`):
   case14 goes 0.4167 (β=1) → **0.4428** (β=5), which nearly closes the gap to its own oracle
   (0.4477) — but **0.4428 is still below the rule baseline (0.5392) and barely above all-positive
   (0.4345).** Threshold choice does **not** rescue case14; it only shows the held-vs-oracle gap
   there is mostly threshold transfer. The failure stands and must be reported as a failure.
5. **Expert rules written after looking at the audit output** — §24.3's contamination trap.
6. **Claiming A2 as "we beat Pavão".** It is their architecture on our benchmark. Nothing more.

---

## D. Proposed shape of the results section

Two tables, and they must stay separate.

**Table 1 — method comparison, held threshold, per topology.** One row per arm, one column group per
grid. Arms, in ascending order of information used:

```
all-positive
best single rule (rho of removed line)
DC-LODF screening                      [A1]  <- RETIRED 2026-09-20; see §A1
logistic / GBT on endpoint features    [A3]  <- DONE; 0.9190 / 0.5845 / 0.5177 held
MLP, Pavao feature set                 [A2]  <- DROPPED, see thesis_findings.md §26.5
GNN head-only (no message passing)
GNN full
GNN + shield
```

with a separate, clearly-labelled **oracle** column showing the best-threshold ceiling per grid.

**Table 2 — qualitative related work**, in the Younesi Table 1 shape: Ref · Year · Method ·
Validation · What it does that we do not / we do that it does not. Carries B1–B4 **and B7** — B7 is
the Component D row and should sit adjacent to B2's ref [12] — and states in its caption that no
entry reports a comparable metric.

**Plus a disclosure paragraph** covering: eval batch size, held-vs-oracle threshold discipline, the
case14 failure, and — per §24.5 — any control that was specified and not run, **in those words**.

---

## E. Execution order for the next session

Dependency-ordered. Each step is independently reportable, so stopping early still leaves the
results section better than it is now.

1. ✅ **DONE 2026-09-18 — resolve A1's open questions** (PTDF source, topology handling, limits,
   islanding). All four settled by measurement; see §A1 and re-derive with §S4. **Do not redo this.**
   ⚠️ The fourth resolution changed the arm: ~22–26% of positives are solve-failure cases a linear
   method cannot produce, so step 2 must report **two** F1s (overload-only and full-label).
2. 🚫 **RETIRED 2026-09-20 — A1, DC-LODF screening.** It ran on all three grids and won on all
   three; the arm was then removed from the thesis's scope and its code and artifacts deleted.
   Replaced by the inference-cost benchmark, `thesis_findings.md` §25. See §A1.
3. ✅ **DONE 2026-09-19 — A3, tabular baselines**, re-scoped to all three grids.
   **Logistic, GBT and the GNN all lose 31–50% off-distribution.** `thesis_findings.md` §26.
   ⚠️ The "…and LODF loses 4%" half of this line went with the retired arm (§A1); the
   three-learned-architectures finding does not depend on it.
4. ❌ **DROPPED — A2, Pavão architecture replication.** It measures model capacity, not
   transfer, and needs features our records do not carry. §26.5. The spec stays in §A2 in case
   the architecture ladder is wanted for a different reason.
5. **A5 — collect the existing arms** into one table; re-measure the train/val/test row.
6. **A4 / §24** — the ceiling tree and, if the contamination ordering can be honoured, the expert
   rules.
7. **B5 verification** — read Appendix B of the Ahmadi review and decide whether the
   classify-task-framing argument survives.
8. **Write Table 2** from B1–B4 and B7.
9. **Verify the two papers read only at abstract level** — §A1's Alcántara & Chatzivasileiadis
   (DCPF protocol, definition of "critical violation") and §B7's Malik (shield mechanism,
   stress-test protocol). Neither number may be quoted until this is done.

**Where results go:** new measured numbers to `thesis_findings.md` as a new section; this file gets
each arm marked DONE with a pointer. Do not let this file accumulate results — it is a plan, and a
plan that also holds findings is how `component_d_plan.md` went stale.

---

## F. The novelty claim for Components B, C and D — what may and may not be said

§A builds comparators for Component A because comparators for Component A exist. **For B, C and D
no comparator was found at all**, and that asymmetry has to be stated deliberately rather than left
as a silence in the results section. This section fixes the wording.

### F1. The sentence the thesis may write

> No prior work was found that extracts operational rules from published power-system standards
> using a language model and uses the resulting corpus to gate the predictions of a learned
> contingency screener. The search behind that statement is recorded in §B8 and §F2; it was targeted,
> not systematic. **No comparative number is therefore reported for Components B, C or D** — the
> results for those components are internal measurements against our own ablations, and should be
> read as such.

**Three things that sentence does deliberately.** It says *"no prior work was found"*, not *"no prior
work exists"*. It names where we looked, so the claim is auditable. And it states the consequence —
no comparative number — **in those words**, which is the same discipline `thesis_findings.md` §24.5
imposes on controls that were specified and not run.

### F2. The sentence the thesis may NOT write

🚨 **Never: "this is the first system to …" or "no one has done this."** One targeted search does not
license either. If a reviewer produces a counterexample, an unhedged claim costs the whole chapter's
credibility, and the hedged version costs nothing — it was never the contribution.

**Search coverage actually achieved (2026-09-18):** the LLM-rule-extraction direction was searched
for power-system standards specifically and returned **nothing domain-specific**. The general
technique is well populated — LLMs generating candidate formalisations from statutory and
transactional text — so the novelty is in the *application and the downstream use*, not in the
extraction idea. Say it that way; it is both more defensible and more accurate.

⚠️ **To upgrade F1 to a stronger claim, this is the bar:** a recorded systematic search — named
databases, date range, query strings, inclusion criteria, counts at each screening stage — that a
reader could repeat. Roughly half a day. **Until that exists, F1's wording is the ceiling.** Do not
quietly strengthen it during writing-up; that is how an unsupportable sentence gets into a thesis.

### F3. The antecedents that must be acknowledged anyway

An absence of comparators is not an absence of related work, and the thesis is stronger for naming
the near misses than for pretending to an empty field:

| component | nearest antecedent | the difference to state |
|---|---|---|
| **D** (shield) | §B7 Malik 2026 — same simulator family, same ρ ≥ 1.0 gate | their rules are hand-designed and enforced by forward simulation; ours are extracted from standards and evaluated as stated predicates. ⚠️ Their gate is *better informed* than ours — see §B7 point 3 |
| **D** (shield) | §B2 ref [12] — shielded safe RL | reactive, on a continuous RL policy; ours is post-hoc on a supervised per-line classifier |
| **D** (design space) | §B3 CCN+ | constraints injected at **training** time; ours gates post-hoc, which is why §22.3's identical-metrics-across-two-corpora result is measurable at all |
| **B + C** | §B4 Bohne et al. | neuro-symbolic fault diagnosis over a knowledge graph — right architecture pattern, different domain, no rule extraction from standards |

### F4. The honest reading, which must travel with the claim

**An absent baseline is not a result.** That B, C and D have no comparator makes their numbers
*harder* to interpret, not better — there is nothing external to say whether a 0.16% extraction
yield or a +0.0082 shield delta is good, bad or expected. The contribution is that the pipeline was
built, measured end to end, and its failure modes characterised; it is **not** that it beat anything.
State the limitation in the results section rather than letting a reader discover it.

This is the same standard applied to Component A throughout: on case14 the model scores 0.77× the
rule baseline and **loses to answering "violation" every time**, and that is reported as a failure.
B, C and D get the same treatment.
