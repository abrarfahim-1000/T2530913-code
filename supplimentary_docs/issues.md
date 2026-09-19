# Outstanding issues in `results_and_analysis.md`

*Originally compiled 2026-09-19 by reading [`results_and_analysis.md`](results_and_analysis.md) end
to end against [`thesis_findings.md`](thesis_findings.md) and the artifacts in `results/`.
**Rewritten 2026-09-20**, after the work ran, to carry **only what is still open.***

This file is not a list of errors in the chapter. Where a figure was wrong it was corrected in
place, and **§5.6 of the chapter is the record of every claim that changed** — including those that
changed on 2026-09-20. This is the list of things that are still true and still open.

Issue IDs are unchanged from the 2026-09-19 numbering so earlier references still resolve. Gaps in
the sequence are items that were closed.

> **Closed on 2026-09-20 and removed from this file:** **A2** (the threshold's exchange rate is now
> measured), the oracle-selection and pair-resolution halves of **A3**, **B5** (the evaluator admits
> `abs`/`min`/`max`, and the fourteenth expert rule now evaluates and loses), the auditability half
> of **B6**, **B4b** and **B4c** (the mis-stated reason and the duration miscount), **C1** (retired
> claims are now enforced by a test), **C2** (`thesis_findings.md` §17.1), and **C5** (retired by
> design). Each is recorded in the chapter's §5.6 with what replaced it. The test suite went from
> **302 to 469 passing**.

| level | meaning |
|---|---|
| **A** | A reader or examiner could press on this and the chapter has no answer. Decide before submission. |
| **B** | Disclosed gap. The chapter states it honestly; closing it would strengthen the work but nothing depends on it. |
| **C** | Hygiene — a trap, a standing decision, or housekeeping. |

---

## A. Open questions

### A1. The expert-rule arm is not blind, and its framing upstream is explicitly unresolved

**Where:** chapter §5.2.11 ("Fourteen rules written by hand"), §5.4.7, §5.5.3 · findings §29.1, §30.8

**This is the one live decision in the project.**

The fourteen hand-written rules were written after their author had already seen the served corpus,
the gate's precision figures and the tree's splits. §24.3's pre-registration ordering could not be
honoured, so §24.3's own fallback applies. The md5 fingerprint (`05a29bd7…`) prevents tuning against
the result but does not reconstruct blindness.

**The unresolved part is upstream.** `thesis_findings.md` §30.8 states that §29 is *deliberately left
unrevised* and that **"nothing in §29 should be quoted as final until it is reconciled with this
section."** The chapter could not leave a hole there, so it reconciled §29 on its own initiative: it
keeps the not-blind disclosure, adds the shared-100 finding, and argues in §5.5.3 that the blindness
limitation is *largely moot rather than repaired*, because enumeration answers *could anyone have
written a better rule?* — which strictly contains *could a human have?*.

**Decision needed:** confirm that reconciliation, or settle §29.1 differently and have the chapter
follow. If the decision is to drop the expert arm's independent weight entirely, the chapter loses
nothing load-bearing — enumeration carries the claim — but §5.4.7's route 5 disappears and its count
falls from seven converging procedures to six.

⚠ B5's repair strengthened this arm in passing: its conclusion now holds over all fourteen rules
rather than thirteen. That makes the arm more useful as corroboration and does nothing about the
blindness.

### A3. Three-term rules were never enumerated

**Where:** chapter §5.2.11 · findings §30.9

The other two limits on the enumeration closed on 2026-09-20 — selection no longer uses the answer
key, and the pair grid was re-run at 64 and 128 cutpoints without changing anything. This one
remains, because the search space cubes.

The evidence against three-term rules is indirect and consistent: two terms already add nothing over
one, at three separate resolutions, and on two of three grids the best pair is *worse* than the best
single. **Recommendation: leave it open and state it.** A three-term arm is not a small extension of
the pair arm's two matrix products, and the chapter's current sentence is the honest one.

---

## B. Disclosed gaps — measured honestly, not closed

### A4. The message-passing ablation cannot currently be re-measured

⏸ **Deferred by decision, 2026-09-20.** *(Keeps its A-series ID; it is a disclosed gap while deferred.)*

**Where:** chapter §5.2.3 · CLAUDE.md, Arm A3 note

The chapter reports **0.8987 → 0.8411, +0.058** for message passing. The head-only checkpoint
`gnn_checkpoint_n1_headonly.pt` **is not on disk** (confirmed 2026-09-19), so that figure needs a
*retrain*, not a re-measurement.

✅ **The caveat has been written.** §5.2.3 now states that the artifact is gone, that the figure is
single-seed, and that it clears its in-distribution band by roughly six standard deviations without
having been reproduced.

**When it is picked up**, the right shape is *not* a bare `--head-only` run, which would be compared
against a number from a different training episode. Run **both** arms at the documented protocol
(`--epochs 30 --batch_size 128 --lr 3e-4`) over seeds 42, 0, 1, 2 — the shape of
`evaluation/reactance_transfer.py`, roughly two minutes per run on this XPU. That replaces a
single-seed figure with a banded one rather than merely restoring it.

**Expect the absolute levels to move even if the delta holds.** Both numbers are validation F1 and
`BatchNorm(track_running_stats=False)` uses live batch statistics, so a fresh pair lands wherever the
trainer's validation batch size puts it. §5.2.3 will need both levels re-quoted **with the batch size
stated**. That is bookkeeping, not a finding.

**If the delta comes back materially smaller, the chapter gets more coherent, not less.** §5.1.5
already reports a gradient-boosted tree matching the GNN in-distribution, and §5.4.4 already reports
the graph model transferring worst of three learned architectures. **Not at risk either way:** §5.4.4
rests on the tabular baselines and LODF, not on this ablation.

✅ B2's metrics work is done, so whenever this runs it will record its own per-epoch history.

### B1. The deployed model has still never been retrained at multiple seeds

**Where:** chapter §5.3.6, §5.4.1, §5.5.3, §5.6

✅ **The gate's own figures now carry a measured band** — four architecturally identical control
checkpoints scored through the shield on all three grids, **twelve of twelve deltas positive**,
NeurIPS 2020 at **+0.0082 ± 0.0003**. The deployed checkpoint sits inside that band everywhere and
below the WCCI 2022 mean, so it is not a lucky seed.

**What remains:** the *model* bands (±0.02 generally, **±0.07 on case14**) are still transferred from
the reactance experiment's control arm rather than measured on the deployed checkpoint itself. The
chapter says so and claims no confidence interval on any headline raw-model figure. Every conclusion
drawn clears the band — case14's failure by 0.12, WCCI 2022's margin at roughly three sd — and no
narrow cross-topology comparison is drawn anywhere.

⚠ **One figure was narrowed by the seed work and must be quoted carefully.** case14 intervention
precision spans **0.816–0.924** across seeds, on 15–217 blocks. It is a range, never a point
estimate. The same applies to case14's delta: the sign is safe on all four seeds, the magnitude is
not.

### B2. No per-epoch history exists for the deployed N-1 model

**Where:** chapter §5.3.6 · CLAUDE.md, Model Training

✅ **Closed going forward.** `training/train_gnn.py` now persists per-epoch history, the train/val
gap, the full config actually used, the git commit, and `best_epoch` — the epoch the saved weights
came from, not the last epoch run. The deliverable stays a bare `state_dict` (verified sha256
identical) because `load_state_dict` is strict and five call sites depend on the old form.

**What remains is unrecoverable.** The deployed checkpoint was saved before any of that existed, so
its epoch count and training curve are gone. The practical half is closed — §5.2.4's seed-42 control
shows the documented command reproduces the deployed cross-topology behaviour to within 0.0006 — but
no training curve exists for the model the thesis ships, and none can be produced without retraining.

### B3. Validator-seed stability is unmeasured for the arm that was kept

⏸ **Blocked on hardware.** This machine has no CUDA and no Ollama (verified: `nvidia-smi` absent,
port 11434 dead).

**Where:** chapter §5.3.6, §5.2.8

Both arms of the §5.2.8 prompt A/B ran at a single seed and temperature 0.0. Verdict stability across
repeated runs is established for the `strict` arm and **not** for the `translated` one — the arm that
produced the served corpus.

**This matters more than when it was filed.** B6 established that the `translated` arm also fabricated
reasons, so reason quality in the served arm is a live question rather than a formality.

✅ The harness is written and unit-tested — 26 tests over synthetic fixtures, isolation guards for the
trap where `deduplicate_rules` silently merges every `*_confirmed.jsonl` in a shared directory, and a
preflight that exits 2 with nothing written when Ollama is unreachable. **It is unexecuted.** Run on
the LLM host:

```bash
python evaluation/replicate_validation.py --repeats 3
```

Roughly one minute per repeat; the recorded 2026-08-20 run took 48.9 s. Budget ten minutes cold for
the model load. Verdict stability and reason stability are reported **separately and never blended**,
because §5.2.8 is precisely the case where the two diverge.

⚠ **A structural limit that cannot be removed without re-running stage 3.** `validate.py` writes bare
rules to `*_confirmed.jsonl` with no verdict object and therefore no reason, so reason stability is
recoverable for the **21 rejections** and **not** for the **11 confirmations**. The artifact reports
`n_unmeasurable` explicitly and returns `null`, never `true`.

⚠ **`CLAUDE.md` is wrong about `VALIDATOR_MODEL`.** `extraction/common.py` pins the validator tag as a
literal and reads no environment variable; only `EXTRACTOR_MODEL` does. The runner refuses to start if
the env var disagrees with the pinned constant, rather than validating with a different model than the
operator believes. The one-line fix is noted in the runner's header and has not been applied.

### B4. The voltage ride-through conclusion clears its bar by under 1%

⏸ **Simulation arm deferred by decision, 2026-09-20.** *(B4b and B4c were documentation corrections,
needed no simulator, and are done.)*

**Where:** chapter §5.2.5, §5.5.3 · findings §20.2 ·
`results/audit/andes_{voltage,frequency}_spike.json`

The two halves of the dynamic-simulator rejection are not equally strong, and the chapter says so:

| arm | margin | how to quote it |
|---|---|---|
| frequency | worst N-1 excursion 0.3293 Hz against a 0.6 Hz bar — a factor of ~1.8 | **settled under N-1** |
| voltage | deepest clean-opening dip clears the mildest threshold by ~0.006 pu once rebased | **measured, thin margin** |

**The exposure is bounded, and tightly.** From the harvested threshold ladder:

| under-voltage threshold | rules citing it | distance from the current deepest dip (rebased 0.9557) |
|---:|---:|---:|
| 0.95 | **1** | 0.0057 |
| 0.90 | **16** | 0.056 |
| 0.88 | 1 | 0.076 |
| 0.85 | 7 | 0.106 |

The famously thin margin is thin against a threshold **exactly one rule uses**. Crossing it is a
one-rule event; reaching the sixteen rules at 0.90 pu needs ten times the current margin.

**Coverage is the weaker flank, not depth.** Arm A tripped eight lines, a subset of the branch set, so
the deepest clean-opening dip is a minimum over part of the N-1 set rather than all of it. One
asymmetry to record if a crossing ever occurs: in Arm B the post-clearing tail is milliseconds, which
is why those envelopes are reachable by the event and never by a state. Arm A has no such escape — a
clean opening settles, so a dip below threshold persists for the rest of the window and satisfies
every duration test at once. **Depth is the only gate there; duration comes free.**

**The stress arm, when it runs**, is a parameter sweep rather than new machinery: ANDES 2.0.0 is
installed and `sanity/andes_voltage_spike.py` and `sanity/andes_frequency_spike.py` already exist.
Three knobs — scale load to a heavier dispatch, add the untested trips, extend the N-2 escalation
ladder (which the frequency arm already has, showing N-2 at 0.7975 Hz against the 0.6 Hz bar) to the
voltage arm, and push past the 20-second window.

🚨 **This is the item most able to overturn a conclusion rather than shore one up.** If a heavier case
crosses the voltage bar, §5.2.5's rejection of a dynamic simulator needs **rewording, not a caveat.**
Worth scheduling before submission.

**One sub-gap is not closable by simulation at all:** nine of the 103 ride-through rules carry
free-variable thresholds and can be neither fired nor shown inert.

### B6. Known validator errors, disclosed and not corrected

**Where:** chapter §5.2.8 · `results/audit/rejection_adjudication.json`

✅ **The audit exists and the counts are now evidence-backed.** All 21 shared rejections were re-read
against their source chunks: **11 sound, 6 wrong, and 4 that should have been corrected rather than
rejected.** The artifact separates the validator's verbatim verdict and reason (mechanical, from
disk) from the classification (a reading, with its basis recorded), so a reader can reject an
individual call without discarding the audit.

**What remains open: the errors themselves are not corrected**, and deliberately so — correcting them
means re-running stage 3, which would invalidate the replication result in §5.3.6. The chapter reports
them rather than suppressing them.

⚠ **`thesis_findings.md` §12.4 is now known to be wrong in several specific ways** and has not been
revised. Its table covers only 17 of the 21; four rejections (**R_260, R_1444, R_1575, R_1152**)
appear in no row, and two of those are among the wrong ones. Its *"roughly 17 of 21 rest on a
defensible reading"* gives 11 on this reading; its SHOULD-HAVE-BEEN-CORRECT membership names R_741
where the audit finds R_1108; and at least two of its rows quote the **`strict`** arm's reason while
discussing the served arm. The chapter does not depend on §12.4 — it cites the artifact — but a reader
who checks the findings document will find the older numbers. **This is the same class of problem C2
fixed in §17.1, and it is the author's document to correct.**

---

## C. Standing decisions and hygiene

### C3. Nothing is committed

`supplimentary_docs/results_and_analysis.md` and this file are untracked on branch
`shield-v2-explanation-channels`, alongside the uncommitted 2026-09-19 and 2026-09-20 work —
`evaluation/{exhaustive_rules_n1,expert_rules_n1,ceiling_tree_n1,audit_rejections,replicate_validation}.py`,
`results/{ceiling,seeds,lodf,reactance,tabular}/`, `expert_rules/`, `tests/test_docs_hygiene.py`, and
modifications to `CLAUDE.md`, `extraction/common.py`, `shield/evaluator.py`, `training/train_gnn.py`
and `.gitignore`.

🔸 **The author confirmed on 2026-09-20 that nothing is to be committed.** A standing decision, not a
pending task — recorded so it stays a decision rather than an oversight. Note that each further
session adds to the same pile.

### C4. `ResultsAnalysis.pdf` must not circulate alongside the chapter

Both drafts in the PDF predate the 2026-09-19 comparison arms, and their
`[PLACEHOLDER: EXTERNAL SOTA BASELINES]` blocks are now filled with a result that **reverses the
emphasis** of the older text: DC/LODF beats the model on all three grids. §5.6 of the chapter lists
every claim that changed. The PDF's Version 1 skeleton is also retired — the chapter follows
Version 2, with that draft's `5.6 Discussion` renumbered to 5.5.

### C6. Editing the guarded documents

`tests/test_docs_hygiene.py` scans `results_and_analysis.md`, `thesis_findings.md` and `CLAUDE.md`
for **nine retired claims** and fails if one reappears outside a withdrawal notice. It checks for
**absence only**, so ordinary edits cannot break it. Two mechanics are worth knowing before editing
those documents.

A withdrawal notice is recognised by a marker — *withdrawn*, *superseded*, *must not be quoted* and
so on — within six lines of the quoted claim, and **the marker phrase must sit on a single line**,
because a phrase split across a line break does not match. That is not hypothetical: it failed the
build once on 2026-09-20, and the fix was to reword the notice rather than to weaken the guard.

The pin list, for a retired claim knowingly left live in a file the current task may not edit, is
**empty**, and a test keeps it that way. Adding a pin is therefore a deliberate act that shows up in
the diff with its reason attached.

---

## What still points at the chapter

Nothing in `results_and_analysis.md` is currently waiting on unfinished work. These five are waiting
on a decision, on hardware, or on a deferred run, and each names what it would change.

| item | what would change in the chapter |
|---|---|
| **A1** — the §29.1 blindness decision | If settled differently from the chapter's reconciliation, **§5.2.11, §5.4.7 and §5.5.3** follow, and §5.4.7's count drops from **seven** converging procedures to six. |
| **A4** — the head-only retrain | **§5.2.3's two numbers change** and the caveat comes out. Both levels will need re-quoting with the batch size stated. |
| **B4** — the ANDES stress arm | If the stress arm crosses the voltage bar, **§5.2.5's rejection needs rewording, not just a caveat.** This is the one that could still overturn something. |
| **B3** — the validator replication | Once run on the LLM host, **§5.3.6's disclosure becomes a result** — or, if verdicts prove unstable, a considerably larger problem for §5.2.8. |
| **A3** — three-term rules | Only if they are ever enumerated, which is not recommended. **§5.2.11's closing limitation** would then change. |

Two further items point at other documents rather than at the chapter: **B6**'s §12.4 discrepancies
in `thesis_findings.md`, and **B3**'s `VALIDATOR_MODEL` note in `CLAUDE.md`. Both are one-edit fixes
in documents the author owns.
