# Why we did not use a dynamic simulator

**A standalone account of the ANDES investigation — what it was supposed to buy, what was measured, and why it was rejected.**

Status: **INVESTIGATED AND REJECTED on evidence, 2026-09-17. Do not re-open.**
Written to be read on its own, without the codebase open.
The same material appears as `thesis_findings.md` §20.2; this file is the readable version.

---

## 1. The problem it was meant to solve

The rule corpus extracted from the standards contains **457 rules about frequency** — statements
of the form *"if system frequency falls below 59.4 Hz, disconnect within 2 seconds."* That is the
single largest untapped block of rules in the whole corpus.

None of them could be used, for a blunt reason: **the simulator does not model frequency.**

Grid2Op is a *quasi-static* simulator. It takes a snapshot of the grid every five minutes and
solves for where the power flows. Frequency is not a quantity it represents at all — not
approximately, not badly, simply not at all. A rule that mentions frequency cannot be evaluated
against a grid state that has no frequency in it.

The same held for a second family: **103 voltage ride-through rules**, which say what a generator
must tolerate *during* a disturbance, and are written in terms of both a voltage depth and a
duration in seconds. Grid2Op's clock ticks once per 300 seconds. A rule about what happens in the
first 0.15 seconds has nowhere to live.

**The proposal ("Solution 3")** was to bolt on **ANDES**, a transient/dynamic power system
simulator that models the grid second-by-second and *does* have frequency, and re-evaluate the
corpus against it. On paper the prize was large: **1,252 of 2,463 candidates (50.8%) are blocked
specifically because their stated reason names frequency or sub-second time** — the count a dynamic
simulator would, on paper, recover — against the 58 the current pipeline produces.
(`evaluation/capability_gap.py`; see §5 below — an earlier version of this page quoted 726, produced
by hand and never reproducible.)

---

## 2. What was actually tested

Rather than build the pipeline and find out, the question was turned into a measurement that could
be run in an afternoon: **does the event this thesis is about move these quantities at all?**

Test rig: ANDES 2.0.0, the IEEE 14-bus case, 60 Hz base, governors present (TGOV1), 20-second
window, disturbance applied at t = 1 s and never reverted.

One detail worth recording, because it nearly invalidated the run: the shipped `ieee14_linetrip`
case **reconnects the line 0.1 s later**. That is a fault-clearing cycle, not a line outage. A
permanent outage had to be added by hand to test the right event.

### Arm 1 — frequency

Trip the eight most heavily loaded lines, one at a time, permanently.

| test | worst frequency deviation |
|---|---:|
| **line trip (the N-1 event this thesis models)** | **0.3293 Hz** |
| generator trip (control — this *should* move frequency) | 0.3153 Hz |
| islanding (ANDES's shipped separation case) | 0.1251 Hz |
| **mildest threshold in the corpus (59.4 / 60.6 Hz)** | **needs 0.6000 Hz** |

Not one contingency of any kind reached the easiest of the 457 thresholds.

To make the finding quantitative rather than merely negative, an escalation ladder was run to find
what it *would* take:

| event | deviation | reaches 0.6 Hz? |
|---|---:|---|
| 1 generator out (N-1) | 0.3153 Hz | no |
| **2 generators out (N-2)** | **0.7975 Hz** | **yes** |
| 3 generators out (N-3) | 1.8918 Hz | yes |
| 1 load out (N-1) | 0.2208 Hz | no |
| 3 loads out (N-3) | 1.6229 Hz | yes |

### Arm 2 — voltage ride-through

Two arms, because the first alone would have been unfair to the rules.

**Arm A, clean line opening** — the event class this thesis models. Deepest voltage dip anywhere:
**0.9844 pu** (0.9557 rebased), against a mildest rule threshold of 0.95 pu. The envelopes are
never entered, so the duration half of each rule is never even engaged.

**Arm B, bolted three-phase fault** — the event class the rules are actually *written* for, at 7
buses with two clearing times. Here 110 distinct envelopes *are* entered. But voltage collapses to
~0.0003 pu while the fault is on the wires and is back above 0.9 pu **within about 50 ms of
clearing**. The longest post-clearing excursion below the mildest bar is 0.080 s.

---

## 3. Why it fails, in one paragraph

**Frequency moves when generation and demand fall out of balance.** Losing a transmission line
destroys neither — the same generators are still running and the same loads are still drawing; the
electricity simply takes a different route. The balance is untouched, so frequency barely twitches.
And this thesis is *about* losing one line. The 457 frequency rules are aimed at a different
failure mode than the one being screened for: they need two or three generators to fail at once,
which is not an N-1 contingency at all.

**Voltage ride-through envelopes describe a moment, not a state.** They say what a generator must
survive *while a short circuit is still on the wires* — a window measured in tens of milliseconds.
The task here is the settled condition *after* a line is lost. The grid passes through the
ride-through region in milliseconds and never sits in it. Right physics, wrong instant.

---

## 4. What it would have cost, and what it would have bought

| | |
|---|---:|
| candidates blocked because they name frequency or sub-second time | **1,252** |
| rules that would actually have **fired** | **3** |
| engineering time to build the dynamic pipeline | **2–4 weeks** |

1,252 → 3 is the entire finding. The gap is not a modelling subtlety; it is the difference between
asking *"can this rule be computed?"* and asking *"will this rule ever be true?"*

---

## 5. Two errors made along the way, recorded deliberately

**The first "evaluable" figure was mine, produced by hand, and wrong twice over.** It was originally
quoted as 726, checking *evaluability* — can the simulator supply the variables this rule names —
without checking *firing*. Those are different questions, and the gap between them is large either
way. But the 726 itself was also never backed by a script and did not survive a later check:
`evaluation/capability_gap.py` scans every untranslatable candidate's stated reason directly and
gets **1,252**, which reconciles exactly with the corpus's own frequency/time partition
(`thesis_findings.md` §19.1). The gap between "evaluable" and "fires" is therefore **1,249 rules**,
not 723 — found by measuring, twice, not by thinking harder.

**The frequency margin estimate was also loose.** The prediction before running was that line trips
would move frequency by "hundredths of a Hz." The measured worst case was 0.3293 Hz. The conclusion
survived — 0.33 against a 0.6 bar is still a clear miss — but the margin is about 1.8×, not the 20×
implied. Quote the measured number, never the estimate.

**One near miss worth stating honestly.** In Arm A, the deepest clean-opening voltage dip clears the
mildest threshold by only ~0.006 pu once rebased — under 1%. The frequency arm missed by a factor
of two; this one missed by a hair. A heavier loading condition or a deeper contingency could
plausibly cross it. The voltage conclusion is therefore *narrower* than the frequency one and should
be stated as such.

---

## 6. What to say if asked

> *"Why didn't you use a dynamic simulator to unlock the frequency rules?"*

Because it was tested and it does not pay. No N-1 contingency of any kind — line **or** generator —
reaches the mildest of the 457 frequency thresholds; firing them requires an N-2 event, which is a
different task. Of 1,252 candidates blocked because they name frequency or sub-second time, 3 would
have fired, at a cost of two to four weeks. The result tables are in
`results/audit/andes_frequency_spike.json` and `results/audit/andes_voltage_spike.json`, and both
probes re-run in minutes; the 1,252 figure is in `results/audit/capability_gap.json` and also
re-runs in seconds.

The value of this is not the negative result itself. It is that the answer is a **measurement
rather than an opinion** — and that it was obtained *before* the two-to-four weeks were spent
rather than after.

---

## 7. Reproduce

```bash
.venv\Scripts\python.exe sanity\andes_frequency_spike.py
.venv\Scripts\python.exe sanity\andes_voltage_spike.py
.venv\Scripts\python.exe evaluation\capability_gap.py
```

Artifacts: `results/audit/andes_frequency_spike.json`, `results/audit/andes_voltage_spike.json`,
`results/audit/capability_gap.json` (the 1,252 figure — see §1, §4, §5 above).
Cross-references: `thesis_findings.md` §20.2 (the same account in situ), §19.1 (where the frequency
and time/dynamics rules sit in the full corpus accounting), §5 (the standards/simulator mismatch
this is the sharpest instance of).
