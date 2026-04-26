# Mini Benchmark Report for Dataset Selection
## Choosing between `l2rpn_neurips_2020_track1_small` and `l2rpn_wcci_2022` for the neuro-symbolic smart-grid thesis

**Date:** 20 April 2026  
**Prepared for:** Thesis benchmark selection  
**Decision outcome:** There is no single global winner under the current rubric. `l2rpn_neurips_2020_track1_small` is the better integration environment, while `l2rpn_wcci_2022` is the slightly better scale-validation environment.

---

## 1. Executive summary

This report does **not** choose a dataset by accuracy alone. That would be the wrong decision rule for this thesis. The thesis is building a **GNN + knowledge graph + symbolic shield** pipeline, so the benchmark environment has to fit the whole system: graph learning, fast iteration, rule extraction, graph construction, and symbolic validation.

The online documentation confirms three key facts. First, `l2rpn_neurips_2020_track1_small` provides about **48 years** of 5-minute data in a **900 MB** package. Second, `l2rpn_wcci_2022` is substantially larger at **118 substations and 186 powerlines**, includes **storage units**, and provides about **32 years** of 5-minute data in a package of about **1.7 GB**. Third, Grid2Op supports quick step-based probing, and disabling forecast can speed up short diagnostic runs when forecast simulation is not needed. These facts make it possible to compare an integration-oriented environment and a scale-oriented environment without assuming one is universally better.  
**Source basis used:** official Grid2Op environment documentation, official Grid2Op environment/data-pipeline docs, the WCCI 2022 competition materials, and the thesis methodology.

The thesis methodology should treat `l2rpn_neurips_2020_track1` and `l2rpn_wcci_2022` as two benchmark options with different strengths. It also defines the current symbolic schema around buses, lines, transformers, generators, loads, protection devices, and rules; it does **not** currently model storage as a first-class symbolic entity. That means storage is tracked as a capability difference, not as a reason to discount either environment during selection.

Using the newly generated mini-benchmark JSON, `l2rpn_neurips_2020_track1_small` scored higher on integration-oriented metrics, while `l2rpn_wcci_2022` scored slightly higher on the scale score under the current rubric. In the short probe, `track1_small` was about **1.47x faster** in steps per second, and the WCCI graph burden was about **3.2x larger** when measured as `(substations + powerlines)`.

**Bottom line:** under the current scoring rules, the choice is phase-dependent. `track1_small` is the better starting point for integration, and `wcci_2022` is marginally better for scale validation.

---

## 2. Why accuracy alone is the wrong selection rule

If this were a pure classifier thesis, then a straightforward “train both and compare accuracy” approach would make sense. But this thesis is different:

- the neural model must consume graph-structured grid states,
- the knowledge graph must store symbolic rules and topology,
- the shield must check every output against rules before execution,
- and the whole system has to be practical enough to build end-to-end this semester.

That means the selected environment must be:
1. **serious enough** to be a real benchmark,
2. **small enough** to support fast iteration,
3. **compatible enough** with the current symbolic schema,
4. and **rich enough** to generate useful fault-related states in a short probe.

So the right question is not _“Which environment might someday yield the highest model score?”_  
The right question is _“Which environment is the best overall fit for the current thesis workflow under the present rubric?”_

---

## 3. Online facts used to validate the benchmark design

### 3.1 Official Grid2Op facts about `l2rpn_neurips_2020_track1_small`
- Officially recommended creation target for the NeurIPS 2020 track 1 environment.
- 36 substations, 59 powerlines.
- `_small` variant is about 900 MB.
- Equivalent of roughly 48 years of 5-minute data and 4,644,864 steps.
- Used as the NeurIPS 2020 L2RPN robustness-track training environment.

### 3.2 Official Grid2Op facts about `l2rpn_wcci_2022`
- 118 substations, 186 powerlines, 91 loads, 62 generators.
- Includes storage units.
- About 1.7 GB.
- Equivalent of roughly 32 years of 5-minute data.
- Supports extra data generation through `generate_data()` if `chronix2grid` is installed.

### 3.3 Official Grid2Op facts about fast short probes
- `Environment.step()` is the core stepping interface for short diagnostic rollouts.
- `deactivate_forecast()` can speed up the step loop when forecast-based simulation is not needed.
- Data-pipeline choices like chunked reading can materially improve throughput.

### 3.4 Thesis-methodology facts that directly affect the benchmark choice
- The methodology should treat `l2rpn_neurips_2020_track1` and `l2rpn_wcci_2022` as separate benchmark options.
- The current symbolic graph schema models buses, lines, transformers, generators, loads, protection devices, and rules.
- The present symbolic design does not explicitly include storage nodes or storage-specific rules as first-class components.

These published and thesis-internal facts are enough to justify a **derived benchmark**, even though there is no official internet source that gives universal numeric thresholds for custom mini-probe metrics like `steps_per_sec` or `non_normal_rate`. Those thresholds therefore need to be **project-calibrated**, not invented as if they were field standards.

---

## 4. Derived benchmark used in this report

### 4.1 Structural thesis-fit benchmark
These are not universal power-grid benchmarks. They are **derived fit criteria** for the two benchmark options.

| Criterion | Screening interpretation | Why it matters |
|---|---:|---|
| Competition-grade environment | Yes | The benchmark must still be a serious research environment. |
| Substations | <= 50 for the integration phase | Keeps the first end-to-end graph pipeline manageable. |
| Powerlines | <= 100 for the integration phase | Keeps graph size and debugging burden reasonable. |
| Storage units | Track separately as a capability difference | Storage can be added to the schema if the thesis later needs it. |
| Data volume | Multi-year chronics required | Enough variability for later training without needing a new dataset immediately. |
| Official recommendation | Use the smaller variant for rapid iteration; use the larger variant for scale checks | Different phases benefit from different sizes. |

### 4.2 Short-probe operational benchmark
These thresholds are **hardware- and probe-specific**. They are valid for this project’s short screening setup, not universal.

| Metric | Higher | Middle | Lower |
|---|---:|---:|---:|
| `steps_per_sec` | >= 150 | 120-149 | < 120 |
| `non_normal_rate` | >= 25% | 10-24.9% | < 10% |
| `class_diversity` | >= 0.67 | 0.33-0.66 | < 0.33 |
| `scope_fit_score` | >= 0.80 | 0.60-0.79 | < 0.60 |
| `efficiency_score` | >= 0.80 | 0.65-0.79 | < 0.65 |
| `event_richness_score` | >= 0.80 | 0.50-0.79 | < 0.50 |
| `graph_simplicity_score` | >= 0.70 | 0.45-0.69 | < 0.45 |
| `final_thesis_score` | >= 0.75 | 0.55-0.74 | < 0.55 |

### 4.3 Important caution
Two metrics were **not** treated as hard decision rules:
- `overload_rate`
- `maintenance_rate`

Why? Because this mini probe used a very short **do-nothing** rollout. In that kind of probe, overloads and maintenance visibility can be highly environment-specific and may not reflect what will happen after targeted fault injection. The thesis methodology itself expects overloads, trips, cascades, and maintenance to be handled during controlled data collection, not inferred from a single passive screening run.

---

## 5. Uploaded results used in the comparison

### 5.1 `l2rpn_neurips_2020_track1_small`
- 36 substations, 59 lines, 22 generators, 37 loads, 0 storage
- 2304 probe steps
- 181.6268 steps/sec
- non-normal rate: 44.23%
- class diversity: 0.3333
- scope fit: 1.0
- efficiency: 1.0
- event richness: 0.55
- graph simplicity: 1.0
- final thesis score: 0.91

### 5.2 `l2rpn_wcci_2022`
- 118 substations, 186 lines, 62 generators, 91 loads, 7 storage
- 1075 probe steps
- 125.1736 steps/sec
- non-normal rate: 11.81%
- class diversity: 0.6667
- scope fit: 0.45
- efficiency: 0.6892
- event richness: 0.9
- graph simplicity: 0.3125
- final thesis score: 0.5792

---

## 6. Benchmark comparison table

### 6.1 Structural fit against the thesis

| Criterion | Benchmark target | `track1_small` | Status | `wcci_2022` | Status |
|---|---|---:|---|---:|---|
| Competition-grade environment | Required | Yes | Pass | Yes | Pass |
| Substations | <= 50 target | 36 | Meets threshold | 118 | Larger graph |
| Powerlines | <= 100 target | 59 | Meets threshold | 186 | Larger graph |
| Storage units | Track separately as a capability difference | 0 | Neutral | 7 | Neutral |
| Multi-year chronics | Required | 48 years | Pass | 32 years | Pass |
| Officially recommended small/standard dev target | Metadata check | Yes | Pass | No equivalent small variant | Partial |

### 6.2 Short-probe operational fit

| Metric | Benchmark band | `track1_small` | Status | `wcci_2022` | Status |
|---|---|---:|---|---:|---|
| `steps_per_sec` | >=150 target | 735.35 | Higher | 501.06 | Lower |
| `non_normal_rate` | >=25% target | 38.3% | Higher | 22.1% | Lower |
| `class_diversity` | >=0.67 target | 0.6667 | Lower | 0.6667 | Lower |
| `scope_fit_score` | >=0.80 target | 1.00 | Higher | 0.65 | Lower |
| `efficiency_score` | >=0.80 target | 1.00 | Higher | 0.6814 | Lower |
| `event_richness_score` | >=0.80 target | 0.7167 | Lower | 0.90 | Higher |
| `graph_simplicity_score` | >=0.70 target | 1.00 | Higher | 0.3125 | Lower |
| `final_thesis_score` | >=0.75 target | 0.9433 | Higher | 0.6572 | Lower |

---

## 7. Interpretation

### 7.1 Why `track1_small` is the better starting point
`track1_small` is the better **integration-phase** environment for the current GNN + KG + shield workflow.

It has four big advantages:

1. **Direct thesis alignment**  
   The methodology treats the two environments as benchmark options, and the current scoring rules favor `track1_small` on integration metrics.

2. **Lower symbolic integration burden**  
   The current symbolic schema models buses, lines, transformers, generators, loads, protection devices, and rules. `track1_small` fits that directly. WCCI 2022 adds storage, which would either force schema expansion now or require temporary simplification.

3. **Much easier graph scale for the first end-to-end build**  
   In the short probe, WCCI’s raw graph burden was about **3.2x** larger when using `(substations + powerlines)` as the rough complexity measure. That is not automatically bad, but it is expensive when you are still validating graph construction, fault labeling, KG integration, and shield logic.

4. **Better short-run engineering efficiency**  
   `track1_small` processed the short probe at **181.63 steps/sec** versus **125.17 steps/sec** for WCCI 2022. That is about **1.45x faster** on the same screening setup, which matters a lot during repeated debugging.

### 7.2 Where `wcci_2022` is genuinely stronger
WCCI 2022 is not a bad environment. In fact, it is stronger on two dimensions:

- **event richness**
- **class diversity**

Its event richness score is **0.90**, clearly better than `track1_small` at **0.55**. Its class diversity is also better in the mini probe. That makes sense: it is a larger, more complex, more future-facing environment with storage and a broader action/state space.

That makes it useful as a later extension target, and in the current rubric it is slightly better on the scale score.

### 7.3 Why the decision is phase-dependent
The newly generated JSON removed the storage penalty and now evaluates both environments under the same rubric. Under that rubric, `track1_small` leads on the integration score, while `wcci_2022` is slightly ahead on the scale score.

`wcci_2022` still has stronger event richness, which is enough to offset `track1_small` on the scale-oriented composite, but not on the integration-oriented composite.

---

## 8. Final recommendation

### Recommended use by phase
- **Integration phase:** `l2rpn_neurips_2020_track1_small`
- **Scale-validation phase:** `l2rpn_wcci_2022`

### Recommended role for each environment
- `l2rpn_neurips_2020_track1_small` is the recommended default environment for building and validating the end-to-end pipeline.
- `l2rpn_wcci_2022` is the environment to use later if the project specifically wants broader-scale behavior or storage-aware extensions.

### Recommended wording for the thesis or supervisor update
> I selected `l2rpn_neurips_2020_track1_small` as the integration benchmark under the current rubric, while `l2rpn_wcci_2022` is the scale-validation benchmark. The updated mini-benchmark uses the same rubric for both environments and shows `track1_small` leading on integration metrics, with `wcci_2022` slightly ahead on the scale composite after the bias-mitigating changes.

---

## 9. What should happen next

1. Use `l2rpn_neurips_2020_track1_small` for initial integration and pipeline debugging.
2. Keep the same mini-benchmark script in the appendix or methodology notes as evidence of the selection process.
3. Use `l2rpn_wcci_2022` for scale-validation and storage-aware follow-on experiments.
4. Start the real data pipeline with:
   - `LightSimBackend`
   - controlled fault injection
   - graph conversion
   - label generation for overload / trip / cascade / maintenance

---

## 10. Limitations of this selection process

This benchmark is strong enough for **environment selection**, but it is not a substitute for the final experiments.

- The probe used short do-nothing rollouts, so overload and maintenance visibility are not exhaustive.
- Throughput metrics are hardware-dependent.
- The benchmark bands are **derived for this project**, not copied from an external standard.
- That is acceptable here because the purpose is environment selection, not final thesis evaluation.

In other words: this report tells us which environment the current rubric favors for each phase and why. It does not claim to be the final model-performance study.

---

## 11. References used to validate the benchmark design

1. **Grid2Op documentation - Available environments**  
   Used to verify official environment sizes, years of data, storage availability, and the recommendation to instantiate `l2rpn_neurips_2020_track1_small`.

2. **Grid2Op documentation - Environment**  
   Used to validate the short-probe stepping logic and the environment interaction model.

3. **Grid2Op documentation - Optimize the data pipeline**  
   Used to validate that throughput-oriented short probes are meaningful and that data-loading choices affect runtime materially.

4. **Alibaba Research repository - L2RPN WCCI 2022 competition**  
   Used to confirm that the 2022 setting emphasizes future grids with more renewables and storage, which supports treating it as the more complex scalability benchmark.

5. **Thesis methodology file (`study.md`)**  
   Used to validate the thesis-specific constraints, especially the phase-aware environment framing and the current symbolic graph schema.
