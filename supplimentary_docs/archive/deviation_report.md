# Detailed Update: Research Direction and Implementation Change

---

## 1. Current State of the Project

We have successfully completed the core symbolic components of the Neuro-Symbolic architecture, but we have encountered a roadblock with the neural perception layer that dictates our immediate next steps.

**Completed Milestones:**
1. **Component B (LLM Extraction):** We have successfully run our local multi-LLM pipeline (Qwen3-14B as Extractor, Nemotron-3 Nano 30B as Validator). It has accurately extracted symbolic rules from dense power grid documentation (e.g., IEEE standards).
2. **Component C (Knowledge Graph):** These extracted rules have been populated into a NetworkX Knowledge Graph. The graph successfully maps grid entities (buses, lines) to their governing physical constraints. The "Symbolic Validation Shield" logic is fully capable of parsing grid states and enforcing these rules.

**Immediate Roadblock:**
- **Component A (GNN Baseline):** We have built the initial Graph Neural Network (GNN) using PyTorch Geometric. However, the model is currently **severely underperforming** even on its home topology (the 36-bus system). 
- **Consequence:** We cannot test the neuro-symbolic integration or evaluate any advanced hypotheses until the GNN baseline is competent. Fixing the GNN is our absolute top priority.

---

## 2. Refined Research Direction: The Unified Thesis

Once the GNN is stabilized, we propose shifting the focus of our formal evaluation. Instead of evaluating the system on a single grid topology, we will focus on **Cross-Topology Generalization (Transfer Learning)**. 

We are adopting a unified thesis that combines two distinct research angles:

### Angle 1: The Generalization Thesis (The "Where")
Neural networks are notorious for overfitting to the structural topology of the graphs they are trained on. If you train a GNN on a 36-bus grid, it typically fails when deployed on a 118-bus grid because the node degrees and spatial relationships change.
However, **physical rules do not care about topology.** The IEEE standard that a voltage must not drop below 0.95 pu applies perfectly to both a 14-bus grid and a 118-bus grid. 

We will train the GNN on a 36-bus system, then test the integrated GNN+Shield system on unseen 14, 57, and 118-bus systems. We expect the GNN's accuracy to degrade, but the Symbolic Shield should maintain a near-perfect safety guarantee because the Knowledge Graph rules are universally applicable.

### Angle 2: The Epistemology Thesis (The "Why")
Currently, the shield is designed as a safety tool—it blocks unsafe actions. We will repurpose it to also act as a **diagnostic explainability tool**. 
When the GNN makes an unsafe prediction on a new topology and the shield blocks it, we will ask: *What exactly went wrong inside the neural network?*
By collecting every blocked prediction, we can cluster the GNN's failure modes:
- **Overconfident Wrong Predictions:** The GNN predicts "normal" with high confidence during a physical fault.
- **Class Confusion:** The GNN detects a problem but confuses a simple "overload" with a "cascading failure."
- **Localization Failure:** The GNN correctly identifies the fault type, but points to the wrong bus/line.
- **Novel Topology State:** The GNN fails specifically on sub-graphs it has never seen before.

### The Open-Ended Thesis Question
> *"Is the symbolic shield necessary for transfer learning, or does the GNN intrinsically generalize to unseen topologies?"*

Keeping this question open-ended guarantees a successful, publishable thesis regardless of the experimental outcome:
- **Outcome A (Expected - The Shield is Necessary):** We empirically prove that neural models cannot be trusted with topological changes, and our decoupled symbolic layer is a mandatory safety requirement for real-world utility deployment.
- **Outcome B (Surprise - The GNN Generalizes):** If the GNN performs surprisingly well on the 118-bus system and the shield rarely has to intervene, we will have discovered that GNNs can learn underlying physical power-flow dynamics that generalize structurally. This allows us to map the exact empirical boundaries of where neural perception succeeds in physics.

---

## 3. Coding and Implementation Changes

To support fixing the GNN and executing this new research direction, the coding roadmap requires the following concrete changes:

### Phase 1: Fixing the GNN (Immediate Priority)
We are pausing integration testing to debug the PyTorch Geometric training pipeline. We suspect the underperformance stems from a few specific areas that we will investigate:
- **Feature Normalization:** We need to ensure that the node and edge normalization statistics (`node_mean`, `node_std`) calculated during training are being saved and applied identically during validation/inference. A mismatch here causes catastrophic failure.
- **Class Imbalance:** Normal grid states vastly outnumber fault states in the Grid2Op chronics. We will audit the `CrossEntropyLoss` weights and our stratified sampling logic to ensure the GNN isn't collapsing into always predicting "normal."
- **Message-Passing Depth:** We will tune the GAT/GCN layers (e.g., number of heads, dropout, number of layers) to ensure the receptive field is large enough to capture cascading effects.

### Phase 2: Expanding the Data and Integration Harness
We do not need to write new data generation logic, as Grid2Op natively supports multiple topologies.
- **Multi-Environment Test Loop:** We will heavily expand the `study3(integration).md` evaluation script. Currently, it evaluates the GNN on a held-out test set from the 36-bus environment. We will write parallel evaluation loops that load the `l2rpn_case14_sandbox` (14-bus), `rte_case5_example` (5-bus), and `l2rpn_wcci_2022` (118-bus) environments. The script will feed observations from these unseen grids directly into the 36-bus-trained GNN and the existing Shield.

### Phase 3: Diagnostic Logging Framework
To answer the "Epistemology Thesis," we need rich data about *why* the GNN failed. We will upgrade the `validate()` function within the Symbolic Shield:
- Currently, the shield just outputs `PASS` or `BLOCK` with a simple text explanation.
- **Coding Change:** We will modify the shield to export a detailed JSON log for every single `BLOCK` event. This log will include:
  - The GNN's raw `class_logits` and predicted confidence score.
  - The raw, un-normalized observation vector (voltages, line loads).
  - The specific Rule ID that was violated.
  - The evaluated condition string that triggered the block.
- This JSON dataset will be parsed by a separate analysis script to generate the failure clusters (overconfidence, class confusion, etc.) needed for the final thesis report.
