> ### ⚠️ STATUS 2026-08-20 — PARTLY SUPERSEDED. Read `study.md` §5 for the shield as built.
>
> The integration phase described here happened, but three specifics are now wrong and one is
> dangerous to copy:
>
> - 🚨 **§4.5's flat `voltage_pu = v_or / 150.0` conversion is WRONG and will produce ~100% false
>   blocks.** case14 runs lines at ~20 kV *and* ~138 kV; even the 36-bus grid has 7 lines at
>   ~365 kV. The shield uses **per-line base kV** from `data/grid_dataset_<tag>_basekv.json` with
>   energized-line masking. These grids also operate ~6% above nominal, so a healthy frame reads
>   ~1.06 pu, not ~1.00.
> - **The shield does not query the knowledge graph by entity.** Rule retrieval sits behind a
>   `RuleProvider` protocol; the default serves a flat JSONL corpus and the KG is an opt-in
>   provider that returns an identical set. Entity routing is actively forbidden and
>   test-guarded — identical rules carry different `entity` labels depending on their source
>   standard (`study.md` §4).
> - **The gate is asymmetric and binary**, not a PASS/CORRECT/BLOCK selector over top-k
>   candidates. It escalates an over-permissive `secure` verdict and never blocks a cautious one
>   (`study.md` §5).
>
> What remains accurate and useful: the phase framing, the decoupling invariant, and the
> interaction-contract discussion in §4.7.

---

## Phase 4: Neuro-Symbolic Integration — The Shield Layer

This is the phase where the trained GNN and the populated Knowledge Graph are wired together into a single inference pipeline. Every grid state observation flows through the GNN, produces a prediction, then hits the shield before anything is acted on. Nothing bypasses the shield.

---

### 4.1 What you have coming in

**From the GNN (trained):**
- A `GridGNN` model checkpoint (`gnn_checkpoint_best.pt`)
- For each grid state graph, it outputs:
  - `class_logits` — shape `(1, n_classes)` → softmax → predicted fault type (normal / overload / line_trip / cascade)
  - `loc_logits` — shape `(n_nodes,)` → sigmoid → per-bus fault probability
- The classification head uses `global_max_pool` so the fault signal from one stressed bus isn't diluted across all 36

**From the KG (populated from `all_rules_deduped.jsonl`):**
- A NetworkX `DiGraph` with two kinds of nodes: grid entities (Bus, Line, Generator, etc.) and Rule nodes
- Entity → Rule edges via `has_rule` relation
- Each Rule node carries: `condition` (a parseable logical expression), `action`, `severity`, `source`, `explanation`

---

### 4.2 The observation-to-graph bridge

At inference time, you receive a live Grid2Op `obs` object (or a record from the test set). Before the GNN can run, the observation must be converted to a PyG `Data` object using the exact same `build_node_features()` and `build_edges()` functions used during training — including the same normalization stats (node_mean, node_std, edge_mean, edge_std) computed from the training split. If normalization is applied differently here than during training, the GNN produces garbage. The normalization stats must be saved after training and loaded at inference time.

```python
# Save after training
torch.save({
    "node_mean": node_mean, "node_std": node_std,
    "edge_mean": edge_mean, "edge_std": edge_std,
}, "normalization_stats.pt")

# At inference
norm = torch.load("normalization_stats.pt")
graph = obs_to_pyg(obs, meta)  # builds raw Data object
graph.x = (graph.x - norm["node_mean"]) / norm["node_std"]
graph.edge_attr = (graph.edge_attr - norm["edge_mean"]) / norm["edge_std"]
```

The `obs_to_pyg` function is essentially `build_node_features` + `build_edges` wrapped to accept a live Grid2Op obs instead of a dict record. The feature vector is identical: 5 node features (load_p, gen_p, mean_v, max_rho, connected_line_frac), 4 edge features (rho, p_or, q_or, line_status).

---

### 4.3 GNN inference

```python
model.eval()
with torch.no_grad():
    class_logits, loc_logits = model(
        graph.x, graph.edge_index, graph.edge_attr,
        torch.zeros(graph.x.size(0), dtype=torch.long)  # batch = single graph
    )

probs = F.softmax(class_logits, dim=1).squeeze()       # (n_classes,)
predicted_class = probs.argmax().item()                 # int
predicted_label = idx_to_label[predicted_class]         # "overload", "cascade", etc.
confidence = probs[predicted_class].item()              # float

# Localization: top-k buses most likely to be the fault source
loc_probs = torch.sigmoid(loc_logits).squeeze()         # (36,)
top_k_buses = loc_probs.topk(3).indices.tolist()        # [bus_id, ...]
```

The predicted label and top-k fault buses are passed to the shield along with the raw observation context (rho, voltages, line_status).

---

### 4.4 Prediction context construction

The shield doesn't receive the PyG graph. It receives a flat context dict that mirrors what the rules were written against — the physical quantities directly. This is intentional: the rules from the IEEE standards talk about voltage, loading percentage, and line status, not graph embeddings.

```python
context = {
    "fault_type":        predicted_label,           # GNN output
    "confidence":        confidence,
    "predicted_buses":   top_k_buses,               # from loc head
    "rho":               obs.rho.tolist(),           # (59,) raw, not normalized
    "v_or":              obs.v_or.tolist(),          # (59,) in kV
    "line_status":       obs.line_status.tolist(),   # (59,) bool
    "load_p":            obs.load_p.tolist(),
    "gen_p":             obs.gen_p.tolist(),
    "rho_max":           float(obs.rho.max()),
    "rho_max_line":      int(obs.rho.argmax()),
    "n_tripped_lines":   int((~obs.line_status).sum()),
}
```

The raw (un-normalized) observation values are used here because rules have physical thresholds: `voltage_pu > 1.05`, `loading_pct > 100`, etc. Using normalized values would break rule evaluation.

---

### 4.5 The Shield: rule retrieval and evaluation

The shield does two things: retrieves all rules applicable to the current prediction, then evaluates each rule's condition against the context.

**Rule retrieval from KG:**

```python
def get_applicable_rules(KG, fault_type, context):
    applicable = []
    for node, attrs in KG.nodes(data=True):
        if attrs.get("type") != "Rule":
            continue
        # Rules that apply to this fault type OR are global (entity="Grid")
        entity = attrs.get("entity", "").lower()
        if entity == "grid" or entity == fault_type or entity in ["line", "bus"]:
            applicable.append(attrs)
    return applicable
```

In practice with NetworkX you'd traverse `has_rule` edges from the predicted entity. With the NeurIPS 2020 environment (36 buses, 59 lines), the full rule set is small enough that iterating all Rule nodes per step is fast enough — no Cypher query optimization needed at this scale.

**Condition evaluation:**

The condition strings from extracted rules look like: `"voltage_pu > 1.05 OR voltage_pu < 0.95"`, `"loading_pct > 100"`, `"rho > 1.0 AND line_status == False"`.

These are evaluated by building a local namespace from the context and using Python's `eval()` with a strict whitelist:

```python
import operator, re

ALLOWED_NAMES = {
    "voltage_pu", "loading_pct", "rho", "rho_max",
    "n_tripped_lines", "line_status", "AND", "OR", "NOT",
    "True", "False", "and", "or", "not",
}

def evaluate_condition(condition: str, context: dict) -> bool:
    # Map rule variable names to actual context values
    namespace = {
        "voltage_pu":      min(context["v_or"]) / 150.0,   # kV → pu (150kV nominal)
        "loading_pct":     context["rho_max"] * 100,
        "rho":             context["rho_max"],
        "rho_max":         context["rho_max"],
        "n_tripped_lines": context["n_tripped_lines"],
        "line_status":     not all(context["line_status"]),
    }
    try:
        return bool(eval(condition, {"__builtins__": {}}, namespace))
    except Exception:
        return False  # malformed condition → don't block
```

The voltage_pu mapping is the key thing to get right. Grid2Op gives voltage in kV. The IEEE rules use per-unit. For the NeurIPS 2020 environment, nominal is ~150kV, so dividing by 150 gives approximate pu. If your meta JSON stores nominal voltages per substation, use those instead.

---

### 4.6 Shield decision logic

```python
def validate(prediction_context: dict, KG) -> dict:
    fault_type = prediction_context["fault_type"]
    applicable_rules = get_applicable_rules(KG, fault_type, prediction_context)
    
    violated = []
    for rule in applicable_rules:
        if evaluate_condition(rule["condition"], prediction_context):
            violated.append(rule)
    
    if not violated:
        return {
            "status":      "PASS",
            "fault_type":  fault_type,
            "confidence":  prediction_context["confidence"],
            "explanation": None,
        }
    
    # Sort violated by severity (critical > high > medium > low)
    SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    violated.sort(key=lambda r: SEVERITY_ORDER.get(r.get("severity", "low"), 3))
    
    explanation = build_explanation(violated)
    
    return {
        "status":         "BLOCK",
        "fault_type":     fault_type,
        "confidence":     prediction_context["confidence"],
        "violated_rules": violated,
        "explanation":    explanation,
        "highest_severity": violated[0]["severity"],
    }


def build_explanation(violated_rules: list) -> str:
    lines = []
    for rule in violated_rules:
        lines.append(
            f"Rule {rule['rule_id']} ({rule['severity'].upper()}): "
            f"{rule['explanation']} "
            f"[Source: {rule['source']}]"
        )
    return " | ".join(lines)
```

The PASS/BLOCK output is the final system output. A PASS means the GNN's prediction is consistent with all symbolic constraints and can be acted on. A BLOCK means the prediction — even if the GNN is confident — would violate at least one extracted rule. The explanation cites the exact rule ID and its source document.

---

### 4.7 How the two components actually interact

The interaction is strictly one-directional and post-hoc. The GNN has no knowledge of the rules during inference — it does not see the KG and its weights are not affected by the KG. The KG has no knowledge of GNN internals — it only sees the prediction output and the raw context. This separation is the core design property. It means:

- The GNN can be retrained, swapped, or upgraded without touching the shield
- Rules can be added or removed from the KG without retraining the GNN
- When the shield blocks a prediction, you know the GNN's neural representation was insufficient for that state — this is a diagnostic signal for the GNN, not a failure of the overall system

The information flow per timestep is:
```
obs → [PyG conversion + normalization] → GNN → (class_logits, loc_logits)
    → [argmax + top-k] → (predicted_label, top_buses, confidence)
    → [raw obs context] → Shield → validate()
    → PASS (forward prediction) or BLOCK (generate explanation)
```

---

### 4.8 KG construction from confirmed rules

The `all_rules_deduped.jsonl` from the extraction pipeline is loaded into NetworkX:

```python
import networkx as nx
import json

def build_knowledge_graph(rules_path: str, meta) -> nx.DiGraph:
    KG = nx.DiGraph()
    
    # Add grid topology nodes from meta
    for bus_id in range(meta.n_sub):
        KG.add_node(f"Bus_{bus_id}", type="Bus", bus_id=bus_id)
    
    for line_id in range(meta.n_line):
        or_bus = meta.line_or_bus[line_id]
        ex_bus = meta.line_ex_bus[line_id]
        KG.add_node(f"Line_{line_id}", type="Line", line_id=line_id)
        KG.add_edge(f"Bus_{or_bus}", f"Line_{line_id}", type="connected_to")
        KG.add_edge(f"Bus_{ex_bus}", f"Line_{line_id}", type="connected_to")
    
    # Add extracted rules
    with open(rules_path) as f:
        for line in f:
            rule = json.loads(line)
            rule_node_id = rule["rule_id"]
            KG.add_node(rule_node_id, type="Rule", **rule)
            
            # Connect rule to the entity type it governs
            entity = rule["entity"].lower()
            if entity == "bus":
                for bus_id in range(meta.n_sub):
                    KG.add_edge(f"Bus_{bus_id}", rule_node_id, type="has_rule")
            elif entity == "line":
                for line_id in range(meta.n_line):
                    KG.add_edge(f"Line_{line_id}", rule_node_id, type="has_rule")
            else:  # grid-level rule
                KG.add_node("Grid", type="Grid")
                KG.add_edge("Grid", rule_node_id, type="has_rule")
    
    nx.write_gpickle(KG, "knowledge_graph.gpickle")
    return KG
```

---

### 4.9 Evaluation metrics for this phase

Three things need to be measured for the integrated system, separate from GNN-only metrics:

**Rule compliance rate** — of all PASS decisions, what percentage actually satisfy all rules? This is checked post-hoc by running the rule evaluator over the passed predictions. It should be 100% by construction, but numerical edge cases in condition parsing can break this.

**False block rate** — of ground-truth normal/valid states in the test set, what percentage does the shield incorrectly block? This tells you if the extracted rules are over-constraining. A high false block rate means rules are either too broad (covering normal operating ranges) or the condition thresholds were misextracted.

**Safe action rate vs GNN-only** — compare the rate of rule-violating outputs: GNN alone vs GNN + shield. The delta is the safety contribution of the symbolic layer. Younesi et al. (2026) is your comparison target at 91.7% safe restoration with manually written rules.

**Explanation quality** — for blocked predictions, the explanation cites `rule_id` and `source`. Manual human rating (1–5 scale by team members) of whether the explanation is correct, specific, and actionable. This is the explainability claim of the thesis.

---

### 4.10 End-to-end test harness

The three scenarios from `study.md` Section 8 are the integration smoke test: clean pass, single overload block, cascade risk multi-rule block. These run against `rte_case5_example` (5-bus) using the trained GNN checkpoint and the populated KG. All three must pass before moving to formal evaluation on held-out chronics from `l2rpn_neurips_2020_track1_small`.

The formal evaluation runs the full pipeline over test-set chronics: each step produces an obs, the obs goes through the GNN, the prediction hits the shield, and the output (PASS or BLOCK + explanation) is logged. Metrics are aggregated over all steps.