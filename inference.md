# 🛡️ Inference & Validation Lifecycle

This document describes the end-to-end execution flow of the Neuro-Symbolic Shield. It details how neural predictions from the GNN are intercepted, validated against the Knowledge Graph, and either approved or blocked.

## 🔄 The "Shield" Workflow

1.  **Neural Perception (GNN):** The GNN analyzes the current grid state (voltages, loads, topology) and predicts a fault type or recommends a control action.
2.  **Symbolic Interception:** The Shield intercepts the GNN output before it is executed on the grid.
3.  **KG Querying:** The Shield queries the Knowledge Graph for rules associated with the specific entities (Buses/Lines) and the overall system (Grid).
4.  **Constraint Checking:** The Shield evaluates the GNN's recommendation against the logical conditions of the retrieved rules.
5.  **Final Decision:** If all rules are satisfied, the action is **APPROVED**. If any rule is violated, the action is **BLOCKED** and an explanation is generated.

---

## 🎭 Scenario: The "Dangerous Shortcut"

This scenario demonstrates the system preventing a "greedy" neural decision that stabilizes a local issue but creates a system-wide safety violation.

### 1. Neural Perception (The GNN)
*   **Grid State:** Peak evening load. `Line_12` is at **98% capacity** (extreme overload risk).
*   **Recommendation:** The GNN predicts that disconnecting `Line_5` will force power to redistribute, dropping `Line_12` to a safe 80% load.
*   **GNN Output:** `ACTION: DISCONNECT Line_5`
*   **Confidence:** 94%

### 2. Symbolic Interception
The Validation Shield holds the action and queries the Knowledge Graph.

#### A. Physical Context Query
*   **Node:** `Line_5`
*   **Linked Rule:** `Rule R_614` (Transmission Relay Loadability)
*   **Condition:** `current_pu <= 1.15`
*   **Action:** `BLOCK`
*   **Logic:** A line carrying more than 115% of its rated current is considered a "critical bridge" and cannot be manually tripped without risking a cascade.

#### B. System Context Query
*   **Node:** `Grid`
*   **Linked Rule:** `Rule R_008` (Frequency Response)
*   **Condition:** `frequency_hz >= 49`
*   **Action:** `REDISPATCH` (Instead of Trip)
*   **Logic:** If system frequency is below 49Hz, the grid is unstable; tripping major lines is prohibited to prevent a frequency collapse.

### 3. Simulated Validation
The Shield checks the real-time telemetry against these rules:
*   **Measurement 1:** `Line_5` current is `1.18 pu`. (**Violation of R_614**)
*   **Measurement 2:** Grid frequency is `48.9 Hz`. (**Violation of R_008**)

### 4. The Final Decision

| Component | Status | Reasoning |
| :--- | :--- | :--- |
| **GNN (Neural)** | ✅ **ALLOW** | Local optimization: "Disconnecting Line_5 fixes Line_12." |
| **Shield (Symbolic)** | ❌ **BLOCK** | Global Safety: "Action violates R_614 (Overload) and R_008 (Low Freq)." |

**System Outcome:** 
The action is rejected. The operator receives an alert: 
> `[SHIELD REJECTION]: GNN action 'DISCONNECT Line_5' blocked by safety rules R_614 and R_008. Reason: System frequency too low for line tripping; Line_5 is a critical bridge.`

---

## 💡 Key Architectural Takeaway
The GNN is excellent at **pattern matching** (finding shortcuts), while the Knowledge Graph is the **authority on safety** (enforcing laws). By combining them, we get a system that is both intelligent and guaranteed to be safe.
