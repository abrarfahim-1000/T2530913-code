# Mathematical Formulas

This document compiles the mathematical formulas and formal definitions extracted from the codebase, organized by file. These representations are formatted for inclusion in a thesis report, detailing the data transformations, balancing techniques, and neural network mechanics used in the project.

> ### ⚠️ STATUS 2026-08-20 — audited against the current codebase. Read this map first.
>
> Roughly a third of what follows describes the **retired 4-class classification** pipeline. Those
> blocks are individually marked 🚫 **RETIRED** and kept only because the negative-results chapter
> needs them. Everything unmarked is current and safe to cite.
>
> | | formulas |
> |---|---|
> | ✅ **Still current** | chunking, rho clipping, bus load summation, graph pruning, mean bus voltage, max bus loading, connected-line fraction, GATv2 attention, multi-head concat, LeakyReLU / ReLU / ELU, live batch norm, early stopping, precision / recall |
> | 🚫 **Retired with the 4-class task** | the 4-class labelling function (§ generate_dataset), class-imbalance ratio, ICF weighting, global graph pooling, fault-localization targets, the multi-task loss, macro-F1 |
> | ⚠️ **Superseded — the formula changed** | node/edge feature vectors (4/4 → 8/8), voltage normalization (flat ÷150 → per-line base kV **for the shield**) |
> | ➕ **Added below** | the edge-level readout, masked BCE, per-line base-kV conversion, average precision, intervention precision, reach, structural ceiling, and predicate normalization |
>
> **Three referenced source files no longer exist:** `scripts/audit_datasets.py`,
> `scripts/split.py`, `training/evaluate.py`. Their formulas are marked accordingly — do not cite
> a file path that a reader cannot open.

## `extraction/extract.py`

⭐ **Text Chunking Strategy** (Used in Chapter 5)
To process large documents with local LLMs, the text is partitioned into overlapping segments of fixed size. The start index of the $i$-th chunk is defined as:
$$ \text{start}_i = i \cdot (S - O) \tag{1} $$
$$ \text{end}_i = \text{start}_i + S \tag{2} $$

*   **Notations**:
    *   $\text{start}_i$: The starting character index of the $i$-th chunk.
    *   $\text{end}_i$: The ending character index of the $i$-th chunk.
    *   $i$: The integer index of the current chunk (0, 1, 2, ...).
    *   $S$: The fixed chunk size in characters (defined by `CHUNK_SIZE`, e.g., 4000).
    *   $O$: The overlap size between consecutive chunks in characters (defined by `CHUNK_OVERLAP`, e.g., 500).
*   **Where it is used**: In `extraction/extract.py` within the `chunk_text` function during the rule extraction phase (Component B).
*   **How it works in this project**: Local LLMs (like Qwen3) have finite context windows. This strategy slices long IEEE manuals into manageable segments. The overlap ($O$) ensures that symbolic rules spanning across chunk boundaries are not truncated or lost, maintaining continuity for the LLM.

---

## `scripts/audit_datasets.py`

🚫 **RETIRED — Class Imbalance Ratio** *(4-class task; `scripts/audit_datasets.py` no longer exists)*
To quantify the severity of the class imbalance within the dataset, the ratio between the majority class and the minority class is calculated:
$$ \text{Imbalance Ratio} = \frac{\max_{c \in C} N_c}{\min_{c \in C} N_c + \epsilon} \tag{3} $$

*   **Notations**:
    *   $N_c$: The number of samples belonging to class $c$.
    *   $C$: The set of all classes (normal, overload, line_trip, cascade).
    *   $\epsilon$: A small constant ($10^{-9}$) to prevent division by zero for missing classes.
*   **Where it is used**: In `scripts/audit_datasets.py` within the `audit_dataset` function during data quality auditing.
*   **How it works in this project**: Power grid faults are naturally rare. This ratio identifies if the "normal" class excessively dominates the dataset. A high ratio (e.g., >10) triggers a recommendation to use weighted loss functions during training to prevent the GNN from being biased toward the majority class.

**Statistical Outlier Detection (Rho Percentiles)**
To characterize the distribution of line loadings across the dataset, the $p$-th percentile is calculated to identify typical and extreme stress states:
$$ \text{Percentile}(p) = \text{inf} \{ x \in \mathbb{R} : F(x) \geq p/100 \} \tag{4} $$

*   **Notations**:
    *   $p$: The target percentile (e.g., 95 or 99).
    *   $x$: A loading ratio ($\rho$) value.
    *   $F(x)$: The empirical cumulative distribution function (ECDF) of loading ratios in the dataset.
    *   $\text{inf}$: The infimum (minimum value) of the set.
*   **Where it is used**: In `scripts/audit_datasets.py` to analyze the distribution of the `rho` (loading) field.
*   **How it works in this project**: $\rho$ values above 1.0 indicate overloads. By calculating the 95th and 99th percentiles, the system determines the "tail" of the distribution, ensuring the dataset contains enough high-stress scenarios to effectively train the GNN's fault detection capabilities.

---

## `scripts/generate_dataset.py`

🚫 **RETIRED AS A TARGET — but keep it: this is the closed-form result** *(Chapter: negative results)*
The ground-truth label $y$ for each grid state is determined by a hierarchical physical rule set based on the loading ratio $\rho$ and the connectivity of the $N$ transmission lines:
$$ y = \begin{cases} \text{overload} & \text{if } \max(\rho) \ge 1.0 \\ \text{normal} & \text{else if } \sum_{i=1}^N s_i = N \\ \text{line\_trip} & \text{else if } \sum_{i=1}^N s_i = N - 1 \\ \text{cascade} & \text{else if } \sum_{i=1}^N s_i < N - 1 \end{cases} \tag{5} $$

*   **Notations**:
    *   $y$: The categorical state label.
    *   $\rho$: The vector of line loading ratios.
    *   $s_i$: The operational status of line $i$ (1 = connected, 0 = tripped).
    *   $N$: The total number of transmission lines in the grid (e.g., 59).
*   **Where it is used**: In `scripts/generate_dataset.py` within the `get_state_label` function.
*   **How it works in this project**: 🚫 **This target was abandoned, and equation (5) is the reason.** It is a *closed-form function of the observation the model is given* — every quantity on the right-hand side ($\rho$, $s_i$, $N$) is an input feature. Re-measured over every record of both datasets before they were deleted: **315,000 records, 100.000000% agreement, zero disagreements.** The trained GATv2 reached macro F1 0.8277 on the same task, so four threshold comparisons outperformed the network.
    The honest reading, which must accompany this in the report: it is **not** evidence that symbolic methods beat neural ones — it is evidence that the benchmark was **mis-specified**. The label carried no information the input did not already contain, so any measurement of "does the symbolic layer add value over the GNN?" was rigged before it ran. That is precisely why the task was replaced with N-1 screening, whose label requires a power-flow solve and therefore *cannot* be written as a function of the present observation.

⭐ **Transmission Line Loading Clipping** (Used in Chapter 5)
To prevent extreme values or measurement errors from destabilizing the neural network during data generation and feature extraction, the loading ratio $\rho$ is clipped to a predefined threshold $\rho_{\text{max}} = 2.0$:
$$ \rho' = \min(\rho, \rho_{\text{max}}) \tag{6} $$

*   **Notations**:
    *   $\rho$: The raw loading ratio from the simulation.
    *   $\rho'$: The clipped loading ratio used for model features.
    *   $\rho_{\text{max}}$: The saturation threshold (set to 2.0).
*   **Where it is used**: In `scripts/generate_dataset.py` during feature extraction.
*   **How it works in this project**: During grid failures, loading can spike to extreme values. Clipping prevents these outliers from causing gradient explosions in the GNN, while still preserving the "overloaded" signal ($1.0 \le \rho' \le 2.0$).

---

## `scripts/pyg_data.py`

⭐ **Bus Load Summation** (Used in Chapter 5)
The total active load $P_i$ at bus $i$ is the aggregate of all individual loads $p_l$ physically mapped to that substation:
$$ P_i = \sum_{l \in \mathcal{L}_i} p_l \tag{7} $$

*   **Notations**:
    *   $P_i$: The total active power load at bus $i$.
    *   $p_l$: The active power of an individual load $l$.
    *   $\mathcal{L}_i$: The set of indices of loads connected to bus $i$.
*   **Where it is used**: In `scripts/pyg_data.py` within `build_node_features`.
*   **How it works in this project**: Grid2Op provides loads at a granular level. To construct a graph where nodes represent substations, this formula aggregates all local consumer demands into a single node feature, representing the total power sink at that location.

⭐ **Dynamic Graph Pruning (Topology Masking)** (Used in Chapter 5)
To ensure the GNN only propagates messages across physically intact lines, the adjacency matrix $\mathbf{A}$ and edge attribute matrix $\mathbf{E}$ are dynamically pruned at each time step $t$ using the line status vector $\mathbf{s}_t$:
$$ \mathcal{E}_t = \{ (u, v)_k \in \mathcal{E}_{\text{static}} : s_{t,k} = 1 \} \tag{8} $$

*   **Notations**:
    *   $\mathcal{E}_t$: The set of active edges for the current message-passing step.
    *   $\mathcal{E}_{\text{static}}$: The set of all physical transmission lines in the grid.
    *   $(u, v)_k$: An edge representing line $k$ connecting bus $u$ and bus $v$.
    *   $s_{t,k}$: The status of line $k$ at time $t$ (1 if connected, 0 if disconnected).
*   **Where it is used**: In `scripts/pyg_data.py` within the `build_edges` function.
*   **How it works in this project**: This is a core topological feature. When a line trips in the simulation, its corresponding edge is physically removed from the graph. This forces the GNN to process the grid's state without that path, reflecting actual power flow constraints.

⭐ **Mean Bus Voltage** (Used in Chapter 5)
The mean voltage at each bus $i$ is calculated by averaging the voltage readings of all active lines connected to it:
$$ \bar{V}_i = \frac{1}{|E_i|} \sum_{e \in E_i} V_{\text{or}}^{(e)} \tag{9} $$

*   **Notations**:
    *   $\bar{V}_i$: The average voltage feature for node $i$.
    *   $E_i$: The set of transmission lines incident to bus $i$.
    *   $V_{\text{or}}^{(e)}$: The origin-side voltage of line $e$.
    *   $|E_i|$: The degree (number of lines) of bus $i$.
*   **Where it is used**: In `scripts/pyg_data.py` within `build_node_features`.
*   **How it works in this project**: Grid2Op defines voltages as a line property. To create a substation-level feature, the project averages incident line voltages to provide a localized grid health signal to each node.

⭐ **Max Bus Loading Ratio** (Used in Chapter 5)
To capture localized stress signals, the maximum loading ratio across all transmission lines connected to a substation is extracted for each bus:
$$ \rho_{\max, i} = \max_{e \in E_i} (\rho^{(e)}) \tag{10} $$

*   **Notations**:
    *   $\rho_{\max, i}$: The peak loading signal at bus $i$.
    *   $\rho^{(e)}$: The loading ratio of line $e$.
    *   $E_i$: The set of lines incident to bus $i$.
*   **Where it is used**: In `scripts/pyg_data.py` within `build_node_features`.
*   **How it works in this project**: Faults often start with a single line overload. This feature ensures that the substation node "perceives" the stress of its most critical connected line, aiding the GNN in pinpointing fault origins.

⚠️ **PARTLY SUPERSEDED — Voltage Normalization** *(still used for the GNN's `mean_v` node feature; NOT valid for the shield's per-unit conversion — see Current Formulas)*
To scale the node voltage features into a stable range for the neural network, the raw voltage is normalized by a constant factor (150.0 kV):
$$ V_{\text{norm}} = \frac{V_{\text{raw}}}{150.0} \tag{11} $$

*   **Notations**:
    *   $V_{\text{norm}}$: The normalized voltage feature (typically in range [0, 1.2]).
    *   $V_{\text{raw}}$: The raw voltage value in kV.
    *   150.0: The grid's nominal voltage level in kV.
*   **Where it is used**: In `scripts/pyg_data.py` within `build_node_features`.
*   **How it works in this project**: Z-scoring or simple division is necessary for GNN stability. Since voltages are typically around 150kV in the NeurIPS grid, this scaling brings the feature into a numerically stable range for gradient descent.

⭐ **Connected Line Fraction** (Used in Chapter 5)
A topological stability feature is computed for each bus, indicating the proportion of intact lines:
$$ f_{\text{conn}}^{(i)} = \frac{N_{\text{connected}}^{(i)}}{N_{\text{total}}^{(i)}} \tag{12} $$

*   **Notations**:
    *   $f_{\text{conn}}^{(i)}$: The connectivity ratio of bus $i$.
    *   $N_{\text{connected}}^{(i)}$: The count of connected lines at bus $i$ (status = 1).
    *   $N_{\text{total}}^{(i)}$: The total number of physical lines originally connected to bus $i$.
*   **Where it is used**: In `scripts/pyg_data.py` within `build_node_features`.
*   **How it works in this project**: This acts as a topological "health" indicator. A low fraction suggests that the bus is becoming isolated due to trips or cascades, providing a critical signal for classification.

⚠️ **SUPERSEDED — Node and Edge Feature Vectors** *(now 8 node / 8 edge features; see Current Formulas)*
The GNN input space is formally defined by the following feature compositions for each node $v$ and edge $e$:
$$ \mathbf{x}_v = [P_{\text{load}, v}, \bar{V}_v, \rho_{\max, v}, f_{\text{conn}, v}]^\top \in \mathbb{R}^4 \tag{13} $$
$$ \mathbf{e}_{uv} = [\rho_{uv}, P_{\text{or}, uv}, Q_{\text{or}, uv}, s_{uv}]^\top \in \mathbb{R}^4 \tag{14} $$

*   **Notations**:
    *   $\mathbf{x}_v$: The 4-dimensional feature vector for node $v$.
    *   $\mathbf{e}_{uv}$: The 4-dimensional feature vector for edge $(u,v)$.
    *   $P_{\text{or}}, Q_{\text{or}}$: Active and reactive power flow at the origin.
    *   $s_{uv}$: Operational status (1 or 0).
*   **Where it is used**: In `scripts/pyg_data.py` as the standard input format for the GNN.
*   **How it works in this project**: These vectors bundle electrical properties and topological state into a single format that PyTorch Geometric can process, ensuring the model considers both power flow and physical connectivity.

---

## `scripts/split.py`

🚫 **RETIRED — Inverse Class Frequency (ICF) Weighting** *(4-class loss; `scripts/split.py` no longer exists. N-1 uses masked BCE — see Current Formulas)*
To counteract the class imbalance during training, sample weights are determined using a square-root smoothed Inverse Class Frequency method:
$$ w_c = \sqrt{\frac{N}{C \cdot N_c}} \tag{15} $$

*   **Notations**:
    *   $w_c$: The training weight for class $c$.
    *   $N$: The total number of training samples.
    *   $C$: The number of distinct classes (e.g., 4).
    *   $N_c$: The count of samples belonging to class $c$.
*   **Where it is used**: In `scripts/split.py` via `compute_class_weights` and applied in the loss function in `training/train_gnn.py`.
*   **How it works in this project**: Rare faults (like cascades) have very few samples. This formula increases their importance during training so the GNN learns to recognize them even with limited data, while the square root prevents weights from becoming too extreme.

---

## `training/train_gnn.py`

⭐ **Graph Attention (GAT) with Edge Features** (Used in Chapter 5)
The model utilizes a multi-head attention mechanism that incorporates edge attributes $\mathbf{e}_{ij}$ into the calculation of attention coefficients $\alpha_{ij}$ between nodes $i$ and $j$. For each head $k$, the attention coefficient is:
$$ \alpha_{ij}^{(k)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^{(k)\top} [\mathbf{W}^{(k)} \mathbf{h}_i \, \Vert \, \mathbf{W}^{(k)} \mathbf{h}_j \, \Vert \, \mathbf{W}_e^{(k)} \mathbf{e}_{ij}]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^{(k)\top} [\mathbf{W}^{(k)} \mathbf{h}_i \, \Vert \, \mathbf{W}^{(k)} \mathbf{h}_k \, \Vert \, \mathbf{W}_e^{(k)} \mathbf{e}_{ik}]\right)\right)} \tag{16} $$

*   **Notations**:
    *   $\alpha_{ij}^{(k)}$: The importance of node $j$ to node $i$ in attention head $k$.
    *   $\mathbf{h}_i, \mathbf{h}_j$: The node embeddings for buses $i$ and $j$.
    *   $\mathbf{e}_{ij}$: The attribute vector of the line connecting $i$ and $j$.
    *   $\mathbf{W}, \mathbf{W}_e, \mathbf{a}$: Learnable weight matrices and vectors.
    *   $\Vert$: Concatenation operator.
*   **Where it is used**: In `training/train_gnn.py` within the `GATConv` layers of the `GridGNN` class.
*   **How it works in this project**: This formula allows the GNN to dynamically weight the "influence" of neighboring buses. By including edge features ($\mathbf{e}_{ij}$), the model can pay more attention to lines that are currently overloaded or tripped.

⭐ **Multi-Head Aggregation (Concatenation)** (Used in Chapter 5)
In the hidden layers of the GNN, the outputs from $K$ independent attention heads are concatenated to form the updated node representation, increasing the dimensionality of the feature space:
$$ \mathbf{h}_i^{(l)} = \text{ELU} \left( \text{BatchNorm} \left( \Vert_{k=1}^K \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j^{(l-1)} \right) \right) \tag{17} $$

*   **Notations**:
    *   $\mathbf{h}_i^{(l)}$: The node embedding at layer $l$.
    *   $K$: The number of attention heads (e.g., 4 or 8).
    *   $\text{ELU, BatchNorm}$: Activation and normalization functions applied to the output.
*   **Where it is used**: In `training/train_gnn.py` as the output of the hidden GAT layers.
*   **How it works in this project**: Using multiple heads allows the model to learn different aspects of grid behavior (e.g., one head for voltage drops, another for loading spikes) simultaneously, combining them into a rich feature embedding.

⭐ **LeakyReLU Activation** (Used in Chapter 5)
The Graph Attention layers utilize the LeakyReLU activation function within the attention mechanism to allow a small, non-zero gradient for negative values:
$$ \text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \sigma x & \text{if } x \le 0 \end{cases} \tag{18} $$

*   **Notations**:
    *   $x$: The input to the activation function.
    *   $\sigma$: The negative slope parameter (fixed at 0.2).
*   **Where it is used**: Internal to the `GATConv` layers in `training/train_gnn.py`.
*   **How it works in this project**: It prevents "dead" attention weights by allowing small gradients for negative values, ensuring that the GNN continues to learn even from less important components of the grid.

⭐ **Rectified Linear Unit (ReLU) Activation** (Used in Chapter 5)
The classifier and localizer heads utilize the ReLU activation function to introduce non-linearity while maintaining computational efficiency:
$$ \text{ReLU}(x) = \max(0, x) \tag{19} $$

*   **Notations**:
    *   $x$: The input to the activation function.
*   **Where it is used**: In the `classifier` and `localizer` MLP heads in `training/train_gnn.py`.
*   **How it works in this project**: ReLU is used in the final fully connected layers to model non-linear relationships between graph embeddings and the categorical fault labels.

⭐ **Exponential Linear Unit (ELU) Activation** (Used in Chapter 5)
The network employs the ELU activation function to ensure smooth gradients for negative inputs and avoid the "dying ReLU" problem:
$$ \text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases} \tag{20} $$

*   **Notations**:
    *   $x$: The input to the activation function.
    *   $\alpha$: The alpha parameter (usually 1.0).
*   **Where it is used**: In the hidden layers of `GridGNN` in `training/train_gnn.py`.
*   **How it works in this project**: ELU is preferred for hidden GNN layers because its negative values help push the mean activation closer to zero, accelerating convergence.

⭐ **Live Batch Normalization** (Used in Chapter 5)
To stabilize training without introducing dependency on long-term running statistics (crucial for non-stationary power grid data), the model uses batch normalization without tracking running stats:
$$ \hat{x} = \frac{x - E[x_{\text{batch}}]}{\sqrt{\text{Var}[x_{\text{batch}}] + \epsilon}} \tag{21} $$

*   **Notations**:
    *   $\hat{x}$: The normalized feature.
    *   $E[x_{\text{batch}}], \text{Var}[x_{\text{batch}}]$: The mean and variance of the current batch.
    *   $\epsilon$: A small constant ($10^{-5}$) for stability.
*   **Where it is used**: In `training/train_gnn.py` via `BatchNorm(..., track_running_stats=False)`.
*   **How it works in this project**: Grid states are highly dynamic. By disabling "running stats," the model normalizes each batch independently, making it more robust to shifting grid conditions during inference.

**Gradient Norm Clipping**
To prevent exploding gradients during backpropagation, the global norm of the gradients $\mathbf{g}$ is clipped to a threshold $\tau = 1.0$:
$$ \mathbf{g} \leftarrow \min\left(1, \frac{\tau}{\|\mathbf{g}\|_2}\right) \mathbf{g} \tag{22} $$

*   **Notations**:
    *   $\mathbf{g}$: The gradient vector.
    *   $\tau$: The clipping threshold (1.0).
    *   $\|\mathbf{g}\|_2$: The L2 norm of the gradient.
*   **Where it is used**: In the training loop of `training/train_gnn.py`.
*   **How it works in this project**: Large grid faults can cause massive spikes in the loss function. Clipping ensures these spikes don't destabilize the model's weights during training.

⭐ **Early Stopping Criterion** (Used in Chapter 5)
To prevent overfitting, the training process is terminated if the macro-averaged F1 score on the validation set fails to improve for a set number of epochs:
$$ \text{Terminate if: } \max_{i \in \{t-p, \dots, t\}} (\text{F1}_{\text{val}}^{(i)}) < \max_{j \in \{1, \dots, t-p-1\}} (\text{F1}_{\text{val}}^{(j)}) + \delta \tag{23} $$

*   **Notations**:
    *   $\text{F1}_{\text{val}}^{(i)}$: The validation Macro F1 score at epoch $i$.
    *   $p$: The patience (set to 15 epochs).
    *   $\delta$: The minimum delta (0.001).
*   **Where it is used**: In `training/train_gnn.py` via the `EarlyStopping` class.
*   **How it works in this project**: This ensures the model is saved at its peak generalization capability, preventing it from memorizing the training split.

**Batch Variance and Standard Deviation**
To normalize node and edge features purely based on the training split, the variance is calculated sequentially to avoid precision loss (computed in `float64`):
$$ \mu = \frac{1}{N} \sum_{k=1}^N x_k \tag{24} $$
$$ \sigma^2 = \max\left(0, \left( \frac{1}{N} \sum_{k=1}^N x_k^2 \right) - \mu^2\right) \tag{25} $$
$$ \sigma = \sqrt{\sigma^2} + \epsilon \tag{26} $$

*   **Notations**:
    *   $\mu, \sigma^2, \sigma$: Mean, variance, and standard deviation.
    *   $N$: Total count of nodes or edges in the training set.
    *   $\epsilon$: Stability constant ($10^{-7}$).
*   **Where it is used**: In `training/train_gnn.py` within `compute_normalization_stats`.
*   **How it works in this project**: These are used to calculate the static normalization parameters of the entire training set, which are then applied to all data (train, val, test) to ensure consistent scaling.

**Graph Attention Layer with Activation**
The forward pass applies an Exponential Linear Unit (ELU) alongside non-tracking Batch Normalization over the output of a Graph Attention layer (GATConv):
$$ \mathbf{h}^{(l)} = \text{ELU} \left( \text{BatchNorm} \left( \text{GATConv} (\mathbf{h}^{(l-1)}, \mathbf{e}) \right) \right) \tag{27} $$

*   **Notations**:
    *   $\mathbf{h}^{(l)}$: Node embeddings at layer $l$.
    *   $\mathbf{e}$: Edge attributes.
    *   $\text{GATConv}$: The Graph Attention operator.
*   **Where it is used**: The core layer block in `training/train_gnn.py`.
*   **How it works in this project**: This represents the fundamental "reasoning" step of the GNN, where node features are updated based on topology-aware attention and then normalized/activated for the next layer.

🚫 **RETIRED — Global Graph Pooling (Readout)** *(the N-1 model pools NOTHING; see the edge-level readout in Current Formulas. Pooling would erase the per-line distinction the task is about)*
To aggregate node-level features into a single fixed-size graph embedding $\mathbf{h}_G$, three distinct pooling mechanisms are concatenated (Mean, Max, and Min pooling):
$$ \mathbf{h}_G = \text{Concat}\left( \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v, \max_{v \in V} \mathbf{h}_v, \min_{v \in V} \mathbf{h}_v \right) \tag{28} $$

*   **Notations**:
    *   $\mathbf{h}_G$: The global graph embedding vector.
    *   $V$: The set of all nodes in the graph.
    *   $\mathbf{h}_v$: The final embedding of node $v$.
*   **Where it is used**: In `training/train_gnn.py` after the message-passing layers.
*   **How it works in this project**: To classify the entire grid state, we must reduce the graph to a single vector. Concatenating Mean, Max, and Min ensures the model captures both average health and localized extreme stress (overloads).

**Weighted Cross-Entropy Loss**
The classification task applies the ICF-derived class weights to the standard Cross-Entropy loss formulation:
$$ \mathcal{L}_{\text{cls}} = - \frac{1}{B} \sum_{j=1}^B w_{y_j} \log \left( \frac{\exp(z_{j, y_j})}{\sum_{c=1}^C \exp(z_{j, c})} \right) \tag{29} $$

*   **Notations**:
    *   $\mathcal{L}_{\text{cls}}$: The weighted classification loss.
    *   $w_{y_j}$: The ICF weight for class $y_j$.
    *   $z_{j, c}$: The logit output for class $c$ of sample $j$.
*   **Where it is used**: In `training/train_gnn.py` as the primary objective function.
*   **How it works in this project**: This forces the model to focus on correctly identifying rare but dangerous grid faults (high $w_c$) over common normal states (low $w_c$).

🚫 **RETIRED — Fault Localization Targets** *(the localizer head was removed with the task)*
The localization task uses a node-level target vector $\mathbf{t}$ where a value of 1 indicates the physical location of the fault (the bus or the origin substation of a line) and 0 indicates all other nodes:
$$ t_{g, i} = \begin{cases} 1 & \text{if node } i \in V_g \text{ is the identified fault location} \\ 0 & \text{otherwise} \end{cases} \tag{30} $$

*   **Notations**:
    *   $t_{g, i}$: The binary target for node $i$ in graph $g$.
    *   $V_g$: The set of nodes in graph $g$.
*   **Where it is used**: In `training/train_gnn.py` via `build_loc_targets_fast`.
*   **How it works in this project**: This defines the target for the "where is the fault?" task, marking the specific substation involved in a trip or overload.

⚠️ **SUPERSEDED — Binary Cross-Entropy for Localization** *(BCE is still the loss, but over the per-line N-1 label vector with masking, not over per-bus fault targets. See Current Formulas)*
The localization head is trained using Binary Cross-Entropy with logits, allowing the model to independently estimate the probability of each node being the fault source:
$$ \mathcal{L}_{\text{loc}} = - \frac{1}{N} \sum_{i=1}^N [t_i \log \sigma(z_{\text{loc}, i}) + (1 - t_i) \log(1 - \sigma(z_{\text{loc}, i}))] \tag{31} $$

*   **Notations**:
    *   $\mathcal{L}_{\text{loc}}$: The localization loss.
    *   $z_{\text{loc}, i}$: The logit output for node $i$.
    *   $\sigma$: The sigmoid activation function.
*   **Where it is used**: In `training/train_gnn.py` for the localization head.
*   **How it works in this project**: This trains the GNN to output a probability heat-map across the grid, identifying the most likely origin of a fault.

🚫 **RETIRED — Multi-Task Loss Formulation** *(there is no second head to weight; the N-1 loss is single-task)*
The total objective function optimized during training is a weighted combination of the classification and localization losses:
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{loc}} \tag{32} $$

*   **Notations**:
    *   $\mathcal{L}_{\text{total}}$: The final scalar loss to minimize.
    *   $\lambda$: The localization loss weight (set to 0.3).
*   **Where it is used**: In `training/train_gnn.py` as the optimizer input.
*   **How it works in this project**: By training on both tasks simultaneously, the GNN's node embeddings are forced to capture both the "type" and "location" of grid stress, leading to more robust perception.

**Cosine Annealing Learning Rate Schedule**
The learning rate $\eta$ is decayed following a cosine curve to promote better convergence and avoid local minima:
$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right) \tag{33} $$

*   **Notations**:
    *   $\eta_t$: The learning rate at epoch $t$.
    *   $T_{\text{cur}}, $T_{\text{max}}$: Current and total epochs.
*   **Where it is used**: In `training/train_gnn.py` via `CosineAnnealingLR`.
*   **How it works in this project**: It starts with a high learning rate to explore the loss landscape and smoothly decreases it to settle into a precise, stable minimum.

---

## `training/evaluate.py`

⭐ **Precision and Recall** (Used in Chapter 5)
To evaluate the model's performance on individual fault classes, precision and recall are calculated for each class $c$:
$$ P_c = \frac{TP_c}{TP_c + FP_c} \tag{34} $$
$$ R_c = \frac{TP_c}{TP_c + FN_c} \tag{35} $$

*   **Notations**:
    *   $TP_c, FP_c, FN_c$: True Positives, False Positives, and False Negatives for class $c$.
*   **Where it is used**: In `training/evaluate.py` for per-class performance metrics.
*   **How it works in this project**: Precision measures the reliability of fault predictions (safety), while recall measures the ability to catch all faults (resilience).

**Class-wise F1-score**
The harmonic mean of precision and recall provides a balanced measure of performance for each class:
$$ \text{F1}_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} \tag{36} $$

*   **Notations**:
    *   $P_c, R_c$: Precision and Recall for class $c$.
*   **Where it is used**: In `training/evaluate.py`.
*   **How it works in this project**: This is the primary metric for each individual fault type, ensuring both high accuracy and high coverage.

⚠️ **SUPERSEDED — Macro-Averaged F1 Score** *(`training/evaluate.py` no longer exists. N-1 is binary per contingency, so the reported metric is binary F1 plus average precision)*
The primary performance metric is the Macro F1 score, which treats all grid fault classes with equal importance regardless of their frequency:
$$ \text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} \tag{37} $$

*   **Notations**:
    *   $C$: The total number of classes.
*   **Where it is used**: In `training/train_gnn.py` and `training/evaluate.py` as the headline metric.
*   **How it works in this project**: It prevents the score from being inflated by the "normal" class, providing an honest assessment of how well the GNN detects rare fault scenarios.

**Z-Score Feature Standardization**
During dataset evaluation and testing, the static means and standard deviations computed from the training split are applied to standardize incoming features:
$$ x_{\text{norm}} = \frac{x - \mu}{\sigma} \tag{38} $$
$$ e_{\text{norm}} = \frac{e - \mu_e}{\sigma_e} \tag{39} $$

*   **Notations**:
    *   $\mu, \sigma$: Mean and standard deviation calculated ONLY from the training split.
*   **Where it is used**: In `training/evaluate.py` to prepare test data for inference.
*   **How it works in this project**: This ensures the test data is on the same scale as the training data, which is essential for the GNN's pre-trained weights to function correctly.

**Confusion Matrix**
To visualize the performance of the multi-class classifier, a confusion matrix $\mathbf{C}$ is constructed where each element $C_{i,j}$ represents the number of samples belonging to true class $i$ that were predicted as class $j$:
$$ C_{i,j} = \sum_{k=1}^N \mathbb{1}(y_k = i) \cdot \mathbb{1}(\hat{y}_k = j) \tag{40} $$

*   **Notations**:
    *   $C_{i,j}$: The count of class $i$ samples predicted as class $j$.
    *   $\mathbb{1}$: The indicator function.
*   **Where it is used**: In `training/evaluate.py` to diagnose model errors.
*   **How it works in this project**: It helps identify if the model is confusing specific fault types (e.g., confusing an `overload` for a `cascade`), allowing for targeted improvements in the dataset or architecture.

---

# Current Formulas — N-1 screening, the shield, and the knowledge graph

*Added 2026-08-20. Everything in this section reflects code that exists and results that were
measured. Equation numbers continue from the sections above.*

## `scripts/generate_dataset.py` — the N-1 label

⭐ **N-1 Contingency Label** (the target the model actually learns)

For a frame with observation $o$ and $N$ lines, the label is a **vector**, one entry per line:

$$ y_k = \begin{cases} 1 & \text{if } \max\big(\rho(\,\mathcal{F}(o, \neg k)\,)\big) \ge 1.0 \ \text{ or the episode terminates} \\ 0 & \text{if the post-contingency flow is within all limits} \\ -1 & \text{if line } k \text{ is already out, or the flow diverges} \end{cases} \tag{34} $$

*   **Notations**:
    *   $\mathcal{F}(o, \neg k)$: a **power-flow solve** on the network with line $k$ removed.
    *   $y_k \in \{1, 0, -1\}$: violation, secure, or *not evaluated*.
*   **Where it is used**: `scripts/generate_dataset.py`, written to `n1_violation` per frame.
*   **Why this matters more than any other equation in this document**: $\mathcal{F}$ is a
    **solver**, not an arithmetic expression over $o$. Unlike equation (5), $y_k$ **cannot be
    rewritten as a function of the present observation**, so no rule the shield could hold is
    capable of restating it. This is the property that un-rigs the neuro-symbolic comparison.
*   ⚠️ $-1$ entries are **missing labels, not negatives**, and must be masked out of the loss.

## `training/train_gnn.py` — the edge-level readout

⭐ **Per-Line Readout** (replaces global pooling, eq. 28)

For line $k$ with origin bus $u$ and extremity bus $v$, after $L$ rounds of message passing:

$$ z_k = \text{MLP}\Big( \big[\, \mathbf{h}_u^{(L)} \,\|\, \mathbf{h}_v^{(L)} \,\|\, \mathbf{e}_k \,\|\, \mathbf{x}_u \,\|\, \mathbf{x}_v \,\big] \Big) \in \mathbb{R} \tag{35} $$

*   **Notations**:
    *   $\mathbf{h}^{(L)}$: learned node embeddings after message passing.
    *   $\mathbf{x}$: the **raw** input features of the same nodes — a skip connection.
    *   $z_k$: a single logit; $\sigma(z_k) = P(\text{losing line } k \text{ violates a limit})$.
*   **Why nothing is pooled**: measured over all 22,000 frames, **96.4–98.8% are *mixed*** — some
    contingencies violate and others do not, within the same grid state. A graph-level vector
    cannot represent that. (No frame on any grid is entirely secure: 0.00% everywhere.)
*   **Why the raw skip $\mathbf{x}_u, \mathbf{x}_v$ is there**: post-contingency redistribution is
    governed by whether the endpoints have spare capacity to absorb line $k$'s flow. Three rounds
    of attention and normalization are free to render those quantities unrecognisable; the skip
    lets the head form the flow/spare ratio directly. **Measured ablation:** removing
    $\mathbf{h}^{(L)}$ from (35) costs $0.8987 \rightarrow 0.8411$.
*   **Why it is topology-agnostic**: (35) is defined per line, so a 20-line and a 186-line grid
    both evaluate on 36-bus-trained weights with no architectural change.

⭐ **Masked Binary Cross-Entropy** (the training objective)

$$ \mathcal{L} = -\frac{1}{|\mathcal{M}|} \sum_{k \in \mathcal{M}} \Big[ y_k \log \sigma(z_k) + (1 - y_k)\log\big(1 - \sigma(z_k)\big) \Big], \qquad \mathcal{M} = \{k : y_k \ne -1\} \tag{36} $$

*   **Where it is used**: `training/train_gnn.py`.
*   **How it works in this project**: single-task — there is no localization term and no
    $\alpha$ to tune (contrast the retired eq. 33). $\mathcal{M}$ excludes contingencies the
    simulator could not evaluate; counting them as negatives would teach the model that
    already-tripped lines are safe to lose.

## `shield/context.py` — the per-unit voltage contract

⭐ **Per-Line Base-kV Conversion** (supersedes the flat $\div 150$ of eq. 11 **for the shield**)

$$ V_{\text{pu}}^{\min} = \min_{k \,\in\, \mathcal{E}} \frac{V_{\text{or},k}}{V_{\text{base},k}}, \qquad V_{\text{pu}}^{\max} = \max_{k \,\in\, \mathcal{E}} \frac{V_{\text{or},k}}{V_{\text{base},k}}, \qquad \mathcal{E} = \{k : s_k = 1\} \tag{37} $$

*   **Notations**:
    *   $V_{\text{base},k}$: the **nominal voltage of line $k$**, from the backend, not a constant.
    *   $\mathcal{E}$: energized lines only — de-energized lines read 0 kV and would otherwise
        drive $V_{\text{pu}}^{\min}$ to 0 and fire every undervoltage rule.
*   🚨 **Why a constant divisor is wrong**: case14 operates lines at ~20 kV *and* ~138 kV; the
    36-bus grid has 7 lines at ~365 kV. A flat $\div 150$ yields ~100% false blocks.
*   ⚠️ These grids run **~6% above nominal**, so a healthy frame reads $V_{\text{pu}} \approx 1.06$,
    not $1.00$. A rule written assuming 1.00 will misfire.

## `evaluation/eval_shield_n1.py` — how the gate is scored

⭐ **Intervention Precision** (the headline metric)

$$ \text{IP} = \frac{|\{\text{blocked} \wedge \text{truly a violation}\}|}{|\{\text{blocked}\}|} = \frac{\text{corrections}}{\text{corrections} + \text{regressions}} \tag{38} $$

Measured **93.8% / 92.2% / 93.4%** on the 36-, 14- and 118-bus grids. The flatness is the finding:
the gate enforces a physical doctrine rather than a learned pattern, so its accuracy does not
depend on topology — while the model's F1 collapses $0.90 \rightarrow 0.42 \rightarrow 0.56$.

⭐ **Reach** (what *does* vary with topology)

$$ R = \frac{|\{\text{contingencies where the model says secure and a CONSTRAINT fires on the base case}\}|}{|\{\text{all contingencies scored}\}|} \tag{39} $$

Measured **0.31% / 0.087% / 3.04%**. The gate is not more accurate off-distribution — it simply
gets to speak more often, because the model is wrong more often in the way a rule can see.

⭐ **Structural Ceiling** (the honest bound on the whole approach)

$$ C = \frac{|\{\text{missed violations whose base case violates some rule}\}|}{|\{\text{missed violations}\}|} \tag{40} $$

Measured **16.2% / 0.51% / 30.9%**. The complement is a hard limit: **69–99% of the model's
dangerous errors occur on base cases that are entirely within limits.** Detecting those requires
solving the contingency — the exact computation the model exists to avoid. This answers
"would more rules have helped?" structurally, without building larger rulesets and plotting a curve.

⭐ **Average Precision** (threshold-free ranking quality)

$$ \text{AP} = \sum_n (R_n - R_{n-1})\, P_n \tag{41} $$

Reported alongside F1 so a result cannot be an artifact of a lucky threshold. ⚠️ The decision
threshold is selected on the **36-bus validation split and held fixed across all three grids** —
never re-tuned per topology.

## `kg/build.py` — predicate normalization

⭐ **Canonical Predicate Form**

$$ \pi(c) = \text{unparse}\Big( \mathcal{N}\big(\text{parse}(c)\big) \Big), \qquad \mathcal{N}: \ n \mapsto \lfloor n \rfloor \ \text{ if } n \in \mathbb{Z} \tag{42} $$

*   **Notations**: $c$ a condition string; $\text{parse}/\text{unparse}$ an abstract-syntax-tree
    round trip; $\mathcal{N}$ canonicalises numeric literals.
*   **Why it exists**: deduplication keys on $(\text{entity}, c)$, which splits **one physical
    check into three records** purely from noise in LLM output — `Line` in one standard versus
    `Facility` in another, `100` in one clause versus `100.0` in the next. $\pi$ collapses them.
*   ⚠️ **This is a view, not a substitution.** It never changes what the shield is served;
    collapsing the served rules would move `highest_severity` and every count derived from it.
*   **Result**: the thermal check resolves to **one** predicate, stated in **10 clauses across 4
    documents from 2 identified standards bodies**.
