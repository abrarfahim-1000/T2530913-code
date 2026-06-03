# Mathematical Formulas

This document compiles the mathematical formulas and formal definitions extracted from the codebase, organized by file. These representations are formatted for inclusion in a thesis report, detailing the data transformations, balancing techniques, and neural network mechanics used in the project.

## `extraction/extract.py`

**Text Chunking Strategy**
To process large documents with local LLMs, the text is partitioned into overlapping segments of fixed size. The start index of the $i$-th chunk is defined as:
$$ \text{start}_i = i \cdot (S - O) $$
$$ \text{end}_i = \text{start}_i + S $$
where $S$ is the chunk size (characters) and $O$ is the overlap between consecutive chunks.

## `scripts/audit_datasets.py`

**Class Imbalance Ratio**
To quantify the severity of the class imbalance within the dataset, the ratio between the majority class and the minority class is calculated:
$$ \text{Imbalance Ratio} = \frac{\max_{c \in C} N_c}{\min_{c \in C} N_c + \epsilon} $$
where $N_c$ is the number of samples belonging to class $c \in C$, and $\epsilon = 10^{-9}$ is a small constant added for numerical stability to prevent division by zero.

**Statistical Outlier Detection (Rho Percentiles)**
To characterize the distribution of line loadings across the dataset, the $p$-th percentile is calculated to identify typical and extreme stress states:
$$ \text{Percentile}(p) = \text{inf} \{ x \in \mathbb{R} : F(x) \geq p/100 \} $$
where $F(x)$ is the empirical cumulative distribution function of the $\rho$ values. Specifically, the 95th and 99th percentiles are monitored.

## `scripts/generate_dataset.py`

**Physical State Labeling Logic**
The ground-truth label $y$ for each grid state is determined by a hierarchical physical rule set based on the loading ratio $\rho$ and the connectivity of the $N$ transmission lines:
$$ y = \begin{cases} \text{overload} & \text{if } \max(\rho) \ge 1.0 \\ \text{normal} & \text{else if } \sum_{i=1}^N s_i = N \\ \text{line\_trip} & \text{else if } \sum_{i=1}^N s_i = N - 1 \\ \text{cascade} & \text{else if } \sum_{i=1}^N s_i < N - 1 \end{cases} $$
where $s_i \in \{0, 1\}$ is the operational status of line $i$.

**Transmission Line Loading Clipping**
To prevent extreme values or measurement errors from destabilizing the neural network during data generation and feature extraction, the loading ratio $\rho$ is clipped to a predefined threshold $\rho_{\text{max}} = 2.0$:
$$ \rho' = \min(\rho, \rho_{\text{max}}) $$

## `scripts/pyg_data.py`

**Bus Load Summation**
The total active load $P_i$ at bus $i$ is the aggregate of all individual loads $p_l$ physically mapped to that substation:
$$ P_i = \sum_{l \in \mathcal{L}_i} p_l $$
where $\mathcal{L}_i$ is the set of indices of loads connected to bus $i$.

**Dynamic Graph Pruning (Topology Masking)**
To ensure the GNN only propagates messages across physically intact lines, the adjacency matrix $\mathbf{A}$ and edge attribute matrix $\mathbf{E}$ are dynamically pruned at each time step $t$ using the line status vector $\mathbf{s}_t$:
$$ \mathcal{E}_t = \{ (u, v)_k \in \mathcal{E}_{\text{static}} : s_{t,k} = 1 \} $$
where $\mathcal{E}_t$ is the set of edges included in the message-passing graph for the current state.

**Mean Bus Voltage**
The mean voltage at each bus $i$ is calculated by averaging the voltage readings of all active lines connected to it:
$$ \bar{V}_i = \frac{1}{|E_i|} \sum_{e \in E_i} V_{\text{or}}^{(e)} $$
where $E_i$ represents the set of lines connected to bus $i$, and $V_{\text{or}}^{(e)}$ is the origin voltage of line $e$.

**Max Bus Loading Ratio**
To capture localized stress signals, the maximum loading ratio across all transmission lines connected to a substation is extracted for each bus:
$$ \rho_{\max, i} = \max_{e \in E_i} (\rho^{(e)}) $$
where $\rho^{(e)}$ is the current loading ratio of line $e$.

**Voltage Normalization**
To scale the node voltage features into a stable range for the neural network, the raw voltage is normalized by a constant factor (150.0 kV):
$$ V_{\text{norm}} = \frac{V_{\text{raw}}}{150.0} $$

**Connected Line Fraction**
A topological stability feature is computed for each bus, indicating the proportion of intact lines:
$$ f_{\text{conn}}^{(i)} = \frac{N_{\text{connected}}^{(i)}}{N_{\text{total}}^{(i)}} $$
where $N_{\text{connected}}^{(i)}$ is the number of currently active lines at bus $i$, and $N_{\text{total}}^{(i)}$ is the total number of physical lines originally connected to the bus.

**Node and Edge Feature Vectors**
The GNN input space is formally defined by the following feature compositions for each node $v$ and edge $e$:
$$ \mathbf{x}_v = [P_{\text{load}, v}, \bar{V}_v, \rho_{\max, v}, f_{\text{conn}, v}]^\top \in \mathbb{R}^4 $$
$$ \mathbf{e}_{uv} = [\rho_{uv}, P_{\text{or}, uv}, Q_{\text{or}, uv}, s_{uv}]^\top \in \mathbb{R}^4 $$
where $s_{uv} \in \{0, 1\}$ is the operational status of the transmission line.

## `scripts/split.py`

**Inverse Class Frequency (ICF) Weighting**
To counteract the class imbalance during training, sample weights are determined using a square-root smoothed Inverse Class Frequency method:
$$ w_c = \sqrt{\frac{N}{C \cdot N_c}} $$
where $N$ is the total number of samples across all classes, $C$ is the total number of distinct classes, and $N_c$ is the sample count for class $c$. If $N_c = 0$, $w_c$ is strictly set to $0$.

## `training/train_gnn.py`

**Graph Attention (GAT) with Edge Features**
The model utilizes a multi-head attention mechanism that incorporates edge attributes $\mathbf{e}_{ij}$ into the calculation of attention coefficients $\alpha_{ij}$ between nodes $i$ and $j$. For each head $k$, the attention coefficient is:
$$ \alpha_{ij}^{(k)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^{(k)\top} [\mathbf{W}^{(k)} \mathbf{h}_i \, \Vert \, \mathbf{W}^{(k)} \mathbf{h}_j \, \Vert \, \mathbf{W}_e^{(k)} \mathbf{e}_{ij}]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^{(k)\top} [\mathbf{W}^{(k)} \mathbf{h}_i \, \Vert \, \mathbf{W}^{(k)} \mathbf{h}_k \, \Vert \, \mathbf{W}_e^{(k)} \mathbf{e}_{ik}]\right)\right)} $$
where $\Vert$ denotes concatenation and $\mathbf{W}, \mathbf{W}_e$ are learnable weight matrices.

**Multi-Head Aggregation (Concatenation)**
In the hidden layers of the GNN, the outputs from $K$ independent attention heads are concatenated to form the updated node representation, increasing the dimensionality of the feature space:
$$ \mathbf{h}_i^{(l)} = \text{ELU} \left( \text{BatchNorm} \left( \Vert_{k=1}^K \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j^{(l-1)} \right) \right) $$

**LeakyReLU Activation**
The Graph Attention layers utilize the LeakyReLU activation function within the attention mechanism to allow a small, non-zero gradient for negative values:
$$ \text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \sigma x & \text{if } x \le 0 \end{cases} $$
where the slope parameter $\sigma$ is set to 0.2.

**Rectified Linear Unit (ReLU) Activation**
The classifier and localizer heads utilize the ReLU activation function to introduce non-linearity while maintaining computational efficiency:
$$ \text{ReLU}(x) = \max(0, x) $$

**Exponential Linear Unit (ELU) Activation**
The network employs the ELU activation function to ensure smooth gradients for negative inputs and avoid the "dying ReLU" problem:
$$ \text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases} $$
where $\alpha$ is typically set to 1.0.

**Live Batch Normalization**
To stabilize training without introducing dependency on long-term running statistics (crucial for non-stationary power grid data), the model uses batch normalization without tracking running stats:
$$ \hat{x} = \frac{x - E[x_{\text{batch}}]}{\sqrt{\text{Var}[x_{\text{batch}}] + \epsilon}} $$
where the normalization is performed strictly using the current batch's mean and variance during both training and inference.

**Gradient Norm Clipping**
To prevent exploding gradients during backpropagation, the global norm of the gradients $\mathbf{g}$ is clipped to a threshold $\tau = 1.0$:
$$ \mathbf{g} \leftarrow \min\left(1, \frac{\tau}{\|\mathbf{g}\|_2}\right) \mathbf{g} $$

**Early Stopping Criterion**
To prevent overfitting, the training process is terminated if the macro-averaged F1 score on the validation set fails to improve for a set number of epochs:
$$ \text{Terminate if: } \max_{i \in \{t-p, \dots, t\}} (\text{F1}_{\text{val}}^{(i)}) < \max_{j \in \{1, \dots, t-p-1\}} (\text{F1}_{\text{val}}^{(j)}) + \delta $$
where $p$ is the patience (15 epochs) and $\delta$ is the minimum delta (0.001).

**Batch Variance and Standard Deviation**
To normalize node and edge features purely based on the training split, the variance is calculated sequentially to avoid precision loss (computed in `float64`):
$$ \mu = \frac{1}{N} \sum_{k=1}^N x_k $$
$$ \sigma^2 = \max\left(0, \left( \frac{1}{N} \sum_{k=1}^N x_k^2 \right) - \mu^2\right) $$
$$ \sigma = \sqrt{\sigma^2} + \epsilon $$
where $N$ is the total number of node (or edge) instances across the training graphs, and $\epsilon = 10^{-7}$ prevents division by zero during standardization.

**Graph Attention Layer with Activation**
The forward pass applies an Exponential Linear Unit (ELU) alongside non-tracking Batch Normalization over the output of a Graph Attention layer (GATConv):
$$ \mathbf{h}^{(l)} = \text{ELU} \left( \text{BatchNorm} \left( \text{GATConv} (\mathbf{h}^{(l-1)}, \mathbf{e}) \right) \right) $$
where $\mathbf{h}^{(l)}$ represents the node embeddings at layer $l$, and $\mathbf{e}$ represents the edge attributes.

**Global Graph Pooling (Readout)**
To aggregate node-level features into a single fixed-size graph embedding $\mathbf{h}_G$, three distinct pooling mechanisms are concatenated (Mean, Max, and Min pooling):
$$ \mathbf{h}_G = \text{Concat}\left( \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v, \max_{v \in V} \mathbf{h}_v, \min_{v \in V} \mathbf{h}_v \right) $$
Note: The minimum pooling is practically implemented using the mathematical identity $\min(x) = -\max(-x)$.

**Weighted Cross-Entropy Loss**
The classification task applies the ICF-derived class weights to the standard Cross-Entropy loss formulation:
$$ \mathcal{L}_{\text{cls}} = - \frac{1}{B} \sum_{j=1}^B w_{y_j} \log \left( \frac{\exp(z_{j, y_j})}{\sum_{c=1}^C \exp(z_{j, c})} \right) $$
where $B$ is the batch size, $w_{y_j}$ is the static weight for the ground-truth class $y_j$, and $z_{j,c}$ is the predicted logit for class $c$ of the $j$-th sample.

**Fault Localization Targets**
The localization task uses a node-level target vector $\mathbf{t}$ where a value of 1 indicates the physical location of the fault (the bus or the origin substation of a line) and 0 indicates all other nodes:
$$ t_{g, i} = \begin{cases} 1 & \text{if node } i \in V_g \text{ is the identified fault location} \\ 0 & \text{otherwise} \end{cases} $$
where $V_g$ is the set of nodes in graph $g$.

**Binary Cross-Entropy (BCE) for Localization**
The localization head is trained using Binary Cross-Entropy with logits, allowing the model to independently estimate the probability of each node being the fault source:
$$ \mathcal{L}_{\text{loc}} = - \frac{1}{N} \sum_{i=1}^N [t_i \log \sigma(z_{\text{loc}, i}) + (1 - t_i) \log(1 - \sigma(z_{\text{loc}, i}))] $$
where $N$ is the total number of nodes in the batch, $z_{\text{loc}, i}$ is the localization logit for node $i$, and $\sigma(\cdot)$ is the sigmoid activation function.

**Multi-Task Loss Formulation**
The total objective function optimized during training is a weighted combination of the classification and localization losses:
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{loc}} $$
where $\lambda$ (loc_loss_weight) is a hyperparameter (set to 0.3) that controls the influence of the localization task.

**Cosine Annealing Learning Rate Schedule**
The learning rate $\eta$ is decayed following a cosine curve to promote better convergence and avoid local minima:
$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right) $$
where $T_{\text{max}}$ is the total number of epochs and $T_{\text{cur}}$ is the current epoch.

## `training/evaluate.py`

**Precision and Recall**
To evaluate the model's performance on individual fault classes, precision and recall are calculated for each class $c$:
$$ P_c = \frac{TP_c}{TP_c + FP_c} $$
$$ R_c = \frac{TP_c}{TP_c + FN_c} $$
where $TP_c$, $FP_c$, and $FN_c$ represent the number of true positives, false positives, and false negatives for class $c$, respectively.

**Class-wise F1-score**
The harmonic mean of precision and recall provides a balanced measure of performance for each class:
$$ \text{F1}_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} $$

**Macro-Averaged F1 Score**
The primary performance metric is the Macro F1 score, which treats all grid fault classes with equal importance regardless of their frequency:
$$ \text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} $$
where $P_c$ and $R_c$ are the precision and recall for class $c$, respectively.

**Z-Score Feature Standardization**
During dataset evaluation and testing, the static means and standard deviations computed from the training split are applied to standardize incoming features:
$$ x_{\text{norm}} = \frac{x - \mu}{\sigma} $$
$$ e_{\text{norm}} = \frac{e - \mu_e}{\sigma_e} $$
where $x$ represents the node features and $e$ represents the edge attributes.

**Confusion Matrix**
To visualize the performance of the multi-class classifier, a confusion matrix $\mathbf{C}$ is constructed where each element $C_{i,j}$ represents the number of samples belonging to true class $i$ that were predicted as class $j$:
$$ C_{i,j} = \sum_{k=1}^N \mathbb{1}(y_k = i) \cdot \mathbb{1}(\hat{y}_k = j) $$
where $N$ is the total number of test samples and $\mathbb{1}(\cdot)$ is the indicator function.
