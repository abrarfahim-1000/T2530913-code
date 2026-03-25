# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔧 Common Development Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Verification Scripts
```bash
# Check API key validity
python check.py

# Verify Grid2Op setup
python verify_grid2op.py

# Test stack validation
python verify_stack.py

# Quick CUDA/XPU check
python t.py
```

### Data Generation
```bash
# Generate dataset using Grid2Op (run on Research PC)
# See study.md section 6 for detailed script
python -c "
import grid2op
from lightsim2grid import LightSimBackend
import json

env = grid2op.make('l2rpn_neurips_2020_track1_small', backend=LightSimBackend())
# ... (refer to study.md lines 378-438 for full script)
"
```

### GNN Training
```bash
# Train GNN model (see study.md section 7)
# Execute on Research PC with RTX 4080 Super
python -c "
import torch
from torch_geometric.loader import DataLoader
# ... (refer to study.md lines 463-611 for training loop)
"
```

### LLM Knowledge Extraction
```bash
# Extract rules from documents using Qwen2.5-32B via Ollama
# Run on Research PC (see study.md section 3)
ollama run qwen2.5:32b
# ... (refer to study.md lines 150-174 for extraction pipeline)
```

### End-to-End Testing
```bash
# Run toy scenario validation (see study.md section 8)
python -c "
import grid2op
from lightsim2grid import LightSimBackend
import torch
import networkx as nx
# ... (refer to study.md lines 648-712 for full wiring code)
"
```

## 🏗️ High-Level Architecture

This repository implements a **Neuro-Symbolic AI system for power grid fault detection and action validation** with four tightly integrated components:

### 1. **Graph Neural Network (GNN) - Neural Layer** (`Component A`)
- **Purpose**: Perceives grid state and outputs fault classifications, locations, and recommended actions
- **Input**: Grid snapshot as graph with node features (voltage, load, generation, bus type) and edge features (power flow, loading %, line parameters)
- **Output**:
  - Node-level: fault probability per bus
  - Graph-level: fault type classification (normal/overload/line_trip/cascade/maintenance)
  - Action head: recommended control action
- **Tech**: PyTorch Geometric with GAT/GCN layers
- **Key File References**: See study.md lines 67-113

### 2. **LLM Knowledge Extraction Pipeline** (`Component B`)
- **Purpose**: Automatically extracts symbolic rules from grid standards/documents
- **Process**:
  1. Document ingestion (IEEE standards, grid manuals) via LangChain
  2. Structured rule extraction using Qwen2.5-32B (4-bit quantized) via Ollama
  3. JSON validation with Pydantic schema
  4. Deduplication of overlapping rules
- **Output Schema**: rule_id, source, entity, condition, action, severity, explanation
- **Tech**: Ollama + LangChain + Pydantic
- **Key File References**: See study.md lines 115-196

### 3. **Knowledge Graph (Symbolic Rule Store)** (`Component C`)
- **Purpose**: Stores extracted rules and grid topology as queryable graph
- **Schema**:
  - Nodes: Bus, Line, Transformer, Generator, Load, ProtectionDevice, Rule
  - Edges: connected_to, feeds, protected_by, has_rule, triggers
- **Tech**: NetworkX (with potential Neo4j migration for scale)
- **Key File References**: See study.md lines 198-258

### 4. **Symbolic Validation Shield** (`Component D`)
- **Purpose**: Hard constraint validation layer that blocks GNN predictions violating symbolic rules
- **Critical Principle**: GNN outputs are NEVER final without symbolic validation (see study.md lines 61-63)
- **Process**:
  1. Intercept GNN prediction
  2. Check against all applicable rules in knowledge graph
  3. Return PASS (with validated prediction) or BLOCK (with structured explanation)
- **Tech**: Custom Python logic
- **Key File References**: See study.md lines 260-319

## 📊 Key Technologies & Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Grid Simulation | Grid2Op + LightSim2Grid | RTE-developed framework, 10x faster power flow, L2RPN benchmark |
| GNN Framework | PyTorch + PyTorch Geometric | Standard for GNN research, full architecture flexibility |
| LLM Inference | Ollama + Qwen2.5-32B (4-bit) | Local, reproducible, superior structured extraction |
| LLM Orchestration | LangChain | JSON output parsing, document loaders, prompt chaining |
| Knowledge Graph | NetworkX → Neo4j (if needed) | Zero-setup for research scale |
| Validation Shield | Custom Python | Fully auditable, no external dependencies |
| Experiment Tracking | Weights & Biases | Training metrics, hyperparameter logging |
| Data Validation | Pydantic | Schema enforcement on LLM outputs |

## 💻 Hardware Allocation

### Research PC (Primary - Heavy Workloads)
- **CPU**: Intel Core i7-14700K (20 cores/28 threads)
- **GPU**: NVIDIA RTX 4080 Super (16GB VRAM)
- **RAM**: DDR5 64GB
- **Runs**: GNN training, dataset generation, LLM extraction, hyperparameter sweeps

### Development PC (Light Testing Only)
- **CPU**: AMD Ryzen 5 7500F
- **GPU**: Intel Arc B580 (12GB VRAM)
- **RAM**: DDR5 16GB
- **Runs**: Code development, unit tests, shield logic, smoke tests

## 🧪 Validation & Testing Approach

### Toy Scenario Testing (Prerequisite)
Before full evaluation, validate end-to-end pipeline using `rte_case5_example` (5-bus environment):
1. **Clean pass**: Normal grid state → Expected: PASS
2. **Overload condition**: Line at 115% limit → Expected: BLOCK + explanation
3. **Cascading risk**: Lines at 90% load → Expected: BLOCK + multi-rule explanation

See study.md lines 619-751 for detailed toy scenario implementation.

### Formal Evaluation
After toy scenarios pass, evaluate on held-out chronics from `l2rpn_neurips_2020_track1_small`:
- Compare GNN-only vs GNN+Shield output
- Metrics: safe action rate, blocked action rate, explanation quality
- Target: Match/exceed Younesi et al. (2026) 91.7% safe restoration

See study.md lines 795-833 for evaluation plan details.

## 📚 Important Documentation

- **[study.md](study.md)**: Complete methodology and implementation guide (primary reference)
- **[requirements.txt](requirements.txt)**: Full dependency list
- **Gemini.md**: Additional project context

## ⚠️ Critical Constraints

1. **Shield Non-Negotiability**: GNN outputs must always pass through symbolic validation - no bypass mode exists
2. **Hardware Separation**: Heavy workloads (training, extraction, generation) MUST run on Research PC
3. **Local LLM Requirement**: Use Ollama with Qwen2.5-32B (4-bit) for reproducibility - no API calls
4. **Simulation-Only**: System uses Grid2Op simulation; no real utility deployment in scope
5. **Exclusions**: Foundation model training, transmission grids, blockchain, RL agents (future work)

## 🔍 Navigating the Codebase

While this repository primarily contains documentation and verification scripts, the full implementation would follow the patterns detailed in study.md. Key implementation files would be developed according to:

- **Data Generation**: Lines 378-438 in study.md
- **GNN Architecture**: Lines 515-564 in study.md
- **Training Loop**: Lines 568-611 in study.md
- **End-to-End Wiring**: Lines 648-712 in study.md

When implementing, refer to the specific line numbers in study.md for detailed specifications.