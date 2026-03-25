# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Common Development Commands

### Environment Setup
- Install Python dependencies: `pip install -r requirements.txt`
- Install Grid2Op and lightsim2grid: `pip install grid2op lightsim2grid`
- Install PyTorch Geometric: Follow instructions at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Install Ollama for LLM: https://ollama.com/download
- Pull Qwen2.5-32B model: `ollama pull qwen2.5:32b`

### Data Generation
- Generate dataset using Grid2Op: `python scripts/generate_dataset.py`
- Uses l2rpn_neurips_2020_track1_small environment by default
- For large-scale runs: `python scripts/generate_dataset.py --env l2rpn_neurips_2020_track1_large`

### LLM Knowledge Extraction
- Extract rules from documents: `python scripts/extract_rules.py`
- Processes PDFs/text files in `data/documents/`
- Outputs structured JSON rules to `data/extracted_rules.json`

### GNN Training
- Train the GNN model: `python scripts/train_gnn.py`
- Uses stratified train/val/test split (70/15/15)
- Logs metrics to Weights & Biases (requires wandb login)
- Model checkpoints saved to `models/`

### Knowledge Graph Construction
- Build knowledge graph from extracted rules: `python scripts/build_kg.py`
- Creates NetworkX graph pickle at `data/knowledge_graph.gpickle`

### Symbolic Validation Shield
- Run shield validation: `python scripts/validate_shield.py`
- Takes GNN predictions and validates against knowledge graph rules

### End-to-End Testing
- Run toy scenario tests: `python scripts/test_toy_scenarios.py`
- Tests clean pass, overload, and cascading risk scenarios on rte_case5_example
- Run formal evaluation: `python scripts/evaluate_end_to_end.py`
- Evaluates on held-out chronics from l2rpn_neurips_2020_track1_small

### Experiment Tracking
- View training metrics: `wandb ui` (after launching training)
- Compare runs: Weights & Biases dashboard

## High-Level Architecture

This repository implements a neuro-symbolic system for power grid fault detection and validation with four main components:

1. **Graph Neural Network (Neural Layer)** - Component A
   - Uses PyTorch Geometric with GAT/GCN architectures
   - Processes grid state as graph (buses=nodes, lines=edges)
   - Outputs fault classification and localization recommendations
   - Input features: voltage, power flows, load/generation, bus types
   - Two-headed design: classification head (fault type) + localization head (fault location)

2. **LLM Knowledge Extraction Pipeline** - Component B
   - Uses Qwen2.5-32B via Ollama (4-bit quantized) for local, reproducible inference
   - Extracts structured safety rules from IEEE standards and grid documentation
   - Pipeline: Document chunking → LLM prompting → JSON parsing → Validation → Deduplication
   - Output schema includes rule_id, source, entity, condition, action, severity, explanation

3. **Knowledge Graph (Symbolic Rule Store)** - Component C
   - Built with NetworkX (planned migration to Neo4j if scale requires)
   - Stores extracted rules and grid topology as queryable graph
   - Node types: Bus, Line, Transformer, Generator, Load, ProtectionDevice, Rule
   - Edge types: connected_to, feeds, protected_by, has_rule, triggers
   - Enables relational rule traversal (e.g., "if Line overloaded, check protected breaker")

4. **Symbolic Validation Shield** - Component D
   - Hard constraint validation layer (not a loss function penalty)
   - Intercepts every GNN prediction and validates against KG rules
   - Returns PASS (with validated prediction) or BLOCK (with structured explanation)
   - Provides traceable explanations citing specific rule IDs and source documents

### Key Design Principles
- **GNN outputs are never final without symbolic validation** - The shield is always active
- **Local LLM preference** - For reproducibility and data sensitivity (Qwen2.5-32B via Ollama)
- **Grid2Op simulation** - Using l2rpn_neurips_2020_track1 as primary environment (36 subs, 59 lines)
- **LightSimBackend** - ~10x faster power flow solver for data generation
- **Explainability focus** - Every blocked decision cites specific rule and source document

### Technology Stack
- Grid simulation: Grid2Op + lightsim2grid backend
- GNN framework: PyTorch + PyTorch Geometric
- LLM inference: Ollama + Qwen2.5-32B (4-bit)
- LLM orchestration: LangChain
- Knowledge graph: NetworkX → Neo4j (if needed)
- Shield logic: Custom Python
- Data validation: Pydantic
- Experiment tracking: Weights & Biases
- Version control: Git + GitHub

### Hardware Allocation
- **Research PC**: Intel i7-14700K, RTX 4080 Super (16GB VRAM), 64GB RAM
  - Handles GNN training, dataset generation, LLM extraction
- **Personal PC**: AMD Ryzen 5 7500F, Intel Arc B580 (12GB VRAM), 16GB RAM
  - Code development, unit tests, shield logic, small-scale tests only

### Evaluation Plan
- GNN metrics: Accuracy, Macro F1-score, Precision/Recall per fault class
- Shield metrics: Rule compliance rate, False block rate, Block precision
- LLM extraction: Rule coverage, Rule precision, Parse success rate
- End-to-end: Safe action rate, Blocked action rate, Explanation quality (1-5 human rating)