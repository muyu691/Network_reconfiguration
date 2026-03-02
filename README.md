# Dual-Topology GNN for OD-Free Traffic Flow Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.x-3c78d8)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

---

## Overview
This project tackles that problem head-on: given the **old network state** (topology + historical equilibrium flows) and the **new network topology**, predict the new equilibrium flows **with zero OD information**.

**Core contributions:**
- A **Dual-Topology GNN** that treats `(G, G′)` network pairs as first-class citizens
- An Implicit Demand Virtual Routing Layer using Multi-Head Self-Attention to dynamically establish "virtual edges", enabling single-hop global flow redistribution and bypassing the myopic nature of GNNs.
- A vectorized **EdgeAlignmentModule** with O(1) hash-based matching of heterogeneous edge sets
- A **Physics-Informed loss** (Kirchhoff's conservation law) that acts as a structurally grounded regularizer

---

## Architecture Overview

### Phase 1 — Data Generation

```
Base Network G
    │
    ├─ Run Frank-Wolfe SUE solver  →  flow_old  [E_old, 1]  (demand proxy)
    │
    ├─ Apply random mutation  →  G′
    │     • topology_only  (40%): add/delete edges
    │     • attribute_only (30%): scale capacity / speed  ∈ [0.3×, 2.0×]
    │     • both           (30%): topology + attribute changes
    │
    └─ Re-run SUE on G′ (same OD, never exposed to model)  →  flow_new  [E_new, 1]  (GT)
```

Each training sample is a `(G, flow_old, G′) → flow_new` tuple.

---

### Phase 2 — Model Architecture

```
Old Graph G                              New Graph G′
 ─────────────────                        ────────────────────
 edge_index_old                           edge_index_new
 edge_attr_old  [E_old, 3]               edge_attr_new  [E_new, 3]
 flow_old       [E_old, 1]
        │                                        │
        ▼                                        │
 ┌──────────────────────┐                        │
 │   OldGraphEncoder    │                        │
 │   cat(attr, flow)    │                        │
 │   → GatedGCN × L    │                        │
 └──────────┬───────────┘                        │
            │  h_nodes_old  [N, H]               │
            │                  ┌─────────────────┘
            ▼                  ▼
    ┌────────────────────────────────────┐
    │       EdgeAlignmentModule          │
    │  Hash key = src_id × N + dst_id    │  ← O(1), zero Python loops
    │                                    │
    │  retained edge → [attr_old | flow_old | attr_new | is_new=0]
    │  new edge      → [zeros    | zero     | attr_new | is_new=1]
    └──────────────────┬─────────────────┘
                       │  aligned_features  [E_new, 8]
                       ▼
    ┌────────────────────────────────────┐
    │       NewGraphReasoner             │
    │                                    │
    │  1. Node Fusion (no residual):     │
    │    cat([ones(N,H), h_nodes_old])   │
    │    → Linear → ReLU → Dropout      │  ← forced re-learning on G′
    │                                    │
    │  2. Implicit Virtual Routing:      │  ← breaks GNN myopia
    │    Global Multi-Head Attention.    │
    │    Infers implicit OD & transfers  │
    │    global demand via virtual edges.│
    │                                    │
    │  3. GatedGCN × L  on G′ topology   │  ← local physical distribution
    │                                    │
    │  4. Edge Decoder:                  │
    │    [x_src ‖ x_dst ‖ aligned_feat] │
    │    → MLP → flow_pred  [E_new, 1]  │  ← unbounded (no activation)
    └────────────────────────────────────┘
```
**Loss Function:**

```
L_total = L_sup(pred, true) + λ · (1/|V|) Σ_v [ (inflow_v - outflow_v - Δ_v) / σ ]²

where:
  • First term  — supervised regression in normalized space
  • Second term — Global Kirchhoff flow conservation applied to all nodes
  • Δ_v         — The net physical demand of node v
  • ÷ σ         — dimensionless rescaling; prevents ~10⁶× gradient imbalance
```

---

## Baseline Models

### Baseline Hierarchy

| Tier | Model | Config | What it tests | Topology? | Learnable? |
|------|-------|--------|---------------|-----------|------------|
| 0 | **Test-Set Mean** | `configs/Baselines/test-mean-baseline.yaml` | Trivial lower bound (literature standard) | No | No |
| 1 | **Capacity-Proportional** | `configs/Baselines/capacity-proportional-baseline.yaml` | Can saturation ratio heuristic estimate flows? | No | No |
| 2 | **Edge MLP** | `configs/Baselines/mlp-baseline.yaml` | Can edge features alone (no message passing) predict flows? | No | Yes |
| 3 | **Single-Topology GatedGCN** | `configs/Baselines/single-topology-gatedgcn.yaml` | Is dual-graph encoding necessary, or is G' alone sufficient? | G' only | Yes |
| -- | **Dual-Topology GNN (Ours)** | `configs/GatedGCN/network-pairs-topology.yaml` | Full model with old graph encoding + virtual routing | G + G' | Yes |

### Running Baselines

```bash
# Tier 0: Test-Set Mean (1 epoch)
python main.py --cfg configs/Baselines/test-mean-baseline.yaml

# Tier 1: Capacity-Proportional (1 epoch)
python main.py --cfg configs/Baselines/capacity-proportional-baseline.yaml

# Tier 2: Edge MLP (200 epochs)
python main.py --cfg configs/Baselines/mlp-baseline.yaml

# Tier 3: Single-Topology GatedGCN (200 epochs)
python main.py --cfg configs/Baselines/single-topology-gatedgcn.yaml
```
### Results

| **Model**           | **MAE**   | **RMSE**  | **R2**    | **WMAPE (%)** | **Spearman** |
| ------------------- | ---------- | ---------- | ---------- | -------------- | ------------- |
| **Mean Baseline**   | 0.7942     | 1.0046     | 0.0006     | 119.81         | 0.0211        |
| **HA Baseline**     | 0.3534     | 0.6013     | 0.6420     | 53.31          | 0.8153        |
| **MLP Baseline**    | 0.2947     | 0.5291     | 0.7228     | 44.46          | 0.8731        |
| **Single-Top GNN**  | 0.2531     | 0.4674     | 0.7837     | 38.18          | 0.90575       |
| **Ours (GNN+PINN)** | **0.1839** | **0.3437** | **0.8830** | **27.75**      | **0.9356**    |
---

## Installation

```bash
# 1. Create environment
conda create -n dual-topo-gnn python=3.9 -y
conda activate dual-topo-gnn

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric + scatter
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. Install remaining dependencies
pip install -e .
pip install scikit-learn networkx tqdm wandb
```

---

## Usage

### Step 1 — Generate Network Pair Dataset

```bash
cd create_sioux_data

# Generate (G, G') scenario pairs and solve SUE for each
python solve_network_pairs.py

# Build PyG Data objects and train/val/test splits
python build_network_pairs_dataset.py \
    --output_dir processed_data/pyg_dataset
```

This produces:
```
create_sioux_data/processed_data/pyg_dataset/
├── train_dataset.pt
├── val_dataset.pt
├── test_dataset.pt
└── scalers/
    ├── flow_scaler.pkl   ← loaded automatically at training time
    └── attr_scaler.pkl
```

### Step 2 — Train

```bash
cd ..   # back to project root

python main.py --cfg configs/GatedGCN/network-pairs-topology.yaml
```

Key config options (edit `configs/GatedGCN/network-pairs-topology.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topology_gnn.hidden_dim` | 128 | GNN hidden dimension |
| `topology_gnn.num_layers_old` | 3 | GatedGCN layers in OldGraphEncoder |
| `topology_gnn.num_layers_new` | 3 | GatedGCN layers in NewGraphReasoner |
| `topology_gnn.num_heads` | 4 | Attention heads for Implicit Virtual Routing |
| `model.lambda_cons` | 0.1 | Conservation loss weight λ |
| `optim.max_epoch` | 200 | Training epochs |
| `train.batch_size` | 32 | Graphs per batch |

### Step 3 — Monitor Training

```bash
tensorboard --logdir results/
```

---

## Project Structure

```
GraphGPS_implement-main/
├── configs/
│   ├── GatedGCN/
│   │   └── network-pairs-topology.yaml          # Main model config
│   └── Baselines/
│       ├── test-mean-baseline.yaml              # Tier 0
│       ├── capacity-proportional-baseline.yaml   # Tier 1
│       ├── mlp-baseline.yaml                     # Tier 2
│       └── single-topology-gatedgcn.yaml         # Tier 3
├── create_sioux_data/
│   ├── generate_scenarios.py          # Base network + mutation generation
│   ├── solve_network_pairs.py         # Frank-Wolfe SUE solver
│   └── build_network_pairs_dataset.py # PyG Data object builder
├── graphgps/
│   ├── config/
│   │   └── topology_gnn_config.py     # Hyperparameter registration
│   ├── loader/dataset/
│   │   └── network_pairs_topology.py  # Dataset loader
│   ├── loss/
│   │   └── flow_conservation_loss.py  # PINN conservation loss
│   ├── network/
│   │   ├── topology_model.py          # Core dual-topology model (3 modules)
│   │   ├── heuristic_baselines.py     # Tier 0 & 1: non-learnable baselines
│   │   ├── mlp_baseline.py            # Tier 2: edge-level MLP
│   │   └── single_topology_gatedgcn.py # Tier 3: single-graph GatedGCN
│   └── train/
│       └── custom_train.py            # Training loop with PINN loss dispatch
└── main.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
