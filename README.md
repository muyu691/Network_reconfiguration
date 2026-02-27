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
    │  Node Fusion (no residual):        │
    │    cat([ones(N,H), h_nodes_old])   │
    │    → Linear → ReLU → Dropout      │  ← forced re-learning on G′
    │                                    │
    │  GatedGCN × L  on G′ topology      │
    │                                    │
    │  Edge Decoder:                     │
    │    [x_src ‖ x_dst ‖ aligned_feat] │
    │    → MLP → flow_pred  [E_new, 1]  │  ← unbounded (no activation)
    └────────────────────────────────────┘
```

**Loss Function:**

```
L_total = L1(pred, true)  +  λ · (1/|V_nc|) Σ_v [(inflow_v - outflow_v) / σ]²

where:
  • First term  — supervised regression in normalized space
  • Second term — Kirchhoff conservation on non-centroid nodes (nodes 12–24)
  • ÷ σ         — dimensionless rescaling; prevents ~10⁶× gradient imbalance
```

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
python main_create_dataset.py

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
├── configs/GatedGCN/
│   └── network-pairs-topology.yaml    # Main training config
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
│   │   └── topology_model.py          # Core model (3 modules)
│   └── train/
│       └── custom_train.py            # Training loop with PINN loss dispatch
└── main.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
