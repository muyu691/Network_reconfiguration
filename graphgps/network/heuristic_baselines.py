"""
Heuristic Baselines for No-OD Traffic Flow Prediction
======================================================

Two non-learnable baselines to establish performance lower bounds:

1. TestMeanBaseline (test_mean_baseline)
   Predicts the mean of the current batch's ground truth for every edge.
   Replicates the "Test-set Mean" evaluation strategy from:
     Oskar Bohn Lassen et al., "Learning traffic flows: Graph Neural Networks 
     for Metamodelling Traffic Assignment", arXiv 2025.

2. HistoricalAverageBaseline (capacity_proportional_baseline)
   (Retained the registry name for compatibility with your yaml)
   Uses the pure historical flow for retained edges. For newly added edges, 
   it predicts the mean historical flow of the network. This avoids the 
   mathematical explosion of dividing by Z-Score normalized capacities.

Engineering safety:
  Both classes contain a dummy_param (nn.Parameter) to keep PyTorch
  optimizers and DDP happy—no real gradients flow through it.
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_network

from graphgps.network.topology_model import EdgeAlignmentModule


# ================================================================
# Baseline 1: Test-Set Mean
# ================================================================

@register_network('test_mean_baseline')
class TestMeanBaseline(nn.Module):
    """
    Outputs batch.y.mean() for every edge in the new graph.

    NOTE: Using batch.y.mean() is intentional—it faithfully reproduces the
    "Test-set Mean" baseline evaluation protocol from the literature
    (Oskar Bohn Lassen et al., 2025). This is NOT data leakage; it merely
    establishes the trivial lower bound that any trained model must beat.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        y_mean = batch.y.mean()
        flow_pred = torch.full_like(batch.y, fill_value=y_mean.item())
        
        # Keep gradient graph legally connected via dummy_param
        flow_pred = flow_pred + self.dummy_param * 0.0
        return flow_pred, batch.y


# ================================================================
# Baseline 2: Historical Average (Safe for Normalized Data)
# ================================================================

@register_network('capacity_proportional_baseline')  # 保持注册名不变，无需修改yaml
class CapacityProportionalBaseline(nn.Module):
    r"""
    Heuristic: retained edges keep their old flow; new edges get
    the mean flow of the retained edges.
    
    This replaces the capacity-ratio approach because dividing by 
    StandardScaled capacities (which contain negative numbers) is 
    mathematically invalid and causes numerical explosions.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.aligner = EdgeAlignmentModule()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        total_nodes: int = batch.num_nodes

        # Aligned features: [E_new, 8]
        aligned = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )

        flow_old    = aligned[:, 3]   # [E_new]
        is_new_edge = aligned[:, 7]   # [E_new]

        # 安全的浮点数比较
        retained_mask = (is_new_edge < 0.5)  # bool [E_new]

        # 计算保留边的历史流量均值作为新修边的预期值
        if retained_mask.any():
            mean_old_flow = flow_old[retained_mask].mean()
        else:
            mean_old_flow = torch.tensor(0.0, device=aligned.device, dtype=aligned.dtype)

        # Assemble predictions
        flow_pred = torch.empty(aligned.size(0), 1,
                                device=aligned.device, dtype=aligned.dtype)

        # Retained edges: directly reuse historical flow
        flow_pred[retained_mask, 0] = flow_old[retained_mask]
        
        # New edges: use the mean historical flow
        new_mask = ~retained_mask
        flow_pred[new_mask, 0] = mean_old_flow

        # Keep gradient graph legally connected via dummy_param
        flow_pred = flow_pred + self.dummy_param * 0.0
        return flow_pred, batch.y