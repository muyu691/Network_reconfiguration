"""
MLP Baseline — Edge-level MLP without graph topology message passing.

Architecture:
  EdgeAlignmentModule → aligned_features [E_new, 8]
  → Linear(8, H) → ReLU → Linear(H, H) → ReLU → Linear(H, 1)
  → flow_pred [E_new, 1]
"""

import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.network.topology_model import EdgeAlignmentModule


@register_network('mlp_baseline')
class MLPBaseline(nn.Module):

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.aligner = EdgeAlignmentModule()

        h = cfg.gnn.dim_inner
        drop_rate = cfg.gnn.dropout

        self.mlp = nn.Sequential(
            nn.Linear(8, h),
            nn.LayerNorm(h),          
            nn.ReLU(),
            nn.Dropout(drop_rate),  
            
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(h, 1)
        )

    def forward(self, batch):
        total_nodes: int = batch.num_nodes

        aligned_features = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )

        flow_pred = self.mlp(aligned_features)  # [E_new, 1]
        return flow_pred, batch.y
