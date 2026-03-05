"""
Ablation Baseline: Node-Centric GNN (SAGEConv)
================================================

Replaces the edge-centric GatedGCN with a standard node-aggregation GNN
(GraphSAGE) to demonstrate the necessity of explicit edge feature updates.

Pipeline:
  OldGraphEncoder (SAGEConv)  → h_nodes_old [N, H]
  EdgeAlignmentModule         → aligned_features [E_new, 8]
  NewGraphReasoner (SAGEConv) → updated x [N, H]
  EdgeDecoder:  [x_src || x_dst || aligned_features] → flow_pred [E_new, 1]

Key difference from the main model:
  - SAGEConv aggregates neighbour *node* features only; edge attributes
    are reduced to a scalar edge_weight via an MLP, losing rich edge info.
  - No explicit edge embedding update per layer (edge-centric property lost).
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_batch

from graphgps.network.topology_model import EdgeAlignmentModule


class NodeCentricOldEncoder(nn.Module):
    """Encode old graph using SAGEConv (node-centric aggregation)."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_to_weight = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index_old, edge_attr_old, flow_old, num_nodes):
        device = edge_attr_old.device
        dtype = edge_attr_old.dtype
        x = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        e_raw = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]
        edge_weight = self.edge_to_weight(e_raw).squeeze(-1)  # [E_old]

        for layer, norm in zip(self.layers, self.norms):
            x_res = x
            x = layer(x, edge_index_old)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + x_res
        return x


class NodeCentricNewReasoner(nn.Module):
    """Reason on new graph using SAGEConv + edge decoder."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_to_weight = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layer = nn.Dropout(dropout)

        decoder_in_dim = hidden_dim * 2 + 8
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_index_new, aligned_features, h_nodes_old, num_nodes):
        device = h_nodes_old.device
        dtype = h_nodes_old.dtype

        x_init = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)
        x = self.node_fusion(torch.cat([x_init, h_nodes_old], dim=-1))

        edge_weight = self.edge_to_weight(aligned_features).squeeze(-1)

        for layer, norm in zip(self.layers, self.norms):
            x_res = x
            x = layer(x, edge_index_new)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout_layer(x)
            x = x + x_res

        src_idx, dst_idx = edge_index_new[0], edge_index_new[1]
        edge_repr = torch.cat([x[src_idx], x[dst_idx], aligned_features], dim=-1)
        flow_pred = self.edge_decoder(edge_repr)
        return flow_pred


@register_network('NodeCentricGNN')
class NodeCentricGNN(nn.Module):
    """
    Ablation baseline replacing GatedGCN with standard SAGEConv.

    Demonstrates that node-only aggregation without explicit edge feature
    updates is insufficient for traffic flow prediction tasks.
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        hidden_dim = cfg.topology_gnn.hidden_dim
        num_layers_old = cfg.topology_gnn.num_layers_old
        num_layers_new = cfg.topology_gnn.num_layers_new
        dropout = cfg.topology_gnn.dropout

        self.encoder = NodeCentricOldEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers_old,
            dropout=dropout,
        )
        self.aligner = EdgeAlignmentModule()
        self.reasoner = NodeCentricNewReasoner(
            hidden_dim=hidden_dim,
            num_layers=num_layers_new,
            dropout=dropout,
        )

    def forward(self, batch):
        total_nodes: int = batch.num_nodes

        h_nodes_old = self.encoder(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            num_nodes=total_nodes,
        )

        aligned_features = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )

        flow_pred = self.reasoner(
            edge_index_new=batch.edge_index_new,
            aligned_features=aligned_features,
            h_nodes_old=h_nodes_old,
            num_nodes=total_nodes,
        )

        return flow_pred, batch.y
