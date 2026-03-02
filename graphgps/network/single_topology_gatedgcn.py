"""
Single-Topology GatedGCN Baseline
==================================

Ablation model that performs message passing ONLY on the new graph G'
(edge_index_new) without any dual-graph encoding of the old topology.

Purpose: Demonstrates that without explicitly encoding historical
congestion patterns from the old graph G, a single-topology GNN
cannot fully capture global flow redistribution effects.

Pipeline:
  EdgeAlignmentModule → aligned_features [E_new, 8]
  node_enc(batch.x)   → x [N, H]
  edge_enc(aligned)    → e [E_new, H]
  GatedGCN stack on edge_index_new → updated x, e
  edge_decoder(e)      → flow_pred [E_new, 1]
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.network.topology_model import EdgeAlignmentModule


class _GNNBatch:
    __slots__ = ('x', 'edge_attr', 'edge_index', 'num_nodes', 'batch')

    def __init__(self, x, edge_attr, edge_index, num_nodes=None, batch_idx=None):
        self.x = x
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch = batch_idx


@register_network('single_topology_gatedgcn')
class SingleTopologyGatedGCN(nn.Module):

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        H = cfg.gnn.dim_inner
        num_layers = cfg.topology_gnn.num_layers_new
        dropout = cfg.topology_gnn.dropout
        residual = cfg.topology_gnn.residual

        self.aligner = EdgeAlignmentModule()

        self.node_enc = nn.Linear(1, H)
        self.edge_enc = nn.Linear(8, H)

        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(
                in_dim=H, out_dim=H,
                dropout=dropout, residual=residual,
            )
            for _ in range(num_layers)
        ])

        self.edge_decoder = nn.Sequential(
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
        )

        # Replace any BatchNorm injected by GatedGCNLayer with LayerNorm
        def _replace_bn(module):
            for name, child in module.named_children():
                if 'BatchNorm' in child.__class__.__name__:
                    setattr(module, name, nn.LayerNorm(child.num_features))
                else:
                    _replace_bn(child)
        _replace_bn(self)

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

        x = self.node_enc(batch.x)           # [N, H]
        e = self.edge_enc(aligned_features)   # [E_new, H]

        for layer in self.gnn_layers:
            mini = _GNNBatch(x, e,
            batch.edge_index_new, 
            num_nodes=total_nodes, 
            batch_idx=batch.batch
            )
            mini = layer(mini)
            x = mini.x
            e = mini.edge_attr

        flow_pred = self.edge_decoder(e)  # [E_new, 1]
        return flow_pred, batch.y
