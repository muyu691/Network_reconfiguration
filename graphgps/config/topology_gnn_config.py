"""
Phase 2 config: NetworkPairsTopologyModel hyperparameter registration

All parameters are mounted under the cfg.topology_gnn namespace,
and can be overridden via YAML config file or command line.
"""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('topology_gnn')
def topology_gnn_cfg(cfg):
    """
    Register dedicated config group for NetworkPairsTopologyModel.

    Example YAML config snippet:
      topology_gnn:
        hidden_dim: 128
        num_layers_old: 3
        num_layers_new: 3
        dropout: 0.1
        residual: true
    """
    cfg.topology_gnn = CN()

    # Unified hidden dimension for all GNN layers (OldGraphEncoder + NewGraphReasoner)
    # Node and edge embeddings are within this space
    cfg.topology_gnn.hidden_dim = 128

    # Number of stacked GatedGCN layers in OldGraphEncoder
    cfg.topology_gnn.num_layers_old = 3

    # Number of stacked GatedGCN layers in NewGraphReasoner
    cfg.topology_gnn.num_layers_new = 3

    # Dropout probability, applied within GatedGCN layers and node_fusion / edge_decoder
    cfg.topology_gnn.dropout = 0.1

    # Whether to use residual connections in GatedGCN layers
    # Note: This residual refers to the skip connection within GatedGCN layers themselves,
    #       and is unrelated to the prohibition of residuals in the node_fusion layer of NewGraphReasoner
    cfg.topology_gnn.residual = True

    # Number of attention heads in ImplicitVirtualRoutingLayer
    # Controls the granularity of implicit demand virtual links:
    #   more heads → model can capture more diverse OD/rerouting patterns in parallel
    # Must satisfy: hidden_dim % num_heads == 0
    cfg.topology_gnn.num_heads = 4
