"""
Edge Regression Head for traffic flow prediction.

This head is designed for edge-level regression tasks where we predict
a continuous value for each edge in the graph (e.g., traffic flow).
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


@register_head('edge_regression')
class EdgeRegressionHead(nn.Module):
    """
    GNN prediction head for edge-level regression tasks.
    
    This head predicts a continuous value for each edge based on the 
    concatenation or interaction of source and target node embeddings.
    
    Args:
        dim_in (int): Input dimension (node embedding dimension)
        dim_out (int): Output dimension (1 for single value regression)
    """
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Edge decoding strategy from config
        # 'concat': concatenate source and target node embeddings + edge features
        # 'dot': dot product of source and target node embeddings
        # 'add': element-wise addition
        
        if cfg.model.edge_decoding == 'concat':
            # Concatenate source node, target node, AND edge embeddings
            # Following paper equation (8): z_ij = CONCAT(h_i^(L), h_j^(L), e_ij)
            # Input: [batch_edges, dim_in * 2 + edge_dim]
            # Output: [batch_edges, dim_out]
            edge_dim = cfg.dataset.edge_dim if cfg.dataset.edge_encoder else 0
            concat_dim = dim_in * 2 + edge_dim
            self.layer_post_mp = MLP(
                new_layer_config(concat_dim, dim_out, cfg.gnn.layers_post_mp,
                                 has_act=False, has_bias=True, cfg=cfg))
            # Note: decode_module now takes 3 inputs (v1, v2, edge_feat)
            self.decode_module = lambda v1, v2, e: \
                self.layer_post_mp(torch.cat((v1, v2, e), dim=-1))
                
        elif cfg.model.edge_decoding == 'dot':
            # Project node embeddings then compute dot product
            # Note: 'dot' mode doesn't use edge features
            self.layer_post_mp = MLP(
                new_layer_config(dim_in, dim_in, cfg.gnn.layers_post_mp,
                                 has_act=False, has_bias=True, cfg=cfg))
            # Add a final linear layer to map dot product to output dimension
            self.final_layer = nn.Linear(1, dim_out) if dim_out > 1 else nn.Identity()
            self.decode_module = lambda v1, v2, e=None: self.final_layer(
                torch.sum(v1 * v2, dim=-1, keepdim=True))
                
        elif cfg.model.edge_decoding == 'add':
            # Element-wise addition followed by MLP
            # Note: 'add' mode doesn't use edge features
            self.layer_post_mp = MLP(
                new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                                 has_act=False, has_bias=True, cfg=cfg))
            self.decode_module = lambda v1, v2, e=None: \
                self.layer_post_mp(v1 + v2)
                
        else:
            raise ValueError(
                f'Unknown edge decoding method: {cfg.model.edge_decoding}. '
                'Supported: concat, dot, add')
    
    def forward(self, batch):
        """
        Forward pass for edge regression.
        
        Args:
            batch: PyG Batch object containing:
                - x: node embeddings [num_nodes, dim_in]
                - edge_index: edge connectivity [2, num_edges]
                - edge_attr: edge features/embeddings [num_edges, edge_dim]
                - y: edge labels (ground truth flows) [num_edges]
                
        Returns:
            pred: predictions [num_edges, dim_out]
            label: ground truth [num_edges, dim_out] or [num_edges]
        """
        # Apply post-MP layers if needed (for non-concat)
        if cfg.model.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch)
        
        # Get source and target node embeddings for each edge
        # edge_index[0]: source nodes
        # edge_index[1]: target nodes
        src_nodes = batch.x[batch.edge_index[0]]  # [num_edges, dim_in]
        tgt_nodes = batch.x[batch.edge_index[1]]  # [num_edges, dim_in]
        
        # Decode edge predictions
        if cfg.model.edge_decoding == 'concat':
            # Following paper equation (8-9): include edge features in concatenation
            edge_feats = batch.edge_attr  # [num_edges, edge_dim]
            pred = self.decode_module(src_nodes, tgt_nodes, edge_feats)
        else:
            # For 'dot' and 'add' modes, edge features are not used
            pred = self.decode_module(src_nodes, tgt_nodes, None)
        
        # Get labels
        label = batch.y
        
        # Ensure dimensions match
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)  # [num_edges]
        if label.dim() == 1 and pred.dim() == 2:
            label = label.unsqueeze(1)  # [num_edges, 1]
            
        return pred, label
