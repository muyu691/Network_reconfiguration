"""
Phase 2: Dual Graph Neural Network —— NetworkPairsTopologyModel
======================================================

Core innovation: Predict new equilibrium flows (flows_new) after network topology reconfiguration using only historical flows (flows_old) as demand proxy, with no need for OD matrix.

Three main submodules:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Module 1: OldGraphEncoder                                       │
  │   Aggregates historical flows over the old graph G topology,    │
  │   outputs node-level "historical congestion memory"             │
  │   h_nodes_old : [N, H]                                          │
  ├─────────────────────────────────────────────────────────────────┤
  │ Module 2: EdgeAlignmentModule                                   │
  │   Fully vectorized (no for loops) matching of edges between G   │
  │   and G', producing an 8-dim aligned feature for each new edge  │
  │   aligned_features : [E_new, 8]                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ Module 3: NewGraphReasoner                                      │
  │   Fuses historical memory, reasons for new equilibrium flows    │
  │   over the topology of G'                                       │
  │   flow_pred : [E_new, 1]  (Unbounded, suitable for StandardScaler) │
  └─────────────────────────────────────────────────────────────────┘

Batch processing compatibility note:
  PyG's Batch.from_data_list() adds node offsets automatically to fields ending with 'index'
  (by __inc__ mechanism). Therefore, edge_index_old and
  edge_index_new remain globally correct node indices after batching.
  The Hash Key in EdgeAlignmentModule uses total_nodes (global node count)
  as encoding base to ensure key uniqueness across graphs.
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gatedgcn_layer import GatedGCNLayer


# ============================================================
# Lightweight GNN Batch container
# ============================================================

class _GNNBatch:
    """
    Minimalist data container—intended solely to drive GatedGCNLayer.forward().

    GatedGCNLayer.forward(batch) only accesses these three fields:
      batch.x          : [N, H]  node features
      batch.edge_attr  : [E, H]  edge features
      batch.edge_index : [2, E]  connectivity

    When GatedGCNLayer is initialized with equivstable_pe=False,
    code path `batch.pe_EquivStableLapPE` will never be executed
    (Python short-circuit evaluation), so this container does not need it.
    """

    __slots__ = ('x', 'edge_attr', 'edge_index')

    def __init__(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        self.x = x
        self.edge_attr = edge_attr
        self.edge_index = edge_index


# ============================================================
# Module 1: OldGraphEncoder (historical hub memory encoder)
# ============================================================

class OldGraphEncoder(nn.Module):
    """
    Encodes historical traffic congestion patterns on the old graph G.

    Design notes:
      - Nodes are initialized as "blank slate" ones(N, H), starting directly in hidden space,
        avoiding a meaningless 1 → H linear mapping.
      - Edge inputs = cat([edge_attr_old(3), flow_old(1)]) = [E_old, 4],
        then projected into hidden space and fed to the GatedGCN stack.
      - The gated aggregation of GatedGCN allows the model to distinguish high/low flow links,
        effectively learning "which nodes are congestion hubs."

    Tensor shape conventions (H = hidden_dim):
      Input:
        edge_index_old  : [2, E_old]
        edge_attr_old   : [E_old, 3]  - [capacity, speed, length] (normalized)
        flow_old        : [E_old, 1]  - historical equilibrium flows (normalized)
        num_nodes       : int         - batch total node count

      Output:
        h_nodes_old     : [num_nodes, H]  - node historical memory embedding
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Edge feature projection: [capacity, speed, length, flow_old] → hidden space
        # Shape change: [E_old, 4] → [E_old, H]
        # GatedGCNLayer requires node and edge features to have the same dimension (H)
        self.edge_proj = nn.Linear(4, hidden_dim)

        # Stacked GatedGCN layers: multi-round message passing on old graph topology
        # Each layer: in_dim = out_dim = H, keep dimensions invariant
        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        edge_index_old: torch.Tensor,   # [2, E_old]
        edge_attr_old: torch.Tensor,    # [E_old, 3]
        flow_old: torch.Tensor,         # [E_old, 1]
        num_nodes: int,
    ) -> torch.Tensor:                  # → [num_nodes, H]

        device = edge_attr_old.device
        dtype  = edge_attr_old.dtype

        # -- Blank node initialization -------------------------------------
        # Directly initialize in hidden space H, avoiding meaningless 1→H layer.
        # All nodes start in the same "no prior" state; history is injected solely
        # via message passing from edge flows.
        # Shape: [num_nodes, H]
        x = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # -- Edge feature projection ---------------------------------------
        # cat([edge_attr_old(3), flow_old(1)]) → [E_old, 4]
        # then project to hidden space      → [E_old, H]
        e_raw = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]
        e = self.edge_proj(e_raw)                              # [E_old, H]

        # -- Stacked message passing (old graph topology) -----------------
        for layer in self.gnn_layers:
            mini_batch = _GNNBatch(x, e, edge_index_old)
            mini_batch = layer(mini_batch)
            x = mini_batch.x           # [num_nodes, H]
            e = mini_batch.edge_attr   # [E_old, H]

        # Return node embedding as "historical congestion memory", discard edge embeddings
        return x   # h_nodes_old : [num_nodes, H]


# ============================================================
# Module 2: EdgeAlignmentModule (heterogeneous edge feature alignment)
# ============================================================

class EdgeAlignmentModule(nn.Module):
    """
    Vectorized heterogeneous edge feature alignment module (no learnable params).

    Core algorithm: O(E_old + E_new) vectorized matching based on node-pair hash.

    Alignment rules (strictly follow .cursorrules Phase 2 spec):
    ┌──────────────────────────────────────────────────────────────┐
    │ Retained edge (present in both G and G'):                   │
    │   [edge_attr_old(3), flow_old(1), edge_attr_new(3), 0(1)]   │
    │   → 8-dim total, is_new_edge = 0                            │
    ├──────────────────────────────────────────────────────────────┤
    │ Added/new edge (only in G'):                                │
    │   [zeros(3),         zero(1),    edge_attr_new(3), 1(1)]    │
    │   → 8-dim total, is_new_edge = 1                            │
    └──────────────────────────────────────────────────────────────┘

    Hash coding scheme:
      key(src, dst) = src × total_nodes + dst
      Since src, dst ∈ [0, total_nodes), key is unique for each node-pair.

    Batch correctness:
      PyG applies node offset to edge indices (__inc__) such that node IDs in different graphs
      do not overlap. Thus, using total_nodes (batch-global node count) as hash base, keys are unique.

    Tensor shape conventions (H = hidden_dim):
      Input:
        edge_index_old  : [2, E_old]
        edge_attr_old   : [E_old, 3]
        flow_old        : [E_old, 1]
        edge_index_new  : [2, E_new]
        edge_attr_new   : [E_new, 3]
        total_nodes     : int

      Output:
        aligned_features : [E_new, 8]
    """

    def forward(
        self,
        edge_index_old: torch.Tensor,   # [2, E_old]
        edge_attr_old: torch.Tensor,    # [E_old, 3]
        flow_old: torch.Tensor,         # [E_old, 1]
        edge_index_new: torch.Tensor,   # [2, E_new]
        edge_attr_new: torch.Tensor,    # [E_new, 3]
        total_nodes: int,
    ) -> torch.Tensor:                  # → [E_new, 8]

        device = edge_attr_new.device
        dtype  = edge_attr_new.dtype
        E_old  = edge_index_old.shape[1]
        E_new  = edge_index_new.shape[1]

        # -- Step 1: Encode edge node-pairs as unique integer keys ----------
        #
        # Formula: key(src, dst) = src × total_nodes + dst
        #
        # Rationale: Since 0 ≤ src, dst < total_nodes,
        #   any distinct (src, dst) produces distinct key—equiv. to
        #   flattening 2D index to 1D (row-major order).
        #
        # Time complexity: O(E_old + E_new), no Python for-loops.
        old_keys = edge_index_old[0] * total_nodes + edge_index_old[1]  # [E_old]
        new_keys = edge_index_new[0] * total_nodes + edge_index_new[1]  # [E_new]

        # -- Step 2: Build reverse lookup table key → old_edge_index -------
        #
        # key_to_old_idx[key] = index of this edge in old graph, -1 means not exists.
        # For Sioux Falls (24 nodes, batch size 64):
        #   total_nodes = 24 × 64 = 1536, max_key ≈ 2.36M, about 9MB RAM, feasible.
        max_key = total_nodes * total_nodes
        key_to_old_idx = torch.full(
            (max_key,), fill_value=-1, dtype=torch.long, device=device
        )

        old_edge_pos = torch.arange(E_old, dtype=torch.long, device=device)
        # Vectorized hash assignment: map each edge key in old graph to its row in edge_attr_old
        # (If duplicate edges exist, latter overwrite former; for directed graphs, shouldn't happen)
        key_to_old_idx[old_keys] = old_edge_pos                          # [max_key]

        # -- Step 3: For each new edge, lookup if it exists in old graph --
        #
        # match_idx[i] = j: edge_index_new[:, i] matches edge_attr_old[j]
        # match_idx[i] = -1: this is an added edge, does not exist in old graph
        match_idx = key_to_old_idx[new_keys]  # [E_new], range {-1, ..., E_old-1}

        # -- Step 4: Construct aligned features from old graph side (4 dims)
        #
        # Retained edge: use real old features + old flows
        # Added edge: fill zeros (representing "did not exist before")
        #
        # old_feats[j] = cat([edge_attr_old[j], flow_old[j]]) shape [4]
        old_feats = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]

        # Initialize zero matrix (default: all-added edges are zero from old side)
        aligned_old = torch.zeros(E_new, 4, dtype=dtype, device=device)

        # Boolean mask: mark which new edges exist in old graph (retained)
        retained_mask = match_idx >= 0  # [E_new], bool

        # Only fill for retained edges, avoid -1 index out of bound
        if retained_mask.any():
            # aligned_old[retained_mask] ← old_feats[match_idx[retained_mask]]
            # pure tensor indexing, no Python loops
            aligned_old[retained_mask] = old_feats[match_idx[retained_mask]]

        # -- Step 5: Build is_new_edge indicator ---------------------------
        #
        # is_new_edge = 1 marks new edge (not exist in old graph), model can distinguish
        # is_new_edge = 0 marks retained edge (exist in old graph)
        # Shape: [E_new, 1], float dtype for use in linear layers
        is_new_edge = (~retained_mask).to(dtype).unsqueeze(1)  # [E_new, 1]

        # -- Step 6: Concatenate final aligned feature ---------------------
        #
        # Retained edge: [edge_attr_old(3), flow_old(1), edge_attr_new(3), 0(1)] = 8
        # New edge:      [zeros(3),        zero(1),      edge_attr_new(3), 1(1)] = 8
        # Shape strictly [E_new, 8]
        aligned_features = torch.cat(
            [aligned_old, edge_attr_new, is_new_edge], dim=-1
        )  # [E_new, 8]

        return aligned_features


# ============================================================
# Implicit Demand Virtual Routing Layer (Implicit Virtual Edge Layer)
# ============================================================

class ImplicitVirtualRoutingLayer(nn.Module):
    r"""
    Implicit demand virtual routing layer -- uses global self-attention to break the locality limitation of GNN receptive field.

    Physical meaning:
      When a traffic network experiences disconnects or dramatic changes in attributes, flows get redistributed globally.
      Traditional GNN aggregates only 1-hop neighbors per layer, thus for flow transfer between distant OD pairs,
      many layers must be stacked to propagate the signal, but this can cause over-smoothing.

      This layer computes self-attention weights among all nodes within the same graph,
      equivalent to dynamically establishing "implicit virtual edges" (Implicit Virtual Links):
        - High attention score $\alpha_{ij}$ → strong association between node $i$ and $j$
          (potential OD relationship, alternative path relationship, flow conservation constraint relationship)
        - These virtual edges allow global flow redistribution signals to be transmitted in a single layer

      Mathematical expression:
        $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

      Architecture (Post-LN Transformer Block):
        $$\mathbf{x}_{\text{mid}} = \text{LN}\!\left(\mathbf{x} + \text{MHA}(\mathbf{x})\right)$$
        $$\mathbf{x}_{\text{out}} = \text{LN}\!\left(\mathbf{x}_{\text{mid}} + \text{FFN}(\mathbf{x}_{\text{mid}})\right)$$

    Batch safety (strictly prevents cross-graph contamination):
      PyG's to_dense_batch transforms flat [N, H] → [B, Max_N, H] dense tensors,
      together with key_padding_mask=~mask passed to MultiheadAttention, ensures:
        1. Padding positions (dummy nodes) do not participate in attention computation
        2. Nodes from different graphs will never communicate with each other
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head self-attention: establish global implicit virtual edges within the same graph
        # batch_first=True → input/output format [B, Seq, H], consistent with to_dense_batch output
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-Attention LayerNorm (normalize after residual fusion)
        self.norm1 = nn.LayerNorm(hidden_dim)
        # Post-FFN LayerNorm
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed Forward Network (FFN): two Linear + ReLU layers, standard 4x expansion ratio
        # Function: applies nonlinear transform to globally aggregated information from attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,      # [N, hidden_dim]  flat node features
        batch: torch.Tensor,   # [N]             node belong-to index
    ) -> torch.Tensor:         # → [N, hidden_dim]
        """
        Forward pass: global self-attention → residual + LayerNorm → FFN → residual + LayerNorm.

        Args:
            x     : Flat node features. Nodes of all graphs in batch concatenated together. Shape [N, H]
            batch : Node assignment indices; the i-th node belongs to batch[i]-th graph. Shape [N]

        Returns:
            x_flat : Node features with global perspective (contains implicit virtual edge information), shape [N, H]
        """
        # -- Step A: Dense transformation --------------------------------
        # Transform flat [N, H] into 3D dense tensor [B, Max_N, H]
        # mask: [B, Max_N], True = real node, False = padding (filled with 0)
        # Physical meaning: aligns node lists for each graph in mini-batch to same-length sequences
        #                   for Self-Attention matrix ops
        x_dense, mask = to_dense_batch(x, batch)  # [B, Max_N, H], [B, Max_N]

        # -- Step B: Global self-attention (establish implicit virtual edges) -----------
        # key_padding_mask=~mask: Positions with value True are ignored by attention
        #   → Padding nodes neither generate nor receive attention
        #   → Nodes in different graphs never communicate (because they're on different batch dimension)
        # Physical meaning: each node calculates attention with all other nodes in the same graph,
        #                   building feature similarity-based "implicit virtual edges",
        #                   so that potential OD and alternative path relations are captured instantly
        attn_out, _ = self.attn(
            query=x_dense,
            key=x_dense,
            value=x_dense,
            key_padding_mask=~mask,
        )  # [B, Max_N, H]

        # -- Step C: Residual connection + LayerNorm (Post-Attention) -----
        # $\mathbf{x}_{\text{mid}} = \text{LN}(\mathbf{x} + \text{MHA}(\mathbf{x}))$
        # Residual connection retains original local features, with attention output adding global info
        x_dense = self.norm1(x_dense + attn_out)  # [B, Max_N, H]

        # -- Step D: FFN + residual + LayerNorm ---------------------------
        # $\mathbf{x}_{\text{out}} = \text{LN}(\mathbf{x}_{\text{mid}} + \text{FFN}(\mathbf{x}_{\text{mid}}))$
        ffn_out = self.ffn(x_dense)                # [B, Max_N, H]
        x_dense = self.norm2(x_dense + ffn_out)    # [B, Max_N, H]

        # -- Step E: Restore as flat format -------------------------------
        # Use mask boolean indexing to select real nodes, filter out padding
        # The number of True in mask matches N (number of all real nodes), dimension matches strictly
        x_flat = x_dense[mask]  # [N, H]

        return x_flat


# ============================================================
# Module 3: NewGraphReasoner (reasoner for new topology)
# ============================================================

class NewGraphReasoner(nn.Module):
    """
    Fuses historical memory and reasons future equilibrium flows over new graph G'.

    Node Fusion principle (critical constraint):
      x_new_init = ones(N, H)           ← blank node for new graph
      x_fused    = cat([x_new_init, h_nodes_old])  → [N, 2H]
      x          = node_fusion(x_fused)            → [N, H]

      ⚠️  Strictly forbid residual connections after node_fusion!
          Residual x = x_fused_proj + h_nodes_old makes a "memory shortcut",
          letting the model copy old flows and bypass learning new topology.
          Force nonlinear compression (Linear→ReLU→Dropout) to ensure the model
          rediscover flow patterns by new graph message passing.

    Edge Decoder design:
      Input = cat([x_src(H), x_dst(H), aligned_features(8)]) → [2H + 8]
      Output = flow_pred [E_new, 1]
      Last layer is a pure linear, no activation (Unbounded Output).
      Target has been StandardScaler-normalized (mean 0, std 1, can be negative),
      any bounded activation (like ReLU/Sigmoid) introduces distribution bias.

    Tensor shape conventions (H = hidden_dim):
      Input:
        edge_index_new   : [2, E_new]
        aligned_features : [E_new, 8]
        h_nodes_old      : [num_nodes, H]
        num_nodes        : int

      Output:
        flow_pred : [E_new, 1]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        residual: bool,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # -- Node Fusion layer --------------------------------------------
        # Input: cat([x_new_init(H), h_nodes_old(H)]) → [N, 2H]
        # Output: [N, H]
        #
        # Principle: Nonlinear compression forces a dynamic balance between
        # the "blank" (for new topology adaptation) and "historical memory" (old flow pattern).
        # Strictly no residual — avoids degenerate strategies that just outputs old flows.
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -- Implicit Demand Virtual Routing Layer (Implicit Virtual Routing) -------------
        # After node_fusion output and before GatedGCN local message passing,
        # inject full-graph perspective to each node via global self-attention,
        # overcoming GNN shortsightedness, allowing far-end flow redistribution signals
        # to propagate instantly.
        # Controlled by cfg.dataset.use_virtual_links (default True).
        self.use_virtual_links = cfg.dataset.use_virtual_links
        if self.use_virtual_links:
            self.virtual_routing = ImplicitVirtualRoutingLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Project the aligned edge features: [E_new, 8] → [E_new, H]
        # GatedGCNLayer requires same dimension for node and edge features
        self.edge_proj = nn.Linear(8, hidden_dim)

        # Stacked GatedGCN layers: multi-round message passing on new graph topology
        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

        # -- Edge-level flow decoder --------------------------------------
        # Input: cat([x_src(H), x_dst(H), aligned_features(8)]) → [2H + 8]
        #
        # Principle: Explicitly fuses "source node state + target node state + edge features";
        # better captures directional flow laws than decoding from edge only.
        # aligned_features retains original 8 dims (including is_new_edge bit);
        # allows decoder to distinguish new/retained edges and predict them differently.
        #
        # Last layer: nn.Linear(H, 1), no activation (Unbounded Output).
        decoder_in_dim = hidden_dim * 2 + 8
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # ⚠️  Intentionally omit activation here!
            # StandardScaler-processed targets span (−∞, +∞),
            # any bounded activation (ReLU/Tanh/Sigmoid) will bias predictions.
        )

    def forward(
        self,
        edge_index_new: torch.Tensor,    # [2, E_new]
        aligned_features: torch.Tensor,  # [E_new, 8]
        h_nodes_old: torch.Tensor,       # [num_nodes, H]
        num_nodes: int,
        batch_vec: torch.Tensor,         # [num_nodes] node assignment index (must be provided by PyG Batch)
    ) -> torch.Tensor:                   # → [E_new, 1]

        device = h_nodes_old.device
        dtype  = h_nodes_old.dtype

        # -- Blank node initialization (new graph) ------------------------
        # x_new_init represents "no prior knowledge" for new graph.
        # After fusing with h_nodes_old, model must (re)assign flows by new topology,
        # not just inherit old states.
        # Shape: [num_nodes, H]
        x_new_init = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # -- Historical memory fusion (Node Fusion) ------------------------
        # Concatenate blank node with historical embedding: [N, H] || [N, H] → [N, 2H]
        # Then nonlinearly compress: [N, 2H] → [N, H]
        #
        # ⚠️  No residual connection! x must be entirely determined by node_fusion nonlinear transform,
        #     h_nodes_old can't shortcut to downstream network.
        x_cat = torch.cat([x_new_init, h_nodes_old], dim=-1)  # [N, 2H]
        x = self.node_fusion(x_cat)                            # [N, H]

        # -- Implicit Demand Virtual Routing Layer (Implicit Virtual Routing) -------------------
        if self.use_virtual_links:
            assert batch_vec is not None, (
                "batch_vec is None! ImplicitVirtualRoutingLayer requires PyG's batch.batch vector."
            )
            assert batch_vec.shape[0] == num_nodes, (
                f"batch_vec length ({batch_vec.shape[0]}) does not match num_nodes ({num_nodes})!"
            )
            x = self.virtual_routing(x, batch_vec)  # [N, H], global implicit OD information fused

        # -- Aligned edge feature projection -------------------------------
        # Project 8-dim aligned feature into hidden space for GatedGCNLayer
        # Shape: [E_new, 8] → [E_new, H]
        e = self.edge_proj(aligned_features)  # [E_new, H]

        # -- Stacked message passing over new graph topology ---------------
        # This is where model "understands" new topology:
        #   - Added edges (is_new_edge=1) can propagate their physical attrs to neighbors
        #   - Deleted edges disappear, flow redistributes over retained paths
        for layer in self.gnn_layers:
            mini_batch = _GNNBatch(x, e, edge_index_new)
            mini_batch = layer(mini_batch)
            x = mini_batch.x           # [num_nodes, H]
            e = mini_batch.edge_attr   # [E_new, H]

        # -- Edge-level flow decoding --------------------------------------
        # For each new edge, extract src/dst node embeddings,
        # concat with raw 8-dim aligned features (not projected), feed to decoder.
        #
        # Keep original aligned_features (not projected/e),
        #   1. aligned_features contains is_new_edge indicator (explicitly shows new/retained)
        #   2. Raw physical attrs (capacity, speed, length) have flow-related dimension,
        #      directly feed un-distorted.
        src_idx, dst_idx = edge_index_new[0], edge_index_new[1]
        edge_repr = torch.cat(
            [x[src_idx], x[dst_idx], aligned_features], dim=-1
        )  # [E_new, 2H + 8]

        flow_pred = self.edge_decoder(edge_repr)  # [E_new, 1]

        return flow_pred


# ============================================================
# Main model: NetworkPairsTopologyModel
# ============================================================

@register_network('topology_gnn')
class NetworkPairsTopologyModel(nn.Module):
    """
    Main dual graph neural network model for traffic network topology reconfiguration.

    Strictly enforces project constraints:
      - No OD matrix input (batch.x contains only all-ones placeholder, not used by model)
      - flows_old as proxy for demand (historical flow is a key model input)
      - Final prediction is unbounded real number in StandardScaler space; loss is computed externally

    Expected PyG Batch fields (generated by Phase 1 build_network_pairs_dataset.py):
      batch.edge_index_old : [2, E_old_total]   old graph connectivity (node offset done)
      batch.edge_attr_old  : [E_old_total, 3]   old graph physical attrs (normalized)
      batch.flow_old       : [E_old_total, 1]   old graph flows (normalized)
      batch.edge_index_new : [2, E_new_total]   new graph connectivity (node offset done)
      batch.edge_attr_new  : [E_new_total, 3]   new graph physical attrs (normalized)
      batch.y              : [E_new_total, 1]   new graph eq. flows (normalized, ground truth)
      batch.num_nodes      : int                total node count (PyG autocompute sum)

    Args:
        dim_in  : placeholder (for GraphGym API), not used internally
        dim_out : placeholder (for GraphGym API), output dim always 1 by design
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        # Read all hyperparams from cfg.topology_gnn
        # Config file: graphgps/config/topology_gnn_config.py
        hidden_dim     = cfg.topology_gnn.hidden_dim
        num_layers_old = cfg.topology_gnn.num_layers_old
        num_layers_new = cfg.topology_gnn.num_layers_new
        dropout        = cfg.topology_gnn.dropout
        residual       = cfg.topology_gnn.residual

        self.encoder  = OldGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers_old,
            dropout=dropout,
            residual=residual,
        )
        self.aligner  = EdgeAlignmentModule()

        num_heads       = cfg.topology_gnn.num_heads

        self.reasoner = NewGraphReasoner(
            hidden_dim=hidden_dim,
            num_layers=num_layers_new,
            dropout=dropout,
            residual=residual,
            num_heads=num_heads,
        )
        def replace_bn_with_ln(module):
            for name, child in module.named_children():
                # If finds layer containing 'BatchNorm' in name
                if 'BatchNorm' in child.__class__.__name__:
                    # Replace with LayerNorm
                    setattr(module, name, nn.LayerNorm(child.num_features))
                else:
                    # Recursively process child modules
                    replace_bn_with_ln(child)

        replace_bn_with_ln(self)

    def forward(self, batch):
        """
        Full forward pass.

        Pipeline:
          1. OldGraphEncoder     → h_nodes_old : [N_total, H]
          2. EdgeAlignmentModule → aligned_features : [E_new_total, 8]
          3. NewGraphReasoner    → flow_pred : [E_new_total, 1]

        Args:
            batch : PyG Batch object (contains a mini-batch of (G, G') pairs)

        Returns:
            pred : torch.Tensor [E_new_total, 1]  predicted flows (in normalized space)
            true : torch.Tensor [E_new_total, 1]  true flows (in normalized space)
        """
        # PyG automatically sums num_nodes across all graphs in the batch,
        # resulting in current batch's global node count (for GNN aggregation & hash matching)
        total_nodes: int = batch.num_nodes

        # -- Module 1: Historical congestion memory encoding on old graph -----
        # Output: h_nodes_old [total_nodes, H]
        h_nodes_old = self.encoder(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            num_nodes=total_nodes,
        )

        # -- Module 2: Vectorized alignment of new/old edge features ----------
        # Output: aligned_features [E_new_total, 8]
        aligned_features = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )

        # -- Module 3: Fusing history, reasoning flows on new graph -----------
        # Output: flow_pred [E_new_total, 1]
        # batch.batch: [N_total] node assignment index, passed to ImplicitVirtualRoutingLayer
        #   Ensures global self-attention only occurs between nodes in the same graph
        flow_pred = self.reasoner(
            edge_index_new=batch.edge_index_new,
            aligned_features=aligned_features,
            h_nodes_old=h_nodes_old,
            num_nodes=total_nodes,
            batch_vec=batch.batch,
        )

        return flow_pred, batch.y
