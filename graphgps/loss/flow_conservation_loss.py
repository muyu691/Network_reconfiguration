"""
Phase 3: Physics-Informed Neural Network (PINN) Conservation Loss
=================================================================

Implements the Global Flow Conservation Law (Kirchhoff's First Law in Traffic Networks):

  For non-centroid nodes (pure intersections, nodes 12-24 in Sioux Falls, 0-indexed 11-23):
      sum(inflow(v)) - sum(outflow(v)) = 0

  For centroid nodes (OD zones, nodes 1-11 in Sioux Falls, 0-indexed 0-10):
      sum(inflow(i)) - sum(outflow(i)) = Delta_i
  where Delta_i = dest_demand(i) - origin_demand(i)  (derived from OD matrix, NOT a model input)

  Unified formula covering all nodes:
      error(v) = [sum(inflow(v)) - sum(outflow(v))] - Delta_v
  Since Delta_v = 0 for non-centroid nodes, the same expression covers both types.

This file contains:
  FlowConservationLoss    Conservation penalty (pure physics, reusable)
  CombinedPINNLoss        Full combination: supervised loss + lambda x conservation loss
  compute_pinn_loss()     Top-level entry, called by custom_train.py (Phase 4)

Design notes:
  The standard GraphGPS @register_loss interface only accepts (pred, true) and cannot carry
  batch context. Because conservation loss needs batch.edge_index_new and batch.net_demand,
  this file uses a standalone nn.Module + top-level function design that is called directly
  as compute_pinn_loss(pred, batch) from the modified custom_train.py.

Configuration dependencies (must be set before running):
  cfg.dataset.flow_mean   : float  - StandardScaler mean  (loaded from flow_scaler.pkl)
  cfg.dataset.flow_std    : float  - StandardScaler std   (loaded from flow_scaler.pkl)
  cfg.model.lambda_cons   : float  - Conservation loss weight lambda (default 0.1)
  cfg.model.cons_norm     : str    - Conservation error norm, 'l2' (MSE) or 'l1' (MAE)
  cfg.model.loss_fun      : str    - Supervised loss type, 'l1' or 'mse' (default 'l1')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg


# ============================================================
# Core Module 1: FlowConservationLoss (Global Conservation Penalty)
# ============================================================

class FlowConservationLoss(nn.Module):
    """
    Computes the global flow conservation penalty for ALL nodes.

    Extended Global Conservation Constraint
    ----------------------------------------
    The constraint is unified across all node types via a single formula:

        error(v) = [sum(inflow(v)) - sum(outflow(v))] - Delta_v

    where Delta_v is the real-space net demand for node v (unit: vehicles/hour):
      - Non-centroid nodes (0-indexed 11-23):  Delta_v = 0  (pure intersections)
      - Centroid nodes     (0-indexed 0-10):   Delta_v = dest_demand - origin_demand

    Since Delta_v = 0 for non-centroid nodes, the formula naturally reduces to the
    original Kirchhoff constraint (sum inflow = sum outflow) for those nodes.

    Why denormalize BEFORE computing the error?
    -------------------------------------------
    With StandardScaler: f_hat = (f - mu) / sigma.

    If the centroid constraint were checked in normalized space:
        sum(f_hat_in) - sum(f_hat_out)  =?  Delta_hat_v

    After substitution the right-hand side becomes:
        Delta_hat_v + (d_in - d_out) * mu / sigma

    For d_in != d_out and mu != 0 this is degree-dependent and physically incorrect.

    Correct three-step approach:
      1. Denormalize:    real_flow = pred_scaled * sigma + mu          (veh/hr)
      2. Compute error:  error(v) = [sum_in(v) - sum_out(v)] - Delta_v (veh/hr)
         Delta_v is already in real flow space -- direct subtraction is valid.
      3. Scale back:     error_scaled = error / sigma                  (dimensionless)
         Brings L_cons to the same O(1) scale as L_sup, preventing gradient imbalance.

    Tensor shapes:
      pred           : [E_new, 1]   model output (StandardScaler space)
      edge_index_new : [2, E_new]   new graph connectivity (batch node offsets applied)
      net_demand     : [N_total]    float32, real-space Delta_v per node (veh/hr)
                                    After PyG batching: shape [24 * batch_size]
      num_nodes      : int          total nodes in the batch
      flow_mean      : float        StandardScaler mean mu
      flow_std       : float        StandardScaler std sigma

      returns -> scalar  (dimensionless conservation penalty)
    """

    def forward(
        self,
        pred: torch.Tensor,           # [E_new, 1]
        edge_index_new: torch.Tensor, # [2, E_new]
        net_demand: torch.Tensor,     # [N_total], float32, real-space Delta_v (veh/hr)
        num_nodes: int,
        flow_mean: float,
        flow_std: float,
    ) -> torch.Tensor:                # -> scalar

        # Step 1: Denormalization -------------------------------------------------
        # real_flow = pred_scaled * sigma + mu
        # flow_mean / flow_std are Python floats, auto-broadcast to [E_new, 1].
        # Linear transform is fully differentiable; gradients are unaffected.
        real_flow = pred * flow_std + flow_mean  # [E_new, 1]

        # Flatten to 1D for scatter_add
        real_flow_flat = real_flow.squeeze(-1)   # [E_new]

        # Step 2: Vectorized aggregation (scatter_add) ----------------------------
        # For each directed edge (src -> dst):
        #   outflow[src] += real_flow_flat[e]
        #   inflow[dst]  += real_flow_flat[e]
        # O(E_new), no Python loops.

        src_nodes = edge_index_new[0]  # [E_new]
        dst_nodes = edge_index_new[1]  # [E_new]

        device = real_flow_flat.device
        dtype  = real_flow_flat.dtype

        outflow = torch.zeros(num_nodes, device=device, dtype=dtype)
        inflow  = torch.zeros(num_nodes, device=device, dtype=dtype)

        outflow.scatter_add_(0, src_nodes, real_flow_flat)  # [N_total]
        inflow.scatter_add_(0, dst_nodes, real_flow_flat)   # [N_total]

        # Step 3: Global conservation error ---------------------------------------
        # Unified formula: error(v) = [inflow(v) - outflow(v)] - Delta_v
        #
        # net_demand shape [N_total] aligns with inflow/outflow [N_total] after batching.
        # Cast net_demand to match real_flow dtype (float32 -> float32 typically).
        # Subtraction is done BEFORE dividing by sigma to maintain dimensional consistency.
        diff = inflow - outflow                           # [N_total], veh/hr
        net_demand_dev = net_demand.to(device=device, dtype=dtype)
        error = diff - net_demand_dev                    # [N_total], veh/hr
        # For non-centroid nodes: net_demand = 0 => error = inflow - outflow  (original constraint)
        # For centroid nodes:     net_demand = Delta_v => error = inflow - outflow - Delta_v

        # Step 4: Dimensionless scaling (scale alignment) -------------------------
        # Divide by sigma AFTER computing (diff - net_demand).
        # Both terms are in veh/hr, so subtraction is valid; dividing by sigma maps
        # the error to dimensionless units comparable to L_sup ~ O(1).
        # Dividing by a positive scalar preserves gradients.
        error_scaled = error / flow_std   # [N_total], dimensionless

        # Step 5: Conservation penalty (L1 or L2 over ALL nodes) ------------------
        # Average over all N_total nodes (centroid + non-centroid).
        if cfg.model.cons_norm == 'l1':
            loss_cons = error_scaled.abs().mean()
        else:
            loss_cons = (error_scaled ** 2).mean()

        return loss_cons


# ============================================================
# Core Module 2: CombinedPINNLoss (Supervised Loss + Conservation Loss)
# ============================================================

class CombinedPINNLoss(nn.Module):
    """
    Combined loss: L_total = L_sup + lambda * L_cons

    Where:
      L_sup  = L1Loss or MSELoss(pred, y)   computed in normalized space
      L_cons = global conservation penalty over ALL nodes
               computed in real flow space (denormalized), then scaled back

    Three-step design rationale:
      1. L_sup in normalized space: pred, y ~ N(0,1), residues O(1).
      2. L_cons denormalizes to real space first: eliminates (d_in - d_out)*mu drift
         that would otherwise make the constraint physically wrong for d_in != d_out.
      3. L_cons divided by sigma: maps veh/hr error to dimensionless O(1),
         preventing gradient magnitude imbalance between L_sup and L_cons.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conservation = FlowConservationLoss()

    def forward(
        self,
        pred: torch.Tensor,  # [E_new, 1]  model output (normalized space)
        batch,               # PyG Batch object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss : scalar        weighted total loss (for backward)
            pred       : [E_new, 1]   predictions (for logger metrics)
        """
        true = batch.y  # [E_new, 1], normalized Ground Truth

        # Supervised loss (normalized space) -------------------------------------
        if cfg.model.loss_fun == 'mse':
            loss_sup = F.mse_loss(pred, true)
        else:
            loss_sup = F.l1_loss(pred, true)

        # Conservation loss (real flow space) ------------------------------------
        # flow_mean and flow_std must be set in cfg before training
        # (loaded from scalers/flow_scaler.pkl in the training entry point)
        flow_mean = cfg.dataset.flow_mean  # mu
        flow_std  = cfg.dataset.flow_std   # sigma

        # batch.net_demand: shape [N_total] after PyG collate (24 * batch_size)
        # Directly aligned with scatter_add outputs inside FlowConservationLoss.
        loss_cons = self.conservation(
            pred=pred,
            edge_index_new=batch.edge_index_new,
            net_demand=batch.net_demand,
            num_nodes=batch.num_nodes,
            flow_mean=flow_mean,
            flow_std=flow_std,
        )

        # Weighted sum -----------------------------------------------------------
        # L_total = L_sup + lambda * L_cons
        lam = cfg.model.lambda_cons
        total_loss = loss_sup + lam * loss_cons

        return total_loss, pred


# ============================================================
# Top-level function: compute_pinn_loss (entry for custom_train.py)
# ============================================================

# Module-level singleton: avoids re-instantiation on every call (no learnable params)
_pinn_loss_fn = CombinedPINNLoss()


def compute_pinn_loss(
    pred: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Entry point for Phase 4: replaces compute_loss(pred, true) in custom_train.py.

    Usage (in custom_train.py):

        from graphgps.loss.flow_conservation_loss import compute_pinn_loss

        # Replace:
        #   loss, pred_score = compute_loss(pred, true)
        # With:
        #   loss, pred_score = compute_pinn_loss(pred, batch)

    Args:
        pred  : [E_new, 1]  model predictions (from NetworkPairsTopologyModel.forward)
        batch : PyG Batch object, must contain:
                  batch.y              [E_new, 1]   normalized Ground Truth
                  batch.edge_index_new [2, E_new]   new graph topology
                  batch.net_demand     [N_total]    real-space Delta_v per node (veh/hr)
                  batch.num_nodes      int           total batch node count

    Returns:
        total_loss  : scalar        weighted total loss for backprop
        pred_score  : [E_new, 1]   predictions (for logger MAE/MSE metrics)
    """
    return _pinn_loss_fn(pred, batch)
