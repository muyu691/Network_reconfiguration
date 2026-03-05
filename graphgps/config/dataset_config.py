from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    # ── Phase 3: Flow de-normalization parameters ──────────────────────────────────────────
    # These two values must be loaded from the scaler generated in Phase 1 and written to cfg before training starts.
    # De-normalization formula: real_flow = pred_scaled * flow_std + flow_mean
    # Loading method (execute in master_loader.py or at the training entry point):
    #   import pickle
    #   with open('processed_data/pyg_dataset/scalers/flow_scaler.pkl', 'rb') as f:
    #       scaler = pickle.load(f)
    #   cfg.dataset.flow_mean = float(scaler.mean_[0])
    #   cfg.dataset.flow_std  = float(scaler.scale_[0])
    cfg.dataset.flow_mean = 0.0   # Placeholder, must be overwritten with real value before run
    cfg.dataset.flow_std  = 1.0   # Placeholder, must be overwritten with real value before run

    # ── Ablation Study Flags ──────────────────────────────────────────────────
    # Whether to use Implicit Virtual Routing (global self-attention) in NewGraphReasoner.
    # Set to False to ablate the contribution of implicit virtual links.
    cfg.dataset.use_virtual_links = True

    # Whether to zero-out Capacity column in edge_attr (feature importance ablation).
    cfg.dataset.mask_capacity = False

    # Whether to zero-out Free-Flow Time column in edge_attr (feature importance ablation).
    cfg.dataset.mask_fft = False
