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

    # ── Phase 3：流量反归一化参数 ──────────────────────────────────────────
    # 这两个值必须在训练启动前从 Phase 1 生成的 scaler 中读取并写入 cfg，
    # 反归一化公式：real_flow = pred_scaled * flow_std + flow_mean
    # 加载方式（在 master_loader.py 或训练入口中执行）：
    #   import pickle
    #   with open('processed_data/pyg_dataset/scalers/flow_scaler.pkl', 'rb') as f:
    #       scaler = pickle.load(f)
    #   cfg.dataset.flow_mean = float(scaler.mean_[0])
    #   cfg.dataset.flow_std  = float(scaler.scale_[0])
    cfg.dataset.flow_mean = 0.0   # 占位符，运行前必须覆盖为真实值
    cfg.dataset.flow_std  = 1.0   # 占位符，运行前必须覆盖为真实值
