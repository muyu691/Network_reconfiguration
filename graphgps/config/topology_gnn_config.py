"""
Phase 2 配置：NetworkPairsTopologyModel 超参数注册

所有参数均挂载在 cfg.topology_gnn 命名空间下，
通过 YAML 配置文件或命令行覆盖。
"""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('topology_gnn')
def topology_gnn_cfg(cfg):
    """
    为 NetworkPairsTopologyModel 注册专属配置组。

    推荐的 YAML 配置片段：
      topology_gnn:
        hidden_dim: 128
        num_layers_old: 3
        num_layers_new: 3
        dropout: 0.1
        residual: true
    """
    cfg.topology_gnn = CN()

    # 所有 GNN 层（旧图编码器 + 新图推理器）的统一隐层维度
    # 节点嵌入、边嵌入均在此空间内
    cfg.topology_gnn.hidden_dim = 128

    # OldGraphEncoder 中堆叠的 GatedGCN 层数
    cfg.topology_gnn.num_layers_old = 3

    # NewGraphReasoner 中堆叠的 GatedGCN 层数
    cfg.topology_gnn.num_layers_new = 3

    # Dropout 概率，应用于 GatedGCN 层内部及 node_fusion / edge_decoder
    cfg.topology_gnn.dropout = 0.1

    # GatedGCN 层是否使用残差连接
    # 注意：此残差指 GatedGCN 层自身的跳跃连接，
    #       与 NewGraphReasoner 的 node_fusion 层禁止残差无关
    cfg.topology_gnn.residual = True
