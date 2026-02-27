"""
Phase 2：双重图神经网络 —— NetworkPairsTopologyModel
======================================================

核心创新：无需 OD 矩阵，仅凭历史流量（flows_old）作为需求代理，
          预测网络拓扑重构后的新均衡流量（flows_new）。

三大子模块：
  ┌─────────────────────────────────────────────────────────────────┐
  │ Module 1: OldGraphEncoder                                       │
  │   在旧图 G 的拓扑上聚合历史流量，输出节点级"历史拥堵记忆"       │
  │   h_nodes_old : [N, H]                                          │
  ├─────────────────────────────────────────────────────────────────┤
  │ Module 2: EdgeAlignmentModule                                   │
  │   纯向量化（无 for 循环）匹配 G 与 G' 的边，                    │
  │   为每条新图边生成 8 维对齐特征向量                              │
  │   aligned_features : [E_new, 8]                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ Module 3: NewGraphReasoner                                       │
  │   融合历史记忆，在 G' 拓扑上推理重构后的均衡流量                │
  │   flow_pred : [E_new, 1]  （Unbounded，适应 StandardScaler）    │
  └─────────────────────────────────────────────────────────────────┘

批处理兼容性说明：
  PyG 的 Batch.from_data_list() 会对所有含 'index' 后缀的字段
  自动加节点偏移量（__inc__ 机制），因此 edge_index_old 和
  edge_index_new 在批处理后依然是全局正确的节点索引。
  EdgeAlignmentModule 中的 Hash Key 使用 total_nodes（全局节点数）
  作为编码基数，确保跨图的键不冲突。
"""

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer


# ============================================================
# 轻量级 GNN Batch 容器
# ============================================================

class _GNNBatch:
    """
    极简数据容器，专门用于驱动 GatedGCNLayer.forward()。

    GatedGCNLayer.forward(batch) 只访问三个字段：
      batch.x          : [N, H]  节点特征
      batch.edge_attr  : [E, H]  边特征
      batch.edge_index : [2, E]  图连接关系

    当 GatedGCNLayer 以 equivstable_pe=False 初始化时，
    代码路径 `batch.pe_EquivStableLapPE` 永远不会被执行
    （Python 短路求值），因此此容器无需包含该字段。
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
# Module 1: OldGraphEncoder（历史集线器记忆编码器）
# ============================================================

class OldGraphEncoder(nn.Module):
    """
    在旧图 G 的拓扑上编码历史交通拥堵模式。

    设计要点：
      - 节点初始化为"白板" ones(N, H)，直接在隐藏空间起步，
        省去意义不明的 1 → H 线性映射层。
      - 边输入 = cat([edge_attr_old(3), flow_old(1)]) = [E_old, 4]，
        经投影层映射到隐藏空间后送入 GatedGCN 栈。
      - GatedGCN 的门控聚合机制使模型能区分高/低流量路段，
        有效捕捉"哪些节点是拥堵枢纽"。

    张量形状约定（H = hidden_dim）：
      输入：
        edge_index_old  : [2, E_old]
        edge_attr_old   : [E_old, 3]  - [capacity, speed, length]（已归一化）
        flow_old        : [E_old, 1]  - 历史均衡流量（已归一化）
        num_nodes       : int         - 当前批次总节点数

      输出：
        h_nodes_old     : [num_nodes, H]  - 节点历史记忆嵌入
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

        # 边特征投影：将 [capacity, speed, length, flow_old] 映射到隐藏空间
        # 形状变化：[E_old, 4] → [E_old, H]
        # GatedGCNLayer 要求节点特征与边特征维度严格相同（均为 H）
        self.edge_proj = nn.Linear(4, hidden_dim)

        # 堆叠 GatedGCN 层：在旧图拓扑上做多轮消息传递
        # 每一层的 in_dim = out_dim = H，保持维度不变
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

        # ── 白板节点初始化 ──────────────────────────────────────────
        # 直接在隐藏空间 H 维中初始化，避免引入无物理意义的 1→H 线性层。
        # 所有节点以相同的"无先验"状态出发，历史流量信息完全依靠
        # GatedGCN 的消息传递从边流向节点。
        # 形状：[num_nodes, H]
        x = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # ── 边特征投影 ──────────────────────────────────────────────
        # cat([edge_attr_old(3), flow_old(1)]) → [E_old, 4]
        # 然后投影到隐藏空间      → [E_old, H]
        e_raw = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]
        e = self.edge_proj(e_raw)                              # [E_old, H]

        # ── 堆叠消息传递（旧图拓扑）────────────────────────────────
        for layer in self.gnn_layers:
            mini_batch = _GNNBatch(x, e, edge_index_old)
            mini_batch = layer(mini_batch)
            x = mini_batch.x           # [num_nodes, H]
            e = mini_batch.edge_attr   # [E_old, H]

        # 返回节点嵌入作为"历史拥堵记忆"，边嵌入不再需要
        return x   # h_nodes_old : [num_nodes, H]


# ============================================================
# Module 2: EdgeAlignmentModule（异构边特征对齐）
# ============================================================

class EdgeAlignmentModule(nn.Module):
    """
    向量化的异构边特征对齐模块（无可学习参数）。

    核心算法：基于节点对 Hash 的 O(E_old + E_new) 向量化匹配。

    对齐规则（严格遵循 .cursorrules Phase 2 规范）：
    ┌──────────────────────────────────────────────────────────────┐
    │ 保留边（G 和 G' 中均存在）：                                 │
    │   [edge_attr_old(3), flow_old(1), edge_attr_new(3), 0(1)]   │
    │   → 维度合计 8，is_new_edge = 0                             │
    ├──────────────────────────────────────────────────────────────┤
    │ 新修边（仅 G' 中存在）：                                     │
    │   [zeros(3),         zero(1),    edge_attr_new(3), 1(1)]    │
    │   → 维度合计 8，is_new_edge = 1                             │
    └──────────────────────────────────────────────────────────────┘

    Hash 编码方案：
      key(src, dst) = src × total_nodes + dst
      由于 src, dst ∈ [0, total_nodes)，任意不同节点对的 key 唯一。

    批处理正确性：
      PyG Batch 对边索引施加的节点偏移（__inc__）使得跨图的节点 ID
      互不重叠，因此以 total_nodes（批次全局节点数）为编码基数时，
      不同图的边 key 天然不冲突。

    张量形状约定（H = hidden_dim）：
      输入：
        edge_index_old  : [2, E_old]
        edge_attr_old   : [E_old, 3]
        flow_old        : [E_old, 1]
        edge_index_new  : [2, E_new]
        edge_attr_new   : [E_new, 3]
        total_nodes     : int

      输出：
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

        # ── Step 1：将边的节点对编码为唯一整数 Key ──────────────────
        #
        # 公式：key(src, dst) = src × total_nodes + dst
        #
        # 原理：由于 0 ≤ src, dst < total_nodes，
        #       不同 (src, dst) 对映射的 key 一定不同，等价于
        #       将二维坐标展开为一维坐标（行主序）。
        #
        # 时间复杂度：O(E_old + E_new)，零 Python for 循环。
        old_keys = edge_index_old[0] * total_nodes + edge_index_old[1]  # [E_old]
        new_keys = edge_index_new[0] * total_nodes + edge_index_new[1]  # [E_new]

        # ── Step 2：建立反向查找表 key → old_edge_index ─────────────
        #
        # key_to_old_idx[key] = 旧图中该边的索引，-1 表示不存在。
        # 对于 Sioux Falls（24 节点，批量 64 图）：
        #   total_nodes = 24 × 64 = 1536，max_key ≈ 2.36M，内存约 9MB，可行。
        max_key = total_nodes * total_nodes
        key_to_old_idx = torch.full(
            (max_key,), fill_value=-1, dtype=torch.long, device=device
        )

        old_edge_pos = torch.arange(E_old, dtype=torch.long, device=device)
        # 向量化散列写入：将旧图每条边的 key 映射到其在 edge_attr_old 中的行号
        # （若存在重复边，后写者覆盖先写者；有向图理论上不存在此情况）
        key_to_old_idx[old_keys] = old_edge_pos                          # [max_key]

        # ── Step 3：为每条新图边查找其在旧图中的匹配索引 ─────────────
        #
        # match_idx[i] = j  表示 edge_index_new[:, i] 对应 edge_attr_old[j]
        # match_idx[i] = -1 表示该边是新修边，旧图中不存在
        match_idx = key_to_old_idx[new_keys]  # [E_new]，值域 {-1, 0, ..., E_old-1}

        # ── Step 4：构建旧图侧的对齐特征（4 维）──────────────────────
        #
        # 保留边：使用真实的旧属性 + 旧流量
        # 新修边：补零（代表"此前不存在"）
        #
        # old_feats[j] = cat([edge_attr_old[j], flow_old[j]])，形状 [4]
        old_feats = torch.cat([edge_attr_old, flow_old], dim=-1)  # [E_old, 4]

        # 初始化为零矩阵（默认：新修边对应旧侧全补零）
        aligned_old = torch.zeros(E_new, 4, dtype=dtype, device=device)

        # 布尔掩码：标记哪些新图边在旧图中存在（保留边）
        retained_mask = match_idx >= 0  # [E_new], bool

        # 仅对保留边执行向量化填充，避免 -1 索引导致越界
        if retained_mask.any():
            # aligned_old[retained_mask] ← old_feats[match_idx[retained_mask]]
            # 此操作为纯张量索引，无 Python 循环
            aligned_old[retained_mask] = old_feats[match_idx[retained_mask]]

        # ── Step 5：构建 is_new_edge 指示变量 ─────────────────────────
        #
        # is_new_edge = 1 表示新修边（旧图不存在），模型可据此差异化处理
        # is_new_edge = 0 表示保留边（旧图已存在）
        # 形状：[E_new, 1]，float 类型以便参与线性层计算
        is_new_edge = (~retained_mask).to(dtype).unsqueeze(1)  # [E_new, 1]

        # ── Step 6：拼接最终对齐特征 ────────────────────────────────
        #
        # 保留边：[edge_attr_old(3), flow_old(1), edge_attr_new(3), 0(1)] = 8 维
        # 新修边：[zeros(3),         zero(1),     edge_attr_new(3), 1(1)] = 8 维
        # 形状严格为 [E_new, 8]
        aligned_features = torch.cat(
            [aligned_old, edge_attr_new, is_new_edge], dim=-1
        )  # [E_new, 8]

        return aligned_features


# ============================================================
# Module 3: NewGraphReasoner（破旧立新的推理器）
# ============================================================

class NewGraphReasoner(nn.Module):
    """
    在新图 G' 的拓扑上融合历史记忆并推理未来均衡流量。

    Node Fusion 设计原则（关键约束）：
      x_new_init = ones(N, H)           ← 新图白板节点
      x_fused    = cat([x_new_init, h_nodes_old])  → [N, 2H]
      x          = node_fusion(x_fused)            → [N, H]

      ⚠️  严格禁止在 node_fusion 后添加残差连接！
          残差 x = x_fused_proj + h_nodes_old 会形成"记忆捷径"，
          让模型通过直接复制旧流量绕过对新拓扑的学习。
          强制非线性压缩（Linear→ReLU→Dropout）迫使模型
          从新图的消息传递中重新发现流量分布。

    Edge Decoder 设计：
      输入 = cat([x_src(H), x_dst(H), aligned_features(8)]) → [2H + 8]
      输出 = flow_pred [E_new, 1]
      最后一层为纯线性层，无激活函数（Unbounded Output）。
      标签经过 StandardScaler（零均值，单位方差，含负值），
      有界激活（如 ReLU/Sigmoid）会造成系统性预测偏差。

    张量形状约定（H = hidden_dim）：
      输入：
        edge_index_new   : [2, E_new]
        aligned_features : [E_new, 8]
        h_nodes_old      : [num_nodes, H]
        num_nodes        : int

      输出：
        flow_pred : [E_new, 1]
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

        # ── Node Fusion 层 ─────────────────────────────────────────
        # 输入：cat([x_new_init(H), h_nodes_old(H)]) → [N, 2H]
        # 输出：[N, H]
        #
        # 设计理念：通过非线性压缩，模型被迫在"白板"（适应新拓扑）
        # 与"历史记忆"（旧流量模式）之间找到动态平衡。
        # 严格禁止残差——避免模型退化为直接输出旧流量的捷径策略。
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 对齐边特征投影：[E_new, 8] → [E_new, H]
        # GatedGCNLayer 要求节点与边特征维度相同
        self.edge_proj = nn.Linear(8, hidden_dim)

        # 堆叠 GatedGCN 层：在新图拓扑上做多轮消息传递
        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

        # ── 边级流量解码器 ─────────────────────────────────────────
        # 输入：cat([x_src(H), x_dst(H), aligned_features(8)]) → [2H + 8]
        #
        # 设计理念：显式地将"源节点状态 + 目标节点状态 + 边本身特征"
        # 三者融合，比仅用边特征的解码器更能捕捉方向性流量规律。
        # aligned_features 保留原始 8 维（含 is_new_edge 标志位），
        # 让解码器区分新修边与保留边，差异化地预测其流量。
        #
        # 最后一层：nn.Linear(H, 1)，无激活函数（Unbounded Output）。
        decoder_in_dim = hidden_dim * 2 + 8
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # ⚠️  此处故意省略激活函数！
            # StandardScaler 处理后的标签分布在 (-∞, +∞)，
            # 任何有界激活（ReLU/Tanh/Sigmoid）都会引入偏差。
        )

    def forward(
        self,
        edge_index_new: torch.Tensor,    # [2, E_new]
        aligned_features: torch.Tensor,  # [E_new, 8]
        h_nodes_old: torch.Tensor,       # [num_nodes, H]
        num_nodes: int,
    ) -> torch.Tensor:                   # → [E_new, 1]

        device = h_nodes_old.device
        dtype  = h_nodes_old.dtype

        # ── 白板节点初始化（新图）────────────────────────────────────
        # x_new_init 代表"对新图一无所知"的初始状态。
        # 与 h_nodes_old 融合后，模型需要通过新图的拓扑自行重分配流量，
        # 而不是简单地继承旧状态。
        # 形状：[num_nodes, H]
        x_new_init = torch.ones(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # ── 历史记忆融合（Node Fusion）────────────────────────────────
        # 拼接白板节点与历史记忆：[N, H] ‖ [N, H] → [N, 2H]
        # 经非线性融合层压缩：[N, 2H] → [N, H]
        #
        # ⚠️  无残差连接！x 必须完全由 node_fusion 的非线性变换决定，
        #     不允许 h_nodes_old 绕过变换直接流入下游网络。
        x_cat = torch.cat([x_new_init, h_nodes_old], dim=-1)  # [N, 2H]
        x = self.node_fusion(x_cat)                            # [N, H]

        # ── 对齐边特征投影 ─────────────────────────────────────────
        # 将 8 维对齐特征投影到隐藏空间，满足 GatedGCNLayer 维度约束
        # 形状：[E_new, 8] → [E_new, H]
        e = self.edge_proj(aligned_features)  # [E_new, H]

        # ── 在新图拓扑上堆叠消息传递 ─────────────────────────────────
        # 此阶段是模型"理解新拓扑"的关键：
        #   - 新修边（is_new_edge=1）会将其物理属性传播给相邻节点
        #   - 删除的边不再存在，流量自然重分布到保留路径
        for layer in self.gnn_layers:
            mini_batch = _GNNBatch(x, e, edge_index_new)
            mini_batch = layer(mini_batch)
            x = mini_batch.x           # [num_nodes, H]
            e = mini_batch.edge_attr   # [E_new, H]

        # ── 边级流量解码 ───────────────────────────────────────────
        # 为每条新图边提取源节点、目标节点的嵌入，
        # 与原始 8 维对齐特征（未经投影）拼接，送入解码器。
        #
        # 保留原始 aligned_features（而非投影后的 e）：
        #   1. aligned_features 包含 is_new_edge 标志位，
        #      让解码器显式感知边的"新旧"属性
        #   2. 原始物理属性（capacity, speed, length）的量纲
        #      与 flow 直接相关，不经投影失真地喂给解码器
        src_idx, dst_idx = edge_index_new[0], edge_index_new[1]
        edge_repr = torch.cat(
            [x[src_idx], x[dst_idx], aligned_features], dim=-1
        )  # [E_new, 2H + 8]

        flow_pred = self.edge_decoder(edge_repr)  # [E_new, 1]

        return flow_pred


# ============================================================
# 主模型：NetworkPairsTopologyModel
# ============================================================

@register_network('topology_gnn')
class NetworkPairsTopologyModel(nn.Module):
    """
    双重图神经网络主模型，针对交通网络拓扑重构场景。

    严格遵守项目约束：
      - 无 OD 矩阵输入（batch.x 中仅含全 1 占位符，模型内部不使用）
      - flows_old 作为需求代理（历史流量，模型的核心输入之一）
      - 最终预测为 StandardScaler 空间的无界实数，损失函数在调用方计算

    期望的 PyG Batch 字段（由 Phase 1 的 build_network_pairs_dataset.py 生成）：
      batch.edge_index_old : [2, E_old_total]   旧图连接（已施加批处理节点偏移）
      batch.edge_attr_old  : [E_old_total, 3]   旧图物理属性（归一化）
      batch.flow_old       : [E_old_total, 1]   旧图历史流量（归一化）
      batch.edge_index_new : [2, E_new_total]   新图连接（已施加批处理节点偏移）
      batch.edge_attr_new  : [E_new_total, 3]   新图物理属性（归一化）
      batch.y              : [E_new_total, 1]   新图均衡流量（归一化，Ground Truth）
      batch.num_nodes      : int                批次总节点数（PyG 自动求和）

    Args:
        dim_in  : 占位参数（GraphGym 接口约定），模型内部不使用
        dim_out : 占位参数（GraphGym 接口约定），输出维度由架构固定为 1
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()

        # 从 cfg.topology_gnn 读取所有超参数
        # 对应配置文件：graphgps/config/topology_gnn_config.py
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

        self.reasoner = NewGraphReasoner(
            hidden_dim=hidden_dim,
            num_layers=num_layers_new,
            dropout=dropout,
            residual=residual,
        )
        def replace_bn_with_ln(module):
            for name, child in module.named_children():
                # 如果发现名字里带有 BatchNorm 的层
                if 'BatchNorm' in child.__class__.__name__:
                    # 用 LayerNorm 替换它
                    setattr(module, name, nn.LayerNorm(child.weight.shape[0]))
                else:
                    # 递归处理子模块
                    replace_bn_with_ln(child)
                    
        replace_bn_with_ln(self)

    def forward(self, batch):
        """
        完整前向传播。

        流程：
          1. OldGraphEncoder  → h_nodes_old : [N_total, H]
          2. EdgeAlignmentModule → aligned_features : [E_new_total, 8]
          3. NewGraphReasoner → flow_pred : [E_new_total, 1]

        Args:
            batch : PyG Batch 对象（包含一个 mini-batch 的 (G, G') 对）

        Returns:
            pred : torch.Tensor [E_new_total, 1]  预测流量（标准化空间）
            true : torch.Tensor [E_new_total, 1]  真实流量（标准化空间）
        """
        # PyG 在批处理时自动将各图的 num_nodes 相加，
        # 得到当前批次的全局节点总数（用于 GNN 聚合和 Hash 匹配）
        total_nodes: int = batch.num_nodes

        # ── Module 1：在旧图上编码历史拥堵记忆 ───────────────────────
        # 输出：h_nodes_old [total_nodes, H]
        h_nodes_old = self.encoder(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            num_nodes=total_nodes,
        )

        # ── Module 2：向量化对齐新旧图边特征 ─────────────────────────
        # 输出：aligned_features [E_new_total, 8]
        aligned_features = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )

        # ── Module 3：融合历史记忆，在新图上推理流量 ──────────────────
        # 输出：flow_pred [E_new_total, 1]
        flow_pred = self.reasoner(
            edge_index_new=batch.edge_index_new,
            aligned_features=aligned_features,
            h_nodes_old=h_nodes_old,
            num_nodes=total_nodes,
        )

        return flow_pred, batch.y
