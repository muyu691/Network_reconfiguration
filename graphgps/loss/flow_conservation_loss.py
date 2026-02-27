"""
Phase 3：物理信息神经网络（PINN）守恒损失
==========================================

实现核心物理约束：流量守恒定律（Kirchhoff's First Law in Traffic Networks）
  对于每个非质心节点 v：∑ inflow(v) = ∑ outflow(v)

文件包含：
  ┌──────────────────────────────────────────────────────────────────┐
  │  FlowConservationLoss    守恒惩罚项（纯物理逻辑，可独立复用）     │
  │  CombinedPINNLoss        监督损失 + λ × 守恒损失的完整组合        │
  │  compute_pinn_loss()     顶层函数，供 custom_train.py 调用         │
  └──────────────────────────────────────────────────────────────────┘

⚠️ 关于 @register_loss 接口的设计决策：
  GraphGPS 的标准 register_loss 签名为 (pred, true)，无法携带 batch 上下文。
  而守恒损失需要 batch.edge_index_new 和 batch.non_centroid_mask。
  因此本文件采用独立 nn.Module + 顶层函数的设计，
  由 Phase 4 修改后的 custom_train.py 直接调用 compute_pinn_loss(pred, batch)。

配置依赖（运行前必须确保已设置正确值）：
  cfg.dataset.flow_mean   : float  - StandardScaler 的均值（来自 flow_scaler.pkl）
  cfg.dataset.flow_std    : float  - StandardScaler 的标准差（来自 flow_scaler.pkl）
  cfg.model.lambda_cons   : float  - 守恒损失权重系数 λ（默认 0.1）
  cfg.model.cons_norm     : str    - 守恒误差范数类型，'l2'（MSE）或 'l1'（MAE）
  cfg.model.loss_fun      : str    - 监督损失类型，'l1' 或 'mse'（默认 'l1'）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg


# ============================================================
# 核心模块一：FlowConservationLoss（流量守恒惩罚项）
# ============================================================

class FlowConservationLoss(nn.Module):
    """
    计算非质心节点的流量守恒惩罚。

    ════════════════════════════════════════════════════════
    为什么必须先反归一化再计算守恒误差？
    ════════════════════════════════════════════════════════

    设某非质心节点 v 有 d_in 条入边，d_out 条出边。
    设 flow_scaler 的 mean = μ，std = σ。

    在 StandardScaler 的归一化空间中，真实流量 f 被映射为：
      f̂ = (f - μ) / σ

    若直接在归一化空间计算守恒约束，条件为：
      ∑_{e→v} f̂_e  =  ∑_{v→e} f̂_e

    将反归一化代入：
      ∑_{e→v} (f_e - μ)/σ  =  ∑_{v→e} (f_e - μ)/σ

      [∑ f_in  - d_in  × μ] / σ  =  [∑ f_out - d_out × μ] / σ

      ∑ f_in - ∑ f_out  =  (d_in - d_out) × μ

    当 d_in ≠ d_out（入度 ≠ 出度，有向图中极为常见），
    且 μ ≠ 0 时，归一化空间中的守恒约束不等价于真实空间的守恒约束！

    换言之：强迫模型在归一化空间满足 ∑ pred_in = ∑ pred_out，
    等效于要求真实流量满足 ∑ f_in - ∑ f_out = (d_in - d_out) × μ，
    这是物理上完全错误的约束，会系统性地引导模型产生有偏预测。

    正确做法：
      real_flow = pred_scaled × σ + μ    （反归一化）
      constraint: ∑ real_flow_in(v) = ∑ real_flow_out(v)

    ════════════════════════════════════════════════════════

    张量形状约定：
      输入：
        pred             : [E_new, 1]    模型输出（StandardScaler 空间）
        edge_index_new   : [2, E_new]    新图连接关系（已施加批处理节点偏移）
        non_centroid_mask: [N_total]     bool，True 表示非质心节点
        num_nodes        : int           批次总节点数
        flow_mean        : float         StandardScaler 均值 μ
        flow_std         : float         StandardScaler 标准差 σ

      输出：
        loss_cons        : scalar        守恒惩罚（MSE 或 MAE）
    """

    def forward(
        self,
        pred: torch.Tensor,              # [E_new, 1]
        edge_index_new: torch.Tensor,    # [2, E_new]
        non_centroid_mask: torch.Tensor, # [N_total], bool
        num_nodes: int,
        flow_mean: float,
        flow_std: float,
    ) -> torch.Tensor:                   # → scalar

        # ── Step 1：反归一化 ────────────────────────────────────────────
        # 将模型输出从 StandardScaler 空间还原到真实车流量空间（辆/小时）。
        #
        # 公式：real_flow = pred_scaled × σ + μ
        #
        # 注意：flow_mean 和 flow_std 是 Python float，
        #       直接参与 Tensor 运算，自动广播到 [E_new, 1]。
        #       反归一化操作可微（线性变换），不影响梯度传播。
        real_flow = pred * flow_std + flow_mean  # [E_new, 1]

        # 压缩为 1D，方便后续 scatter_add 操作
        real_flow_flat = real_flow.squeeze(-1)   # [E_new]

        # ── Step 2：向量化图聚合（scatter_add）──────────────────────────
        #
        # scatter_add 的语义：
        #   output[index[i]] += src[i]
        #
        # 对于有向边 (src → dst)：
        #   outflow[src_node] += real_flow_flat[e]   （每条边的流量流出源节点）
        #   inflow[dst_node]  += real_flow_flat[e]   （每条边的流量流入目标节点）
        #
        # 时间复杂度：O(E_new)，完全向量化，无 Python 循环。
        #
        # 与 torch_scatter.scatter_add 的等价性：
        #   torch_scatter.scatter_add(real_flow_flat, src_nodes, dim=0, dim_size=num_nodes)
        #   ≡  torch.zeros(num_nodes).scatter_add_(0, src_nodes, real_flow_flat)
        # 此处使用原生 PyTorch API，避免额外依赖（torch_scatter 已在 GatedGCN 中引入）。

        src_nodes = edge_index_new[0]   # [E_new] - 各边的源节点索引
        dst_nodes = edge_index_new[1]   # [E_new] - 各边的目标节点索引

        device = real_flow_flat.device
        dtype  = real_flow_flat.dtype

        # 初始化聚合容器：大小为 num_nodes（批次总节点数，含所有图的所有节点）
        outflow = torch.zeros(num_nodes, device=device, dtype=dtype)
        inflow  = torch.zeros(num_nodes, device=device, dtype=dtype)

        # 向量化累加：每条边的流量同时贡献到其源节点的 outflow 和目标节点的 inflow
        outflow.scatter_add_(0, src_nodes, real_flow_flat)  # [N_total]
        inflow.scatter_add_(0, dst_nodes, real_flow_flat)   # [N_total]

        # ── Step 3：施加物理掩码（仅非质心节点）────────────────────────
        #
        # 质心节点（Sioux Falls 中的节点 1-11，0-indexed 为 0-10）是 OD 区域：
        #   - 它们可以"无中生有"地产生出行需求（起点）或吸收到达流量（终点）
        #   - 对质心节点施加守恒约束在物理上是错误的
        #
        # 非质心节点（节点 12-24，0-indexed 为 11-23）是普通交叉口：
        #   - 车辆只能"过境"，不能在此凭空产生或消失
        #   - Kirchhoff 第一定律在此节点严格成立：∑ inflow = ∑ outflow
        #
        # batch.non_centroid_mask 在批处理时由 PyG 沿 dim=0 拼接，
        # 形状从单图的 [24] 自动扩展为批次的 [24 × batch_size]，无需手动处理。
        diff = inflow - outflow             # [N_total]，单位：辆/小时（真实空间）
        diff_nc = diff[non_centroid_mask]   # [N_non_centroid_total]，仅保留非质心节点

        # ── Step 4：无量纲化缩放（Scale Alignment）────────────────────
        #
        # ⚠️  关键修复：消除"真实流量空间 vs 归一化空间"的量纲失衡！
        #
        # 问题根源：
        #   L_sup  在归一化空间计算（pred, y ~ N(0,1)），数量级约 O(1)
        #   diff_nc 在真实流量空间（单位：辆/小时），数量级约 O(σ²) ~ O(10⁶)
        #   若直接相加：total_loss = L_sup + λ × L_cons，即便 λ = 0.01，
        #   L_cons 产生的梯度也会彻底淹没 L_sup 的梯度 → 梯度爆炸或模型崩溃。
        #
        # 解决方案：将真实空间的守恒误差除以 σ（flow_std），回到"标准差单位"。
        #   diff_nc_scaled = diff_nc / σ
        #
        # 数学合法性证明：
        #   守恒约束等价类：∑ inflow = ∑ outflow
        #                   ⟺  (∑ inflow) / σ = (∑ outflow) / σ
        #                   ⟺  ∑ [(f_in × σ + μ) / σ] = ∑ [(f_out × σ + μ) / σ]
        #                           （代入反归一化公式，化简后）
        #                   ⟺  ∑ f̂_in + d_in × (μ/σ) = ∑ f̂_out + d_out × (μ/σ)
        #   这与在真实空间的约束完全等价（两边同除以 σ 不改变零点位置）。
        #
        # 物理含义变化：
        #   diff_nc        → "此交叉口的不守恒量（辆/小时）"
        #   diff_nc_scaled → "此交叉口的不守恒量（以 σ 为单位，无量纲）"
        #   后者与 L_sup 的残差处于同一量级：O(1)，保证梯度平衡。
        #
        # 注意：除以 σ（标量）是可微操作，梯度链路完整保留。
        diff_nc_scaled = diff_nc / flow_std  # [N_non_centroid_total]，无量纲

        # ── Step 5：计算守恒惩罚（可选 L1 或 L2 范数）──────────────────
        #
        # L2（MSE）：对偏差的平方取均值，对大违规更敏感，梯度更平滑
        # L1（MAE）：对偏差的绝对值取均值，对离群值更鲁棒
        #
        # 范数类型由 cfg.model.cons_norm 控制，默认 'l2'（如 cursorrules 所定义）
        if cfg.model.cons_norm == 'l1':
            loss_cons = diff_nc_scaled.abs().mean()
        else:
            loss_cons = (diff_nc_scaled ** 2).mean()

        return loss_cons


# ============================================================
# 核心模块二：CombinedPINNLoss（监督损失 + 守恒损失）
# ============================================================

class CombinedPINNLoss(nn.Module):
    """
    组合损失：监督损失 + λ × 物理守恒损失

    公式：
      L_total = L_sup + λ × L_cons

    其中：
      L_sup  = L1Loss(pred, y)  或  MSELoss(pred, y)  （在归一化空间直接计算）
      L_cons = MSE 或 MAE of (inflow - outflow) on non-centroid nodes
               （必须在真实流量空间计算，需先反归一化）

    设计理念（三步走，缺一不可）：
      1. L_sup 直接在 StandardScaler 归一化空间计算：此时 pred 和 y 均已归一化，
         两者均值为 0，方差为 1，MSE/L1 度量的是归一化残差，数量级 O(1)。
      2. L_cons 必须先反归一化到真实流量空间做加减法：消除
         (d_in - d_out) × μ 的常数项漂移，确保守恒约束物理正确。
      3. L_cons 最终除以 σ 回到无量纲尺度：将真实误差（辆/小时，量级 O(σ²)）
         压缩回 O(1)，与 L_sup 对齐，防止梯度爆炸与监督信号被吞噬。

    Args:
        dim_in  : 未使用（兼容 GraphGPS 模型工厂接口约定）
        dim_out : 未使用
    """

    def __init__(self) -> None:
        super().__init__()
        self.conservation = FlowConservationLoss()

    def forward(
        self,
        pred: torch.Tensor,   # [E_new, 1]  模型输出（归一化空间）
        batch,                # PyG Batch 对象
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss : scalar  加权总损失（用于反向传播）
            pred       : [E_new, 1]  预测值（供 logger 记录指标，保持 compute_loss 约定）
        """
        true = batch.y   # [E_new, 1]，归一化后的 Ground Truth

        # ── 监督损失（归一化空间）─────────────────────────────────────
        # pred 和 true 均在同一归一化空间，可直接计算残差。
        # 使用 L1Loss 作为默认（对交通流量的重尾分布更鲁棒）。
        if cfg.model.loss_fun == 'mse':
            loss_sup = F.mse_loss(pred, true)
        else:
            loss_sup = F.l1_loss(pred, true)

        # ── 守恒损失（真实流量空间）───────────────────────────────────
        # 读取 Phase 1 中由 StandardScaler 计算并写入 cfg 的统计量
        # 注意：这两个值在训练入口（master_loader.py 或 main.py）中
        #       从 scalers/flow_scaler.pkl 加载后设置，不可使用默认占位符
        flow_mean = cfg.dataset.flow_mean   # μ
        flow_std  = cfg.dataset.flow_std    # σ

        loss_cons = self.conservation(
            pred=pred,
            edge_index_new=batch.edge_index_new,
            non_centroid_mask=batch.non_centroid_mask,
            num_nodes=batch.num_nodes,
            flow_mean=flow_mean,
            flow_std=flow_std,
        )

        # ── 加权组合 ────────────────────────────────────────────────
        # Total_Loss = L_sup + λ × L_cons
        # λ 由 cfg.model.lambda_cons 控制，建议从小值开始（0.01 ~ 0.1）调参
        lam = cfg.model.lambda_cons
        total_loss = loss_sup + lam * loss_cons

        return total_loss, pred


# ============================================================
# 顶层函数：compute_pinn_loss（供 custom_train.py 调用）
# ============================================================

# 模块级单例，避免每次调用重新构建（内部无参数，开销极小）
_pinn_loss_fn = CombinedPINNLoss()


def compute_pinn_loss(
    pred: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Phase 4 的调用入口：在 custom_train.py 中替换 compute_loss(pred, true)。

    使用方式（custom_train.py 中）：

        from graphgps.loss.flow_conservation_loss import compute_pinn_loss

        # 替换原有的：
        #   loss, pred_score = compute_loss(pred, true)
        # 改为：
        #   loss, pred_score = compute_pinn_loss(pred, batch)

    Args:
        pred  : [E_new, 1]  模型预测值（来自 NetworkPairsTopologyModel.forward）
        batch : PyG Batch 对象，需包含字段：
                  batch.y                - [E_new, 1]  归一化 Ground Truth
                  batch.edge_index_new   - [2, E_new]  新图拓扑
                  batch.non_centroid_mask- [N_total]   非质心节点布尔掩码
                  batch.num_nodes        - int          批次总节点数

    Returns:
        total_loss  : scalar          反向传播用的加权总损失
        pred_score  : [E_new, 1]      预测值（供 logger 计算 MAE/MSE 指标）
    """
    return _pinn_loss_fn(pred, batch)
