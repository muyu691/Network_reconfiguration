"""
Phase 1 最后一步：将 (G, G') 数据对转换为 PyG Data 对象并划分数据集

输入：
  network_pairs_dataset.pkl（由 solve_network_pairs.py 生成）
  每个 pair 包含：G, G_prime, flows_old, flows_new, edge_list_old, edge_list_new

输出：
  train_dataset.pt / val_dataset.pt / test_dataset.pt
  scalers/attr_scaler.pkl / scalers/flow_scaler.pkl

每个 PyG Data 对象的字段（严格按照 .cursorrules Phase 1 规格）：
  x              : [24, 1]      全 1 占位符节点特征（模型内部初始化用）
  edge_index_old : [2, E_old]   旧图连接关系，0-indexed
  edge_index_new : [2, E_new]   新图连接关系，0-indexed
  edge_attr_old  : [E_old, 3]   旧图归一化物理属性 [capacity, speed, length]
  flow_old       : [E_old, 1]   旧图归一化历史流量（需求代理，模型输入之一）
  edge_attr_new  : [E_new, 3]   新图归一化物理属性（模型输入之一）
  y              : [E_new, 1]   新图归一化均衡流量（Ground Truth）
  num_nodes      : int          节点数（调试用）
  num_edges_old  : int          旧图边数（调试用）
  num_edges_new  : int          新图边数（调试用）
  mutation_type  : str          变异类型（调试用）

严格约束：
  - x 中绝不包含 OD 数据
  - Scaler 只在训练集上 fit，val/test 直接用训练集的 scaler transform（防止数据泄露）
  - attr_scaler 对 [capacity, speed, length] 统一归一化（旧图和新图属性共享同一 scaler）
  - flow_scaler 对 [flows_old, flows_new] 统一归一化（两者是同一物理量，共享同一分布）
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

NUM_NODES = 24  # Sioux Falls 节点总数，固定不变


# ============================================================
# 工具函数：原始特征提取
# ============================================================

def extract_edge_attrs(G, edge_list: list) -> np.ndarray:
    """
    按 edge_list 的顺序从 NetworkX DiGraph 中提取边的物理属性。

    提取的三列依次为：capacity（通行能力）、speed（限速）、length（物理长度）。
    不提取 free_flow_time，因为它是 length/speed 的导出量，属于冗余特征。

    Args:
        G:         NetworkX DiGraph，边上含 'capacity', 'speed', 'length' 属性
        edge_list: list[(u, v)]，决定输出行的顺序（与 flows_old/flows_new 索引一致）

    Returns:
        attrs: np.ndarray [E, 3]，列序为 [capacity, speed, length]
    """
    caps   = np.array([G[u][v]['capacity'] for u, v in edge_list], dtype=np.float64)
    speeds = np.array([G[u][v]['speed']    for u, v in edge_list], dtype=np.float64)
    lens   = np.array([G[u][v]['length']   for u, v in edge_list], dtype=np.float64)
    # 按列拼接，顺序固定为 [capacity, speed, length]
    return np.column_stack([caps, speeds, lens])  # [E, 3]


def edge_list_to_index(edge_list: list) -> np.ndarray:
    """
    将 edge_list 转换为 PyG 格式的 edge_index [2, E]。

    Sioux Falls 节点编号为 1-24（1-indexed），PyG 要求 0-indexed，因此统一减 1。

    Args:
        edge_list: list[(u, v)]，节点 ID 为 1-indexed

    Returns:
        edge_index: np.ndarray [2, E]，dtype=int64，节点 ID 为 0-indexed
    """
    if len(edge_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(edge_list, dtype=np.int64)  # [E, 2]
    arr -= 1                                    # 1-indexed → 0-indexed
    return arr.T                                # [2, E]


# ============================================================
# Scaler：拟合与保存
# ============================================================

def fit_scalers(pairs: list, train_idx: np.ndarray) -> tuple:
    """
    仅在训练集上拟合归一化器，防止验证集/测试集信息泄露。

    拟合策略：
    - attr_scaler（StandardScaler for [capacity, speed, length]）：
        将训练集所有样本的旧图和新图的边属性【垂直堆叠】后拟合。
        这是因为旧图和新图的属性属于同一物理量（容量/速度/长度），
        应服从同一分布假设，使用同一个 scaler 能保证变换的一致性。

    - flow_scaler（StandardScaler for flows）：
        将训练集所有样本的 flows_old 和 flows_new【合并】后拟合。
        flows_old 和 flows_new 均为交通均衡流量，物理含义相同，共享同一分布。

    Args:
        pairs:     全量数据对列表（只用 train_idx 对应的样本）
        train_idx: 训练集样本的索引数组

    Returns:
        attr_scaler: 已拟合的 StandardScaler（适用于 [capacity, speed, length]）
        flow_scaler: 已拟合的 StandardScaler（适用于 flows，reshape(-1, 1)后传入）
    """
    print("  收集训练集特征以拟合 StandardScaler...")

    all_attrs = []   # 存储所有训练边的属性 [capacity, speed, length]
    all_flows = []   # 存储所有训练边的流量

    for idx in train_idx:
        pair = pairs[idx]

        # --- 旧图边属性 ---
        old_attrs = extract_edge_attrs(pair['G'], pair['edge_list_old'])  # [E_old, 3]
        all_attrs.append(old_attrs)

        # --- 新图边属性 ---
        new_attrs = extract_edge_attrs(pair['G_prime'], pair['edge_list_new'])  # [E_new, 3]
        all_attrs.append(new_attrs)

        # --- 旧图流量 ---
        all_flows.append(pair['flows_old'].reshape(-1, 1))   # [E_old, 1]

        # --- 新图流量（Ground Truth） ---
        all_flows.append(pair['flows_new'].reshape(-1, 1))   # [E_new, 1]

    # 垂直堆叠所有训练样本的特征
    all_attrs_np = np.vstack(all_attrs)  # [N_train_edges_total, 3]
    all_flows_np = np.vstack(all_flows)  # [N_train_flows_total, 1]

    print(f"  训练集边属性矩阵形状: {all_attrs_np.shape}")
    print(f"  训练集流量矩阵形状:   {all_flows_np.shape}")

    # 拟合 attr_scaler
    attr_scaler = StandardScaler()
    attr_scaler.fit(all_attrs_np)

    # 拟合 flow_scaler
    flow_scaler = StandardScaler()
    flow_scaler.fit(all_flows_np)

    print(f"  attr_scaler 均值: {attr_scaler.mean_}")
    print(f"  attr_scaler 标准差: {attr_scaler.scale_}")
    print(f"  flow_scaler 均值: {flow_scaler.mean_[0]:.2f}")
    print(f"  flow_scaler 标准差: {flow_scaler.scale_[0]:.2f}")

    return attr_scaler, flow_scaler


def save_scalers(attr_scaler: StandardScaler, flow_scaler: StandardScaler,
                 output_dir: str) -> None:
    """
    将拟合好的 scaler 对象持久化，供模型评估时反归一化使用。

    保存路径：
      {output_dir}/scalers/attr_scaler.pkl  ← 适用于 edge_attr_old/edge_attr_new
      {output_dir}/scalers/flow_scaler.pkl  ← 适用于 flow_old/y（反变换时恢复真实流量值）

    Args:
        attr_scaler: 拟合好的属性 scaler
        flow_scaler: 拟合好的流量 scaler
        output_dir:  数据集根目录
    """
    scalers_dir = os.path.join(output_dir, 'scalers')
    os.makedirs(scalers_dir, exist_ok=True)

    with open(os.path.join(scalers_dir, 'attr_scaler.pkl'), 'wb') as f:
        pickle.dump(attr_scaler, f)
    with open(os.path.join(scalers_dir, 'flow_scaler.pkl'), 'wb') as f:
        pickle.dump(flow_scaler, f)

    print(f"  Scaler 已保存至: {scalers_dir}/")


# ============================================================
# 核心：单样本 PyG Data 对象构建
# ============================================================

def build_single_data_object(
    pair: dict,
    attr_scaler: StandardScaler,
    flow_scaler: StandardScaler,
) -> Data:
    """
    将一个 (G, G') 数据对转换为 PyG Data 对象。

    字段说明：
    ┌────────────────────┬──────────────┬────────────────────────────────────────────┐
    │ 字段名             │ 形状         │ 说明                                        │
    ├────────────────────┼──────────────┼────────────────────────────────────────────┤
    │ x                  │ [24, 1]      │ 全 1 占位符；OD 数据严禁写入此字段          │
    │ edge_index_old     │ [2, E_old]   │ 旧图连接，0-indexed                         │
    │ edge_index_new     │ [2, E_new]   │ 新图连接，0-indexed                         │
    │ edge_attr_old      │ [E_old, 3]   │ 归一化旧图物理属性 [cap, spd, len]          │
    │ flow_old           │ [E_old, 1]   │ 归一化历史流量，作为需求代理（模型输入）    │
    │ edge_attr_new      │ [E_new, 3]   │ 归一化新图物理属性（模型输入）              │
    │ y                  │ [E_new, 1]   │ 归一化新图均衡流量，Ground Truth            │
    │ non_centroid_mask  │ [24]         │ bool，标记非质心节点（12-24），用于守恒损失  │
    │ num_nodes          │ int          │ = 24（调试/断言用）                         │
    │ num_edges_old      │ int          │ = E_old                                     │
    │ num_edges_new      │ int          │ = E_new                                     │
    │ mutation_type      │ str          │ 变异类型（调试/分析用）                     │
    └────────────────────┴──────────────┴────────────────────────────────────────────┘

    归一化步骤：
    1. 旧图属性 [E_old, 3] → attr_scaler.transform → float32
    2. 新图属性 [E_new, 3] → attr_scaler.transform → float32
    3. flows_old [E_old, 1] → flow_scaler.transform → float32
    4. flows_new [E_new, 1] → flow_scaler.transform → float32

    Args:
        pair:         solve_network_pairs.py 输出的单个数据对 dict
        attr_scaler:  已拟合的边属性 StandardScaler
        flow_scaler:  已拟合的流量 StandardScaler

    Returns:
        data: torch_geometric.data.Data 对象
    """
    # ── 提取边属性 ────────────────────────────────────────────
    # extract_edge_attrs 按 edge_list 的固定顺序提取，与 flows_old/new 索引对齐
    raw_attr_old = extract_edge_attrs(pair['G'],       pair['edge_list_old'])  # [E_old, 3]
    raw_attr_new = extract_edge_attrs(pair['G_prime'], pair['edge_list_new'])  # [E_new, 3]

    # ── 归一化 ──────────────────────────────────────────────────────────────
    # 注意：Scaler 已在训练集 fit，这里直接 transform（val/test 同样用训练集 scaler）

    # 边物理属性归一化
    norm_attr_old = attr_scaler.transform(raw_attr_old).astype(np.float32)  # [E_old, 3]
    norm_attr_new = attr_scaler.transform(raw_attr_new).astype(np.float32)  # [E_new, 3]

    # 流量归一化：先 reshape 为 [E, 1] 再 transform，保持 2D 输出形状
    flows_old_2d = pair['flows_old'].reshape(-1, 1)                         # [E_old, 1]
    flows_new_2d = pair['flows_new'].reshape(-1, 1)                         # [E_new, 1]
    norm_flow_old = flow_scaler.transform(flows_old_2d).astype(np.float32)  # [E_old, 1]
    norm_flow_new = flow_scaler.transform(flows_new_2d).astype(np.float32)  # [E_new, 1]

    # ── edge_index（1-indexed → 0-indexed）────────────────────────────────
    ei_old = edge_list_to_index(pair['edge_list_old'])  # [2, E_old]
    ei_new = edge_list_to_index(pair['edge_list_new'])  # [2, E_new]

    # ── 节点特征：全 1 占位符（24 × 1）────────────────────────────────────
    # 严格按 .cursorrules Phase 2 规格：模型内部会用 hidden_dim 重新初始化，
    # 此处 x 仅作为占位以满足 PyG Data 接口要求，绝不包含 OD 信息。
    x = torch.ones((NUM_NODES, 1), dtype=torch.float)

    # ── 节点掩码：标记非质心节点（用于物理守恒损失计算）─────────────────────
    # Sioux Falls 拓扑定义：节点 1-11 是质心（OD 源汇，可凭空产生/吸收车流）
    #                      节点 12-24 是普通交叉口（必须满足流量守恒）
    # 0-indexed 后，节点 12-24 对应数组索引 11-23，即 [11:]
    # 此掩码在 Phase 3 损失函数中用于 L_cons 的计算，避免硬编码索引散落各处。
    non_centroid_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    non_centroid_mask[11:] = True

    # ── 构建 PyG Data 对象 ────────────────────────────────────────────────
    data = Data(
        # 节点特征
        x=x,                                                         # [24, 1]

        # 旧图结构与特征
        edge_index_old=torch.from_numpy(ei_old).long(),              # [2, E_old]
        edge_attr_old=torch.from_numpy(norm_attr_old),               # [E_old, 3]
        flow_old=torch.from_numpy(norm_flow_old),                    # [E_old, 1]

        # 新图结构与特征
        edge_index_new=torch.from_numpy(ei_new).long(),              # [2, E_new]
        edge_attr_new=torch.from_numpy(norm_attr_new),               # [E_new, 3]

        # Ground Truth
        y=torch.from_numpy(norm_flow_new),                           # [E_new, 1]

        # 物理约束掩码
        non_centroid_mask=non_centroid_mask,                          # [24]，bool

        # 元信息（用于调试和统计分析，不参与前向计算）
        num_nodes=NUM_NODES,
        num_edges_old=len(pair['edge_list_old']),
        num_edges_new=len(pair['edge_list_new']),
        mutation_type=pair['mutation_type'],
    )

    return data


# ============================================================
# 数据集构建主函数
# ============================================================

def build_full_dataset(
    pairs: list,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    attr_scaler: StandardScaler,
    flow_scaler: StandardScaler,
) -> tuple:
    """
    对三个数据集分区（train/val/test）分别构建 PyG Data 对象列表。

    val 和 test 分区使用与 train 完全相同的 scaler（即训练集拟合的 scaler），
    这是防止数据泄露的标准机器学习规范。

    Args:
        pairs:       全量数据对列表
        train_idx:   训练集索引
        val_idx:     验证集索引
        test_idx:    测试集索引
        attr_scaler: 已在训练集上 fit 的属性 scaler
        flow_scaler: 已在训练集上 fit 的流量 scaler

    Returns:
        train_dataset: list[Data]
        val_dataset:   list[Data]
        test_dataset:  list[Data]
    """
    def _build_split(idx_arr: np.ndarray, split_name: str) -> list:
        """内部辅助：对给定索引列表构建 Data 对象。"""
        dataset = []
        for idx in tqdm(idx_arr, desc=f"  构建 {split_name} 集"):
            data = build_single_data_object(pairs[idx], attr_scaler, flow_scaler)
            dataset.append(data)
        return dataset

    print(f"\n  构建训练集（{len(train_idx)} 个样本）...")
    train_dataset = _build_split(train_idx, 'Train')

    print(f"\n  构建验证集（{len(val_idx)} 个样本）...")
    val_dataset = _build_split(val_idx, 'Val')

    print(f"\n  构建测试集（{len(test_idx)} 个样本）...")
    test_dataset = _build_split(test_idx, 'Test')

    return train_dataset, val_dataset, test_dataset


def split_indices(
    num_samples: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    生成打乱后的 train/val/test 索引划分。

    先打乱全部索引（保证随机性），再按比例切分（保证精确的计数）。
    test_ratio = 1 - train_ratio - val_ratio，不传入是为了避免浮点误差。

    Args:
        num_samples: 总样本数
        train_ratio: 训练集比例（默认 0.6）
        val_ratio:   验证集比例（默认 0.2）
        seed:        随机种子

    Returns:
        train_idx, val_idx, test_idx: np.ndarray of int
    """
    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(num_samples)  # 打乱

    n_train = int(num_samples * train_ratio)
    n_val   = int(num_samples * val_ratio)
    # test 取余量，自动吸收 int() 截断误差
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train : n_train + n_val]
    test_idx  = all_idx[n_train + n_val:]

    return train_idx, val_idx, test_idx


# ============================================================
# 统计与验证
# ============================================================

def print_dataset_stats(
    train_dataset: list,
    val_dataset: list,
    test_dataset: list,
) -> None:
    """
    打印三个数据集分区的统计摘要，用于快速验证数据质量。

    覆盖以下信息：
    - 样本数量与 G'/G 边数分布（验证拓扑变化是否生效）
    - 归一化后 y（flows_new）的数值范围（验证 scaler 效果）
    - 变异类型分布（验证 40/30/30 比例是否保持）
    """
    from collections import Counter

    def _stats_one_split(dataset: list, name: str):
        e_old = [d.num_edges_old for d in dataset]
        e_new = [d.num_edges_new for d in dataset]
        y_all = torch.cat([d.y for d in dataset]).numpy()
        mutation_dist = Counter(d.mutation_type for d in dataset)

        print(f"\n  [{name}] {len(dataset)} 个样本")
        print(f"    G  边数: 固定 {e_old[0]}")
        print(f"    G' 边数: min={min(e_new)}, max={max(e_new)}, mean={np.mean(e_new):.1f}")
        print(f"    y (归一化 flows_new): "
              f"min={y_all.min():.3f}, max={y_all.max():.3f}, "
              f"mean={y_all.mean():.3f}, std={y_all.std():.3f}")
        print(f"    变异类型分布: {dict(mutation_dist)}")

    print(f"\n{'='*60}")
    print("数据集统计摘要")
    print(f"{'='*60}")
    _stats_one_split(train_dataset, 'Train')
    _stats_one_split(val_dataset,   'Val')
    _stats_one_split(test_dataset,  'Test')


def validate_single_data_object(data: Data) -> None:
    """
    对单个 Data 对象执行字段完整性和数值有效性的断言检查。

    在调试阶段调用此函数（每个 split 抽查第一个样本），
    生产环境可以略过以节省时间。

    检查项：
    - 所有必需字段存在
    - 形状与预期一致
    - 无 NaN/Inf
    - edge_index 中节点编号在 [0, 23] 范围内
    - non_centroid_mask 内容正确
    """
    E_old = data.num_edges_old
    E_new = data.num_edges_new

    # 必需字段形状检查
    assert data.x.shape            == (NUM_NODES, 1),    f"x 形状错误: {data.x.shape}"
    assert data.edge_index_old.shape == (2, E_old),      f"edge_index_old 形状错误"
    assert data.edge_index_new.shape == (2, E_new),      f"edge_index_new 形状错误"
    assert data.edge_attr_old.shape  == (E_old, 3),      f"edge_attr_old 形状错误"
    assert data.flow_old.shape       == (E_old, 1),      f"flow_old 形状错误"
    assert data.edge_attr_new.shape  == (E_new, 3),      f"edge_attr_new 形状错误"
    assert data.y.shape              == (E_new, 1),      f"y 形状错误"
    assert data.non_centroid_mask.shape == (NUM_NODES,), f"non_centroid_mask 形状错误"

    # 节点 ID 范围检查（必须在 [0, NUM_NODES-1] 内）
    for name, ei in [('edge_index_old', data.edge_index_old),
                     ('edge_index_new', data.edge_index_new)]:
        assert ei.min() >= 0,            f"{name} 存在负节点 ID"
        assert ei.max() < NUM_NODES,     f"{name} 节点 ID 超出范围（max={ei.max()}）"

    # 数值有效性（NaN/Inf）
    for name, tensor in [('edge_attr_old', data.edge_attr_old),
                         ('flow_old',      data.flow_old),
                         ('edge_attr_new', data.edge_attr_new),
                         ('y',             data.y)]:
        assert not torch.isnan(tensor).any(), f"{name} 含 NaN"
        assert not torch.isinf(tensor).any(), f"{name} 含 Inf"

    # x 必须全为 1（确认 OD 数据没有被误写入）
    assert torch.all(data.x == 1.0), "x 不是全 1 占位符，请检查是否误写入 OD 数据！"

    # non_centroid_mask 验证：前 11 个必须是 False（质心），后 13 个必须是 True（非质心）
    assert not data.non_centroid_mask[:11].any(), "前 11 个节点（质心）应为 False"
    assert data.non_centroid_mask[11:].all(),     "后 13 个节点（非质心）应为 True"


# ============================================================
# 主流程
# ============================================================

def run(args) -> None:
    """
    完整的 PyG 数据集构建流程：

      Step 1 — 加载 pkl 数据对
      Step 2 — 划分 train/val/test 索引
      Step 3 — 拟合 StandardScaler（仅训练集）
      Step 4 — 构建并归一化三个分区的 Data 对象
      Step 5 — 验证样本正确性（每个分区抽查首个样本）
      Step 6 — 打印统计摘要
      Step 7 — 保存 .pt 文件和 scaler

    Args:
        args: argparse.Namespace，见 parse_args()
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1：加载数据对 ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 1 — 加载 (G, G') 数据对")
    print(f"{'='*60}")
    print(f"  文件路径: {args.input_pkl}")

    with open(args.input_pkl, 'rb') as f:
        payload = pickle.load(f)

    # solve_network_pairs.py 保存格式：{'pairs': list, 'failed_indices': list}
    # 兼容旧格式（直接保存 list 的情况）
    if isinstance(payload, dict):
        pairs = payload['pairs']
        print(f"  废弃样本数（求解失败）: {len(payload.get('failed_indices', []))}")
    else:
        pairs = payload

    num_samples = len(pairs)
    print(f"  有效数据对数量: {num_samples}")

    # 快速打印第一个 pair 的字段，方便确认输入格式正确
    first = pairs[0]
    print(f"\n  第一个 pair 字段:")
    print(f"    od_matrix     : {first['od_matrix'].shape}")
    print(f"    G  边数       : {len(first['edge_list_old'])}")
    print(f"    G' 边数       : {len(first['edge_list_new'])}")
    print(f"    flows_old     : {first['flows_old'].shape}")
    print(f"    flows_new     : {first['flows_new'].shape}")
    print(f"    mutation_type : {first['mutation_type']}")

    # ── Step 2：索引划分 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2 — 划分 Train / Val / Test 索引（60/20/20）")
    print(f"{'='*60}")

    train_idx, val_idx, test_idx = split_indices(
        num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  Train : {len(train_idx)} 样本 ({len(train_idx)/num_samples*100:.1f}%)")
    print(f"  Val   : {len(val_idx)}   样本 ({len(val_idx)/num_samples*100:.1f}%)")
    print(f"  Test  : {len(test_idx)}  样本 ({len(test_idx)/num_samples*100:.1f}%)")

    # ── Step 3：拟合 Scaler（仅训练集）───────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3 — 拟合 StandardScaler（训练集，防止数据泄露）")
    print(f"{'='*60}")

    attr_scaler, flow_scaler = fit_scalers(pairs, train_idx)
    save_scalers(attr_scaler, flow_scaler, args.output_dir)

    # ── Step 4：构建 PyG Data 对象 ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4 — 构建 PyG Data 对象")
    print(f"{'='*60}")

    train_dataset, val_dataset, test_dataset = build_full_dataset(
        pairs, train_idx, val_idx, test_idx, attr_scaler, flow_scaler
    )

    # ── Step 5：正确性验证（每个分区抽查第一个样本）──────────────────────
    print(f"\n{'='*60}")
    print("Step 5 — 正确性验证")
    print(f"{'='*60}")

    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        try:
            validate_single_data_object(dataset[0])
            print(f"  [{name}] 首个样本验证通过 ✓")
            _print_data_summary(dataset[0], name)
        except AssertionError as e:
            print(f"  [{name}] 验证失败: {e}")
            raise

    # ── Step 6：打印统计摘要 ──────────────────────────────────────────────
    print_dataset_stats(train_dataset, val_dataset, test_dataset)

    # ── Step 7：保存 .pt 文件 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 7 — 保存数据集")
    print(f"{'='*60}")

    _save_split(train_dataset, os.path.join(args.output_dir, 'train_dataset.pt'), 'Train')
    _save_split(val_dataset,   os.path.join(args.output_dir, 'val_dataset.pt'),   'Val')
    _save_split(test_dataset,  os.path.join(args.output_dir, 'test_dataset.pt'),  'Test')

    print(f"\n  所有数据集已保存至: {args.output_dir}/")
    print(f"  Scaler 已保存至:    {args.output_dir}/scalers/")


def _print_data_summary(data: Data, split_name: str) -> None:
    """打印单个 Data 对象的字段形状摘要（调试用）。"""
    print(f"\n  [{split_name}] 首个样本字段摘要:")
    print(f"    x                  : {tuple(data.x.shape)}")
    print(f"    edge_index_old     : {tuple(data.edge_index_old.shape)}")
    print(f"    edge_attr_old      : {tuple(data.edge_attr_old.shape)}")
    print(f"    flow_old           : {tuple(data.flow_old.shape)}")
    print(f"    edge_index_new     : {tuple(data.edge_index_new.shape)}")
    print(f"    edge_attr_new      : {tuple(data.edge_attr_new.shape)}")
    print(f"    y                  : {tuple(data.y.shape)}")
    print(f"    non_centroid_mask  : {tuple(data.non_centroid_mask.shape)} "
          f"(True count: {data.non_centroid_mask.sum()})")
    print(f"    mutation_type      : {data.mutation_type}")


def _save_split(dataset: list, path: str, name: str) -> None:
    """保存 PyG dataset list 到 .pt 文件。"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(dataset, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  {name:6s}: {len(dataset):5d} 样本 → {path} ({size_mb:.1f} MB)")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='构建 (G, G\') 网络对的 PyG 数据集',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_pkl', type=str,
                        default='processed_data/pairs/network_pairs_dataset.pkl',
                        help='solve_network_pairs.py 输出的 pkl 文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='processed_data/pyg_dataset',
                        help='输出目录（存放 .pt 文件和 scaler）')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例（test = 1 - train - val）')
    parser.add_argument('--seed', type=int, default=42,
                        help='划分随机种子')
    return parser.parse_args()


def main():
    print("\n" + "=" * 60)
    print("  Phase 1 最后一步：构建 PyG 网络对数据集")
    print("=" * 60)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    print(f"\n  运行参数:")
    for k, v in vars(args).items():
        print(f"    {k:20s}: {v}")

    try:
        run(args)
    except KeyboardInterrupt:
        print("\n\n  用户中断，退出。")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\n  [错误] 数据集构建失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
