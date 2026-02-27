"""
生成网络重构场景：(G, G') 网络对数据

此模块完成以下任务：
1. 使用拉丁超立方采样 (LHS) 生成基础场景的 OD 矩阵和 G 的边属性
2. 实现三种网络变异操作（加边、删边、改属性）
3. 按照 40%/30%/30% 分布生成变异类型，构建 (G, G') 网络对

严格约束：
- G 和 G' 共享完全相同的 OD 矩阵（OD 仅用于 SUE 求解，绝不作为节点特征）
- G 和 G' 的节点集完全一致（Sioux Falls 的 24 个节点），只有边和边属性发生变化
- 变异类型分布：40% 仅拓扑变化，30% 仅属性变化，30% 两者都变化
"""

import numpy as np
import networkx as nx
from scipy.stats import qmc
from copy import deepcopy
from tqdm import tqdm


# ============================================================
# Part 1: 基础场景生成（LHS 采样）
# ============================================================

def generate_lhs_base_scenarios(
    num_samples: int = 2000,
    num_centroids: int = 11,
    num_edges: int = 76,
    seed: int = 42
) -> tuple:
    """
    使用拉丁超立方采样（LHS）生成基础场景的 OD 矩阵和 G 的边属性。

    参数范围（来自论文规格）：
    - OD 需求：0 - 1500 辆/OD对
    - 容量 (capacity)：4000 - 26000
    - 速度 (speed)：45 - 80 km/h

    Args:
        num_samples:   生成场景数量（默认 2000）
        num_centroids: 质心节点数（Sioux Falls 为 11）
        num_edges:     边数量（Sioux Falls 为 76）
        seed:          随机种子

    Returns:
        od_matrices: np.ndarray [num_samples, num_centroids, num_centroids]
        capacities:  np.ndarray [num_samples, num_edges]   -- G 的容量
        speeds:      np.ndarray [num_samples, num_edges]   -- G 的速度
    """
    print(f"\n{'='*60}")
    print(f"使用 LHS 采样生成 {num_samples} 个基础场景")
    print(f"{'='*60}")

    num_od_pairs = num_centroids * num_centroids          # 11*11 = 121
    num_dims = num_od_pairs + num_edges * 2               # 121 + 76 + 76 = 273

    sampler = qmc.LatinHypercube(d=num_dims, seed=seed)
    samples = sampler.random(n=num_samples)               # [num_samples, 273]

    # 按维度切分
    od_raw  = samples[:, :num_od_pairs]                   # [N, 121]
    cap_raw = samples[:, num_od_pairs : num_od_pairs + num_edges]   # [N, 76]
    spd_raw = samples[:, num_od_pairs + num_edges:]       # [N, 76]

    # 映射到实际范围
    od_matrices = (od_raw * 1500.0).reshape(num_samples, num_centroids, num_centroids)
    capacities  = cap_raw * (26000 - 4000) + 4000
    speeds      = spd_raw * (80 - 45) + 45

    print(f"  OD 矩阵形状:  {od_matrices.shape}")
    print(f"  容量矩阵形状: {capacities.shape}")
    print(f"  速度矩阵形状: {speeds.shape}")
    print(f"  OD 范围: [{od_matrices.min():.1f}, {od_matrices.max():.1f}]")
    print(f"  容量范围: [{capacities.min():.1f}, {capacities.max():.1f}]")
    print(f"  速度范围: [{speeds.min():.1f}, {speeds.max():.1f}]")
    print(f"  LHS 采样完成！")

    return od_matrices, capacities, speeds


# ============================================================
# Part 2: 三种拓扑变异算法
# ============================================================

def mutate_add_edges(
    G: nx.DiGraph,
    flows_old: np.ndarray,
    rng: np.random.Generator,
    top_k_range: tuple = (5, 10),
    edges_per_node_range: tuple = (1, 3)
) -> tuple:
    """
    变异操作一：加边
    在 G 中流量最高的前 top_k 个节点上添加新边。

    算法逻辑：
    1. 计算每个节点的流量重要性：将与该节点相关的所有有向边的流量求和，
       得到节点级别的流量强度 node_flow_sum[v]。
    2. 随机选择 top_k（在 [top_k_range[0], top_k_range[1]] 内均匀采样），
       取流量最高的 top_k 个节点作为候选添加边的源节点。
    3. 对每个候选节点，随机添加 1-3 条出边：
       - 目标节点：从 G 中所有节点里排除自身和已有出边目标，均匀随机采样
       - 新边属性：在现有边属性的 [min, max] 范围内均匀采样，保持属性合理性
       - 自动计算 free_flow_time = (length / speed) * 60

    Args:
        G:                  当前场景的 NetworkX DiGraph（含完整边属性）
        flows_old:          np.ndarray [num_edges_G]，按 list(G.edges()) 顺序索引
        rng:                numpy 随机数生成器
        top_k_range:        随机选取 top_k 节点的范围（包含两端）
        edges_per_node_range: 每个高流量节点添加边数的范围（包含两端）

    Returns:
        G_new:       添加了新边的 DiGraph
        added_edges: list of (u, v) 新增的有向边
    """
    edges    = list(G.edges())
    node_ids = list(G.nodes())   # Sioux Falls: 节点 1-24

    # --- 步骤1：计算节点流量强度 ---
    # 对每个节点，累加其所有入边和出边上的流量
    node_flow_sum = {n: 0.0 for n in node_ids}
    for idx, (u, v) in enumerate(edges):
        node_flow_sum[u] += flows_old[idx]
        node_flow_sum[v] += flows_old[idx]

    # --- 步骤2：选取前 top_k 个高流量节点 ---
    top_k = int(rng.integers(top_k_range[0], top_k_range[1] + 1))
    sorted_nodes = sorted(node_ids, key=lambda n: node_flow_sum[n], reverse=True)
    top_nodes = sorted_nodes[:top_k]

    # --- 步骤3：为每个高流量节点添加 1-3 条新有向边 ---
    G_new = deepcopy(G)
    added_edges = []

    # 计算现有边属性范围，新边在此范围内均匀采样
    all_caps = [G[u][v]['capacity']      for u, v in edges]
    all_spds = [G[u][v]['speed']         for u, v in edges]
    all_lens = [G[u][v]['length']        for u, v in edges]
    cap_min, cap_max = min(all_caps), max(all_caps)
    spd_min, spd_max = min(all_spds), max(all_spds)
    len_min, len_max = min(all_lens), max(all_lens)

    for node in top_nodes:
        # 排除自身和已经存在出边的目标节点（避免重复边）
        existing_targets = set(G_new.successors(node))
        candidates = [n for n in node_ids if n != node and n not in existing_targets]

        if not candidates:
            # 该节点已经连接了所有其他节点，跳过
            continue

        num_to_add = int(rng.integers(edges_per_node_range[0], edges_per_node_range[1] + 1))
        num_to_add = min(num_to_add, len(candidates))

        # 从候选节点中随机选取目标
        targets = rng.choice(candidates, size=num_to_add, replace=False)

        for target in targets:
            # 新边属性：在现有边属性的 [min, max] 范围内均匀采样
            new_cap = float(rng.uniform(cap_min, cap_max))
            new_spd = float(rng.uniform(spd_min, spd_max))
            new_len = float(rng.uniform(len_min, len_max))
            # free_flow_time (分钟) = length (km) / speed (km/h) * 60
            new_fft = (new_len / new_spd) * 60.0

            G_new.add_edge(
                int(node), int(target),
                capacity=new_cap,
                speed=new_spd,
                length=new_len,
                free_flow_time=new_fft
            )
            added_edges.append((int(node), int(target)))

    return G_new, added_edges


def mutate_delete_edges(
    G: nx.DiGraph,
    rng: np.random.Generator,
    num_delete_range: tuple = (5, 10)
) -> tuple:
    """
    变异操作二：删边
    随机删除 G 中的 5-10 条有向边，同时保证图的弱连通性。

    算法逻辑：
    1. 随机确定目标删除数量 num_delete ∈ [5, 10]
    2. 对所有边进行随机排列（打乱顺序），逐边尝试删除：
       - 临时删除该边
       - 检查图是否仍然弱连通（weakly connected）
         * 弱连通：将所有有向边视为无向边后，任意两节点间存在路径
         * 这是保证 SUE 可解性的最低要求
       - 若连通：确认删除，累计到已删除列表
       - 若不连通：恢复该边（将其重新加回，含所有原始属性）
    3. 达到目标删除数量后停止

    注意：遍历完所有边后若仍未达到目标数量，以实际删除数量为准（不强制报错）

    Args:
        G:                当前场景的 NetworkX DiGraph
        rng:              numpy 随机数生成器
        num_delete_range: 删除边数的范围（包含两端）

    Returns:
        G_new:         删除边后的 DiGraph
        deleted_edges: list of (u, v) 被删除的有向边
    """
    edges = list(G.edges())
    num_delete = int(rng.integers(num_delete_range[0], num_delete_range[1] + 1))

    G_new = deepcopy(G)
    deleted_edges = []

    # 随机打乱边的遍历顺序，保证删除结果的随机性
    shuffled_indices = rng.permutation(len(edges))

    for idx in shuffled_indices:
        if len(deleted_edges) >= num_delete:
            break

        u, v = edges[idx]

        # 临时删除该边
        edge_data = dict(G_new[u][v])  # 保存完整属性，用于可能的恢复
        G_new.remove_edge(u, v)

        # 检查强连通性：有向图中任意两节点间均存在有向路径
        # 必须使用强连通（is_strongly_connected）而非弱连通（is_weakly_connected）！
        # 弱连通仅保证"忽略方向后"连通，无法阻止单向死胡同的出现。
        # SUE 求解器（Frank-Wolfe）依赖有向最短路，若某 OD 对间无有向路径，
        # nx.shortest_path 会抛出 NetworkXNoPath 异常，导致整批求解崩溃。
        # Sioux Falls 基础网络是强连通的，删边后必须保持这一性质。
        if nx.is_strongly_connected(G_new):
            # 删除合法，确认保留
            deleted_edges.append((u, v))
        else:
            # 删除此边会破坏强连通性，恢复该边及其所有原始属性
            G_new.add_edge(u, v, **edge_data)

    return G_new, deleted_edges


def mutate_attributes(
    G: nx.DiGraph,
    rng: np.random.Generator,
    cap_scale_range: tuple = (0.3, 2.0),
    spd_scale_range: tuple = (0.3, 2.0)
) -> tuple:
    """
    变异操作三：改属性
    对 G 中每条边独立地随机缩放容量（capacity）和速度（speed）。

    算法逻辑：
    1. 遍历 G 的每一条有向边 (u, v)
    2. 独立采样容量缩放因子 λ_cap ~ Uniform(0.3, 2.0)
       独立采样速度缩放因子 λ_spd ~ Uniform(0.3, 2.0)
    3. 更新属性：
       new_capacity      = old_capacity * λ_cap
       new_speed         = old_speed    * λ_spd
       new_free_flow_time = (length / new_speed) * 60   （根据新速度重新计算）
    4. 返回修改后的图和每条边的缩放记录

    注意：length 不变（物理距离不受重构影响）；free_flow_time 依赖速度需重新计算。

    Args:
        G:               当前场景的 NetworkX DiGraph
        rng:             numpy 随机数生成器
        cap_scale_range: 容量缩放因子的范围（包含两端）
        spd_scale_range: 速度缩放因子的范围（包含两端）

    Returns:
        G_new:       属性修改后的 DiGraph
        attr_changes: dict {(u, v): {'cap_scale': float, 'spd_scale': float}}
    """
    G_new = deepcopy(G)
    attr_changes = {}

    for u, v in list(G_new.edges()):
        cap_scale = float(rng.uniform(cap_scale_range[0], cap_scale_range[1]))
        spd_scale = float(rng.uniform(spd_scale_range[0], spd_scale_range[1]))

        old_cap = G_new[u][v]['capacity']
        old_spd = G_new[u][v]['speed']
        old_len = G_new[u][v]['length']   # 物理距离不变

        new_cap = old_cap * cap_scale
        new_spd = old_spd * spd_scale
        # free_flow_time (分钟) = length (km) / speed (km/h) * 60
        new_fft = (old_len / new_spd) * 60.0

        G_new[u][v]['capacity']       = new_cap
        G_new[u][v]['speed']          = new_spd
        G_new[u][v]['free_flow_time'] = new_fft

        attr_changes[(u, v)] = {'cap_scale': cap_scale, 'spd_scale': spd_scale}

    return G_new, attr_changes


# ============================================================
# Part 3: 构建单一场景的具体 G 和生成 G'
# ============================================================

def build_scenario_graph(G_topo: nx.DiGraph, capacities_i: np.ndarray, speeds_i: np.ndarray) -> nx.DiGraph:
    """
    给 Sioux Falls 基础拓扑（G_topo）赋予第 i 个场景的具体边属性，
    返回该场景的具体 G_i（可直接传入 SUE 求解器）。

    边的顺序与 list(G_topo.edges()) 完全一致，
    因此 capacities_i[j] 对应 list(G_topo.edges())[j]。

    Args:
        G_topo:       Sioux Falls 基础有向图（含 length 属性）
        capacities_i: np.ndarray [num_edges_G]，第 i 场景下 G 的边容量
        speeds_i:     np.ndarray [num_edges_G]，第 i 场景下 G 的边速度

    Returns:
        G_i: 含具体属性的 NetworkX DiGraph
    """
    G_i = deepcopy(G_topo)
    edges = list(G_topo.edges())

    for j, (u, v) in enumerate(edges):
        length = G_topo[u][v]['length']         # 物理距离不变
        spd    = float(speeds_i[j])
        cap    = float(capacities_i[j])
        fft    = (length / spd) * 60.0          # 分钟

        G_i[u][v]['capacity']       = cap
        G_i[u][v]['speed']          = spd
        G_i[u][v]['free_flow_time'] = fft

    return G_i


def _apply_topology_mutation(
    G: nx.DiGraph,
    flows_old: np.ndarray,
    rng: np.random.Generator,
    mutation_info: dict
) -> nx.DiGraph:
    """
    内部辅助函数：对 G 施加拓扑变异（加边、删边，或两者都做）。

    拓扑变异的子操作以等概率随机选择：
    - 'add'   (33%)：仅加边
    - 'delete'(33%)：仅删边
    - 'both'  (34%)：先加边后删边

    Args:
        G:           当前图
        flows_old:   G 上的历史流量（用于加边时识别高流量节点）
        rng:         numpy 随机数生成器
        mutation_info: 用于记录变异详情的字典（原地修改）

    Returns:
        G_mutated: 拓扑变异后的图
    """
    # 注意：此处 flows_old 的索引依赖于 list(G.edges()) 的顺序，
    # 必须在拓扑不变的 G（即 G_scenario）上调用此函数，不能在已变异的图上再次调用。
    topo_op = rng.choice(['add', 'delete', 'both'])

    G_mut = G
    if topo_op in ('add', 'both'):
        G_mut, added_edges = mutate_add_edges(G_mut, flows_old, rng)
        mutation_info['added_edges'] = added_edges

    if topo_op in ('delete', 'both'):
        # 删边时 flows_old 已经是针对原 G 的，删边不改变 flows_old 索引含义
        G_mut, deleted_edges = mutate_delete_edges(G_mut, rng)
        mutation_info['deleted_edges'] = deleted_edges

    mutation_info['topo_op'] = str(topo_op)
    return G_mut


# ============================================================
# Part 4: 网络对生成主函数
# ============================================================

def generate_network_pairs(
    G_topo: nx.DiGraph,
    od_matrices: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    flows_old: np.ndarray,
    seed: int = 42
) -> list:
    """
    主函数：为每个基础场景生成对应的变异网络 G'，构建完整的 (G, G') 对列表。

    变异类型分布（按 .cursorrules 规格严格执行）：
    - 40%  topology_only  : 仅拓扑变化（加/删边）
    - 30%  attribute_only : 仅属性变化（容量/速度缩放 0.3x-2.0x）
    - 30%  both           : 先拓扑变化，再属性变化

    每个返回元素的数据结构（scenario_pair dict）：
    ```
    {
        'od_matrix'     : np.ndarray [11, 11],  # 共享 OD 矩阵（仅用于 SUE，非模型输入！）
        'G'             : nx.DiGraph,           # 场景 G（固定拓扑 + LHS 属性）
        'G_prime'       : nx.DiGraph,           # 变异网络 G'（可能有不同边集和属性）
        'mutation_type' : str,                  # 'topology_only'|'attribute_only'|'both'
        'mutation_info' : dict,                 # 变异详情：added/deleted edges，attr scales
    }
    ```

    节点集一致性保证：
    - G 和 G' 的节点集完全相同（Sioux Falls 的 24 个节点：1-24）
    - 节点从不增加或删除，只有边集和边属性发生变化

    Args:
        G_topo:      Sioux Falls 基础有向图（从 .tntp 文件加载，含 length 属性）
        od_matrices: np.ndarray [N, 11, 11]，LHS 生成的 OD 矩阵
        capacities:  np.ndarray [N, 76]，LHS 生成的 G 的容量
        speeds:      np.ndarray [N, 76]，LHS 生成的 G 的速度
        flows_old:   np.ndarray [N, 76]，在 G 上运行 SUE 得到的历史流量
                     （用于加边变异时定位高流量节点）
        seed:        随机种子

    Returns:
        scenario_pairs: list of dict，长度 = N
    """
    num_samples = od_matrices.shape[0]
    rng = np.random.default_rng(seed)

    # 按照 40/30/30 分布精确分配变异类型，采用"精确计数 + 洗牌"策略。
    # 不使用概率采样（rng.choice with p=[...]），因为概率采样存在随机波动，
    # 无法保证论文中声明的严格 40/30/30 比例完全精确（可复现）。
    # 策略：先构造精确数量的标签数组，再原地随机打乱其顺序。
    # num_both 使用剩余数量，自动吸收 int() 截断产生的 1 个样本误差。
    num_topo = int(num_samples * 0.40)
    num_attr = int(num_samples * 0.30)
    num_both = num_samples - num_topo - num_attr   # 剩余全归 both，处理截断误差

    mutation_types = np.array(
        ['topology_only'] * num_topo +
        ['attribute_only'] * num_attr +
        ['both'] * num_both,
        dtype=object
    )
    rng.shuffle(mutation_types)  # 原地随机打乱，保证场景顺序无系统性偏差

    print(f"\n{'='*60}")
    print(f"生成 {num_samples} 个 (G, G') 网络对")
    print(f"{'='*60}")
    count = {t: int(np.sum(mutation_types == t)) for t in ['topology_only', 'attribute_only', 'both']}
    print(f"  变异类型分布（精确计数）:")
    print(f"    仅拓扑变化 (topology_only) : {count['topology_only']} ({count['topology_only']/num_samples*100:.1f}%)")
    print(f"    仅属性变化 (attribute_only): {count['attribute_only']} ({count['attribute_only']/num_samples*100:.1f}%)")
    print(f"    拓扑+属性都变 (both)       : {count['both']} ({count['both']/num_samples*100:.1f}%)")

    scenario_pairs = []
    num_nodes = G_topo.number_of_nodes()

    for i in tqdm(range(num_samples), desc="  生成网络对"):
        mutation_type = mutation_types[i]
        mutation_info = {'type': mutation_type}

        # --- 步骤1：构建第 i 个场景的具体 G（固定拓扑 + LHS 属性）---
        G_i = build_scenario_graph(G_topo, capacities[i], speeds[i])

        # --- 步骤2：以 G_i 为起点生成 G' ---
        G_prime = deepcopy(G_i)

        if mutation_type == 'topology_only':
            # 仅施加拓扑变异（加边/删边），不改变属性
            G_prime = _apply_topology_mutation(G_prime, flows_old[i], rng, mutation_info)

        elif mutation_type == 'attribute_only':
            # 仅施加属性变异，拓扑不变
            G_prime, attr_changes = mutate_attributes(G_prime, rng)
            mutation_info['attr_changes_count'] = len(attr_changes)  # 不存储全量避免内存爆炸

        elif mutation_type == 'both':
            # 先拓扑变化，再对变化后的图施加属性变化
            # 注意：先拓扑后属性，属性变化会覆盖所有边（包括新加的边）
            G_prime = _apply_topology_mutation(G_prime, flows_old[i], rng, mutation_info)
            G_prime, attr_changes = mutate_attributes(G_prime, rng)
            mutation_info['attr_changes_count'] = len(attr_changes)

        # --- 步骤3：节点集一致性验证 ---
        # G 和 G' 的节点集必须完全一致，只有边集变化
        assert set(G_i.nodes()) == set(G_prime.nodes()), (
            f"场景 {i}: G 和 G' 的节点集不一致！G: {set(G_i.nodes())}，G': {set(G_prime.nodes())}"
        )
        assert G_prime.number_of_nodes() == num_nodes, (
            f"场景 {i}: G' 节点数 {G_prime.number_of_nodes()} 不等于预期 {num_nodes}！"
        )

        scenario_pairs.append({
            'od_matrix'    : od_matrices[i].copy(),   # [11, 11]，OD 矩阵（仅 SUE 用）
            'G'            : G_i,                     # 场景 G（NetworkX DiGraph）
            'G_prime'      : G_prime,                 # 变异网络 G'（NetworkX DiGraph）
            'mutation_type': mutation_type,
            'mutation_info': mutation_info,
        })

    # --- 汇总统计 ---
    edge_counts_G      = [len(list(p['G'].edges()))       for p in scenario_pairs]
    edge_counts_Gprime = [len(list(p['G_prime'].edges())) for p in scenario_pairs]

    print(f"\n  生成完成！")
    print(f"  G  边数: 固定 {edge_counts_G[0]} 条（所有场景一致）")
    print(f"  G' 边数: min={min(edge_counts_Gprime)}, "
          f"max={max(edge_counts_Gprime)}, "
          f"mean={np.mean(edge_counts_Gprime):.1f}")

    return scenario_pairs


# ============================================================
# Part 5: 存取工具函数
# ============================================================

def save_scenarios(od_matrices, capacities, speeds, save_path='processed_data/raw/base_scenarios.npz'):
    """
    保存 LHS 基础场景数据（numpy 格式，紧凑压缩）。

    Args:
        od_matrices: [N, 11, 11]
        capacities:  [N, 76]
        speeds:      [N, 76]
        save_path:   目标文件路径（.npz）
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, od_matrices=od_matrices, capacities=capacities, speeds=speeds)
    print(f"  基础场景已保存: {save_path}")


def load_scenarios(load_path='processed_data/raw/base_scenarios.npz'):
    """
    加载 LHS 基础场景数据。

    Returns:
        od_matrices, capacities, speeds
    """
    data = np.load(load_path)
    print(f"  基础场景已加载: {load_path}")
    return data['od_matrices'], data['capacities'], data['speeds']


def save_scenario_pairs(scenario_pairs: list, save_path='processed_data/raw/scenario_pairs.pkl'):
    """
    保存 (G, G') 对列表（含 NetworkX 图对象，使用 pickle 序列化）。

    注意：每个 scenario_pair 包含两个 NetworkX DiGraph，内存占用较大。
    对于大规模数据集（> 5000 样本），建议分批保存。

    Args:
        scenario_pairs: list of dict，长度 = N
        save_path:      目标文件路径（.pkl）
    """
    import os
    import pickle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scenario_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  网络对数据已保存: {save_path}（{len(scenario_pairs)} 对）")


def load_scenario_pairs(load_path='processed_data/raw/scenario_pairs.pkl') -> list:
    """
    加载 (G, G') 对列表。

    Returns:
        scenario_pairs: list of dict
    """
    import pickle
    with open(load_path, 'rb') as f:
        scenario_pairs = pickle.load(f)
    print(f"  网络对数据已加载: {load_path}（{len(scenario_pairs)} 对）")
    return scenario_pairs
