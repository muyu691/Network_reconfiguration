"""
双重流量求解流水线：(G, G') 网络对完整数据生成

完整执行步骤：
  Step 1 — LHS 采样         : 生成基础场景参数（OD 矩阵、容量、速度）
  Step 2 — 第一次 SUE 求解  : 在 G 上运行 Frank-Wolfe → flows_old [N, E_old]
  Step 3 — 生成网络对       : 调用 generate_network_pairs，传入 flows_old 定位高流量节点
  Step 4 — 第二次 SUE 求解  : 在每个 G' 上运行 Frank-Wolfe → flows_new [E_new_i]
  Step 5 — 保存结果         : 输出 .pkl 文件，供 build_pyg_data.py 使用

严格约束：
  - G 和 G' 共享完全相同的 OD 矩阵（OD 绝不作为模型节点特征）
  - G' 的边数/边序可能与 G 不同，flows_new 使用 list(G'.edges()) 索引
  - 第二次求解失败的样本（异常/NaN/Inf）会被跳过并计入废弃日志

输出数据结构（每个 completed_pair 为一个 dict）：
  {
    'od_matrix'    : np.ndarray [11, 11],   # 共享 OD（仅 SUE 用）
    'G'            : nx.DiGraph,            # 基础场景图（固定拓扑 + LHS 属性）
    'G_prime'      : nx.DiGraph,            # 变异网络
    'mutation_type': str,
    'mutation_info': dict,
    'flows_old'    : np.ndarray [E_old],    # G 上的均衡流量（按 edge_list_old 索引）
    'flows_new'    : np.ndarray [E_new],    # G' 上的均衡流量（按 edge_list_new 索引）
    'edge_list_old': list[tuple],           # list(G.edges())  ← 显式保存，消除歧义
    'edge_list_new': list[tuple],           # list(G'.edges()) ← 显式保存，消除歧义
  }

用法：
  python solve_network_pairs.py --num_samples 2000 --network_file ../sioux_data/SiouxFalls_net.tntp
  python solve_network_pairs.py --num_samples 2000 --skip_first_solve  # 断点续跑
"""

import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

# 兼容直接运行 (python solve_network_pairs.py) 和模块导入两种方式
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_sioux import load_sioux_falls_network
from generate_scenarios import (
    generate_lhs_base_scenarios,
    save_scenarios,
    load_scenarios,
    generate_network_pairs,
    save_scenario_pairs,
)
from sue_solver import frank_wolfe_sue, solve_sue_batch
from utils import compute_free_flow_times


# ============================================================
# 核心工具函数
# ============================================================

def _extract_graph_arrays(G) -> tuple:
    """
    从 NetworkX DiGraph 的边属性中直接提取 SUE 求解所需的两个数组。

    G' 的拓扑和属性已在 generate_network_pairs 中完整嵌入图对象，
    本函数按 list(G.edges()) 的顺序提取：
      - capacities      : [E] 每条边的通行能力
      - free_flow_times : [E] 每条边的自由流行驶时间（分钟）

    为什么不用 utils.compute_free_flow_times？
    该函数需要额外的 speeds 数组，而 G' 的 free_flow_time 在变异时已根据
    新速度重新计算并写入图属性，直接读取更准确且避免重复计算。

    Args:
        G: 含完整边属性（capacity, free_flow_time）的 NetworkX DiGraph

    Returns:
        edges:           list[(u, v)]，长度 E，定义 flows 的索引顺序
        capacities:      np.ndarray [E]
        free_flow_times: np.ndarray [E]
    """
    edges = list(G.edges())
    capacities      = np.array([G[u][v]['capacity']       for u, v in edges], dtype=np.float64)
    free_flow_times = np.array([G[u][v]['free_flow_time'] for u, v in edges], dtype=np.float64)
    return edges, capacities, free_flow_times


def solve_single_graph_sue(
    G,
    od_matrix: np.ndarray,
    max_iter: int = 100,
    convergence_threshold: float = 1e-4,
) -> np.ndarray:
    """
    对拓扑/属性任意的单个图 G 运行 Frank-Wolfe SUE 求解。

    与 solve_sue_batch 的区别：
    - solve_sue_batch 假设所有样本共享同一拓扑，属性通过 numpy 数组批量传入。
    - 本函数直接从图的边属性提取 capacities 和 free_flow_times，
      适用于 G'（每个 G' 可能有不同数量和顺序的边）。

    返回的 flows 索引顺序与 list(G.edges()) 完全一致。
    调用方应在调用前保存 list(G.edges()) 以便后续索引还原。

    Args:
        G:                    NetworkX DiGraph（含 capacity、free_flow_time 属性）
        od_matrix:            np.ndarray [num_centroids, num_centroids]
        max_iter:             Frank-Wolfe 最大迭代次数
        convergence_threshold: 收敛判据（相对间隙）

    Returns:
        flows: np.ndarray [E]，均衡流量，按 list(G.edges()) 索引

    Raises:
        ValueError: 若求解结果包含 NaN 或 Inf（视为求解失败）
    """
    edges, capacities, free_flow_times = _extract_graph_arrays(G)

    flows = frank_wolfe_sue(
        G,
        od_matrix,
        capacities,
        free_flow_times,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        verbose=False,
    )

    # 数值有效性检查：NaN 或 Inf 意味着求解发散或图存在结构问题
    if np.any(np.isnan(flows)) or np.any(np.isinf(flows)):
        raise ValueError(
            f"SUE 求解结果包含 NaN/Inf（图有 {G.number_of_edges()} 条边）。"
        )

    return flows


# ============================================================
# 第一次 SUE 求解（批量，固定拓扑 G）
# ============================================================

def run_first_sue_solve(
    G_topo,
    od_matrices: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    max_iter: int = 100,
    convergence_threshold: float = 1e-4,
) -> np.ndarray:
    """
    第一次 SUE 批量求解：在固定拓扑 G 上计算 flows_old。

    所有基础场景共享相同的 G_topo 拓扑，只有容量和速度不同，
    因此可复用现有的 solve_sue_batch 批量接口实现高效求解。

    flows_old[i] 按 list(G_topo.edges()) 的固定顺序索引，
    后续将用于 generate_network_pairs 中识别高流量节点。

    Args:
        G_topo:               基础拓扑图（来自 .tntp 文件，含 length 属性）
        od_matrices:          [N, 11, 11]
        capacities:           [N, 76]  ← 按 list(G_topo.edges()) 索引
        speeds:               [N, 76]  ← 按 list(G_topo.edges()) 索引
        max_iter:             每个场景的 Frank-Wolfe 最大迭代次数
        convergence_threshold: 收敛判据

    Returns:
        flows_old: np.ndarray [N, 76]，按 list(G_topo.edges()) 索引
    """
    num_samples = od_matrices.shape[0]

    print(f"\n{'='*60}")
    print(f"Step 2 — 第一次 SUE 批量求解（在 G 上，共 {num_samples} 个场景）")
    print(f"{'='*60}")

    # 使用 utils.compute_free_flow_times 计算自由流时间：
    # 该函数用 G_topo[u][v]['length'] 除以各场景速度，与 build_scenario_graph
    # 中的计算方式完全一致，保证结果可重现。
    free_flow_times = compute_free_flow_times(G_topo, speeds)  # [N, 76]

    edges_G = list(G_topo.edges())
    num_edges_G = len(edges_G)
    flows_old = np.zeros((num_samples, num_edges_G), dtype=np.float64)

    failed_count = 0
    for i in tqdm(range(num_samples), desc="  第一次 SUE 求解"):
        try:
            flows_i = frank_wolfe_sue(
                G_topo,
                od_matrices[i],
                capacities[i],
                free_flow_times[i],
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                verbose=False,
            )
            if np.any(np.isnan(flows_i)) or np.any(np.isinf(flows_i)):
                raise ValueError("flows_old 含 NaN/Inf")
            flows_old[i] = flows_i
        except Exception as e:
            # 基础 G 强连通，第一次求解失败极为罕见，记录后置零继续
            failed_count += 1
            flows_old[i] = 0.0
            if failed_count <= 5:  # 避免日志爆炸，只打印前 5 条
                print(f"  [警告] 场景 {i} 第一次求解失败: {e}")

    print(f"\n  第一次 SUE 求解完成！")
    print(f"  flows_old 形状: {flows_old.shape}")
    print(f"  flows_old 统计: min={flows_old.min():.1f}, max={flows_old.max():.1f}, "
          f"mean={flows_old.mean():.1f}")
    if failed_count:
        print(f"  [警告] 第一次求解失败场景数: {failed_count}（flows_old 置零）")

    return flows_old


# ============================================================
# 第二次 SUE 求解（逐图，G' 拓扑各异）
# ============================================================

def run_second_sue_solve(
    scenario_pairs: list,
    max_iter: int = 100,
    convergence_threshold: float = 1e-4,
    checkpoint_path: str = None,
    checkpoint_interval: int = 200,
) -> tuple:
    """
    第二次 SUE 求解：对每个 G' 独立求解，完成后合并成最终数据对。

    与第一次的关键区别：
    - 每个 G' 可能有不同的边数和边序（拓扑变异导致）
    - 必须从 G' 的图属性直接提取 capacity 和 free_flow_time
    - flows_new[i] 按 list(G_prime.edges()) 索引（与 flows_old 索引方案不同！）
    - edge_list_new[i] 必须显式保存，build_pyg_data.py 依赖它还原 edge_index_new

    失败处理策略：
    - 捕获所有异常（NetworkXNoPath、NaN/Inf、收敛失败等）
    - 失败样本记入 failed_indices 并跳过，不进入 completed_pairs
    - 在 Sioux Falls 强连通保证下，失败率应接近 0%

    Args:
        scenario_pairs:       generate_network_pairs 的输出，list of dict
                              每个 dict 含 'G', 'G_prime', 'od_matrix' 等字段
        max_iter:             Frank-Wolfe 最大迭代次数
        convergence_threshold: 收敛判据
        checkpoint_path:      中间结果保存路径（每隔 checkpoint_interval 个样本保存一次）
                              设为 None 则不保存检查点
        checkpoint_interval:  检查点保存间隔（样本数）

    Returns:
        completed_pairs: list of dict，每个 dict 包含完整的双图数据
        failed_indices:  list of int，求解失败的原始样本索引
    """
    num_pairs = len(scenario_pairs)

    print(f"\n{'='*60}")
    print(f"Step 4 — 第二次 SUE 求解（在各 G' 上，共 {num_pairs} 个图）")
    print(f"{'='*60}")
    if checkpoint_path:
        print(f"  检查点保存路径: {checkpoint_path}（每 {checkpoint_interval} 个样本）")

    completed_pairs = []
    failed_indices  = []

    for i, pair in enumerate(tqdm(scenario_pairs, desc="  第二次 SUE 求解")):
        G_prime   = pair['G_prime']
        od_matrix = pair['od_matrix']

        try:
            # --- 核心：对 G' 求解 SUE ---
            # G' 的边数/边序已与 G 不同（加/删边导致），
            # solve_single_graph_sue 直接从 G' 的图属性提取 capacity 和 free_flow_time，
            # 返回的 flows_new 按当前 list(G'.edges()) 排列。
            flows_new = solve_single_graph_sue(
                G_prime,
                od_matrix,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
            )

            # --- 构建完整数据对 ---
            # 显式保存 edge_list_old/new，消除后续 build_pyg_data.py 中的边序歧义。
            # 不能依赖重新调用 list(G.edges()) 来还原顺序，因为 pickle 反序列化后
            # 图对象的迭代顺序理论上可能与保存时不同（虽然 NetworkX 通常稳定）。
            edge_list_old = list(pair['G'].edges())
            edge_list_new = list(G_prime.edges())

            completed_pairs.append({
                'od_matrix'    : pair['od_matrix'],         # [11, 11]，仅 SUE 用
                'G'            : pair['G'],                 # NetworkX DiGraph
                'G_prime'      : G_prime,                   # NetworkX DiGraph
                'mutation_type': pair['mutation_type'],
                'mutation_info': pair['mutation_info'],
                'flows_old'    : pair['flows_old'],         # [E_old]，按 edge_list_old
                'flows_new'    : flows_new,                 # [E_new]，按 edge_list_new
                'edge_list_old': edge_list_old,             # list[(u,v)]，E_old 条
                'edge_list_new': edge_list_new,             # list[(u,v)]，E_new 条
            })

        except Exception as e:
            # 记录失败样本但不中断流水线
            failed_indices.append(i)
            if len(failed_indices) <= 10:  # 只打印前 10 条，避免日志爆炸
                print(f"\n  [跳过] 样本 {i} (mutation={pair['mutation_type']}): {e}")

        # 定期保存检查点（防止长时运行中途崩溃丢失进度）
        if (checkpoint_path is not None
                and (i + 1) % checkpoint_interval == 0
                and completed_pairs):
            _save_checkpoint(completed_pairs, failed_indices, checkpoint_path, i + 1)

    # 最终统计报告
    total   = num_pairs
    success = len(completed_pairs)
    failed  = len(failed_indices)

    print(f"\n  第二次 SUE 求解完成！")
    print(f"  总样本数 : {total}")
    print(f"  成功     : {success} ({success/total*100:.1f}%)")
    print(f"  失败跳过 : {failed}  ({failed/total*100:.1f}%)")

    if failed_indices:
        print(f"  失败样本索引（前20个）: {failed_indices[:20]}")

    # 打印完成样本中各变异类型的实际分布
    if completed_pairs:
        from collections import Counter
        dist = Counter(p['mutation_type'] for p in completed_pairs)
        print(f"\n  完成样本变异类型分布:")
        for t, cnt in sorted(dist.items()):
            print(f"    {t:20s}: {cnt:5d} ({cnt/success*100:.1f}%)")

    return completed_pairs, failed_indices


def _save_checkpoint(completed_pairs: list, failed_indices: list,
                     base_path: str, step: int) -> None:
    """
    保存流水线中间状态到检查点文件。

    文件名格式：{base_path}.ckpt_{step}.pkl
    例如：processed_data/raw/pairs.pkl.ckpt_400.pkl

    Args:
        completed_pairs: 当前已完成的数据对列表
        failed_indices:  当前已记录的失败索引列表
        base_path:       最终输出文件的路径（用于构造检查点文件名）
        step:            当前处理进度（已处理的样本总数）
    """
    ckpt_path = f"{base_path}.ckpt_{step}.pkl"
    os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else '.', exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'completed_pairs': completed_pairs, 'failed_indices': failed_indices}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  [检查点] 已保存 {len(completed_pairs)} 对到 {ckpt_path}")


# ============================================================
# 主流水线编排函数
# ============================================================

def run_pipeline(args) -> list:
    """
    完整的"生成 → 求解 → 重构 → 求解"流水线主控函数。

    执行顺序：
      Step 1 — 加载 Sioux Falls 基础拓扑
      Step 2 — LHS 采样基础场景（或从文件加载）
      Step 3 — 第一次 SUE 批量求解（或从文件加载 flows_old）
      Step 4 — 生成 (G, G') 网络对（传入 flows_old 以识别高流量节点）
      Step 5 — 第二次 SUE 逐图求解（G' 拓扑各异，逐一处理）
      Step 6 — 保存最终数据集

    断点续跑支持（--skip_first_solve 标志）：
      若 flows_old 已保存，可跳过 Step 2-3，直接从 Step 4 开始。

    Args:
        args: argparse.Namespace，见 parse_args()

    Returns:
        completed_pairs: 最终完成的数据对列表
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: 加载基础拓扑 ----
    print(f"\n{'='*60}")
    print("Step 1 — 加载 Sioux Falls 基础拓扑")
    print(f"{'='*60}")
    G_topo, centroids = load_sioux_falls_network(args.network_file)
    edges_G_topo = list(G_topo.edges())
    num_edges_G  = len(edges_G_topo)
    print(f"  节点数: {G_topo.number_of_nodes()}")
    print(f"  边数:   {num_edges_G}")
    print(f"  质心:   {centroids}")

    # ---- Step 2: LHS 采样基础场景 ----
    scenarios_path = os.path.join(args.output_dir, 'base_scenarios.npz')

    if args.skip_first_solve and os.path.exists(scenarios_path):
        print(f"\n{'='*60}")
        print("Step 2 — 加载已有基础场景（跳过 LHS 采样）")
        print(f"{'='*60}")
        od_matrices, capacities, speeds = load_scenarios(scenarios_path)
    else:
        print(f"\n{'='*60}")
        print("Step 2 — LHS 采样基础场景")
        print(f"{'='*60}")
        od_matrices, capacities, speeds = generate_lhs_base_scenarios(
            num_samples=args.num_samples,
            num_centroids=len(centroids),
            num_edges=num_edges_G,
            seed=args.seed,
        )
        save_scenarios(od_matrices, capacities, speeds, scenarios_path)

    # ---- Step 3: 第一次 SUE 批量求解 ----
    flows_old_path = os.path.join(args.output_dir, 'flows_old.npy')

    if args.skip_first_solve and os.path.exists(flows_old_path):
        print(f"\n{'='*60}")
        print("Step 3 — 加载已有 flows_old（跳过第一次 SUE 求解）")
        print(f"{'='*60}")
        flows_old = np.load(flows_old_path)
        print(f"  flows_old 形状: {flows_old.shape}")
    else:
        print(f"\n{'='*60}")
        print("Step 3 — 第一次 SUE 批量求解")
        print(f"{'='*60}")
        flows_old = run_first_sue_solve(
            G_topo, od_matrices, capacities, speeds,
            max_iter=args.max_iter,
            convergence_threshold=args.convergence_threshold,
        )
        np.save(flows_old_path, flows_old)
        print(f"  flows_old 已保存: {flows_old_path}")

    # ---- Step 4: 生成 (G, G') 网络对 ----
    print(f"\n{'='*60}")
    print("Step 4 — 生成 (G, G') 网络对")
    print(f"{'='*60}")
    # 将 flows_old 注入 generate_network_pairs，用于识别高流量节点（加边变异依赖此信息）
    scenario_pairs = generate_network_pairs(
        G_topo=G_topo,
        od_matrices=od_matrices,
        capacities=capacities,
        speeds=speeds,
        flows_old=flows_old,
        seed=args.seed,
    )

    # 将第一次求解的 flows_old[i] 嵌入每个 pair，方便后续统一访问
    # 注意：flows_old[i] 按 list(G_topo.edges()) 索引，
    # 与 pair['G'] 的边序完全一致（build_scenario_graph 不改变拓扑边序）
    for i, pair in enumerate(scenario_pairs):
        pair['flows_old'] = flows_old[i].copy()

    # ---- Step 5: 第二次 SUE 逐图求解 ----
    checkpoint_path = os.path.join(args.output_dir, 'pairs_completed.pkl')

    completed_pairs, failed_indices = run_second_sue_solve(
        scenario_pairs=scenario_pairs,
        max_iter=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        checkpoint_path=checkpoint_path if args.checkpoint else None,
        checkpoint_interval=args.checkpoint_interval,
    )

    # ---- Step 6: 保存最终数据集 ----
    print(f"\n{'='*60}")
    print("Step 6 — 保存最终数据集")
    print(f"{'='*60}")

    output_path = os.path.join(args.output_dir, 'network_pairs_dataset.pkl')
    _save_final_dataset(completed_pairs, failed_indices, output_path)

    # 打印最终汇总
    _print_final_summary(completed_pairs, failed_indices, output_path, args)

    return completed_pairs


def _save_final_dataset(completed_pairs: list, failed_indices: list,
                        output_path: str) -> None:
    """
    将完整数据集持久化为 pickle 文件。

    文件包含两个字段：
      - 'pairs'          : list of dict，完整的 (G, G') 数据对
      - 'failed_indices' : list of int，求解失败的原始样本索引（用于审计）

    Args:
        completed_pairs: 完成的数据对列表
        failed_indices:  失败样本索引列表
        output_path:     输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    payload = {
        'pairs'         : completed_pairs,
        'failed_indices': failed_indices,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  数据集已保存: {output_path}（{size_mb:.1f} MB）")
    print(f"  共 {len(completed_pairs)} 个有效对，{len(failed_indices)} 个废弃样本")


def _print_final_summary(completed_pairs: list, failed_indices: list,
                         output_path: str, args) -> None:
    """打印流水线最终执行摘要。"""
    print(f"\n{'='*60}")
    print("流水线执行摘要")
    print(f"{'='*60}")
    print(f"  目标样本数      : {args.num_samples}")
    print(f"  有效数据对      : {len(completed_pairs)}")
    print(f"  废弃样本数      : {len(failed_indices)}")
    print(f"  有效率          : {len(completed_pairs)/args.num_samples*100:.1f}%")
    print(f"  输出文件        : {output_path}")

    if completed_pairs:
        # 边数统计
        e_old = [len(p['edge_list_old']) for p in completed_pairs]
        e_new = [len(p['edge_list_new']) for p in completed_pairs]
        print(f"\n  G  边数 (固定)  : {e_old[0]}")
        print(f"  G' 边数统计     : min={min(e_new)}, max={max(e_new)}, "
              f"mean={np.mean(e_new):.1f}, std={np.std(e_new):.1f}")

        # 流量统计
        all_flows_new = np.concatenate([p['flows_new'] for p in completed_pairs])
        print(f"\n  flows_new 统计  : min={all_flows_new.min():.1f}, "
              f"max={all_flows_new.max():.1f}, "
              f"mean={all_flows_new.mean():.1f}")

        all_flows_old = np.concatenate([p['flows_old'] for p in completed_pairs])
        print(f"  flows_old 统计  : min={all_flows_old.min():.1f}, "
              f"max={all_flows_old.max():.1f}, "
              f"mean={all_flows_old.mean():.1f}")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='(G, G\') 网络对双重 SUE 求解流水线',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('--network_file', type=str,
                        default='../sioux_data/SiouxFalls_net.tntp',
                        help='Sioux Falls 网络文件路径 (.tntp)')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='生成的场景对数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='全局随机种子')
    parser.add_argument('--output_dir', type=str,
                        default='processed_data/pairs',
                        help='所有输出文件的根目录')

    # SUE 求解器参数
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Frank-Wolfe 最大迭代次数')
    parser.add_argument('--convergence_threshold', type=float, default=1e-4,
                        help='Frank-Wolfe 收敛判据（相对间隙）')

    # 断点续跑
    parser.add_argument('--skip_first_solve', action='store_true',
                        help='跳过第一次 SUE 求解，直接加载已有的 flows_old.npy')

    # 检查点
    parser.add_argument('--checkpoint', action='store_true',
                        help='开启第二次求解的检查点保存')
    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='检查点保存间隔（样本数）')

    return parser.parse_args()


def main():
    print("\n" + "=" * 60)
    print("  (G, G') 网络对双重 SUE 求解流水线")
    print("=" * 60)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    print(f"\n  运行参数:")
    for k, v in vars(args).items():
        print(f"    {k:30s}: {v}")

    try:
        completed_pairs = run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n  用户中断，退出。")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\n  [错误] 流水线异常终止: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    return completed_pairs


if __name__ == '__main__':
    main()
