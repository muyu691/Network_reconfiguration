"""
高阶随机用户均衡 (SUE) 求解器：Markov-Logit 网络加载 + MSA-SR 外层寻优

理论核心（与 Dial 节点遍历加载法区分）：
1) Markov Routing / Equivalent Absorbing Markov Chain:
   - 对每个目的地 d，构造路段转移概率 p_{uv}^d；
   - 利用访问量固定点 x = q + P^T x（等价于 (I - P^T)x = q）进行全网络并行加载；
   - 全程使用矩阵/稀疏算子，避免显式节点遍历循环。
2) MSA-SR（自调节连续平均法）：
   - 步长基础形式 alpha_k = beta / k^gamma；
   - 结合相对收敛间隙趋势进行自调节，显著优于标准 1/k 的固定衰减。

注意：
- 保留原项目兼容接口：frank_wolfe_sue(...) 与 solve_sue_batch(...)。
- 保留 BPR 拥堵更新逻辑（仅强化数值稳定性）。
- 严格不把 OD 作为模型输入特征；OD 仅用于 SUE 物理求解。
"""

import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


EPS = 1e-12
V_MAX = 1e6
EXP_CLIP_MIN = -700.0
EXP_CLIP_MAX = 60.0


def bpr_travel_time(flow, capacity, free_flow_time, alpha=0.15, beta=4.0):
    """
    BPR 路段阻抗函数：
    t = t0 * (1 + alpha * (flow / capacity)^beta)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap_safe = np.maximum(np.asarray(capacity, dtype=np.float64), EPS)
        ratio = np.clip(np.asarray(flow, dtype=np.float64) / cap_safe, 0.0, 1e6)
        return np.asarray(free_flow_time, dtype=np.float64) * (1.0 + alpha * (ratio ** beta))


def _build_sparse_edge_incidence(num_nodes, tails, heads):
    """
    构造稀疏关联矩阵：
    - out_mat: [N, E]，按尾节点聚合（每列在 tail 处为 1）
    - in_mat : [N, E]，按头节点聚合（每列在 head 处为 1）
    """
    num_edges = tails.shape[0]
    edge_ids = np.arange(num_edges, dtype=np.int64)
    ones = np.ones(num_edges, dtype=np.float64)
    out_mat = sp.csr_matrix((ones, (tails, edge_ids)), shape=(num_nodes, num_edges))
    in_mat = sp.csr_matrix((ones, (heads, edge_ids)), shape=(num_nodes, num_edges))
    return out_mat, in_mat


def _centroid_destination_nodes(od_matrix, num_nodes):
    """
    约定：OD 矩阵大小为 [C, C]，对应图中的节点 ID 1..C（0-index 后为 0..C-1）。
    """
    num_centroids = int(od_matrix.shape[0])
    if num_centroids > num_nodes:
        raise ValueError(
            f"OD centroid count ({num_centroids}) exceeds graph nodes ({num_nodes})."
        )
    return np.arange(num_centroids, dtype=np.int64)


def _solve_recursive_logit_values(
    travel_times,
    tails,
    heads,
    out_mat,
    dest_nodes,
    theta,
    max_iter=200,
    tol=1e-8,
):
    """
    向量化 Bellman-LogSum 固定点（并行处理全部目的地）：

      V_u^d = -1/theta * log( sum_{(u,v)} exp( -theta * (c_uv + V_v^d) ) )
      V_d^d = 0

    这是 Markov Routing 下构建转移概率的“值函数”步骤。
    """
    num_nodes = int(out_mat.shape[0])
    num_edges = tails.shape[0]
    num_dests = dest_nodes.shape[0]
    tt = np.asarray(travel_times, dtype=np.float64).reshape(num_edges)

    V = np.zeros((num_nodes, num_dests), dtype=np.float64)
    V[dest_nodes, np.arange(num_dests)] = 0.0

    for _ in range(max_iter):
        # utility_e,d = -theta * (c_e + V_head,d)
        utility = -theta * (tt[:, None] + V[heads, :])
        utility = np.clip(utility, EXP_CLIP_MIN, EXP_CLIP_MAX)
        z = np.exp(utility)  # [E, D]

        # S_u,d = sum_{e from u} z_e,d
        S = out_mat @ z  # [N, D]
        V_new = -np.log(np.maximum(S, EPS)) / theta

        # 目的地边界条件：V_d^d = 0
        V_new[dest_nodes, np.arange(num_dests)] = 0.0

        # 不可达数值保护（S 过小表示近乎不可达）
        unreachable = S <= EPS
        if np.any(unreachable):
            V_new[unreachable] = V_MAX
            V_new[dest_nodes, np.arange(num_dests)] = 0.0

        rel = np.linalg.norm(V_new - V) / (np.linalg.norm(V) + 1.0)
        V = V_new
        if rel < tol:
            break

    return V


def _markov_logit_network_loading(
    travel_times,
    od_matrix,
    tails,
    heads,
    out_mat,
    in_mat,
    theta=0.8,
    value_iter=250,
    value_tol=1e-8,
    flow_iter=500,
    flow_tol=1e-9,
):
    """
    Markov-Logit 全网络随机加载（无显式节点遍历）：

    Step A: 求值函数 V^d（Bellman-LogSum 并行固定点）
    Step B: 构造转移概率 p_e^d
    Step C: 求节点访问量固定点 x = q + P^T x（Richardson 近似解）
    Step D: 路段流量 f_e = sum_d x_tail(e)^d * p_e^d
    """
    num_nodes = out_mat.shape[0]
    num_edges = tails.shape[0]

    od = np.asarray(od_matrix, dtype=np.float64)
    dest_nodes = _centroid_destination_nodes(od, num_nodes)
    num_dests = dest_nodes.shape[0]

    # A. 并行求值函数
    V = _solve_recursive_logit_values(
        travel_times=travel_times,
        tails=tails,
        heads=heads,
        out_mat=out_mat,
        dest_nodes=dest_nodes,
        theta=theta,
        max_iter=value_iter,
        tol=value_tol,
    )  # [N, D]

    # B. 构造边转移概率 p_e^d
    utility = -theta * (np.asarray(travel_times, dtype=np.float64)[:, None] + V[heads, :])
    utility = np.clip(utility, EXP_CLIP_MIN, EXP_CLIP_MAX)
    z = np.exp(utility)  # [E, D]
    S = out_mat @ z      # [N, D]
    p = z / np.maximum(S[tails, :], EPS)

    # 目的地吸收：目的节点不再外流（对应吸收马尔可夫链边界）
    is_out_of_dest = tails[:, None] == dest_nodes[None, :]
    p[is_out_of_dest] = 0.0
    p = np.clip(p, 0.0, 1.0)

    # C. 目的地并行注入向量 q：q[i,d] = OD(i,d)，且 q[d,d] = 0
    q = np.zeros((num_nodes, num_dests), dtype=np.float64)
    q[:num_dests, :] = od
    q[dest_nodes, np.arange(num_dests)] = 0.0

    x = q.copy()  # 节点访问量（每个目的地一列）
    for _ in range(flow_iter):
        edge_flow_by_dest = x[tails, :] * p        # [E, D]
        x_new = q + (in_mat @ edge_flow_by_dest)   # [N, D]

        # 防止漂移导致负值（理论上不应出现）
        x_new = np.maximum(x_new, 0.0)

        rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1.0)
        x = x_new
        if rel < flow_tol:
            break
    else:
        warnings.warn(
            f"Markov loading did not converge within {flow_iter} iterations. "
            "Increase flow_iter for larger networks."
        )

    # D. 聚合到路段流量
    edge_flow_by_dest = x[tails, :] * p
    flows = np.sum(edge_flow_by_dest, axis=1)  # [E]
    flows = np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(flows, 0.0)


def _relative_gap(flows, aux_flows, travel_times):
    """
    相对收敛间隙（Relative Gap）：
    - flow_gap: ||x^{k+1}-x^k|| / ||x^k||
    - cost_gap: |t(x)^T x - t(x)^T y| / max(t(x)^T x, eps)
      其中 y 为当前阻抗下的随机加载解（搜索方向目标）
    """
    x = np.asarray(flows, dtype=np.float64)
    y = np.asarray(aux_flows, dtype=np.float64)
    t = np.asarray(travel_times, dtype=np.float64)

    flow_gap = np.linalg.norm(y - x) / (np.linalg.norm(x) + EPS)
    cx = float(np.dot(t, x))
    cy = float(np.dot(t, y))
    cost_gap = abs(cx - cy) / max(abs(cx), EPS)
    return flow_gap, cost_gap


def _msa_sr_step(iteration, flow_gap, prev_flow_gap, beta=1.2, gamma=0.72):
    """
    自调节 MSA-SR 步长：
      alpha_k = beta / (k^gamma)
    并按 gap 变化趋势修正：
      - 若下降停滞（ratio 接近 1），适度增大步长；
      - 若震荡/反弹（ratio > 1），收缩步长稳定迭代。
    """
    k = max(int(iteration), 1)
    alpha = beta / (k ** gamma)

    if np.isfinite(prev_flow_gap) and prev_flow_gap > 0.0:
        ratio = flow_gap / prev_flow_gap
        if ratio > 1.02:
            alpha *= 0.60
        elif ratio > 0.98:
            alpha *= 0.85
        elif ratio < 0.70:
            alpha *= 1.08

    return float(np.clip(alpha, 0.02, 0.80))


def markov_logit_sue_solver(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    max_iter=120,
    convergence_threshold=1e-5,
    verbose=True,
    theta=0.8,
    bpr_alpha=0.15,
    bpr_beta=4.0,
):
    """
    高性能 Markov-Logit SUE 主求解器（推荐入口）。

    输入/输出签名与旧版 frank_wolfe_sue 保持一致：
      输入: G, od_matrix, capacities, free_flow_times
      输出: 与 list(G.edges()) 对齐的一维路段流量数组 [E]

    外层：MSA-SR
      x^{k+1} = (1-alpha_k) x^k + alpha_k * y^k
      其中 y^k 为当前阻抗下的 Markov-Logit 全网络随机加载解。
    """
    edges = list(G.edges())
    num_edges = len(edges)
    if num_edges == 0:
        return np.zeros(0, dtype=np.float64)

    # Graph edge arrays (严格按 list(G.edges()) 顺序)
    tails = np.asarray([u - 1 for u, _ in edges], dtype=np.int64)
    heads = np.asarray([v - 1 for _, v in edges], dtype=np.int64)
    num_nodes = int(max(np.max(tails), np.max(heads)) + 1)

    out_mat, in_mat = _build_sparse_edge_incidence(num_nodes, tails, heads)

    cap = np.asarray(capacities, dtype=np.float64).reshape(num_edges)
    t0 = np.asarray(free_flow_times, dtype=np.float64).reshape(num_edges)
    cap = np.maximum(cap, EPS)
    t0 = np.maximum(t0, EPS)

    # 初值：自由流阻抗下做一次 Markov-Logit 加载
    flows = _markov_logit_network_loading(
        travel_times=t0,
        od_matrix=od_matrix,
        tails=tails,
        heads=heads,
        out_mat=out_mat,
        in_mat=in_mat,
        theta=theta,
    )

    prev_flow_gap = np.inf
    for it in range(1, max_iter + 1):
        travel_times = bpr_travel_time(flows, cap, t0, alpha=bpr_alpha, beta=bpr_beta)

        aux_flows = _markov_logit_network_loading(
            travel_times=travel_times,
            od_matrix=od_matrix,
            tails=tails,
            heads=heads,
            out_mat=out_mat,
            in_mat=in_mat,
            theta=theta,
        )

        flow_gap, cost_gap = _relative_gap(flows, aux_flows, travel_times)
        step = _msa_sr_step(it, flow_gap, prev_flow_gap, beta=1.2, gamma=0.72)

        new_flows = (1.0 - step) * flows + step * aux_flows
        new_flows = np.maximum(np.nan_to_num(new_flows, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        update_gap = np.linalg.norm(new_flows - flows) / (np.linalg.norm(flows) + EPS)
        flows = new_flows
        prev_flow_gap = flow_gap

        if verbose and (it == 1 or it % 5 == 0):
            print(
                f"    Iter {it:03d} | step={step:.4f} | "
                f"flow_gap={flow_gap:.6e} | cost_gap={cost_gap:.6e} | update_gap={update_gap:.6e}"
            )

        # 严格终止：同时满足行为 gap 与更新 gap
        if max(flow_gap, cost_gap, update_gap) < convergence_threshold:
            if verbose:
                print(f"  SUE converged at iter {it}, gap={max(flow_gap, cost_gap, update_gap):.3e}")
            break

    return flows


def advanced_sue_solver(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    max_iter=120,
    convergence_threshold=1e-5,
    verbose=True,
    theta=0.8,
):
    """
    与 markov_logit_sue_solver 等价的学术入口别名。
    """
    return markov_logit_sue_solver(
        G=G,
        od_matrix=od_matrix,
        capacities=capacities,
        free_flow_times=free_flow_times,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        verbose=verbose,
        theta=theta,
    )


def frank_wolfe_sue(
    G,
    od_matrix,
    capacities,
    free_flow_times,
    max_iter=120,
    convergence_threshold=1e-5,
    verbose=True,
):
    """
    兼容旧调用名：内部切换到高级 Markov-Logit SUE（MSA-SR）。
    """
    return markov_logit_sue_solver(
        G=G,
        od_matrix=od_matrix,
        capacities=capacities,
        free_flow_times=free_flow_times,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        verbose=verbose,
    )


def solve_sue_batch(G, od_matrices, capacities, speeds, method='frank_wolfe', verbose=True):
    """
    批量求解 SUE（兼容旧接口）。
    """
    try:
        from .utils import compute_free_flow_times
    except ImportError:
        from utils import compute_free_flow_times

    num_samples = int(od_matrices.shape[0])
    num_edges = len(list(G.edges()))

    print(f"\n{'=' * 60}")
    print(f"Solving SUE for {num_samples} scenarios using '{method}' method")
    print(f"{'=' * 60}")

    print("  Computing free-flow times...")
    free_flow_times = compute_free_flow_times(G, speeds)

    all_flows = np.zeros((num_samples, num_edges), dtype=np.float64)

    method = str(method).lower()
    if method in ('frank_wolfe', 'advanced', 'markov_logit', 'msa_sr'):
        solver_func = frank_wolfe_sue
    else:
        raise ValueError(f"Unknown method: {method}")

    print("  Running traffic assignment...")
    for i in tqdm(range(num_samples), desc="  Progress", disable=not verbose):
        all_flows[i] = solver_func(
            G=G,
            od_matrix=od_matrices[i],
            capacities=capacities[i],
            free_flow_times=free_flow_times[i],
            max_iter=120,
            convergence_threshold=1e-5,
            verbose=False,
        )

    print("\n SUE solving completed!")
    print("  Flow statistics:")
    print(f"    Min: {all_flows.min():.2f}")
    print(f"    Max: {all_flows.max():.2f}")
    print(f"    Mean: {all_flows.mean():.2f}")
    print(f"    Std: {all_flows.std():.2f}")
    return all_flows


def save_flows(flows, save_path='processed_data/raw/flows.npz'):
    """保存求解流量结果。"""
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, flows=flows)
    print(f"\n Flows saved to: {save_path}")


def load_flows(load_path='processed_data/raw/flows.npz'):
    """加载已保存的流量结果。"""
    data = np.load(load_path)
    print(f"\n Flows loaded from: {load_path}")
    return data['flows']


if __name__ == '__main__':
    from load_sioux import load_sioux_falls_network
    from generate_scenarios import generate_lhs_samples

    print("Testing advanced Markov-Logit SUE solver...")
    G, _ = load_sioux_falls_network('../sioux_data/SiouxFalls_net.tntp')
    od_mats, caps, speeds = generate_lhs_samples(num_samples=5)
    flows = solve_sue_batch(G, od_mats, caps, speeds, method='markov_logit')
    print("\n Test completed!")
    print(f"  Generated flows shape: {flows.shape}")
