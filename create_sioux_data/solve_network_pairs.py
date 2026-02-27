"""
Dual flow solving pipeline: (G, G') network pair complete data generation

Full procedure steps:
  Step 1 — LHS sampling        : Generate base scenario parameters (OD matrix, capacity, speed)
  Step 2 — First SUE solving   : Run Frank-Wolfe on G → flows_old [N, E_old]
  Step 3 — Network pair creation: Call generate_network_pairs, passing flows_old to locate high-flow nodes
  Step 4 — Second SUE solving  : Run Frank-Wolfe on each G' → flows_new [E_new_i]
  Step 5 — Save results        : Output .pkl file, for use by build_pyg_data.py

Strict constraints:
  - G and G' share exactly the same OD matrix (OD is never used as a model node feature)
  - The number and order of edges in G' may differ from G, flows_new is indexed using list(G'.edges())
  - Samples that fail the second solve (abnormal/NaN/Inf) will be skipped and logged as discarded

Output data structure (each completed_pair is a dict):
  {
    'od_matrix'    : np.ndarray [11, 11],   # shared OD (only used for SUE)
    'G'            : nx.DiGraph,            # base scenario graph (fixed topology + LHS attributes)
    'G_prime'      : nx.DiGraph,            # mutated network
    'mutation_type': str,
    'mutation_info': dict,
    'flows_old'    : np.ndarray [E_old],    # equilibrium flows on G (indexed by edge_list_old)
    'flows_new'    : np.ndarray [E_new],    # equilibrium flows on G' (indexed by edge_list_new)
    'edge_list_old': list[tuple],           # list(G.edges())  ← explicit save, eliminates ambiguity
    'edge_list_new': list[tuple],           # list(G'.edges()) ← explicit save, eliminates ambiguity
  }

Usage:
  python solve_network_pairs.py --num_samples 2000 --network_file ../sioux_data/SiouxFalls_net.tntp
  python solve_network_pairs.py --num_samples 2000 --skip_first_solve  # resume job from checkpoint
"""

import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

# Compatible with both direct execution (python solve_network_pairs.py) and module import
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
# Core utility functions
# ============================================================

def _extract_graph_arrays(G) -> tuple:
    """
    Directly extract the two arrays required for SUE solving from the edge attributes of a NetworkX DiGraph.

    The topology and attributes of G' have already been fully embedded in the graph object in generate_network_pairs,
    This function extracts in the order of list(G.edges()):
      - capacities      : [E] capacity of each edge
      - free_flow_times : [E] free flow travel time for each edge (minutes)

    Why not use utils.compute_free_flow_times?
    That function requires an extra speeds array, but G's free_flow_time has already been recalculated according to
    the new speed and written into the graph attributes during mutation, so direct reading is more accurate and avoids redundant calculation.

    Args:
        G: NetworkX DiGraph with complete edge attributes (capacity, free_flow_time)

    Returns:
        edges:           list[(u, v)], length E, defines flows' index order
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
    Run Frank-Wolfe SUE solve for an arbitrary single graph G (topology/attributes).

    Difference from solve_sue_batch:
    - solve_sue_batch assumes all samples share the same topology, attributes are passed in batch via numpy arrays.
    - This function extracts capacities and free_flow_times directly from the graph's edge attributes,
      suitable for G' (each G' may have different numbers and order of edges).

    The returned flows index order matches list(G.edges()) exactly.
    Callers should save list(G.edges()) before calling in order to restore indices later.

    Args:
        G:                    NetworkX DiGraph (with capacity, free_flow_time attributes)
        od_matrix:            np.ndarray [num_centroids, num_centroids]
        max_iter:             Frank-Wolfe maximum iterations
        convergence_threshold: convergence criterion (relative gap)

    Returns:
        flows: np.ndarray [E], equilibrium flows, indexed by list(G.edges())

    Raises:
        ValueError: If the solving result contains NaN or Inf (considered failed solve)
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

    # Check numerical validity: NaN or Inf means divergent solve or a structural problem in the graph
    if np.any(np.isnan(flows)) or np.any(np.isinf(flows)):
        raise ValueError(
            f"SUE result contains NaN/Inf (graph has {G.number_of_edges()} edges)."
        )

    return flows


# ============================================================
# First SUE solving (batch, fixed topology G)
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
    First SUE batch solving: compute flows_old on fixed topology G.

    All base scenarios share the same G_topo topology, only capacity and speed differ,
    so the existing solve_sue_batch batch interface can be reused for efficient solving.

    flows_old[i] is indexed by list(G_topo.edges()) in a fixed order,
    and will be used in generate_network_pairs to identify high-flow nodes.

    Args:
        G_topo:               base topology graph (from the .tntp file, containing length attribute)
        od_matrices:          [N, 11, 11]
        capacities:           [N, 76]  ← indexed by list(G_topo.edges())
        speeds:               [N, 76]  ← indexed by list(G_topo.edges())
        max_iter:             Frank-Wolfe max iterations per scenario
        convergence_threshold: convergence criterion

    Returns:
        flows_old: np.ndarray [N, 76], indexed by list(G_topo.edges())
    """
    num_samples = od_matrices.shape[0]

    print(f"\n{'='*60}")
    print(f"Step 2 — First SUE batch solve (on G, total {num_samples} scenarios)")
    print(f"{'='*60}")

    # Use utils.compute_free_flow_times to compute free flow time:
    # That function uses G_topo[u][v]['length'] divided by each scenario's speed, matching
    # the calculation in build_scenario_graph exactly, ensuring reproducibility.
    free_flow_times = compute_free_flow_times(G_topo, speeds)  # [N, 76]

    edges_G = list(G_topo.edges())
    num_edges_G = len(edges_G)
    flows_old = np.zeros((num_samples, num_edges_G), dtype=np.float64)

    failed_count = 0
    for i in tqdm(range(num_samples), desc="  First SUE solve"):
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
                raise ValueError("flows_old contains NaN/Inf")
            flows_old[i] = flows_i
        except Exception as e:
            # The base G is strongly connected, first solve failures are extremely rare, log and continue zeroed
            failed_count += 1
            flows_old[i] = 0.0
            if failed_count <= 5:  # Avoid logs exploding, print only first 5 entries
                print(f"  [Warning] Scenario {i} first solve failed: {e}")

    print(f"\n  First SUE solve complete!")
    print(f"  flows_old shape: {flows_old.shape}")
    print(f"  flows_old stats: min={flows_old.min():.1f}, max={flows_old.max():.1f}, "
          f"mean={flows_old.mean():.1f}")
    if failed_count:
        print(f"  [Warning] First solve failed for {failed_count} scenarios (flows_old set to zero)")

    return flows_old


# ============================================================
# Second SUE solving (per-graph, G' topology varies)
# ============================================================

def run_second_sue_solve(
    scenario_pairs: list,
    max_iter: int = 100,
    convergence_threshold: float = 1e-4,
    checkpoint_path: str = None,
    checkpoint_interval: int = 200,
) -> tuple:
    """
    Second SUE solving: solve each G' independently, then merge into final data pairs.

    Key difference from the first solve:
    - Each G' may have different number/order of edges (due to topology mutations)
    - Must extract capacity and free_flow_time directly from G' graph attributes
    - flows_new[i] indexed by list(G_prime.edges()) (different from flows_old indexing!)
    - edge_list_new[i] must be saved explicitly, build_pyg_data.py relies on it to recover edge_index_new

    Failure handling:
    - Catch all exceptions (NetworkXNoPath, NaN/Inf, convergence failure, etc)
    - Failed sample is logged in failed_indices and skipped, not added to completed_pairs
    - Given strong connectivity in Sioux Falls, failure rate should be near 0%

    Args:
        scenario_pairs:       output of generate_network_pairs, list of dict
                              each dict contains 'G', 'G_prime', 'od_matrix', etc
        max_iter:             Frank-Wolfe max iterations
        convergence_threshold: convergence criterion
        checkpoint_path:      path for saving intermediate checkpoint (saved every checkpoint_interval samples)
                              if None, no checkpoint is saved
        checkpoint_interval:  checkpoint save interval (number of samples)

    Returns:
        completed_pairs: list of dict, each dict contains the full dual-graph data
        failed_indices:  list of int, indices of original samples which failed to solve
    """
    num_pairs = len(scenario_pairs)

    print(f"\n{'='*60}")
    print(f"Step 4 — Second SUE solve (on each G', total {num_pairs} graphs)")
    print(f"{'='*60}")
    if checkpoint_path:
        print(f"  Checkpoint save path: {checkpoint_path} (every {checkpoint_interval} samples)")

    completed_pairs = []
    failed_indices  = []

    for i, pair in enumerate(tqdm(scenario_pairs, desc="  Second SUE solve")):
        G_prime   = pair['G_prime']
        od_matrix = pair['od_matrix']

        try:
            # --- Main: Solve SUE on G' ---
            # G' may have different number/order of edges (due to add/delete edge mutations),
            # solve_single_graph_sue extracts capacity and free_flow_time from G', 
            # and returns flows_new as indexed by current list(G'.edges()).
            flows_new = solve_single_graph_sue(
                G_prime,
                od_matrix,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
            )

            # --- Build complete data pair ---
            # Explicitly save edge_list_old/new to eliminate edge order ambiguity in later build_pyg_data.py.
            # Can't rely on re-calling list(G.edges()) to recover order, since after pickle deserialization 
            # the graph object's iteration order may theoretically differ from when saved (even though NetworkX is usually stable).
            edge_list_old = list(pair['G'].edges())
            edge_list_new = list(G_prime.edges())

            completed_pairs.append({
                'od_matrix'    : pair['od_matrix'],         # [11, 11], only for SUE
                'G'            : pair['G'],                 # NetworkX DiGraph
                'G_prime'      : G_prime,                   # NetworkX DiGraph
                'mutation_type': pair['mutation_type'],
                'mutation_info': pair['mutation_info'],
                'flows_old'    : pair['flows_old'],         # [E_old], indexed by edge_list_old
                'flows_new'    : flows_new,                 # [E_new], indexed by edge_list_new
                'edge_list_old': edge_list_old,             # list[(u,v)], E_old entries
                'edge_list_new': edge_list_new,             # list[(u,v)], E_new entries
            })

        except Exception as e:
            # Log failed samples but do not interrupt the pipeline
            failed_indices.append(i)
            if len(failed_indices) <= 10:  # Only print first 10, avoid exploding logs
                print(f"\n  [Skipped] Sample {i} (mutation={pair['mutation_type']}): {e}")

        # Periodically save checkpoint (to prevent long-running crashes losing progress)
        if (checkpoint_path is not None
                and (i + 1) % checkpoint_interval == 0
                and completed_pairs):
            _save_checkpoint(completed_pairs, failed_indices, checkpoint_path, i + 1)

    # Final statistics report
    total   = num_pairs
    success = len(completed_pairs)
    failed  = len(failed_indices)

    print(f"\n  Second SUE solve complete!")
    print(f"  Total samples : {total}")
    print(f"  Success       : {success} ({success/total*100:.1f}%)")
    print(f"  Failed/skipped: {failed}  ({failed/total*100:.1f}%)")

    if failed_indices:
        print(f"  Failed sample indices (first 20): {failed_indices[:20]}")

    # Print true distribution of mutation types in finished samples
    if completed_pairs:
        from collections import Counter
        dist = Counter(p['mutation_type'] for p in completed_pairs)
        print(f"\n  Completed sample mutation type distribution:")
        for t, cnt in sorted(dist.items()):
            print(f"    {t:20s}: {cnt:5d} ({cnt/success*100:.1f}%)")

    return completed_pairs, failed_indices


def _save_checkpoint(completed_pairs: list, failed_indices: list,
                     base_path: str, step: int) -> None:
    """
    Save intermediate state of the pipeline to a checkpoint file.

    Filename format: {base_path}.ckpt_{step}.pkl
    For example: processed_data/raw/pairs.pkl.ckpt_400.pkl

    Args:
        completed_pairs: List of data pairs completed so far
        failed_indices:  List of failure indices recorded so far
        base_path:       Final output file path (used to construct checkpoint file name)
        step:            Current progress (total number of processed samples)
    """
    ckpt_path = f"{base_path}.ckpt_{step}.pkl"
    os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else '.', exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump({'completed_pairs': completed_pairs, 'failed_indices': failed_indices}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  [Checkpoint] Saved {len(completed_pairs)} pairs to {ckpt_path}")


# ============================================================
# Main pipeline workflow function
# ============================================================

def run_pipeline(args) -> list:
    """
    Full "generate → solve → reconstruct → solve" pipeline control function.

    Execution order:
      Step 1 — Load Sioux Falls base topology
      Step 2 — LHS sample base scenarios (or load from file)
      Step 3 — First SUE batch solve (or load flows_old from file)
      Step 4 — Generate (G, G') network pairs (pass in flows_old to locate high-flow nodes)
      Step 5 — Second SUE solve for each graph (G' topologies vary, each solved individually)
      Step 6 — Save final dataset

    Resume job support (--skip_first_solve flag):
      If flows_old is already saved, can skip Step 2-3 and start from Step 4 directly.

    Args:
        args: argparse.Namespace, see parse_args()

    Returns:
        completed_pairs: List of all completed data pairs
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: Load base topology ----
    print(f"\n{'='*60}")
    print("Step 1 — Load Sioux Falls base topology")
    print(f"{'='*60}")
    G_topo, centroids = load_sioux_falls_network(args.network_file)
    edges_G_topo = list(G_topo.edges())
    num_edges_G  = len(edges_G_topo)
    print(f"  Number of nodes: {G_topo.number_of_nodes()}")
    print(f"  Number of edges:   {num_edges_G}")
    print(f"  Centroids:   {centroids}")

    # ---- Step 2: LHS sample base scenarios ----
    scenarios_path = os.path.join(args.output_dir, 'base_scenarios.npz')

    if args.skip_first_solve and os.path.exists(scenarios_path):
        print(f"\n{'='*60}")
        print("Step 2 — Load existing base scenarios (skip LHS sampling)")
        print(f"{'='*60}")
        od_matrices, capacities, speeds = load_scenarios(scenarios_path)
    else:
        print(f"\n{'='*60}")
        print("Step 2 — LHS sample base scenarios")
        print(f"{'='*60}")
        od_matrices, capacities, speeds = generate_lhs_base_scenarios(
            num_samples=args.num_samples,
            num_centroids=len(centroids),
            num_edges=num_edges_G,
            seed=args.seed,
        )
        save_scenarios(od_matrices, capacities, speeds, scenarios_path)

    # ---- Step 3: First SUE batch solve ----
    flows_old_path = os.path.join(args.output_dir, 'flows_old.npy')

    if args.skip_first_solve and os.path.exists(flows_old_path):
        print(f"\n{'='*60}")
        print("Step 3 — Load existing flows_old (skip first SUE solve)")
        print(f"{'='*60}")
        flows_old = np.load(flows_old_path)
        print(f"  flows_old shape: {flows_old.shape}")
    else:
        print(f"\n{'='*60}")
        print("Step 3 — First SUE batch solve")
        print(f"{'='*60}")
        flows_old = run_first_sue_solve(
            G_topo, od_matrices, capacities, speeds,
            max_iter=args.max_iter,
            convergence_threshold=args.convergence_threshold,
        )
        np.save(flows_old_path, flows_old)
        print(f"  flows_old saved: {flows_old_path}")

    # ---- Step 4: Generate (G, G') network pairs ----
    print(f"\n{'='*60}")
    print("Step 4 — Generate (G, G') network pairs")
    print(f"{'='*60}")
    # Inject flows_old into generate_network_pairs for high-flow node identification (add-edge mutations depend on this info)
    scenario_pairs = generate_network_pairs(
        G_topo=G_topo,
        od_matrices=od_matrices,
        capacities=capacities,
        speeds=speeds,
        flows_old=flows_old,
        seed=args.seed,
    )

    # Embed the first-solve flows_old[i] into each pair for unified access in subsequent steps
    # Note: flows_old[i] uses list(G_topo.edges()) as index,
    # which matches exactly the edge order of pair['G'] (build_scenario_graph doesn't change edge order)
    for i, pair in enumerate(scenario_pairs):
        pair['flows_old'] = flows_old[i].copy()

    # ---- Step 5: Second SUE solve per graph ----
    checkpoint_path = os.path.join(args.output_dir, 'pairs_completed.pkl')

    completed_pairs, failed_indices = run_second_sue_solve(
        scenario_pairs=scenario_pairs,
        max_iter=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        checkpoint_path=checkpoint_path if args.checkpoint else None,
        checkpoint_interval=args.checkpoint_interval,
    )

    # ---- Step 6: Save final dataset ----
    print(f"\n{'='*60}")
    print("Step 6 — Save final dataset")
    print(f"{'='*60}")

    output_path = os.path.join(args.output_dir, 'network_pairs_dataset.pkl')
    _save_final_dataset(completed_pairs, failed_indices, output_path)

    # Print final summary
    _print_final_summary(completed_pairs, failed_indices, output_path, args)

    return completed_pairs


def _save_final_dataset(completed_pairs: list, failed_indices: list,
                        output_path: str) -> None:
    """
    Persist the full dataset to a pickle file.

    The file contains two fields:
      - 'pairs'          : list of dict, the complete list of (G, G') data pairs
      - 'failed_indices' : list of int, indices of failed samples (for auditing)

    Args:
        completed_pairs: The list of completed data pairs
        failed_indices:  List of indices for failed samples
        output_path:     Output file path
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    payload = {
        'pairs'         : completed_pairs,
        'failed_indices': failed_indices,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Dataset saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  {len(completed_pairs)} valid pairs, {len(failed_indices)} discarded samples")


def _print_final_summary(completed_pairs: list, failed_indices: list,
                         output_path: str, args) -> None:
    """Print final pipeline execution summary."""
    print(f"\n{'='*60}")
    print("Pipeline execution summary")
    print(f"{'='*60}")
    print(f"  Target number of samples   : {args.num_samples}")
    print(f"  Valid data pairs           : {len(completed_pairs)}")
    print(f"  Discarded samples          : {len(failed_indices)}")
    print(f"  Validity rate              : {len(completed_pairs)/args.num_samples*100:.1f}%")
    print(f"  Output file                : {output_path}")

    if completed_pairs:
        # Edge statistics
        e_old = [len(p['edge_list_old']) for p in completed_pairs]
        e_new = [len(p['edge_list_new']) for p in completed_pairs]
        print(f"\n  G  number of edges (fixed): {e_old[0]}")
        print(f"  G' edge stats             : min={min(e_new)}, max={max(e_new)}, "
              f"mean={np.mean(e_new):.1f}, std={np.std(e_new):.1f}")

        # Flow statistics
        all_flows_new = np.concatenate([p['flows_new'] for p in completed_pairs])
        print(f"\n  flows_new stats           : min={all_flows_new.min():.1f}, "
              f"max={all_flows_new.max():.1f}, "
              f"mean={all_flows_new.mean():.1f}")

        all_flows_old = np.concatenate([p['flows_old'] for p in completed_pairs])
        print(f"  flows_old stats           : min={all_flows_old.min():.1f}, "
              f"max={all_flows_old.max():.1f}, "
              f"mean={all_flows_old.mean():.1f}")


# ============================================================
# CLI Entry
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='(G, G\') Network pair dual SUE solving pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Basic arguments
    parser.add_argument('--network_file', type=str,
                        default='../sioux_data/SiouxFalls_net.tntp',
                        help='Path to Sioux Falls network file (.tntp)')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of scenario pairs to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    parser.add_argument('--output_dir', type=str,
                        default='processed_data/pairs',
                        help='Root directory for all output files')

    # SUE solver parameters
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Frank-Wolfe max iterations')
    parser.add_argument('--convergence_threshold', type=float, default=1e-4,
                        help='Frank-Wolfe convergence criterion (relative gap)')

    # Resume job support
    parser.add_argument('--skip_first_solve', action='store_true',
                        help='Skip the first SUE solve, directly load existing flows_old.npy')

    # Checkpoint
    parser.add_argument('--checkpoint', action='store_true',
                        help='Enable checkpoint saving for the second solve')
    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='Checkpoint save interval (number of samples)')

    return parser.parse_args()


def main():
    print("\n" + "=" * 60)
    print("  (G, G') Network pair dual SUE solving pipeline")
    print("=" * 60)
    print(f"  Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    print(f"\n  Run parameters:")
    for k, v in vars(args).items():
        print(f"    {k:30s}: {v}")

    try:
        completed_pairs = run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n  User interrupt, exiting.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\n  [Error] Pipeline terminated with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    return completed_pairs


if __name__ == '__main__':
    main()
