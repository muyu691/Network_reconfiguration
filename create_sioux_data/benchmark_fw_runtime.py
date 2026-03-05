"""
Frank-Wolfe (Markov-Logit SUE) Runtime Benchmark
=================================================

Standalone script to measure the average wall-clock time of the traditional
SUE solver on the test split of the (G, G') network pairs dataset.

This provides a fair comparison point against learned methods (GNN / MLP / baselines),
which are timed inside the inference loop of custom_train.py.

Usage:
    cd create_sioux_data
    python benchmark_fw_runtime.py \
        --input_pkl processed_data/pairs/network_pairs_dataset.pkl \
        --num_test_graphs 50 \
        --seed 42

Output:
    Per-graph solve time statistics (min / mean / median / max / total).
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sue_solver import markov_logit_sue_solver
from build_network_pairs_dataset import split_indices


def _extract_graph_arrays(G):
    """Extract capacity and free_flow_time arrays aligned with list(G.edges())."""
    edges = list(G.edges())
    caps = np.array([G[u][v]['capacity'] for u, v in edges], dtype=np.float64)
    ffts = np.array([G[u][v]['free_flow_time'] for u, v in edges], dtype=np.float64)
    return edges, caps, ffts


def benchmark_sue_on_test_split(args):
    """
    Load the pkl dataset, identify the test split, then run the Markov-Logit
    SUE solver on each test G' to collect per-graph wall-clock times.
    """
    # ── Load pairs ──────────────────────────────────────────────────────
    print(f"Loading pairs from: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        pairs = payload['pairs']
    else:
        pairs = payload

    num_samples = len(pairs)
    print(f"Total valid pairs: {num_samples}")

    # ── Reproduce the exact same test split used by build_network_pairs_dataset ──
    _, _, test_idx = split_indices(
        num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Test split size: {len(test_idx)}")

    if args.num_test_graphs > 0:
        test_idx = test_idx[:args.num_test_graphs]
    print(f"Benchmarking on {len(test_idx)} test graphs ...\n")

    # ── Solve each G' and record time ───────────────────────────────────
    times_sec = []

    for rank, idx in enumerate(test_idx):
        pair = pairs[idx]
        G_prime = pair['G_prime']
        od_matrix = pair['od_matrix']
        _, caps, ffts = _extract_graph_arrays(G_prime)

        t0 = time.perf_counter()
        _ = markov_logit_sue_solver(
            G=G_prime,
            od_matrix=od_matrix,
            capacities=caps,
            free_flow_times=ffts,
            max_iter=args.max_iter,
            convergence_threshold=args.conv_thr,
            verbose=False,
            theta=args.theta,
        )
        elapsed = time.perf_counter() - t0
        times_sec.append(elapsed)

        if (rank + 1) % 10 == 0 or (rank + 1) == len(test_idx):
            print(f"  [{rank+1:>4d}/{len(test_idx)}]  last={elapsed:.3f}s  "
                  f"avg={np.mean(times_sec):.3f}s")

    # ── Statistics ──────────────────────────────────────────────────────
    times = np.array(times_sec)
    print(f"\n{'='*60}")
    print(f"  Frank-Wolfe (Markov-Logit SUE) Runtime Benchmark")
    print(f"{'='*60}")
    print(f"  Graphs tested     : {len(times)}")
    print(f"  Total time        : {times.sum():.2f} s")
    print(f"  Min  per graph    : {times.min():.4f} s  ({times.min()*1000:.2f} ms)")
    print(f"  Mean per graph    : {times.mean():.4f} s  ({times.mean()*1000:.2f} ms)")
    print(f"  Median per graph  : {np.median(times):.4f} s  ({np.median(times)*1000:.2f} ms)")
    print(f"  Max  per graph    : {times.max():.4f} s  ({times.max()*1000:.2f} ms)")
    print(f"  Std               : {times.std():.4f} s")
    print(f"{'='*60}")

    # Percentiles
    for q in [25, 50, 75, 95]:
        v = np.percentile(times, q)
        print(f"  P{q:>2d}  : {v:.4f} s  ({v*1000:.2f} ms)")
    print(f"{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description='Benchmark Frank-Wolfe SUE solver runtime on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input_pkl', type=str,
                   default='processed_data/pairs/network_pairs_dataset.pkl',
                   help='Path to the (G, G\') pairs pkl file')
    p.add_argument('--num_test_graphs', type=int, default=0,
                   help='Number of test graphs to benchmark (0 = all)')
    p.add_argument('--train_ratio', type=float, default=0.6)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (must match build_network_pairs_dataset)')
    p.add_argument('--max_iter', type=int, default=120,
                   help='SUE solver max outer iterations')
    p.add_argument('--conv_thr', type=float, default=1e-5,
                   help='SUE convergence threshold')
    p.add_argument('--theta', type=float, default=0.8,
                   help='Logit dispersion parameter')
    return p.parse_args()


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  Frank-Wolfe Runtime Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    args = parse_args()
    benchmark_sue_on_test_split(args)
