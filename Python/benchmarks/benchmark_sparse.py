#!/usr/bin/env python
"""Benchmark sparse similarity computation across backends, metrics, and sizes.

Reproducible: fixed seeds, deterministic configuration. Compares scipy vs
torch backends for Euclidean and Correlation metrics at increasing N.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import phate
from phate import sparse_similarity

# ── Configuration (change these to reproduce different scenarios) ──────────

SIZES = [500, 2000, 5000, 10000, 20000, 50000, 100000]
SEEDS = [42, 123, 456, 789, 1024]
K = 10
DECAY = 40
BATCH_SIZE = 256
METRIC = "euclidean"
N_FEATURES = 100

BACKENDS = [
    ("scipy", "euclidean", "cpu"),
    ("torch", "euclidean", "cpu"),
    ("torch", "correlation", "cpu"),
]

# ── Helpers ────────────────────────────────────────────────────────────────


def gen_data(n_samples, n_features=N_FEATURES, seed=42):
    return phate.tree.gen_dla(
        n_dim=n_features,
        n_branch=max(1, n_samples // 100),
        branch_length=100,
        seed=seed,
    )


def run_benchmark():
    print(f"PHATE Sparse Similarity Benchmark")
    print(f"k={K}, decay={DECAY}, batch_size={BATCH_SIZE}, features={N_FEATURES}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} runs per config)")
    print(f"Sizes: {SIZES}")
    print()
    print(
        f"{'N':>8} {'Backend':>6} {'Metric':>12} "
        f"{'Mean(s)':>10} {'Std(s)':>10} {'Min(s)':>10} {'Max(s)':>10} {'NNZ':>10} {'Density%':>10}"
    )
    print("-" * 100)

    for n in SIZES:
        for backend, metric, device in BACKENDS:
            # Skip scipy at large N (too slow)
            if backend == "scipy" and n > 20000:
                continue

            times = []
            nnz = None

            for seed in SEEDS:
                data, _ = gen_data(n, seed=seed)

                t0 = time.time()
                if backend == "scipy":
                    S = sparse_similarity.compute_sparse_similarity(
                        data, k=K, metric=metric, decay=DECAY,
                        batch_size=BATCH_SIZE, verbose=0,
                    )
                else:
                    S = sparse_similarity.compute_sparse_similarity_torch(
                        data, k=K, metric=metric, decay=DECAY,
                        batch_size=BATCH_SIZE, device=device, verbose=0,
                    )
                times.append(time.time() - t0)
                nnz = S.nnz

            t_mean = np.mean(times)
            t_std = np.std(times, ddof=1)
            t_min = np.min(times)
            t_max = np.max(times)
            density = 100 * nnz / (n * n)

            print(
                f"{n:>8} {backend:>6} {metric:>12} "
                f"{t_mean:>10.3f} {t_std:>10.3f} {t_min:>10.3f} {t_max:>10.3f} "
                f"{nnz:>10} {density:>9.3f}%"
            )
        print()


def benchmark_minimal_k():
    """Binary search for minimal k achieving graph connectivity."""
    print("=" * 80)
    print("Minimal k Connectivity Search")
    print("=" * 80)

    n = 500
    data, _ = gen_data(n, seed=42)
    print(f"Dataset: N={n}, features={N_FEATURES}")

    t0 = time.time()
    k_opt, S = sparse_similarity.find_minimal_k(
        data, k_max=50, metric="euclidean", batch_size=256, verbose=1
    )
    elapsed = time.time() - t0

    n_comp, _ = sparse_similarity.check_connectivity(S)
    print(f"Minimal k for connectivity: {k_opt}")
    print(f"Components at k_opt: {n_comp}")
    print(f"Search time: {elapsed:.2f}s")
    print(f"Sparsity: {100 * S.nnz / (n * n):.3f}% ({S.nnz} non-zeros)")

    if k_opt > 1:
        t0 = time.time()
        S_prev = sparse_similarity.compute_sparse_similarity(
            data, k=k_opt - 1, metric="euclidean", decay=DECAY,
            batch_size=BATCH_SIZE, verbose=0,
        )
        n_comp_prev, _ = sparse_similarity.check_connectivity(S_prev)
        print(f"Components at k_opt-1={k_opt - 1}: {n_comp_prev}")
        print(f"Check time: {time.time() - t0:.2f}s")

    return k_opt


if __name__ == "__main__":
    run_benchmark()
    benchmark_minimal_k()
