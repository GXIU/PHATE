#!/usr/bin/env python
"""Systematic sparse PHATE benchmarks with full configuration support.

Usage:
    python benchmarks/benchmark_sparse_phate.py                    # defaults
    python benchmarks/benchmark_sparse_phate.py --sparse-k 20      # override k
    python benchmarks/benchmark_sparse_phate.py --config my.json   # JSON config

All configuration is driven by the Config dataclass below.
"""

from dataclasses import dataclass, field, asdict
import time
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.sparse import issparse
import phate
from phate import sparse_similarity
from phate import vne


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class Config:
    """Benchmark configuration — all tunable parameters in one place."""

    # ── Data generation ──
    sizes: list = field(default_factory=lambda: [500, 2000, 5000, 10000, 20000, 50000, 100000])
    n_features: int = 512
    seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # ── Sparse similarity ──
    sparse_k: int = 10
    sparse_metric: str = "euclidean"       # "euclidean", "cosine", "correlation"
    decay: float = 40.0
    batch_size: int = 256

    # ── Backend ──
    sparse_backend: str = "torch"          # "scipy" or "torch"
    sparse_device: str = "cpu"             # "cpu", "cuda", "mps"

    # ── Connectivity ──
    tighten_k: bool = True                 # if True, find minimal k via binary search
    k_max: int = 200                       # upper bound for k search (capped at N-1)

    # ── PHATE parameters ──
    knn: int = 5
    t: int = 20
    gamma: float = 1.0
    mds_solver: str = "sgd"               # "sgd" or "smacof"
    mds_how: str = "metric"               # "classic", "metric", "nonmetric"
    n_components: int = 2

    # ── I/O ──
    input_dir: str = ""                    # data directory (empty = use generated data)
    output_dir: str = "benchmark_results"  # where to save CSVs and plots
    plot_on: bool = True                   # generate comparison plots

    # ── Benchmark selection ──
    run_similarity: bool = True
    run_connectivity: bool = True
    run_spectral: bool = True
    run_fill_in: bool = True
    run_mds: bool = True
    run_end_to_end: bool = True

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Helpers ──────────────────────────────────────────────────────────────


def gen_data(config, n, seed=42):
    return phate.tree.gen_dla(
        n_dim=config.n_features,
        n_branch=max(1, n // 100),
        branch_length=100,
        seed=seed,
    )


def mean_std(arr):
    return np.mean(arr), np.std(arr, ddof=1)


def fmt_ms(arr):
    m, s = mean_std(arr)
    return f"{m:.3f} ± {s:.3f}"


def resolve_k(config, data, verbose=True):
    """Resolve k: use tighten_k to find minimal connected k, or use sparse_k directly."""
    if config.tighten_k:
        k_opt, S = sparse_similarity.find_minimal_k(
            data,
            k_max=min(config.k_max, data.shape[0] - 1),
            k_min=1,
            metric=config.sparse_metric,
            decay=config.decay,
            batch_size=config.batch_size,
            verbose=1 if verbose else 0,
        )
        if verbose:
            print(f"  Minimal connected k: {k_opt}")
        return k_opt, S
    else:
        return config.sparse_k, None


# ══════════════════════════════════════════════════════════════════════════
# 1. Similarity computation timing
# ══════════════════════════════════════════════════════════════════════════


def benchmark_similarity(config):
    print("=" * 100)
    print("1. SIMILARITY COMPUTATION TIMING")
    print(f"   dim={config.n_features}, k={config.sparse_k}, decay={config.decay}")
    print("=" * 100)
    print(
        f"{'N':>8} {'Backend':>6} {'Metric':>12} "
        f"{'Time':>18} {'NNZ':>10} {'Density%':>10}"
    )
    print("-" * 68)

    backends = [
        ("scipy", "euclidean"),
        ("torch", "euclidean"),
        ("torch", "correlation"),
    ]

    for n in config.sizes:
        for backend, metric in backends:
            if backend == "scipy" and n > 20000:
                continue
            times = []
            nnz = None
            for seed in config.seeds:
                data, _ = gen_data(config, n, seed)
                t0 = time.time()
                if backend == "scipy":
                    S = sparse_similarity.compute_sparse_similarity(
                        data, k=config.sparse_k, metric=metric,
                        decay=config.decay, batch_size=config.batch_size, verbose=0,
                    )
                else:
                    S = sparse_similarity.compute_sparse_similarity_torch(
                        data, k=config.sparse_k, metric=metric, decay=config.decay,
                        batch_size=config.batch_size, device=config.sparse_device, verbose=0,
                    )
                times.append(time.time() - t0)
                nnz = S.nnz
            print(
                f"{n:>8} {backend:>6} {metric:>12} "
                f"{fmt_ms(times):>18} {nnz:>10} {100*nnz/(n*n):>9.3f}%"
            )
        print()


# ══════════════════════════════════════════════════════════════════════════
# 2. Connectivity scaling
# ══════════════════════════════════════════════════════════════════════════


def benchmark_connectivity(config):
    sizes = [s for s in config.sizes if s <= 10000]
    if not sizes:
        return
    print("=" * 100)
    print("2. CONNECTIVITY SCALING: minimal k vs N")
    print(f"   metric={config.sparse_metric}, decay={config.decay}")
    print("=" * 100)
    print(f"{'N':>8} {'k_min':>12} {'Time(s)':>12}")
    print("-" * 34)

    all_n, all_k = [], []
    for n in sizes:
        k_opts, times = [], []
        for seed in config.seeds[:3]:
            data, _ = gen_data(config, n, seed)
            t0 = time.time()
            k_opt, _ = sparse_similarity.find_minimal_k(
                data, k_max=min(config.k_max, n - 1), k_min=1,
                metric=config.sparse_metric, decay=config.decay,
                batch_size=config.batch_size, verbose=0,
            )
            times.append(time.time() - t0)
            k_opts.append(k_opt)
            all_n.append(np.log(n))
            all_k.append(np.log(k_opt))
        km, ks = mean_std(k_opts)
        tm, _ = mean_std(times)
        print(f"{n:>8} {km:>8.1f} ± {ks:>4.1f}  {tm:>10.2f}")

    if len(all_n) > 3:
        coeffs = np.polyfit(all_n, all_k, 1)
        print(f"\nPower-law: k_min ~ N^{coeffs[0]:.3f}")
        print(f"Rule of thumb: k >= {max(2, int(np.ceil(np.exp(coeffs[1]) * 5000**coeffs[0])))} for N=5000")


# ══════════════════════════════════════════════════════════════════════════
# 3. Spectral fidelity
# ══════════════════════════════════════════════════════════════════════════


def benchmark_spectral(config):
    sizes = [s for s in config.sizes if 500 <= s <= 5000]
    if not sizes:
        return
    print("=" * 100)
    print("3. SPECTRAL FIDELITY: sparse eigsh(100) vs dense SVD")
    print("=" * 100)
    print(
        f"{'N':>8} {'Top10_err':>12} {'Top50_err':>12} "
        f"{'VNE_corr':>10} {'Knee_delta':>12} {'SpecMass':>10}"
    )
    print("-" * 68)

    for n in sizes:
        data, _ = gen_data(config, n, seed=42)
        opt_k, _ = resolve_k(config, data, verbose=False)
        S = sparse_similarity.compute_sparse_similarity(
            data, k=opt_k, metric=config.sparse_metric, decay=config.decay,
            batch_size=config.batch_size, verbose=0,
        )
        P = sparse_similarity.compute_sparse_diffusion_operator(S)
        P_dense = P.toarray()

        k_eigs = min(100, n - 2)
        eig_sp = np.sort(np.abs(sparse.linalg.eigsh(
            (P + P.T) * 0.5, k=k_eigs, which="LM", return_eigenvectors=False,
        )))[::-1]
        eig_de = np.sort(np.abs(np.linalg.svd(P_dense, compute_uv=False)))[::-1]

        top10_err = np.mean(np.abs(eig_sp[:10] - eig_de[:10]) / (eig_de[:10] + 1e-10))
        top50_err = np.mean(np.abs(eig_sp[:50] - eig_de[:50]) / (eig_de[:50] + 1e-10))

        h_sp = vne.compute_von_neumann_entropy_sparse(P, t_max=20)
        h_de = vne.compute_von_neumann_entropy(P_dense, t_max=20)
        vne_c = np.corrcoef(h_sp, h_de)[0, 1]

        k_sp = vne.find_knee_point(y=h_sp, x=np.arange(1, len(h_sp) + 1))
        k_de = vne.find_knee_point(y=h_de, x=np.arange(1, len(h_de) + 1))

        spec_mass = np.sum(eig_sp) / (np.sum(eig_de) + 1e-10)
        print(
            f"{n:>8} {top10_err:>12.4f} {top50_err:>12.4f} "
            f"{vne_c:>10.4f} {abs(k_sp - k_de):>12} {spec_mass:>10.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════
# 4. Diffusion power fill-in
# ══════════════════════════════════════════════════════════════════════════


def benchmark_fill_in(config):
    t_vals = [1, 2, 4, 8, 16, 32]
    k_vals = [5, 10, 20, 50]
    n = 5000
    print("=" * 100)
    print(f"4. DIFFUSION POWER FILL-IN (N={n}, {config.n_features}-dim)")
    print("=" * 100)
    header = f"{'k':>4}" + "".join(f"{'t=' + str(t):>10}" for t in t_vals)
    print(header)
    print("-" * len(header))

    data, _ = gen_data(config, n, seed=42)
    for k_val in k_vals:
        S = sparse_similarity.compute_sparse_similarity(
            data, k=k_val, metric=config.sparse_metric, decay=config.decay,
            batch_size=config.batch_size, verbose=0,
        )
        P = sparse_similarity.compute_sparse_diffusion_operator(S)
        row = f"{k_val:>4}"
        P_t = P.copy()
        prev_step = 1
        for t in t_vals:
            steps_needed = t - prev_step
            for _ in range(steps_needed):
                P_t = P_t @ P
            density = P_t.nnz / (n * n)
            row += f"{100*density:>9.2f}%"
            prev_step = t
        print(row)


# ══════════════════════════════════════════════════════════════════════════
# 5. MDS optimization
# ══════════════════════════════════════════════════════════════════════════


def benchmark_mds(config):
    sizes = [s for s in config.sizes if 1000 <= s <= 10000]
    if not sizes:
        return
    print("=" * 100)
    print("5. MDS OPTIMIZATION: is_pairwise vs dense reference")
    print(f"   solver={config.mds_solver}, gamma={config.gamma}")
    print("=" * 100)
    print(f"{'N':>8} {'Method':>22} {'Time':>15}")
    print("-" * 47)

    for n in sizes:
        for label, sparse_flag in [("sparse+is_pairwise", True), ("dense (graphtools)", False)]:
            times = []
            for seed in config.seeds[:3]:
                data, _ = gen_data(config, n, seed)
                if sparse_flag:
                    op = phate.PHATE(
                        knn=config.knn, t=config.t, gamma=config.gamma,
                        sparse_k=config.sparse_k, sparse_metric=config.sparse_metric,
                        sparse_backend=config.sparse_backend,
                        sparse_device=config.sparse_device,
                        sparse_batch_size=config.batch_size,
                        mds_solver=config.mds_solver, n_components=config.n_components,
                        verbose=False, random_state=seed,
                    )
                else:
                    if n > 5000:
                        continue
                    op = phate.PHATE(
                        knn=config.knn, t=config.t, gamma=config.gamma,
                        mds_solver=config.mds_solver, n_components=config.n_components,
                        verbose=False, random_state=seed,
                    )
                t0 = time.time()
                try:
                    emb = op.fit_transform(data)
                    times.append(time.time() - t0)
                except Exception as e:
                    print(f"{n:>8} {label:>22} FAILED: {e}")
                    break
            if times:
                print(f"{n:>8} {label:>22} {fmt_ms(times):>15}")
        print()


# ══════════════════════════════════════════════════════════════════════════
# 6. End-to-end
# ══════════════════════════════════════════════════════════════════════════


def benchmark_end_to_end(config):
    sizes = [s for s in config.sizes if s <= 20000]
    if not sizes:
        return
    print("=" * 100)
    print("6. END-TO-END PIPELINE")
    print(f"   {config.n_features}-dim, backend={config.sparse_backend}, "
          f"device={config.sparse_device}, solver={config.mds_solver}")
    print("=" * 100)
    print(f"{'N':>8} {'Config':>24} {'Total(s)':>15}")
    print("-" * 49)

    for n in sizes:
        data, _ = gen_data(config, n, seed=42)

        # Sparse path
        if config.tighten_k:
            opt_k, _ = resolve_k(config, data, verbose=False)
            actual_k = opt_k
        else:
            actual_k = config.sparse_k

        times = []
        for seed in config.seeds[:3]:
            op = phate.PHATE(
                knn=config.knn, t=config.t, gamma=config.gamma,
                sparse_k=actual_k, sparse_metric=config.sparse_metric,
                sparse_backend=config.sparse_backend,
                sparse_device=config.sparse_device,
                sparse_batch_size=config.batch_size,
                mds_solver=config.mds_solver, n_components=config.n_components,
                verbose=False, random_state=seed,
            )
            t0 = time.time()
            emb = op.fit_transform(data)
            times.append(time.time() - t0)
        print(f"{n:>8} {'sparse (k=' + str(actual_k) + ')':>24} {fmt_ms(times):>15}")

        # Dense path (for reference, skip large N)
        if n <= 5000:
            times2 = []
            for seed in config.seeds[:3]:
                op2 = phate.PHATE(
                    knn=config.knn, t=config.t, gamma=config.gamma,
                    mds_solver=config.mds_solver, n_components=config.n_components,
                    verbose=False, random_state=seed,
                )
                t0 = time.time()
                emb2 = op2.fit_transform(data)
                times2.append(time.time() - t0)
            print(f"{n:>8} {'dense (graphtools)':>24} {fmt_ms(times2):>15}")
        print()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def parse_args():
    """Simple CLI: python benchmark_sparse_phate.py --sparse-k 20 --sparse-device mps"""
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--config":
            return Config.from_json(sys.argv[i + 1])
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:].replace("-", "_")
            val = sys.argv[i + 1]
            # Type coercion
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif val.replace(".", "").isdigit():
                val = float(val) if "." in val else int(val)
            args[key] = val
            i += 2
        else:
            i += 1
    if args:
        return Config(**args)
    return Config()


if __name__ == "__main__":
    config = parse_args()

    # Ensure output dir
    if config.output_dir and config.output_dir != "benchmark_results":
        os.makedirs(config.output_dir, exist_ok=True)

    print("PHATE Systematic Sparse Benchmarks")
    print(f"Config: {config.n_features}-dim, k={config.sparse_k}, "
          f"decay={config.decay}, backend={config.sparse_backend}, "
          f"device={config.sparse_device}, tighten_k={config.tighten_k}")
    print(f"Sizes: {config.sizes}, seeds={len(config.seeds)} runs")
    print()

    benchmarks = [
        (config.run_similarity, benchmark_similarity, "Similarity timing"),
        (config.run_connectivity, benchmark_connectivity, "Connectivity"),
        (config.run_spectral, benchmark_spectral, "Spectral fidelity"),
        (config.run_fill_in, benchmark_fill_in, "Fill-in"),
        (config.run_mds, benchmark_mds, "MDS"),
        (config.run_end_to_end, benchmark_end_to_end, "End-to-end"),
    ]

    for enabled, fn, name in benchmarks:
        if enabled:
            fn(config)
        else:
            print(f"[SKIP] {name}")

    print("\nDone.")
