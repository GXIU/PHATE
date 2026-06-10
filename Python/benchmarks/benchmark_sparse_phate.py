#!/usr/bin/env python
"""Systematic sparse PHATE benchmarks — config-driven CLI.

Usage:
    python benchmarks/benchmark_sparse_phate.py                      # defaults, all benchmarks
    python benchmarks/benchmark_sparse_phate.py --sparse-k 20        # override k
    python benchmarks/benchmark_sparse_phate.py --config my.yml      # load YAML/JSON config
    python benchmarks/benchmark_sparse_phate.py --show-config        # print config and exit
    python benchmarks/benchmark_sparse_phate.py --save-config cfg.yml  # save as YAML/JSON

All Config dataclass fields are exposed as CLI flags.  Use --help for the full list.
"""

from dataclasses import dataclass, field, asdict, fields as dc_fields
import argparse
import time
import sys
import os
import json
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.sparse import issparse
import phate
from phate import sparse_similarity
from phate import vne


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    """Benchmark configuration — all tunable parameters in one place."""

    # Data generation
    sizes: list = field(default_factory=lambda: [500, 2000, 5000, 10000, 20000, 50000, 100000])
    n_features: int = 512
    float_precision: str = "float32"       # "float16", "float32", "float64"
    seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Sparse similarity
    sparse_k: int = 10
    sparse_metric: str = "euclidean"
    decay: float = 40.0
    batch_size: int = 256

    # Backend
    sparse_backend: str = "torch"
    sparse_device: str = "cpu"

    # Connectivity
    tighten_k: bool = False
    k_max: int = 200

    # PHATE parameters
    knn: int = 5
    t: int = 2                    # diffusion steps (keep low to preserve sparsity)
    gamma: float = 1.0
    mds_solver: str = "sgd"
    mds_how: str = "metric"
    n_components: int = 2

    # I/O
    input_dir: str = ""
    output_dir: str = "benchmark_results"
    plot_on: bool = True
    plot_format: str = "png"             # "png" (matplotlib) or "html" (plotly interactive)

    # Benchmark selection
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
        return cls._from_dict(d)

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls._from_dict(d)

    def save(self, path):
        """Save config to JSON or YAML (detected from file extension)."""
        if path.endswith((".yaml", ".yml")):
            self.to_yaml(path)
        else:
            self.to_json(path)

    @classmethod
    def load(cls, path):
        """Load config from JSON or YAML (detected from file extension)."""
        if path.endswith((".yaml", ".yml")):
            return cls.from_yaml(path)
        else:
            return cls.from_json(path)

    @classmethod
    def _from_dict(cls, d):
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_dtype(config):
    """Convert float_precision string to numpy dtype."""
    return {"float16": np.float16, "float32": np.float32, "float64": np.float64}[config.float_precision]


def gen_data(config, n, seed=42):
    data, clusters = phate.tree.gen_dla(
        n_dim=config.n_features,
        n_branch=max(1, n // 100),
        branch_length=100,
        seed=seed,
    )
    return data.astype(_resolve_dtype(config)), clusters


def load_tabular(path, dtype=np.float64):
    """Load a single tabular data file.

    Supports .csv, .tsv, .parquet, .npy, .h5ad.
    Returns (data, labels) where data is (n_samples, n_features) array
    and labels is a 1D array of strings or None.
    """
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    labels = None

    if ext == ".npy":
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
        data, labels = _df_to_array(df)
    elif ext in (".csv", ".tsv", ".txt", ".gz"):
        sep = "\t" if ext in (".tsv", ".txt") else ","
        df = pd.read_csv(path, sep=sep)
        data, labels = _df_to_array(df)
    elif ext in (".h5ad", ".h5"):
        import anndata
        adata = anndata.read_h5ad(path)
        data = adata.X
        if hasattr(data, "toarray"):
            data = data.toarray()
        labels = adata.obs_names.values if hasattr(adata, "obs_names") else None
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Cast to requested precision
    data = np.asarray(data, dtype=dtype, order="C")
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data, labels


def _df_to_array(df):
    """Extract numeric data from DataFrame. First column used as labels if non-numeric."""
    labels = None
    first_col = df.columns[0]
    try:
        df[first_col].astype(float)
    except (ValueError, TypeError):
        labels = df[first_col].values
        df = df.drop(columns=[first_col])
    data = df.select_dtypes(include=[np.number]).values
    return data, labels


def discover_datasets(input_dir):
    """Scan input_dir for loadable data files. Returns list of (name, path) tuples."""
    if not input_dir or not os.path.exists(input_dir):
        return []
    if os.path.isfile(input_dir):
        name = os.path.splitext(os.path.basename(input_dir))[0]
        return [(name, input_dir)]
    supported = {".csv", ".tsv", ".parquet", ".npy", ".h5ad", ".txt"}
    datasets = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            name = os.path.splitext(fname)[0]
            datasets.append((name, os.path.join(input_dir, fname)))
    return datasets


class Dataset:
    """A benchmark dataset — either synthetic (generated per-size) or real (loaded from file)."""

    def __init__(self, name, n, gen_fn=None, data=None, labels=None):
        self.name = name
        self.n = n
        self._gen_fn = gen_fn          # callable(seed) -> (data, labels)
        self._data = data              # preloaded data (for real files)
        self._labels = labels
        self.dim = data.shape[1] if data is not None else None

    def get(self, seed=42):
        """Return (data, labels) for this dataset."""
        if self._data is not None:
            return self._data, self._labels
        return self._gen_fn(seed)


def iter_datasets(config):
    """Yield Dataset objects for benchmarking.

    When input_dir is set: yields one Dataset per discovered file (real data).
    When input_dir is empty: yields synthetic Datasets at each configured size.
    """
    files = discover_datasets(config.input_dir)
    if files:
        dtype = _resolve_dtype(config)
        for name, path in files:
            data, labels = load_tabular(path, dtype=dtype)
            yield Dataset(name, data.shape[0], data=data, labels=labels)
    else:
        for n in config.sizes:
            def _gen(cfg, size, s):
                return lambda seed: gen_data(cfg, size, seed)
            yield Dataset(f"N={n}", n, gen_fn=_gen(config, n, None))
    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint — save/load intermediate results so we can resume from any stage
# ═══════════════════════════════════════════════════════════════════════════


def _ckpt_path(output_dir, name, tag):
    """Path for a checkpoint file: output_dir/name.tag.npz"""
    safe = name.replace("/", "_").replace(" ", "_")
    return os.path.join(output_dir, f"{safe}.{tag}.npz")


def _save_ckpt(path, matrix, meta=None):
    """Save a scipy sparse matrix (or numpy array) to .npz with optional metadata."""
    from scipy import sparse as sp_sparse
    if sp_sparse.issparse(matrix):
        sp_sparse.save_npz(path, matrix)
    else:
        np.savez_compressed(path, data=matrix, **(meta or {}))


def _load_ckpt(path):
    """Load a checkpoint. Returns (matrix, meta_dict) or (None, {}) if not found."""
    from scipy import sparse as sp_sparse
    if not os.path.exists(path):
        return None, {}
    try:
        mat = sp_sparse.load_npz(path)
        return mat, {}
    except (ValueError, TypeError):
        pass
    try:
        f = np.load(path, allow_pickle=True)
        if "data" in f:
            return f["data"], {k: f[k] for k in f.files if k != "data"}
        return f[list(f.files)[0]], {}
    except Exception:
        return None, {}


# ═══════════════════════════════════════════════════════════════════════════


def compute_sparse(config, data, k=None):
    """Dispatch sparse similarity to the configured backend."""
    k = k or config.sparse_k
    if config.sparse_backend == "torch":
        return sparse_similarity.compute_sparse_similarity_torch(
            data, k=k, metric=config.sparse_metric, decay=config.decay,
            batch_size=config.batch_size, device=config.sparse_device, verbose=0,
        )
    else:
        return sparse_similarity.compute_sparse_similarity(
            data, k=k, metric=config.sparse_metric, decay=config.decay,
            batch_size=config.batch_size, verbose=0,
        )


def mean_std(arr):
    return np.mean(arr), np.std(arr, ddof=1)


def fmt_ms(arr):
    m, s = mean_std(arr)
    return f"{m:.3f} ± {s:.3f}"


def resolve_k(config, data, verbose=True):
    if config.tighten_k:
        k_opt, S, _ = sparse_similarity.find_minimal_k(
            data,
            k_start=config.sparse_k,
            k_max=min(config.k_max, data.shape[0] - 1),
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. Similarity computation timing
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# 2. Connectivity scaling
# ═══════════════════════════════════════════════════════════════════════════


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
            k_opt, _, _ = sparse_similarity.find_minimal_k(
                data, k_start=10, k_max=min(config.k_max, n - 1),
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


# ═══════════════════════════════════════════════════════════════════════════
# 3. Spectral fidelity
# ═══════════════════════════════════════════════════════════════════════════


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

    from scipy import sparse as sp_sparse

    for n in sizes:
        data, _ = gen_data(config, n, seed=42)
        opt_k, _ = resolve_k(config, data, verbose=False)
        S = compute_sparse(config, data, k=opt_k)
        P = sparse_similarity.compute_sparse_diffusion_operator(S)
        P_dense = P.toarray()

        k_eigs = min(100, n - 2)
        eig_sp = np.sort(np.abs(sp_sparse.linalg.eigsh(
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
        knee_delta = abs(k_sp - k_de)

        spec_mass = np.sum(eig_sp) / (np.sum(eig_de) + 1e-10)
        print(
            f"{n:>8} {top10_err:>12.4f} {top50_err:>12.4f} "
            f"{vne_c:>10.4f} {knee_delta:>12} {spec_mass:>10.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Diffusion power fill-in
# ═══════════════════════════════════════════════════════════════════════════


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
        S = compute_sparse(config, data, k=k_val)
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


# ═══════════════════════════════════════════════════════════════════════════
# 5. MDS optimization
# ═══════════════════════════════════════════════════════════════════════════


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
                        mds=config.mds_how,
                        mds_solver=config.mds_solver, n_components=config.n_components,
                        verbose=False, random_state=seed,
                    )
                else:
                    if n > 5000:
                        continue
                    op = phate.PHATE(
                        knn=config.knn, t=config.t, gamma=config.gamma,
                        mds=config.mds_how,
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


# ═══════════════════════════════════════════════════════════════════════════
# 6. End-to-end
# ═══════════════════════════════════════════════════════════════════════════


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
                mds=config.mds_how,
                mds_solver=config.mds_solver, n_components=config.n_components,
                verbose=False, random_state=seed,
            )
            t0 = time.time()
            emb = op.fit_transform(data)
            times.append(time.time() - t0)
        print(f"{n:>8} {'sparse (k=' + str(actual_k) + ')':>24} {fmt_ms(times):>15}")

        if n <= 5000:
            times2 = []
            for seed in config.seeds[:3]:
                op2 = phate.PHATE(
                    knn=config.knn, t=config.t, gamma=config.gamma,
                    mds=config.mds_how,
                    mds_solver=config.mds_solver, n_components=config.n_components,
                    verbose=False, random_state=seed,
                )
                t0 = time.time()
                emb2 = op2.fit_transform(data)
                times2.append(time.time() - t0)
            print(f"{n:>8} {'dense (graphtools)':>24} {fmt_ms(times2):>15}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks that produce plots for visual comparison
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_plot_comparison(config):
    """Generate sparse-vs-dense comparison plots.

    For N ≤ 5000: runs both sparse and dense paths, aligned via Procrustes.
    For N > 5000: sparse path only (dense path would OOM).
    Output: PNG (matplotlib) or HTML (plotly).
    """
    ndim = config.n_components
    fmt = config.plot_format
    dtype = _resolve_dtype(config)

    datasets = discover_datasets(config.input_dir)
    if datasets:
        name, path = datasets[0]
        data, labels = load_tabular(path, dtype=dtype)
        n = data.shape[0]
        desc = f"{name} ({n} × {data.shape[1]})"
    else:
        n = min(5000, config.sizes[-1] if config.sizes else 5000)
        data, _ = gen_data(config, n, seed=42)
        desc = f"N={n}, {config.n_features}-dim"

    print(f"Plot comparison: {desc}, k={config.sparse_k}, metric={config.sparse_metric}, "
          f"ndim={ndim}, fmt={fmt}, precision={config.float_precision}")
    print(f"  Data: {data.shape}, dtype={data.dtype}, memory={data.nbytes / 1e6:.1f} MB")

    os.makedirs(config.output_dir, exist_ok=True)

    # Sparse path — always run
    t0 = time.time()
    emb_s = phate.PHATE(
        knn=config.knn, t=config.t, gamma=config.gamma,
        sparse_k=config.sparse_k, sparse_metric=config.sparse_metric,
        sparse_backend=config.sparse_backend, sparse_device=config.sparse_device,
        sparse_batch_size=config.batch_size,
        mds=config.mds_how,
        mds_solver=config.mds_solver, n_components=ndim,
        verbose=False, random_state=42,
    ).fit_transform(data)
    t_sparse = time.time() - t0
    print(f"  Sparse path: {t_sparse:.1f}s")

    # Dense path — only for tractable N
    if n <= 5000:
        t0 = time.time()
        emb_d = phate.PHATE(
            knn=config.knn, t=config.t, gamma=config.gamma,
            mds=config.mds_how,
            mds_solver=config.mds_solver, n_components=ndim,
            verbose=False, random_state=42,
        ).fit_transform(data)
        t_dense = time.time() - t0
        print(f"  Dense path: {t_dense:.1f}s")
    else:
        emb_d = None
        print(f"  Dense path: skipped (N={n} > 5000, would OOM)")

    if fmt == "html":
        _plot_html(emb_s, emb_d, ndim, n, config.output_dir)
    else:
        _plot_png(emb_s, emb_d, ndim, n, config.output_dir)


def _plot_png(emb_s, emb_d, ndim, n, output_dir):
    """Matplotlib static plot — side-by-side comparison or sparse-only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_dense = emb_d is not None

    if has_dense:
        from scipy.linalg import orthogonal_procrustes
        R, _ = orthogonal_procrustes(emb_s, emb_d)
        emb_s_aligned = emb_s @ R
        corr = np.corrcoef(emb_d.ravel(), emb_s_aligned.ravel())[0, 1]

    if ndim == 3:
        if has_dense:
            fig = plt.figure(figsize=(18, 5.5))
            panels = [
                ("Dense PHATE", emb_d, False),
                ("Sparse PHATE", emb_s, False),
                (f"Overlay (corr={corr:.4f})", np.vstack([emb_d, emb_s_aligned]), True),
            ]
            for i, (title, emb, is_overlay) in enumerate(panels):
                ax = fig.add_subplot(1, 3, i + 1, projection="3d")
                xs, ys, zs = emb[:, 0], emb[:, 1], emb[:, 2]
                if is_overlay:
                    half = n
                    ax.scatter(xs[:half], ys[:half], zs[:half], c="blue", s=4, alpha=0.5, label="dense")
                    ax.scatter(xs[half:], ys[half:], zs[half:], c="red", s=4, alpha=0.5, label="sparse")
                    ax.legend()
                else:
                    ax.scatter(xs, ys, zs, c=np.arange(len(emb)), cmap="viridis", s=5, alpha=0.7)
                ax.set_title(title)
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(emb_s[:, 0], emb_s[:, 1], emb_s[:, 2], c=np.arange(n), cmap="viridis", s=3, alpha=0.7)
            ax.set_title(f"Sparse PHATE (N={n})\nstd={np.std(emb_s):.4f}")
    else:
        if has_dense:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            axes[0].scatter(emb_d[:, 0], emb_d[:, 1], c=np.arange(n), cmap="viridis", s=5, alpha=0.7)
            axes[0].set_title(f"Dense PHATE\nstd={np.std(emb_d):.4f}")
            axes[0].set_aspect("equal")
            axes[1].scatter(emb_s[:, 0], emb_s[:, 1], c=np.arange(n), cmap="viridis", s=5, alpha=0.7)
            axes[1].set_title(f"Sparse PHATE\nstd={np.std(emb_s):.4f}")
            axes[1].set_aspect("equal")
            axes[2].scatter(emb_d[:, 0], emb_d[:, 1], c="blue", s=4, alpha=0.5, label="dense")
            axes[2].scatter(emb_s_aligned[:, 0], emb_s_aligned[:, 1], c="red", s=4, alpha=0.5, label="sparse")
            axes[2].set_title(f"Overlay (Procrustes)\ncorr={corr:.4f}")
            axes[2].legend()
            axes[2].set_aspect("equal")
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(emb_s[:, 0], emb_s[:, 1], c=np.arange(n), cmap="viridis", s=3, alpha=0.7)
            ax.set_title(f"Sparse PHATE (N={n})\nstd={np.std(emb_s):.4f}")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

    path = os.path.join(output_dir, "sparse_vs_dense_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_html(emb_s, emb_d, ndim, n, output_dir):
    """Plotly interactive HTML — sparse (+ dense if available)."""
    import plotly.graph_objects as go

    has_dense = emb_d is not None
    fig = go.Figure()

    trace_cls = go.Scatter3d if ndim == 3 else go.Scatter
    marker = dict(size=2, opacity=0.7)
    coords = (lambda e: dict(x=e[:, 0], y=e[:, 1], z=e[:, 2])) if ndim == 3 else \
             (lambda e: dict(x=e[:, 0], y=e[:, 1]))

    if has_dense:
        from scipy.linalg import orthogonal_procrustes
        R, _ = orthogonal_procrustes(emb_s, emb_d)
        emb_s_aligned = emb_s @ R
        corr = np.corrcoef(emb_d.ravel(), emb_s_aligned.ravel())[0, 1]
        fig.add_trace(trace_cls(
            **coords(emb_d), mode="markers", name="Dense PHATE",
            marker=dict(**marker, color=np.arange(n), colorscale="Viridis"), visible=True,
        ))
        fig.add_trace(trace_cls(
            **coords(emb_s_aligned), mode="markers", name="Sparse PHATE (aligned)",
            marker=dict(**marker, color=np.arange(n), colorscale="Plasma"), visible="legendonly",
        ))
        title = f"Sparse vs Dense PHATE — Procrustes corr: {corr:.4f}"
    else:
        fig.add_trace(trace_cls(
            **coords(emb_s), mode="markers", name="Sparse PHATE",
            marker=dict(**marker, color=np.arange(n), colorscale="Viridis"),
        ))
        title = f"Sparse PHATE (N={n})"

    fig.update_layout(
        title=title,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=900, height=700,
    )
    if ndim == 3:
        fig.update_layout(scene=dict(**axis_kw, aspectmode="data"))

    path = os.path.join(output_dir, "sparse_vs_dense_comparison.html")
    fig.write_html(path)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI — auto-generated from Config dataclass fields
# ═══════════════════════════════════════════════════════════════════════════


FIELD_META = {
    "sizes":            {"type": "int_list", "help": "N values to benchmark"},
    "n_features":       {"type": "int",      "help": "Dimensionality of generated data"},
    "float_precision":  {"type": "str",      "help": "Floating-point precision", "choices": ["float16", "float32", "float64"]},
    "seeds":            {"type": "int_list", "help": "Random seeds for reproducibility"},
    "sparse_k":         {"type": "int",      "help": "Top-k neighbors per row in sparse similarity"},
    "sparse_metric":    {"type": "str",      "help": "Distance/similarity metric", "choices": ["euclidean", "cosine", "correlation"]},
    "decay":            {"type": "float",    "help": "Kernel decay (sigma); None for raw distances"},
    "batch_size":       {"type": "int",      "help": "Batch size for sparse similarity computation"},
    "sparse_backend":   {"type": "str",      "help": "Backend for sparse similarity", "choices": ["scipy", "torch"]},
    "sparse_device":    {"type": "str",      "help": "Device for torch backend", "choices": ["cpu", "cuda", "mps"]},
    "tighten_k":        {"type": "bool",     "help": "Binary-search for minimal connected k"},
    "k_max":            {"type": "int",      "help": "Upper bound for k search"},
    "knn":              {"type": "int",      "help": "kNN for dense PHATE path"},
    "t":                {"type": "int",      "help": "Diffusion time (fixed, or max if auto)"},
    "gamma":            {"type": "float",    "help": "Information distance parameter (-1 to 1)"},
    "mds_solver":       {"type": "str",      "help": "MDS solver", "choices": ["sgd", "smacof"]},
    "mds_how":          {"type": "str",      "help": "MDS type", "choices": ["classic", "metric", "nonmetric"]},
    "n_components":     {"type": "int",      "help": "Embedding dimensions"},
    "input_dir":        {"type": "str",      "help": "Directory with input data (empty = generated)"},
    "output_dir":       {"type": "str",      "help": "Directory for CSVs and plots"},
    "plot_on":          {"type": "bool",     "help": "Generate comparison plots"},
    "plot_format":      {"type": "str",      "help": "Plot output format", "choices": ["png", "html"]},
    "run_similarity":   {"type": "bool",     "help": "Run similarity timing benchmark"},
    "run_connectivity": {"type": "bool",     "help": "Run connectivity scaling benchmark"},
    "run_spectral":     {"type": "bool",     "help": "Run spectral fidelity benchmark"},
    "run_fill_in":      {"type": "bool",     "help": "Run diffusion fill-in benchmark"},
    "run_mds":          {"type": "bool",     "help": "Run MDS optimization benchmark"},
    "run_end_to_end":   {"type": "bool",     "help": "Run end-to-end pipeline benchmark"},
}


def _parse_list(s, convert=int):
    """Parse a comma-separated list like '500,2000,5000'."""
    return [convert(x.strip()) for x in s.split(",") if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="benchmark_sparse_phate",
        description="Systematic sparse PHATE benchmarks — config-driven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All Config fields can be overridden from CLI. Use --config to load a JSON baseline.",
    )

    # ── Special flags (not dataclass fields) ──
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (loaded first, then overridden by any other flags)")
    parser.add_argument("--show-config", action="store_true",
                        help="Print resolved config and exit")
    parser.add_argument("--save-config", type=str, default=None, metavar="PATH",
                        help="Save resolved config to JSON and exit")
    parser.add_argument("--plot-comparison", action="store_true",
                        help="Run visual sparse-vs-dense comparison plot")
    parser.add_argument("--all", action="store_true", dest="_all_benchmarks",
                        help="Run all benchmarks (default if none selected)")

    # Shorthand aliases for benchmark selection
    _bench_aliases = {
        "similarity": "run_similarity", "connectivity": "run_connectivity",
        "spectral": "run_spectral", "fill-in": "run_fill_in",
        "mds": "run_mds", "end-to-end": "run_end_to_end",
    }
    for short, full in _bench_aliases.items():
        parser.add_argument(f"--{short}", action="store_true", default=None, dest=full,
                            help=f"Run {short} benchmark")
        parser.add_argument(f"--no-{short}", action="store_false", default=None, dest=full,
                            help=f"Skip {short} benchmark")

    # ── Auto-generate flags from Config dataclass ──
    for f in dc_fields(Config):
        meta = FIELD_META.get(f.name, {})
        ftype = meta.get("type", f.type.__name__ if hasattr(f.type, '__name__') else str(f.type))
        flag = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not f.default_factory else f.default_factory()

        if ftype == "bool":
            # Boolean: --flag / --no-flag
            parser.add_argument(flag, action="store_true", default=None, dest=f.name,
                                help=meta.get("help", ""))
            parser.add_argument("--no-" + f.name.replace("_", "-"), action="store_false",
                                dest=f.name, help=f"Disable {f.name}")
        elif ftype == "int_list":
            parser.add_argument(flag, type=lambda s: _parse_list(s, int), default=None,
                                help=meta.get("help", ""))
        elif ftype == "str":
            kwargs = {"type": str, "default": None, "help": meta.get("help", "")}
            if "choices" in meta:
                kwargs["choices"] = meta["choices"]
            parser.add_argument(flag, **kwargs)
        elif ftype == "float":
            parser.add_argument(flag, type=float, default=None,
                                help=meta.get("help", ""))
        elif ftype == "int":
            parser.add_argument(flag, type=int, default=None,
                                help=meta.get("help", ""))
        else:
            parser.add_argument(flag, type=str, default=None,
                                help=meta.get("help", ""))

    return parser


def resolve_config(args) -> Config:
    """Build Config: JSON baseline → dataclass defaults → CLI overrides."""
    # Start with config file if provided (JSON or YAML, auto-detected)
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # Apply CLI overrides (only non-None values)
    for f in dc_fields(Config):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(config, f.name, val)

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args)

    # --all enables every benchmark
    if getattr(args, "_all_benchmarks", False):
        for f in dc_fields(Config):
            if f.name.startswith("run_"):
                setattr(config, f.name, True)

    # --show-config
    if args.show_config:
        print(json.dumps(asdict(config), indent=2, default=str))
        sys.exit(0)

    # --save-config
    if args.save_config:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")
        sys.exit(0)

    # Ensure output dir
    os.makedirs(config.output_dir, exist_ok=True)

    print("PHATE Systematic Sparse Benchmarks")
    print(f"Config: {config.n_features}-dim, k={config.sparse_k}, "
          f"decay={config.decay}, backend={config.sparse_backend}, "
          f"device={config.sparse_device}, tighten_k={config.tighten_k}")
    print(f"Sizes: {config.sizes}, seeds={len(config.seeds)} runs")
    print()

    # --plot-comparison (standalone plot, or with benchmarks)
    if args.plot_comparison:
        if config.plot_on:
            benchmark_plot_comparison(config)
        else:
            print("[SKIP] Plot comparison (plot_on=False)")

    # Run selected benchmarks
    benchmarks = [
        (config.run_similarity, benchmark_similarity, "Similarity timing"),
        (config.run_connectivity, benchmark_connectivity, "Connectivity"),
        (config.run_spectral, benchmark_spectral, "Spectral fidelity"),
        (config.run_fill_in, benchmark_fill_in, "Fill-in"),
        (config.run_mds, benchmark_mds, "MDS"),
        (config.run_end_to_end, benchmark_end_to_end, "End-to-end"),
    ]

    any_ran = False
    for enabled, fn, name in benchmarks:
        if enabled:
            fn(config)
            any_ran = True
        else:
            print(f"[SKIP] {name}")

    if not any_ran and not args.plot_comparison:
        print("No benchmarks selected. Use --help to see options.")

    print("\nDone.")
