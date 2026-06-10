"""Sparse similarity computation with batched top-k selection.

Computes pairwise similarities in batches to avoid materializing a dense
N x N matrix. For each batch of rows, the full N-column similarity is
computed, then only the top-k entries per row are retained. The result is
a scipy.sparse.csr_matrix.

Memory: O(batch_size * N) for the batch similarity matrix + O(N * k) for
the sparse result. For N=20000, batch_size=256, float64: ~40 MB peak.
"""

import numpy as np
import tasklogger
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from scipy import sparse
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

_logger = tasklogger.get_tasklogger("graphtools")


def compute_sparse_similarity(
    X,
    k=10,
    metric="euclidean",
    decay=40,
    batch_size=256,
    verbose=1,
):
    """Compute a sparse similarity matrix using batched top-k selection.

    For each batch of rows, pairwise distances/similarities are computed
    against ALL columns. Only the top-k entries per row are retained.
    Distances are converted to affinities via an alpha-decaying kernel:
    ``affinity = exp(-distance / decay)``. All other entries are zero.

    For distance metrics: keeps the k smallest distances, applies kernel.
    For similarity metrics: keeps the k largest values.

    Parameters
    ----------
    X : ndarray, shape=[n_samples, n_features]
        Input data matrix.
    k : int
        Number of top entries to retain per row.
    metric : str
        Distance/similarity metric for ``scipy.spatial.distance.cdist``.
    decay : float or None
        Alpha decay parameter for kernel: ``exp(-d / decay)``.
        If None, distances are stored as-is.
    batch_size : int
        Number of rows to process per batch. Controls peak memory.
    verbose : int
        Verbosity level.

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape=[n_samples, n_samples]
        Sparse similarity matrix. Symmetrized: (S + S.T) / 2.
    """
    n_samples, n_features = X.shape
    k = min(k, n_samples)

    # Determine whether metric is a similarity (keep largest)
    similarity_metrics = {"cosine", "correlation"}
    is_similarity = metric in similarity_metrics

    # Accumulate COO entries: (row, col, value)
    all_rows = []
    all_cols = []
    all_vals = []

    kernel_str = f"exp(-d/{decay})" if decay else "raw"
    _logger.log_info(
        f"Computing sparse {k}-NN similarity on {n_samples} samples "
        f"with metric='{metric}', kernel={kernel_str}, "
        f"batch_size={batch_size}..."
    )

    n_batches = (n_samples + batch_size - 1) // batch_size
    progress = None
    if verbose > 0:
        progress = Progress(
            TextColumn("  [bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>5.0f}%"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Sparse similarity", total=n_batches)
        progress.start()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        batch = X[batch_start:batch_end]  # (batch_sz, n_features)

        # Compute distances/similarities for this batch against all samples
        pair = cdist(batch, X, metric=metric)  # (batch_sz, n_samples)

        for i in range(pair.shape[0]):
            global_i = batch_start + i
            row = pair[i]

            if is_similarity:
                if k < n_samples:
                    idx = np.argpartition(-row, k)[:k]
                else:
                    idx = np.arange(n_samples)
                top_vals = row[idx].astype(np.float64, copy=True)
            else:
                if k < n_samples:
                    idx = np.argpartition(row, k)[:k]
                else:
                    idx = np.arange(n_samples)
                top_vals = row[idx].astype(np.float64, copy=True)
                if decay is not None:
                    np.exp(-top_vals / decay, out=top_vals)

            all_rows.extend([global_i] * len(idx))
            all_cols.extend(idx)
            all_vals.extend(top_vals)

        if progress is not None:
            progress.update(task, advance=1)
        elif verbose > 0 and (batch_end % (batch_size * 10) == 0 or batch_end == n_samples):
            _logger.log_info(f"  Processed {batch_end}/{n_samples} samples...")

    if progress is not None:
        progress.stop()

    n_entries = len(all_rows)
    _logger.log_info(f"Building sparse matrix with {n_entries} entries...")

    S_coo = coo_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_samples, n_samples),
    )
    S = S_coo.tocsr()

    # Symmetrize: average with transpose for undirected graph
    S = (S + S.T) * 0.5

    _logger.log_info(
        f"Result: {S.nnz} non-zeros ({100 * S.nnz / (n_samples * n_samples):.3f}% dense)"
    )

    return S


def compute_sparse_similarity_torch(
    X,
    k=10,
    metric="euclidean",
    decay=40,
    batch_size=256,
    device="cpu",
    verbose=1,
):
    """Torch-accelerated sparse similarity computation.

    Uses PyTorch for GPU/optimized-CPU pairwise distance computation
    and top-k selection. Falls back to scipy if torch is unavailable.

    For ``metric='correlation'``, uses matrix multiplication after
    z-score normalization (much faster than ``cdist``).

    Parameters
    ----------
    X : ndarray, shape=[n_samples, n_features]
        Input data matrix.
    k : int
        Number of top entries to retain per row.
    metric : str
        ``'euclidean'``, ``'cosine'``, or ``'correlation'``.
    decay : float or None
        Alpha decay parameter for kernel.
    batch_size : int
        Rows per batch.
    device : str
        Torch device (``'cpu'``, ``'cuda'``, ``'mps'``).
    verbose : int
        Verbosity level.

    Returns
    -------
    S : scipy.sparse.csr_matrix
    """
    n_samples = X.shape[0]
    k = min(k, n_samples)
    is_similarity = metric in {"cosine", "correlation"}

    X_t = torch.from_numpy(X.astype(np.float32)).to(device)

    kernel_str = f"exp(-d/{decay})" if decay else "raw"
    _logger.log_info(
        f"Computing sparse {k}-NN similarity (torch, {device}) on {n_samples} samples "
        f"with metric='{metric}', kernel={kernel_str}, batch_size={batch_size}..."
    )

    all_rows = []
    all_cols = []
    all_vals = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    progress = None
    if verbose > 0:
        progress = Progress(
            TextColumn("  [bold green]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>5.0f}%"),
            TimeRemainingColumn(),
        )
        task = progress.add_task(f"Sparse similarity (torch, {device})", total=n_batches)
        progress.start()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        batch = X_t[batch_start:batch_end]

        if metric == "correlation":
            b_centered = batch - batch.mean(dim=1, keepdim=True)
            b_norm = b_centered / (b_centered.std(dim=1, keepdim=True) + 1e-10)
            x_centered = X_t - X_t.mean(dim=1, keepdim=True)
            x_norm = x_centered / (x_centered.std(dim=1, keepdim=True) + 1e-10)
            pair = 1 - b_norm @ x_norm.T
        elif metric == "cosine":
            b_norm = batch / (batch.norm(dim=1, keepdim=True) + 1e-10)
            x_norm = X_t / (X_t.norm(dim=1, keepdim=True) + 1e-10)
            pair = 1 - b_norm @ x_norm.T
        else:
            pair = torch.cdist(batch, X_t)

        if is_similarity:
            top_vals, top_idx = torch.topk(pair, k, dim=1, largest=True)
        else:
            top_vals, top_idx = torch.topk(pair, k, dim=1, largest=False)

        top_vals = top_vals.cpu().numpy().astype(np.float64)
        top_idx = top_idx.cpu().numpy()

        if decay is not None and not is_similarity:
            top_vals = np.exp(-top_vals / decay)

        for i in range(top_idx.shape[0]):
            global_i = batch_start + i
            all_rows.extend([global_i] * k)
            all_cols.extend(top_idx[i])
            all_vals.extend(top_vals[i])

        if progress is not None:
            progress.update(task, advance=1)
        elif verbose > 0 and (batch_end % (batch_size * 10) == 0 or batch_end == n_samples):
            _logger.log_info(f"  Processed {batch_end}/{n_samples} samples...")

    if progress is not None:
        progress.stop()

    n_entries = len(all_rows)
    _logger.log_info(f"Building sparse matrix with {n_entries} entries...")

    S_coo = coo_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_samples, n_samples),
    )
    S = S_coo.tocsr()
    S = (S + S.T) * 0.5

    _logger.log_info(
        f"Result: {S.nnz} non-zeros ({100 * S.nnz / (n_samples * n_samples):.3f}% dense)"
    )

    return S


def compute_sparse_diffusion_operator(S):
    """Row-normalize a sparse similarity matrix into a diffusion operator.

    The diffusion operator P is the row-stochastic transition probability
    matrix: P = D^{-1} S, where D is the diagonal degree matrix.

    This is a sparse-aware implementation that avoids densifying.

    Parameters
    ----------
    S : scipy.sparse.csr_matrix
        Sparse similarity/affinity matrix.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        Row-normalized diffusion operator.
    """
    if not issparse(S):
        return _dense_row_normalize(S)

    # Compute row sums (degrees)
    row_sums = np.array(S.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero

    # D^{-1} S: scale each row by its inverse degree
    inv_degrees = 1.0 / row_sums
    D_inv = sparse.diags(inv_degrees)
    P = D_inv @ S

    return P.tocsr()


def _dense_row_normalize(S):
    """Fallback row normalization for dense input."""
    row_sums = S.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return S / row_sums[:, np.newaxis]


def check_connectivity(S):
    """Check if the graph represented by sparse matrix S is connected.

    Parameters
    ----------
    S : scipy.sparse.csr_matrix or ndarray
        Adjacency/similarity matrix.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : ndarray
        Component label for each node.
    """
    if issparse(S):
        n_components, labels = connected_components(
            S, directed=False, return_labels=True
        )
    else:
        from scipy.sparse.csgraph import connected_components as cc

        n_components, labels = cc(S, directed=False, return_labels=True)
    return n_components, labels


def find_minimal_k(
    X,
    k_start=100,
    k_max=1000,
    metric="euclidean",
    decay=40,
    batch_size=256,
    verbose=1,
):
    """Scale-up search for a k that maintains graph connectivity.

    Starts with a sufficient assumption (k_start) and doubles until
    the graph is connected or k_max is reached. Much faster than
    binary search when the default k_start is already sufficient.

    Parameters
    ----------
    X : ndarray, shape=[n_samples, n_features]
        Input data matrix.
    k_start : int
        Initial k to try (should be generous enough for typical data).
    k_max : int
        Upper bound for k (capped at n_samples - 1).
    metric : str
        Distance/similarity metric.
    decay : float or None
        Alpha decay parameter for kernel.
    batch_size : int
        Batch size for sparse similarity computation.
    verbose : int
        Verbosity level.

    Returns
    -------
    k_opt : int
        k achieving connectivity (first value that works).
    S : scipy.sparse.csr_matrix
        Sparse similarity matrix at k_opt.
    n_components : int
        Number of connected components at k_opt (1 if connected).
    """
    n_samples = X.shape[0]
    k_max = min(k_max, n_samples - 1)
    k = min(k_start, k_max)

    _logger.log_info(
        f"Finding k for connectivity on {n_samples} samples "
        f"(start={k_start}, max={k_max})..."
    )

    n_comp = None
    while k <= k_max:
        _logger.log_info(f"Testing k={k}...")
        S = compute_sparse_similarity(
            X, k=k, metric=metric, decay=decay,
            batch_size=batch_size, verbose=max(verbose - 1, 0)
        )
        n_comp, labels = check_connectivity(S)
        if n_comp == 1:
            _logger.log_info(f"Connected at k={k} ✓")
            return k, S, 1
        _logger.log_info(f"  {n_comp} components, scaling up...")
        if k == k_max:
            break
        k = min(k * 2, k_max)

    _logger.log_info(
        f"Still disconnected at k_max={k_max} ({n_comp} components). "
        f"Data may have genuinely disconnected manifolds."
    )
    return k, S, n_comp
