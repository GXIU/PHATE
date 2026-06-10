# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

"""SGD-MDS with optional sparse on-the-fly distance computation."""

import numpy as np
import tasklogger

_logger = tasklogger.get_tasklogger("graphtools")


def _sparse_pairwise_distances(X, i_array, j_array, row_norms_sq, batch_size=5000):
    """Compute Euclidean distances between rows of sparse matrix X.

    Uses dist²(i,j) = ||row_i||² + ||row_j||² - 2·row_i·row_j
    with batched sparse-dense dot products. All heavy lifting in scipy C code.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix, shape (N, N)
    i_array, j_array : ndarray of int, shape (P,)
    row_norms_sq : ndarray, shape (N,) — precomputed ||row||²
    batch_size : int — rows per sparse slice

    Returns
    -------
    dists : ndarray, shape (P,) — Euclidean distances
    """
    P = len(i_array)
    dists = np.empty(P, dtype=np.float32)

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        ib = i_array[start:end]
        jb = j_array[start:end]

        # dot(row_i, row_j) = (X[ib] @ X.T)[p, jb[p]] — compute via
        # sparse * sparse transpose, then gather the diagonal elements.
        Xi = X[ib]       # sparse (B, N)
        Xj = X[jb]       # sparse (B, N)

        # Element-wise multiply + row sum gives the dot products
        # dot[p] = sum_c Xi[p,c] * Xj[p,c]
        dot_prod = np.array(Xi.multiply(Xj).sum(axis=1)).ravel()

        d2 = row_norms_sq[ib] + row_norms_sq[jb] - 2 * dot_prod
        np.maximum(d2, 0, out=d2)
        dists[start:end] = np.sqrt(d2)

    return dists


def sgd_mds(
    D,
    n_components=2,
    learning_rate=0.001,
    n_iter=500,
    init=None,
    random_state=None,
    verbose=0,
    pairs_per_iter=None,
    sparse=False,
    sparse_X=None,
):
    """SGD-MDS with optional sparse on-the-fly distance computation.

    Parameters
    ----------
    D : ndarray (N,N) or None (when sparse_X is set)
    sparse_X : scipy.sparse.csr_matrix, optional
        Sparse N×N potential matrix. When set, pairwise distances are
        computed on-the-fly from sparse rows — no dense N×N matrix needed.
    sparse : bool
        If True and sparse_X is None, sample from non-zero entries of D.
    """
    rng = (np.random.RandomState(random_state) if isinstance(random_state, int)
           else random_state or np.random.RandomState())

    n_samples = D.shape[0] if D is not None else sparse_X.shape[0]

    # ── On-the-fly sparse mode ──
    if sparse_X is not None:
        # Precompute row norms squared: ||row_i||²
        X_sq = sparse_X.copy()
        X_sq.data **= 2
        row_norms_sq = np.array(X_sq.sum(axis=1)).ravel().astype(np.float32)
        D_max = float(np.max(sparse_X.data)) if sparse_X.nnz > 0 else 1.0
        _otf = True
        _sp_nnz = False
    elif sparse and hasattr(D, 'nnz'):
        _otf = False
        _sp_nnz = True
        _nnz_rows, _nnz_cols = D.nonzero()
        _nnz_data = D.data
        _n_edges = len(_nnz_data)
        D_max = np.max(_nnz_data) if _n_edges > 0 else 1.0
        if D_max > 0:
            D_norm = D / D_max
        else:
            D_norm = D / 1.0
    else:
        _otf = False
        _sp_nnz = False
        D_max = np.max(D)
        if D_max > 0:
            D_norm = D / D_max
        else:
            D_norm = D.copy()

    # Initialize embedding
    if init is None:
        Y = rng.randn(n_samples, n_components).astype(np.float32) * 0.01
    else:
        Y = np.asarray(init, dtype=np.float32).copy()
        s = np.std(Y)
        if s > 0:
            Y /= s

    if pairs_per_iter is None:
        pairs_per_iter = int(n_samples * np.log(n_samples))

    total_pairs = n_samples * (n_samples - 1) / 2
    sampling_ratio = pairs_per_iter / total_pairs
    batch_scale = np.sqrt(1.0 / sampling_ratio)
    eta_max = learning_rate * batch_scale
    eta_min = learning_rate * 0.01 * batch_scale
    lambd = np.log(eta_max / eta_min) / max(n_iter - 1, 1)

    if verbose > 0:
        mode = "sparse-otf" if _otf else ("sparse-nnz" if _sp_nnz else "dense")
        _logger.log_debug(
            f"SGD-MDS [{mode}]: n={n_samples}, pairs/iter={pairs_per_iter}")

    prev_stress = None
    stress_history = []
    inv_D_max = 1.0 / D_max if D_max > 0 else 1.0

    for iteration in range(n_iter):
        lr = eta_max * np.exp(-lambd * iteration)

        # ── Sample pairs ──
        if _otf:
            i_s = rng.randint(0, n_samples, pairs_per_iter)
            j_s = rng.randint(0, n_samples, pairs_per_iter)
            v = i_s != j_s
            i_s, j_s = i_s[v], j_s[v]
            if len(i_s) == 0:
                continue
            # Vectorized sparse distance computation (scipy C code)
            target_dists = _sparse_pairwise_distances(
                sparse_X, i_s, j_s, row_norms_sq) * inv_D_max

        elif _sp_nnz:
            e = rng.randint(0, _n_edges, pairs_per_iter)
            i_s = _nnz_rows[e]
            j_s = _nnz_cols[e]
            target_dists = _nnz_data[e] * inv_D_max
            v = i_s != j_s
            i_s, j_s, target_dists = i_s[v], j_s[v], target_dists[v]
        else:
            i_s = rng.randint(0, n_samples, pairs_per_iter)
            j_s = rng.randint(0, n_samples, pairs_per_iter)
            v = i_s != j_s
            i_s, j_s = i_s[v], j_s[v]
            if len(i_s) == 0:
                continue
            target_dists = D_norm[i_s, j_s]

        if len(i_s) == 0:
            continue

        # Compute current embedding distances
        diff = Y[i_s] - Y[j_s]
        emb_dists = np.linalg.norm(diff, axis=1)
        emb_dists = np.maximum(emb_dists, 1e-10)

        # Gradient: ∇stress = -2(d_target - d_emb) * (y_i - y_j)/d_emb
        errors = target_dists - emb_dists
        weights = -2.0 * errors / emb_dists
        grad_contrib = diff * weights[:, np.newaxis]

        gradients = np.zeros_like(Y)
        np.add.at(gradients, i_s, grad_contrib)
        np.add.at(gradients, j_s, -grad_contrib)
        Y -= lr * gradients

        stress = np.mean(errors ** 2)
        stress_history.append(stress)

        if verbose > 0 and iteration % 100 == 0:
            _logger.log_debug(
                f"  iter {iteration}: stress={stress:.6f} lr={lr:.6f} "
                f"|Y|={np.mean(np.abs(Y)):.4f}")

        if iteration > 0:
            rel = abs(stress - prev_stress) / (prev_stress + 1e-10)
            if rel < 1e-6 and iteration > 50:
                if verbose > 0:
                    _logger.log_info(f"Converged at iter {iteration}")
                break
        prev_stress = stress

    if len(stress_history) > 10:
        recent = stress_history[-max(1, len(stress_history) // 10):]
        trend = (recent[-1] - recent[0]) / (recent[0] + 1e-10)
        if abs(trend) > 0.01:
            _logger.log_warning(
                f"SGD-MDS may not have converged: stress trend {trend*100:.1f}%")

    if D_max > 0:
        Y *= D_max

    return Y


def sgd_mds_metric(
    D,
    n_components=2,
    init=None,
    random_state=None,
    verbose=0,
    sparse=False,
    sparse_X=None,
):
    """Auto-tuned SGD-MDS."""
    n = D.shape[0] if D is not None else sparse_X.shape[0]

    if n < 1000:
        n_iter, ppi = 300, n * n // 10
    elif n < 5000:
        n_iter, ppi = 500, int(n * np.log(n) * 2)
    else:
        n_iter, ppi = 750, int(n * np.log(n) * 2)

    if sparse and hasattr(D, 'nnz'):
        ppi = min(ppi, D.nnz)

    return sgd_mds(
        D=D, n_components=n_components, learning_rate=0.001,
        n_iter=n_iter, init=init, random_state=random_state,
        verbose=verbose, pairs_per_iter=ppi,
        sparse=sparse, sparse_X=sparse_X,
    )
