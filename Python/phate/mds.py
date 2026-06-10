# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

import numpy as np
import scipy.spatial
import tasklogger
from deprecated import deprecated
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from sklearn.decomposition import PCA

from . import sgd_mds as sgd_mds_module

_logger = tasklogger.get_tasklogger("graphtools")


# Fast classical MDS using random svd
@deprecated(version="1.0.0", reason="Use phate.mds.classic instead")
def cmdscale_fast(D, ndim):
    return classic(D=D, n_components=ndim)


def classic(D, n_components=2, random_state=None):
    """Fast CMDS using random SVD

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    _logger.log_debug(
        f"Performing classic MDS on {type(D).__name__} of shape {D.shape}..."
    )
    D = D**2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    )
    Y = pca.fit_transform(D)
    return Y


@deprecated(version="1.0.0", reason="Use phate.mds.smacof instead")
def sgd(D, n_components=2, random_state=None, init=None):
    """Metric MDS using stochastic gradient descent

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int or None, optional (default: None)
        numpy random state

    init : array-like or None
        Initialization algorithm or state to use for MMDS

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    return smacof(
        D=D,
        n_components=n_components,
        random_state=random_state,
        init=init,
        metric=True,
    )


def smacof(
    D,
    n_components=2,
    metric=True,
    init=None,
    random_state=None,
    verbose=0,
    max_iter=3000,
    eps=1e-6,
    n_jobs=1,
):
    """Metric and non-metric MDS using SMACOF

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    metric : bool, optional (default: True)
        Use metric MDS. If False, uses non-metric MDS

    init : array-like or None, optional (default: None)
        Initialization state

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    verbose : int or bool, optional (default: 0)
        verbosity

    max_iter : int, optional (default: 3000)
        maximum iterations

    eps : float, optional (default: 1e-6)
        stopping criterion

    Returns
    -------
    Y : array-like, shape=[n_samples, n_components]
        embedded data
    """
    _logger.log_debug(
        "Performing non-metric MDS on " f"{type(D)} of shape {D.shape}..."
    )
    # Metric MDS from sklearn
    Y, _ = manifold.smacof(
        D,
        n_components=n_components,
        metric=metric,
        max_iter=max_iter,
        eps=eps,
        random_state=random_state,
        n_jobs=n_jobs,
        n_init=1,
        init=init,
        verbose=verbose,
    )
    return Y


def _sparse_row_pdist(X):
    """Compute pairwise Euclidean distances between rows of a sparse matrix.

    Uses the identity  dist²(i,j) = ||row_i||² + ||row_j||² - 2·row_i·row_j
    to avoid materializing a dense copy of X.  The result is a dense N×N
    float32 array.
    """
    # Row norms squared (keep sparse)
    X_sq = X.copy()
    X_sq.data **= 2
    row_norms_sq = np.array(X_sq.sum(axis=1)).ravel().astype(np.float32)

    # Dot products between rows: X @ X.T
    # Use float32 for the dense result to halve memory
    dot = (X @ X.T).toarray().astype(np.float32)

    # D[i,j] = sqrt(max(0, ||row_i||² + ||row_j||² - 2·row_i·row_j))
    D_sq = row_norms_sq[:, np.newaxis] + row_norms_sq[np.newaxis, :]
    D_sq -= 2 * dot
    np.maximum(D_sq, 0, out=D_sq)
    return np.sqrt(D_sq, dtype=np.float32)


def embed_MDS(
    X,
    ndim=2,
    how="metric",
    distance_metric="euclidean",
    solver="sgd",
    n_jobs=1,
    seed=None,
    verbose=0,
    is_pairwise=False,
    _is_sparse_mds=False,
):
    """Performs classic, metric, and non-metric MDS

    Metric MDS is initialized using classic MDS,
    non-metric MDS is initialized using metric MDS.

    Parameters
    ----------
    X: ndarray [n_samples, n_features]
        2 dimensional input data array with n_samples

    n_dim : int, optional, default: 2
        number of dimensions in which the data will be embedded

    how : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    distance_metric : string, optional, default: 'euclidean'
        choose from ['cosine', 'euclidean']
        distance metric for MDS

    solver : {'sgd', 'smacof'}, optional (default: 'sgd')
        which solver to use for metric MDS. SGD is 5-10x faster than SMACOF
        while producing nearly identical results (correlation > 0.99).
        Note that SMACOF was used for all figures in the original PHATE paper.

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for log_debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    seed: integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    is_pairwise : bool, optional, default: False
        If True, X is treated as a precomputed pairwise dissimilarity matrix
        (shape [n_samples, n_samples]). Distance computation is skipped.

    Returns
    -------
    Y : ndarray [n_samples, n_dim]
        low dimensional embedding of X using MDS
    """

    if how not in ["classic", "metric", "nonmetric"]:
        raise ValueError(
            "Allowable 'how' values for MDS: 'classic', "
            "'metric', or 'nonmetric'. "
            f"'{how}' was passed."
        )
    if solver not in ["sgd", "smacof"]:
        raise ValueError(
            "Allowable 'solver' values for MDS: 'sgd' or "
            "'smacof'. "
            f"'{solver}' was passed."
        )

    # ── Sparse input: randomized SVD directly on the sparse matrix ──
    # Classical MDS on Euclidean distances = PCA on the data rows.
    # Randomized SVD on sparse X creates only N×k intermediates,
    # never a dense N×N matrix.  O(N·nnz·k) time, O(N·k + nnz) memory.
    _X_is_sparse = sparse.issparse(X)
    if _X_is_sparse and (how == "classic" or _is_sparse_mds):
        from sklearn.utils.extmath import randomized_svd

        U, S, Vt = randomized_svd(
            X, n_components=ndim, random_state=seed, n_iter=4)
        return U[:, :ndim] * S[:ndim]

    if _X_is_sparse and not is_pairwise:
        X_dist = _sparse_row_pdist(X)
    elif is_pairwise:
        X_dist = np.asarray(X.toarray() if _X_is_sparse else X)
    elif distance_metric == "euclidean" and X.shape[0] > 1000:
        from sklearn.metrics.pairwise import euclidean_distances

        X_dist = euclidean_distances(X, X)
    else:
        X_dist = squareform(pdist(X, distance_metric))

    # Check for degenerate distance matrix
    if sparse.issparse(X_dist):
        _dist_check = X_dist.data
    else:
        _dist_check = X_dist
    if _dist_check.std() < 1e-10 or len(np.unique(_dist_check)) <= 1:
        import warnings
        warnings.warn(
            f"Degenerate distance matrix detected (std={_dist_check.std():.2e}, "
            f"unique_values={len(np.unique(_dist_check))}). "
            "This typically occurs when hyperparameters cause complete diffusion homogeneity "
            "(e.g., KNN close to dataset size). "
            "Returning zero embedding.",
            RuntimeWarning
        )
        return np.zeros((X_dist.shape[0], ndim))

    # Classic MDS requires dense
    X_dense = X_dist.toarray() if sparse.issparse(X_dist) else X_dist
    Y_classic = classic(X_dense, n_components=ndim, random_state=seed)
    if how == "classic":
        return Y_classic

    # metric MDS using SGD or SMACOF
    if solver == "sgd":
        _is_sparse = sparse.issparse(X_dist)
        Y = sgd_mds_module.sgd_mds_metric(
            X_dist,
            n_components=ndim,
            random_state=seed,
            init=Y_classic,
            verbose=verbose,
            sparse=_is_sparse,
        )
    elif solver == "smacof":
        Y = smacof(
            X_dense, n_components=ndim, random_state=seed, init=Y_classic, metric=True
        )
    else:
        raise RuntimeError

    if how == "metric":
        # re-orient to classic
        _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
        return Y

    # nonmetric is slowest
    Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y, metric=False)
    # re-orient to classic
    _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
    return Y
