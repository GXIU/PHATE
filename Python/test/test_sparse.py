"""Tests for sparse PHATE functionality.

Covers: sparse similarity, connectivity, minimal-k search,
sparse diffusion operator, sparse VNE, MDS is_pairwise flag,
and end-to-end sparse PHATE workflow.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phate
from phate import sparse_similarity
from scipy import sparse
from scipy.spatial.distance import pdist, squareform


def create_test_data(n=300, dim=50, seed=42):
    tree_data, tree_clusters = phate.tree.gen_dla(
        n_dim=dim, n_branch=max(1, n // 100), branch_length=100, seed=seed
    )
    return tree_data, tree_clusters


# ── Sparse similarity tests ──────────────────────────────────────────────


def test_compute_sparse_similarity_basic():
    """Sparse similarity produces CSR, correct shape, non-empty."""
    data, _ = create_test_data(n=200)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    assert sparse.issparse(S)
    assert S.shape == (200, 200)
    assert S.nnz > 0
    assert S.nnz <= 200 * 10 * 2  # at most ~2k entries per row after symmetrization


def test_compute_sparse_similarity_symmetry():
    """Symmetrized matrix is symmetric."""
    data, _ = create_test_data(n=100)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    # (S - S.T) should be near-zero (floating point tolerance)
    diff_mat = (S - S.T).toarray()
    assert np.abs(diff_mat).max() < 1e-10


def test_compute_sparse_similarity_kernel():
    """Kernel formula: affinity = exp(-distance / decay)."""
    data, _ = create_test_data(n=100)
    decay = 40
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="euclidean", decay=decay, batch_size=256, verbose=0
    )
    # All values should be in (0, 1] since exp(-d/decay) ∈ (0, 1]
    assert S.data.min() > 0
    assert S.data.max() <= 1.0


def test_compute_sparse_similarity_no_decay():
    """With decay=None, stores raw distances."""
    data, _ = create_test_data(n=100)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="euclidean", decay=None, batch_size=256, verbose=0
    )
    # Raw distances should be non-negative
    assert S.data.min() >= 0


def test_compute_sparse_similarity_cosine():
    """Cosine metric works and produces values in valid range."""
    data, _ = create_test_data(n=100)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="cosine", decay=40, batch_size=256, verbose=0
    )
    assert sparse.issparse(S)
    assert S.nnz > 0


def test_compute_sparse_similarity_k_equals_n():
    """k=n_samples: all entries are non-zero (or nearly all after symmetrization)."""
    data, _ = create_test_data(n=50)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=50, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    # With k=n, symmetrization gives full or nearly full matrix
    density = S.nnz / (50 * 50)
    assert density > 0.9


def test_compute_sparse_similarity_connectivity():
    """With sufficient k, the graph is connected."""
    data, _ = create_test_data(n=200)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=20, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    n_comp, _ = sparse_similarity.check_connectivity(S)
    assert n_comp == 1


# ── Diffusion operator tests ─────────────────────────────────────────────


def test_compute_sparse_diffusion_operator():
    """Diffusion operator rows sum to 1.0 (row-stochastic)."""
    data, _ = create_test_data(n=100)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    P = sparse_similarity.compute_sparse_diffusion_operator(S)
    row_sums = np.array(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ── Connectivity tests ───────────────────────────────────────────────────


def test_check_connectivity_connected():
    """Known connected graph returns 1 component."""
    data, _ = create_test_data(n=100)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=20, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    n_comp, labels = sparse_similarity.check_connectivity(S)
    assert n_comp == 1
    assert len(labels) == 100


def test_check_connectivity_disconnected():
    """Known disconnected graph returns >1 components."""
    # Create two separate clusters that are far apart
    rng = np.random.RandomState(42)
    cluster1 = rng.randn(50, 10)
    cluster2 = rng.randn(50, 10) + 100  # far away
    data = np.vstack([cluster1, cluster2])
    # k=1 gives very sparse graph, likely disconnected
    S = sparse_similarity.compute_sparse_similarity(
        data, k=1, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    n_comp, _ = sparse_similarity.check_connectivity(S)
    assert n_comp > 1


def test_find_minimal_k():
    """find_minimal_k returns connected graph when connectivity is achievable."""
    data, _ = create_test_data(n=200)
    k_opt, S, n_comp = sparse_similarity.find_minimal_k(
        data, k_start=10, k_max=100, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    assert k_opt >= 10
    assert n_comp == 1
    # Result must be connected
    n_comp_check, _ = sparse_similarity.check_connectivity(S)
    assert n_comp_check == 1


# ── Sparse VNE tests ─────────────────────────────────────────────────────


def test_sparse_vne_vs_dense():
    """On small N, sparse VNE approximates dense VNE."""
    from phate import vne

    data, _ = create_test_data(n=100)
    S = sparse_similarity.compute_sparse_similarity(
        data, k=10, metric="euclidean", decay=40, batch_size=256, verbose=0
    )
    P = sparse_similarity.compute_sparse_diffusion_operator(S)
    P_dense = P.toarray()

    # Compute VNE both ways
    h_sparse = vne.compute_von_neumann_entropy_sparse(P, t_max=10)
    h_dense = vne.compute_von_neumann_entropy(P_dense, t_max=10)

    # VNE curves should be highly correlated
    corr = np.corrcoef(h_sparse, h_dense)[0, 1]
    assert corr > 0.9


# ── MDS is_pairwise tests ────────────────────────────────────────────────


def test_embed_mds_is_pairwise():
    """MDS with is_pairwise=True skips distance computation and produces valid output."""
    from phate import mds

    data, _ = create_test_data(n=100)
    # Create a synthetic pairwise dissimilarity matrix
    D = squareform(pdist(data, "euclidean"))

    # Run MDS with is_pairwise=True (treat D as precomputed pairwise)
    emb_pairwise = mds.embed_MDS(
        D, ndim=2, how="metric", solver="sgd",
        is_pairwise=True, seed=42, verbose=0,
    )

    # Run MDS with is_pairwise=False (compute pdist on the rows of D)
    emb_normal = mds.embed_MDS(
        D, ndim=2, how="metric", solver="sgd",
        is_pairwise=False, seed=42, verbose=0,
    )

    # Both should produce valid embeddings of correct shape
    assert emb_pairwise.shape == (100, 2)
    assert emb_normal.shape == (100, 2)
    assert not np.any(np.isnan(emb_pairwise))
    assert not np.any(np.isinf(emb_pairwise))


def test_embed_mds_is_pairwise_sparse():
    """MDS with is_pairwise=True handles sparse input correctly."""
    from phate import mds

    data, _ = create_test_data(n=100)
    D = squareform(pdist(data, "euclidean"))
    D_sparse = sparse.csr_matrix(D)

    # Should not raise — sparse + is_pairwise path
    emb = mds.embed_MDS(
        D_sparse, ndim=2, how="metric", solver="sgd",
        is_pairwise=True, seed=42, verbose=0,
    )
    assert emb.shape == (100, 2)
    assert not np.any(np.isnan(emb))


# ── End-to-end sparse PHATE tests ────────────────────────────────────────


def test_phate_sparse_basic_workflow():
    """Sparse PHATE runs end-to-end and produces valid embedding."""
    data, _ = create_test_data(n=200)
    phate_op = phate.PHATE(
        knn=5,
        t=20,
        sparse_k=10,
        sparse_metric="euclidean",
        sparse_batch_size=256,
        verbose=False,
        random_state=42,
    )
    emb = phate_op.fit_transform(data)
    assert emb.shape == (200, 2)
    assert not np.any(np.isnan(emb))
    assert not np.any(np.isinf(emb))
    # Embedding should have non-zero variance
    assert np.std(emb) > 0


def test_phate_sparse_vs_dense_consistency():
    """Sparse and dense PHATE embeddings share meaningful neighbor structure.

    Despite using different MDS solvers (sparse uses randomized SVD for
    classic MDS, dense uses SGD metric MDS), both embeddings should
    preserve local neighborhood relationships.
    """
    data, _ = create_test_data(n=200)
    k_nn = 10

    emb_dense = phate.PHATE(
        knn=5, t=20, verbose=False, random_state=42,
    ).fit_transform(data)
    emb_sparse = phate.PHATE(
        knn=5, t=20, sparse_k=50, sparse_metric="euclidean",
        sparse_batch_size=256, verbose=False, random_state=42,
    ).fit_transform(data)

    assert emb_dense.shape == emb_sparse.shape == (200, 2)

    # k-NN in each embedding
    def knn_indices(emb, k):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(emb)
        return nn.kneighbors(emb, return_distance=False)

    nn_dense = knn_indices(emb_dense, k_nn)
    nn_sparse = knn_indices(emb_sparse, k_nn)

    # Jaccard overlap per point: |dense_nn ∩ sparse_nn| / (2k - |intersection|)
    overlaps = []
    for i in range(200):
        inter = len(set(nn_dense[i]) & set(nn_sparse[i]))
        overlaps.append(inter / (2 * k_nn - inter))
    mean_overlap = np.mean(overlaps)

    # Should share at least 20% of neighbors on average
    assert mean_overlap > 0.20, f"neighbor overlap={mean_overlap:.3f}"


def test_phate_sparse_torch_backend():
    """Sparse PHATE with torch backend produces valid embedding."""
    data, _ = create_test_data(n=200)
    emb = phate.PHATE(
        knn=5, t=20, sparse_k=10, sparse_metric="euclidean",
        sparse_backend="torch", sparse_device="cpu",
        sparse_batch_size=256, verbose=False, random_state=42,
    ).fit_transform(data)
    assert emb.shape == (200, 2)
    assert not np.any(np.isnan(emb))


def test_phate_sparse_cosine_metric():
    """Sparse PHATE with cosine metric produces valid embedding."""
    data, _ = create_test_data(n=200)
    emb = phate.PHATE(
        knn=5, t=20, sparse_k=10, sparse_metric="cosine",
        sparse_batch_size=256, verbose=False, random_state=42,
    ).fit_transform(data)
    assert emb.shape == (200, 2)
    assert not np.any(np.isnan(emb))


def test_phate_sparse_connectivity_warning():
    """PHATE warns when sparse_k is too small for connectivity."""
    data, _ = create_test_data(n=300)
    with pytest.warns(RuntimeWarning, match="disconnected"):
        phate.PHATE(
            knn=5, t=20, sparse_k=1, sparse_metric="euclidean",
            sparse_batch_size=256, verbose=False, random_state=42,
        ).fit(data)


def test_phate_sparse_different_metrics():
    """Sparse PHATE works with all supported metrics."""
    data, _ = create_test_data(n=100)
    for metric in ["euclidean", "cosine"]:
        emb = phate.PHATE(
            knn=5, t=20, sparse_k=30, sparse_metric=metric,
            sparse_batch_size=256, verbose=False, random_state=42,
        ).fit_transform(data)
        assert emb.shape == (100, 2)


def test_phate_sparse_with_gamma():
    """Sparse PHATE works with different gamma values."""
    data, _ = create_test_data(n=100)
    for gamma in [-1, 0, 1]:
        emb = phate.PHATE(
            knn=5, t=20, sparse_k=30, gamma=gamma,
            sparse_metric="euclidean", sparse_batch_size=256,
            verbose=False, random_state=42,
        ).fit_transform(data)
        assert emb.shape == (100, 2)
        assert not np.any(np.isnan(emb))
