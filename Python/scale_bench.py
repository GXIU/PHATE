#!/usr/bin/env python
"""Step-by-step sparse PHATE scale test at N=10k, 20k, 50k, 100k, 200k."""
import sys, os, time, gc
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from phate import sparse_similarity, mds

def run_scale(N, d=100, k=10, t=2):
    print(f"\n{'='*60}")
    print(f"N={N}, d={d}, k={k}, t={t}")
    print(f"{'='*60}")

    # Generate data
    t0 = time.time()
    rng = np.random.RandomState(42)
    data = rng.randn(N, d).astype(np.float32)
    print(f"  data: {data.nbytes/1e6:.0f} MB, {time.time()-t0:.1f}s")

    # Step 1: Sparse similarity
    t0 = time.time()
    S = sparse_similarity.compute_sparse_similarity_torch(
        data, k=k, metric="euclidean", decay=40, batch_size=256, device="cpu", verbose=0
    )
    t1 = time.time()
    mem_mb = (S.data.nbytes + S.indptr.nbytes + S.indices.nbytes) / 1e6
    density = 100 * S.nnz / (N * N)
    print(f"  sim:  {t1-t0:.1f}s, {S.nnz} nnz ({density:.3f}% dense), {mem_mb:.0f} MB")

    # Step 2: Diffusion operator
    t0 = time.time()
    P = sparse_similarity.compute_sparse_diffusion_operator(S)
    print(f"  diffop: {time.time()-t0:.2f}s")

    # Step 3: Matrix power P^t (stepwise sparse)
    t0 = time.time()
    P_pow = P
    for step in range(t - 1):
        P_pow = P_pow @ P
    t1 = time.time()
    nnz_pow = P_pow.nnz
    density_pow = 100 * nnz_pow / (N * N)
    print(f"  P^{t}:  {t1-t0:.2f}s, {nnz_pow} nnz ({density_pow:.3f}% dense)")

    # Step 4: Log potential
    t0 = time.time()
    diff_potential = P_pow.copy()
    diff_potential.data = np.log1p(diff_potential.data)
    diff_potential.data = -diff_potential.data
    diff_potential.eliminate_zeros()
    print(f"  log-pot: {time.time()-t0:.2f}s")

    # Step 5: MDS (randomized SVD on sparse)
    t0 = time.time()
    emb = mds.embed_MDS(
        diff_potential, ndim=2, how="classic", solver="sgd",
        is_pairwise=True, _is_sparse_mds=True, seed=42,
    )
    t1 = time.time()
    print(f"  MDS: {t1-t0:.1f}s, emb={emb.shape}")

    # Cleanup
    del data, S, P, P_pow, diff_potential
    gc.collect()

    return emb

if __name__ == "__main__":
    for N in [10000, 20000, 50000]:
        emb = run_scale(N, d=100, k=10, t=2)

    # 100K — try if memory allows
    try:
        emb = run_scale(100_000, d=100, k=10, t=2)
    except Exception as e:
        print(f"  N=100000 FAILED: {e}")

    # 200K
    try:
        emb = run_scale(200_000, d=100, k=10, t=2)
    except Exception as e:
        print(f"  N=200000 FAILED: {e}")

    print(f"\n{'='*60}")
    print("Done.")
