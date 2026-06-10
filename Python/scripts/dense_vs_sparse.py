#!/usr/bin/env python
"""Dense vs Sparse PHATE comparison on small-N tree data, t=1,2,4,8.

Usage:
    uv run python scripts/dense_vs_sparse.py
"""

import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import phate
from matplotlib import pyplot as plt

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

# Small tree for quick dense runs
data, clusters = phate.tree.gen_dla(
    n_dim=100, n_branch=20, branch_length=100, seed=42
)
print(f"Data: {data.shape}, branches: {len(np.unique(clusters))}")

t_values = [1, 2, 4, 8]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))

for col, t in enumerate(t_values):
    # Dense path
    print(f"Dense  t={t}...", end=" ", flush=True)
    t0 = time.time()
    emb_d = phate.PHATE(knn=5, t=t, n_components=2, verbose=False, random_state=42
    ).fit_transform(data)
    td = time.time() - t0

    # Sparse path
    print(f"Sparse t={t}...", end=" ", flush=True)
    t0 = time.time()
    emb_s = phate.PHATE(sparse_k=30, t=t, n_components=2, verbose=False
    ).fit_transform(data)
    ts = time.time() - t0

    print(f"dense={td:.1f}s sparse={ts:.1f}s")

    axes[0, col].scatter(emb_d[:, 0], emb_d[:, 1], c=clusters,
                         cmap="Spectral", s=2, rasterized=True)
    axes[0, col].set_title(f"Dense t={t} ({td:.1f}s)")
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

    axes[1, col].scatter(emb_s[:, 0], emb_s[:, 1], c=clusters,
                         cmap="Spectral", s=2, rasterized=True)
    axes[1, col].set_title(f"Sparse t={t} ({ts:.1f}s)")
    axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

for ax_row, label in zip(axes, ["Dense (kNN=5)", "Sparse (k=30)"]):
    ax_row[0].set_ylabel(label, fontsize=11)

plt.suptitle("Dense vs Sparse PHATE — tree data (N=2000, d=100)", fontsize=12, y=1.01)
plt.tight_layout()
path = os.path.join(OUTDIR, "dense_vs_sparse.png")
plt.savefig(path, dpi=120, bbox_inches="tight")
print(f"Saved: {path}")
