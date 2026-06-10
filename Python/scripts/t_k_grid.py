#!/usr/bin/env python
"""Grid search: t=1..10 × k=[10, 50, 100] — PHATE embeddings on tree data.

Usage:
    uv run python scripts/t_k_grid.py
"""

import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import phate
from matplotlib import pyplot as plt

INPUT = "./tree_50k.parquet"
OUTDIR = "./plots"
T_VALUES = list(range(1, 11))
K_VALUES = [10, 50, 100]

os.makedirs(OUTDIR, exist_ok=True)

# Load
df = pd.read_parquet(INPUT)
clusters = df["branch"].values
data = df.drop(columns=["branch"]).values.astype(np.float64)
print(f"Data: {data.shape}, branches: {len(np.unique(clusters))}")

results = {}

for k in K_VALUES:
    for t in T_VALUES:
        print(f"k={k:3d}, t={t:2d}...", end=" ", flush=True)
        t0 = time.time()
        emb = phate.PHATE(
            sparse_k=k, t=t, n_components=2, verbose=False,
        ).fit_transform(data)
        elapsed = time.time() - t0
        results[(k, t)] = emb
        print(f"{elapsed:.1f}s")

# Plot grid
fig, axes = plt.subplots(
    len(K_VALUES), len(T_VALUES),
    figsize=(len(T_VALUES) * 2.2, len(K_VALUES) * 2.2),
)

for row, k in enumerate(K_VALUES):
    for col, t in enumerate(T_VALUES):
        ax = axes[row, col]
        emb = results[(k, t)]
        ax.scatter(emb[:, 0], emb[:, 1], c=clusters, cmap="Spectral",
                   s=0.3, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"t={t}", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"k={k}", fontsize=9, rotation=0, labelpad=20)

plt.suptitle("Sparse PHATE: t=1..10 × k=[10, 50, 100]", fontsize=12, y=1.01)
plt.tight_layout()
path = os.path.join(OUTDIR, "t_k_grid.png")
plt.savefig(path, dpi=120, bbox_inches="tight")
print(f"Saved: {path}")
