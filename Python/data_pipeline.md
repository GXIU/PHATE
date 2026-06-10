# Sparse PHATE — Data Pipeline & Benchmarks

## Architecture

```
┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐
│  Input   │    │   Sparse      │    │  Diffusion   │    │    MDS      │    │  Output  │
│  data    │───→│   Similarity  │───→│  Potential   │───→│  (rand SVD) │───→│  embed   │
│  N × d   │    │  (CSR, ~N·k)  │    │  (CSR, P^t)  │    │  (CSR→N×2)  │    │  N × 2   │
└──────────┘    └───────────────┘    └──────────────┘    └────────────┘    └──────────┘
                     O(N·d·k)           O(N·k²·t)           O(N·k·ndim)

  No dense N×N matrix at any step. All intermediates stay sparse CSR.
```

**Key design decisions:**
- Batched top-k similarity via `torch.cdist` — 50× faster than `scipy.cdist`
- Diffusion operator stays sparse CSR throughout — stepwise `P @ P` with no densification threshold
- MDS via `sklearn.randomized_svd` on sparse CSR — creates only N×ndim intermediates, never N×N
- All dependencies are core (no optional extras): torch, rich, anndata, pygsp, pandas, pyarrow

---

## 1. Generate Tree Data → Parquet

```bash
uv run python -c "
import pandas as pd, numpy as np
from phate.tree import gen_dla

data, clusters = gen_dla(
    n_dim=512, n_branch=100, branch_length=500, seed=42
)
# (50000, 512) — 100 branches × 500 points each, ~5s generation

df = pd.DataFrame(data.astype(np.float32))
df.insert(0, 'branch', clusters.astype(np.int32))
df.to_parquet('tree_50k.parquet', index=False)
"
```

| Output | Rows | Columns | Size |
|--------|------|---------|------|
| `tree_50k.parquet` | 50,000 | 513 (branch + 512 features) | 153 MB |
| `tree_500k.parquet` | 500,000 | 513 | 1.3 GB |

`gen_dla` uses pre-allocation — O(N·d) time, scales to N=500K in ~5s.

---

## 2. Configure via YAML

`pipeline_50k.yml`:

```yaml
input: tree_50k.parquet
output: embedding_50k.parquet
sparse_k: 100
sparse_metric: euclidean
decay: 40.0
batch_size: 256
sparse_device: cpu    # or cuda, mps
t: 2
gamma: 1.0
n_components: 2
plot_dir: plots
plot_format: png       # or html (interactive plotly)
verbose: true
```

## 3. Run CLI

```bash
uv run python cli.py -c pipeline_50k.yml

# Override any field from CLI
uv run python cli.py -c pipeline_50k.yml -t 8 --sparse-k 50 --device cuda

# Print resolved config
uv run python cli.py -c pipeline_50k.yml --show-config
```

---

## 4. Benchmarks

### Scale (d=100, k=10, t=2)

| N | Similarity | P² | MDS | **Total** | S nnz | P^t nnz | P^t density |
|---|-----------|-----|-----|-------|--------|----------|-------------|
| 10K | 0.1s | 0.04s | 0.3s | **0.5s** | 172K | 7.5M | 7.5% |
| 20K | 0.3s | 0.10s | 0.6s | **1.1s** | 346K | 16.9M | 4.2% |
| 50K | 2.2s | 0.31s | 1.6s | **4.4s** | 869K | 44.4M | 1.8% |
| 100K | 8.2s | 0.85s | 4.0s | **13.8s** | 1.7M | 104M | 1.0% |
| 200K | 34.2s | 2.40s | 19.3s | **57.6s** | 3.5M | 234M | 0.6% |

### Dense (original, knn=5) vs Sparse (k=30) — N=2000, d=100

| t | Dense | Sparse | Speedup |
|---|-------|--------|---------|
| 1 | 3.3s | 0.1s | 33× |
| 2 | 3.2s | 0.1s | 32× |
| 4 | 4.8s | 0.4s | 12× |
| 8 | 18.4s | 2.1s | 9× |

Plot: `plots/dense_vs_sparse.png` — side-by-side comparison colored by branch.

### t=1..10 × k=[10, 50, 100] — N=50,000, d=512

| k\t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|------|----|
| 10 | 9.7s | 9.7s | 10.0s | 10.2s | 13.1s | 11.3s | 11.3s | 11.7s | 11.9s | 12.2s |
| 50 | 10.9s | 11.3s | 11.7s | 12.7s | 15.6s | 16.5s | 19.0s | 21.4s | 27.6s | 30.9s |
| 100 | 17.5s | 17.6s | 24.7s | 25.8s | 32.9s | 55.3s | 106.9s | 129.1s | 60.2s | 77.3s |

Plot: `plots/t_k_grid.png` — 3×10 grid, each panel is a PHATE embedding.

**Observations:**
- k=10: minimal fill-in, t has little effect on runtime (density stays low)
- k=50: moderate fill-in, runtime grows 3× from t=1 to t=10
- k=100: heavy fill-in, P^t approaches dense matrix — runtime spikes at t=7-8
- Recommendation: for tree data, k=50 at t≤4 gives best speed/quality tradeoff

---

## 5. Scripts

```bash
# Grid search: t=1..10 × k=[10, 50, 100]
uv run python scripts/t_k_grid.py

# Dense vs sparse comparison
uv run python scripts/dense_vs_sparse.py
```

---

## 6. Build & Install

```bash
# Build wheel
uv build --wheel
# → dist/phate-2.0.0-py3-none-any.whl

# Install
uv pip install dist/phate-2.0.0-py3-none-any.whl
```

---

## 7. GPU Acceleration

Set `sparse_device: cuda` in YAML (or `--device cuda` on CLI). The similarity computation (torch.cdist) is the dominant cost at large N and benefits most from GPU.

```yaml
sparse_device: cuda
batch_size: 1024   # larger batches on GPU
```
