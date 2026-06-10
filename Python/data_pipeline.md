# Sparse PHATE Data Pipeline

## 1. Generate Tree Data → Parquet

```bash
uv run python -c "
import pandas as pd, numpy as np
from phate.tree import gen_dla

data, clusters = gen_dla(
    n_dim=512, n_branch=100, branch_length=500, seed=42
)
# data: (50000, 512) float64 — 100 branches × 500 points each

df = pd.DataFrame(data.astype(np.float32))
df.insert(0, 'branch', clusters.astype(np.int32))
df.to_parquet('tree_50k.parquet', index=False)
"
```

Output: `tree_50k.parquet` — 153 MB, 50,000 rows × 513 columns (`branch` + 512 features).

## 2. YAML Configuration

`pipeline_50k.yml`:

```yaml
input: tree_50k.parquet
output: embedding_50k.parquet
sparse_k: 50
sparse_metric: euclidean
decay: 40.0
batch_size: 256
sparse_device: cpu
t: 2
gamma: 1.0
n_components: 2
plot_dir: plots
plot_format: png
verbose: true
```

## 3. Run Sparse PHATE

```bash
uv run python cli.py -c pipeline_50k.yml
```

Output:
```
Loaded tree_50k.parquet: (50000, 513), float32, 0.1s
PHATE: k=50, t=2, gamma=1.0, metric=euclidean, ndim=2, device=cpu
  Sparse similarity (torch, cpu) ─── 100%
    Result: 2,756,490 non-zeros (0.110% dense)
  Calculated sparse similarity and diffusion operator in 5.13 seconds.
  Calculated diffusion potential in 0.19 seconds.
  Calculated metric MDS in 0.22 seconds.
Done: 6.9s, embedding=(50000, 2)
Saved: embedding_50k.parquet
Plot: plots/phate_embedding.png
```

**No dense N×N matrix at any step.** All intermediate matrices stay sparse CSR.

## 4. Output

**`embedding_50k.parquet`** — 50,000 rows × 2 columns (`x_0`, `x_1`):

| x_0 | x_1 |
|-----|-----|
| 0.65 | 4.61 |
| 0.70 | 5.23 |
| ... | ... |

Embedding range: x_0 ∈ [-6.5, 96.9], x_1 ∈ [-6.5, 95.2].

**Plot:** `plots/phate_embedding.png` — scatter colored by branch label.

## Pipeline Summary

```
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│ tree_50k     │ ──→ │ sparse PHATE     │ ──→ │ embedding  │
│ .parquet     │     │ (all sparse CSR) │     │ .parquet   │
│ 50K × 512    │     │                  │     │ 50K × 2    │
│ 153 MB       │     │ sim:   5.1s      │     │ + plot.png │
│              │     │ diff:  0.2s      │     │            │
│              │     │ MDS:   0.2s      │     │            │
│              │     │ total: 5.6s      │     │            │
└──────────────┘     └──────────────────┘     └────────────┘
```

### Scaling to N=500,000

```bash
uv run python -c "
from phate.tree import gen_dla
import pandas as pd

data, clusters = gen_dla(
    n_dim=512, n_branch=500, branch_length=1000, seed=42
)
df = pd.DataFrame(data.astype(np.float32))
df.insert(0, 'branch', clusters.astype(np.int32))
df.to_parquet('tree_500k.parquet', index=False)
"
```

Generates `tree_500k.parquet` (1.3 GB, 500,000 × 512) in ~5 seconds.

For GPU acceleration, set `sparse_device: cuda` in the YAML:
```yaml
sparse_device: cuda
batch_size: 1024
```
