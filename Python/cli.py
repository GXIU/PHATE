#!/usr/bin/env python
"""Run sparse PHATE — config-driven with YAML support.

Usage:
    uv run python cli.py -c config.yml
    uv run python cli.py -c config.yml --sparse-k 20 --t 4   # CLI overrides
    uv run python cli.py --show-config -c config.yml          # print resolved config

Config file (YAML):
    input: data.parquet
    output: embedding.parquet
    sparse_k: 10
    t: 2
    gamma: 1.0
    decay: 40.0
    sparse_metric: euclidean
    n_components: 2
    sparse_device: cpu
    batch_size: 256
    verbose: true
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_DEFAULTS = {
    "input": None,
    "output": None,
    "sparse_k": 10,
    "t": 2,
    "gamma": 1.0,
    "decay": 40.0,
    "sparse_metric": "euclidean",
    "n_components": 2,
    "sparse_device": "cpu",
    "batch_size": 256,
    "no_header": False,
    "verbose": True,
    "plot_dir": None,
    "plot_format": "png",
}

CONFIG_TYPES = {
    "sparse_k": int,
    "t": int,
    "gamma": float,
    "decay": float,
    "n_components": int,
    "batch_size": int,
    "no_header": bool,
    "verbose": bool,
}

CONFIG_CHOICES = {
    "sparse_metric": ["euclidean", "cosine", "correlation"],
    "sparse_device": ["cpu", "cuda", "mps"],
    "plot_format": ["png", "html"],
}


def load_config(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def resolve_config(cli_args):
    """YAML baseline → defaults → CLI overrides."""
    config = dict(CONFIG_DEFAULTS)

    if cli_args.config:
        yaml_cfg = load_config(cli_args.config)
        config.update(yaml_cfg)

    for key in CONFIG_DEFAULTS:
        val = getattr(cli_args, key, None)
        if val is not None:
            config[key] = val

    for key, typ in CONFIG_TYPES.items():
        if key in config and config[key] is not None:
            config[key] = typ(config[key])

    return config


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════


def load_data(path, no_header=False):
    ext = os.path.splitext(path)[1].lower()
    labels = None

    if ext == ".npy":
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
        data, labels = _df_to_array(df, no_header)
    elif ext in (".csv", ".tsv", ".txt", ".gz"):
        sep = "\t" if ext in (".tsv", ".txt") else ","
        header = "infer" if not no_header else None
        df = pd.read_csv(path, sep=sep, header=header)
        data, labels = _df_to_array(df, no_header)
    elif ext in (".h5ad", ".h5"):
        import anndata
        adata = anndata.read_h5ad(path)
        data = adata.X
        if hasattr(data, "toarray"):
            data = data.toarray()
        labels = getattr(adata, "obs_names", None)
    else:
        supported = ".parquet, .csv, .tsv, .npy, .h5ad"
        raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}")

    data = np.asarray(data, dtype=np.float32, order="C")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data, labels


def _df_to_array(df, no_header):
    labels = None
    if not no_header:
        first_col = df.columns[0]
        try:
            df[first_col].astype(float)
        except (ValueError, TypeError):
            labels = df[first_col].values
            df = df.drop(columns=[first_col])
    data = df.select_dtypes(include=[np.number]).values
    return data, labels


def save_parquet(path, embedding, labels=None):
    cols = [f"x_{i}" for i in range(embedding.shape[1])]
    df = pd.DataFrame(embedding, columns=cols)
    if labels is not None:
        df.insert(0, "label", labels)
    df.to_parquet(path, index=False)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def build_parser():
    p = argparse.ArgumentParser(
        description="Run sparse PHATE — config-driven with YAML support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All config keys can be overridden via CLI flags.",
    )
    p.add_argument("-c", "--config", type=str, help="YAML config file")
    p.add_argument("--show-config", action="store_true", help="Print resolved config and exit")
    p.add_argument("--input", type=str, help="Input file path")
    p.add_argument("--output", type=str, help="Output parquet path")
    p.add_argument("--sparse-k", type=int, default=None, dest="sparse_k", help="Top-k neighbors")
    p.add_argument("-t", "--time", type=int, default=None, dest="t", help="Diffusion time steps")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--decay", type=float, default=None)
    p.add_argument("--metric", type=str, default=None, dest="sparse_metric",
                   choices=["euclidean", "cosine", "correlation"])
    p.add_argument("--ndim", type=int, default=None, dest="n_components", help="Embedding dimensions")
    p.add_argument("--device", type=str, default=None, dest="sparse_device",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--no-header", action="store_true", default=None)
    p.add_argument("--no-verbose", action="store_false", default=None, dest="verbose")
    p.add_argument("--plot-dir", type=str, default=None, dest="plot_dir", help="Output directory for plots")
    p.add_argument("--plot-format", type=str, default=None, dest="plot_format",
                   choices=["png", "html"], help="Plot format: png or html")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    args = build_parser().parse_args()
    config = resolve_config(args)

    if args.show_config:
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        sys.exit(0)

    if not config["input"] or not config["output"]:
        print("ERROR: --input and --output (or 'input'/'output' in YAML) are required", file=sys.stderr)
        sys.exit(1)

    # Load
    t0 = time.time()
    data, labels = load_data(config["input"], no_header=config["no_header"])
    print(f"Loaded {config['input']}: {data.shape}, {data.dtype}, {time.time() - t0:.1f}s", file=sys.stderr)

    # Run
    summary = (
        f"k={config['sparse_k']}, t={config['t']}, gamma={config['gamma']}, "
        f"metric={config['sparse_metric']}, ndim={config['n_components']}, "
        f"device={config['sparse_device']}"
    )
    print(f"PHATE: {summary}", file=sys.stderr)
    t0 = time.time()

    import phate
    embedding = phate.PHATE(
        sparse_k=config["sparse_k"],
        t=config["t"],
        gamma=config["gamma"],
        decay=config["decay"],
        sparse_metric=config["sparse_metric"],
        sparse_backend="torch",
        sparse_device=config["sparse_device"],
        sparse_batch_size=config["batch_size"],
        n_components=config["n_components"],
        verbose=config["verbose"],
    ).fit_transform(data)

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s, embedding={embedding.shape}", file=sys.stderr)

    # Save
    save_parquet(config["output"], embedding, labels)
    print(f"Saved: {config['output']}", file=sys.stderr)

    # Plot
    if config["plot_dir"]:
        os.makedirs(config["plot_dir"], exist_ok=True)
        _plot(embedding, labels, config)


def _plot(embedding, labels, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = config["plot_dir"]
    fmt = config["plot_format"]
    ndim = embedding.shape[1]

    if fmt == "html":
        import plotly.graph_objects as go
        fig = go.Figure()
        Trace = go.Scatter3d if ndim == 3 else go.Scatter
        coords = (lambda e: dict(x=e[:, 0], y=e[:, 1], z=e[:, 2])) if ndim == 3 else \
                 (lambda e: dict(x=e[:, 0], y=e[:, 1]))
        color = labels if labels is not None else np.arange(len(embedding))
        fig.add_trace(Trace(**coords(embedding), mode="markers",
                           marker=dict(size=2, color=color, colorscale="Viridis")))
        path = os.path.join(plot_dir, f"phate_embedding.{fmt}")
        fig.write_html(path)
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        c = labels if labels is not None else np.arange(len(embedding))
        if ndim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=c, cmap="viridis", s=3)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], c=c, cmap="viridis", s=3)
            ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        path = os.path.join(plot_dir, f"phate_embedding.{fmt}")
        fig.savefig(path, dpi=150)
        plt.close()
    print(f"Plot: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
