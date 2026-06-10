# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Plotting convenience functions
import anndata

from .phate import PHATE


def _get_plot_data(data, ndim=None):
    """Get plot data out of an input object

    Parameters
    ----------
    data : array-like, `phate.PHATE` or `scanpy.AnnData`
    ndim : int, optional (default: None)
        Minimum number of dimensions
    """
    out = data
    if isinstance(data, PHATE):
        out = data.transform()
    elif isinstance(data, anndata.AnnData):
        try:
            out = data.obsm["X_phate"]
        except KeyError:
            raise RuntimeError(
                "data.obsm['X_phate'] not found. "
                "Please run `sc.tl.phate(adata)` before plotting."
            )
    if ndim is not None and out[0].shape[0] < ndim:
        if isinstance(data, PHATE):
            data.set_params(n_components=ndim)
            out = data.transform()
        else:
            raise ValueError(
                f"Expected at least {ndim}-dimensional data, got {out[0].shape[0]}"
            )
    return out
