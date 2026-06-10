# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Generating random fractal tree via DLA
import numpy as np
from scipy.io import loadmat

# random tree via diffusion limited aggregation


def gen_dla(
    n_dim=100, n_branch=20, branch_length=100, rand_multiplier=2, seed=37, sigma=4
):
    """Generate fractal tree data via Diffusion-Limited Aggregation.

    O(N·d) time/memory via pre-allocation. Scales to N=500K+.
    """
    np.random.seed(seed)
    N = n_branch * branch_length
    M = np.empty((N, n_dim))

    steps = -1 + rand_multiplier * np.random.rand(branch_length, n_dim)
    M[:branch_length] = np.cumsum(steps, axis=0)

    for i in range(1, n_branch):
        start = i * branch_length
        ind = np.random.randint(start)
        new_branch = np.cumsum(
            -1 + rand_multiplier * np.random.rand(branch_length, n_dim), axis=0
        )
        M[start : start + branch_length] = new_branch + M[ind]

    M += np.random.normal(0, sigma, M.shape)
    C = np.repeat(np.arange(n_branch), branch_length)

    return M, C


def artificial_tree():
    tree = loadmat("../../data/TreeData.mat")
    return tree["M"], tree["C"]
