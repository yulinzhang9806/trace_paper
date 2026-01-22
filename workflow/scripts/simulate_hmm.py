#!python3
"""Simulate data under the mixture of Gammas HMM model."""

import sys

import numpy as np
import pandas as pd

sys.path.append(".")
from arg_hmm.arg_hmm import ARG_HMM_BRANCH  # noqa


def simulate_hmm_data(
    n=10, nsites=100, a1=6.6, b1=1e-4, a2=4.0, b2=1e-4, p=1e-2, q=1e-1, w=1e-1, seed=42
):
    """Simulate data from an HMM with a mixture of Gamma distributions."""
    arg_hmm = ARG_HMM_BRANCH(
        alpha_upper_normal=a1,
        beta_upper_normal=b1,
        alpha_upper_intro=a2,
        beta_upper_intro=b2,
        admix_prop=w,
    )
    arg_hmm.m = nsites
    txs_test = np.zeros(shape=(n, arg_hmm.m))
    zs_test = np.zeros(shape=(n, arg_hmm.m))
    for i in range(n):
        xs, zs = arg_hmm.simulate_data(seed=i + seed + 1, p=p, q=q)
        txs_test[i, :] = xs
        zs_test[i, :] = zs
    return txs_test, zs_test


if __name__ == "__main__":
    txs_sim, zs_test = simulate_hmm_data(
        n=snakemake.params["n"],  # noqa
        nsites=snakemake.params["l"],  # noqa
        a1=snakemake.params["a1"],  # noqa
        b1=snakemake.params["b1"],  # noqa
        a2=snakemake.params["a2"],  # noqa
        b2=snakemake.params["b2"],  # noqa
        p=snakemake.params["p"],  # noqa
        q=snakemake.params["q"],  # noqa
        w=snakemake.params["w"],  # noqa
        seed=int(snakemake.wildcards["seed"]),  # noqa
    )
    np.savez_compressed(snakemake.output["hmm_data"], txs=txs_sim, zs=zs_test)  # noqa
