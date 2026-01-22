#!python3

"""Perform parameter estimation under the current HMM model."""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.append(".")
from arg_hmm.arg_hmm import ARG_HMM_BRANCH  # noqa


def initialize_gamma_parameters(txs, t, niter=30):
    """Estimate gamma parameters from estimated tmrcas."""
    xs = txs - t
    assert np.all(xs > 0)
    half_quantile = np.quantile(xs, 0.5)
    # We do just kind of arbitrarily split it up here for initialization purposes...
    xs0 = xs[xs < half_quantile]
    xs1 = xs[xs > half_quantile]
    a1, b1 = ARG_HMM_BRANCH().est_gamma_mle(xs0, niter=niter)
    a2, b2 = ARG_HMM_BRANCH().est_gamma_mle(xs1, niter=niter)
    return a1, b1, a2, b2


def infer_hmm_params(txs, t=1000, mode="tmrca", niter=10):
    """Infer the parameters of the HMM.

    Args:
        txs (`np.array`): multi-dimensional array of times
        t (`int`): admixture time.
        mode (`string`): mode for operations
        niter (`int`): number of iterations for inference methods

    Returns:
        est_params (`dict`): estimated parameters.

    """
    assert txs.ndim == 2
    assert txs.shape[0] > 1
    assert t >= 0
    (a1, b1, a2, b2) = initialize_gamma_parameters(txs, t)
    arg_hmm = ARG_HMM_BRANCH(
        t_admix=t,
        alpha_upper_normal=a1,
        beta_upper_normal=b1,
        alpha_upper_intro=a2,
        beta_upper_intro=b2,
        admix_prop=0.5,
    )
    arg_hmm.p = 0.01
    arg_hmm.q = 0.01
    arg_hmm.m = txs.shape[1]
    if mode == "tmrca":
        arg_hmm.t_bulk = txs
    else:
        arg_hmm.bulk_t_upper = txs
    arg_hmm.t = txs[0, :]
    arg_hmm.set_emission_binning()
    arg_hmm.set_emission_dist()
    est_params = arg_hmm.composite_baum_welch(mode=mode, niter=niter)
    return est_params


def main():
    """Estimate parameters for a simulated set of TMRCAs or Branch Lengths."""
    parser = argparse.ArgumentParser(
        description="Simulating data under a joint gamma HMM model."
    )
    parser.add_argument("--input_txs", "-i", help="Number of samples as replicates.")
    parser.add_argument(
        "--t", "-t", help="Inferred admixture timing (in generations)", type=float
    )
    parser.add_argument(
        "--niter", "-n", help="Number of iterations of parameter evaluation.", type=int
    )
    parser.add_argument("--out", "-o", help="output numpy arrays stored.")
    args = parser.parse_args()
    data = np.load(args.input)
    txs_sim = data["txs"]
    est_params = infer_hmm_params(txs_sim, args.t, args.niter)
    pd.DataFrame.from_dict(est_params).to_csv(args.out, sep="\t", index=None)


if __name__ == "__main__":
    data = np.load(snakemake.input["hmm_data"])  # noqa
    txs_sim = data["txs"]
    est_params = infer_hmm_params(
        txs=txs_sim,
        t=snakemake.params["t"],  # noqa
        mode=snakemake.params["mode"],  # noqa
        niter=snakemake.params["niter"],  # noqa
    )
    pd.DataFrame.from_dict(est_params).to_csv(
        snakemake.output["inf_hmm_params"], sep="\t", index=None  # noqa
    )
