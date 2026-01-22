#!python3

"""Perform parameter estimation under a mixture of Gamma distribution model."""

import argparse
import sys

import numpy as np
import yaml

sys.path.append(".")
from arg_hmm.utils import ARG_Prior_utils  # noqa


def infer_gamma_mixture_params(
    txs, t=0.0, pseudocount=1e-3, stop_threshold=1e-5, inertia=0.0, verbose=0, n_jobs=4
):
    """Infer the parameters of the HMM.

    txs (`np.array`): multi-dimensional array of times
    """
    assert txs.ndim == 2
    assert txs.shape[0] > 1
    assert t >= 0
    arg_prior_utils = ARG_Prior_utils()
    concat_txs = np.concatenate(txs, axis=None) - t
    a1_est, b1_est, a2_est, b2_est, w = arg_prior_utils.estimate_gamma_mixture(
        concat_txs,
        pseudocount=pseudocount,
        stop_threshold=stop_threshold,
        inertia=inertia,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    mu1 = a1_est / b1_est
    mu2 = a2_est / b2_est
    if mu1 > mu2:
        est_params = {
            "a1": float(a2_est),
            "b1": float(b2_est),
            "a2": float(a1_est),
            "b2": float(b1_est),
            "w": float(1 - w),
            "p": 0.01,
            "q": 0.05,
        }
    else:
        est_params = {
            "a1": float(a1_est),
            "b1": float(b1_est),
            "a2": float(a2_est),
            "b2": float(b2_est),
            "w": float(1 - w),
            "p": 0.01,
            "q": 0.05,
        }
    return est_params


def main():
    """Estimate parameters for a simulated set of TMRCAs or Branch Lengths."""
    parser = argparse.ArgumentParser(
        description="Simulating data under a joint gamma HMM model."
    )
    parser.add_argument("--input_txs", "-i", help="Number of samples as replicates.")
    parser.add_argument("--t", "-t", help="Inferred admixture timing", type=float)
    parser.add_argument("--n", "-n", help="Number of jobs", default=4, type=int)
    parser.add_argument("--out", "-o", help="output numpy arrays stored.")
    args = parser.parse_args()
    data = np.load(args.input)
    txs_sim = data["txs"]
    est_params = infer_gamma_mixture_params(txs_sim, args.t)
    with open(args.out, "w") as outfile:
        yaml.dump(est_params, outfile, default_flow_style=False)


if __name__ == "__main__":
    data = np.load(snakemake.input["hmm_data"])  # noqa
    txs_sim = data["txs"]
    est_params = infer_gamma_mixture_params(
        txs_sim, snakemake.params["t"], n_jobs=snakemake.threads  # noqa
    )
    with open(snakemake.output["inf_gamma_params"], "w") as outfile:  # noqa
        yaml.dump(est_params, outfile, default_flow_style=False)
