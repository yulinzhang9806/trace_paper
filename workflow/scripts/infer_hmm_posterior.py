#!python3

"""Perform parameter estimation under the current HMM model."""

import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(".")
from arg_hmm.arg_hmm import ARG_HMM_BRANCH  # noqa


def read_hmm_params(hmm_params_yaml):
    """Read the HMM parameters from a yaml file."""
    with open(hmm_params_yaml, "r") as params:
        param_dict = yaml.safe_load(params)
    if "p" not in param_dict:
        param_dict["p"] = 0.01
    if "q" not in param_dict:
        param_dict["q"] = 0.01
    return param_dict


def set_binary_emissions(
    emission_bins, t_null_prob, t_admix_prob, t_archaic=10000, eps=1e-8
):
    """Set a binary HMM emission model."""
    n = emission_bins.size
    emissions = np.zeros(shape=(2, n))
    emissions[0, np.where(emission_bins < t_archaic)] = t_null_prob
    emissions[1, np.where(emission_bins >= t_archaic)] = t_admix_prob
    emissions += eps
    return emissions


def infer_hmm_posterior(txs, pos, param_dict, t=1000):
    """Infer the parameters of the HMM.

    txs (`np.array`): multi-dimensional array of times
    pos (`np.array`): positions in the HMM.
    param_dict (`dict`): dictionary of parameters for inference under the HMM model.
    t (`int`): admixture time (or cutoff time for branches)
    """
    assert txs.ndim == 2
    assert txs.shape[0] >= 1
    assert t >= 0
    arg_hmm = ARG_HMM_BRANCH(
        t_admix=t,
        alpha_upper_normal=param_dict["a1"],
        beta_upper_normal=param_dict["b1"],
        alpha_upper_intro=param_dict["a2"],
        beta_upper_intro=param_dict["b2"],
        admix_prop=param_dict["w"],
    )
    arg_hmm.p = param_dict["p"]
    arg_hmm.q = param_dict["q"]
    arg_hmm.m = txs.shape[1]
    arg_hmm.bulk_t_upper = txs
    arg_hmm.t = txs[0, :]
    arg_hmm.pos = pos
    arg_hmm.set_constant_recomb()
    arg_hmm.set_emission_binning()
    if "t_null_prob" in param_dict:
        # Setting the binary emissions for branch lengths ...
        emissions = set_binary_emissions(
            arg_hmm.emission_bins,
            t_null_prob=param_dict["t_null_prob"],
            t_admix_prob=param_dict["t_admix_prob"],
            t_archaic=param_dict["t_archaic"],
        )
        arg_hmm.emissions = emissions
    else:
        arg_hmm.set_emission_dist()
    posterior = np.zeros(shape=(txs.shape[0], arg_hmm.m))
    for i in tqdm(range(txs.shape[0])):
        arg_hmm.t = txs[i, :]
        gammas, _, _ = arg_hmm.forward_backward()
        # Only take the posterior probability of being in the archaic state.
        posterior[i, :] = np.exp(gammas[1, :])
    return posterior


if __name__ == "__main__":
    data = np.load(snakemake.input["hmm_data"])  # noqa
    txs_sim = data["txs"]
    pos = data["pos"]
    param_dict = read_hmm_params(snakemake.input["hmm_params"])  # noqa
    est_posterior = infer_hmm_posterior(
        txs_sim, pos, param_dict, t=snakemake.params["t"]  # noqa
    )
    np.savez_compressed(
        snakemake.output["posterior_data"], posteriors=est_posterior, pos=pos  # noqa
    )
