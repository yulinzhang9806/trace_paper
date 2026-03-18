"""Utility functions for introgression in Tree-Sequences."""

import os

import numpy as np
import pandas as pd
import tskit
from intervaltree import Interval, IntervalTree
from numpy import mean, median, var
from scipy.optimize import differential_evolution, minimize
from scipy.stats import gamma
from tqdm import tqdm
import sys
import argparse


class Figure_utils:
    """Utility for data processing and figure generation."""

    def bootstrap(self, data, nboot, ntime, func=np.mean, replace=True):
        """Bootstrapping function. Returns a 1-dimensional numpy array with ntime entries.

        data: 1-dimensional numpy array including all data points.
        nboot: bootstrap sample size.
        ntime: times repeating the bootstrap process.
        func: function applied to the bootstrapping samples.
        replace: sample with replacement or not.
        """
        out = []
        for i in range(ntime):
            sample = np.random.choice(data, nboot, replace=replace)
            out.append(func(sample))
        return np.array(out)

    def merge_seeds(self, filepath, t_admix, t_archaic, seeds):
        """Merge datafiles from simulations of different seeds."""
        out_df = pd.read_csv(
            str(filepath)
            + str(seeds[0])
            + "_"
            + str(t_admix)
            + "_"
            + str(t_archaic)
            + "_data.tsv",
            sep="\t",
        )
        out_df["seed"] = seeds[0]
        for s in range(1, len(seeds)):
            df = pd.read_csv(
                str(filepath)
                + str(seeds[s])
                + "_"
                + str(t_admix)
                + "_"
                + str(t_archaic)
                + "_data.tsv",
                sep="\t",
            )
            df["seed"] = seeds[s]
            out_df = pd.concat([out_df, df])
        return out_df

    def cut_dataframe(self, df, pp_cutoff, l_cutoff):
        """Cut dataframe according to posterior probability threshold and haplotype length threshold. Remove na."""
        out_df = df[
            (df["mean_posterior_cutoff"] == pp_cutoff)
            & (df["length_cutoff"] == l_cutoff)
        ]
        out_df = out_df.dropna(subset=["recall"])
        return out_df

    def get_datapoints(self, df, colname, ntime, pp_cutoff, l_cutoff):
        """Get datapoints for different length cutoffs with stderr for plotting."""
        out = []
        outerr = []
        for ll in l_cutoff:
            subdf = self.cut_dataframe(df=df, pp_cutoff=pp_cutoff, l_cutoff=ll)
            pres = self.bootstrap(
                data=subdf[colname].to_numpy(),
                nboot=len(subdf),
                ntime=ntime,
                func=np.nanmean,
                replace=True,
            )
            out.append(np.nanmean(pres))
            outerr.append(
                np.sqrt((np.std(pres) ** 2) / (len(pres) - np.sum(np.isnan(pres))))
            )
        return np.array(out), np.array(outerr)

    def remove_spines(self, ax, right=True, top=True):
        """Remove right and/or upper spines of subplots."""
        if right:
            ax.spines["right"].set_visible(False)
        if top:
            ax.spines["top"].set_visible(False)
