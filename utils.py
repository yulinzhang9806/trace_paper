"""Utility functions for introgression in Tree-Sequences."""

import os

import numpy as np
import pandas as pd
import polars as pl
import tskit
from intervaltree import Interval, IntervalTree
from numpy import mean, median, var
from scipy.optimize import differential_evolution, minimize
from scipy.stats import gamma
import matplotlib.pyplot as plt
import warnings
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

    def remove_border(self, ax):
        """
        Remove border spines completely from a matplotlib axis.

        :param matplotlib.pyplot.axis ax: Input axis.
        """
        for i in ["top", "bottom", "left", "right"]:
            ax.spines[i].set_visible(False)

    def remove_ticks(self, ax):
        """
        Remove the x and y ticks for conceptual plots.

        :param matplotlib.pyplot.axis ax: Input axis.
        """
        ax.set_yticks([])
        ax.set_xticks([])

    def create_ideogram(self, chrom_df=None, **kwargs):
        if chrom_df is None:
            raise ValueError("Chromosome lengths need to be specified")
        else:
            assert "chrom" in chrom_df.columns
            assert "size" in chrom_df.columns
            fig, axs = plt.subplots(22, 1, **kwargs)
            m_size = 0
            for i in range(1, 23):
                l = chrom_df.filter(pl.col("chrom") == f"chr{i}")["size"].to_numpy()[0]
                if l >= m_size:
                    m_size = l
                self.remove_border(axs[i - 1])
                self.remove_ticks(axs[i - 1])
                axs[i - 1].add_patch(
                    plt.Rectangle(
                        (0, 0), l / 1e6, 1, ls="-", lw=1, ec="black", fc="none"
                    )
                )
                axs[i - 1].plot([0, l / 1e6], [1, 1], color="none")
            return fig, axs, m_size

    def draw_deserts(self, axs, deserts_df=None, category="conservative", **kwargs):
        """Draw deserts onto specific chromosomes."""
        assert len(axs) == 22
        # assert category in ['default', 'kerdoncuff', 'vernot', 'sankararaman', 'conservative']
        assert deserts_df is not None
        filt_deserts = deserts_df.filter(pl.col("type") == category)
        for chrom, start, end in zip(
            filt_deserts["chrom"].to_numpy(),
            filt_deserts["start"].to_numpy(),
            filt_deserts["end"].to_numpy(),
        ):
            try:
                axid = int(chrom[3:]) - 1
                axs[axid].axvspan(start / 1e6, end / 1e6, **kwargs)
            except ValueError:
                warnings.warn(f"{chrom} is not currently parseable!")
        return axs
