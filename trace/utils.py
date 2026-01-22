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

class Output_utils:
    """Utility to filter and generate hmm output files."""

    def __init__(
        self,
        samplefile=None,
        samplename=None,
    ):
        """Initialize the ARG-output class."""
        self.samplefile = samplefile
        self.samplename = samplename

    def read_samplename(self):
        """Read from samplename file, return two dictionaries."""
        assert self.samplefile is not None
        samplename_to_tsid = {}
        tsid_to_samplename = {}
        print(f"Reading sample name file {self.samplefile}")
        infile = open(self.samplefile)
        lines = infile.readlines()
        infile.close()
        for i in range(len(lines)):
            s = lines[i].strip('\n').split() # split by tab or space, check this
            if i == 0:
                try:
                    samplename_to_tsid[str(s[1])] = int(s[0])
                    tsid_to_samplename[int(s[0])] = str(s[1])
                except:
                    continue
            elif len(s) >= 2:
                samplename_to_tsid[str(s[1])] = int(s[0])
                tsid_to_samplename[int(s[0])] = str(s[1])
            elif len(s) == 0 and i == len(lines) - 1:
                continue
            else:
                print("Error: empty row")
                sys.exit(0)
        return samplename_to_tsid, tsid_to_samplename

    def write_raw_pp(self, exp_pp, treespan_phy, treespan_cM, chrom, outpref, subrange):
        assert self.samplename is not None
        out = 'chromosome\tstart(physical)\tend(physical)\tstart(cM)\tend(cM)\tArchaic\tHuman\n'
        for i in range(treespan_phy.shape[0]):
            out += chrom + '\t' + str(int(treespan_phy[i, 0])) + '\t' + str(int(treespan_phy[i, 1])) + '\t'
            out += str(treespan_cM[i, 0]) + '\t' + str(treespan_cM[i, 1]) + '\t'
            out += str(round(exp_pp[1, i], 3)) + '\t' + str(round(exp_pp[0, i], 3)) + '\n'
        outf = f"{outpref}{self.samplename}.posteriorProb.chr{chrom}_{int(subrange[0])}_{int(subrange[1])}.txt"
        print(f"Writing to {outf}")
        outfile = open(outf, "w")
        outfile.write(out)
        outfile.close()

    def filter_tracts(
        self,
        indiv_pp,
        treespan,
        treespan_phy,
        pp_cutoff=0.9,
        arc_cutoff=0.5,
        phy_cutoff=5e4,
        l_cutoff=0.05,
        remove_margin=0,
    ):
        tracts = []
        states = np.zeros(indiv_pp.shape[1])
        i = 0
        while i < indiv_pp.shape[1]:
            if indiv_pp[1][i] >= pp_cutoff:
                j = i
                temp_pos = []
                temp_pp = []
                while (
                    j < len(indiv_pp[1]) and indiv_pp[1][j] >= arc_cutoff
                ):
                    temp_pos.append(j)
                    temp_pp.append(indiv_pp[1][j])
                    j += 1
                if (
                    np.mean(temp_pp) >= pp_cutoff 
                    and treespan[np.max(temp_pos)][1] 
                    - treespan[np.min(temp_pos)][0]
                    >= l_cutoff
                    and treespan_phy[np.max(temp_pos)][1]
                    - treespan_phy[np.min(temp_pos)][0]
                    >= phy_cutoff
                    ):
                    start = treespan_phy[temp_pos[0]][0]
                    end = treespan_phy[temp_pos[-1]][1]
                    tracts.append([start, end, np.mean(temp_pp), end - start, treespan[temp_pos[-1]][1] - treespan[temp_pos[0]][0]])
                    states[np.array(temp_pos[remove_margin:(len(temp_pos) - remove_margin)])] = 1
                i = j
            else:
                i += 1
        return tracts, states

    def summarize(
        self,
        pp,
        treespan,
        treespan_phy,
        outpref,
        chrom="chr1",
        pp_cutoff=0.9,
        phy_cutoff=5e4,
        l_cutoff=0.05,
        remove_margin=0,
    ):
        """Summarize all pp results, do posterior cutoff based on thresholds shown."""

        # read pp matrix, 0 is human, 1 is archaic, for each chromosome separately
        out = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
        tracts, states = self.filter_tracts(
            pp, treespan, treespan_phy, pp_cutoff=pp_cutoff, phy_cutoff=phy_cutoff, l_cutoff=l_cutoff, remove_margin=remove_margin
        )
        for i in range(len(tracts)):
            out += (
                chrom
                + "\t"
                + str(int(tracts[i][0]))
                + "\t"
                + str(int(tracts[i][1]))
                + "\t"
                + str(round(tracts[i][2], 2))
                + "\t"
                + str(int(tracts[i][3]))
                + "\t"
                + str(round(tracts[i][4], 3))
                + "\n"
            )
        outfile = open(str(outpref) + ".summary.txt", "w")
        outfile.write(out)
        outfile.close()
        return states

class ExplicitDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    format the help menu such that defaults are printed to help
    only when they are explicitly stated in the parser
    (e.g. do not print actions or Nones)
    """

    def _get_help_string(self, action):
        """
        returns the help string
        """

        if action.default in (None, False):
            return action.help
        return super()._get_help_string(action)
