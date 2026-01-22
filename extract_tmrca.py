import demes
import msprime
import tszip
import yaml
from arg_hmm.utils import *
import tskit
import numpy as np
import pandas as pd
import pybedtools
import sys
import os

def extract_tmrca(input_tsz, inds, other_samples, outpref, windowsize=1000):
    def get_pairwise_times(ts, windowsize, k1, k2):
        """Function for getting pairwise average tmrca over the genome."""
        windows = np.arange(0, ts.sequence_length, windowsize)
        windows = np.append(windows, ts.sequence_length)
        times = ts.divergence(sample_sets=[k1, k2], windows=windows, mode='branch')
        return windows, 0.5*times
    if input_tsz.endswith(".tsz"):
        ts = tszip.decompress(input_tsz)
    else:
        ts = tskit.load(input_tsz)
    out = np.zeros(shape=(len(inds), len(other_samples), int(ts.sequence_length / windowsize) + (1 if ts.sequence_length % windowsize > 0 else 0)))
    # out = np.zeros(shape=(len(inds), int(ts.sequence_length / windowsize) + (1 if ts.sequence_length % windowsize > 0 else 0)))
    for i in range(len(inds)):
        k1 = inds[i]
        os = [s for s in other_samples if s != k1]
        for j in range(len(os)):
            k2 = os[j]
            windows, times = get_pairwise_times(ts, windowsize, [k1], [k2])
            out[i, j, :] = times
        # windows, times = get_pairwise_times(ts, windowsize, [k1], other_samples)
        # out[i] = times
    np.savez_compressed(f"{outpref}.npz", tmrca=out, windows=windows, inds=inds, other_samples=other_samples)


tsz = sys.argv[1]
outpref = sys.argv[2]
start_ind = int(sys.argv[3])
end_ind = int(sys.argv[4])
inds = range(start_ind, end_ind)
other_samples = list(range(0, 182))
extract_tmrca(tsz, inds, other_samples, outpref, windowsize=10000)