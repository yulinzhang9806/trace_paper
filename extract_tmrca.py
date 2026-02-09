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


tsz = sys.argv[1]
outpref = sys.argv[2]
start_ind = int(sys.argv[3])
end_ind = int(sys.argv[4])
inds = range(start_ind, end_ind)
other_samples = list(range(0, 182))
extract_tmrca(tsz, inds, other_samples, outpref, windowsize=10000)