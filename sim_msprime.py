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

m = sys.argv[1]  # model name
s = int(sys.argv[2])  # seed
model = demes.load(f"data/template_models/Demography1_{m}.yaml")
demo = msprime.Demography.from_demes(model)
pop_dict = {i.name: i.id for i in demo.populations}
n = 100
# seeds = range(1, 11)
# for s in seeds:

ts = msprime.sim_ancestry(
    samples = [
        msprime.SampleSet(n, population=pop_dict["NonAfrican"]),
        msprime.SampleSet(n, population=pop_dict["African"]),
        msprime.SampleSet(n, population=pop_dict["East_African"]),
        msprime.SampleSet(1, population=pop_dict["Seq_NEA"], time=1600),
        msprime.SampleSet(1, population=pop_dict["Seq_DEN"], time=1500),
    ],
    sequence_length=50e6,
    recombination_rate=1e-8,
    # recombination_rate=msprime.RateMap.read_hapmap("results/realdata/1000g_hg38_2022/genetic_map_hg38/genetic_map_hg38_chr21.txt", sequence_length=50e6),
    random_seed=s,
    demography=demo,
    record_migrations = True,
)
mts = msprime.sim_mutations(ts, rate=1.2e-8, random_seed=s)
with open(f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}.vcf", "w") as vcf:
    mts.write_vcf(vcf, contig_id=1)
tszip.compress(ts, f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}.tsz")

arg_utils = ARG_utils(
    # total_sample_size=4 * n + 4,
    total_sample_size=6 * n + 4,
    afr_poplabel=pop_dict["NonAfrican"],
    ghost_poplabel=pop_dict["Intro_NEA"],
)
arg_utils.add_tree_sequence(ts)
arg_utils.extract_ghost_intro_all(
    from_pop=pop_dict["NonAfrican"],
    to_pop=pop_dict["Intro_NEA"],
)
arg_utils.write_bed_output(
    f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}.NEA.indiv.bed", arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
)
arg_utils.ne_seg = {i:[] for i in range(arg_utils.total_sample_size)}
arg_utils.extract_ghost_intro_all(
    from_pop=pop_dict["NonAfrican"],
    to_pop=pop_dict["Intro_DEN"],
)
arg_utils.write_bed_output(
    f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}.DEN.indiv.bed", arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
)
arg_utils.ne_seg = {i:[] for i in range(arg_utils.total_sample_size)}
arg_utils.extract_ghost_intro_all(
    from_pop=pop_dict["African"],
    to_pop=pop_dict["Ghost"],
)
arg_utils.write_bed_output(
    f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}.Ghost.indiv.bed", arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
)
# arg_utils.ne_seg = {i:[] for i in range(arg_utils.total_sample_size)}
# arg_utils.extract_ghost_intro_all(
#     from_pop=pop_dict["DEN"],
#     to_pop=pop_dict["SUPER"],
# )
# arg_utils.write_bed_output(
#     f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}.SUPER.indiv.bed", arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
# )
with open(f"results/realdata/test_simulation/{m}_model/msprime/n100_seed{s}_samples.json", "w") as f:
    f.write('{\n\t"ingroup": [\n\t\t')
    for i in range(n - 1):
        f.write('"' + f"tsk_{i}" + '",\n\t\t')
    f.write('"' + f"tsk_{n-1}" + '"\n\t],\n\t"outgroup": [\n\t\t')
    for i in range(n, 2 * n - 1):
        f.write('"' + f"tsk_{i}" + '",\n\t\t')
    f.write('"' + f"tsk_{2*n-1}" + '"]\n')
    f.write("}")