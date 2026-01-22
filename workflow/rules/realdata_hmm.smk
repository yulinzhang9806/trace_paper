#!python3

import numpy as np
import tskit
import tszip
import msprime
import sys
import os
import pandas as pd
from workflow.scripts.utils import Analysis_workflow_utils
from arg_hmm.utils import Performance_utils
import demes

rule singer_inputs_extract:
    """Generate input files for the hmm."""
    input:
        singer_tsz = "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_{post}.tsz",
        indiv_file = "results/realdata/{dataset}/GhostHMM/analysis_samples.txt",
        pilot_mask = paths['pilot_mask_hg38'],
    params:
        t_archaic = lambda wildcards: wildcards.tarchaic,
        outpref = "results/realdata/{dataset}/GhostHMM/singer/chr{chrom}_t{tarchaic}_{post}",
        chrom = lambda wildcards: wildcards.chrom,
        window_size = 1000,
        func = "mean",
        sample_names = None,
    output:
        npzfile = "results/realdata/{dataset}/GhostHMM/singer/chr{chrom}_t{tarchaic}_{post}.npz",
    wildcard.constraints:
        post = "|".join(range(190, 200))
    shell:
        """
        python getdata_singer.py --tree-file {input.singer_tsz} --individual-file {input.indiv_file} \
        --chrom {params.chrom} --t-archaic {params.t_archaic} --outpref {params.outpref} \
        --include-regions {input.pilot_mask} --window-size {params.window_size} --func {params.func} \
        --sample-names {params.sample_names}
        """


rule hmm_singer:
    """Run hmm on singer posterior samples."""
    input:
        npzfiles = expand("results/realdata/{dataset}/GhostHMM/singer/chr{chrom}_t{tarchaic}_{post}.npz", post = range(190, 200), allow_missing = True), 
        data_file="results/realdata/{dataset}/GhostHMM/singer/chr1_full_files",
        genetic_map="results/realdata/{dataset}/genetic_map_hg38/genetic_map_hg38_chr{chrom}.txt",
    output:
        raw_output="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.xss.npz",
    params:
        indiv = lambda wildcards: wildcards.ind,
        chrom = lambda wildcards: wildcards.chrom,
        t_archaic = lambda wildcards: wildcards.tarchaic,
        subrange = None,
        include_regions = None,
        func = "mean",
        outpref = "results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind",
    shell:
        """
        python run_arghmm.py --data-file {input.data_file} --individual {params.indiv} \
        --genetic-map {input.genetic_map} --chrom {params.chrom} --t-archaic {params.t_archaic} \
        --subrange {params.subrange} --include-regions {params.include_regions} --func {params.func} \
        --outpref {params.outpref}
        """

rule hmm_singer_filter:
    """Filter hmm output."""
    input:
        raw_output="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.xss.npz",
    output:
        tracts="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.summary.txt",
    params:
        chrom = lambda wildcards: wildcards.chrom,
        pp = 0.9,
        phy_cutoff = 50000,
        l_cutoff = 0.05,
        outpref = "results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}",
    shell:
        """
        python summarize.py --file {input.raw_output} --chrom {params.chrom} \
        --posterior-threshold {params.pp} --physical-length-threshold {params.phy_cutoff} --genetic-distance-threshold {params.l_cutoff} \
        --out {params.outpref}
        """

rule annotate_hmmix_ibdmix:
    """Annotate with hmmix and ibdmix output."""
    input:
        tracts="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.summary.txt",
        hmmix=HMMIX_FILE[int(lambda wildcards: wildcards.ind)],
        ibdmix_nea=IBDMIX_NEA_FILE[int(lambda wildcards: wildcards.ind)],
        ibdmix_den=IBDMIX_DEN_FILE[int(lambda wildcards: wildcards.ind)],
    output:
        tracts_annotated="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.summary.hmmix.ibdmix.txt",
    params:
        chrom = lambda wildcards: wildcards.chrom,
        hmmix_pref = "results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.hmmix",
        ibdmix_pref = "results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.hmmix.ibdmix",
        indID = individualID[int(lambda wildcards: wildcards.ind)],
        ibdmix_annotated = IBDMIX_PREF + "/chr{chrom}.txt",
    run:
        a = pd.read_csv(ibdmix_nea, sep="\s+")
        b = pd.read_csv(ibdmix_den, sep="\s+")
        a["archaic"] = "Neanderthal"
        b["archaic"] = "Denisova"
        c = pd.concat([a, b])
        c = c[c["chrom"]=="chr1"]
        c.to_csv(params.ibdmix_annotated, index=False, sep="\t")
        SUMMARIZE().append_hmmix_info(hmmixfile, summaryfile, params.hmmix_pref, inference = "hmmix", individualID = None, pp_cutoff = 0.8, l_cutoff = 5e4)
        SUMMARIZE().append_ibdmix_info(ibdmixfile, summaryfile, params.ibdmix_pref, inference = "ibdmix", individualID = params.indID, pp_cutoff = 0.8, l_cutoff = 5e4)

rule ind_count:
    """Get individual mutation counts."""
    input:
        tracts_annotated="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.summary.hmmix.ibdmix.txt",
        snpinfo = expand("results/realdata/{dataset}/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.afr.outgroup.chr{i}.txt", i = range(1,23), allow_missing = True),
        bcffile = expand("results/realdata/{dataset}/vcf_files/1000g_hg38_chr{i}.bcf", i = range(1,23), allow_missing = True),
    params:
        ind = lambda wildcards: wildcards.ind,
        samplename = individualID[int(lambda wildcards: wildcards.ind)],
        snpinfo = "results/realdata/{dataset}/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.afr.outgroup",
        bcfpref = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr",
        outpref = "results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.summary.hmmix.ibdmix.counts",
    output:
        txtfile="results/realdata/{dataset}/GhostHMM/singerave_t{tarchaic}_ind{ind}.chr{chrom}.summary.hmmix.ibdmix.counts.txt",
    run:
        hap = "left" if params.ind%2 == 0 else "right"
        SUMMARIZE().final_ind_count(samplename = params.samplename, summary = input.tracts_annotated, hap = hap, snpinfo = params.snpinfo, bcfpref = params.bcfpref, outpref)